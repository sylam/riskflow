# Calc Lifecycle

The internal object walk behind a calculation. The public entry points and output shape are in [API Overview](../api_overview.md); this page traces what happens *between* `run_job` and the result.

## Dispatch

`Context.run_job` branches on `Calculation['Object']` into `run_cmc` / `run_baseval` / `run_hedgemontecarlo`, which set device/seed defaults then call `construct_calculation`. That constructor is `globals().get(calc_type)(config, **kwargs)` — a class-name `globals()` dispatch. The three classes are `Credit_Monte_Carlo`, `Base_Revaluation`, and `Hedge_Monte_Carlo`, all in `calculation.py`.

## Compile phase 1 — `calculate_dependencies`

Discovers the factor universe, wires the dependency DAG, topologically orders it, and splits stochastic vs static. This is a subsystem in itself — see [Dependency System](dependency_system.md). It returns `dependent_factors` (factor → max date), `stochastic_factors` (process-factor → price-factor), `additional_factors` (implied factors), plus reset/settlement date sets.

!!! danger "Invariant — `calculate_dependencies` is not idempotent"
    It **mutates** `self.params`: `add_rates_for_factor` writes a default block into `Price Factors` for a missing `DiscountRate`, and `find_models` injects `Price Models[model] = None` for implied models. A second call sees the injected defaults/dummies. Do not call it twice expecting identical output, and do not treat `Price Factors` / `Price Models` as pristine afterward.

## Compile phase 2 — `_build_factor_state`

Constructs the factor objects, mints the AAD leaves, and builds the processes. Key steps:

- **Factor objects.** Stochastic factors → `construct_process(model.type, factor_obj, Price Models[...], implied_obj)`; static factors = `set(dependent_factors) - stochastic_factors.values()` → `construct_factor`. `all_factors` = stochastic + static + implied.
- **AAD leaves.** Each stochastic factor's `current_value` (offset by `Tenor_Offset`) becomes a `torch.tensor(..., requires_grad=calc_grad)` in `stoch_var`; implied params in `implied_var`; static factors in `static_var`.
- **Implied-leaf dedupe.** A factor that is *both* a static dependent (e.g. a `…ModelParameters` block pulled in by a conditional field) and a spot process's implied factor must share **one** tensor. `implied_leaves` is consulted so the static leaf reuses the single implied tensor; `value.backward()` then sums both consumers' sensitivities into that one leaf.
- **Sizing.** `self.num_factors = sum(v.num_factors() for v in stoch_factors.values())` sizes the correlated random block.
- **precalculate, then calc_references.** First loop: cache `_factor_precalc_args[key]`, set `value.factor_key = key`, call `value.precalculate(..., self.process_ofs[key], ...)` (which sets `z_offset`, `spot0`). Second loop: `value.calc_references(...)` resolves cross-process links (e.g. `BasisLinkedSpotModel.calc_references` sets `linked_key` from the name-prefix parent). Order is load-bearing — references need the keys precalc set.

!!! warning "Invariant — implied-leaf dedupe"
    A factor that is simultaneously a static dependent and a spot process's implied factor must map to **one** `torch` tensor. Minting a second leaf under the same scope name splits gradients between the pricer path (`t_Static_Buffer`) and the scenario path (`implied_tensor`) and desyncs a bump.

!!! warning "Invariant — precalculate before calc_references"
    `precalculate` sets `value.factor_key`, `z_offset`, `spot0` and caches `_factor_precalc_args`; only then can `calc_references` resolve links that need `all_factors`. Both loops iterate `stoch_factors` in the same topological order, which is also what makes publish-as-you-go safe.

## Compile phase 3 — correlation + cholesky + `process_ofs`

`get_cholesky_decomp` iterates `stoch_factors.items()` (topological order). For each, `value.correlation_name` returns `(corr_type, [sub_factor_tuples])`; `self.process_ofs.setdefault(key, len(correlation_factors))` records the **row offset** of this factor's random substream, and each sub-factor appends a `Factor(corr_type, key.name+sub)` to `correlation_factors`. The symmetric correlation matrix is built from `Correlations`, healed to PSD if needed, and `torch.linalg.cholesky`'d.

!!! warning "Invariant — `num_factors()` must equal `len(correlation_name[1])`"
    The first sizes the rows of the correlated random block; the second sizes the correlation matrix and the `process_ofs` stride. They are wired together only by convention — no assert. A new process whose two counts disagree silently misaligns every downstream process's `z_offset`, so each reads another factor's substream. Numbers come out wrong, not erroring.

## Execute — the per-batch generate loop

`Credit_Monte_Carlo.execute` builds the `DealStructure` tree via `set_deal_structures` (each deal's `calc_dependencies` produces `Factor_dep`, `calc_time_dependency` produces `Time_dep`), then loops `Simulation_Batches`:

1. **`shared_mem.reset(num_factors, time_grid, antithetic)`** — draws `torch.randn(num_factors, sample*T)`, correlates by the cholesky, reshapes to `(num_factors, T, B)` into `t_random_numbers`; resets cashflows; **clears `t_Buffer`**. Outer draws are pseudo-random.
2. **Publish-as-you-go generate** — `for key, value in stoch_factors.items(): t_Scenario_Buffer[key] = value.generate(shared_mem)`. Iteration is topological, so a linked factor reads its parent's already-published path from `t_Scenario_Buffer`. Each process reads its substream via `t_random_numbers[self.z_offset, ...]` where `z_offset = process_ofs`.
3. **`resolve_structure`** — walks the netting tree, calls `deal.calculate` → `generate` + `pricing.interpolate`, accumulates MTM, runs `post_process`; owns cashflow reset/save for accumulating sub-structures and the `FLIP`-prefix sign inversion.
4. CVA/FVA/CollVA/IM adjustments follow. This block is **do-not-touch** ([Change Scope](conventions.md#change-scope)).

!!! warning "Invariant — `t_Buffer` is the memo table"
    `t_Buffer` is the memoized eval cache. It must be cleared between batches (`reset`) and between inner forks; never carry it across a random-number reset or a batch-size change.

!!! warning "Invariant — publish-as-you-go + the `(factor_key, kind)` convention"
    Every process publishes its own path to `t_Scenario_Buffer[key]` as `generate` returns, and any sufficient statistic under `(factor_key, kind)` (e.g. `(key,'regimes')`, `(key,'garch_log_h')`). The calc/solver never name a regime/belief/variance directly — they iterate the model-agnostic verbs. Base implementations of those verbs are inert no-ops.

## RNG-substream ordering (the reproducibility surface)

`process_ofs` is the row a factor reads from `t_random_numbers`; the iteration order is `stoch_factors` insertion order, which is the `topological_sort` output, which tie-breaks equal-depth factors by dict-insertion order (deal-walk order + `factor_fields` order + `dependant_fields` list order). The correlation structure is preserved under a permutation (cholesky rows permute consistently), but the **realized draws change bit-for-bit**. This is the single most important reproducibility invariant — stated in full on [Dependency System](dependency_system.md#rng-ordering). The HMM regime path draws from a **separate Sobol `quasi_rng` stream** whose batch counter advances across batches and is never reset by `reset()` — a second, parallel substream-assignment surface.

## Deal `Time_dep` / `Factor_dep` + pricing dispatch

- `Factor_dep` is the compiled factor-offset lookup a deal builds once in `calc_dependencies`, via the generic `get_*` layer → `calc_factor_code_chain` → `calc_factor_index`. It is stored verbatim as `DealDataType.Factor_dep` and consumed unchanged by `generate`. See [Resolver Layer](resolver_layer.md).
- `Time_dep` (`DealTimeDependencies`) precomputes interp indices/alphas against the mtm grid; `calculate` prices on the deal grid, `pricing.interpolate` gathers to the mtm grid and saves `Calc_res['Value']` (and, when `shared.keep_tensor`, `Calc_res['tensor']`).

## Valuation modes

**`Base_Revaluation`** is the degenerate lifecycle: a single time point (`TimeGrid({base_date}, …)`), no stochastic factors, everything a static leaf — no cholesky, no generate loop. `resolve_structure` runs once; greeks via `pricing.greeks`. It is the compile-plus-single-eval reference for reconciliation.

**`Hedge_Monte_Carlo`** inherits the full CMC scenario engine and diverges in what happens to the marks:

- **Own dependency assembly** (`update_factors`): merges deal-driven factors with the JSON `Scenario_Factors` list (factors no deal reaches, e.g. a basis consumed only by a composed spot), collapses per-factor tenors to a single horizon, caps the time grid at the liability terminal, then calls `_build_factor_state` directly.
- **Inner-MC shared state** (`_init_shared_mem`): builds `CMC_State_Inner` so one `shared_mem` hosts outer mode (`reset()`, pseudo-random `(F,T,B)`) and inner mode (`reset_inner()`, Sobol quasi-random `(F,T,B,B2)`); processes dispatch on `Z.ndim`.
- **Generate loop** adds: optional `Randomize_Initial_State` burn-in; `Observed_Scenario` path substitution + `reseed_from_path` (walk-forward replay); leafing the declared spot (`requires_grad_(True)`) for base-delta AAD; snapshotting the full outer `t_Scenario_Buffer` for on-demand inner forking. Marks are **harvested, not aggregated**: liability MTM via `resolve_hedge_structure` (post-process-free — no per-batch GPU→CPU copy), tradable tensors via `tensor_marks`.
- **Bundle + runtime**: `Bundle.from_batch` + `construct_hedge_runtime` + `run_hedge_execution`; in `solve_hedge` mode the bundle carries `inner_mc` / `inner_mc_grad` closures that fork inner MC on demand from the cached outer buffer.
- **A solve is a stream**: a bundle per batch, handed to a persistent solver as it is built — `StreamingSolve.warmup` on batch 1 (which locks the frame), `step` on each later batch, `finish` on a held-out final batch. Fork width follows `Batch_Size`, not the whole simulation. `Simulation_Batches` is therefore a stream length under `solve_hedge` and a path multiplier under `simulate_only` — the one genuinely different meaning between the two verbs. The end-to-end reproduction gate is `gates/wf_smoke_gate.sh` (trade 202001, 512x5 batches, seed 7).

!!! warning "Invariant — a frozen eval is the stream of length one"
    `DiffV2_Load_Value_Fn` means EVALUATION: the policy is the file's and there is nothing to fit,
    so `Simulation_Batches` must be 1 and that single batch is both the warmup bundle and the
    held-out world (frozen nets saw none of it). Two defences hold it: the contract refuses `N > 1`
    with a checkpoint loaded, so there are no `step` batches to sweep, and `step` refuses to sweep
    a loaded net anyway. `finish` skips its re-bind/re-fork when handed the bundle it is already
    bound to — without that the Sobol stream advances and the verdict moves.

!!! warning "Invariant — `Z.ndim` dispatch (outer vs inner MC)"
    `generate()` must handle both outer (`Z.ndim==2`, `(T,B)`) and inner (`Z.ndim==3`, `(T,B,B2)`) modes, with the per-outer-path initial state broadcast on the **middle** axis in inner mode. This is what lets one process instance serve both loops.

!!! warning "Invariant — inner-MC batch state + fail-loud pricing"
    `shared_mem.simulation_batch` and `shared_mem.fillvalue` must track the current flat batch during an inner fork (set before `reset_inner` and before the pricing pass) and be restored to `B_outer` afterward — in a `finally`, so a mid-fork raise (CUDA OOM, degenerate pricing) cannot leave the state flat-sized and make the *next* chunk fail on shapes instead of the real cause; `fillvalue` is frozen at construction and used as the empty-cat fallback in energy-leg / cash-settle code. Inner-MC pricing must fail loud rather than let `Deal.calculate`'s guard swallow a failure into a scalar-0 mark — inside a fork that silently corrupts the solver's training labels. Both halves are checked: the liability on its flat shape, and every tradable still live in the fork's dependency list on having produced a `tensor_marks` entry (a missing one is indistinguishable from an expired contract, and the solver's `live` mask retires it).

!!! warning "Invariant — `keep_tensor` gates the hedge tradable series"
    `keep_tensor` governs whether `pricing.interpolate` stores `Calc_res['tensor']`; the hedge path sets `Keep_Tensor='Yes'` and harvests those via `tensor_marks`. Removing/altering that store breaks the hedge bundle's tradable series with no error — only missing marks.

## Inner-MC subsystem

`_run_inner_mc_at_t` forks the simulator from each outer-path state at outer step `t`: truncates the grid (`TimeGrid.truncate_to`), windows every deal's `Time_dep` to `{t,t+1}` (`copy_window`), and runs ONE pass at `Batch_Size x Inner_Sub_Batch` flat samples (no partition: peak memory is a function of those two JSON fields, and an over-wide config raises CUDA OOM naming the fork). The pass: `reset_inner` (Sobol), per-process `precalculate` from `outer_buf[key][t]`, `inner_fork_seed` / `reseed_inner_state` for the sufficient statistic, generate, then **publishes each factor's grid as a `ScenarioSource`** — the outer-realized past at `B_outer` followed by the forked rows flattened `(B,B2)→B*B2` — for one real pricing pass on restricted `DealStructure`s. It uses the model-agnostic verb protocol so the loop is uniform across model worlds — see [The process protocol](dependency_system.md#the-process-protocol).

!!! note "Four objects, one query: rows route by block, tenors route by segment"
    The curve read splits into a **query**, **logical scenario storage** and **one physical
    interpolation**, and nothing holds two of those jobs at once.

    - `CurveTensor` — query coordinates. It keeps scenario ROWS (`index`, `index_next`, `alpha`),
      never a flattened `row * n_tenors` offset, because a tenor segment's stride is its own.
    - `ScenarioBlock` / `ScenarioSource` — logical storage. A block is one physical tensor plus
      `first_row` (where it starts in the logical grid) and `batch_index` (which of ITS columns
      supplies each logical column). A fork publishes two blocks; ordinary generation publishes a
      bare tensor and no source at all.
    - `Interpolation` — one physical tensor and whatever its kind derives from it. It knows
      nothing about blocks, logical rows or batch fan-out, and flattens rows against its OWN
      stride. Base valuation, credit Monte Carlo and the outer hedge loop build only this.
    - `SegmentedInterpolation` — a SIBLING, not a subclass: composes leaves over the TENOR axis
      for a `Near_Interpolation` curve.
    - `RoutedInterpolation` — composes strategies over the SCENARIO axis for a fork.

    `build_interpolation` is the single recursive constructor: bare tensor + kind → leaf; bare
    tensor + segment list → segmented; `ScenarioSource` + either → routed, whose per-block children
    it builds by calling itself. So a segmented curve inside a fork is a `RoutedInterpolation` of
    `SegmentedInterpolation`s and needs no special case — the two compositions are orthogonal.

    **Why a fork publishes blocks.** Every realized-past row is identical across the inner draws,
    so joining them into one tensor writes the past out `Inner_Sub_Batch` times: 98% of the
    stuffed buffer at the production operating point, dragging a same-shaped slab of Hermite
    coefficients with it. Each block interpolates at its OWN width and
    `ScenarioBlock.project` takes the RESULT up to the logical width — never the stored tensor,
    which would hand back exactly the memory the split exists to save.

    **Order is load-bearing.** A read is raw (`read_at`), then blended over time, then `combine`d
    (RT scaling, and the segmented tenor select). `combine` and `project` are both linear, so they
    commute with the blend — which is what lets the routed path be the same arithmetic in the same
    order as an unrouted one, and is why the whole thing is bitwise.

    A time-interpolated read reaches `index + 1`, so a row just below a cut reads ACROSS it and
    names two blocks — `route` classifies on where a read ENDS, not where it starts.

    **Invariant — a source is write-once.** Built after every process's `generate` has published,
    and nothing writes into `t_Scenario_Buffer` afterwards. It answers only `shape` / `new` / the
    RT tenor rescale, so a late write fails loud rather than silently materializing the grid it
    exists to avoid.

    Measured on the production walk-forward book (trade 202001, garch, seed 7), like for like:

    | | 1280x64 | 2048x64 | wall @1280 | kB per flat sample |
    | --- | ---: | ---: | ---: | ---: |
    | joined grid | 6.33 GiB | 10.11 GiB | 116.2 s | 80.62 |
    | block sequence | **1.09 GiB** | **1.71 GiB** | 105.9 s | **13.23** |

    `peak_alloc = 0.057 GiB + 13.23 kB · B_flat` (two-point fit; the 4096x64 rung measured
    3.36 GiB against 3.36 predicted). At a 19.6 GiB allocated ceiling that moves max `B_flat`
    from 254 k to 1.55 M — `Batch_Size` 3 977 → 24 208 at `Inner_Sub_Batch` 64, or
    `Inner_Sub_Batch` 199 → 1 210 at `Batch_Size` 1280.

!!! note "Hermite coefficients are built eagerly, and the block split is why that is affordable"
    An intermediate design deferred the `g,c` pair to the gather that read it, on the argument
    that a fork reads ~11% of a block's rows. That was true of the JOINED grid, where the past's
    coefficients cost `B_flat` columns — 1.03 GiB at 1280x64. Blocks store those rows at
    `B_outer`, so eager costs single-digit MB and the deferral stopped paying for itself.
    Measured both ways on the production walk-forward: deferred 193 s / 0.595 GiB, eager
    **179 s / 0.617 GiB** — 7% faster for 22 MiB of a 0.6 GiB peak, because the build COUNT was
    unchanged (3 737 vs 3 769 — objects widen once) while the span bookkeeping added a device
    sync per gather. On deep credit MC eager wins on both axes (132.4 MiB / 0.68 s against
    152.0 / 0.82). A full-horizon fork was the one shape the deferral would still have helped,
    and that switch is retired: a fork prices exactly `{t, t+1}`, which is every field the
    bootstrap reads. A wider fork should be justified by being the right fork, not by the reader
    hedging against one nobody measured.
