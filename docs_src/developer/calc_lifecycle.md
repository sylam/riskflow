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
- **Streaming** (`Solver.DiffV2_Streaming_Batches='Yes'`, the adopted production mode): a bundle per batch, handed to a persistent solver as it is built — `StreamingSolve.warmup` on batch 1 (which locks the frame), `step` on each later batch, `finish` on a held-out final batch. Fork width follows `Batch_Size`, not the whole simulation. The end-to-end reproduction gate is `tb_wf_smoke_gate.sh` (trade 202001, 512x5 batches, seed 7); it replaced the non-streaming `--batch 2048` anchor, which no longer fits single-pass.

!!! note "Queued memory work — row-restricted Hermite coefficients (MEASURED, not merged)"
    `make_curve_tensor` builds the Hermite `g,c` pair for every `(scen x n_tenors)` row of a curve
    block at cache-population time. Measured at the recommended operating point (1280x64,
    `tb_hermite_census.py`): **13.32 GiB** of `g,c` allocated over a run, largest single entry
    **2.23 GiB**, and consumers gather a mean **2.61%** of the rows — so **~13 GiB is recoverable**
    and ~2.2 GiB of the 9.26 GiB peak is the largest entry's unused rows. Deferring the build to
    first gather saves **0.002 GiB** (a null: the un-gathered entries are tiny), so the fix is row
    restriction, not laziness. Projected: brings `2048x64` back to ~19 GiB on the production world
    (from the 23.5 GiB that OOMs) and buys ~+308 outer paths at 1280. NOT implemented: it needs
    index translation in `Interpolation.eval`'s gather (`i00 = t_index + i1` remapped into the
    restricted block) inside the curve stack `base_valuation` and `credit_monte_carlo` also price
    through — a maintainer decision, with the numbers above as the case.

!!! warning "Invariant — `Z.ndim` dispatch (outer vs inner MC)"
    `generate()` must handle both outer (`Z.ndim==2`, `(T,B)`) and inner (`Z.ndim==3`, `(T,B,B2)`) modes, with the per-outer-path initial state broadcast on the **middle** axis in inner mode. This is what lets one process instance serve both loops.

!!! warning "Invariant — inner-MC batch state + fail-loud pricing"
    `shared_mem.simulation_batch` and `shared_mem.fillvalue` must track the current flat batch during an inner fork (set before `reset_inner` and before the pricing pass) and be restored to `B_outer` afterward — in a `finally`, so a mid-fork raise (CUDA OOM, degenerate pricing) cannot leave the state flat-sized and make the *next* chunk fail on shapes instead of the real cause; `fillvalue` is frozen at construction and used as the empty-cat fallback in energy-leg / cash-settle code. Inner-MC liability pricing must fail loud on a degenerate shape rather than let `Deal.calculate`'s guard swallow an OOM into a scalar-0 mark — inside a fork that silently corrupts the solver's training labels.

!!! warning "Invariant — `keep_tensor` gates the hedge tradable series"
    `keep_tensor` governs whether `pricing.interpolate` stores `Calc_res['tensor']`; the hedge path sets `Keep_Tensor='Yes'` and harvests those via `tensor_marks`. Removing/altering that store breaks the hedge bundle's tradable series with no error — only missing marks.

## Inner-MC subsystem

`_run_inner_mc_at_t` forks the simulator from each outer-path state at outer step `t`: truncates the grid (`TimeGrid.truncate_to`), optionally windows to `{t,t+1}` (`copy_window`) for the one-step diff-ML bootstrap, and runs ONE pass at `Batch_Size x Inner_Sub_Batch` flat samples (no partition: peak memory is a function of those two JSON fields, and an over-wide config raises CUDA OOM naming the fork). The pass: `reset_inner` (Sobol), per-process `precalculate` from `outer_buf[key][t]`, `inner_fork_seed` / `reseed_inner_state` for the sufficient statistic, generate, then stuffs the outer-realized past (broadcast across `B_inner`) and flattens `(B,B2)→B*B2` for one real pricing pass on restricted `DealStructure`s. It uses the model-agnostic verb protocol so the loop is uniform across model worlds — see [The process protocol](dependency_system.md#the-process-protocol).
