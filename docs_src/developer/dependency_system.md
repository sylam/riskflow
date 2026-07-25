# Dependency System

`Context.calculate_dependencies` (`config.py`) is the compiler front end: it discovers every price factor the book touches, wires each to its sub-factor dependencies, orders them, and splits stochastic vs static. Discovery is table-driven — three plain-dict **registries** decide which factors exist; the code around them just walks the tables.

The already-present [Cross Factor](../calibration/cross_factor.md) page covers the composed-spot name-prefix chain and the sim-time buffer publish/consume from the calibration angle; this page is the discovery/ordering intro. Read them together — do not expect either to restate the other.

## The three registries

All three are plain dicts defined inside `calculate_dependencies`. Extend by adding a row.

**`dependant_fields`** — `{factor_type: [(price_factor_key, linked_factor_type), …]}`. The edge generator of the dependency DAG: for a factor it reads `Price Factors[name][price_factor_key]`, and if present builds `Factor(linked_type, check_rate_name(value))` as a dependency, **recursing** if the linked type is itself a `dependant_fields` key. Chains like `CommodityPrice → {Interest_Rate, Forward_Rate, Currency}` and `ReferencePrice → ForwardPrice → FxRate → InterestRate`. There is **no visited-set** — termination relies on the registry being acyclic.

**`nested_fields`** — `{head_type: tail_period_type}`. Governs the positional **name-prefix chain**. Identity for a curve (`InterestRate: InterestRate`); type-switching for a 0D spot (`CommodityPrice`/`EquityPrice`/`FxRate → ObservedBasis`). Consumed by `update_nested_rates`: for each prefix length it registers a period keyed to a single-parent dependency (`head → tail(2) → tail(3) …`). On a genuine type switch it **pops the full-name head key** — so `CommodityPrice.PLATINUM_CME.LME_CME` never exists as a factor; the real pair is `CommodityPrice.PLATINUM_CME` (the spot, carrying the dependant/conditional keys) and `ObservedBasis.PLATINUM_CME.LME_CME` (the tail, depending on the spot). This is what topo-orders `ObservedBasis` **after** its parent, which `BasisLinkedSpotModel.generate` relies on.

**`conditional_fields`** — `{factor_type: lambda(instrument, factor_fields, params) → [Factor, …]}`. Instrument-dependent extra factors (FXVol, Correlation, `<SpotModel>ModelParameters`). Each returned factor is appended as a dependency **and** registered as its own key with an empty dep list. Because the lambda reads the instrument, these are re-evaluated per occurrence.

!!! warning "Invariant — `dependant_fields` must stay acyclic"
    `get_rates` recurses over `dependant_fields` with no visited-set guard. A cyclic entry is unbounded recursion (`RecursionError`), **not** the clean `RuntimeError` that `topological_sort` raises for graph cycles.

!!! warning "Invariant — type-switch head/tail rule"
    For a type-switching `nested_fields` entry (`tail_type != head_type`, e.g. `CommodityPrice → ObservedBasis`), the **head** period must carry the dependant/conditional keys and the tail periods depend positionally on their own prefix. The **virtual full-name head key must be dropped everywhere** — `get_rates` pops it, `get_price_factors` pops its tenor. Code that reads `dependent_factors` must not assume the dotted full name is a key; only the head spot and the `ObservedBasis` tail exist.

!!! warning "Invariant — `conditional_fields` types must not overlap `dependant_fields` keys"
    The conditional branch overwrites a factor's dep list with `[]` (via `update`, not `setdefault`). If a conditional-factor type were also a `dependant_fields` key, its real dependencies would be silently erased. Also: any conditional lambda for a type reachable via a bare-`{}` sentinel factor (`FxRate`, `InterestRate`, `SurvivalProb`) must `getattr`-guard `instrument.options` / `.field` — a raw `AttributeError` there is not caught by `add_rates_for_factor`'s `KeyError` handler.

## The walk

`walk_groups` recurses the deal tree **depth-first, children before parent**, skipping `Ignore=='True'` nodes. Per instrument it runs `instrument.reset(holidays)` and `finalize_dates(...)` (which fill `reval_dates` / `settlement_currencies`), then `get_price_factors`. That iterates the class attribute `instrument.factor_fields` (`{field_name: [factor_type, …]}`); `iter_factors` pulls the field value(s) via `utils.get_fieldname` (handles nested-tuple keys), flattens, and yields `Factor(type, check_rate_name(v))`. For each factor: record its per-deal max date (`max(instrument.get_reval_dates())`) into `dependent_factor_tenors`, and add its rates unless already present (or its type is conditional).

`add_rates_for_factor` calls `get_rates`; on a `KeyError` (missing `Price Factors` block) it logs a warning and — **only for `DiscountRate`** — auto-creates a default block and retries. **Any other type is silently skipped.**

!!! warning "Invariant — the self-heal is `DiscountRate`-only"
    A non-`DiscountRate` factor whose `Price Factors` block is missing is dropped (two log lines), absent from `dependent_factors`, never simulated or valued. If a new derived type needs a default block, extend `add_rates_for_factor` explicitly — do not rely on the silent skip.

!!! warning "Invariant — dates before tenors"
    `get_reval_dates` / `finalize_dates` must run (via `walk_groups`) before tenor collection: the per-factor max date and the reset/settlement sets all come from `instrument.reset()` + `finalize_dates`. A deal whose `reset()` leaves `reval_dates` empty contributes no tenor, and its directly-referenced factors default to `max(reset_dates)`.

The main body seeds base-currency FX first, walks the book, adds report currency (linked to base), then optional CVA `SurvivalProb`, FVA/deflation curves (`add_interest_rate` pins a curve plus all transitive dependents to `reset_dates`).

!!! warning "Invariant — base currency sorts first, stays static"
    Base-currency FX is appended to every other `FxRate`'s dependency list and excluded from the stochastic set (`find_models`). Keep base a static, dependency-of-all-FX anchor.

## Ordering — `topological_sort`

Edges collected as `dependent_factors` (factor → list-of-prerequisites) are `topological_sort`'d (`utils.topological_sort`): a repeated-pass Kahn variant that, **within a pass, emits nodes in dict-insertion order** and moves every node whose edges all point outside the still-unsorted set. Dependencies land first; dependents follow. Cycles raise `RuntimeError`. The input dict is destroyed.

`traverse_dependents` (`utils.py`) fans a factor's tenor out to all transitive dependents — BFS, `seen`-guarded (cycle-safe), and does **not** yield the start node.

!!! warning "Invariant — throwaway graph, cycle behavior"
    `topological_sort` rejects cycles with `RuntimeError` and **destroys its input dict** — pass a rebuilt/throwaway graph. `traverse_dependents` is `seen`-guarded and yields transitive dependents **excluding** the start node; do not rely on the node appearing in its own output.

## Stochastic vs static split — `find_models`

`find_models` walks the topo order, resolving each factor's process via `Model Configuration.search` (`modelfilters` first-match, else `modeldefaults`; subtype-aware). A factor is **stochastic** iff a process was found, `name[0] != Base_Currency`, and `Factor(stoch_proc, name)` is in `Price Models`. Implied models pull an additional static factor and inject a dummy `Price Models[model] = None`. Downstream, `static_factors = set(dependent_factors) - stochastic_factors.values()`.

!!! warning "Invariant — a lost process silently becomes static"
    A factor that resolves to a process but whose `Factor(stoch_proc, name)` is **absent** from `Price Models` is not added to `stochastic_factors`; via the set-difference it falls into `static_factors` and is frozen at its current value. Only a warning (`len(name)>1`) / error (`len==1`) log distinguishes "intended static" from "lost its process."

## RNG ordering {#rng-ordering}

!!! danger "Invariant (WARNING) — insertion order into `dependent_factors` is load-bearing for reproducibility"
    `topological_sort` tie-breaks equal-depth factors by **insertion order**. That order flows through `find_models` (`setdefault`) → `stoch_factors` → `get_cholesky_decomp`'s `process_ofs` → the RNG-substream / correlation column each process reads from `t_random_numbers`. Permuting equal-depth factors — by changing **deal-walk order, `factor_fields` dict order, or `dependant_fields` list order** — yields *different-but-valid* finite-path MC draws: the correlation structure is preserved (cholesky rows permute consistently) but realized results move bit-for-bit. **Preserve the DFS children-before-parent walk, `factor_fields` order, and `dependant_fields` list order, or treat any reordering as a results-changing event.** The HMM regime path adds a second, parallel surface: a separate Sobol `quasi_rng` stream whose batch counter advances across batches and is never reset by `reset()`.

## The process protocol {#the-process-protocol}

Every stochastic process implements a small verb set so the calc/solver loop uniformly across model worlds and never branch on model type. Core verbs (all processes): `num_factors()`, `precalculate(...process_ofs...)` (sets `z_offset`), `correlation_name`, `generate(shared_mem)` (dispatch on `Z.ndim`). Extension verbs, inert no-ops on the base: `reveal_state_at`, `inner_fork_seed`, `outer_reseed`, `reseed_from_path`, `reseed_inner_state`, `diff_state_leaves`, `privileged_factors` / `privileged_layout`, `calc_references` / `link_references`, `copy`. Every model-specific buffer key is owned by the process under the `(factor_key, kind)` convention. The outer/inner MC contract for `generate` is on [Calc Lifecycle](calc_lifecycle.md#valuation-modes).

## Extension recipes {#extension-recipes}

Registries, not functions. Each recipe is "which registry / attribute to touch."

| Add a… | Touch |
| --- | --- |
| **factor type** | If it embeds other factors: a `dependant_fields` row. If its name is a positional chain: a `nested_fields` row (identity = curve, distinct tail = 0D spot) — and the matching head/tail literal at each `calc_factor_code_chain` call site in `instruments.py`. If it is instrument-conditional (vol/correlation/model params): a `conditional_fields` lambda. Register its process in `Model Configuration` (`modeldefaults` / `modelfilters`). |
| **process** | A class in `stochasticprocess.py` implementing the [process protocol](#the-process-protocol) (`num_factors`, `precalculate`, `correlation_name`, `generate`, plus any extension verbs it needs; leave the rest as base no-ops). It is constructed by name via `globals()` dispatch from `Price Models`. Ensure `num_factors() == len(correlation_name[1])`. Pair it with a `*Calibration` class registered in `calibration_config.json` — see [the calibration contract](../calibration/contract.md). |
| **deal** | A `Deal` subclass in `instruments.py` with a `factor_fields` class attribute (field → factor type) so discovery finds its factors, `calc_dependencies` (builds `Factor_dep` via the `get_*` resolver layer), `calc_time_dependency`, and `calculate`. Registered by name via `globals()` dispatch. Declare its JSON via `fields.mapping` (see [Conventions — documentation attributes](conventions.md#documentation-and-doc-generation)). |
| **valuation option** | A field on the `Calculation` block, honored inside the relevant `Calculation` class. New whole-mode = a class dispatched by `run_job` / `construct_calculation`. Do **not** touch `Credit_Monte_Carlo`'s CVA/FVA block ([Change Scope](conventions.md#change-scope)). |

Before writing any of these, search `utils.py` and the package for an existing equivalent — see [Conventions — look before you write](conventions.md#look-before-you-write).
