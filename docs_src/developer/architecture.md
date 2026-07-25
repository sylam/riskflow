# Architecture

riskflow is a **financial virtual machine**. A job is a program; the engine compiles it, then executes it against Monte-Carlo scenarios.

| VM concept | riskflow |
| --- | --- |
| program | the job JSON — `Calculation`, `Deals`, market data (`Price Factors`, `Price Models`, `Correlations`) |
| loader | `Context.load_json` |
| compile | `Context.calculate_dependencies` (discover + order factors) + each process's `precalculate` |
| instructions | `StochasticProcess.generate` (per factor) and `Deal.calculate` / `pricing.*` (per deal) |
| execute | the per-batch generate loop in `Calculation.execute` |
| registers / heap | `shared_mem.t_Scenario_Buffer` (simulated paths), `t_Static_Buffer` (static leaves) |
| memoized eval cache | `shared_mem.t_Buffer` |

The public surface (`Context`, `load_json`, `run_job`, the three calculation objects, output shape) is documented once in [API Overview](../api_overview.md); this section adds the **internal** view the public page omits.

## Reading order

This section reads Architecture → [Calc Lifecycle](calc_lifecycle.md) → [Dependency System](dependency_system.md) → [Resolver Layer](resolver_layer.md) → [Conventions](conventions.md). (mkdocs sorts the nav alphabetically; follow the prose order, not the sidebar.)

## The spine: one `Factor` keys everything

`Factor = namedtuple('Factor', 'type name')` (`utils.Factor`) — `type` a `str`, `name` a tuple of uppercase strings — is the identity used by **every** dict in the pipeline: the discovery graph, `stochastic_factors` / `static_factors`, `all_factors`, `all_tenors`, and the runtime `t_Scenario_Buffer` / `t_Static_Buffer`. One key across many dicts is what lets the layers compose without a translation table.

Names stay **atomic tuples** everywhere. A dotted market-data name (`"PLATINUM_CME.LME_CME"`) is split into `('PLATINUM_CME','LME_CME')` at exactly one boundary — `utils.check_rate_name` / `check_tuple_name` — inside the [Resolver Layer](resolver_layer.md). Deal code and processes carry the whole `Factor` by reference and never index into `name`.

!!! warning "Invariant — the `Factor` identity"
    `Factor = (type:str, name:tuple[str])`. The name is atomic; string↔tuple conversion happens **only** in `check_rate_name` / `check_tuple_name` at the resolver boundary. One `Factor` value keys four dict families identically — the offset maps (`static_factors`/`stoch_factors`), the object graph (`all_factors`), the tenor payloads (`all_tenors`), and the runtime buffers (`t_Static_Buffer`/`t_Scenario_Buffer`). Buffers are populated under the process's own `factor_key`. Keep the name atomic and this holds; split it early and it breaks silently.

## The three phases

**1. Compile — `calculate_dependencies`.** Walk the deal tree, discover every price factor a deal touches, wire each factor to its sub-factors, collect the max date each is needed to, topologically order them, and split into **stochastic** (simulated) vs **static** (frozen leaf). Discovery is driven by three plain-dict registries, not by branching code — see [Dependency System](dependency_system.md).

**2. Compile — `_build_factor_state` + `precalculate`.** Construct the factor objects, mint AAD leaves (`torch.tensor(..., requires_grad=…)`), build each stochastic process, assemble the correlation matrix and its cholesky, and assign each process its RNG-substream offset (`process_ofs`). Detailed in [Calc Lifecycle](calc_lifecycle.md).

**3. Execute — the generate loop.** Per simulation batch: draw the correlated random block, then iterate `stoch_factors` in topological order publishing each path into `t_Scenario_Buffer` as it is produced (so a linked factor reads its parent's already-published path), then price the deal tree, accumulating MTM.

## Why registries, not functions

Extension points are **data**, not control flow. Adding a factor type, a process, a deal, or a valuation option means adding a row to a registry (or a class attribute the engine iterates), never editing a dispatcher. The dispatchers themselves are `globals()`-keyed on class name. This is the house pattern; the mechanics live in [Conventions](conventions.md) and the extension recipes in [Dependency System](dependency_system.md#extension-recipes).

## Where valuation modes diverge

`run_job` (`Context.run_job`) is a 3-way branch on `Calculation['Object']`: `Base_Valuation` (single-date static reval), `Credit_Monte_Carlo` (the full scenario engine — CVA/FVA/exposure), and `Hedge_Monte_Carlo` (inherits the CMC engine but harvests raw marks for the diff-ML hedge solver and forks an inner Monte-Carlo). All three share the compile phases above; they differ only in what the execute phase does with the priced tensors. See [Calc Lifecycle](calc_lifecycle.md#valuation-modes).

## Design direction

The long-term shape is a stateless streaming compute: JSON events in, stateless compute, JSON results out, with state living outside the process. Everything above is consistent with that — the engine holds no cross-job state beyond the market-data cache, and the job JSON is the whole contract. See [Conventions — JSON is the contract](conventions.md#json-is-the-contract).
