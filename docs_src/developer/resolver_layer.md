# Resolver Layer

The generic `get_*` layer in `instruments.py` is the **only** place a factor name is decomposed. A deal's `calc_dependencies` runs here to turn `Factor` keys into the flat **code** tuples that pricing consumes off the runtime buffers. The [Cross Factor](../calibration/cross_factor.md) calibration page walks the positional name-prefix chain and the sim-time summation from the calibration side; this page is the resolver-internals view.

## Name normalization

`utils.check_rate_name` (dotted string or tuple → uppercase tuple) and `check_tuple_name` (inverse, `type.name0.name1…`) are the sole string↔tuple bridge. `check_tuple_name` also builds every fail-loud error string and every `Price Factors` / `Price Models` lookup key. `BASIS_COMPOSABLE_TYPES = ('FxRate','EquityPrice','CommodityPrice')` marks the 0D spot types whose name may carry a composed `primary + ObservedBasis…` chain.

## The code tuple

`calc_factor_index(field, static_offsets, stochastic_offsets, all_tenors={})` is the atom. It returns `tuple([stoch_bool, field(=Factor), subtype] + all_tenors.get(field, []))`, else raises `Exception(check_tuple_name(field))`. The slots are named constants in `utils.py`:

| index | constant | meaning |
| --- | --- | --- |
| 0 | `FACTOR_INDEX_Stoch` | `True` stochastic / `False` static |
| 1 | `FACTOR_INDEX_Offset` | the **`Factor` key** (legacy misnomer — not an int) |
| 2 | `FACTOR_INDEX_SubType` | factor subtype |
| 3 | `Tenor_Index` | `CurveTenor` payload (curves only) |
| 4 | `Daycount` | daycount closure (curves only) |

!!! warning "Invariant — the code-tuple layout is fixed"
    `[0]` stoch bool, `[1]` a single `Factor` key, `[2]` subtype, `[3+]` tenor payload from `all_tenors`. `calc_factor_index` is the sole producer. The stable hashable identity is `x[:2] = (stoch, Factor)`; indices `3+` (CurveTenor / daycount closure) are unhashable.

## The positional chain — `calc_factor_code_chain`

`calc_factor_code_chain(head, tail, fieldname, …)` resolves a prefix chain: `[calc_factor_index(Factor(head if x==1 else tail, fieldname[:x]), …) for x in 1..len]`. Period 1 = head type on `name[:1]`; each longer prefix = tail type on `name[:x]`. A single-element name yields a single-element code — the **bit-identical legacy path**. Four call sites hardcode the head/tail pair: `get_interest_factor` (`head==tail=='InterestRate'`, the spread chain), and `get_fxrate_factor` / `get_commodity_rate_factor` / `get_equity_rate_factor` (`(<Spot>, 'ObservedBasis')`).

!!! warning "Invariant — the positional chain"
    Period 1 = head type on `name[:1]`; prefix `k>1` = tail type on `name[:k]`. Curve ⇒ `head==tail` (spread chain); 0D spot ⇒ `tail=ObservedBasis`. These four hardcoded head/tail pairs are a second copy of `config.nested_fields` — discovery and resolution must agree, and the virtual full-name head key for a type-switched spot (`CommodityPrice.X.Y`) must never be looked up (discovery intentionally drops it).

## Object-graph hops — `get_factor_component`

`get_factor_component(componentname, all_factors)` implements the **primary-spot** rule: `primary = name[:1] if type in BASIS_COMPOSABLE_TYPES else name`, returning the underlying factor **object**. Repo/carry/currency read off it (`get_equity_currency_factor`, `get_equity_zero_rate_factor`, `get_commodity_zero_rate_factor` → `.get_repo_curve_name()` / `.get_currency()` → `get_interest_factor`). So repo/carry/dividend/currency live on the **ultimate** primary `name[:1]`.

!!! warning "Invariant — two distinct parent rules, both correct"
    Object lookups (repo/carry/dividend/currency) resolve off the **ultimate** primary `name[:1]` via `get_factor_component`. The `ObservedBasis` process links off its **immediate** parent `name[:-1]` (`BasisLinkedSpotModel.calc_references`). Both are correct for their jobs; do not conflate them.

## Getter taxonomy

The two trailing args tell you the axis:

- takes `all_tenors` → **curve-shaped**, needs a tenor payload appended (interest / forward-rate / zero / discount / survival / inflation / dividend / all vol getters).
- takes `all_factors` → must inspect a factor **object** to hop to a linked factor (repo, currency, recovery, spot value, model params).
- takes neither → pure 0D **spot** code, no tenor (`get_fxrate_factor`, `get_commodity_rate_factor`, `get_equity_rate_factor`, `get_price_index_factor`).

## The list-at-index-1 idiom — `get_spot_model_params_factor`

`get_spot_model_params_factor(spot_model, name, all_factors, …)`: `spot_model=='None'` → `None` (GBM); unknown → `ValueError`; switch-on-but-absent → `KeyError`. Otherwise it returns a single-element code whose **index 1 is a *list* of per-parameter sub-factors** and whose **index 2 (subtype slot) is the model name** — the same shape as the SVI/Skew vol code in `get_equity_price_vol_factor`. These list-shaped codes never take the generic `x[:2]` cache path.

!!! warning "Invariant — `t_Buffer` cache-key discipline"
    Cache keys slice `x[:2] = (stoch, Factor)` and **exclude** indices `3+`. That requires index 1 be a single `Factor`. SVI/HN codes carry a **list** at index 1 and are consumed only by dedicated vol/HN paths that tuple-flatten the list (`calc_time_grid_vol_rate`) or read `t_Static_Buffer` directly. Do not route a list-shaped code through the generic `(stoch, Factor)` key path.

## `field_index` == `Factor_dep`

Each deal's `calc_dependencies` returns a `field_index` dict of these codes; `add_deal_to_structure` stores it **verbatim** as `DealDataType.Factor_dep`, and `generate` reads `deal_data.Factor_dep['Commodity']` etc. unchanged.

!!! warning "Invariant — `field_index` is stored and consumed verbatim"
    The dict a deal's `calc_dependencies` returns **is** `DealDataType.Factor_dep`, stored and consumed unchanged by `generate`.

## Pricing consumption

- `calc_time_grid_spot_rate(rate, …)`: spot = **sum** over code elements of the gathered component (stochastic via `gather_scenario_interp`, static via `reshape(1,-1)`), reading `t_*_Buffer[r[FACTOR_INDEX_Offset]]`. Cache key `('spot', tuple(tuple(r[:2]) for r in rate), time_hash)`.
- `calc_time_grid_curve_rate(code, …, n_batch_dims=1)`: builds a `TensorBlock`; `make_curve_tensor` collapses extra trailing batch axes (inner-MC `(B,B2)→B*B2`) when `n_batch_dims>1`, keeping the interp stack single-trailing-batch.
- `TensorBlock.gather_weighted_curve`: sums `zip(curve_tensors, code)` (parent curve + basis spreads); the risk-neutral branch keys on `not curve_component[FACTOR_INDEX_Stoch]`.

!!! warning "Invariant — composed value is a sum of gathered components"
    Spot = primary + basis; curve = parent + spreads. The sum is order-agnostic; ordering (primary first) matters only for the single-element bit-identity guarantee and the risk-neutral static-branch check.

!!! warning "Invariant — fail-loud, end to end"
    An unknown/missing factor raises `Exception` / `KeyError` / `ValueError` naming it via `check_tuple_name`. There is no silent GBM/zero fallback anywhere in the resolver or the pricer.

## The compressed contract

```
JSON string
  → check_rate_name → Factor(name tuple)
  → [discovery: nested/dependant/conditional + topo]        (Dependency System)
  → [construct: stoch/static/all_factors/all_tenors]        (Calc Lifecycle)
  → [resolve: get_* → code [stoch, Factor, subtype, *tenor]] (this page)
  → field_index == Factor_dep
  → [generate: t_Scenario_Buffer[Factor]]
  → [price: calc_time_grid_*_rate sums components, caches in t_Buffer by (stoch,Factor)]
```
