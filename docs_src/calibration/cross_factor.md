# Cross-Factor Calibration

Some factors depend on a sibling factor's data — the calibration of an
[`InterestRate`](../json/price_models.md) carry curve observed at floating-tenor knots
needs the day-by-day tenor values; a [`ObservedBasis`](../json/price_factors_overview.md)
needs the linked commodity's spot series. The framework handles both with a single
mechanism: **archive column subkeys**.

## The subkey convention

A column header of the form `<archive_name>,<sub_key>` declares that this column's
observations are paired with another factor whose name encodes `<sub_key>`. The framework
auto-discovers the partner archive column at calibration time and pulls it into the same
`data_frame` passed to the calibration class.

Two patterns share this mechanism:

| Primary archive column | Sub-key | Auto-pulled partner |
|---|---|---|
| `InterestRate.PLATINUM_CARRY,PLATINUM_TAU1` | `PLATINUM_TAU1` | `Tenor.PLATINUM_TAU1` |
| `ObservedBasis.LME_CME,PLATINUM_LME` | `PLATINUM_LME` | `CommodityPrice.PLATINUM_LME` |

The matching rule is: for a non-numeric sub-key, look up any other archive entry whose
name ends in `.<sub_key>`. (Numeric sub-keys are interpreted as fixed tenors and need no
partner — they're the standard tenor-grid convention used by IR curves.)

This keeps the dependency declaration in the **archive header**, not in JSON config. A
calibration class receives both the primary column(s) and the partner column(s) in its
`data_frame`, splits them by archive name prefix, and is otherwise self-contained.

## Sim-time wiring

An `ObservedBasis` needs its linked *simulated path* at runtime. That link is carried in the
factor's NAME, not a field: the parent is the name minus its last period (positional, like the
InterestRate parent chain — see the composed-spot section below). The name-prefix nesting registers
the parent and orders it first, so its simulated path (and HMM regime path, if any) is in
`shared_mem.t_Scenario_Buffer` when the basis's `generate()` runs.

Inside `generate()`, the dependent factor reads:

```python
linked_path    = shared_mem.t_Scenario_Buffer[linked_key]              # (T, B)
linked_regimes = shared_mem.t_Scenario_Buffer[(linked_key, 'regimes')] # (T, B), if HMM
```

`linked_key` is derived in `calc_references` from the factor's own name (`factor.name[:-1]`),
resolving the parent's type — `ObservedBasis` if the parent is itself a basis chain, else the one
composable spot type (`FxRate`/`EquityPrice`/`CommodityPrice`) it exists under.

## Auxiliary publish convention

A process that exposes more than its primary path (e.g. an HMM regime path) writes the
extras to `t_Scenario_Buffer` itself inside `generate()`, keyed by `(self.factor_key, kind)`:

```python
def generate(self, shared_mem):
    ...
    shared_mem.t_Scenario_Buffer[(self.factor_key, 'regimes')] = regime_path
    return spot_path
```

The framework hands `self.factor_key` to each process before `precalculate`. Consumers
read using the same key shape — `(linked_key, 'regimes')` — so the publish/consume
convention is symmetric and lives entirely on the process classes (no framework-side
branching for specific process subtypes).

## Deal-side resolution

A deal that prices off a composed spot names it directly in its `Commodity` field — a plain
spot, or a composed name (`PLATINUM_CME.LME_CME`). `get_commodity_rate_factor` is basis-aware
(the positional chain below), so the deal carries **no** basis fields and no linked-name lookup:
both `FloatingEnergyDeal` and `CommodityFutureDeal` just declare `Commodity`.

## Composed spot: the positional name-prefix chain

A composed spot (primary + basis) is carried **in the name**, positionally, exactly like an
interest-rate curve and its basis spread. `InterestRate.USD_SOFR.FUNDING` is the SOFR curve
(period 1) plus its FUNDING basis curve (period 2); `CommodityPrice.PLATINUM_CME.LME_CME` is the
CME spot plus the LME_CME basis observed against it. Bases stack
(`...PLATINUM_CME.LME_CME.SHF` adds a third). A deal referencing the composed name carries **no**
composition fields — the name is the whole story.

**Resolution is positional, not by probing.** One rule, `instruments.calc_factor_code_chain`:
period 1 is the *head* factor; every longer prefix is a *tail* factor named by that whole prefix
chain. For a curve `head == tail == InterestRate` (so `get_interest_factor` is just its identity
case); for a 0D spot `head` is the spot type and `tail = ObservedBasis`. The resulting multi-element
code is summed by `calc_time_grid_spot_rate` (composed spot = primary + Σ bases). A plain
single-period name yields a one-element code — bit-identical to the pre-chain lookup. Repo / carry /
dividend / currency lookups take the head (`fieldname[:1]`), since those live on the primary.

**Discovery is the existing curve mechanism, generalized as data.** `nested_fields` is a
`{head type: tail type}` map (identity for `InterestRate`, `ObservedBasis` for the 0D spot types);
`update_nested_rates` walks the name prefixes, giving period 1 the head type and the tail periods
the mapped type, each linked to its parent prefix so `topological_sort` orders them. Because
`CommodityPrice` (unlike `InterestRate`) also carries `dependant_fields`, `get_rates` pulls the
chain first and then applies the dependant/conditional fields to the head; the type-switched
full-name key (`Factor('CommodityPrice', ('PLATINUM_CME','LME_CME'))`) is never a real factor, so
it is dropped from both `rates_to_add` and the tenor map.

**No `Observed_Factor` field, no `Implied_Basis` field.** The link is the name prefix, so
`BasisLinkedSpotModel.calc_references` derives its linked parent as `factor.name[:-1]` (its type
resolved from `all_factors`), and both deal types just name the composed spot in `Commodity`. The
`dependant_fields['ObservedBasis']` entry, the `CommodityFutureDeal` `Implied_Basis` field, the
`CME_FLAT` identity basis (a composed name needs no `+0` factor) and the linked getters all go
away with the redundancy.

**Collision note.** The positional rule reinterprets any multi-period 0D factor name as
primary + basis chain, so a genuine multi-part `CommodityPrice`/`EquityPrice`/`FxRate` name would
now be split. A sweep of the shipped configs found none.

## Design notes — when calibrations need sibling state

The mechanisms above (archive subkeys / name-prefix pulls, and the name-prefix sim chain) handle
the cases where a factor needs sibling **archive data** (calibration-time) or sibling **simulated
path** (sim-time). Neither lets a calibration class see another factor's *calibrated parameters*.
That's deliberate: the framework's calibration loop runs each class self-contained, with
inputs limited to its `data_frame` slice plus its own `param` from
`calibration_config.json`. No reach-through to `Config.params['Price Models']`.

This rule has a real cost. Some calibrations would, in principle, like to use a sibling's
fitted state. The canonical example is `BasisLinkedSpotCalibration`: it partitions η by
regime to fit `Sigma_By_State`, and the *correct* partition is "the LME HMM's posterior
state on this day." Currently the calibration uses a rolling-vol tercile of `ΔLME` as a
proxy — correlated with the HMM partition but not identical. The validation tests
([Test D](contract.md#output-calibrationinfo)) confirm the proxy is empirically close
(0.6–0.8% recovery error) but it's not the partition the simulator runtime uses.

We considered three ways to fix this:

1. **Pass the LME HMM params via `calibration_config.json`** — manual sync, brittle.
2. **Run an inline HMM fit on `dlme` inside the basis calibration** — self-contained but
   wasteful and may converge to a different local optimum than the production LME HMM.
3. **Framework-level: pass `params['Price Models']` into `calibrate(...)`** — semantically
   right (basis can run Viterbi against the actual production HMM), but breaks the
   self-containment rule.

**Decision (2026-05): defer (3) until the next calibration class needs sibling state.**
The current proxy is empirically validated; closing the theoretical gap is not blocking
any failing test. When (3) eventually lands, it should come with a topo-sorted calibrate
loop and an explicit `DEPENDS_ON` declaration on the calibration class — so dependency
ordering is part of the contract, not an implicit alphabetical sort. That's the right
moment to also drop the rolling-vol tercile in
[`BasisLinkedSpotCalibration`](../json/price_models/basislinkedspotmodel.md) in favour of
HMM-Viterbi-partitioned σ.

Until then: **stick with self-contained calibrations**. If you find yourself wanting
sibling fitted state, add a punchlist entry rather than reaching across — the cumulative
cost of cross-talking calibrations is much higher than the cost of one well-correlated
proxy.
