# Cross-Factor Calibration

Some factors depend on a sibling factor's data — the calibration of an
[`InterestRate`](../json/price_models.md) carry curve observed at floating-tenor knots
needs the day-by-day tenor values; a [`CommodityBasis`](../json/price_factors_overview.md)
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
| `CommodityBasis.LME_CME,PLATINUM_LME` | `PLATINUM_LME` | `CommodityPrice.PLATINUM_LME` |

The matching rule is: for a non-numeric sub-key, look up any other archive entry whose
name ends in `.<sub_key>`. (Numeric sub-keys are interpreted as fixed tenors and need no
partner — they're the standard tenor-grid convention used by IR curves.)

This keeps the dependency declaration in the **archive header**, not in JSON config. A
calibration class receives both the primary column(s) and the partner column(s) in its
`data_frame`, splits them by archive name prefix, and is otherwise self-contained.

## Sim-time wiring

For factors that need a sibling's *simulated path* at runtime (not just archive data at
calibration time), declare the dependency in
[`config.py`](../json/model_configuration.md)'s `dependant_fields`:

```python
dependant_fields = {
    ...
    'CommodityBasis': [('Observed_Commodity', 'CommodityPrice')],
}
```

This says: *a CommodityBasis factor depends on the CommodityPrice factor named in its
own `Observed_Commodity` field*. The framework's
[`calculate_dependencies`](../json/model_configuration.md) walker uses this to:

1. Pull the linked CommodityPrice factor into the simulation set whenever a
   CommodityBasis is requested.
2. Generate the linked factor first so its simulated path and (if applicable) HMM regime
   path are available in `shared_mem.t_Scenario_Buffer` when the dependent factor's
   `generate()` runs.

Inside `generate()`, the dependent factor reads:

```python
linked_path    = shared_mem.t_Scenario_Buffer[linked_key]              # (T, B)
linked_regimes = shared_mem.t_Scenario_Buffer[(linked_key, 'regimes')] # (T, B), if HMM
```

`linked_key` is the `utils.Factor` form of the linked factor's name; resolve it from
`self.factor.param[<dep_field>]` (e.g. `self.factor.param['Observed_Commodity']`).

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

When a deal references a factor that itself has a linked dependency (e.g.
`CommodityFutureDeal` references `Implied_Basis: PLATINUM_BASIS`, and the basis carries
its own `Observed_Commodity`), the deal does **not** declare the linked factor in its
`factor_fields`. Instead, the deal's `calc_dependencies` looks up the linked name through
a small helper:

```python
basis_field    = utils.check_rate_name(self.field['Implied_Basis'])
observed_field = utils.check_rate_name(get_observed_commodity_name(basis_field, all_factors))
```

`get_observed_commodity_name` reads the basis Price Factor's `Observed_Commodity` field
and returns the linked CommodityPrice's name. This mirrors the existing pattern used by
`get_inflation_index_name`, `get_interest_rate_currency`, and similar two-tier resolvers
in `instruments.py`.

The benefit: the deal's JSON definition only needs the *direct* linkage
(`Implied_Basis`); the framework's dependency walker pulls the rest in via the
`dependant_fields` declaration.

## Design notes — when calibrations need sibling state

The two mechanisms above (archive subkeys, `dependant_fields`) handle the cases where a
factor needs sibling **archive data** (calibration-time) or sibling **simulated path**
(sim-time). Neither lets a calibration class see another factor's *calibrated parameters*.
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
