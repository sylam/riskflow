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
