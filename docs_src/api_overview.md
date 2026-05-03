# API

Everything in RiskFlow is based off a *Context*. All calculations are constructed with reference to one.
Note that the interest rate curves start one day from now i.e. $1/365\approx 0.00274$. No interest rate
curve can start at time 0 (although the rate at time 0 is flat extrapolated from the first timepoint).

## The Context

A `Context` holds the loaded JSON config, market data, deal hierarchy, and calendar metadata. All
calculations read from it. Once a context is loaded, you can reuse it for multiple calculations
(e.g. revalue, then run a Monte Carlo simulation, then run a hedge optimisation) without re-parsing
the JSON.

```python
import riskflow as rf

cx = rf.Context()
cx.load_json('fxfwd.json')
```

A context can hold *multiple* loaded configurations — each `load_json` call adds another
`Config` to `cx.config_cache`, and the most recently loaded one is set as `cx.current_cfg`. All
calculation methods read from `cx.current_cfg`, so to switch which configuration is active you
re-load (or assign `cx.current_cfg` directly to a previously-cached `Config`).

The active configuration exposes:

- `cx.current_cfg.params` — the merged market-data dictionary (`'System Parameters'`,
  `'Price Factors'`, `'Price Models'`, `'Correlations'`, etc.). Mutate this directly to override
  loaded market data.
- `cx.current_cfg.deals` — the deal hierarchy (`'Calculation'`, `'Deals'`, `'Attributes'`).
- `cx.holiday_cfg_cache` — calendar definitions parsed from referenced XML calendar files (this
  one lives on the context itself, since calendars are shared across configurations).

## Running a calculation

Three calculation types are supported, each with both an explicit method and a JSON-driven
dispatcher:

| Method | JSON `Calculation.Object` | Purpose |
|---|---|---|
| `cx.Base_Valuation(overrides)` | `BaseValuation` | Single-point MTM revaluation |
| `cx.Credit_Monte_Carlo(overrides)` | `CreditMonteCarlo` | Path-dependent simulation (CVA / FVA / PFE) |
| `cx.Hedge_Monte_Carlo(overrides)` | `HedgeMonteCarlo` | Same simulation engine, used as the env for an RL hedger |
| `cx.run_job(overrides)` | (any of the above) | Dispatches based on the loaded JSON's `Calculation.Object` |

Use `run_job()` when the JSON itself fully specifies which calculation to run:

```python
cx = rf.Context()
cx.load_json('BaseValuation.Test1.json')
calc, out = cx.run_job(overrides={})
```

Use the explicit methods when you want to run a different calculation than the JSON specifies (for
example, running a Credit Monte Carlo against a JSON originally written for Base Valuation).

Each method returns a `(calc, out)` tuple — the calculation object (useful for inspecting state
post-run) and the output dictionary.

### Base Valuation

A single-point theoretical price. Cheapest of the three calculations.

```python
calc, out = cx.Base_Valuation(overrides={'Run_Date': '2024-08-01', 'Currency': 'USD'})
out['Results']['mtm']
```

returns a pandas DataFrame:

```
Test	NettingCollateralSet	0.0
	341	FXNonDeliverableForward	-343.123474121
```

So the market value at 1 August 2024 of the forward is -343 USD. The output structure depends on
the deals loaded and any tags defined; `out['Results'].keys()` enumerates everything that's
available.

When `Greeks` is set on the calculation, additional dataframes appear in `out['Results']` for
first-order (and optionally second-order) sensitivities by risk factor.

### Credit Monte Carlo

Monte Carlo simulation over a configurable time grid. Used for path-dependent metrics like
exposure profiles, CVA, FVA.

```python
params = {
    'Time_grid': '0d 2d 1w(1w) 3m(1m) 2y(3m)',
    'Run_Date': '2024-08-01',
    'Currency': 'ZAR',
    'Simulation_Batches': 2,
    'Batch_Size': 512,
    'Random_Seed': 6126,
    'Calc_Scenarios': 'No',
    'Generate_Cashflows': 'Yes',
    'Dynamic_Scenario_Dates': 'Yes',
}
calc, out = cx.Credit_Monte_Carlo(overrides=params)
```

```python
out['Results']['exposure_profile']
```

```
             EE         PFE
2024-08-01    0.000000    0.000000
2024-08-03    1.809047    0.000000
2024-08-08   25.414378  201.102859
...
```

**EE** is the Expected Exposure, **PFE** is the Peak Exposure (95% by default). The result is a
pandas DataFrame and so can be plotted via `.plot()`.

For multi-GPU machines, `Credit_Monte_Carlo(runparallel=True)` shards the simulation across all
visible CUDA devices and merges the results.

### Hedge Monte Carlo

A specialisation of Credit Monte Carlo wired into a TorchRL training loop. The same scenario engine
generates trajectories which are consumed by a structured policy that learns to hedge a portfolio
of liabilities by trading a configured set of futures or other instruments. See the
[Hedging_Problem](../json/index.md#calculation) JSON section for the configuration contract.

```python
calc, out = cx.Hedge_Monte_Carlo(overrides={'Random_Seed': 42})
out['Results'].keys()
```

When `Execution_Mode` is `optimize_policy`, the optimiser trains the policy in-process and
`out['Results']` contains the trained policy artifact alongside diagnostic metrics. When
`Execution_Mode` is `simulate_only`, only the scenario bundle is computed (useful for offline
analysis).

## Overrides

Every calculation method accepts an `overrides` dict that updates the JSON's `Calculation` section
just before execution. Common overrides:

- `Run_Date` / `Base_Date` — switch the valuation date without editing the JSON
- `Currency` — change the reporting currency
- `Random_Seed`, `Batch_Size`, `Simulation_Batches` — control Monte Carlo reproducibility and size
- `Greeks` — enable sensitivities (`'No'` / `'Factors'` / `'All'`)
- `Time_Grid` — re-shape the simulation time grid (Credit / Hedge Monte Carlo)

Overrides are merged shallowly into the loaded `Calculation` object, so nested fields (e.g.
`Hedging_Problem.Optimizer.Epochs`) need to be passed as a complete sub-dict if you want to change
just one entry.

## Inspecting and modifying loaded data

`cx.current_cfg.params` exposes the full market-data tree of the active configuration as nested
dicts. To override a single price factor's spot before running a calculation:

```python
cx.current_cfg.params['Price Factors']['EquityPrice.AAPL']['Spot'] = 200.0
calc, out = cx.run_job(overrides={'Run_Date': '2024-08-01'})
```

Similarly, `cx.current_cfg.deals['Deals']['Children']` is a list of deal definitions you can
append to or mutate. Calling `cx.run_job()` after mutating either of these picks up the changes —
there's no implicit cache that needs to be invalidated.

## Output structure

The return value of every calculation method is `(calc, out)` where `out` is a dict with three
top-level keys:

- `'Netting'` — the internal `DealStructure` tree. Useful for developers walking the hierarchy.
- `'Stats'` — a dict of timing and counter statistics from the run.
- `'Results'` — the user-facing dataframes / arrays. Keys vary by calculation type; see the
  [Output](output.md) page.

---
