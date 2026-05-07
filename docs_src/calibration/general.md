# Historical Calibration

Calibration is RiskFlow's process for fitting [price model](../json/price_models.md) parameters
from historical timeseries. Distinct from
[bootstrapping](../bootstrapping/general.md), which fits to current market quotes for
risk-neutral simulation, calibration produces real-world (P-measure) parameters used in PFE,
capital, and policy-training simulations.

## Inputs

A calibration run consumes three inputs:

- **A historical archive** — a wide CSV table (one row per business day, columns named after
  the factor they observe). Loaded once via `MarketDataArchiveFile` in
  [calibration_config.json](config.md).
- **A static MarketData JSON** — declares the factors and their stochastic processes via
  `Model Configuration` (see [Model Configuration](../json/model_configuration.md)). Price
  Factors typically remain empty at this layer; live values are merged in at job time by
  the calculation cfg's `ExplicitMarketData`.
- **`calibration_config.json`** — declares which method to apply to each model class and where
  the archive lives.

## Output

`Config.calibrate_factors` populates two sections of the MarketData JSON in memory:

- `Price Models` — one entry per calibrated factor, keyed `<ModelName>.<FactorName>`,
  containing the parameters the simulator's process class reads at `precalculate` time.
- `Correlations` — pairwise innovation correlations across all calibrated factors,
  auto-discovered from the per-factor residual columns (see
  [Calibration Class Contract](contract.md)). Entries below the configured cutoff
  (typically 0.10) are dropped.

A subsequent `Config.write_marketdata_json` call serialises the result to disk. The
calibrated MarketData is the deliverable consumed by downstream simulations.

## Workflow

```python
from riskflow.config import Config
import pandas as pd

aa = Config()
aa.parse_json('./data/MarketDataRF_platinum.json')          # static source-of-truth
aa.parse_json('./artifacts/calibration_config.json')        # methods + archive

factors = aa.fetch_all_calibration_factors()                # discovered from archive
keep = {**factors['present'], **factors['absent']}          # routed via Model Configuration

start = aa.archive.index.min()
end   = aa.archive.index.max()
aa.calibrate_factors(start, end, keep, smooth=0.0, correlation_cuttoff=0.1)
aa.write_marketdata_json('./data/MarketDataRF_platinum_calibrated.json')
```

The factor list comes from intersecting (a) factors in the static `Price Factors` block,
(b) factors discovered in the archive, and (c) routes available in `Model Configuration`.
Factors absent from one of these are simply skipped — calibration is opt-in by data
availability.

## Iteration order

Factors are calibrated in alphabetical order of their fully-qualified name
(`<Type>.<Name>`). A factor whose calibration class needs data from a sibling factor's
archive should rely on the archive-pull mechanism described in
[Cross-Factor Calibration](cross_factor.md), not on iteration order — the data flows in
through `data_frame`, not through any cross-class lookup.

If a calibration must use another factor's already-fitted *parameters* (rare), the factor
naming convention happens to put commodity dependencies (e.g. `CommodityPrice.X`) before
their basis derivatives (e.g. `CommodityBasis.X_Y`) alphabetically, but this is not a
guarantee the framework provides. Prefer self-contained calibration classes.
