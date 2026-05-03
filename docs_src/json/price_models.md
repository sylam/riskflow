# Price Models

Each stochastic process has a JSON entry under `Price Models` that specifies the parameters used
to simulate the corresponding price factor. The key follows the pattern `<ModelName>.<FactorName>`:

```json
{
  "Price Models": {
    "GBMAssetPriceModel.ZAR": {
      "Vol": 0.171569979672,
      "Drift": -0.0436796256753
    }
  }
}
```

The `<FactorName>` portion must match the underlying name in the corresponding entry under
`Price Factors` (e.g. `FxRate.ZAR` here). The model name itself is consumed by the simulator's
process registry — see [Theory](../theory/asset_pricing.md) for the supported models.

A factor can have multiple candidate models registered in [Model Configuration](model_configuration.md);
the active one for any given factor is selected by the model-filters and process-factor maps in
that section. Behaviour when two factors share a name (e.g. an `EquityPrice.ZAR` defined alongside
`FxRate.ZAR`) is undefined — internally vanilla python dictionaries are used to track factors, so
all factor names must be unique.

## How parameters are obtained

Parameters can come from any of:

- **Historical calibration** — a separate offline process fits the model to a long timeseries
  (typically maximum-likelihood or method-of-moments). Used for real-world (P-measure) simulations
  for PFE, capital, etc.
- **Implied / market-data calibration** — the [bootstrappers](bootstrapper_configuration.md) fit
  parameters to current market quotes (ATM volatilities, swap rates, futures option prices). Used
  for risk-neutral (Q-measure) simulations for CVA / FVA.
- **Manual override** — for testing or stress runs, parameters can be edited directly in the JSON.

The detailed parameter schema for each model is generated automatically from the codebase below.
