#  Price Models

Parameters for the stochastic processes used to simulate its corresponding price factor is to be
provided here e.g. for ZAR,

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

Note that the name of the model needs to match the price factor. This can become confusing if, 
e.g. you have a stock called ZAR (i.e. an EquityPrice.ZAR defined) and the ZAR FX rate defined 
(as FxRate.ZAR). Behaviour when this occurs is undefined. It is up to the user to make sure that 
all price factors have a unique name (internally, vanilla python dictionaries are used to keep 
track of all price factors).

Price Models can be calibrated based on historical data (i.e a large timeseries database) or 
implied from current market data. 