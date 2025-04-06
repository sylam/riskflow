#  Price Factors

Prices factors representing a snapshot of a market (and hence taken at the same time) are to be provided
here. e.g. for the ZAR FxRate with USD as the **Base Currency**

```json
{
  "Price Factors": {
    "FxRate.USD": {
      "Domestic_Currency": null,
      "Interest_Rate": "USD-MASTER",
      "Spot": 1
    },
    "FxRate.ZAR": {
      "Domestic_Currency": "USD",
      "Interest_Rate": "ZAR-SWAP.ZAR-USD-BASIS",
      "Spot": 0.075475
    }
  }
}
```

Note that the corresponding USD-MASTER and both the ZAR-SWAP and ZAR-USD-BASIS interest rate curves
need to be defined.

Price Factors like interest rates, volatility surfaces etc., are regarded as being independently 
defined (as opposed to FxRates or EquityRates which depend on Interest Rates for their cost of carry).

Interest Rate price factors uses the .Curve format. e.g.

```json
{
  "Price Factors": {
    "InterestRate.ZAR-JIBAR-3M": {
      "Day_Count": "ACT_365",
      "Currency": "ZAR",
      "Curve": {
        ".Curve": {
          "meta": [],
          "data": [
            [
              0.25,
              0.07
            ],
            [
              15.0,
              0.092
            ]
          ]
        }
      }
    }
  }
}
```

This specifies that the curve starts roughly 3m (0.25 years) from now at 7% and goes to 9.2% 15 years
from now. Curve Interpolation is discussed [here](../theory/asset_pricing.md#interest-and-inflation-rate-interpolation).

Generally, in Riskflow, all time periods (unless otherwise specified) are assumed to be annual.

Basis spreads are defined as InterestRates but have a subtype specified i.e.

```json
{
  "InterestRate.ZAR-SWAP.ZAR-USD-BASIS": {
    "Day_Count": "ACT_365",
    "Currency": "ZAR",
    "Curve": {
      ".Curve": {
        "meta": [],
        "data": [
          [
            0.0027397260273972603,
            0.0001897614337292436
          ],
          [
            30.024657534246575,
            -0.008421900202085705
          ]
        ]
      }
    },
    "Sub_Type": "BasisSpread"
  }
}
```

In general, when defining price models, combine the name of the Price Factor with the **Sub_Type** 
(if the **Sub_Type** is defined for that price factor).

A full list of all the required fields for each Price Factor is defined below:
