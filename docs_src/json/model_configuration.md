#  Model Configuration

This specifies an assignment of Risk Factor to Process Model. There are two subsections:

**modeldefaults** contain mappings of price factors to models e.g. FxRate:GBMAssetPriceModel defaults
all FX rates to use the GBMAssetPriceModel.

**modelfilters** allow exceptions to the default model. Filters are allowed based on attributes of the
price factor (or its *ID*) e.g.

```json
{
  "Model Configuration": {
    ".ModelParams": {
      "modeldefaults": {
        "EquityPrice": "GBMAssetPriceModel",
        "FxRate": "GBMAssetPriceModel",
        "InterestRate": "PCAInterestRateModel"
      },
      "modelfilters": {
        "InterestRate": [
          [
            [
              "Currency",
              "CHF"
            ],
            "HullWhite1FactorInterestRateModel"
          ],
          [
            [
              "Currency",
              "DKK"
            ],
            "HullWhite1FactorInterestRateModel"
          ]
        ],
        "InterestRateBasisSpread": [
          [
            [
              "ID",
              "ZAR-SWAP.ZAR-USD-BASIS"
            ],
            "HullWhite1FactorInterestRateModel"
          ]
        ]
      }
    }
  }
}
```

is interpreted as follows:
- All EquityPrice and FxRate price factors are to be simulated using GBMAssetPriceModel
- All InterestRates are to use the PCAInterestRateModel.

Except for the following (the filters section)
- For InterestRates where the currency is CHF or DKK, use the HullWhite1FactorInterestRateModel
- For InterestRateBasisSpread when the ID is ZAR-SWAP.ZAR-USD-BASIS, use the HullWhite1FactorInterestRateModel

You are permitted to model InterestRate spreads (called InterestRateBasisSpread) separately to the base InterestRate
curve.


