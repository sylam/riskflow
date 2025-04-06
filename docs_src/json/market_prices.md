# Market Prices

All risk neutral models need to derived from observable market prices. This section specifies both the
model used and the necessary data required to correctly simulate a risk-neutral model.

Currently only FX, Equities, Commodities and IR rates may be risk neutral:

- FX may be simulated via *GBMTSModelPrices*  and requires only a corresponding FX vol surface to
  establish average ATM vols used by the corresponding *GBMAssetPriceTSModelImplied* model. It is
  specified as follows:

```json
{
"GBMTSModelPrices.AUD":
  {
    "instrument": {
      "Asset_Price_Volatility": "AUD.USD"
    },
    "Children": []
  }
}
```

- The only risk neutral IR currently available may be simulated via 
  *HullWhite2FactorInterestRateModelPrices* and requires both a corresponding swaption volatility
   surface and a set of **instrument definitions** that define the forward starting swaps that
   reference the swaption vol surface. Note that again, only ATM vols are used. They are specified as:

```json
{
  "HullWhite2FactorInterestRateModelPrices.ZAR-JIBAR-3M": {
    "instrument": {
      "Swaption_Volatility": "ZAR_SMILE_ICE",
      "Property_Aliases": null,
      "Generation_Parameters": {
        "Last_Tenor": {
          ".DateOffset": "9Y"
        },
        "Floating_Frequency": {
          ".DateOffset": "6M"
        },
        "First_Tenor": {
          ".DateOffset": "1Y"
        },
        "Day_Count": "ACT_365",
        "Last_Maturity": {
          ".DateOffset": "10Y"
        },
        "First_Start": {
          ".DateOffset": "1Y"
        },
        "Fixed_Frequency": {
          ".DateOffset": "6M"
        },
        "Index_Offset": 0,
        "Last_Start": {
          ".DateOffset": "9Y"
        },
        "First_Maturity": {
          ".DateOffset": "10Y"
        }
      },
      "Generate_Instruments": "No",
      "Instrument_Definitions": [
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "1Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "2Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "5Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "10Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "6M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "1Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "10Y"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "2Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "10Y"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "5Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Holiday_Calendar": null,
          "Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "10Y"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "10Y"
          },
          "Market_Volatility_Type": "Lognormal",
          "Index_Offset": 0,
          "Market_Volatility": {
            ".Percent": 0
          }
        }
      ]
    },
    "Children": []
  }
}
```

Note that although *Generation paramaters* can be specified, instrument definitions are preferred.