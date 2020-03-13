There are two fundemental data files that need to be setup prior to performing any calculations viz.
Market Data and Trade Data files:

## Market Data

A market data file is a JSON object that contains the following sections:

### System Parameters

Here the **Base Currency** and the **Correlations Healing Method** is specified. All FX rates are
expressed relative to the base currency. Since the correlations between risk factors may not always
form a positive-definite matrix, a technique called *Eigenvalue Raising* is used to ensure that all
eigenvalues of the correlation matrix are positive. The matrix is then rescaled to ensure that there
are one's along the diagonal.

###  Model Configuration

This specifies a assignment of Risk Factor to Process Model. There are two subsections:

**modeldefaults** contain mappings of price factors to models e.g. FxRate:GBMAssetPriceModel defaults
all FX rates to use the GBMAssetPriceModel.

**modelfilters** allow exceptions to the default model. Filters are allowed based on attributes of the
price factor (or its *ID*) e.g.

```json
"InterestRateBasisSpread": [
  [
	  [
		"Currency",
		"ZAR"
	  ],
	  "HullWhite1FactorInterestRateModel"
	]
]
```

is interpreted as using the HullWhite1FactorInterestRateModel for all InterestRateBasisSpreads where
the Currency is ZAR.

###  Price Factors

Prices factors representing a snapshot of a market (and hence taken at the same time) are to be provided
here. e.g. for the ZAR FxRate with USD as the **Base Currency**

```json
"Price Factors": {
  "FxRate.USD": {
    "Domestic_Currency": null,
    "Interest_Rate": "USD-MASTER",
    "Spot": 1
   },
  "FxRate.ZAR": {
    "Domestic_Currency": USD,
    "Interest_Rate": "ZAR-SWAP.ZAR-USD-BASIS",
    "Spot": 0.075475
   }
}
```
Note that the corresponding USD-MASTER and both the ZAR-SWAP and ZAR-USD-BASIS interest rate curves
need to be defined.

###  Price Models

Parameters for the stochastic processes used to simulate it's corresponding price factor is to be
provided here e.g. for ZAR,

```json
"Price Models": {
  "GBMAssetPriceModel.ZAR": {
    "Vol": 0.171569979672,
    "Drift": -0.0436796256753
   }
}
```

Note that the name of the model needs to match the price factor.

###  Correlations

Correlations between Price Models is to be provided here. e.g.
```json
"Correlations": {
      "HWInterestRate.ZAR-SWAP.F1": {
        "HWInterestRate.USD-MASTER.F1": 0.2,
        "LognormalDiffusionProcess.ZAR": 0.5
      },
      "HWInterestRate.USD-MASTER.F1": {
        "LognormalDiffusionProcess.ZAR": -0.1
      }
    }
```
specifies that the correlation between the one factor hull white model for ZAR-SWAP and the lognormal
process for ZAR is 0.5 while the correlation between ZAR-SWAP and
the one factor hull white model for the USD-MASTER is 0.2.

### Market Prices

All risk neutral models need to derived from observable market prices. This section specifies both the
model used and the necessary data required to correctly simulate a risk-neutral model.

Currently only FX and IR rates may be risk neutral:

- FX may be simulated via *GBMTSModelPrices*  and requires only a corresponding FX vol surface to
  establish average ATM vols used by the corresponding *GBMAssetPriceTSModelImplied* model. It is
  specified as follows:

```json
"GBMTSModelPrices.AUD",
  {
    "instrument": {
      "Asset_Price_Volatility": "AUD.USD"
    },
    "Children": []
  }
```

- The only risk neutral IR currently available may be simulated via 
  *HullWhite2FactorInterestRateModelPrices* and requires both a corresponding swaption volatility
   surface as well as a set of **instrument definitions** that define the forward starting swaps that
   reference the swaption vol surface. Note that again, only ATM vols are used. They are specified as:

```json
"HullWhite2FactorInterestRateModelPrices.ZAR-JIBAR-3M",
{
	"instrument": {
		"Swaption_Volatility": "ZAR_SMILE_ICE",
		"Property_Aliases": null,
		"Generation_Parameters": {
			"Last_Tenor": {
				".DateOffset": {
					"years": 9
				}
			},
			"Floating_Frequency": {
				".DateOffset": {
					"months": 6
				}
			},
			"First_Tenor": {
				".DateOffset": {
					"years": 1
				}
			},
			"Day_Count": "ACT_365",
			"Last_Maturity": {
				".DateOffset": {
					"years": 10
				}
			},
			"First_Start": {
				".DateOffset": {
					"years": 1
				}
			},
			"Fixed_Frequency": {
				".DateOffset": {
					"months": 6
				}
			},
			"Index_Offset": 0.0,
			"Last_Start": {
				".DateOffset": {
					"years": 9
				}
			},
			"First_Maturity": {
				".DateOffset": {
					"years": 10
				}
			}
		},
		"Generate_Instruments": "No",
		"Instrument_Definitions": [
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"months": 3
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 1
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			},
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"months": 3
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 2
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			},
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"months": 3
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 5
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			},
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"months": 3
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 10
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			},
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"months": 6
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 1
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			},
			...
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"years": 10
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 2
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			},
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"years": 10
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 5
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			},
			{
				"Floating_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Weight": 1.0,
				"Holiday_Calendar": null,
				"Day_Count": "ACT_365",
				"Start": {
					".DateOffset": {
						"years": 10
					}
				},
				"Fixed_Frequency": {
					".DateOffset": {
						"months": 3
					}
				},
				"Tenor": {
					".DateOffset": {
						"years": 10
					}
				},
				"Market_Volatility_Type": "Lognormal",
				"Index_Offset": 0.0,
				"Market_Volatility": {
					".Percent": 0.0
				}
			}
		]
	},
	"Children": []
}
```
Note that although *Generation paramaters* can be specified, instrument definitions are preferred.

### Bootstrapper Configuration

This section links the price factors to the market prices from the previous section as well as
adding any extra parameters required for calibration

The 2 bootstrapper configurations so far supported are:

```json
"GBMTSImpliedParameters": "GBMTSModelPrices"
"HullWhite2FactorModelParameters": "HullWhite2FactorInterestRateModelPrices"
```

## Trade Data

This is a JSON object representing the portfolio of trades belonging to a single netting set. As as
example, consider booking a **FXNonDeliverableForward** inside
an uncollateralized netting set. 
```json
{
  "Deals": {
    "Children": [
      {
        "instrument": {
          ".Deal": {
            "Object": "NettingCollateralSet",
            "Reference": "Test",
            "Tags": "",
            "Collateralized": "False",
            "Netted": "True",
          }
        },
        "Children": [
          {
            "instrument": {
              ".Deal": {
                "Object": "FXNonDeliverableForward",
                "Reference": "341",
                "Tags": "",
                "MtM": "",
                "Sell_Currency": "USD",
                "Sell_Amount": 1000,
                "Settlement_Date": {
                  ".Timestamp": "2017-08-31"
                },
                "Settlement_Currency": "ZAR",
                "Buy_Amount": 14000,
                "Discount_Rate": "ZAR-SWAP",
                "Buy_Currency": "ZAR"
              }
            }
          }
        ]
      }
    ]
  }
}
```
The JSON is parsed into python dictionaries and lists. However, note that objects beginning with a
period (e.g. .Deal, .Timestamp etc.) will convert the enclosed dictionary/list into the corresponding
python object. When defining Deals, any number of key-value attributes may be specified as it's loaded
directly as a python dictionary.

## Calendars

Optionally, an XML calendar file may be loaded defining one more set of business days i.e.

```xml
<Calendars>
  <Calendar Location="Johannesburg" Weekends="Saturday and Sunday" Holidays="2016-01-01|New Year's Day,2016-03-21|Human Rights Day, ... ,2047-12-26|Day of Goodwill" />
  <Calendar Location="Kabul" Weekends="Friday" Holidays="2016-02-15|Liberation Day, 2016-03-20|Noruz (New Year) 2, ... , 2047-10-30|Ashoora*" />
</Calendars>
```


---
