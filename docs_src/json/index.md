# Riskflow JSON data format
Riskflow uses standard JSON as its source of input. The structure of a riskflow JSON calculation is as follows:

```json
{
  "Calc": {
    "Calculation": "<calculation section>",
    "Deals": "<Deals section>",
    "MergeMarketData": "<Market data section>",
    "CalendDataFile": "<name of the xml file defining the business day calendars to use>"
  }
}
```

## Calculation
This defines the parameters of the calculation. Parameters are dependent on the calculation type 
and is documented [here](../api_usage/calculations.md). Note, however, the type of the calculation is provided as 
an *Object* attribute.

e.g.
```json
{
  "Calc": {
    "Calculation": {
      "Object": "CreditMonteCarlo",
      "Base_Date": {
        ".Timestamp": "2022-12-01"
      },
      "Currency": "ZAR",
      "Time_Grid": {
        ".Grid": [
          {
            ".DateOffset": "0d"
          },
          {
            ".DateOffset": "2d"
          },
          {
            ".DateOffset": "1w",
            ".Offset": "1w"
          },
          {
            ".DateOffset": "3m",
            ".Offset": "1m"
          },
          {
            ".DateOffset": "2y",
            ".Offset": "3m"
          }
        ]
      },
      "Random_Seed": 1,
      "Dynamic_Scenario_Dates": "Yes",
      "Deflation_Interest_Rate": "ZAR-SWAP",
      "Credit_Valuation_Adjustment": {
        "Calculate": "Yes",
        "Counterparty": "Sketchy Traders LTD",
        "Bank": "Bank of River",
        "Deflate_Stochastically": "Yes",
        "Stochastic_Hazard_Rates": "No",
        "CDS_Tenors": [0.5, 1, 3, 5, 10],
        "Gradient": "No"
      }
    }
  }
}
```
Extra information (e.g. returning simulated cashflows, exposure profiles etc. is specified here).

## Deals
This section only has 3 attributes:
### Reference
The name of this collection of deals (used later when generating the output)

e.g.
```json
{
  "Calc": {
    "Deals": {
      "Reference": "WaterfallTrading"
    }
  }
}
```

### Tag_Titles
It is sometimes useful to add extra information to each deal over an above that which is solely required
for pricing (e.g. Portfolio or Originating desk). Later, when defining these tags per deal, these are the 
titles that will be used for reporting.

e.g. 
```json
{
  "Calc": {
    "Deals": {
      "Tag_Titles": "Portfolio,Desk",
      "Deals": {
        "Children": [
          {          
            "Instrument": {
              ".Deal": {
                "Object": "FXNonDeliverableForward",
                "Reference": "12345678",
                "Tags": [
                  "River Forex Trading,STIRT"
                ],
                "Buy_Currency": "USD",
                "Sell_Currency": "ZAR",
                "Buy_Amount": -10000000,
                "Sell_Amount": -182805000,
                "Settlement_Date": {
                  ".Timestamp": "2022-12-28"
                },
                "Settlement_Currency": "ZAR",
                "Discount_Rate": "ZAR-SWAP"
              }
            }
          }           
        ]
      }
    }
  }
}
```
Notice how the *Tags* attribute corresponds to the *Tag_Titles* attribute.

### Deals
In the example above, we see the general layout for defining deals. A *Deals* attribute is always 
followed a single *Children* which is then defined to be an array. Further *Instrument* objects (of 
type *.Deal*) are then defined as children. Note that should the *Instrument* be a Container object,
it may then define its own *Children* attribute as an array. Nested structures thus form a tree 
structure.

## MergeMarketData
There are just 2 attributes here. 

### MarketDataFile
This is the json file that specifies all [system parameters](./system_parameters.md), [correlations](./correlations.md), 
[price models](./price_models.md), [model configuration](./model_configuration.md) and [market prices](./market_prices.md)
used for bootstrapping.  

This can become quite a large file if several hundred risk factors are used
(as the number of correlations increase as the square of the number of factors). A usual practise is to have one 
MarketData file for risk neutral calculations (like CVA or FVA) and another for real-world calculations (like PFE).

### ExplicitMarketData
Since the data in the marketdata file is static, it cannot contain current market prices. This section acts as an
override for any section already defined in the MarketDataFile specified above. Typically only the 
[Price Factors](./price_factors_overview.md) section would be defined (to contain the latest market data).

```json
{
  "Calc": {
    "MergeMarketData": {
      "MarketDataFile": "mymarketdatafile.json",
      "ExplicitMarketData": {
        "Price Factors": {
          "InterestRate.ZAR-JIBAR-3M": {
            "Day_Count": "ACT_365",
            "Currency": "ZAR",
            "Curve": {
              ".Curve": {
                "meta": [],
                "data": [
                  [
                    0.2465753424657534,
                    0.0713
                  ],
                  [
                    25.0,
                    0.0764
                  ]
                ]
              }
            },
            "Sub_Type": null
          },
          "InterestRate.ZAR-SWAP": {
            "Day_Count": "ACT_365",
            "Currency": "ZAR",
            "Curve": {
              ".Curve": {
                "meta": [],
                "data": [
                  [
                    0.0027397260273972603,
                    0.069
                  ],
                  [
                    10.0,
                    0.079
                  ]                  
                ]
              }
            },
            "Sub_Type": null
          },
          "InterestRate.USD-SOFR": {
            "Day_Count": "ACT_365",
            "Currency": "USD",
            "Curve": {
              ".Curve": {
                "meta": [],
                "data": [
                  [
                    0.0027397260273972603,
                    0.038
                  ],
                  [
                    20.0,
                    0.032
                  ]
                ]
              }
            },
            "Sub_Type": null
          },
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
          },
          "FxRate.ZAR": {
            "Domestic_Currency": "USD",
            "Interest_Rate": "ZAR-SWAP.ZAR-USD-BASIS",
            "Spot": 0.0566318758176227
          },
          "FXVol.USD.ZAR": {
            "Surface": {
              ".Curve": {
                "meta": [
                  2,
                  "Flat"
                ],
                "data": [
                  [
                    1.0,
                    1.0,
                    0.28
                  ]
                ]
              }
            }
          }
        }
      }
    }
  }
}
```

Note that when it comes to exporting any calculation, it is assumed that if a MarketDataFile is specified, then only the 
*Price Factors* section should be exported (as Price Models, Correlations etc. are assumed to remain unchanged). If a 
MarketDataFile is not specified, then all sections of the current configuration are exported.

The idea here is to explicitly save model parameters, correlations etc. directly to the MarketDataFile, leave it static, and 
then define current market prices in the Price Factors section. Of course, for this to work, a MarketDataFile needs to have an 
empty *Price Factors* section  

This makes working with riskflow JSON files easier as we can load up and cache static MarketDataFile once and only read the 
trade, calculation and *Price Factor* data per JSON file.    

## CalendDataFile

An XML calendar file may be loaded defining one more set of business days i.e.

```xml
<Calendars>
  <Calendar Location="Johannesburg" Weekends="Saturday and Sunday" Holidays="2016-01-01|New Year's Day,2016-03-21|Human Rights Day, ... ,2047-12-26|Day of Goodwill" />
  <Calendar Location="Kabul" Weekends="Friday" Holidays="2016-02-15|Liberation Day, 2016-03-20|Noruz (New Year) 2, ... , 2047-10-30|Ashoora*" />
</Calendars>
```

The format is straightforward with the **Location**, **Weekends** rule and Public **Holidays** specified as above. 
If this file was called *calendars.cal*, then this section would be defined thus

```json
{
  "Calc": {
    "CalendDataFile": "calendars.cal"  
  }
}
```
