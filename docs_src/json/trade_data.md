# Trade Data

This is a JSON object representing the portfolio of trades belonging to a single netting set. As an
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
            "Netted": "True"
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

A list of all supported deals (as well as the corresponding datatypes) is as follows: