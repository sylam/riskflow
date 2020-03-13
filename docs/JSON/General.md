All data types used in RiskFlow are the standard JSON types (string, float, integers) with the exception
of the following:

- ModelParams. An object with the following fields:
    - modeldefaults: a JSON object with price factor type as the fieldname and the factor model as the value.
    - modelfilters: a JSON object with price factor type as the fieldname and an array of fieldname, 
value pairs followed by a factor model.
- Curve. An object with two fields:
    - meta: set to []. Reserved for future use.
    - data: an array with floats or integers. Can be a list of pairs, triples or quads.
- Percent. Float value interpreted as being entered in percentage points.
- Basis. Float value interpreted as being entered in basis points.
- DateOffset. An object that must have at least one or several of the following distinct fields:
    - days: integer
    - weeks: integer
    - months: integer
    - years: integer
- Timestamp: String value interpreted as a date in YYYY-MM-DD date format

Note that all objects must be preceded with a '.'(period) as the fieldname in the JSON.

### Correlation names

Each factor model has its own correlation name. They are as follows:

| Factor Model      | Correlation Name     | Number of factors  | Sub Components  |
| -----------------:|:--------------------:| ------------------:| ---------------:|
| GBMAssetPriceModel  | LognormalDiffusionProcess | 1 | <NA> |
| GBMPriceIndexModel  | LognormalDiffusionProcess | 1 | <NA> |
| HullWhite1FactorInterestRateModel  | HWInterestRate | 1 | F1|
| HullWhite2FactorImpliedInterestRateModel  | HWImpliedInterestRate | 2 | F1, F2 |
| HWHazardRateModel | HullWhiteProcess | 1 | <NA> |
| CSForwardPriceModel | ClewlowStricklandProcess | 1 | <NA> |
| PCAInterestRateModel | InterestRateOUProcess | 3 | PC1,PC2,PC3 |

The sections that follow refer to the *Price Factor* section of the market data file.

---


## Correlation

- **Value**: Float. The market implied correlation between rates. Specified as "pricefactor1/pricefactor2" e.g. Correlation.FxRate.USD.ZAR/ReferencePrice.BRENT_OIL-IPE.USD