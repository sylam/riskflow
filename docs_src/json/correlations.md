#  Correlations

Correlations are defined just once between 2 price factors (as the correlation matrix is symmetric). 
The convention is to provide the name of the first correlation factor and then a list of all 
subsequent correlations as a standard JSON object.

However, the naming of these correlation values needs some consideration as its possible to have 2 or
more difference Price Models for the same factor. Also, Each Price model could have several factors of
randomness i.e. a Hull White 2 factor model has 2 sources of randomness whereas Geometric Brownian 
Motion on an Equity just has 1.

An example correlation specification is presented below i.e.

```json
{
  "Correlations": {
      "HWInterestRate.ZAR-SWAP.F1": {
        "HWInterestRate.USD-MASTER.F1": 0.2,
        "LognormalDiffusionProcess.ZAR": 0.5
      },
      "HWInterestRate.USD-MASTER.F1": {
        "LognormalDiffusionProcess.ZAR": -0.1
      }
    }
}
```

The correlation between the one factor hull white model for ZAR-SWAP and the lognormal
process for ZAR is 0.5 while the correlation between ZAR-SWAP and the one factor hull white 
model for the USD-MASTER is 0.2. There is a correlation of -0.1 between the USD-MASTER and
the GBM process called ZAR (here interpreted as an FxRate)

# Correlation names

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

These names may change in future.