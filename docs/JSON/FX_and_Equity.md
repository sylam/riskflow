

---


## DividendRate

- **Currency**: String.
- **Curve**: *Curve* object specifying the continuous dividend yield

## EquityPrice

- **Currency**: String
- **Interest_Rate**: String representing the equity repo curve
- **Spot**:Spot rate in the specified *Currency*

## EquityPriceVol

- **Surface**: *Curve* object consisting of (moneyness, expiry, volatility) triples. Flat
extrapolated and linearly interpolated. All Floats.

## FXVol

- **Surface**: *Curve* object consisting of (moneyness, expiry, volatility) triples. Flat
extrapolated and linearly interpolated. All Floats.

## FxRate

- **Interest_Rate**: String. Associated interest rate curve name.
- **Spot**:Float. Spot rate in base currency.