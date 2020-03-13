

---


## DiscountRate

- **Interest_Rate**: String. Name of the *Interest Rate* price factor for discounting

## InterestRate

- **Currency**: String. The associated currency for this curve
- **Curve**: *Curve* object specifying the continuously compounded interest rate
- **Day_Count**: String. Either ACT_365 or ACT_360. The daycount convention for this curve
- **Sub_Type**: Optional String can be null or set to **BasisSpread** if this curve is a spread
over its parent

## InterestRateVol

- **Surface**: *Curve* object consisting of (moneyness, expiry, tenor, volatility) quads. Flatextrapolated and linearly interpolated. All Floats.

## InterestYieldVol

- **Surface**: *Curve* object consisting of (moneyness, expiry, tenor, volatility) quads. Flat
extrapolated and linearly interpolated. All Floats.
- **Property_Aliases**: list of key value pairs specifying additional options e.g. Specification
of a shifted black scholes value via BlackScholesDisplacedShiftValue