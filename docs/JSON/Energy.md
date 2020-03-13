

---


## ForwardPrice

- **Currency**: String. The associated currency for this curve
- **Curve**: *Curve* object of date, rate pairs specifying forward price at the corresponding
excel date

## ForwardPriceSample

- **Offset**: Integer specifying a calendar day offset
- **Holiday_Calendar**: String specifying the name of the calendar to use in the calendar xml file
- **Sampling_Convention**: String. Either **ForwardPriceSampleDaily** or 
**ForwardPriceSampleBullet**

## ForwardPriceVol

- **Surface**: *Curve* object consisting of (moneyness, expiry, delivery, volatility) quads. Flat
extrapolated and linearly interpolated. All Floats.

## ReferencePrice

- **Currency**: String. The associated currency for this curve.
- **Fixing_Curve**: *Curve* object of date, reference date pairs specifying the delivery date for
a particular date. Both dates are in excel format.
- **ForwardPrice**: String. The name of associated ForwardPrice factor

## ReferenceVol

- **ForwardPriceVol**: String. Name of the *ForwardPriceVol* price factor to use
- **ReferencePrice**: String: Name of the *ReferencePrice* price factor that defines the reference
lookup