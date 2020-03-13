

---


## InflationRate

- **Currency**: String. The associated currency for this curve
- **Reference_Name**: String. allowed values listed [here](../Theory/Inflation/#price-index-references)
- **Curve**: *Curve* object specifying the continuously compounded inflation growth rate
- **Day_Count**: String. Either ACT_365 or ACT_360. The daycount convention for this curve
- **Price_Index**: String. Name of associated PriceIndex factor

## PriceIndex

- **Index**: A *Curve* object representing a series of (date, value) pairs where the date is an excel integer
- **Next_Publication_Date**: *TimeStamp* object
- **Last_Period_Start**: *TimeStamp* object
- **Publication_Period**: String. Either **Monthly** or **Quarterly**