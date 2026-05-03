# System Parameters

Global parameters that apply across the entire calculation. These live in the market-data section
of the JSON, alongside `Price Factors`, `Price Models`, etc.

- **Base_Currency** — Defines the base currency against which all calculations will internally use
  (usually `USD`). Any FX rates loaded are interpreted against this currency.
- **Base_Date** — Default valuation date (i.e. the calculation `Run_Date` if no override is
  supplied). If `null`, the current system date is used at run time.
- **Exclude_Deals_With_Missing_Market_Data** — `Yes` (default) or `No`. When `Yes`, deals that
  reference price factors not in the market data are silently skipped from the calculation. When
  `No`, the calculation will raise rather than continue with an incomplete portfolio.
- **Correlations_Healing_Method** — How non-positive-definite correlation matrices are repaired:
    - `Eigenvalue_Raising` (default) — small negative eigenvalues are floored to a tiny positive
      value, then the matrix is rescaled to keep ones on the diagonal.
    - `Alternating_Projections` — iteratively projects onto the cone of positive-semidefinite
      matrices and onto the unit-diagonal subspace until convergence.

```json
{
  "System Parameters": {
    "Base_Currency": "USD",
    "Base_Date": null,
    "Exclude_Deals_With_Missing_Market_Data": "Yes",
    "Correlations_Healing_Method": "Eigenvalue_Raising"
  }
}
```
