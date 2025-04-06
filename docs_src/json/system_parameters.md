# System Parameters

Here we set global system parameters

- **Base_Currency** - Defines the base currency against which all calculations will internally use (Usually set to USD)  
- **Base_Date** - Defines the Default Calculation Date (i.e. the Run_Date). If null, the current system date is used.
- **Correlations_Healing_Method**: 
  - Since the correlations between risk factors may not always form a positive-definite matrix,
  a technique called *Eigenvalue Raising* is used to ensure that all eigenvalues of the correlation matrix are positive. 
  The matrix is then rescaled to ensure that there are one's along the diagonal.
                   
```json
{
  "System Parameters": {
    "Base_Currency": "USD",
    "Base_Date": null,
    "Correlations_Healing_Method": "Eigenvalue_Raising"
  }
}
```

Other correlation healing methodologies can be added in future.