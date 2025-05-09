## Notebook

The outputs obtained from running the jupyter extensions for riskflow are self-explanatory and can be 
downloaded by right-clicking and selecting "Save As". (see [here](../quickstart))

## Base Revaluation

Assuming a correct JSON file was loaded, a base valuation (theoretical price) calculation can be
explicitly run by 

```python
calc, out = cx.Base_Valuation(overrides={})
```

The out variable is a dictionary containing 3 items:

- Netting - this contains the riskflow internal calculation structure and is intended for Developers
- Stats - a dictionary containing statistics (Deals loaded/skipped and execution time etc.)
- Results - the dictionary that contains the results seen in the GUI. 
  - mtm - the mark to market of all instruments loaded along with any tagged data
  - Greeks_First - the analytic sensitivities of the portfolio by risk factor

## Credit Monte Carlo

Once again, assuming a valid JSON file, a Monte Carlo calculation can be explicitly run by:

```python
calc, out = cx.Credit_Monte_Carlo(overrides={})
```

The out variable will contain the same 3 dictionary items as before except that the Results key will 
contain more information depending on the sub calculations requested

The Results dictionary may contain:

  - mtm - the calculated theoretical prices per scenario per time point
  - exposure_profile - the percentiles of the mtm calculation
  - fva - the funding valuation adjustment 
  - grad_fva - the sensitivities of the fva
  - cva - the credit valuation adjustment
  - grad_cva - the sensitivities of the cva

