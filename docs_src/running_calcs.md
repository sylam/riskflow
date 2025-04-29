## In Jupyter

We've already seen an example of a *Base_Valuation* calculation. This simply calculates the 
theoretical price of a portfolio of derivatives. In order to calculate expectations of future 
simulations of our portfolio, we also need to define stochastic processes. This is done 
in the *Settings* tab under *Model Configuration*. In the clip below, we assign both FxRates 
and EquityPrices to be simulated using Geometric Brownian Motion (GBMAssetPriceModel). 

Notice that the pairing of Risk Factor with a compatible stochastic process is fixed.   


<video width="1000" height="500" controls>
  <source src="credit_monte_carlo.mp4" type="video/mp4">
</video>

The parameters are explained [here](../api_usage/calculations). Of course, there is also the 
problem of capturing correlations between risk factors. Defining these Correlations and 
calculating the stochastic process parameters can be eased by loading up a file of 
historical data.

## In Python

A JSON file defining a calculation can also be directly run in python

```python
import riskflow as rf
cx = rf.Context()
cx.load_json('BaseValuation.Test1.json')
calc, out = cx.run_job(overrides={})
```

This will return references to the calculation object (calc) and the output itself (out)
