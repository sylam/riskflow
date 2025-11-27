# API

Everything in RiskFlow is based off a *Context*. All calculations are constructed with reference to one.
Note that the interest rate curves start one day from now i.e. $1/365\approx 0.00274$. No interest rate
curve can start at time 0 (although the rate at time 0 is flat extrapolated from the first timepoint).

Assume the Market Data file is called *MarketData.json*, the trade data is called *fxfwd.json* and the
calendar file is called *calendars.xml*. Here's how to perform a *Base Revaluation*:

```python
import riskflow as rf

# create a context
cx      = rf.Context()

# load a json 
# note - the json should contain references to calendars and 
# a valid marketdata file

cx.load_json('fxfwd.json')

# Once loaded, we can provide overrides to the calculation 
# this returns the results and a reference to the calculation itself
calc, out = cx.Base_Valuation(overrides={'Run_Date':'2024-08-01', 'Currency':'USD'})
```

```out``` will now contain the stats for the computation in a dictionary with the field
*Result* containing the results.
To check the results, use the following code:
```python
out['Results']['mtm']
```
should return a pandas dataframe and look like:
```
Test	NettingCollateralSet	0.0
	341	FXNonDeliverableForward	-343.123474121
```

So the market value at 1 August 2024 of the forward is -343 USD. Note that the 
output could vary depending on the structure of the json and any tags you may 
have defined. The results may also contain other outputs (e.g. Sensitivities) so 
a quick 
```python
dir(out['Results'])
```
is recommended to see all the output results. If a credit simulation was
needed, we could reuse the context and create a new override

```python

# first define the parameters (i.e. the overrides)
time_grid = '0d 2d 1w(1w) 3m(1m) 2y(3m)'

params 	= { 'Time_grid':time_grid, 'Run_Date':'2024-08-01', 'Currency':'ZAR', 'Simulation_Batches':2,
             'Batch_Size':512, 'Random_Seed':6126, 'Calc_Scenarios':'No',
             'Generate_Cashflows': 'Yes', 'Dynamic_Scenario_Dates': 'Yes'
           }

# execute the calculation with the new parameters
calc, out = cx.Credit_Monte_Carlo(overrides=params)
```

To see the output, access the ```Results``` key i.e.
```python
out['Results']['exposure_profile'])
```
should output
```
             EE         PFE
2017-08-01    0.000000    0.000000
2017-08-03    1.809047    0.000000
2017-08-08   25.414378  201.102859
2017-08-15   62.647971  414.040855
2017-08-22   96.029947  574.368042
2017-08-29  130.653986  737.920593
2017-08-31  132.900141  745.158432
2017-09-01    0.000000    0.000000
```
Where **EE** is the Expected Exposure and **PFE** is the Peak Exposure at 95% (the default). Note that
this returns a pandas dataframe and as such, can be plotted by using the ```.plot()``` method
(assuming *matplotlib* is installed).

This assumes that all suitable stochastic processes are defined.

---
