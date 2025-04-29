## Jupyter

The fastest way to use riskflow is to install jupyter and [riskflow_widgets](https://github.com/sylam/riskflow_widgets).
The User Interface for riskflow is called [riskflow_jupyter](https://github.com/sylam/riskflow/raw/refs/heads/master/riskflow_jupyter.py) 
and is not included in the core riskflow package (as it's solely a GUI and not necessary to run any calculations)

Once installed, you can run the workbench by entering this into a cell 

```python
import pandas as pd
import riskflow_jupyter
```

Then run the workbench

```python
wb = riskflow_jupyter.Workbench(default_rundate=pd.Timestamp('2025-04-02'))
```

Here's a quick video showing how an equity option can be booked. Notice that each risk factor needs to be correctly 
defined. All interest rates in Riskflow are assumed to be NACC (Nominal Annual Compounded Continuously).  

<video width="1000" height="500" controls>
  <source src="quickstart.mp4" type="video/mp4">
</video>

Once the calculation is executed, it can be exported to json. Note that the wb variable contains the state of the library 
and can be used to learn/interrogate the data in the workbench. In particular, the context member variable stores the current
context of the library. 

The GUI is also designed to work with [voila](https://github.com/voila-dashboards/voila) as a stand-alone dashboard if
desired. 

An example base_valuation calculation is in the ./examples folder

## Design Philosophy

The basic idea in riskflow is to define all inputs that would normally be associated with pricing a financial instrument 
separately as a *price factor*. This includes things like Fx rates, equity prices, volatilities, interest rates etc. 

The portfolio of instruments is then declared to reference these price factors. This separates the definition of the 
financial instrument from the market variables that would typically be used to calculate its theoretical (risk neutral) price.

Later, a stochastic process model can be attached to a price factor that specifies how that price factor changes through time.
This is the basis for xVA calculations (as long as you can specify a risk neutral calibration). Of course, you are free to specify any 
process model you like to test the performance of your portfolio of derivatives.

