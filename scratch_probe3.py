import importlib.util, os, torch
spec = importlib.util.spec_from_file_location("hntest", os.path.join(os.getcwd(),"tests","test_hn_implied_process.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
import riskflow.utils as utils
from riskflow.config import Config
from riskflow.calculation import construct_calculation
from riskflow.instruments import construct_instrument
import pandas as pd

# Rebuild _dedupe_calc but DO NOT route EquityPrice to the HN implied process (static-only pricer dep)
def build(route):
    hn = dict(T.IMPLIED_PARAM); hn.pop('Steps_Per_Year', None)
    pf = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency':'USD','Day_Count':'ACT_365','Sub_Type':None,'Curve':utils.Curve([], [[0.0,0.02],[5.0,0.02]])},
        'DiscountRate.USD': {'Interest_Rate':'USD'},
        'EquityPrice.EQ': {'Spot':100.0,'Currency':'USD','Interest_Rate':'USD','Issuer':'','Respect_Default':'No','Jump_Level':0.0},
        'DividendRate.EQ': {'Currency':'USD','Floor':None,'Curve':utils.Curve([], [[0.01,0.01],[5.0,0.01]])},
        'EquityPriceVol.EQ': {'Surface_Type':'Explicit','Moneyness_Rule':'Sticky_Moneyness','Surface':utils.Curve([], [[m,t,0.25] for m in (0.8,1.0,1.2) for t in (0.02,2.0)])},
        'HestonNandiModelParameters.EQ': dict(hn, Property_Aliases=None),
    }
    c = Config()
    c.params['System Parameters']['Base_Currency']='USD'; c.params['System Parameters']['Base_Date']=T.BASE_HN
    c.params['Price Factors']=pf; c.params['Price Models']={}
    if route:
        c.params['Model Configuration'].append('EquityPrice', (), 'HestonNandiImpliedSpotModel')
    horizon=30; bdates=[T.BASE_HN+pd.Timedelta(days=d) for d in range(1,horizon+1)]
    field={'Object':'EquityBarrierOption','Reference':'BARR1','Currency':'USD','Payoff_Currency':'USD','Equity':'EQ','Dividends':'EQ','Discount_Rate':'USD','Equity_Volatility':'EQ','Buy_Sell':'Buy','Option_Type':'Call','Strike_Price':100.0,'Expiry_Date':T.BASE_HN+pd.Timedelta(days=horizon),'Units':10.0,'Barrier_Type':'Down_And_Out','Barrier_Price':1.0,'Cash_Rebate':0.0,'Barrier_Dates':[[d,1.0] for d in bdates],'Barrier_Monitoring_Frequency':pd.DateOffset(days=1)}
    val={'EquityBarrierOption':{'SpotModel':'HestonNandi'}}
    c.params['Valuation Configuration']=val
    inst=construct_instrument(field,val)
    c.deals={'Attributes':{'Reference':'test','Tag_Titles':''},'Deals':{'Children':[{'Instrument':inst}]},'Calculation':{'Base_Date':T.BASE_HN,'Currency':'USD'}}
    calc=construct_calculation('Credit_Monte_Carlo',c,device=torch.device('cpu'),prec=T.DTYPE)
    calc.input_time_grid='0d 2d(1w) 1m'; calc.batch_size=64
    params={'Run_Date':'2024-06-28','Time_grid':'0d 2d(1w) 1m','Batch_Size':64,'Simulation_Batches':1,'Random_Seed':1,'Currency':'USD','MCMC_Simulations':0,'Tenor_Offset':0.0,'CVA':{'Gradient':'Yes'}}
    calc.params=params
    shared=calc.update_factors(params,T.BASE_HN,0,1)
    return calc,shared

calc,shared=build(route=False)   # static-only, process NOT routed
sk=utils.Factor('EquityPrice',('EQ',))
fk=utils.Factor('HestonNandiModelParameters',('EQ','Gamma_Star'))
print("routed to implied process:", sk in calc.implied_var)
print("HN param in static_var:", fk in calc.static_var, " requires_grad:", calc.static_var[fk].requires_grad if fk in calc.static_var else None)
print("HN param in t_Static_Buffer:", fk in shared.t_Static_Buffer)
print("static leaf is a FRESH tensor (not aliased to any implied leaf):", not any(fk in v for v in calc.implied_var.values()))
