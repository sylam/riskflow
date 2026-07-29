import importlib.util, os, torch
spec = importlib.util.spec_from_file_location("hntest", os.path.join(os.getcwd(),"tests","test_hn_implied_process.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
import riskflow.utils as utils
from riskflow.pricing import SensitivitiesEstimator

calc, shared = T._dedupe_calc()   # Gradient_Variables defaults to 'All'
shared.reset(calc.num_factors, calc.time_grid)
sk = utils.Factor('EquityPrice', ('EQ',))
fk = utils.Factor('HestonNandiModelParameters', ('EQ','Gamma_Star'))
L = calc.implied_var[sk][fk]
proc = calc.stoch_factors[sk]

# a value that depends on the leaf via BOTH the scenario path (coef 1) and the pricer read (coef 3)
path = proc.generate(shared)
value = path.sum() + (shared.t_Static_Buffer[fk] * 3.0).sum()

# analytic per-path contributions measured separately
L.grad=None; path.sum().backward(retain_graph=True); g_sc = float(L.grad.clone())
L.grad=None; (shared.t_Static_Buffer[fk]*3.0).sum().backward(retain_graph=True); g_pr = float(L.grad.clone())
L.grad=None

est = SensitivitiesEstimator(value, calc.all_var, create_graph=False)
rep = est.report_grad()
# how many entries mention this scope name?
hits = [k for k in rep if k=='HestonNandiModelParameters.EQ.Gamma_Star']
print("entries for Gamma_Star scope:", len(hits))
print("reported grad :", float(rep['HestonNandiModelParameters.EQ.Gamma_Star']))
print("g_scenario+g_pricer:", g_sc + g_pr)
print("len(est.params) (deduped leaves):", len(est.params), " len(all_var):", len(calc.all_var))
print("match:", abs(float(rep['HestonNandiModelParameters.EQ.Gamma_Star']) - (g_sc+g_pr)) < 1e-9)
