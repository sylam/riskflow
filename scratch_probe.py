import importlib.util, os
spec = importlib.util.spec_from_file_location("hntest", os.path.join(os.getcwd(),"tests","test_hn_implied_process.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
import riskflow.utils as utils
from collections import Counter

def run(gradvar):
    calc, shared = T._dedupe_calc()
    p = dict(calc.params); p['Gradient_Variables'] = gradvar
    try:
        shared2 = calc.update_factors(p, T.BASE_HN, 0, 1)
        av = calc.all_var
        sk = utils.Factor('EquityPrice', ('EQ',))
        fk = utils.Factor('HestonNandiModelParameters', ('EQ','Gamma_Star'))
        L = calc.implied_var[sk][fk]
        print(f"[{gradvar}] all_var type={type(av).__name__} len={len(av) if hasattr(av,'__len__') else 'NA'} "
              f"| leaf requires_grad={L.requires_grad} static_is_implied={calc.static_var[fk] is L}")
        if hasattr(av,'__iter__') and not isinstance(av,dict):
            names = Counter(utils.check_scope_name(k) for k,_ in av)
            dups = {n:c for n,c in names.items() if c>1}
            print(f"        duplicate scope names in all_var: {dups if dups else 'none'}")
    except Exception as e:
        print(f"[{gradvar}] EXCEPTION: {type(e).__name__}: {e}")

for gv in ('All','Implied','Factors'):
    run(gv)
