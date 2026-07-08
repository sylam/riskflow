"""Minimal debug harness for the platinum hedge — load the canonical fixture, run one
solve_hedge job, print the verdict. Tweak the constants below and step through.

Useful breakpoints:
  riskflow/hedge_solver.py   DiffSolverV2.solve        (driver: bank -> forks -> fit -> verdict)
  riskflow/hedge_solver.py   DiffSolverV2._fit_step    (per-t bootstrap + twin-loss fit)
  riskflow/hedge_solver.py   DiffSolverV2._decide      (argmax; cost-aware branch)
  riskflow/hedge_solver.py   DiffSolverV2._verdict     (OOS rollout vs textbook/no-hedge)
  riskflow/calculation.py    _run_inner_mc_at_t        (the inner-MC fork)
"""
import json
import logging
import os

import riskflow as rf

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tests', 'fixtures', 'policy_test_simulate_only.json')

# ---- knobs (defaults = small & fast for debugging; shipping values in comments) ----
BATCH = 48                # shipping: 256
INNER = 8                 # shipping: 64   (even if INNER_ANTITHETIC)
T_MIN = 100               # shipping: 60   (sweep covers t in [T_MIN, ~116])
LEVELS = 5                # shipping: 9
FIT_ITERS = 30            # shipping: 60
SEED = 1234
INNER_ANTITHETIC = 'Yes'
INNER_DRAWS = 'sobol'     # 'sobol' | 'random'  (iid randn inner draws)
COST_AWARE = 'No'         # shipping: 'Yes'  (charge k|dq| at the argmax in the verdict)


def main():
    cfg = json.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'] = BATCH
    calc['Inner_Sub_Batch'] = INNER
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = INNER_ANTITHETIC
    calc['Inner_Draws'] = INNER_DRAWS
    calc['Random_Seed'] = SEED
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': LEVELS,
        'Training_Action_Chunk_Size': 64,
        'T_Min': T_MIN,
        'DiffV2_Fit_Iters': FIT_ITERS,
        'DiffV2_OOS_Frac': 0.5,
        'DiffV2_Cost_Aware_Argmax': COST_AWARE,
    }

    cx = rf.Context()
    cx.load_json((json.dumps(cfg), 'harness.json'))
    _, result = cx.run_job()

    diag = (result.evaluation_summary or {}).get('diagnostics') or {}
    v = diag.get('verdict') or {}
    print('\nV_0 =', diag.get('V_0'), '| bounded =', diag.get('bounded'),
          '| max|Y_boot| =', diag.get('max_abs_Y_boot'))
    for p in ('greedy', 'textbook', 'nohedge'):
        s = v.get(p) or {}
        print(f"{p:<10} u={s.get('u_mean'):.4f} mean={s.get('wT_mean'):,.0f} "
              f"p5={s.get('wT_p5'):,.0f} cvar5={s.get('wT_cvar5'):,.0f}"
              + (f" | net u={s['u_mean_net']:.4f} cost=${s['turnover_cost_mean']:,.0f}"
                 if 'u_mean_net' in s else ''))
    print('greedy mean|q| =', v.get('greedy_mean_abs_q'), '| q@t0 =', v.get('greedy_q_first'))
    return result


if __name__ == '__main__':
    main()
