"""Minimal debug harness for the platinum hedge — load the canonical fixture, run one
solve_hedge job, print the verdict. Tweak the constants below and step through.

PRODUCTION RECIPE (validated 2026-07-09, Components world):
  TRAIN: BATCH=8192, INNER=64, T_MIN=0, LEVELS=9, FIT_ITERS=60, COST_AWARE='Yes',
  one-step forks + trust-region clamp + per-column grad norm (all default 'Yes').
  Single-seed is now RELIABLE: u=0.380-0.392 across seeds (the clamp killed the
  phantom-extrapolation bimodality; per-column lambda_j added +0.01-0.017/seed).
  ENSEMBLE (optional upside + winner's-curse reduction): DiffV2_Load_Value_Fn=[ckpts];
  deploys at Inner_Sub_Batch=16-32 with no quality loss (fresh-world u≈0.38,
  E[W_T]≈$1.14M, p5≈−116k vs textbook 0.064/$191k/−879k; OOD gate passed).
  Notes: single-net selection floor is inner=64 (twin July legs); peak ≈ 8.1 GB at
  8192x64. Backtest: backtest_walk_forward.py (Historical_Replay realized-path mode).

Useful breakpoints:
  riskflow/hedge_solver.py   DiffSolverV2.solve        (driver: bank -> forks -> fit -> verdict)
  riskflow/hedge_solver.py   DiffSolverV2._fit_step    (per-t bootstrap + twin-loss fit)
  riskflow/hedge_solver.py   DiffSolverV2._decide      (argmax; cost-aware branch)
  riskflow/hedge_solver.py   DiffSolverV2._verdict     (OOS rollout vs textbook/no-hedge)
  riskflow/calculation.py    _run_inner_mc_at_t        (the inner-MC fork; one-step window)
  riskflow/pricing.py        pv_energy_cashflows       (ForwardCurve=Components liability)
"""
import json
import logging
import os

import riskflow as rf

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tests', 'fixtures', 'policy_test_simulate_only.json')

# ---- knobs (defaults = small & fast for debugging; production values in comments) ----
BATCH = 256                # production: 8192
INNER = 8                 # production: 64  (selection floor on the dual strip)
T_MIN = 100               # production: 0   (full ~116-step window)
LEVELS = 5                # production: 9
FIT_ITERS = 30            # production: 60
SEED = 1234
INNER_ANTITHETIC = 'Yes'
INNER_DRAWS = 'sobol'     # 'sobol' | 'random'  (iid randn inner draws)
COST_AWARE = 'No'         # production: 'Yes' (charge k|dq| at the argmax)
ONE_STEP = 'Yes'          # production: 'Yes' ({t,t+1} fork window; 'No' = legacy full forks)
SAVE_CKPT = ''            # production: per-seed path, then ensemble via LOAD_CKPTS
LOAD_CKPTS = []           # eval-only: list of pruned member checkpoints (ensemble argmax)


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
        'DiffV2_One_Step_Fork': ONE_STEP,
    }
    if SAVE_CKPT:
        hp['Solver']['DiffV2_Save_Value_Fn'] = SAVE_CKPT
    if LOAD_CKPTS:
        hp['Solver']['DiffV2_Load_Value_Fn'] = LOAD_CKPTS

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
