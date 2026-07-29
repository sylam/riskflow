"""OOD STRESS GATE: train the dual-strip diff-ML hedge under the CALIBRATED regime world,
freeze it (DiffV2_Save_Value_Fn), then evaluate the frozen policy (DiffV2_Load_Value_Fn,
training skipped) under STRESSED worlds it never saw — deployment robustness, not
in-distribution fit ([[feedback_rl_ood_robustness]]: stress-not-realistic params).

Stresses (ExplicitMarketData Price Models override of MarkovHMMSpotModel.PLATINUM_LME):
  control      — same world, fresh seed (frozen-policy in-distribution baseline)
  vol_x1.5     — per-regime Sigma * 1.5
  bear_drift   — per-regime Mu - 0.15 (log-space annualised; ~one regime's worth bearish)
  crash_tilt   — transition rows tilted +5pp toward the crash regime (state 2)

textbook / no-hedge are policy-free and re-derived under each stressed world — fair
comparators. All three policies are scored with the TRAIN-time utility (the checkpoint
restores utility_scale, so u() is the same function everywhere).

JSON-is-the-contract: load_json + run_job only.
"""
import copy
import csv
import datetime
import json
import logging
import os
import sys

import riskflow as rf

from validate_cross_market import FIXTURE, PROFILES, add_lme_leg

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

CALIBRATED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'tests', 'fixtures', 'data', 'MarketDataRF_platinum_calibrated.json')
HMM_KEY = 'MarkovHMMSpotModel.PLATINUM_LME'


def stressed_hmm(base, stress):
    p = copy.deepcopy(base)
    if stress == 'vol_x1.5':
        for s in p['States']:
            s['Sigma'] *= 1.5
    elif stress == 'bear_drift':
        for s in p['States']:
            s['Mu'] -= 0.15
    elif stress == 'crash_tilt':
        P = p['Transition_Matrix']
        for i in range(len(P)):
            if i != 2:
                shift = min(0.05, P[i][i])
                P[i][i] -= shift
                P[i][2] += shift
    elif stress != 'control':
        raise ValueError(stress)
    return p


def build_cfg(template, prof, seed, save=None, load=None, hmm_override=None):
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    # a solve is a stream: fit batches, then a held-out one. (B, 1)+OOS 0.5 -> (B/2, 2)
    calc['Batch_Size'], calc['Simulation_Batches'] = prof['batch'] // 2, 2
    calc['Inner_Sub_Batch'] = prof['inner']
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = 'Yes'
    calc['Random_Seed'] = seed
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': prof['t_min'],
        'DiffV2_Fit_Iters': prof['iters'],
    }
    if save:
        hp['Solver']['DiffV2_Save_Value_Fn'] = save
    if load:
        hp['Solver']['DiffV2_Load_Value_Fn'] = load
    add_lme_leg(cfg)
    if hmm_override is not None:
        emd = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
        emd.setdefault('Price Models', {})[HMM_KEY] = hmm_override
    return cfg


def run_one(cfg, name, run_dir):
    json.dump(cfg, open(os.path.join(run_dir, name + '.json'), 'w'), indent=1, default=str)
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), name + '.json'))
    _, result = cx.run_job()
    diag = (result.evaluation_summary or {}).get('diagnostics') or {}
    json.dump(diag, open(os.path.join(run_dir, f'diag_{name}.json'), 'w'), indent=1, default=str)
    return diag


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    train_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    eval_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 777
    prof = PROFILES[profile]
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'daily_runs', f'{stamp}_ood_stress_{profile}_s{train_seed}')
    os.makedirs(run_dir, exist_ok=True)
    template = json.load(open(FIXTURE))
    base_hmm = json.load(open(CALIBRATED))['MarketData']['Price Models'][HMM_KEY]
    ckpt = os.path.abspath(os.path.join(run_dir, 'value_fn.pt'))

    logging.info('=== TRAIN (calibrated world, seed=%d, %s) ===', train_seed, profile)
    train_diag = run_one(build_cfg(template, prof, train_seed, save=ckpt), 'train', run_dir)

    rows = []
    for stress in ('control', 'vol_x1.5', 'bear_drift', 'crash_tilt'):
        logging.info('=== EVAL frozen policy under %s (seed=%d) ===', stress, eval_seed)
        cfg = build_cfg(template, prof, eval_seed, load=ckpt,
                        hmm_override=stressed_hmm(base_hmm, stress))
        diag = run_one(cfg, f'eval_{stress}', run_dir)
        v = diag.get('verdict') or {}
        g, t, n = v.get('greedy') or {}, v.get('textbook') or {}, v.get('nohedge') or {}
        rows.append({'stress': stress,
                     'g_u': g.get('u_mean'), 'g_p5': g.get('wT_p5'), 'g_cvar5': g.get('wT_cvar5'),
                     'tb_u': t.get('u_mean'), 'tb_p5': t.get('wT_p5'), 'tb_cvar5': t.get('wT_cvar5'),
                     'nh_u': n.get('u_mean'), 'nh_p5': n.get('wT_p5'), 'nh_cvar5': n.get('wT_cvar5'),
                     'mean_abs_q': v.get('greedy_mean_abs_q')})

    print('\n===== OOD STRESS GATE (%s, train_seed=%d eval_seed=%d) =====' % (profile, train_seed, eval_seed))
    print(f"train V_0={train_diag.get('V_0'):.4f} OOS u={((train_diag.get('verdict') or {}).get('greedy') or {}).get('u_mean'):.4f}")
    print(f"{'stress':<12}{'greedy u/p5/cvar5':>32}{'textbook u/p5/cvar5':>32}{'nohedge u':>11}")
    for r in rows:
        print(f"{r['stress']:<12}{r['g_u']:>10.4f}{r['g_p5']:>11.0f}{r['g_cvar5']:>11.0f}"
              f"{r['tb_u']:>10.4f}{r['tb_p5']:>11.0f}{r['tb_cvar5']:>11.0f}{r['nh_u']:>11.4f}")

    with open(os.path.join(run_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
