"""(B) A/B validation of the antithetic inner-MC (Inner_Antithetic) on the real platinum
regime deal — same fixture/profile/seed as validate_regime_kappa.py, crossing
Inner_Antithetic {No, Yes} with kappa {0, 0.5}.

Toy evidence (diffml_hedge_hmm.py): sign-flipping the inner emission draws ~halves the
label+argmax variance of E[C], collapsing the winner's curse -> better OOS tail at fixed
inner budget. Production wrinkle: inner draws are Sobol QMC; the port folds the sequence
with its mirror (z, -z pairs on the inner axis), regime uniforms stay iid.

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

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tests', 'fixtures', 'policy_test_simulate_only.json')
# default grid; override with argv[3] as comma-separated Anti:kappa pairs, e.g. "Yes:0.25,No:0.25"
VARIANTS = [('No', 0.0), ('Yes', 0.0), ('No', 0.5), ('Yes', 0.5)]

PROFILES = {
    'smoke': dict(batch=48, inner=8, t_min=100, iters=30),
    'full': dict(batch=192, inner=32, t_min=90, iters=50),
    'deep': dict(batch=256, inner=32, t_min=60, iters=60),
}


def build_cfg(template, antithetic, kappa, prof, seed):
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    # a solve is a stream: fit batches, then a held-out one. (B, 1)+OOS 0.5 -> (B/2, 2)
    calc['Batch_Size'], calc['Simulation_Batches'] = prof['batch'] // 2, 2
    calc['Inner_Sub_Batch'] = prof['inner']
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = antithetic
    calc['Random_Seed'] = seed
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': prof['t_min'],
        'DiffV2_Fit_Iters': prof['iters'],
        'DiffV2_Risk_Kappa': kappa,
    }
    return cfg


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    variants = ([(a, float(k)) for a, k in (v.split(':') for v in sys.argv[3].split(','))]
                if len(sys.argv) > 3 else VARIANTS)
    prof = PROFILES[profile]
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'daily_runs', f'{stamp}_inner_antithetic_{profile}_s{seed}')
    os.makedirs(run_dir, exist_ok=True)
    template = json.load(open(FIXTURE))

    rows = []
    for anti, kappa in variants:
        name = f'anti{anti}_k{kappa:g}'
        cfg = build_cfg(template, anti, kappa, prof, seed)
        json.dump(cfg, open(os.path.join(run_dir, name + '.json'), 'w'), indent=1, default=str)
        logging.info('=== Inner_Antithetic=%s kappa=%g (%s, seed=%d) ===', anti, kappa, profile, seed)
        cx = rf.Context()
        cx.load_json((json.dumps(cfg), name + '.json'))
        _, result = cx.run_job()
        diag = (result.evaluation_summary or {}).get('diagnostics') or {}
        v = diag.get('verdict') or {}
        g = v.get('greedy') or {}
        rows.append({
            'antithetic': anti, 'kappa': kappa, 'V_0': diag.get('V_0'),
            'max_abs_Y_boot': diag.get('max_abs_Y_boot'),
            'u_mean': g.get('u_mean'), 'wT_mean': g.get('wT_mean'),
            'wT_p5': g.get('wT_p5'), 'wT_cvar5': g.get('wT_cvar5'),
            'mean_abs_q': v.get('greedy_mean_abs_q'),
        })
        json.dump(diag, open(os.path.join(run_dir, f'diag_{name}.json'), 'w'), indent=1, default=str)

    print('\n===== ANTITHETIC A/B (%s, seed=%d) =====' % (profile, seed))
    print(f"{'variant':<22}{'u_mean':>10}{'wT_mean':>12}{'wT_p5':>12}{'wT_cvar5':>12}{'V_0':>9}")
    for r in rows:
        print(f"anti={r['antithetic']:<4} k={r['kappa']:<10g}{r['u_mean']:>10.4f}{r['wT_mean']:>12.0f}"
              f"{r['wT_p5']:>12.0f}{r['wT_cvar5']:>12.0f}{r['V_0']:>9.4f}")

    with open(os.path.join(run_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
