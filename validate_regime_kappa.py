"""(A) Regime-deal validation of RISK_KAPPA + belief-MDP on the REAL platinum problem.

The canonical fixture already simulates PLATINUM_LME under the Baum-Welch-calibrated
3-regime MarkovHMMSpotModel (tests/fixtures/data/MarketDataRF_platinum_calibrated.json),
with the outer forward belief filter feeding the value function's market state and
Inner_Belief_Filter (default True) keeping the bootstrap honest. This script asks the
question Port #1 left open: does downside-dispersion action selection (DiffV2_Risk_Kappa)
actually protect the tail on a deal with a genuine regime-lag tail to protect?

Per kappa: solve_hedge on the fixture (identical seed/paths across kappas), read the OOS
verdict {u_mean, wT_mean, wT_p5, wT_cvar5} for greedy vs textbook vs no-hedge. textbook /
no-hedge are kappa-independent -> printed once and cross-checked identical across runs
(internal consistency gate). Resolved per-variant configs + summary land in the run dir.

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
KAPPAS = [0.0, 0.5, 1.0]

# smoke: fast plumbing check (test-sized); full: the actual verdict run
PROFILES = {
    'smoke': dict(batch=48, inner=8, t_min=100, iters=30),
    'full': dict(batch=192, inner=32, t_min=90, iters=50),
}


def build_cfg(template, kappa, prof, seed):
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    # a solve is a stream: fit batches, then a held-out one. (B, 1)+OOS 0.5 -> (B/2, 2)
    calc['Batch_Size'], calc['Simulation_Batches'] = prof['batch'] // 2, 2
    calc['Inner_Sub_Batch'] = prof['inner']
    calc['Inner_MC_Enabled'] = 'Yes'
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


def run_one(cfg, name):
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), name))
    _, result = cx.run_job()
    return (result.evaluation_summary or {}).get('diagnostics') or {}


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    prof = PROFILES[profile]
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'daily_runs', f'{stamp}_regime_kappa_{profile}_s{seed}')
    os.makedirs(run_dir, exist_ok=True)
    template = json.load(open(FIXTURE))

    rows, benchmarks = [], {}
    for kappa in KAPPAS:
        cfg = build_cfg(template, kappa, prof, seed)
        cfg_path = os.path.join(run_dir, f'kappa_{kappa:g}.json')
        json.dump(cfg, open(cfg_path, 'w'), indent=1, default=str)
        logging.info('=== kappa=%g (%s, seed=%d) ===', kappa, profile, seed)
        diag = run_one(cfg, os.path.basename(cfg_path))
        v = diag.get('verdict') or {}
        g = v.get('greedy') or {}
        rows.append({
            'kappa': kappa, 'V_0': diag.get('V_0'), 'bounded': diag.get('bounded'),
            'u_mean': g.get('u_mean'), 'wT_mean': g.get('wT_mean'),
            'wT_p5': g.get('wT_p5'), 'wT_cvar5': g.get('wT_cvar5'),
            'mean_abs_q': v.get('greedy_mean_abs_q'),
        })
        for bench in ('textbook', 'nohedge'):
            prev = benchmarks.setdefault(bench, v.get(bench))
            if prev != v.get(bench):
                logging.warning('CONSISTENCY: %s verdict differs across kappa runs: %s vs %s',
                                bench, prev, v.get(bench))
        json.dump(diag, open(os.path.join(run_dir, f'diag_kappa_{kappa:g}.json'), 'w'),
                  indent=1, default=str)

    # ladder
    print('\n===== OOS VERDICT LADDER (%s, seed=%d) =====' % (profile, seed))
    hdr = f"{'policy':<16}{'u_mean':>10}{'wT_mean':>12}{'wT_p5':>12}{'wT_cvar5':>12}"
    print(hdr)
    for bench in ('nohedge', 'textbook'):
        b = benchmarks.get(bench) or {}
        print(f"{bench:<16}{b.get('u_mean', float('nan')):>10.4f}{b.get('wT_mean', float('nan')):>12.0f}"
              f"{b.get('wT_p5', float('nan')):>12.0f}{b.get('wT_cvar5', float('nan')):>12.0f}")
    for r in rows:
        print(f"greedy k={r['kappa']:<8g}{r['u_mean']:>10.4f}{r['wT_mean']:>12.0f}"
              f"{r['wT_p5']:>12.0f}{r['wT_cvar5']:>12.0f}   V_0={r['V_0']:.4f} q={r['mean_abs_q']}")

    with open(os.path.join(run_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({'benchmarks': benchmarks, 'rows': rows},
              open(os.path.join(run_dir, 'summary.json'), 'w'), indent=1, default=str)
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
