"""Tier-1 lever sweep on the dual-strip platinum hedge — find the settings that maximise
expected utility with what's already built (no new features).

Levers (one-factor-at-a-time from the current best, then combine winners):
  levels  — action-grid Levels_Per_Axis (5 current; 9/11 = finer positions)
  inner   — Inner_Sub_Batch (32 current; 64 = winner's-curse lever, antithetic pairs)
  batch   — outer paths (256 current; 512 = better fit + tighter verdict)
  iters/hidden — convergence check (if these move the verdict, we weren't converged)
  t_min   — window depth (60 current; 30/10 = hedge more of the deal)

Comparability: batch/t_min change the path set / rollout window, so each variant reports
greedy vs ITS OWN textbook+nohedge (edge = greedy_u - textbook_u) alongside raw stats.
Wall-time recorded — 'optimal' accounts for cost.

Usage:  python sweep_levers.py <seed> [variant ...]
  variant = 'name:key=val[,key=val...]' with keys in
            {levels, inner, batch, iters, hidden, t_min}
  no variants -> the standard OFAT battery.
JSON-is-the-contract: load_json + run_job only.
"""
import copy
import csv
import datetime
import json
import logging
import os
import sys
import time

import torch

import riskflow as rf

from validate_cross_market import FIXTURE, add_lme_leg

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

BASE = dict(levels=5, inner=32, batch=256, iters=60, hidden=32, t_min=60,
            lr=2.0e-3, cost_aware=0, random_draws=0, save=0, one_step=1,
            noise=0.15, percol=0)
FLOAT_KEYS = ('lr', 'noise')
BATTERY = [
    'base:',
    'IT120:iters=120',
    'H64:hidden=64',
    'I64:inner=64',
    'W30:t_min=30',
    'B512:batch=512',
    'W10:t_min=10',
    'L9:levels=9',
    'L11:levels=11',
]


def build_cfg(template, p, seed):
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    # a solve is a stream: fit batches, then a held-out one. (B, 1)+OOS 0.5 -> (B/2, 2)
    calc['Batch_Size'], calc['Simulation_Batches'] = p['batch'] // 2, 2
    calc['Inner_Sub_Batch'] = p['inner']
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = 'Yes'
    calc['Inner_Draws'] = 'random' if p['random_draws'] else 'sobol'
    calc['Random_Seed'] = seed
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': p['levels'],
        'Training_Action_Chunk_Size': 64,
        'T_Min': p['t_min'],
        'DiffV2_Fit_Iters': p['iters'],
        'DiffV2_Hidden': p['hidden'],
        'DiffV2_LR': p['lr'],
        'DiffV2_Cost_Aware_Argmax': 'Yes' if p['cost_aware'] else 'No',
        'DiffV2_One_Step_Fork': 'Yes' if p['one_step'] else 'No',
        'DiffV2_Bank_Noise_Frac': p['noise'],
        'DiffV2_Per_Column_Grad_Norm': 'Yes' if p['percol'] else 'No',
    }
    if p['save']:
        hp['Solver']['DiffV2_Save_Value_Fn'] = os.path.abspath(
            os.path.join(p['run_dir'], f"value_fn_s{seed}.pt"))
    add_lme_leg(cfg)
    return cfg


def parse_variant(spec):
    name, _, kv = spec.partition(':')
    p = dict(BASE)
    for pair in filter(None, kv.split(',')):
        k, _, v = pair.partition('=')
        if k not in p:
            raise ValueError(f'unknown lever {k!r} in {spec!r}')
        p[k] = float(v) if k in FLOAT_KEYS else int(v)
    return name, p


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    specs = sys.argv[2:] or BATTERY
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'daily_runs', f'{stamp}_lever_sweep_s{seed}')
    os.makedirs(run_dir, exist_ok=True)
    template = json.load(open(FIXTURE))

    rows = []
    for spec in specs:
        name, p = parse_variant(spec)
        p['run_dir'] = run_dir
        cfg = build_cfg(template, p, seed)
        json.dump(cfg, open(os.path.join(run_dir, name + '.json'), 'w'), indent=1, default=str)
        logging.info('=== %s %s (seed=%d) ===', name, p, seed)
        t0 = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cx = rf.Context()
        cx.load_json((json.dumps(cfg), name + '.json'))
        _, result = cx.run_job()
        secs = time.monotonic() - t0
        peak_gb = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) \
            if torch.cuda.is_available() else 0.0
        diag = (result.evaluation_summary or {}).get('diagnostics') or {}
        v = diag.get('verdict') or {}
        g, tb, nh = v.get('greedy') or {}, v.get('textbook') or {}, v.get('nohedge') or {}
        rows.append({
            'variant': name, **p, 'secs': round(secs, 1), 'peak_gb': peak_gb,
            'V_0': diag.get('V_0'), 'actions': diag.get('action_grid_size'),
            'u': g.get('u_mean'), 'p5': g.get('wT_p5'), 'cvar5': g.get('wT_cvar5'),
            'edge_u': (g.get('u_mean') or 0) - (tb.get('u_mean') or 0),
            'edge_p5': (g.get('wT_p5') or 0) - (tb.get('wT_p5') or 0),
            'edge_cvar5': (g.get('wT_cvar5') or 0) - (tb.get('wT_cvar5') or 0),
            'nh_u': nh.get('u_mean'), 'mean_abs_q': v.get('greedy_mean_abs_q'),
        })
        json.dump(diag, open(os.path.join(run_dir, f'diag_{name}.json'), 'w'),
                  indent=1, default=str)
        logging.info('%s done in %.0fs: u=%.4f edge_u=%.4f', name, secs, rows[-1]['u'],
                     rows[-1]['edge_u'])

    print('\n===== LEVER SWEEP (seed=%d) =====' % seed)
    print(f"{'variant':<8}{'secs':>6}{'gb':>6}{'K':>7}{'u':>9}{'edge_u':>9}{'p5':>10}{'cvar5':>10}"
          f"{'edge_p5':>9}{'edge_cv5':>9}")
    for r in rows:
        print(f"{r['variant']:<8}{r['secs']:>6.0f}{r['peak_gb']:>6.2f}{r['actions']:>7}{r['u']:>9.4f}"
              f"{r['edge_u']:>9.4f}{r['p5']:>10.0f}{r['cvar5']:>10.0f}"
              f"{r['edge_p5']:>9.0f}{r['edge_cvar5']:>9.0f}")

    with open(os.path.join(run_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
