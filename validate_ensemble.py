"""ENSEMBLE-ARGMAX validation: train K shipping-config value functions (different seeds =
different outer worlds + different net inits), freeze them, then on ONE fresh eval world
compare the frozen SINGLE policy vs the frozen ENSEMBLE policy (argmax over the mean of K
continuations, each member in its own standardization frame).

Mechanism: selection noise in the argmax (winner's curse) is the measured tail lever
(antithetic, n_inner); averaging K independently-fitted value fns is the classic cross-fit
reduction of what remains. Paired comparison: identical eval paths for single vs ensemble.

Usage: python validate_ensemble.py <profile> <eval_seed> [train_seeds...]
JSON-is-the-contract: load_json + run_job only.
"""
import copy
import datetime
import json
import logging
import os
import sys

import riskflow as rf

from sweep_levers import BASE, build_cfg

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

FIXTURE_TEMPLATE_KEYS = dict(BASE, inner=64, levels=9)   # the shipping config


def run_one(cfg, name, run_dir):
    json.dump(cfg, open(os.path.join(run_dir, name + '.json'), 'w'), indent=1, default=str)
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), name + '.json'))
    _, result = cx.run_job()
    diag = (result.evaluation_summary or {}).get('diagnostics') or {}
    json.dump(diag, open(os.path.join(run_dir, f'diag_{name}.json'), 'w'), indent=1, default=str)
    return diag


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else 'ship'   # 'ship' = shipping levers
    eval_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    train_seeds = [int(s) for s in sys.argv[3:]] or [1234, 7, 42]
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'daily_runs', f'{stamp}_ensemble_e{eval_seed}')
    os.makedirs(run_dir, exist_ok=True)
    template = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'tests', 'fixtures', 'policy_test_simulate_only.json')))
    p = dict(FIXTURE_TEMPLATE_KEYS)

    ckpts = []
    for s in train_seeds:
        cfg = build_cfg(template, p, s)
        ck = os.path.abspath(os.path.join(run_dir, f'value_fn_s{s}.pt'))
        cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Save_Value_Fn'] = ck
        logging.info('=== TRAIN member seed=%d ===', s)
        run_one(cfg, f'train_s{s}', run_dir)
        ckpts.append(ck)

    results = {}
    for name, load in (('single', ckpts[0]), ('ensemble', ckpts)):
        cfg = build_cfg(template, p, eval_seed)
        cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Load_Value_Fn'] = load
        logging.info('=== EVAL %s on fresh world seed=%d ===', name, eval_seed)
        diag = run_one(cfg, f'eval_{name}', run_dir)
        results[name] = diag.get('verdict') or {}

    print('\n===== ENSEMBLE vs SINGLE (frozen, fresh world seed=%d, K=%d) =====' % (
        eval_seed, len(ckpts)))
    print(f"{'policy':<10}{'u':>9}{'wT_mean':>12}{'p5':>11}{'cvar5':>11}")
    for name in ('single', 'ensemble'):
        g = results[name].get('greedy') or {}
        print(f"{name:<10}{g.get('u_mean'):>9.4f}{g.get('wT_mean'):>12.0f}"
              f"{g.get('wT_p5'):>11.0f}{g.get('wT_cvar5'):>11.0f}")
    tb = results['single'].get('textbook') or {}
    print(f"{'textbook':<10}{tb.get('u_mean'):>9.4f}{tb.get('wT_mean'):>12.0f}"
          f"{tb.get('wT_p5'):>11.0f}{tb.get('wT_cvar5'):>11.0f}")
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
