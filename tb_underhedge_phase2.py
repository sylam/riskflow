"""UNDER-HEDGE PHASE 2 -- ROLL-LAYER probes, EVAL-ONLY on frozen control checkpoints (tb_, no commit).

Phase 1 falsified every TRAINING-side lever (objective aversion 2->100 moves u_pre only +0.02..0.06;
Huber_Delta / selection n_inner / grid are roll-inert). NEW PRIME SUSPECT = the ROLL LAYER: the roll
argmax ranks one-step E[u(W1)] under a utility scale c calibrated to TERMINAL wealth spread (~$1M), so
one-step wealth moves (~$10k) sit in the near-LINEAR region of the utility (x=W1/c~0.01) => u~identity
=> E[u(W1)]~E[W1] => martingale world + cost-aware argmax => cost-dominated => rationally under-trades
AT ANY AVERSION. This driver rolls the FROZEN control checkpoint (no training, ~seconds) under roll-time
overrides to test that mechanism directly.

PROBES (per month, on control_<m>/value_fn_<m>_garch_s7.pt):
  R0_ctrl_repro  base roll (scale x1, cost-aware Yes, bid-offer 10bps, roll-inner 512) -- MUST reproduce
                 the phase-1 control u_pre exactly (harness-parity sanity; else the phase-2 roll is wrong).
  R1_scale10     roll utility scale /10   -- one-step curvature partially visible. PRED: u_pre RISES.
  R1_scale100    roll utility scale /100  -- one-step curvature fully visible.     PRED: u_pre RISES more.
  R2a_costoff    DiffV2_Cost_Aware_Argmax=No (cost out of the DECISION, still charged in realized P&L).
                 PRED: u_pre RISES if cost-domination binds the argmax.
  R2b_kappa0     Bid_Offer_Spread_Bps=0 (cost out of decision AND realized). PRED: u_pre RISES if cost binds.
  R3_inner16     roll-inner 16 (the CORRECT H2 -- the ROLL's own inner-MC). Does roll-inner move the u_pre
                 LEVEL (not just dispersion)?  (roll-inner 512 == R0, so 512 is the control point.)
  R4_combo       scale/100 + cost-aware No + bid-offer 0. PRED: if u_pre -> ramp(~1.0), MECHANISM CONFIRMED
                 (under-hedge = myopic roll + terminal-scale utility + costs); fix space is roll-time.

Utility-scale override is via CHECKPOINT SURGERY (the load path overwrites runtime utility_scale from the
checkpoint, hedge_solver.py:1148 -- config Utility_Scale_Explicit is discarded on load). A scaled COPY of
the frozen artifact is a data transform (like the driftless-md splice), not a framework edit/monkey-patch.
ENGAGEMENT PROOF is OUTCOME-BASED: the roll's logged "utility_scale" differs (R1/R4), and u_pre/greedy/churn
move; R0 must equal control. All eval-only (no checkpoint is retrained; mtime-gated).
"""
import argparse
import copy
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

import production_solver as ps
import production_walk_forward as pwf
from production_walk_forward import (build_corrected_archive, build_deal_config,
                                     observed_scenario_npz, pf_bound)
from production_solver import apply_config, run

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, 'data', 'pl_exp.csv')
TEMPLATE = os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
BASE = os.path.join(WF, 'underhedge_probes')
P2 = os.path.join(BASE, 'phase2')
CS = 50.0
SFX = '_garch'

# (name, utility_scale divisor, DiffV2_Cost_Aware_Argmax, Bid_Offer_Spread_Bps, roll_inner)
PROBES = [
    ('R0_ctrl_repro', 1,   'Yes', 10.0, 512),
    ('R1_scale10',    10,  'Yes', 10.0, 512),
    ('R1_scale100',   100, 'Yes', 10.0, 512),
    ('R2a_costoff',   1,   'No',  10.0, 512),
    ('R2b_kappa0',    1,   'Yes', 0.0,  512),
    ('R3_inner16',    1,   'Yes', 10.0, 16),
    ('R4_combo',      100, 'No',  0.0,  512),
]


def trade_date_of(m):
    return (pd.Timestamp(f'{m[:4]}-{m[4:]}-01') + pd.offsets.BDay(0)).normalize()


def f0_of(td):
    a = (td + pd.offsets.MonthBegin(3)).normalize()
    return (pd.bdate_range(a, (a + pd.offsets.MonthEnd(0)).normalize())[0] - td).days


def u_pre_of(diag, f0):
    sv = diag.get('stepper_verdict') or {}
    q = np.array(sv.get('greedy_q_traj') or [])
    t = np.array(sv.get('greedy_q_t') or [])
    if q.ndim != 2 or not len(t):
        return None
    u = np.abs(q.sum(1)) / CS
    pre = t < f0
    return round(float(u[pre].mean()), 4) if pre.any() else None


def scaled_ckpt(src, divisor, dst):
    """Write a COPY of the frozen checkpoint with utility_scale/divisor (all else byte-identical).
    Returns the new scale. divisor==1 -> return src unchanged (no copy)."""
    if divisor == 1:
        return src, None
    ck = torch.load(src, map_location='cpu', weights_only=False)
    new_scale = float(ck['utility_scale']) / divisor
    ck['utility_scale'] = new_scale
    torch.save(ck, dst)
    return dst, new_scale


def run_probe(month, probe, arch, md, trade_date, f0):
    name, div, cost_aware, bid_offer, roll_inner = probe
    row_path = os.path.join(P2, f'tb_row_{name}_{month}.json')
    if os.path.exists(row_path):
        logging.info('P2 %s/%s SKIP (sidecar exists)', name, month)
        return json.load(open(row_path))

    run_dir = os.path.join(P2, f'{name}_{month}')
    os.makedirs(run_dir, exist_ok=True)
    ctrl_ckpt = os.path.abspath(os.path.join(BASE, f'control_{month}', f'value_fn_{month}{SFX}_s7.pt'))
    assert os.path.exists(ctrl_ckpt), f'missing frozen control checkpoint {ctrl_ckpt}'
    mt0 = os.path.getmtime(ctrl_ckpt)

    ckpt, new_scale = scaled_ckpt(ctrl_ckpt, div, os.path.join(run_dir, 'value_fn_scaled.pt'))

    # deal cfg at base objective (unfenced); override Evaluator bid-offer for R2b/R4
    cfg, info = build_deal_config(json.load(open(TEMPLATE)), arch, trade_date, md, 8.0, 2500.0,
                                  delta_corridor=None, spot_model='garch')
    cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']['Bid_Offer_Spread_Bps'] = bid_offer

    obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{month}{SFX}.npz'))
    observed_scenario_npz(arch, trade_date, obs_npz)
    roll = apply_config(copy.deepcopy(cfg), batch=1, seed=7, load=[ckpt],
                        stepper_rollout=True, randomize_initial_state=False)
    rc = roll['Calc']['Calculation']
    rc['Inner_Sub_Batch'] = roll_inner
    rc['Observed_Scenario'] = obs_npz
    rc['Hedging_Problem']['Solver']['DiffV2_Cost_Aware_Argmax'] = cost_aware  # roll-time cost knob
    logging.info('=== P2 %s/%s: scale/%d(=%.6g) cost_aware=%s bid_offer=%.1f roll_inner=%d ===',
                 name, month, div, new_scale if new_scale else 922786.26, cost_aware, bid_offer, roll_inner)
    rdiag = run(roll, f'p2_{name}_{month}')
    json.dump(rdiag, open(os.path.join(run_dir, f'diag_{month}{SFX}.json'), 'w'), indent=1, default=str)

    assert os.path.getmtime(ctrl_ckpt) == mt0, 'control checkpoint was REWRITTEN -- not eval-only!'

    sv = rdiag.get('stepper_verdict') or {}
    gr = (sv.get('greedy') or {}).get('wT_mean')
    nh = (sv.get('nohedge') or {}).get('wT_mean')
    q = np.array(sv.get('greedy_q_traj') or [[0.0]])
    bound = pf_bound(arch, trade_date, info['mats'], info['pay'])
    u_pre = u_pre_of(rdiag, f0)
    out = {'probe': name, 'month': month, 'is_loser': month == '202105',
           'scale_div': div, 'roll_scale': (new_scale if new_scale else 922786.26),
           'cost_aware': cost_aware, 'bid_offer': bid_offer, 'roll_inner': roll_inner,
           'u_pre': u_pre,
           'greedy_usd_oz': None if gr is None else round(gr / 2500.0, 2),
           'nohedge_usd_oz': None if nh is None else round(nh / 2500.0, 2),
           'churn': round(float(np.abs(np.diff(q, axis=0)).sum()), 1),
           'bound_pass': (None if (gr is None or nh is None)
                          else bool(gr / 2500.0 <= nh / 2500.0 + bound + 1e-6)),
           'q_max_l1_pre': (float(np.abs(np.array(sv['greedy_q_traj'])[
               np.array(sv['greedy_q_t']) < f0]).sum(1).max() / CS) if q.ndim == 2 and len(q) > 1 else None)}
    tmp = row_path + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, row_path)
    logging.info('P2 %s/%s DONE: u_pre=%s greedy=%s churn=%s roll_scale=%.6g cost_aware=%s bid=%.1f PASS=%s',
                 name, month, u_pre, out['greedy_usd_oz'], out['churn'], out['roll_scale'], cost_aware,
                 bid_offer, out['bound_pass'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', required=True)
    ap.add_argument('--probes', nargs='*', default=None)
    args = ap.parse_args()
    os.makedirs(P2, exist_ok=True)

    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    md = os.path.abspath(os.path.join(BASE, 'md', args.month, f'md_{args.month}_garch.json'))
    assert os.path.exists(md), f'missing cached driftless md {md} (run phase 1 first)'
    td = trade_date_of(args.month)
    f0 = f0_of(td)
    probes = [p for p in PROBES if (args.probes is None or p[0] in args.probes)]
    logging.info('PHASE2 month=%s f0=%d n=%d CUDA=%s riskflow=%s',
                 args.month, f0, len(probes), os.environ.get('CUDA_VISIBLE_DEVICES'), riskflow.__file__)
    for probe in probes:
        run_probe(args.month, probe, arch, md, td, f0)
    logging.info('PHASE2 month %s COMPLETE (%d probes)', args.month, len(probes))


if __name__ == '__main__':
    main()
