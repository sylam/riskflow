"""UNDER-HEDGE DIFFERENTIAL DIAGNOSIS -- phase 1 discriminating probes (throwaway, tb_ prefix, no commit).

WHY does the platinum hedge policy systematically UNDER-COVER pre-fixing (unfenced u_pre ~0.41-0.51
where the causal delta ramp prescribes 1.0)? Established: it under-covers ~everywhere; the spurious
GARCH drift explained only ~0.024 of the ~0.59 gap (falsified, 4f08450); the corridor fence fixes
coverage mechanically. This driver runs ONE-VARIABLE-AT-A-TIME probes off a fixed base config in the
DRIFTLESS GARCH world of record (Convexity_Correction=Yes via garchify_md), UNFENCED (delta_corridor
=None -- coverage gap is an unfenced phenomenon), batch 2048 / fit-iters 40 / roll-inner 512, seed 7
single-seed (scoping, stated). Months: 202105 (loser) + 202011 (winner) for contrast.

METRIC (mirrors the driftless experiment's u_pre_free): u_pre = mean |sum_i q_i / 50| over the
pre-first-fixing window (greedy_q_t < f0). Also greedy $/oz, churn, bound-PASS.

PROBE MATRIX (base config = production_walk_forward.OBJECTIVE + production_solver.BEST_*):
  control                          -- base config (anchors all deltas)
  H1a_SR0.1  Surplus_Reward 1->0.1 -- KILLER: make surplus ~worthless. If coverage then jumps to ~1,
                                      the under-hedge was RATIONAL upside-chasing (objective, not solver).
  H1b_SR4    Surplus_Reward 1->4   -- coverage should DROP further if H1 (directional).
  H1c_Hub24  Huber_Aversion 6->24  -- more risk-averse: coverage up if H1.
  H1d_Floor40 Floor_Penalty 10->40 -- heavier floor: coverage up if H1.
  H2_inner16  Inner_Sub_Batch 64->16  } selection n_inner (known dominant lever). H2 predicts
  H2_inner256 Inner_Sub_Batch 64->256 } coverage MOVES with n_inner; H1 predicts it does NOT.
  H3_grid21  Training_Action_Grid_Levels_Per_Axis 9->21 -- finer grid (6.25c steps -> ~2.4c).
                                      H3 predicts finer grid RAISES coverage; H1/H2 predict no change.
  H4_bounds0 Position_Bounds_Penalty 0.25->0 AND Per_Instrument_Bounds_Penalty 0.5->0 -- are the soft
                                      bound penalties taxing large books into under-coverage?

Config overrides live in THIS driver (module globals of the driver scripts -- the task's designated
knob points: OBJECTIVE + BEST_*); NO framework edit, NO monkey-patch of riskflow internals, NO edit of
the fixture. Each (month, probe) gets its own run_dir (checkpoint names collide across probes) + sidecar
(idempotent restart). Calibrate+garchify ONCE per month, reused by all 9 probes.
"""
import argparse
import copy
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root first (shadow-import trap)
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

import production_solver as ps
import production_walk_forward as pwf
from production_walk_forward import (build_corrected_archive, one_trade, calibrate, garchify_md,
                                     LME_COL, CME_COL, SOFR_PREFIX)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, 'data', 'pl_exp.csv')
TEMPLATE = os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
BASE = os.path.join(WF, 'underhedge_probes')
CALSRC = os.path.join(ROOT, 'artifacts', 'calibration_config.json')
MDSRC = os.path.join(ROOT, 'data', 'MarketDataRF_platinum.json')
CS = 50.0
SFX = '_garch'

# pristine base configs captured before any probe mutates the driver-module globals
BASE_OBJ = copy.deepcopy(pwf.OBJECTIVE)
BASE_CALC = copy.deepcopy(ps.BEST_CALC)
BASE_SOLVER = copy.deepcopy(ps.BEST_SOLVER)

# probe = (name, objective-override, BEST_CALC-override, BEST_SOLVER-override)
#
# DEAD-KNOB FINDING (phase-1 diagnosis): the DiffSolverV2 objective AsymmetricUtility_Huber reads
# ONLY {Object, Huber_Aversion, Huber_Delta, CARA_Gamma, Utility_Scale_*} (hedge_runtime.py
# construct_hedge_runtime). Surplus_Reward, Floor_Penalty, Power, Expiry_*, Post_Deal_Trade_
# Penalty, Position_Bounds_Penalty, Per_Instrument_Bounds_Penalty are VESTIGIAL RL-era keys with
# ZERO framework references (the RL-era TerminalFloorThenSurplusUtility that consumed them is gone).
# So the ONLY live H1 lever is the utility SHAPE (Huber_Aversion sweep). H1a (Surplus) + H4 (bounds
# penalties) are kept as DEAD-KNOB CONTROLS -- they MUST reproduce control bit-for-bit (proof the
# knob is inert, not the splice). H1's real test = the Huber_Aversion sweep {2, 6(ctrl), 24, 100}:
# higher aversion => quadratic loss penalty steeper => the linear-gain asymmetry that lets a policy
# rationally under-hedge (upside is free) is overwhelmed => coverage should climb toward the ramp.
PROBES = [
    ('control',     {},                                                             {},                        {}),
    ('H1a_SR0.1',   {'Surplus_Reward': 0.1},                                        {},                        {}),   # DEAD-KNOB control (proven == control)
    # --- H1 OBJECTIVE (live utility SHAPE knobs the AsymmetricUtility_Huber path actually reads) ---
    ('H1c_Hub24',   {'Huber_Aversion': 24.0},                                       {},                        {}),   # LIVE: 4x aversion (loss curvature up)
    ('H1e_Hub2',    {'Huber_Aversion': 2.0},                                        {},                        {}),   # LIVE: 1/3 aversion (down)
    ('H1f_Hub100',  {'Huber_Aversion': 100.0},                                      {},                        {}),   # LIVE: extreme aversion (killer)
    ('H1g_Delta025', {'Huber_Delta': 0.25},                                         {},                        {}),   # LIVE: tighter gain/loss crossover (more asymmetric)
    ('H1h_Delta4',  {'Huber_Delta': 4.0},                                           {},                        {}),   # LIVE: wider crossover (nearer symmetric-quadratic)
    # --- H2 selection / H3 grid ---
    ('H2_inner16',  {},                                                             {'Inner_Sub_Batch': 16},   {}),   # LIVE selection lever
    ('H2_inner256', {},                                                             {'Inner_Sub_Batch': 256},  {}),   # LIVE selection lever
    ('H3_grid21',   {},                                                             {},                        {'Training_Action_Grid_Levels_Per_Axis': 21}),  # LIVE grid geometry
    # H4_bounds0 / H1b / H1d DROPPED: Position_Bounds_Penalty, Per_Instrument_Bounds_Penalty,
    # Surplus_Reward, Floor_Penalty are DEAD KEYS (0 framework refs) -- no-ops by construction.
    ('H4_bounds0',  {'Position_Bounds_Penalty': 0.0, 'Per_Instrument_Bounds_Penalty': 0.0}, {},               {}),   # kept in list for schema; NOT queued
]


def trade_date_of(month):
    return (pd.Timestamp(f'{month[:4]}-{month[4:]}-01') + pd.offsets.BDay(0)).normalize()


def f0_of(trade_date):
    avg_start = (trade_date + pd.offsets.MonthBegin(3)).normalize()
    fixings = pd.bdate_range(avg_start, (avg_start + pd.offsets.MonthEnd(0)).normalize())
    return (fixings[0] - trade_date).days, fixings


def u_pre_of(diag, f0):
    """mean |total net position / 50| over the pre-first-fixing window (greedy_q_t < f0)."""
    sv = diag.get('stepper_verdict') or {}
    q = np.array(sv.get('greedy_q_traj') or [])
    t = np.array(sv.get('greedy_q_t') or [])
    if q.ndim != 2 or not len(t):
        return None
    u = np.abs(q.sum(1)) / CS
    pre = t < f0
    return round(float(u[pre].mean()), 4) if pre.any() else None


def calib_inputs():
    """Build the corrected calibration inputs once (mirrors production_walk_forward.main)."""
    cdir = os.path.join(BASE, '_calib_inputs')
    os.makedirs(cdir, exist_ok=True)
    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    arch_csv = os.path.abspath(os.path.join(cdir, 'archive_cme.csv'))
    arch.drop(columns=[LME_COL]).to_csv(arch_csv)
    md_src = json.load(open(MDSRC))
    md_src['MarketData']['Model Configuration']['.ModelParams']['modelfilters'] = {
        'CommodityPrice': [[['ID', 'PLATINUM_LME'], 'BasisComposedSpotModel']]}
    md_cal = os.path.abspath(os.path.join(cdir, 'marketdata_corrected.json'))
    json.dump(md_src, open(md_cal, 'w'), indent=1)
    cal_src = json.load(open(CALSRC))
    cal_src['CalibrationConfig']['MarketDataArchiveFile']['name'] = arch_csv
    cal_cfg = os.path.abspath(os.path.join(cdir, 'calibration_config.json'))
    json.dump(cal_src, open(cal_cfg, 'w'), indent=1)
    return arch, md_cal, cal_cfg


def driftless_md_for(month, arch, md_cal, cal_cfg):
    """Calibrate at the trade date (causal, no lookahead) + garchify -> driftless (Convexity=Yes) md.
    Cached per month (skip if already built)."""
    mdir = os.path.join(BASE, 'md', month)
    os.makedirs(mdir, exist_ok=True)
    garch_md = os.path.abspath(os.path.join(mdir, f'md_{month}_garch.json'))
    if os.path.exists(garch_md):
        blk = json.load(open(garch_md))['MarketData']['Price Models']['GARCHSpotModel.PLATINUM_CME']
        assert blk.get('Convexity_Correction') == 'Yes', f'{garch_md} not driftless'
        logging.info('MD %s: reuse cached driftless md (alpha=%.4f beta=%.4f nu=%.2f)',
                     month, blk['Alpha'], blk['Beta'], blk['Nu'])
        return garch_md
    cal_end = trade_date_of(month).strftime('%Y-%m-%d')
    hmm_md = os.path.abspath(os.path.join(mdir, f'md_{month}.json'))
    calibrate(md_cal, cal_cfg, cal_end, hmm_md)
    blk = garchify_md(hmm_md, arch, cal_end, garch_md)
    assert blk.get('Convexity_Correction') == 'Yes'
    return garch_md


def engaged_config(arch, trade_date, md):
    """Build the deal+solver config with the CURRENT driver globals and return the ACTUAL baked
    Objective + the calc/solver knobs the training will use. ENGAGEMENT PROOF (a): reports what
    REACHES the config, not what we intended. build_deal_config + apply_config are pure config
    transforms (no training), so this is exactly the config one_trade will train on."""
    cfg, _ = pwf.build_deal_config(json.load(open(TEMPLATE)), arch, trade_date, md, 8.0, 2500.0,
                                   delta_corridor=None, spot_model='garch')
    train = ps.apply_config(copy.deepcopy(cfg), batch=2048, seed=7,
                            save=os.path.join(BASE, '_scratch.pt'))  # save path unused (no run())
    calc = train['Calc']['Calculation']
    return {'Objective': calc['Hedging_Problem']['Objective'],
            'Inner_Sub_Batch': calc['Inner_Sub_Batch'],
            'Training_Action_Grid_Levels_Per_Axis':
                calc['Hedging_Problem']['Solver']['Training_Action_Grid_Levels_Per_Axis']}


def run_job(month, probe, arch, md, trade_date, f0):
    name, obj_ov, calc_ov, solver_ov = probe
    row_path = os.path.join(BASE, f'tb_row_{name}_{month}.json')
    run_dir = os.path.join(BASE, f'{name}_{month}')
    os.makedirs(run_dir, exist_ok=True)

    # set the driver-module globals for this probe (reconstruct fresh from pristine base each time)
    pwf.OBJECTIVE = {**BASE_OBJ, **obj_ov}
    ps.BEST_CALC = {**BASE_CALC, **calc_ov}
    ps.BEST_SOLVER = {**BASE_SOLVER, **solver_ov}
    # ENGAGEMENT PROOF (a): dump the ACTUAL baked Objective + effective solver knobs to the run dir.
    eng = engaged_config(arch, trade_date, md)
    json.dump(eng, open(os.path.join(run_dir, f'engaged_config_{month}.json'), 'w'), indent=1, default=str)
    obj = eng['Objective']
    logging.info('ENGAGED %s/%s: Huber_Aversion=%s Huber_Delta=%s Surplus_Reward=%s(dead) '
                 'Position_Bounds_Penalty=%s(dead) Inner_Sub_Batch=%s grid=%s',
                 name, month, obj['Huber_Aversion'], obj['Huber_Delta'], obj.get('Surplus_Reward'),
                 obj.get('Position_Bounds_Penalty'), eng['Inner_Sub_Batch'],
                 eng['Training_Action_Grid_Levels_Per_Axis'])

    if os.path.exists(row_path):
        logging.info('JOB %s/%s: SKIP (sidecar exists); engaged-proof re-emitted', name, month)
        return json.load(open(row_path))

    args = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40, seeds=[7],
                              roll_inner=512, delta_corridor=None, spot_model='garch')
    logging.info('=== JOB %s / %s  OBJ_ov=%s CALC_ov=%s SOLVER_ov=%s ===',
                 name, month, obj_ov, calc_ov, solver_ov)
    rec = one_trade(json.load(open(TEMPLATE)), arch, trade_date, md, args, run_dir, month)

    diag = json.load(open(os.path.join(run_dir, f'diag_{month}{SFX}.json')))
    u_pre = u_pre_of(diag, f0)
    sv = diag.get('stepper_verdict') or {}
    q = np.array(sv.get('greedy_q_traj') or [])
    out = {'probe': name, 'month': month, 'is_loser': month == '202105',
           'engaged_huber': obj['Huber_Aversion'], 'engaged_surplus': obj.get('Surplus_Reward'),
           'engaged_pos_bp': obj.get('Position_Bounds_Penalty'),
           'engaged_inner': eng['Inner_Sub_Batch'],
           'engaged_grid': eng['Training_Action_Grid_Levels_Per_Axis'],
           'u_pre': u_pre, 'greedy_usd_oz': rec['greedy_usd_oz'], 'nohedge_usd_oz': rec['nohedge_usd_oz'],
           'churn': rec['churn'], 'bound_pass': rec['bound_pass'], 'bound': rec['pf_bound'],
           'train_u': rec['train_u'], 'V_0': rec['V_0'], 'fair': rec['fair'],
           'q_last_l1': float(np.abs(q[-1]).sum()) if q.ndim == 2 and len(q) else None,
           'q_max_l1_pre': (float(np.abs(np.array(sv['greedy_q_traj'])[
               np.array(sv['greedy_q_t']) < f0]).sum(1).max() / CS)
               if q.ndim == 2 and len(q) else None)}
    tmp = row_path + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, row_path)
    logging.info('JOB %s/%s DONE: u_pre=%s greedy=%s churn=%s PASS=%s train_u=%s',
                 name, month, u_pre, rec['greedy_usd_oz'], rec['churn'], rec['bound_pass'], rec['train_u'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', required=True)
    ap.add_argument('--probes', nargs='*', default=None, help='subset of probe names; default all')
    args = ap.parse_args()
    os.makedirs(BASE, exist_ok=True)

    arch, md_cal, cal_cfg = calib_inputs()
    md = driftless_md_for(args.month, arch, md_cal, cal_cfg)
    trade_date = trade_date_of(args.month)
    f0, _ = f0_of(trade_date)
    probes = [p for p in PROBES if (args.probes is None or p[0] in args.probes)]
    logging.info('UNDERHEDGE PROBES month=%s f0=%d n_probes=%d CUDA=%s riskflow=%s',
                 args.month, f0, len(probes), os.environ.get('CUDA_VISIBLE_DEVICES'), riskflow.__file__)
    for probe in probes:
        run_job(args.month, probe, arch, md, trade_date, f0)
    logging.info('MONTH %s COMPLETE (%d probes)', args.month, len(probes))


if __name__ == '__main__':
    main()
