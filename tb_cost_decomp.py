"""EXACT net-of-cost decomposition via fixed-schedule replay (throwaway; no commit).

For a given (cost, band) roll, capture the frozen greedy decision path (greedy_q_traj) and the
built bundle+runtime, then REPLAY that exact integer trade schedule through a BundleStepper under
runtime-accounting VARIANTS that toggle individual cost components. Because IM funding and the
calendar rebate are realized-only (they do NOT feed the cost-aware argmax) and vol_t is sourced
from the bundle (not runtime), replaying the SAME schedule under a mutated runtime isolates each
component EXACTLY on the true decision path. Uses only framework primitives (BundleStepper), no
framework edits, no monkey-patching.

Components (all in $/oz, volume=2500), on the realistic-BASE decision path:
  net_full      A = replay(spread=realistic, cal=on,  IM=on)      [== diag greedy, validated]
  IM_drag       B - A       B = replay(spread=realistic, cal=on,  IM=off)
  cal_rebate    B - C       C = replay(spread=realistic, cal=off, IM=off)   (credit, +)
  turnover_tot  D - C       D = replay(spread=0,         cal=off, IM=off)   (gross)
  turnover_Mi   D - Mi      Mi = replay(only instrument i spread on, cal/IM off)
STEP-0 mode: flat10 fixed-decision realized cost = replay(spread=0) - net_full  (>0 => NET).
"""
import os, sys, json, copy, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import riskflow as rf
assert 'PycharmProjects' in rf.__file__, f'wrong riskflow: {rf.__file__}'

from production_walk_forward import build_corrected_archive, build_deal_config, observed_scenario_npz
from production_solver import apply_config
from tb_garch_corridor import (ROOT, ARCHIVE, TEMPLATE, WF, SEEDS, SFX,
                               build_month_map, link_checkpoints)
from tb_train_in_corridor import trade_date_of
from tb_cost_frontier import cost_blocks

from riskflow.hedge_bundle import BundleStepper

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
VOL = 2500.0


def run_full(cfg, name='roll'):
    """Run a solve_hedge job; return the HedgeRuntimeExecutionResult (bundle+runtime+summary)."""
    cx = rf.Context()
    cx.load_json((json.dumps(cfg, default=str), name + '.json'))
    _, result = cx.run_job()
    return result


def build_roll_cfg(month, band, block, arch, mmap, run_dir):
    """Reproduce one_trade's ROLL config with the cost block overlaid on the Evaluator."""
    src, md = mmap[month]
    link_checkpoints(src, month, run_dir)
    trade_date = trade_date_of(month)
    template = json.load(open(TEMPLATE))
    template['Calc']['Calculation']['Hedging_Problem']['Evaluator'].update(copy.deepcopy(block))
    cfg, info = build_deal_config(template, arch, trade_date, md, 8.0, VOL,
                                  delta_corridor=band, spot_model='garch')
    ckpts = [os.path.abspath(os.path.join(run_dir, f'value_fn_{month}{SFX}_s{s}.pt')) for s in SEEDS]
    obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{month}{SFX}.npz'))
    observed_scenario_npz(arch, trade_date, obs_npz)
    roll = apply_config(copy.deepcopy(cfg), batch=1, seed=SEEDS[0], load=ckpts,
                        stepper_rollout=True, randomize_initial_state=False)
    roll['Calc']['Calculation']['Inner_Sub_Batch'] = 512
    roll['Calc']['Calculation']['Observed_Scenario'] = obs_npz
    return roll


def variant_runtime(runtime, **acc_overrides):
    """Shallow-copy runtime with an overridden accounting sub-dict (bundle untouched)."""
    rt = dict(runtime)
    acc = dict(runtime['accounting'])
    acc.update(acc_overrides)
    rt['accounting'] = acc
    return rt


def replay_net(bundle, runtime, q_map, hedges):
    """Drive a BundleStepper along the observed path, forcing the recorded greedy book at each
    swept decision step (delta = target - current, integer-rounded by the env exactly as the
    live roll did). Returns realized net wealth (float, batch=1)."""
    hist = bundle.initial_time_index
    stepper = BundleStepper(bundle, runtime)
    last = None
    while not stepper.done:
        t = stepper.time_index - hist
        if stepper.is_decision_step and t in q_map:
            cur = stepper._state['positions']
            B = next(iter(cur.values())).shape[0]
            tgt = q_map[t]
            delta = {n: torch.full((B,), float(tgt[j]), device=cur[n].device) - cur[n]
                     for j, n in enumerate(hedges)}
            last = stepper.step(delta)
        else:
            last = stepper.step(None)
    return float((last['transition_pnl_excess'] + last['transition_liability_value']).sum())


def spec_only(spec, keep):
    """Copy a bid_offer spread spec keeping ONLY instrument `keep`'s per-instrument bps (others 0)."""
    s = copy.deepcopy(spec)
    s['per_instrument'] = {k: (v if k == keep else 0.0) for k, v in s['per_instrument'].items()}
    return s


def decompose(month, band, arch, mmap, base):
    run_dir = os.path.join(base, f'run_decomp_{"free" if band is None else int(round(band*100)):}')
    os.makedirs(run_dir, exist_ok=True)
    block = cost_blocks()['base']
    roll = build_roll_cfg(month, band, block, arch, mmap, run_dir)
    res = run_full(roll, f'decomp_{month}')
    bundle, runtime = res.bundle, res.runtime
    diag = (res.evaluation_summary or {}).get('diagnostics') or {}
    sv = diag.get('stepper_verdict') or {}
    hedges = list(runtime['names']['hedges'])
    q_traj, q_t = sv['greedy_q_traj'], sv['greedy_q_t']
    q_map = {int(t): q for t, q in zip(q_t, q_traj)}
    diag_greedy = sv['greedy']['wT_mean']
    spec = runtime['accounting']['bid_offer_spread_spec']

    A = replay_net(bundle, runtime, q_map, hedges)                                   # full
    B = replay_net(bundle, variant_runtime(runtime, im_funding_spread_bps=0.0), q_map, hedges)
    C = replay_net(bundle, variant_runtime(runtime, im_funding_spread_bps=0.0,
                                           roll_as_calendar_spread=False), q_map, hedges)
    D = replay_net(bundle, variant_runtime(runtime, im_funding_spread_bps=0.0,
                                           roll_as_calendar_spread=False,
                                           bid_offer_spread_spec=None,
                                           bid_offer_spread_bps=0.0), q_map, hedges)
    Mi = {}
    for n in hedges:
        Mi[n] = replay_net(bundle, variant_runtime(
            runtime, im_funding_spread_bps=0.0, roll_as_calendar_spread=False,
            bid_offer_spread_spec=spec_only(spec, n)), q_map, hedges)

    row = {'tag': month, 'band': ('free' if band is None else round(band, 2)),
           'diag_greedy_oz': round(diag_greedy / VOL, 4),
           'replay_full_oz': round(A / VOL, 4),
           'im_drag_oz': round((B - A) / VOL, 4),
           'cal_rebate_oz': round((B - C) / VOL, 4),
           'turnover_tot_oz': round((D - C) / VOL, 4),
           'gross_oz': round(D / VOL, 4)}
    for n in hedges:
        row[f'turnover_{n}_oz'] = round((D - Mi[n]) / VOL, 4)
    logging.info('DECOMP %s band=%s: full=%.3f (diag=%.3f) turnover_tot=%.3f [%s] cal=%.3f im=%.3f',
                 month, row['band'], row['replay_full_oz'], row['diag_greedy_oz'],
                 row['turnover_tot_oz'],
                 ' '.join(f'{n}={row[f"turnover_{n}_oz"]:.3f}' for n in hedges),
                 row['cal_rebate_oz'], row['im_drag_oz'])
    return row


def step0_flat_cost(month, arch, mmap, base):
    """Fixed-decision realized cost of the flat-10bps book (free band): replay flat10 decisions
    with the realized spread zeroed. gross - net = the cost the frozen decisions actually paid."""
    run_dir = os.path.join(base, 'run_step0_flat_free')
    os.makedirs(run_dir, exist_ok=True)
    block = cost_blocks()['flat10']
    roll = build_roll_cfg(month, None, block, arch, mmap, run_dir)
    res = run_full(roll, f'step0_{month}')
    bundle, runtime = res.bundle, res.runtime
    sv = ((res.evaluation_summary or {}).get('diagnostics') or {}).get('stepper_verdict') or {}
    hedges = list(runtime['names']['hedges'])
    q_map = {int(t): q for t, q in zip(sv['greedy_q_t'], sv['greedy_q_traj'])}
    net = replay_net(bundle, runtime, q_map, hedges)
    gross = replay_net(bundle, variant_runtime(runtime, bid_offer_spread_bps=0.0), q_map, hedges)
    row = {'tag': month, 'net_oz': round(net / VOL, 4), 'gross_oz': round(gross / VOL, 4),
           'realized_cost_oz': round((gross - net) / VOL, 4)}
    logging.info('STEP0 %s (flat10, fixed decisions): net=%.3f gross=%.3f realized_cost=%.3f/oz',
                 month, row['net_oz'], row['gross_oz'], row['realized_cost_oz'])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['decomp', 'step0'], default='decomp')
    ap.add_argument('--bands', nargs='*', default=['0.15', '0.60'])
    ap.add_argument('--months', nargs='*', default=None)
    ap.add_argument('--base', default=os.path.join(WF, 'net_of_cost'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    os.makedirs(args.base, exist_ok=True)

    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    mmap = build_month_map()
    from tb_garch_corridor import month_list
    months = args.months or (month_list('2020-01', 24) + month_list('2022-01', 24))

    rows = []
    if args.mode == 'step0':
        for m in months:
            rows.append(step0_flat_cost(m, arch, mmap, args.base))
    else:
        for band in args.bands:
            b = None if band == 'free' else float(band)
            for m in months:
                rows.append(decompose(m, b, arch, mmap, args.base))
    out = args.out or os.path.join(args.base, f'decomp_{args.mode}.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    logging.info('WROTE %s (%d rows)', out, len(rows))


if __name__ == '__main__':
    main()
