"""Daily PPO training driver for the platinum hedger.

Defaults are the Phase-4 winner from the HP sweep (artifacts/hp_phase{1,2,3,4}.csv):
  lr            = 3e-4
  reward_scale  = 1e-7
  naked_penalty = 5
  entropy_coef  = 3e-4
  value_coef    = 0.1
  ppo_epochs    = 4
  minibatch     = 128
  batch         = 512
  epochs        = 200      (Phase 4 reached +$737K at 150 still climbing — 200 gives margin)

Usage:
  python train_daily.py                                # all defaults
  python train_daily.py --epochs 300 --tag exp_a       # longer run + tag
  python train_daily.py --reward_scale 3e-7 --tag rs3  # override one knob

Outputs (one timestamped subdir under artifacts/daily_runs/):
  trained_policy.json      — saved policy artifact (loadable for re-eval / production)
  ml_paths.csv             — per-day per-instrument breakdown for the trained policy
  textbook_paths.csv       — same breakdown under the textbook hedge baseline
  summary.csv              — terminal P&L per (policy, case)
  config.json              — exact hyperparameters used (reproducibility)
  training.log             — stdout from the run
"""
import argparse
import json as jsonlib
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import riskflow as rf
from riskflow import torchrl_hedge, calculation as _calc_mod
from riskflow.torchrl_hedge import (
    build_shared_state, step_runtime_state, reward_and_terminal_payoff,
    _decision_time_indices, _optimizer_settings, _ensure_decision_views,
    _last_time_index, _bundle_scenario_dates, _realized_structured_action,
)
from riskflow.structured_policy import save_policy_artifact

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lr',             type=float, default=3.0e-4)
    p.add_argument('--reward_scale',   type=float, default=1.0e-7)
    p.add_argument('--naked_penalty',  type=float, default=5.0)
    p.add_argument('--expiry_penalty', type=float, default=1.0,
                   help='Coef on the exponential-ramp expiry penalty. The penalty per step is '
                        '`coef * exp(threshold - days_to_expiry) * |position notional|`. '
                        'Set 0 to disable (legacy behavior — policy can fake-profit past expiry).')
    p.add_argument('--expiry_threshold_days', type=float, default=4.0,
                   help='Days before expiry at which the exponential ramp starts (multiplier=1).')
    p.add_argument('--entropy_coef',   type=float, default=3.0e-4)
    p.add_argument('--value_coef',     type=float, default=0.1)
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--ppo_epochs',     type=int,   default=4)
    p.add_argument('--minibatch_size', type=int,   default=128)
    p.add_argument('--batch_size',     type=int,   default=512)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--tag',            type=str,   default='',
                   help='Optional tag appended to the output dir name.')
    p.add_argument('--output_root',    type=str,   default='artifacts/daily_runs')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rollout helpers — adapted from diagnose_vs_textbook.py for 3-instrument output
# ---------------------------------------------------------------------------
def rollout_policy(policy, bundle, runtime):
    """Deterministic ML-policy rollout capturing position/trade/price for every
    instrument in the action space. Returns per-decision-step tensors keyed by
    instrument name plus terminal P&L."""
    settings = _optimizer_settings(runtime)
    decision_indices = _decision_time_indices(bundle, settings, epoch=None, evaluation=True)
    decision_set = set(int(i) for i in decision_indices)
    instrument_order = tuple(runtime['names']['action_instruments'])
    state = build_shared_state(bundle, runtime)
    last_idx = _last_time_index(bundle)
    batch_size = int(next(iter(state['positions'].values())).shape[0])
    device = bundle['time_grid_days'].device
    times = []
    positions = {n: [] for n in instrument_order}
    trades    = {n: [] for n in instrument_order}
    prices    = {n: [] for n in instrument_order}
    transition = None
    while int(state['time_index']) < last_idx:
        t = int(state['time_index'])
        if t in decision_set:
            state = _ensure_decision_views(state, bundle, runtime)
            output = policy.sample(state['entity_state'], deterministic=True)
            mapped = _realized_structured_action(
                policy.map_actions(output), state['positions'], runtime,
                batch_size=batch_size, device=device,
            )
            times.append(t)
            for name in instrument_order:
                positions[name].append(state['positions'][name].detach().cpu())
                trades[name].append(mapped['trade_deltas'][name].detach().cpu())
                prices[name].append(state['tradable_values'][name].detach().cpu())
        else:
            mapped = None
        next_state = step_runtime_state(state, mapped, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        state = next_state
    return {
        'times':     times,
        'position':  {n: torch.stack(positions[n], dim=0) for n in instrument_order},
        'trade':     {n: torch.stack(trades[n], dim=0)    for n in instrument_order},
        'price':     {n: torch.stack(prices[n], dim=0)    for n in instrument_order},
        'pnl_excess': transition['pnl_excess'].detach().cpu(),
        'liability':  transition['liability_value'].detach().cpu(),
        'net_pnl':    (transition['pnl_excess'] + transition['liability_value']).detach().cpu(),
    }


def rollout_textbook_3instr(bundle, runtime, hedge_instrument='PL_JUL_2026'):
    """Textbook hedge: short Volume/contract_size of `hedge_instrument` from t=0,
    linearly buy back across business days in the averaging window, then flat. Other
    instruments stay at zero throughout. Returns same shape as rollout_policy so the
    CSV builder can treat them uniformly."""
    settings = _optimizer_settings(runtime)
    decision_indices = _decision_time_indices(bundle, settings, epoch=None, evaluation=True)
    decision_set = set(int(i) for i in decision_indices)
    instrument_order = tuple(runtime['names']['action_instruments'])
    dates = _bundle_scenario_dates(bundle)

    liabilities = runtime.get('liabilities') or {}
    first_liab = next(iter(liabilities.values())) if liabilities else {}
    payments = ((first_liab or {}).get('params') or {}).get('Payments') or {}
    items = payments.get('Items') or []
    first_pmt = items[0] if items else {}
    period_start = pd.Timestamp(first_pmt['Period_Start'])
    period_end = pd.Timestamp(first_pmt['Period_End'])
    tn_idx = next(i for i, d in enumerate(dates) if d >= period_start)

    bus_day_offset = bundle.get('meta', {}).get('business_day')
    tradable_meta = (runtime.get('tradables') or {}).get(hedge_instrument, {}) or {}
    last_trade_date = tradable_meta.get('last_trade_date')
    window_end = pd.Timestamp(last_trade_date) if last_trade_date is not None else period_end
    window_end = min(window_end, period_end)

    def _is_bd(d):
        if bus_day_offset is not None and hasattr(bus_day_offset, 'is_on_offset'):
            return bool(bus_day_offset.is_on_offset(pd.Timestamp(d)))
        return pd.Timestamp(d).weekday() < 5

    buyback_indices = [i for i, d in enumerate(dates)
                       if period_start <= d <= window_end and _is_bd(d)]
    deal_volume = float(first_pmt.get('Volume', 0.0))
    contract_size = float((runtime.get('tradables') or {}).get(hedge_instrument, {})
                          .get('contract_size', 1.0))
    short_at_start = int(round(deal_volume / contract_size))

    target_at_idx = {}
    if buyback_indices:
        N = len(buyback_indices)
        for i, idx in enumerate(buyback_indices):
            u = i / max(N - 1, 1)
            target_at_idx[idx] = -short_at_start * (1.0 - u)

    state = build_shared_state(bundle, runtime)
    last_idx = _last_time_index(bundle)
    batch_size = int(next(iter(state['positions'].values())).shape[0])
    device = bundle['time_grid_days'].device
    deltas_map = runtime.get('policy', {}).get('action_space', {}).get('trade_deltas', {}) or {}
    allowed = sorted(set(int(d) for d in deltas_map.get(hedge_instrument, range(-10, 11))))
    allowed_arr = np.asarray(allowed, dtype=np.int64)

    times = []
    positions = {n: [] for n in instrument_order}
    trades    = {n: [] for n in instrument_order}
    prices    = {n: [] for n in instrument_order}
    transition = None
    while int(state['time_index']) < last_idx:
        t = int(state['time_index'])
        if t in decision_set:
            if t < tn_idx:
                target = -short_at_start
            elif t in target_at_idx:
                target = target_at_idx[t]
            else:
                target = 0.0
            cur = int(round(state['positions'][hedge_instrument][0].item()))
            desired = int(round(target)) - cur
            trade = int(allowed_arr[int(np.argmin(np.abs(allowed_arr - desired)))])
            action = {
                'trade_deltas': {
                    name: torch.full(
                        (batch_size,),
                        float(trade if name == hedge_instrument else 0),
                        dtype=torch.float32, device=device,
                    )
                    for name in instrument_order
                }
            }
            times.append(t)
            for name in instrument_order:
                positions[name].append(state['positions'][name].detach().cpu())
                trades[name].append(action['trade_deltas'][name].detach().cpu())
                prices[name].append(state['tradable_values'][name].detach().cpu())
        else:
            action = None
        next_state = step_runtime_state(state, action, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        state = next_state
    return {
        'times':     times,
        'position':  {n: torch.stack(positions[n], dim=0) for n in instrument_order},
        'trade':     {n: torch.stack(trades[n], dim=0)    for n in instrument_order},
        'price':     {n: torch.stack(prices[n], dim=0)    for n in instrument_order},
        'pnl_excess': transition['pnl_excess'].detach().cpu(),
        'liability':  transition['liability_value'].detach().cpu(),
        'net_pnl':    (transition['pnl_excess'] + transition['liability_value']).detach().cpu(),
    }


# ---------------------------------------------------------------------------
# Per-day cashflow expansion (multi-instrument)
# ---------------------------------------------------------------------------
def expand_per_day(rollout, bundle, runtime):
    """Build per-instrument (T, B) per-day cashflow tensors plus portfolio totals."""
    factors = bundle['factors']
    spot_key = next(k for k in factors if 'CommodityPrice' in k and 'PLATINUM' in k)
    mtm_running = bundle['liability_mtm'].detach().cpu().float()
    spot = factors[spot_key].detach().cpu().float()
    instrument_order = tuple(runtime['names']['action_instruments'])
    spread_bps = float((runtime.get('accounting') or {}).get('bid_offer_spread_bps', 0.0))
    unit_cost = float((runtime.get('accounting') or {}).get('transaction_cost_per_unit', 0.0))
    T, B = mtm_running.shape

    # Forward-fill realised liability cashflow past settlement.
    nonzero = (mtm_running != 0)
    last_nz = (nonzero * torch.arange(T).unsqueeze(1)).max(dim=0).values
    rows = torch.arange(T).unsqueeze(1).expand(T, B)
    fill_mask = rows > last_nz.unsqueeze(0)
    realised = mtm_running.gather(0, last_nz.unsqueeze(0)).expand(T, B)
    mtm = torch.where(fill_mask, realised, mtm_running)

    times = [int(t) for t in rollout['times']]
    per_instr = {}
    for name in instrument_order:
        cs = float(runtime['tradables'][name]['contract_size'])
        fut = bundle['tradables'][name].detach().cpu().float()
        # Forward-fill position from decision steps.
        pos = torch.zeros((T, B)); trd = torch.zeros((T, B))
        cur = torch.zeros(B); j = 0
        rollout_pos = rollout['position'][name]
        rollout_trd = rollout['trade'][name]
        for t in range(T):
            if j < len(times) and t == times[j]:
                trd[t] = rollout_trd[j].cpu().float()
                cur = rollout_pos[j].cpu().float() + rollout_trd[j].cpu().float()
                j += 1
            pos[t] = cur
        trade_cash = -trd * fut * cs
        trade_cost = trd.abs() * (unit_cost + fut * cs * 0.5 * spread_bps * 1.0e-4)
        position_mtm = pos * fut * cs
        cum_cash = trade_cash.cumsum(0)
        cum_cost = trade_cost.cumsum(0)
        per_instr[name] = {
            'fut': fut, 'cs': cs, 'pos': pos, 'trd': trd,
            'trade_cash': trade_cash, 'trade_cost': trade_cost, 'position_mtm': position_mtm,
            'cum_cash': cum_cash, 'cum_cost': cum_cost,
        }
    # Portfolio totals across all instruments.
    portfolio_pos_mtm = sum(p['position_mtm'] for p in per_instr.values())
    portfolio_cum_cash = sum(p['cum_cash'] for p in per_instr.values())
    portfolio_cum_cost = sum(p['cum_cost'] for p in per_instr.values())
    hp = portfolio_cum_cash + portfolio_pos_mtm - portfolio_cum_cost
    total = mtm + hp
    return {
        'spot': spot,
        'spread_bps': spread_bps,
        'per_instr': per_instr,
        'mtm': mtm,
        'portfolio_position_mtm':   portfolio_pos_mtm,
        'portfolio_cum_trade_cash': portfolio_cum_cash,
        'portfolio_cum_trade_cost': portfolio_cum_cost,
        'hedge_portfolio_ex_funding': hp,
        'total_ex_funding_discount': total,
    }


def write_csv(fields, day_strs, policy_label, runtime, csv_path):
    """Per-day rows with all per-instrument breakdowns + portfolio totals.
    5 cases (worst, p5, mean, p95, best) selected by terminal `total_ex_funding_discount`."""
    total = fields['total_ex_funding_discount']
    T, B = total.shape
    sidx = torch.argsort(total[-1])
    cases = {
        'worst': int(sidx[0]),
        'p5':    int(sidx[round(0.05 * (B - 1))]),
        'p95':   int(sidx[round(0.95 * (B - 1))]),
        'best':  int(sidx[-1]),
    }
    instrument_order = tuple(runtime['names']['action_instruments'])
    rows = []

    def _row(t, case, path_idx, getter):
        row = {
            'policy':   policy_label,
            'case':     case,
            'path_idx': int(path_idx),
            'day':      day_strs[t],
            'spot':                            getter(fields['spot'][t]).item(),
            'spread_bps':                      fields['spread_bps'],
        }
        for name in instrument_order:
            p = fields['per_instr'][name]
            row[f'{name}_futures']      = getter(p['fut'][t]).item()
            row[f'{name}_contract_size'] = p['cs']
            row[f'{name}_position']     = float(getter(p['pos'][t]).item())
            row[f'{name}_trade']        = float(getter(p['trd'][t]).item())
            row[f'{name}_trade_cash']   = float(getter(p['trade_cash'][t]).item())
            row[f'{name}_trade_cost']   = float(getter(p['trade_cost'][t]).item())
            row[f'{name}_position_mtm'] = float(getter(p['position_mtm'][t]).item())
        row['portfolio_position_mtm']         = float(getter(fields['portfolio_position_mtm'][t]).item())
        row['portfolio_cum_trade_cash']       = float(getter(fields['portfolio_cum_trade_cash'][t]).item())
        row['portfolio_cum_trade_cost']       = float(getter(fields['portfolio_cum_trade_cost'][t]).item())
        row['hedge_portfolio_ex_funding']     = float(getter(fields['hedge_portfolio_ex_funding'][t]).item())
        row['mtm_ex_post_settle_discount']    = float(getter(fields['mtm'][t]).item())
        row['total_ex_funding_discount']      = float(getter(fields['total_ex_funding_discount'][t]).item())
        return row

    for case_name, idx in cases.items():
        for t in range(T):
            rows.append(_row(t, case_name, idx, lambda x, idx=idx: x[idx]))
    # Mean: cross-path average per day; path_idx = -1.
    for t in range(T):
        rows.append(_row(t, 'mean', -1, lambda x: x.mean()))

    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'_{args.tag}' if args.tag else ''
    out_dir = os.path.join(args.output_root, f'{timestamp}{tag}')
    os.makedirs(out_dir, exist_ok=True)

    # Save the resolved config first thing so a crashed run still leaves a record.
    config_dict = {
        'lr': args.lr, 'reward_scale': args.reward_scale,
        'naked_penalty': args.naked_penalty, 'expiry_penalty': args.expiry_penalty,
        'expiry_threshold_days': args.expiry_threshold_days,
        'entropy_coef': args.entropy_coef,
        'value_coef': args.value_coef, 'epochs': args.epochs,
        'ppo_epochs': args.ppo_epochs, 'minibatch_size': args.minibatch_size,
        'batch_size': args.batch_size, 'seed': args.seed, 'tag': args.tag,
        'timestamp': timestamp,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        jsonlib.dump(config_dict, f, indent=2)
    print(f'Output: {out_dir}\n')
    for k, v in config_dict.items():
        print(f'  {k:<20} {v}')
    print()

    # Pull policy_test JSON and override knobs
    with open('policy_test.py') as f:
        src = f.read()
    m = re.search(r"json\s*=\s*'''(.*?)'''", src, re.DOTALL)
    cfg = jsonlib.loads(m.group(1))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'optimize_policy'
    calc['Simulation_Batches'] = 1
    calc['Batch_Size'] = args.batch_size
    calc['Random_Seed'] = args.seed
    opt = calc['Hedging_Problem']['Optimizer']
    opt['Epochs'] = args.epochs
    opt['PPO_Epochs'] = args.ppo_epochs
    opt['Minibatch_Size'] = args.minibatch_size
    opt['Learning_Rate'] = args.lr
    opt['Reward_Scale'] = args.reward_scale
    opt['Entropy_Coef'] = args.entropy_coef
    opt['Value_Coef'] = args.value_coef
    calc['Hedging_Problem']['Objective']['Naked_Penalty'] = args.naked_penalty
    calc['Hedging_Problem']['Objective']['Expiry_Penalty'] = args.expiry_penalty
    calc['Hedging_Problem']['Objective']['Expiry_Threshold_Days'] = args.expiry_threshold_days

    # Hijack execution to capture the trained policy + bundle + runtime
    holder = {}
    orig_th = torchrl_hedge.run_torchrl_execution
    orig_calc = _calc_mod.run_torchrl_execution

    def capture(bundle, runtime):
        holder['bundle'] = bundle
        holder['runtime'] = runtime
        result = torchrl_hedge.train_torchrl_policy(bundle, runtime)
        holder['policy'] = result['policy']
        holder['result'] = result
        return result

    torchrl_hedge.run_torchrl_execution = capture
    _calc_mod.run_torchrl_execution = capture

    t_start = time.monotonic()
    try:
        cx = rf.Context()
        cx.load_json((jsonlib.dumps(cfg), 'train_daily.json'))
        cx.run_job()
    finally:
        torchrl_hedge.run_torchrl_execution = orig_th
        _calc_mod.run_torchrl_execution = orig_calc
    elapsed = time.monotonic() - t_start

    bundle, runtime, policy = holder['bundle'], holder['runtime'], holder['policy']

    # Save the trained policy artefact for re-evaluation later
    artifact_path = os.path.join(out_dir, 'trained_policy.json')
    save_policy_artifact(policy.to_artifact(), artifact_path)
    print(f'\nTraining done in {elapsed/60:.1f} min')
    print(f'Saved policy: {artifact_path}')

    # Evaluate the trained policy deterministically + textbook on the same bundle
    print('\nRunning ML rollout (deterministic)...')
    ml_rollout = rollout_policy(policy, bundle, runtime)
    print('Running textbook rollout...')
    tb_rollout = rollout_textbook_3instr(bundle, runtime, hedge_instrument='PL_JUL_2026')

    ml_fields = expand_per_day(ml_rollout, bundle, runtime)
    tb_fields = expand_per_day(tb_rollout, bundle, runtime)

    day_strs = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in _bundle_scenario_dates(bundle)]
    ml_path = os.path.join(out_dir, 'ml_paths.csv')
    tb_path = os.path.join(out_dir, 'textbook_paths.csv')
    write_csv(ml_fields, day_strs, 'ml',       runtime, ml_path)
    write_csv(tb_fields, day_strs, 'textbook', runtime, tb_path)
    print(f'\nWrote {ml_path}')
    print(f'Wrote {tb_path}')

    # Summary CSV: terminal P&L per (policy, case)
    def _terminal_summary(label, rollout, fields):
        net = rollout['net_pnl'].numpy()
        total = fields['total_ex_funding_discount'][-1].numpy()
        rows = [
            {'policy': label, 'metric': 'mean',  'net_pnl': float(net.mean()),
             'total_ex_funding': float(total.mean())},
            {'policy': label, 'metric': 'std',   'net_pnl': float(net.std()),
             'total_ex_funding': float(total.std())},
            {'policy': label, 'metric': 'min',   'net_pnl': float(net.min()),
             'total_ex_funding': float(total.min())},
            {'policy': label, 'metric': 'p5',    'net_pnl': float(np.percentile(net, 5)),
             'total_ex_funding': float(np.percentile(total, 5))},
            {'policy': label, 'metric': 'p95',   'net_pnl': float(np.percentile(net, 95)),
             'total_ex_funding': float(np.percentile(total, 95))},
            {'policy': label, 'metric': 'max',   'net_pnl': float(net.max()),
             'total_ex_funding': float(total.max())},
        ]
        return rows

    summary_rows = (_terminal_summary('ml', ml_rollout, ml_fields)
                    + _terminal_summary('textbook', tb_rollout, tb_fields))
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.2f')
    print(f'Wrote {summary_path}')

    # Headline print
    ml_mean = float(ml_rollout['net_pnl'].mean())
    tb_mean = float(tb_rollout['net_pnl'].mean())
    print('\n' + '=' * 60)
    print(f'  ML policy net_pnl mean:      ${ml_mean:>+12,.0f}')
    print(f'  Textbook    net_pnl mean:    ${tb_mean:>+12,.0f}')
    print(f'  ML beats textbook by:        ${ml_mean - tb_mean:>+12,.0f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
