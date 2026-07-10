"""Run the textbook averaging hedge against the standard fixture via `BundleStepper`.

The textbook hedge is a deterministic policy expressed entirely in client code:
short the full leg volume from t=0, then ramp linearly back to zero across the
averaging window (period_start → window_end). Used as a baseline benchmark for
the trained ML policy.

Same shape as train_daily.py — derives cfg from the test fixture, applies CLI
overrides only where the user explicitly passed a flag, drives the framework via
cx.run_job(), then drives the stepper from the result. No internal imports; the
hedge logic only reads runtime / bundle dicts (the public contract).

CSVs written: <out>/textbook_paths.csv, <out>/textbook_summary.csv
"""
import argparse
import json as jsonlib
import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import torch

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'fixtures',
                        'policy_test_simulate_only.json')


def _resolve_config_path(source_job_path: str, referenced_path: Optional[str]) -> Optional[str]:
    if not referenced_path:
        return referenced_path

    normalized = os.path.normpath(str(referenced_path))
    if os.path.isabs(normalized) and os.path.exists(normalized):
        return normalized

    job_dir = os.path.dirname(os.path.abspath(source_job_path))
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for base_dir in (job_dir, repo_root):
        candidate = os.path.normpath(os.path.join(base_dir, normalized))
        if os.path.exists(candidate):
            return candidate

    return referenced_path


def _normalize_external_references(cfg: dict[str, Any], source_job_path: str) -> dict[str, Any]:
    calc = cfg.get('Calc', {})
    merge_market_data = calc.get('MergeMarketData', {})
    if isinstance(merge_market_data, dict):
        merge_market_data['MarketDataFile'] = _resolve_config_path(
            source_job_path, merge_market_data.get('MarketDataFile'))
        merge_market_data['StressedMarketDataFile'] = _resolve_config_path(
            source_job_path, merge_market_data.get('StressedMarketDataFile'))

    if calc.get('CalendDataFile') is not None:
        calc['CalendDataFile'] = _resolve_config_path(source_job_path, calc.get('CalendDataFile'))

    return cfg


def load_job_config(job_path: Optional[str] = None) -> dict[str, Any]:
    resolved_job_path = job_path or FIXTURE
    with open(resolved_job_path, encoding='utf-8') as handle:
        cfg = jsonlib.load(handle)
    return _normalize_external_references(cfg, resolved_job_path)


def apply_simulation_overrides(
    cfg: dict[str, Any],
    *,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    strike: Optional[float] = None,
    liability_name: Optional[str] = None,
) -> dict[str, Any]:
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'simulate_only'
    calc['Simulation_Batches'] = 1
    if batch_size is not None:
        calc['Batch_Size'] = batch_size
    if seed is not None:
        calc['Random_Seed'] = seed
    if strike is not None:
        liability_key = liability_name or next(iter(calc['Hedging_Problem']['Liabilities']['FloatingEnergyDeal']))
        calc['Hedging_Problem']['Liabilities']['FloatingEnergyDeal'][liability_key]['Payments']['Items'][0]['Fixed_Basis'] = -strike
    return cfg


def load_simulation_config(
    job_path: Optional[str] = None,
    *,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    strike: Optional[float] = None,
    liability_name: Optional[str] = None,
) -> dict[str, Any]:
    cfg = load_job_config(job_path)
    return apply_simulation_overrides(
        cfg,
        batch_size=batch_size,
        seed=seed,
        strike=strike,
        liability_name=liability_name,
    )


def run_simulation_config(cfg: dict[str, Any]):
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'textbook.json'))
    _, result = cx.run_job()
    return result


def runtime_instrument_order(result) -> list[str]:
    policy = result.runtime.get('policy', {})
    action_space = policy.get('action_space', {})
    order = list(action_space.get('instrument_order', ()))
    if order:
        return order
    return list(result.runtime.get('tradables', {}).keys())


def build_schedule_rows(result, instruments: Optional[list[str]] = None) -> list[dict[str, Any]]:
    bundle = result.bundle
    dates = bundle['scenario_dates']
    business_day = bundle['meta']['business_day']
    decision_indices = set(bundle.get('business_indices', ()))
    chosen_instruments = instruments or runtime_instrument_order(result)
    rows: list[dict[str, Any]] = []
    for idx, date in enumerate(dates):
        row = {
            'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
            'is_business_day': bool(business_day.is_on_offset(date)),
            'is_decision_step': idx in decision_indices,
        }
        for instrument in chosen_instruments:
            row[instrument] = ''
        rows.append(row)
    return rows


def position_targets_from_rows(rows: list[dict[str, Any]], instruments: list[str]) -> dict[str, dict[str, int]]:
    targets: dict[str, dict[str, int]] = {}
    current = {instrument: 0 for instrument in instruments}
    for row in rows:
        date_key = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
        row_targets = dict(current)
        for instrument in instruments:
            value = row.get(instrument, '')
            if value in ('', None):
                continue
            row_targets[instrument] = int(round(float(value)))
        current = row_targets
        targets[date_key] = row_targets
    return targets


def run_position_target_schedule(result, target_positions_by_date: dict[str, dict[str, int]]):
    stepper = result.create_stepper()
    last = None
    instruments = runtime_instrument_order(result)
    while not stepper.done:
        if not stepper.is_decision_step:
            last = stepper.step(None)
            continue
        observed = stepper.observe()
        date_key = pd.Timestamp(result.bundle['scenario_dates'][stepper.time_index]).strftime('%Y-%m-%d')
        targets = target_positions_by_date.get(date_key)
        if not targets:
            last = stepper.step(None)
            continue
        deltas = {}
        for instrument in instruments:
            current_pos = float(observed['positions'][instrument][0].item())
            target = int(targets.get(instrument, round(current_pos)))
            delta = target - int(round(current_pos))
            if delta:
                deltas[instrument] = delta
        last = stepper.step(deltas or None)
    return stepper, last


def summarize_terminal_pnl(last):
    pnl = (last['transition_pnl_excess'] + last['transition_liability_value']).to(dtype=torch.float64)
    return {
        'mean': float(pnl.mean()),
        'median': float(torch.quantile(pnl, 0.5)),
        'min': float(pnl.min()),
        'std': float(pnl.std()),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch_size', type=int, default=None,
                   help='Batch size for the simulation (default: from fixture)')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed (default: from fixture)')
    p.add_argument('--hedge_instrument', type=str, default='PL_JUL_2026',
                   help='Contract to short and ramp back (default: %(default)s)')
    p.add_argument('--strike', type=float, default=2055.0,
                   help='fixed strike (default: %(default)s)')
    p.add_argument('--output_root', type=str, default='artifacts/textbook_runs')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def _build_simulate_config(args):
    return load_simulation_config(
        batch_size=args.batch_size,
        seed=args.seed,
        strike=args.strike,
        liability_name=args.hedge_instrument,
    )


def _textbook_targets(result, hedge_instrument):
    """Compute per-decision-step target positions for the textbook ramp hedge.

    Returns `(short_at_start, tn_idx, target_at_idx)`:
      - short_at_start: integer position to hold from t=0 until averaging begins
      - tn_idx: first scenario-date index inside the averaging window
      - target_at_idx: {decision_t_idx: target_position} across the averaging-window
        business days; linearly ramps from -short_at_start back to 0.
    """
    runtime = result.runtime
    bundle = result.bundle
    liab = next(iter(runtime['liabilities'].values()))
    items = liab['params']['Payments']['Items']
    deal_volume = float(items[0]['Volume'])
    contract_size = float(runtime['tradables'][hedge_instrument]['contract_size'])
    short_at_start = int(round(deal_volume / contract_size))

    period_start = pd.Timestamp(items[0]['Period_Start'])
    period_end = pd.Timestamp(items[0]['Period_End'])
    last_trade = runtime['tradables'][hedge_instrument].get('last_trade_date')
    window_end = min(pd.Timestamp(last_trade), period_end) if last_trade else period_end
    dates = bundle['scenario_dates']
    bday = bundle['meta']['business_day']
    buyback_indices = [i for i, d in enumerate(dates)
                       if period_start <= d <= window_end and bday.is_on_offset(d)]
    tn_idx = next(i for i, d in enumerate(dates) if d >= period_start)
    N = len(buyback_indices)
    target_at_idx = {idx: -short_at_start * (1.0 - (i + 1) / N)
                     for i, idx in enumerate(buyback_indices)}
    return short_at_start, tn_idx, target_at_idx


def _run_textbook(stepper, hedge_instrument, short_at_start, tn_idx, target_at_idx):
    last = None
    while not stepper.done:
        if not stepper.is_decision_step:
            last = stepper.step(None)
            continue
        t = stepper.time_index
        if t < tn_idx:
            target = -short_at_start
        elif t in target_at_idx:
            target = target_at_idx[t]
        else:
            target = 0.0
        cur_pos = float(stepper.observe()['positions'][hedge_instrument][0].item())
        delta = int(round(target)) - int(round(cur_pos))
        last = stepper.step({hedge_instrument: delta})
    return last


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'_{args.tag}' if args.tag else ''
    out_dir = os.path.join(args.output_root, f'{timestamp}{tag}')
    os.makedirs(out_dir, exist_ok=True)

    cfg = _build_simulate_config(args)
    cfg_path = os.path.join(out_dir, 'textbook.json')
    with open(cfg_path, 'w') as f:
        jsonlib.dump(cfg, f, indent=2)
    print(f'Output:    {out_dir}')
    print(f'Config:    {cfg_path}\n')

    t_start = time.monotonic()
    result = run_simulation_config(cfg)
    sim_elapsed = time.monotonic() - t_start

    short_at_start, tn_idx, target_at_idx = _textbook_targets(result, args.hedge_instrument)
    print(f'Textbook ramp: short {short_at_start} {args.hedge_instrument} contracts from t=0, '
          f'ramp to 0 over {len(target_at_idx)} business days in the averaging window\n')

    stepper = result.create_stepper()
    last = _run_textbook(stepper, args.hedge_instrument, short_at_start, tn_idx, target_at_idx)
    stepper.write_diagnostic_csvs(out_dir, label='textbook')

    summary = summarize_terminal_pnl(last)
    elapsed = time.monotonic() - t_start
    print('=' * 78)
    print(f"  textbook P&L:  mean=${summary['mean']:>+12,.0f}  "
          f"median=${summary['median']:>+12,.0f}  "
          f"worst=${summary['min']:>+12,.0f}  "
          f"std=${summary['std']:>10,.0f}")
    print(f'  paths CSV:    {os.path.join(out_dir, "textbook_paths.csv")}')
    print(f'  summary CSV:  {os.path.join(out_dir, "textbook_summary.csv")}')
    print(f'  (simulate {sim_elapsed:.1f}s, total {elapsed:.1f}s)')
    print('=' * 78)


if __name__ == '__main__':
    main()
