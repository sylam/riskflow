"""HP sweep for the symlog-stack training run.

For each row in the sweep grid, build a per-row JSON (fixture + overrides), drive the
framework via `rf.Context().run_job()`, and record the evaluation_summary's headline
metrics + position audit + the policy-vs-no_trade delta into one CSV. No internal
imports, no monkey-patching.

Default sweep is OFAT (one-factor-at-a-time) around the fixture's symlog defaults,
50 epochs, seed=42. Adjust the `GRID` dict below to add axes / values / seeds.

Output:
  artifacts/sweeps/<timestamp>_<tag>/
    sweep.csv               aggregated headline rows
    <row_tag>/              per-row outputs (config, live_diag, summary, ml_paths, policy)
"""
import argparse
import json as jsonlib
import os
import time
from datetime import datetime

import pandas as pd

import riskflow as rf


FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'fixtures',
                        'policy_test_simulate_only.json')


# Sweep axes — OFAT around the fixture defaults. To add a new axis, append a dict
# with `axis`, `optimizer_or_objective`, `json_key`, `values`. The "default" (axis=None)
# row is always row 0 and used as the baseline reference for delta columns.
GRID = {
    'epochs': 100,
    'seeds': (42,),
    'axes': [
        {'axis': 'floor_penalty',                'block': 'objective',
         'json_key': 'Floor_Penalty',                'values': (5.0, 20.0)},
        {'axis': 'dense_tracking_reward_scale',  'block': 'optimizer',
         'json_key': 'Dense_Tracking_Reward_Scale',  'values': (1.0, 3.0)},
        {'axis': 'entropy_coef',                 'block': 'optimizer',
         'json_key': 'Entropy_Coef',                 'values': (3.0e-4, 3.0e-3)},
        {'axis': 'learning_rate',                'block': 'optimizer',
         'json_key': 'Learning_Rate',                'values': (1.0e-3,)},
        # Row-1 (default, fp=10) at 100ep produced frac_post_settle=1.0 — coef=1.0 in
        # utility units doesn't bite. Bracket upward to find the binding strength.
        {'axis': 'post_deal_trade_penalty',      'block': 'objective',
         'json_key': 'Post_Deal_Trade_Penalty',       'values': (2.0, 5.0)},
        # val_l still ~e9 at end of 30ep smoke; bracket value_coef to test V capacity.
        {'axis': 'value_coef',                   'block': 'optimizer',
         'json_key': 'Value_Coef',                    'values': (0.05, 0.2)},
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--epochs', type=int, default=None,
                   help='Override per-row epochs (default: from GRID dict)')
    p.add_argument('--seeds', type=str, default=None,
                   help='Comma-separated seed list (default: from GRID dict)')
    p.add_argument('--row_start', type=int, default=0,
                   help='First row index in the materialized plan (inclusive)')
    p.add_argument('--row_end', type=int, default=None,
                   help='Last row index (exclusive). Default = all rows.')
    p.add_argument('--output_root', type=str, default='artifacts/sweeps')
    p.add_argument('--tag', type=str, default='symlog_phaseA')
    return p.parse_args()


def _row_configs():
    """Materialize the OFAT plan: 1 default row + N override rows per axis. Each row is
    `{'row_tag': str, 'overrides': [(block, key, value), ...]}`. Default row has no
    overrides (uses fixture defaults verbatim except seed/epochs)."""
    rows = [{'row_tag': 'default', 'overrides': []}]
    for axis in GRID['axes']:
        for v in axis['values']:
            rows.append({
                'row_tag': f"{axis['axis']}={v:g}",
                'overrides': [(axis['block'], axis['json_key'], v)],
            })
    return rows


def _build_config(*, epochs, seed, overrides, live_diag_path):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'optimize_policy'
    calc['Simulation_Batches'] = 1
    calc['Random_Seed'] = seed
    opt = calc['Hedging_Problem']['Optimizer']
    obj = calc['Hedging_Problem']['Objective']
    opt['Epochs'] = epochs
    opt['Seed'] = seed
    opt['Live_Diag_Path'] = live_diag_path
    for block, key, value in overrides:
        target = opt if block == 'optimizer' else obj
        target[key] = value
    return cfg


def _summarize(result):
    """Pull the headline row from result.evaluation_summary. Adds policy_minus_no_trade
    deltas and the post-settle position audit. All values are floats / ints — straight
    pandas-friendly."""
    summary = result.evaluation_summary or {}
    metrics = summary.get('metrics', {})
    diff = summary.get('reference', {}).get('policy_minus_no_trade', {})
    post = summary.get('diagnostics', {}).get('post_settle', {})
    no_trade = summary.get('reference', {}).get('no_trade', {}).get('metrics', {})
    return {
        'mean_pnl': float(metrics.get('average_net_pnl', 0.0)),
        'worst_pnl': float(metrics.get('worst_net_pnl', 0.0)),
        'median_pnl': float(metrics.get('median_net_pnl', 0.0)),
        'no_trade_worst': float(no_trade.get('worst_net_pnl', 0.0)),
        'no_trade_mean': float(no_trade.get('average_net_pnl', 0.0)),
        'mean_delta_vs_no_trade': float(diff.get('average_net_pnl', 0.0)),
        'worst_delta_vs_no_trade': float(diff.get('worst_net_pnl', 0.0)),
        'max_per_inst_pos': float(post.get('max_per_instrument_position', 0.0)),
        'min_per_inst_pos': float(post.get('min_per_instrument_position', 0.0)),
        'max_total_abs_pos': float(post.get('max_total_abs_position', 0.0)),
        'frac_post_settle': float(post.get('frac_post_settle_holding', 0.0)),
    }


def main():
    args = parse_args()
    epochs = args.epochs if args.epochs is not None else GRID['epochs']
    seeds = (tuple(int(s) for s in args.seeds.split(','))
             if args.seeds is not None else GRID['seeds'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = os.path.join(args.output_root, f'{timestamp}_{args.tag}')
    os.makedirs(sweep_dir, exist_ok=True)

    all_rows = _row_configs()
    end = args.row_end if args.row_end is not None else len(all_rows)
    rows = all_rows[args.row_start:end]
    n_total = len(rows) * len(seeds)
    print(f'Sweep dir: {sweep_dir}')
    print(f'Rows {args.row_start}-{end} ({len(rows)} of {len(all_rows)}) × seeds {seeds} = {n_total} runs at {epochs} epochs each\n')

    records = []
    for seed in seeds:
        for i, row in enumerate(rows):
            row_tag = f"{row['row_tag']}_seed{seed}"
            row_dir = os.path.join(sweep_dir, row_tag)
            os.makedirs(row_dir, exist_ok=True)
            live_diag = os.path.join(row_dir, 'live_diag.json')
            cfg = _build_config(epochs=epochs, seed=seed, overrides=row['overrides'],
                                live_diag_path=live_diag)
            cfg_path = os.path.join(row_dir, 'train.json')
            with open(cfg_path, 'w') as f:
                jsonlib.dump(cfg, f, indent=2)

            print(f"[{i+1}/{len(rows)} seed={seed}] {row['row_tag']}")
            t_start = time.monotonic()
            cx = rf.Context()
            cx.load_json(cfg_path)
            _, result = cx.run_job()
            elapsed = time.monotonic() - t_start

            with open(os.path.join(row_dir, 'trained_policy.json'), 'w') as f:
                jsonlib.dump(dict(result.policy_artifact), f, indent=2, default=str)
            result.write_diagnostic_csvs(row_dir)

            metrics = _summarize(result)
            metrics.update({
                'row_tag': row['row_tag'],
                'seed': seed,
                'epochs': epochs,
                'overrides': ';'.join(f"{b}.{k}={v:g}" for b, k, v in row['overrides']),
                'elapsed_min': elapsed / 60.0,
            })
            records.append(metrics)
            # Re-write CSV after every row so partial sweeps still produce a usable file.
            pd.DataFrame(records).to_csv(os.path.join(sweep_dir, 'sweep.csv'), index=False)
            print(f"  → mean=${metrics['mean_pnl']:+,.0f} worst=${metrics['worst_pnl']:+,.0f} "
                  f"vs_nt: mean_d=${metrics['mean_delta_vs_no_trade']:+,.0f} "
                  f"worst_d=${metrics['worst_delta_vs_no_trade']:+,.0f}  "
                  f"frac_post={metrics['frac_post_settle']:.3f}  ({elapsed/60:.1f} min)\n")

    print('=' * 78)
    print(f'Sweep complete. Aggregate CSV: {os.path.join(sweep_dir, "sweep.csv")}')


if __name__ == '__main__':
    main()
