"""Thin CLI shim for training a hedger via the riskflow framework.

Loads `tests/fixtures/policy_test_simulate_only.json` and applies CLI overrides ONLY for
flags the user explicitly passed. The fixture is the source of truth for default
hyperparameters; this script's job is to (a) flip Execution_Mode to optimize_policy and
(b) wire up per-run output paths (Live_Diag_Path, Random_Seed, Batch_Size). Every other
field passes through from the fixture unchanged unless the corresponding `--<flag>` was
provided on the command line.

No internal imports. No monkey-patching. The framework's public surface is the JSON
schema + `rf.Context`. Re-evaluate a saved artifact via `eval_policy_artifact.py`.
"""
import argparse
import json as jsonlib
import os
import time
from datetime import datetime

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'fixtures',
                        'policy_test_simulate_only.json')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Optimizer — None sentinel: only override the fixture when the flag is explicitly passed.
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--reward_scale', type=float, default=None)
    p.add_argument('--entropy_coef', type=float, default=None)
    p.add_argument('--value_coef', type=float, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--ppo_epochs', type=int, default=None)
    p.add_argument('--minibatch_size', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--dense_tracking_reward_clip', type=float, default=None)
    p.add_argument('--dense_tracking_reward_scale', type=float, default=None)
    # Objective
    p.add_argument('--objective_object', type=str, default=None)
    p.add_argument('--utility_scale_explicit', type=float, default=None)
    p.add_argument('--floor_penalty', type=float, default=None)
    p.add_argument('--expiry_penalty', type=float, default=None)
    p.add_argument('--expiry_threshold_days', type=float, default=None)
    p.add_argument('--post_deal_trade_penalty', type=float, default=None)
    p.add_argument('--position_bounds_penalty', type=float, default=None)
    # Run admin
    p.add_argument('--output_root', type=str, default='artifacts/daily_runs')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def build_training_config(args, *, out_dir: str) -> dict:
    """Layer CLI overrides on top of the fixture. Only fields whose CLI flag was explicitly
    passed (i.e. not None) get overridden — every other field stays as the fixture set it.

    Always-applied (per-run plumbing, not user-tunable defaults):
      - Execution_Mode = optimize_policy (this script trains; eval has its own entry point)
      - Optimizer.Live_Diag_Path: atomic per-epoch JSON dump to <out_dir>/live_diag.json
      - Random_Seed / Simulation_Batches / Batch_Size: only if the CLI flag was passed
    """
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'optimize_policy'
    calc['Simulation_Batches'] = 1
    if args.batch_size is not None:
        calc['Batch_Size'] = args.batch_size
    if args.seed is not None:
        calc['Random_Seed'] = args.seed
    opt = calc['Hedging_Problem']['Optimizer']
    if args.epochs is not None: opt['Epochs'] = args.epochs
    if args.ppo_epochs is not None: opt['PPO_Epochs'] = args.ppo_epochs
    if args.minibatch_size is not None: opt['Minibatch_Size'] = args.minibatch_size
    if args.lr is not None: opt['Learning_Rate'] = args.lr
    if args.reward_scale is not None: opt['Reward_Scale'] = args.reward_scale
    if args.entropy_coef is not None: opt['Entropy_Coef'] = args.entropy_coef
    if args.value_coef is not None: opt['Value_Coef'] = args.value_coef
    if args.dense_tracking_reward_clip is not None:
        opt['Dense_Tracking_Reward_Clip'] = args.dense_tracking_reward_clip
    if args.dense_tracking_reward_scale is not None:
        opt['Dense_Tracking_Reward_Scale'] = args.dense_tracking_reward_scale
    opt['Live_Diag_Path'] = os.path.join(out_dir, 'live_diag.json')
    obj = calc['Hedging_Problem']['Objective']
    if args.objective_object is not None: obj['Object'] = args.objective_object
    if args.utility_scale_explicit is not None: obj['Utility_Scale_Explicit'] = args.utility_scale_explicit
    if args.floor_penalty is not None: obj['Floor_Penalty'] = args.floor_penalty
    if args.expiry_penalty is not None: obj['Expiry_Penalty'] = args.expiry_penalty
    if args.expiry_threshold_days is not None: obj['Expiry_Threshold_Days'] = args.expiry_threshold_days
    if args.post_deal_trade_penalty is not None: obj['Post_Deal_Trade_Penalty'] = args.post_deal_trade_penalty
    if args.position_bounds_penalty is not None: obj['Position_Bounds_Penalty'] = args.position_bounds_penalty
    return cfg


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'_{args.tag}' if args.tag else ''
    out_dir = os.path.join(args.output_root, f'{timestamp}{tag}')
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_training_config(args, out_dir=out_dir)
    config_path = os.path.join(out_dir, 'train_daily.json')
    with open(config_path, 'w') as f:
        jsonlib.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, 'cli_args.json'), 'w') as f:
        jsonlib.dump({**vars(args), 'timestamp': timestamp}, f, indent=2)
    print(f'Output:    {out_dir}')
    print(f'Config:    {config_path}\n')

    t_start = time.monotonic()
    cx = rf.Context()
    cx.load_json(config_path)
    _, result = cx.run_job()
    elapsed = time.monotonic() - t_start

    artifact_path = os.path.join(out_dir, 'trained_policy.json')
    with open(artifact_path, 'w') as f:
        jsonlib.dump(dict(result.policy_artifact), f, indent=2, default=str)

    # Post-processing: per-day per-instrument breakdown CSVs. The framework's compute path
    # is now side-effect-free; CSV writing is the caller's choice via the result method.
    result.write_diagnostic_csvs(out_dir)

    summary = result.evaluation_summary or {}
    metrics = summary.get('metrics', {})
    post = summary.get('diagnostics', {}).get('post_settle', {})
    print(f'Training done in {elapsed/60:.1f} min')
    print(f'Saved policy: {artifact_path}')
    print()
    print('=' * 78)
    print(f"  argmax-eval P&L:  mean={metrics.get('average_net_pnl', 0):>+12,.0f}  "
          f"median={metrics.get('median_net_pnl', 0):>+12,.0f}  "
          f"worst={metrics.get('worst_net_pnl', 0):>+12,.0f}")
    print(f"  position audit:   max_per_inst={post.get('max_per_instrument_position', 0):>+6.1f}  "
          f"min_per_inst={post.get('min_per_instrument_position', 0):>+6.1f}  "
          f"max_total_abs={post.get('max_total_abs_position', 0):>6.1f}  "
          f"frac_post_settle_holding={post.get('frac_post_settle_holding', 0):.4f}")
    if 'reference' in summary:
        diff = summary['reference'].get('policy_minus_no_trade', {})
        print(f"  vs no_trade:      mean_delta={diff.get('average_net_pnl', 0):>+12,.0f}  "
              f"worst_delta={diff.get('worst_net_pnl', 0):>+12,.0f}")
    print('=' * 78)


if __name__ == '__main__':
    main()
