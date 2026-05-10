"""Re-evaluate a saved policy artifact through the riskflow framework.

Reads the run's saved JSON config (`train_daily.json`), overrides Execution_Mode and
points Policy.Artifact_Path at the trained policy, calls `cx.run_job()`. Reports the
framework's evaluation_summary side-by-side with the position-limit audit.

Pattern: load JSON, modify, run_job. No internal imports, no monkey-patching.
"""
import argparse
import glob
import json as jsonlib
import os

import riskflow as rf


def find_run_dir(tag_substr: str) -> str:
    candidates = sorted(glob.glob(f'artifacts/**/*{tag_substr}*', recursive=True))
    candidates = [d for d in candidates if os.path.exists(os.path.join(d, 'trained_policy.json'))]
    if not candidates:
        raise SystemExit(f"no run dir matching {tag_substr!r} with trained_policy.json")
    return candidates[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', help='substring of run dir name (searches under artifacts/)')
    p.add_argument('--run_dir', help='explicit run dir (alternative to --tag)')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not args.tag and not args.run_dir:
        raise SystemExit('--tag or --run_dir required')

    run_dir = args.run_dir or find_run_dir(args.tag)
    artifact = os.path.join(run_dir, 'trained_policy.json')
    # Phase A/B/C/D sweeps name the config 'train.json'; train_daily.py names it 'train_daily.json'.
    cfg_path = next(p for p in (
        os.path.join(run_dir, 'train.json'),
        os.path.join(run_dir, 'train_daily.json'),
    ) if os.path.exists(p))
    print(f'Run dir:   {run_dir}')
    print(f'Artifact:  {artifact}')

    # Load the training-time config and override only what changes for eval.
    cfg = jsonlib.load(open(cfg_path))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'simulate_only'
    calc['Batch_Size'] = args.batch_size
    calc['Random_Seed'] = args.seed
    calc['Hedging_Problem']['Policy']['Artifact_Path'] = os.path.abspath(artifact)
    # Drop the diagnostic-output switch — eval doesn't need to re-write CSVs.
    calc['Hedging_Problem']['Optimizer'].pop('Diagnostic_Output_Dir', None)
    calc['Hedging_Problem']['Optimizer'].pop('Live_Diag_Path', None)

    eval_cfg_path = os.path.join(run_dir, 'eval_artifact.json')
    with open(eval_cfg_path, 'w') as f:
        jsonlib.dump(cfg, f, indent=2)

    cx = rf.Context()
    cx.load_json(eval_cfg_path)
    _, result = cx.run_job()

    summary = result.evaluation_summary or {}
    metrics = summary.get('metrics', {})
    post = summary.get('diagnostics', {}).get('post_settle', {})

    import torch
    net = summary.get('final_state', {}).get('net_pnl')
    print()
    print(f"Argmax-eval over {args.batch_size} paths (seed={args.seed}):")
    print(f"  mean   = {metrics.get('average_net_pnl', 0):>+12,.0f}")
    print(f"  median = {metrics.get('median_net_pnl', 0):>+12,.0f}")
    print(f"  worst  = {metrics.get('worst_net_pnl', 0):>+12,.0f}")
    if net is not None:
        net_t = net if isinstance(net, torch.Tensor) else torch.as_tensor(net)
        print(f"  std    = {float(net_t.std()):>12,.0f}")
        print(f"  p5     = {float(torch.quantile(net_t.to(dtype=torch.float64), 0.05)):>+12,.0f}")
        print(f"  p95    = {float(torch.quantile(net_t.to(dtype=torch.float64), 0.95)):>+12,.0f}")
    print()
    print('Position-limit audit:')
    print(f"  frac_post_settle_holding    = {post.get('frac_post_settle_holding', 0):.4f}")
    print(f"  max_per_instrument_position = {post.get('max_per_instrument_position', 0):>+6.1f}")
    print(f"  min_per_instrument_position = {post.get('min_per_instrument_position', 0):>+6.1f}")
    print(f"  max_total_abs_position      = {post.get('max_total_abs_position', 0):>6.1f}")


if __name__ == '__main__':
    main()
