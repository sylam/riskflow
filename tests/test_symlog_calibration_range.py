"""Pre-flight calibration check: drives the framework with a JSON fixture, retrieves the
resolved utility_scale `c` from result.torchrl_bundle, then verifies the symlog terminal
utility band is in the [10, 50] range the spec calls "fittable by a linear value head"
across a panel of representative terminal net_pnl dollar values.

JSON in, result out. No internal framework imports — symlog is computed locally as
sign(x)·fp·log1p(|x|/c), the same closed form the framework's `_evaluate_objective`
implements. If those two formulas ever drift, the smoke test (which exercises the
framework's own evaluator) will catch it.
"""
import json as jsonlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'policy_test_simulate_only.json')


def _symlog_floor_utility(net_pnl: torch.Tensor, c: float, fp: float) -> torch.Tensor:
    """Symlog floor-only utility: -fp · log1p(|loss|/c) on the downside, 0 on the upside."""
    loss = (-net_pnl).clamp_min(0.0)
    return -fp * torch.log1p(loss / c)


def main():
    fp = 10.0
    cfg = jsonlib.load(open(FIXTURE))
    obj = cfg['Calc']['Calculation']['Hedging_Problem']['Objective']
    obj.update({'Object': 'AsymmetricUtility_Symlog', 'Floor_Penalty': fp,
                'Surplus_Reward': 1.0, 'Power': 1.0})

    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'calib_range.json'))
    _, result = cx.run_job()

    c = float(result.torchrl_bundle['utility_scale'])
    print(f"Resolved utility_scale c = ${c:,.0f}")

    pnl_panel = torch.tensor([
        -2.59e8, -1.00e8, -1.00e7, -1.84e6, -8.85e5, -1.30e5, -3.05e4,
         0.0,
        +5.00e3, +5.00e6, +5.00e7, +1.56e8,
    ], dtype=torch.float32)
    util = _symlog_floor_utility(pnl_panel, c, fp)

    print(f"\n{'pnl ($)':>15} | {'util_floor':>11}")
    print("-" * 31)
    for x, u in zip(pnl_panel.tolist(), util.tolist()):
        print(f"{x:>15,.0f} | {u:>11.3f}")

    util_max = float(util.abs().max().item())
    print(f"\nmax |util_floor| = {util_max:.2f}")
    if util_max < 10.0:
        print(f"FAIL: max |util| = {util_max:.2f} < 10 — V's target distribution too compressed.")
        raise SystemExit(1)
    if util_max > 50.0:
        print(f"FAIL: max |util| = {util_max:.2f} > 50 — c too small for the eval distribution.")
        raise SystemExit(1)
    print("PASS: max |util| in [10, 50] — symlog targets fittable by a linear value head.")


if __name__ == '__main__':
    main()
