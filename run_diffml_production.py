"""Production driver — DifferentialSolver on the real platinum prepay/offtake.

Per validation_sandwich_spec.md §1, §6: belief-state policy on the participant-
visible coordinates (the regime substitution lives in DifferentialSolver itself,
commit 7467fb1). This script just loads the latest daily-run JSON, overrides
Solver.Object + Execution_Mode to the diff-ML stack, and calls cx.run_job.

The L/π/U sandwich diagnostic is the next thing to wire into the solver; once
it lands, the validation readout becomes `gap = U − L` + dollar floor (§3, §5),
not toy V_0. For now this script just confirms the diff-ML stack runs to
completion on the production deal structure (3 platinum futures, daily grid,
2500 oz PLAT_JUL29 liability).

Pattern: load JSON, modify, cx.run_job. No internal imports, no monkey-patching.
"""
import argparse
import json as jsonlib
import logging
import os
import sys

import riskflow as rf

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                    default='artifacts/daily_runs/20260510_173249_symlog_smoke/train_daily.json',
                    help='Production daily-run JSON to base the solve on.')
    p.add_argument('--batch-size', type=int, default=256,
                    help='Outer paths for the smoke. Daily 60-step horizon × 3-leg '
                         '1331-action grid is heavy; start small.')
    p.add_argument('--bank-b-endo', type=int, default=2,
                    help='Per-outer endogenous span. 2 = spec §6 default.')
    p.add_argument('--margin-usd-per-oz', type=float, default=6.0,
                    help='Dealer margin to clear ($/oz). Spec §0: $6-8/oz; '
                         'default $6 per validation_sandwich_spec.md §7 ship criterion.')
    p.add_argument('--randomize-t0', dest='randomize_t0', action='store_true',
                    default=True,
                    help='Per-path t=0 initial state for Huge-Savine diff-ML, drawn '
                         'from the process\'s own T-step pushforward via a per-batch '
                         'burn-in. Without it the t=0 exogenous slice has zero '
                         'variance and the differential label is degenerate.')
    p.add_argument('--no-randomize-t0', dest='randomize_t0', action='store_false')
    args = p.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f'config not found: {args.config}')

    with open(args.config) as f:
        cfg = jsonlib.load(f)

    calc = cfg['Calc']['Calculation']
    hp = calc['Hedging_Problem']

    # Switch to the diff-ML stack — solve_hedge mode + DifferentialSolver.
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'] = int(args.batch_size)
    calc['Inner_Sub_Batch'] = 128
    calc['Inner_MC_Enabled'] = 'Yes'

    # Huge-Savine diff-ML needs variance in the t=0 exogenous slice. Single JSON
    # switch; the per-batch burn-in inside HedgeMonteCarlo.execute does the rest.
    if args.randomize_t0:
        hp['Randomize_Initial_State'] = True

    # Sum the deal volume from cashflows so $/oz conversion has its denominator.
    total_volume_oz = 0.0
    for deal_type, by_name in hp['Liabilities'].items():
        for name, params in by_name.items():
            items = params.get('Payments', {}).get('Items') or []
            for cf in items:
                total_volume_oz += float(cf.get('Volume', 0.0))
    print(f'Total liability volume: {total_volume_oz:.1f} oz across all cashflows')
    hp['Solver'] = {
        'Object': 'DifferentialSolver',
        'Bank_Sampling': {'B_Endo': int(args.bank_b_endo)},
        'Value_Fn': {
            'MLP_Hidden': [128, 128, 128],
            'MLP_Train_Steps_Per_Solve': 2000,
            'MLP_Adam_LR': 1.0e-3,
            'MLP_Minibatch': 4096,
        },
        # Default off until validated: λ-mix produces a calibration↔policy tradeoff
        # (see project_lambda_mix_tradeoff). Belief substitution may obviate it
        # entirely per spec §5.
        'Lambda_Mix': 0.0,
        'Use_Advantage_Decomp': True,
        'T_Min': 0,
    }

    # Write to a temp file and load via cx.load_json (it expects a path).
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        jsonlib.dump(cfg, f, indent=2)
        cfg_path = f.name

    cx = rf.Context()
    cx.load_json(cfg_path)
    _, result = cx.run_job()

    print()
    print('=' * 78)
    print('Production diff-ML smoke — DifferentialSolver on PLAT_JUL29 deal')
    print('=' * 78)
    print(f'Margin floor (ship if cleared): ${args.margin_usd_per_oz:.2f}/oz')
    # Result schema depends on solve_hedge output; print what's there.
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (str, int, float, bool)):
                print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
