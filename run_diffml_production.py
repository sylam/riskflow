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
    p.add_argument('--batch-size', type=int, default=4096,
                    help='Outer paths for the diff-ML bank. Huge-Savine style uses a '
                         'fat outer batch with a tiny inner rollout fan.')
    p.add_argument('--inner-sub-batch', type=int, default=4,
                    help='Inner paths per outer state. DifferentialSolver is designed '
                         'for tiny-inner labels/rollouts; 2 or 4 is usually enough.')
    p.add_argument('--bank-b-endo', type=int, default=2,
                    help='Per-outer endogenous span. 2 = spec §6 default.')
    p.add_argument('--train-steps', type=int, default=2000,
                    help='Maximum MLP training steps per C_t fit. Lower this for wiring smokes.')
    p.add_argument('--loss-tol', type=float, default=1.0e-4,
                    help='Early-stop tolerance on standardized twin loss. Set 0 to force max steps.')
    p.add_argument('--action-grid-levels', type=int, default=11,
                    help='Training/decision action grid levels per live hedge axis.')
    p.add_argument('--t-min', type=int, default=0,
                    help='Backward sweep stop index. 0 = full sweep to initial decision.')
    p.add_argument('--audit-max-rounds', type=int, default=3,
                    help='Max audit/correction rounds per fitted timestep.')
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
    calc['Inner_Sub_Batch'] = int(args.inner_sub_batch)
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
        'Training_Action_Grid_Levels_Per_Axis': int(args.action_grid_levels),
        'Decision_Action_Grid_Levels_Per_Axis': int(args.action_grid_levels),
        'Audit_Max_Rounds': int(args.audit_max_rounds),
        'Value_Fn': {
            'MLP_Hidden': [128, 128, 128],
            'MLP_Train_Steps_Per_Solve': int(args.train_steps),
            'MLP_Loss_Tol': float(args.loss_tol),
            'MLP_Adam_LR': 1.0e-3,
            'MLP_Minibatch': 4096,
        },
        # Default off until validated: λ-mix produces a calibration↔policy tradeoff
        # (see project_lambda_mix_tradeoff). Belief substitution may obviate it
        # entirely per spec §5.
        'Lambda_Mix': 0.0,
        'Use_Advantage_Decomp': True,
        'T_Min': int(args.t_min),
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
    diagnostics = None
    if hasattr(result, 'optimizer_diagnostics'):
        diagnostics = result.optimizer_diagnostics
    elif isinstance(result, dict):
        diagnostics = result.get('optimizer_diagnostics') or result.get('diagnostics')
    if isinstance(diagnostics, dict):
        for key in (
            'V_0', 'M1_v0_decision_at_t_min', 'M1_L_cost_per_oz_usd',
            'M1_penalty_zero_mean_z', 'M1_penalty_boundary_hit_cap',
            'M1_C_fitted_count'):
            if key in diagnostics:
                print(f'  {key}: {diagnostics[key]}')


if __name__ == '__main__':
    main()
