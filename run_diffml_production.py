"""Production driver — DifferentialSolver on the real platinum prepay/offtake.

Per validation_sandwich_spec.md §1, §6: belief-state policy on the participant-
visible coordinates (the regime substitution lives in DifferentialSolver itself,
commit 7467fb1). This script just loads the latest daily-run JSON, overrides
Solver.Object + Execution_Mode to the diff-ML stack, and calls cx.run_job.

The L/π/U validation sandwich is wired into DifferentialSolver, so the readout is
`gap = U − L` (utility units) + the dollar floor (§3, §5), alongside toy V_0. U is
the penalized clairvoyant upper bound; L the realised policy value; a WIDE gap is
AMBIGUOUS (perfect-foresight + grid/turnover relaxation slack), NOT necessarily
policy suboptimality. The penalty zero-mean gate must be green (boundary not at cap)
for U to be a valid bound. Runs on the production deal (3 platinum futures, daily
grid, 2500 oz PLAT_JUL29 liability).

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
    p.add_argument('--run-upper-bound', dest='run_upper_bound', action='store_true',
                    default=False,
                    help='Compute the penalized clairvoyant UPPER bound U (gap = U − L). '
                         'OPT-IN: the wealth-grid DP is O(P·G·K²·I) per step, so it is costly '
                         'with many live hedge axes (K = levels^n_live near t0). The cheap '
                         'penalty zero-mean gate always runs regardless.')
    p.add_argument('--upper-bound-wealth-grid', type=int, default=81,
                    help='Wealth-grid nodes for U. More = finer (watch σ/Δgrid ≥ 0.3) but slower.')
    p.add_argument('--upper-bound-max-paths', type=int, default=128,
                    help='Outer paths the U DP runs on (capped subset of Batch_Size).')
    p.add_argument('--save-value-fn', type=str, default=None,
                    help='Persist the fitted C-stack to this path after the sweep (for OOS reuse).')
    p.add_argument('--load-value-fn', type=str, default=None,
                    help='Load a saved C-stack, SKIP training, and run the L/π/U sandwich on '
                         'this run\'s batch. Pair with a DIFFERENT --random-seed for genuine '
                         'out-of-sample validation.')
    p.add_argument('--random-seed', type=int, default=None,
                    help='Outer-MC seed. Use a different value from the training run when '
                         'loading a C-stack so the OOS sandwich runs on a fresh batch.')
    p.add_argument('--oracle-action-match', type=str, default=None,
                    help='Path to an exact-DP oracle npz (e.g. artifacts/gate2_exact_dp.npz.toy_t30) '
                         'to score the fitted policy per-depth vs the optimal action. Only '
                         'meaningful on the matching toy deal (configs_gate2_toy.json).')
    p.add_argument('--label-audit', type=str, default=None,
                    help='Comma-separated timesteps (e.g. "0,6,12") at which to snapshot the '
                         'bootstrap labels the net fits: Y_boot, baseline B_t, and the residual '
                         'target (should be SMALL under advantage decomp). Confirms the labels '
                         'are sane vs exploding (the label-bias signature).')
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
        # BSS sandwich upper bound (opt-in; gap = U − L surfaces in the readout below).
        'Run_Upper_Bound': bool(args.run_upper_bound),
        'Upper_Bound_Wealth_Grid': int(args.upper_bound_wealth_grid),
        'Upper_Bound_Max_Paths': int(args.upper_bound_max_paths),
    }
    # C-stack persistence for out-of-sample validation (train→save, then eval→load + new seed).
    if args.save_value_fn:
        hp['Solver']['Save_Value_Fn_Path'] = args.save_value_fn
    if args.load_value_fn:
        hp['Solver']['Load_Value_Fn_Path'] = args.load_value_fn
    if args.oracle_action_match:
        hp['Solver']['Oracle_Action_Match_Path'] = args.oracle_action_match
    if args.label_audit:
        hp['Solver']['Label_Audit_T_Steps'] = [int(x) for x in args.label_audit.split(',') if x.strip()]
    if args.random_seed is not None:
        calc['Random_Seed'] = int(args.random_seed)

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
            'loaded_value_fn', 'C_loaded_count',   # set on an OOS (load) run
            'V_0', 'M1_v0_decision_at_t_min', 'M1_L_cost_per_oz_usd',
            'M1_penalty_zero_mean_z', 'M1_penalty_boundary_hit_cap',
            # BSS sandwich: lower L (utility), upper U = min(naive, penalized), gap, and the
            # discretised-DP off-grid clamp gate (a high fraction means U is grid-biased).
            'M1_U_lower_bound_L_util', 'M1_U_upper_bound', 'M1_U_gap',
            'M1_U_naive', 'M1_U_penalized', 'M1_U_mean_ge_L',
            'M1_U_off_grid_frac', 'M1_U_vol_over_gridstep', 'M1_U_skipped',
            'M1_C_fitted_count'):
            if key in diagnostics:
                print(f'  {key}: {diagnostics[key]}')

        # --- ASSUMPTIONS CHECK: is this run trustworthy? (mirrors the solver's consolidated
        # log line). A FAIL means the V_0 / sandwich / action-match below it are suspect. ---
        d = diagnostics
        def _g(k, default=None):
            return d.get(k, d.get('M1_' + k, default))
        underfit = _g('conv_underfit_t', []); escalated = _g('conv_escalated_t', [])
        conv_ok = (not underfit and not escalated) if ('M1_conv_n_fitted' in d or 'conv_n_fitted' in d) else None
        v0_over = _g('v0_over_utility_bound')
        pz = _g('penalty_zero_mean_z'); pen_cap = _g('penalty_boundary_hit_cap')
        u_ge_l = _g('U_mean_ge_L')
        print('  ── ASSUMPTIONS ──')
        if conv_ok is not None:
            print(f"  C_t converged: {'OK' if conv_ok else 'FAIL'}  "
                  f"(underfit t={underfit}, escalated t={escalated}, "
                  f"max val_loss={_g('conv_val_loss_max')} @t={_g('conv_val_loss_max_t')})")
        if v0_over is not None:
            print(f"  V_0 within utility bound: {'OK' if not v0_over else 'FAIL (over-optimism)'}")
        if pz is not None:
            print(f"  penalty dual-feasible: {'OK' if (pz < 3.0 and not pen_cap) else 'FAIL'}  (z={pz:.2f})")
        if u_ge_l is not None:
            print(f"  U ≥ L: {'OK' if u_ge_l else 'FAIL'}")

        # --- LABEL AUDIT: the actual bootstrap labels the net is asked to fit, per t.
        # residual = Y_boot − B_t is what the NN regresses to; under advantage decomp it
        # should be SMALL. A large/growing residual (esp. |residual| ≫ |B|) is label-bias. ---
        la = _g('label_audit_at_t')
        if isinstance(la, dict) and la:
            print('  ── LABEL AUDIT (what the net fits, per t) ──')
            print(f"  {'t':>3} {'live':>4} {'Y_boot|mean|':>13} {'B|mean|':>12} "
                  f"{'residual|mean|':>14} {'residual_max':>13} {'gradN_med':>10}")
            for t_key in sorted(la, key=lambda k: int(k)):
                a = la[t_key]
                print(f"  {int(t_key):>3} {a['live_axes']:>4} "
                      f"{a['Y_boot']['abs_mean']:>13.4g} {a['baseline_B']['abs_mean']:>12.4g} "
                      f"{a['residual_y_target']['abs_mean']:>14.4g} "
                      f"{a['residual_y_target']['max']:>13.4g} "
                      f"{a['grad_label_norm']['median']:>10.4g}")
            print("  (residual should be SMALL/O(1); |residual| ≫ |B| or residual growing "
                  "with bank ⇒ label-bias, not a net-size problem.)")

        # --- §7 DOWNSIDE-PROTECTION VERDICT: learned vs unhedged vs best-static on the
        # SAME shared-shock batch. The practical "did riskflow solve the hedge" check
        # (there is no exact-DP oracle for the platinum deal). $/oz; down_sd = RMS of the
        # loss side only. SOLVED iff learned down_sd ≤ unhedged AND ≤ static. ---
        vol = _g('bench_volume_oz') or _g('L_volume_oz')
        if vol and _g('bench_learned_downside_sd_usd') is not None:
            print('  ── DOWNSIDE-PROTECTION VERDICT ($/oz) ──')
            print(f"  {'policy':10s} {'mean':>10s} {'down_sd':>10s} {'5%worst':>10s} {'95%best':>10s}")
            for name in ('unhedged', 'static', 'learned'):
                print(f"  {name:10s} "
                      f"{_g(f'bench_{name}_mean_usd')/vol:>+10.4f} "
                      f"{_g(f'bench_{name}_downside_sd_usd')/vol:>10.4f} "
                      f"{_g(f'bench_{name}_p5_usd')/vol:>+10.4f} "
                      f"{_g(f'bench_{name}_p95_usd')/vol:>+10.4f}")
            beats_un = _g('bench_learned_beats_unhedged_downside')
            beats_st = _g('bench_learned_beats_static_downside')
            print(f"  SOLVED: learned cuts downside vs unhedged={'OK' if beats_un else 'FAIL'}, "
                  f"vs static={'OK' if beats_st else 'FAIL'}  (n*_static={_g('bench_n_star_static')})")
            L_cost = _g('L_cost_per_oz_usd')
            if L_cost is not None:
                print(f"  SHIP CRITERION: L cost={L_cost:+.4f} $/oz vs margin "
                      f"${args.margin_usd_per_oz:.2f}/oz → "
                      f"{'CLEARS' if L_cost <= args.margin_usd_per_oz else 'MISSES'}")

        # --- Exact-DP action-match verdict (if --oracle-action-match was set) ---
        if _g('oracle_match_skipped') is False:
            print('  ── EXACT-DP ACTION-MATCH ──')
            print(f"  V_0={_g('v0_decision_at_t_min', d.get('V_0'))} vs oracle V_0={_g('oracle_V0')}")
            print(f"  exact-match: all={100*_g('oracle_match_exact_frac',0):.1f}%  "
                  f"preA={100*_g('oracle_match_exact_frac_preA',0):.1f}%  "
                  f"postA={100*_g('oracle_match_exact_frac_postA',0):.1f}%  "
                  f"mean|Δq|={_g('oracle_match_mean_abs_dq')}  "
                  f"err-vs-depth slope={_g('oracle_match_err_vs_depth_slope')}")


if __name__ == '__main__':
    main()
