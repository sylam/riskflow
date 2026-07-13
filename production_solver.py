"""Production DiffSolverV2 hedge solver — trains (or evaluates) a day-1 hedge policy with the
validated best configuration.

BEST CONFIG (validated 2026-07): full decision window (T_Min=0), 8k outer paths, 64 inner
draws, one-step forks + trust-region clamp + per-column differential normalization (Huge-Savine
lambda_j) + cost-aware argmax + antithetic inner draws, on the Components forward-curve world.
Single-seed training is reliable; pass several seeds for an ensemble-argmax deployment (each
member evaluated in its own frame, continuations averaged — cross-fit winner's-curse reduction).

Two modes:
  TRAIN   — solve the backward DP, save the fitted value function(s) (per seed), print the
            out-of-sample verdict (greedy vs textbook vs no-hedge).
  EVAL    — load frozen checkpoint(s) and score them on fresh paths (no training). A LIST of
            checkpoints runs the ensemble argmax.

Usage:
    # train one policy, save the checkpoint:
    python production_solver.py --config policy.json --save value_fn.pt

    # train an ensemble (one checkpoint per seed -> value_fn_s7.pt, value_fn_s42.pt, ...):
    python production_solver.py --config policy.json --save value_fn.pt --seeds 7 42 314

    # scale the batch up on a bigger GPU (the recipe is stable to at least 8192; 16k needs care):
    python production_solver.py --config policy.json --save value_fn.pt --batch 16384

    # evaluate a frozen ensemble on fresh paths (no training):
    python production_solver.py --config policy.json --load value_fn_s7.pt value_fn_s42.pt

`policy.json` is a complete HedgeMonteCarlo config: MergeMarketData (the calibrated market data,
in the Components forward-curve world) + Hedging_Problem (the deal + tradables + evaluator).
The best solver/calc knobs below are applied on top, so the JSON need not carry a Solver block.

This solver is config-agnostic — it applies the same best block to whatever world the JSON
describes. The shipping deliverable (artifacts/platinum_hedge_shipping.json) is the corrected
composed-spot platinum world: CommodityPrice.PLATINUM_CME = P is the martingale primary,
CommodityBasis.LME_CME (Linked_Commodity=PLATINUM_CME) carries the published basis b = P - S,
and CommodityPrice.PLATINUM_LME = P - b is the composed LBMA fixing (BasisComposedSpotModel,
routed by modelfilters, never calibrated). The world's invariant is E[dF|b] ≈ 0 — every tradeable
future references P (martingale), so its expected one-step change conditional on the basis is ~0
(no unexecutable reversion for the solver to harvest); enforce it at calibration.

JSON-is-the-contract: import riskflow, load_json, run_job. No internal imports, no monkey-patching.
"""
import argparse
import copy
import json
import logging
import os

import riskflow as rf

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

# ---- validated best configuration -------------------------------------------------------------
BEST_CALC = {
    'Batch_Size': 8192,        # outer paths; scale up on a bigger GPU (memory ~ linear in B)
    'Inner_Sub_Batch': 64,     # inner draws — the selection floor for near-identical hedges
    'Inner_MC_Enabled': 'Yes',
    'Inner_Antithetic': 'Yes', # mirrored (z,-z) inner pairs — halves argmax selection variance
}
BEST_SOLVER = {
    'Object': 'DiffSolverV2',
    'Training_Action_Grid_Levels_Per_Axis': 9,   # action-grid resolution per hedge axis
    'Training_Action_Chunk_Size': 64,
    'T_Min': 0,                                   # full window from day 0 = a day-1 policy
    'DiffV2_Fit_Iters': 60,
    'DiffV2_Hidden': 32,
    'DiffV2_OOS_Frac': 0.5,                       # held-out split for the honest verdict
    'DiffV2_LR': 0.002,
    'DiffV2_Cost_Aware_Argmax': 'Yes',            # charge repositioning cost at the argmax
    'DiffV2_One_Step_Fork': 'Yes',                # {t,t+1} fork window (default) — required < inner=64
    'DiffV2_Per_Column_Grad_Norm': 'Yes',         # per-input-column differential normalization
}


def apply_config(cfg, *, batch=None, seed=7, save=None, load=None,
                 stepper_rollout=False, randomize_initial_state=True):
    """Apply the validated best config to a policy JSON (mutated in place) and return it.

    save                 — checkpoint path to write the fitted value function to (train mode).
    load                 — list of checkpoints to restore and evaluate frozen (eval mode); a
                           list of >1 runs the ensemble argmax.
    stepper_rollout      — also roll the frozen policy day-by-day via BundleStepper (real futures
                           accounting) and expose diagnostics['stepper_verdict']. Used by the
                           walk-forward backtest; ignored in plain train mode.
    randomize_initial_state — 'Yes' for training (Huge-Savine boundary variance), 'No' for a
                           deterministic single-path evaluation.
    """
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc.update(BEST_CALC)
    if batch is not None:
        calc['Batch_Size'] = batch
    calc['Random_Seed'] = seed
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes' if randomize_initial_state else 'No'
    solver = dict(BEST_SOLVER)
    if save:
        solver['DiffV2_Save_Value_Fn'] = os.path.abspath(save)
    if load:
        solver['DiffV2_Load_Value_Fn'] = [os.path.abspath(p) for p in load]
    if stepper_rollout:
        solver['DiffV2_Stepper_Rollout'] = 'Yes'
    hp['Solver'] = solver
    return cfg


def run(cfg, name='policy'):
    """Run one solve_hedge job via the JSON contract; return its diagnostics dict."""
    cx = rf.Context()
    cx.load_json((json.dumps(cfg, default=str), name + '.json'))
    _, result = cx.run_job()
    return (result.evaluation_summary or {}).get('diagnostics') or {}


def print_verdict(diag, key='verdict'):
    """Pretty-print the greedy/textbook/no-hedge terminal-wealth verdict."""
    v = diag.get(key) or {}
    print(f"\n[{key}] V_0={diag.get('V_0')} bounded={diag.get('bounded')} "
          f"max|Y_boot|={diag.get('max_abs_Y_boot')}")
    for p in ('greedy', 'textbook', 'nohedge'):
        s = v.get(p) or {}
        if s.get('wT_mean') is not None:
            print(f"  {p:<9} u={s.get('u_mean', 0):+.4f}  E[W_T]={s['wT_mean']:+,.0f}  "
                  f"p5={s.get('wT_p5', 0):+,.0f}  cvar5={s.get('wT_cvar5', 0):+,.0f}")
    print('  greedy q@t0 =', v.get('greedy_q_first'))


def _seeded_ckpt(save, seed, multi):
    if not save or not multi:
        return save
    stem = save[:-3] if save.endswith('.pt') else save
    return f'{stem}_s{seed}.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Complete policy JSON (market data + deal + calc).')
    ap.add_argument('--save', help='Checkpoint path (a _s<seed> suffix is added for multi-seed).')
    ap.add_argument('--load', nargs='*', help='Frozen checkpoint(s) to evaluate (>1 = ensemble argmax).')
    ap.add_argument('--seeds', type=int, nargs='+', default=[7], help='Training seed(s) / ensemble members.')
    ap.add_argument('--batch', type=int, help='Override Batch_Size (scale up on a big GPU).')
    args = ap.parse_args()
    template = json.load(open(args.config))

    if args.load:  # ---- EVAL: frozen policy on fresh paths, no training ----
        cfg = apply_config(copy.deepcopy(template), batch=args.batch, seed=args.seeds[0],
                           load=args.load, randomize_initial_state=False)
        logging.info('=== EVAL frozen %s ===', args.load)
        print_verdict(run(cfg, 'eval'))
        return

    multi = len(args.seeds) > 1
    for seed in args.seeds:  # ---- TRAIN: one checkpoint per seed ----
        ckpt = _seeded_ckpt(args.save, seed, multi)
        cfg = apply_config(copy.deepcopy(template), batch=args.batch, seed=seed, save=ckpt)
        logging.info('=== TRAIN seed=%d save=%s ===', seed, ckpt)
        print_verdict(run(cfg, f'train_s{seed}'))


if __name__ == '__main__':
    main()
