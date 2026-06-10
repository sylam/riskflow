"""Gate 4 — drive the differential-ML solver on the toy that matches
`gate2_exact_dp.py` exactly, so it can be scored cell-by-cell against the DP oracle.

Per `gate4_diffml_solver_spec.md` §3: weekly dt = 5/252, T_dec = 30 weekly decision
points, T_A = 15, q ∈ [-5, 5], λ_turn = 1e-4. The fixture is a TEMPLATE; weekly
economics live here, not in the JSON file.

The oracle at `artifacts/gate2_exact_dp.npz` must have been produced with the same
T_dec; the existing 150 MB artifact in the tree is T_dec=10 (a smaller debug run) —
regenerate with `python gate2_exact_dp.py` (defaults are T_dec=30) before any §13
cell-by-cell comparison is meaningful.
"""
import argparse
import json as jsonlib
import logging
import re
from pathlib import Path

import riskflow as rf

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


# Gate 2 toy parameters (must match gate2_exact_dp.py ToyConfig defaults at S_0=100).
TOY_S0 = 100.0
TOY_K = 100.0
TOY_C = 100.0                                    # symlog scale
TOY_LAMBDA_TURN = 1.0e-4                         # L1 turnover per unit per spot

# Date arithmetic at weekly grid. Base + N*7 days lands on the same weekday, so a
# weekly Time_Grid lands exactly on each contract's expiry without business-day jitter.
BASE_DATE = '2026-04-10'                         # Friday — keeps weekly steps on Fridays
# Liability has a degenerate averaging window (7 days before T_dec) — matches gate3a's
# pattern and lets the framework's energy-leg machinery price the K - S_T_dec payoff
# via Avg ≈ S_T_dec.
LIABILITY_AVG_WINDOW_DAYS = 7


def load_template():
    """Extract the inline JSON literal from policy_test.py — single source of truth for
    the LsmDpSolver config; we then override what we need for the weekly toy."""
    source = Path(__file__).parent.joinpath('policy_test.py').read_text()
    m = re.search(r"json\s*=\s*'''(.*?)'''", source, re.DOTALL)
    if not m:
        raise SystemExit('could not find inline JSON in policy_test.py')
    return jsonlib.loads(m.group(1))


def _ts(days):
    """Format an offset-from-base date as a riskflow `{".Timestamp": "YYYY-MM-DD"}`."""
    import pandas as pd
    return {'.Timestamp': (pd.Timestamp(BASE_DATE) + pd.Timedelta(days=days))
            .strftime('%Y-%m-%d')}


def build_toy_cfg(solver_object='DifferentialSolver', lambda_mix=0.0,
                  bank_total=16384, bank_b_exo=8192, bank_b_endo=2,
                  t_dec_weeks=30, t_a_weeks=15, t_min=0):
    t_a_days = int(t_a_weeks * 7)
    t_dec_days = int(t_dec_weeks * 7)
    t_dec_period_start_days = t_dec_days - LIABILITY_AVG_WINDOW_DAYS
    cfg = load_template()
    calc = cfg['Calc']['Calculation']

    # Weekly sim grid — the spec is explicit (§3): weekly dt, weekly P applied per
    # step, NO daily/CTMC re-discretisation. The HMM's CTMC machinery is a no-op
    # here because Calibration_DT_Years = sim dt = 5/252; P_per_step == P_calib.
    # Time_Grid syntax: `<first>(<step>)` — `'0d 7d(7d)'` puts the first scenario at
    # base+0d, then steps by 7d from base+7d through max_date (the deal terminal).
    calc['Time_Grid'] = '0d 7d(7d)'
    calc['Base_Date'] = {'.Timestamp': BASE_DATE}
    # B_outer = exogenous bank size. Inner_Sub_Batch=128 — the framework enforces
    # ≥128 (hedge_runtime.py:486). The diff-ML design (spec §6/§12) only NEEDS M=1
    # or 2 for the bootstrap label, but the framework's existing inner-MC machinery
    # is built around larger M; we just use `market_t1[:, 0, :]` and discard the
    # other 127 draws. The wasted compute is bounded by the single-chunk constraint
    # B_outer × B_inner ≤ _INNER_MC_FLAT_LIMIT (default 32768; the driver caller
    # should set RF_INNER_MC_FLAT_LIMIT for larger B_outer).
    calc['Batch_Size'] = int(bank_b_exo)
    calc['Inner_Sub_Batch'] = 128 if solver_object == 'DifferentialSolver' else 256

    hp = calc['Hedging_Problem']

    # Two staggered futures, no carry/basis (forward = spot per gate2). PL_A expires
    # at 15w, PL_B at 30w == T_dec. Drop platinum's third contract.
    hp['Tradable_Instruments']['CommodityFutureDeal'] = {
        'PL_A': {
            'Maturity_Date': _ts(t_a_days),
            'Currency': 'USD',
            'Carry': 'PLATINUM_CARRY',
            'Repo_Rate': 'USD-SOFR',
            'Implied_Basis': 'LME_CME',
            'Contract_Size': 1.0,
        },
        'PL_B': {
            'Maturity_Date': _ts(t_dec_days),
            'Currency': 'USD',
            'Carry': 'PLATINUM_CARRY',
            'Repo_Rate': 'USD-SOFR',
            'Implied_Basis': 'LME_CME',
            'Contract_Size': 1.0,
        },
    }

    ps = hp['Portfolio_State']
    ps['Positions'] = {'PL_A': 0, 'PL_B': 0}
    ps['Cash_Balances'] = {'USD_CASH': TOY_S0}
    ps['Settlement_Prices'] = {'PL_A': TOY_S0, 'PL_B': TOY_S0}
    if 'Initial_Margin' in ps:
        ps['Initial_Margin'] = {
            'PL_A': {'Method': 'per_contract', 'Amount': 0.0},
            'PL_B': {'Method': 'per_contract', 'Amount': 0.0},
        }
    if 'Margin_Balances' in ps:
        ps['Margin_Balances'] = {'USD_CASH': 0.0}

    # Liability K - S_T_dec. Volume=-1, Fixed_Basis=K → payment = K - Avg ≈ K - S_T_dec
    # over the 7d averaging window; framework requires a non-degenerate window.
    hp['Liabilities']['FloatingEnergyDeal'] = {
        'PLAT_TERMINAL': {
            'Currency': 'USD',
            'Sampling_Type': 'USD',
            'FX_Sampling_Type': 'USD',
            'Discount_Rate': 'USD-SOFR',
            'Commodity': 'PLATINUM_LME',
            'Reference_Type': 'PLATINUM',
            'Payer_Receiver': 'Receiver',
            'Payments': {
                'Items': [{
                    'Payment_Date': _ts(t_dec_days),
                    'Period_Start': _ts(t_dec_period_start_days),
                    'Period_End': _ts(t_dec_days),
                    'Volume': -1.0,
                    'Fixed_Basis': TOY_K,
                    'Price_Multiplier': 1.0,
                    'FX_Period_Start': _ts(t_dec_period_start_days),
                    'Realized_Average': 0.0,
                    'FX_Period_End': _ts(t_dec_days),
                    'FX_Realized_Average': 0.0,
                }],
            },
        },
    }

    # Symlog with explicit c=100 (matches gate2). Strip the production penalty stack —
    # toy has no margin/position-limit shaping.
    hp['Objective'] = {
        'Object': 'AsymmetricUtility_Symlog',
        'Utility_Scale_Mode': 'vol_scaled_notional',
        'Utility_Scale_Explicit': TOY_C,
        'Floor_Penalty': 1.0,
        'Surplus_Reward': 1.0,
        'Power': 1.0,
        'Expiry_Penalty': 0.0,
        'Expiry_Threshold_Days': 0.0,
        'Post_Deal_Trade_Penalty': 0.0,
        'Position_Bounds_Penalty': 0.0,
        'Per_Instrument_Bounds_Penalty': 0.0,
    }

    # Position limits ±5 per contract (matches gate2's q_grid). λ_turn = 1e-4 per spec §3.
    hp['Evaluator']['Position_Limits'] = {
        'PL_A': {'Min_Position': -5, 'Max_Position': 5},
        'PL_B': {'Min_Position': -5, 'Max_Position': 5},
    }
    hp['Evaluator']['Total_Position_Abs_Limit'] = 10
    hp['Evaluator']['Transaction_Cost_Per_Unit'] = TOY_LAMBDA_TURN
    hp['Evaluator']['Bid_Offer_Spread_Bps'] = 0.0

    # Solver dispatch. Default to the new DifferentialSolver; allow override for
    # cross-checking against LsmDpSolver on the same toy economics.
    hp['Solver']['Object'] = solver_object

    # Differential-ML bank construction (spec §6). Only ONE swappable axis lives at the
    # framework boundary — the exogenous forward sweep, exposed via the solver's
    # `sample_exogenous(n, seed)` seam. Path-count breadth at the TRUE t0 is the
    # correct query distribution for exogenous coordinates (the argmax doesn't control
    # them; perturbing t0 would sample regions the deployed policy never visits and
    # invalidate audit-rollout validity). The ENDOGENOUS span (inventory, wealth) is
    # the only knob configured per axis — layered on top of each exogenous slice
    # inside the solver. Stratified-regime or t0-spot-perturbation: NOT part of the
    # design; only added if a regime-occupancy diagnostic shows starvation on this
    # specific toy (the calibrated P sits near the 77/23 stationary distribution, so
    # both regimes are well-sampled at B_exo = 8k).
    hp['Solver']['Bank_Sampling'] = {
        'B_Endo': int(bank_b_endo),
    }
    # Gate 4 twin-net defaults (spec §9 / 3090 budget): 3×128 softplus MLP, Adam 1e-3,
    # minibatch 4096, 2000 SGD steps per C_t fit. These override LsmDpSolver's smaller
    # (64,64,64) head and zero-train default — DifferentialSolver is a different fitter.
    hp['Solver']['Value_Fn'] = {
        'MLP_Hidden': [128, 128, 128],
        'MLP_Train_Steps_Per_Solve': 2000,
        'MLP_Adam_LR': 1.0e-3,
        'MLP_Minibatch': 4096,
    }
    hp['Solver']['T_Min'] = int(t_min)
    hp['Solver']['Lambda_Mix'] = float(lambda_mix)

    # MarketDataRF overrides — 2-state HMM with the gate2 parameters, zero basis vol,
    # near-zero carry vol (forward ≈ spot), spot rescaled to S_0 = 100.
    md = cfg['Calc']['MergeMarketData']
    em = md.setdefault('ExplicitMarketData', {})
    pm = em.setdefault('Price Models', {})
    pf = em.setdefault('Price Factors', {})

    pm['MarkovHMMSpotModel.PLATINUM_LME'] = {
        'Log_Price': True,
        'States': [
            {'Mu': 0.50, 'Sigma': 0.15, 'Nu': 1.0e6},     # Nu high ⇒ Gaussian innovations
            {'Mu': -0.50, 'Sigma': 0.40, 'Nu': 1.0e6},
        ],
        'Transition_Matrix': [[0.97, 0.03], [0.10, 0.90]],
        'Initial_State_Probs': [0.5, 0.5],
        'Calibration_DT_Years': 5.0 / 252.0,              # = sim dt ⇒ P per step == P_calib
    }
    pm['BasisLinkedSpotModel.LME_CME'] = {
        'A': 0.0, 'Phi': 0.99, 'Nu': 30.0, 'Mu': 0.0,
        'Sigma_By_State': [0.0, 0.0],
        'Calibration_DT_Years': 5.0 / 252.0,
    }
    pm['CSForwardPriceModel.PLATINUM'] = {
        'Sigma': 1.0e-4, 'Alpha': 1.0, 'Drift': 0.0,
    }
    # PLATINUM_CARRY is the ForwardRate factor the futures contracts hook into. The
    # platinum market data assigns it a VARMixedFactorInterestRateModel calibrated at
    # daily dt; that model rejects weekly sim steps (Brownian approximation valid only
    # for δ ≈ δ_calib). Override Calibration_DT_Years to weekly so the sim/calib ratio
    # is 1.0 (check passes), and zero σ so forward ≈ spot per gate2's "no carry/basis"
    # spec. Phi/Mean keep their calibrated shape — irrelevant once Sigma=0.
    pm['VARMixedFactorInterestRateModel.PLATINUM_CARRY'] = {
        'Mean': [0.0, 0.0, 0.0],
        'Phi': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'Sigma': [0.0, 0.0, 0.0],
        'Calibration_Tenors': [0.086111111, 0.336111111, 0.591666667],
        'Contract_Cycle_Years': 0.255555555,
        'Calibration_DT_Years': 5.0 / 252.0,
    }
    # USD-SOFR uses PCAInterestRateModel by default — also needs weekly check; let it
    # discount near-flat by overriding to no-vol.
    pm['PCAInterestRateModel.USD-SOFR'] = {
        'Reversion_Speed': 0.1,
        'Yield_Volatility': {'.Curve': {'meta': [], 'data': [[0.01, 1.0e-6], [30.0, 1.0e-6]]}},
        'Eigenvectors': [
            {'Eigenvector': {'.Curve': {'meta': [], 'data': [[0.01, 1.0], [30.0, 1.0]]}}, 'Eigenvalue': 1.0},
        ],
        'Rate_Drift_Model': 'Drift_To_Forward',
        'Princ_Comp_Source': 'Covariance',
        'Distribution_Type': 'Lognormal',
    }
    pf['CommodityPrice.PLATINUM_LME'] = {
        'Currency': 'USD',
        'Interest_Rate': 'USD-SOFR',
        'Spot': TOY_S0,
        'Property_Aliases': '',
    }
    pf['CommodityBasis.LME_CME'] = {
        'Spot': 0.0,
        'Observed_Commodity': 'PLATINUM_LME',
    }

    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--solver', type=str, default='DifferentialSolver',
                   choices=['DifferentialSolver', 'LsmDpSolver'],
                   help='Which solver to dispatch. Use LsmDpSolver for a sanity-check'
                        ' run on the weekly economics before the new solver is wired.')
    p.add_argument('--bank-b-exo', type=int, default=8192)
    p.add_argument('--bank-b-endo', type=int, default=2)
    p.add_argument('--t-dec-weeks', type=int, default=30,
                   help='Deal terminal T_dec in weekly decision points. Spec default 30; '
                        'use 10 to match the existing artifacts/gate2_exact_dp.npz oracle.')
    p.add_argument('--t-a-weeks', type=int, default=15,
                   help='Contract A expiry T_A in weeks. Spec default 15; use 5 for T_dec=10.')
    p.add_argument('--lambda-mix', type=float, default=0.0,
                   help='Spec §14 λ-mix: blend (1-λ)·Y_boot + λ·Y_rollout for the '
                        'value label. 0 = pure bootstrap (banked default). Indicated '
                        'when a horizon-stable bounded residual gap survives advantage '
                        'decomposition.')
    p.add_argument('--t-min', type=int, default=0,
                   help='Backward sweep target (smallest t). 0 = full sweep down to '
                        'the initial decision (Milestone 3); set T_A-1 for M1.5 — '
                        'validates the multi-contract code path across the T_A '
                        'transition without the full 30-step compute.')
    args = p.parse_args()

    cfg = build_toy_cfg(solver_object=args.solver,
                        bank_total=args.bank_b_exo * args.bank_b_endo,
                        bank_b_exo=args.bank_b_exo, bank_b_endo=args.bank_b_endo,
                        t_dec_weeks=args.t_dec_weeks, t_a_weeks=args.t_a_weeks,
                        t_min=args.t_min, lambda_mix=args.lambda_mix)
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'gate4_diffml_toy.json'))

    print(f'Running Gate 4 toy via cx.run_job '
          f'(solver={args.solver}, B_exo={args.bank_b_exo}, B_endo={args.bank_b_endo}, '
          f'T_dec={args.t_dec_weeks}w, T_A={args.t_a_weeks}w)...')
    _, result = cx.run_job()

    es = result.evaluation_summary or {}
    comp = es.get('comparison') or {}

    print('\n' + '=' * 78)
    print(f'Gate 4 — toy in framework (solver={args.solver})')
    print('=' * 78)
    print('DP oracle target (gate2 T_dec=30 defaults): V_0 ≈ +0.864 at c=100')
    print('NOTE: artifacts/gate2_exact_dp.npz currently holds the T_dec=10 debug run; '
          're-run gate2_exact_dp.py before §13 cell-by-cell comparison.')
    print()
    for name in ('DifferentialSolver', 'LsmDpSolver', 'MpcSolver',
                 'HindsightDpSolver', 'textbook'):
        d = comp.get(name, {})
        v0 = d.get('v0_mean')
        v_std = d.get('v0_std')
        if v0 is not None:
            line = f'  {name:<22}: V_0 = {v0:+.4f}'
            if v_std is not None:
                line += f' ± {v_std:.4f}'
            n_star = d.get('n_star')
            if n_star is not None:
                line += f'   n*_0 = {n_star}'
            print(line)


if __name__ == '__main__':
    main()
