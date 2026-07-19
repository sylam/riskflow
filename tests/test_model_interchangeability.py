"""Model interchangeability: ONE calc config, TWO spot-model worlds.

The property under test — "switch a model in or out and the rest of the calc works as expected."
The base config (deal + tradables + calc + solver knobs) is IDENTICAL across both runs; the ONLY
difference is the `MergeMarketData.MarketDataFile` pointing at a different `Price Models` block:

  * HMM world   — artifacts/MarketDataRF_platinum_calibrated_cme.json (MarkovHMMSpotModel primary)
  * GARCH world — artifacts/MarketDataRF_platinum_garch.json           (GARCHSpotModel primary)

Zero calc/config code changes are needed to swap: the calc speaks only the model-agnostic
StochasticProcess protocol (privileged_layout / reveal_state_at / inner_fork_seed / outer_reseed /
reseed_from_path / reseed_inner_state / diff_state_leaves), and every model-specific buffer key and
recursion lives inside the process class.

Covers:
  1. simulate_only + a full stepper replay to done, both worlds (exercises generate + the observed
     replay reseed path with zero config change).
  2. The deep-state privileged layout tracks the swapped model (regime one-hot/belief → log-variance).
  3. A few-iter solve_hedge, both worlds: bounded, and the V̂ market width resizes with the model
     (HMM market_dim 9 = belief(3)+price(1)+shared(5); GARCH 7 = log_h(1)+price(1)+shared(5)).
"""
import json
import os

import torch

import riskflow as rf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPING = os.path.join(REPO, 'artifacts', 'platinum_hedge_shipping.json')
HMM_MKT = os.path.join(REPO, 'artifacts', 'MarketDataRF_platinum_calibrated_cme.json')
GARCH_MKT = os.path.join(REPO, 'artifacts', 'MarketDataRF_platinum_garch.json')


def _cfg(market_file, mode):
    """The shipping config with ONLY the market-data file swapped (+ tiny CPU sizing)."""
    cfg = json.load(open(SHIPPING))
    cfg['Calc']['MergeMarketData']['MarketDataFile'] = market_file
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = mode
    calc['Batch_Size'] = 24
    calc['Simulation_Batches'] = 1
    calc['Random_Seed'] = 1
    if mode == 'simulate_only':
        calc['Hedging_Problem'].pop('Solver', None)
    else:
        calc['Inner_Sub_Batch'] = 4
        calc['Inner_MC_Enabled'] = 'Yes'
        calc['Hedging_Problem']['Randomize_Initial_State'] = 'No'
        sol = calc['Hedging_Problem']['Solver']
        sol['DiffV2_Fit_Iters'] = 1
        sol['Training_Action_Grid_Levels_Per_Axis'] = 3
        sol['DiffV2_Hidden'] = 8
    return cfg


def _run(market_file, mode):
    cx = rf.Context()
    cx.load_json((json.dumps(_cfg(market_file, mode), default=str), 'interchange.json'))
    _, res = cx.run_job()
    return res


def test_simulate_only_and_stepper_both_worlds():
    layouts = {}
    for name, mkt in (('HMM', HMM_MKT), ('GARCH', GARCH_MKT)):
        res = _run(mkt, 'simulate_only')
        # A full stepper replay to done — exercises the observed-path reseed with zero config change.
        stepper = res.create_stepper()
        steps = 0
        while not stepper.done:
            last = stepper.step(None)
            steps += 1
        assert steps > 0, f'{name}: stepper did not advance'
        pnl = (last['transition_pnl_excess'] + last['transition_liability_value'])
        assert torch.isfinite(pnl).all(), f'{name}: non-finite stepper P&L'
        layouts[name] = res.runtime['privileged_layout']

    # The deep-state privileged layout tracks the swapped model, with zero calc change.
    assert layouts['HMM'] == {'platinum_cme_regime_onehot': 3, 'platinum_cme_regime_belief': 3}, layouts['HMM']
    assert layouts['GARCH'] == {'platinum_cme_log_h': 1}, layouts['GARCH']


def test_solve_reveal_width_9_vs_7():
    dims = {}
    for name, mkt in (('HMM', HMM_MKT), ('GARCH', GARCH_MKT)):
        diag = (_run(mkt, 'solve_hedge').evaluation_summary or {}).get('diagnostics') or {}
        assert diag.get('bounded') is True, f'{name}: solve not bounded'
        dims[name] = diag.get('market_dim')
    # V̂ market width resizes with the model: HMM belief(3)+price(1)+shared(5)=9; GARCH log_h(1)+price(1)+5=7.
    assert dims['HMM'] == 9, dims['HMM']
    assert dims['GARCH'] == 7, dims['GARCH']
    assert dims['HMM'] - dims['GARCH'] == 2, dims
