"""Spot_Price_History is OPTIONAL in the hedging JSON contract.

With no `Portfolio_State.Spot_Price_History`, the solver must still train sane:
  * `referenced_commodities` is derived from the deal's live CommodityPrice factors (never
    from history keys), so the declared-underlying set is unchanged;
  * the utility scale falls back to CALIBRATED market data — spot from the CommodityPrice
    factor, σ from the underlying MarkovHMMSpotModel's stationary regime-weighted vol
    (`calibrated_annual_vol`) — instead of the realized-vol read off a history window;
  * the history prefix no-ops (`initial_time_index == 0`), value bounded, artifact present.

JSON-is-the-contract: load_json + run_job, history removed in code (the fixture template is
never edited). Companion to test_utility_scale_unit.py (the fail-loud unit coverage)."""
import json as jsonlib
import math
import os

import numpy as np

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg_without_history():
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'] = 48
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Random_Seed'] = 1234
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Portfolio_State'].pop('Spot_Price_History', None)   # <-- the whole point
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': 100,
        'DiffV2_Fit_Iters': 5,
        'DiffV2_OOS_Frac': 0.5,
    }
    return cfg


def test_spot_price_history_optional_trains_via_calibrated_scale():
    assert 'PycharmProjects' in rf.__file__, rf.__file__   # guard the reference-riskflow shadow
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(_cfg_without_history()), 'spot_history_optional.json'))
    _, result = cx.run_job()
    bundle, runtime = result.bundle, result.runtime
    diag = (result.evaluation_summary or {}).get('diagnostics') or {}

    # Underlying set derived from the deal (live CommodityPrice factors), not history keys.
    assert runtime['referenced_commodities'] == ('CommodityPrice.PLATINUM_LME',), \
        runtime['referenced_commodities']

    # History is genuinely absent: no history tensor, prefix no-ops.
    assert 'spot_price_history' not in bundle
    assert int(bundle['initial_time_index']) == 0

    # Utility scale came from the calibrated fallback (spot + stationary σ), not the floor.
    calib = bundle.get('calibrated_utility_inputs')
    assert calib is not None, 'calibrated fallback not assembled'
    commodity, spot, sigma = calib
    assert commodity == 'CommodityPrice.PLATINUM_LME'
    assert spot > 0.0 and sigma > 0.0
    c = bundle['utility_scale']
    assert c > 1.0e3, f'utility_scale collapsed to the floor: {c}'
    # c = total_leg_volume · spot · σ · √τ — reproduce it from the parts.
    tau = max(float(int(bundle['last_settlement_index']) - int(bundle['initial_time_index'])) / 252.0,
              1.0 / 252.0)
    expected_c = float(bundle['total_leg_volume']) * spot * sigma * (tau ** 0.5)
    assert math.isclose(c, expected_c, rel_tol=1e-6), (c, expected_c)

    # σ is the stationary regime-weighted vol of the calibrated 3-state HMM (Log_Price=True).
    P = np.array([[0.9903413013204756, 0.009412628327557179, 0.0002460703519671834],
                  [0.007917212174787837, 0.9783132181049096, 0.013769569720302594],
                  [0.0011542393670975122, 0.32217292523078384, 0.6766728354021186]])
    sig = np.array([0.15171719016534066, 0.24138966217119134, 0.6216348894195312])
    evals, evecs = np.linalg.eig(P.T)
    pi = np.real(evecs[:, int(np.argmin(np.abs(evals - 1.0)))]); pi = pi / pi.sum()
    expected_sigma = float(np.sqrt(float((pi * sig * sig).sum())))
    assert math.isclose(sigma, expected_sigma, rel_tol=1e-9), (sigma, expected_sigma)

    # Trains sane: bounded value, artifact present.
    assert diag.get('bounded') is True
    assert math.isfinite(float(diag['V_0'])) and abs(float(diag['V_0'])) < 50.0
    assert result.policy_artifact is not None
