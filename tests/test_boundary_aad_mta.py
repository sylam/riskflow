"""Boundary-aware AAD for the MTA transfer decision.

A minimum transfer amount makes the collateral balance jump discontinuously, so ordinary AAD
differentiates the netting set with the decision FROZEN and drops the term that flows through the
decision itself:

    dE[J]/dtheta = E[ dJ/dtheta | D ]  +  f_G(0) * E[ dJ * dG/dtheta | G = 0 ]
                   ^ what riskflow computes      ^ the missing boundary term

The correction is injected as a term whose FORWARD VALUE IS EXACTLY ZERO, so the reported XVA
cannot move — only the reverse sweep sees it. That property is the backbone of these gates: every
one of them asserts the forward numbers are untouched, which is a far stronger statement than a
tolerance and holds at every stage of the build.

This file also carries the only collateralised netting set in the suite, with an economically
live MTA (transfers actually get suppressed), which is what the boundary term needs to be
non-trivial.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import riskflow
from riskflow import utils
from riskflow.config import Config
from riskflow import calculation
from riskflow.instruments import construct_instrument, scan_collateral_balance

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64


def _cfg(min_transfer, collateralised=True):
    """One FX forward inside a collateralised netting set. `min_transfer` is the MTA in agreement
    currency — large enough and most margin calls are suppressed, which is exactly the regime
    where the frozen-decision derivative is wrong."""
    swap = {
        'Object': 'FXForwardDeal', 'Reference': 'FWD1', 'Buy_Currency': 'EUR',
        'Sell_Currency': 'USD', 'Buy_Amount': 10_000_000.0, 'Sell_Amount': 11_000_000.0,
        'Buy_Discount_Rate': 'EUR', 'Sell_Discount_Rate': 'USD',
        'Settlement_Date': BASE + pd.Timedelta(days=730), 'Discount_Rate': 'USD',
    }
    netting = {
        'Object': 'NettingCollateralSet', 'Reference': 'CSA1', 'Agreement_Currency': 'USD',
        'Apply_Closeout_When_Uncollateralized': 'No', 'Balance_Currency': 'USD',
        'Opening_Balance': 0.0, 'Base_Collateral_Call_Date': BASE,
        'Calendars': None, 'Collateral_Assets': {'Cash_Collateral': [
            {'Currency': 'USD', 'Amount': 1.0, 'Haircut_Posted': 0.0,
             'Collateral_Rate': 'USD', 'Funding_Rate': 'USD'}]},
        'Collateral_Call_Frequency': pd.DateOffset(days=7),
        'Collateralized': 'True', 'Netted': 'True',
        'Credit_Support_Amounts': {
            # CreditSupportList maps rating -> amount and value() returns the amount
            'Received_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
            'Posted_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
            'Independent_Amount': utils.CreditSupportList([[0.0, 0.0]]),
            'Minimum_Received': utils.CreditSupportList([[0.0, min_transfer]]),
            'Minimum_Posted': utils.CreditSupportList([[0.0, min_transfer]])},
        'Funding_Rate': 'USD', 'Liquidation_Period': 10.0, 'Settlement_Period': 0.0,
    }
    if not collateralised:
        netting['Collateralized'] = 'False'

    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Priority': 1, 'Spot': 1.1},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.03], [10.0, 0.03]])},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.02], [10.0, 0.02]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'DiscountRate.EUR': {'Interest_Rate': 'EUR'},
        'SurvivalProb.CPTY': {'Recovery_Rate': 0.4,
                              'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])},
        'FxRateVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.12] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.1, 3.0)])},
    }
    c.params['Price Models'] = {
        'GBMAssetPriceModel.EUR': {'Vol': 0.12, 'Drift': 0.0},
    }
    c.params['Model Configuration'].append('FxRate', (), 'GBMAssetPriceModel')
    c.params['Valuation Configuration'] = {}
    c.deals = {
        'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
        'Deals': {'Children': [
            {'Instrument': construct_instrument(netting, {}),
             'Children': [{'Instrument': construct_instrument(swap, {})}]}]},
        'Calculation': {'Base_Date': BASE, 'Currency': 'USD'},
    }
    return c


def _params(gradient, seed=1, batch=256, batches=1):
    p = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1m(3m) 2y(3m)',
         'Batch_Size': batch, 'Simulation_Batches': batches, 'Random_Seed': seed,
         'Currency': 'USD', 'MCMC_Simulations': 0, 'Tenor_Offset': 0.0,
         'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
         'Credit_Valuation_Adjustment': {
             'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
             'Stochastic_Hazard_Rates': 'No',
             'Gradient': 'Yes' if gradient else 'No'}}
    return p


def _run(min_transfer, gradient, seed=1):
    """Asking for sensitivities is what turns the boundary machinery on - there is no separate
    setting, because a term worth exactly zero in the forward pass has nothing a user would want
    to disable."""
    _, out = riskflow.run_cmc(_cfg(min_transfer), prec=DTYPE,
                              overrides=_params(gradient, seed=seed))
    return out


# ---------------------------------------------------------------------------
# the scan itself, independent of the engine
# ---------------------------------------------------------------------------

def test_scan_reproduces_the_loop_it_replaced():
    """One recursion now serves the daily and the call-mask schedules, which previously had a loop
    each. An all-True mask IS the daily schedule, so the two must agree exactly."""
    torch.manual_seed(0)
    T, B = 24, 512
    required = torch.randn(T, B, dtype=DTYPE).cumsum(0)
    recv, post = required - 0.5, required + 0.5
    opening = torch.zeros(B, dtype=DTYPE)

    def original(call_mask):
        sim = [opening]
        for i in range(1, T):
            if not call_mask[i]:
                sim.append(sim[-1])
                continue
            mask = (sim[-1] < recv[i]) | (sim[-1] > post[i])
            sim.append(required[i] * mask + sim[-1] * (~mask))
        return torch.stack(sim)

    for mask in (np.ones(T, dtype=bool), np.array([i % 3 == 0 for i in range(T)])):
        got, _ = scan_collateral_balance(opening, required, recv, post, mask)
        assert torch.equal(got, original(mask)), 'refactored scan is not the loop it replaced'


def test_gaps_agree_with_the_transfer_decision():
    """receive_gap > 0 or post_gap > 0 must be exactly the transfer the forward path took —
    the gaps are recorded alongside the decision, never used to re-derive it."""
    torch.manual_seed(1)
    T, B = 16, 256
    required = torch.randn(T, B, dtype=DTYPE).cumsum(0)
    recv, post = required - 0.3, required + 0.3
    opening = torch.zeros(B, dtype=DTYPE)
    mask = np.ones(T, dtype=bool)
    path, gaps = scan_collateral_balance(opening, required, recv, post, mask, collect_gaps=True)
    assert len(gaps) == T - 1
    for index, recv_gap, post_gap, previous, _ in gaps:
        transfer = (previous < recv[index]) | (previous > post[index])
        assert torch.equal((recv_gap > 0) | (post_gap > 0), transfer), \
            f'call {index}: gap sign disagrees with the transfer taken'


# ---------------------------------------------------------------------------
# the engine: the forward must not move
# ---------------------------------------------------------------------------

def test_mta_is_economically_live():
    """The fixture only tests anything if the MTA actually suppresses transfers. A zero MTA
    transfers at every call and a large one does not, so the exposure profiles must differ."""
    loose = _run(0.0, gradient=False)['Results']['mtm'].values
    tight = _run(2_000_000.0, gradient=False)['Results']['mtm'].values
    assert not np.allclose(loose, tight), \
        'MTA is not binding — the boundary term would be trivially zero'


@pytest.mark.parametrize('min_transfer', [0.0, 2_000_000.0])
def test_asking_for_sensitivities_does_not_move_the_exposure(min_transfer):
    """THE gate for the whole build, and the property that matters operationally: requesting risk
    must not change the numbers being reported. The correction is worth exactly zero forward by
    construction, so the profile has to be identical bitwise, not within a tolerance."""
    off = _run(min_transfer, gradient=False)['Results']['mtm'].values
    on = _run(min_transfer, gradient=True)['Results']['mtm'].values
    assert np.array_equal(off, on), f'exposure moved: max |d| {np.abs(off - on).max():.3e}'


def test_events_are_registered_only_when_sensitivities_are_wanted():
    """Keeps the bit-identity gate from passing vacuously: if the netting set were skipped, or the
    scan never learned that greeks were wanted, 'unchanged forward' would be true and
    meaningless."""
    import riskflow.instruments as instruments
    original = instruments.scan_collateral_balance
    tally = {}

    def counted(*args, **kwargs):
        path, gaps = original(*args, **kwargs)
        tally['gaps'] = tally.get('gaps', 0) + len(gaps)
        return path, gaps

    instruments.scan_collateral_balance = counted
    try:
        for gradient, expected in ((False, False), (True, True)):
            tally.clear()
            out = _run(2_000_000.0, gradient=gradient)
            assert out['Results']['mtm'].shape[0] > 1, 'netting set did not price'
            assert bool(tally.get('gaps')) is expected, \
                f'gradient={gradient}: collected {tally.get("gaps", 0)} gaps'
    finally:
        instruments.scan_collateral_balance = original


def test_cva_per_scenario_reproduces_the_reported_cva():
    """The counterfactual objective must be the same quantity the engine reports. It is not
    BITWISE equal — summing over time before averaging over paths reverses the reduction order —
    so the reported number keeps its original grouping and this pins the two together."""
    torch.manual_seed(0)
    pv = torch.rand(120, 256, dtype=DTYPE) * 1e7
    prob = torch.rand(119, 1, dtype=DTYPE) * 1e-3
    recovery = 0.4
    reported = (1.0 - recovery) * (0.5 * (pv[1:] + pv[:-1]) * prob).mean(axis=1).sum()
    per_path = calculation.cva_per_scenario(pv, prob, recovery)
    assert per_path.shape == (256,), f'expected a per-path vector, got {tuple(per_path.shape)}'
    assert abs(per_path.mean() - reported) < 1e-12 * abs(reported), 'objective drifted from CVA'


def _run_capturing_shared(min_transfer, seed=1):
    """Grab the shared state so the replay context can be inspected after the run. boundary_sets
    are cleared per batch, so with one batch the last batch's context survives."""
    from riskflow.calculation import Credit_Monte_Carlo
    original = Credit_Monte_Carlo._init_shared_mem
    grabbed = {}

    def capture(self, *args, **kwargs):
        shared = original(self, *args, **kwargs)
        grabbed['shared'] = shared
        return shared

    Credit_Monte_Carlo._init_shared_mem = capture
    try:
        _, out = riskflow.run_cmc(_cfg(min_transfer), prec=DTYPE,
                                  overrides=_params(True, seed=seed))
    finally:
        Credit_Monte_Carlo._init_shared_mem = original
    return out, grabbed['shared']


def test_replay_reproduces_the_reported_mtm():
    """The counterfactual is only meaningful if the replay is the SAME arithmetic the forward pass
    ran. Handed the balance path that actually occurred, it must return the reported netting-set
    MTM — anything else means the closure captured the wrong pieces, and every jump built on it
    would be wrong in a way no forward number could reveal."""
    out, shared = _run_capturing_shared(2_000_000.0)
    assert shared.boundary_sets, 'no boundary context was recorded'
    bset = shared.boundary_sets[0]
    replayed = bset.replay(bset.balance)
    reported = torch.as_tensor(out['Results']['mtm'].values, dtype=replayed.dtype,
                               device=replayed.device)
    assert replayed.shape == reported.shape, f'{tuple(replayed.shape)} vs {tuple(reported.shape)}'
    scale = reported.abs().max().clamp_min(1.0)
    assert (replayed - reported).abs().max() < 1e-9 * scale, \
        f'replay differs from the reported MTM by {(replayed - reported).abs().max():.3e}'


def test_replay_from_a_forced_balance_changes_the_mtm():
    """A forced transfer must actually move the exposure, or the jump is zero and the correction
    cannot be tested. Guards against a replay that silently ignores its opening balance."""
    _, shared = _run_capturing_shared(2_000_000.0)
    bset = shared.boundary_sets[0]
    event = bset.events[len(bset.events) // 2]
    with torch.no_grad():
        suffix, _ = scan_collateral_balance(
            event.required_balance, bset.required, bset.recv_band, bset.post_band,
            bset.call_mask, start=event.call_index)
        forced = torch.cat([bset.balance[:event.call_index], suffix], dim=0)
    assert forced.shape == bset.balance.shape
    assert not torch.equal(forced, bset.balance), 'forcing a transfer changed nothing'
    assert not torch.equal(bset.replay(forced), bset.replay(bset.balance)), \
        'a different balance path produced an identical MTM'


def test_correction_moves_gradients_but_not_the_forward():
    """There is no longer a setting that yields greeks WITHOUT the correction, so the uncorrected
    gradient is obtained by suppressing the term itself. A correction that moved the forward would
    be a bug; one that left gradients alone would be doing nothing."""
    original = calculation.mta_boundary_correction
    calculation.mta_boundary_correction = lambda *a, **kw: None
    try:
        off = _run(2_000_000.0, gradient=True)['Results']
    finally:
        calculation.mta_boundary_correction = original
    on = _run(2_000_000.0, gradient=True)['Results']
    assert np.array_equal(off['mtm'].values, on['mtm'].values), 'forward exposure moved'

    g_off, g_on = off['grad_cva'], on['grad_cva']
    assert set(g_off) == set(g_on), 'the gradient vector changed shape'
    delta = (g_on['Gradient'] - g_off['Gradient']).abs()
    assert (delta > 0).any(), 'boundary correction left every gradient untouched — it is inert'

    # The correction reaches a factor only through the decision GAP, which is built from the
    # portfolio MTM. The counterparty's own survival curve is not in that gap, so its sensitivity
    # must be untouched to the bit — this is what says the term is landing where it should rather
    # than smearing across the whole vector.
    survival = [i for i in g_off.index if 'SurvivalProb' in str(i[0])]
    assert survival, 'fixture lost its survival factor'
    for row in survival:
        assert g_on['Gradient'][row] == g_off['Gradient'][row], \
            f'{row}: survival sensitivity moved, but it is not in the transfer decision'
