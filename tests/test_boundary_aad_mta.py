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


def _params(boundary_aad, seed=1, batch=256, batches=1):
    p = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1m(3m) 2y(3m)',
         'Batch_Size': batch, 'Simulation_Batches': batches, 'Random_Seed': seed,
         'Currency': 'USD', 'MCMC_Simulations': 0, 'Tenor_Offset': 0.0,
         'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
         'Credit_Valuation_Adjustment': {
             'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
             'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}}
    if boundary_aad:
        p['Boundary_AAD'] = 'Yes'
    return p


def _run(min_transfer, boundary_aad, seed=1):
    _, out = riskflow.run_cmc(_cfg(min_transfer), prec=DTYPE,
                              overrides=_params(boundary_aad, seed=seed))
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
    """The fixture is only a test of anything if the MTA actually suppresses transfers. A zero
    MTA transfers at every call; a large one does not, and the CVAs must differ."""
    loose = _run(0.0, boundary_aad=False)['Results']['grad_cva']
    tight = _run(2_000_000.0, boundary_aad=False)['Results']['grad_cva']
    assert loose is not None and tight is not None
    same = all(np.allclose(loose[k], tight[k]) for k in loose if k in tight)
    assert not same, 'MTA is not binding — the boundary term would be trivially zero'


@pytest.mark.parametrize('min_transfer', [0.0, 2_000_000.0])
def test_forward_cva_is_bit_identical_with_the_switch_on(min_transfer):
    """THE gate for the whole build. The correction's forward value is exactly zero by
    construction, so switching boundary AAD on may not move any reported number at all — not
    within a tolerance, bitwise."""
    off = _run(min_transfer, boundary_aad=False)['Results']
    on = _run(min_transfer, boundary_aad=True)['Results']
    assert set(off) == set(on)
    a, b = off['mtm'].values, on['mtm'].values
    assert np.array_equal(a, b), f'exposure profile moved: max |d| {np.abs(a - b).max():.3e}'


def test_events_are_registered_only_when_the_switch_is_on():
    """Keeps the bit-identity gates from passing vacuously: if the netting set were skipped, or
    the switch never reached the scan, 'unchanged forward' would be true and meaningless."""
    import riskflow.instruments as instruments
    original = instruments.scan_collateral_balance
    tally = {}

    def counted(*args, **kwargs):
        path, gaps = original(*args, **kwargs)
        tally['gaps'] = tally.get('gaps', 0) + len(gaps)
        return path, gaps

    instruments.scan_collateral_balance = counted
    try:
        for boundary_aad, expected in ((False, False), (True, True)):
            tally.clear()
            out = _run(2_000_000.0, boundary_aad=boundary_aad)
            assert out['Results']['mtm'].shape[0] > 1, 'netting set did not price'
            assert bool(tally.get('gaps')) is expected, \
                f'boundary_aad={boundary_aad}: collected {tally.get("gaps", 0)} gaps'
    finally:
        instruments.scan_collateral_balance = original
