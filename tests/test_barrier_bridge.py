"""A barrier's survival between grid dates is a probability, not an endpoint check.

`pv_barrier_option` prices the remaining life with Reiner-Rubinstein, which assumes CONTINUOUS
monitoring. The historical path state asked only whether the spot sat beyond the barrier ON a grid
date, so every path that crossed and came back counted as still alive - a state inconsistent with
the formula being applied to it, and one that got worse the coarser the grid.

The gate needs no external reference. Set r = q = 0 and the simulation drift to match, and the
option value is a MARTINGALE: E[MTM_t] must equal the t=0 value at every t, on every grid. The t=0
row is the pure closed form because no history has accumulated yet, so the test compares the
pricer against itself and still pins the thing that was wrong. Endpoint-only survival fails it by
+11% at 3m and +26% at 9m on the quarterly grid, and the error moves with the grid, which is what
`test_bridge_is_grid_independent` reads directly.
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
from riskflow.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
VOL = 0.25
SPOT = 100.0


def _cfg():
    """Down-and-out call, continuously monitored, in a zero-rate zero-dividend world so that the
    value is a martingale under the simulation measure. The equity is driven by GBM, whose
    lognormal interval law is what publishes a bridge variance rate at all."""
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': 1.0,
        'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 90.0, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [], 'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
    }
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, VOL] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
    }
    # Drift 0 with r = q = 0 makes the SIMULATED spot a martingale, which is what lets the option
    # value be one too - the pricing measure and the simulation measure have to be the same one.
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': VOL, 'Drift': 0.0}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


ONE_TOUCH = {
    'Object': 'EquityOneTouchOption', 'Reference': 'OT1', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
    'Buy_Sell': 'Buy', 'Cash_Payoff': 100.0, 'Payoff_Type': 'Cash', 'Barrier_Price': 90.0,
    'Barrier_Type_One': 'Down', 'Expiry_Date': BASE + pd.Timedelta(days=365),
    'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
}


def _profile(grid, seed=1, batch=8192, deal=None):
    params = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': grid, 'Batch_Size': batch,
              'Simulation_Batches': 1, 'Random_Seed': seed, 'Currency': 'USD',
              'Tenor_Offset': 0.0, 'Deflation_Interest_Rate': 'USD'}
    c = _cfg()
    if deal is not None:
        c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides=params)
    return out['Results']['mtm']


def _cva(spot, deal, gradient):
    """CVA, and its AAD gradient when asked for. A counterparty is what gives the barrier a
    sensitivity worth measuring: the exposure profile is where the touch state accumulates, which
    base valuation - one deal-time row, no interval, no history - structurally cannot show."""
    c = _cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': 4096,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}})
    if not gradient:
        return float(out['Results']['cva'])
    g = out['Results']['grad_cva']['Gradient']
    return float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])


def _analytic_touch_probability():
    """P(the minimum of the GBM breaches the barrier before expiry), by reflection. Independent of
    riskflow - the whole point is that the pricer is checked against something it did not produce."""
    import math
    mu, sig, b = -0.5 * VOL ** 2, VOL, math.log(90.0 / SPOT)
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    return phi((b - mu) / sig) + math.exp(2.0 * mu * b / sig ** 2) * phi((b + mu) / sig)


def test_variance_rate_reproduces_the_processes_own_variance():
    """The rate is only exact if it is the SIMULATION variance. A process discretises the scenario
    grid into per-step vols; a rate against elapsed time must sum back to the same total, or the
    bridge is being handed a vol that means something else - the pricing implied vol for the
    option's remaining life is exactly such a quantity, carrying the same units."""
    from riskflow.stochasticprocess import GBMAssetPriceModel
    import types

    # UNEVEN scenario dates, and they must be the SCENARIO set - a single date leaves dt all zero,
    # which makes both sides of the comparison zero and the assertion true for any rate at all.
    dates = {BASE + pd.Timedelta(days=d) for d in (0, 30, 90, 365)}
    grid = utils.TimeGrid(dates, dates, dates)
    grid.set_base_date(BASE)
    p = GBMAssetPriceModel(factor=types.SimpleNamespace(param={}), param={'Vol': VOL, 'Drift': 0.0})
    p.precalculate(BASE, grid, torch.tensor([SPOT], dtype=DTYPE), None, 0)

    stepwise = float((p.vol * p.vol).sum())
    elapsed = grid.time_grid_years[-1]
    assert stepwise > 0.0 and elapsed > 0.0, 'degenerate grid - the comparison below is vacuous'
    assert p.bridge_variance_rate * elapsed == pytest.approx(stepwise, rel=1e-12), (
        'rate x elapsed must equal the variance the process actually simulates')


@pytest.mark.parametrize('grid,label', [('0d 3m(3m)', 'quarterly'),
                                        ('0d 1m(1m)', 'monthly'),
                                        ('0d 1w(1w)', 'weekly')])
def test_bridge_is_grid_independent(grid, label):
    """With r = 0 the value is a martingale, so every date on every grid must report the t=0
    price. The endpoint-only state fails this by +11% at 3m and +26% at 9m on the quarterly grid,
    and by a DIFFERENT amount on each grid - a barrier's exposure profile should not depend on how
    often the engine happens to look at it.

    4% is measured headroom, not a round number: the bridge's worst reading across these grids and
    several seeds is ~2.1% at 8192 paths, and the defect it has to catch is an order of magnitude
    outside that."""
    mtm = _profile(grid)
    t0 = mtm.values[0].mean()
    assert t0 > 0.0, 'a bought down-and-out call should be worth something at inception'
    drift = np.abs(mtm.values.mean(axis=1) - t0) / t0
    assert drift.max() < 0.04, (
        f'{label}: exposure profile drifts {drift.max():.1%} from the t=0 value {t0:.4f} '
        f'at row {drift.argmax()} of {len(drift)} - survival is not being carried as a probability')


def test_one_touch_paid_at_expiry_holds_its_value_until_then():
    """A one-touch paying at EXPIRY owes the nominal on every path that has touched, so between
    the touch and expiry such a path holds a CERTAIN claim worth its discounted value. That value
    was carried as zero: a touched deal reported nothing for the rest of its life and then jumped
    to the nominal on the last date. With r=0 the correct value is a martingale, and equals the
    nominal times the touch probability at t=0."""
    mtm = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Option_Payment_Timing='Expiry'))
    v = mtm.values.mean(axis=1)
    expected = 100.0 * _analytic_touch_probability()
    assert v[0] == pytest.approx(expected, rel=2e-3), (
        f'inception value {v[0]:.3f} should be the analytic {expected:.3f}')
    assert np.abs(v - v[0]).max() / v[0] < 0.03, (
        f'value paid at expiry is not being held: profile {np.round(v, 2)}')


def test_one_touch_paid_on_touch_settles_and_leaves():
    """The counterpart, and the reason the fix above is confined to Expiry timing: paid ON touch,
    the cash settles and the path stops carrying it, so this profile SHOULD decay. Both timings
    must still agree at inception, since with r=0 there is nothing to discount between them."""
    on_touch = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Option_Payment_Timing='Touch'))
    at_expiry = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Option_Payment_Timing='Expiry'))
    v = on_touch.values.mean(axis=1)
    assert v[0] == pytest.approx(at_expiry.values[0].mean(), rel=2e-3), (
        'with r=0 the two payment timings are worth the same at inception')
    assert v[-1] < 0.25 * v[0], f'paid-on-touch value should run off, got {np.round(v, 2)}'


BARRIER_DEAL = {
    'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
    'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 100.0,
    'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': 1.0, 'Barrier_Type': 'Down_And_Out',
    'Barrier_Price': 90.0, 'Cash_Rebate': 0.0, 'Barrier_Dates': [],
    'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
}


@pytest.mark.parametrize('deal,label', [
    (BARRIER_DEAL, 'barrier'),
    (dict(ONE_TOUCH, Option_Payment_Timing='Expiry'), 'one_touch')])
def test_aad_delta_matches_bump_and_reprice(deal, label):
    """The gradient has to be the derivative of the value actually reported. Under common random
    numbers - same seed, so the same normals, the draws depending on seed and factor ordering
    rather than on the spot - a central difference estimates the same derivative without touching
    the tape, so agreement is evidence and not a restatement.

    An INDICATOR has zero derivative almost everywhere, so the knock-out channel contributed
    nothing and AAD reported the wrong number while looking perfectly well-behaved: measured 9-19%
    off for the barrier and 31-44% off for the one-touch, with no plateau across the bump ladder
    because bumping flips discrete touch states. Carrying survival as a probability closes it to
    0.00%. 2% is headroom over that, and an order of magnitude inside the defect."""
    aad = _cva(SPOT, deal, gradient=True)
    assert abs(aad) > 1e-6, 'a barrier with a live knock-out should have a spot delta'
    for rel in (5e-4, 1e-3):
        h = SPOT * rel
        crn = (_cva(SPOT + h, deal, False) - _cva(SPOT - h, deal, False)) / (2.0 * h)
        assert crn == pytest.approx(aad, rel=0.02), (
            f'{label}: AAD {aad:+.5f} vs bump-and-reprice {crn:+.5f} at {rel:.0e} - '
            f'a channel through which spot moves the value is not being differentiated')
