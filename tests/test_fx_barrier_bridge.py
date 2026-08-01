"""An FX barrier is monitored on a CROSS, so the bridge cannot always be applied.

The equity deals name their underlying outright, because an equity barrier watches one simulated
factor. `utils.calc_fx_cross` divides the underlying leg by the quote leg, and the log-variance of
that ratio is `v_u + v_c - 2*rho*sqrt(v_u*v_c)` - not any single factor's. `t_Bridge_Variance_Rate`
is keyed per factor and `pv_barrier_option` asks for exactly one, so the honest answer is a rate
only where the quote leg contributes nothing.

That is why `get_fx_barrier_underlying` can return None, and why returning None matters: naming the
underlying leg unconditionally would hand the bridge an understated variance, which understates
crossings while LOOKING like the fix had been applied. Not bridging is a known fallback; bridging
with the wrong variance is a silent wrong answer.

Both halves are gated here - the quote-leg-static case must bridge, the simulated-quote-leg case
must decline to.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
VOL = 0.12
SPOT = 1.1
# The barrier has to BITE. At 12% vol a barrier 10% out is almost never touched, and a fixture
# where survival is ~1 makes the bridge and the endpoint indicator agree - it passed a deliberate
# mutant that disabled the bridge entirely. 4.5% out at 12% vol is the same standardised distance
# as the equity gate's 10% out at 25% vol, where the endpoint error is 11-26%.
BARRIER = 1.05


def _deal(**kw):
    field = {
        'Object': 'FXBarrierOption', 'Reference': 'FXB1', 'Currency': 'USD',
        'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
        'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 1.1, 'Underlying_Amount': 1.0,
        'Expiry_Date': BASE + pd.Timedelta(days=365), 'Barrier_Type': 'Down_And_Out',
        'Barrier_Price': BARRIER, 'Cash_Rebate': 0.0,
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
    }
    field.update(kw)
    return field


def _cfg(deal=None, third_currency=False):
    """Zero rates in both legs plus zero GBM drift makes the EURUSD spot a martingale under the
    simulation measure, so the barrier value is one too. With `third_currency` the deal is quoted
    in a SIMULATED currency, which is the genuine cross the bridge must decline."""
    quote = 'GBP' if third_currency else 'USD'
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Priority': 1, 'Spot': SPOT},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'DiscountRate.EUR': {'Interest_Rate': 'EUR'},
        'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, VOL] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.1, 3.0)])},
    }
    c.params['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': VOL, 'Drift': 0.0}}
    if third_currency:
        c.params['Price Factors']['FxRate.GBP'] = {
            'Domestic_Currency': 'USD', 'Interest_Rate': 'GBP', 'Priority': 1, 'Spot': 1.3}
        c.params['Price Factors']['InterestRate.GBP'] = {
            'Currency': 'GBP', 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])}
        c.params['Price Factors']['DiscountRate.GBP'] = {'Interest_Rate': 'GBP'}
        c.params['Price Models']['GBMAssetPriceModel.GBP'] = {'Vol': 0.10, 'Drift': 0.0}
        c.params['Price Factors']['FXVol.EUR.GBP'] = {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, VOL] for m in (0.8, 1.0, 1.2) for t in (0.1, 3.0)])}
    c.params['Model Configuration'].append('FxRate', (), 'GBMAssetPriceModel')
    c.params['Valuation Configuration'] = {}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(
                   deal or _deal(Currency=quote, Discount_Rate=quote), {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _profile(grid, deal=None, third_currency=False, batch=8192):
    _, out = riskflow.run_cmc(_cfg(deal, third_currency), prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': grid, 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Deflation_Interest_Rate': 'USD'})
    return out['Results']['mtm']


def test_quote_leg_static_publishes_the_underlying():
    """USD-quoted under a USD base: the quote leg is not simulated, so the cross carries the EUR
    leg's own variance and the bridge is exact."""
    from riskflow.instruments import get_fx_barrier_underlying
    field = {'Currency': ('USD',), 'Underlying_Currency': ('EUR',)}
    assert get_fx_barrier_underlying(field, {}) == utils.Factor('FxRate', ('EUR',))


def test_a_simulated_quote_leg_declines_the_bridge():
    """A genuine cross. The variance the bridge needs is not any single factor's, so the honest
    answer is None and the pricer falls back to observing endpoints."""
    from riskflow.instruments import get_fx_barrier_underlying
    field = {'Currency': ('GBP',), 'Underlying_Currency': ('EUR',)}
    stochastic = {utils.Factor('FxRate', ('GBP',)): 0, utils.Factor('FxRate', ('EUR',)): 1}
    assert get_fx_barrier_underlying(field, stochastic) is None, (
        'bridging a cross off one leg understates the variance - worse than not bridging')


def test_a_basis_chain_declines_the_bridge():
    """A chained name is several factors; the head factor's rate is not the whole variance."""
    from riskflow.instruments import get_fx_barrier_underlying
    field = {'Currency': ('USD',), 'Underlying_Currency': ('EUR', 'BASIS')}
    assert get_fx_barrier_underlying(field, {}) is None


@pytest.mark.parametrize('grid,label', [('0d 3m(3m)', 'quarterly'), ('0d 1m(1m)', 'monthly')])
def test_fx_barrier_profile_is_grid_independent(grid, label):
    """The value gate. Zero rates make it a martingale, so every date on every grid must report the
    inception value; the endpoint indicator overstates survival by more the coarser the grid, which
    is exactly what a grid-dependent profile detects."""
    mtm = _profile(grid)
    v = mtm.values.mean(axis=1)
    assert v[0] > 0.0, 'a bought down-and-out call should be worth something at inception'
    drift = np.abs(v - v[0]) / v[0]
    assert drift.max() < 0.04, (
        f'{label}: profile drifts {drift.max():.1%} from inception {v[0]:.6f} '
        f'at row {drift.argmax()} of {len(drift)}\n{np.round(v, 6)}')


def test_the_cross_case_still_prices():
    """Declining the bridge must be a fallback, not a failure - the endpoint path is what the
    engine did everywhere before, and a genuine cross has to keep pricing."""
    mtm = _profile('0d 3m(3m)', third_currency=True, deal=_deal(
        Currency='GBP', Payoff_Currency='GBP', Discount_Rate='GBP', FX_Volatility='EUR.GBP',
        Strike_Price=0.85, Barrier_Price=0.76))
    assert np.isfinite(mtm.values).all() and (mtm.values > 0).any()
