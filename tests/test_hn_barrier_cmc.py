"""A barrier deal priced through Credit_Monte_Carlo — the path that had no test at all.

Every barrier gate in this suite priced under BASE VALUATION, which has one deal-time row. That
single fact hid a bug for as long as the deal has existed: four equity deals corrected spot's
shape by testing its COLUMNS and then repeating its ROWS, which is inert at one row and squares
the grid at 37 (1369 vs 37). `Deal.calculate` swallowed the resulting RuntimeError into a
skipped deal, so a barrier in an exposure calculation silently produced nothing.

Kept deliberately small — it exists to prove a barrier still prices across an exposure grid,
which is the one thing base valuation structurally cannot check.
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
import hn_reference as hnref

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
_SP = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=252.0)
HN = {'Omega': float(_SP['omega']), 'Alpha': float(_SP['alpha']), 'Beta': float(_SP['beta']),
      'Gamma_Star': float(_SP['gamma_star']),
      'H0': 1.6 * float(utils.hn_stationary_var(
          _SP['omega'], _SP['alpha'], _SP['beta'], _SP['gamma_star']))}


def _cfg(hn):
    """Monthly-monitored down-and-out barrier. The rate and dividend curves are STATIC while the
    equity is simulated — the exact combination that triggered the shape bug, because a simulated
    spot's B columns against a static curve's 1 is an ordinary broadcast pair, not a defect."""
    bdates = [BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': 100.0,
        'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 80.0, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [[d, 80.0] for d in bdates],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    val = {'EquityBarrierOption': {'SpotModel': 'HestonNandi'}} if hn else {}
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.02]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, 0.01], [5.0, 0.01]])},
        'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.25] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'HestonNandiModelParameters.EQ': dict(HN, Property_Aliases=None),
    }
    c.params['Price Models'] = {}
    c.params['Model Configuration'].append('EquityPrice', (), 'HestonNandiImpliedSpotModel')
    c.params['Valuation Configuration'] = val
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, val)}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _profile(hn, seed=1, batch=64, sims=1024):
    params = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2d 1w(1w) 3m(1m)',
              'Batch_Size': batch, 'Simulation_Batches': 1, 'Random_Seed': seed,
              'Currency': 'USD', 'MCMC_Simulations': sims, 'Tenor_Offset': 0.0,
              'Deflation_Interest_Rate': 'USD'}
    _, out = riskflow.run_cmc(_cfg(hn), prec=DTYPE, overrides=params)
    return out['Results']['mtm']


@pytest.mark.parametrize('hn', [True, False], ids=['heston_nandi', 'gbm'])
def test_barrier_prices_across_the_exposure_grid(hn):
    """The regression gate for the shape bug: one row per report date, one column per path. The
    bug produced len(deal_time)**2 rows and the deal was skipped, which surfaced only as a
    reporting error further downstream. It fired for GBM too, hence both ids."""
    mtm = _profile(hn)
    assert mtm.shape[0] > 1, 'exposure profile collapsed to a single row — deal skipped?'
    assert mtm.shape[1] == 64, f'expected one column per path, got {mtm.shape[1]}'
    assert np.isfinite(mtm.values).all(), 'NaN in the exposure profile'
    assert (mtm.values > 0).any(), 'a bought down-and-out barrier should carry positive exposure'
    # monotone decay is not guaranteed, but the profile must not be constant across time
    assert mtm.values.std(axis=1).min() > 0.0, 'no dispersion across paths at some date'
