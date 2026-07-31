"""Can the CMC exposure numbers be trusted with the tabulated interval draw?

The scenario engine is where the quantile table actually engages (it is the calc that owns a
t_PreCalc), and that is the one place none of the unit gates reach: they exercise the substep
function directly on synthetic tensors, not a priced exposure profile through the real pricers.

So this reprices a genuine Credit_Monte_Carlo book three ways and compares against the
SEED-TO-SEED noise of the exact walk, which is the only meaningful yardstick for two Monte
Carlo estimates:

    walk    every unmonitored day walked exactly                     (the oracle)
    table   the aggregate drawn from the exact tabulated inverse CDF (what ships)
    hybrid  the aggregate from the table, the terminal variance from the walk

The hybrid isolates the one piece that is still approximate. The table fixes the aggregate
return's marginal exactly; h_end is a regression on the realised aggregate plus a
moment-matched residual, and its standard deviation runs 1-4% off the walk. If table and
hybrid agree, that residual approximation does not reach the exposure profile and the ship
decision is clean; if they diverge, h_end needs the same treatment the aggregate just got.

BLOCKED as of 2026-07-31, and the blocker is the finding. EquityBarrierOption cannot price
under Credit_Monte_Carlo at all: it raises "size of tensor a (1369) must match tensor b (37)"
- 1369 is 37 squared, the report grid against itself - and Deal.calculate swallows that into a
skipped deal, after which report() dies on the shape. It fails IDENTICALLY with SpotModel off,
so it is a pre-existing bug in the barrier deal's scenario path, not an HN or sub-stepping one.

Which means the table has never been exercised end to end on a barrier, because the only calc
that engages it is the one the deal cannot run in. Fix that first; this harness is ready to
answer the question the moment it does.

Run:  CUDA_VISIBLE_DEVICES=0 python gates/hn_cmc_acceptance.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))

import numpy as np
import pandas as pd
import torch

import riskflow
from riskflow import utils
from riskflow.config import Config
from riskflow.instruments import construct_instrument
import hn_reference as hnref

assert os.path.realpath(os.path.dirname(os.path.dirname(riskflow.__file__))) == os.path.realpath(REPO), riskflow.__file__

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
_SP = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=252.0)
HN = {'Omega': float(_SP['omega']), 'Alpha': float(_SP['alpha']), 'Beta': float(_SP['beta']),
      'Gamma_Star': float(_SP['gamma_star']),
      'H0': 1.6 * float(utils.hn_stationary_var(
          _SP['omega'], _SP['alpha'], _SP['beta'], _SP['gamma_star']))}


def build(horizon_days, monitor_every, batch, batches, seed):
    """A monthly-monitored down-and-out barrier on an HN-driven equity, priced through the full
    scenario engine. Monthly monitoring is the case the table is for: n_sub ~ 21 per interval."""
    bdates = [BASE + pd.Timedelta(days=d)
              for d in range(monitor_every, horizon_days + 1, monitor_every)]
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=horizon_days),
        'Units': 100.0, 'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 80.0,
        'Cash_Rebate': 0.0, 'Barrier_Dates': [[d, 80.0] for d in bdates],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    val = {'EquityBarrierOption': {'SpotModel': 'HestonNandi'}}
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
    return c, {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2d 1w(1w) 3m(1m) 2y(3m)',
               'Batch_Size': batch, 'Simulation_Batches': batches, 'Random_Seed': seed,
               'Currency': 'USD', 'MCMC_Simulations': 0, 'Tenor_Offset': 0.0,
               'Deflation_Interest_Rate': 'USD'}


_WALK = utils.hn_unmonitored_substeps
_TABLE = utils.hn_table_substeps


def hybrid(Sj, h, b_step, n_steps, hn_params, shared, num_sims, antithetic):
    """Aggregate from the table, terminal variance from the walk — isolates h_end's residual."""
    St, _ = _TABLE(Sj, h, b_step, n_steps, hn_params, shared, num_sims, antithetic)
    _, hw = _WALK(Sj, h, b_step, n_steps, hn_params, shared, num_sims, antithetic)
    return St, hw


def run(mode, seed, batch=512, batches=2):
    # a measurement harness may swap the internal; shipped code never does
    utils.hn_table_substeps = {'walk': _WALK, 'table': _TABLE, 'hybrid': hybrid}[mode]
    try:
        cfg, params = build(365, 30, batch, batches, seed)
        t0 = time.perf_counter()
        _, out, profile = riskflow.run_cmc(cfg, prec=DTYPE, overrides=params)
        wall = time.perf_counter() - t0
        # the exposure profile IS the CMC deliverable: compare EPE and PFE95, not one mark
        mtm = out['Results']['mtm']
        epe = float(np.maximum(mtm.values, 0.0).mean(axis=1).mean())
        pfe = float(np.percentile(mtm.values, 95, axis=1).mean())
        return (epe, pfe), wall
    finally:
        utils.hn_table_substeps = _TABLE


SEEDS = (1, 2, 3, 4, 5, 6)
print(f'Credit_Monte_Carlo, HN down-and-out barrier, monthly monitoring (n_sub~21)')
print(f'512 x 2 batches x {len(SEEDS)} seeds\n')
res, wall = {}, {}
for mode in ('walk', 'table', 'hybrid'):
    vals, walls = zip(*[run(mode, s) for s in SEEDS])
    res[mode] = np.array(vals)                              # (seeds, 2) = EPE, PFE95
    wall[mode] = float(np.mean(walls))
    print(f'  {mode:7} EPE {res[mode][:, 0].mean():11.5f}  PFE95 {res[mode][:, 1].mean():11.5f}  '
          f'wall {wall[mode]:6.2f}s')

for k, name in ((0, 'EPE'), (1, 'PFE95')):
    w = res['walk'][:, k]
    se = w.std(ddof=1) / np.sqrt(len(SEEDS))
    print(f'\n  {name}: exact-walk seed se {se:.6f}')
    for mode in ('table', 'hybrid'):
        bias = res[mode][:, k].mean() - w.mean()
        print(f'    {mode:7} bias {bias:+.6f} = {abs(bias) / max(se, 1e-12):5.2f} se   '
              f'{bias / max(abs(w.mean()), 1e-12) * 1e4:+8.2f} bp')
    d = res['table'][:, k].mean() - res['hybrid'][:, k].mean()
    print(f'    h_end residual alone (table - hybrid): {d:+.6f} = {abs(d) / max(se, 1e-12):.2f} se')
print(f'\n  speed: table {wall["walk"] / wall["table"]:.2f}x the exact walk')
