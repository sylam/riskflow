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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import riskflow
from riskflow import utils
from riskflow.config import Config
from riskflow.instruments import construct_instrument
from crn_ladder import ladder

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


def _cva(spot, deal, gradient, batch=4096, mcmc=None):
    """CVA, and its AAD gradient when asked for. A counterparty is what gives the barrier a
    sensitivity worth measuring: the exposure profile is where the touch state accumulates, which
    base valuation - one deal-time row, no interval, no history - structurally cannot show."""
    c = _cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        **({'MCMC_Simulations': mcmc} if mcmc else {}),
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
    off for the barrier and 31-44% off for the one-touch, and - the discriminating signal - the
    ladder SCATTERED instead of converging, because shrinking the bump changes how many paths sit
    on the far side of the jump rather than refining a limit. Carrying survival as a probability
    gives 0.00% disagreement at 0.00% flatness."""
    aad = _cva(SPOT, deal, gradient=True)
    assert abs(aad) > 1e-6, 'a barrier with a live knock-out should have a spot delta'
    r = ladder(price=lambda s: _cva(s, deal, False), aad=aad, base=SPOT,
               rungs=(2e-4, 5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), (
        f'{label}: a channel through which spot moves the value is not being differentiated\n{r}')


MONTHLY_BARRIER = [[BASE + pd.Timedelta(days=d), 90.0] for d in range(30, 366, 30)]


def test_discrete_barrier_is_observed_only_on_its_own_dates():
    """A DISCRETELY monitored barrier is observed on the dates its terms name, and nowhere else.

    pv_discrete_barrier_option latched the crossing with a cumsum over every MTM row of each block,
    so on this fixture it monitored 37 reporting rows against 12 barrier dates - knocking scenarios
    out on dates the deal never observes, monitoring expiry although BarrierDates flags it -1, and
    (because the guard tested samples passed BEFORE the block) missing the first barrier date
    entirely while lagging one block thereafter.

    With r = q = 0 the value is a martingale, so the profile mean must equal the inception value.
    Inception is unaffected by the defect - no scenario has hit yet, and the OSS handles all twelve
    future observations analytically - which is what makes it a valid reference for the rest of the
    profile. Against an independent 2m-path simulation observing exactly those twelve dates
    (V_0 = 8.4844 +- 0.0118) inception prices to -0.02%.

    Measured: the defect drove the profile to -2.79% and its worst row to -4.88%; the corrected
    form sits at -0.02%. 1% is headroom over a measured 0.18% worst case across three seeds."""
    c = _cfg()
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(
        dict(BARRIER_DEAL, Barrier_Dates=MONTHLY_BARRIER), {})}]
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2d 1w(1w) 3m(1m)',
        'Batch_Size': 4096, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD',
        'Tenor_Offset': 0.0, 'MCMC_Simulations': 256, 'Deflation_Interest_Rate': 'USD'})
    v = out['Results']['mtm'].values.mean(axis=1)
    assert len(v) > 12, 'the grid must be FINER than the barrier schedule or nothing is being tested'
    assert abs(v.mean() - v[0]) / v[0] < 0.01, (
        f'profile mean {v.mean():.4f} vs inception {v[0]:.4f} '
        f'({(v.mean() - v[0]) / v[0]:+.2%}) - the barrier is being observed on the wrong dates\n'
        f'{np.round(v, 3)}')


def _rebate_run(rebate, units):
    c = _cfg()
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(dict(
        BARRIER_DEAL, Barrier_Price=95.0, Cash_Rebate=rebate, Units=units,
        Barrier_Dates=[[BASE + pd.Timedelta(days=d), 95.0] for d in range(30, 366, 30)]), {})}]
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1m(1m)', 'Batch_Size': 2048,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': 256, 'Generate_Cashflows': 'Yes', 'Deflation_Interest_Rate': 'USD'})
    cf = out['Results']['cashflows']
    return (out['Results']['mtm'].values.mean(axis=1)[0],
            sum(float(np.nansum(v.values)) for v in cf.values()))


def test_discrete_barrier_rebate_is_paid_and_is_absolute_cash():
    """Two defects in one field. The knock-out rebate was PRICED - sim_spot_oss accrues it, and it
    is in the mtm of the row that knocks out - but never settled: hit_value is zero from the next
    row on and the single cash_settle fires on the last row only, so the total settled cash was
    bit-identical to the same deal with no rebate at all.

    And it was scaled wrongly. pv_barrier_option reads Cash_Rebate as ABSOLUTE cash - it hands the
    closed form cash_rebate/nominal and multiplies back - while everything sim_spot_oss returns is
    scaled by nominal, so the same field on the same deal class meant Units times more cash under
    discrete monitoring than under continuous. Doubling Units must not double a cash rebate."""
    p0, c0 = _rebate_run(0.0, 1.0)
    p5, c5 = _rebate_run(5.0, 1.0)
    assert c5 - c0 > 0.0, 'the rebate is priced into the mtm but never settled'

    p0x, _ = _rebate_run(0.0, 2.0)
    p5x, _ = _rebate_run(5.0, 2.0)
    assert (p5x - p0x) == pytest.approx(p5 - p0, rel=1e-9), (
        f'a cash rebate must not scale with Units: adds {p5 - p0:.4f} at Units=1 but '
        f'{p5x - p0x:.4f} at Units=2')


def _digital(H, btype='Down_And_Out'):
    return {'Object': 'EquityBarrierBinaryOption', 'Reference': 'DIG1', 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365),
            'Cash_Payoff': 100.0, 'Barrier_Type': btype, 'Barrier_Price': H,
            'Settlement_Date': BASE + pd.Timedelta(days=365),
            'Barrier_Dates': [[BASE + pd.Timedelta(days=d), H] for d in range(30, 366, 30)]}


def test_digital_terminal_step_is_integrated_not_sampled():
    """A digital's payoff was an indicator on the DRAWN terminal spot, whose derivative is zero
    almost everywhere - so the density term that is most of a digital's delta and vega never
    reached the tape at all.

    The barrier is put out of reach so the outer already-hit latch never fires and this isolates
    the terminal step. Before the fix AAD reported EXACTLY zero here, and the equity, vol and
    dividend factors were absent from the greeks report rather than showing zero rows - a silent
    total loss of sensitivity. Now: 0.00% against bump-and-reprice at 0.01% flatness.

    NOTE the same deal WITH a live barrier still disagrees (33.7% at 9.96% flatness). That residual
    is the outer barrier_hit latch in pv_discrete_barrier_option, a genuine jump in the value
    function needing the boundary-flux machinery, not this terminal step - which is exactly why
    this gate uses an unreachable barrier."""
    deal = _digital(1e-6)
    # the OSS forks an inner Monte Carlo per outer path, so the outer batch stays small here
    kw = dict(batch=1024, mcmc=256)
    aad = _cva(SPOT, deal, gradient=True, **kw)
    assert abs(aad) > 1e-6, 'a digital must have a spot delta'
    r = ladder(price=lambda s: _cva(s, deal, False, **kw), aad=aad, base=SPOT,
               rungs=(5e-4, 1e-3, 2e-3, 5e-3))
    assert r.agrees(tol=0.02), f'digital terminal step is not being integrated\n{r}'


def test_digital_reports_its_equity_and_vol_factors():
    """The failure mode was not a wrong number but a MISSING one: with a zero gradient the factor's
    .grad is None and pricing.report_grad drops it, so the risk report simply had no equity row."""
    c = _cfg()
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(_digital(1e-6), {})}]
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': 256,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': 128, 'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}})
    factors = {str(i[0]).split('.')[0] for i in out['Results']['grad_cva']['Gradient'].index}
    for needed in ('EquityPrice', 'EquityPriceVol'):
        assert needed in factors, f'{needed} missing from the greeks report; got {sorted(factors)}'


def _settled(deal_overrides, batch=512):
    c = _cfg()
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(
        dict(BARRIER_DEAL, **deal_overrides), {})}]
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1m(1m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': 128, 'Generate_Cashflows': 'Yes', 'Deflation_Interest_Rate': 'USD'})
    cf = out['Results']['cashflows']
    return (out['Results']['mtm'].values.mean(axis=1)[0],
            sum(float(np.nansum(v.values)) for v in cf.values()))


def test_a_sold_knock_out_pays_its_rebate_rather_than_receiving_it():
    """`nominal` in the discrete pricer ALREADY carries Buy_Sell, unlike pv_barrier_option where
    buy_or_sell is a separate factor. Dividing the rebate by it therefore cancelled the direction,
    and every rebate leg came back as +cash_rebate whichever way the deal was done - a seller who
    must PAY on knock-out booked it as a receipt, in the reported price and in the settled cash.

    Buy and Sell must be exact mirror images. The original rebate gate only ever ran Buy, which is
    why this survived it."""
    kw = dict(Barrier_Price=95.0,
              Barrier_Dates=[[BASE + pd.Timedelta(days=d), 95.0] for d in range(30, 361, 30)])
    buy = np.subtract(_settled(dict(kw, Buy_Sell='Buy', Cash_Rebate=5.0)),
                      _settled(dict(kw, Buy_Sell='Buy', Cash_Rebate=0.0)))
    sell = np.subtract(_settled(dict(kw, Buy_Sell='Sell', Cash_Rebate=5.0)),
                       _settled(dict(kw, Buy_Sell='Sell', Cash_Rebate=0.0)))
    assert buy[0] > 0.0 and buy[1] > 0.0, 'a bought knock-out receives its rebate'
    assert sell[0] == pytest.approx(-buy[0], rel=1e-9), (
        f'rebate does not flip with direction: buy {buy[0]:+.4f} vs sell {sell[0]:+.4f}')
    assert sell[1] == pytest.approx(-buy[1], rel=1e-9), (
        f'settled rebate cash does not flip: buy {buy[1]:+.2f} vs sell {sell[1]:+.2f}')


def test_a_barrier_date_on_expiry_settles_its_rebate_once():
    """The per-observation settle fires on every barrier date, and the single settle after the loop
    pays the whole terminal row - which already contains that rebate, because sim_spot_oss accrued
    it. A deal whose last barrier date IS expiry therefore paid twice, and instruments.py unions
    Expiry_Date into the observation dates, so that is the common case rather than a corner.

    pv_barrier_option guards the identical double count with `expiry[index] > 0.0`. Strike is put
    out of reach here so the rebate is the only cash in the run."""
    expiry = BASE + pd.Timedelta(days=365)
    at_expiry = _settled({'Barrier_Price': 95.0, 'Cash_Rebate': 5.0, 'Strike_Price': 1e6,
                          'Barrier_Dates': [[expiry, 95.0]]})[1]
    earlier = _settled({'Barrier_Price': 95.0, 'Cash_Rebate': 5.0, 'Strike_Price': 1e6,
                        'Barrier_Dates': [[BASE + pd.Timedelta(days=330), 95.0]]})[1]
    assert at_expiry < 1.5 * earlier, (
        f'rebate settled twice: {at_expiry:.2f} against {earlier:.2f} for a barrier date 35 days '
        f'earlier - a single count differs only by the extra knock-out probability')


@pytest.mark.parametrize('freq_days,label', [(0, 'continuous'), (30, 'monthly'), (7, 'weekly')])
def test_the_bridge_honours_the_monitoring_frequency(freq_days, label):
    """A discretely monitored barrier is priced by a CONTINUOUS closed form against a barrier
    shifted away from the live region (Broadie-Glasserman-Kou). The bridge was handed the RAW
    barrier while the formula three lines later priced the shifted one, so the path state monitored
    continuously a barrier the product observes monthly, and the two disagreed about the same deal.

    With r = q = 0 the value is a martingale at ANY monitoring frequency. Measured before the fix:
    monthly monitoring decayed -11.58% over the profile; continuous was unaffected, which is why
    the original gate - written at the default 0d - could not see it.

        continuous  +0.82%      monthly  -0.07%      weekly  +0.46%

    Inception rises with coarser monitoring (8.485 monthly against 7.176 continuous) because fewer
    observations means fewer chances to knock out - a second reason a frequency-blind gate is weak."""
    deal = dict(BARRIER_DEAL, Barrier_Monitoring_Frequency=pd.DateOffset(days=freq_days))
    v = _profile('0d 1m(1m)', deal=deal).values.mean(axis=1)
    drift = np.abs(v - v[0]).max() / v[0]
    assert drift < 0.05, (
        f'{label}: profile drifts {drift:.1%} from inception {v[0]:.4f} - the bridge and the '
        f'closed form are testing different barriers\n{np.round(v, 3)}')
