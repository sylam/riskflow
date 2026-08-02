"""A physically-settled swaption's exercise decision, and whether its derivative reaches the tape.

`SwaptionDeal.post_process` freezes exercise on the first post-expiry row:

    Ut_mask = Ut_swap * (Ut_swap[0] >= 0)

a bool taken on SIMULATED state and broadcast over every later row. The value is right - physical
settlement really is path dependent, the holder either owns the swap for the rest of its life or
owns nothing - so the indicator must NOT be smoothed. What ordinary AAD drops is the flux: as a
factor moves, scenarios cross the exercise boundary, and an indicator has zero derivative almost
everywhere.

This matters more than the barrier sites it follows: Settlement_Style DEFAULTS to 'Physical'
(fields.py), so the frozen branch is what a book gets by omission, and NO test in this suite
exercised SwaptionDeal at all before this file.

Same two kinds of test as test_boundary_pricer_events.py. SAFETY: asking for sensitivities must not
move a reported number, bit-for-bit. ACCEPTANCE: AAD against a common-random-numbers bump ladder,
reporting agreement and flatness separately - a ladder that scatters with the bump size is
differencing across the jump rather than converging on a derivative.

Both netting shapes are measured, and the collateralised one is the larger defect rather than the
harder one - unlike the barrier's, whose collateralised ladder is still a strict xfail in
test_boundary_pricer_events.
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
CURVE = 0.03            # flat continuously-compounded zero curve
SWAP_RATE = 3.05        # % - near the money, so the exercise boundary is populated
VOL = 0.20
PRINCIPAL = 10000000.0
EXPIRY = BASE + pd.DateOffset(years=1)
MATURITY = BASE + pd.DateOffset(years=6)
GRID = '0d 6y(6m)'


def _legs(rate):
    """Semi-annual fixed and floating cashflow items for the underlying swap.

    Both legs share payment and accrual dates, which makes K(t) the constant fixed rate and the
    swaption's effective strike exactly `rate` - one less moving part between the fixture and the
    thing being measured."""
    dates = [EXPIRY + pd.DateOffset(months=6 * i)
             for i in range(1 + round((MATURITY - EXPIRY).days / 182.625))]
    fixed, float_ = [], []
    for start, end in zip(dates[:-1], dates[1:]):
        accrual = (end - start).days / 365.0
        fixed.append({'Payment_Date': end, 'Accrual_Start_Date': start, 'Accrual_End_Date': end,
                      'Accrual_Year_Fraction': accrual, 'Notional': PRINCIPAL,
                      'Rate': utils.Percent(rate), 'Fixed_Amount': 0.0})
        float_.append({'Payment_Date': end, 'Accrual_Start_Date': start, 'Accrual_End_Date': end,
                       'Accrual_Year_Fraction': accrual, 'Notional': PRINCIPAL,
                       'Fixed_Amount': 0.0, 'Margin': utils.Basis(0.0),
                       'Resets': [[start, start, end, accrual, pd.DateOffset(months=6),
                                   'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]]})
    return fixed, float_


def _swaption(rate=SWAP_RATE):
    """The parent swaption plus the two children post_process actually prices.

    post_process reads `child_map['CFFixedInterestListDeal']` and
    `child_map['CFFloatingInterestListDeal']` - the parent's own FixedCashflows/FloatCashflows are
    built by calc_dependencies and then never read - so the children are not decoration."""
    fixed, float_ = _legs(rate)
    parent = {
        'Object': 'SwaptionDeal', 'Reference': 'SWPT1', 'Currency': 'EUR',
        'Discount_Rate': 'EUR', 'Forecast_Rate': 'EUR', 'Forecast_Rate_Volatility': 'EUR',
        'Buy_Sell': 'Buy', 'Payer_Receiver': 'Payer', 'Settlement_Style': 'Physical',
        'Option_Expiry_Date': EXPIRY, 'Settlement_Date': EXPIRY,
        'Swap_Effective_Date': EXPIRY, 'Swap_Maturity_Date': MATURITY,
        'Swap_Rate': rate, 'Principal': PRINCIPAL,
        'Pay_Frequency': pd.DateOffset(months=6), 'Receive_Frequency': pd.DateOffset(months=6),
        'Index_Tenor': pd.DateOffset(months=6), 'Index_Day_Count': 'ACT_365',
        'Pay_Day_Count': 'ACT_365', 'Receive_Day_Count': 'ACT_365', 'Floating_Margin': 0.0,
        'Pay_Amortisation': None, 'Receive_Amortisation': None,
    }
    children = [
        {'Instrument': construct_instrument({
            'Object': 'CFFixedInterestListDeal', 'Reference': 'SWPT1_FIX', 'Currency': 'EUR',
            'Discount_Rate': 'EUR', 'Buy_Sell': 'Buy',
            'Cashflows': {'Compounding': 'No', 'Items': fixed}}, {})},
        {'Instrument': construct_instrument({
            'Object': 'CFFloatingInterestListDeal', 'Reference': 'SWPT1_FLT', 'Currency': 'EUR',
            'Discount_Rate': 'EUR', 'Forecast_Rate': 'EUR', 'Buy_Sell': 'Buy',
            'Cashflows': {'Compounding_Method': 'None', 'Averaging_Method': 'None',
                          'Properties': [], 'Items': float_}}, {})}]
    return {'Instrument': construct_instrument(parent, {}), 'Children': children}


def _cfg(curve=CURVE, rate=SWAP_RATE, collateralised=False):
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    # The deal is in EUR because find_models refuses to simulate the BASE currency's curve: with
    # everything in USD the interest rate is static, nothing crosses the exercise boundary and the
    # fixture would measure a frozen indicator that never moves.
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        # Spot is deliberately NOT 1: the branch values are scored against a netting MTM in
        # REPORTING currency, so an fx factor left off them is a real error that a same-currency
        # fixture cannot see (test_the_registered_branches_reproduce_the_reported_value pins it)
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Priority': 1, 'Spot': 1.1},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.03], [10.0, 0.03]])},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             # knots start at 0.25, not 0: HullWhite1Factor divides the simulated
                             # curve by its own tenor, so a zero-tenor knot is 0/0 and NaNs the
                             # whole netting set
                             'Curve': utils.Curve([], [[t, curve] for t in (0.25, 1.0, 3.0, 5.0)])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'DiscountRate.EUR': {'Interest_Rate': 'EUR'},
        'InterestYieldVol.EUR': {
            'Property_Aliases': None,
            'Surface': utils.Curve([], [[m, e, t, VOL] for m in (-0.01, 0.0, 0.01)
                                        for e in (0.5, 1.0, 2.0) for t in (1.0, 2.0, 5.0)])},
        'SurvivalProb.CPTY': {'Recovery_Rate': 0.4,
                              'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])},
    }
    c.params['Price Models'] = {
        'HullWhite1FactorInterestRateModel.EUR': {
            'Lambda': 0.0, 'Alpha': 0.05, 'Quanto_FX_Correlation': 0.0,
            'Quanto_FX_Volatility': None, 'Sigma': utils.Curve([], [[0.0, 0.008]])}}
    c.params['Model Configuration'].append('InterestRate', (), 'HullWhite1FactorInterestRateModel')
    node = _swaption(rate)
    if collateralised:
        # A collateralised set is a DIFFERENT route to the same correction: the deal's delta
        # reaches the net through Vte AND through the balance the scan derives from it, so it goes
        # in as a gross via the chain the netting set publishes rather than being added on.
        netting = {
            'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
            'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Funding_Rate': 'USD',
            'Balance_Currency': 'USD', 'Liquidation_Period': 10.0, 'Settlement_Period': 0.0,
            'Credit_Support_Amounts': {
                'Received_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
                'Posted_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
                'Independent_Amount': utils.CreditSupportList([[0.0, 0.0]]),
                'Minimum_Received': utils.CreditSupportList([[0.0, 0.0]]),
                'Minimum_Posted': utils.CreditSupportList([[0.0, 0.0]])}}
        node = {'Instrument': construct_instrument(netting, {}), 'Children': [node]}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [node]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _run(curve=CURVE, gradient=False, batch=1024, batches=1, rate=SWAP_RATE, bandwidth=None,
         seed=1, collateralised=False):
    """One CMC run -> (netting mtm, cva, d cva / d(parallel curve shift) or None).

    The curve is the factor to bump: it moves the forward swap rate, so scenarios cross the
    exercise boundary as it shifts. AAD reports one gradient per curve knot, so the parallel-shift
    derivative the CRN ladder measures is their SUM."""
    c = _cfg(curve, rate, collateralised)
    overrides = {
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': GRID, 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': seed, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}}
    if bandwidth is not None:
        overrides['Boundary_AAD_Bandwidth'] = bandwidth
    _, out = riskflow.run_cmc(c, prec=DTYPE, overrides=overrides)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    grad = None
    if gradient:
        g = out['Results']['grad_cva']['Gradient']
        grad = float(sum(g.loc[i] for i in g.index if str(i[0]).startswith('InterestRate.EUR')))
    return out['Results']['mtm'].values, float(out['Results']['cva']), grad


# ------------------------------------------------------------------ the fixture reaches the site

def test_the_fixture_actually_exercises_the_frozen_branch():
    """Before measuring anything: confirm this deal reaches the code under test.

    Three things have to hold or the fixture is testing nothing. The deal must take the PHYSICAL
    branch (Cash_Settled False); the post-expiry block must be NON-EMPTY, since `if
    Ut_swap.shape[0]` skips the frozen mask entirely when it is not; and with greeks requested the
    decision must actually have been recorded. The fourth condition - that the decision is
    genuinely uncertain - is the test below."""
    seen = {}
    import riskflow.instruments as instruments
    original = instruments.SwaptionDeal.post_process

    def spy(self, accum, shared, time_grid, deal_data, child_dependencies):
        result = original(self, accum, shared, time_grid, deal_data, child_dependencies)
        tenor = deal_data.Factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](
            deal_data.Factor_dep['Expiry'] -
            time_grid.time_grid[deal_data.Time_dep.deal_time_grid][:, utils.TIME_GRID_MTM])
        seen['cash_settled'] = deal_data.Factor_dep['Cash_Settled']
        seen['post_expiry_rows'] = int((tenor < 0.0).sum())
        seen['boundary_sets'] = len(getattr(shared, 'boundary_sets', []))
        return result

    instruments.SwaptionDeal.post_process = spy
    try:
        _run(gradient=True, batch=256)
    finally:
        instruments.SwaptionDeal.post_process = original

    assert seen['cash_settled'] is False, 'the fixture is not on the physical branch'
    assert seen['post_expiry_rows'] > 0, (
        'no reporting row falls after option expiry, so the frozen mask is never applied')
    assert seen['boundary_sets'] == 1, (
        f'post_process registered {seen["boundary_sets"]} boundary sets with greeks requested - '
        f'the exercise decision is not reaching the correction at all')


def test_the_exercise_decision_is_genuinely_uncertain():
    """A boundary correction can only recover what crosses. If every scenario exercises (or none
    does) the indicator is constant, its flux is zero and the fixture would measure nothing while
    still looking like a swaption test."""
    fired = {}
    import riskflow.pricing as pricing
    original = pricing.interpolate

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        if deal_data.Instrument.field.get('Reference') == 'SWPT1':
            tenor = deal_data.Factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](
                deal_data.Factor_dep['Expiry'] -
                time_grid.time_grid[deal_data.Time_dep.deal_time_grid][:, utils.TIME_GRID_MTM])
            post = mtm[int((tenor >= 0.0).sum()):]
            fired['share'] = float((post[0] != 0.0).to(torch.float64).mean())
        return original(mtm, shared, time_grid, deal_data, interpolate_grid)

    pricing.interpolate = spy
    try:
        _run(batch=1024)
    finally:
        pricing.interpolate = original

    assert 0.1 < fired['share'] < 0.9, (
        f'{fired["share"]:.1%} of scenarios exercise - the boundary is not populated, so this '
        f'fixture cannot see the defect')


def test_the_registered_branches_reproduce_the_reported_value():
    """The two branches, selected by the recorded flag, must be the deal's reported profile EXACTLY.

    One comparison pins three things that are individually easy to get wrong and individually
    silent: UNITS (the counterfactual is scored against a netting MTM in reporting currency, so the
    branches carry fx_rep), GRID (they go through the same interpolate-and-pad the reported value
    did, so a row cannot slip), and SIGN (which branch is `triggered`).

    Silent because a boundary correction is worth exactly zero in the forward pass - none of these
    would move a reported number, only a gradient, and only by a factor that looks like ordinary
    Monte Carlo error. torch.equal, not allclose: every one of these is an exact identity."""
    import riskflow.pricing as pricing
    original = pricing.interpolate
    seen = {}

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        result = original(mtm, shared, time_grid, deal_data, interpolate_grid)
        if deal_data.Instrument.field.get('Reference') == 'SWPT1':
            seen['reported'] = result
            seen['sets'] = list(shared.boundary_sets)
        return result

    pricing.interpolate = spy
    try:
        _run(gradient=True, batch=256)
    finally:
        pricing.interpolate = original

    bset, = seen['sets']
    reported = seen['reported'].detach()
    selected = torch.where(bset.fired[0], bset.triggered, bset.untriggered)
    assert torch.equal(selected, reported), (
        'the registered branches do not reconstruct the reported deal value - the counterfactual '
        f'is being scored in the wrong units, on the wrong grid, or with the branches swapped; '
        f'max |d| {float((selected - reported).abs().max()):.6g}')


# ---------------------------------------------------------------- safety, must pass now and after

@pytest.mark.parametrize('collateralised', [False, True],
                         ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_swaption_exposure(collateralised):
    """BIT-identical, not approximately. The correction is `gap - gap.detach()`, worth exactly zero
    forward, so this holds by construction - but the registration code that feeds it does not, and
    runs only when greeks are wanted, which is exactly when nobody is checking the value.

    Both netting shapes, because they are different assembly routes: uncollateralised adds the
    deal's delta to the reported MTM, collateralised pushes it through the gross-to-net chain the
    netting set publishes. Both are measured against an oracle further down."""
    kw = dict(batch=512, collateralised=collateralised)
    mtm_off, cva_off, _ = _run(**kw)
    mtm_on, cva_on, grad = _run(gradient=True, **kw)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no interest rate gradient was reported at all'


def test_the_frozen_exercise_is_what_the_residual_is():
    """Attribution, so the fix is aimed at the right thing - and the fixture's noise floor, so the
    acceptance tolerance below means something.

    Struck at 1% against a 3% curve, every scenario exercises: the indicator is CONSTANT, there is
    no flux across the boundary, and the same machinery already agrees. Measured 0.00% at 0.02%
    flatness, which is what makes the 2.46% seen at the money signal rather than Monte Carlo error.
    Any claimed fix has to close the second reading without disturbing this one."""
    kw = dict(batch=1024, rate=1.0)
    aad = _run(gradient=True, **kw)[2]
    r = ladder(price=lambda x: _run(curve=x, **kw)[1], aad=aad, base=CURVE,
               rungs=(1e-4, 2e-4, 5e-4), absolute=True)
    assert r.agrees(tol=0.005), f'a swaption that always exercises should already agree\n{r}'


# ------------------------------------------------------------------------------------- acceptance

def test_physical_exercise_gradient_matches_bump_and_reprice():
    """The frozen exercise indicator in SwaptionDeal.post_process.

    A physically settled swaption really is worth the swap or nothing from expiry on, so the jump
    is genuine product economics and must not be smoothed - the flux of scenarios across the
    exercise boundary is what has to reach the tape.

    Uncorrected this reads 2.06-2.92% LOW across four seeds, always the same sign, against a ladder
    that is already flat (0.38-1.14%) - so this defect does NOT announce itself by scattering the
    way the barrier's did, and the always-exercises control above is what separates it from noise.
    Corrected: 0.02-0.49% on the same four seeds. 1.5% sits between the two, three times the worst
    corrected reading and below the best uncorrected one.

    The rungs stop at 5e-4 (5bp on a 3% curve) because the CVA's curvature in a parallel shift
    starts to show above that; the reading is quoted where the ladder is flat, which is the whole
    point of measuring flatness separately."""
    kw = dict(batch=1024)
    aad = _run(gradient=True, **kw)[2]
    assert abs(aad) > 1e-6, 'a swaption CVA must have an interest rate delta'
    r = ladder(price=lambda x: _run(curve=x, **kw)[1], aad=aad, base=CURVE,
               rungs=(5e-5, 1e-4, 2e-4, 5e-4), absolute=True)
    assert r.agrees(tol=0.015), f'{r}'


def test_collateralised_physical_exercise_gradient_matches_bump_and_reprice():
    """The same defect with collateral in the way, which is the harder half: a gross-mtm delta
    reaches the net through Vte AND through the balance the collateral scan derives from it, so a
    correction that only handles the additive path passes the test above and fails this one.

    It is also the LARGER half. Collateral removes most of the smooth exposure and leaves the
    boundary term as a much bigger share of what is left: uncorrected 8.13-8.80% low across three
    seeds against 2.06-2.92% uncollateralised. Corrected 0.72-1.31% on the same seeds, at 3.6-5.5%
    flatness - the residual CVA here is ~440x smaller, so the ladder is correspondingly noisier and
    needs the extra paths to hold still at all. 4% is three times the worst corrected reading and
    less than half the best uncorrected one."""
    kw = dict(batch=1024, batches=2, collateralised=True)
    aad = _run(gradient=True, **kw)[2]
    r = ladder(price=lambda x: _run(curve=x, **kw)[1], aad=aad, base=CURVE,
               rungs=(5e-5, 1e-4, 2e-4, 5e-4), absolute=True)
    assert r.agrees(tol=0.04), f'{r}'


@pytest.mark.parametrize('bandwidth', [0.005, 0.01, 0.02])
def test_the_correction_holds_still_across_the_usable_bandwidth(bandwidth):
    """No single bandwidth can be argued for on its own, so the estimate has to hold still over a
    range of them - that is what the local-linear weights buy and the only acceptance criterion
    worth having for the kernel itself.

    It holds on 0.005-0.02 (0.10-0.32% from the oracle at 4096 paths) and NOT beyond: 0.05 reads
    +1.5%, 0.10 +3.5%, 0.20 +7.3%, and quadrupling the paths does not shrink any of them, so that
    is estimator BIAS and not Monte Carlo error. The swaption's jump curves in its gap much more
    sharply than a barrier's does - the exercised branch keeps growing with the swap value - so the
    local-linear residual bites at a narrower bandwidth here than at the barrier site, which stayed
    inside 1% out to 0.20. The default of 0.01 sits in the middle of the usable range."""
    kw = dict(batch=1024)
    aad = _run(gradient=True, bandwidth=bandwidth, **kw)[2]
    r = ladder(price=lambda x: _run(curve=x, **kw)[1], aad=aad, base=CURVE,
               rungs=(5e-5, 1e-4, 2e-4, 5e-4), absolute=True)
    assert r.agrees(tol=0.015), f'bandwidth {bandwidth}\n{r}'
