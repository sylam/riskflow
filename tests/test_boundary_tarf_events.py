"""The two decisions inside the TARF one-step-survival pricer that jump, and reach the tape.

`pv_MC_Tarf` runs an INNER Monte Carlo of `2 * MCMC_Simulations` paths per reporting row, and two
things inside it are decided on simulated state and carry a real value jump:

  KNOCK-IN   `barrier_hit = (barrier - Sj) * callOrPut >= 0` switches the OTM leg on for one INNER
             path. Not smoothable - the client either owes the leveraged leg or does not - and the
             jump is `N_otm * |K - barrier|`, which on this fixture is 60,000 on a deal worth 8,262.
  TARGET PIN `q = remaining_target == 0` zeroes the survival weight, so the deal is worth nothing
             from there on. `calc_accum_value` ends in `.clamp(max=targetValue)`, so this exact
             float equality fires on a POSITIVE-MEASURE set - measured below at 27.7%-61.3% of
             outer paths - and it is a redemption, not a rounding artifact.

TWO REACHABILITY TRAPS, both of which make a fixture measure nothing if you miss them.

  `Barrier` was NOT in `FXTARFOptionDeal.Fields`, so no schema-authored deal could emit it and the
  knock-in was reachable only because `instruments.py:440` keeps the params dict unfiltered. Added,
  on the `Barrier_Price`/EquityBarrierBinaryOption precedent, and gated below.

  `LeverageNotional` (N_otm) defaults to 0, and BOTH sites are dead at that default.
  `cf_otm = relu(-intr) * N_otm * barrier_hit` multiplies the knock-in by zero. And the target pin
  becomes CONTINUOUS: as `remaining_target -> 0` the KO term `(1-p)*L*R`, the clamped intrinsic and
  every surviving cashflow all go to zero with it, so zeroing the weight costs nothing. Measured -
  same fixture, same 61.3% firing rate, `LeverageNotional=0`: the uncorrected AAD agrees with
  bump-and-reprice to 0.00%-1.14% out to 3e-4. The pin is a boundary only when the OTM leg pays.

WHY THE LADDERS START AT 3e-4. Differencing across a jump does not converge as h shrinks - it
changes how many paths sit on the wrong side. Below ~1e-4 the CRN readings scatter over 1/h
(measured: -69k, -1.59e6, -525k, -222k, -152k as h goes 1e-6 -> 1e-4), which is the signature of a
discontinuity and not something to average. The rungs kept are the ones where the oracle plateaus.
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
from riskflow import run_baseval, utils
from riskflow.config import Config
from riskflow.instruments import construct_instrument
from crn_ladder import ladder

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
SPOT = STRIKE = 0.65
SIGMA = 0.12
N1 = 1_000_000.0
N2 = 2_000_000.0          # LeverageNotional -> the OTM leg. Zero kills BOTH sites; see the header.
BARRIER = 0.62            # < K, so the knock-in bites where the OTM leg pays
MONTHLY = [30, 60, 90]
BIMONTHLY = [60 * (i + 1) for i in range(6)]
UNREACHABLE_TARGET = 1e9


def _price_factors(spot):
    return {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'FxRate.AUD': {'Domestic_Currency': 'USD', 'Interest_Rate': 'AUD', 'Priority': 1,
                       'Spot': spot},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'InterestRate.AUD': {'Currency': 'AUD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'FXVol.AUD.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.5, 1.0, 1.5)
                                                      for t in (0.02, 2.0)])},
    }


def _tarf(target, fix_days, barrier=None, leverage=N2, buy_sell='Buy'):
    fix_dates = [BASE + pd.Timedelta(days=d) for d in fix_days]
    deal = {
        'Object': 'FXTARFOptionDeal', 'Reference': 'TARF1', 'Currency': 'USD',
        'Underlying_Currency': 'AUD', 'Discount_Rate': 'USD', 'FX_Volatility': 'AUD.USD',
        'Buy_Sell': buy_sell, 'Expiry_Date': fix_dates[-1], 'Underlying_Amount': N1,
        'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Settlement_Style': 'Physical',
        'Option_Style': 'European', 'InvertedTarget': False, 'LeverageNotional': leverage,
        'TargetAdjustment': '', 'TargetLevel': target,
        'TARF_ExpiryDates': [[d, d, None] for d in fix_dates]}
    if barrier is not None:
        deal['Barrier'] = barrier
    return deal


def _cfg(deal, spot, counterparty=False, simulate_fx=False):
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = _price_factors(spot)
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    if counterparty:
        c.params['Price Factors']['SurvivalProb.CPTY'] = {
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    if simulate_fx:
        c.params['Price Models'] = {'GBMAssetPriceModel.AUD': {'Vol': SIGMA, 'Drift': 0.0}}
        c.params['Model Configuration'].append('FxRate', (), 'GBMAssetPriceModel')
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _baseval(deal, spot=SPOT, greeks=False, sims=1 << 16, bandwidth=None):
    """(price, d(price)/d(FxRate.AUD spot)). One date, one scenario - and still a full inner MC
    underneath, which is where the knock-in is decided."""
    overrides = {'MCMC_Simulations': sims, 'Random_Seed': 1,
                 'Greeks': 'First' if greeks else 'No'}
    if bandwidth is not None:
        overrides['Boundary_AAD_Bandwidth'] = bandwidth
    _, out = run_baseval(_cfg(deal, spot), overrides=overrides)
    rows = out['Results']['mtm']
    price = float(rows[rows['Reference'] == 'TARF1']['Value'].iloc[0])
    grad = None
    if greeks:
        frame = out['Results']['Greeks_First']
        # two columns: 'Value' is the FACTOR LEVEL (display_val=True), the other is the gradient
        column = [x for x in frame.columns if x != 'Value'][0]
        index, = [i for i in frame.index if str(i[0]) == 'FxRate.AUD']
        grad = float(frame.loc[index, column])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return price, grad


def _cmc(deal, spot=SPOT, gradient=False, batches=4, batch=512, mcmc=128, bandwidth=None):
    """(cva, mtm profile, d(cva)/d(FxRate.AUD spot))."""
    overrides = {
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2m(2m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}}
    if bandwidth is not None:
        overrides['Boundary_AAD_Bandwidth'] = bandwidth
    _, out = riskflow.run_cmc(
        _cfg(deal, spot, counterparty=True, simulate_fx=True), prec=DTYPE, overrides=overrides)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    grad = None
    if gradient:
        g = out['Results']['grad_cva']['Gradient']
        # absent rather than zero when the book has no exposure to the factor at all - which is
        # the case for the zero-leverage control, whose registration is still what is being read
        index = [i for i in g.index if str(i[0]) == 'FxRate.AUD']
        grad = float(g.loc[index[0]]) if index else None
    return float(out['Results']['cva']), out['Results']['mtm'].values, grad


KNOCK_IN = _tarf(UNREACHABLE_TARGET, MONTHLY, barrier=BARRIER)
NO_BARRIER = _tarf(UNREACHABLE_TARGET, MONTHLY)
# Sell, or the exposure of this book is ~0 on every path and the CVA gradient is noise
KNOCK_IN_CMC = _tarf(UNREACHABLE_TARGET, BIMONTHLY, barrier=BARRIER, buy_sell='Sell')
PIN_CMC = _tarf(0.02, BIMONTHLY, buy_sell='Sell')
PIN_UNREACHABLE_CMC = _tarf(UNREACHABLE_TARGET, BIMONTHLY, buy_sell='Sell')
PIN_NO_LEVERAGE_CMC = _tarf(0.02, BIMONTHLY, leverage=0.0, buy_sell='Sell')


# ---------------------------------------------------------------- reachability

def test_the_knock_in_barrier_is_reachable_from_the_schema():
    """`Barrier` is read with a hard key by pricing.pv_MC_Tarf. It was not in the deal's Fields, so
    a deal authored to the schema could not switch the leveraged leg on at all - the same defect
    that made a schema-authored EquityBarrierBinaryOption silently skip, and the same fix.

    Asserting the FIELD LIST and not just that the pricer sees a hand-written dict: the dict path
    worked before too (`instruments.py` keeps params unfiltered), so a test that only priced one
    would have passed against the defect."""
    from riskflow.fields import mapping
    instrument = mapping['Instrument']
    assert 'Barrier' in instrument['sections']['FXTARFOptionDeal.Fields'], (
        'Barrier is not in FXTARFOptionDeal.Fields, so no schema-authored TARF can carry a '
        'knock-in and the OTM leg is unreachable')
    assert 'Barrier' in instrument['fields'], 'the Barrier widget descriptor is missing'
    priced, _ = _baseval(KNOCK_IN, sims=1 << 12)
    unbarriered, _ = _baseval(NO_BARRIER, sims=1 << 12)
    assert priced != unbarriered, 'the Barrier key did not reach the pricer at all'


def test_a_single_scenario_gap_does_not_poison_the_scalar_being_differentiated():
    """Base valuation runs ONE scenario, so a decision taken per scenario registers a gap with a
    single element - and `torch.std` returns NaN on one sample rather than raising. That NaN
    reaches the kernel width, the density, and then the scalar handed to backward().

    It hid because `0 * NaN` is NaN in the forward pass but reaches nothing in the backward one
    while the degenerate gap happens to carry no graph, which is the case for the historic-accrual
    decision that produced it. A gap in the same shape that DID carry one would have turned every
    reported greek into NaN, and the reported price - read from the stored result, not from this
    scalar - would still have looked perfect.

    Exactly zero, not merely finite: one sample supports no local-linear fit, so the honest answer
    is that this decision contributes nothing."""
    import riskflow.pricing as pricing
    gap = torch.tensor([0.3], dtype=DTYPE, requires_grad=True)
    jump = torch.tensor([5.0], dtype=DTYPE)
    correction = pricing.stochastic_boundary_correction(gap, jump, 0.01)
    assert torch.isfinite(correction), f'a one-sample gap produced {correction!r}'
    assert float(correction) == 0.0, f'expected exactly zero, got {float(correction)!r}'


# ---------------------------------------------------------------- safety: the value must not move

@pytest.mark.parametrize('deal,label', [(KNOCK_IN, 'knock-in'), (NO_BARRIER, 'control')])
def test_asking_for_sensitivities_does_not_move_the_tarf_price(deal, label):
    """BIT-identical, not approximately. A boundary correction is `gap - gap.detach()`, worth
    exactly zero forward, so this holds by construction - but the registration that feeds it does
    not: the target-pin counterfactual carries a SECOND survival weight through the same loop, and
    a second accumulator that consumed one random number would move the reported price."""
    off, _ = _baseval(deal, sims=1 << 14)
    on, grad = _baseval(deal, greeks=True, sims=1 << 14)
    assert off == on, f'{label}: price moved when sensitivities were requested: {off!r} -> {on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no FX gradient was reported at all'


def test_asking_for_sensitivities_does_not_move_the_tarf_exposure():
    """The same statement under exposure, where the pin has multiple rows to latch across and the
    counterfactual weight runs the whole block loop beside the reported one."""
    cva_off, mtm_off, _ = _cmc(PIN_CMC, batches=1)
    cva_on, mtm_on, grad = _cmc(PIN_CMC, gradient=True, batches=1)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no FX gradient was reported at all'


# ---------------------------------------------------------------- B1: the OTM-leg knock-in

def test_the_knock_in_gradient_matches_bump_and_reprice_under_exposure():
    """The acceptance gate for the knock-in, and the one whose oracle resolves cleanly.

    Measured on this fixture, identical CRN readings before and after: AAD -59,971.64 against an
    oracle of -83,856.45 at 0.90% flatness, i.e. 39.83% SHORT; corrected, -83,663.21, i.e. 0.23%.

    An intermediate reading is worth recording because it is what a plausible half-fix produces.
    Registering only the fixings this row has yet to observe (`dt > 0`) left 5.50% behind: under
    CMC a deal's own dates are folded into the mtm grid, so EVERY fixing date is also a reporting
    row, and on that row the first fixing is a past reset whose knock-in was going unregistered."""
    _, _, aad = _cmc(KNOCK_IN_CMC, gradient=True)
    r = ladder(price=lambda s: _cmc(KNOCK_IN_CMC, spot=s)[0], aad=aad, base=SPOT,
               rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.02), f'the knock-in flux is not reaching the tape\n{r}'


def test_the_knock_in_gradient_matches_bump_and_reprice_at_base_valuation():
    """Base valuation, which is where this was first measured and where the pricer's inner MC is
    the ONLY simulation there is - one date, one scenario, 2x65536 inner paths.

    It is also the route that had no boundary correction at all: `Base_Reval_State` never recorded
    a decision, so a deal priced by Monte Carlo reported a gradient with the flux missing and
    nothing in the calculation could have told you. Measured: AAD 2,421,013 against a CRN plateau
    of ~3.766e6, 35.7% short; corrected, 3,768,262, i.e. 0.07%. The control with no Barrier key
    agrees to 0.00% at 0.01% flatness both before and after, which is what says the machinery costs
    nothing where there is no boundary to cross."""
    _, aad = _baseval(KNOCK_IN, greeks=True)
    r = ladder(price=lambda s: _baseval(KNOCK_IN, spot=s)[0], aad=aad, base=SPOT,
               rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.02), f'the knock-in flux is not reaching the tape\n{r}'


def test_the_control_with_no_barrier_was_already_right_and_stays_right():
    """Attribution: with no knock-in there is no boundary, so ordinary AAD is already the
    derivative of the reported value and must remain so. This is the reading that says the 35.7%
    above is the barrier and not something else in the pricer - and the ladder here is flat to
    0.01%, four decades of it, which the barriered one never is."""
    _, aad = _baseval(NO_BARRIER, greeks=True)
    r = ladder(price=lambda s: _baseval(NO_BARRIER, spot=s)[0], aad=aad, base=SPOT,
               rungs=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.01), f'an unbarriered TARF should already agree\n{r}'


def test_the_knock_in_correction_holds_still_across_the_usable_bandwidth():
    """No single bandwidth can be argued for, so the acceptance is that the estimate does not
    depend on it. Measured across a 40x range, 0.005 to 0.2: 3,780,461 / 3,768,262 / 3,771,541 /
    3,782,522 / 3,790,539 / 3,791,727 - a 0.62% spread on a number that was 35.7% wrong.

    The price is checked at every bandwidth too: a correction is worth exactly zero forward whatever
    its width, so any movement means the registration perturbed the valuation."""
    price, values = None, []
    for bandwidth in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        px, aad = _baseval(KNOCK_IN, greeks=True, bandwidth=bandwidth)
        price = px if price is None else price
        assert px == price, f'the price moved with the bandwidth: {price!r} -> {px!r}'
        values.append(aad)
    spread = (max(values) - min(values)) / abs(np.median(values))
    assert spread < 0.02, (
        f'the correction tracks the bandwidth ({spread:.2%} over 40x) - that is estimator bias, '
        f'not a converged estimate: {[round(v) for v in values]}')


# ---------------------------------------------------------------- B2: the target pin

def _pin_registration(deal, batch=512):
    """Run, and return (the deal's own reported profile, the sets registered for it)."""
    import riskflow.pricing as pricing
    original = pricing.interpolate
    seen = {}

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        result = original(mtm, shared, time_grid, deal_data, interpolate_grid)
        if deal_data.Instrument.field.get('Reference') == 'TARF1':
            seen['reported'] = result.detach()
            seen['sets'] = [x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)]
        return result

    pricing.interpolate = spy
    try:
        _cmc(deal, gradient=True, batches=1, batch=batch)
    finally:
        pricing.interpolate = original
    return seen


def test_the_pin_fires_on_a_material_share_of_paths_and_its_branches_are_the_reported_value():
    """Two statements, and the first is what stops the second from being vacuous.

    The pin must actually FIRE. `q = remaining_target == 0.0` is an exact float equality, and the
    only reason it is a boundary rather than a curiosity is that `calc_accum_value` clamps AT the
    target, so it fires on a positive-measure set. Measured on this fixture, per block:
    0% / 27.7% / 43.2% / 51.8% / 57.4% / 61.3% of outer paths. A TargetLevel that no path reaches
    would make every assertion below true and measure nothing.

    Then the branches, selected by the recorded flags, must reconstruct the deal's reported profile
    EXACTLY - which pins the grid, the currency, the latch bookkeeping and which branch is which,
    all of which are invisible in a forward pass worth zero. torch.equal, not allclose."""
    seen = _pin_registration(PIN_CMC)
    latched = [x for x in seen['sets'] if isinstance(x, utils.LatchedBoundarySet)]
    assert len(latched) == 1, f'expected one target-pin registration, got {seen["sets"]}'
    bset, = latched
    fired = [float(f.to(DTYPE).mean()) for f in bset.fired]
    assert max(fired) > 0.2, (
        f'the pin fires on at most {max(fired):.1%} of paths - this fixture gates nothing; '
        f'lower TargetLevel until it does')
    assert fired[0] == 0.0, 'block 0 reads the HISTORIC accrual, which cannot differ by scenario'

    prefix = [torch.zeros_like(bset.fired[0])]
    for flag in bset.fired:
        prefix.append(prefix[-1] | flag)
    selected = bset.to_mtm(torch.where(
        torch.stack(prefix)[bset.obs_before], bset.triggered, bset.untriggered))
    assert torch.equal(selected, seen['reported']), (
        'the registered branches do not reconstruct the reported deal value; max |d| '
        f'{float((selected - seen["reported"]).abs().max()):.6g} against a reported |mean| of '
        f'{float(seen["reported"].abs().mean()):.6g}')
    assert float(bset.triggered.abs().max()) == 0.0, (
        'a redeemed TARF is worth nothing - the triggered branch should be identically zero')
    assert float(bset.untriggered.abs().max()) > 0.0, (
        'the untriggered branch is identically zero, so the counterfactual weight never ran')

    for k, (gap, flag) in enumerate(zip(bset.gaps[1:], bset.fired[1:]), start=1):
        assert torch.equal(gap.detach() >= 0.0, flag), (
            f'decision {k}: gap > 0 does not mean the target FILLED, so the correction is scored '
            f'against the wrong branch and will pull the wrong way')
        # An ATOM at the boundary is the signature of building the gap from the accrual the pricer
        # reports, which `calc_accum_value` clamps AT the target: every filled path then sits at
        # exactly zero with a derivative to match, so the one quantity that has to carry the flux
        # carries none of it. The UNCLAMPED running accrual is the same test and the same sign.
        assert float((gap.detach() == 0.0).to(DTYPE).mean()) == 0.0, (
            f'decision {k}: {float((gap.detach() == 0.0).to(DTYPE).mean()):.1%} of paths sit at '
            f'gap == 0 exactly. A crossing is a measure-zero event; an atom there means the gap '
            f'came off the CLAMPED accrual, which is flat past the decision it is meant to detect')
        assert gap.requires_grad, f'decision {k}: the gap carries no graph at all'


def test_the_pin_correction_vanishes_where_the_target_cannot_fill():
    """The other side of the fixture check, and a tight gate in its own right: with a target no
    path reaches, every gap sits far from zero, the kernel underflows to zero and the correction
    must contribute NOTHING. So the gradient has to agree with bump-and-reprice on its own terms.

    This is the reading that says the pin machinery is inert where there is no boundary - and it
    runs the SAME code path (the second survival weight, the whole block loop) as the live one."""
    _, _, aad = _cmc(PIN_UNREACHABLE_CMC, gradient=True, batches=2)
    r = ladder(price=lambda s: _cmc(PIN_UNREACHABLE_CMC, spot=s, batches=2)[0], aad=aad,
               base=SPOT, rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.02), f'an unreachable target should already agree\n{r}'


@pytest.mark.parametrize('deal,leverage', [(PIN_NO_LEVERAGE_CMC, 0.0), (PIN_CMC, N2)])
def test_the_pin_is_a_boundary_only_when_the_leveraged_leg_pays(deal, leverage):
    """A design finding, gated so it cannot be forgotten, and read exactly where it lives: the
    value a PINNED path would still have been worth had the target not quite filled.

    With LeverageNotional = 0 that value is identically zero. As remaining_target -> 0 the KO term
    (1-p)*L*R goes to zero with R, the intrinsic is clamped at a vanishing remaining target so
    cf_itm -> 0, and cf_otm = relu(-intr) * N_otm is zero because N_otm is. Nothing is lost by
    redeeming, so there is no jump and the pin is CONTINUOUS - measured, the uncorrected gradient
    already agreed with bump-and-reprice to 0.00%-1.14% on the identical fixture at a 61.3% firing
    rate. Turn the leg on and the same pinned path was worth tens of thousands: that is the
    obligation the redemption cancels, and it is the whole jump.

    So any future fixture for this site that drops the leverage leg measures nothing, whatever it
    asserts. Both directions are gated, because only the pair says which way round it is."""
    seen = _pin_registration(deal)
    bset, = [x for x in seen['sets'] if isinstance(x, utils.LatchedBoundarySet)]
    fired = [float(f.to(DTYPE).mean()) for f in bset.fired]
    assert max(fired) > 0.2, (
        f'the pin fires on at most {max(fired):.1%} of paths, so this says nothing either way')
    last = len(bset.fired) - 1
    rows = np.flatnonzero(np.asarray(bset.obs_before) == last + 1)
    alive = bset.untriggered[rows][:, bset.fired[last]].abs().max()
    if leverage:
        assert float(alive) > 0.0, (
            'a pinned path is worth nothing had it NOT redeemed, so redemption costs nothing and '
            'the pin is not a boundary - but the OTM leg is switched on, so it should be')
    else:
        assert float(alive) == 0.0, (
            f'with no OTM leg a pinned path should have had nothing left to lose, yet the '
            f'counterfactual is worth {float(alive):.6g} - the pin IS a boundary at zero leverage '
            f'and the header of this file is wrong')
