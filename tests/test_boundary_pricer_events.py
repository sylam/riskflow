"""Acceptance criteria for extending the boundary correction from collateral to PRICER events.

Written BEFORE the change, so "done" is defined by measurement rather than by the change looking
plausible. Four sites - the discrete barrier's already-hit latch, the autocall coupon digital, the
autocall put barrier, the TARF knock-in - are all the same defect: a trigger OBSERVED at a
reporting row, whose value jump is real and whose flux across the trigger is missing from the tape.

Two kinds of test here, and the distinction matters.

SAFETY (must pass now and after): asking for sensitivities must not move a reported number. The
correction is `gap - gap.detach()`, worth exactly zero in the forward pass, so this holds by
construction - which is precisely the sort of claim worth pinning, because the registration code
that feeds it does NOT hold by construction and runs only when greeks are wanted.

ACCEPTANCE (xfail now, must pass after): AAD against a common-random-numbers bump ladder. Marked
strict, so they turn the suite RED the moment they start passing and the marker has to come off -
an xfail left lying around after the fix would quietly stop being a gate.
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
from riskflow.instruments import construct_instrument
from crn_ladder import ladder
import test_barrier_bridge as bb

MONTHLY = [[bb.BASE + pd.Timedelta(days=d), 90.0] for d in range(30, 366, 30)]
DISCRETE_BARRIER = dict(bb.BARRIER_DEAL, Barrier_Dates=MONTHLY)

QUARTERLY = [bb.BASE + pd.Timedelta(days=d) for d in (91, 182, 273, 365)]


def _autocall(threshold, barrier=0.0):
    """A quarterly autocall. Every coupon fixing lands on a reporting row - which is not a choice:
    a deal's own dates are folded into the grid - and an ALIGNED fixing is exactly the hard case.
    It is decided by the scenario's own spot, before any inner draw has advanced it, so the pricer
    takes the indicator branch; every FUTURE fixing is a survival probability through norm_cdf and
    was differentiable all along. `Barrier` is a ratio of the strike."""
    return {
        'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': QUARTERLY[-1], 'Units': 1.0,
        'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'Option_Style': 'European',
        'Barrier': barrier, 'Payoff_Type': None,
        'Price_Fixing': [[d, 0.0] for d in QUARTERLY],
        'Autocall_Coupons': [[d, 0.05] for d in QUARTERLY],
        'Autocall_Thresholds': [[d, threshold] for d in QUARTERLY],
        'Barrier_Dates': [[d, barrier] for d in QUARTERLY] if barrier else [],
        'Autocall_Floating': []}


AUTOCALL = _autocall(1.02)
# Both of this deal's indicators saturated: a threshold at 5x spot is reached by no path in a year
# at 25% vol, and a put barrier at 2x the strike is breached by every one. The branches are still
# taken - the registration still runs - but no scenario sits near either boundary, so there is no
# flux to recover and ordinary AAD is already the derivative of the reported value.
AUTOCALL_NO_TRIGGER = _autocall(5.0, barrier=2.0)


def _run(deal, spot=bb.SPOT, gradient=False, batch=512, mcmc=128, collateralised=False,
         batches=1, exclude_paid_today=False):
    """One CMC run returning (netting mtm, cva, equity-spot gradient or None)."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    child = {'Instrument': construct_instrument(deal, {})}
    if collateralised:
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
        # Exclude_Paid_Today lives in the VALUATION CONFIGURATION - NettingCollateralSet reads it
        # from valuation_options, so putting it on the deal dict is silently ignored
        c.deals['Deals']['Children'] = [
            {'Instrument': construct_instrument(
                netting, {'NettingCollateralSet': {'Exclude_Paid_Today': exclude_paid_today}}),
             'Children': [child]}]
    else:
        c.deals['Deals']['Children'] = [child]
    _, out = riskflow.run_cmc(c, prec=bb.DTYPE, overrides={
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # the OSS forks an inner MC per path; runs here are sequential
    grad = None
    if gradient:
        g = out['Results']['grad_cva']['Gradient']
        grad = float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])
    return out['Results']['mtm'].values, float(out['Results']['cva']), grad


# ---------------------------------------------------------------- safety, must pass now and after

@pytest.mark.parametrize('collateralised', [False, True], ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_barrier_exposure(collateralised):
    """BIT-identical, not approximately - a boundary correction is worth exactly zero forward, so
    any drift means the registration path perturbed the valuation rather than observed it.

    Both netting shapes, because they are different code paths: the collateralised branch runs the
    gross/net split in post_process, the uncollateralised one returns an interpolation of the
    accumulated deal mtm with no split at all."""
    mtm_off, cva_off, _ = _run(DISCRETE_BARRIER, collateralised=collateralised)
    mtm_on, cva_on, grad = _run(DISCRETE_BARRIER, gradient=True, collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no equity gradient was reported at all'


@pytest.mark.parametrize('collateralised', [False, True], ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_autocall_exposure(collateralised):
    """The autocall records both branches of its coupon trigger from ONE forward pass, and the
    branch it did not take is carried on a second accumulator rather than a second simulation.
    That is what makes this checkable: a re-run would have consumed the random stream the reported
    value was built from, and the exposure would drift. BIT-identical, not approximately."""
    mtm_off, cva_off, _ = _run(AUTOCALL, collateralised=collateralised)
    mtm_on, cva_on, grad = _run(AUTOCALL, gradient=True, collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no equity gradient was reported at all'


def test_the_autocall_trigger_is_what_the_residual_is():
    """Attribution, so the fix is aimed at the right thing. With no scenario anywhere near either
    of this deal's indicators the registration still runs and still costs nothing, and the
    uncorrected gradient already agrees with bump-and-reprice."""
    kw = dict(batch=1024, mcmc=256, batches=16)
    aad = _run(AUTOCALL_NO_TRIGGER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(AUTOCALL_NO_TRIGGER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), f'an unreachable trigger should already agree\n{r}'


def test_autocall_coupon_digital_gradient_matches_bump_and_reprice():
    """The aligned coupon digital in pv_MC_AutoCallSwap. An autocall observed on its coupon date
    really has redeemed, so the jump is product economics and must NOT be smoothed away - what has
    to reach the tape is the flux of scenarios across the threshold.

    Uncorrected this reads 54% LOW (AAD +1.489e-05 against an oracle of +2.287e-05 at 65536 paths,
    where the ladder is flat to 0.9%). Corrected it lands within 0.02%-0.74% over bandwidths from
    0.005 to 0.2, a 40x range: the estimate holds still, which is the only acceptance worth having
    when no single bandwidth can be argued for. Measured at 39%-65% low and closed to under 2% at
    thresholds 0.95, 1.02 and 1.15 on two seeds."""
    kw = dict(batch=1024, mcmc=256, batches=16)
    aad = _run(AUTOCALL, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(AUTOCALL, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.05), f'{r}'


@pytest.mark.xfail(strict=True, reason='the MTM counterfactual is right (0.82% with cash_settle '
                                       'stubbed out) but an autocall SETTLES its coupon when it '
                                       'fires, and that cash reaches a collateralised net through '
                                       'C_ts_te, which gross_to_net does not carry')
def test_collateralised_autocall_gradient_matches_bump_and_reprice():
    """The same trigger with collateral in the way, and the one thing this port did NOT close.

    A gross-mtm delta reaches the net through Vte and through the balance the collateral scan
    derives from it, and `gross_to_net` runs that whole chain - which works: stub `cash_settle` out
    and the corrected gradient lands 0.82% from the oracle on a ladder flat to 1.15% (uncorrected
    +8.2e-08 against +3.69e-06, so the correction is supplying essentially all of it).

    Shipped, it reads 92% low: +5.31e-06 corrected, +2.31e-06 uncorrected, oracle +1.018e-05. The
    missing channel is CASH. Firing pays the coupon, `cash_settle` books it, and a collateralised
    exposure reads that ledger through C_ts_te - so the counterfactual has to move the settled cash
    as well as the mtm, and `gross_to_net` takes only an mtm delta. Left failing deliberately: the
    number is not noise (flatness 1.68%) and a tolerance widened until a test passes measures
    nothing."""
    kw = dict(batch=1024, mcmc=256, batches=16, collateralised=True)
    aad = _run(AUTOCALL, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(AUTOCALL, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.05), f'{r}'


def test_the_barrier_latch_is_what_the_residual_is():
    """Attribution, so the fix is aimed at the right thing. With the barrier UNREACHABLE the latch
    never fires and the same machinery agrees with bump-and-reprice; with it live, it does not.
    Any claimed fix has to close the second reading without disturbing the first."""
    far = dict(bb.BARRIER_DEAL, Barrier_Price=1e-6,
               Barrier_Dates=[[d, 1e-6] for d, _ in MONTHLY])
    kw = dict(batch=1024, mcmc=256)
    aad = _run(far, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(far, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), f'an unreachable barrier should already agree\n{r}'


# ------------------------------------------------------- acceptance, xfail until the change lands

def test_discrete_barrier_latch_gradient_matches_bump_and_reprice():
    """The already-hit latch in pv_discrete_barrier_option. A discretely monitored knock-out really
    is worth nothing once it crosses, so the jump is genuine product economics and must NOT be
    smoothed away - the flux of paths across the barrier is what has to reach the tape.

    Uncorrected this read 13% off. The rungs stop at 2e-3 because 5e-3 is a half-point bump on a
    spot of 100, where the ORACLE stops converging at these path counts - the flatness check
    refuses that reading, which is the behaviour wanted from it."""
    kw = dict(batch=1024, mcmc=256)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.05), f'{r}'


@pytest.mark.xfail(strict=True, reason='route is BUILT and converging - AAD +0.000904 vs CRN '
                                       '+0.000963, flatness 5.25% - but 6.48% at the path count '
                                       'this GPU allows, against a 5% bar. Not loosened to pass: '
                                       'confirm at higher paths, do not tune the tolerance')
def test_collateralised_barrier_latch_gradient_matches_bump_and_reprice():
    """The same defect with collateral in the way, which is the harder half: a gross-mtm delta
    reaches the net through Vte AND through the balance the collateral scan produces, so a fix
    that only handles the additive path will pass the test above and fail this one - which is
    exactly what happened, and what sent the gross-to-net chain into post_process.

    It now runs that chain and converges (flatness 5.25%), landing 6.48% from the oracle where the
    bar is 5%. The exposure here is almost entirely collateralised away, so the number is ~16x
    smaller than the uncollateralised one and correspondingly noisier, and this GPU will not hold
    enough paths to separate residual bias from Monte Carlo error. Left failing deliberately: a
    tolerance widened until a test passes measures nothing."""
    kw = dict(batch=1024, mcmc=256, collateralised=True)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1],
               aad=aad, base=bb.SPOT, rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.05), f'{r}'


def _fva(spot, gradient, batch=1024, mcmc=192):
    """FVA and its equity-spot gradient. A funding SPREAD is what makes it non-zero - with the cost,
    benefit and risk-free curves all equal the adjustment is identically zero and measures nothing."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    c.params['Price Factors']['InterestRate.FUND'] = {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, 0.02], [10.0, 0.02]])}
    c.params['Price Factors']['DiscountRate.FUND'] = {'Interest_Rate': 'FUND'}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(DISCRETE_BARRIER, {})}]
    _, out = riskflow.run_cmc(c, prec=bb.DTYPE, overrides={
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Funding_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Funding_Cost_Interest_Curve': 'FUND',
            'Funding_Benefit_Interest_Curve': 'FUND', 'Risk_Free_Curve': 'USD',
            'Counterparty': 'CPTY', 'Gradient': 'Yes' if gradient else 'No'}})
    if not gradient:
        return float(out['Results']['fva'])
    g = out['Results']['grad_fva']['Gradient']
    return float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])


def test_fva_gradient_carries_the_boundary_term_too():
    """FVA reads the same exposure as CVA, so it drops the same boundary terms - and it is the path
    that matters in production, because the shipped batch job DELETES the CVA section, so a
    correction assembled only over there could never fire for it.

    Measured on this fixture: 3.86% off uncorrected, 0.65% with the correction, the CRN ladder clean
    at 1.27% flatness both times - so the movement is the correction and not the oracle wandering."""
    assert _fva(bb.SPOT, False) > 0.0, 'no funding spread - the adjustment is identically zero'
    aad = _fva(bb.SPOT, gradient=True)
    r = ladder(price=lambda s: _fva(s, False), aad=aad, base=bb.SPOT, rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), f'the fva gradient is missing its boundary term\n{r}'


def test_the_correction_generalises_to_the_other_barrier_direction():
    """The gap must be signed so gap > 0 means CROSSED, and that sign flips with the barrier
    direction - a DOWN barrier is crossed from above, an UP barrier from below. Getting it backwards
    still converges, it just pulls the wrong way, so a second direction has to be measured rather
    than reasoned about.

    Up-and-IN is the variant with material exposure: it knocks in as the call goes into the money.
    Its mirror images are deliberately NOT gated - an up-and-OUT call is knocked out exactly when it
    becomes valuable, and a down-and-in call knocks in deep out of the money, so both carry a CVA
    delta near -0.0003 where the CRN oracle itself stops converging (measured flatness 28% and 82%).
    A gate there would be pinning Monte Carlo noise."""
    H = 110.0
    deal = dict(bb.BARRIER_DEAL, Barrier_Type='Up_And_In', Barrier_Price=H,
                Barrier_Dates=[[bb.BASE + pd.Timedelta(days=d), H] for d in range(30, 366, 30)])
    kw = dict(batch=1024, mcmc=256)
    aad = _run(deal, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(deal, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), f'up-barrier gap sign or counterfactual is wrong\n{r}'


def test_the_correction_covers_heston_nandi_barriers():
    """instruments.py refuses the CONTINUOUS barrier variant for SpotModel='HestonNandi', so every
    HN barrier deal routes through the discrete pricer and its already-hit latch - the audit put
    that at 100% of them. The registration sits in the shared pricer, which should cover HN, but
    'should' is not a measurement: HN takes a different branch through sim_spot_oss, and its
    hit_value for a knock-out is zeros rather than a closed form.

    Measured 0.23% against bump-and-reprice at 0.77% flatness, on a CVA delta of 1.46 - large enough
    that the oracle resolves it cleanly, unlike the mirror-image barrier variants."""
    import test_hn_barrier_cmc as hb

    def run(spot, gradient):
        c = hb._cfg(True)
        c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
        c.params['Price Factors']['SurvivalProb.CPTY'] = {
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
        _, out = riskflow.run_cmc(c, prec=hb.DTYPE, overrides={
            'Run_Date': hb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': 512,
            'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
            'MCMC_Simulations': 256, 'Deflation_Interest_Rate': 'USD',
            'Gradient_Variables': 'Factors',
            'Credit_Valuation_Adjustment': {
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}})
        if not gradient:
            return float(out['Results']['cva'])
        g = out['Results']['grad_cva']['Gradient']
        return float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])

    aad = run(100.0, True)
    r = ladder(price=lambda s: run(s, False), aad=aad, base=100.0, rungs=(5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), f'the HN barrier path is not carrying the boundary term\n{r}'


def test_the_correction_scales_correctly_across_simulation_batches():
    """`boundary_sets` is cleared per batch while `.grad` ACCUMULATES across them, and report()
    divides by Simulation_Batches. A correction added once but averaged over N - or accumulated N
    times without averaging - would make the reported gradient scale with the batch count, which no
    single-batch test can see. Measured 0.85% / 0.37% / 0.01% at 1 / 2 / 4 batches, tracking the
    oracle at each and tightening with paths as it should."""
    kw = dict(batch=512, mcmc=192)
    aad = _run(DISCRETE_BARRIER, gradient=True, batches=2, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, batches=2, **kw)[1],
               aad=aad, base=bb.SPOT, rungs=(1e-3, 2e-3))
    assert r.agrees(tol=0.03), f'the correction does not survive multiple batches\n{r}'


@pytest.mark.parametrize('exclude_paid_today', [False, True], ids=['plain', 'exclude_paid_today'])
def test_a_zero_gross_delta_reproduces_the_reported_net(exclude_paid_today):
    """The invariant the collateral counterfactual rests on, which nothing asserted.

    `gross_to_net` pushes a gross-mtm delta through At -> required balance -> bands -> scan -> the
    netting arithmetic. Feed it ZERO and it must return the netting set's reported net EXACTLY - if
    it does not, every correction is measured against a rebased baseline and is the wrong size while
    still converging and still looking bandwidth-stable.

    It did not. `Vte` was re-derived as `g_Vt[Te]` rather than taken from the reported `b_Vte`, and
    under Exclude_Paid_Today the two carry DIFFERENT cashflow adjustments - a local-grid one and a
    Te-grid one. Measured with the mutation restored: max|diff| 120.51 against a reported |mean| of
    2.61, a 46x rebasing. Taking b_Vte makes the invariant true by construction.

    Both settings are gated because the defect is INVISIBLE at the default: with the option off the
    two forms coincide exactly, and no value gate can see it either way - the reported mtm is
    unchanged, which is what let it hide. Note Exclude_Paid_Today is read from the VALUATION
    CONFIGURATION, not the deal's fields; setting it on the netting dict is silently ignored and
    makes this test vacuous."""
    import riskflow.calculation as C
    seen = {}
    original = C.pricer_boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        chain = getattr(shared, 'gross_to_net', None)
        bset = next((x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)), None)
        if chain is not None and bset is not None:
            with torch.no_grad():
                seen['diff'] = float((chain(torch.zeros_like(bset.untriggered))
                                      - reported_mtm).abs().max())
        return original(shared, objective, reported_mtm, bandwidth)

    C.pricer_boundary_correction = probe
    try:
        _run(dict(DISCRETE_BARRIER, Cash_Rebate=5.0), gradient=True, collateralised=True,
             batch=256, mcmc=64, exclude_paid_today=exclude_paid_today)
    finally:
        C.pricer_boundary_correction = original

    assert 'diff' in seen, 'the collateral chain never ran - the fixture is not exercising it'
    assert seen['diff'] == 0.0, (
        f'a zero gross delta moved the net by {seen["diff"]:.4e}: the counterfactual is rebased, '
        f'so every correction against it is mis-sized')
