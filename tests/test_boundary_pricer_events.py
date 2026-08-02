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


def _run(deal, spot=bb.SPOT, gradient=False, batch=512, mcmc=128, collateralised=False):
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
        c.deals['Deals']['Children'] = [
            {'Instrument': construct_instrument(netting, {}), 'Children': [child]}]
    else:
        c.deals['Deals']['Children'] = [child]
    _, out = riskflow.run_cmc(c, prec=bb.DTYPE, overrides={
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD',
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
