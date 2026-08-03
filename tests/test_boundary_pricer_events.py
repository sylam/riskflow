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


NETTING = {
    'Object': 'NettingCollateralSet', 'Netted': 'True', 'Agreement_Currency': 'USD',
    'Funding_Rate': 'USD', 'Balance_Currency': 'USD', 'Liquidation_Period': 10.0,
    'Settlement_Period': 0.0,
    'Credit_Support_Amounts': {
        'Received_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
        'Posted_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
        'Independent_Amount': utils.CreditSupportList([[0.0, 0.0]]),
        'Minimum_Received': utils.CreditSupportList([[0.0, 0.0]]),
        'Minimum_Posted': utils.CreditSupportList([[0.0, 0.0]])}}

# A deal that contributes an mtm date NOBODY else has. The barrier's own reval dates are the 3m
# reporting grid plus its monitoring dates, so with it alone the deal grid IS the mtm grid and the
# interpolation `Deal.calculate` performs is the identity - the state in which a branch registered
# on the deal grid and padded at the tail happens to be right. Day 137 is the parameter the defect
# lives in: it makes `gather_interp_matrix` insert a row in the MIDDLE.
INTERPOLATING_DEAL = {
    'Object': 'EquityForwardDeal', 'Reference': 'FWD1', 'Currency': 'USD', 'Equity': 'EQ',
    'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': 'Buy', 'Units': 1.0,
    'Forward_Price': 100.0, 'Maturity_Date': bb.BASE + pd.Timedelta(days=137)}


def _foreign_report_currency(c, ccy='EUR'):
    """Report in `ccy` while every deal still pays USD, so `fx_rep` is a simulated (T, B) cross
    rather than `shared.one`. Nothing else about the portfolio changes."""
    c.params['Price Factors']['FxRate.' + ccy] = {
        'Domestic_Currency': None, 'Interest_Rate': ccy, 'Priority': 1, 'Spot': 1.25}
    c.params['Price Factors']['FxRate.USD']['Domestic_Currency'] = ccy
    c.params['Price Factors']['InterestRate.' + ccy] = {
        'Currency': ccy, 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])}
    c.params['Price Factors']['DiscountRate.' + ccy] = {'Interest_Rate': ccy}
    c.params['Price Models']['GBMAssetPriceModel.USD'] = {'Vol': 0.12, 'Drift': 0.0}
    c.params['Model Configuration'].append('FxRate', (), 'GBMAssetPriceModel')
    return ccy


def _run(deal, spot=bb.SPOT, gradient=False, batch=512, mcmc=128, collateralised=False,
         batches=1, exclude_paid_today=False, extra_deals=(), report_currency='USD',
         children=None):
    """One CMC run returning (netting mtm, cva, equity-spot gradient or None)."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    if report_currency != 'USD':
        _foreign_report_currency(c, report_currency)
    kids = [{'Instrument': construct_instrument(d, {})} for d in (deal,) + tuple(extra_deals)]
    if children is not None:
        c.deals['Deals']['Children'] = children(c)
    elif collateralised:
        # Exclude_Paid_Today lives in the VALUATION CONFIGURATION - NettingCollateralSet reads it
        # from valuation_options, so putting it on the deal dict is silently ignored
        c.deals['Deals']['Children'] = [
            {'Instrument': construct_instrument(
                dict(NETTING, Reference='NS1', Collateralized='True'),
                {'NettingCollateralSet': {'Exclude_Paid_Today': exclude_paid_today}}),
             'Children': kids}]
    else:
        c.deals['Deals']['Children'] = kids
    _, out = riskflow.run_cmc(c, prec=bb.DTYPE, overrides={
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': report_currency,
        'Tenor_Offset': 0.0,
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


def _spy_registration(reference, **run_kw):
    """Run, and return the deal's OWN reported profile alongside the sets registered for it.

    `pricing.interpolate` is the last thing `Deal.calculate` does, so its return value IS what the
    deal contributes to the netting MTM - already gathered onto the MTM grid and already padded.
    That makes it the only honest thing to compare a branch against: the netting mtm is a SUM, so
    once a second deal is on the grid it stops being this deal's value at all."""
    import riskflow.pricing as pricing
    original = pricing.interpolate
    seen = {}

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        result = original(mtm, shared, time_grid, deal_data, interpolate_grid)
        if deal_data.Instrument.field.get('Reference') == reference:
            seen['reported'] = result.detach()
            seen['sets'] = [x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)]
            seen['mtm_dates'] = time_grid.mtm_time_grid
            seen['deal_dates'] = time_grid.time_grid[
                deal_data.Time_dep.deal_time_grid][:, utils.TIME_GRID_MTM]
        return result

    pricing.interpolate = spy
    try:
        _run(gradient=True, **run_kw)
    finally:
        pricing.interpolate = original
    return seen


def _latched_reported(bset):
    """The value the registration says was reported: the latch state after every recorded decision,
    selecting between the branches, through the deal's own map. This is verbatim what
    `LatchedBoundarySet.branch_deltas` computes as its baseline."""
    prefix = [torch.zeros_like(bset.fired[0])]
    for flag in bset.fired:
        prefix.append(prefix[-1] | flag)
    return bset.to_mtm(torch.where(
        torch.stack(prefix)[bset.obs_before], bset.triggered, bset.untriggered))


# ------------------------------------------------- the branch has to land where the value landed

@pytest.mark.parametrize('report_currency', ['USD', 'EUR'])
@pytest.mark.parametrize('interpolated', [False, True])
def test_the_registered_barrier_branches_reproduce_the_reported_value(interpolated,
                                                                      report_currency):
    """The branches, selected by the recorded flags, must be the deal's reported profile EXACTLY.

    Both defects this pins are invisible in a forward pass - a boundary correction is worth zero
    there - so only a gradient moves, and only by a factor that reads as Monte Carlo error.

    GRID. The pricer builds its profile over `deal_time_grid`; `Deal.calculate` puts it on the MTM
    grid with `gather_interp_matrix`, which INSERTS rows in the middle wherever another deal
    contributes an mtm date inside this deal's life. Padding the tail instead lands deal row i on
    mtm row i, which is the same row only while no such date exists - hence the `interpolated`
    parameter, and hence a fixture with a second deal in it. Measured on this one: mtm grid
    [0 30 60 90 92 120 137 150 180 183 210 240 270 273 300 330 360 365], the barrier's own rows
    everything but 137, so every branch value from day 150 on sat one row early and the expiry row
    was left as the zero pad.

    UNITS. `fx_rep` is `shared.one` only when the payoff and reporting currencies match; otherwise
    it is a simulated (T, B) cross, so a branch registered without it is a delta in the wrong
    currency AND leaves the fx factor's own flux off the tape. Measured: the branch was exactly
    0.8x the reported value, the USD/EUR spot.

    torch.equal, not allclose: both are exact identities."""
    seen = _spy_registration(
        'BARR1', deal=DISCRETE_BARRIER, batch=256, mcmc=64, report_currency=report_currency,
        extra_deals=(INTERPOLATING_DEAL,) if interpolated else ())
    bset, = seen['sets']
    selected = _latched_reported(bset)
    assert torch.equal(selected, seen['reported']), (
        'the registered branches do not reconstruct the reported deal value - the counterfactual '
        'is being scored on the wrong grid or in the wrong currency; max |d| '
        f'{float((selected - seen["reported"]).abs().max()):.6g} against a reported |mean| of '
        f'{float(seen["reported"].abs().mean()):.6g}')


@pytest.mark.parametrize('report_currency', ['USD', 'EUR'])
def test_the_autocall_row_delta_lands_where_the_reported_value_did(report_currency):
    """The same defect on the ROW-local shape, where it presents differently: an autocall's jump is
    a single row, so a mis-mapped row index puts the whole jump on the WRONG DATE rather than
    rescaling it - and on an interpolated grid the jump has to SPLIT across the two mtm rows its
    deal row sits between, which no integer row index can express at all.

    The oracle is the deal's OWN reported profile, captured independently of the registration.
    Comparing a delta against another use of the same map proves nothing: both move together, and
    a first attempt at this test passed the padding mutant for exactly that reason."""
    seen = _spy_registration('AC1', deal=AUTOCALL, batch=256, mcmc=64,
                             report_currency=report_currency,
                             extra_deals=(INTERPOLATING_DEAL,))
    bset, = seen['sets']
    assert torch.equal(bset.to_mtm(bset.reported), seen['reported']), (
        'the registered baseline is not the deal value that was reported - every delta measured '
        'against it is on the wrong grid or in the wrong currency; max |d| '
        f'{float((bset.to_mtm(bset.reported) - seen["reported"]).abs().max()):.6g}')
    # The jump for a decision recorded on pricer row r belongs on the DATE that row falls on, and
    # on the mtm rows either side of it that interpolate through it - never anywhere else. Dates,
    # not indices: an index equal to the registered one is what the padding form produced, and it
    # was the wrong date. `assert bset.rows` because a fixture where nothing fires gates nothing.
    assert bset.rows, 'no autocall decision was recorded - this fixture cannot see the defect'
    mtm_dates, deal_dates = seen['mtm_dates'], seen['deal_dates']
    for (gap, on, off), row in zip(bset.branch_deltas(), bset.rows):
        hit = mtm_dates[(on - off).abs().sum(dim=1).gt(0).cpu().numpy()]
        assert len(hit) and hit.min() < deal_dates[row] + 1 and hit.max() > deal_dates[row] - 1, (
            f'the jump recorded on pricer row {row} (day {deal_dates[row]:.0f}) landed on mtm days '
            f'{hit.tolist()} - it is on the wrong date, not merely the wrong index')
        assert deal_dates[row] in hit, (
            f'the coupon date itself (day {deal_dates[row]:.0f}) carries none of its own jump')


def test_a_registration_does_not_hold_the_calculation_state():
    """A boundary set outlives the pricing call - it is held until the batch's backward pass - so
    what its grid map closes over is a memory contract, and no reported number can show you when
    it is wrong.

    Closing over `shared` makes a cycle: shared -> boundary_sets -> the closure -> shared.
    Refcounting cannot break it, so the calculation state and everything reachable from it
    survives the run and waits on the cyclic collector. MEASURED on the collateralised barrier at
    batch 1024: 19.6 GB still resident after ONE run where the same run had held 32 MiB before,
    and the next run OOMed. The suite only ever saw it as some other test failing, in whichever
    file happened to run last, which is why it is gated at the cause and not at a byte count.

    `interp_to_mtm_grid` never used `shared` either, so the fix removed a parameter rather than
    working around one."""
    seen = _spy_registration('BARR1', deal=DISCRETE_BARRIER, batch=256, mcmc=64,
                             collateralised=True)
    bset, = seen['sets']
    held = [cell.cell_contents for cell in (bset.to_mtm.__closure__ or ())]
    assert held, 'the grid map closes over nothing at all - this is not reading the real map'
    assert not any(isinstance(x, utils.Calculation_State) for x in held), (
        'the grid map closes over the calculation state, which is a reference CYCLE through '
        'shared.boundary_sets - the whole calculation survives the run')
    for name in ('untriggered', 'triggered'):
        assert not getattr(bset, name).requires_grad, f'{name} carries a graph'


def test_the_grid_map_detaches_the_fx_cross_it_captures():
    """The other half, and a unit test because no fixture in this file can reach it: `fx_rep` is a
    SIMULATED (T, B) cross only when the payoff and reporting currencies differ AND the fx factor
    is stochastic; everywhere here it resolves to a static rate, which carries no graph, so an
    integration gate cannot tell a detached capture from a live one. It could not - the mutant
    that keeps the graph survived the EUR-reported fixture.

    Live, it would pin the deal's whole tape for as long as the set exists. Branch values are
    coefficients: the rule that they stay detached is a memory contract as much as a correctness
    one, and `.detach()` on the way OUT of the map is too late for the thing it captured."""
    import riskflow.pricing as pricing
    cross = torch.ones(3, 2, dtype=bb.DTYPE, requires_grad=True)
    to_mtm = pricing.deal_to_mtm_grid(None, None, cross)
    captured = [c.cell_contents for c in to_mtm.__closure__ if torch.is_tensor(c.cell_contents)]
    assert captured, 'the map captured no tensor at all, so the fx cross went somewhere else'
    assert not any(t.requires_grad for t in captured), (
        'the grid map captured the fx cross with its graph attached, pinning the deal tape for '
        'the life of the registration')


def test_the_netting_set_a_registration_sits_under_is_the_one_that_scores_it():
    """`boundary_sets` accumulates from every deal in every netting set, so a single slot on
    `shared` cannot say which set a given registration belonged to. It was one slot: an
    UNCOLLATERALISED set's barrier was pushed through a collateralised set's gross-to-net chain,
    and with two collateralised sets the last one to run spoke for both.

    Both failure modes need a portfolio with more than one netting set in it, which is why no
    single-set fixture could see either. Measured on this one: before, one chain object served all
    three registrations; after, the uncollateralised set's barrier carries None and the two
    collateralised sets carry two DIFFERENT chains."""
    def portfolio(c):
        def netting(ref, collateralised, barrier_ref):
            return {'Instrument': construct_instrument(
                dict(NETTING, Reference=ref, Collateralized=collateralised), {}),
                'Children': [{'Instrument': construct_instrument(
                    dict(DISCRETE_BARRIER, Reference=barrier_ref), {})}]}
        return [netting('NS_UNCOL', 'False', 'B_UNCOL'),
                netting('NS_COL_A', 'True', 'B_COL_A'),
                netting('NS_COL_B', 'True', 'B_COL_B')]

    import riskflow.calculation as C
    seen = {}
    original = C.pricer_boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        seen['chains'] = [x.net_from_gross for x in shared.boundary_sets
                          if isinstance(x, utils.BoundarySet)]
        return original(shared, objective, reported_mtm, bandwidth)

    C.pricer_boundary_correction = probe
    try:
        _run(DISCRETE_BARRIER, gradient=True, batch=256, mcmc=64, children=portfolio)
    finally:
        C.pricer_boundary_correction = original

    chains = seen.get('chains')
    assert chains is not None and len(chains) == 3, f'expected three registrations, got {chains}'
    uncollateralised, col_a, col_b = chains
    assert uncollateralised is None, (
        'a barrier in an UNCOLLATERALISED netting set was handed a gross-to-net chain, so its '
        'delta is scored through a collateral scan that never touched it')
    assert col_a is not None and col_b is not None, 'a collateralised set published no chain'
    assert col_a is not col_b, (
        'both collateralised sets are scored through the SAME chain - the last one to run is '
        'speaking for the other one as well')


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


def test_collateralised_barrier_latch_gradient_matches_bump_and_reprice():
    """The same defect with collateral in the way, which is the harder half: a gross-mtm delta
    reaches the net through Vte AND through the balance the collateral scan produces, so a fix
    that only handles the additive path will pass the test above and fail this one - which is
    exactly what happened, and what sent the gross-to-net chain into post_process.

    This was an xfail at 6.48%, blamed on path count. It was not path count. A COLLATERALISED
    netting set puts its own margin-call schedule on the mtm grid - measured here, 86 mtm rows
    against the barrier's own 51, with 81 interpolated - so this is the fixture in the repo where
    the branch profile was WORST mis-mapped, and the uncollateralised one above (17 rows, 17 deal
    rows, no interpolation) could not see it. Putting the branches through the deal's own grid map
    moved the AAD from +0.000904075 to +0.000941522 against an unchanged oracle of +0.000962679:
    6.48% to 2.25%, on a ladder whose three CRN readings are bit-identical between the two runs.

    Still the noisiest of these gates - the exposure is almost entirely collateralised away, so the
    number is ~16x smaller than the uncollateralised one and the ladder's flatness is 5.25%."""
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
        bset = next((x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)), None)
        if bset is not None and bset.net_from_gross is not None:
            with torch.no_grad():
                # a zero delta on the MTM grid - which is the grid the chain consumes, and which
                # only `to_mtm` knows how to reach from the pricer's own rows
                seen['diff'] = float(
                    (bset.net_from_gross(bset.to_mtm(torch.zeros_like(bset.untriggered)))
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
