"""Heston-Nandi spot model wired into the two remaining one-step-survival (OSS) Monte Carlo
pricers - the DISCRETE BARRIER option (riskflow/pricing.py pv_discrete_barrier_option) and the
AUTOCALL (pv_MC_AutoCallSwap) - extending the committed TARF pattern (commit bf449d1).

RESOLUTION (settled design). Opt-in per deal TYPE via the Valuation Configuration switch
``SpotModel`` (default 'None' = GBM/surface). There is NO deal field: the params factor is found
by NAMING CONVENTION off the equity underlying the deal already references -
``<SpotModel>ModelParameters.<equity>`` - and pulled in as a dependent STATIC factor by the
EquityPrice conditional in config.py when the switch is on. So the HN path is activated by
(a) the switch AND (b) the HestonNandiModelParameters.<equity> factor in the market data.
  * switch off / absent          -> GBM, byte-identical, regardless of what factors exist
  * switch on + factor present   -> HN
  * switch on + factor MISSING   -> loud failure (deal skipped with ERROR naming the factor)
  * unknown SpotModel value       -> ValueError naming the value + accepted set (deal skipped)

SHARED PER-STEP HELPER + MUTATION KILL MATRIX. All three pricers route every daily HN step -
the unmonitored sub-steps AND the survival-truncated final step of each interval - through the
single ``pricing.hn_daily_advance`` (the ONLY copy of the h-recursion
``h = Omega + Beta*h + Alpha*(z - Gamma_Star*sqrt(h))**2``). Four one-line mutants of it were
applied to the shared helper (source edit, run, revert BY HAND) and each is killed by BOTH
recursion closed-form gates below (measured reldiff vs the hn_garch closed form; correct code
passes at <=2.2e-3, gate tol 5e-3):

    mutant                         barrier_ko_never   autocall_single_coupon
    (correct)                          2.1e-3              1.3e-3
    (a) leverage-sign  Z -> Z+ga*sh    2.0e-2              2.2e-1
    (b) frozen h       h = h           1.5e-1              7.6e-2
    (c) 10x alpha                      1.0 (var explodes)  1.0
    (d) dropped beta   -beta*h         5.0e-1              4.9e-1

The autocall single-coupon gate prices a 5%-OTM tail probability Q(S_T>K), which is acutely
skew-sensitive, so it kills the leverage-sign flip at 2.2e-1 (no brute-force reference needed -
the aggregate-variance-normal rejection is the TARF's; here the closed form IS hn_garch).
"""
import io
import logging
import os
import sys

# reference-riskflow shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
import torch

import riskflow
from riskflow import hn_garch, run_baseval, utils
from riskflow.config import Config
from riskflow.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
SPOT = 100.0
STRIKE = 100.0
SIGMA = 0.25
N = 10.0

# strong-leverage GARCH with a term structure (H0 != stationary) - the recursion is genuinely
# exercised AND the leverage cross-term is engaged (kills the leverage-sign mutant via the skew).
_STRONG = hn_garch.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=252.0)
_STRONG_H0 = 1.6 * float(_STRONG.stationary_var)
STRONG = {'Omega': float(_STRONG.omega), 'Alpha': float(_STRONG.alpha), 'Beta': float(_STRONG.beta),
          'Gamma_Star': float(_STRONG.gamma), 'H0': _STRONG_H0}
# the degenerate GARCH that collapses to constant-variance GBM at a 1-day step: no ARCH, no
# leverage, no GARCH memory, so h == Omega == H0 == sigma^2 * (1 day) for all t.
_H0_DAY = SIGMA ** 2 / 365.0
DEGEN = {'Omega': _H0_DAY, 'Alpha': 0.0, 'Beta': 0.0, 'Gamma_Star': 0.0, 'H0': _H0_DAY}


def _flat_vol(sig):
    return utils.Curve([], [[m, t, sig] for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)])


def _price_factors(sig, hn_params, r, q):
    pf = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, r], [5.0, r]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, q], [5.0, q]])},
        'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': _flat_vol(sig)},
    }
    if hn_params is not None:
        pf['HestonNandiModelParameters.EQ'] = dict(hn_params, Property_Aliases=None)
    return pf


def _cfg(field, ref, sig=SIGMA, hn_params=None, r=0.0, q=0.0, spot_model='auto'):
    """spot_model='auto' turns the switch on iff hn_params is supplied; pass an explicit string
    (incl. 'None') to decouple the switch from factor presence for the resolution-semantics tests."""
    if spot_model == 'auto':
        spot_model = 'HestonNandi' if hn_params is not None else None
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = _price_factors(sig, hn_params, r, q)
    c.params['Price Models'] = {}
    val = {field['Object']: {'SpotModel': spot_model}} if spot_model else {}
    c.params['Valuation Configuration'] = val
    inst = construct_instrument(field, val)
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': inst}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c, ref


def _run(cfg_ref, seed=1, sims=1 << 14):
    cfg, ref = cfg_ref
    calc, out = run_baseval(cfg, overrides={'MCMC_Simulations': sims, 'Random_Seed': seed})
    df = out['Results']['mtm']
    rows = df[df['Reference'] == ref]
    mtm = float(rows['Value'].iloc[0]) if len(rows) else None
    return mtm, calc


def _mtm(cfg_ref, seed=1, sims=1 << 14):
    return _run(cfg_ref, seed, sims)[0]


# ======================================================================================
# builders
# ======================================================================================

def _barrier_field(btype, bprice, bdays, horizon):
    bdates = [BASE + pd.Timedelta(days=d) for d in bdays]
    return {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1',
        'Currency': 'USD', 'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ',
        'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
        'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': STRIKE,
        'Expiry_Date': BASE + pd.Timedelta(days=horizon), 'Units': N,
        'Barrier_Type': btype, 'Barrier_Price': bprice, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [[d, bprice] for d in bdates],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }


def _barrier_cfg(btype, bprice, bdays, horizon, **kw):
    return _cfg(_barrier_field(btype, bprice, bdays, horizon), 'BARR1', **kw)


def _autocall_field(coupon_days, thresholds, coupons, horizon):
    cdates = [BASE + pd.Timedelta(days=d) for d in coupon_days]
    return {
        'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC1',
        'Currency': 'USD', 'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ',
        'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
        'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': STRIKE,
        'Expiry_Date': BASE + pd.Timedelta(days=horizon), 'Units': N,
        'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'Option_Style': 'European',
        'Barrier': 0.0, 'Payoff_Type': None,
        'Price_Fixing': [[d, 0.0] for d in cdates],
        'Autocall_Coupons': [[d, c] for d, c in zip(cdates, coupons)],
        'Autocall_Thresholds': [[d, t] for d, t in zip(cdates, thresholds)],
        'Barrier_Dates': [], 'Autocall_Floating': [],
    }


def _autocall_cfg(coupon_days, thresholds, coupons, horizon, **kw):
    return _cfg(_autocall_field(coupon_days, thresholds, coupons, horizon), 'AC1', **kw)


def _factor_dep(calc):
    return calc.netting_sets.dependencies[0].Factor_dep


def _capture_errors(fn):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    prev = root.level
    root.setLevel(logging.ERROR)
    try:
        fn()
    finally:
        root.removeHandler(handler)
        root.setLevel(prev)
    return stream.getvalue()


def test_uses_repo_under_test():
    assert riskflow.__file__ == os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'riskflow', '__init__.py')


# ======================================================================================
# resolution semantics (both deals): the SpotModel switch + naming-convention factor
# ======================================================================================

BDAYS30 = list(range(1, 31))  # 30 daily observation dates => 30 one-day HN intervals


@pytest.mark.parametrize('build', [
    lambda **kw: _barrier_cfg('Down_And_Out', 1.0, BDAYS30, 30, **kw),
    lambda **kw: _autocall_cfg([30], [1.05], [0.05], 30, **kw),
])
def test_switch_routes_the_factor_dependency(build):
    """Switch off/absent -> GBM factor_dep (Volatility, no HN_Params). Switch on + factor present
    -> HN_Params carrying the five GARCH scalars, resolved off the factor's own current_value()
    keys; the GBM Volatility path is left in place (unused)."""
    _, gbm = _run(build(), sims=64)
    assert 'HN_Params' not in _factor_dep(gbm) and 'Volatility' in _factor_dep(gbm)

    _, hn = _run(build(hn_params=DEGEN), sims=64)
    dep = _factor_dep(hn)
    assert 'Volatility' in dep and 'HN_Params' in dep
    assert [x.name[-1] for x in dep['HN_Params'][0][utils.FACTOR_INDEX_Offset]] == \
        ['Omega', 'Alpha', 'Beta', 'Gamma_Star', 'H0']


@pytest.mark.parametrize('build,name', [
    (lambda **kw: _barrier_cfg('Down_And_Out', 1.0, BDAYS30, 30, **kw), 'EquityBarrierOption'),
    (lambda **kw: _autocall_cfg([30], [1.05], [0.05], 30, **kw), 'QEDI_CustomAutoCallSwap'),
])
def test_unknown_spot_model_fails_loudly(build, name):
    """A typo'd SpotModel raises ValueError naming the value and the accepted set; the engine's
    dependency loop logs+skips the raising deal (never a silent GBM fallback)."""
    txt = _capture_errors(lambda: run_baseval(
        build(hn_params=STRONG, spot_model='Heston_Nandi')[0],
        overrides={'MCMC_Simulations': 64, 'Random_Seed': 1}))
    assert 'Heston_Nandi' in txt and "('None', 'HestonNandi')" in txt


@pytest.mark.parametrize('build', [
    lambda **kw: _barrier_cfg('Down_And_Out', 1.0, BDAYS30, 30, **kw),
    lambda **kw: _autocall_cfg([30], [1.05], [0.05], 30, **kw),
])
def test_switch_on_but_factor_missing_fails_loud(build):
    """SpotModel on but the params factor absent from the market data -> the deal is SKIPPED with an
    ERROR naming the expected factor (never a silent GBM price)."""
    txt = _capture_errors(lambda: run_baseval(
        build(spot_model='HestonNandi')[0],  # switch on, no HestonNandiModelParameters.EQ factor
        overrides={'MCMC_Simulations': 64, 'Random_Seed': 1}))
    assert 'HestonNandiModelParameters.EQ' in txt


@pytest.mark.parametrize('build', [
    lambda **kw: _barrier_cfg('Down_And_Out', 92.0, BDAYS30, 30, **kw),
    lambda **kw: _autocall_cfg([5, 10, 15], [1.02, 1.02, 1.02], [0.05, 0.05, 0.05], 15, **kw),
])
def test_switch_off_with_factor_present_is_gbm(build):
    """Factor present but SpotModel off (default) -> byte-identical to the no-factor GBM price
    (an unused static factor loaded into t_Static_Buffer must not perturb the GBM path)."""
    gbm = _mtm(build(), sims=1 << 12)
    off = _mtm(build(hn_params=STRONG, spot_model='None'), sims=1 << 12)
    assert off == gbm


# ======================================================================================
# BIT-IDENTITY: degenerate HN reproduces GBM to float64 epsilon (n_sub=1 per daily interval)
# ======================================================================================

@pytest.mark.parametrize('btype,bprice', [
    ('Down_And_Out', 90.0), ('Up_And_In', 110.0), ('Down_And_In', 90.0), ('Up_And_Out', 110.0)])
@pytest.mark.parametrize('seed', [1, 7])
def test_barrier_bit_identity_degenerate_hn_reproduces_gbm(btype, bprice, seed):
    """Alpha=Gamma_Star=Beta=0, Omega=H0=sigma^2*(1 day) and daily (n_sub=1) observation dates make
    the HN branch and the GBM branch the SAME computation; they agree to float64 machine epsilon
    (frequently exactly). At n_sub=1 there are zero unconditional sub-steps, so the branches consume
    identical RNG - which is what makes the bit-identity gate meaningful."""
    bdays = list(range(1, 21))
    gbm = _mtm(_barrier_cfg(btype, bprice, bdays, 20), seed=seed, sims=1 << 13)
    hn = _mtm(_barrier_cfg(btype, bprice, bdays, 20, hn_params=DEGEN), seed=seed, sims=1 << 13)
    assert hn == pytest.approx(gbm, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize('thr', [1.05, 0.98])  # 0.98 knocks readily (exercises p<1); 1.05 rarely
@pytest.mark.parametrize('seed', [1, 7])
def test_autocall_bit_identity_degenerate_hn_reproduces_gbm(thr, seed):
    """Daily coupons (n_sub=1) with the degenerate GARCH reproduce the GBM autocall to float64
    epsilon (identical RNG at n_sub=1, per the barrier note)."""
    days = [1, 2, 3, 4, 5]
    gbm = _mtm(_autocall_cfg(days, [thr] * 5, [0.05] * 5, 5), seed=seed, sims=1 << 13)
    hn = _mtm(_autocall_cfg(days, [thr] * 5, [0.05] * 5, 5, hn_params=DEGEN), seed=seed, sims=1 << 13)
    assert hn == pytest.approx(gbm, rel=1e-11, abs=1e-11)


# ======================================================================================
# CLOSED-FORM GATES (non-degenerate strong-leverage params) - also the mutation kill matrix
# ======================================================================================

def _hn_call_ref(n_total, h0):
    return float(hn_garch.hn_call(
        SPOT, STRIKE, hn_garch.HNParams(
            _STRONG.omega, _STRONG.alpha, _STRONG.beta, _STRONG.gamma, 0.0).as_tensors(),
        n_total, h0)) * N


def test_barrier_ko_never_knock_matches_hn_call():
    """KILL MATRIX GATE 1 (spot recursion). A knock-OUT call whose barrier is far out of the way
    (never knocks) pays the survivors the vanilla payoff, so its price is the HN vanilla call under
    the pricer's 30-step daily recursion == hn_garch.hn_call (zero carry). A wrong final-step
    h-update desynchronises the whole 30-step variance path and breaks the match (tol 5e-3; correct
    ~2.1e-3, every mutant >= 2.0e-2)."""
    price = _mtm(_barrier_cfg('Down_And_Out', 1.0, BDAYS30, 30, hn_params=STRONG), seed=3, sims=1 << 17)
    assert price == pytest.approx(_hn_call_ref(30, _STRONG_H0), rel=5e-3)


def test_barrier_ki_always_knock_matches_hn_call_closed_form():
    """KI leg via in-out parity. An up-and-IN call whose barrier sits well below spot knocks in on
    the first observation (survival ~ 0), so the price collapses to the analytic parity vanilla -
    which under HN is the hn_garch CLOSED FORM (NOT a normal at aggregate variance). This pins that
    closed-form leg deterministically (matches to ~1e-12, not just MC error)."""
    price = _mtm(_barrier_cfg('Up_And_In', 50.0, BDAYS30, 30, hn_params=STRONG), seed=3, sims=1 << 15)
    assert price == pytest.approx(_hn_call_ref(30, _STRONG_H0), rel=1e-6)


def test_autocall_single_coupon_matches_hn_cdf():
    """KILL MATRIX GATE 2 (survival construction + recursion). A single-coupon autocall with no
    floating/barrier pays coupon * P(S_T > K) discounted; with zero rates that is
    N * coupon * (1 - hn_cdf_logret(log(K/S))) under the pricer's daily HN recursion + final-step
    survival truncation. The 5%-OTM knock probability is a skew-sensitive tail, so this gate kills
    every h-recursion mutant - incl. the leverage-sign flip at 2.2e-1 (tol 5e-3, correct ~1.3e-3)."""
    horizon, coup, thr = 30, 0.05, 1.05
    n_total = max(round(horizon / 365 * 252), 1)
    q_below = float(hn_garch.hn_cdf_logret(
        torch.log(torch.tensor(thr * STRIKE / SPOT)),
        hn_garch.HNParams(_STRONG.omega, _STRONG.alpha, _STRONG.beta, _STRONG.gamma, 0.0).as_tensors(),
        n_total, _STRONG_H0))
    ref = N * coup * (1.0 - q_below)
    price = _mtm(_autocall_cfg([horizon], [thr], [coup], horizon, hn_params=STRONG), seed=3, sims=1 << 17)
    assert price == pytest.approx(ref, rel=5e-3)


def test_non_degenerate_hn_moves_the_price_off_gbm():
    """Sanity: the HN branch is really taken - a strong-leverage GARCH price differs materially from
    the flat-vol GBM barrier price (a bypassed/constant-h implementation would land on GBM)."""
    horizon = 30
    gbm = _mtm(_barrier_cfg('Down_And_Out', 1.0, BDAYS30, horizon, sig=0.30), sims=1 << 15)
    hn = _mtm(_barrier_cfg('Down_And_Out', 1.0, BDAYS30, horizon, hn_params=STRONG), seed=3, sims=1 << 15)
    assert abs(hn - gbm) / abs(gbm) > 1e-2
