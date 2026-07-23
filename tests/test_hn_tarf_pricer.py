"""Heston-Nandi spot model wired into the TARF one-step-survival Monte Carlo pricer
(riskflow/pricing.py sim_spot_tarf), opt-in per deal via the ``SpotModel='HestonNandi'``
Valuation Configuration switch (riskflow/instruments.py FXTARFOptionDeal).

Everything is driven end-to-end through the framework's Base_Revaluation pricer (run_baseval,
float64) on a self-built FX TARF - no monkey-patching of the pricer, the HN parameters ride the
AAD graph out of t_Static_Buffer exactly like a real bootstrapped factor.

THE GATES
  * options switch      - absent/'None' resolves the GBM factor_dep (Volatility, no HN_Params);
                          'HestonNandi' resolves HN_Params (the five GARCH scalars) and leaves the
                          Volatility path in place but unused. A non-degenerate GARCH price also
                          differs materially from the GBM price (the branch is really taken).
  * bit-identity        - Alpha=Gamma_Star=Beta=0, Omega=H0=sigma^2*dt and a fixing schedule with
                          exactly ONE daily sub-step per fixing (n_sub=1) makes the HN branch
                          reproduce the GBM branch. It is NOT torch.equal - see the test docstring.
  * closed form         - a single-fixing, never-knocking HN TARF == N1 * hn_garch.hn_call (the
                          validated 98-test semi-analytic HN pricer) to Monte-Carlo error, over two
                          persistences (so the daily h-recursion is exercised, not bypassed).
  * dynamics engaged    - the GARCH term structure / leverage move the price off a flat-vol GBM at
                          the SAME long-run vol, i.e. h actually recurses.

RNG / ANTITHETIC (stated once). A fixing spans n_sub daily HN steps. The final (monitored) step
keeps the existing Sobol+antithetic truncated draw (u then 1-u) - that is what OSS's variance
reduction targets. The n_sub-1 unconditional sub-steps are drawn from the REGULAR normal stream
(torch.randn) and made antithetic by NEGATING the normal on the paired half (z, -z), matching the
u<->1-u pairing on the truncated draw; without that negation the antithetic variance reduction is
silently lost. At n_sub=1 there are zero unconditional sub-steps, so the HN and GBM branches consume
identical RNG - which is what makes the bit-identity gate meaningful.
"""
import logging
import os
import sys

# reference-riskflow shadow-import guard (see MEMORY: `import riskflow` can resolve to an old
# ~/VSCode snapshot); pin the package under test to THIS repo before importing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import riskflow
from riskflow import hn_garch, run_baseval, utils
from riskflow.config import Config
from riskflow.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
SPOT = 0.65
STRIKE = 0.65
SIGMA = 0.12
DT_DAYS = 30
N1 = 1_000_000.0


def _flat_vol_surface(sigma):
    return utils.Curve([], [[m, t, sigma] for m in (0.5, 1.0, 1.5) for t in (0.02, 2.0)])


def _price_factors(sigma, hn_params, r_dom=0.0, r_for=0.0):
    # r_dom/r_for are flat annualised USD/AUD rates. Zero (default) => driftless FX forward =>
    # hn_garch.HNParams(..., r=0) is the reference law. With r_for=0 the carry (r_dom-r_for) and
    # the USD discount coincide, so a single hn_call r = r_dom/365 (per ACT/365 day) pins BOTH the
    # forward drift and the discount (see test_never_knock_daily_hn_matches_summed_closed_form).
    pf = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'FxRate.AUD': {'Domestic_Currency': 'USD', 'Interest_Rate': 'AUD', 'Priority': 1, 'Spot': SPOT},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, r_dom], [5.0, r_dom]])},
        'InterestRate.AUD': {'Currency': 'AUD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, r_for], [5.0, r_for]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'FXVol.AUD.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': _flat_vol_surface(sigma)},
    }
    if hn_params is not None:
        pf['HestonNandiModelParameters.AUD'] = dict(hn_params, Property_Aliases=None)
    return pf


def _tarf(target, fix_days):
    # No deal field: HN is activated purely by the SpotModel switch (in _cfg's Valuation
    # Configuration) + the HestonNandiModelParameters.AUD factor (in _price_factors), resolved by
    # naming convention off Underlying_Currency='AUD'. Matches the equity OSS deals (test_hn_oss_pricers).
    fix_dates = [BASE + pd.Timedelta(days=d) for d in fix_days]
    return {
        'Object': 'FXTARFOptionDeal', 'Reference': 'TARF1',
        'Currency': 'USD', 'Underlying_Currency': 'AUD', 'Discount_Rate': 'USD',
        'FX_Volatility': 'AUD.USD', 'Buy_Sell': 'Buy', 'Expiry_Date': fix_dates[-1],
        'Underlying_Amount': N1, 'Option_Type': 'Call', 'Strike_Price': STRIKE,
        'Settlement_Style': 'Physical', 'Option_Style': 'European',
        'InvertedTarget': False, 'LeverageNotional': 0.0, 'TargetAdjustment': '',
        'TargetLevel': target, 'TARF_ExpiryDates': [[d, d, None] for d in fix_dates],
    }


def _cfg(hn, target, fix_days, sigma=SIGMA, hn_params=None, steps_per_year=None, r_dom=0.0, r_for=0.0,
         spot_model='HestonNandi'):
    cfg = Config()
    cfg.params['System Parameters']['Base_Currency'] = 'USD'
    cfg.params['System Parameters']['Base_Date'] = BASE
    cfg.params['Price Factors'] = _price_factors(sigma, hn_params if hn else None, r_dom, r_for)
    cfg.params['Price Models'] = {}
    val = {}
    if hn:
        val['FXTARFOptionDeal'] = {'SpotModel': spot_model}
        if steps_per_year is not None:
            val['FXTARFOptionDeal']['Steps_Per_Year'] = steps_per_year
    cfg.params['Valuation Configuration'] = val
    inst = construct_instrument(_tarf(target, fix_days), val)
    cfg.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                 'Deals': {'Children': [{'Instrument': inst}]},
                 'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return cfg


def _run(cfg, seed, sims):
    calc, out = run_baseval(cfg, overrides={'MCMC_Simulations': sims, 'Random_Seed': seed})
    df = out['Results']['mtm']
    mtm = float(df[df['Reference'] == 'TARF1']['Value'].iloc[0])
    return mtm, calc


def _mtm(cfg, seed=1, sims=1 << 14):
    return _run(cfg, seed, sims)[0]


# monthly fixings all exactly DT_DAYS apart => every interval has dt = DT_DAYS/365 (ACT/365)
_MONTHLY = [DT_DAYS * (i + 1) for i in range(3)]
# steps_per_year chosen so round(dt * spy) == 1 for every fixing => one HN sub-step per fixing
_SPY_ONE_SUBSTEP = 365.0 / DT_DAYS
# the degenerate GARCH that collapses to constant-variance GBM: no ARCH, no leverage, no GARCH
# memory, so h_{t+1} = Omega = H0 for all t, with H0 = sigma^2 * dt matching the GBM step variance
_H0_DEGEN = SIGMA ** 2 * (DT_DAYS / 365.0)
_DEGENERATE = {'Omega': _H0_DEGEN, 'Alpha': 0.0, 'Beta': 0.0, 'Gamma_Star': 0.0, 'H0': _H0_DEGEN}


def _cum_days(fix_days, spy=252.0):
    """Cumulative HN daily sub-step counts at each fixing - the pricer's n_sub per interval is
    max(round(dt*spy), 1) with dt the ACT/365 interval fraction, so this must mirror it exactly."""
    prev, cum = 0, []
    for d in fix_days:
        cum.append((cum[-1] if cum else 0) + max(int(round((d - prev) / 365.0 * spy)), 1))
        prev = d
    return cum


def _real_garch(persistence=0.96):
    p = hn_garch.hn_params_from_targets(
        ann_vol=0.28, persistence=persistence, gamma=250.0, leverage_share=0.5, steps_per_year=252.0)
    return {'Omega': float(p.omega), 'Alpha': float(p.alpha), 'Beta': float(p.beta),
            'Gamma_Star': float(p.gamma), 'H0': 1.2 * float(p.stationary_var)}, p


def test_uses_repo_under_test():
    assert riskflow.__file__ == os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'riskflow', '__init__.py')


# ======================================================================================
# the opt-in switch
# ======================================================================================

def test_options_switch_routes_the_factor_dependency():
    """SpotModel absent -> GBM factor_dep (Volatility, no HN_Params). SpotModel='HestonNandi' ->
    HN_Params carrying the five GARCH scalars as static sub-factors, resolved generically off the
    factor's own current_value() keys; the GBM Volatility path is left in place (unused)."""
    _, gbm_calc = _run(_cfg(False, 0.10, _MONTHLY), seed=1, sims=64)
    gbm_dep = gbm_calc.netting_sets.dependencies[0].Factor_dep
    assert 'HN_Params' not in gbm_dep and 'Volatility' in gbm_dep

    _, hn_calc = _run(_cfg(True, 0.10, _MONTHLY, hn_params=_DEGENERATE,
                           steps_per_year=_SPY_ONE_SUBSTEP), seed=1, sims=64)
    hn_dep = hn_calc.netting_sets.dependencies[0].Factor_dep
    assert 'Volatility' in hn_dep  # the GBM path is untouched, just not consumed
    assert 'HN_Params' in hn_dep
    param_keys = [x.name[-1] for x in hn_dep['HN_Params'][0][utils.FACTOR_INDEX_Offset]]
    assert param_keys == ['Omega', 'Alpha', 'Beta', 'Gamma_Star', 'H0']


def test_unknown_spot_model_fails_loudly(caplog):
    """A typo'd SpotModel (e.g. 'Heston_Nandi') must raise ValueError naming the offending value and
    the accepted set, NOT silently fall back to GBM with the HN factor present. The calc engine's
    dependency loop logs+skips the raising deal (framework behaviour), so we assert the logged error
    carries both. Only 'None' (default/absent) and 'HestonNandi' are accepted."""
    cfg = _cfg(True, 0.10, _MONTHLY, hn_params=_DEGENERATE, steps_per_year=_SPY_ONE_SUBSTEP,
               spot_model='Heston_Nandi')
    with caplog.at_level(logging.ERROR):
        run_baseval(cfg, overrides={'MCMC_Simulations': 64, 'Random_Seed': 1})
    assert 'Heston_Nandi' in caplog.text and "('None', 'HestonNandi')" in caplog.text


def test_non_degenerate_hn_actually_changes_the_price():
    """The HN branch is really taken: a real GARCH price differs materially from the GBM price."""
    hn_params, _ = _real_garch()
    gbm = _mtm(_cfg(False, 0.10, _MONTHLY, sigma=0.28), sims=1 << 15)
    hn = _mtm(_cfg(True, 0.10, _MONTHLY, hn_params=hn_params, steps_per_year=252.0), sims=1 << 15)
    assert abs(hn - gbm) / abs(gbm) > 1e-2


# ======================================================================================
# BIT-IDENTITY GATE
# ======================================================================================

@pytest.mark.parametrize('target', [0.10, 0.02])  # 0.02 knocks (exercises p<1, the KO term); 0.10 rarely
@pytest.mark.parametrize('seed', [1, 7])
def test_bit_identity_degenerate_hn_reproduces_gbm(seed, target):
    """With Alpha=Gamma_Star=Beta=0, Omega=H0=sigma^2*dt and n_sub=1, the HN branch and the GBM
    branch are the SAME computation and reproduce each other to float64 machine epsilon (frequently
    exactly, absdiff==0).

    Why not asserted with torch.equal / == : two reasons, both precise.
      (1) At n_sub=1 the branches consume IDENTICAL RNG (no unconditional sub-steps), but assemble
          the per-step drift and vol by different float operation sequences - sqrt(sigma^2*dt) vs
          sigma*sqrt(dt), and fwd_carry*dt - 0.5*H0 vs (fwd_carry - 0.5*sigma^2)*dt - which differ
          in the last ULP by floating-point NON-ASSOCIATIVITY and propagate through norm_icdf/exp.
      (2) For any fixing spanning n_sub>1 daily steps, the HN branch draws n_sub-1 extra
          unconditional normals: a DIFFERENT set of random variables (same law), so exact identity
          is structurally impossible there regardless of parameters.
    The tolerance below is a float64-epsilon guard on (1); it is orders of magnitude tighter than
    any economically meaningful difference, so a genuine divergence of the two branches still fails."""
    gbm = _mtm(_cfg(False, target, _MONTHLY), seed=seed)
    hn = _mtm(_cfg(True, target, _MONTHLY, hn_params=_DEGENERATE, steps_per_year=_SPY_ONE_SUBSTEP),
              seed=seed)
    assert hn == pytest.approx(gbm, rel=1e-10, abs=1e-6)


# ======================================================================================
# CLOSED-FORM GATE + h-recursion (two persistences => the daily recursion is exercised)
# ======================================================================================

@pytest.mark.parametrize('persistence', [0.90, 0.96])
def test_single_fixing_hn_matches_hn_garch_closed_form(persistence):
    """A single-fixing, never-knocking (huge target) HN TARF pays N1*relu(S_n-K) discounted; with
    zero carry that is exactly N1 * hn_garch.hn_call under the HN law simulated by the pricer's daily
    recursion. Running it at two persistences (different h term structures) pins that the h-recursion
    is engaged AND that the OSS final-step draw feeds it (the never-knock terminal law is the law of
    the recursion where the last draw feeds h - it would not match hn_call otherwise)."""
    hn_params, p = _real_garch(persistence)
    horizon = 60
    n_sub = max(int(round(horizon / 365.0 * 252.0)), 1)
    ref = float(hn_garch.hn_call(
        SPOT, STRIKE, hn_garch.HNParams(p.omega, p.alpha, p.beta, p.gamma, 0.0).as_tensors(),
        n_sub, hn_params['H0'])) * N1
    price = _mtm(_cfg(True, 1e9, [horizon], hn_params=hn_params, steps_per_year=252.0),
                 seed=3, sims=1 << 17)  # ~7e-4 MC error observed; tol is ~4x that
    assert price == pytest.approx(ref, rel=3e-3)


def test_hn_garch_dynamics_are_engaged_vs_flat_vol_gbm():
    """h really recurses: the GARCH term structure + leverage move the single-fixing price off a
    flat-vol GBM struck at the SAME annualised long-run vol (a bypassed/constant-h implementation
    would land on the GBM number)."""
    hn_params, p = _real_garch()
    horizon = 60
    hn = _mtm(_cfg(True, 1e9, [horizon], hn_params=hn_params, steps_per_year=252.0),
              seed=3, sims=1 << 16)
    gbm = _mtm(_cfg(False, 1e9, [horizon], sigma=float(p.ann_vol(252.0))), seed=3, sims=1 << 16)
    assert abs(hn - gbm) / gbm > 3e-3  # ~1% observed; comfortably above MC noise (~2e-3)


# ======================================================================================
# FINAL-STEP h-RECURSION GATE  (pricing.py sim_spot_tarf, the survival-conditioned update
#   h = Omega + Beta*h + Alpha*(Z - Gamma_Star*sqrt(h))**2  fed forward ACROSS fixings)
#
# The single-fixing closed-form gate above does NOT cover that update: with one fixing the final
# h is produced but never consumed. Four one-line mutants of it - (a) Z- -> Z+ leverage-sign flip,
# (b) freeze h (h=h), (c) 10*Alpha, (d) drop Beta*h - all survive every test above. The two tests
# below kill them (measured mutation matrix, reldiff vs reference; correct code in [brackets]):
#     never-knock daily (test A): a 1.2e-2  b 5.4e-2  c 2.8e0  d 5.0e-1   [~1.7e-3]
#     knocking strong-lev (B)   : a 5.1e-2  b 4.2e-3  c 7.8e-2 d 9.3e-2   [~2e-4]
# A alone kills (b),(c),(d) (and here (a)); B is the dedicated (a) killer - its survival-truncated
# draw gives E[Z]<0, so the leverage cross-term 2*Gamma_Star*sqrt(h)*E[Z] makes E[h'] FIRST-ORDER
# sign-sensitive, which the near-unconditional never-knock law only sees as a weaker skew effect.
# ======================================================================================

# DAILY fixings (1 day apart, spy=252 => round(1/365*252)=1 => n_sub=1 per fixing): EVERY step is
# the monitored final draw, so the cross-fixing final-step update is under test at every step.
_DAILY_NK = list(range(1, 21))  # cum_days = 1..20


@pytest.mark.parametrize('r_dom', [0.0, 0.05])  # r_dom=0.05 (r_for=0) exercises b_step != 0 (Task 3)
def test_never_knock_daily_hn_matches_summed_closed_form(r_dom):
    """Never-knock (huge target) 20-fixing DAILY TARF: survival ~ 1, so each fixing pays
    N1*relu(S_d - K) and the total equals N1 * sum_d hn_garch.hn_call at the cumulative day counts
    with h1=H0. Because n_sub=1, every daily step feeds h through the survival-conditioned final-step
    recursion, and h is carried continuously across all 20 fixings - so a wrong final-step update
    (frozen/mis-scaled/sign-flipped) desynchronises h for every later fixing and breaks the match.
    Non-zero carry (r_dom, r_for=0 => carry=discount) threads into hn_call as r = r_dom/365 (one
    ACT/365 day per step), pinning b_step = fwd_carry*dt/n_sub."""
    hn_params, p = _real_garch(0.94)
    hn_params = dict(hn_params, H0=1.5 * float(p.stationary_var))  # H0 != stationary => term structure
    ptens = hn_garch.HNParams(p.omega, p.alpha, p.beta, p.gamma, r_dom / 365.0).as_tensors()
    ref = sum(float(hn_garch.hn_call(SPOT, STRIKE, ptens, c, hn_params['H0']))
              for c in _cum_days(_DAILY_NK)) * N1
    price = _mtm(_cfg(True, 1e9, _DAILY_NK, hn_params=hn_params, steps_per_year=252.0, r_dom=r_dom),
                 seed=3, sims=1 << 17)  # correct reldiff ~1.7e-3; tol ~3x, mutants >= 1.2e-2
    assert price == pytest.approx(ref, rel=5e-3)


# Strong-leverage KNOCKING config (gamma*=400, leverage_share=0.8): the KO-truncated final draw
# biases E[Z]<0, engaging the leverage cross-term so the (Z - Gamma_Star*sh) SIGN moves E[h'].
def _knock_params():
    p = hn_garch.hn_params_from_targets(ann_vol=0.30, persistence=0.94, gamma=400.0,
                                        leverage_share=0.8, steps_per_year=252.0)
    return {'Omega': float(p.omega), 'Alpha': float(p.alpha), 'Beta': float(p.beta),
            'Gamma_Star': float(p.gamma), 'H0': 1.5 * float(p.stationary_var)}


_KNOCK_FIX = list(range(1, 41))  # 40 daily fixings
_KNOCK_TARGET = 0.04
# REGRESSION CONSTANT: value of the KNOCKING config under an INDEPENDENT brute-force daily-HN MC
# that simulates the full daily path and monitors KO/accrual only at the fixings (_brute_force_tarf
# below - it never touches the pricer's line-1545 update, so it is the invariant ground truth that
# the mutants are scored against). Provenance: mean of _brute_force_tarf at paths=6_000_000 over
# seeds (20240628, 777, 12345) = 33024.6 (per-seed spread 5.3, i.e. 1.6e-4 rel). Recompute with
# _regen_knock_reference() if the config changes. NOT re-run in CI.
_KNOCK_BRUTE_REF = 33024.6


def _brute_force_tarf(hn_params, fix_days, target, seed, paths, spy=252.0, r_daily=0.0):
    """Plain (un-truncated) daily Heston-Nandi MC of the same TARF product the OSS pricer values:
    log S_{t+1} = log S_t + r_daily - 0.5 h_t + sqrt(h_t) z_t,  h_{t+1} = Omega + Beta h_t +
    Alpha (z_t - Gamma_Star sqrt(h_t))**2, monitoring only at the fixings, paying N1*min(relu(S-K),
    remaining) and knocking when the target is exhausted. Independent of pricing.py."""
    g = torch.Generator().manual_seed(seed)
    Om, Al, Be, Ga = (float(hn_params[k]) for k in ('Omega', 'Alpha', 'Beta', 'Gamma_Star'))
    fix_at = set(_cum_days(fix_days, spy))
    logS = torch.zeros(paths, dtype=torch.float64)
    h = torch.full((paths,), float(hn_params['H0']), dtype=torch.float64)
    rem = torch.full((paths,), float(target), dtype=torch.float64)
    pv = torch.zeros(paths, dtype=torch.float64)
    for day in range(1, max(fix_at) + 1):
        z = torch.randn(paths, generator=g, dtype=torch.float64)
        sh = h.sqrt()
        logS = logS + (r_daily - 0.5 * h) + sh * z
        h = Om + Be * h + Al * (z - Ga * sh) ** 2
        if day in fix_at:
            pay = torch.minimum(torch.relu(SPOT * torch.exp(logS) - STRIKE), rem)
            pv, rem = pv + pay, rem - pay  # zero rates => discount = 1
    return float(N1 * pv.mean())


def _regen_knock_reference(paths=6_000_000, seeds=(20240628, 777, 12345)):
    return float(np.mean([_brute_force_tarf(_knock_params(), _KNOCK_FIX, _KNOCK_TARGET, s, paths)
                          for s in seeds]))


def test_knocking_hn_matches_brute_force_daily_mc():
    """KNOCKING strong-leverage TARF vs the hardcoded brute-force regression constant. Under the
    correct survival-conditioned update the OSS estimator is an exact importance-weighted estimator
    of the SAME product, so it reproduces the brute-force value to Monte-Carlo error (correct reldiff
    ~2e-4 across seeds; brute-force reference itself is 1.6e-4-tight). The leverage-sign mutant (a)
    biases the survival-truncated h-path and shifts the price by ~5e-2 - killed by the tol below."""
    price = _mtm(_cfg(True, _KNOCK_TARGET, _KNOCK_FIX, hn_params=_knock_params(), steps_per_year=252.0),
                 seed=3, sims=1 << 17)
    assert price == pytest.approx(_KNOCK_BRUTE_REF, rel=5e-3)
