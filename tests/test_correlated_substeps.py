"""Coarse-grid sub-stepping — the PFE/CVA node law (utils.substep_schedule / substep_normals /
hn_correlated_substeps / garch_correlated_substeps and their routing in GARCHSpotModel /
HestonNandiImpliedSpotModel._simulate_returns).

A scenario grid step spanning more than one calibration step walks the sub-steps that span it —
whole trading days, then the fractional remainder — through the same recursion the fine grid
takes. This replaced a deterministic-h aggregate-variance bridge over round(f) whole days, which
was wrong twice over: 29%→2000% off on tail probabilities at |z|=2–3 (gates/hn_aggregate_bias.csv),
and carrying round(f) days of variance instead of f (−13% on the framework's own default CVA
grid, since a calendar grid never yields an integer f).

Every gate below was checked to FAIL under the mutation it exists to catch:

  (a) substep_schedule sums to f EXACTLY — the clock gate. Dies under round(f)/floor(f).
      Deterministic, no MC noise, and the sharpest statement of what was broken.
  (b) E[Σ h_j·dt_j] = h_0·f at stationary h_0, on a NON-INTEGER f. Dies under round(f) by
      round(f)/f − 1 (up to 13%, ~40 MC sigma).
  (c) substep_normals contract: w'Z = z_fw to float precision AND Cov(Z_j, z_fw) = w_j with
      w ∝ √var — the weights ARE the return loading. Dies under uniform weights (the mutation
      that survived every distributional gate, because marginals are invariant to w).
  (d) Z is iid N(0,1) — tightened until dropping the projection fails it.
  (e) HN coarse marginal IS the exact n-step law: matches the Fourier-inversion oracle
      utils.hn_cdf_logret at n ∈ {21,63}, at tail points where the old bridge's normal CDF is
      proven outside the MC band (the gate asserts the bridge would fail — power, not faith).
  (f) h_end is DISPERSED and matches a daily-grid run (the bridge collapsed it to its mean).
  (g) Multi-node coarse == daily terminal quantiles, tolerance set from a measured 8-seed
      spread, not fitted to the shipped seeds.
  (h) GARCH sub-innovations are EXACTLY standardized Student-t (direct kurtosis gate on the
      primitive: t_8 → 1.5, Gaussian → 0).
  (i) reseed_from_path refuses a coarse grid; inner-MC mode forks on the middle axis.

Deterministic (seeded CMC_State + float64). CPU sizes.
"""
import os
import sys
import types

# reference-riskflow shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import kurtosis, norm

import riskflow
from riskflow import utils
import hn_reference as hnref
from riskflow.calculation import CMC_State, CMC_State_Inner
from riskflow.stochasticprocess import GARCHSpotModel, HestonNandiImpliedSpotModel

DTYPE = torch.float64
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0
CALENDAR = 1.0 / 365.25                      # the production clock: f = 252k/365.25, never integer

# Strong-leverage HN with a term structure (H0 != stationary) so the walk is genuinely exercised.
_SP = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=252.0)
H0_STAT = float(utils.hn_stationary_var(_SP['omega'], _SP['alpha'], _SP['beta'], _SP['gamma_star']))
H0_TS = 1.6 * H0_STAT
HN_IMPLIED = {'Omega': float(_SP['omega']), 'Alpha': float(_SP['alpha']), 'Beta': float(_SP['beta']),
              'Gamma_Star': float(_SP['gamma_star']), 'H0': H0_TS, 'Steps_Per_Year': 252.0}
GARCH_PARAM = {'Omega': 4.0e-06, 'Alpha': 0.06, 'Beta': 0.92, 'Nu': 8.0, 'Mu': 0.0, 'H0': 1.5e-04,
               'Log_Price': True, 'Calibration_DT_Years': DT_C, 'Convexity_Correction': 'Yes'}


def test_uses_repo_under_test():
    assert riskflow.__file__ == os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'riskflow', '__init__.py')


# ---------------------------------------------------------------------------
# Harness (the test_hn_implied_process idiom: synthetic grid + seeded CMC_State)
# ---------------------------------------------------------------------------

def _time_grid(T, day_step=1.0, year_per_day=DT_C):
    days = np.cumsum(np.full(T, day_step, dtype=np.float64))
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days * year_per_day
    tg.CurrencyMap = {}
    scen = np.zeros((T, 3), dtype=np.float64)
    scen[:, utils.TIME_GRID_MTM] = days
    scen[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(T)
    tg.scenario_grid = scen
    return tg


def _shared(B, tg, seed=7, sub=None, cholesky=None):
    one = torch.ones(1, 1, dtype=DTYPE)
    chol = torch.eye(1, dtype=DTYPE) if cholesky is None else cholesky
    nf = chol.shape[0]
    if sub is None:
        sh = CMC_State(cholesky=chol, static_buffer={}, batch_size=B, one=one, mcmc_sims=0,
                       report_currency=None, seed=seed, job_id=0, num_jobs=1)
        sh.reset(num_factors=nf, time_grid=tg)
    else:
        sh = CMC_State_Inner(cholesky=chol, static_buffer={}, batch_size=B, one=one, mcmc_sims=0,
                             report_currency=None, seed=seed, job_id=0, num_jobs=1,
                             simulation_sub_batch=sub)
        sh.reset_inner(num_factors=nf, time_grid=tg)
    return sh


def _rate_code(name, level):
    factor = utils.Factor('InterestRate', (name,))
    tp = np.array([0.0, 1.0, 5.0, 30.0], dtype=np.float64)
    td = utils.tenor_diff(tp, 'Linear')
    dc = (lambda days: utils.get_day_count_accrual(REF_DATE, days, utils.DAYCOUNT_ACT365))
    return [(False, factor, None, td, dc)], factor, torch.full((tp.size,), level, dtype=DTYPE)


def _make_hn(tg, sh, spot0=None, param=None):
    p = HestonNandiImpliedSpotModel(
        factor=types.SimpleNamespace(param={}), param=None,
        implied_factor=types.SimpleNamespace(param=dict(param or HN_IMPLIED)))
    p.factor_type = 'EquityPrice'
    p.factor_key = utils.Factor('EquityPrice', ('EQ',))
    rc, rf, rcur = _rate_code('R', 0.0)
    sh.t_Static_Buffer[rf] = rcur
    p.r_t = p.q_t = rc
    p.precalculate(REF_DATE, tg, spot0 if spot0 is not None else torch.tensor([100.0], dtype=DTYPE),
                   sh, process_ofs=0)
    return p


def _make_garch(tg, sh, spot0=None, param=None):
    p = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(param or GARCH_PARAM))
    p.factor_key = utils.Factor('CommodityPrice', ('PT',))
    p.precalculate(REF_DATE, tg, spot0 if spot0 is not None else torch.tensor([1000.0], dtype=DTYPE),
                   sh, process_ofs=0)
    return p


def _hn_cf_params():
    return hnref.as_tensors({'omega': _SP['omega'], 'alpha': _SP['alpha'], 'beta': _SP['beta'],
                             'gamma_star': _SP['gamma_star'], 'r': 0.0})


def _t(x):
    return torch.tensor(float(x), dtype=DTYPE)


# ---------------------------------------------------------------------------
# (a) THE CLOCK — the schedule spans exactly the interval it is given
# ---------------------------------------------------------------------------

def test_substep_schedule_spans_exactly_f():
    """The defect this replaced: n_sub = round(f) gave the interval round(f) whole days of
    variance instead of f. Deterministic gate — no MC noise — and it dies on any rounding."""
    f = np.array([0.0, 0.69, 1.0, 1.38, 2.0, 2.7598, 3.4497, 4.8296, 21.3881, 63.4677])
    for x, steps in zip(f, utils.substep_schedule(f)):
        assert abs(sum(steps) - x) < 1e-12 or x == 0.0, f'f={x}: schedule {steps} sums to {sum(steps)}'
        assert all(s <= 1.0 + 1e-12 for s in steps), f'f={x}: a sub-step exceeds one calibration step'
        assert len(steps) == max(int(np.ceil(x)), 1), f'f={x}: {len(steps)} sub-steps'
    # whole steps first, remainder last — the OSS pricer's convention
    assert utils.substep_schedule([3.25])[0] == (1.0, 1.0, 1.0, 0.25)


@pytest.mark.parametrize('day_step', [4.0, 5.0, 7.0])
def test_calendar_grid_walks_its_own_trading_time(day_step):
    """A CALENDAR grid (the production clock) has non-integer f at every coarse step. The walked
    trading time must equal the elapsed trading time exactly — under round(f) this is off by
    -13.0% at 5 calendar days and +8.7% at 4, which is a step function of grid spacing."""
    T = 8
    tg = _time_grid(T, day_step, year_per_day=CALENDAR)
    p = _make_hn(tg, _shared(16, tg))
    assert p.n_sub[1] > 1, f'day_step={day_step} did not produce a coarse step'
    walked = sum(sum(s) for s in p.sub_dt[1:])
    elapsed = float(p.f[1:].sum())
    assert abs(walked / elapsed - 1.0) < 1e-12, \
        f'walked {walked:.6f} trading days, grid elapsed {elapsed:.6f}'


@pytest.mark.parametrize('make,name', [(_make_hn, 'hn'), (_make_garch, 'garch')], ids=['hn', 'garch'])
def test_integrated_variance_is_f_not_round_f(make, name):
    """At STATIONARY h_0 the mean recursion is flat, so E[Σ h_j·dt_j] = h_0·f exactly for any
    schedule. On a non-integer f (2.7598 = 4 calendar days) round(f)=3 would inflate this by
    +8.7%, ~40 sigma at this batch. The one gate that reads the realized integrated variance."""
    B = 400000
    if name == 'hn':
        param = {**HN_IMPLIED, 'H0': H0_STAT}
        h0 = H0_STAT
    else:
        param = dict(GARCH_PARAM)
        h0 = float(GARCH_PARAM['Omega'] / (1.0 - GARCH_PARAM['Alpha'] - GARCH_PARAM['Beta']))
        param['H0'] = h0
    tg = _time_grid(2, 4.0, year_per_day=CALENDAR)
    sh = _shared(B, tg, seed=17)
    p = make(tg, sh, param=param)
    f = float(p.f[1])
    assert abs(f - 2.7598) < 1e-3 and p.n_sub[1] == 3, (f, p.n_sub)
    spot = p.generate(sh)
    # Var(interval return) = E[Σ h_j dt_j] for a zero-mean/martingale step (GARCH mu=0, HN carry=0)
    d = (spot.log()[1] - spot.log()[0])
    realized = d.var().item()
    se = 3.0 * realized * np.sqrt(2.0 / B)                       # ~3sigma on a variance estimate
    assert abs(realized - h0 * f) < 4.0 * se, \
        f'{name}: Var {realized:.6e} != h0*f {h0 * f:.6e} (round(f)*h0 would be {h0 * round(f):.6e})'


# ---------------------------------------------------------------------------
# (c)/(d) the projection: reproduces the framework draw, weights ARE the loading
# ---------------------------------------------------------------------------

def test_substep_normals_reproduces_framework_draw():
    torch.manual_seed(0)
    n, B = 63, 200000
    sqrt_var = (torch.rand(n, B, dtype=DTYPE) * 2.0 + 0.2).sqrt()
    z_fw = torch.randn(B, dtype=DTYPE)
    Z = utils.substep_normals(sqrt_var, z_fw)
    w = sqrt_var / (sqrt_var ** 2).sum(0, keepdim=True).sqrt()
    assert ((w * Z).sum(0) - z_fw).abs().max().item() < 1e-12, "w'Z != z_fw"


def test_substep_normals_loading_is_the_weights():
    """Cov(Z_j, z_fw) = w_j, i.e. the framework draw enters as the √var-weighted combination.
    The weights leave every MARGINAL invariant, so this is the ONLY gate that sees them —
    without it, replacing √E[h·dt] with uniform weights passes the whole suite."""
    torch.manual_seed(1)
    n, B = 16, 400000
    # a strongly-sloped variance profile so uniform and √var weights are far apart
    prof = torch.linspace(4.0, 0.25, n, dtype=DTYPE).view(n, 1).sqrt()
    sqrt_var = prof.expand(n, B).contiguous()
    z_fw = torch.randn(B, dtype=DTYPE)
    Z = utils.substep_normals(sqrt_var, z_fw)
    w = (prof / (prof ** 2).sum().sqrt()).view(-1).numpy()
    cov = (Z * z_fw).mean(dim=1).numpy()                          # z_fw is unit variance
    se = 1.0 / np.sqrt(B)
    assert np.abs(cov - w).max() < 5.0 * se, f'loading != weights: max dev {np.abs(cov - w).max():.5f}'
    uniform = 1.0 / np.sqrt(n)
    assert np.abs(cov - uniform).max() > 30.0 * se, 'profile too flat — uniform weights would pass'


def test_substep_normals_iid():
    torch.manual_seed(2)
    n, B = 63, 200000
    sqrt_var = (torch.rand(n, B, dtype=DTYPE) * 2.0 + 0.2).sqrt()
    Z = utils.substep_normals(sqrt_var, torch.randn(B, dtype=DTYPE))
    assert Z.mean(dim=1).abs().max().item() < 0.015
    # tightened until dropping the projection (Z = e + w*z_fw, variance 1 + w^2) fails it
    assert (Z.var(dim=1) - 1.0).abs().max().item() < 0.020
    C = np.corrcoef(Z.numpy())
    off = np.abs(C - np.eye(n)).max()
    assert off < 0.016, f'sub-steps correlated: max |off-diagonal| {off:.4f}'


# ---------------------------------------------------------------------------
# (e) HN coarse marginal == the exact n-step law (and the bridge is refuted)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n', [21, 63])
def test_hn_coarse_marginal_matches_closed_form(n):
    B = 400000
    tg = _time_grid(2, day_step=float(n))                          # f = n exactly ⇒ the integer oracle applies
    sh = _shared(B, tg, seed=11)
    p = _make_hn(tg, sh)
    assert list(p.n_sub) == [1, n]
    R = (p.generate(sh).log()[1] - np.log(100.0)).numpy()
    cf = _hn_cf_params()
    # the old bridge's law: N(-V/2, V), V the deterministic mean-forwarded Σ E[h_j]
    psi = float(_SP['beta'] + _SP['alpha'] * _SP['gamma_star'] ** 2)
    V, m = 0.0, H0_TS
    for _ in range(n):
        V, m = V + m, float(_SP['omega'] + _SP['alpha']) + psi * m
    bridge_beaten = 0
    for zq in (-3.0, -2.0, 0.0, 2.0):
        b = float(R.mean() + zq * R.std())
        oracle = float(utils.hn_cdf_logret(b, n, H0_TS, **cf))
        sim = float((R <= b).mean())
        se = float(np.sqrt(max(oracle * (1.0 - oracle), 1e-12) / B))
        assert abs(sim - oracle) < 4.0 * se, \
            f'n={n} z={zq:+.0f}: sim {sim:.5f} != oracle {oracle:.5f} ({abs(sim - oracle) / se:.1f} se)'
        bridge_beaten += abs(float(norm.cdf((b + V / 2.0) / np.sqrt(V))) - oracle) > 8.0 * se
    assert bridge_beaten >= 2, 'bridge CDF inside the MC band everywhere — the gate has no power'


def test_hn_coarse_h_end_dispersion_matches_daily():
    """The bridge forwarded h deterministically: every path left the interval with the SAME h."""
    n, B = 21, 200000
    tg_c = _time_grid(2, day_step=float(n))
    sh_c = _shared(B, tg_c, seed=5)
    p_c = _make_hn(tg_c, sh_c)
    p_c.generate(sh_c)
    h_c = p_c.last_log_h[1].exp().numpy()
    tg_d = _time_grid(n + 1)
    sh_d = _shared(B, tg_d, seed=6)
    p_d = _make_hn(tg_d, sh_d)
    p_d.generate(sh_d)
    h_d = p_d.last_log_h[n].exp().numpy()
    cv_c, cv_d = h_c.std() / h_c.mean(), h_d.std() / h_d.mean()
    assert cv_c > 0.2, f'h_end degenerate (cv={cv_c:.3f}) — the bridge is back'
    assert abs(cv_c / cv_d - 1.0) < 0.10, f'h_end spread off daily: coarse cv {cv_c:.3f} vs {cv_d:.3f}'
    for q in (5, 50, 95):
        qc, qd = np.percentile(h_c, q), np.percentile(h_d, q)
        assert abs(qc / qd - 1.0) < 0.05, f'h_end q{q}: coarse {qc:.3e} vs daily {qd:.3e}'


# ---------------------------------------------------------------------------
# (g) multi-node coarse grid == daily grid, terminal quantiles
# ---------------------------------------------------------------------------

def _terminal_quantiles(make, coarse_T, n, daily_T, spot0, seed_c, seed_d):
    B = 400000
    tg_c = _time_grid(coarse_T, day_step=float(n))
    sh_c = _shared(B, tg_c, seed=seed_c)
    p_c = make(tg_c, sh_c)
    R_c = (p_c.generate(sh_c).log()[coarse_T - 1] - np.log(spot0)).numpy()
    tg_d = _time_grid(daily_T)
    sh_d = _shared(B, tg_d, seed=seed_d)
    p_d = make(tg_d, sh_d)
    R_d = (p_d.generate(sh_d).log()[daily_T - 1] - np.log(spot0)).numpy()
    return R_c, R_d


@pytest.mark.parametrize('make,spot0', [(_make_hn, 100.0), (_make_garch, 1000.0)], ids=['hn', 'garch'])
def test_multinode_terminal_quantiles_match_daily(make, spot0):
    """3 coarse intervals of 21 days == 63 daily steps. Tolerances come from a MEASURED 8-seed
    spread of |Δq| (q01: mean 0.0022, sd 0.0017, max 0.0061 at this batch), not from the shipped
    seeds — an earlier 3.0e-3 on q01 failed 5 of 12 seed pairs."""
    R_c, R_d = _terminal_quantiles(make, 4, 21, 64, spot0, seed_c=3, seed_d=4)
    for q, tol in ((1, 1.0e-2), (5, 4.0e-3), (50, 2.0e-3), (95, 4.0e-3), (99, 1.0e-2)):
        qc, qd = np.percentile(R_c, q), np.percentile(R_d, q)
        assert abs(qc - qd) < tol, f'q{q:02d}: coarse {qc:+.5f} vs daily {qd:+.5f} (tol {tol})'
    assert abs(R_c.std() / R_d.std() - 1.0) < 0.01, 'terminal dispersion off the daily witness'


# ---------------------------------------------------------------------------
# (h) GARCH sub-innovations are exactly standardized Student-t
# ---------------------------------------------------------------------------

def test_garch_substep_innovation_is_standardized_t():
    """Direct gate on the primitive: one sub-step, so r_sum/√var_sum IS the innovation ε.
    Standardized t_8 has unit variance and excess kurtosis 6/(ν−4) = 1.5; a Gaussian sub-step
    (the mutation: skip the t-scaling) gives 0. The distributional gates alone cannot see this."""
    B = 2000000
    torch.manual_seed(5)
    h = torch.full((B,), 1.5e-4, dtype=DTYPE)
    _, var_sum, r_sum = utils.garch_correlated_substeps(
        h, torch.randn(B, dtype=DTYPE), (1.0,),
        _t(GARCH_PARAM['Omega']), _t(GARCH_PARAM['Alpha']), _t(GARCH_PARAM['Beta']), _t(8.0))
    eps = (r_sum / var_sum.sqrt()).numpy()
    assert abs(eps.var() - 1.0) < 0.01, f'ε not unit variance: {eps.var():.4f}'
    k = kurtosis(eps, fisher=True)
    assert 1.0 < k < 2.1, f'ε excess kurtosis {k:.3f} — t_8 is 1.50, Gaussian 0'


def test_hn_single_substep_is_the_framework_draw():
    """One sub-step ⇒ w = 1 ⇒ Z = z_fw identically: the walk degenerates to the fine step, which
    is why the fine branch stays bit-identical."""
    B = 200000
    torch.manual_seed(7)
    h = torch.full((B,), H0_STAT, dtype=DTYPE)
    z_fw = torch.randn(B, dtype=DTYPE)
    _, var_sum, r_sum = utils.hn_correlated_substeps(
        h, z_fw, (1.0,), _t(_SP['omega']), _t(_SP['alpha']), _t(_SP['beta']), _t(_SP['gamma_star']))
    assert ((r_sum / var_sum.sqrt()) - z_fw).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
# (i) replay refuses a coarse grid; inner-MC forks on the middle axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('make', [_make_hn, _make_garch], ids=['hn', 'garch'])
def test_reseed_from_path_coarse_raises(make):
    tg = _time_grid(3, day_step=21.0)
    sh = _shared(64, tg, seed=1)
    p = make(tg, sh)
    path = p.generate(sh)
    with pytest.raises(ValueError, match='n_sub == 1'):
        p.reseed_from_path(path, sh)
    # positive control: the daily grid still replays
    tg_d = _time_grid(20)
    sh_d = _shared(64, tg_d, seed=1)
    p_d = make(tg_d, sh_d)
    p_d.reseed_from_path(p_d.generate(sh_d), sh_d)


@pytest.mark.parametrize('make', [_make_hn, _make_garch], ids=['hn', 'garch'])
def test_inner_mode_coarse_forks_on_middle_axis(make):
    """The fork seed is per-OUTER-path and must broadcast across B2, through a coarse walk."""
    B, B2, T = 5, 8, 3
    tg = _time_grid(T, day_step=21.0)
    sh = _shared(B, tg, seed=42, sub=B2)
    p = make(tg, sh, spot0=torch.linspace(80.0, 120.0, B, dtype=DTYPE))
    assert (p.n_sub[1:] > 1).all(), 'grid is not coarse'
    seed = torch.linspace(1.0, 3.0, B, dtype=DTYPE) * H0_STAT     # distinct entry variance per outer
    sh.t_Scenario_Buffer[(p.factor_key, 'h0_inner')] = seed
    out = p.generate(sh)
    assert out.shape == (T, B, B2) and torch.isfinite(out).all()
    assert torch.equal(p.last_log_h[0], seed.view(B, 1).expand(B, B2).log()), \
        'inner h0 != the passed seed (fork did not land on the middle axis)'
    # the walk actually moved the state and the price off their t=0 values
    assert not torch.allclose(p.last_log_h[T - 1], p.last_log_h[0]), 'variance never advanced'
    assert (out[T - 1] != out[0]).all(), 'price never advanced'
