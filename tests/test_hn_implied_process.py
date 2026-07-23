"""Acceptance tests for HestonNandiImpliedSpotModel — the Heston-Nandi GARCH(1,1) risk-neutral
IMPLIED spot process, the final piece of the HN build: the calibrated HestonNandiModelParameters
factor that the semi-analytic option pricer consumes now ALSO drives the outer-scenario evolution of
its own underlying (CVA / credit-MC).

Covered:
  (a) construct_process dispatches the class by name; find_models routes EquityPrice ->
      HestonNandiImpliedSpotModel with the implied HestonNandiModelParameters additional factor and
      the dummy Price Models entry.
  (b) PRICE-space martingale: E[S_t e^{-∫(r-q)}] flat (carry=0 ⇒ E[S_t]=S_0), AND the by-log-h-state
      version (the LOG-space test is structurally blind to a Jensen price drift — the platinum lesson).
  (c) hn_garch ORACLE: the simulated terminal distribution matches the closed-form hn_moments /
      hn_call prices / hn_cdf_logret probabilities within MC error (the closed form validates the
      whole process end-to-end).
  (d) reveal / fork: privileged_layout width, reveal_state_at packing (log_h first / price last),
      inner_fork_seed fidelity (inner h[0] == exp(outer log h_t) EXACTLY), reseed_from_path
      forward/replay consistency, revealed_annual_vol.
  THE CLOCK: business-daily (f=1, EXACT hn_garch), calendar-daily (f≈0.69) and 2-bd (n_sub=2 bridge)
      grids; grid-invariant annualized vol (the platinum GARCH trap NOT re-created).
  DRIFT / MEASURE: the risk-neutral (r-q) drift is read from the underlying's OWN interest-rate and
      dividend curves (E[S_T/S_0] = exp(∫(r-q)) end-to-end).
  DEDUPE: the pricer's static HestonNandiModelParameters leaf and the process's implied leaf are ONE
      tensor (greeks on), and the gradient flows to BOTH the scenario path and the pricer.

Deterministic (seeded CMC_State + torch default double). CPU-fast sizes.
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
from scipy.stats import skew, kurtosis

import riskflow
from riskflow import hn_garch, utils
from riskflow.calculation import CMC_State, CMC_State_Inner, construct_calculation, construct_process
from riskflow.config import Config
from riskflow.instruments import construct_instrument
from riskflow.stochasticprocess import (
    HestonNandiImpliedSpotModel, REVEAL_SUFFICIENT, REVEAL_CONTINUOUS)

DTYPE = torch.float64
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0

# Strong-leverage HN with a term structure (H0 != stationary): the recursion is genuinely exercised
# AND the leverage cross-term engaged (skew ≈ -0.9), so the oracle match is a real test.
_SP = hn_garch.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=252.0)
H0_STAT = float(_SP.stationary_var)
H0_TS = 1.6 * H0_STAT
IMPLIED_PARAM = {'Omega': float(_SP.omega), 'Alpha': float(_SP.alpha), 'Beta': float(_SP.beta),
                 'Gamma_Star': float(_SP.gamma), 'H0': H0_TS, 'Steps_Per_Year': 252.0}


def _hn_params(h0=None):
    """hn_garch.HNParams (tensors, r=0) mirroring IMPLIED_PARAM for the closed-form oracle."""
    return hn_garch.HNParams(_SP.omega, _SP.alpha, _SP.beta, _SP.gamma, 0.0).as_tensors()


def test_uses_repo_under_test():
    assert riskflow.__file__ == os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'riskflow', '__init__.py')


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _time_grid(T, year_per_day=DT_C, day_step=1.0):
    """`year_per_day=DT_C` ⇒ business-daily grid (dt=dt_c, f=1, EXACT hn_garch). `1/365.25` ⇒ the
    calendar-daily production convention (f≈0.69). `day_step` scales spacing (the 2-bd grid)."""
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


def _rate_code(name, level):
    """A flat static zero curve (is_stoch=False) + its (n_tenors,) buffer tensor. r_t and q_t are
    read via exactly this code the way GBMAssetPriceTSModelImplied reads its drift curves."""
    factor = utils.Factor('InterestRate', (name,))
    tp = np.array([0.0, 1.0, 5.0, 30.0], dtype=np.float64)
    td = utils.tenor_diff(tp, 'Linear')
    dc = (lambda days: utils.get_day_count_accrual(REF_DATE, days, utils.DAYCOUNT_ACT365))
    return [(False, factor, None, td, dc)], factor, torch.full((tp.size,), level, dtype=DTYPE)


def _make(T, B, tg=None, sub=None, seed=7, param=None, r_level=0.0, q_level=None):
    """Build a precalculated process + seeded shared state. q_level=None ⇒ r_t and q_t share ONE
    code (carry ≡ 0 to the bit); else a distinct q curve (a real (r-q) carry)."""
    tg = tg or _time_grid(T)
    one = torch.ones(1, 1, dtype=DTYPE)
    if sub is None:
        sh = CMC_State(cholesky=torch.eye(1, dtype=DTYPE), static_buffer={}, batch_size=B, one=one,
                       mcmc_sims=0, report_currency=None, seed=seed, job_id=0, num_jobs=1)
        sh.reset(num_factors=1, time_grid=tg)
        tensor = torch.tensor([100.0], dtype=DTYPE)
    else:
        sh = CMC_State_Inner(cholesky=torch.eye(1, dtype=DTYPE), static_buffer={}, batch_size=B,
                             one=one, mcmc_sims=0, report_currency=None, seed=seed, job_id=0,
                             num_jobs=1, simulation_sub_batch=sub)
        sh.reset_inner(num_factors=1, time_grid=tg)
        tensor = torch.full((B,), 100.0, dtype=DTYPE)
    p = HestonNandiImpliedSpotModel(
        factor=types.SimpleNamespace(param={}), param=None,
        implied_factor=types.SimpleNamespace(param=dict(param or IMPLIED_PARAM)))
    p.factor_type = 'EquityPrice'
    p.factor_key = utils.Factor('EquityPrice', ('EQ',))
    rc, rf, rcur = _rate_code('R', r_level)
    sh.t_Static_Buffer[rf] = rcur
    if q_level is None:
        p.r_t = p.q_t = rc
    else:
        qc, qf, qcur = _rate_code('Q', q_level)
        sh.t_Static_Buffer[qf] = qcur
        p.r_t, p.q_t = rc, qc
    p.precalculate(REF_DATE, tg, tensor, sh, process_ofs=0)
    return p, sh


# ---------------------------------------------------------------------------
# (a)/(e) registration + routing
# ---------------------------------------------------------------------------

def test_construct_process_dispatch():
    proc = construct_process('HestonNandiImpliedSpotModel', types.SimpleNamespace(param={}),
                             None, implied_factor=types.SimpleNamespace(param=dict(IMPLIED_PARAM)))
    assert isinstance(proc, HestonNandiImpliedSpotModel)
    assert HestonNandiImpliedSpotModel.num_factors() == 1
    assert proc.correlation_name == ('HestonNandiSpotProcess', [()])


def test_find_models_routes_implied_process():
    """A market-data JSON where EquityPrice is mapped to HestonNandiImpliedSpotModel and the
    HestonNandiModelParameters factor is present routes the factor to the implied process, resolves
    the implied additional factor by naming convention, and stamps the dummy Price Models entry."""
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['Model Configuration'].append('EquityPrice', (), 'HestonNandiImpliedSpotModel')
    c.params['Price Factors'] = {
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'HestonNandiModelParameters.EQ': dict(IMPLIED_PARAM),
    }
    c.params['Price Models'] = {}
    spot = utils.Factor('EquityPrice', ('EQ',))
    stoch, add = c.find_models([spot])
    proc_key = utils.Factor('HestonNandiImpliedSpotModel', ('EQ',))
    assert stoch.get(proc_key) == spot, 'EquityPrice not routed to HestonNandiImpliedSpotModel'
    assert add.get(proc_key) == utils.Factor('HestonNandiModelParameters', ('EQ',))
    assert utils.check_tuple_name(proc_key) in c.params['Price Models'], 'no dummy Price Models entry'


# ---------------------------------------------------------------------------
# THE CLOCK
# ---------------------------------------------------------------------------

def test_nsub_business_and_calendar_daily_grid():
    p_bd, _ = _make(120, 8)
    assert (p_bd.n_sub == 1).all(), 'business-daily grid should give n_sub == 1'
    assert torch.allclose(p_bd.f[1:], torch.ones_like(p_bd.f[1:])), 'business-daily grid should give f==1'
    p_cal, _ = _make(120, 8, tg=_time_grid(120, year_per_day=1.0 / 365.25))
    assert (p_cal.n_sub == 1).all(), 'calendar-daily production grid should still give n_sub == 1'
    assert abs(float(p_cal.f[1]) - 252.0 / 365.25) < 1e-9, 'calendar step should be f≈0.69 business days'


def test_nsub_two_business_day_bridge_analytic():
    # dt_arr = [0, 2·dt_c] ⇒ n_sub = [1, 2]; the aggregate variance is h0 + E[h_1] with
    # E[h_1] = ω + α + ψ·h0 (the HN mean recursion).
    p, sh = _make(2, 200000, tg=_time_grid(2, day_step=2.0), seed=13)
    assert list(p.n_sub) == [1, 2], f'expected n_sub [1,2], got {list(p.n_sub)}'
    spot = p.generate(sh)
    d = spot.log()[1] - spot.log()[0]
    psi = float(_SP.beta + _SP.alpha * _SP.gamma ** 2)
    v_analytic = H0_TS + (float(_SP.omega) + float(_SP.alpha) + psi * H0_TS)
    assert abs(d.var().item() / v_analytic - 1.0) < 0.02, \
        f'2-bd aggregate variance {d.var().item():.4e} != analytic {v_analytic:.4e}'


def test_clock_grid_invariant_annualized_vol():
    # Fractional trading clock ⇒ grid-invariant annualized vol: the business-daily grid (f=1) and
    # the calendar-daily production grid (f≈0.69) both annualize (each by 1/dt_grid) to ≈30%.
    T, B = 800, 4000
    vols = {}
    for name, ypd in (('business', DT_C), ('calendar', 1.0 / 365.25)):
        p, sh = _make(T, B, tg=_time_grid(T, year_per_day=ypd), param={**IMPLIED_PARAM, 'H0': H0_STAT})
        logspot = p.generate(sh).log()
        r = (logspot[1:] - logspot[:-1]).numpy()[100:]
        vols[name] = float(np.sqrt(r.var() / ypd))
        assert abs(vols[name] - 0.30) < 0.01, f'{name}-clock annualized vol {vols[name]:.4f} off 30%'
    assert abs(vols['business'] - vols['calendar']) < 0.01, \
        f'vol not grid-invariant: business {vols["business"]:.4f} vs calendar {vols["calendar"]:.4f}'


# ---------------------------------------------------------------------------
# (c) hn_garch ORACLE
# ---------------------------------------------------------------------------

def test_oracle_terminal_moments():
    # Business-daily grid (f=1) ⇒ the process IS the exact hn_garch daily recursion (r=0). The
    # simulated T-1-step terminal log-return matches the closed-form hn_moments within MC error.
    T, B = 40, 60000
    p, sh = _make(T, B, seed=7)                                    # carry=0 (r_t==q_t)
    R = (p.generate(sh).log()[T - 1] - float(np.log(100.0)))
    k1, k2, k3, k4 = hn_garch.hn_moments(_hn_params(), T - 1, H0_TS)
    se_mean = R.std().item() / np.sqrt(B)
    assert abs(R.mean().item() - k1) < 4.0 * se_mean, f'mean {R.mean().item():.4e} != oracle {k1:.4e}'
    assert abs(R.var().item() / k2 - 1.0) < 0.03, f'var {R.var().item():.4e} != oracle {k2:.4e}'
    assert abs(skew(R.numpy()) - k3) < 0.1, f'skew {skew(R.numpy()):.3f} != oracle {k3:.3f}'
    assert abs(kurtosis(R.numpy(), fisher=True) - k4) < 0.35, 'excess kurtosis off the oracle'
    assert k3 < -0.5, 'leverage cross-term should give a clearly negative skew (test has power)'


def test_oracle_call_prices():
    # E[(S_{T-1}-K)+] (r=0 ⇒ undiscounted) matches hn_call at several strikes within MC error.
    T, B = 40, 60000
    p, sh = _make(T, B, seed=11)
    ST = p.generate(sh)[T - 1]
    for K in (95.0, 100.0, 105.0):
        oracle = float(hn_garch.hn_call(100.0, K, _hn_params(), T - 1, H0_TS))
        payoff = torch.clamp(ST - K, min=0.0)
        sim, se = payoff.mean().item(), payoff.std().item() / np.sqrt(B)
        assert abs(sim - oracle) < 4.0 * se, f'call K={K}: sim {sim:.5f} != oracle {oracle:.5f} ({abs(sim-oracle)/se:.1f} se)'


def test_oracle_terminal_cdf():
    # Q(R_{T-1} <= b) matches the exact Fourier-inversion hn_cdf_logret across the body.
    T, B = 40, 80000
    p, sh = _make(T, B, seed=5)
    R = (p.generate(sh).log()[T - 1] - float(np.log(100.0))).numpy()
    for b in (-0.05, 0.0, 0.05):
        oracle = float(hn_garch.hn_cdf_logret(b, _hn_params(), T - 1, H0_TS))
        sim = float((R <= b).mean())
        se = float(np.sqrt(oracle * (1 - oracle) / B))
        assert abs(sim - oracle) < 4.0 * se, f'cdf b={b:+.2f}: sim {sim:.5f} != oracle {oracle:.5f}'


# ---------------------------------------------------------------------------
# (b) PRICE-space martingale
# ---------------------------------------------------------------------------

def test_price_martingale_flat():
    # carry=0 ⇒ the discounted spot is E[S_t]=S_0 at every horizon (the -½h Gaussian convexity makes
    # the PRICE a martingale by construction). A LOG-space E[Δlog S]=−½h would look zero-drift while
    # the PRICE drifts +½·var (the Jensen drift the platinum GARCH lesson warns of).
    T, B = 120, 60000
    p, sh = _make(T, B, seed=3)
    spot = p.generate(sh)
    for t in (1, T // 2, T - 1):
        ratio = spot[t] / spot[0]
        se = ratio.std().item() / np.sqrt(B)
        assert abs(ratio.mean().item() - 1.0) < 4.0 * se, \
            f'price not a martingale at t={t}: E[S_t/S_0]={ratio.mean().item():.5f}'


def test_price_martingale_by_state():
    # PRICE-space martingale conditional on the revealed variance decile (log h_t ∈ F_t), pooled over
    # the horizon: E[S_{t+1}/S_t − 1 | log h_t bucket] = 0. The LOG-space analogue is identically
    # zero and so blind to the Jensen drift — this is the test with power.
    T, B = 140, 60000
    p, sh = _make(T, B, param={**IMPLIED_PARAM, 'H0': H0_STAT}, seed=11)
    spot = p.generate(sh)
    gross = (spot[1:] / spot[:-1]).reshape(-1) - 1.0
    h = p.last_log_h[:-1].reshape(-1)
    edges = torch.quantile(h, torch.linspace(0.0, 1.0, 11, dtype=DTYPE))
    worst = 0.0
    for i in range(10):
        m = (h >= edges[i]) & (h <= edges[i + 1])
        se = gross[m].std().item() / np.sqrt(int(m.sum()))
        worst = max(worst, abs(gross[m].mean().item()) / se)
    assert worst < 3.5, f'price-martingale-by-state violated: worst |E[gross-1|log h decile]|/se={worst:.2f}'


def test_risk_neutral_drift_from_curves():
    # The risk-neutral drift is the underlying's OWN cost of carry: r_t (equity zero) − q_t (dividend),
    # integrated over calendar time (ACT/365, the curve's own daycount). E[S_T/S_0] = exp(∫(r-q)).
    T, B = 120, 60000
    r, q = 0.05, 0.01
    p, sh = _make(T, B, tg=_time_grid(T, year_per_day=1.0 / 365.25), r_level=r, q_level=q)
    ST = p.generate(sh)[T - 1]
    ratio = (ST / 100.0)
    t_years = utils.get_day_count_accrual(REF_DATE, float(T), utils.DAYCOUNT_ACT365)
    target = float(np.exp((r - q) * t_years))
    se = ratio.std().item() / np.sqrt(B)
    assert abs(ratio.mean().item() - target) < 4.0 * se, \
        f'E[S_T/S_0]={ratio.mean().item():.5f} != exp((r-q)·T)={target:.5f}'


# ---------------------------------------------------------------------------
# NO-LOOKAHEAD (log h_t is F_t-measurable)
# ---------------------------------------------------------------------------

def test_no_lookahead():
    T, B, k = 40, 128, 20
    p, sh = _make(T, B, seed=1)
    Z = sh.t_random_numbers[0, :T].clone()

    def run(Zin):
        sh.t_random_numbers[0, :T] = Zin
        p.generate(sh)
        return p.last_log_h.clone()

    base = run(Z)
    Zm = Z.clone(); Zm[k, 0] = Zm[k, 0] + 1.5
    pert = run(Zm)
    assert torch.equal(pert[:k], base[:k]), 'log h leaked future information (s < k changed)'
    assert not torch.equal(pert[k, 0], base[k, 0]), 'log h_k did not respond to the step-k shock'
    other = torch.arange(1, B)
    assert torch.equal(pert[:, other], base[:, other]), 'unperturbed paths changed'


# ---------------------------------------------------------------------------
# (d) reveal / fork / reseed
# ---------------------------------------------------------------------------

def test_privileged_layout_and_reveal_width():
    T, B, B2 = 20, 8, 12
    po, sho = _make(T, B, seed=3)
    sho.t_Scenario_Buffer[po.factor_key] = po.generate(sho)
    assert HestonNandiImpliedSpotModel.privileged_layout(None) == {'log_h': 1}
    assert sum(HestonNandiImpliedSpotModel.privileged_layout(None).values()) + 1 == 2  # log_h(1)+price(1)
    segs = po.reveal_state_at(5, sho.t_Scenario_Buffer)
    assert [k for _, k in segs] == [REVEAL_SUFFICIENT, REVEAL_CONTINUOUS], 'log_h first, price last'
    assert segs[0][0].shape == (1, B) and segs[1][0].shape == (1, B)
    # privileged_factors (T,B,1)
    assert po.privileged_factors(sho.t_Scenario_Buffer[po.factor_key])['log_h'].shape == (T, B, 1)
    # inner mode packs (1,B,B2)
    pi, shi = _make(T, B, sub=B2, seed=4)
    shi.t_Scenario_Buffer[pi.factor_key] = pi.generate(shi)
    in_segs = pi.reveal_state_at(5, shi.t_Scenario_Buffer)
    assert in_segs[0][0].shape == (1, B, B2) and in_segs[1][0].shape == (1, B, B2)


def test_inner_fork_seed_fidelity():
    T, B, B2 = 60, 400, 300
    po, sho = _make(T, B, seed=5)
    sho.t_Scenario_Buffer[po.factor_key] = po.generate(sho)
    t = 30
    seed = po.inner_fork_seed(po.factor_key, sho.t_Scenario_Buffer, t)
    key = (po.factor_key, 'h0_inner')
    assert set(seed) == {key} and seed[key].shape == (B,)
    assert torch.allclose(seed[key], po.last_log_h[t].exp()), 'h0_inner != exp(outer log h_t)'

    pi, shi = _make(T, B, sub=B2, seed=17)
    shi.t_Scenario_Buffer[key] = seed[key]
    pi.generate(shi)
    # inner h at the fork start equals the passed seed EXACTLY (on the middle axis, ⊗ B2)
    assert torch.equal(pi.last_log_h[0], seed[key].view(B, 1).expand(B, B2).log()), \
        'inner h0 != passed h0_inner (fork not landing on the middle axis)'


def test_outer_reseed_terminal_h():
    T, B = 30, 200
    p, sh = _make(T, B, seed=9)
    p.generate(sh)
    reseed = p.outer_reseed()
    key = (p.factor_key, 'h0_outer')
    assert set(reseed) == {key}
    assert torch.equal(reseed[key], p.last_log_h[-1].exp()), 'outer_reseed != terminal h'


@pytest.mark.parametrize('year_per_day', [DT_C, 1.0 / 365.25])      # business-daily f=1 AND the
def test_reseed_from_path_forward_replay(year_per_day):             # production calendar grid f≈0.69:
    # Observed-path replay reruns the HN variance recursion on the REALIZED returns; with the same
    # carry/convexity the innovation is recovered exactly, so the replayed log h matches the forward
    # sim to float precision. The calendar-grid row is the audit's regression gate: the forward
    # update must apply the SAME fractional blend as the replay (unblended forward desyncs by
    # max|Δlog h|≈1.5 at f≈0.69 — the platinum clock trap, caught 2026-07-23).
    T, B = 120, 800
    tg = _time_grid(T, year_per_day=year_per_day)
    pf, shf = _make(T, B, tg=tg, seed=5)
    path = pf.generate(shf)
    logh_fwd = pf.last_log_h.clone()
    pr, shr = _make(T, B, tg=tg, seed=99)                           # different seed: replay must not depend on it
    pr.reseed_from_path(path, shr)
    logh_re = shr.t_Scenario_Buffer[(pr.factor_key, 'hn_log_h')].squeeze(1)
    assert torch.allclose(logh_fwd, logh_re, atol=1e-10), \
        f'reseed desync: max|Δlog h|={ (logh_fwd - logh_re).abs().max().item():.2e}'
    assert torch.allclose(shr.t_Scenario_Buffer[(pr.factor_key, 'h0_outer')], logh_re[-1].exp())


def test_revealed_and_calibrated_annual_vol():
    p, sh = _make(20, 64)
    # long-run annualized vol √((ω+α)/(1−ψ)/dt_c) ≈ 30% (the calibration target)
    assert abs(p.calibrated_annual_vol() - 0.30) < 0.005, p.calibrated_annual_vol()
    # revealed σ_t at the long-run variance annualizes to ≈30% too
    log_h = torch.log(torch.tensor([H0_STAT], dtype=DTYPE))
    assert abs(float(p.revealed_annual_vol(log_h)) - 0.30) < 0.01


# ---------------------------------------------------------------------------
# (d) inner-MC row is added to tests/test_inner_mc_processes.py; here is the
# standalone shape/init-on-middle-axis check (mirrors the contract).
# ---------------------------------------------------------------------------

def test_inner_mc_shape_and_middle_axis():
    B, B2, T = 5, 8, 10
    tg = _time_grid(T)
    one = torch.ones(1, 1, dtype=DTYPE)
    sh = CMC_State_Inner(cholesky=torch.eye(1, dtype=DTYPE), static_buffer={}, batch_size=B, one=one,
                         mcmc_sims=0, report_currency=None, seed=42, job_id=0, num_jobs=1,
                         simulation_sub_batch=B2)
    sh.reset_inner(num_factors=1, time_grid=tg)
    p = HestonNandiImpliedSpotModel(factor=types.SimpleNamespace(param={}), param=None,
                                    implied_factor=types.SimpleNamespace(param=dict(IMPLIED_PARAM)))
    p.factor_type = 'EquityPrice'
    p.factor_key = utils.Factor('EquityPrice', ('EQ',))
    rc, rf, rcur = _rate_code('R', 0.0)
    sh.t_Static_Buffer[rf] = rcur
    p.r_t = p.q_t = rc
    spot0 = torch.linspace(80.0, 120.0, B, dtype=DTYPE)            # distinct per outer path
    p.precalculate(REF_DATE, tg, spot0, sh, process_ofs=0)
    out = p.generate(sh)
    assert out.shape == (T, B, B2), f'expected (T,B,B2), got {tuple(out.shape)}'
    assert torch.isfinite(out).all()
    level0 = out[0].mean(dim=-1)                                   # per-outer (B,)
    assert (level0[1:] - level0[:-1] >= -1e-6).all(), \
        't=0 level not monotone in per-outer init — init landed on the wrong axis'


# ---------------------------------------------------------------------------
# (f) DEDUPE: one leaf for both the pricer (static) and the process (implied)
# ---------------------------------------------------------------------------

BASE_HN = pd.Timestamp('2024-06-28')


def _dedupe_calc():
    """A Credit_Monte_Carlo whose EquityPrice.EQ is routed to HestonNandiImpliedSpotModel AND is
    referenced by a barrier deal with SpotModel='HestonNandi' (the OSS pricer pulls
    HestonNandiModelParameters.EQ in as a static dependent factor). update_factors builds the factor
    state (greeks on) — the collision point. Returns (calc, shared)."""
    hn = dict(IMPLIED_PARAM)
    hn.pop('Steps_Per_Year', None)
    pf = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.02]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, 0.01], [5.0, 0.01]])},
        'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.25] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'HestonNandiModelParameters.EQ': dict(hn, Property_Aliases=None),
    }
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE_HN
    c.params['Price Factors'] = pf
    c.params['Price Models'] = {}
    # route the equity to the HN implied SPOT process
    c.params['Model Configuration'].append('EquityPrice', (), 'HestonNandiImpliedSpotModel')

    horizon = 30
    bdates = [BASE_HN + pd.Timedelta(days=d) for d in range(1, horizon + 1)]
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 100.0,
        'Expiry_Date': BASE_HN + pd.Timedelta(days=horizon), 'Units': 10.0,
        'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 1.0, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [[d, 1.0] for d in bdates],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    val = {'EquityBarrierOption': {'SpotModel': 'HestonNandi'}}
    c.params['Valuation Configuration'] = val
    inst = construct_instrument(field, val)
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': inst}]},
               'Calculation': {'Base_Date': BASE_HN, 'Currency': 'USD'}}

    calc = construct_calculation('Credit_Monte_Carlo', c, device=torch.device('cpu'), prec=DTYPE)
    calc.input_time_grid = '0d 2d(1w) 1m'
    calc.batch_size = 64
    params = {'Run_Date': '2024-06-28', 'Time_grid': '0d 2d(1w) 1m', 'Batch_Size': 64,
              'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'MCMC_Simulations': 0,
              'Tenor_Offset': 0.0, 'CVA': {'Gradient': 'Yes'}}         # a Gradient='Yes' dict ⇒ greeks on
    calc.params = params
    shared = calc.update_factors(params, BASE_HN, 0, 1)
    return calc, shared


def test_dedupe_single_leaf_and_gradient_flow():
    calc, shared = _dedupe_calc()
    spot_key = utils.Factor('EquityPrice', ('EQ',))
    assert spot_key in calc.implied_var, 'EquityPrice not routed to the implied HN process'
    # stoch_factors is keyed by the underlying (spot) factor; the process is the value.
    assert isinstance(calc.stoch_factors[spot_key], HestonNandiImpliedSpotModel), \
        'HN implied process not constructed on the EquityPrice factor'

    fkeys = [utils.Factor('HestonNandiModelParameters', ('EQ', p))
             for p in ('Omega', 'Alpha', 'Beta', 'Gamma_Star', 'H0')]
    # (1) exactly one tensor per param: the static leaf IS the implied leaf, and the pricer's
    #     t_Static_Buffer resolves to that same tensor.
    for fk in fkeys:
        L = calc.implied_var[spot_key][fk]
        assert calc.static_var[fk] is L, f'{fk.name}: static leaf is a DUPLICATE of the implied leaf'
        assert shared.t_Static_Buffer[fk] is L, f'{fk.name}: pricer t_Static_Buffer != the implied leaf'
        assert L.requires_grad, f'{fk.name}: implied leaf not differentiable under greeks'

    # (2) gradient flows to BOTH consumers through the single leaf. Use Gamma_Star (drives the
    #     variance recursion → the scenario path) and H0 (the entry variance → both). update_factors
    #     builds the state; the simulation loop's reset (random numbers) is done here manually.
    shared.reset(calc.num_factors, calc.time_grid)
    proc = calc.stoch_factors[spot_key]
    L = calc.implied_var[spot_key][fkeys[3]]                        # Gamma_Star leaf
    L.grad = None
    path = proc.generate(shared)                                   # SCENARIO path depends on the leaf
    path.sum().backward(retain_graph=True)
    g_scenario = L.grad.clone()
    assert g_scenario is not None and float(g_scenario.abs()) > 0, 'no gradient to the scenario path'

    L.grad = None
    (shared.t_Static_Buffer[fkeys[3]] * 3.0).sum().backward()      # PRICER read of the same leaf
    g_pricer = L.grad.clone()
    assert float(g_pricer.abs()) > 0, 'no gradient to the pricer path'

    # combined loss accumulates BOTH into the one leaf (torch sums path contributions)
    L.grad = None
    path2 = proc.generate(shared)
    (path2.sum() + (shared.t_Static_Buffer[fkeys[3]] * 3.0).sum()).backward()
    assert float(L.grad.abs()) > 0, 'combined backward produced no gradient on the shared leaf'
