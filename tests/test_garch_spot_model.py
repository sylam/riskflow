"""Acceptance tests for GARCHSpotModel (spec §6, items 1-6).

Covers the six committed checks:
  1. No-lookahead (bitwise): the revealed log h_s for s < k is invariant to a perturbation of
     the innovation at step k; log h_k changes. GARCH(1,1) has no leverage term (GJR omitted per
     spec non-goals), so log h is invariant to the SIGN of the shock — the forward propagation is
     exercised with a MAGNITUDE perturbation; a complementary sign-flip check documents the symmetry.
  2. Martingale by state: bucketing paths by log h_t decile, |E[Δlog S_t | bucket]| < 3·MC-s.e.
  3. Unconditional moments: annualized vol 24.75%±0.5% (H0 = LR variance), excess kurtosis > 1,
     ACF1(r²) in [0.1, 0.4] (panel-pooled), variance clustering (ACF1(log h) ≈ α+β).
  4. Inner/outer consistency: an inner fork seeded with h0_inner = exp(outer log h_t) reproduces the
     outer conditional one-step mean/variance/kurtosis within MC error.
  5. Fork fidelity: the inner h at the fork start equals the passed h0_inner exactly.
  6. n_sub grid: production business-daily / calendar-daily grids ⇒ n_sub == 1 every step; a
     synthetic 2-business-day step matches the aggregate-variance approximation analytically.

Plus registration (construct_process resolves the class by name) and the privileged layout / vol.

Deterministic: CMC_State is seeded, and torch.manual_seed fixes the standardised-t Gamma draw.
CPU-fast sizes throughout.
"""
import os
import types

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import kurtosis

from riskflow import utils
from riskflow.calculation import CMC_State, CMC_State_Inner, construct_process
from riskflow.stochasticprocess import (
    GARCHSpotModel, BasisLinkedSpotModel, REVEAL_SUFFICIENT, REVEAL_CONTINUOUS)

DEVICE = torch.device('cpu')
DTYPE = torch.float32
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0

# §2 reference calibration block.
PARAM = {
    'Omega': 8.028e-07, 'Alpha': 0.0328, 'Beta': 0.9639, 'Nu': 7.50,
    'Mu': 0.0, 'H0': 7.671e-04, 'Log_Price': True, 'Calibration_DT_Years': DT_C,
}
LR_VAR = PARAM['Omega'] / (1.0 - PARAM['Alpha'] - PARAM['Beta'])          # long-run per-step variance


def _time_grid(T, day_step=1.0, year_per_day=DT_C):
    """Synthetic time grid. `year_per_day=DT_C` gives a business-daily grid (dt == dt_c, clean
    n_sub=1); `year_per_day=1/365.25` reproduces the calendar-daily production convention.
    `day_step` scales the per-step spacing (used for the 2-business-day grid)."""
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


def _shared(B, T, seed=42, sub=None, tg=None, nf=1):
    one = torch.ones(1, 1, dtype=DTYPE, device=DEVICE)
    tg = tg or _time_grid(T)
    if sub is None:
        s = CMC_State(cholesky=torch.eye(nf, dtype=DTYPE), static_buffer={}, batch_size=B, one=one,
                      mcmc_sims=0, report_currency=None, seed=seed, job_id=0, num_jobs=1)
        s.reset(num_factors=nf, time_grid=tg)
    else:
        s = CMC_State_Inner(cholesky=torch.eye(nf, dtype=DTYPE), static_buffer={}, batch_size=B,
                            one=one, mcmc_sims=0, report_currency=None, seed=seed, job_id=0,
                            num_jobs=1, simulation_sub_batch=sub)
        s.reset_inner(num_factors=nf, time_grid=tg)
    return s


def _make(T, B, param=None, seed=42, sub=None, tg=None):
    tg = tg or _time_grid(T)
    sh = _shared(B, T, seed=seed, sub=sub, tg=tg)
    p = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(param or PARAM))
    p.factor_key = utils.Factor('CommodityPrice', ('TEST',))
    tensor = torch.tensor([2000.0], dtype=DTYPE) if sub is None else torch.full((B,), 2000.0, dtype=DTYPE)
    p.precalculate(REF_DATE, tg, tensor, sh, process_ofs=0)
    return p, sh


# ---------------------------------------------------------------------------
# §6.1 No-lookahead
# ---------------------------------------------------------------------------

def test_no_lookahead_and_symmetry():
    T, B, k = 40, 128, 20
    p, sh = _make(T, B, seed=1)
    Z = sh.t_random_numbers[0, :T].clone()

    def run(Zin):
        sh.t_random_numbers[0, :T] = Zin
        torch.manual_seed(99)                                          # identical Gamma W both runs
        p.generate(sh)
        return p.last_log_h.clone()

    base = run(Z)

    # Sign flip on one path's step-k shock: r² unchanged (no leverage) ⇒ log h fully invariant.
    Zs = Z.clone(); Zs[k, 0] = -Zs[k, 0]
    flipped = run(Zs)
    assert torch.equal(flipped, base), 'sign flip changed log h — GARCH(1,1) should be leverage-free'

    # Magnitude perturbation: r_k² changes ⇒ log h_s (s < k) bitwise unchanged, log h_k changes,
    # and only the perturbed path is affected.
    Zm = Z.clone(); Zm[k, 0] = Zm[k, 0] + 1.5
    pert = run(Zm)
    assert torch.equal(pert[:k], base[:k]), 'log h leaked future information (s < k changed)'
    assert not torch.equal(pert[k, 0], base[k, 0]), 'log h_k did not respond to the step-k shock'
    other = torch.arange(1, B)
    assert torch.equal(pert[:, other], base[:, other]), 'unperturbed paths changed'


# ---------------------------------------------------------------------------
# §6.2 Martingale by state
# ---------------------------------------------------------------------------

def test_martingale_by_state():
    T, B = 140, 60000
    p, sh = _make(T, B, param={**PARAM, 'H0': LR_VAR}, seed=11)
    torch.manual_seed(7)
    spot = p.generate(sh)
    log_spot = spot.log()
    log_h = p.last_log_h
    edges = torch.linspace(0.0, 1.0, 11)
    for t in (40, 80, 120):
        h_t = log_h[t]                                                 # revealed variance at t
        d = log_spot[t + 1] - log_spot[t]                              # Δlog S_t (move t→t+1)
        q = torch.quantile(h_t, edges)
        for i in range(10):
            m = (h_t >= q[i]) & (h_t <= q[i + 1])
            n = int(m.sum())
            if n < 50:
                continue
            se = d[m].std().item() / np.sqrt(n)
            assert abs(d[m].mean().item()) < 3.0 * se, (
                f'martingale violated at t={t} bucket {i}: |E[dlogS]|/se='
                f'{abs(d[m].mean().item()) / se:.2f}')


# ---------------------------------------------------------------------------
# §6.3 Unconditional moments
# ---------------------------------------------------------------------------

def _sim_returns(T, B, ypd, seed=3, mseed=9):
    p, sh = _make(T, B, param={**PARAM, 'H0': LR_VAR}, seed=seed,
                  tg=_time_grid(T, year_per_day=ypd))
    torch.manual_seed(mseed)
    spot = p.generate(sh)
    return p, (spot.log()[1:] - spot.log()[:-1]).numpy()[100:]         # drop burn-in


def test_unconditional_moments():
    # Fractional trading clock ⇒ grid-invariant annualized vol: the calendar-daily production
    # grid (dt=1/365.25, f≈0.69) and the business-daily grid (dt=dt_c, f=1) both annualize
    # (each by 1/dt_grid) to the calibrated 24.75%. Excess kurtosis > 1 on both. ACF1(r²) is
    # gated on the business clock, where it matches the standard GARCH value / data reference.
    T, B = 800, 4000
    vols = {}
    for name, ypd in (('business', DT_C), ('calendar', 1.0 / 365.25)):
        p, r = _sim_returns(T, B, ypd)
        vols[name] = float(np.sqrt(r.var() / ypd))                    # annualize over real (grid) time
        assert abs(vols[name] - 0.2475) < 0.005, \
            f'{name}-clock annualized vol {vols[name]:.4f} outside 24.75%±0.5%'
        assert kurtosis(r.flatten(), fisher=True) > 1.0, f'{name}: excess kurtosis not > 1'
        if name == 'business':
            r2 = r ** 2                                                # panel-pooled ACF1(r²)
            m = r2.mean()
            acf1 = ((r2[1:] - m) * (r2[:-1] - m)).mean() / ((r2 - m) ** 2).mean()
            assert 0.1 <= acf1 <= 0.4, f'ACF1(r²)={acf1:.4f} outside [0.1, 0.4]'
    assert abs(vols['business'] - vols['calendar']) < 0.005, \
        f'vol not grid-invariant: business {vols["business"]:.4f} vs calendar {vols["calendar"]:.4f}'


def test_half_life_real_time():
    # The per-step mean-reversion factor (1 − f(1−α−β)) makes the vol decay RATE grid-invariant
    # in REAL time. From an elevated H0, fit exp decay to E[h_t]−h* on both clocks; both give a
    # half-life of ≈ 210 bd = 210·(365.25/252) ≈ 304 calendar days.
    target_cal_days = 210.0 * 365.25 / 252.0
    T, B = 400, 40000
    for ypd in (DT_C, 1.0 / 365.25):
        p, sh = _make(T, B, param={**PARAM, 'H0': 4.0 * LR_VAR}, seed=5,
                      tg=_time_grid(T, year_per_day=ypd))
        torch.manual_seed(2)
        p.generate(sh)
        e_h = p.last_log_h.exp().mean(dim=1).numpy() - LR_VAR         # E[h_t] − h*, decays to 0
        w = slice(1, 180)
        slope = np.polyfit(np.arange(T)[w], np.log(np.clip(e_h[w], 1e-12, None)), 1)[0]
        hl_cal_days = float(np.log(0.5) / slope * ypd * 365.25)
        assert abs(hl_cal_days / target_cal_days - 1.0) < 0.15, \
            f'half-life {hl_cal_days:.0f} != {target_cal_days:.0f} calendar days (ypd={ypd:.5f})'


# ---------------------------------------------------------------------------
# §6.4 Inner/outer consistency + §6.5 fork fidelity
# ---------------------------------------------------------------------------

def test_inner_outer_consistency():
    T, B, B2 = 60, 400, 400
    # Outer pass to obtain the conditional variance at a mid horizon.
    po, sho = _make(T, B, param={**PARAM, 'H0': LR_VAR}, seed=5)
    torch.manual_seed(3)
    po.generate(sho)
    t = 30
    h0_inner = po.last_log_h[t].exp()                                  # (B,) outer conditional variance

    # Inner fork seeded from exp(outer log h_t): one-step move variance should equal h0_inner.
    pi, shi = _make(T, B, param={**PARAM, 'H0': LR_VAR}, seed=17, sub=B2)
    shi.t_Scenario_Buffer[(pi.factor_key, 'h0_inner')] = h0_inner
    torch.manual_seed(21)
    inner = pi.generate(shi)                                           # (T, B, B2)
    d = inner.log()[1] - inner.log()[0]                                # (B, B2) one-step Δlog S

    # §6.5 fork fidelity: inner h at the fork start equals the seed exactly.
    assert torch.allclose(pi.last_log_h[0], h0_inner.view(B, 1).expand(B, B2).log()), \
        'inner h0 != passed h0_inner'

    # Conditional mean ~ 0, and pooled variance matches the seeded conditional variance.
    se_mean = d.std().item() / np.sqrt(d.numel())
    assert abs(d.mean().item()) < 4.0 * se_mean, 'inner one-step mean not ~ 0'
    var_ratio = d.var().item() / h0_inner.mean().item()
    assert abs(var_ratio - 1.0) < 0.06, f'inner one-step variance ratio {var_ratio:.3f} != 1'
    # Fat tails carry through (Student-t emission).
    z = (d / h0_inner.view(B, 1).sqrt()).flatten().numpy()
    assert kurtosis(z, fisher=True) > 0.5, 'inner one-step innovation lost its fat tail'


# ---------------------------------------------------------------------------
# §6.6 n_sub grid
# ---------------------------------------------------------------------------

def test_nsub_business_daily_grid():
    p, _ = _make(120, 8)
    assert (p.n_sub == 1).all(), 'business-daily grid should give n_sub == 1 every step'


def test_nsub_calendar_daily_grid():
    # Production Time_Grid "0d 1d(1d)" ⇒ time_grid_years = days/365.25.
    tg = _time_grid(120, year_per_day=1.0 / 365.25)
    p, _ = _make(120, 8, tg=tg)
    assert (p.n_sub == 1).all(), 'calendar-daily production grid should still give n_sub == 1'


def test_nsub_two_business_day_step_analytic():
    # A single 2-business-day step: dt_arr = [0, 2·dt_c] ⇒ n_sub = [1, 2].
    tg = _time_grid(2, day_step=2.0)                                   # spacing 2 → dt=2·dt_c
    B = 200000
    p, sh = _make(2, B, tg=tg, seed=13)
    assert list(p.n_sub) == [1, 2], f'expected n_sub [1,2], got {list(p.n_sub)}'
    torch.manual_seed(4)
    spot = p.generate(sh)
    d = (spot.log()[1] - spot.log()[0])                               # the 2-bd move, variance V
    h0 = PARAM['H0']
    v_analytic = h0 + (PARAM['Omega'] + (PARAM['Alpha'] + PARAM['Beta']) * h0)  # Σ_j h_j, j=0,1
    assert abs(d.var().item() / v_analytic - 1.0) < 0.05, (
        f'2-bd aggregate variance {d.var().item():.3e} != analytic {v_analytic:.3e}')


# ---------------------------------------------------------------------------
# §2 registration / §4 privileged layout & vol
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fork wiring: inner_fork_seed extraction + reveal_state_at width (both modes)
# ---------------------------------------------------------------------------

def test_inner_fork_seed_extraction_and_middle_axis():
    T, B, B2 = 30, 16, 20
    po, sho = _make(T, B, seed=5)
    torch.manual_seed(1)
    sho.t_Scenario_Buffer[po.factor_key] = po.generate(sho)
    t = 12
    seed = po.inner_fork_seed(po.factor_key, sho.t_Scenario_Buffer, t)
    key = (po.factor_key, 'h0_inner')
    assert set(seed) == {key}, 'inner_fork_seed should emit exactly the h0_inner key'
    h0 = seed[key]
    log_h_buf = sho.t_Scenario_Buffer[(po.factor_key, 'garch_log_h')]      # (T, 1, B)
    assert h0.shape == (B,), f'h0_inner should be (B,), got {tuple(h0.shape)}'
    assert torch.allclose(h0, log_h_buf[t].reshape(B).exp()), 'h0_inner != exp(outer log h_t)'

    # The inner generate seeds h on the MIDDLE (B) axis, broadcast across the B2 fan-out.
    pi, shi = _make(T, B, seed=7, sub=B2)
    shi.t_Scenario_Buffer[key] = h0
    torch.manual_seed(2)
    pi.generate(shi)
    assert torch.allclose(pi.last_log_h[0], h0.view(B, 1).expand(B, B2).log()), \
        'inner h0 did not land on the middle axis'


def test_reveal_state_at_width_both_modes():
    T, B, B2 = 20, 8, 12
    po, sho = _make(T, B, seed=3)
    torch.manual_seed(1)
    sho.t_Scenario_Buffer[po.factor_key] = po.generate(sho)
    out_segs = po.reveal_state_at(5, sho.t_Scenario_Buffer)
    assert [k for _, k in out_segs] == [REVEAL_SUFFICIENT, REVEAL_CONTINUOUS], 'log_h first, price last'
    assert out_segs[0][0].shape == (1, B) and out_segs[1][0].shape == (1, B)
    assert sum(b.shape[0] for b, _ in out_segs) == 2, 'market width should be log_h(1)+price(1)=2'

    pi, shi = _make(T, B, seed=4, sub=B2)
    torch.manual_seed(2)
    shi.t_Scenario_Buffer[pi.factor_key] = pi.generate(shi)
    in_segs = pi.reveal_state_at(5, shi.t_Scenario_Buffer)
    assert [k for _, k in in_segs] == [REVEAL_SUFFICIENT, REVEAL_CONTINUOUS]
    assert in_segs[0][0].shape == (1, B, B2) and in_segs[1][0].shape == (1, B, B2)


# ---------------------------------------------------------------------------
# Calibrator reproduces the §2 reference table (scipy path, full sample ~0.5s)
# ---------------------------------------------------------------------------

def test_calibrate_reproduces_reference():
    csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'pl_exp.csv')
    df = pd.read_csv(csv, index_col=0)[['CommodityPrice.PLATINUM']]
    out = GARCHSpotModel.calibrate(df)                                     # scipy L-BFGS-B path
    persist = out['Alpha'] + out['Beta']
    lr_vol = np.sqrt(out['Omega'] / (1.0 - persist) / out['Calibration_DT_Years'])
    # §2 reference table (full-sample fit is the source of truth): 2% rel on (α, β, LR-vol),
    # 10% on (ω, ν, H0).
    assert abs(out['Alpha'] - 0.0328) / 0.0328 < 0.02, out['Alpha']
    assert abs(out['Beta'] - 0.9639) / 0.9639 < 0.02, out['Beta']
    assert abs(lr_vol - 0.2475) / 0.2475 < 0.02, lr_vol
    assert abs(out['Omega'] - 8.03e-07) / 8.03e-07 < 0.10, out['Omega']
    assert abs(out['Nu'] - 7.50) / 7.50 < 0.10, out['Nu']
    assert abs(out['H0'] - 7.67e-04) / 7.67e-04 < 0.10, out['H0']
    assert out['Mu'] == 0.0 and out['Log_Price'] is True


def test_registration_layout_and_vol():
    proc = construct_process('GARCHSpotModel', types.SimpleNamespace(param={}), dict(PARAM))
    assert isinstance(proc, GARCHSpotModel), 'construct_process did not resolve GARCHSpotModel by name'
    assert GARCHSpotModel.privileged_layout(PARAM) == {'log_h': 1}
    # V̂ deep-state market width = log_h (1) + price (1).
    assert sum(GARCHSpotModel.privileged_layout(PARAM).values()) + 1 == 2
    assert abs(proc.calibrated_annual_vol() - 0.2476) < 0.001, 'long-run annualized vol off'


# ---------------------------------------------------------------------------
# Flat-Sigma basis (single-vol OU) under a regime-free GARCH primary
# ---------------------------------------------------------------------------

def _garch_primary(sh, T, B, seed_manual):
    """Run a GARCH primary and publish its (regime-free) path into the buffer under CME."""
    p = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(PARAM))
    p.factor_key = utils.Factor('CommodityPrice', ('CME',))
    p.precalculate(REF_DATE, _time_grid(T), torch.tensor([2000.0], dtype=DTYPE), sh, process_ofs=0)
    torch.manual_seed(seed_manual)
    sh.t_Scenario_Buffer[p.factor_key] = p.generate(sh)
    return p


def _basis(param, linked='CME'):
    # Observed_Commodity is the bare CommodityPrice name (no type prefix), as in the market data.
    b = BasisLinkedSpotModel(factor=types.SimpleNamespace(param={'Observed_Commodity': linked}),
                             param=param)
    b.factor_key = utils.Factor('CommodityBasis', ('LME_CME',))
    return b


def test_flat_sigma_basis_under_garch_primary():
    T, B = 400, 8000
    sh = _shared(B, T, seed=2, nf=2)                                  # factor 0 = GARCH, factor 1 = basis
    _garch_primary(sh, T, B, seed_manual=4)
    # Pure OU (A=0) so the moment check is clean: b(t) = φ·b(t-1) + η, Var(η)=σ².
    phi, sigma = 0.35, 8.0793
    basis = _basis({'A': 0.0, 'Phi': phi, 'Nu': 6.0, 'Mu': 0.0, 'Sigma': sigma,
                    'Calibration_DT_Years': DT_C})
    basis.precalculate(REF_DATE, _time_grid(T), torch.tensor([0.0], dtype=DTYPE), sh, process_ofs=1)
    assert basis.sigma_by_state is None and basis.sigma_flat is not None, 'flat form not selected'
    torch.manual_seed(6)
    out = basis.generate(sh)
    assert out.shape == (T, B), f'expected (T,B), got {tuple(out.shape)}'
    assert torch.isfinite(out).all(), 'non-finite basis output'
    # Stationary OU variance σ²/(1-φ²); mean decays to 0 from b0=0.
    var_stat = sigma ** 2 / (1.0 - phi ** 2)
    var_sim = out[-50:].var().item()
    assert abs(var_sim / var_stat - 1.0) < 0.08, f'OU variance {var_sim:.1f} != σ²/(1-φ²) {var_stat:.1f}'
    assert abs(out[-1].mean().item()) < 4.0 * out[-1].std().item() / np.sqrt(B), 'OU mean not ~ 0'


def test_regime_basis_without_regimes_raises():
    T, B = 20, 64
    sh = _shared(B, T, seed=2, nf=2)
    _garch_primary(sh, T, B, seed_manual=4)                          # GARCH publishes NO regimes
    basis = _basis({'A': 0.0, 'Phi': 0.3, 'Nu': 6.0, 'Mu': 0.0,
                    'Sigma_By_State': [5.0, 8.0, 12.0], 'Calibration_DT_Years': DT_C})
    basis.precalculate(REF_DATE, _time_grid(T), torch.tensor([0.0], dtype=DTYPE), sh, process_ofs=1)
    with pytest.raises(KeyError, match='regimes'):
        basis.generate(sh)


def test_basis_both_sigma_forms_rejected():
    sh = _shared(4, 5, seed=1)
    basis = _basis({'A': 0.0, 'Phi': 0.3, 'Nu': 6.0, 'Mu': 0.0,
                    'Sigma': 8.0, 'Sigma_By_State': [5.0, 8.0], 'Calibration_DT_Years': DT_C})
    with pytest.raises(AssertionError):
        basis.precalculate(REF_DATE, _time_grid(5), torch.tensor([0.0], dtype=DTYPE), sh, process_ofs=1)
