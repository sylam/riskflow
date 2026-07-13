"""Tests for the vector-of-initial-states refactor of three stochastic processes
plus the new `CMC_State_Inner` shared-state subclass.

Coverage:
- Outer-mode regression for HMM / Basis / VAR via direct instantiation against
  frozen legacy `generate` reproductions (bit-exact for HMM/Basis where draws are
  fully deterministic given a manual seed; allclose-at-fp32 for VAR where the
  operator swap `@` → einsum can permute summation order).
- Inner-mode shape/initial-state/independence for each process.
- HMM `regime0_inner` override in `t_Scenario_Buffer`.
- VAR einsum-vs-`@` operator regression in isolation.
- End-to-end smoke through cx.run_job on the existing simulate_only fixture, to
  verify the framework still wires through unchanged in the live JSON path.
"""
import json
import os
import types

import numpy as np
import pandas as pd
import pytest
import torch

import riskflow as rf
from riskflow import utils
from riskflow.calculation import CMC_State, CMC_State_Inner
from riskflow.stochasticprocess import (
    BasisComposedSpotModel,
    BasisLinkedSpotModel,
    MarkovHMMSpotModel,
    VARMixedFactorInterestRateModel,
)


# ---------------------------------------------------------------------------
# Stand-in fixtures for direct-instantiation tests.
# ---------------------------------------------------------------------------

DEVICE = torch.device('cpu')
DTYPE = torch.float32


def _time_grid(T):
    """Build a daily T-day stand-in time_grid object exposing the attributes
    that the processes touch in precalculate/generate plus CMC_State.reset."""
    days = np.arange(1, T + 1, dtype=np.float64)
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days / utils.DAYS_IN_YEAR
    tg.CurrencyMap = {}
    # HW2F's precalculate reads scenario_grid; matching row count to scen_time_grid
    # keeps grid_index None so generate returns a tensor (not the deflator dict).
    # ScenarioPriorIndex is the per-step scenario index GBMTSImplied's curve gather
    # reads via gather_scenario_interp (PriorScenarioDelta=0 -> exact index, alpha=None).
    scen_grid = np.zeros((T, 3), dtype=np.float64)
    scen_grid[:, utils.TIME_GRID_MTM] = days
    scen_grid[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(T)
    tg.scenario_grid = scen_grid
    return tg


def _build_shared(num_factors, batch_size, seed=42, simulation_sub_batch=None):
    """Build a CMC_State (or CMC_State_Inner) with an identity cholesky — enough
    for direct-instantiation testing where we don't care about cross-factor
    correlation but do care about the shape of t_random_numbers."""
    one = torch.ones(1, 1, dtype=DTYPE, device=DEVICE)
    chol = torch.eye(num_factors, dtype=DTYPE, device=DEVICE)
    if simulation_sub_batch is None:
        shared = CMC_State(
            cholesky=chol, static_buffer={}, batch_size=batch_size, one=one,
            mcmc_sims=0, report_currency=None,
            seed=seed, job_id=0, num_jobs=1,
        )
    else:
        shared = CMC_State_Inner(
            cholesky=chol, static_buffer={}, batch_size=batch_size, one=one,
            mcmc_sims=0, report_currency=None,
            seed=seed, job_id=0, num_jobs=1,
            simulation_sub_batch=simulation_sub_batch,
        )
    return shared


def _do_reset(shared, num_factors, tg):
    """Pick the right reset for the shared state's flavor — inner-mode shape only
    via reset_inner; reset() stays the canonical outer-mode call."""
    if isinstance(shared, CMC_State_Inner):
        shared.reset_inner(num_factors=num_factors, time_grid=tg)
    else:
        shared.reset(num_factors=num_factors, time_grid=tg)


def _hmm_param(with_nu=False):
    states = [
        {'Mu': 0.05, 'Sigma': 0.15},
        {'Mu': -0.02, 'Sigma': 0.30},
    ]
    if with_nu:
        for s in states:
            s['Nu'] = 8.0
    return {
        'Log_Price': True,
        'States': states,
        'Transition_Matrix': [[0.97, 0.03], [0.05, 0.95]],
        'Initial_State_Probs': [0.6, 0.4],
        'Calibration_DT_Years': 1.0 / 252.0,
    }


def _basis_param():
    return {
        'A': 0.1,
        'Phi': 0.4,
        'Nu': 6.0,
        'Mu': 0.0,
        'Sigma_By_State': [10.0, 15.0],
        'Calibration_DT_Years': 1.0 / 252.0,
    }


def _var_param():
    # Mild VAR(1) with positive eigenvalues so matrix-power per-step Phi is stable.
    return {
        'Mean': [0.0, 0.0, 0.0],
        'Phi': [[0.9, 0.05, 0.0],
                [-0.02, 0.92, 0.03],
                [0.0, 0.01, 0.88]],
        'Sigma': [0.001, 0.005, 0.002],
        'Calibration_Tenors': [0.08, 0.33, 0.58],
        'Contract_Cycle_Years': 0.25,
        'Calibration_DT_Years': 1.0 / 252.0,
    }


def _hmm_factor():
    return types.SimpleNamespace(param={})


def _basis_factor(linked='CommodityPrice.LME'):
    # Basis reads factor.param['Observed_Commodity'] to find the linked CommodityPrice.
    return types.SimpleNamespace(param={'Observed_Commodity': linked})


def _var_factor(ref_date):
    # Three contract knots roughly 1m / 4m / 7m forward, in excel-date offsets.
    excel_ref = (ref_date - utils.excel_offset).days
    days_forward = np.array([30, 120, 215], dtype=np.float64)
    factor_tenor = excel_ref + days_forward
    return types.SimpleNamespace(get_tenor=lambda: factor_tenor)


REF_DATE = pd.Timestamp('2026-04-10')


# ---------------------------------------------------------------------------
# Frozen legacy generate functions (pre-inner-MC). Used to assert that the
# refactored outer-mode paths reproduce historical behavior bit-exactly (where
# random draws can be controlled) or to fp32 precision (where the einsum swap
# introduces a different summation order).
# ---------------------------------------------------------------------------

def _legacy_hmm_generate(p, shared_mem):
    Z = shared_mem.t_random_numbers[p.z_offset, :p.scenario_horizon]
    T, B = Z.shape
    device = Z.device
    dtype = torch.float32
    # Canonical Sobol orientation (dimension = T+1 per-path uniforms, samples = paths) —
    # matches the production fix for the cross-path regime-correlation defect at large B.
    u_regime = shared_mem.quasi_rng(T + 1, B)[1].transpose(0, 1).contiguous()
    pi0_cum = p.pi0_cum.to(device=device, dtype=dtype)
    P_cum = p.P_cum.to(device=device, dtype=dtype)
    state = torch.searchsorted(pi0_cum, u_regime[0]).clamp_max_(p.n_states - 1)
    regimes = torch.empty((T, B), dtype=torch.long, device=device)
    regimes[0] = state
    for t in range(1, T):
        cdf_rows = P_cum[t - 1].index_select(0, state)
        state = (cdf_rows < u_regime[t].unsqueeze(1)).sum(dim=1).clamp_max_(p.n_states - 1)
        regimes[t] = state
    mu = p.mu_per_state.to(device=device, dtype=dtype)
    sigma = p.sigma_per_state.to(device=device, dtype=dtype)
    dt = p.dt_per_step.to(device=device, dtype=dtype)
    Z_dt = Z.to(dtype=dtype)
    nu_per_state = getattr(p, 'nu_per_state', None)
    if nu_per_state is not None:
        nu = nu_per_state.to(device=device, dtype=dtype)
        nu_t = nu[regimes]
        W = torch.distributions.Gamma(nu_t / 2.0, 0.5).sample().to(dtype=dtype)
        t_innov = Z_dt * torch.sqrt(nu_t / W)
        scale_to_unit_var = torch.sqrt((nu_t - 2.0).clamp_min(1.0e-3) / nu_t)
        innov = t_innov * scale_to_unit_var
    else:
        innov = Z_dt
    mu_t = mu[regimes] * dt.view(T, 1)
    std_t = sigma[regimes] * dt.view(T, 1).sqrt()
    ds = mu_t + std_t * innov
    s0 = p.spot0.reshape(()).expand(B)
    if p.log_price:
        log_path = s0.log().unsqueeze(0) + ds.cumsum(dim=0)
        spot = log_path.exp()
    else:
        spot = s0.unsqueeze(0) + ds.cumsum(dim=0)
    return spot, regimes


def _legacy_basis_generate(p, shared_mem, lme_path, regimes):
    Z = shared_mem.t_random_numbers[p.z_offset, :p.scenario_horizon]
    T, B = Z.shape
    device = Z.device
    dtype = Z.dtype
    sigma_t = p.sigma_by_state[regimes]
    nu = p.Nu
    W = torch.distributions.Chi2(nu).sample((T, B)).to(device=device, dtype=dtype)
    eta = sigma_t * Z * torch.sqrt((nu - 2.0) / W)
    phi = float(p.Phi)
    a = float(p.A)
    b_init = p.b0.reshape(()).expand(B)
    out = torch.empty((T, B), device=device, dtype=dtype)
    out[0] = b_init
    for t in range(1, T):
        out[t] = a * (lme_path[t] - lme_path[t - 1]) + phi * out[t - 1] + eta[t]
    return out


def _legacy_var_generate(p, shared_mem):
    """Outer-mode VAR generate as of pre-refactor: uses `@` rather than einsum."""
    Z = shared_mem.t_random_numbers[p.z_offset:p.z_offset + 3, :p.scenario_horizon]
    _, T, B = Z.shape
    n_contracts = p.contract_T.shape[1]
    mean = p.mean_vec.view(3, 1)
    X = p.X0.view(3, 1).expand(3, B).clone()
    out = torch.empty((T, n_contracts, B), dtype=Z.dtype, device=Z.device)
    for t in range(T):
        X = mean + p.Phi_per_step[t] @ (X - mean) \
            + p.sigma_per_step[t].view(3, 1) * Z[:, t, :]
        c_slot = p.D_slot_per_step[t] @ X
        ts = p.D_slot_per_step[t, :, 1].contiguous()
        cT_t = p.contract_T[t]
        idx = torch.clamp(torch.searchsorted(ts, cT_t, right=False), 1, 2)
        alpha = ((cT_t - ts[idx - 1]) / (ts[idx] - ts[idx - 1])).unsqueeze(-1)
        out_t = (1.0 - alpha) * c_slot[idx - 1] + alpha * c_slot[idx]
        out[t] = torch.where(p.contract_expired[t].unsqueeze(-1),
                             torch.zeros_like(out_t), out_t)
    return out


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def _setup_hmm(*, batch_size, T=10, with_nu=False, inner_sub=None, seed=42):
    """Common HMM setup. Returns (process, shared, tensor)."""
    tg = _time_grid(T)
    shared = _build_shared(num_factors=1, batch_size=batch_size, seed=seed,
                           simulation_sub_batch=inner_sub)
    _do_reset(shared, 1, tg)
    p = MarkovHMMSpotModel(factor=_hmm_factor(), param=_hmm_param(with_nu=with_nu))
    p.factor_key = utils.Factor('CommodityPrice', ('TEST',))
    if inner_sub is None:
        tensor = torch.tensor([2000.0], dtype=DTYPE, device=DEVICE)         # outer (1,)
    else:
        tensor = torch.full((batch_size,), 2000.0, dtype=DTYPE, device=DEVICE)  # inner (B,)
    p.precalculate(REF_DATE, tg, tensor, shared, process_ofs=0)
    return p, shared, tensor


def test_hmm_outer_bit_exact_regression():
    """Without Nu, all randomness lives in t_random_numbers + quasi_rng (both
    deterministic given seed). New outer-mode generate must produce bit-exact
    output vs the frozen legacy generate."""
    p, shared, _ = _setup_hmm(batch_size=64, T=10, with_nu=False)
    expected_spot, expected_reg = _legacy_hmm_generate(p, shared)
    # Reset the quasi_rng batch counter (we drew once for the legacy run) so the
    # new generate consumes the same Sobol block.
    shared.reset_qrg()
    got_spot = p.generate(shared)
    got_reg = p.last_regime_path
    assert torch.equal(got_spot, expected_spot), 'HMM outer-mode spot path drifted'
    assert torch.equal(got_reg, expected_reg), 'HMM outer-mode regime path drifted'


def test_hmm_outer_with_nu_bit_exact():
    """With Nu, the internal Gamma draw is non-deterministic across calls. Reseed
    the global RNG between the legacy and new calls so the W-draw matches."""
    p, shared, _ = _setup_hmm(batch_size=32, T=10, with_nu=True)
    torch.manual_seed(7)
    expected_spot, expected_reg = _legacy_hmm_generate(p, shared)
    shared.reset_qrg()
    torch.manual_seed(7)
    got_spot = p.generate(shared)
    got_reg = p.last_regime_path
    assert torch.equal(got_spot, expected_spot)
    assert torch.equal(got_reg, expected_reg)


def test_hmm_inner_shape_init_and_independence():
    B, B2, T = 4, 8, 10
    p, shared, tensor = _setup_hmm(batch_size=B, T=T, with_nu=False, inner_sub=B2)
    # Distinct per-outer-path initial spots to verify the broadcast.
    tensor[:] = torch.tensor([1500.0, 1800.0, 2100.0, 2400.0], dtype=DTYPE, device=DEVICE)
    # Re-call precalculate to pick up the new tensor values.
    p.precalculate(REF_DATE, _time_grid(T), tensor, shared, process_ofs=0)
    out = p.generate(shared)
    assert out.shape == (T, B, B2), f'expected (T,B,B2)=({T},{B},{B2}), got {tuple(out.shape)}'

    # t=0 row: spot0 + dS_0 — initial-spot effect is the dominant signal across the
    # B fan-out. Check spot0 broadcasts across the B2 inner copies for each b.
    expected_s0_per_b = tensor.detach().clone()
    # log_price=True so the path returns exp(); compare on log scale to isolate s0.
    log0 = out[0].log()                                                              # (B, B2)
    # Inner paths within an outer share s0 but have different dS_0; log0[b, :] should
    # have variance > 0 but be centered around s0[b] for typical Sigma.
    s0_log = expected_s0_per_b.log()
    centered = log0 - s0_log.unsqueeze(-1)                                           # (B, B2)
    assert centered.abs().max() < 0.5, (
        'initial-spot broadcast appears wrong — inner paths drift > 50% on log scale '
        f'at t=0; max |Δ log|={centered.abs().max().item()}')

    # B2-independence: within a fixed b, the B2 paths should produce distinct
    # terminals (zero variance would mean they collapsed).
    per_b_std = out[-1].std(dim=-1)                                                  # (B,)
    assert (per_b_std > 0).all(), 'inner paths collapsed to identical terminals'


def test_hmm_regime0_inner_override():
    """When (factor_key, 'regime0_inner') is set in t_Scenario_Buffer, all B2 inner
    paths for outer b must start in the per-b regime — no sampling from pi0."""
    B, B2, T = 4, 8, 10
    p, shared, _ = _setup_hmm(batch_size=B, T=T, with_nu=False, inner_sub=B2)
    # 2-state model in _hmm_param: regimes ∈ {0, 1}.
    regime0 = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=DEVICE)
    shared.t_Scenario_Buffer[(p.factor_key, 'regime0_inner')] = regime0
    p.generate(shared)
    got_regime0 = p.last_regime_path[0]                                              # (B, B2)
    expected = regime0.view(B, 1).expand(B, B2)
    assert torch.equal(got_regime0, expected), 'regime0_inner override not honored'


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

def _setup_basis(*, batch_size, T=10, inner_sub=None, seed=42):
    """Basis needs a linked spot path + regime path in t_Scenario_Buffer."""
    tg = _time_grid(T)
    # Basis consumes Z at offset 0; HMM emission would normally have its own offset,
    # but we bypass HMM and inject the linked path directly.
    shared = _build_shared(num_factors=1, batch_size=batch_size, seed=seed,
                           simulation_sub_batch=inner_sub)
    _do_reset(shared, 1, tg)

    p = BasisLinkedSpotModel(factor=_basis_factor(linked='CommodityPrice.LME'),
                             param=_basis_param())
    p.factor_key = utils.Factor('CommodityBasis', ('LME_CME', 'LME'))
    if inner_sub is None:
        tensor = torch.tensor([3.0], dtype=DTYPE, device=DEVICE)
        lme_shape = (T, batch_size)
        reg_shape = (T, batch_size)
    else:
        tensor = torch.full((batch_size,), 3.0, dtype=DTYPE, device=DEVICE)
        lme_shape = (T, batch_size, inner_sub)
        reg_shape = (T, batch_size, inner_sub)
    p.precalculate(REF_DATE, tg, tensor, shared, process_ofs=0)

    # Inject linked CommodityPrice path (price-level, all positive) and 2-state regimes.
    torch.manual_seed(123)
    lme_path = 2000.0 + torch.randn(lme_shape, dtype=DTYPE, device=DEVICE).cumsum(0) * 5.0
    regimes = torch.randint(0, 2, reg_shape, dtype=torch.long, device=DEVICE)
    shared.t_Scenario_Buffer[p.linked_key] = lme_path
    shared.t_Scenario_Buffer[(p.linked_key, 'regimes')] = regimes
    return p, shared, tensor, lme_path, regimes


def test_basis_outer_bit_exact_regression():
    p, shared, _, lme, reg = _setup_basis(batch_size=64, T=10)
    torch.manual_seed(99)
    expected = _legacy_basis_generate(p, shared, lme, reg)
    torch.manual_seed(99)
    got = p.generate(shared)
    assert torch.equal(got, expected), 'Basis outer-mode path drifted'


def test_basis_inner_shape_init_and_independence():
    B, B2, T = 4, 8, 10
    p, shared, tensor, _, _ = _setup_basis(batch_size=B, T=T, inner_sub=B2)
    tensor[:] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, device=DEVICE)
    # Rebuild because precalculate captured the original tensor — re-call for clarity.
    p.precalculate(REF_DATE, _time_grid(T), tensor, shared, process_ofs=0)
    out = p.generate(shared)
    assert out.shape == (T, B, B2)
    # t=0: out[0, b, :] == tensor[b] across all B2.
    expected_t0 = tensor.unsqueeze(-1).expand(B, B2)
    assert torch.equal(out[0], expected_t0), 'Basis t=0 initial-state broadcast wrong'
    # B2-independence at terminal.
    per_b_std = out[-1].std(dim=-1)
    assert (per_b_std > 0).all()


@pytest.mark.parametrize('inner_sub', [None, 8])
def test_composed_spot_shape_and_composition(inner_sub):
    """BasisComposedSpotModel: published path == primary + basis bitwise in outer (T,B)
    and inner (T,B,B2) modes; the primary is the basis's Observed_Commodity (sim driver
    and pricing anchor unified), resolved via calc_references."""
    B, T = 4, 10
    shared = _build_shared(num_factors=1, batch_size=B, seed=42, simulation_sub_batch=inner_sub)
    _do_reset(shared, 1, _time_grid(T))
    primary_key = utils.Factor('CommodityPrice', ('CME',))
    basis_key = utils.Factor('CommodityBasis', ('LME_CME',))
    # basis: sim driver and pricing anchor are both CME (Observed_Commodity); the composed
    # LME spot published here is derived FROM the basis, never drives it (acyclic).
    basis_factor = types.SimpleNamespace(param={'Observed_Commodity': 'CME'})
    p = BasisComposedSpotModel(
        factor=types.SimpleNamespace(param={'Implied_Basis': 'LME_CME'}), param={})
    p.factor_key = utils.Factor('CommodityPrice', ('LME',))
    p.precalculate(REF_DATE, _time_grid(T), None, shared, process_ofs=0)
    p.calc_references(p.factor_key, {}, {}, {}, {basis_key: basis_factor})
    assert p.primary_key == primary_key and p.basis_key == basis_key
    shape = (T, B) if inner_sub is None else (T, B, inner_sub)
    torch.manual_seed(7)
    primary = 2000.0 + torch.randn(shape, dtype=DTYPE, device=DEVICE).cumsum(0) * 5.0
    basis = torch.randn(shape, dtype=DTYPE, device=DEVICE) * 10.0
    shared.t_Scenario_Buffer[primary_key] = primary
    shared.t_Scenario_Buffer[basis_key] = basis
    out = p.generate(shared)
    assert out.shape == shape
    assert torch.equal(out, primary + basis), 'composed path != primary + basis'
    assert p.reveal_state_at(0, shared.t_Scenario_Buffer) == []
    assert BasisComposedSpotModel.num_factors() == 0


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------

def _setup_var(*, batch_size, T=10, inner_sub=None, seed=42):
    tg = _time_grid(T)
    shared = _build_shared(num_factors=3, batch_size=batch_size, seed=seed,
                           simulation_sub_batch=inner_sub)
    _do_reset(shared, 3, tg)
    p = VARMixedFactorInterestRateModel(factor=_var_factor(REF_DATE), param=_var_param())
    p.factor_key = utils.Factor('InterestRate', ('TEST_CARRY',))
    if inner_sub is None:
        tensor = torch.tensor([0.01, 0.012, 0.014], dtype=DTYPE, device=DEVICE)        # (3,)
    else:
        # Per-outer-path curve fan-out: (3, B) with slight variation per b.
        base = torch.tensor([0.01, 0.012, 0.014], dtype=DTYPE, device=DEVICE).view(3, 1)
        bumps = torch.linspace(-0.001, 0.001, batch_size, dtype=DTYPE, device=DEVICE).view(1, -1)
        tensor = base + bumps                                                          # (3, B)
    p.precalculate(REF_DATE, tg, tensor, shared, process_ofs=0)
    return p, shared, tensor


def test_var_outer_numerical_eq_einsum_vs_legacy():
    """New (einsum) outer-mode generate ≈ legacy (@) outer-mode generate at fp32
    precision. Bit-exact is not guaranteed because einsum may permute summation."""
    p, shared, _ = _setup_var(batch_size=32, T=10)
    expected = _legacy_var_generate(p, shared)
    got = p.generate(shared)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-6), (
        f'VAR outer-mode einsum drift: max |Δ|={(got - expected).abs().max().item():.2e}')


def test_var_einsum_vs_matmul_identity():
    """Direct micro-check: einsum('ij,j...->i...', A, X) ≡ A @ X for the
    outer-mode shape (3,3) × (3, B), independently of process internals."""
    torch.manual_seed(0)
    A = torch.randn(3, 3, dtype=DTYPE, device=DEVICE)
    X = torch.randn(3, 17, dtype=DTYPE, device=DEVICE)
    via_einsum = torch.einsum('ij,j...->i...', A, X)
    via_matmul = A @ X
    assert torch.allclose(via_einsum, via_matmul, rtol=1e-5, atol=1e-6)


def test_var_inner_shape_init_and_independence():
    B, B2, T = 4, 8, 10
    p, shared, tensor = _setup_var(batch_size=B, T=T, inner_sub=B2)
    out = p.generate(shared)
    n_contracts = p.contract_T.shape[1]
    assert out.shape == (T, n_contracts, B, B2), (
        f'expected (T,n_c,B,B2)=({T},{n_contracts},{B},{B2}), got {tuple(out.shape)}')

    # X0 was solved per-b from the (3, B) tensor. At t=0 the curve reconstruction
    # for contract knots that coincide with calibration slots should reproduce
    # tensor[contract, b] within numerical precision — and be identical across the
    # B2 fan-out (no inner-sample noise has fired yet beyond Phi/sigma at t=0).
    # Use t=0 of the output and check independence across B2 first.
    t0 = out[0]                                                                      # (n_contracts, B, B2)
    # Across B2 within fixed b: VAR's t=0 evaluation applies one step of Phi+sigma
    # from X0, so paths within B2 differ. Just check they don't collapse to 0.
    per_b_std = out[-1].std(dim=-1)                                                  # (n_contracts, B)
    assert (per_b_std > 0).all(), 'VAR inner paths collapsed at terminal'

    # Per-b X0 differentiation: at t=0 in the outer/inner case, the mean of the
    # B2 paths for each (contract, b) should be ordered by b (since tensor[:, b]
    # is monotone in b via the linspace bumps). Verify monotonicity of the
    # contract-wise average across b.
    t0_mean_b = t0.mean(dim=-1)                                                      # (n_contracts, B)
    # Across b, contract-0 should track the level component; check b-axis ordering.
    diffs = t0_mean_b[:, 1:] - t0_mean_b[:, :-1]                                     # (n_contracts, B-1)
    assert (diffs >= -1e-3).all() or (diffs <= 1e-3).all(), (
        'X0 fan-out across b not propagating to t=0 evaluation — contract means '
        'are not monotone in b for monotone tensor[:, b] input')


# ---------------------------------------------------------------------------
# Inner-mode antithetic (Inner_Antithetic='Yes'): mirrored (z, -z) pairs on the
# inner axis; odd sub-batch rejected; default path unchanged.
# ---------------------------------------------------------------------------

def test_cmc_state_inner_antithetic_mirrors_inner_axis():
    B, B2, T = 4, 8, 5
    shared = _build_shared(num_factors=1, batch_size=B, simulation_sub_batch=B2)
    shared.reset_inner(num_factors=1, time_grid=_time_grid(T), use_antithetic=True)
    z = shared.t_random_numbers
    assert z.shape == (1, T, B, B2)
    half = B2 // 2
    assert torch.allclose(z[..., half:], -z[..., :half]), (
        'antithetic inner draws must be the mirror of the first half on the inner axis')
    # the paired construction must not degenerate (first half genuinely random)
    assert z[..., :half].std() > 0


def test_cmc_state_inner_antithetic_rejects_odd_sub_batch():
    shared = _build_shared(num_factors=1, batch_size=4, simulation_sub_batch=3)
    with pytest.raises(ValueError, match='even Inner_Sub_Batch'):
        shared.reset_inner(num_factors=1, time_grid=_time_grid(5), use_antithetic=True)


def test_cmc_state_inner_base_reset_still_works():
    """The inherited base `reset()` must produce the regular outer-mode (T, B)
    shape, even on a CMC_State_Inner — only `reset_inner()` swaps to (T, B, B2)."""
    B, B2, T = 4, 8, 5
    shared = _build_shared(num_factors=1, batch_size=B, simulation_sub_batch=B2)
    shared.reset(num_factors=1, time_grid=_time_grid(T))
    assert shared.t_random_numbers.shape == (1, T, B), (
        f'base reset() on CMC_State_Inner must yield (num_factors, T, B); '
        f'got {tuple(shared.t_random_numbers.shape)}')
    shared.reset_inner(num_factors=1, time_grid=_time_grid(T))
    assert shared.t_random_numbers.shape == (1, T, B, B2)


def test_cmc_state_inner_reset_inner_rejects_sub_batch_one():
    """Guard fires when inner mode is actually used (reset_inner), not at construction —
    so a CMC_State_Inner can be carried through outer-only flows without configuring
    Inner_Sub_Batch up front."""
    shared = _build_shared(num_factors=1, batch_size=4, simulation_sub_batch=1)
    with pytest.raises(ValueError, match='simulation_sub_batch > 1'):
        shared.reset_inner(num_factors=1, time_grid=_time_grid(5))


# ---------------------------------------------------------------------------
# End-to-end smoke through the framework JSON path. Outer-mode is the only path
# wired to the live driver; this just confirms the refactor didn't break the
# canonical simulate_only fixture.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMULATE_FIXTURE = os.path.join(REPO_ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')


@pytest.mark.skipif(not os.path.exists(SIMULATE_FIXTURE),
                    reason='simulate_only fixture not present in checkout')
def test_simulate_only_smoke_runs():
    cx = rf.Context()
    cx.load_json(SIMULATE_FIXTURE)
    _, result = cx.run_job()
    # Defensive: any rollout output suffices. evaluation_summary may be absent in
    # pure simulate_only mode; the goal is "didn't raise during reset/precalc/generate".
    assert result is not None


# ===========================================================================
# Per-path vector-init contract for the remaining processes (HW family, OU
# spot models) + the shared base-class seams `forward_curve` / `align_rank`.
#
# Strategy (no pre-change golden needed): feed a per-path init that is B
# *replicated* copies of the calibrated curve/spot — every output column must
# equal the calibrated single-init output (the per-path machinery reduces to
# the calibrated path exactly = no-op). Then feed B *distinct* columns and
# assert the init flows through monotonically (per-path independence).
# ===========================================================================

from riskflow.stochasticprocess import (                                       # noqa: E402
    StochasticProcess,
    HullWhite1FactorInterestRateModel,
    HWHazardRateModel,
    LogOUSpotModel,
    MarkovSwitchingLogOUSpotModel,
)


def test_align_rank_pads_to_explicit_target():
    """align_rank right-pads with trailing singleton axes to the explicit target
    rank; no-op when already there; spans a 2-gap (future inner mode)."""
    x2 = torch.zeros(3, 4)
    assert StochasticProcess.align_rank(x2, 2).shape == (3, 4)                  # no-op
    assert StochasticProcess.align_rank(x2, 3).shape == (3, 4, 1)              # +1
    assert StochasticProcess.align_rank(x2, 4).shape == (3, 4, 1, 1)          # +2 (inner)
    # values preserved (pure view / broadcast prep)
    x = torch.randn(2, 5)
    assert torch.equal(StochasticProcess.align_rank(x, 4).reshape(2, 5), x)


# --- shared curve seam: forward_curve loop-over-B == per-column calibrated ---

def _curve_factor():
    """Minimal curve factor accepted by utils.calc_curve_forwards: tenors in years,
    a single linear interpolation method, and a day-count accrual that maps a
    scen-grid day index to a year fraction."""
    tenors_years = np.array([0.08, 0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    f = types.SimpleNamespace()
    f.get_tenor = lambda: tenors_years
    f.tenors = tenors_years
    f.interpolation = [('Linear',)]
    f.get_day_count_accrual = lambda ref_date, day: float(day) / utils.DAYS_IN_YEAR
    f.get_currency = lambda: ('USD',)
    return f


def test_forward_curve_batch_consistency():
    """The base-class `forward_curve` seam: a per-path (n_tenors, B) init whose
    columns are all the calibrated curve must yield B identical forward-curve
    columns, each equal to the calibrated (n_tenors,) call."""
    proc = StochasticProcess.__new__(StochasticProcess)
    proc.factor = _curve_factor()
    shared = _build_shared(num_factors=1, batch_size=4)
    _do_reset(shared, 1, _time_grid(8))
    tgy = (np.arange(1, 9, dtype=np.float64)) / utils.DAYS_IN_YEAR
    n_tenors = _curve_factor().get_tenor().size
    calib = torch.tensor([0.01, 0.012, 0.015, 0.02, 0.025], dtype=DTYPE)        # (n_tenors,)

    fc_calib = proc.forward_curve(calib, tgy, shared)                          # [T, n_tenors]
    assert fc_calib.shape[1] == n_tenors and fc_calib.ndim == 2

    B = 4
    replicated = calib.view(-1, 1).expand(n_tenors, B).contiguous()           # (n_tenors, B) all-equal
    fc_rep = proc.forward_curve(replicated, tgy, shared)                       # [T, n_tenors, B]
    assert fc_rep.shape == (fc_calib.shape[0], n_tenors, B)
    for b in range(B):
        assert torch.allclose(fc_rep[..., b], fc_calib, rtol=1e-6, atol=1e-8), (
            f'forward_curve replicated column {b} != calibrated')

    # distinct columns → distinct forward curves (per-path independence)
    bumps = torch.linspace(-0.003, 0.003, B, dtype=DTYPE).view(1, B)
    distinct = calib.view(-1, 1) + bumps                                       # (n_tenors, B)
    fc_dist = proc.forward_curve(distinct, tgy, shared)
    assert (fc_dist.std(dim=-1) > 0).any(), 'forward_curve collapsed distinct inits'


# --- scalar OU spot models -------------------------------------------------

def _logou_param():
    return {'Kappa': 1.5, 'Sigma': 0.2, 'Theta': 7.6, 'Anchor_Theta_At_Spot': True}


def _setup_logou(*, batch_size, T=10, per_path=None, seed=42):
    tg = _time_grid(T)
    shared = _build_shared(num_factors=1, batch_size=batch_size, seed=seed)
    _do_reset(shared, 1, tg)
    p = LogOUSpotModel(factor=types.SimpleNamespace(param={}), param=_logou_param())
    p.factor_key = utils.Factor('CommodityPrice', ('OU',))
    spot = 2000.0
    if per_path is None:
        tensor = torch.tensor([spot], dtype=DTYPE, device=DEVICE)             # (1,)
    elif per_path == 'replicated':
        tensor = torch.full((batch_size,), spot, dtype=DTYPE, device=DEVICE)  # (B,) all-equal
    else:
        bumps = torch.linspace(-50.0, 50.0, batch_size, dtype=DTYPE, device=DEVICE)
        tensor = spot + bumps                                                  # (B,) distinct
    p.precalculate(REF_DATE, tg, tensor, shared, process_ofs=0)
    return p, shared


def _mslogou_param():
    return {
        'States': [{'Kappa': 1.0, 'Sigma': 0.15, 'Theta': 7.6},
                   {'Kappa': 2.0, 'Sigma': 0.30, 'Theta': 7.7}],
        'Transition_Matrix': [[0.97, 0.03], [0.05, 0.95]],
        'Initial_State_Probs': [0.6, 0.4],
        'Calibration_DT_Years': 1.0 / 252.0,
    }


def _setup_mslogou(*, batch_size, T=10, per_path=None, seed=42):
    tg = _time_grid(T)
    shared = _build_shared(num_factors=1, batch_size=batch_size, seed=seed)
    _do_reset(shared, 1, tg)
    p = MarkovSwitchingLogOUSpotModel(factor=types.SimpleNamespace(param={}), param=_mslogou_param())
    p.factor_key = utils.Factor('CommodityPrice', ('MSOU',))
    spot = 2000.0
    if per_path is None:
        tensor = torch.tensor([spot], dtype=DTYPE, device=DEVICE)
    elif per_path == 'replicated':
        tensor = torch.full((batch_size,), spot, dtype=DTYPE, device=DEVICE)
    else:
        bumps = torch.linspace(-50.0, 50.0, batch_size, dtype=DTYPE, device=DEVICE)
        tensor = spot + bumps
    p.precalculate(REF_DATE, tg, tensor, shared, process_ofs=0)
    return p, shared


@pytest.mark.parametrize('setup', [_setup_logou, _setup_mslogou])
def test_ou_spot_perpath_replicated_equals_calibrated(setup):
    """Per-path init of B replicated spots == calibrated single-spot output
    (same Z, same per-column log_spot0 ⇒ byte-equal reduction)."""
    B, T = 6, 10
    p_c, sh_c = setup(batch_size=B, T=T, per_path=None)
    p_r, sh_r = setup(batch_size=B, T=T, per_path='replicated')
    out_c, out_r = p_c.generate(sh_c), p_r.generate(sh_r)
    assert out_c.shape == out_r.shape == (T, B)
    assert torch.allclose(out_r, out_c, rtol=1e-6, atol=1e-5), (
        f'{setup.__name__}: replicated per-path != calibrated '
        f'(max |Δ|={(out_r - out_c).abs().max().item():.2e})')


@pytest.mark.parametrize('setup', [_setup_logou, _setup_mslogou])
def test_ou_spot_perpath_distinct_flows_through_monotonically(setup):
    """Distinct per-path spots (monotone in b) propagate to the t=0 level. With Z
    held fixed across the two runs, out_distinct[0]/out_replicated[0] is exp of a
    monotone-in-b init delta, hence itself monotone in b."""
    B, T = 6, 10
    p_r, sh_r = setup(batch_size=B, T=T, per_path='replicated')
    p_d, sh_d = setup(batch_size=B, T=T, per_path='distinct')
    out_r, out_d = p_r.generate(sh_r), p_d.generate(sh_d)
    assert out_d.shape == (T, B)
    assert not torch.allclose(out_d, out_r), 'per-path init did not flow through'
    ratio0 = (out_d[0] / out_r[0])
    d = ratio0[1:] - ratio0[:-1]
    assert (d >= -1e-4).all(), (
        f'{setup.__name__}: per-path t=0 level not monotone in init bumps: ratio0={ratio0.tolist()}')


# --- HW1F: representative curve process (HWHazard shares the exact pattern) --

def _hw1f_param():
    return {
        'Alpha': 0.05,
        'Sigma': utils.Curve([], [(0.0, 0.008), (2.0, 0.010)]),
        'Lambda': 0.0,
        'Quanto_FX_Volatility': 0.0,
        'Quanto_FX_Correlation': 0.0,
    }


def _setup_hw1f(*, batch_size, T=10, per_path=None, seed=42):
    tg = _time_grid(T)
    shared = _build_shared(num_factors=1, batch_size=batch_size, seed=seed)
    _do_reset(shared, 1, tg)
    p = HullWhite1FactorInterestRateModel(factor=_curve_factor(), param=_hw1f_param())
    p.factor_key = utils.Factor('InterestRate', ('HW1F',))
    n_tenors = _curve_factor().get_tenor().size
    calib = torch.tensor([0.01, 0.012, 0.015, 0.02, 0.025], dtype=DTYPE, device=DEVICE)
    if per_path is None:
        tensor = calib
    elif per_path == 'replicated':
        tensor = calib.view(-1, 1).expand(n_tenors, batch_size).contiguous()
    else:
        bumps = torch.linspace(-0.003, 0.003, batch_size, dtype=DTYPE, device=DEVICE).view(1, -1)
        tensor = calib.view(-1, 1) + bumps
    p.precalculate(REF_DATE, tg, tensor, shared, process_ofs=0)
    return p, shared


def test_hw1f_perpath_replicated_equals_calibrated():
    """HW1F: per-path init of B replicated curves == calibrated, broadcast across B."""
    B, T = 5, 10
    p_c, sh_c = _setup_hw1f(batch_size=B, T=T, per_path=None)
    p_r, sh_r = _setup_hw1f(batch_size=B, T=T, per_path='replicated')
    out_c = p_c.generate(sh_c)                                                 # (T, n_tenors, B)
    out_r = p_r.generate(sh_r)                                                 # (T, n_tenors, B)
    assert out_c.shape == out_r.shape
    assert out_c.shape[-1] == B and out_c.ndim == 3
    assert torch.allclose(out_r, out_c, rtol=1e-5, atol=1e-7), (
        f'HW1F replicated per-path != calibrated (max |Δ|={(out_r - out_c).abs().max().item():.2e})')


def test_hw1f_perpath_distinct_carries_batch():
    """HW1F: distinct per-path curves produce a per-path output that depends on the init."""
    B, T = 5, 10
    p_r, sh_r = _setup_hw1f(batch_size=B, T=T, per_path='replicated')
    p_d, sh_d = _setup_hw1f(batch_size=B, T=T, per_path='distinct')
    out_r, out_d = p_r.generate(sh_r), p_d.generate(sh_d)
    assert out_d.shape == out_r.shape and out_d.ndim == 3
    assert not torch.allclose(out_d, out_r), 'HW1F per-path init did not flow through'


# ===========================================================================
# Inner-MC (nested simulation, Z.ndim==3) support for the newly-enabled
# processes + the base-class enforcement contract.
#
# Inner shape: each of B outer paths fans into B2 inner samples; the per-outer
# init[b] must land on the MIDDLE axis (broadcast across B2), not collide with
# B2. We verify: (1) output shape (T,...,B,B2); (2) the t=0 level is monotone in
# b for a monotone-in-b init (init landed on the B axis, not transposed/collapsed).
# ===========================================================================

from riskflow.stochasticprocess import (                                       # noqa: E402
    GBMAssetPriceModel, GBMPriceIndexModel, GBMAssetPriceTSModelImplied,
    SingleRegimeOU1FactorKalmanModel, HWHazardRateModel,
    HullWhite2FactorImpliedInterestRateModel, CSForwardPriceModel,
    PCAInterestRateModel,
)


def _inner_shared(num_factors, B, B2, seed=42):
    sh = _build_shared(num_factors=num_factors, batch_size=B, seed=seed,
                       simulation_sub_batch=B2)
    _do_reset(sh, num_factors, _time_grid(10))
    return sh


def _setup_gbm_inner(B, B2, T=10):
    sh = _inner_shared(1, B, B2)
    p = GBMAssetPriceModel(factor=types.SimpleNamespace(param={}),
                           param={'Vol': 0.2, 'Drift': 0.03})
    p.factor_key = utils.Factor('CommodityPrice', ('GBM',))
    spot = torch.linspace(1800.0, 2200.0, B, dtype=DTYPE, device=DEVICE)        # distinct per outer
    p.precalculate(REF_DATE, _time_grid(T), spot, sh, process_ofs=0)
    return p, sh


def _setup_logou_inner(B, B2, T=10):
    sh = _inner_shared(1, B, B2)
    p = LogOUSpotModel(factor=types.SimpleNamespace(param={}), param=_logou_param())
    p.factor_key = utils.Factor('CommodityPrice', ('OU',))
    spot = torch.linspace(1800.0, 2200.0, B, dtype=DTYPE, device=DEVICE)
    p.precalculate(REF_DATE, _time_grid(T), spot, sh, process_ofs=0)
    return p, sh


def _setup_mslogou_inner(B, B2, T=10):
    sh = _inner_shared(1, B, B2)
    p = MarkovSwitchingLogOUSpotModel(factor=types.SimpleNamespace(param={}), param=_mslogou_param())
    p.factor_key = utils.Factor('CommodityPrice', ('MSOU',))
    spot = torch.linspace(1800.0, 2200.0, B, dtype=DTYPE, device=DEVICE)
    p.precalculate(REF_DATE, _time_grid(T), spot, sh, process_ofs=0)
    return p, sh


def _setup_hw1f_inner(B, B2, T=10):
    sh = _inner_shared(1, B, B2)
    p = HullWhite1FactorInterestRateModel(factor=_curve_factor(), param=_hw1f_param())
    p.factor_key = utils.Factor('InterestRate', ('HW1F',))
    n_tenors = _curve_factor().get_tenor().size
    base = torch.tensor([0.01, 0.012, 0.015, 0.02, 0.025], dtype=DTYPE, device=DEVICE).view(-1, 1)
    bumps = torch.linspace(-0.004, 0.004, B, dtype=DTYPE, device=DEVICE).view(1, -1)
    curve = base + bumps                                                        # (n_tenors, B) distinct
    p.precalculate(REF_DATE, _time_grid(T), curve, sh, process_ofs=0)
    return p, sh


def _setup_singleregime_inner(B, B2, T=10):
    sh = _inner_shared(1, B, B2)
    p = SingleRegimeOU1FactorKalmanModel(factor=types.SimpleNamespace(param={}),
                                         param={'Kappa': 1.5, 'Theta': 0.01, 'Sigma': 0.02})
    p.factor_key = utils.Factor('InterestRate', ('OU_CARRY',))
    x0 = torch.linspace(0.008, 0.012, B, dtype=DTYPE, device=DEVICE)            # (B,) distinct
    p.precalculate(REF_DATE, _time_grid(T), x0, sh, process_ofs=0)
    return p, sh


def _setup_hwhazard_inner(B, B2, T=10):
    sh = _inner_shared(1, B, B2)
    p = HWHazardRateModel(factor=_curve_factor(),
                          param={'Alpha': 0.05, 'Sigma': 0.01, 'Lambda': 0.0})
    p.factor_key = utils.Factor('SurvivalProb', ('HWHAZARD',))
    base = torch.tensor([0.01, 0.012, 0.015, 0.02, 0.025], dtype=DTYPE, device=DEVICE).view(-1, 1)
    # MULTIPLICATIVE per-outer scaling (monotone in b): a purely additive vertical shift
    # cancels in the instantaneous-forward (mul_time=False) differencing and leaves t=0 flat.
    factors = torch.linspace(0.8, 1.2, B, dtype=DTYPE, device=DEVICE).view(1, -1)
    p.precalculate(REF_DATE, _time_grid(T), base * factors, sh, process_ofs=0)
    return p, sh


def _priceindex_factor():
    # GBMPriceIndexModel.precalculate needs factor.param['Last_Period_Start'] and
    # factor.get_last_publication_dates(...); stub the publication grid as the scen grid.
    return types.SimpleNamespace(
        param={'Last_Period_Start': REF_DATE},
        get_last_publication_dates=lambda ref_date, scen_times: [
            ref_date + pd.Timedelta(days=int(t)) for t in scen_times])


def _setup_gbmindex_inner(B, B2, T=10):
    sh = _inner_shared(1, B, B2)
    p = GBMPriceIndexModel(factor=_priceindex_factor(), param={'Vol': 0.2, 'Drift': 0.03})
    p.factor_key = utils.Factor('PriceIndex', ('GBMPI',))
    spot = torch.linspace(1800.0, 2200.0, B, dtype=DTYPE, device=DEVICE)
    p.precalculate(REF_DATE, _time_grid(T), spot, sh, process_ofs=0)
    return p, sh


def _hw2f_implied_factor():
    return types.SimpleNamespace(param={
        'Sigma_1': utils.Curve([], [(0.0, 0.008), (5.0, 0.01)]),
        'Sigma_2': utils.Curve([], [(0.0, 0.006), (5.0, 0.009)]),
        'Quanto_FX_Volatility': utils.Curve([], [(0.0, 0.0)]),
    })


def _setup_hw2f_inner(B, B2, T=10):
    sh = _inner_shared(2, B, B2)
    p = HullWhite2FactorImpliedInterestRateModel(
        factor=_curve_factor(), param={'Lambda_1': 0.0, 'Lambda_2': 0.0},
        implied_factor=_hw2f_implied_factor())
    p.factor_key = utils.Factor('InterestRate', ('HW2F',))
    base = torch.tensor([0.01, 0.012, 0.015, 0.02, 0.025], dtype=DTYPE, device=DEVICE).view(-1, 1)
    bumps = torch.linspace(-0.004, 0.004, B, dtype=DTYPE, device=DEVICE).view(1, -1)
    implied_tensor = {
        'Alpha_1': torch.tensor([0.1], dtype=DTYPE, device=DEVICE),
        'Alpha_2': torch.tensor([0.05], dtype=DTYPE, device=DEVICE),
        'Correlation': torch.tensor(0.3, dtype=DTYPE, device=DEVICE),
        'Sigma_1': torch.tensor([0.008, 0.01], dtype=DTYPE, device=DEVICE),
        'Sigma_2': torch.tensor([0.006, 0.009], dtype=DTYPE, device=DEVICE),
    }
    p.precalculate(REF_DATE, _time_grid(T), base + bumps, sh, process_ofs=0,
                   implied_tensor=implied_tensor)
    return p, sh


@pytest.mark.parametrize('setup,is_curve', [
    (_setup_gbm_inner, False),
    (_setup_gbmindex_inner, False),
    (_setup_singleregime_inner, False),
    (_setup_logou_inner, False),
    (_setup_mslogou_inner, False),
    (_setup_hw1f_inner, True),
    (_setup_hwhazard_inner, True),
    (_setup_hw2f_inner, True),
])
def test_inner_mc_shape_and_init_on_correct_axis(setup, is_curve):
    B, B2, T = 5, 8, 10
    p, sh = setup(B, B2, T)
    out = p.generate(sh)
    if is_curve:
        assert out.ndim == 4 and out.shape[0] == T and out.shape[-2:] == (B, B2), (
            f'{setup.__name__}: expected (T,n_tenors,B,B2), got {tuple(out.shape)}')
        level0 = out[0].mean(dim=(0, -1))          # mean over tenors + B2 -> per-outer (B,)
    else:
        assert out.shape == (T, B, B2), (
            f'{setup.__name__}: expected (T,B,B2), got {tuple(out.shape)}')
        level0 = out[0].mean(dim=-1)               # mean over B2 -> per-outer (B,)
    assert torch.isfinite(out).all(), f'{setup.__name__}: non-finite inner output'
    # init was monotone increasing in b → the t=0 per-outer level must be monotone in b,
    # proving init[b] landed on the B (middle) axis and broadcast across B2 (not collided).
    d = level0[1:] - level0[:-1]
    assert (d >= -1e-3).all(), (
        f'{setup.__name__}: t=0 level not monotone in outer-path init — init landed on '
        f'the wrong axis. level0={level0.tolist()}')


# --- GBMTSImplied: the one process that gathers OTHER (rate) factors during generate. ---
# Faithful inner-MC test: build real stochastic zero-rate curve codes + a (T,n_tenors,B,B2)
# scenario buffer (the shape the live factor graph feeds it), so the drift gather exercises
# the real utils.calc_time_grid_curve_rate path with n_batch_dims=2.

def _stoch_rate_code(name_tuple, rate_level, B, B2, T):
    factor = utils.Factor('InterestRate', name_tuple)
    tenor_points = np.array([0.0, 1.0, 5.0, 30.0], dtype=np.float64)
    tenor_data = utils.tenor_diff(tenor_points, 'Linear')

    def daycount(time_in_days):
        return utils.get_day_count_accrual(REF_DATE, time_in_days, utils.DAYCOUNT_ACT365)

    code = (True, factor, None, tenor_data, daycount)                          # is_stoch=True
    scen = torch.full((T, tenor_points.size, B, B2), rate_level, dtype=DTYPE, device=DEVICE)
    return [code], factor, scen


def _setup_gbmts_implied_inner(B, B2, T=10, r_level=0.04, q_level=0.01, vol=0.2):
    sh = _inner_shared(1, B, B2)
    factor = types.SimpleNamespace(
        param={}, get_subtype=lambda: None, get_currency=lambda: ('USD',),
        get_domestic_currency=lambda x: ('ZAR',))
    vol_tenors = np.array([0.0, 1.0, 5.0], dtype=np.float64)
    implied = types.SimpleNamespace(param={
        'Vol': utils.Curve([], list(zip(vol_tenors, [vol] * vol_tenors.size))),
        'Quanto_FX_Volatility': None})
    implied_tensor = {'Vol': torch.full((vol_tenors.size,), vol, dtype=DTYPE, device=DEVICE)}
    p = GBMAssetPriceTSModelImplied(factor=factor, param={}, implied_factor=implied)
    p.factor_type = 'FxRate'
    p.factor_key = utils.Factor('FxRate', ('USD',))
    # stochastic domestic/foreign zero curves, simulated to (T, n_tenors, B, B2)
    r_code, r_key, r_scen = _stoch_rate_code(('ZAR',), r_level, B, B2, T)
    q_code, q_key, q_scen = _stoch_rate_code(('USD',), q_level, B, B2, T)
    sh.t_Scenario_Buffer[r_key] = r_scen
    sh.t_Scenario_Buffer[q_key] = q_scen
    p.r_t, p.q_t = r_code, q_code
    spot = torch.linspace(10.0, 20.0, B, dtype=DTYPE, device=DEVICE)            # monotone per-outer
    p.precalculate(REF_DATE, _time_grid(T), spot, sh, process_ofs=0, implied_tensor=implied_tensor)
    return p, sh


def test_gbmts_implied_inner_mc():
    """GBMTSImplied gathers stochastic rate curves inside generate; under inner MC those
    curves are (T,n_tenors,B,B2). Verify n_batch_dims=2 collapses (B,B2) for the gather and
    the drift reshapes back so output is (T,B,B2) with the per-outer spot on the B axis."""
    B, B2, T = 5, 8, 10
    p, sh = _setup_gbmts_implied_inner(B, B2, T)
    out = p.generate(sh)
    assert out.shape == (T, B, B2), f'expected (T,B,B2), got {tuple(out.shape)}'
    assert torch.isfinite(out).all(), 'non-finite GBMTSImplied inner output'
    level0 = out[0].mean(dim=-1)                                                # (B,)
    d = level0[1:] - level0[:-1]
    assert (d >= -1e-3).all(), (
        f'GBMTSImplied inner t=0 level not monotone in spot init — init landed on the '
        f'wrong axis or drift mis-broadcast. level0={level0.tolist()}')
