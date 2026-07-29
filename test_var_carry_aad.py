"""AAD test for VARMixedFactorInterestRateModel: confirm tensor → X_0 → out flows
gradients end-to-end. Compares autograd dV/d(curve_0) against finite differences.

Build the model with curve_0 having requires_grad=True, run generate(), reduce to a
scalar V, backward, inspect curve_0.grad. Then bump curve_0 and finite-difference.
"""
import types
import numpy as np
import pandas as pd
import torch

from riskflow.stochasticprocess import (
    VARMixedFactorInterestRateModel,
    VARMixedFactorInterestRateCalibration,
)
from riskflow.utils import excel_offset, DAYS_IN_YEAR

torch.set_default_dtype(torch.float64)


# --- Calibrate (same as smoke) ---
df = pd.read_csv('data/plat_archive.csv', index_col=0)
carry_cols = [c for c in df.columns if 'ForwardRate.PLATINUM_CARRY' in c]
tenor_cols = [c for c in df.columns if c.startswith('Tenor.PLATINUM')]
sub = df[carry_cols + tenor_cols].dropna()
cal = VARMixedFactorInterestRateCalibration(model=None, param={})
info = cal.calibrate(sub, vol_shift=0.0)
param = info.param


class StubFactor:
    def __init__(self, tenor):
        self._tenor = np.array(tenor, dtype=np.float64)
    def get_tenor(self):
        return self._tenor


ref_date = pd.Timestamp('2025-12-30')
contract_dates_excel = np.array([46141.0, 46232.0, 46324.0])                   # APR26 / JUL26 / OCT26
factor = StubFactor(contract_dates_excel)

scen_days = np.arange(0.0, 90 + 1e-9, 1.0)                                     # short horizon for speed
sim_t_years = scen_days / DAYS_IN_YEAR
time_grid = types.SimpleNamespace(time_grid_years=sim_t_years, scen_time_grid=scen_days)


class Shared:
    one = torch.tensor(1.0, dtype=torch.float64)


shared = Shared()


def reduce_path(model, T, B, seed=0):
    """Build a deterministic Z, run generate, reduce to a scalar V (sum over T,K,B)."""
    g = torch.Generator(); g.manual_seed(seed)
    Z = torch.randn(3, T, B, dtype=torch.float64, generator=g)
    shared_mem = types.SimpleNamespace(t_random_numbers=Z)
    out = model.generate(shared_mem)
    return out.sum()                                                           # scalar


T = len(sim_t_years)
B = 4

# --- 1. Build with requires_grad and check graph ---
curve0 = torch.tensor([0.06662, 0.06673, 0.06710], dtype=torch.float64, requires_grad=True)
model = VARMixedFactorInterestRateModel(factor=factor, param=param)
model.precalculate(ref_date=ref_date, time_grid=time_grid, tensor=curve0,
                   shared=shared, process_ofs=0)

print('1. Graph integrity:')
print(f'   X_0.requires_grad        = {model.X0.requires_grad}     (must be True)')
print(f'   X_0.grad_fn              = {model.X0.grad_fn}')
assert model.X0.requires_grad, 'X_0 detached from input curve — graph broken!'

V = reduce_path(model, T, B, seed=42)
print(f'   V (out.sum())            = {V.item():.6f}')
print(f'   V.requires_grad          = {V.requires_grad}')

V.backward()
g_autograd = curve0.grad.detach().clone().numpy()
print(f'\n2. Autograd gradient ∂V/∂curve_0:')
print(f'   {g_autograd}')
assert torch.isfinite(curve0.grad).all().item(), 'gradient has NaN/Inf'
assert (curve0.grad.abs() > 1e-12).any().item(), 'gradient is identically zero — chain broken!'

# --- 3. Finite-difference comparison ---
print(f'\n3. Finite differences (h=1e-6):')
h = 1.0e-6
g_fd = np.zeros(3)
curve0_base = torch.tensor([0.06662, 0.06673, 0.06710], dtype=torch.float64)
for k in range(3):
    bumped_up = curve0_base.clone()
    bumped_up[k] += h
    m_up = VARMixedFactorInterestRateModel(factor=factor, param=param)
    m_up.precalculate(ref_date=ref_date, time_grid=time_grid, tensor=bumped_up,
                      shared=shared, process_ofs=0)
    V_up = reduce_path(m_up, T, B, seed=42).item()

    bumped_dn = curve0_base.clone()
    bumped_dn[k] -= h
    m_dn = VARMixedFactorInterestRateModel(factor=factor, param=param)
    m_dn.precalculate(ref_date=ref_date, time_grid=time_grid, tensor=bumped_dn,
                      shared=shared, process_ofs=0)
    V_dn = reduce_path(m_dn, T, B, seed=42).item()

    g_fd[k] = (V_up - V_dn) / (2 * h)

print(f'   {g_fd}')
print(f'\n4. Comparison:')
print(f'   {"k":>3}  {"autograd":>14}  {"finite-diff":>14}  {"abs_err":>10}  {"rel_err":>10}')
for k in range(3):
    abs_err = abs(g_autograd[k] - g_fd[k])
    rel_err = abs_err / (abs(g_fd[k]) + 1e-12)
    print(f'   {k:>3}  {g_autograd[k]:>14.6f}  {g_fd[k]:>14.6f}  {abs_err:>10.2e}  {rel_err:>10.2e}')

max_rel_err = max(abs(g_autograd[k] - g_fd[k]) / (abs(g_fd[k]) + 1e-12) for k in range(3))
print(f'\n   Max rel err = {max_rel_err:.2e}')
if max_rel_err < 1e-3:
    print('   AAD vs FD agree to better than 0.1% — graph is intact end-to-end. ✓')
else:
    print('   AAD vs FD DISAGREE — investigate.')
