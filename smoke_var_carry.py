"""Smoke test for VARMixedFactorInterestRateModel as a forward-curve-shaped factor:
calibrate → precalculate → generate against an absolute-date tenor grid (matching
how the platinum CommodityFutureDeal reads the carry factor)."""
import types
import numpy as np
import pandas as pd
import torch

from riskflow.stochasticprocess import (
    VARMixedFactorInterestRateModel,
    VARMixedFactorInterestRateCalibration,
)
from riskflow.utils import excel_offset, DAYS_IN_YEAR


# --- Calibrate (slot-data calibration is unchanged) ---
df = pd.read_csv('data/plat_archive.csv', index_col=0)
carry_cols = [c for c in df.columns if 'ForwardRate.PLATINUM_CARRY' in c]
tenor_cols = [c for c in df.columns if c.startswith('Tenor.PLATINUM')]
sub = df[carry_cols + tenor_cols].dropna()
cal = VARMixedFactorInterestRateCalibration(model=None, param={})
info = cal.calibrate(sub, vol_shift=0.0)
param = info.param
print('Calibrated:')
print(' Calibration_Tenors:', param['Calibration_Tenors'])
print(' Contract_Cycle_Years:', param['Contract_Cycle_Years'])
print()


# --- Stub factor with absolute-date tenors ---
class StubFactor:
    def __init__(self, tenor):
        self._tenor = np.array(tenor, dtype=np.float64)
    def get_tenor(self):
        return self._tenor

# Three platinum contract expiries (Excel offsets for APR26 / JUL26 / OCT26).
ref_date = pd.Timestamp('2025-12-30')
contract_dates_excel = np.array([46141.0, 46232.0, 46324.0])                   # APR26 / JUL26 / OCT26
factor = StubFactor(contract_dates_excel)

# Today's curve at the dated knots (carry rates AT contract expiry dates today).
curve0_vals = np.array([0.06662, 0.06673, 0.06710])

# Sim grid: 1 year of daily steps starting at t=0.
sim_t = np.arange(0.0, 1.0 + 1e-9, 1.0 / DAYS_IN_YEAR, dtype=np.float64)        # calendar-time
scen_days = sim_t * DAYS_IN_YEAR
time_grid = types.SimpleNamespace(
    time_grid_years=sim_t,
    scen_time_grid=scen_days,
)

class Shared:
    one = torch.tensor(1.0, dtype=torch.float64)

shared = Shared()
torch.set_default_dtype(torch.float64)

model = VARMixedFactorInterestRateModel(factor=factor, param=param)

tensor0 = torch.tensor(curve0_vals)
model.precalculate(ref_date=ref_date, time_grid=time_grid, tensor=tensor0,
                   shared=shared, process_ofs=0)
print(f'X0: {model.X0.numpy()}')
print(f'slot τ at t=0:        {model.tau_slot_per_step[0].numpy()}')
print(f'contract T at t=0:    {model.contract_T[0].numpy()}    (should ~ [0.33, 0.58, 0.83])')
mid_idx = len(sim_t) // 2
print(f'slot τ at t={sim_t[mid_idx]:.2f}:    {model.tau_slot_per_step[mid_idx].numpy()}')
print(f'contract T at t={sim_t[mid_idx]:.2f}: {model.contract_T[mid_idx].numpy()}')
last_idx = len(sim_t) - 1
print(f'slot τ at t={sim_t[last_idx]:.2f}:    {model.tau_slot_per_step[last_idx].numpy()}')
print(f'contract T at t={sim_t[last_idx]:.2f}: {model.contract_T[last_idx].numpy()}')
print(f'expired at t={sim_t[last_idx]:.2f}:   {model.contract_expired[last_idx].numpy()}')
print()

# --- Generate ---
T = len(sim_t)
B = 64
shared_mem = types.SimpleNamespace(
    t_random_numbers=torch.randn(3, T, B, dtype=torch.float64),
)
out = model.generate(shared_mem)                                                # (T, n_contracts, B)
print('out shape:', tuple(out.shape))
print('out[0] mean (per contract):', out[0].mean(dim=-1).numpy(), '(should ~ curve0)')
print('out[T-1] mean (per contract):', out[-1].mean(dim=-1).numpy())
print('out[T-1] std  (per contract):', out[-1].std(dim=-1).numpy())

# --- Round-trip check: t=0 batch mean should equal curve0 (deterministic state at t=0) ---
roundtrip_err = np.abs(out[0].mean(dim=-1).numpy() - curve0_vals)
print()
print('t=0 round-trip max error:', roundtrip_err.max())

# --- No NaN/Inf ---
assert torch.isfinite(out).all(), 'non-finite output'
print('All paths finite.')

# --- Expired contracts are zeroed at the end of sim ---
expired_at_end = model.contract_expired[last_idx].numpy()
if expired_at_end.any():
    expired_vals = out[last_idx][expired_at_end].abs().max().item()
    print(f'Expired contracts at end: {expired_at_end} — max |value| = {expired_vals:.2e}')
    assert expired_vals < 1e-12

print('PASS.')
