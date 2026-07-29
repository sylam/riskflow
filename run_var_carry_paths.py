"""Simulate the platinum carry curve via VARMixedFactorInterestRateModel and dump
a CSV of paths.

Output: artifacts/var_carry_paths.csv — long format
    columns: sim_date, contract, mean, p5, p25, p50, p75, p95, std,
             scen_0 ... scen_(N_VIS-1)
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


# --- Setup ---
SEED = 7
N_PATHS = 256
N_VIS = 16                                                                 # paths to dump in CSV
HORIZON_DAYS = 252                                                         # one year of business days
REF_DATE = pd.Timestamp('2025-12-30')

CONTRACTS = [
    ('PL_APR_2026', pd.Timestamp('2026-04-29')),
    ('PL_JUL_2026', pd.Timestamp('2026-07-29')),
    ('PL_OCT_2026', pd.Timestamp('2026-10-29')),
]
CURVE_TODAY = np.array([0.06662, 0.06673, 0.06710])                        # carry rates today, per contract


# --- Calibrate ---
df = pd.read_csv('data/plat_archive.csv', index_col=0)
carry_cols = [c for c in df.columns if 'ForwardRate.PLATINUM_CARRY' in c]
tenor_cols = [c for c in df.columns if c.startswith('Tenor.PLATINUM')]
sub = df[carry_cols + tenor_cols].dropna()
cal = VARMixedFactorInterestRateCalibration(model=None, param={})
info = cal.calibrate(sub, vol_shift=0.0)
param = info.param

print('Calibrated:')
print(' Calibration_Tenors:    ', param['Calibration_Tenors'])
print(' Contract_Cycle_Years:  ', param['Contract_Cycle_Years'])
print(' σ (latent innovations):', param['Sigma'])


# --- Stub factor + grid (mirrors the framework's calc_dependencies setup) ---
class StubFactor:
    def __init__(self, tenor):
        self._tenor = np.array(tenor, dtype=np.float64)
    def get_tenor(self):
        return self._tenor

contract_dates_excel = np.array([(d - excel_offset).days for _, d in CONTRACTS], dtype=np.float64)
factor = StubFactor(contract_dates_excel)

scen_days = np.arange(0.0, HORIZON_DAYS + 1, 1.0, dtype=np.float64)        # daily steps in calendar days
sim_t_years = scen_days / DAYS_IN_YEAR
sim_dates = [REF_DATE + pd.Timedelta(days=int(d)) for d in scen_days]
time_grid = types.SimpleNamespace(time_grid_years=sim_t_years, scen_time_grid=scen_days)


class Shared:
    one = torch.tensor(1.0, dtype=torch.float64)


shared = Shared()
torch.set_default_dtype(torch.float64)

# --- Run ---
model = VARMixedFactorInterestRateModel(factor=factor, param=param)
model.precalculate(ref_date=REF_DATE, time_grid=time_grid,
                   tensor=torch.tensor(CURVE_TODAY), shared=shared, process_ofs=0)

torch.manual_seed(SEED)
shared_mem = types.SimpleNamespace(
    t_random_numbers=torch.randn(3, len(sim_t_years), N_PATHS, dtype=torch.float64),
)
out = model.generate(shared_mem).cpu().numpy()                             # (T, n_contracts, B)
print(f'\nout shape: {out.shape}    (T={out.shape[0]}, contracts={out.shape[1]}, paths={out.shape[2]})')


# --- Build long-format CSV ---
rows = []
for c_idx, (cname, cdate) in enumerate(CONTRACTS):
    for t in range(out.shape[0]):
        paths_t = out[t, c_idx, :]
        row = {
            'sim_date': sim_dates[t].strftime('%Y-%m-%d'),
            'contract': cname,
            'expiry':   cdate.strftime('%Y-%m-%d'),
            'T_years':  float(model.contract_T[t, c_idx].item()),
            'expired':  bool(model.contract_expired[t, c_idx].item()),
            'mean':     float(paths_t.mean()),
            'std':      float(paths_t.std()),
            'p5':       float(np.percentile(paths_t, 5)),
            'p25':      float(np.percentile(paths_t, 25)),
            'p50':      float(np.percentile(paths_t, 50)),
            'p75':      float(np.percentile(paths_t, 75)),
            'p95':      float(np.percentile(paths_t, 95)),
        }
        for k in range(N_VIS):
            row[f'scen_{k}'] = float(paths_t[k])
        rows.append(row)

out_df = pd.DataFrame(rows)
out_path = 'artifacts/var_carry_paths.csv'
out_df.to_csv(out_path, index=False)
print(f'\nWrote {out_path}    ({out_df.shape[0]} rows × {out_df.shape[1]} cols)')


# --- Summary printout ---
print('\nSummary at key sim dates:')
print(out_df.groupby('contract').apply(
    lambda g: g.iloc[[0, len(g) // 4, len(g) // 2, 3 * len(g) // 4, -1]][
        ['sim_date', 'T_years', 'expired', 'mean', 'std', 'p5', 'p95']
    ], include_groups=False
).to_string())
