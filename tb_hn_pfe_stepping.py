"""How wrong was the aggregate-variance bridge AT THE PFE QUANTILES, and does the correlated
sub-step walk (utils.hn_correlated_substeps / garch_correlated_substeps) close it?

The scenario processes now walk the sub-steps that SPAN each coarse interval -- whole trading
days, then the fractional remainder -- with the framework draw riding the sqrt(E[h.dt])-weighted
combination of the sub-step normals.  Section HN below runs an INTEGER f (n whole days) so the
closed form applies as an oracle; the separate CALENDAR section runs the production clock, where
f is never integer and the previous round(f) truncation cost up to 13% of interval variance.

This table is the acceptance measurement: per node spacing n (monthly/quarterly/semiannual
exposure grids), the return quantiles PFE reads, four ways --

    oracle      exact n-step HN law (Fourier inversion of hn_cdf_logret, bisected)
    exact-sim   the NEW coarse-grid scenario path (one interval, n_sub = n)
    daily-sim   the daily-grid witness (n daily steps, n_sub = 1)
    bridge      the OLD scheme, closed form: N(-V/2, V), V = E[sum h_j] deterministic

plus a GARCH(1,1)-t section (no closed form: coarse vs daily witness) and generate() wall time.

Run:  CUDA_VISIBLE_DEVICES=0 python tb_hn_pfe_stepping.py     (~2 min GPU)
Out:  tb_hn_pfe_stepping.csv
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))

import numpy as np
import pandas as pd
import torch

import riskflow
from riskflow import utils
from riskflow.calculation import CMC_State
from riskflow.stochasticprocess import GARCHSpotModel, HestonNandiImpliedSpotModel
import hn_reference as hnref
from scipy.stats import norm

assert 'PycharmProjects' in riskflow.__file__, riskflow.__file__
DTYPE = torch.float64
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0
B = 1_000_000
N_LIST = (21, 63, 126)
Q_LIST = (1.0, 2.5, 5.0, 95.0, 97.5, 99.0)

_SP = hnref.hn_params_from_targets(ann_vol=0.30, persistence=0.94, gamma=350.0,
                                   leverage_share=0.7, steps_per_year=252.0)
H0_STAT = float(utils.hn_stationary_var(_SP['omega'], _SP['alpha'], _SP['beta'], _SP['gamma_star']))
H0 = 1.6 * H0_STAT
HN_IMPLIED = {'Omega': float(_SP['omega']), 'Alpha': float(_SP['alpha']), 'Beta': float(_SP['beta']),
              'Gamma_Star': float(_SP['gamma_star']), 'H0': H0, 'Steps_Per_Year': 252.0}
GARCH_PARAM = {'Omega': 4.0e-06, 'Alpha': 0.06, 'Beta': 0.92, 'Nu': 8.0, 'Mu': 0.0, 'H0': 1.5e-04,
               'Log_Price': True, 'Calibration_DT_Years': DT_C, 'Convexity_Correction': 'Yes'}


def _time_grid(T, day_step=1.0):
    days = np.cumsum(np.full(T, day_step, dtype=np.float64))
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days * DT_C
    tg.CurrencyMap = {}
    scen = np.zeros((T, 3))
    scen[:, utils.TIME_GRID_MTM] = days
    scen[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(T)
    tg.scenario_grid = scen
    return tg


def _shared(tg, seed):
    one = torch.ones(1, 1, dtype=DTYPE, device=DEV)
    sh = CMC_State(cholesky=torch.eye(1, dtype=DTYPE, device=DEV), static_buffer={}, batch_size=B,
                   one=one, mcmc_sims=0, report_currency=None, seed=seed, job_id=0, num_jobs=1)
    sh.reset(num_factors=1, time_grid=tg)
    return sh


def _rate_code():
    factor = utils.Factor('InterestRate', ('R',))
    tp = np.array([0.0, 1.0, 5.0, 30.0])
    td = utils.tenor_diff(tp, 'Linear')
    dc = (lambda days: utils.get_day_count_accrual(REF_DATE, days, utils.DAYCOUNT_ACT365))
    return [(False, factor, None, td, dc)], factor, torch.zeros(tp.size, dtype=DTYPE, device=DEV)


def make_hn(tg, sh):
    p = HestonNandiImpliedSpotModel(factor=types.SimpleNamespace(param={}), param=None,
                                    implied_factor=types.SimpleNamespace(param=dict(HN_IMPLIED)))
    p.factor_type = 'EquityPrice'
    p.factor_key = utils.Factor('EquityPrice', ('EQ',))
    rc, rf, rcur = _rate_code()
    sh.t_Static_Buffer[rf] = rcur
    p.r_t = p.q_t = rc
    p.precalculate(REF_DATE, tg, torch.tensor([100.0], dtype=DTYPE, device=DEV), sh, process_ofs=0)
    return p


def make_garch(tg, sh):
    p = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(GARCH_PARAM))
    p.factor_key = utils.Factor('CommodityPrice', ('PT',))
    p.precalculate(REF_DATE, tg, torch.tensor([1000.0], dtype=DTYPE, device=DEV), sh, process_ofs=0)
    return p


def oracle_quantile(q, n, cf):
    lo, hi = -1.5, 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(utils.hn_cdf_logret(mid, n, H0, **cf)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bridge_V(n):
    psi = float(_SP['beta'] + _SP['alpha'] * _SP['gamma_star'] ** 2)
    V, m = 0.0, H0
    for _ in range(n):
        V, m = V + m, float(_SP['omega'] + _SP['alpha']) + psi * m
    return V


def sim_returns(make, T, day_step, seed):
    tg = _time_grid(T, day_step)
    sh = _shared(tg, seed)
    p = make(tg, sh)
    if DEV.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    spot = p.generate(sh)
    if DEV.type == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return (spot.log()[T - 1] - spot.log()[0]).cpu().numpy(), wall


rows = []
cf = hnref.as_tensors({'omega': _SP['omega'], 'alpha': _SP['alpha'], 'beta': _SP['beta'],
                       'gamma_star': _SP['gamma_star'], 'r': 0.0})
print(f'HN one interval, h1 = 1.6x stationary, B = {B:,}, {DEV}')
for n in N_LIST:
    R_coarse, w_c = sim_returns(make_hn, 2, float(n), seed=11)
    R_daily, w_d = sim_returns(make_hn, n + 1, 1.0, seed=12)
    V = bridge_V(n)
    for q in Q_LIST:
        qo = oracle_quantile(q / 100.0, n, cf)
        qe = float(np.percentile(R_coarse, q))
        qd = float(np.percentile(R_daily, q))
        qb = float(norm.ppf(q / 100.0) * np.sqrt(V) - V / 2.0)
        # error in the survival prob at the oracle quantile is the OSS-style view; report
        # quantile displacement in ANNUALIZED VOL units (what a PFE number moves by)
        rows.append(dict(model='HN', n=n, q=q, oracle=qo, exact_sim=qe, daily_sim=qd, bridge=qb,
                         err_exact=qe - qo, err_daily=qd - qo, err_bridge=qb - qo,
                         wall_coarse_s=w_c, wall_daily_s=w_d))
    e = [r for r in rows if r['model'] == 'HN' and r['n'] == n]
    print(f'  n={n:3d}  wall coarse {w_c:.2f}s / daily {w_d:.2f}s   '
          f'max|err| exact {max(abs(r["err_exact"]) for r in e):.5f}  '
          f'bridge {max(abs(r["err_bridge"]) for r in e):.5f}')

print(f'GARCH-t one interval, B = {B:,} (daily witness = oracle proxy)')
for n in (21, 63):
    R_coarse, w_c = sim_returns(make_garch, 2, float(n), seed=21)
    R_daily, w_d = sim_returns(make_garch, n + 1, 1.0, seed=22)
    for q in Q_LIST:
        qe, qd = float(np.percentile(R_coarse, q)), float(np.percentile(R_daily, q))
        rows.append(dict(model='GARCH', n=n, q=q, oracle=np.nan, exact_sim=qe, daily_sim=qd,
                         bridge=np.nan, err_exact=qe - qd, err_daily=0.0, err_bridge=np.nan,
                         wall_coarse_s=w_c, wall_daily_s=w_d))
    e = [r for r in rows if r['model'] == 'GARCH' and r['n'] == n]
    print(f'  n={n:3d}  wall coarse {w_c:.2f}s / daily {w_d:.2f}s   '
          f'max|coarse-daily| {max(abs(r["err_exact"]) for r in e):.5f}')

print('CALENDAR clock (f never integer): walked trading time vs elapsed, and the '
      'round(f) truncation this replaced')
CAL = 1.0 / 365.25
for step in (4, 5, 7, 30):
    tgc = _time_grid(3, float(step))
    tgc.time_grid_years = tgc.scen_time_grid * CAL
    p = make_hn(tgc, _shared(tgc, 5))
    f = float(p.f[1])
    walked = sum(p.sub_dt[1])
    print(f'  {step:3d} cal-d/step: f={f:8.4f} sub-steps={len(p.sub_dt[1]):3d} walked={walked:8.4f} '
          f'({100 * (walked / f - 1):+.2e}%)   round(f)={round(f):3d} would walk '
          f'{100 * (round(f) / f - 1):+6.2f}%')

df = pd.DataFrame(rows)
df.to_csv('tb_hn_pfe_stepping.csv', index=False)
pd.set_option('display.width', 200)
print(df[df.model == 'HN'][['n', 'q', 'oracle', 'exact_sim', 'bridge', 'err_exact', 'err_bridge']]
      .to_string(index=False, float_format=lambda x: f'{x:+.5f}'))
print('\nwritten tb_hn_pfe_stepping.csv')
