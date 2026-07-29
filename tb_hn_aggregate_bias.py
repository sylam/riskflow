"""DELIVERABLE 2 -- how wrong is the "mean-matched normal bridge" for OSS survival?

CONTEXT.  ``pv_MC_Tarf`` / ``pv_discrete_barrier_option`` in riskflow/pricing.py use
one-step survival (OSS): at each fixing they convert the per-path barrier into a
z-score and take p_survive = Phi(z_max) EXACTLY, because under GBM the step is exactly
lognormal.  Under Heston-Nandi that exactness survives only for a SINGLE model step:
HN is conditionally Gaussian GIVEN h.  A weekly/monthly fixing spans n ~ 5-21 daily HN
steps and the aggregate is a normal MIXTURE, not a normal.

The proposed cheap fix is to keep one OSS step per fixing but widen it to
V = E_t[Sum_{k=1..n} h_{t+k}] (affine, closed-form), i.e. use N(n*r - V/2, V).
This script quantifies the resulting error in the ONE NUMBER OSS CONSUMES: the survival
probability.

The oracle is ``utils.hn_cdf_logret`` -- the EXACT Q(R_n <= b) by Fourier inversion,
validated in tests/test_hn_garch.py against Black-Scholes (1e-12), against the phi=1
martingale identity (1e-14), and against brute-force Monte Carlo.  Monte Carlo is
carried alongside here purely as an independent witness.

Run:  python tb_hn_aggregate_bias.py            (~3 min on a 3090, ~25 min CPU)
Outputs: tb_hn_aggregate_bias.csv       (main survival-probability error table)
         tb_hn_aggregate_moments.csv    (E[Sum h] validation + exact aggregate moments)
         tb_hn_daily_monitoring.csv     (daily monitoring is a different product)
         tb_hn_last_substep_oss.csv     (the exact drop-in)
         tb_hn_compounding.csv          (per-fixing error compounded over a schedule)
"""

import math
import os
import sys
import time

import pandas as pd
import torch

# HN core now lives in riskflow.utils; the test-only reference helpers in tests/hn_reference.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
from riskflow import utils
import hn_reference as hnref

DT = torch.float64
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SPY = 252.0                                   # steps per year (daily HN clock)
R_ANN = 0.03
N_MC = 20_000_000                             # paths for the brute-force witness
N_LIST = (5, 21, 63)                          # weekly / monthly / quarterly fixings
Z_GRID = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
H_STATES = (('calm', 0.5), ('neutral', 1.0), ('stressed', 2.0))

# realistic equity/FX-ish risk-neutral HN sets: (name, ann_vol, persistence, gamma*, lev)
PARAM_SETS = (
    ('A_equity_psi095', 0.20, 0.95, 250.0, 0.10),
    ('B_equity_psi098', 0.20, 0.98, 400.0, 0.12),
    ('C_fx_psi099_lowskew', 0.25, 0.99, 200.0, 0.05),
)


def _t(x):
    return torch.tensor(float(x), dtype=DT)


def _ncdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _npdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def edgeworth_cdf(z, skew, exkurt):
    """4th-order Edgeworth CDF of the STANDARDISED aggregate.

    F(z) ~ Phi(z) - phi(z) * [ g1/6 * He2 + g2/24 * He3 + g1^2/72 * He5 ]
    with He2 = z^2-1, He3 = z^3-3z, He5 = z^5-10z^3+15z.  Not a proper CDF -- it can
    leave [0,1] and lose monotonicity in the far tail; that is reported, not hidden.
    """
    he2 = z * z - 1.0
    he3 = z ** 3 - 3.0 * z
    he5 = z ** 5 - 10.0 * z ** 3 + 15.0 * z
    return _ncdf(z) - _npdf(z) * (skew / 6.0 * he2 + exkurt / 24.0 * he3
                                  + skew ** 2 / 72.0 * he5)


# ======================================================================================
# section 1 -- E_t[Sum h] closed form vs brute force
# ======================================================================================

def section1_expected_sum_h(sets):
    print('\n' + '=' * 100)
    print('SECTION 1 -- E_t[Sum_{k=1..n} h_{t+k}] closed form vs brute-force MC')
    print('  E_t[h_{t+k}] = m + psi^(k-1) (h1 - m),  m = (omega+alpha)/(1-psi)')
    print('  E_t[Sum]     = n*m + (h1-m)(1-psi^n)/(1-psi)         EXACT at n=1')
    print('=' * 100)
    rows = []
    for name, p, m in sets:
        for hname, hmult in H_STATES:
            h1 = hmult * m
            for n in (1, 5, 21, 63):
                cf = float(hnref.hn_expected_sum_h(p, n, _t(h1)))
                mc, se = hnref.hn_simulate_sum_h(p, n, h1, 4_000_000, seed=17, device=DEV)
                z = (cf - mc) / se if se > 0 else 0.0
                rows.append(dict(param_set=name, h_state=hname, n=n, closed_form=cf,
                                 mc=mc, mc_se=se, z=z))
                print('  %-20s %-9s n=%2d  cf=%.8e  mc=%.8e  se=%.1e  z=%+6.2f%s'
                      % (name, hname, n, cf, mc, se, z,
                         '   <-- EXACT (deterministic)' if n == 1 else ''))
    return pd.DataFrame(rows)


# ======================================================================================
# section 2 -- moments of the true aggregate vs the bridge
# ======================================================================================

def section2_moments(sets):
    print('\n' + '=' * 100)
    print('SECTION 2 -- moments of the TRUE n-step aggregate vs the mean-matched bridge')
    print('  bridge = N(n*r - V/2, V),  V = E[Sum h].  kappa1 matches BY CONSTRUCTION.')
    print('  kappa2/V > 1 => the bridge is NOT even variance-matched.')
    print('=' * 100)
    print('  %-20s %-9s %3s %12s %12s %8s %9s %9s'
          % ('set', 'h_state', 'n', 'V=E[Sum h]', 'kappa2', 'k2/V', 'skew', 'ex_kurt'))
    rows = []
    for name, p, m in sets:
        for hname, hmult in H_STATES:
            h1 = hmult * m
            for n in N_LIST:
                v = float(hnref.hn_expected_sum_h(p, n, _t(h1)))
                k1, k2, sk, ek = hnref.hn_moments(p, n, _t(h1))
                rows.append(dict(param_set=name, h_state=hname, n=n, V=v, kappa1=k1,
                                 kappa2=k2, var_ratio=k2 / v, skew_true=sk, ex_kurt=ek,
                                 mean_check=k1 - (float(p.r) * n - 0.5 * v)))
                print('  %-20s %-9s %3d %12.6e %12.6e %8.5f %+9.4f %9.4f'
                      % (name, hname, n, v, k2, k2 / v, sk, ek))
    return pd.DataFrame(rows)


# ======================================================================================
# section 3 -- THE error table: survival probability, true vs bridge
# ======================================================================================

def section3_survival(sets):
    print('\n' + '=' * 100)
    print('SECTION 3 -- survival probability P(R_n <= b), true vs mean-matched normal')
    print('  b = mu + z*sqrt(V), mu = n*r - V/2.  z<0 == DOWN barrier, z>0 == UP barrier.')
    print('  p_ko is the KNOCK-OUT probability (the small tail): P(R>b) for z>0,')
    print('  P(R<=b) for z<0.  rel_err is on p_ko -- that is what an OSS step pays out.')
    print('=' * 100)
    rows = []
    for name, p, m in sets:
        for hname, hmult in H_STATES:
            h1 = hmult * m
            for n in N_LIST:
                v = float(hnref.hn_expected_sum_h(p, n, _t(h1)))
                sd = math.sqrt(v)
                mu = float(p.r) * n - 0.5 * v
                _, k2, sk, ek = hnref.hn_moments(p, n, _t(h1))
                sd_t = math.sqrt(k2)
                bs = [mu + z * sd for z in Z_GRID]
                true = utils.hn_cdf_logret(torch.tensor(bs, dtype=DT), p, n, h1).tolist()
                # independent witness
                r = hnref.hn_simulate(p, n, h1, N_MC, seed=23, device=DEV)
                for z, b, pt in zip(Z_GRID, bs, true):
                    ind = (r <= b).to(DT)
                    mc, se = float(ind.mean()), float(ind.std() / math.sqrt(len(ind)))
                    p_norm = _ncdf(z)                                  # mean-matched
                    p_vm = _ncdf((b - mu) / sd_t)                      # variance-matched
                    p_eg = edgeworth_cdf((b - mu) / sd_t, sk, ek)      # Edgeworth-4
                    up = z > 0
                    ko = (lambda x: 1.0 - x) if up else (lambda x: x)
                    rows.append(dict(
                        param_set=name, h_state=hname, n=n, z=z, barrier_logret=b,
                        barrier=('UP' if up else 'DOWN'),
                        p_surv_true=pt, p_surv_bridge=p_norm, p_surv_varmatch=p_vm,
                        p_surv_edgeworth=p_eg, p_surv_mc=mc, mc_se=se,
                        p_ko_true=ko(pt), p_ko_bridge=ko(p_norm),
                        p_ko_varmatch=ko(p_vm), p_ko_edgeworth=ko(p_eg),
                        abs_err_bridge=ko(p_norm) - ko(pt),
                        rel_err_bridge=(ko(p_norm) - ko(pt)) / ko(pt),
                        rel_err_varmatch=(ko(p_vm) - ko(pt)) / ko(pt),
                        rel_err_edgeworth=(ko(p_eg) - ko(pt)) / ko(pt),
                        mc_z=(pt - mc) / se if se > 0 else 0.0))
    df = pd.DataFrame(rows)
    bad = df.loc[df.mc_z.abs() > 4.5]
    print('  MC witness: max |z| = %.2f over %d rows (%d rows beyond 4.5 sigma)'
          % (df.mc_z.abs().max(), len(df), len(bad)))
    return df


def print_error_tables(df):
    for name in df.param_set.unique():
        for n in N_LIST:
            sub = df[(df.param_set == name) & (df.n == n) & (df.h_state == 'neutral')]
            if not len(sub):
                continue
            print('\n  --- %s, n=%d, h1 = stationary ---' % (name, n))
            print('   %5s %5s %13s %13s %11s %10s | %10s %10s'
                  % ('z', 'side', 'p_ko_true', 'p_ko_bridge', 'abs_err',
                     'rel_err', 'rel_varm', 'rel_edgew'))
            for _, r in sub.iterrows():
                print('   %+5.1f %5s %13.6e %13.6e %+11.3e %+9.2f%% | %+9.2f%% %+9.2f%%'
                      % (r.z, r.barrier, r.p_ko_true, r.p_ko_bridge, r.abs_err_bridge,
                         100 * r.rel_err_bridge, 100 * r.rel_err_varmatch,
                         100 * r.rel_err_edgeworth))


# ======================================================================================
# section 4 -- daily monitoring is a DIFFERENT product (reference, not a drop-in)
# ======================================================================================

def section4_daily_monitoring(sets):
    print('\n' + '=' * 100)
    print('SECTION 4 -- daily sub-stepped OSS applies the barrier EVERY DAY.')
    print('  That prices a daily-monitored barrier, not a discrete fixing.  Below:')
    print('  P(R_n <= b) [the fixing] vs P(max_k R_k <= b) [daily monitoring].')
    print('=' * 100)
    rows = []
    for name, p, m in sets:
        h1 = m
        for n in N_LIST:
            v = float(hnref.hn_expected_sum_h(p, n, _t(h1)))
            mu = float(p.r) * n - 0.5 * v
            sd = math.sqrt(v)
            r, rmax, rmin = hnref.hn_simulate(p, n, h1, N_MC // 4, seed=29, device=DEV,
                                           track_extrema=True)
            for z in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0):
                b = mu + z * sd
                up = z > 0
                run = rmax if up else rmin
                ko_fix = float(((r > b) if up else (r <= b)).to(DT).mean())
                ko_day = float(((run > b) if up else (run <= b)).to(DT).mean())
                exact = utils.hn_cdf_logret(_t(b), p, n, h1)
                ko_exact = float(1.0 - exact) if up else float(exact)
                rows.append(dict(param_set=name, n=n, z=z,
                                 barrier=('UP' if up else 'DOWN'),
                                 p_ko_fixing_exact=ko_exact, p_ko_fixing_mc=ko_fix,
                                 p_ko_daily_monitored=ko_day,
                                 ratio_daily_over_fixing=ko_day / max(ko_fix, 1e-12)))
                print('  %-20s n=%2d z=%+4.1f %4s  KO(fixing)=%.5e  KO(daily-mon)=%.5e'
                      '  ratio=%6.2fx' % (name, n, z, 'UP' if up else 'DOWN', ko_exact,
                                          ko_day, ko_day / max(ko_fix, 1e-12)))
    return pd.DataFrame(rows)


# ======================================================================================
# section 5 -- the fix that keeps OSS exact: truncate on the LAST daily sub-step
# ======================================================================================

def last_substep_oss(p, n, h1, b, n_paths, seed, device=DEV):
    """OSS variant that is EXACT for a discretely-monitored fixing.

    Simulate the first n-1 daily HN steps unconditionally (they are not monitored), then
    apply the analytic one-step survival on the FINAL daily step, where the model IS
    conditionally Gaussian:

        p_i = Phi( (b - x_{n-1} - (r - h_n/2)) / sqrt(h_n) )

    E[p_i] = P(R_n <= b) exactly, and the surviving draw is a plain truncated normal, so
    the whole OSS machinery is unchanged -- only the number of diffusion steps grows.
    Returns (estimate, standard_error).
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    om, al, be, ga, r = (float(x) for x in (p.omega, p.alpha, p.beta, p.gamma, p.r))
    h = torch.full((n_paths,), float(h1), dtype=DT, device=device)
    x = torch.zeros((n_paths,), dtype=DT, device=device)
    for _ in range(int(n) - 1):
        z = torch.randn((n_paths,), generator=g, dtype=DT, device=device)
        sh = h.sqrt()
        x = x + (r - 0.5 * h + sh * z)
        h = om + be * h + al * (z - ga * sh) ** 2
    zmax = (b - x - (r - 0.5 * h)) / h.sqrt()
    ps = 0.5 * torch.erfc(-zmax / math.sqrt(2.0))
    return float(ps.mean()), float(ps.std() / math.sqrt(n_paths))


def section5_exact_oss(sets):
    print('\n' + '=' * 100)
    print('SECTION 5 -- the drop-in that stays EXACT: move the OSS truncation to the')
    print('  final daily sub-step of each fixing interval (product UNCHANGED).')
    print('=' * 100)
    rows = []
    for name, p, m in sets:
        h1 = m
        for n in N_LIST:
            v = float(hnref.hn_expected_sum_h(p, n, _t(h1)))
            mu = float(p.r) * n - 0.5 * v
            for z in (-2.0, 2.0):
                b = mu + z * math.sqrt(v)
                exact = float(utils.hn_cdf_logret(_t(b), p, n, h1))
                est, se = last_substep_oss(p, n, h1, b, 4_000_000, seed=31)
                bridge = _ncdf(z)
                rows.append(dict(param_set=name, n=n, z=z, p_surv_exact=exact,
                                 p_surv_last_substep_oss=est, se=se,
                                 z_score=(est - exact) / se, p_surv_bridge=bridge))
                print('  %-20s n=%2d z=%+4.1f  exact=%.8f  last-substep OSS=%.8f (se %.1e,'
                      ' z=%+5.2f)  bridge=%.8f (err %+9.2e)'
                      % (name, n, z, exact, est, se, (est - exact) / se, bridge,
                         bridge - exact))
    return pd.DataFrame(rows)


def section6_cf_cost(sets):
    """Timing of the third option: exact CF inversion per fixing, vectorised over paths."""
    print('\n' + '=' * 100)
    print('SECTION 6 -- cost of the EXACT route (CF inversion, vectorised over paths)')
    print('  A(phi), B(phi) depend only on (n, params) -- NOT on h1 or on the barrier --')
    print('  so one n-step recursion serves the whole path batch; per path it is one')
    print('  complex exp over the quadrature grid.')
    print('=' * 100)
    name, p, m = sets[1]
    for dt, lab in ((torch.float64, 'complex128'), (torch.float32, 'complex64')):
        pg = utils.HNParams(*[torch.tensor(float(x), dtype=dt, device=DEV)
                           for x in (p.omega, p.alpha, p.beta, p.gamma, p.r)])
        for n in (5, 21):
            for npaths in (10_000, 100_000):
                for nodes in (128, 512):
                    h1 = torch.linspace(0.5 * m, 2.0 * m, npaths, dtype=dt, device=DEV)
                    b = torch.linspace(-0.2, 0.2, npaths, dtype=dt, device=DEV)
                    kw = dict(phi_max=512.0, panels=nodes // 8)
                    utils.hn_cdf_logret(b, pg, n, h1, **kw)
                    if DEV == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.time()
                    for _ in range(5):
                        utils.hn_cdf_logret(b, pg, n, h1, **kw)
                    if DEV == 'cuda':
                        torch.cuda.synchronize()
                    print('  %-10s n=%2d paths=%7d nodes=%4d  %8.2f ms per fixing (%s)'
                          % (lab, n, npaths, nodes, 1000 * (time.time() - t0) / 5, DEV))


# ======================================================================================

def build_sets():
    out = []
    for name, vol, psi, gam, lev in PARAM_SETS:
        p = hnref.hn_params_from_targets(vol, psi, gam, lev, r=R_ANN / SPY,
                                      steps_per_year=SPY).as_tensors(DT)
        m = float(p.stationary_var)
        out.append((name, p, m))
        print('  %-20s ann_vol=%.1f%%  psi=%.4f  gamma*=%6.0f  omega=%.4e alpha=%.4e '
              'beta=%.4f  m=%.6e' % (name, 100 * float(p.ann_vol(SPY)),
                                     float(p.persistence), float(p.gamma),
                                     float(p.omega), float(p.alpha), float(p.beta), m))
    return out


def section7_compounding(df3):
    """The per-fixing error is not the end of it: a TARF's survival weight is
    L = prod_i p_i over the whole schedule, so a systematic per-fixing bias compounds
    geometrically.  This is the number that actually reaches the PV."""
    print('\n' + '=' * 100)
    print('SECTION 7 -- compounding: TARF survival weight L = prod(p_i) over F fixings')
    print('  reported as (L_approx / L_true - 1); survival = 1 - p_ko.')
    print('=' * 100)
    d = df3.copy()
    d['surv_true'] = 1 - d.p_ko_true
    d['surv_bridge'] = 1 - d.p_ko_bridge
    d['surv_edge'] = 1 - d.p_ko_edgeworth
    rows = []
    for n, f, lab in ((5, 52, 'weekly, 1y  '), (21, 12, 'monthly, 1y '),
                      (21, 24, 'monthly, 2y ')):
        s = d[(d.n == n) & (d.h_state == 'neutral')]
        for z in (-3.0, -2.0, 2.0, 3.0):
            r = s[s.z == z]
            br = ((r.surv_bridge / r.surv_true) ** f - 1) * 100
            eg = ((r.surv_edge / r.surv_true) ** f - 1) * 100
            rows.append(dict(schedule=lab.strip(), n=n, fixings=f, z=z,
                             barrier=r.barrier.iloc[0], bridge_lo=br.min(),
                             bridge_hi=br.max(), edgeworth_lo=eg.min(),
                             edgeworth_hi=eg.max()))
            print('  %s z=%+.1f %4s  bridge %+7.1f%% .. %+7.1f%%   |  Edgeworth-4'
                  ' %+6.2f%% .. %+6.2f%%'
                  % (lab, z, r.barrier.iloc[0], br.min(), br.max(), eg.min(), eg.max()))
    return pd.DataFrame(rows)


def verdict(df3, df2):
    print('\n' + '=' * 100)
    print('VERDICT SUMMARY -- worst |relative error| in the KO probability')
    print('=' * 100)
    print('  %4s %28s %10s %10s %10s' % ('n', 'approximation', '|z|<=1', '|z|=2', '|z|=3'))
    for n in N_LIST:
        for col, lab in (('rel_err_bridge', 'mean-matched normal'),
                         ('rel_err_varmatch', 'variance-matched normal'),
                         ('rel_err_edgeworth', 'Edgeworth-4 (exact k1..k4)')):
            s = df3[df3.n == n]
            a = 100 * s[s.z.abs() <= 1.0][col].abs().max()
            b = 100 * s[s.z.abs() == 2.0][col].abs().max()
            c = 100 * s[s.z.abs() == 3.0][col].abs().max()
            print('  %4d %28s %9.2f%% %9.2f%% %9.2f%%' % (n, lab, a, b, c))
    print('\n  worst variance shortfall of the bridge (kappa2/V - 1):')
    for n in N_LIST:
        s = df2[df2.n == n]
        print('    n=%2d  %+.3f%%   (true skew range %+.3f..%+.3f, ex-kurt up to %.3f)'
              % (n, 100 * (s.var_ratio.max() - 1), s.skew_true.min(), s.skew_true.max(),
                 s.ex_kurt.max()))


if __name__ == '__main__':
    torch.manual_seed(0)
    print('device=%s  MC paths=%d' % (DEV, N_MC))
    sets = build_sets()
    df1 = section1_expected_sum_h(sets)
    df2 = section2_moments(sets)
    df3 = section3_survival(sets)
    print_error_tables(df3)
    df4 = section4_daily_monitoring(sets)
    df5 = section5_exact_oss(sets)
    section6_cf_cost(sets)
    df7 = section7_compounding(df3)
    verdict(df3, df2)

    df3.to_csv('tb_hn_aggregate_bias.csv', index=False)
    df1['section'] = 'expected_sum_h'
    df2['section'] = 'moments'
    pd.concat([df1, df2], ignore_index=True).to_csv('tb_hn_aggregate_moments.csv',
                                                    index=False)
    df4.to_csv('tb_hn_daily_monitoring.csv', index=False)
    df5.to_csv('tb_hn_last_substep_oss.csv', index=False)
    df7.to_csv('tb_hn_compounding.csv', index=False)
    print('\nwrote tb_hn_aggregate_bias.csv, tb_hn_aggregate_moments.csv, '
          'tb_hn_daily_monitoring.csv, tb_hn_last_substep_oss.csv, tb_hn_compounding.csv')
