"""Validation tests A-D for BasisLinkedSpotModel against basis_model_spec.md section 6.

Calibrates the model directly from plat_archive.csv via the framework's
BasisLinkedSpotCalibration class, then runs spec tests A-D.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from riskflow.stochasticprocess import BasisLinkedSpotCalibration


# --- Calibrate ---------------------------------------------------------------

df = pd.read_csv('data/plat_archive.csv', index_col=0)
basis_col = 'CommodityBasis.LME_CME,PLATINUM_LME'
lme_col = 'CommodityPrice.PLATINUM_LME'
sub = df[[basis_col, lme_col]].dropna()

cal = BasisLinkedSpotCalibration(model=None, param={})
info = cal.calibrate(sub, vol_shift=0.0)
P = info.param

A_, PHI_B, NU = P['A'], P['Phi'], P['Nu']
SIGMA = P['Sigma_By_State']
SIGMA_POOLED = float(np.sqrt(np.mean([s ** 2 for s in SIGMA])))

print('Calibrated parameters')
print('  A          =', f'{A_:.10f}')
print('  Phi        =', f'{PHI_B:.10f}')
print('  Nu         =', f'{NU:.6f}')
print('  Sigma[0,1,2]=', [f'{s:.4f}' for s in SIGMA])
print('  Sigma_pool =', f'{SIGMA_POOLED:.4f}')
print()
print('Spec reference values (basis_model_spec.md)')
print('  A          = -0.0966959839')
print('  Phi        =  0.4100073467')
print('  Nu         =  5.21680652')
print('  Sigma[0,1,2]= [11.0837994, 14.6830903, 19.5087432]')
print('  Sigma_pool = 15.4652445')
print()


# --- Step function (matches spec §4 NumPy ref, parameterized) ---------------

def t_innovation(rng, sigma, nu):
    z = rng.standard_normal()
    w = rng.chisquare(nu)
    return sigma * z * np.sqrt((nu - 2.0) / w)


def basis_step(b_prev, dlme, regime_state, rng, regime_keyed=True):
    sigma = SIGMA[regime_state] if regime_keyed else SIGMA_POOLED
    return A_ * dlme + PHI_B * b_prev + t_innovation(rng, sigma, NU)


# --- Test A: round-trip moments using historical ΔLME -----------------------

def test_A():
    LME = sub[lme_col].values
    basis_emp = sub[basis_col].values
    dlme_hist = np.diff(LME)
    n = len(LME)
    rng = np.random.default_rng(42)
    n_paths = 200
    stds, kurts = [], []
    for _ in range(n_paths):
        b = np.zeros(n)
        for t in range(1, n):
            b[t] = basis_step(b[t - 1], dlme_hist[t - 1], regime_state=1, rng=rng,
                              regime_keyed=False)
        stds.append(b[100:].std())
        kurts.append(scipy_stats.kurtosis(b[100:], fisher=False) - 0.0)
    sim_std, sim_kurt = float(np.mean(stds)), float(np.mean(kurts))
    emp_std = float(basis_emp.std())
    rel_err = abs(sim_std - emp_std) / emp_std
    pass_std = rel_err < 0.10
    pass_kurt = 2.0 < sim_kurt < 8.0
    print(f'Test A — Round-trip moments (historical ΔLME, pooled σ, {n_paths} paths)')
    print(f'  sim std      = {sim_std:.3f}    emp std = {emp_std:.3f}    rel_err = {rel_err:.2%}    [{"PASS" if pass_std else "FAIL"}, tol 10%]')
    print(f'  sim kurt     = {sim_kurt:.3f}   range = (2, 8)                              [{"PASS" if pass_kurt else "FAIL"}]')
    return pass_std and pass_kurt


# --- Test B: quantile match on long-run bootstrap ---------------------------

def test_B():
    LME = sub[lme_col].values
    basis_emp = sub[basis_col].values
    dlme_pool = np.diff(LME)
    rng = np.random.default_rng(7)
    T = 100_000
    dlme_boot = rng.choice(dlme_pool, T - 1)
    b = np.zeros(T)
    for t in range(1, T):
        b[t] = basis_step(b[t - 1], dlme_boot[t - 1], regime_state=1, rng=rng,
                          regime_keyed=False)
    b = b[1000:]
    targets = {0.005: -55.6, 0.01: -44.4, 0.05: -25.3,
               0.95:  25.1, 0.99:  44.8, 0.995: 60.9}
    emp_q = {q: float(np.quantile(basis_emp, q)) for q in targets}
    print('Test B — Quantile match (bootstrap ΔLME, pooled σ, T=100k)')
    print(f'  {"q":>6} {"sim":>10} {"target":>10} {"empirical":>11} {"rel_err":>10} {"tol":>6} {"verdict":>8}')
    all_pass = True
    for q, target in targets.items():
        sim_q = float(np.quantile(b, q))
        rel_err = abs(sim_q - target) / abs(target)
        tol = 0.10 if q <= 0.5 else 0.18
        verdict = 'PASS' if rel_err < tol else 'FAIL'
        if rel_err >= tol:
            all_pass = False
        print(f'  {q:>6.3f} {sim_q:>10.2f} {target:>10.2f} {emp_q[q]:>11.2f} {rel_err:>9.2%} {tol:>5.0%} {verdict:>8}')
    return all_pass


# --- Test C: innovation independence from ΔLME ------------------------------

def test_C():
    rng = np.random.default_rng(0)
    T = 20_000
    dlme_seq = rng.standard_normal(T) * 16.0
    b = np.zeros(T)
    for t in range(1, T):
        b[t] = basis_step(b[t - 1], dlme_seq[t - 1], regime_state=1, rng=rng,
                          regime_keyed=False)
    inno = b[1:] - A_ * dlme_seq[:-1] - PHI_B * b[:-1]
    rho = float(np.corrcoef(inno, dlme_seq[:-1])[0, 1])
    verdict = 'PASS' if abs(rho) < 0.05 else 'FAIL'
    print('Test C — Innovation independence from ΔLME (synthetic ΔLME, T=20k)')
    print(f'  corr(η, ΔLME) = {rho:+.4f}    |·| < 0.05    [{verdict}]')
    return abs(rho) < 0.05


# --- Test D: regime-keyed σ recovery ----------------------------------------

def test_D():
    rng = np.random.default_rng(0)
    T = 20_000
    print('Test D — Regime-keyed σ recovery (zero ΔLME, T=20k per state)')
    print(f'  {"state":>6} {"target σ":>10} {"sim σ":>10} {"rel_err":>10} {"verdict":>8}')
    all_pass = True
    for state in (0, 1, 2):
        dlme = np.zeros(T)
        b = np.zeros(T)
        for t in range(1, T):
            b[t] = basis_step(b[t - 1], dlme[t - 1], regime_state=state, rng=rng,
                              regime_keyed=True)
        inno = b[1:] - PHI_B * b[:-1]
        target = SIGMA[state]
        sim_sigma = float(inno.std())
        rel_err = abs(sim_sigma - target) / target
        verdict = 'PASS' if rel_err < 0.05 else 'FAIL'
        if rel_err >= 0.05:
            all_pass = False
        print(f'  {state:>6} {target:>10.4f} {sim_sigma:>10.4f} {rel_err:>9.2%} {verdict:>8}')
    return all_pass


# --- Run all -----------------------------------------------------------------

results = {
    'A': test_A(),
    'B': test_B(),
    'C': test_C(),
    'D': test_D(),
}
print()
print('=' * 60)
print('Summary:', ', '.join(f'{k}={"PASS" if v else "FAIL"}' for k, v in results.items()))
print('Overall:', 'PASS' if all(results.values()) else 'FAIL')
