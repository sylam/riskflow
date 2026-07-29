"""Compare the cross-product (1, τ, w(τ)) basis vs Nelson-Siegel (1, τ, h(τ; α)) basis
for the platinum carry curve VAR(1) model. In-sample fit is exact at the 3 slots for
both bases; the comparison focuses on the dynamics — innovation autocorrelation, σ,
Φ structure — and off-grid predictions.

NS hump basis:  h(τ; α) = (1 - exp(-α τ)) / (α τ)  -  exp(-α τ)
                peak at τ ≈ 1/α
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# -- Load --------------------------------------------------------------------

df = pd.read_csv('data/plat_archive.csv', index_col=0)
carry_cols = sorted([c for c in df.columns if 'PLATINUM_CARRY' in c])
tenor_cols = ['Tenor.PLATINUM_TAU1', 'Tenor.PLATINUM_TAU2', 'Tenor.PLATINUM_TAU3']
sub = df[carry_cols + tenor_cols].dropna()

tau_floor = 0.03
carry = sub[carry_cols].values.astype(np.float64)
tau = sub[tenor_cols].values.astype(np.float64)
mask = tau[:, 0] > tau_floor
carry = carry[mask]
tau = tau[mask]
print(f'Days: {len(carry)}    τ range across slots: '
      f'[{tau[:, 0].min():.3f}, {tau[:, 2].max():.3f}]')


# -- Bases -------------------------------------------------------------------

def w_cross(t):
    """Unit 3-vector orthogonal to (1,1,1) and τ; sign w_2 > 0."""
    w = np.array([t[2] - t[1], t[0] - t[2], t[1] - t[0]])
    n = np.linalg.norm(w)
    w /= n
    if w[1] < 0:
        w = -w
    return w


def h_ns(tau_arr, alpha):
    """NS hump: ((1 - exp(-ατ))/(ατ)) - exp(-ατ). Hump location ≈ 1/α."""
    e = np.exp(-alpha * tau_arr)
    return (1.0 - e) / (alpha * tau_arr) - e


def design_cross(t):
    return np.column_stack([np.ones(3), t, w_cross(t)])


def design_ns(t, alpha):
    return np.column_stack([np.ones(3), t, h_ns(t, alpha)])


# -- Per-day decomposition ---------------------------------------------------

def daily_decompose(design_fn, *args):
    """Returns X_daily (n, 3) and per-day in-sample residual (should be ~0)."""
    X = np.empty((len(carry), 3))
    res = np.empty(len(carry))
    for i in range(len(carry)):
        D = design_fn(tau[i], *args)
        X[i] = np.linalg.solve(D, carry[i])
        res[i] = np.linalg.norm(D @ X[i] - carry[i])
    return X, res


# -- VAR(1) fit --------------------------------------------------------------

def var1_fit(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    X_lag, X_next = Xc[:-1], Xc[1:]
    Phi = np.linalg.solve(X_lag.T @ X_lag, X_lag.T @ X_next).T            # (3, 3)
    innov = X_next - X_lag @ Phi.T                                        # (n-1, 3)
    sigma = innov.std(axis=0)
    return mu, Phi, sigma, innov


def innov_autocorr(innov, lag=1):
    n = innov.shape[1]
    return np.array([
        np.corrcoef(innov[:-lag, k], innov[lag:, k])[0, 1] for k in range(n)
    ])


def report(label, X, mu, Phi, sigma, innov, res):
    print(f'\n=== {label} ===')
    print(f'  per-day in-sample residual:  max={res.max():.2e}    mean={res.mean():.2e}')
    print(f'  μ = [{mu[0]:+.6f}, {mu[1]:+.6f}, {mu[2]:+.6f}]')
    print(f'  σ_innov = [{sigma[0]:.6f}, {sigma[1]:.6f}, {sigma[2]:.6f}]')
    print(f'  Φ eigenvalues: {np.sort(np.abs(np.linalg.eigvals(Phi)))[::-1]}')
    print(f'  innov autocorr lag-1:  [{innov_autocorr(innov, 1)[0]:+.4f}, '
          f'{innov_autocorr(innov, 1)[1]:+.4f}, {innov_autocorr(innov, 1)[2]:+.4f}]')
    print(f'  innov autocorr lag-5:  [{innov_autocorr(innov, 5)[0]:+.4f}, '
          f'{innov_autocorr(innov, 5)[1]:+.4f}, {innov_autocorr(innov, 5)[2]:+.4f}]')
    inn_corr = np.corrcoef(innov.T)
    print(f'  innov cross-corr:  ρ(0,1)={inn_corr[0, 1]:+.3f}  ρ(0,2)={inn_corr[0, 2]:+.3f}  ρ(1,2)={inn_corr[1, 2]:+.3f}')
    # Latent state std (informational)
    print(f'  X std: [{X[:, 0].std():.4f}, {X[:, 1].std():.4f}, {X[:, 2].std():.4f}]')


# -- Compare -----------------------------------------------------------------

X_cp, res_cp = daily_decompose(design_cross)
mu_cp, Phi_cp, sigma_cp, innov_cp = var1_fit(X_cp)
report('cross-product basis (1, τ, w_cross(τ))', X_cp, mu_cp, Phi_cp, sigma_cp, innov_cp, res_cp)

for alpha in [2.0, 3.0, 4.0, 5.0]:
    X_ns, res_ns = daily_decompose(design_ns, alpha)
    mu_ns, Phi_ns, sigma_ns, innov_ns = var1_fit(X_ns)
    report(f'Nelson-Siegel basis (α={alpha:.1f}; hump at τ≈{1/alpha:.3f})',
           X_ns, mu_ns, Phi_ns, sigma_ns, innov_ns, res_ns)


# -- Off-grid: evaluate both on a dense τ grid ------------------------------

print('\n\n=== Off-grid prediction: c(τ_dense, t) ===')
tau_dense = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])

# Pick a representative day (last calibration day) for an example fit.
i = len(carry) - 1
print(f'Evaluating on day {i} (τ={tau[i]} → carry={carry[i]})')

# Cross-product: would need the OLD-code linear interpolation hack
# of w_calib onto tau_dense, since w is a 3-vector tied to slot τ.
w_calib = w_cross(tau[i])
order = np.argsort(tau[i])
w_dense_cp = np.interp(tau_dense, tau[i][order], w_calib[order])
c_dense_cp = X_cp[i, 0] + X_cp[i, 1] * tau_dense + X_cp[i, 2] * w_dense_cp

# NS: native function evaluation
alpha = 3.0
X_ns3, _ = daily_decompose(design_ns, alpha)
c_dense_ns = X_ns3[i, 0] + X_ns3[i, 1] * tau_dense + X_ns3[i, 2] * h_ns(tau_dense, alpha)

print(f'  τ          {"   ".join(f"{t:.2f}" for t in tau_dense)}')
print(f'  cross-prod {"  ".join(f"{c:+.4f}" for c in c_dense_cp)}')
print(f'  NS(α=3)    {"  ".join(f"{c:+.4f}" for c in c_dense_ns)}')

# Sanity: at the slot τs, both should match the data exactly.
print(f'\nAt slot τs = {tau[i]}:')
c_at_slots_cp = design_cross(tau[i]) @ X_cp[i]
c_at_slots_ns = design_ns(tau[i], alpha) @ X_ns3[i]
print(f'  cross-prod  → {c_at_slots_cp}    err = {np.linalg.norm(c_at_slots_cp - carry[i]):.2e}')
print(f'  NS(α=3)     → {c_at_slots_ns}    err = {np.linalg.norm(c_at_slots_ns - carry[i]):.2e}')
print(f'  observed    → {carry[i]}')
