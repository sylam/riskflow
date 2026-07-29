# Platinum LBMA-CME Simulator Specification

Empirical findings from `plat.csv` (3,856 daily rows, 2009-11-10 → 2025-12-30) and the resulting joint stochastic process specification for use in a vectorized MC simulator (RiskFlow-compatible).

## 1. Model architecture

Three independent stochastic primitives, each calibrated separately:

1. **CME synthetic spot `S(t)`** — 3-state HMM with state-conditional Gaussian emissions on daily diffs
2. **Carry term structure `c(τ, t)`** — 2-factor OU in PCA space with correlated Brownian innovations
3. **Basis residual `ξ(t)`** — fast OU around zero, with σ scaled per spot-regime

Two derived observables per simulated day:

```
LBMA(t)  = 0.40·S(t) + 0.60·S(t-1) + ξ(t)
F_i(t)   = S(t) · exp((r_i(t) + c_i(t)) · τ_i(t))      for i = 1,2,3
```

with `c(t) = c̄ + V · X(t)`, where `X(t) = (X₁, X₂)` is the 2-factor latent state and `V` is the 3×2 PCA loading matrix.

## 2. Spot dynamics — 3-state HMM

Calibrate Baum-Welch on `df['synthetic_spot'].diff()` (or `df['LBME_PLATINUM'].diff()`; nearly equivalent). Per-state parameters `(μ_k, σ_k)` for k ∈ {1,2,3} plus 3×3 row-stochastic transition matrix `P`.

**Empirical targets the fit must reproduce:**

| Statistic | Synthetic spot | LBMA |
|---|---|---|
| Daily diff std | $19.36 | $16.74 |
| Excess kurtosis | 6.80 | 5.93 |
| Skew | — | +0.21 |
| AC(\|Δ\|, lag=1) | — | 0.21 |
| AC(Δ², lag=1) | — | 0.28 |

**Skellam vs Gaussian**: at λ ≈ 14,000 ticks/side/day, Skellam excess kurt = 1/(2λ) ≈ 4×10⁻⁵, indistinguishable from Gaussian. Use Gaussian emissions for fitting. Optional post-sampling: round simulated diffs to $0.10 grid (matches empirical tick structure: 100% PL futures, 98.75% LBMA).

**State labels (expected, after fit):** "calm" (σ ~$10), "normal" (σ ~$18), "stressed" (σ ~$35-50). Fat tails come from time-in-stressed-state, not from a single non-Gaussian distribution.

## 3. Carry term structure — 2-factor OU

Calibrate via PCA on `df[['PL1_carry', 'PL2_carry', 'PL3_carry']]`, after filtering `df['PL1_tau'] > 0.03` (drops 356 near-expiry rows where 1/τ amplification destabilizes the carry estimate).

**Empirical PCA results:**

| Factor | Variance share | Loadings (PL1, PL2, PL3) | Half-life | Stationary σ |
|---|---|---|---|---|
| PC1 | 95.8% | (−0.294, +0.660, +0.691) | 21.5 days | 132.5 bps |
| PC2 | 3.3% | (−0.384, −0.744, +0.547) | 7.2 days | 24.5 bps |
| PC3 | 0.9% | (negligible) | — | — |

Drop PC3. The negative PL1 loading on PC1 is partly an OLS-fit artifact: the spot-fit imposes 2 linear constraints on the 3 carries, leaving 2 effective DOF.

**SDE form:**
```
dX₁ = −κ₁ X₁ dt + σ₁ dW₁         κ₁ = 0.032 / day      σ₁ = 33.6 bps/√day
dX₂ = −κ₂ X₂ dt + σ₂ dW₂         κ₂ = 0.097 / day      σ₂ = 10.8 bps/√day
d⟨W₁, W₂⟩ = ρ dt                  ρ  = −0.68
```

**Daily-step AR(1) implementation form (use this for the simulator):**
```
X₁(t) = 0.968 · X₁(t−1) + ε₁(t)
X₂(t) = 0.908 · X₂(t−1) + ε₂(t)
(ε₁, ε₂) ~ N(0, Σ),    Σ = [[33.4², ρ·33.4·10.5],
                              [ρ·33.4·10.5, 10.5²]]   bps²
```

ρ = −0.68 is **load-bearing**. Independent Brownians would mis-specify joint dynamics. Either Cholesky the 2×2 innovation covariance, or re-orthogonalize via PCA on innovations (yields different loadings, equivalent model).

**Reconstruction:**
```
c̄ = (+3.9, −3.9, −10.4) bps                      (post-filter mean)
V = [[−0.294, −0.384],
     [+0.660, −0.744],
     [+0.691, +0.547]]
c(t) = c̄ + V · X(t)
```

## 4. Basis residual

Decomposition: `basis_t = β · ΔS_t + ξ_t`, where `basis_t = LBMA_t − S_t`.

| Parameter | Value | Source |
|---|---|---|
| β | −0.60 | OLS, no intercept; explains 49% of basis variance |
| AR(1) of ξ | 0.43 | half-life 0.8 days |
| σ_ξ stationary (unconditional) | $11.83 | residual std |
| σ_ξ innovation (daily AR(1) form) | $10.8 | from stationary × √(1−φ²) |

**Daily-step form:**
```
ξ(t) = 0.43 · ξ(t−1) + η(t),    η(t) ~ N(0, $10.8²)
```

**Regime coupling — non-optional.** σ_ξ scales by ~1.84× from low-vol to high-vol regimes (year-by-year basis std ranges $7.21 in 2017 to $30.05 in 2020). Fit σ_ξ per spot-regime jointly with the HMM, or as a multiplier on the spot-regime σ. Three free parameters: `(σ_ξ^calm, σ_ξ^normal, σ_ξ^stressed)`. Approximate values to test against: $7-10, $12-15, $22-30.

**Structural justification for β:** LBMA PM fix at 14:00 GMT precedes CME PL settle at ~18:00 GMT. Most platinum daily price discovery occurs in the NY session, post-LBMA-fix. β reflects the share of daily move that lands between the two snapshots. Confirms via: corr(basis_t, ΔCME_t) = −0.70, corr(basis_t, ΔLBMA_{t+1}) = −0.64, corr(basis_t, ΔCME_{t+1}) ≈ 0.

## 5. Per-day simulation algorithm

Given previous-day state `(S_{t-1}, X_{t-1}, ξ_{t-1}, regime k_{t-1})`:

```
1.  Sample regime k_t ~ P[· | k_{t-1}]
2.  Sample spot innovation: dS ~ N(μ_{k_t}, σ_{k_t})
3.  S(t) = S(t-1) + dS
4.  Sample (ε₁, ε₂) ~ N(0, Σ),  step X(t) = diag(0.968, 0.908)·X(t-1) + (ε₁, ε₂)
5.  Sample η ~ N(0, σ_ξ(k_t)²),  step ξ(t) = 0.43·ξ(t-1) + η
6.  c(t) = c̄ + V · X(t)
7.  LBMA(t) = 0.40·S(t) + 0.60·S(t-1) + ξ(t)
8.  For i ∈ {1,2,3}: F_i(t) = S(t) · exp((r_i(t) + c_i(t)) · τ_i(t))
```

Vectorize across paths along the batch axis.

**Roll handling**: when `τ_i(t) ≤ 0`, advance contract i to the next-out maturity. Reset that tenor's τ to the new contract's day-count. Track `vendor_rolled` flag in calibration data — 121 such days (3.1%) — to handle observation-side discontinuities, but the model itself is roll-aware via τ_i(t).

**SOFR**: deterministic curve `r_i(t)`, taken from `df[f'PL{i}_rf']` for historical replay or extrapolated forward from a fitted curve for forward simulation.

## 6. Joint correlation summary

| Coupling | Status | Value |
|---|---|---|
| PC1 ↔ PC2 innovation | implemented | ρ = −0.68 |
| Spot regime ↔ basis σ_ξ | implemented | per-regime fit |
| Spot ΔS ↔ basis | implemented (deterministic) | β·ΔS term |
| Spot innovation ↔ carry factor innovations | **not yet measured** | open |
| Spot regime ↔ term-structure σ | **not yet measured** | open |

The open items above should be checked before final commit. A non-zero spot ↔ carry correlation would mean a 3×3 (or larger) Cholesky inside the inner loop. Likely small for platinum but should be quantified.

## 7. Validation anchors

Sampled paths from the calibrated simulator must reproduce:

- **LBMA daily diff**: std $16.7 ± 5%, excess kurt 5–7, skew ~0.2
- **Vol clustering**: AC(\|Δ\|, lag=1) ∈ [0.18, 0.25], AC(Δ², lag=1) ∈ [0.25, 0.32]
- **Basis std**: $16.5 ± 10% unconditional; ratio σ_basis^high / σ_basis^low ≈ 1.8 across vol quartiles
- **Carry stds (bps)**: PL1 ~40, PL2 ~95, PL3 ~95
- **Carry correlations**: PL1-PL2 ≈ −0.81, PL1-PL3 ≈ −0.90, PL2-PL3 ≈ +0.93
- **Year-by-year basis std**: simulated 16-year tracks should span $7-30 with vol-correlated pattern

If any of these miss by more than the tolerance above, the most likely culprits are (in order): (1) too few HMM states, (2) ρ between PC innovations not implemented, (3) σ_ξ fit globally rather than per-regime.

## 8. Calibration recipe (reference)

```
# 1. HMM on spot
from hmmlearn.hmm import GaussianHMM
diffs = df['synthetic_spot'].diff().dropna().values.reshape(-1,1)
hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200).fit(diffs)
# extract: hmm.transmat_, hmm.means_, np.sqrt(hmm.covars_)
# Posterior decode: states = hmm.predict(diffs)

# 2. PCA on carries
mask = df['PL1_tau'] > 0.03
C = df.loc[mask, ['PL1_carry','PL2_carry','PL3_carry']].values
Cc = C - C.mean(axis=0)
eigvals, eigvecs = np.linalg.eigh(np.cov(Cc.T))
idx = np.argsort(eigvals)[::-1]                 # descending
V = eigvecs[:, idx[:2]]                         # 3x2 loadings
scores = Cc @ V                                 # n x 2
# Per-PC AR(1):
phi = [(scores[:-1,k]*scores[1:,k]).sum() / (scores[:-1,k]**2).sum() for k in (0,1)]
innov = np.column_stack([scores[1:,k] - phi[k]*scores[:-1,k] for k in (0,1)])
Sigma_innov = np.cov(innov.T)
rho = Sigma_innov[0,1] / np.sqrt(Sigma_innov[0,0]*Sigma_innov[1,1])

# 3. Basis decomposition
basis = (df['LBME_PLATINUM'] - df['synthetic_spot']).values
dS = df['synthetic_spot'].diff().values
m = ~np.isnan(dS)
beta = (basis[m]*dS[m]).sum() / (dS[m]**2).sum()
xi = basis - beta*dS
phi_xi = np.corrcoef(xi[m][:-1], xi[m][1:])[0,1]
sigma_xi_stat = xi[m].std()
# Per-regime σ_ξ: groupby regime label from HMM posterior, std() of xi within each
```

## 9. Data file column reference

`plat.csv` columns used:

| Column | Meaning |
|---|---|
| `synthetic_spot` | Back-solved CME spot from 3-tenor futures fit |
| `LBME_PLATINUM` | LBMA PM fix |
| `PL{1,2,3}` | CME futures settlement prices |
| `PL{1,2,3}_carry` | Implied carry above SOFR per tenor |
| `PL{1,2,3}_tau` | Time to expiry in years (act/360) |
| `PL{1,2,3}_rf` | SOFR curve at each tenor |
| `vendor_rolled` | True on contract roll dates (121/3856) |
| `spot_fit_rmse` | Per-day RMSE of the futures→spot fit |

Filter `PL1_tau > 0.03` for any carry-PCA work. Keep all rows for spot HMM and basis decomposition.
