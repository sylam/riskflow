# GARCHSpotModel — implementation spec

Replaces `MarkovHMMSpotModel` as the model for `CommodityPrice.PLATINUM_CME` (the
martingale primary of the platinum hedging world). Written against the framework
contracts as of `SOLVER_VERSION = diffsolverv2/2026-07`; the reference class to mirror
throughout is `MarkovHMMSpotModel` (`stochasticprocess.py:2214`).

## 0. Purpose and design goals

1. **Martingale primary by construction.** `E[Δlog S | everything] = μ·dt` with `Mu`
   defaulting to `0.0`. No regime drifts. All solver edge becomes state-dependent risk
   sizing + cost timing + averaging structure.
2. **Exactly observable sufficient statistic.** The conditional variance `h_t` is a
   deterministic recursion on realized returns — no belief filter, no filter
   misspecification on real data. Revealed to V̂ as `log h_t`.
3. **Drop-in.** Same outer/inner generate contract, same buffer conventions, same
   privileged-state plumbing as the HMM, so `solve_hedge`, the inner-MC forker, and the
   stepper run unchanged. Everything else in the factor graph stays as is:
   `CME_FLAT` (identity basis), `LME_CME` basis, `PLATINUM_CARRY` VAR, `USD-SOFR` PCA,
   `BasisComposedSpotModel` routing for `PLATINUM_LME`.

**Non-goals (measured or deferred):** GJR asymmetry (fitted γ = −0.007, t = −0.9 on the
16y sample — insignificant for platinum; omit the term), jumps (ν ≈ 7.5 tails cover daily
gaps; revisit only if the stepper backtest shows gap under-coverage), component GARCH,
expert-view drift machinery (hook only, §7).

## 1. Model definition

All quantities in **fraction log-return units** at the calibration step
`dt_c = Calibration_DT_Years` (1/252, business-daily).

```
ε_k  ~ standardized Student-t(ν), unit variance
r_k  = sqrt(h_k) · ε_k                          # per-step log return innovation
Δlog S over step k = μ · dt_c + r_k             # μ annualized; default μ = 0
h_{k+1} = ω + α · r_k² + β · h_k                # h in per-step variance units
h_0 = H0                                        # from calibration (filtered variance at base date)
```

Standardized t exactly as the HMM does it (`stochasticprocess.py:2348-2355`):
`W ~ Gamma(ν/2, rate=1/2)` with `.clamp_min(1.0e-6)`, `ε = Z·sqrt(ν/W)·sqrt((ν−2).clamp_min(1e-3)/ν)`,
one framework Gaussian `Z` per step from `t_random_numbers` — **no extra uniform stream**
(unlike the HMM there is no regime sampling; do not call `quasi_rng`).

**No-lookahead invariant (the h index convention).** `h_t` is the variance of the step
`t → t+1` and is a function of returns strictly before `t` (i.e. `r_0..r_{t−1}`), so it
is known at decision time `t`. The revealed coordinate at grid point `t` is `log h_t`.
This invariant has a dedicated test (§6.1).

**dt handling.** The recursion is defined at `dt_c`. Per grid step of `dt` years:
`n_sub = max(1, round(dt / dt_c))`. For `n_sub > 1` (only one Gaussian is
available per grid step): forward `h` deterministically through the sub-steps using
`E[r²] = h` (`h_{j+1} = ω + (α+β)·h_j`), and draw the aggregate return with variance
`Σ_j h_j` from the single `Z`; documented approximation otherwise
(loses within-step vol-of-vol). For `n_sub == 1` see the clock correction below.

### §1a Clock correction (2026-07-18)

The original claim above — "on the production business-daily grid `n_sub = 1`, exact" —
is **wrong about the grid**. The production `Time_Grid "0d 1d(1d)"` is **calendar-daily**:
`config.parse_grid` does not business-day adjust, and `time_grid_years = days / 365.25`, so
each grid step is `dt = 1/365.25` while the recursion is calibrated per business day
`dt_c = 1/252`. Injecting a full per-business-day variance `h` at every calendar step
(≈365/yr) instead of every business day (≈252/yr) inflates annualized vol to
`24.75% × √(365.25/252) ≈ 29.6%` and terminal variance by ×1.45 over a 4-month window.

**Fix — fractional trading clock (implemented).** Let `f_t = dt_t / dt_c` be the
trading-time length of a grid step (`≈0.6897` on the production calendar grid, `1` on a
business-daily grid). When `n_sub == 1` (covers `round(f) == 1`, i.e. the production
`f ≈ 0.69` and the exact `f = 1`), the step is:

```
r_t     = sqrt(h_t · f_t) · ε_t         (+ μ·dt_t)
h_{t+1} = h_t + f_t·(ω − (1−β)·h_t) + α·r_t²
```

Properties (all verified in code + tests):
1. **Standard at `f = 1`** — reduces to `h_{t+1} = ω + α·r_t² + β·h_t` exactly.
2. **Grid-invariant level** — E-fixed-point `ω/(1−α−β)` for ANY `f`, so the annualized
   vol is `24.75%` whether annualized over calendar or business years (regression-gated
   on both clocks + a grid-invariance check).
3. **Grid-invariant rate** — per-step mean-reversion factor `(1 − f(1−α−β))`, so the vol
   half-life in REAL time is grid-independent: `210 bd ≈ 210·(365.25/252) ≈ 304 calendar
   days` on both clocks (regression-gated to 15%).

`H0` stays in per-business-day variance units, consumed directly. The integer `n_sub ≥ 2`
aggregate-variance bridge (grids coarser than ~1.5 bd/step) is unchanged. The `f = 1` step
is thus exact; the `n_sub ≥ 2` bridge is the only remaining approximation, so the coarse-grid
diagnostic fires only there (INFO).

**Log-price floor.** Same as HMM: `log_path.clamp_min(-10.0).exp()` before returning the
spot path (`stochasticprocess.py:2365-2369`).

## 2. JSON schema and reference calibration

```json
"GARCHSpotModel.PLATINUM_CME": {
  "Omega":  8.028e-07,
  "Alpha":  0.0328,
  "Beta":   0.9639,
  "Nu":     7.50,
  "Mu":     0.0,
  "H0":     7.671e-04,
  "Log_Price": true,
  "Calibration_DT_Years": 0.003968253968253968
}
```

Units: `Omega`, `H0` are per-calibration-step variance of **fraction** log returns
(beware the ×10⁴ trap if the fit ran on percent returns); `Mu` is annualized; `Nu` > 2.

Validation on load: `ω > 0`, `α ≥ 0`, `β ≥ 0`, `α + β ≤ 0.999`, `ν > 2.05`, `H0 > 0`.

Reference fit (16.1y of `CommodityPrice.PLATINUM` from `pl_exp.csv`, zero mean,
`|r| < 0.25` outlier guard, GARCH(1,1)-t) — the calibrator (§5) must reproduce these to
tolerance:

| quantity | value |
|---|---|
| ω (fraction units) | 8.03e-07 |
| α | 0.0328 (t = 2.8) |
| β | 0.9639 (t = 72) |
| ν | 7.50 (t = 9.2) |
| persistence α+β | 0.9967 (half-life ≈ 210 bd) |
| long-run ann vol √(ω/(1−α−β)·252) | 24.75% |
| H0 at 2025-12-30 | 7.67e-04 (≈ 44% annualized) |

Note H0 ≈ 1.8× long-run vol: the filter is carrying the 2025 squeeze — the solver's
day-1 state will correctly say "high-vol regime, hedge tight."

**Model configuration wiring** (`ModelParams` in `MarketDataRF_platinum_calibrated_cme.json`):
change `modeldefaults.CommodityPrice` from `MarkovHMMSpotModel` to `GARCHSpotModel`; keep
the existing `modelfilters` entry routing `PLATINUM_LME → BasisComposedSpotModel`. Remove
(or leave orphaned) the `MarkovHMMSpotModel.PLATINUM_CME` Price Models block and add the
`GARCHSpotModel.PLATINUM_CME` block above. `construct_process` resolves model classes by
name string (`calculation.py:815`), so adding the class to `stochasticprocess.py` is the
only registration step — confirm resolution in the smoke test.

## 3. Class contract (`stochasticprocess.py`)

`class GARCHSpotModel(StochasticProcess)` — mirror `MarkovHMMSpotModel` member for
member. Deltas only, below.

### 3.1 Statics

- `num_factors() -> 1`.
- `correlation_name -> ('GARCHSpotProcess', [()])`.
- `privileged_layout(cls, param) -> {'log_h': 1}`.

### 3.2 `precalculate(ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None)`

- Store `z_offset = process_ofs`, `scenario_horizon = time_grid.scen_time_grid.size`.
- dt array anchored at `time_grid_years[0]` exactly as the HMM does
  (`stochasticprocess.py:2259-2262`) — **required** for inner-MC kept-base mode where
  `scen_time_grid[0] > 0`.
- Per-step `n_sub` (int array) and the μ·dt drift array, via `shared.one.new_tensor`
  (dtype/device discipline: no casts in `generate`).
- `self.spot0 = tensor` kept on the autograd graph, `(1,)` outer / `(B,)` inner —
  identical convention to the HMM (`stochasticprocess.py:2300-2305`). Note: in log mode
  `h` depends only on generated innovations, never on `spot0`, so price-AAD w.r.t.
  `spot0` is unaffected by the vol recursion.
- `self.h0_default = param['H0']` scalar tensor.

### 3.3 `generate(shared_mem)`

`Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]`.

**Outer mode** `Z.ndim == 2`, shape `(T, B)`:

- `h0`: if `shared_mem.t_Scenario_Buffer` carries `(self.factor_key, 'h0_outer')`
  (shape `(B,)`, variance units), use it (this is the diff-ML t=0 randomization hook,
  mirroring `regime0_outer`, `stochasticprocess.py:2329-2337`); else `H0` expanded.
- Sequential loop over `T` (h feeds back through realized r; T ≈ 85–120, the HMM loops
  too): draw `ε_t` from `Z[t]` via the standardized-t transform; `r_t = sqrt(h_t)·ε_t`;
  accumulate `Δlog S`; update `h_{t+1}`. Record `log h_t` **before** the step's shock is
  applied (the no-lookahead convention).
- Path: `spot_path = (log s0 + cumsum).clamp_min(-10.0).exp()`, `(T, B)`.
- Publish the revealed state, detached (consumed as a state coordinate, not
  differentiated through — same rationale as the belief detach,
  `stochasticprocess.py:2372-2383`):
  `shared_mem.t_Scenario_Buffer[(self.factor_key, 'garch_log_h')] = log_h.detach().unsqueeze(1)`
  with B-LAST shape `(T, 1, B)` so the buffer's `dim=-1` concat works.
- Stash `self.last_log_h = log_h.detach()` `(T, B)` for `privileged_factors`.

**Inner mode** `Z.ndim == 3`, shape `(T, B, B2)`:

- `h0`: buffer key `(self.factor_key, 'h0_inner')`, shape `(B,)`, expanded
  `.view(B, 1).expand(B, B2)` — mirrors `regime0_inner`
  (`stochasticprocess.py:2394-2401`). Fallback `H0` if absent.
- `spot0` is `(B,)`, broadcast `s0.view(B, 1)` as the HMM does.
- Same loop; publish `(T, 1, B, B2)` log-h to the buffer and stash.

Return `spot_path`.

### 3.4 Reveal plumbing

- `privileged_factors(self, simulated)` →
  `{'log_h': self.last_log_h.to(torch.float32).unsqueeze(-1)}` (shape `(T, B, 1)`,
  matching the HMM's `(T, B, n)` accumulator convention).
- `reveal_state_at(self, t, buffer)`: **log_h-first / price-last**, mirroring the HMM's
  belief-first/price-last ordering that `hedge_solver.py:676` documents. Rank-check the
  buffered tensor against the current mode exactly as the HMM does
  (outer `(T,1,B)` vs inner `(T,1,B,B2)`); defensive fallback: broadcast
  `log(long-run variance)` if the buffer key is absent.

### 3.5 Fork-site changes (outside `stochasticprocess.py`)

The inner-MC forker and the t=0 randomizer currently seed the HMM via
`regime0_inner` / `regime0_outer`. Locate the producers
(`grep -rn "regime0_inner\|regime0_outer" --include='*.py' .` — they live in the
bundle/inner-MC setup in `hedge_bundle.py` / `calculation.py`) and mirror:

- **Required (correctness):** at an inner fork at outer time `t`, write
  `(factor_key, 'h0_inner') = exp(outer garch_log_h[t])` per outer path `(B,)`. Without
  this the inner fan-out reprices from `H0` instead of the forked path's vol state and
  the one-step bootstrap labels are wrong.
- **Optional (off by default):** t=0 jitter of `log h0` for
  `Randomize_Initial_State` — do not build unless the bank-support diagnostics ask
  for it.

The producer code should key on a small capability probe
(`hasattr(proc, 'privileged_layout')` + which keys the process consumes) rather than
`isinstance` checks, so HMM and GARCH worlds both run.

Also update `hedge_bundle.py:388-389`: the "martingale primary sorts first" candidate
rule keys on `hasattr(proc, '_forward_belief')`; generalize the predicate to "process
exposes a revealed sufficient statistic" (e.g. `hasattr(proc, 'privileged_layout')`) so
the GARCH primary sorts first in a cross-market strip exactly as the HMM did.

## 4. Solver interaction (should be automatic — verify, don't code)

The V̂ deep-state market block sizes itself off `privileged_layout`, so the market input
width changes from `n_states(=3 belief) + price` to `1(log h) + price`. No solver code
changes; the smoke test (§6.7) asserts the width and that checkpoints stamp the new
recipe hash (`_config_hash` covers it via the solver cfg — world changes flow through
the corridor/provenance checks already in place).

## 5. `calibrate` classmethod

Follow the GBM pattern (`stochasticprocess.py:297-305`): input a business-daily
2pm-London-synced close series; steps:

1. `r = diff(log px)`, drop NaN, outlier guard `|r| < 0.25`.
2. Zero-mean GARCH(1,1)-t MLE. Use `arch` (`arch_model(100·r, mean='Zero',
   vol='GARCH', p=1, q=1, dist='t')`) if importable, else scipy L-BFGS on the standard
   log-likelihood with the same standardized-t density. Convert percent-fit units back
   to fraction (`ω ×1e-4`, `h ×1e-4`).
3. Enforce `α + β ≤ 0.999` (scale β down, log a warning) and `ν ≥ 2.05`.
4. `H0` = filtered conditional variance at the final observation.
5. Return the §2 param dict; log parameters with standard errors, persistence,
   half-life, and long-run annualized vol. Must reproduce the §2 reference table on
   `pl_exp.csv` within 2% relative on (α, β, LR vol) and 10% on (ω, ν).

Recalibration cadence: quarterly, with the rest of the world.

## 6. Acceptance tests

1. **No-lookahead:** flip the sign of `Z[t]` on one path; revealed `log h_s` for
   `s ≤ t` is bitwise unchanged, `log h_{t+1}` changes.
2. **Martingale by state:** simulate an outer bundle; at several `t`, bucket paths by
   `log h_t` decile; assert `|E[Δlog S_t | bucket]| < 3·MC-s.e.` per bucket. (This is
   the `E[dF|state] = 0` invariant, now directly assertable — the test the HMM world
   could never pass.)
3. **Moments (§1a clock):** simulated unconditional annualized vol = 24.75% ± 0.5% on the
   CALENDAR-daily production clock (annualized over calendar years, H0 = LR variance),
   AND grid-invariant vs a business-daily grid; excess kurtosis > 1 (data reference:
   sample kurtosis 4.3); ACF₁(r²) positive, in [0.1, 0.4] (data reference: 0.238), gated
   on the business clock. Half-life in real time ≈ 210·(365.25/252) ≈ 304 calendar days on
   both clocks. (The `ACF₁(log h) ≈ α+β` shorthand is dropped — it conflates the recursion
   AR coefficient with the realized log-h ACF1.)
4. **Inner/outer consistency:** fork inner at `t` with `h0_inner = exp(outer log h_t)`;
   inner one-step mean/variance/kurtosis match the outer conditional within MC error
   (same style as the existing FD/consistency tests, e.g. `test_diffml_spot_grad_fd`).
5. **Fork fidelity:** inner `h` at the fork start equals the passed `h0_inner` exactly.
6. **Grid:** the production `platinum_hedge_shipping.json` grid is CALENDAR-daily
   (`dt = 1/365.25`), so `f ≈ 0.6897` and `n_sub == round(f) == 1` every step (assert) —
   handled EXACTLY by the §1a fractional clock, not the (wrong) original "business-daily,
   exact" assumption; one synthetic 2-business-day step (`f = 2 ⇒ n_sub = 2`) matches the
   aggregate-variance bridge analytically.
7. **End-to-end smoke:** swap the model in the market-data JSON, run
   `production_solver.py` (1 seed, `DiffV2_Fit_Iters` ≈ 5): verdict prints, `bounded`
   is True, V̂ market width = price + 1, checkpoint saves and reloads through the
   provenance check, `Random_Seed` determinism holds across two runs.

## 7. Hooks and follow-ons (explicitly out of this build)

- **Expert-view drift overlay.** `Mu` stays a scalar field, default 0, with a docstring
  warning that drift is the one parameter 16y of daily data cannot identify
  (s.e. ≈ σ/√years ≈ 5.6% unconditional). The intended design when the expert signal
  exists: an **exogenous** expert state (not price-filtered), a per-state drift sized by
  the experts' historical hit rate or a stated confidence, and posterior mixing over the
  view strength so the policy's tilt shrinks with view uncertainty. Separate spec;
  humans own the mean, GARCH owns the variance.
- **Basis recalibration (config-only, once synced data lands).** The current
  `BasisLinkedSpotModel.LME_CME` numbers are calibrated on asynchronous closes
  (LBMA 2pm London vs later CME settle) and therefore mostly measure the futures move
  in the 2pm→settle window, not executable basis. Re-estimate on the 6-month
  2pm-synced Bloomberg sample: a spread *vol* identifies fine from n ≈ 126
  (relative s.e. ≈ 1/√(2n) ≈ 6%); keep the OU form; expect σ to collapse and reversion
  to weaken. Cross-check with the variance decomposition
  `Var(async basis) ≈ Var(true basis) + Var(2pm→settle futures move)` computed from the
  same intraday sample. Refresh quarterly like everything else.
- **Carry–spot correlation** (2025 squeeze realism: price up + backwardation up
  together) — next modeling ticket, touches `VARMixedFactorInterestRateModel`
  innovation wiring, not this class.
