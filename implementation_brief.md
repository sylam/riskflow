# Implementation brief — differential-ML dynamic hedge (Phase 1: single-underlying validation harness)

**Read this first; `differential_ml_redesign_v14.md` is the reference spec behind it.** The spec has a 14-deep revision-note stack and mixes resolved / live / deferred items — do **not** infer scope from skimming it. Scope, order, and exclusions are below.

## Scope of Phase 1
Build a **falsification harness for ONE underlying** (the platinum deal), against the **real riskflow interfaces** (not mocks, not the production loop, not the book). Goal: decide whether the approach holds *before* a full build. The verdict metric is the **policy rolled forward under fresh MC** (action ranking), not the fitted value `D_0`.

## Build order — each gate must pass before the next
1. **Gate 0 — arbitrary-state differentiable fork smoke.** Sample a batch of designer states; fork **one step with autograd ON**. PASS = finite pathwise differentials of the one-step output w.r.t. the state; VAR `X_0` round-trip guard holds (`stochasticprocess.py:2552`); spot/regime/basis fork jointly. (Inspection says all three processes support per-path batched init — `964–969/1015–1020`, `2537/2549`, `2771–2772` — but run it.)
2. **Belief filter + calibration.** Forward HMM filter → `P(regime_t | prices_{0..t})`; use it as `market_t`'s regime coordinate **in place of the privileged true regime** (`calculation.py:2110–2113`). PASS = (a) calibrated (bucket by predicted belief, empirical regime frequency matches) AND (b) `C_T`, `C_{T-1}` have non-trivial gradient in the belief coordinate on held-out data. This is the gating empirical unknown (R1b).
3. **Staggered-expiry exact-DP toy.** Tiny exact grid DP, **≥2 hedge contracts with staggered expiry** (grid steps e.g. 121 → 11 → unwind), terminal at `T_dec` (forced flat, known liability). This is ground truth.
4. **Post-decision fit with λ-return.** `C_t = B_t (exact analytic baseline) + A_t (MLP)`; value label = λ-return `y_boot + λ(y_roll − y_boot)`, differential label = **one-step bootstrap only**; on-policy rollout rows are **value-loss only**. Compare fitted policy to exact DP.
5. **Sweep + diagnostics across full depth.** Plot, vs depth and especially in the early all-contracts-live window: `δ_norm = (y_roll − y_boot)/max(std(y_roll), floor)`, action-error rate (vs exact DP), policy-evaluated value (vs exact DP and static-hold). Sweep M, operator/β (hard / LCB / soft), and λ; pick the smallest/simplest that keeps action-error flat.
6. **Real T/T-1/T-2 smoke** on the real problem (real interfaces, last few steps), before any full-horizon run.

## Build against these real interfaces (no mocks)
`build_action_grid` / `search_action_grid` (`hedge_solver.py:41/59`) — already the `D_t = max_q C_t` operator; `ValueFunctionApproximator` `mlp_differential` dispatch + `_compute_target_gradients` + hull clamp + C¹ (`263+`); process `generate()` per-path init; the existing `no_grad` inner-MC pattern (`calculation.py:2281`) for **value-only** rollouts.

## DO NOT build in Phase 1 (explicitly deferred)
- **Book / multi-underlying / global-budget coordination** (§9 "Book extension") — Phase 2, only after the single-underlying toy holds; validate it with a *two-underlying* exact-DP toy then.
- **Rollout gradients** (`∂y_roll/∂post`) — never; one-step differential only, rollout is value + diagnostic.
- **Differential PCA, shared trunk / temporal regularization, active resampling** — Tier 3; only if the toy curves demand them.
- **Production safety layer** — deployment hardening, not validation.
- **Hedge roll machinery beyond cash-settle-at-expiry** — R9 is fixed-expiry, three contracts, no roll into successors.

## Correctness traps (where an agent will go wrong — each cost a review round to find)
- `market_t` carries the **filtered belief, not the true regime** (R1b).
- The argmax maximizes **`B_t + A_t`**, not `A_t` alone — include the (analytic, action-dependent) baseline difference.
- Differential label is the **one-step bootstrap only**; do not differentiate the rollout.
- Terminal is **`T_dec` = last fixing** (force flat after, `q=0` for `t ≥ T_dec`), **not** the October expiry.
- Action grid is **time-varying `11^(live contracts)`** (1331 → 121 → 11 as contracts expire + cash-settle), not a flat 1331.
- Cash coordinate is **`loss_budget_remaining`** with a **positive `loss_scale = max(|unhedged_liability_PV|, floor)`**; enforce the monotonicity guard. The objective is `AsymmetricUtility_Symlog` (not CARA) — cash cannot be collapsed by translation.
- `δ` is read **scale-normalized**.
- MC structure is **fat outer (≥16k) × tiny inner (M=1, or 2 antithetic, ≤4)** — never a fan; `B_inner` is a free parameter capped only by a 32k flat-memory limit (`calculation.py:2229/2250`).

## One open input still needed from the desk
Confirmed already: hedges are 3 fixed-expiry CME futures (last = October = horizon); last fixing before expiry; loss budget is **global** (book-time only). Nothing else blocks Phase 1.
