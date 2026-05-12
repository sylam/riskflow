# Status note — HP search for "beat textbook", 2026-05-11

## TL;DR

**Pure HP/coefficient tuning cannot beat the textbook hedge on this single-leg deal.**

After 8 rounds and ~55 training runs covering every coefficient axis in the framework,
the best HP combination (`dense_5_ent_3e-3_power_2` on seed=42) achieves
Sortino = −$515k vs textbook's −$93k — a gap of $422k that does not close further with
HP tweaking. Multi-seed validation confirms this is a local optimum (4-seed avg
Sortino = −$624k — seed=42 was a lucky outlier).

The architectural changes needed to genuinely beat textbook (warm-start from textbook
policy, bad-path-anticipation features, finer action-space granularity, possibly
behavior-cloning + PPO fine-tuning) are out of scope for pure HP search.

For production on this single-leg deal: **use `run_textbook_hedge.py`**. The ML stack
remains the right target for multi-leg / basis-risk cases where no closed-form
benchmark exists.

---

## Objective

Maximize **Sortino = mean_pnl − max(0, −p5_pnl)** on the standard fixture's val_bundle.

| benchmark | mean | p5 | **Sortino** |
|-----------|------|-----|-------------|
| no_trade | +$93k | ≈−$1M | ≈−$900k |
| **textbook ramp** | **−$31k** | **−$63k** | **−$93k** |
| target | … | … | **beat −$93k** |

Textbook captures the deal's economics so well that it's both *good Sortino* and
*almost optimal trade-off* — its mean cost ($124k below no_trade) buys $937k of
p5 protection.

---

## 8-round HP search summary

All rounds were single-seed (42) at 100 epochs unless noted. **No limit changes** —
only coefficients and architectural parameters that aren't position bounds.

### Round 1 — first-pass axis perturbations (6 rows)
| row | Sortino |
|-----|---------|
| **baseline** (`dense_3`) | **−$594k** |
| pbrs (Dense_Reward_Mode=potential_based) | −$647k |
| bounds_low | −$710k |
| fp_50 | −$665k (no_trade collapse) |
| ~~totlim_100~~ (disqualified, limit change) | — |
| ~~fp_50_totlim_100~~ (disqualified, limit change) | — |

### Round 2 — utility-shape axis (8 rows)
| row | Sortino |
|-----|---------|
| **dense_5** | **−$579k** ✓ |
| baseline (dense_3) | −$594k |
| fp_2 | −$677k |
| power_2 | −$684k |
| fp_5_dense_1 | −$692k |
| dense_0 | −$725k |
| surplus_5 | −$727k |
| fp_5, fp_2_dense_0 | no_trade collapses |

**Insight**: higher `dense_scale` pushes the agent toward more hedging.

### Round 3 — push dense axis + dense_5 combos (8 rows)
| row | Sortino | max_total |
|-----|---------|-----------|
| **d5_ent_3e-3** | **−$555k** ✓ | 34 |
| d5_fp_30 | −$645k | 17 |
| d5_power_2 | −$660k | 18 |
| d5_e3_p3 | −$675k | 15 |
| d5_e3_p5 (Power 5) | −$669k | 15 |
| dense_8/10/15/20 | −$688k to −$725k | drops to 7 |
| d5_lr_1e-3 | **−$1,814k** | 130 (blow-up) |

**Insight**: dense axis has an optimum near 5; higher entropy unlocks bigger hedge.

### Round 4 — dense_5 × entropy/utility combinations (8 rows)
| row | Sortino | max_total |
|-----|---------|-----------|
| **d5_e3_p2** | **−$515k** ✓ | **42** |
| d5_e3_p2_postdeal_3 | −$720k | 21 |
| d5_e3_p2_fp_5 | −$717k | 4 |
| dense_5_ent_3e-2 | −$683k | 30 |
| dense_5_ent_3e-3_fp_20 | −$638k | 24 |
| dense_3_ent_3e-3 | −$624k | 19 |
| dense_5_ent_1e-2 | −$605k | 33 |
| dense_5_ent_5e-3 | −$566k | 20 |

**Insight**: adding `Power=2` to dense_5_ent_3e-3 unlocks max_total=42 → −$40k Sortino improvement.

### Round 5 — push Power axis + sr/fp combos around d5_e3_p2 (8 rows)
| row | Sortino |
|-----|---------|
| **d5_e3_p2 (R4)** | **−$515k** |
| d5_e3_p3 (Power=3) | −$648k |
| d5_e3_p2_fp15 | −$661k |
| d5_e3_p2_sr0.5 | −$659k |
| d5_e3_p5 (Power=5) | −$669k |
| d5_e5_p2 | −$669k |
| d5_e3_p1.5 | −$673k |
| d5_e3_p4 (Power=4) | −$700k |
| d5_e3_p2_sr2 | −$706k |

**Insight**: Power=2 is the unique sweet spot; all variations worse.

### Round 6 — optimizer axes (8 rows)
| row | Sortino |
|-----|---------|
| **d5_e3_p2 (R4)** | **−$515k** |
| d5_e3_p2_cvar20 | −$579k |
| d5_e3_p2_rs2 | −$687k |
| d5_e3_p2_vc0.5 | −$693k |
| d5_e3_p2_clip0.1 | −$705k |
| d5_e3_p2_vasym2 | −$702k |
| d5_e3_p2_rs0.5 | −$715k |
| d5_e3_p2_vc0.05 | −$724k |
| d5_e3_p2_cvar10 | −$740k |

**Insight**: d5_e3_p2 is robust at default optimizer settings. None of CVaR, Value_Coef,
Reward_Scale, Clip_Eps, Value_Loss_Asym_Weight perturbations help.

### Round 7 — non-optimizer axes (8 rows)
| row | Sortino | max_total |
|-----|---------|-----------|
| **d5_e3_p2 (R4)** | **−$515k** | 42 |
| d5_e3_p2_lam0.95 | −$532k | 36 |
| d5_e3_p2_ep120 | −$592k | 24 |
| d5_e3_p2_ep80 | −$662k | 51 (!) |
| d5_e3_p2_g0.99 | −$672k | 13 |
| d5_e3_p2_ppo8 | −$675k | 83 (!) |
| d5_e3_p2_ppo2 | −$825k | 33 |
| d5_e3_p2_ep150 | −$828k | 65 |
| d5_e3_p2_g0.95 | −$1,011k | 74 (catastrophe) |

**Insight**: Training-duration trajectory is unimodal — agent discovers hedging by 80ep
(max_total=51), refines to 42 by 100ep (best Sortino), then either keeps refining
smaller (24 at 120ep) or diverges (150ep+, 200ep+). Bigger hedges at 80/150ep but
worse tail variance.

### Round 9 — entropy schedule + entropy floor on seeds 7 & 42 (6 runs)

Hypothesis: maybe the seed-variance pattern (seed=42 max_total=42, others 3-18) is
because the policy gets stuck in a sub-optimal basin during early training. Higher
initial exploration (entropy schedule annealing high → low, or entropy floor) might
break this.

| config | seed=7 Sortino | seed=42 Sortino |
|---|---|---|
| **d5_e3_p2 baseline** | **−$679k** | **−$515k** (R4) |
| d5_ent_sched (anneal 0.01→0.001) | −$744k | −$597k |
| d5_ent_sched_aggressive (0.03→0.001) | −$699k | −$707k |
| d5_e3_p2_entfloor (H_min=8, coef=0.01) | **−$630k** ✓ | −$660k |

**Insight**: entropy floor improves seed=7 (+$49k) but degrades seed=42 (−$145k).
Net effect is roughly neutral. No config dominates the baseline across seeds.

### Round 8 — multi-seed validation + fine-grain (5 rows + 3 multi-seed)

**Multi-seed validation of d5_e3_p2**:
| seed | Sortino | max_total |
|-----|---------|-----------|
| 42 (R4) | −$515k | 42 |
| 7 | −$679k | 18 |
| 123 | −$624k | 9 |
| 456 | −$679k | 3 |
| **4-seed avg** | **−$624k** | 18 |

**Conclusion**: seed=42 was a lucky outlier. The true performance of d5_e3_p2 is
Sortino ≈ −$624k, not −$515k.

**Fine-grain rows on seed=42** (all worse than R4):
- d5_e3_p2_ep90: −$652k
- d4_e3_p2: −$720k
- d6_e3_p2: −$654k
- d5_e2_p2 (ent_2e-3): −$718k
- d5_e3_p2.5: −$647k

---

## Why textbook is unbeatable here (post-mortem)

1. **Perfect instrument match**: 50 JUL contracts × $102.5k = $5.125M, exactly matches
   the deal's $5.125M notional (2500 oz × $2050). No basis risk.
2. **Zero policy noise**: textbook is open-loop deterministic. ML's stochastic
   actions add variance even when the mean strategy is right.
3. **Symmetric in spot direction**: the linear ramp captures the deal's
   averaging-window structure exactly.

To beat it, ML must be *strictly better than open-loop*: e.g. over-hedge on bad-path
indicators and under-hedge when spot trends favorably. This requires:

- **State features** that anticipate bad paths (realized vol, drawdown-since-inception,
  spot-minimum-so-far). Current state has mechanics but no bad-path indicators.
- **Behavior cloning + PPO fine-tuning**: initialize policy from textbook actions,
  then let PPO add the closed-loop refinement. Standard imitation-learning recipe.
- **Finer action space at large positions**: current Trade_Deltas jumps by 5 past
  ±10. Agent can't fine-tune at hedge size 40-50.
- **Privileged training** (V sees full path, policy doesn't): teach V what
  bad paths look like, let policy learn confidence from V's value targets.

These are **architectural** changes, not HP changes.

---

## What was tested (full coefficient list)

Coefficients and parameters tested:

- `Dense_Tracking_Reward_Scale`: {0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20}
- `Entropy_Coef`: {3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2}
- `Floor_Penalty`: {2, 5, 10, 15, 20, 30, 50}
- `Power`: {1, 1.5, 2, 2.5, 3, 4, 5}
- `Surplus_Reward`: {0.5, 1, 2, 5}
- `Per_Instrument_Bounds_Penalty`: {0.05, 0.5}
- `Position_Bounds_Penalty`: {0.05, 0.25}
- `Post_Deal_Trade_Penalty`: {1, 2, 3, 5}
- `Gamma`: {0.95, 0.99, 1.0}
- `GAE_Lambda`: {0.95, 0.995}
- `PPO_Epochs`: {2, 4, 8}
- `Value_Coef`: {0.05, 0.1, 0.2, 0.5}
- `Reward_Scale`: {0.5, 1, 2}
- `Clip_Eps`: {0.1, 0.2}
- `Value_Loss_Asym_Weight`: {1, 2}
- `CVaR_Alpha` × `CVaR_Lambda`: {(0.1, 1.0), (0.2, 2.0)}
- `Dense_Reward_Mode`: {asymmetric, potential_based}
- `Learning_Rate`: {3e-4, 1e-3}
- `Epochs`: {80, 90, 100, 120, 150, 200}
- ~20 named 2- and 3-coefficient combinations

**Not tested** (per user constraint or out of scope):
- `Total_Position_Abs_Limit` (limit — user said don't touch)
- `Position_Limits` per instrument (limit — same)
- `Trade_Deltas` action-space granularity (changes action space, arguably a "limit"
  on tradable deltas)
- Network architecture (`Token_Dim`, `Emb_Dim`, `N_Heads`, `N_Layers`)
- Decision interval curriculum
- KL penalty (not in framework)
- Warm-start / behavior cloning (architectural)
- State features (architectural)

---

## Production recommendation

**Ship `run_textbook_hedge.py` for this single-leg deal**. It's a one-line python
script call away. The ML policy improves Δworst vs no_trade by +$579k on a 16k-path
held-out eval, but textbook achieves +$2.09M on the same eval — 3.6× more tail
protection at $30k more mean cost.

**Keep the ML stack** for:
- Multi-leg books (the actual production target per `project_objective` memory)
- Basis-risk hedges (hedge instrument ≠ underlying)
- v2-simulator OOD experiments

**HP recipe for ML training** (if used for one of the above):
- `Dense_Tracking_Reward_Scale=5`
- `Entropy_Coef=0.003`
- `Power=2.0`
- All other fixture defaults
- `Epochs=100` (DO NOT increase — 200ep is unstable)

This config achieves Sortino ≈ −$624k 4-seed avg on the test fixture. For other
deals, re-run the HP sweep — the optimum depends on the specific deal economics.

---

## Files / git status (unchanged from earlier note)

See `STATUS_NOTE_pre_symlog.md` for the previous architecture audit, and the prior
edit of this file for the earlier overnight findings (now superseded).

---

## Subsequent framework / cleanup fixes (post-HP-search)

After the 9-round HP search concluded with the textbook-gap conclusion, a deeper
audit prompted by the question "why can't ML even find textbook?" surfaced
**4 framework bugs** that had been silently degrading the agent. All fixed.

### 1. Expiry penalty fires during deal-active window (root cause)
`_expiry_holding_penalty` fired starting 4 BUSINESS DAYS before the contract's
`last_trade_date`. For an averaging deal whose averaging window ends ~contract
expiry, that punishes textbook's natural ramp-down by design. Measured: framework
reward for textbook policy was −14.6 vs no_trade −4.15 (textbook *lost* to no_trade
under the framework's own reward).

**Fix**: gate the expiry penalty on `current_idx > last_settlement_index`. Now fires
only after the deal is settled; during deal life, holding contracts near their
expiry is treated as legitimate hedging.

Post-fix framework reward: textbook −0.79 vs no_trade −4.15. **+3.36 utility edge
for textbook** — training now correctly points at it.

Multi-seed verification of d5_e3_p2 under the fix:

| seed | before fix | after fix | delta |
|---|---|---|---|
| 42 | −$515k | −$521k | −$6k |
| 7 | −$679k | −$532k | **+$147k** |
| 123 | −$624k | −$677k | −$53k |
| 456 | −$679k | −$646k | +$33k |
| **avg** | **−$624k** | **−$594k** | **+$30k** |

Helped seed 7 a lot, hurt seed 123 slightly. Net +$30k on the 4-seed avg. The
textbook-target gap closes from $531k to ~$500k.

### 2. PBRS Dense_Tracking_Reward_Clip silently differs from asymmetric mode (20×)
In asymmetric mode the clip applies before the `rs · fp` scaling — effective
post-scale clip = `rs · fp · rc`. In PBRS mode the `fp` is baked into the shaping
*before* the clip — effective post-scale clip = `rc`. A user switching modes with
the same `rc=5, fp=10, rs=2` would silently see effective clips of 100 vs 5.

**Fix**: in PBRS mode, multiply the clip bound by `fp` so the SAME `rc` value means
the SAME pre-scale-clip-on-down-change in both modes (matches the documented
"asymmetric semantics" for `rc`).

### 3. `positions=None` dead-code threaded through 4 method signatures
After per-instrument bounds went reward-side, the `_feasible_mask` was removed but
its `positions` parameter remained in 4 `StructuredRebalancePolicy` methods. Every
decision step still built a `(B, I)` int64 tensor and passed it through the policy
sample, the rollout return dict, the PPO update's flatten, and the minibatch
slicing — for nothing.

**Fix**: removed `positions=` from `_policy_outputs`, `forward`, `sample`,
`evaluate_action`; dropped the `flat_positions` reshape and `mb_positions` slicing
in `_ppo_update`; dropped the spurious `positions_t` build in `_diag_rollout_policy`.
Diagnostic position recording for `_post_settle_holding_metrics` kept intact.

### 4. Per-step `.any()` GPU-CPU sync in `_per_instrument_bounds_penalty`
`if not bool((violation > 0).any()): continue` guarded a tensor `exp_linear_ramp`
that correctly returns 0 when violation is 0. The guard added a per-instrument-per-step
CUDA-CPU sync — **3 instruments × ~250 decision steps = 750 syncs per epoch**.

**Fix**: remove the guard. Always run the ramp. The wasted compute is negligible vs
the sync overhead.

### 5. `_realized_structured_action` rebuilds the deltas tensor every decision
Per-decision `torch.tensor(padded_deltas, dtype=int64, device=...)` was allocating
a `(I, max_bins)` tensor from a Python list every decision step. ~252 allocations
+ host→device uploads per epoch.

**Fix**: cache the tensor on `runtime["policy"]["action_space"]["_padded_deltas_dev_<device>"]`
on first call. Same for the `min_trade_delta` fallback path. Built once, reused.

## Verdict: textbook STILL dominates ML on this single-leg fixture

Even with all 5 framework fixes, the multi-seed average of d5_e3_p2 is Sortino
**−$594k** — still 6× textbook's **−$93k**. Rounds 10-12 explored alternative
reward axes (utility_scale c, small-c × low-dense combinations) and none beat the
post-fix baseline.

The remaining gap is *architectural*, not coefficient-tunable:
- Warm-start the policy from textbook actions (behavior cloning + PPO refinement)
- Add bad-path-anticipation features to the state (realized vol, drawdown,
  running spot minimum)
- Finer action-space granularity at hedge sizes 40-50
- Privileged value-head that sees full trajectories during training

These are the next 2-3 weeks of effort if revisiting ML on single-leg. For the
production multi-leg deal (per `project_objective` memory), re-run HP search on
the new fixture — textbook may not dominate there.

## Compute used today

| round | runs | wallclock |
|-------|------|-----------|
| Phase A/B/C clean (yesterday) | 32 | ~140 min |
| Round 1 | 6 | ~21 min |
| Round 2 | 8 | ~28 min |
| Round 3 | 8 | ~28 min |
| Round 4 | 8 | ~28 min |
| Round 5 | 8 | ~28 min |
| Round 6 | 8 | ~28 min |
| Round 7 | 8 | ~30 min |
| Round 8 | 8 + 3 multi-seed | ~50 min |
| **Total today** | **~55 (today) + 32 (yesterday)** | **~240 min today / ~380 min total** |

All artifacts under `artifacts/sweeps/2026051*_*beat_textbook_r*/` and
`artifacts/sweeps/2026051*_*phase{A,B,C}_clean*/`.
