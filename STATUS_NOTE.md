# Status note — overnight run, 2026-05-11

## TL;DR

The fully-soft, dollar-wrap-free bounds-penalty architecture lands. Multi-seed HP sweep
identifies `Dense_Tracking_Reward_Scale=3` as the robust 100-epoch winner across 4 seeds,
but **the textbook averaging hedge dominates the ML policy on this single-leg deal**.
Recommendation: ship the textbook hedge for single-leg cases; keep the ML policy in
reserve for multi-leg / basis-risk scenarios where there's no closed-form benchmark.

**Critical caveat: training past 100 epochs is unstable** — Phase C (200ep × 4 seeds)
showed both top configs blowing up on 3 of 4 seeds (one path reached −$74M). Don't
train longer than 100 epochs with the current reward stack.

(Previous status notes preserved: `STATUS_NOTE_pre_symlog.md`,
`STATUS_NOTE_calibration_audit.md`.)

---

## What landed this session (architecture changes)

1. **Per-instrument bounds: hard mask → soft penalty.**
   `StructuredRebalancePolicy._feasible_mask` removed. Per-instrument enforcement is now
   reward-side via `_per_instrument_bounds_penalty`. Symmetric in long/short: equal
   violation counts produce equal penalties (was asymmetric before because Min=−50,
   Max=0 are asymmetric configured limits).

2. **Both bounds penalties are pure-count, no dollar wrap.**
   penalty = `coef × ramp(violation, threshold)`. No `log1p(notional/c)` factor.
   Violation is dimensionless (integer contracts), `coef` carries the utility-unit
   conversion. Both `_per_instrument_bounds_penalty` and `_position_bounds_penalty`
   use the same formula, just fed different violation counts.

3. **Fixture coefficients tuned.** `Per_Instrument_Bounds_Penalty=0.5`,
   `Position_Bounds_Penalty=0.25`. Per-instrument penalty is 2× the strength of
   portfolio-total at every violation level (a single-contract bound is the harder rule
   to break than a soft portfolio aggregate).

4. **Bug found and fixed earlier in session: portfolio bounds penalty was scaling with
   `avg_notional` (= constant per-contract notional) instead of `abs_notional`**.
   Crushed the penalty by ~200× — that's why earlier runs were allowing
   `max_total_abs=150` (3× the limit). Fixed by switching to abs_notional, then
   simplified to pure-count form as above.

5. **Stripped redundant `.to(device=device, dtype=torch.float32)`** from all four
   penalty functions. State tensors are already on bundle's device/float32; the `.to()`
   was a no-op.

6. **New tool**: `run_textbook_hedge.py` — CLI shim that drives the textbook averaging
   ramp via `BundleStepper`. Reuses the exact logic in
   `tests/test_stepper.py::test_textbook_hedge_as_stepper_loop`.

7. **`eval_policy_artifact.py`** now accepts `--run_dir` (in addition to `--tag`) and
   prints the no_trade reference + delta for the eval bundle.

---

## HP search outcome (4 phases, all on the cleaned architecture)

### Phase A clean — 12 configs, seed=42, 100ep (OFAT axis exploration)

Ranked by Δworst vs no_trade:

| rank | config | Δmean | Δworst | max_total |
|------|--------|-------|--------|-----------|
| 1 | `Dense_Tracking_Reward_Scale=3` | −$31k | **+$331k** | 30 |
| 2 | `Floor_Penalty=20` | −$47k | +$228k | 22 |
| 3 | `Entropy_Coef=0.003` | −$13k | +$210k | 25 |
| 4 | `Dense_Tracking_Reward_Scale=1` | −$39k | +$207k | 17 |
| 5 | `Value_Coef=0.05` | −$28k | +$30k | 3 |
| 6 | default | −$30k | +$29k | 6 |
| 7 | `Value_Coef=0.2` | −$28k | +$16k | 12 |
| 8 | `Floor_Penalty=5` | −$20k | +$5k | 16 |
| 9 | `Post_Deal_Trade_Penalty=2` | $0 | $0 | 0 (no_trade collapse) |
| 10 | `Entropy_Coef=0.0003` | −$8k | −$5k | 18 |
| 11 | `Learning_Rate=0.001` | −$73k | −$23k | **210** ⚠ |
| 12 | `Post_Deal_Trade_Penalty=5` | −$44k | −$25k | 15 |

Position-bound audit healthy: max_total_abs in [3, 30] for all reasonable configs
(50 limit respected). lr=1e-3 overshoots dramatically (max_total=210) but Δworst
only −$23k vs −$3.17M under the previous architecture — the cleaned penalty contains
it.

### Phase B clean — top-4 × seeds {7, 123, 456} × 100ep (multi-seed validation)

Combined with Phase A's seed=42 → 4 seeds per config:

| config | Δmean avg | Δworst avg | Δworst min | seeds positive |
|--------|-----------|------------|------------|----------------|
| **dense_3** | −$62k | **+$309k** | +$122k | **4/4** |
| entropy_3e-3 | −$41k | +$183k | +$128k | 4/4 |
| fp_20 | −$72k | +$176k | +$39k | 4/4 |
| dense_1 | −$67k | +$159k | +$28k | 4/4 |

**dense_3 wins multi-seed**: highest avg Δworst, highest minimum across seeds.
All four configs survive multi-seed validation at 100ep.

### Phase C clean — top-2 × all 4 seeds × 200ep (production-grade validation)

| config | seed=7 | seed=42 | seed=123 | seed=456 |
|--------|--------|---------|----------|----------|
| dense_3 200ep | +$268k | **−$74M** ⚠ | −$606k | −$1.58M |
| entropy_3e-3 200ep | +$424k | −$4.4M | −$5.7M | −$828k |

**200ep is unstable**. Both configs blow up on 3 of 4 seeds.
**Don't train longer than 100 epochs.**

### Held-out bigval eval — 16,384 fresh seed=999 paths

| metric | no_trade | textbook | dense_3 100ep seed=7 | entropy_3e-3 100ep seed=7 |
|--------|----------|----------|----------------------|----------------------------|
| mean | +$93k | −$31k | +$0.25k | +$33k |
| worst | −$2.21M | **−$121k** | −$1.63M | −$1.93M |
| std | — | $20k | $434k | $503k |
| Δmean vs nt | — | −$124k | −$93k | −$60k |
| Δworst vs nt | — | **+$2.09M** | +$579k | +$279k |
| Sortino (Δworst/Δmean) | — | **~17:1** | ~6:1 | ~4.7:1 |

Both ML policies generalize well to fresh paths. dense_3 beats entropy_3e-3 on Sortino
in this bigval test (Δworst +$579k vs +$279k). Both are dominated by the textbook
benchmark for this single-leg deal.

### Decomposition diagnostic (dense_3 100ep, pre/post-last-cashflow segments)

Across all 4 seeds:
- Pre-settle Δhedge **positive on worst path** (+$121k to +$484k) — hedge absorbs loss
- Pre-settle Δhedge **negative on best path** (−$262k to −$840k) — hedge pays cost
- Post-settle mean across cases **negative on all 4 seeds** (−$7k to −$27k) — small
  expected cost paid for tail-protection (NOT directional alpha)

3 of 4 seeds clearly LOSS-ABSORBING; 1 of 4 (seed 123) AMBIGUOUS at boundary because
its pre-settle hedge was smaller, not because post-settle was directional.
**Falsification test passes**: dense_3 100ep is doing real hedging, not speculation.

---

## Recommendations

1. **Production hedge for this single-leg deal**: **use the textbook ramp**.
   `run_textbook_hedge.py` ships ready. The ML policy doesn't beat it on Sortino.

2. **Keep the ML stack** for cases where:
   - Multi-leg deals where there's no closed-form benchmark
   - Basis-risk hedges (hedge instrument ≠ underlying)
   - Constrained capital / binding position limits
   - Future v2-simulator OOD experiments

3. **HP recipe for ML training** (in fixture vocabulary):
   - `Dense_Tracking_Reward_Scale=3.0` (was 2.0 in fixture)
   - `Epochs=100` (do NOT increase — 200ep is unstable)
   - All other fixture defaults

4. **Why 200ep is unstable (investigation)** — examined live_diag.json for
   `dense_3 200ep seed=42` (the −$74M catastrophe) vs the same seed's healthy 100ep
   run:

   | epoch | net_pnl | worst | \|trade\| | entropy | val_loss |
   |-------|---------|-------|-----------|---------|----------|
   | 90 | −$115k | −$2.17M | 9.2 | 8.46 | — |
   | 100 | −$147k | −$1.69M | 11.7 | **9.13** | 7.7e9 |
   | 130 | −$267k | −$4.45M | 13.2 | 9.29 | 8.3e10 |
   | 150 | −$341k | −$8.49M | 12.7 | 9.16 | 1.5e11 |
   | 180 | −$745k | −$28.2M | 13.2 | 9.14 | 6.5e12 |
   | 199 | −$1.07M | −$27.2M | 13.6 | 8.93 | **1.66e13** |

   **Pattern**: around epoch ~100 entropy starts climbing (9.13 → 9.45, was decreasing
   8.46 just before), `|trade|` jumps from 9 → 13, value loss explodes 4 orders of
   magnitude (7.7e9 → 1.66e13). Classic PPO **policy divergence past the natural
   convergence point** — without a strong enough KL penalty or early-stopping criterion,
   the ratio drift compounds and the policy "explores" itself off the cliff.

   The fixture has `LR_Schedule: cosine` with `T_max=Epochs`, so when Epochs is bumped
   200, the LR stays high longer (~50% of peak at ep=100). That extends the unstable
   window. **Fixes to try**:

   - Set `T_max=80` (or use a fixed cosine target irrespective of Epochs)
   - Add a KL-divergence penalty to the PPO loss
   - Early-stopping when `val_loss > 10× prior epoch` for 5 consecutive epochs
   - Entropy schedule that *decreases* late in training (currently constant)

---

## Files / git status

### Modified (should commit — framework changes)

- `riskflow/torchrl_hedge.py` — bounds penalties, redundant `.to()` cleanup,
  per-instrument penalty addition, calibration log updates
- `riskflow/hedge_runtime.py` — `Per_Instrument_Bounds_Penalty` config + validation
- `riskflow/structured_policy.py` — removed `_feasible_mask`, kept no-trade-bin invariant
- `riskflow/calculation.py` — consolidation (already from earlier session)
- `tests/fixtures/policy_test_simulate_only.json` — coefficient tuning
- `train_daily.py` — None-sentinel CLI (already from earlier session)
- `tests/test_symlog_smoke.py` — updated assertions to reflect soft enforcement
- `eval_policy_artifact.py` — added `--run_dir` flag, no_trade reference printing

### New (consider committing)

- `run_textbook_hedge.py` — textbook hedge CLI
- `sweep_symlog.py` — Phase A HP sweep harness
- `sweep_symlog_phaseB_clean.py` — multi-seed validation harness
- `sweep_symlog_phaseC_clean.py` — production validation harness
- `decomp_phaseD.py` — pre/post-settle decomposition diagnostic

### Suggested for `.gitignore` (cruft cleaning)

- `STATUS_NOTE*.md` (these are ephemeral status reports)
- `MEMORY.md`, `.claude/` (Claude auto-memory)
- `__pycache__/`, `tests/_stepper_out*/`, `.ipynb_checkpoints/`
- `artifacts/sweeps/`, `artifacts/textbook_runs/`, `artifacts/daily_runs/`
- 17 old `sweep_*.py` files (sweep_aps, sweep_dim10, sweep_hp_*, sweep_lr,
  sweep_postfix) — stale pre-refactor artifacts

### Untracked but probably to keep tracked

- `tests/fixtures/data/AACalendars.cal` (the JSON references this; tests fail without it)

---

## Compute used overnight

| phase | runs | epochs | wallclock (2-GPU) |
|-------|------|--------|-------------------|
| Phase A clean | 12 | 100 | ~42 min |
| Phase B clean | 12 | 100 | ~42 min |
| Phase C clean | 8 | 200 | ~56 min |
| Textbook + bigval | 3 | — | ~3 min |
| **Total** | **35** | — | **~143 min** |

All artifacts under `artifacts/sweeps/2026051*_symlog_phase{A,B,C}_clean*/`.
The combined CSV is at `artifacts/sweeps/symlog_phaseAB_clean_combined.csv`.

---

## How to reproduce / continue

```bash
# Re-train the winning config (~7 min on one GPU):
python train_daily.py --epochs 100 --dense_tracking_reward_scale 3 --tag winner

# Re-run textbook benchmark on a fresh seed (~30s):
python run_textbook_hedge.py --batch_size 16384 --seed 999 --tag baseline

# Re-eval a trained policy on a held-out bundle (~30s):
python eval_policy_artifact.py --run_dir artifacts/sweeps/<phase_B_dir>/<row> \
                                --batch_size 16384 --seed 999

# Full multi-seed validation sweep (4 configs × 4 seeds × 100ep, ~42 min wallclock):
python sweep_symlog_phaseB_clean.py --row_start 0 --row_end 12

# Decomposition (pre/post-settle hedge contribution) on a trained policy:
python decomp_phaseD.py --phase_dir_glob 'artifacts/sweeps/<dir>/<row_pattern>'
```
