# Spec — completing the differential coverage of C_t

Implements the "second issue": the twin-loss differential label currently supervises only
`(wealth, live tradable prices)`. This extends it to the coordinates that actually carry the
objective's pathwise sensitivity — **without** over-extending it to redundant features.

## Governing principle

Differentiate the **primitive continuous coordinates** — the ones that are (i) genuine stochastic
state, (ii) causally drive the objective, (iii) have a computable pathwise gradient through the
one-step dynamics — and make those differentials **complete** (they must propagate through the
**liability**, not just the hedge instruments). Do **not** add separate differentials for features
that are deterministic functions of the primitives; that double-counts and feeds the net correlated
slope targets, adding noise, not shape.

This is the golden rule applied to the Jacobian: cover what the objective depends on, nothing more.

## What's primitive vs derived (grounded in `hedge_features.py`)

**Primitive continuous coordinates** (differentiate these):
- `price` per tradable — *covered.*
- `wealth` / loss-budget — *covered.*
- `accumulation_fraction` (leg block, the realized-averaging progress) — **MISSING.** Primitive: a
  path integral of past fixings, not a function of the others. Linear, clean gradient.

**Derived features — do NOT differentiate (keep value-supervised only):**
- `liability_mtm`, `hedge_ratio_*`, `total_position_value`, `near/mid/far_position_value`,
  `coverage_ratio`, `position_value`. Each is a deterministic function of `(price, accumulation_fraction,
  inventory, time)`. Their sensitivity must reach C_t **through** the primitives (next section), not as
  independent slope targets.

**Controls / non-stochastic (correctly excluded):** `inventory`, `time_to_*`.

## The "realized average" and "liability sensitivity to each tradable" — what they actually are

The user's two questions map onto the framework as follows, and they are **not** two new columns:

- **Realized average** has two facets: the *progress* (`accumulation_fraction`, Work Item A) and the
  *realized value*, which enters the objective through `liability_mtm`. The progress is differentiated
  directly; the value flows through the liability (Item B).
- **Liability sensitivity to each tradable** is the liability's delta to each future's price. It is
  **already** the chain `price_k → liability_mtm → C`. It does not need its own column — it needs the
  liability to be **on the one-step AAD tape** so the existing `price` differential carries it. If the
  liability is off-tape, every price differential is missing the liability half of the cross-exposure
  — which is precisely the exposure the hedge exists to neutralize.

## Work Item A — add `accumulation_fraction` to the differential set

1. Ensure `accumulation_fraction_t` is a leaf with `requires_grad` in the one-step fork, and its
   one-step update (accrue the new fixing) is on the tape, so `∂Y_boot/∂accumulation_fraction` exists.
2. Extend the baseline gradient: add the analytic `∂B/∂accumulation_fraction` term, FD-guarded the same
   way as the existing `∂B/∂z` (`fd_check_dB`).
3. Unmask the `accumulation_fraction` column in the differential mask (`hedge_solver.py:~2507-2525`)
   and set its `w_diff` weight to 1.0, consistent with the column-mask convention already in place.

## Work Item B — make the price/wealth differential flow through the liability (CONDITIONAL)

The one-step fork uses the F_t1 short-circuit, which skips the pricing chain. So the liability may be
**frozen** in the fork, exactly the failure class as the original F_t1 wealth bug — and if so, no
price differential carries the liability response.

1. **Diagnostic first (cheap, decides whether B is needed):** in the one-step fork, perturb one
   tradable's price and check whether `liability_mtm` (and `wealth`, if it includes the liability)
   responds. This is the **nonzero probe** — the same check that would have caught F_t1.
   - **If frozen →** put the liability's delta on the tape. The liability is an average-rate swap,
     **linear in the remaining fixings**, so its delta to each tradable's underlying is **analytic**
     (the remaining-fixing weight). Inject it into the fork so `∂liability_mtm/∂price_k` is on the tape
     and the existing `price` (and `accumulation_fraction`) differentials become complete. This needs
     no full reprice — only the analytic delta, which is already upstream of the skipped pricing chain.
   - **If live →** the differential already carries the liability; **no work** — only Item A remains.
2. Do **not** add a `liability_mtm` differential column either way. Once it's on the tape, its
   sensitivity is delivered through the primitives; a separate column double-counts.

## Work Item C — the redundancy guard (run before differentiating anything not listed in A)

Before unmasking any derived feature, confirm it carries information the primitives don't:
- Regress the candidate (e.g. `liability_mtm`) on `(price, accumulation_fraction, inventory, time)`
  across the bank. `R² ≈ 1` ⇒ redundant ⇒ keep value-only. `R² < 1` (material path-dependence not in
  the primitives) ⇒ it is itself a primitive and may be differentiated.
- Expected outcome for this book: all the derived features are redundant; the differential set after
  this spec is exactly `{price, wealth, accumulation_fraction}` with the liability on the tape.

## Correctness checks (how to verify — required gates)

1. **Nonzero probe (per new column, and the Item-B liability):** the measured gradient must be nonzero
   on a representative batch. A zero means the coordinate isn't propagating to `z_{t+1}` — a wiring/
   frozen-coordinate bug (this is the F_t1-class detector; run it first, it's three lines).
2. **FD vs AAD (per new column):** finite-difference `Y_boot` w.r.t. each newly differentiated
   coordinate; compare to the autograd gradient. Pass at the existing tolerance (`fd_check_dB` uses
   ~1e-4; the prior `_dB_dz` checks ran 2–5e-6). Belief-style soft inputs and one-hot edge cases both.
3. **Baseline-gradient FD (`∂B/∂accumulation_fraction`):** extend `fd_check_dB.py` to the new coord. A
   wrong analytic baseline gradient is what blew up training in the `w_diff=0` episode — gate on it.
4. **Sign / magnitude sanity:** `∂C/∂accumulation_fraction` and the liability-routed `∂C/∂price` must
   carry the economically correct sign (more locked-in liability / adverse price → lower continuation
   value). A sign flip is the bug that inverted the bull regime earlier.
5. **Redundancy check (Item C):** the regression-`R²` gate above, before any derived feature is
   differentiated.
6. **Integration — grade on the depth profile, not on V_0:** re-run the depth profile after the change.
   Completing the differential set should tighten the audit residual / shrink over-optimism in the
   directions that were under-supervised. When the sandwich exists, grade on the gap. Do **not** read
   success as "V_0 moved toward any number."

## Coordination

Item B touches the **same one-step fork** as the belief-filter work in issue (a). Sequence them so the
fork is modified once: add the inner belief update (a) and the analytic liability delta (B) in one pass,
then run the nonzero + FD checks across belief, accumulation_fraction, and the liability together. The
belief differential (if (a) makes `belief_{t+1}` differentiable) and this spec's coverage are the same
mechanism — extend the column mask once, verify once.
