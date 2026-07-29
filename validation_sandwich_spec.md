# Validation architecture: belief state + the BSS duality sandwich

## The golden rule (read before every step in this milestone)

> **What am I actually trying to achieve, and is what I'm doing taking me closer — or is there
> something else that would get me there faster?**

This milestone exists because we violated that rule for three days. The toy's full-information DP
gave an exact `V_0 = 0.92`; the solver produced `3.57`; we spent three days chasing that number
through λ-mix, gradient-label fixes, and baseline tweaks. **The number was the wrong target.** The
real solver cannot and should not reproduce a full-information oracle, because the real solver does
not observe the regime. We were optimising agreement with a yardstick that doesn't apply.

So before any commit in this milestone, answer the rule. If the thing you're tuning isn't moving the
**duality gap** (§3) or the **dollar floor** (§0), stop — it's the V_0 chase in a new costume.

## 0. What we are actually trying to achieve

A hedging policy whose **realised cost in $/oz beats the $6–8 dealer margin**, with a **certificate
that it is near-optimal** — produced without any privileged-information oracle, so it scales to the
real problem where no DP exists. Two numbers define success, both in dollars on shared paths:

- **Floor** = realised cost of our policy (achievable, lower bound on what's possible).
- **Certificate** = the policy is provably within ε of optimal when the duality gap is tight.

Ship when the floor clears the dealer margin **and** the gap is tight enough to certify we're not
leaving material value on the table. Nothing else is the target.

## 1. What changes, and why

Two changes, one decision each:

1. **State becomes belief, not regime.** A market participant never observes "we are in regime 1";
   it observes prices. Feeding the true regime one-hot is privileged information — a policy trained
   on it solves an easier problem than the one we face and won't transfer. Replace it with the **HMM
   filter posterior** computed from the observed price path. (The belief-weighted drift also fixes
   the V_0 over-optimism's mechanism for free — see §5.)

2. **Validation target becomes the Brown–Smith–Sun information-relaxation sandwich, not the toy
   oracle.** Once we use belief, the full-information DP is the *wrong* yardstick (partial info is
   worth strictly less; the solver can't match it and shouldn't try). The sandwich validates a
   partial-information policy with no oracle and scales to production dimensionality.

The toy is **not deleted** — it did its job (method validation: it caught the F_t1 bug, arbitrated
the depth gate). It is **demoted to a fast regression test of the machinery** (§6). We stop treating
its `V_0` as a target.

## 2. Belief state

`z_t` carries only participant-visible coordinates:

- **belief** — HMM filter posterior over regimes, computed from the observed price path (a few
  numbers; smooth in the price observation, hence differentiable — unlike the one-hot).
- price / live-future prices, running average, inventory, wealth, time, and the book exposure block.
- **No true-regime one-hot anywhere in `z_t`.** The true regime is used only to *simulate* paths,
  never as a network input.

The baseline drift in `B_t` becomes the **belief-weighted expected drift to terminal** (propagate
belief under the transition matrix; take expected cumulative drift). This is the regime-aware
baseline from the prior discussion, with filter probabilities instead of privileged ones — more
correct *and* more honest, same change.

## 3. The sandwich (Brown, Smith & Sun information-relaxation duality)

For any policy and **any** dual-feasible penalty `π`:

```
   U  =  E_paths[ max over action sequences, FUTURE KNOWN, of ( Σ r_t − π ) ]     (upper bound)
  V*  =  true optimal value
   L  =  E_paths[ realised Σ r_t under our policy (argmax C_t) ]                   (lower bound)

   U  ≥  V*  ≥  L           gap = U − L
```

The certificate: **valid for any C_t**. A bad `C_t` still gives a *true* upper bound, just a loose
one. So `gap` is an unconditional certificate — when it's tight, the policy is within `gap` of
optimal and any `C_0` outside `[L, U]` is provably wrong; when it's wide, the bound is telling you
**your `C_t` is bad**, which is the diagnostic you want. The instrument that validates the solver and
the object the solver produces are the same object — the bound tightens exactly when the solver works.

**The three ingredients — all already computed by the current code:**

- **Lower bound `L`** — realised-MC of the policy you already roll forward in the audit. Have it.
- **Penalty `π`** — the value-function-generated martingale penalty:
  ```
  π(path) = Σ_t [ C_{t+1}(s_{t+1}) − Ê[ C_{t+1}(s_{t+1}') | s_t, a_t ] ]
  ```
  Each term is mean-zero given `(s_t, a_t)` (a martingale difference), so `π` is dual-feasible. The
  conditional expectation `Ê[C_{t+1} | s_t, a_t]` is **exactly the bootstrap target** the solver
  already computes — reuse it (afford more inner draws here than in training; this is offline).
- **Upper bound `U`** — the clairvoyant inner optimisation you already have the pieces for. With the
  path's noise known, `s_{t+1}` is a deterministic function of `(s_t, a_t)`, so the clairvoyant's
  problem is a **deterministic backward DP along the single known path** over the action grid,
  maximising `Σ r_t − π`. Cheap. Average over paths → `U`.

`gap = U − L`, in the same utility units; report alongside the **dollar** floor (`$/oz` realised).

## 4. Guards (BSS is easy to wire subtly wrong)

1. **Dual-feasibility is the whole certificate — verify it.** `π` is a valid penalty only if each
   `Δ_t = C_{t+1}(s_{t+1}) − Ê[C_{t+1}|s_t,a_t]` is genuinely mean-zero under the **true** dynamics.
   Before trusting any gap: simulate paths, compute `Σ Δ_t`, confirm the sample mean ≈ 0 within MC
   error. If it isn't zero-mean, `U` is **not** an upper bound and the certificate is void. This is
   the one place to be rigorous, not quick.
2. **Information sets must not cross.** The clairvoyant sees the future **price path** (the noise).
   The penalty's conditional expectation and the policy operate on the **belief-filtered** set. Never
   let `π` or the policy peek at the hidden regime — letting the penalty see privileged regime
   relaxes the wrong constraint and silently breaks the bound. This is the classic BSS bug.
3. **Grade in dollars, on the gap.** Ship criterion = floor (`L`, in `$/oz`) clears the dealer
   margin **and** `gap` is tight. Use "gap narrowing across iterations" as the progress signal; never
   optimise to close the *clairvoyant* slack for its own sake — a tight bound comes from a good `C_t`,
   not from chasing `U` down directly.

## 5. How this arbitrates every open question (the point)

`V_0` calibration, the regime-aware baseline, λ-mix — all of them collapse to a single scalar:
**does it tighten `gap = U − L`?**

- The `3.57` over-optimism is now testable without an oracle: if `3.57 > U`, it is **certified**
  over-optimistic; if it sits inside `[L, U]` and the gap is tight, it's fine. No toy required.
- The **belief-weighted baseline** is expected to be the real fix — single-regime drift over a
  switching horizon forces `A_t` to absorb the regime-mixing, which is why over-optimism grew with
  depth and was worse in bear. A belief-weighted baseline makes `A_t` a genuinely small residual,
  which is the regime where the differential label stabilises and the gap tightens. Grade it on the
  gap, not on matching `0.92`.
- **λ-mix stays shelved.** This morning showed it buys calibration by eroding the regime policy
  (its rollout reuses the flawed downstream stack). It earns a place only if a *bounded* residual gap
  survives a correct belief-weighted baseline — and then only if it tightens the sandwich without
  costing the floor.

One scalar, with a proof attached, answers what three days of tuning couldn't.

## 6. Build order

1. **Belief filter** — HMM posterior from the observed price path into `z_t`; drop the true-regime
   one-hot. Belief-weighted drift in `B_t`.
2. **Lower bound `L`** — realised `$/oz` of the policy on a fresh MC set (you have the rollout).
3. **Penalty `π`** — assemble from `C_{t+1}` increments and the existing one-step conditional
   expectation. **Run guard 1 (zero-mean check) before proceeding.**
4. **Upper bound `U`** — clairvoyant deterministic backward DP along each known path with `−π`;
   average. Confirm `U ≥ L` on every batch (if `U < L`, a guard is violated — stop).
5. **Report** `gap = U − L` and the dollar floor. This replaces the toy-oracle comparison as the
   validation readout.
6. **Demote the toy** to a regression unit test — assert the machinery still runs and its *policy*
   matches the exact DP at small horizon. Stop asserting on its `V_0`.

## 7. Stopping rule (the golden rule, operationalised)

- **Ship** when: dollar floor clears the dealer margin **and** `gap` is tight (policy certified
  near-optimal).
- **Continue** (one lever, graded on the gap) only while the gap is materially loose: belief-weighted
  baseline first; λ-mix only on a surviving bounded residual.
- **Stop and re-read this rule** the moment you find yourself tuning a number that isn't the gap or
  the floor. That is the failure mode this milestone was created to end.

## 8. What this retires

- Toy full-information DP as the *validation target* → demoted to a machinery regression test.
- `V_0`-vs-`0.92` as a success metric → replaced by `gap` and the dollar floor.
- λ-mix as a calibration fix → shelved; the belief-weighted baseline is the calibration mechanism.
- The privileged regime one-hot in `z_t` → replaced by belief everywhere.
