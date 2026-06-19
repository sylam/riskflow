# Implementation brief — BSS validation sandwich (upper bound `U` + gap)

## What this is

The diff-ML solver is closed and fork-free; the labels are validated on their own terms. The
next step is the **validation sandwich** — the actual verdict, per `validation_sandwich_spec.md`.
This is **not** the toy/exact-DP curve. The toy is demoted to a machinery regression test
(spec §6.6, §8); asserting on its `V_0` is the failure mode the spec exists to end.

Two of the three sandwich ingredients already exist. This brief covers the missing third —
the **penalty `π`**, its **feasibility guard**, and the **penalized upper bound `U`** — then
the `gap = U − L` readout. That is the whole job.

## The golden rule (read before every commit — spec §0)

> What am I actually trying to achieve, and is what I'm doing taking me closer?

Three things to hold fixed, because they are easy to lose:

1. **The sandwich MEASURES the gap. It does not close it.** Building it tells you *whether* a
   gap exists and *where*. The thing that closes a wide gap is a better `C_t` — not the sandwich.
2. **The bracket `U ≥ V* ≥ L` holds for ANY `C_t`, even a bad one.** A bad `C_t` gives a *true*
   bound, just a loose one. Never impose `V(t) = ` the rolled-forward chain to `V(T)`. That is a
   Bellman equality that holds only at exact convergence; demanding it is the `V_0` chase in a new
   costume. Consistency *emerges* when a good `C_t` makes the gap small — it is never enforced.
3. **If the thing you are tuning is not moving `gap` or the dollar floor, stop.**

## Already built — reuse, do not rebuild

- **Lower bound `L`** — `DifferentialSolver._compute_dollar_floor` (`hedge_solver.py:3037`):
  realised cost in `$/oz` of the policy on fresh paths. Reported at `~3417`. This is `L`. Have it.
- **Belief state + belief-weighted baseline** — done (the fork-free belief work; spec §1–2,
  baseline refs at `~1826`, `~1970`).
- **Clairvoyant DP skeleton** — `HindsightDpSolver` (`hedge_solver.py:1492`): backward max-plus DP
  over the action grid along each realised path, with L1 turnover. **But this is the naïve
  perfect-information bound — it has NO penalty.** It is the *skeleton* for `U`, not `U` itself
  (see below).
- **Inner-MC machinery** — the fork we deliberately kept for offline yardsticks (`inner_mc_fn` /
  `inner_mc_grad_fn`). The penalty's conditional expectation runs on this, offline, with more
  draws than training. This is why the fork was retained.

## Why the naïve clairvoyant is not enough (the load-bearing point)

`HindsightDpSolver` gives `U_perfect = E[max over actions, future KNOWN, of Σ r_t]`. Because
partial information is worth strictly less than full information, `U_perfect − L` stays **wide
even when your policy is optimal for the partial-info problem you actually face**. That gap
conflates "my `C_t` is bad" with "knowing the regime is worth a lot here" — it is uninformative,
and it is exactly the trap spec §0/§1.2 warn against.

The **penalty `π`** is the entire mechanism that pulls `U` down from the perfect-info bound toward
`V*`. With a good `C_t`, `π` nearly cancels the value of foresight and the gap goes tight. Build
`π`, or you have not built the sandwich.

## What to build

### 1. Penalty `π` (spec §3, §6.3)

The value-function-generated martingale penalty:

```
π(path) = Σ_t [ C_{t+1}(s_{t+1})  −  Ê[ C_{t+1}(s'_{t+1}) | s_t, a_t ] ]
```

- `C_{t+1}(s_{t+1})` — the trained continuation `self.C[t+1]` evaluated at the **realised**
  next state on this path under action `a_t`.
- `Ê[C_{t+1} | s_t, a_t]` — the conditional expectation over the next-step noise. This is the
  **bootstrap target the solver already computes**; reuse that path. Compute it on the retained
  **inner MC** with **more draws than training** (offline — afford it).
- Each term `Δ_t = C_{t+1}(s_{t+1}) − Ê[C_{t+1}|s_t,a_t]` is mean-zero given `(s_t, a_t)` by
  construction (a martingale difference), which makes `π` dual-feasible. Both terms depend on the
  action `a_t`, so `π_t` is evaluated per `(path, t, grid-action)`.

### 2. Guard 1 — zero-mean dual feasibility (spec §4.1) — **FIRST checkpoint, hard gate**

This is the one place to be rigorous, not quick. The whole certificate rests on it.

- Simulate paths, compute `Σ_t Δ_t`, confirm the sample mean ≈ 0 within MC error.
- This *is* the legitimate one-step-consistency check (the real version of the "rolling forward
  should be consistent" intuition) — but it is a **feasibility test on `π`**, not an equality
  forced on `V`.
- **If it is not zero-mean, `U` is not an upper bound and the certificate is void. Stop here.**
  Do not proceed to `U` until this is green.

### 3. Upper bound `U` (spec §3, §6.4)

Extend the `HindsightDpSolver` max-plus DP: along each known path, run the deterministic backward
DP over the action grid maximising the **penalized** reward,

```
J_t(n_prev) = max over n [ r_t(n)  −  π_t(n)  +  J_{t+1}(n) ] ,   roll back to J_0, average over paths → U
```

i.e. subtract `π_t` inside the existing per-step objective (`_step_pnl(t)` term). Then:

- **Confirm `U ≥ L` on every batch.** If `U < L`, a guard is violated (almost always an
  information-set leak — see guard 2). Stop and find it; do not "fix" it by clamping.

### 4. Report `gap = U − L`, in utility units, alongside the dollar floor `L` (`$/oz`)

This replaces the toy-oracle comparison as the validation readout.

### 5. (Optional, useful) Per-`t` gap decomposition

`π` and the clairvoyant slack are sums of per-step terms, so looseness can be attributed to
specific `t` — a "where is `C_t` weak" profile. Build this as a **deliberate decomposition** for
localisation; it is not the default readout. Use it to target the lever, not to chase a number.

## Guards (spec §4 — BSS is easy to wire subtly wrong)

1. **Dual feasibility (zero-mean)** — §2 above. The whole certificate.
2. **Information sets must not cross.** The clairvoyant sees the future **price path** (the
   noise). The penalty's conditional expectation and the policy operate on the **belief-filtered**
   set. **Never let `π` or the policy peek at the hidden regime** — that relaxes the wrong
   constraint and silently breaks the bound. This is the classic BSS bug and the most likely cause
   of any `U < L`.
3. **Grade in dollars, on the gap.** Progress signal = gap narrowing across iterations. **Never
   optimise `U` downward for its own sake** — a tight bound comes from a good `C_t`, not from
   chasing the clairvoyant slack.

## Reading the result, and the one lever

- **Gap tight AND floor clears the dealer margin (`$6–8/oz`)** → ship. The policy is certified
  within `gap` of optimal.
- **Gap wide** → that is the diagnosis that `C_t` is weak (where, if you built §5). The fix is a
  better `C_t`, **not reflexively more data** — more data cannot fix a mis-specified baseline.
  Reach for the **belief-weighted baseline first** (largely in already; grade it on the gap).
  **λ-mix stays shelved** — it earns a place only if a *bounded* residual gap survives a correct
  belief-weighted baseline, and then only if it tightens the gap without costing the floor (§5).

## Don'ts (scope discipline)

- Don't impose `V(t) =` the forward chain to `V(T)`. The bracket holds for any `C_t`.
- Don't optimise `U` down directly; don't clamp `U ≥ L` — investigate the leak instead.
- Don't let the clairvoyant or `π` see the hidden regime.
- Don't re-chase the toy `V_0`. The toy is a machinery regression test now.
- `U`/`π` are an **offline yardstick** — never a solver dependency, never shipped (same status as
  LSM/hindsight). Keep them out of the shippable path.

## Build order + checkpoints

1. **`π`** — assemble from `C_{t+1}` increments and the existing one-step conditional expectation
   (inner MC, extra draws, offline).
2. **Guard 1 (zero-mean)** — hard gate. Green before proceeding, or stop.
3. **`U`** — penalized clairvoyant DP (extend `HindsightDpSolver`). Confirm `U ≥ L` per batch.
4. **Report** `gap = U − L` + dollar floor. This is the verdict readout.
5. *(optional)* per-`t` gap decomposition for localisation.
6. Then, only if the gap is materially loose: the belief-weighted-baseline lever, graded on the gap.
