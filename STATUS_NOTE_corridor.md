# Status note — delta-corridor & train-in-corridor goal, 2026-07-17

## TL;DR

**Goal**: implement the corridor in training, generalize it, audit, validate; expected
improvement in E[pnl] and lower variance.

**Resolution**: implementation, generalization and audits are complete and committed.
The validation verdict is a rigorous NEGATIVE on the training half, with a structural
explanation: **training cannot improve the corridor — the corridor's entire effect is the
fence.** And at n=48 the fence itself moves only VARIANCE (−32%, certain), not expected
pnl (t=0.16, noise) — exactly what martingale discipline predicts.

**Operating point (unchanged, shipped)**: corridor-FREE checkpoints + roll-time
`Total_Position_Schedule` at **band 0.60** — mean +23.10 $/oz, std 60.4 vs baseline
+22.09 / 88.2 (48 trades, 2020–2023; 48/48 causal-bound PASS, 0 corridor breaches).
Band 0.40 (+18.64 / 47.4) if variance matters more. Quote the corridor as a **risk tool**,
never as a pnl improvement.

## The four falsification layers (all committed)

| # | Route for training to matter | Result | Evidence |
|---|---|---|---|
| 1 | Realized rolls, corridor-trained vs roll-clip | **Bit-identical** (trajectory L1=0.0, wT float-exact, churn equal, 10/10 months×bands) | f8c067b, `train_in_corridor_subset.csv` |
| 2 | In-sim distributions under the same fence | Δu ≈ 2e-5 (noise), tails unchanged | f8c067b, `tb_insim_*.json` |
| 3 | Adaptive band via fenced in-model u | u MONOTONE in tightness 46/48 (regularization artifact); selection skill ≈ chance (29–31% vs 33%); Spearman(u-margin, realized) ≈ 0 | 5460692, `adaptive_corridor_sweep.csv` + `adaptive_corridor_verdict.md` |
| 4 | Trade-date in-sim risk/mean forecasts | cvar5/p5/ew vs realized month: ρ ≈ +0.09 (p=0.56); best of 12 cells p=0.03 with perverse sign — not actionable | sweep CSV, pandas |

**Mechanism (layer 1)**: the roll argmax ranks candidate actions by E_inner[C_{t+1}];
the action enters only through one-step wealth W1, and u(W1) from the inner-MC fork
dominates the checkpoint-dependent A-net residual at dollar scale. Same fence + same
draws ⇒ identical trajectory for ANY checkpoint set.

**Structural capstone (layers 3–4)**: the corrected world is built martingale
(E[dF]=0 guard — the basis-saga discipline). No trade-date statistic derived from the
trained model can predict a single month's realized direction, by construction. The
corridor works because it encodes prior knowledge (the causal delta ramp), not prediction.

## Statistics of the fence itself (48 paired months)

- Mean: 0.60-vs-free paired delta +1.01/oz, paired SE 6.22, **t=0.16** (noise);
  0.40-vs-0.60 −4.46 ± 2.94 (t=−1.5).
- Variance: std(fenced) < std(free) in **100%** of 20k paired bootstraps
  (60.4 vs 88.2 at 0.60; 47.4 at 0.40).
- Oracle per-month band selection = +43.56/72.2 over {0.40, 0.60, free} — real headroom,
  but capturing it causally requires information a martingale model cannot contain.

## What shipped and stays

- Corridor machinery generalizes (c87284d): `Total_Position_Schedule` per-step
  [Min_Total, Max_Total] knots on the signed total — long-only, short-only and
  sign-crossing net schedules (book-of-APS ready); enforced once in
  `HedgeActionSpace.grid_at(t)`; bank exploration corridor-projected; dynamic textbook
  benchmark fenced; artifacts stamp the schedule with loud-fail load provenance;
  83 tests incl. bitwise gates.
- Deployment simplification: **one corridor-free checkpoint set serves any band**
  (provably equivalent to retraining per fence). Only retrain inside a fence if a
  fenced V_0 is quoted as a reserve.

## Open item (user decision)

Band adaptivity with EXTERNAL information (realized-vol / regime conditioning) is the
one untested route to the oracle headroom. New goal if wanted; must survive
leave-one-out on 48 months. Recommendation: ship fixed 0.60; revisit adaptivity in the
book-of-APS phase where netting changes the corridor geometry anyway.
