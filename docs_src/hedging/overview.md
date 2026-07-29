# Hedging Overview

`Execution_Mode` selects one of two verbs on the same simulated world:

- **`simulate_only`** — roll the bundle forward with zero trades and report the terminal P&L
  summary, the unhedged baseline. Callers drive their own policy on top via
  `HedgeRuntimeExecutionResult.create_stepper()`.
- **`solve_hedge`** — fit a value function and report a verdict against benchmark tracks.

## One boundary

`construct_hedge_runtime` (`hedge_runtime.py`) is the only place the `Hedging_Problem` JSON block
is read and validated. It returns the normalized **runtime** every consumer indexes by key:
canonical lowercased modes, the instrument / cash-account / hedge name sets, per-instrument
metadata, the accounting rules (position limits, turnover cost, spreads, margin funding, corridor),
the objective, the solver config and the portfolio state.

Past that boundary the runtime **is** the contract — downstream code indexes it directly rather
than re-checking it. A malformed schedule, a partial spot history, an unknown `Execution_Mode` or a
`Simulation_Batches` that contradicts the mode all fail there, naming the field.

!!! warning "`Simulation_Batches` means two different things"
    Under `simulate_only` it is a **path multiplier**: every batch is accumulated into one bundle of
    `Batch_Size x Simulation_Batches` paths. Under `solve_hedge` it is a **stream length**: a bundle
    per batch, `N-1` of them fitted and the last held out for the verdict. Minimum 2. That is a
    genuine semantic difference between two verbs, not a hidden mode flag — size a solve like a
    simulation and you will train on a fraction of the paths you meant to.

## Objective

The DP recursion lives in utility space, so `solve_hedge` requires a utility `Objective.Object` —
one of `AsymmetricUtility_Symlog`, `AsymmetricUtility_Huber`, `AsymmetricUtility_CARA`. The scale
`c` is resolved once when the bundle is built and never recomputed: a per-rollout `c` silently
rescales every reward, so a loaded checkpoint evaluates in **its own** frame.

## Privileged factors

What each stochastic process publishes as market state is a naming convention owned here:
`derive_privileged_layout` asks every live process what it emits, and multi-commodity runs prefix
the attribute with the factor name to disambiguate. Adding a process with its own privileged
surface flows through automatically — no registration step.
