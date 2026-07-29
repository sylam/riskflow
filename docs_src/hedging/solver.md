# Solver

`Solver.Object` must be `DiffSolverV2` — the differential-ML value-function solver. `HindsightDpSolver`
remains available as the clairvoyant benchmark track (`Run_Hindsight_Diagnostic`), not as a primary
solver.

A solve is a **stream**: `StreamingSolve` takes one bundle per simulation batch as the calc builds
it, fits on all but the last, and reports the verdict on the last — a world no fit step saw. Every
track (the solver, the benchmarks, the stepper rollout) optimizes over the same
`HedgeActionSpace`, so they trade the same positions and pay the same frictions.
