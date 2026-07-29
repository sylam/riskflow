# Bundle and Environment

The **bundle** is the simulated world as tensors; the **stepper** is that world as an environment
you can advance a day at a time. The solver reads the first and the backtest drives the second.

Time tensors carry a `History_Lookback_Business_Days` prefix of realized rows in front of the
simulated grid, so full-grid indexing is `initial_time_index + t` while the `*_sim` views strip the
prefix for code that indexes by simulation-grid `t`. Mixing the two is the single easiest mistake to
make against this contract.
