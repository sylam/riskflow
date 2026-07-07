"""Differential-ML dynamic-hedging solver (Execution_Mode='solve_hedge').

`DiffSolverV2` is the production solver: a backward-DP / differential-ML value function
fit by the Huge–Savine twin loss (value + AAD pathwise-gradient), consuming the simulated
scenario bundle and forking inner MC on demand via the closure `bundle['inner_mc_fn']`
attached by `HedgeMonteCarlo.execute`. `HindsightDpSolver` (clairvoyant oracle, the
upper-bound track) and `run_textbook_benchmark` (averaging / min-var lower-bound track)
are kept as benchmarks; `solve_hedge` dispatches the primary solver and assembles the
comparison table + acceptance ladder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from . import utils
from .hedge_bundle import (
    _utility_wrap_signed, _mirror_utility_scale_to_runtime,
)


@dataclass
class SolverResult:
    """High-level result of a hedge solver. Field shapes vary by solver — MPC produces
    a single decision state (`actions` is one `(n_hedge,)` target-position vector,
    `values` a scalar expected utility); the DP solvers produce per-(t, outer-path) grids."""
    solver_name: str
    actions: Any
    values: Any
    terminal_pnl: Optional[Any] = None
    terminal_utility: Optional[Any] = None
    value_fn_artifacts: Optional[Any] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def build_action_grid(runtime, levels_per_axis, device):
    """Discrete action grid in raw contract units — one axis per hedge instrument. Each
    axis spans `[Min_Position_i, Max_Position_i]` (from
    `runtime['accounting']['position_limits']`) with `levels_per_axis` evenly-spaced
    levels, endpoints included. The action is the target inventory `(n_1, …, n_hedge)`,
    not a trade delta. Returns `(n_actions, n_hedge)`."""
    hedges = runtime["names"]["hedges"]
    limits = runtime["accounting"]["position_limits"]
    axes = []
    for name in hedges:
        lim = limits.get(name, {})
        lo = float(lim.get("min_position", 0.0))
        hi = float(lim.get("max_position", 0.0))
        axes.append(torch.linspace(lo, hi, levels_per_axis, device=device))
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.reshape(-1, len(hedges))


def _turnover_cost(delta_contracts, kappa):
    """L1 turnover cost: Σ_i |Δcontracts_i| · kappa_i. `delta_contracts` is
    `(..., n_hedge)`, `kappa` is `(n_hedge,)` per-contract cost. Hard (no smoothing) —
    action search has no gradients on the action."""
    return (delta_contracts.abs() * kappa).sum(dim=-1)


def _per_contract_kappa(tradables_sim, runtime, hedges, t_index, device):
    """Per-contract trading cost per hedge instrument: a flat `Transaction_Cost_Per_Unit`
    plus a half-spread charge `0.5 · Bid_Offer_Spread_Bps · 1e-4 · F_i(t) · contract_size_i`.
    `tradables_sim` is the history-stripped tradables view; `t_index` is simulation-grid."""
    acc = runtime["accounting"]
    contract_size = torch.tensor(
        [float(runtime["tradables"][ref]["contract_size"]) for ref in hedges],
        device=device)
    f_t = torch.tensor(
        [float(tradables_sim[ref][t_index].mean()) for ref in hedges],
        device=device)
    kappa = (acc["transaction_cost_per_unit"]
             + 0.5 * acc["bid_offer_spread_bps"] * 1.0e-4 * f_t * contract_size)
    return kappa, contract_size


def _bundle_sim_views(bundle):
    """History-stripped views of the bundle's time-indexed tensors. `build_hedge_bundle`
    prepends a `History_Lookback_Business_Days` prefix to `tradables` / `liability_mtm`, but
    `inner_mc_fn` works in simulation-grid coords. Stripping the prefix lets the solver index
    every time tensor by the same sim-grid `t`. Returns `(tradables_sim, n_outer_steps)`."""
    hist = int(bundle.get("initial_time_index", 0))
    tradables = {k: v[hist:] for k, v in bundle["tradables"].items()}
    n_outer_steps = int(bundle["liability_mtm"][hist:].shape[0])
    return tradables, n_outer_steps


def _realized_paths(bundle, runtime):
    """Realized outer-path data the no-inner-MC tracks (hindsight, textbook) consume:
    `F` `(n_hedge, t_outer, B_outer)` hedge prices, `L_T` `(B_outer,)` the liability
    terminal MTM, and `t_outer`. The liability terminal mirrors the DP's `inner_mtm[-2]`
    convention — the pre-settlement terminal (index -1 is the appended clean-exit zero)."""
    tradables_sim, t_outer = _bundle_sim_views(bundle)
    hedges = list(runtime["names"]["hedges"])
    F = torch.stack([tradables_sim[h] for h in hedges], dim=0)        # (n_h, t_outer, B)
    hist = int(bundle.get("initial_time_index", 0))
    liab = bundle["liability_mtm"][hist:]                             # (>=t_outer, B)
    L_T = liab[min(t_outer - 2, liab.shape[0] - 1)]                   # (B,)
    return F, L_T, t_outer


def _axis_levels(runtime, levels, device):
    """Per-hedge-axis 1-D level values — the `linspace(min, max, levels)` that
    `build_action_grid` takes the meshgrid of. Returned as a list of `(levels,)` tensors."""
    limits = runtime["accounting"]["position_limits"]
    return [torch.linspace(float(limits.get(h, {}).get("min_position", 0.0)),
                           float(limits.get(h, {}).get("max_position", 0.0)),
                           levels, device=device)
            for h in runtime["names"]["hedges"]]


def _maxplus_grid(A, kappa, axis_vals, levels, n_hedge):
    """`J[n_prev, b] = max_n [ A[n, b] - Σ_i kappa_i·|val_i(n) - val_i(n_prev)| ]` over the
    discrete grid. The L1 turnover cost is separable across hedges, so the max-plus
    factorizes into `n_hedge` sequential 1-D transforms — `O(levels^(n_hedge+1)·B)` rather
    than the `O(n_actions²·B)` full action-pair table. `-inf` entries of `A` (grid points
    masked out by a total-position limit) propagate harmlessly."""
    g = A.reshape(*([levels] * n_hedge), A.shape[-1])
    for i in range(n_hedge):
        v = axis_vals[i]
        cost_i = float(kappa[i]) * (v.unsqueeze(1) - v.unsqueeze(0)).abs()   # (L_p, L_q)
        g = g.movedim(i, 0)                                                 # (L_p, rest…, B)
        cand = g.unsqueeze(1) - cost_i.view(levels, levels, *([1] * (g.dim() - 1)))
        g = cand.amax(dim=0).movedim(0, i)                                  # (L_q, rest…, B)
    return g.reshape(levels ** n_hedge, A.shape[-1])


def run_textbook_benchmark(bundle, runtime):
    """Static-hedge reference: the single best constant position, held over the whole
    horizon with no rebalancing, evaluated on the realized outer paths. A valid lower
    bound for the dynamic DP — dynamic rebalancing can only add value. No inner MC, no
    V̂. Static hold telescopes the per-step P&L to `position · (F_T − F_0)`."""
    F, L_T, t_outer = _realized_paths(bundle, runtime)
    hedges = list(runtime["names"]["hedges"])
    device = F.device
    acc = runtime["accounting"]
    solver_cfg = runtime["solver"]
    tradables_sim, _ = _bundle_sim_views(bundle)
    kappa0, contract_size = _per_contract_kappa(tradables_sim, runtime, hedges, 0, device)
    kappa_T, _ = _per_contract_kappa(tradables_sim, runtime, hedges, t_outer - 1, device)
    grid = build_action_grid(
        runtime, solver_cfg["training_action_grid_levels_per_axis"], device)

    total_move = F[:, -1, :] - F[:, 0, :]                              # (n_h, B) telescoped
    g_t = torch.einsum("ai,ib->ab", grid * contract_size, total_move)  # (n_actions, B)
    cost = _turnover_cost(grid, kappa0)                                # entry turnover
    if acc["force_flat_at_end"]:
        cost = cost + _turnover_cost(grid, kappa_T)                    # terminal unwind
    u = _utility_wrap_signed(L_T.unsqueeze(0) + g_t - cost.unsqueeze(-1), runtime)
    obj = u.mean(dim=-1)                                               # (n_actions,)
    if acc["total_position_abs_limit"] > 0.0:
        violators = grid.abs().sum(dim=-1) > acc["total_position_abs_limit"]
        obj = obj.masked_fill(violators, float("-inf"))
    best = int(obj.argmax())
    return {"v0_mean": float(obj[best]), "v0_std": 0.0,
            "n_star": grid[best].detach().cpu().tolist(),
            "terminal_utility": u[best].detach().cpu()}


class HindsightDpSolver:
    """Clairvoyant upper-bound diagnostic. For each outer path it solves the deterministic
    optimal position trajectory GIVEN the realized future — a backward max-plus DP over
    the discrete action grid with L1 turnover cost. No inner MC, no V̂: the realized path
    is its own one-sample future.

    `u_signed` is monotone and the liability terminal `L_T(b)` is path-fixed, so
    maximizing `u_signed(W_T)` ≡ maximizing the additive cash `G_T` — hence a pure
    max-plus DP, then `u_signed` applied once at `t=0`. `mean_b V_0(b)` is an upper bound
    on any deployable (non-clairvoyant) policy's value — the reference the DP is measured
    against."""

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime

    def solve(self):
        bundle, runtime = self.bundle, self.runtime
        hedges = list(runtime["names"]["hedges"])
        n_h = len(hedges)
        acc = runtime["accounting"]
        solver_cfg = runtime["solver"]
        force_flat = acc["force_flat_at_end"]
        total_abs_limit = acc["total_position_abs_limit"]

        F, L_T, t_outer = _realized_paths(bundle, runtime)             # F (n_h,t_outer,B)
        device = F.device
        b_outer = F.shape[-1]
        tradables_sim, _ = _bundle_sim_views(bundle)
        levels = solver_cfg["training_action_grid_levels_per_axis"]
        grid = build_action_grid(runtime, levels, device)              # (n_actions, n_h)
        n_actions = grid.shape[0]
        axis_vals = _axis_levels(runtime, levels, device)
        valid = (grid.abs().sum(dim=-1) <= total_abs_limit
                 if total_abs_limit > 0.0 else None)

        # Terminal: J_{T}(n_prev) = -[force_flat] · cost(0, n_prev) — the unwind charge.
        kappa_T, contract_size = _per_contract_kappa(
            tradables_sim, runtime, hedges, t_outer - 1, device)
        J = (-_turnover_cost(grid, kappa_T).unsqueeze(-1).expand(n_actions, b_outer)
             if force_flat else torch.zeros(n_actions, b_outer, device=device))

        def _step_pnl(t):
            dF = F[:, t + 1, :] - F[:, t, :]                           # (n_h, B)
            return torch.einsum("ai,ib->ab", grid * contract_size, dF)  # (n_actions, B)

        # Backward DP, t = t_outer-2 … 1: J_t(n_prev) = max_n[pnl_t(n) - cost + J_{t+1}(n)].
        for t in range(t_outer - 2, 0, -1):
            kappa_t, _ = _per_contract_kappa(tradables_sim, runtime, hedges, t, device)
            A = _step_pnl(t) + J
            if valid is not None:
                A = A.masked_fill(~valid.unsqueeze(-1), float("-inf"))
            J = _maxplus_grid(A, kappa_t, axis_vals, levels, n_h)
        # t=0 decision from the flat start (n_{-1}=0): max over n_0 directly.
        kappa0, _ = _per_contract_kappa(tradables_sim, runtime, hedges, 0, device)
        A0 = _step_pnl(0) + J
        if valid is not None:
            A0 = A0.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        A0 = A0 - _turnover_cost(grid, kappa0).unsqueeze(-1)           # cost(n_0, 0)
        g0, n0_idx = A0.max(dim=0)                                     # (B,), (B,)
        v0 = _utility_wrap_signed(L_T + g0, runtime)                   # (B,)

        return SolverResult(
            solver_name="HindsightDpSolver",
            actions=grid[n0_idx].detach().cpu(),
            values=v0.detach().cpu(),
            terminal_pnl=(L_T + g0).detach().cpu(),
            terminal_utility=v0.detach().cpu(),
            diagnostics={
                "V_0": float(v0.mean()),
                "n_star_0": grid[n0_idx].float().mean(dim=0).detach().cpu().tolist(),
                "v0_abs_max": float(v0.abs().max()),
                "action_grid_size": n_actions,
            },
        )


def _result_v0(result):
    """Scalar V_0 of a track's `SolverResult` — every solver writes it to `diagnostics`."""
    return float(result.diagnostics["V_0"])


def _multiseed_summary(runs):
    """Aggregate a track's repeated solves into `v0_mean ± v0_std` (population std).
    Multi-seed repeats re-use the cached outer paths but advance the inner-MC Sobol
    stream, so deterministic tracks (hindsight, textbook) report `std == 0`."""
    v0 = [_result_v0(r) for r in runs]
    mean = sum(v0) / len(v0)
    std = (sum((x - mean) ** 2 for x in v0) / len(v0)) ** 0.5 if len(v0) > 1 else 0.0
    return {"v0_mean": mean, "v0_std": std, "v0_seeds": v0,
            "n_star": runs[0].diagnostics.get("n_star_0")
                      or runs[0].diagnostics.get("n_star")}


class _DiffV2Residual(torch.nn.Module):
    """Zero-init residual head A_t(market, q, W). The continuation is C = u(W) + A, so a
    zero final layer makes C start exactly at the bounded utility anchor (A ≡ 0) and the
    net only ever LEARNS the correction off that anchor — the toy's run-away guard."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        for p in self.body[-1].parameters():
            torch.nn.init.zeros_(p)

    def forward(self, x):
        return self.body(x).squeeze(-1)


class DiffSolverV2:
    """Clean-room differential-ML hedging solver — rebuilt from the toy (`diffml_hedge_huber.py`
    via `diffsolver_v2.py`, validated BOUNDED at T=119) and wired to the OFFICIAL riskflow
    framework. All dynamics come from the bundle's inner-MC closures
    (`inner_mc_fn` / `inner_mc_grad_fn`), which fork the simulator and price via
    `resolve_structure` (tradeables) + `resolve_hedge_structure` (liability) — no analytic
    transition, no Jacobian reconstruction; the framework prices everything.

    Spirit carried over verbatim from the toy:
      * C_t(market, q, W) = u(W) + A_t(market, q, W).  u = the bounded utility anchor
        (`_utility_wrap_signed`, the symlog/Huber/CARA transform normalised by c); A is a
        zero-init residual net — the only learned part. The bounded anchor is what keeps the
        backward recursion from running away (the old solver's 1e8 bug).
      * External argmax — the Bellman max lives OUTSIDE the fitted value (a discrete grid
        search over target positions, not inside the net).
      * Advantage decomposition — fit A = C − u(W) (value AND the wealth-channel pathwise
        gradient: a Huge–Savine twin loss), so the unbounded residual can't drift off u.
      * Operating-region bank — roll the OUTER paths forward exploring q AROUND the per-t
        replication (diagonal min-var) hedge, so wealth stays in-band and A stays on-support.
      * Position-free value (toy-faithful) — V(market, W) does NOT take the position as input;
        with no turnover cost the held position is a freely-reset control, so it enters the
        value only through next-step wealth W1 = W + Σ q_i·cs_i·dF_i + dL. The n_hedge
        instruments live in the ACTION grid + the wealth step (the net learns there are 3 via
        the routing of W1), not as a state coordinate. Adding q as a state is the right move
        ONLY once turnover cost makes the incoming position a real state variable. The action
        grid spans all hedges; a single-future-of-three test pins inactive axes to 0 via
        `Active_Hedge_Indices` (e.g. [2] ⇒ [0,0,-50]…[0,0,0]).

    Wealth convention: net wealth W_t = cumulative hedge P&L + the
    marked liability L_t; W_{t+1} = W_t + Σ_i q_i·cs_i·(F_{t+1,i} − F_{t,i}) + (L_{t+1} − L_t);
    terminal utility u(W_{T_dec}) with W_{T_dec} = total hedge P&L + L_T.

    INCREMENT 1 (this build): value bootstrap + the WEALTH-channel pathwise-gradient twin
    loss. W is the solver's own autograd leaf, so ∂Y_boot/∂W is exact with pure torch (no
    framework AAD needed). INCREMENT 2 adds the market-state (spot/belief) gradient via
    `inner_mc_grad_fn`'s `state_t_leaves` (privileged-layout leaf projection; FD-checked by
    `test_diffml_spot_grad_fd`). Turnover cost is ignored here (the toy has none) — a
    documented next-increment slot.
    """

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime
        self.cfg = runtime["solver"]
        self.device = bundle["time_grid_days"].device
        acc = runtime["accounting"]
        self.hedges = list(runtime["names"]["hedges"])
        self.n_hedge = len(self.hedges)
        self.contract_size = torch.tensor(
            [float(runtime["tradables"][ref]["contract_size"]) for ref in self.hedges],
            device=self.device)                                       # (n_hedge,)
        limits = acc["position_limits"]
        self.q_lo = torch.tensor(
            [float(limits.get(r, {}).get("min_position", 0.0)) for r in self.hedges],
            device=self.device)
        self.q_hi = torch.tensor(
            [float(limits.get(r, {}).get("max_position", 0.0)) for r in self.hedges],
            device=self.device)
        self.total_abs_limit = float(acc["total_position_abs_limit"])
        # Active hedge axes: which instruments the action grid varies (rest pinned to 0).
        active = self.cfg.get("active_hedge_indices")
        self.active = list(range(self.n_hedge)) if active is None else [int(i) for i in active]
        self.n_active = len(self.active)
        self.levels = int(self.cfg["training_action_grid_levels_per_axis"])
        self.chunk = int(self.cfg["training_action_chunk_size"])
        self.fit_iters = int(self.cfg.get("diffv2_fit_iters", 150))
        self.lr = float(self.cfg.get("diffv2_lr", 2.0e-3))
        self.noise_frac = float(self.cfg.get("diffv2_bank_noise_frac", 0.15))
        self.t_min = int(self.cfg.get("t_min", 0))
        self.use_adv = bool(self.cfg.get("use_advantage_decomp", True))
        self.lambda_diff = float(self.cfg.get("lambda_diff", 0.0))    # market-grad gate (incr 2)
        # Downside-aware action selection (toy: RISK_KAPPA). 0 = plain E[C] argmax (bit-identical).
        self.risk_kappa = float(self.cfg.get("diffv2_risk_kappa", 0.0))
        # History-stripped, sim-grid-indexed views of the outer-realised paths.
        self.tradables_sim, self.n_steps = _bundle_sim_views(bundle)
        hist = int(bundle.get("initial_time_index", 0))
        self.liability_sim = bundle["liability_mtm"][hist:]           # (n_steps, B_outer)
        self.B_outer = int(self.liability_sim.shape[-1])
        # Effective terminal: the LAST sim index is a post-settlement drop when the liability
        # mark there collapses to ~0 (the deal pays out — e.g. the platinum average-rate
        # forward marks its realised payoff at T-1, then 0 at the payment date T). The
        # framework's own `inner["L_T"] = inner_mtm[-2]` encodes exactly this. Telescoping
        # wealth THROUGH that drop cancels the liability's settlement risk (no-hedge W_T≡0),
        # so the meaningful terminal is the last LIVE mark. Decisions run 0..T_dec-1; the
        # terminal continuation marks the liability at T_dec.
        T_raw = self.n_steps - 1
        last = float(self.liability_sim[T_raw].abs().mean())
        prev = float(self.liability_sim[T_raw - 1].abs().mean()) if T_raw >= 1 else last
        self._settled_terminal = last < 1.0e-3 * max(prev, 1.0)
        self.T_dec = T_raw - 1 if self._settled_terminal else T_raw

    # ---- utility anchor ------------------------------------------------------
    def _u(self, W):
        """Bounded terminal-utility anchor u(W) — the framework's normalised utility."""
        return _utility_wrap_signed(W, self.runtime)

    # ---- action grid (subset-active; inactive axes pinned to 0) --------------
    def _action_grid(self):
        axes = []
        for i in range(self.n_hedge):
            if i in self.active:
                axes.append(torch.linspace(float(self.q_lo[i]), float(self.q_hi[i]),
                                            self.levels, device=self.device))
            else:
                axes.append(torch.zeros(1, device=self.device))
        mesh = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)      # (K, n_hedge)
        if self.total_abs_limit > 0.0:                                # drop infeasible rows
            grid = grid[grid.abs().sum(-1) <= self.total_abs_limit + 1e-9]
        return grid

    # ---- input standardization ----------------------------------------------
    def _standardize(self, market, W):
        """Standardized state (market | W) for the residual net — POSITION-FREE (toy-faithful:
        with no turnover cost the held position is a freely-reset control, not a state; it
        enters the value only through next-step wealth W1). Market/wealth use bank mean/std."""
        m = (market - self.m_mean) / self.m_std
        wn = ((W - self.w_mean) / self.w_std).unsqueeze(-1)
        return torch.cat([m, wn], dim=-1)

    def _continuation(self, nets, market, W, t, chunk=400_000):
        """C_t = u(W) + A_t(market, W); terminal C_{T_dec} = u(W). Row-chunked net eval."""
        base = self._u(W)
        if t >= self.T_dec:
            return base
        x = self._standardize(market, W)
        if x.shape[0] <= chunk:
            return base + nets[t](x)
        out = torch.empty_like(base)
        for i in range(0, x.shape[0], chunk):
            out[i:i + chunk] = nets[t](x[i:i + chunk])
        return base + out

    # ---- one-step wealth move ------------------------------------------------
    def _wealth_step(self, W, q, dF, dL):
        """W_{t+1} = W + Σ_i q_i·cs_i·dF_i + dL. q (...,n_hedge); dF (...,n_hedge); dL (...)."""
        return W + (q * self.contract_size * dF).sum(dim=-1) + dL

    # ---- inner-MC one-step quantities at outer t -----------------------------
    def _inner_step(self, t):
        """Fork inner MC at t (resolve_structure / resolve_hedge_structure under the hood)
        and return the one-step move tensors the bootstrap needs:
          dF   (B_outer, B_inner, n_hedge)  per-instrument futures move t→t+1
          dL   (B_outer, B_inner)           liability mark change t→t+1
          m1   (B_outer, B_inner, market_dim) market state at t+1
        plus the bank-state market at t (B_outer, market_dim)."""
        inner = self.bundle["inner_mc_fn"](t)
        F_t = torch.stack([self.tradables_sim[ref][t] for ref in self.hedges], dim=-1)   # (B_outer, n_hedge)
        F_t1 = torch.stack([inner["F_t1"][ref] for ref in self.hedges], dim=-1)          # (B_outer, B_inner, n_hedge)
        # EXPIRED-CONTRACT GUARD: the framework returns inner F_t1=0 for a tradable that has
        # expired before the fork's t+1, while the OUTER tradables_sim FREEZES at the last
        # traded price. So a naive F_t1−F_t mints a spurious ~−F_t "move" on a dead contract
        # — shorting it would mine fake P&L, which is exactly what drove the corner-saturation
        # and value inflation. A dead contract can't be traded ⇒ its one-step move is 0
        # (matching the outer's frozen convention). `live_i` = the contract still prices at t+1.
        live = (F_t1.abs().amax(dim=(0, 1)) > 0).to(F_t1.dtype)                          # (n_hedge,)
        dF = (F_t1 - F_t.unsqueeze(1)) * live
        dL = inner["L_t1"] - inner["L_t"]                                                # (B_outer, B_inner)
        return dF, dL, inner["market_t1"], inner["market_t"], live

    # ---- external argmax (Bellman max outside the fitted value) --------------
    def _decide(self, nets, market_t1, dF, dL, W, t, grid):
        """Pick the grid action maximising E_inner[C_{t+1}] per outer path. No grad."""
        with torch.no_grad():
            B, Bi, md = market_t1.shape
            best_val = None
            best_q = None
            for s in range(0, grid.shape[0], self.chunk):
                acts = grid[s:s + self.chunk]                                            # (c, n_hedge)
                c = acts.shape[0]
                q = acts[None, :, None, :]                                               # (1,c,1,n_hedge)
                W1 = self._wealth_step(W[:, None, None], q, dF[:, None], dL[:, None])     # (B,c,Bi)
                C1f = self._continuation(
                    nets,
                    market_t1[:, None].expand(B, c, Bi, md).reshape(-1, md),
                    W1.reshape(-1), t + 1).reshape(B, c, Bi)                              # (B,c,Bi)
                C1 = C1f.mean(-1)                                                         # E_inner[C] (B,c)
                if self.risk_kappa > 0.0:        # downside-aware: penalise per-action downside dispersion
                    dev = (C1f - C1.unsqueeze(-1)).clamp(max=0.0)                         # negatives only
                    C1 = C1 - self.risk_kappa * (dev ** 2).mean(-1).sqrt()
                cval, carg = C1.max(dim=1)
                cact = acts[carg]                                                        # (B,n_hedge)
                if best_val is None:
                    best_val, best_q = cval, cact
                else:
                    upd = cval > best_val
                    best_val = torch.where(upd, cval, best_val)
                    best_q = torch.where(upd.unsqueeze(-1), cact, best_q)
            return best_q, best_val

    # ---- operating-region bank ----------------------------------------------
    def _replication_hedge(self, t):
        """Per-instrument diagonal min-var hedge from the OUTER one-step moves at t:
        q_i = −Cov(dL, dF_i)/(cs_i·Var(dF_i)) / n_active, clamped to [lo,hi]. Inactive → 0."""
        L = self.liability_sim
        dL = (L[t + 1] - L[t])                                                            # (B_outer,)
        q = torch.zeros(self.n_hedge, device=self.device)
        for i in self.active:
            ref = self.hedges[i]
            dF = self.tradables_sim[ref][t + 1] - self.tradables_sim[ref][t]              # (B_outer,)
            var = dF.var()
            if float(var) <= 1e-12:
                continue
            beta = ((dL - dL.mean()) * (dF - dF.mean())).mean() / var
            q[i] = (-beta / (self.contract_size[i] * max(self.n_active, 1)))
        return torch.minimum(torch.maximum(q, self.q_lo), self.q_hi)

    def _build_bank(self, gen):
        """Operating-region bank — roll the OUTER paths forward (cheap; outer-realised
        moves only, no inner MC): hold q = clamp(q_rep_t + noise) each step, accumulate
        W = cum hedge P&L + L_t. Returns per-t lists W_t (B_outer,) and q_prev (B_outer,
        n_hedge). The bank-state `market_t` is read lazily from the SAME inner-MC fork the
        backward sweep makes at t (no extra inner-MC passes)."""
        L = self.liability_sim
        W = L[0].clone()                                                                 # (B_outer,) = cum_pnl(0)+L_0
        q_prev = torch.zeros(self.B_outer, self.n_hedge, device=self.device)
        W_list, q_list = [], []
        rng = (self.q_hi - self.q_lo)
        mask = torch.zeros(self.n_hedge, device=self.device)
        mask[self.active] = 1.0
        for t in range(self.T_dec):
            W_list.append(W.clone())
            q_list.append(q_prev.clone())
            q_rep = self._replication_hedge(t)                                            # (n_hedge,)
            noise = self.noise_frac * rng * torch.randn(
                self.B_outer, self.n_hedge, generator=gen, device=self.device)
            q = torch.minimum(torch.maximum(q_rep[None] + noise * mask, self.q_lo), self.q_hi)
            dF = torch.stack(
                [self.tradables_sim[ref][t + 1] - self.tradables_sim[ref][t] for ref in self.hedges],
                dim=-1)                                                                  # (B_outer, n_hedge)
            W = self._wealth_step(W, q, dF, L[t + 1] - L[t])
            q_prev = q
        return W_list, q_list

    # ---- project per-process state-at-t leaf grads → market_t columns --------
    def _project_leaf_grads(self, leaf_grads, widths, rows, n, md):
        """Map ∂Y/∂(state_t leaf) for each simulated factor into the `(n, md)` gradient w.r.t.
        the privileged market_t columns the value net consumes. `widths` is in market-column
        order (factor iteration order); a regime-switching spot occupies [belief(width-1),
        price(1)] (calc `_extract_outer_state_at` privileged layout: belief-first, price-last),
        its belief columns supervised by the `(key,'regime_belief')` belief leaf and its price
        column by the raw price leaf. Other factors map 1:1 (raw == privileged). Unmeasured /
        unconnected leaves leave their columns at 0 (masked — only the value supervises them)."""
        g = torch.zeros(n, md, device=self.device)
        col = 0
        for key, width in widths:
            if width <= 0:                                      # privileged-empty factor (not in market_t)
                continue
            gb = leaf_grads.get((key, 'regime_belief'))
            gr = leaf_grads.get(key)
            if gb is not None and gb[..., rows].numel() == (width - 1) * n:   # spot: belief + price
                nb = width - 1
                g[:, col:col + nb] = gb[..., rows].reshape(nb, n).transpose(0, 1)
                if gr is not None and gr[..., rows].numel() == n:
                    g[:, col + nb] = gr[..., rows].reshape(-1)
            elif gr is not None and gr[..., rows].numel() == width * n:        # 1:1 raw → privileged
                g[:, col:col + width] = gr[..., rows].reshape(width, n).transpose(0, 1)
            # else: leaf shape doesn't match the privileged block → leave 0 (masked, value-only)
            col += width
        return g

    # ---- one backward step: bootstrap + advantage twin fit -------------------
    def _fit_step(self, nets, W_bank, t, inner, grid, rows=slice(None)):
        dF_ng, dL_ng, m1_ng, market0, live = inner                                       # no-grad cache
        market0 = market0[rows]
        W0_bank = W_bank[t][rows]
        # SELECT the action on the NO-GRAD inner draws; EVALUATE its value + pathwise gradients
        # on a fresh GRAD inner (independent draws → cross-fit, no winner's-curse max-bias).
        q_star, _ = self._decide(nets, m1_ng[rows], dF_ng[rows], dL_ng[rows], W0_bank, t, grid)
        q_star = q_star * live          # expired contracts: dF=0 ⇒ wealth-neutral; report 0, not the tie

        # GRAD inner-MC fork: AAD-live one-step F_t1/L_t1/market_t1 + per-process state-at-t
        # LEAVES. Bootstrap value Y AND its pathwise gradients w.r.t. W0 (wealth) and the
        # market state (spot/belief) come from the SAME forward — the full Huge–Savine twin
        # loss. ∂Y/∂market_t is the differential constraint that regularizes the market
        # dimension (where a value-only / W-only fit overfits the few outer paths).
        ig = self.bundle["inner_mc_grad_fn"](t)
        leaves, widths = ig["state_t_leaves"], ig["state_t_leaf_widths"]
        if not getattr(self, "_proj_checked", False):
            self._proj_checked = True                  # one-time self-check of the label projection
            mt, col, errs = ig["market_t"].detach(), 0, []      # detach: numeric self-check only
            for key, width in widths:
                if width <= 0:
                    continue
                bl, pl = leaves.get((key, "regime_belief")), leaves.get(key)
                bl = bl.detach() if bl is not None else None
                pl = pl.detach() if pl is not None else None
                if bl is not None:
                    nb = width - 1
                    be = float((mt[:, col:col + nb] - bl.reshape(nb, -1).transpose(0, 1)).abs().max())
                    pe = float((mt[:, col + nb] - pl.reshape(-1)).abs().max()) if pl is not None else -1.0
                    errs.append(f"{utils.check_tuple_name(key)}[belief={be:.1g},price={pe:.1g}]")
                elif pl is not None:
                    e = float((mt[:, col:col + width] - pl.reshape(width, -1).transpose(0, 1)).abs().max())
                    errs.append(f"{utils.check_tuple_name(key)}[1:1={e:.1g}]")
                col += width
            logging.info("DiffSolverV2 differential-label projection check (privileged market_t "
                         "cols vs state_t leaves; ≈0 ⇒ ∂Y/∂market_col == ∂Y/∂leaf): %s",
                         " ".join(errs))
        F_t = torch.stack([self.tradables_sim[r][t] for r in self.hedges], dim=-1)        # (B_outer,n_hedge)
        F_t1 = torch.stack([ig["F_t1"][r] for r in self.hedges], dim=-1) * live           # AAD-live
        dF_g = (F_t1 - F_t.unsqueeze(1))[rows]
        dL_g = (ig["L_t1"] - ig["L_t"])[rows]
        m1_g = ig["market_t1"][rows]
        W0 = W0_bank.clone().requires_grad_(True)
        q = q_star[:, None, :]                                                           # (B,1,n_hedge)
        W1 = self._wealth_step(W0[:, None], q, dF_g, dL_g)                                # (B,Bi)
        B, Bi_e, md = m1_g.shape
        Y = self._continuation(
            nets, m1_g.reshape(-1, md), W1.reshape(-1), t + 1).reshape(B, Bi_e).mean(1)   # (B,)
        grads = torch.autograd.grad(Y.sum(), [W0] + list(leaves.values()), allow_unused=True)
        gW = grads[0].detach()
        leaf_grads = {k: (g.detach() if g is not None else None)
                      for k, g in zip(leaves.keys(), grads[1:])}
        g_market = self._project_leaf_grads(leaf_grads, widths, rows, B, md)             # ∂Y/∂market_t (B,md)
        Y = Y.detach()

        # Advantage decomposition: fit A = C − u(W0); subtract the anchor's wealth slope.
        if self.use_adv:
            Wb = W0_bank.clone().requires_grad_(True)
            (dB_dW,) = torch.autograd.grad(self._u(Wb).sum(), Wb)
            a_val = Y - self._u(W0_bank)
            a_gW = gW - dB_dW.detach()
        else:
            a_val, a_gW = Y, gW

        net = nets[t]
        # Twin loss in STANDARDIZED space (g_zn = std·g_raw). Raw-space matching is mis-scaled:
        # ∂A/∂W ~ 1e-6 in dollars, ∂A/∂spot ~ 1e-4 — both inert against the O(0.1) value term.
        # Standardized, ∂A/∂wn and ∂A/∂mn are O(1) and the gradient match actually regularizes
        # — the principled regularizer of differential ML (NOT weight decay).
        lam_g = float(self.cfg.get("diffv2_lambda_grad", 1.0))
        g_zn_W = self.w_std * a_gW                                                       # (B,)
        g_zn_m = self.m_std * g_market                                                   # (B,md); u indep of market
        # Huge–Savine term BALANCING: with W~$1e6 and utility~O(1) the standardized W-gradient
        # label is ~600× the value label, so an unnormalized sum lets the W-gradient drown the
        # value fit AND the market gradient. Normalize each term by its label variance so all
        # are O(1) and lam_g balances value-vs-gradient as intended.
        nrm_v = a_val.var() + 1e-8
        nrm_w = g_zn_W.var() + 1e-8
        nrm_m = g_zn_m.var() + 1e-8
        opt = torch.optim.Adam(net.parameters(), lr=self.lr,
                               weight_decay=float(self.cfg.get("diffv2_weight_decay", 0.0)))
        for _ in range(self.fit_iters):
            mn = ((market0 - self.m_mean) / self.m_std).detach().requires_grad_(True)
            wn = ((W0_bank - self.w_mean) / self.w_std).detach().requires_grad_(True)
            a = net(torch.cat([mn, wn.unsqueeze(-1)], dim=-1))
            da_m, da_w = torch.autograd.grad(a.sum(), [mn, wn], create_graph=True)
            loss = (((a - a_val) ** 2).mean() / nrm_v
                    + lam_g * ((da_w - g_zn_W) ** 2).mean() / nrm_w
                    + lam_g * ((da_m - g_zn_m) ** 2).mean() / nrm_m)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            a_fit = net(self._standardize(market0, W0_bank))
            val_loss = float(((a_fit - a_val) ** 2).mean())
        return {
            "t": t, "val_loss": val_loss,
            "Y_absmean": float(Y.abs().mean()),
            "A_absmean": float(a_fit.abs().mean()),
            "q_star_mean": q_star.mean(0).detach().cpu().tolist(),
            "Y_mean": float(Y.mean()),
        }

    # ---- greedy-rollout downside verdict -------------------------------------
    def _verdict(self, nets, inner_cache, grid, sweep_ts, rows=slice(None)):
        """Roll the fitted argmax policy forward over [t_min, T_dec] on the OUTER paths in
        `rows` (wealth advanced by the outer-realised dF/dL), starting FLAT at t_min, and
        compare terminal-wealth downside against a textbook diagonal-min-var delta hedge and
        no hedge. The argmax uses the cached inner-MC E[C_{t+1}] estimate; the realised
        outcome uses the outer path. Pass held-out `rows` for an honest OUT-OF-SAMPLE verdict.

        Returns per-policy {u_mean (the objective), wT_mean, wT_p5, wT_cvar5}. The verdict:
        the greedy policy should DOMINATE no-hedge on downside and be competitive with
        textbook — a SPECULATING policy (the old solver's failure) shows up as worse p5/CVaR
        than textbook and a wide wT spread."""
        L = self.liability_sim
        t0 = self.t_min
        n = L[t0][rows].shape[0]
        W = {p: L[t0][rows].clone() for p in ("greedy", "textbook", "nohedge")}
        q_traj = {"greedy": [], "textbook": []}                                          # mean |q| per step
        with torch.no_grad():
            for t in sweep_ts:
                dF_o = torch.stack(
                    [self.tradables_sim[r][t + 1][rows] - self.tradables_sim[r][t][rows]
                     for r in self.hedges], dim=-1)                                       # (n, n_hedge)
                dL_o = (L[t + 1] - L[t])[rows]
                dF, dL, m1, _, live = inner_cache[t]
                q_g, _ = self._decide(nets, m1[rows], dF[rows], dL[rows], W["greedy"], t, grid)
                q_g = q_g * live          # zero positions on expired contracts (wealth-neutral)
                q_tb = self._replication_hedge(t)[None].expand(n, self.n_hedge)
                z = torch.zeros(n, self.n_hedge, device=self.device)
                W["greedy"] = self._wealth_step(W["greedy"], q_g, dF_o, dL_o)
                W["textbook"] = self._wealth_step(W["textbook"], q_tb, dF_o, dL_o)
                W["nohedge"] = self._wealth_step(W["nohedge"], z, dF_o, dL_o)
                q_traj["greedy"].append(q_g.mean(0).tolist())
                q_traj["textbook"].append(q_tb.mean(0).tolist())

        def stats(wT):
            p5 = torch.quantile(wT, 0.05)
            cvar5 = wT[wT <= p5].mean() if (wT <= p5).any() else p5
            return {"u_mean": float(self._u(wT).mean()), "wT_mean": float(wT.mean()),
                    "wT_p5": float(p5), "wT_cvar5": float(cvar5)}
        out = {p: stats(W[p]) for p in W}
        # greedy position summary: mean over the rollout of |q| per instrument (is it hedging?)
        gq = torch.tensor(q_traj["greedy"])                                              # (n_steps, n_hedge)
        out["greedy_mean_abs_q"] = gq.abs().mean(0).tolist()
        out["greedy_q_first"] = q_traj["greedy"][0] if q_traj["greedy"] else None
        out["greedy_q_mid"] = (q_traj["greedy"][len(q_traj["greedy"]) // 2]
                               if q_traj["greedy"] else None)
        return out

    # ---- driver --------------------------------------------------------------
    def solve(self):
        # Bank RNG is deterministic; multi-seed repeats (solve_hedge calls solve() N times)
        # advance the framework's inner-MC Sobol stream, so V_0 spread reflects inner-MC noise.
        gen = torch.Generator(device=self.device)
        gen.manual_seed(0)
        logging.info(
            "DiffSolverV2 setup: n_hedge=%d active=%s T_dec=%d (of %d sim steps; "
            "settled_terminal=%s) B_outer=%d levels=%d fit_iters=%d lr=%.3g | "
            "contract_size=%s | q∈[%s, %s] total_abs_limit=%.3g",
            self.n_hedge, self.active, self.T_dec, self.n_steps, self._settled_terminal,
            self.B_outer, self.levels, self.fit_iters, self.lr, self.contract_size.tolist(),
            self.q_lo.tolist(), self.q_hi.tolist(), self.total_abs_limit)

        W_bank, _q_bank = self._build_bank(gen)
        # Cache the framework inner-MC one-step quantities over the swept range — one
        # inner-MC fork per swept t, reused for the argmax, the bootstrap, AND market_t.
        sweep_ts = list(range(self.t_min, self.T_dec))
        inner_cache = {t: self._inner_step(t) for t in sweep_ts}

        # OUT-OF-SAMPLE split: hold out a fraction of outer paths from the bank/fit; the
        # verdict's headline rolls the policy on the HELD-OUT paths so a +EV that's just
        # overfitting to the fitted paths is exposed (in-sample is reported alongside).
        oos_frac = float(self.cfg.get("diffv2_oos_frac", 0.5))
        n_tr = self.B_outer if oos_frac <= 0 else max(1, int(self.B_outer * (1.0 - oos_frac)))
        train, test = slice(0, n_tr), slice(n_tr, self.B_outer)
        has_oos = n_tr < self.B_outer

        # Standardization stats: market/wealth from the TRAIN swept states (no test peeking).
        M = torch.cat([inner_cache[t][3][train] for t in sweep_ts], 0)                    # (n_swept*n_tr, md)
        self.m_mean, self.m_std = M.mean(0), M.std(0).clamp_min(1e-6)
        Wall = torch.stack([W_bank[t][train] for t in sweep_ts], 0).reshape(-1)
        self.w_mean, self.w_std = Wall.mean(), Wall.std().clamp_min(1e-6)
        md = M.shape[-1]
        logging.info(
            "DiffSolverV2 bank: market_dim=%d | swept W∈[%.4g, %.4g] mean=%.4g std=%.4g | "
            "q_rep(t=0)=%s", md, float(Wall.min()), float(Wall.max()),
            float(self.w_mean), float(self.w_std),
            self._replication_hedge(0).detach().cpu().tolist())

        grid = self._action_grid()
        hidden = int(self.cfg.get("diffv2_hidden", 128))
        load_path = str(self.cfg.get("diffv2_load_value_fn", "") or "")
        loaded = None
        if load_path:
            # Frozen-policy eval: restore the fitted nets AND the function's frame — the
            # train-time standardization stats and utility scale are part of the value
            # function; recomputing them from the (possibly stressed) eval world would
            # silently change what the nets compute. EVERY eval path is unseen by the nets,
            # so the verdict rolls over all paths.
            loaded = torch.load(load_path, map_location=self.device)
            for key, want in (("t_min", self.t_min), ("T_dec", self.T_dec),
                              ("md", md), ("hedges", list(self.hedges))):
                if loaded[key] != want:
                    raise ValueError(
                        f"DiffV2_Load_Value_Fn checkpoint mismatch on {key!r}: "
                        f"saved {loaded[key]!r} vs this run {want!r}")
            # OOD drift diagnostic: how far the eval world's market states sit from the
            # train-world standardization frame (in train-σ units, per-dim max).
            drift = ((M.mean(0) - loaded["m_mean"]).abs() / loaded["m_std"]).max()
            logging.info(
                "DiffSolverV2 LOADED value fn from %s (train V_0=%+.6g) | eval-world market "
                "drift vs train frame: max %.3g σ | utility_scale restored to %.6g",
                load_path, loaded["V_0"], float(drift), loaded["utility_scale"])
            self.m_mean, self.m_std = loaded["m_mean"], loaded["m_std"]
            self.w_mean, self.w_std = loaded["w_mean"], loaded["w_std"]
            self.runtime["objective"]["utility_scale"] = float(loaded["utility_scale"])
            hidden = int(loaded["hidden"])
        nets = [_DiffV2Residual(md + 1, hidden=hidden).to(self.device)  # position-free: (market | W)
                for _ in range(self.T_dec)]
        logging.info("DiffSolverV2 action grid: K=%d actions (levels=%d ^ active=%d)",
                     int(grid.shape[0]), self.levels, self.n_active)

        rows = []
        if loaded is not None:
            for net, sd in zip(nets, loaded["state_dicts"]):
                net.load_state_dict(sd)
                net.eval()
            worst = float(loaded["max_abs_Y_boot"])
            root = {"t": self.t_min, "Y_mean": float(loaded["V_0"]),
                    "q_star_mean": list(loaded["n_star_0"])}
        else:
            for t in reversed(sweep_ts):
                r = self._fit_step(nets, W_bank, t, inner_cache[t], grid, rows=train)
                rows.append(r)
                logging.info(
                    "DiffSolverV2 C[t=%d] fitted: val_loss=%.4g |Y_boot|=%.4g |A|=%.4g "
                    "Y_mean=%+.4g q*_mean=%s", r["t"], r["val_loss"], r["Y_absmean"],
                    r["A_absmean"], r["Y_mean"],
                    ["%.3f" % v for v in r["q_star_mean"]])
            worst = max((r["Y_absmean"] for r in rows), default=0.0)
            root = rows[-1] if rows else {"t": self.t_min, "Y_mean": 0.0, "q_star_mean":
                                          [0.0] * self.n_hedge}
        V_0 = float(root["Y_mean"])
        n_star_0 = root["q_star_mean"]
        bounded = worst < 1.0e4
        if loaded is None:
            logging.info(
                "DiffSolverV2 sweep complete: t=%d→%d | max|Y_boot|=%.4g (%s) | "
                "V_0=%+.6g | n_star@t=%d=%s", self.T_dec - 1, self.t_min, worst,
                "BOUNDED" if bounded else "EXPLODED", V_0, root["t"], n_star_0)
        save_path = str(self.cfg.get("diffv2_save_value_fn", "") or "")
        if save_path and loaded is None:
            torch.save({
                "state_dicts": [net.state_dict() for net in nets],
                "m_mean": self.m_mean, "m_std": self.m_std,
                "w_mean": self.w_mean, "w_std": self.w_std,
                "utility_scale": float(self.runtime["objective"]["utility_scale"]),
                "t_min": self.t_min, "T_dec": self.T_dec, "md": md, "hidden": hidden,
                "hedges": list(self.hedges),
                "V_0": V_0, "n_star_0": list(n_star_0), "max_abs_Y_boot": worst,
            }, save_path)
            logging.info("DiffSolverV2 SAVED value fn to %s (V_0=%+.6g)", save_path, V_0)

        # Downside verdict: greedy policy vs textbook delta hedge vs no hedge. HEADLINE is the
        # OUT-OF-SAMPLE rollout (held-out paths the nets never saw); in-sample reported too.
        if loaded is not None:
            # Frozen nets never saw ANY of this run's paths — the whole batch is out-of-sample.
            verdict = verdict_is = self._verdict(nets, inner_cache, grid, sweep_ts,
                                                 rows=slice(None))
            has_oos = True
        else:
            verdict = self._verdict(nets, inner_cache, grid, sweep_ts,
                                    rows=(test if has_oos else train))
            verdict_is = (self._verdict(nets, inner_cache, grid, sweep_ts, rows=train)
                          if has_oos else verdict)
        if has_oos and loaded is None:
            logging.info(
                "DiffSolverV2 IN-SAMPLE vs OOS u(W_T): greedy IS=%+.5f OOS=%+.5f | "
                "textbook IS=%+.5f OOS=%+.5f (gap IS−OOS greedy=%+.5f → overfit if large)",
                verdict_is["greedy"]["u_mean"], verdict["greedy"]["u_mean"],
                verdict_is["textbook"]["u_mean"], verdict["textbook"]["u_mean"],
                verdict_is["greedy"]["u_mean"] - verdict["greedy"]["u_mean"])
        g, tb, nh = verdict["greedy"], verdict["textbook"], verdict["nohedge"]
        # PRIMARY metric = the optimization target E[u(W_T)] (already encodes downside aversion
        # via the concave utility). CVaR5 is a secondary tail diagnostic (noisy at small B).
        beats_nh = g["u_mean"] >= nh["u_mean"]
        beats_tb = g["u_mean"] >= tb["u_mean"]
        tail_vs_tb = g["wT_cvar5"] >= tb["wT_cvar5"] - abs(tb["wT_cvar5"]) * 0.05
        logging.info(
            "DiffSolverV2 VERDICT (%s rollout t=%d→T over %d outer paths, start flat):\n"
            "  policy    u(W_T)mean    W_T mean       W_T p5         W_T CVaR5\n"
            "  greedy    %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  textbook  %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  nohedge   %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  → on the OBJECTIVE E[u(W_T)]: beats no-hedge=%s, beats textbook=%s | "
            "tail(CVaR5) competitive w/ textbook=%s",
            "OUT-OF-SAMPLE" if has_oos else "in-sample", self.t_min,
            self.B_outer if loaded is not None else (
                self.B_outer - n_tr if has_oos else self.B_outer),
            g["u_mean"], g["wT_mean"], g["wT_p5"], g["wT_cvar5"],
            tb["u_mean"], tb["wT_mean"], tb["wT_p5"], tb["wT_cvar5"],
            nh["u_mean"], nh["wT_mean"], nh["wT_p5"], nh["wT_cvar5"],
            beats_nh, beats_tb, tail_vs_tb)
        logging.info(
            "DiffSolverV2 greedy positions: mean|q| per instrument=%s | q@t0=%s | q@mid=%s "
            "(textbook q@t0=%s)", ["%.2f" % v for v in verdict["greedy_mean_abs_q"]],
            verdict.get("greedy_q_first"), verdict.get("greedy_q_mid"),
            self._replication_hedge(self.t_min).detach().cpu().tolist())

        return SolverResult(
            solver_name="DiffSolverV2",
            actions=torch.tensor(n_star_0),
            values=V_0,
            diagnostics={
                "V_0": V_0,
                "n_star_0": n_star_0,
                "n_star": n_star_0,
                "max_abs_Y_boot": worst,
                "bounded": bool(bounded),
                "root_t": int(root["t"]),
                "per_t": rows,
                "action_grid_size": int(grid.shape[0]),
                "market_dim": int(M.shape[-1]),
                "verdict": verdict,                       # OUT-OF-SAMPLE (held-out paths)
                "verdict_in_sample": verdict_is,
                "verdict_is_oos": bool(has_oos),
                "verdict_beats_nohedge_on_utility": bool(beats_nh),
                "verdict_beats_textbook_on_utility": bool(beats_tb),
                "verdict_tail_competitive_vs_textbook": bool(tail_vs_tb),
            },
        )


# The differential-ML solver `DiffSolverV2` is the production deliverable; the
# clairvoyant `HindsightDpSolver` is kept as the upper-bound (oracle) benchmark
# track. `run_textbook_benchmark` supplies the lower-bound (min-var / averaging)
# track. The legacy DP/MPC/DifferentialSolverOld stack was removed.
_SOLVERS: Dict[str, Callable] = {
    "hindsightdpsolver": HindsightDpSolver,
    "diffsolverv2": DiffSolverV2,
}


def _acceptance_ladder(comparison):
    """The acceptance ordering — hindsight ≥ DiffSolverV2 ≥ textbook — over whatever
    tracks are present. `holds` allows a tiny tolerance for Monte-Carlo noise."""
    order = [("HindsightDpSolver", "hindsight"), ("DiffSolverV2", "DiffSolverV2"),
             ("textbook", "textbook")]
    rungs = [(label, comparison[key]["v0_mean"])
             for key, label in order if key in comparison]
    holds = all(rungs[i][1] >= rungs[i + 1][1] - 1.0e-6
                for i in range(len(rungs) - 1))
    return {"order": rungs, "holds": holds}


def solve_hedge(bundle, runtime):
    """Dispatcher + orchestration for `Execution_Mode='solve_hedge'`. Runs the configured
    `Solver.Object`; when that is the `DiffSolverV2` deliverable it also assembles the
    benchmark tracks (hindsight upper bound / textbook lower bound) enabled by the `Run_*`
    flags into a `comparison` table — V_0 mean ± std per track — plus the acceptance ladder.
    Multi-seed repeats re-use the cached outer paths but advance the inner-MC Sobol stream.

    Returns the dict shape `HedgeMonteCarlo.execute` unpacks (`evaluation_output` /
    `optimizer_diagnostics` / `policy_artifact`)."""
    _mirror_utility_scale_to_runtime(bundle, runtime)
    solver_cfg = runtime["solver"]
    obj = solver_cfg["object"]
    if obj not in _SOLVERS:
        raise ValueError(
            f"Unknown Solver Object {obj!r}; available: {sorted(_SOLVERS)}")
    n_seed = max(1, int(solver_cfg.get("multi_seed_count", 1)))
    have_liability = bundle.get("liability_mtm") is not None

    # Primary solver — multi-seed repeats advance the inner-MC Sobol stream.
    primary_runs = [_SOLVERS[obj](bundle, runtime).solve() for _ in range(n_seed)]
    primary = primary_runs[0]
    comparison = {primary.solver_name: _multiseed_summary(primary_runs)}

    # Benchmark tracks — assembled alongside the DiffSolverV2 deliverable.
    if obj == "diffsolverv2":
        for flag, label in (("run_hindsight_diagnostic", "hindsight"),
                             ("run_textbook_benchmark", "textbook")):
            if solver_cfg.get(flag) and not have_liability:
                logging.warning("solve_hedge: %s requested but bundle has no "
                                 "liability_mtm — track skipped", label)
        if solver_cfg.get("run_hindsight_diagnostic") and have_liability:
            comparison["HindsightDpSolver"] = _multiseed_summary(
                [HindsightDpSolver(bundle, runtime).solve()])     # deterministic — one run
        if solver_cfg.get("run_textbook_benchmark") and have_liability:
            comparison["textbook"] = run_textbook_benchmark(bundle, runtime)

    ladder = _acceptance_ladder(comparison)
    logging.info("solve_hedge tracks: %s | ladder holds=%s",
                 {k: round(v["v0_mean"], 4) for k, v in comparison.items()},
                 ladder["holds"])
    return {
        "policy": None,
        "evaluation_output": {
            "solver_name": primary.solver_name,
            "solver_result": primary,
            "actions": primary.actions,
            "values": primary.values,
            "diagnostics": primary.diagnostics,
            "comparison": comparison,
            "ladder": ladder,
        },
        "optimizer_diagnostics": {**primary.diagnostics,
                                  "comparison": comparison, "ladder": ladder},
        "policy_artifact": primary.value_fn_artifacts,
    }
