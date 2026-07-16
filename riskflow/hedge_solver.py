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

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from . import utils
from .hedge_bundle import (
    _utility_wrap_signed, _mirror_utility_scale_to_runtime, wealth_step,
)
from .hedge_runtime import per_contract_kappa, initial_q_from_runtime

# Bumped whenever the fitted-value-function on-disk/artifact contract changes shape. Stamped
# into every artifact so a loader can tell which solver produced a checkpoint.
SOLVER_VERSION = "diffsolverv2/2026-07"


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


def _turnover_cost(delta_contracts, kappa):
    """L1 turnover cost: Σ_i |Δcontracts_i| · kappa_i. `delta_contracts` is
    `(..., n_hedge)`, `kappa` is `(n_hedge,)` per-contract cost. Hard (no smoothing) —
    action search has no gradients on the action."""
    return (delta_contracts.abs() * kappa).sum(dim=-1)


def _config_hash(solver_cfg):
    """sha1 of the stable-JSON normalized solver cfg — a stamp identifying the TRAINING
    RECIPE. The persistence paths (save/load) are excluded so the same recipe hashes
    identically regardless of where its checkpoint lives."""
    stable = {k: v for k, v in solver_cfg.items()
              if k not in ("diffv2_save_value_fn", "diffv2_load_value_fn")}
    return hashlib.sha1(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


def _schedule_key(schedule):
    """Canonical, comparison-stable form of a `Total_Position_Schedule` (None or the sorted
    `(step, min, max)` knot tuple) — flattens tuple-vs-list and int/float drift so a checkpoint's
    stored corridor compares bitwise against the run's corridor regardless of round-trip form."""
    return None if schedule is None else tuple(
        (int(step), float(lo), float(hi)) for step, lo, hi in schedule)


class HedgeActionSpace:
    """The solver-owned ACTION UNIVERSE + friction access, built ONCE from (runtime, device)
    and shared by every track (DiffSolverV2, HindsightDpSolver, run_textbook_benchmark, and
    the stepper rollout) so they optimize over exactly the same positions and price the same
    frictions. One object replaces the old scattered `build_action_grid` (varied ALL hedges,
    letting the benchmarks trade axes an `Active_Hedge_Indices` run had pinned off) + the
    private `DiffSolverV2._action_grid` + the free `_per_contract_kappa`/`_axis_levels`.

    Surface:
      * `axis_levels()` / `grid()` — the MASK-AWARE target-position grid: the active hedge
        axes span `[min, max]` at `levels` points, inactive axes are pinned to a single 0,
        and rows over the total-position cap are dropped (identical to the old private grid).
      * `kappa(tradables_sim, t)` — the per-hedge turnover kappa at sim-grid `t` (each
        instrument's mean mark), off the single `hedge_runtime.per_contract_kappa` rule.
      * `initial_q(batch, device)` — the opening book `q0` (see `initial_q_from_runtime`)."""

    def __init__(self, runtime, device):
        self.runtime = runtime
        self.device = device
        self.hedges = list(runtime["names"]["hedges"])
        self.n_hedge = len(self.hedges)
        acc = runtime["accounting"]
        limits = acc["position_limits"]
        self.q_lo = torch.tensor(
            [float(limits.get(r, {}).get("min_position", 0.0)) for r in self.hedges],
            device=device)
        self.q_hi = torch.tensor(
            [float(limits.get(r, {}).get("max_position", 0.0)) for r in self.hedges],
            device=device)
        self.contract_size = torch.tensor(
            [float(runtime["tradables"][r]["contract_size"]) for r in self.hedges],
            device=device)
        self.total_abs_limit = float(acc["total_position_abs_limit"])
        # Optional per-decision-step corridor on the SIGNED total position Σq_i (sorted
        # (step, min_total, max_total) knots or None). `grid_at(t)` filters the base grid to it.
        self.schedule = acc.get("total_position_schedule")
        self._grid_cache = None
        solver_cfg = runtime["solver"]
        self.levels = int(solver_cfg["training_action_grid_levels_per_axis"])
        active = solver_cfg.get("active_hedge_indices")
        self.active = (list(range(self.n_hedge)) if active is None
                       else [int(i) for i in active])
        self.n_active = len(self.active)

    def axis_levels(self):
        """Per-hedge 1-D level values: `linspace(lo, hi, levels)` on ACTIVE axes, a single 0
        on inactive axes (pinned). The product of these is the action grid."""
        return [torch.linspace(float(self.q_lo[i]), float(self.q_hi[i]), self.levels,
                               device=self.device)
                if i in self.active else torch.zeros(1, device=self.device)
                for i in range(self.n_hedge)]

    def grid(self):
        """Mask-aware target-position grid `(n_actions, n_hedge)`: inactive axes pinned to 0,
        rows over the total-position cap dropped. Cached (deterministic) — `grid_at(t)` reslices
        this base grid per decision step without rebuilding the meshgrid."""
        if self._grid_cache is None:
            mesh = torch.meshgrid(*self.axis_levels(), indexing="ij")
            grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
            if self.total_abs_limit > 0.0:
                grid = grid[grid.abs().sum(-1) <= self.total_abs_limit + 1e-9]
            if grid.shape[0] == 0:
                raise ValueError(
                    "action grid empty — Position_Limits infeasible under Total_Position_Abs_Limit")
            self._grid_cache = grid
        return self._grid_cache

    def _corridor_at(self, t):
        """`(min_total, max_total)` at decision step t — the rightmost `Total_Position_Schedule`
        knot with `Step <= t` (piecewise-constant; clamps to the first knot for t before it)."""
        lo, hi = self.schedule[0][1], self.schedule[0][2]
        for step, mn, mx in self.schedule:
            if step > t:
                break
            lo, hi = mn, mx
        return lo, hi

    def grid_at(self, t, live=None):
        """Action grid at decision step t: the base `grid()` filtered to rows whose SIGNED total
        position lies in the `Total_Position_Schedule` corridor at t. No schedule → the base grid
        unchanged (bit-identical to today). The single per-t filter site every track shares
        (DiffSolverV2 argmax, hindsight, textbook). Empty after filtering ⇒ infeasible corridor at
        t, failed loud.

        The corridor bounds the REALIZED signed total — the position that survives expiry masking
        (the argmax callers apply `q = q * live` afterwards). Passing the step-t `live` leg mask
        filters on Σ(q_i·live_i), so a corridor-satisfying short can't be parked on an expired leg
        (a dF=0 wealth-neutral tie) only to be masked to 0 and silently under-hedge. Absent live ⇒
        Σq_i over all legs (entry step / static diagnostics, where nothing has expired)."""
        grid = self.grid()
        if self.schedule is None:
            return grid
        lo, hi = self._corridor_at(t)
        tot = (grid * live).sum(-1) if live is not None else grid.sum(-1)
        grid_t = grid[(tot >= lo - 1e-9) & (tot <= hi + 1e-9)]
        if grid_t.shape[0] == 0:
            raise ValueError(
                f"action grid empty at step {t} — Total_Position_Schedule corridor "
                f"[{lo}, {hi}] infeasible on live legs (grid Σq·live spans "
                f"[{float(tot.min())}, {float(tot.max())}])")
        return grid_t

    def project_to_corridor(self, q, t):
        """Project the SIGNED total of the ACTIVE legs of `q` `(B, n_hedge)` into the
        `Total_Position_Schedule` corridor at decision step t, keeping every leg inside its own
        `[Min,Max]` via headroom water-filling: shift the active legs by the corridor deficit
        `d = clamp(Σq, lo, hi) − Σq`, distributed in proportion to each leg's remaining room in
        d's direction, so `Σ_active` lands exactly on the nearest corridor edge and no leg leaves
        its box. Feasible whenever the corridor is grid-feasible (headroom ≥ |d|, the same
        feasibility `grid_at` fails loud on). No schedule → returns `q` unchanged (bit-identical).

        Unlike `grid_at` (which FILTERS a discrete action universe to the corridor for the argmax),
        this CONTINUOUSLY nudges an exploration/benchmark position — the exploration bank rolls
        continuous q around the min-var hedge, so a filter has nothing to select from. Shared by
        the bank (so the fitted value trains on IN-corridor wealth states, not unreachable ones)
        and the verdict/stepper textbook (a fair in-corridor min-var comparison)."""
        if self.schedule is None:
            return q
        lo, hi = self._corridor_at(t)
        idx = self.active
        qa = q[:, idx]                                                   # (B, n_active)
        tot = qa.sum(-1, keepdim=True)                                   # (B,1) signed active total
        d = tot.clamp(lo, hi) - tot                                     # (B,1) corridor deficit
        head = torch.where(d >= 0, self.q_hi[idx] - qa, qa - self.q_lo[idx]).clamp_min(0.0)
        out = q.clone()
        out[:, idx] = qa + d * head / head.sum(-1, keepdim=True).clamp_min(1e-9)
        return out

    def kappa(self, tradables_sim, t_index):
        """Per-hedge kappa `(n_hedge,)` at sim-grid `t_index` — each instrument's mean mark
        through `hedge_runtime.per_contract_kappa` (the single turnover-cost rule)."""
        return torch.stack(
            [per_contract_kappa(self.runtime, tradables_sim[r][t_index].mean(), r)
             for r in self.hedges])

    def initial_q(self, batch, device):
        """The opening book `q0` `(batch, n_hedge)` (hedge legs only, `names['hedges']`
        order) from the normalized `Portfolio_State` positions."""
        return initial_q_from_runtime(self.runtime, batch, device)


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
    terminal MTM, and `t_outer`. The liability terminal is the bundle's pre-settlement
    `last_live_mtm_index` (index -1 is the appended clean-exit zero)."""
    tradables_sim, t_outer = _bundle_sim_views(bundle)
    hedges = list(runtime["names"]["hedges"])
    F = torch.stack([tradables_sim[h] for h in hedges], dim=0)        # (n_h, t_outer, B)
    hist = int(bundle.get("initial_time_index", 0))
    liab = bundle["liability_mtm"][hist:]                             # (>=t_outer, B)
    L_T = liab[bundle["last_live_mtm_index"]]                         # (B,)
    return F, L_T, t_outer


def run_textbook_benchmark(bundle, runtime):
    """Static-hedge reference: the single best CONSTANT position, held over the whole horizon
    with no rebalancing, evaluated FRICTIONLESS on the realized outer paths (the shared DP
    objective — greedy/hindsight are frictionless too). A valid lower bound for the dynamic
    DP: dynamic rebalancing can only add value. No inner MC, no V̂. Static hold telescopes
    the per-step P&L to `position · (F_T − F_0)`. Uses the shared `HedgeActionSpace` so it
    respects `Active_Hedge_Indices` (inactive axes pinned to 0) exactly like the greedy grid.

    The turnover a real execution of this constant hold would pay — entry from the OPENING
    book `q0` (not from flat) + terminal unwind — is a shared-kappa net-of-cost DIAGNOSTIC
    (`turnover_cost_mean` / `v0_mean_net`), never charged against the V_0 track."""
    F, L_T, t_outer = _realized_paths(bundle, runtime)
    device = F.device
    acc = runtime["accounting"]
    aspace = HedgeActionSpace(runtime, device)
    tradables_sim, _ = _bundle_sim_views(bundle)
    # The static hold is chosen ONCE at entry and held; a per-t corridor is a dynamic constraint a
    # constant hold can't track, so the single well-defined filter is the entry-step corridor
    # grid_at(0) (no schedule ⇒ the base grid, unchanged). Keeps textbook a within-entry-mandate
    # static lower bound and shares the one filter site.
    grid = aspace.grid_at(0)                                           # (n_actions, n_hedge)
    b_outer = F.shape[-1]
    q0 = aspace.initial_q(b_outer, device)                            # (B, n_hedge) opening book

    total_move = F[:, -1, :] - F[:, 0, :]                              # (n_h, B) telescoped
    g_t = torch.einsum("ai,ib->ab", grid * aspace.contract_size, total_move)  # (n_actions, B)
    u = _utility_wrap_signed(L_T.unsqueeze(0) + g_t, runtime)         # FRICTIONLESS objective
    obj = u.mean(dim=-1)                                               # (n_actions,)
    best = int(obj.argmax())
    n_star = grid[best]                                                # (n_hedge,)
    # Net-of-cost diagnostic (shared kappa): entry |n_star − q0| + terminal unwind |0 − n_star|.
    kappa0 = aspace.kappa(tradables_sim, 0)
    cost = _turnover_cost(n_star.unsqueeze(0) - q0, kappa0)            # (B,) entry from q0
    if acc["force_flat_at_end"]:
        kappa_T = aspace.kappa(tradables_sim, t_outer - 1)
        cost = cost + _turnover_cost(n_star, kappa_T)                  # unwind to flat (scalar)
    u_net = _utility_wrap_signed(L_T + g_t[best] - cost, runtime)
    return {"v0_mean": float(obj[best]), "v0_std": 0.0,
            "v0_mean_net": float(u_net.mean()),
            "turnover_cost_mean": float(cost.mean()),
            "n_star": n_star.detach().cpu().tolist(),
            "terminal_utility": u[best].detach().cpu()}


class HindsightDpSolver:
    """Clairvoyant upper-bound diagnostic, FRICTIONLESS (the shared DP objective). For each
    realized outer path it picks, at EVERY step independently, the grid position maximizing
    that step's realized P&L `q·cs·(F_{t+1}−F_t)` — perfect foresight + free repositioning.
    No inner MC, no V̂: the realized path is its own one-sample future.

    `u_signed` is monotone and the liability terminal `L_T(b)` is path-fixed, so maximizing
    `u_signed(W_T)` ≡ maximizing the additive cash `G_T`; with no turnover cost the per-step
    choices decouple, so the max-plus DP collapses to a per-step argmax. `mean_b V_0(b)` is
    an upper bound on any deployable (non-clairvoyant) policy's value — the reference the DP
    is measured against. The turnover a real execution of this bang-bang trajectory would pay
    (entry from the OPENING book, per-step repositioning, terminal unwind) is a shared-kappa
    net-of-cost DIAGNOSTIC, never charged against the V_0 track. Uses the shared
    `HedgeActionSpace`, so it respects `Active_Hedge_Indices` exactly like the greedy grid."""

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime
        self.aspace = HedgeActionSpace(runtime, bundle["time_grid_days"].device)

    def solve(self):
        bundle, runtime, aspace = self.bundle, self.runtime, self.aspace
        acc = runtime["accounting"]

        F, L_T, t_outer = _realized_paths(bundle, runtime)             # F (n_h,t_outer,B)
        device = F.device
        b_outer = F.shape[-1]
        tradables_sim, _ = _bundle_sim_views(bundle)
        cs = aspace.contract_size

        # Frictionless clairvoyant: independent per-step argmax over the realized move, within the
        # per-t action universe `grid_at(t)` (base grid, corridor-filtered when a schedule is set).
        # The net-of-cost trajectory (from the opening book q0) is accumulated for the diagnostic.
        q_prev = aspace.initial_q(b_outer, device)                     # (B, n_h) opening book
        n_star_0 = None
        G = torch.zeros(b_outer, device=device)
        cost = torch.zeros(b_outer, device=device)
        for t in range(t_outer - 1):
            grid = aspace.grid_at(t)                                   # (n_actions_t, n_h) per-t
            dF = F[:, t + 1, :] - F[:, t, :]                           # (n_h, B)
            step_pnl = torch.einsum("ai,ib->ab", grid * cs, dF)        # (n_actions_t, B)
            best_pnl, best_idx = step_pnl.max(dim=0)                   # (B,), (B,)
            q_now = grid[best_idx]                                     # (B, n_h)
            cost = cost + _turnover_cost(q_now - q_prev, aspace.kappa(tradables_sim, t))
            G = G + best_pnl
            q_prev = q_now
            if t == 0:
                n_star_0 = q_now
        if acc["force_flat_at_end"]:
            cost = cost + _turnover_cost(q_prev, aspace.kappa(tradables_sim, t_outer - 1))
        v0 = _utility_wrap_signed(L_T + G, runtime)                    # (B,) FRICTIONLESS
        v0_net = _utility_wrap_signed(L_T + G - cost, runtime)         # (B,) net-of-cost diagnostic

        return SolverResult(
            solver_name="HindsightDpSolver",
            actions=n_star_0.detach().cpu(),
            values=v0.detach().cpu(),
            terminal_pnl=(L_T + G).detach().cpu(),
            terminal_utility=v0.detach().cpu(),
            diagnostics={
                "V_0": float(v0.mean()),
                "V_0_net": float(v0_net.mean()),
                "turnover_cost_mean": float(cost.mean()),
                "n_star_0": n_star_0.float().mean(dim=0).detach().cpu().tolist(),
                "v0_abs_max": float(v0.abs().max()),
                "action_grid_size": int(aspace.grid().shape[0]),
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
        # The shared action universe (mask-aware grid + per-contract kappa + opening book) —
        # the SAME object the benchmark tracks and the stepper rollout consume, so every track
        # optimizes over identical positions and prices identical frictions.
        self.aspace = HedgeActionSpace(runtime, self.device)
        self.hedges = self.aspace.hedges
        self.n_hedge = self.aspace.n_hedge
        self.contract_size = self.aspace.contract_size                # (n_hedge,)
        self.q_lo = self.aspace.q_lo
        self.q_hi = self.aspace.q_hi
        self.total_abs_limit = self.aspace.total_abs_limit
        # Active hedge axes: which instruments the action grid varies (rest pinned to 0).
        self.active = self.aspace.active
        self.n_active = self.aspace.n_active
        self.levels = self.aspace.levels
        self.chunk = int(self.cfg["training_action_chunk_size"])
        self.fit_iters = int(self.cfg.get("diffv2_fit_iters", 150))
        self.lr = float(self.cfg.get("diffv2_lr", 2.0e-3))
        self.noise_frac = float(self.cfg.get("diffv2_bank_noise_frac", 0.15))
        self.t_min = int(self.cfg.get("t_min", 0))
        self.use_adv = bool(self.cfg.get("use_advantage_decomp", True))
        # Downside-aware action selection (toy: RISK_KAPPA). 0 = plain E[C] argmax (bit-identical).
        self.risk_kappa = float(self.cfg.get("diffv2_risk_kappa", 0.0))
        # Cost-aware EXECUTION: the verdict rollout charges κ·|q − q_prev| at the argmax
        # (hysteresis); training/selection stay cost-free. Default off = bit-identical.
        self.cost_aware = bool(self.cfg.get("diffv2_cost_aware_argmax", False))
        # One-step forks: window inner generation+pricing to {t, t+1} (the bootstrap only
        # reads t/t+1 fields) — fork cost stops scaling with the remaining horizon.
        self.one_step = bool(self.cfg.get("diffv2_one_step_fork", True))
        # History-stripped, sim-grid-indexed views of the outer-realised paths.
        self.tradables_sim, self.n_steps = _bundle_sim_views(bundle)
        hist = int(bundle.get("initial_time_index", 0))
        self.liability_sim = bundle["liability_mtm"][hist:]           # (n_steps, B_outer)
        self.B_outer = int(self.liability_sim.shape[-1])
        # Effective terminal = the last LIVE liability mark. The time grid appends one
        # post-settlement clean-exit row (the deal pays out — the platinum average-rate forward
        # marks its realised payoff at T-1, then 0 at the payment date T), so the meaningful
        # terminal is the bundle's `last_live_mtm_index` (the structural pre-settlement `[-2]`).
        # Telescoping wealth THROUGH the settlement drop cancels the liability's settlement risk
        # (no-hedge W_T≡0). Decisions run 0..T_dec-1; the terminal continuation marks at T_dec.
        self.T_dec = int(bundle["last_live_mtm_index"])
        if self.t_min >= self.T_dec:
            raise ValueError(
                f"Solver.T_Min={self.t_min} must be < decision horizon T_dec={self.T_dec}")

    # ---- utility anchor ------------------------------------------------------
    def _u(self, W):
        """Bounded terminal-utility anchor u(W) — the framework's normalised utility."""
        return _utility_wrap_signed(W, self.runtime)

    # ---- action grid (shared, mask-aware; inactive axes pinned to 0) ---------
    def _action_grid(self):
        return self.aspace.grid()

    # ---- input standardization ----------------------------------------------
    def _standardize(self, market, W):
        """Standardized state (market | W) for the residual net — POSITION-FREE (toy-faithful:
        with no turnover cost the held position is a freely-reset control, not a state; it
        enters the value only through next-step wealth W1). Market/wealth use bank mean/std."""
        m = (market - self.m_mean) / self.m_std
        wn = ((W - self.w_mean) / self.w_std).unsqueeze(-1)
        return torch.cat([m, wn], dim=-1)

    def _continuation(self, nets, market, W, t, chunk=400_000):
        """C_t = u(W) + A_t(market, W); terminal C_{T_dec} = u(W). Row-chunked net eval.
        Ensemble mode (list-of-checkpoints load): A = mean over members, each evaluated in
        its OWN standardization frame — the frame is part of the function.
        A_t is CLAMPED to its fitted-target trust region (one range-width of headroom):
        off-support the zero-init MLP extrapolates freely — measured printing −5 where its
        targets spanned ±0.5 — and the argmax then chases the phantom direction, poisoning
        the t−1 bootstrap labels (the dead-net and corner-over-leverage basins at large B).
        Outside the clamp the gradient is zero, so the differential labels ignore phantom
        directions too."""
        base = self._u(W)
        if t >= self.T_dec:
            return base
        if getattr(self, "_ensemble", None):
            acc = torch.zeros_like(base)
            for m_nets, m_mean, m_std, w_mean, w_std, m_bounds in self._ensemble:
                x = torch.cat([(market - m_mean) / m_std,
                               ((W - w_mean) / w_std).unsqueeze(-1)], dim=-1)
                b = m_bounds[t] if m_bounds is not None else None
                for i in range(0, x.shape[0], chunk):
                    a = m_nets[t](x[i:i + chunk])
                    acc[i:i + chunk] += a if b is None else torch.clamp(a, b[0], b[1])
            return base + acc / len(self._ensemble)
        x = self._standardize(market, W)
        b = self.a_bounds[t]
        if x.shape[0] <= chunk:
            a = nets[t](x)
            return base + (a if b is None else torch.clamp(a, b[0], b[1]))
        out = torch.empty_like(base)
        for i in range(0, x.shape[0], chunk):
            a = nets[t](x[i:i + chunk])
            out[i:i + chunk] = a if b is None else torch.clamp(a, b[0], b[1])
        return base + out

    # ---- one-step wealth move ------------------------------------------------
    def _wealth_step(self, W, q, dF, dL):
        """W_{t+1} = W + Σ_i q_i·cs_i·dF_i + dL — the frictionless analytic wealth law, owned by
        `hedge_bundle.wealth_step` (the single source `futures_account_step` discretizes; the
        solver's bank/verdict/inner-labels all funnel through here). q (...,n_hedge); dF
        (...,n_hedge); dL (...). Expiry is composed by callers via the `live` mask on dF."""
        return wealth_step(W, q, self.contract_size, dF, dL)

    # ---- inner-MC one-step quantities at outer t -----------------------------
    def _inner_step(self, t):
        """Fork inner MC at t (resolve_structure / resolve_hedge_structure under the hood)
        and return the one-step move tensors the bootstrap needs:
          dF   (B_outer, B_inner, n_hedge)  per-instrument futures move t→t+1
          dL   (B_outer, B_inner)           liability mark change t→t+1
          m1   (B_outer, B_inner, market_dim) market state at t+1
        plus the bank-state market at t (B_outer, market_dim)."""
        inner = self.bundle["inner_mc_fn"](t, one_step=self.one_step)
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
    def _decide(self, nets, market_t1, dF, dL, W, t, q_prev=None, kappa=None, live=None):
        """Pick the grid action maximising E_inner[C_{t+1}] per outer path. No grad. The action
        universe is `aspace.grid_at(t, live)` — the base grid, further filtered to the
        `Total_Position_Schedule` corridor at t when one is configured (else the base grid). The
        `live` leg mask (the callers zero `q*live` after expiry) enters the filter so the corridor
        bounds the REALIZED Σ(q_i·live_i), not a target the expiry mask then guts.
        `q_prev`+`kappa` (cost-aware execution): charge the L1 repositioning cost
        κ·|q − q_prev| against the wealth entering the continuation, so the argmax trades
        off expected value against the cost of getting there (hysteresis instead of churn).
        The value function itself stays cost-free; this is a decision-time correction."""
        grid = self.aspace.grid_at(t, live)
        with torch.no_grad():
            B, Bi, md = market_t1.shape
            best_val = None
            best_q = None
            for s in range(0, grid.shape[0], self.chunk):
                acts = grid[s:s + self.chunk]                                            # (c, n_hedge)
                c = acts.shape[0]
                q = acts[None, :, None, :]                                               # (1,c,1,n_hedge)
                W1 = self._wealth_step(W[:, None, None], q, dF[:, None], dL[:, None])     # (B,c,Bi)
                if q_prev is not None:
                    tc = ((acts[None, :, :] - q_prev[:, None, :]).abs() * kappa).sum(-1)  # (B,c) $
                    W1 = W1 - tc[:, :, None]
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
            # Skip degenerate OR non-finite instruments (an expired-contract row can carry a
            # NaN mark; without this guard clamp(NaN)=NaN poisons the whole textbook book).
            if not float(var) > 1e-12 or not torch.isfinite(dF).all():
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
        # Seed q_prev from the OPENING book (position-free value: the bank wealth W below
        # never reads q_prev, so this only labels q_list[0]; q_prev would carry real weight
        # here only if turnover cost entered the value — see initial_q_from_runtime).
        q_prev = self.aspace.initial_q(self.B_outer, self.device)
        W_list, q_list = [], []
        rng = (self.q_hi - self.q_lo)
        mask = torch.zeros(self.n_hedge, device=self.device)
        mask[self.active] = 1.0
        oob = []                                                                         # per-t bank corridor-breach diagnostic
        for t in range(self.T_dec):
            W_list.append(W.clone())
            q_list.append(q_prev.clone())
            q_rep = self._replication_hedge(t)                                            # (n_hedge,)
            noise = self.noise_frac * rng * torch.randn(
                self.B_outer, self.n_hedge, generator=gen, device=self.device)
            q = torch.minimum(torch.maximum(q_rep[None] + noise * mask, self.q_lo), self.q_hi)
            if self.aspace.schedule is not None:
                # Keep the exploration IN the corridor so the value fn trains on reachable wealth
                # states only (un-projected, ~50% of bank paths breach — the nets would fit
                # unreachable W). Diagnostic below then confirms ~0 residual breach.
                q = self.aspace.project_to_corridor(q, t)
                lo, hi = self.aspace._corridor_at(t)                                      # signed-total corridor at t
                tot = q[:, self.active].sum(-1)                                           # bank signed total (active legs)
                tol = 1e-3                     # float32 headroom-fill lands on the edge to ~1e-6
                oob.append((t, lo, hi, float((tot < lo - tol).float().mean()
                                             + (tot > hi + tol).float().mean()),
                            float(tot.min()), float(tot.max())))
            dF = torch.stack(
                [self.tradables_sim[ref][t + 1] - self.tradables_sim[ref][t] for ref in self.hedges],
                dim=-1)                                                                  # (B_outer, n_hedge)
            W = self._wealth_step(W, q, dF, L[t + 1] - L[t])
            q_prev = q
        if oob:
            worst = max(oob, key=lambda r: r[3])
            logging.info(
                "DiffSolverV2 bank IN Total_Position_Schedule (post-projection): residual "
                "frac(Σq outside corridor) min=%.3f mean=%.3f max=%.3f | worst t=%d "
                "corridor=[%.4g, %.4g] bank Σq∈[%.4g, %.4g] frac_oob=%.3f (≈0 ⇒ projection clean)",
                min(r[3] for r in oob), sum(r[3] for r in oob) / len(oob),
                worst[3], worst[0], worst[1], worst[2], worst[4], worst[5], worst[3])
        return W_list, q_list

    # ---- project per-process state-at-t leaf grads → market_t columns --------
    def _project_leaf_grads(self, leaf_grads, widths, rows, n, md):
        """Map ∂Y/∂(state_t leaf) for each simulated factor into the `(n, md)` gradient w.r.t.
        the privileged market_t columns the value net consumes. `widths` is in market-column
        order (factor iteration order); a regime-switching spot occupies [belief(width-1),
        price(1)] (`MarkovHMMSpotModel.reveal_state_at` layout: belief-first, price-last),
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
    def _fit_step(self, nets, W_bank, t, inner, rows=slice(None)):
        dF_ng, dL_ng, m1_ng, market0, live = inner                                       # no-grad cache
        market0 = market0[rows]
        W0_bank = W_bank[t][rows]
        # SELECT the action on the NO-GRAD inner draws; EVALUATE its value + pathwise gradients
        # on a fresh GRAD inner (independent draws → cross-fit, no winner's-curse max-bias).
        q_star, _ = self._decide(nets, m1_ng[rows], dF_ng[rows], dL_ng[rows], W0_bank, t, live=live)
        q_star = q_star * live          # expired contracts: dF=0 ⇒ wealth-neutral; report 0, not the tie

        # GRAD inner-MC fork: AAD-live one-step F_t1/L_t1/market_t1 + per-process state-at-t
        # LEAVES. Bootstrap value Y AND its pathwise gradients w.r.t. W0 (wealth) and the
        # market state (spot/belief) come from the SAME forward — the full Huge–Savine twin
        # loss. ∂Y/∂market_t is the differential constraint that regularizes the market
        # dimension (where a value-only / W-only fit overfits the few outer paths).
        # Row-aware grad-slice width: the grad fork's tape scales with remaining rows × flat —
        # except one-step forks, whose tape is 2 rows regardless of t (the 64 floor then just
        # reproduces the flat cap, giving constant-width slices across the whole sweep).
        rows_t = max(64, 2 if self.one_step else (self.n_steps + 1 - t))
        cell_budget = int(self.bundle.get("inner_mc_cell_budget", 1 << 62))
        inner_sub = int(self.bundle.get("inner_sub_batch", 1))
        grad_chunk = max(1, cell_budget // (inner_sub * rows_t))
        if self.B_outer > grad_chunk:
            # Large-B mode: the grad fork's AAD tape only fits `grad_chunk` outer paths at a
            # time, so fork the TRAIN rows in contiguous sub-slices (labels are per-outer-path
            # — slices are independent; each fork call is single-chunk under the hood).
            r0, r1, _ = rows.indices(self.B_outer)
            Y_parts, gW_parts, gm_parts = [], [], []
            for a in range(r0, r1, grad_chunk):
                b = min(a + grad_chunk, r1)
                ig = self.bundle["inner_mc_grad_fn"](t, outer_rows=(a, b), one_step=self.one_step)
                leaves, widths = ig["state_t_leaves"], ig["state_t_leaf_widths"]
                F_t_c = torch.stack(
                    [self.tradables_sim[r][t][a:b] for r in self.hedges], dim=-1)
                F_t1_c = torch.stack([ig["F_t1"][r] for r in self.hedges], dim=-1) * live
                dF_c = F_t1_c - F_t_c.unsqueeze(1)
                dL_c = ig["L_t1"] - ig["L_t"]
                m1_c = ig["market_t1"]
                W0_c = W0_bank[a - r0:b - r0].clone().requires_grad_(True)
                q_c = q_star[a - r0:b - r0][:, None, :]
                W1_c = self._wealth_step(W0_c[:, None], q_c, dF_c, dL_c)
                n_c, Bi_c, md = m1_c.shape
                Y_c = self._continuation(
                    nets, m1_c.reshape(-1, md), W1_c.reshape(-1), t + 1
                ).reshape(n_c, Bi_c).mean(1)
                grads_c = torch.autograd.grad(
                    Y_c.sum(), [W0_c] + list(leaves.values()), allow_unused=True)
                leaf_grads_c = {k: (g.detach() if g is not None else None)
                                for k, g in zip(leaves.keys(), grads_c[1:])}
                Y_parts.append(Y_c.detach())
                gW_parts.append(grads_c[0].detach())
                gm_parts.append(self._project_leaf_grads(
                    leaf_grads_c, widths, slice(None), n_c, md))
            Y = torch.cat(Y_parts)
            gW = torch.cat(gW_parts)
            g_market = torch.cat(gm_parts)
            return self._fit_from_labels(nets, W0_bank, market0, Y, gW, g_market, t, q_star)
        ig = self.bundle["inner_mc_grad_fn"](t, one_step=self.one_step)
        leaves, widths = ig["state_t_leaves"], ig["state_t_leaf_widths"]
        if not getattr(self, "_proj_checked", False):
            self._proj_checked = True                  # one-time self-check of the label projection
            mt, col, errs = ig["market_t"].detach(), 0, []      # detach: numeric self-check only
            n = mt.shape[0]
            for key, width in widths:
                if width <= 0:
                    continue
                bl, pl = leaves.get((key, "regime_belief")), leaves.get(key)
                bl = bl.detach() if bl is not None else None
                pl = pl.detach() if pl is not None else None
                if bl is not None and bl.numel() == (width - 1) * n:            # spot: belief + price
                    nb = width - 1
                    be = float((mt[:, col:col + nb] - bl.reshape(nb, -1).transpose(0, 1)).abs().max())
                    pe = float((mt[:, col + nb] - pl.reshape(-1)).abs().max()) if pl is not None and pl.numel() == n else -1.0
                    errs.append(f"{utils.check_tuple_name(key)}[belief={be:.1g},price={pe:.1g}]")
                elif pl is not None and pl.numel() == width * n:               # 1:1 raw → privileged
                    e = float((mt[:, col:col + width] - pl.reshape(width, -1).transpose(0, 1)).abs().max())
                    errs.append(f"{utils.check_tuple_name(key)}[1:1={e:.1g}]")
                else:                                                          # belief leaf absent → masked
                    errs.append(f"{utils.check_tuple_name(key)}[unmeasured]")
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
        return self._fit_from_labels(nets, W0_bank, market0, Y, gW, g_market, t, q_star)

    def _fit_from_labels(self, nets, W0_bank, market0, Y, gW, g_market, t, q_star):
        """Shared fit tail: advantage decomposition + standardized twin loss on the
        (value, wealth-grad, market-grad) labels. Called by both the single-fork and the
        sub-sliced large-B label paths of `_fit_step`."""
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
        # Per-column lambda_j (Huge-Savine official): each market column normalized by its
        # own label variance, so one fat-tailed column can't deflate the differential
        # constraint for the rest. 'No' = legacy pooled scalar.
        nrm_m = (g_zn_m.var(dim=0, keepdim=True) + 1e-8
                 if bool(self.cfg.get("diffv2_per_column_grad_norm", False))
                 else g_zn_m.var() + 1e-8)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr,
                               weight_decay=float(self.cfg.get("diffv2_weight_decay", 0.0)))
        for _ in range(self.fit_iters):
            mn = ((market0 - self.m_mean) / self.m_std).detach().requires_grad_(True)
            wn = ((W0_bank - self.w_mean) / self.w_std).detach().requires_grad_(True)
            a = net(torch.cat([mn, wn.unsqueeze(-1)], dim=-1))
            da_m, da_w = torch.autograd.grad(a.sum(), [mn, wn], create_graph=True)
            loss = (((a - a_val) ** 2).mean() / nrm_v
                    + lam_g * ((da_w - g_zn_W) ** 2).mean() / nrm_w
                    + lam_g * (((da_m - g_zn_m) ** 2) / nrm_m).mean())
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            a_fit = net(self._standardize(market0, W0_bank))
            val_loss = float(((a_fit - a_val) ** 2).mean())
            # Trust region for all future EVALUATIONS of this net (argmax, bootstrap labels,
            # verdict): its fitted-target range plus one range-width of headroom each side.
            lo, hi = float(a_val.min()), float(a_val.max())
            pad = max(hi - lo, 1e-3)
            self.a_bounds[t] = (lo - pad, hi + pad)
        return {
            "t": t, "val_loss": val_loss,
            "Y_absmean": float(Y.abs().mean()),
            "A_absmean": float(a_fit.abs().mean()),
            "q_star_mean": q_star.mean(0).detach().cpu().tolist(),
            "Y_mean": float(Y.mean()),
        }

    # ---- greedy-rollout downside verdict -------------------------------------
    def _verdict(self, nets, inner_cache, sweep_ts, rows=slice(None)):
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
        # Parallel turnover-cost accounting (Transaction_Cost_Per_Unit + half the
        # Bid_Offer_Spread_Bps on |Δq| each rebalance, entry measured from the OPENING book
        # q0 — not from flat). The wealth recursion itself stays cost-free (the position-free
        # value design assumes free repositioning) — these are DIAGNOSTICS quantifying that
        # approximation per policy.
        cost = {p: torch.zeros(n, device=self.device) for p in ("greedy", "textbook")}
        q0 = self.aspace.initial_q(n, self.device)
        q_prev = {p: q0.clone() for p in ("greedy", "textbook")}
        with torch.no_grad():
            for t in sweep_ts:
                dF_o = torch.stack(
                    [self.tradables_sim[r][t + 1][rows] - self.tradables_sim[r][t][rows]
                     for r in self.hedges], dim=-1)                                       # (n, n_hedge)
                dL_o = (L[t + 1] - L[t])[rows]
                dF, dL, m1, _, live = inner_cache[t]
                kappa_t = self.aspace.kappa(self.tradables_sim, t)
                q_g, _ = self._decide(
                    nets, m1[rows], dF[rows], dL[rows], W["greedy"], t,
                    q_prev=q_prev["greedy"] if self.cost_aware else None,
                    kappa=kappa_t if self.cost_aware else None, live=live)
                q_g = q_g * live          # zero positions on expired contracts (wealth-neutral)
                # Textbook = diagonal min-var, PROJECTED into the corridor (no schedule ⇒ identity)
                # so the benchmark obeys the same mandate as greedy — a fair in-corridor comparison.
                q_tb = self.aspace.project_to_corridor(
                    self._replication_hedge(t)[None].expand(n, self.n_hedge), t)
                z = torch.zeros(n, self.n_hedge, device=self.device)
                for p, q_now in (("greedy", q_g), ("textbook", q_tb)):
                    cost[p] = cost[p] + _turnover_cost(q_now - q_prev[p], kappa_t)
                    q_prev[p] = q_now
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
        for p in ("greedy", "textbook"):
            net_stats = stats(W[p] - cost[p])
            out[p]["turnover_cost_mean"] = float(cost[p].mean())
            out[p].update({f"{k}_net": v for k, v in net_stats.items()})
        # greedy position summary: mean over the rollout of |q| per instrument (is it hedging?)
        gq = torch.tensor(q_traj["greedy"])                                              # (n_steps, n_hedge)
        out["greedy_mean_abs_q"] = gq.abs().mean(0).tolist()
        out["greedy_q_traj"] = q_traj["greedy"]          # full per-t mean book (audit trail)
        out["textbook_q_traj"] = q_traj["textbook"]      # corridor-projected benchmark book (audit)
        out["greedy_q_first"] = q_traj["greedy"][0] if q_traj["greedy"] else None
        out["greedy_q_mid"] = (q_traj["greedy"][len(q_traj["greedy"]) // 2]
                               if q_traj["greedy"] else None)
        return out

    # ---- frozen-policy daily rollout on a realized path via the stepper -------
    def _rollout_on_stepper(self, nets, inner_cache, sweep_ts):
        """Deployment-faithful backtest: roll the frozen policy day-by-day along the
        bundle's (observed) path through `BundleStepper`, which owns the real futures
        accounting (variation margin, financing, per-instrument expiry, forced-flat).
        Each day the book is chosen by `_decide` from the CAUSAL one-step fork forecast
        (`inner_cache[t]`) and the STEPPER'S OWN net wealth — never a verdict wealth
        recursion, so the decision can't be contaminated by mis-accrued P&L. This is
        the JSON-contract interface for running the precomputed diff-ML nets daily.
        Returns {greedy, textbook, nohedge} terminal-P&L stats in the verdict shape."""
        from .hedge_bundle import BundleStepper, _tracking_error_value
        hist = int(self.bundle.get("initial_time_index", 0))
        sweep_set = set(int(t) for t in sweep_ts)

        q_log = {"greedy": [], "t": []}

        def roll(policy):
            stepper = BundleStepper(self.bundle, self.runtime)
            # Seed the cost-aware decision q_prev from the OPENING book (the stepper's own
            # positions already open here too, so its realized first-step turnover is measured
            # from q0). The frozen value is position-free — q0 only shifts first-step cost/P&L.
            q_prev = self.aspace.initial_q(self.B_outer, self.device)
            last = None
            while not stepper.done:
                t = stepper.time_index - hist
                if stepper.is_decision_step and t in sweep_set:
                    state = stepper._state
                    W = _tracking_error_value(state, self.runtime).to(self.device)     # (B,) net wealth
                    B = W.shape[0]
                    if policy == "nohedge":
                        q = torch.zeros(B, self.n_hedge, device=self.device)
                    elif policy == "textbook":
                        qt = self._replication_hedge(t)     # (n_hedge,) per-instrument clamped
                        # Also honour the TOTAL position cap (replication clamps per-instrument
                        # only; unscaled it can hold n_hedge x the cap and blow up the stepper's
                        # margin accounting). Scale the book down proportionally if over-limit.
                        tot = float(qt.abs().sum())
                        if self.total_abs_limit > 0.0 and tot > self.total_abs_limit:
                            qt = qt * (self.total_abs_limit / tot)
                        # Obey the corridor mandate too (no schedule ⇒ identity), so the stepper
                        # textbook is the same in-corridor min-var benchmark the verdict rolls.
                        q = self.aspace.project_to_corridor(qt[None].expand(B, self.n_hedge), t)
                    else:
                        dF, dL, m1, _, live = inner_cache[t]
                        kappa_t = self.aspace.kappa(self.tradables_sim, t)
                        q, _ = self._decide(nets, m1, dF, dL, W, t,
                                            q_prev=q_prev if self.cost_aware else None,
                                            kappa=kappa_t if self.cost_aware else None, live=live)
                        q = q * live
                        q_log["greedy"].append(q.mean(0).detach().cpu().tolist())
                        q_log["t"].append(int(t))
                    q_prev = q
                    cur = state["positions"]
                    delta = {n: q[:, j] - cur[n].to(dtype=q.dtype, device=q.device)
                             for j, n in enumerate(self.hedges)}
                    last = stepper.step(delta)
                else:
                    last = stepper.step(None)
            return (last["transition_pnl_excess"]
                    + last["transition_liability_value"]).to(torch.float64)

        def stats(wT):
            p5 = torch.quantile(wT, 0.05)
            cvar5 = wT[wT <= p5].mean() if (wT <= p5).any() else p5
            return {"u_mean": float(self._u(wT.to(torch.float32)).mean()),
                    "wT_mean": float(wT.mean()), "wT_p5": float(p5), "wT_cvar5": float(cvar5)}
        out = {p: stats(roll(p)) for p in ("greedy", "textbook", "nohedge")}
        out["greedy_q_traj"] = q_log["greedy"]        # per-decision mean book (audit)
        out["greedy_q_t"] = q_log["t"]
        return out

    # ---- driver --------------------------------------------------------------
    def solve(self):
        # Bank RNG is deterministic; multi-seed repeats (solve_hedge calls solve() N times)
        # advance the framework's inner-MC Sobol stream, so V_0 spread reflects inner-MC noise.
        gen = torch.Generator(device=self.device)
        gen.manual_seed(0)
        logging.info(
            "DiffSolverV2 setup: n_hedge=%d active=%s T_dec=%d (of %d sim steps; last-live "
            "mtm=[-2]) B_outer=%d levels=%d fit_iters=%d lr=%.3g | "
            "contract_size=%s | q∈[%s, %s] total_abs_limit=%.3g",
            self.n_hedge, self.active, self.T_dec, self.n_steps,
            self.B_outer, self.levels, self.fit_iters, self.lr, self.contract_size.tolist(),
            self.q_lo.tolist(), self.q_hi.tolist(), self.total_abs_limit)
        logging.info("DiffSolverV2 inner forks: one_step=%s (window {t, t+1} generation+pricing)"
                     if self.one_step else
                     "DiffSolverV2 inner forks: one_step=%s (full remaining-horizon forks)",
                     self.one_step)

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
        load_cfg = self.cfg.get("diffv2_load_value_fn", "") or ""
        # A load member is either a checkpoint PATH (JSON contract) or an already-materialised
        # artifact DICT (the same dict solve() returns via value_fn_artifacts / torch.saves) —
        # the eval-from-artifact path treats an in-memory artifact exactly like a loaded file.
        load_members = ([(p if isinstance(p, dict) else str(p)) for p in load_cfg]
                        if isinstance(load_cfg, (list, tuple))
                        else ([load_cfg] if isinstance(load_cfg, dict)
                              else ([str(load_cfg)] if load_cfg else [])))
        loaded = None
        if load_members:
            # Frozen-policy eval: restore the fitted nets AND each function's frame — the
            # train-time standardization stats and utility scale are part of the value
            # function; recomputing them from the (possibly stressed) eval world would
            # silently change what the nets compute. EVERY eval path is unseen by the nets,
            # so the verdict rolls over all paths. A LIST of checkpoints = ensemble argmax:
            # each member evaluated in its own frame, continuations averaged (cross-fit
            # winner's-curse reduction on top of antithetic).
            members = []
            for member in load_members:
                # Pre-contract checkpoints predate active_hedge_indices / solver_version /
                # config_hash — they still load: only the frame + net keys below are read.
                ck = member if isinstance(member, dict) else torch.load(member, map_location=self.device)
                src = "<in-memory artifact>" if isinstance(member, dict) else member
                for key, want in (("t_min", self.t_min), ("T_dec", self.T_dec),
                                  ("md", md), ("hedges", list(self.hedges))):
                    if ck[key] != want:
                        raise ValueError(
                            f"DiffV2_Load_Value_Fn checkpoint mismatch on {key!r}: "
                            f"{src} saved {ck[key]!r} vs this run {want!r}")
                # Corridor provenance: a value fn trained INSIDE a Total_Position_Schedule fit only
                # the wealth states that corridor makes reachable. The wealth support is monotone in
                # the corridor: training UNCONSTRAINED spans the widest support, so rolling that
                # policy inside ANY corridor only restricts to a learned subset (valid — this is the
                # roll-only-on-corridor-free validation path). But a policy trained in a specific
                # corridor is queried off-support under a DIFFERENT or absent one → fail loud.
                want_sched = _schedule_key(self.aspace.schedule)
                if "total_position_schedule" in ck:
                    saved_sched = _schedule_key(ck["total_position_schedule"])
                    if saved_sched == want_sched:
                        pass                                         # same corridor — exact match
                    elif saved_sched is None:
                        logging.info(
                            "DiffV2_Load_Value_Fn: %s trained corridor-free (widest wealth "
                            "support); rolling under a Total_Position_Schedule only restricts to a "
                            "learned subset — valid.", src)
                    else:
                        raise ValueError(
                            f"DiffV2_Load_Value_Fn corridor mismatch: {src} was trained under "
                            f"Total_Position_Schedule {saved_sched} but this run rolls under "
                            f"{want_sched}. A policy trained inside a corridor is queried off its "
                            f"learned wealth support under a different (or absent) one — retrain, "
                            f"or match the Evaluator.Total_Position_Schedule.")
                elif want_sched is not None:
                    logging.warning(
                        "DiffV2_Load_Value_Fn: %s predates corridor provenance (no "
                        "total_position_schedule stamp) but this run sets a Total_Position_Schedule "
                        "— cannot verify the frozen policy was trained in it; roll validity "
                        "unverified.", src)
                drift = ((M.mean(0) - ck["m_mean"]).abs() / ck["m_std"]).max()
                logging.info(
                    "DiffSolverV2 LOADED value fn from %s (train V_0=%+.6g) | eval-world "
                    "market drift vs train frame: max %.3g σ | utility_scale %.6g",
                    src, ck["V_0"], float(drift), ck["utility_scale"])
                members.append(ck)
            loaded = members[0]
            scales = [float(ck["utility_scale"]) for ck in members]
            if max(scales) - min(scales) > 0.01 * max(scales):
                logging.warning(
                    "DiffSolverV2 ensemble utility_scale spread %.3g%% — members trained "
                    "against different anchors; averaging is approximate",
                    100.0 * (max(scales) - min(scales)) / max(scales))
            self.m_mean, self.m_std = loaded["m_mean"], loaded["m_std"]
            self.w_mean, self.w_std = loaded["w_mean"], loaded["w_std"]
            self.runtime["objective"]["utility_scale"] = float(sum(scales) / len(scales))
            hidden = int(loaded["hidden"])
        nets = [_DiffV2Residual(md + 1, hidden=hidden).to(self.device)  # position-free: (market | W)
                for _ in range(self.T_dec)]
        # Per-t trust region for A_t evaluation (set at fit time / restored from checkpoint;
        # None = unclamped, e.g. pre-trust-region checkpoints).
        self.a_bounds = (list(loaded["a_bounds"]) if loaded is not None and loaded.get("a_bounds")
                         else [None] * self.T_dec)
        logging.info("DiffSolverV2 action grid: K=%d actions (levels=%d ^ active=%d)",
                     int(grid.shape[0]), self.levels, self.n_active)

        rows = []
        if loaded is not None:
            for net, sd in zip(nets, loaded["state_dicts"]):
                net.load_state_dict(sd)
                net.eval()
            if len(members) > 1:
                # Ensemble: per-member net stacks + frames; _continuation averages members.
                self._ensemble = []
                for ck in members:
                    m_nets = [_DiffV2Residual(md + 1, hidden=int(ck["hidden"])).to(self.device)
                              for _ in range(self.T_dec)]
                    for net, sd in zip(m_nets, ck["state_dicts"]):
                        net.load_state_dict(sd)
                        net.eval()
                    self._ensemble.append(
                        (m_nets, ck["m_mean"], ck["m_std"], ck["w_mean"], ck["w_std"],
                         ck.get("a_bounds")))
                logging.info("DiffSolverV2 ENSEMBLE argmax over %d value fns", len(members))
            worst = float(loaded["max_abs_Y_boot"])
            root = {"t": self.t_min, "Y_mean": float(loaded["V_0"]),
                    "q_star_mean": list(loaded["n_star_0"])}
        else:
            for t in reversed(sweep_ts):
                r = self._fit_step(nets, W_bank, t, inner_cache[t], rows=train)
                rows.append(r)
                logging.info(
                    "DiffSolverV2 C[t=%d] fitted: val_loss=%.4g |Y_boot|=%.4g |A|=%.4g "
                    "Y_mean=%+.4g q*_mean=%s", r["t"], r["val_loss"], r["Y_absmean"],
                    r["A_absmean"], r["Y_mean"],
                    ["%.3f" % v for v in r["q_star_mean"]])
            worst = max((r["Y_absmean"] for r in rows if math.isfinite(r["Y_absmean"])),
                        default=0.0)
            root = rows[-1] if rows else {"t": self.t_min, "Y_mean": 0.0, "q_star_mean":
                                          [0.0] * self.n_hedge}
        V_0 = float(root["Y_mean"])
        n_star_0 = root["q_star_mean"]
        bounded = math.isfinite(V_0) and worst < 1.0e4
        if loaded is None:
            logging.info(
                "DiffSolverV2 sweep complete: t=%d→%d | max|Y_boot|=%.4g (%s) | "
                "V_0=%+.6g | n_star@t=%d=%s", self.T_dec - 1, self.t_min, worst,
                "BOUNDED" if bounded else "EXPLODED", V_0, root["t"], n_star_0)
        # POLICY ARTIFACT (built ONCE, here): the fitted value function + its frame + the
        # provenance stamps. This is the single source returned via SolverResult (→
        # HedgeRuntimeExecutionResult.policy_artifact) AND torch.saved to DiffV2_Save_Value_Fn
        # — the file and the in-memory dict are byte-for-byte the same object, so the
        # eval-from-artifact path (load member = this dict) is identical to loading the file.
        artifact = None
        save_path = str(self.cfg.get("diffv2_save_value_fn", "") or "")
        if loaded is None:
            artifact = {
                "state_dicts": [net.state_dict() for net in nets],
                "m_mean": self.m_mean, "m_std": self.m_std,
                "w_mean": self.w_mean, "w_std": self.w_std,
                "utility_scale": float(self.runtime["objective"]["utility_scale"]),
                "a_bounds": self.a_bounds,
                "hedges": list(self.hedges),
                "active_hedge_indices": list(self.active),
                # Corridor provenance: the Total_Position_Schedule this policy was trained inside
                # (None = unconstrained). A load under a DIFFERENT corridor fails loud (above).
                "total_position_schedule": _schedule_key(self.aspace.schedule),
                "T_dec": self.T_dec, "t_min": self.t_min, "md": md, "hidden": hidden,
                "solver_version": SOLVER_VERSION,
                "config_hash": _config_hash(self.cfg),
                # Headline echoed so a loaded eval reads it back rather than recomputing.
                "V_0": V_0, "n_star_0": list(n_star_0), "max_abs_Y_boot": worst,
            }
            if save_path:
                if not math.isfinite(V_0):
                    raise ValueError(
                        f"refusing to save non-finite value fn to {save_path}: V_0={V_0}")
                tmp = save_path + ".tmp"
                torch.save(artifact, tmp)
                os.replace(tmp, save_path)                       # atomic on POSIX
                logging.info("DiffSolverV2 SAVED value fn to %s (V_0=%+.6g)", save_path, V_0)

        # Downside verdict: greedy policy vs textbook delta hedge vs no hedge. HEADLINE is the
        # OUT-OF-SAMPLE rollout (held-out paths the nets never saw); in-sample reported too.
        if loaded is not None:
            # Frozen nets never saw ANY of this run's paths — the whole batch is out-of-sample.
            verdict = verdict_is = self._verdict(nets, inner_cache, sweep_ts,
                                                 rows=slice(None))
            has_oos = True
        else:
            verdict = self._verdict(nets, inner_cache, sweep_ts,
                                    rows=(test if has_oos else train))
            verdict_is = (self._verdict(nets, inner_cache, sweep_ts, rows=train)
                          if has_oos else verdict)
        if has_oos and loaded is None:
            logging.info(
                "DiffSolverV2 IN-SAMPLE vs OOS u(W_T): greedy IS=%+.5f OOS=%+.5f | "
                "textbook IS=%+.5f OOS=%+.5f (gap IS−OOS greedy=%+.5f → overfit if large)",
                verdict_is["greedy"]["u_mean"], verdict["greedy"]["u_mean"],
                verdict_is["textbook"]["u_mean"], verdict["textbook"]["u_mean"],
                verdict_is["greedy"]["u_mean"] - verdict["greedy"]["u_mean"])
        # Deployment-faithful backtest: when a frozen policy is loaded and stepper rollout is
        # requested, roll it day-by-day on the (observed) path via BundleStepper — real futures
        # accounting + decisions off the stepper's own wealth. This is the trustworthy P&L for a
        # walk-forward backtest (the simplified _verdict wealth recursion mis-accrues expiry).
        stepper_verdict = None
        if loaded is not None and bool(self.cfg.get("diffv2_stepper_rollout", False)):
            stepper_verdict = self._rollout_on_stepper(nets, inner_cache, sweep_ts)
            sg, stb, snh = (stepper_verdict[k] for k in ("greedy", "textbook", "nohedge"))
            logging.info(
                "DiffSolverV2 STEPPER ROLLOUT (frozen policy, realized path, real accounting):\n"
                "  greedy   wT=%+.4e p5=%+.4e cvar5=%+.4e\n"
                "  textbook wT=%+.4e p5=%+.4e cvar5=%+.4e\n"
                "  nohedge  wT=%+.4e p5=%+.4e cvar5=%+.4e",
                sg["wT_mean"], sg["wT_p5"], sg["wT_cvar5"],
                stb["wT_mean"], stb["wT_p5"], stb["wT_cvar5"],
                snh["wT_mean"], snh["wT_p5"], snh["wT_cvar5"])

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
            value_fn_artifacts=artifact,              # the fitted policy (None in eval-from-load runs)
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
                "value_fn_path": save_path or None,       # where the artifact was persisted (if any)
                "verdict": verdict,                       # OUT-OF-SAMPLE (held-out paths)
                "stepper_verdict": stepper_verdict,       # frozen-policy realized-path rollout (real accounting)
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
