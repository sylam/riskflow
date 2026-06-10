"""Model-independent dynamic-hedging solvers (Execution_Mode='solve_hedge').

Replaces the PPO policy track with DP / MPC solvers that consume the same simulated
scenario bundle and inner-MC infrastructure. The solver forks inner MC on demand via
the closure `bundle['inner_mc_fn']` attached by `HedgeMonteCarlo.execute`.

Milestone 1 surface: `MpcSolver` (model-predictive control at a single decision state),
`build_action_grid`, `search_action_grid`, `solve_hedge` dispatcher, `SolverResult`.
`LsmDpSolver` / `HindsightDpSolver` / `ValueFunctionApproximator` / `StatePack` land in
later milestones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch

from . import utils
from .torchrl_hedge import (
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


def search_action_grid(action_grid, objective_fn, *, chunk_size, total_abs_limit):
    """Chunked argmax over a discrete action grid. `objective_fn(action_chunk)` maps a
    `(chunk, n_hedge)` action block to a `(chunk, *rest)` score tensor (`*rest` is empty
    for MPC's single decision state, `(B_outer,)` for the per-outer-path DP step). Actions
    violating `total_abs_limit` (|a|.sum > limit) get `-inf` so they never win. No
    autograd — pure tensor enumeration. Returns `(best_action, best_value)`."""
    best_value = None
    best_action = None
    for start in range(0, action_grid.shape[0], chunk_size):
        chunk = action_grid[start:start + chunk_size]                  # (c, 3)
        obj = objective_fn(chunk)                                      # (c, *rest)
        if total_abs_limit > 0.0:
            valid = chunk.abs().sum(dim=-1) <= total_abs_limit         # (c,)
            mask = ~valid.view((chunk.shape[0],) + (1,) * (obj.ndim - 1))
            obj = obj.masked_fill(mask, float("-inf"))
        chunk_max, chunk_arg = obj.max(dim=0)                          # (*rest,), (*rest,)
        chunk_action = chunk[chunk_arg]                                # (*rest, 3)
        if best_value is None:
            best_value, best_action = chunk_max, chunk_action
        else:
            upd = chunk_max > best_value
            best_value = torch.where(upd, chunk_max, best_value)
            best_action = torch.where(upd.unsqueeze(-1), chunk_action, best_action)
    return best_action, best_value


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
    """History-stripped views of the bundle's time-indexed tensors. `build_torchrl_bundle`
    prepends a `History_Lookback_Business_Days` prefix to `tradables` /
    `static_portfolio_descriptors`, but `inner_mc_fn` works in simulation-grid coords.
    Stripping the prefix lets the solver index every time tensor by the same sim-grid `t`.
    Returns `(tradables_sim, static_sim, n_outer_steps)`."""
    hist = int(bundle.get("initial_time_index", 0))
    tradables = {k: v[hist:] for k, v in bundle["tradables"].items()}
    static = bundle["static_portfolio_descriptors"][hist:]
    return tradables, static, int(static.shape[0])


class MpcSolver:
    """Model-predictive-control solver. At a single decision state (outer time `t`),
    runs inner MC once and picks the discrete-grid action that maximises the inner-MC
    estimate of the expected hedging objective — assuming the position is held flat
    through `T` (receding-horizon assumption). No value function.

    The objective is `mean_inner[u_signed(W_T)]` — the symlog terminal utility (brief
    §MPC). Futures are ~martingales, so a hedge moves no mean of `W_T`; it is the
    concavity of the symlog utility (Jensen — variance aversion) that makes a short
    hedge optimal. `solve_hedge` enforces an `AsymmetricUtility_Symlog` objective, so
    `u_signed` is always the concave transform here, never the identity."""

    def __init__(self, bundle, runtime, t_current=0):
        self.bundle = bundle
        self.runtime = runtime
        self.t_current = t_current

    def solve(self):
        bundle, runtime = self.bundle, self.runtime
        device = bundle["time_grid_days"].device
        solver_cfg = runtime["solver"]
        acc = runtime["accounting"]
        hedges = list(runtime["names"]["hedges"])
        t = self.t_current
        tradables_sim, _, _ = _bundle_sim_views(bundle)

        # Inner MC at the decision time — raw per-(outer, inner) samples.
        inner = bundle["inner_mc_fn"](t)
        L_T = inner["L_T"]                                             # (B_outer, B_inner)
        dF_T = torch.stack([inner["dF_T"][ref] for ref in hedges], dim=0)  # (3, B_outer, B_inner)

        kappa, contract_size = _per_contract_kappa(tradables_sim, runtime, hedges, t, device)
        # Flat book at the decision time — Milestone 1 prices the t=0 decision.
        n_prev = torch.zeros(len(hedges), device=device)
        g_t = 0.0
        force_flat = acc["force_flat_at_end"]
        liability = L_T.unsqueeze(0)                                   # (1, B_outer, B_inner)

        def hedge_pnl_excess(action):
            """Hedge P&L since inception for action `action` (shape (..., 3)): cash carry
            plus Σ_i a_i·contract_size_i·dF_T_i, net of hold + terminal-unwind turnover."""
            scaled = action * contract_size                            # (..., 3)
            hedge_pnl = torch.einsum("...i,ibn->...bn", scaled, dF_T)   # (..., B_outer, B_inner)
            cost = _turnover_cost(action - n_prev, kappa)              # (...,)  hold-constant turnover
            if force_flat:
                cost = cost + _turnover_cost(action, kappa)            # terminal unwind n -> 0
            return g_t + hedge_pnl - cost[..., None, None]

        def objective_fn(action_chunk):                               # (c, 3)
            pnl_excess = hedge_pnl_excess(action_chunk)               # (c, B_outer, B_inner)
            obj = _utility_wrap_signed(pnl_excess + liability, runtime)  # (c, B_outer, B_inner)
            return obj.mean(dim=(1, 2))                                # (c,)

        action_grid = build_action_grid(
            runtime, solver_cfg["training_action_grid_levels_per_axis"], device)
        n_star, best_value = search_action_grid(
            action_grid, objective_fn,
            chunk_size=solver_cfg["training_action_chunk_size"],
            total_abs_limit=acc["total_position_abs_limit"])

        # Terminal samples under n* (diagnostics): raw net P&L and the shaped objective.
        pnl_excess = hedge_pnl_excess(n_star)                          # (B_outer, B_inner)
        net_pnl = pnl_excess + L_T
        obj = _utility_wrap_signed(net_pnl, runtime)

        return SolverResult(
            solver_name="MpcSolver",
            actions=n_star.detach().cpu(),
            values=float(best_value),
            terminal_pnl=net_pnl.detach().cpu(),
            terminal_utility=obj.detach().cpu(),
            diagnostics={
                "decision_time_index": t,
                "n_star": n_star.detach().cpu().tolist(),
                "V_0": float(best_value),
                "expected_objective": float(best_value),
                "mean_terminal_pnl": float(net_pnl.mean()),
                "action_grid_size": int(action_grid.shape[0]),
            },
        )


class StatePack:
    """Layout of the deep-state vector the value function consumes — a flat concatenation
    of blocks whose widths are derived from the problem at runtime:

        positions (n_hedge) | cash (1) | accrued (1) | tradable_prices (n_hedge)
        | market (market_dim) | static (n_static) | time (1)

    The `market` block is opaque — whatever `_run_inner_mc_at_t` packs from the simulated
    factors (factor-type knowledge lives entirely in `_extract_outer_state_at`). Nothing
    here names a regime or a carry factor: the DP works for any factor set, and a new
    factor type is supported by extending `_extract_outer_state_at`, not this layout.

    `accrued` (the leg-accrual recursion) is a forward-compatible slot held at 0 — a
    documented Milestone-2 deferral."""

    def __init__(self, n_hedge, market_dim, n_static):
        self.n_hedge = n_hedge
        self.market_dim = market_dim
        self.n_static = n_static
        self.deep_dim = 2 * n_hedge + 2 + market_dim + n_static + 1
        self.pos_slice = slice(0, n_hedge)
        market_start = 2 * n_hedge + 2
        self.market_slice = slice(market_start, market_start + market_dim)

    @classmethod
    def from_bundle(cls, bundle, runtime, market_dim):
        n_hedge = len(runtime["names"]["hedges"])
        hist = int(bundle.get("initial_time_index", 0))
        n_static = int(bundle["static_portfolio_descriptors"][hist].shape[-1])
        return cls(n_hedge, market_dim, n_static)

    def dim_names(self):
        """Generic per-dim labels for diagnostics — index-derived, no factor semantics."""
        nh = self.n_hedge
        return (tuple(f"position_{i}" for i in range(nh)) + ("cash", "accrued")
                + tuple(f"tradable_price_{i}" for i in range(nh))
                + tuple(f"market_{i}" for i in range(self.market_dim))
                + tuple(f"static_{i}" for i in range(self.n_static)) + ("time_to_T",))


def _assemble_deep_state(positions, g, a, futures, market, static, time_to_t):
    """Concatenate the deep-state blocks along the last axis (StatePack layout order).
    Vector args (`positions`, `futures`, `market`, `static`) are `(..., k)`; scalar args
    (`g`, `a`, `time_to_t`) are `(...,)`. Leading dims broadcast to a common shape."""
    vecs = [positions, futures, market, static]
    scalars = [g, a, time_to_t]
    lead = torch.broadcast_shapes(
        *[v.shape[:-1] for v in vecs], *[s.shape for s in scalars])
    p, f, m, st = [v.expand(*lead, v.shape[-1]) for v in vecs]
    g_e, a_e, ti_e = [s.expand(lead).unsqueeze(-1) for s in scalars]
    return torch.cat([p, g_e, a_e, f, m, st, ti_e], dim=-1)


class ValueFunctionApproximator(torch.nn.Module):
    """Per-timestep value function `V̂(s) = β·φ(s) + MLP(s)` over the deep state. OLS
    branch is a ridge-regularized linear fit on the basis `[1, z, pos², pos⊗market]`
    (`pos⊗market` = the position×market outer product, flattened). The MLP is an optional
    zero-init residual head — built lazily and only when `train_steps > 0`, so OLS-only
    solves (the current default) carry no dead network. Inputs are standardized
    (per-column mean/std from the fit rows) for conditioning. Fit fresh per invocation.

    `forward` clamps the query to the fitted training hull. The true V is bounded
    (symlog-saturating) but a linear/quadratic basis extrapolates without bound, and the
    backward DP recursion compounds any off-hull inflation geometrically. Clamping makes
    V̂ flat outside its hull, keeping the recursion bounded."""

    # Module-level guard for the one-shot "tail saturation active" log.
    _tail_sat_logged = False

    def __init__(self, statepack, mlp_hidden, device,
                 tail_saturating_columns=(), tail_saturation_scale=1.0):
        super().__init__()
        self.pos_slice = statepack.pos_slice
        self.market_slice = statepack.market_slice
        self.deep_dim = statepack.deep_dim
        self.mlp_hidden = list(mlp_hidden)
        self.device = device
        self.mlp = None        # built lazily by fit() — skipped entirely in OLS-only mode
        self.beta = None
        self.r2 = None
        self.z_lo = None
        self.z_hi = None
        self.z_mean = None
        self.z_std = None
        # Tail-saturating columns: indices of the standardized deep-state vector to push
        # through tanh before the basis. Bounds what V̂ sees on load-bearing breach columns
        # — outer-MC training covers the small-σ core; inner-MC queries explore tails to
        # many σ where the linear basis has no training support. tanh saturates beyond
        # ~2σ, silencing the extrapolation. Empty = identity (off).
        # `tail_saturation_scale` (default 1.0) is the multiplier on the standardized input
        # before tanh: scale > 1 tightens the knee inward in raw units (more aggressive
        # saturation closer to the operating point), scale = 1 keeps the knee at 1σ.
        self.tail_saturating_columns = tuple(int(c) for c in tail_saturating_columns)
        self.tail_saturation_scale = float(tail_saturation_scale)
        if self.tail_saturating_columns and not ValueFunctionApproximator._tail_sat_logged:
            logging.info(
                'ValueFunctionApproximator: tail saturation ACTIVE on z columns %s '
                '(scale=%.3g)',
                list(self.tail_saturating_columns), self.tail_saturation_scale)
            ValueFunctionApproximator._tail_sat_logged = True

    def _build_mlp(self):
        """Zero-initialized residual head — `V̂ == β·φ` until the head is trained."""
        layers = []
        prev = self.deep_dim
        for h in self.mlp_hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.GELU()]
            prev = h
        final = torch.nn.Linear(prev, 1)
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = torch.nn.Sequential(*layers).to(self.device)

    def _basis(self, z):
        pos = z[..., self.pos_slice]
        market = z[..., self.market_slice]
        ones = torch.ones_like(z[..., :1])
        pos_market = (pos.unsqueeze(-1) * market.unsqueeze(-2)).flatten(-2)
        return torch.cat([ones, z, pos * pos, pos_market], dim=-1)

    def _basis_jacobian_rows(self, zn, grad_columns):
        """Per-row, per-differentiable-column basis Jacobian `dφ/dzn[b, c, :]` evaluated
        in STANDARDIZED space at `zn[b]` for each fit row `b` and each deep-state column
        `c ∈ grad_columns`. Returns `(N, K, P)` where `K = len(grad_columns)` and `P` is
        the basis dim.

        Operates in standardized space — caller is responsible for transforming raw
        gradient labels via `g_zn[b,c] = z_std[c] · g_raw[b,c]` so the LS constraint
        `β·(dφ/dzn)[b,c,:] = g_zn[b,c]` is dimensionally consistent with the value rows
        `β·φ(zn) = v` (both rhs have value-space scale, both J/φ rows have basis-coef
        scale O(1)). Working in standardized space avoids the `1/z_std` blowup on near-
        degenerate columns that would otherwise let the differential constraint dwarf
        the value constraint at any lambda_diff > 0.

        Tail-saturation chain rule remains: `dφ/dzn` on a tail-saturating column carries
        an extra `scale·sech²(scale·zn)` factor from the tanh squash applied pre-basis.

        Per-basis-block Jacobian in zn-space:
          intercept p=0          → 0 always
          linear p ∈ [1, 1+D)    → sat_chain[b,c] if p-1 == c else 0
          pos² p ∈ [1+D, 1+D+nh) → 2·zn[b,c]·sat_chain[b,c] if c == pos_slice[p-1-D]
                                   else 0
          pos⊗market p ∈ [...]   → zn[b, market[j]]·sat_chain[b,c] if c == pos[i],
                                   zn[b, pos[i]]·sat_chain[b,c]    if c == market[j],
                                   else 0
        where (i, j) indexes the pos⊗market flatten.

        Only emits constraints for `grad_columns` (columns with KNOWN gradient labels).
        """
        N = zn.shape[0]
        D = self.deep_dim
        nh = self.pos_slice.stop - self.pos_slice.start
        md = self.market_slice.stop - self.market_slice.start
        pos_start = self.pos_slice.start
        market_start = self.market_slice.start
        K = len(grad_columns)
        if K == 0:
            return zn.new_zeros((N, 0, 1 + D + nh + nh * md))

        # Tail-saturation chain rule (scale · sech²); identity elsewhere.
        sat_chain = torch.ones_like(zn)
        if self.tail_saturating_columns:
            cols = list(self.tail_saturating_columns)
            scale = self.tail_saturation_scale
            sat_chain[..., cols] = scale * (
                1.0 - torch.tanh(scale * zn[..., cols]) ** 2)

        rows = []
        for c in grad_columns:
            sc = sat_chain[:, c]                                    # (N,) — chain at col c
            intercept_row = zn.new_zeros((N, 1))
            linear_row = zn.new_zeros((N, D))
            linear_row[:, c] = sc
            pos_sq_row = zn.new_zeros((N, nh))
            if pos_start <= c < pos_start + nh:
                i_pos = c - pos_start
                pos_sq_row[:, i_pos] = 2.0 * zn[:, c] * sc
            pos_mkt_row = zn.new_zeros((N, nh * md))
            if pos_start <= c < pos_start + nh:
                i_pos = c - pos_start
                zn_market = zn[:, market_start:market_start + md]   # (N, md)
                pos_mkt_row[:, i_pos * md:(i_pos + 1) * md] = zn_market * sc.unsqueeze(-1)
            elif market_start <= c < market_start + md:
                j_mkt = c - market_start
                zn_pos = zn[:, pos_start:pos_start + nh]            # (N, nh)
                pos_mkt_row[:, j_mkt::md] = zn_pos * sc.unsqueeze(-1)
            rows.append(torch.cat([intercept_row, linear_row, pos_sq_row, pos_mkt_row],
                                  dim=-1))
        return torch.stack(rows, dim=1)                             # (N, K, P)

    def _apply_tail_saturation(self, zn):
        """tanh-saturate the configured columns of the standardized input. No-op when
        `tail_saturating_columns` is empty. Applied identically at fit and query so the
        OLS basis is consistent — otherwise V̂ would be biased. `tail_saturation_scale`
        multiplies the standardized input before tanh: the sech² curvature band moves to
        |z| ∈ [1/scale, 3/scale], shifting V̂'s nonlinearity toward (or away from) the
        operating point of the recovered policy."""
        if not self.tail_saturating_columns:
            return zn
        cols = list(self.tail_saturating_columns)
        out = zn.clone()
        out[..., cols] = torch.tanh(self.tail_saturation_scale * out[..., cols])
        return out

    def fit(self, z, v_target, train_steps, lr, *,
            v_grad_labels=None, grad_columns=None, lambda_diff=0.0):
        """Fit OLS β (per-column ridge — robust to basis columns of disparate norm) then
        the MLP on the OLS residual. Records the training hull + standardization stats.
        Returns the OLS R².

        Optional differential-ML labels (Huge–Savine twin loss): when `v_grad_labels` is
        provided and `lambda_diff > 0`, the OLS is solved as stacked LS
            ‖φ·β - v‖² + λ_diff · Σ_b Σ_{c ∈ grad_columns} ((dφ/dz)[b,c,:]·β - g[b,c])²
        where `g = v_grad_labels` has shape `(N, K)`, `K = len(grad_columns)`, and each
        gradient row enforces the basis prediction's c-th partial derivative at fit row b
        to match the autograd-supplied label. Only `grad_columns` (columns with KNOWN
        gradient labels) contribute — non-differentiable columns are left unconstrained,
        so the pos² and pos⊗market β components are not artificially zeroed out."""
        self.z_lo = z.amin(dim=0)
        self.z_hi = z.amax(dim=0)
        self.z_mean = z.mean(dim=0)
        std = z.std(dim=0)
        # Near-constant columns (a one-hot regime always/never occupied, the held-at-0
        # accrued slot) have ~0 std — standardizing them divides by ~eps and blows the
        # basis up (ill-conditioned Gram → garbage β). Leave those unscaled; the
        # intercept absorbs the constant.
        self.z_std = torch.where(std > 1.0e-6, std, torch.ones_like(std))
        zn = (z - self.z_mean) / self.z_std
        zn = self._apply_tail_saturation(zn)
        phi = self._basis(zn)                                             # (N, P)
        use_diff = (
            v_grad_labels is not None and grad_columns is not None
            and len(grad_columns) > 0 and lambda_diff > 0.0)
        if use_diff:
            # The Jacobian rows are evaluated in standardized space (no 1/z_std factor),
            # so the gradient labels must be transformed: `g_zn[b,c] = z_std[c] · g_raw[b,c]`
            # to keep the LS dimensionally consistent (both phi rows and J rows then have
            # O(1) basis-coef norm; near-degenerate columns don't blow up the LS).
            z_std_cols = self.z_std[torch.tensor(grad_columns, device=self.z_std.device)]
            g_zn = v_grad_labels * z_std_cols                              # (N, K)
            j_rows = self._basis_jacobian_rows(zn, grad_columns)           # (N, K, P)
            n, k, p = j_rows.shape
            w = float(lambda_diff) ** 0.5
            j_flat = (w * j_rows).reshape(n * k, p)
            y_grad_flat = (w * g_zn).reshape(n * k)
            phi_stacked = torch.cat([phi, j_flat], dim=0)
            y_stacked = torch.cat([v_target, y_grad_flat], dim=0)
        else:
            phi_stacked, y_stacked = phi, v_target
        gram = phi_stacked.transpose(-1, -2) @ phi_stacked
        # Scale-aware ridge + absolute floor.
        # Scale-aware part handles disparate column norms.
        # Absolute floor handles zero-variance columns (e.g. bilinear terms where
        # one factor is identically zero on the training set — happens when a
        # rare regime has zero occupancy at this t for this seed).
        diag = gram.diagonal()
        ridge_vec = 1.0e-6 * diag.clamp_min(1.0e-12) + 1.0e-8
        ridge = torch.diag(ridge_vec)

        # Log degenerate columns once if any — useful diagnostic
        n_degenerate = (diag < 1e-10).sum().item()
        if n_degenerate > 0:
            logging.debug(f"VFA fit: {n_degenerate}/{diag.numel()} basis columns near-zero variance")

        try:
            self.beta = torch.linalg.solve(
                gram + ridge, phi_stacked.transpose(-1, -2) @ y_stacked)
        except torch._C._LinAlgError:
            # Fallback: lstsq is rank-deficient-tolerant via SVD. Slower but bulletproof.
            # If we hit this, something is structurally wrong with the basis at this t/seed —
            # worth investigating but not worth crashing the whole sweep.
            logging.warning(f"VFA fit: gram singular even with ridge, falling back to lstsq")
            rhs = phi_stacked.transpose(-1, -2) @ y_stacked
            self.beta = torch.linalg.lstsq(gram + ridge, rhs.unsqueeze(-1)).solution.squeeze(-1)

        # R² and the MLP residual fit are computed on the VALUE rows only — the
        # differential constraint pulls β toward gradient-consistency, but the
        # value-fit quality is the headline diagnostic we report.
        ols_pred = phi @ self.beta
        ss_res = ((v_target - ols_pred) ** 2).sum()
        ss_tot = ((v_target - v_target.mean()) ** 2).sum()
        # R² is undefined for a (near-)constant target — when v_t is pinned at the
        # projection cap there is no variance to explain and 1 - ss_res/~0 explodes.
        # Report 1.0 (the basis intercept fits a constant) rather than a spurious
        # huge-negative value that masquerades as a broken fit.
        if v_target.std() < 1.0e-3 * v_target.abs().mean().clamp_min(1.0e-12):
            self.r2 = 1.0
        else:
            self.r2 = float(1.0 - ss_res / ss_tot.clamp_min(1.0e-12))
        if train_steps > 0:
            if self.mlp is None:
                self._build_mlp()
            residual = (v_target - ols_pred).detach()
            opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
            with torch.enable_grad():
                for _ in range(train_steps):
                    opt.zero_grad()
                    loss = ((self.mlp(zn).squeeze(-1) - residual) ** 2).mean()
                    loss.backward()
                    opt.step()
        return self.r2

    def forward(self, z, clamp=True):
        # `clamp=False` exposes the raw extrapolating surface — used only by the
        # extrapolation diagnostic, which must see V̂ *before* the hull clamp masks it.
        if clamp:
            z = torch.minimum(torch.maximum(z, self.z_lo), self.z_hi)
        zn = (z - self.z_mean) / self.z_std
        zn = self._apply_tail_saturation(zn)
        v = self._basis(zn) @ self.beta
        if self.mlp is not None:
            v = v + self.mlp(zn).squeeze(-1)
        return v


def advance_state(action, g_t, n_prev, inner, tradables_sim, static_sim, t, t_outer,
                   hedges, contract_size, kappa):
    """Build `state_{t+1}` deep state `(K, B_outer, B_inner, deep_dim)`, plus the running
    cash `G_{t+1}` `(K, B_outer, B_inner)`.

    `action` is `(K, M, n_hedge)` post-decision target inventory — `K` is the candidate-
    action axis; `M` is either 1 (a grid chunk: one action broadcast across all outer
    paths) or `B_outer` (the cross-fit re-score: an outer-aligned action). `g_t`/`n_prev`
    are the per-outer-path pre-decision cash/position. `G_{t+1} = G_t + Σ a_i·cs_i·
    (F_{t+1}-F_t) - turnover(a, n_prev)`. The market block is `inner["market_t1"]` —
    opaque, no factor-type knowledge here. `A_{t+1}` held at 0 (Milestone-2 deferral).
    `tradables_sim` / `static_sim` are history-stripped views; `t` is simulation-grid."""
    device = action.device
    n_h = len(hedges)
    f_t = torch.stack([tradables_sim[h][t].to(device) for h in hedges], dim=0)        # (n_h, B_outer)
    f_t1 = torch.stack([inner["F_t1"][h] for h in hedges], dim=0)                     # (n_h, B_outer, B_inner)
    dstep = f_t1 - f_t.unsqueeze(-1)                                                  # (n_h, B_outer, B_inner)

    b_outer = dstep.shape[1]
    scaled = (action * contract_size).expand(action.shape[0], b_outer, n_h)          # (K, B_outer, n_h)
    hedge_pnl = torch.einsum("kbi,ibn->kbn", scaled, dstep)                           # (K, B_outer, B_inner)
    cost = _turnover_cost(action - n_prev, kappa)                                     # (K, B_outer) — n_prev broadcasts M
    g_next = g_t.view(1, -1, 1) + hedge_pnl - cost.unsqueeze(-1)                       # (K, B_outer, B_inner)

    positions = action.unsqueeze(2)                                                  # (K, M, 1, n_h)
    a_next = torch.zeros_like(g_next)                                                 # A_t deferred (held 0)
    futures = f_t1.permute(1, 2, 0).unsqueeze(0)                                       # (1, B_outer, B_inner, n_h)
    market = inner["market_t1"].unsqueeze(0)                                           # (1, B_outer, B_inner, market_dim)
    n_static = static_sim.shape[-1]
    static = static_sim[min(t + 1, t_outer - 1)].to(device).view(1, 1, 1, n_static)
    time_to_t = torch.full((1, 1, 1), float(t_outer - 1 - (t + 1)), device=device)

    deep = _assemble_deep_state(positions, g_next, a_next, futures, market,
                                static, time_to_t)
    return deep, g_next


def _split_inner_axis(inner, sl, b_inner):
    """Slice the inner-MC dict's inner-path axis (dim 1, length `b_inner`) by `sl`. Used
    by the DP cross-fit: one half of the inner paths picks the argmax action, the disjoint
    other half scores it — decoupling selection from evaluation removes the winner's-curse
    max-bias that otherwise compounds geometrically backward.

    Generic by shape, not by key: any tensor carrying an inner axis (dim 1 == `b_inner`)
    is sliced, nested dicts are recursed, and everything else (outer-state tensors whose
    dim 1 is a market/feature count, scalars) passes through. New inner-MC outputs need
    no edit here. `b_inner` (config floor 128) never collides with a market/feature count."""
    def _slice(v):
        if torch.is_tensor(v):
            return v[:, sl] if v.dim() >= 2 and v.shape[1] == b_inner else v
        if isinstance(v, dict):
            return {k: _slice(x) for k, x in v.items()}
        return v
    return {k: _slice(v) for k, v in inner.items()}


def _corr(x, y):
    """Pearson correlation of two 1-D tensors; 0.0 if either is constant. Used to test
    the OLS residual for action-correlated structure — a residual that correlates with a
    position axis signals basis misspecification (a missing term), which Double-V cannot
    de-bias because it is systematic, not variance-driven."""
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(xc.norm() * yc.norm())
    return float(xc @ yc) / denom if denom > 0.0 else 0.0


def _extrapolation_diagnostic(vfa, state_train, deep_q, k=16, max_q=262144):
    """Diagnostic A — per-query extrapolation magnitude of a fitted V̂.

    For each query point `q` (an inner-MC-distributed deep state V̂ is evaluated at one
    step later), the `k` nearest training rows — Euclidean distance in V̂'s standardized
    space — define an in-hull anchor `r̄` = mean neighbour. The *signed* gap
    `V̂(q) - V̂(r̄)`, on the raw unclamped basis, measures how far V̂ has moved from where
    it was actually fit.

    Interpretation (the reason it is logged signed *and* absolute):
      - a heavy, positively-signed tail — especially on the off-hull subset — is the
        signature of distribution-mismatch inflation: V̂ extrapolating upward where
        `max_a` then exploits it. Distribution alignment would be the fix.
      - a roughly sign-symmetric, modest spread says V̂ is not systematically inflating
        off-hull; the residual overestimate is the max-exploits-fit mechanism instead,
        which this gap does not capture."""
    device = state_train.device
    q = deep_q.reshape(-1, deep_q.shape[-1])
    if q.shape[0] > max_q:
        q = q[torch.randperm(q.shape[0], device=device)[:max_q]]
    train_z = (state_train - vfa.z_mean) / vfa.z_std                  # z_std guarded in fit()
    r_bar = torch.empty_like(q)
    for lo in range(0, q.shape[0], 65536):
        qz = (q[lo:lo + 65536] - vfa.z_mean) / vfa.z_std
        nn = torch.cdist(qz, train_z).topk(k, dim=1, largest=False).indices   # (c, k)
        r_bar[lo:lo + 65536] = state_train[nn].mean(dim=1)            # raw-space mean neighbour
    extrap = vfa.forward(q, clamp=False) - vfa.forward(r_bar, clamp=False)     # signed
    off = ((q < vfa.z_lo) | (q > vfa.z_hi)).any(dim=-1)

    def _q(x, ps):
        return {f"p{int(p * 100)}": float(torch.quantile(x, p)) for p in ps}

    out = {
        "n_queries": int(q.shape[0]), "k": k,
        "frac_off_hull": float(off.float().mean()),
        "frac_positive": float((extrap > 0).float().mean()),
        "extrap_mean": float(extrap.mean()),
        "extrap_signed_pctiles": _q(extrap, (0.01, 0.1, 0.5, 0.9, 0.99)),
        "extrap_abs_pctiles": _q(extrap.abs(), (0.5, 0.9, 0.99, 1.0)),
    }
    for label, mask in (("offhull", off), ("inhull", ~off)):
        sub = extrap[mask]
        out[f"extrap_{label}"] = (
            {"frac": float(mask.float().mean()), "mean": float(sub.mean()),
             **_q(sub, (0.5, 0.9, 0.99))}
            if sub.numel() else {"frac": 0.0})
    return out


class TwinNetwork(torch.nn.Module):
    """Differential-ML post-decision continuation function `C_t : z_t → scalar` for the
    DifferentialSolver (`gate4_diffml_solver_spec.md`). Small MLP, **softplus** activation
    throughout so the network is C¹ — required for the twin loss's derivative-matching
    term to be well-defined everywhere on the input domain. **No internal max**: this is
    a smooth post-decision continuation function; the Bellman max is the explicit
    external operator (the decision operator at run/audit time and the `q*_{t+1}`
    argmax inside the value label generator) applied only to the **already-frozen**
    network. That moves-max-out-of-fit property is the architectural payoff over the
    LsmDpSolver postfix pathology.

    Standardization is built in: stored `z_mean`, `z_std`, `y_mean`, `y_std` buffers
    (refreshed once per backward step via `set_normalization` against the freshly-built
    training bank) ensure both the value and gradient losses live in O(1)-norm
    standardized space. Per spec §9:
        dy_norm/dz_norm = dy/dz · z_std / y_std
    keeps the differential constraint dimensionally consistent with the value
    constraint regardless of raw-unit scaling.
    """

    def __init__(self, deep_dim, hidden_sizes=(128, 128, 128), device=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        layers = []
        prev = deep_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.Softplus())
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.mlp = torch.nn.Sequential(*layers).to(device)
        # Normalization buffers — refreshed per backward step from the bank; survive
        # state_dict() saves so warm-start carries the standardization forward.
        self.register_buffer("z_mean", torch.zeros(deep_dim, device=device))
        self.register_buffer("z_std", torch.ones(deep_dim, device=device))
        self.register_buffer("y_mean", torch.zeros((), device=device))
        self.register_buffer("y_std", torch.ones((), device=device))
        self.deep_dim = deep_dim
        self.device = device

    def set_normalization(self, z_mean, z_std, y_mean, y_std):
        """Refresh standardization stats from the current bank's empirical moments.
        Near-constant columns (a one-hot regime that never fires at this t, the
        held-at-zero accrued slot) get z_std clamped to 1.0 — matches the convention
        in ValueFunctionApproximator.fit and prevents the 1/z_std blow-up the OLS path
        learned about the hard way (memory: diff-ml-standardized-jacobian)."""
        eps = 1.0e-6
        self.z_mean.copy_(z_mean.detach())
        self.z_std.copy_(torch.where(z_std > eps, z_std, torch.ones_like(z_std)).detach())
        self.y_mean.copy_(y_mean.detach() if y_mean.dim() == 0 else y_mean.detach().squeeze())
        y_std_clamped = torch.maximum(y_std if y_std.dim() == 0 else y_std.squeeze(),
                                       torch.tensor(eps, device=y_std.device))
        self.y_std.copy_(y_std_clamped.detach())

    def forward(self, z):
        """Returns `C_t(z)` in raw (un-standardized) units. Input `z` is the raw deep
        state; standardization is applied internally so callers pass raw vectors and
        receive raw values. Shape: `z` is `(..., deep_dim)`, output is `(...)`."""
        zn = (z - self.z_mean) / self.z_std
        yn = self.mlp(zn).squeeze(-1)
        return yn * self.y_std + self.y_mean


def twin_loss(net, z, y_target, dy_dz_target, *,
              action_gap=None, gap_threshold=None,
              w_val=1.0, w_diff=1.0):
    """Huge–Savine twin loss (spec §9):
        L = w_val · MSE(C(z), y) + w_diff · MSE(∂C/∂z, ∂y/∂z)
    computed entirely in standardized space so both terms have O(1)-norm contributions
    per row regardless of raw-unit scaling.

    Both `y_target` and `dy_dz_target` are passed in **raw units**; standardization
    happens internally against the network's stored buffers. The gradient prediction
    is computed via `torch.autograd.grad(..., create_graph=True)` over a standardized
    leaf input — that yields `∂yn_pred/∂zn` directly, which is what the standardized
    target is in:
        dy_norm/dz_norm = dy/dz · z_std / y_std        # per spec §9
    The `create_graph=True` keeps the second-order graph alive so the outer optimizer's
    `loss.backward()` reaches the network parameters.

    Action-gap derivative mask: at training rows near action-switch boundaries the
    bootstrap differential is a branchwise derivative and unreliable. Per spec §9, set
    `w_diff_per_row = clamp(action_gap / gap_threshold, 0, 1)` — the value label stays
    active through kinks while the slope target is down-weighted. Pass `action_gap` and
    `gap_threshold` together to activate; pass neither for an unmasked fit.

    Returns `(total_loss, diag_dict)`. `diag_dict` carries per-term losses for
    logging/early-stopping.
    """
    # Standardize inputs once; share the forward pass between value and gradient
    # terms by computing the network on a leaf-version of zn and reusing the result.
    zn_leaf = ((z - net.z_mean) / net.z_std).detach().requires_grad_(True)
    yn_pred = net.mlp(zn_leaf).squeeze(-1)

    # Value branch — standardized MSE.
    yn_target = (y_target - net.y_mean) / net.y_std
    val_resid = yn_pred - yn_target
    val_loss = (val_resid ** 2).mean()

    # Gradient branch — `∂yn_pred/∂zn` via autograd, compared to the standardized
    # gradient target. `create_graph=True` is mandatory: the outer backward needs the
    # path through this gradient computation to reach the network parameters.
    dyn_dzn_pred = torch.autograd.grad(
        yn_pred.sum(), zn_leaf, create_graph=True)[0]               # (N, D)
    dyn_dzn_target = dy_dz_target * (net.z_std / net.y_std)         # (N, D)

    diff_per_row = ((dyn_dzn_pred - dyn_dzn_target) ** 2).mean(dim=-1)  # (N,)
    if action_gap is not None and gap_threshold is not None and gap_threshold > 0:
        mask = (action_gap / float(gap_threshold)).clamp(0.0, 1.0)
        diff_loss = (diff_per_row * mask).mean()
        mask_mean = float(mask.mean())
    else:
        diff_loss = diff_per_row.mean()
        mask_mean = 1.0

    total = w_val * val_loss + w_diff * diff_loss
    return total, {
        "val_loss": float(val_loss.detach()),
        "diff_loss": float(diff_loss.detach()),
        "mask_mean": mask_mean,
    }


def construct_twin_network(deep_dim, runtime, device=None):
    """Factory for the diff-ML continuation function. Reads `Solver.Value_Fn.MLP_Hidden`
    for the layer widths; defaults to `(128, 128, 128)` per the Gate 4 build defaults.
    Device defaults to the bundle's primary device when one is reachable through
    `runtime`; callers pass it explicitly otherwise."""
    vf = runtime["solver"].get("value_fn", {})
    hidden = tuple(vf.get("mlp_hidden", (128, 128, 128)))
    return TwinNetwork(deep_dim, hidden_sizes=hidden, device=device)


class LsmDpSolver:
    """Least-squares-Monte-Carlo dynamic-programming solver — the deliverable. Streaming
    backward sweep: at each outer `t`, fork inner MC, search the discrete action grid
    against the fitted V̂_{t+1} (closed-form V_T at the terminal step), record (n*_t, V_t)
    per outer path, fit V̂_t, free the inner samples, move to t-1.

    Training rows for V̂_t sample the pre-decision inventory `n_prev(b)` (uniform over the
    action grid) and cash `G_t(b)` per outer path — the outer sim carries no policy, so it
    supplies only market state; the (position, cash) subspace the V̂ must generalize over
    is covered by sampling. `accrued_liability_A_t` is held at 0 (Milestone-2 deferral).

    The backward step is de-biased on two axes — `max_a mean_inner[V̂]` over noisy
    estimates is upward-biased (winner's curse) and the bias compounds geometrically
    backward:
      - **Inner cross-fit:** the inner paths are split; one half selects the argmax
        action, the disjoint half scores it — removes the inner-MC selection bias.
      - **Asymmetric Double-V (Hasselt Double-Q decoupling):** V̂_t is two heads. The
        step runs twice — pass 1 argmaxes with head A of V̂_{t+1} and evaluates the
        picked action with head B (→ V_t^(A→B), trains head A); pass 2 swaps the roles
        (→ V_t^(B→A), trains head B). Because the evaluating head never did the argmax,
        the fitted target is not maxed over its own action-surface noise — the bias
        inner splitting cannot touch (V̂ is the same function on both inner halves)."""

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime

    def solve(self):
        bundle, runtime = self.bundle, self.runtime
        device = bundle["time_grid_days"].device
        solver_cfg = runtime["solver"]
        vf_cfg = solver_cfg["value_fn"]
        acc = runtime["accounting"]
        hedges = list(runtime["names"]["hedges"])
        force_flat = acc["force_flat_at_end"]
        total_abs_limit = acc["total_position_abs_limit"]
        c = max(float(bundle.get("utility_scale", 1.0)), 1.0)

        # Range projection: V_t(s) = E[u_signed(W_T)|s] is a conditional expectation of a
        # bounded RV, so V_t ∈ [u_min, u_max]. A linear V̂ violates that under extrapolation
        # and max_a actively seeks the violation — projecting V̂ onto the bound enforces a
        # property the true V provably has. Bounds are the (alpha, 1-alpha) quantiles of the
        # reachable terminal-utility sample (robust to fat-tailed single-path outliers),
        # fixed once at the terminal step. alpha <= 0 disables the projection (ablation).
        alpha = float(solver_cfg.get("range_projection_alpha", 1.0e-3))
        project = alpha > 0.0
        u_min_proj, u_max_proj = float("-inf"), float("inf")
        lambda_return = float(solver_cfg.get("lambda_return", 0.0))
        lambda_diff = float(solver_cfg.get("lambda_diff", 0.0))
        differential_labels = lambda_diff > 0.0 and "inner_mc_grad_fn" in bundle
        if lambda_diff > 0.0 and "inner_mc_grad_fn" not in bundle:
            logging.warning(
                'LsmDpSolver: Lambda_Diff=%g requested but bundle has no '
                'inner_mc_grad_fn — running with value labels only', lambda_diff)
        n_h = len(hedges)

        tradables_sim, static_sim, t_outer = _bundle_sim_views(bundle)
        statepack = None        # built lazily — market_dim is known only once inner MC runs
        action_grid = build_action_grid(
            runtime, solver_cfg["training_action_grid_levels_per_axis"], device)
        n_actions = action_grid.shape[0]
        # Count reachable actions under the L1 position cap for diagnostic logging only.
        # n_prev training rows are NOT restricted to this set: V̂'s OLS basis needs
        # estimation-variance margin around the L1 boundary at |n|_1 = total_abs_limit,
        # which max_a queries densely once the value approximator starts clamp-saturating.
        # Pre-fix-2 (full-grid sampling) keeps the L1 boundary inside the convex hull of
        # the training support (which extends to |n|_1 ≈ 3 * action_grid.max() at the
        # three-corner); valid-only sampling makes the L1 boundary the hull boundary,
        # where OLS variance x'(X'X)^{-1}x is maximal. max_a then preferentially picks
        # the action whose V̂ is variance-inflated upward, hits the projection clamp,
        # and the backward recursion saturates.
        if total_abs_limit > 0.0:
            n_valid_actions = int((action_grid.abs().sum(dim=-1) <= total_abs_limit).sum())
        else:
            n_valid_actions = action_grid.shape[0]
        chunk_size = solver_cfg["training_action_chunk_size"]

        value_fns: Dict[int, ValueFunctionApproximator] = {}
        actions: Dict[int, Any] = {}
        values: Dict[int, Any] = {}
        r2_by_t: Dict[int, float] = {}
        diag_by_t: Dict[int, Dict[str, Any]] = {}
        # Diagnostic A: at a few outer-t's, hold V̂_t + its training rows so the next
        # backward step can measure V̂_t's extrapolation over its inner-MC query cloud.
        diag_extrap_state: Dict[int, Any] = {}
        diag_target_ts = {t_outer // 4, t_outer // 2, (3 * t_outer) // 4}

        with torch.no_grad():
            for t in range(t_outer - 2, -1, -1):
                # Differential branch: fork inner MC with AAD attached to per-process
                # state-at-t leaves. `torch.enable_grad()` re-enables autograd inside the
                # surrounding no_grad — same pattern as vfa.fit's Adam loop.
                if differential_labels:
                    with torch.enable_grad():
                        inner = bundle["inner_mc_grad_fn"](t)
                else:
                    inner = bundle["inner_mc_fn"](t)
                if inner.get("L_T") is None:
                    continue                          # terminal / past-end — no inner horizon
                L_T = inner["L_T"]                    # (B_outer, B_inner)
                b_outer = L_T.shape[0]
                if statepack is None:                 # market_dim known once inner MC runs
                    statepack = StatePack.from_bundle(
                        bundle, runtime, inner["market_t"].shape[-1])
                kappa, contract_size = _per_contract_kappa(
                    tradables_sim, runtime, hedges, t, device)

                # Reachable-G estimate: a bound on one-step hedge P&L over the action grid.
                # V̂_t is queried (at step t-1) at g_t plus one such step, so the fit rows
                # must sample G_t at least this wide — otherwise V̂ is evaluated outside its
                # fitted hull and the backward recursion blows up geometrically.
                f_t_h = torch.stack([tradables_sim[h][t].to(device) for h in hedges], dim=0)
                dstep = torch.stack([inner["F_t1"][h] for h in hedges], dim=0) - f_t_h.unsqueeze(-1)
                # Per-axis-sum bound on one-step hedge P&L over the action grid.
                # NOTE: the L1-dual bound `total_abs_limit * max_i(c_i * |dstep_i|)` is
                # mathematically tighter under the L1 position cap, but in practice
                # narrows V̂'s G-training support to the point where queries at the L1
                # position-boundary land near the edge of the OLS fit hull, where basis
                # estimation variance is maximal. Keeping the looser per-axis-sum bound
                # gives V̂ off-hull margin on the G axis at the cost of overprovisioning.
                per_axis_term = contract_size * dstep.abs().amax(dim=(1, 2))   # (n_hedge,)
                pnl_step = float((action_grid.abs().amax(dim=0) * per_axis_term).sum())
                g_halfwidth = max(4.0 * pnl_step, c)

                # Fit rows: sample per-outer-path pre-decision inventory + cash. At t=0 the
                # true decision state is flat (n_prev=0, G_0=0) — evaluate it exactly.
                # n_prev is drawn from the FULL action grid, not just reachable actions —
                # see comment at the top of the solve loop for the OLS edge-effect rationale.
                if t == 0:
                    n_prev = torch.zeros(b_outer, len(hedges), device=device)
                    g_t = torch.zeros(b_outer, device=device)
                else:
                    n_prev = action_grid[
                        torch.randint(0, action_grid.shape[0], (b_outer,), device=device)
                    ]
                    g_t = (torch.rand(b_outer, device=device) - 0.5) * (2.0 * g_halfwidth)
                zero_b = torch.zeros(b_outer, device=device)

                is_terminal_next = (t + 1 == t_outer - 1)
                vfa_next = value_fns.get(t + 1)        # None at terminal-next, else (head_A, head_B)

                # Cross-fit split: half the inner paths select the argmax action, the
                # disjoint other half scores it (`_split_inner_axis` slices by shape).
                # On the differential path: re-enable grad so the slice preserves the
                # autograd connection back to the inner-MC leaves (slicing inside the
                # outer no_grad would strip requires_grad).
                b_inner = L_T.shape[1]
                half = b_inner // 2
                split_ctx = (torch.enable_grad if differential_labels
                             else torch.no_grad)
                with split_ctx():
                    inner_A = _split_inner_axis(inner, slice(0, half), b_inner)
                    inner_B = _split_inner_axis(inner, slice(half, 2 * half), b_inner)

                # Instrumentation: objective magnitude over the grid (pre-argmax), the
                # per-component deep-state hull queried against V̂_{t+1}, and V̂_{t+1}'s
                # raw per-(action,inner) output range — `vhat_query_abs_max` >> V̂_{t+1}'s
                # training-target range is the signature of `max_a` exploiting
                # off-manifold V̂ extrapolation.
                obj_abs_max = torch.zeros((), device=device)
                vhat_query_abs_max = torch.zeros((), device=device)
                query_lo = torch.full((statepack.deep_dim,), float("inf"), device=device)
                query_hi = torch.full((statepack.deep_dim,), float("-inf"), device=device)
                terminal_u_pool = []        # reachable u_signed sample — terminal step only

                def objective_fn(action, inner_set, vhead):            # action (K, M, n_hedge)
                    nonlocal obj_abs_max, vhat_query_abs_max, query_lo, query_hi
                    deep_next, g_next = advance_state(
                        action, g_t, n_prev, inner_set, tradables_sim,
                        static_sim, t, t_outer, hedges, contract_size, kappa)
                    if is_terminal_next:
                        unwind = (_turnover_cost(action, kappa) if force_flat
                                  else torch.zeros(action.shape[:2], device=device))
                        pnl_excess_t = g_next - unwind.unsqueeze(-1)    # (K, B_outer, B_inner)
                        v = _utility_wrap_signed(
                            pnl_excess_t + inner_set["L_T"].unsqueeze(0), runtime)
                        # Subsample per chunk so the pooled terminal-u sample stays
                        # bounded at any (B_outer, B_inner, n_actions) scale — ~200k per
                        # chunk is ample for the (alpha, 1-alpha) projection quantiles.
                        # Filter to valid actions so the projection quantiles reflect
                        # only positions reachable under the position-limit constraint.
                        if total_abs_limit > 0.0:
                            valid = action.abs().sum(dim=-1) <= total_abs_limit
                            valid_mask = valid.unsqueeze(-1).expand_as(v)
                            if valid_mask.any():
                                filtered = v[valid_mask].reshape(-1)
                                stride = max(1, filtered.numel() // 200_000)
                                terminal_u_pool.append(filtered.detach()[::stride])
                        else:
                            flat = v.detach().reshape(-1)
                            terminal_u_pool.append(flat[::max(1, flat.numel() // 200_000)])
                    else:
                        # Asymmetric Double-V (variant 2): a single V̂_{t+1} head scores
                        # the whole batch — the caller passes the selecting head on the
                        # argmax pass and the *other*, disjoint head on the evaluation
                        # pass, so the fitted target is never maxed over its own noise.
                        # V̂ standardizes its input internally — pass the raw deep state.
                        v_boot = vhead(deep_next)
                        vhat_query_abs_max = torch.maximum(vhat_query_abs_max, v_boot.abs().max())
                        if project:
                            v_boot = v_boot.clamp(u_min_proj, u_max_proj)
                        flat_dn = deep_next.reshape(-1, deep_next.shape[-1])
                        query_lo = torch.minimum(query_lo, flat_dn.amin(dim=0))
                        query_hi = torch.maximum(query_hi, flat_dn.amax(dim=0))
                        if lambda_return > 0.0:
                            # λ-return target: blend bootstrap with a buy-and-hold rollout
                            # to T_dec, computed from inner-MC outputs (L_T, dF_T). At λ=1
                            # the fitted V̂ is never queried inside the target — strips the
                            # recursive max-over-V̂-extrapolation pathology entirely.
                            # Approximation: "frozen-policy rollout" is replaced by "buy-and-
                            # hold from t with the chosen action" — cheaper, value-only, no
                            # autograd through future-policy decisions.
                            cost_kb = _turnover_cost(action - n_prev, kappa)     # (K, B_outer)
                            dF_T_h = torch.stack(
                                [inner_set["dF_T"][h] for h in hedges], dim=0)   # (n_h, B_outer, B_inner)
                            scaled_a = (action * contract_size).expand(
                                action.shape[0], b_outer, n_h)                   # (K, B_outer, n_h)
                            roll_pnl = torch.einsum(
                                "kbi,ibn->kbn", scaled_a, dF_T_h)                # (K, B_outer, B_inner)
                            post_cash = g_t.view(1, -1, 1) - cost_kb.unsqueeze(-1)
                            wealth_T_bh = post_cash + roll_pnl + inner_set["L_T"].unsqueeze(0)
                            v_roll = _utility_wrap_signed(wealth_T_bh, runtime)
                            v = v_boot + lambda_return * (v_roll - v_boot)
                        else:
                            v = v_boot
                    out = v.mean(dim=-1)                               # (K, B_outer)
                    obj_abs_max = torch.maximum(obj_abs_max, out.abs().max())
                    return out

                # Asymmetric Double-V (variant 2) — run the backward step twice.
                #   Pass 1 (A→B): head A argmaxes on inner set A, head B scores the
                #     picked action on the disjoint inner set B → V_t^(A→B), trains A.
                #   Pass 2 (B→A): the heads AND the inner sets both swap → V_t^(B→A),
                #     trains B.
                # Swapping the inner sets injects independent MC noise into the two
                # targets — without it the heads, seeded and fit identically (OLS is
                # deterministic), never diverge and the Double-V collapses to one head.
                # `vfa_next` is None at the terminal-next step (V_{t+1} closed-form);
                # objective_fn ignores `vhead` there, but the two passes still differ
                # because they score on disjoint inner sets, so the heads diverge from
                # the first backward step.
                head_A = None if vfa_next is None else vfa_next[0]
                head_B = None if vfa_next is None else vfa_next[1]
                n_star_AB, v_sel_AB = search_action_grid(
                    action_grid, lambda a: objective_fn(a.unsqueeze(1), inner_A, head_A),
                    chunk_size=chunk_size, total_abs_limit=total_abs_limit)
                # The cross-fit-evaluator score for the selected action is the value
                # target for the OTHER head. On the differential path it must be
                # autograd-attached to the inner-MC leaves so autograd.grad below works;
                # the surrounding torch.no_grad() suppresses tracking by default. The
                # discrete argmax above stays under no_grad (no gradient through actions).
                eval_ctx = (torch.enable_grad if differential_labels
                            else torch.no_grad)
                with eval_ctx():
                    v_t_AB = objective_fn(
                        n_star_AB.unsqueeze(0), inner_B, head_B).squeeze(0)
                n_star_BA, v_sel_BA = search_action_grid(
                    action_grid, lambda a: objective_fn(a.unsqueeze(1), inner_B, head_B),
                    chunk_size=chunk_size, total_abs_limit=total_abs_limit)
                with eval_ctx():
                    v_t_BA = objective_fn(
                        n_star_BA.unsqueeze(0), inner_A, head_A).squeeze(0)
                # Differential ML: extract pathwise gradients ∂v_t/∂state_t through the
                # AAD inner-MC BEFORE detaching v_t_AB/v_t_BA. The grads live on the per-
                # process leaves published in inner['state_t_leaves']; only leaves whose
                # raw shape == privileged-block width (1:1 column mapping) can be
                # projected into the deep-state grad_columns — see the projection block
                # below.
                grad_columns, v_grad_labels_AB, v_grad_labels_BA = (), None, None
                if differential_labels and "state_t_leaves" in inner:
                    leaves = inner["state_t_leaves"]
                    leaf_widths = inner["state_t_leaf_widths"]
                    leaf_keys = [k for k, _ in leaf_widths]
                    leaf_list = [leaves[k] for k in leaf_keys]
                    with torch.enable_grad():
                        g_AB = torch.autograd.grad(
                            v_t_AB.sum(), leaf_list,
                            retain_graph=True, allow_unused=True)
                        g_BA = torch.autograd.grad(
                            v_t_BA.sum(), leaf_list,
                            retain_graph=False, allow_unused=True)
                    # Walk the market block in iteration order; pick out per-leaf grads
                    # whose raw shape lines up with the privileged-block width (i.e. a
                    # 1:1 mapping: ForwardRate per-tenor, CommodityBasis raw scalar).
                    # CommodityPrice/HMM (leaf=raw spot vs priv=regime one-hot) and
                    # ForwardPrice/InterestRate (priv-empty) are SKIPPED — recorded only
                    # in the diagnostic g-norm log.
                    market_offset = statepack.market_slice.start
                    f_t_col_start = market_offset - n_h       # tradable_prices block start
                    cols = []
                    g_AB_cols = []
                    g_BA_cols = []
                    g_norms = {}
                    col = market_offset
                    # Spot-channel gradient lands on the tradable_prices columns rather
                    # than on the market block (whose CommodityPrice slot is the regime
                    # one-hot — non-differentiable w.r.t. raw spot). For a single-spot
                    # setup, the inner-MC fork-init leaf at t IS the outer spot at t, so
                    # ∂v_t/∂spot_leaf ≡ ∂v_t/∂spot_t. Each tradable's F_t is a function
                    # of that spot — to first order ∂F_t[i]/∂spot ≈ 1 for short tenors;
                    # projecting g_spot uniformly across all F_t columns captures the
                    # dominant sensitivity. Multi-spot setups would need a per-tradable
                    # underlying mapping; deferred (the gate3 toy has a single
                    # CommodityPrice).
                    spot_grads_AB, spot_grads_BA = [], []
                    for (key, priv_w), gA, gB, leaf in zip(
                            leaf_widths, g_AB, g_BA, leaf_list):
                        leaf_raw_w = leaf.numel() // b_outer
                        if gA is not None:
                            g_norms[str(key.type)] = float(gA.abs().mean())
                        if priv_w > 0 and leaf_raw_w == priv_w and gA is not None:
                            # 1:1 market-block mapping (ForwardRate per-tenor,
                            # CommodityBasis scalar).
                            gA2 = gA.reshape(priv_w, b_outer)
                            gB2 = (gB if gB is not None
                                   else torch.zeros_like(gA)).reshape(priv_w, b_outer)
                            for i in range(priv_w):
                                cols.append(col + i)
                                g_AB_cols.append(gA2[i].detach())
                                g_BA_cols.append(gB2[i].detach())
                        elif key.type == 'CommodityPrice' and gA is not None:
                            # Spot channel: project to tradable_prices columns below.
                            spot_grads_AB.append(gA.detach())
                            spot_grads_BA.append((gB if gB is not None
                                                  else torch.zeros_like(gA)).detach())
                        col += priv_w
                    if spot_grads_AB:
                        # Aggregate spot gradients across all CommodityPrice leaves (toy
                        # case: one); broadcast to every F_t column.
                        g_spot_AB = torch.stack(spot_grads_AB, dim=0).sum(dim=0)  # (B,)
                        g_spot_BA = torch.stack(spot_grads_BA, dim=0).sum(dim=0)
                        for i in range(n_h):
                            cols.append(f_t_col_start + i)
                            g_AB_cols.append(g_spot_AB)
                            g_BA_cols.append(g_spot_BA)
                    if cols:
                        grad_columns = tuple(cols)
                        v_grad_labels_AB = torch.stack(g_AB_cols, dim=1)  # (B_outer, K)
                        v_grad_labels_BA = torch.stack(g_BA_cols, dim=1)
                    inner["_diff_g_norms"] = g_norms
                    inner["_diff_active_columns"] = len(cols)
                # Detach v_t targets BEFORE any storage / fit — otherwise the inner-MC
                # autograd tape would survive across t iterations and balloon memory.
                v_t_AB = v_t_AB.detach()
                v_t_BA = v_t_BA.detach()
                # n_star reported as a discrete grid action — pick the better
                # cross-fit pass. Value estimate continues to use the cross-fit average.
                chose_AB = float(v_t_AB.mean()) >= float(v_t_BA.mean())
                n_star = n_star_AB if chose_AB else n_star_BA
                v_t = 0.5 * (v_t_AB + v_t_BA)
                actions[t], values[t] = n_star, v_t
                diag_by_t[t] = {
                    "v_t_abs_max": float(v_t.abs().max()),
                    "v_t_mean": float(v_t.mean()),
                    "obj_abs_max_pre_argmax": float(obj_abs_max),
                    "vhat_query_abs_max": float(vhat_query_abs_max),
                    "g_t_train_lo": float(g_t.min()),
                    "g_t_train_hi": float(g_t.max()),
                    # Winner's curse per pass: the selecting head's self-maxed value
                    # minus the disjoint head's honest evaluation of the same action.
                    # This is the upward max-bias the Double-Q decoupling strips out
                    # of the fitted target.
                    "asym_gap_AB_mean": float((v_sel_AB - v_t_AB).mean()),
                    "asym_gap_BA_mean": float((v_sel_BA - v_t_BA).mean()),
                    # Head disagreement: the two cross-fit estimates of V_t. Small =>
                    # the heads agree and the backward recursion is stable.
                    "head_disagree_mean": float((v_t_AB - v_t_BA).mean()),
                    "head_disagree_std": float((v_t_AB - v_t_BA).std()),
                    "chosen_pass": "AB" if chose_AB else "BA",
                }
                # Hull/query diagnostic: pair the deep-state region queried against
                # V̂_{t+1} during this step's action search with V̂_{t+1}'s training
                # hull (recorded when V̂_{t+1} was fit). A query that exceeds the hull
                # on a component means max_a is extrapolating, not interpolating.
                if not is_terminal_next and (t + 1) in diag_by_t:
                    q_lo, q_hi = query_lo.tolist(), query_hi.tolist()
                    diag_by_t[t + 1]["query_hull_lo"] = q_lo
                    diag_by_t[t + 1]["query_hull_hi"] = q_hi
                    if "train_hull_lo" in diag_by_t[t + 1]:
                        t_lo = diag_by_t[t + 1]["train_hull_lo"]
                        t_hi = diag_by_t[t + 1]["train_hull_hi"]
                        names = statepack.dim_names()
                        breaches = []
                        for i in range(statepack.deep_dim):
                            span = max(t_hi[i] - t_lo[i], 1.0e-12)
                            over = max(t_lo[i] - q_lo[i], q_hi[i] - t_hi[i], 0.0) / span
                            if over > 0.0:
                                breaches.append([names[i], over])
                        diag_by_t[t + 1]["hull_breaches"] = sorted(
                            breaches, key=lambda kv: -kv[1])

                # Diagnostic A — extrapolation magnitude of V̂_{t+1} over the inner-MC
                # query cloud this step actually fed it (the A→B-selected action's
                # advance). Runs only at the few `diag_target_ts`.
                if (t + 1) in diag_extrap_state and (t + 1) in diag_by_t:
                    vfa_diag, state_train = diag_extrap_state[t + 1]
                    deep_q, _ = advance_state(
                        n_star_AB.unsqueeze(0), g_t, n_prev, inner_B, tradables_sim,
                        static_sim, t, t_outer, hedges, contract_size, kappa)
                    e = _extrapolation_diagnostic(vfa_diag, state_train, deep_q)
                    diag_by_t[t + 1]["extrapolation"] = e
                    sp = e["extrap_signed_pctiles"]
                    logging.info(
                        'LsmDpSolver extrap-diag t=%d: off_hull=%.2f frac_pos=%.2f '
                        'mean=%.4g signed[p10,p50,p90]=[%.4g,%.4g,%.4g] '
                        'off_hull_mean=%.4g in_hull_mean=%.4g',
                        t + 1, e["frac_off_hull"], e["frac_positive"], e["extrap_mean"],
                        sp["p10"], sp["p50"], sp["p90"],
                        e["extrap_offhull"].get("mean", 0.0),
                        e["extrap_inhull"].get("mean", 0.0))

                # Fix the projection bounds once, from the reachable terminal-utility
                # sample. (alpha, 1-alpha) quantiles — robust to fat-tailed single-path
                # outliers contaminating a literal sample max/min.
                if is_terminal_next and project:
                    pool = torch.cat(terminal_u_pool)
                    # torch.quantile rejects inputs above 2**24 elements — strided-cap
                    # the pool well under that (a few M samples set a 1e-3 quantile fine).
                    if pool.numel() > 8_000_000:
                        pool = pool[::pool.numel() // 8_000_000 + 1]
                    u_min_proj = float(torch.quantile(pool, alpha))
                    u_max_proj = float(torch.quantile(pool, 1.0 - alpha))
                    logging.info('LsmDpSolver: range projection [%.4g, %.4g] '
                                 '(alpha=%.1e, %d terminal-u samples)',
                                 u_min_proj, u_max_proj, alpha, pool.numel())
                    if total_abs_limit > 0.0:
                        logging.info(
                            'LsmDpSolver: terminal pool from %d/%d valid actions (%.1f%%)',
                            n_valid_actions, n_actions,
                            100.0 * n_valid_actions / n_actions)

                logging.info(
                    'LsmDpSolver: V̂_t=%d fit on %d rows, n_prev sampled from full '
                    'action grid (%d points; %d reachable under L1 cap)',
                    t, b_outer, n_actions, n_valid_actions)

                # Fit V̂_t: state_t carries the sampled (n_prev, G_t) + outer-realized
                # market state at t + tradable prices + static descriptors + time-to-T.
                f_t = f_t_h.transpose(0, 1)                       # (B_outer, n_hedge)
                n_static = static_sim.shape[-1]
                static_t = static_sim[t].to(device).expand(b_outer, n_static)
                time_to_t = torch.full((b_outer,), float(t_outer - 1 - t), device=device)
                state_t = _assemble_deep_state(
                    n_prev, g_t, zero_b, f_t, inner["market_t"], static_t, time_to_t)
                # Asymmetric Double-V (variant 2): head A is fit on V_t^(A→B), head B on
                # V_t^(B→A) — each head's target value is the *other* head's evaluation,
                # so neither head ever trains on a target maxed over its own noise.
                tail_sat = vf_cfg.get("tail_saturating_columns", ())
                tail_sat_scale = vf_cfg.get("tail_saturation_scale", 1.0)
                vfa_A = ValueFunctionApproximator(
                    statepack, vf_cfg["mlp_hidden"], device,
                    tail_saturating_columns=tail_sat,
                    tail_saturation_scale=tail_sat_scale)
                vfa_B = ValueFunctionApproximator(
                    statepack, vf_cfg["mlp_hidden"], device,
                    tail_saturating_columns=tail_sat,
                    tail_saturation_scale=tail_sat_scale)
                r2_A = vfa_A.fit(state_t, v_t_AB,
                                 vf_cfg["mlp_train_steps_per_solve"], vf_cfg["mlp_adam_lr"],
                                 v_grad_labels=v_grad_labels_AB,
                                 grad_columns=grad_columns,
                                 lambda_diff=lambda_diff if differential_labels else 0.0)
                r2_B = vfa_B.fit(state_t, v_t_BA,
                                 vf_cfg["mlp_train_steps_per_solve"], vf_cfg["mlp_adam_lr"],
                                 v_grad_labels=v_grad_labels_BA,
                                 grad_columns=grad_columns,
                                 lambda_diff=lambda_diff if differential_labels else 0.0)
                value_fns[t], r2_by_t[t] = (vfa_A, vfa_B), min(r2_A, r2_B)
                # Record V̂_t's training hull — paired against next step's query range.
                diag_by_t[t]["train_hull_lo"] = state_t.amin(dim=0).tolist()
                diag_by_t[t]["train_hull_hi"] = state_t.amax(dim=0).tolist()
                if t in diag_target_ts:
                    diag_extrap_state[t] = (vfa_A, state_t)
                # OLS residual vs every deep-state dim — locates basis misspecification:
                # a residual correlated with a dim is systematic (not variance), and no
                # cross-fit removes it — that dim's basis would need another term.
                resid = v_t_AB - vfa_A.forward(state_t)
                resid_corr = [_corr(resid, state_t[:, i])
                              for i in range(statepack.deep_dim)]
                worst = max(range(statepack.deep_dim), key=lambda i: abs(resid_corr[i]))
                dim_names = statepack.dim_names()
                # Basis layout (see ValueFunctionApproximator._basis):
                #   [0]                                            ones (intercept)
                #   [1 : 1+deep_dim]                               linear in z; pos linear
                #                                                  at [1 : 1+n_hedge]
                #   [1+deep_dim : 1+deep_dim+n_hedge]              pos * pos (per-leg
                #                                                  quadratic; sign tells us
                #                                                  concave vs convex in
                #                                                  inventory)
                #   [1+deep_dim+n_hedge : end]                     pos_market (n_hedge ×
                #                                                  market_dim, flattened)
                # All on standardized z, so coefficients are in standardized units. Sign of
                # the pos² block is the key diagnostic for "is V̂ concave in inventory":
                # negative → interior max exists; positive → max_a pushes to L1 boundary.
                nh = statepack.n_hedge
                dd = statepack.deep_dim
                beta_pos_lin = vfa_A.beta[1 : 1 + nh].detach().cpu()
                beta_pos_sq = vfa_A.beta[1 + dd : 1 + dd + nh].detach().cpu()
                beta_pos_market = vfa_A.beta[1 + dd + nh :].detach().cpu()
                diag_by_t[t]["dVhat_dG_normalized"] = float(vfa_A.beta[1 + nh])
                diag_by_t[t]["beta_pos_lin"] = beta_pos_lin.tolist()
                diag_by_t[t]["beta_pos_sq"] = beta_pos_sq.tolist()
                diag_by_t[t]["beta_pos_market_absmean"] = float(beta_pos_market.abs().mean())
                diag_by_t[t]["beta_pos_market_absmax"] = float(beta_pos_market.abs().max())
                diag_by_t[t]["ols_r2"] = min(r2_A, r2_B)
                diag_by_t[t]["resid_corr"] = resid_corr
                diag_by_t[t]["resid_corr_worst"] = (dim_names[worst], resid_corr[worst])
                if differential_labels:
                    n_active = inner.get("_diff_active_columns", 0)
                    g_norms = inner.get("_diff_g_norms", {})
                    diag_by_t[t]["differential_active_columns"] = n_active
                    diag_by_t[t]["differential_g_norms"] = g_norms
                    label_absmean = (float(v_grad_labels_AB.abs().mean())
                                     if v_grad_labels_AB is not None else 0.0)
                    diag_by_t[t]["differential_label_absmean"] = label_absmean
                    leaf_widths = inner.get("state_t_leaf_widths", [])
                    width_summary = ", ".join(
                        f"{k.type}:{w}" for k, w in leaf_widths)
                    logging.info(
                        'LsmDpSolver t=%d diff: active_cols=%d label_absmean=%.4g '
                        'g_norms=%s | leaf_widths=[%s]',
                        t, n_active, label_absmean,
                        {k: '%.3g' % v for k, v in g_norms.items()},
                        width_summary)
                d = diag_by_t[t]
                logging.info(
                    'LsmDpSolver t=%d: |V_t|max=%.3g V_t.mean=%.3g asym_gap=%.3g '
                    'head_disagree=%.3g Vhat_query|max=%.3g resid_worst=%s:%.3f R2=%.4f',
                    t, d["v_t_abs_max"], d["v_t_mean"],
                    0.5 * (d["asym_gap_AB_mean"] + d["asym_gap_BA_mean"]),
                    d["head_disagree_mean"], d["vhat_query_abs_max"],
                    dim_names[worst], resid_corr[worst], min(r2_A, r2_B))
                logging.info(
                    'LsmDpSolver t=%d beta: pos_lin=%s pos_sq=%s '
                    'pos_market(|.|mean=%.3g, |.|max=%.3g)',
                    t,
                    ['%+.3g' % b for b in beta_pos_lin.tolist()],
                    ['%+.3g' % b for b in beta_pos_sq.tolist()],
                    d["beta_pos_market_absmean"], d["beta_pos_market_absmax"])
                if min(r2_A, r2_B) < 0.7:
                    logging.warning(
                        'LsmDpSolver: OLS R^2=%.3f at t=%d (<0.7) — linear backbone weak',
                        min(r2_A, r2_B), t)

        ts = sorted(actions)
        n_star_0 = (actions[ts[0]].mean(dim=0) if ts
                    else torch.zeros(len(hedges), device=device))
        v0 = float(values[ts[0]].mean()) if ts else 0.0
        actions_stacked = (torch.stack([actions[t] for t in ts], dim=0) if ts
                           else torch.zeros(1, 1, len(hedges)))
        values_stacked = (torch.stack([values[t] for t in ts], dim=0) if ts
                          else torch.zeros(1, 1))
        return SolverResult(
            solver_name="LsmDpSolver",
            actions=actions_stacked.detach().cpu(),
            values=values_stacked.detach().cpu(),
            value_fn_artifacts={t: {"beta_A": vp[0].beta.detach().cpu(),
                                    "beta_B": vp[1].beta.detach().cpu(),
                                    "r2": r2_by_t[t]}
                                for t, vp in value_fns.items()},
            diagnostics={
                "n_star_0": n_star_0.detach().cpu().tolist(),
                "V_0": v0,
                "ols_r2_by_t": r2_by_t,
                "min_ols_r2": min(r2_by_t.values()) if r2_by_t else None,
                "backward_steps": len(ts),
                "action_grid_size": n_actions,
                "range_projection": {"enabled": project, "alpha": alpha,
                                     "u_min": u_min_proj, "u_max": u_max_proj},
                "units_diag_by_t": diag_by_t,
            },
        )


def _realized_paths(bundle, runtime):
    """Realized outer-path data the no-inner-MC tracks (hindsight, textbook) consume:
    `F` `(n_hedge, t_outer, B_outer)` hedge prices, `L_T` `(B_outer,)` the liability
    terminal MTM, and `t_outer`. The liability terminal mirrors the DP's `inner_mtm[-2]`
    convention — the pre-settlement terminal (index -1 is the appended clean-exit zero)."""
    tradables_sim, _, t_outer = _bundle_sim_views(bundle)
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
    tradables_sim, _, _ = _bundle_sim_views(bundle)
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
        tradables_sim, _, _ = _bundle_sim_views(bundle)
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


class DifferentialSolver:
    """Differential-ML dynamic-hedging solver per `gate4_diffml_solver_spec.md`. Casts each
    per-step value function as a smooth post-decision continuation `C_t : z_t → scalar`
    fit by supervised twin loss (value + AAD gradient labels) over a controlled bank,
    with the Bellman max moved **out** of the regression into an explicit external
    operator applied only to the already-frozen `C_{t+1}`. This removes the
    "max-inside-fit + extrapolation clamp" pathology that pins LsmDpSolver to off-
    distribution values (see `class LsmDpSolver` failure mode at `gate2`-toy weekly).

    **The one swappable seam** is `sample_exogenous(n, seed) -> trajectories[all t]`:
    a fat, no-policy forward sweep from the **true** t0 (path-count breadth on
    exogenous coordinates — no t0 perturbation; the argmax doesn't control them so
    the natural forward distribution IS their correct query distribution). Endogenous
    coordinates (inventory, wealth) are spanned across the designer box at each t
    slice inside the solver. The promotion path is (A) "promote `sample_exogenous`
    to a `HedgeMonteCarlo` bundle source" — the framework absorbs that one method;
    everything else (endogenous span, bootstrap label gen, audit rollouts) is
    permanent solver logic that the framework never owns.

    **Current status — Milestone 0**: cold-train `C_T` from inside the solver and exit.
    The bank is the realized terminal slice of the exogenous sweep; labels are the
    deal's closed-form terminal utility `U(W_T(z_T))` over all legal post-decision
    inventories. Backward sweep (Milestone 1+) deferred — the seam is validated here
    before any AAD bootstrap-label or audit machinery lands.
    """

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime
        self.device = bundle["time_grid_days"].device
        self.dtype = bundle["time_grid_days"].dtype
        # Calc handle (calculation.py: bundle['_calc_handle'] = self) — used by
        # `sample_exogenous` to read the already-populated `_outer_scenario_buffer` via
        # the canonical `_extract_outer_state_at` reader. NO monkey-patching.
        self._calc = bundle.get("_calc_handle")
        if self._calc is None:
            raise RuntimeError(
                "DifferentialSolver requires bundle['_calc_handle'] — check "
                "HedgeMonteCarlo.execute attaches it when Solver.Object='DifferentialSolver'.")
        self._outer_buf = self._calc._outer_scenario_buffer
        self._spot_key = self._calc._find_spot_key()

        # Solver config (JSON-driven).
        solver_cfg = runtime["solver"]
        vf_cfg = solver_cfg.get("value_fn", {})
        self.hidden_sizes = tuple(vf_cfg.get("mlp_hidden", (128, 128, 128)))
        self.train_steps_per_solve = int(vf_cfg.get("mlp_train_steps_per_solve", 2000))
        self.train_minibatch = int(vf_cfg.get("mlp_minibatch", 4096))
        self.adam_lr = float(vf_cfg.get("mlp_adam_lr", 1.0e-3))
        # Bank construction knobs (spec §6).
        bank_cfg = solver_cfg.get("bank_sampling", {})
        self.b_endo = int(bank_cfg.get("b_endo", 2))
        # Backward sweep depth — t_min = 0 is the full sweep (Milestone 3); M1.5
        # uses t_min = T_A − 1 to validate the multi-contract path across the T_A
        # transition without committing to the full 30-step sweep. Configurable via
        # JSON `Solver.T_Min`; default 0 = sweep all the way to the initial decision.
        self.t_min = int(solver_cfg.get("t_min", 0))

        self.hedges = list(runtime["names"]["hedges"])
        # T_outer-1 decision points: t=0..T-2 (DP convention). C[T_outer-1] is the
        # boundary anchor — fit to closed-form terminal utility, no bootstrap.
        tradables_sim, static_sim, t_outer = _bundle_sim_views(bundle)
        self._tradables_sim = tradables_sim
        self._static_sim = static_sim
        self.t_outer = t_outer

        # State pack — reuse the existing layout. Market block carries the regime
        # belief for the HMM toy; dim is whatever `_run_inner_mc_at_t` would publish.
        # For Milestone 0 we don't need the inner-MC machinery; only the terminal
        # slice of the outer buffer. Market dim derived from privileged-mode
        # extraction at t=0 (any t works — same per-process widths).
        market_t0 = self._read_privileged_market(t=0)
        market_dim = market_t0.shape[-1]
        self.statepack = StatePack.from_bundle(bundle, runtime, market_dim)

        # Trained continuation functions, one per backward index. None until fit.
        self.C = [None] * t_outer

        # --- Step C: is_live mask + action grid + accounting constants ---
        # `is_live[t, j] = True` iff hedge j is tradable at outer t. Derived from the
        # framework's `runtime['tradables'][h]['last_trade_date']` (hedge_runtime.py:425)
        # + the bundle's history-stripped `time_grid_days_cpu` (torchrl_hedge.py:2124)
        # + `bundle['meta']['base_date']` (torchrl_hedge.py:2090). Convention matches
        # the existing per-step alive flag in hedge_features.py:524 (`current_day <
        # last_trade_day`). This drives the bank's endogenous-span dead-axis collapse
        # (force q_dead = 0) and the decision operator's action-grid restriction.
        hist = int(bundle.get('initial_time_index', 0))
        base_date = bundle['meta']['base_date']
        days = bundle['time_grid_days_cpu'][hist:]
        is_live = torch.zeros(self.t_outer, len(self.hedges),
                              dtype=torch.bool, device=self.device)
        for j, h in enumerate(self.hedges):
            ltd = runtime['tradables'][h]['last_trade_date']
            last_day = (pd.Timestamp(ltd) - pd.Timestamp(base_date)).days
            for ti, d in enumerate(days[:self.t_outer]):
                is_live[ti, j] = int(d) < last_day
        self.is_live = is_live

        # Action grid — integer levels per axis, [Min_Position, Max_Position] per hedge.
        # Reuse the existing helper; spec §4 = ±5 per leg = 11 levels per axis →
        # 11^n_hedge candidates total (121 in the early window, 11 past T_A).
        levels = int(solver_cfg.get("training_action_grid_levels_per_axis", 11))
        self.action_grid = build_action_grid(runtime, levels, self.device)   # (K, n_h)

        # Toy accounting constants. Spec §3: λ_turn = 1e-4, K = 100 strike, S_0 = 100.
        # Promoted to JSON in M2/M3; hard-wired here so the smoke runs without a
        # config update. The wealth convention is gate2-style MTM-invariant:
        #   w_t = cash + (Σq − 1)·S_t + K
        # so under no rebalance: w_{t+1} = w_t + (Σq − 1)·dS; under rebalance:
        # w_post = w_pre − turnover.
        self.lambda_turn = float(runtime["accounting"].get(
            "transaction_cost_per_unit", 1.0e-4))
        self.K_strike = float(solver_cfg.get("k_strike", 100.0))
        self.contract_size = torch.ones(len(self.hedges), device=self.device)
        self.utility_c = max(float(bundle.get("utility_scale", 100.0)), 1.0)

        # ------- Advantage decomposition (spec §14 promoted to main path) -------
        # `C_t(z) = B_t(z) + A_t(z)` with `B_t` a closed-form buy-and-hold baseline
        # (analytic, bounded, action-dependent through Σq) and `A_t` the NN's
        # residual. Bounds magnitudes that otherwise compound geometrically through
        # the backward chain (observed +0.028 error-vs-depth slope at T_dec=30
        # without it; |δ| growing ~3×/step). Decision argmaxes B_t+A_t.
        self._advantage_decomp = bool(solver_cfg.get("use_advantage_decomp", True))
        # λ-mix: blend the bootstrap value target with a no-grad rollout target.
        # Spec §14 deferred lever. Indicated when a horizon-stable bounded residual
        # V_0 gap survives advantage decomposition (the gap exists at T_dec=30:
        # banked V_0 = 3.57 vs oracle 0.92 — bounded but persistent). Mixing in
        # the rollout pulls the value labels toward realized utility, breaking the
        # max-of-noisy compounding chain. Differential label is untouched (rollout
        # has no grad); only the value label is blended. Default 0 = no mix.
        self._lambda_mix = float(solver_cfg.get("lambda_mix", 0.0))
        if self._lambda_mix > 0.0:
            logging.info(
                "DifferentialSolver λ-mix ENABLED: λ=%.3f (value label = "
                "(1−λ)·Y_boot + λ·Y_rollout; gradient label unchanged)",
                self._lambda_mix)
        if self._advantage_decomp:
            # Pre-compute baseline parameters once. The drift comes from the spot
            # factor's per-state μ (annualised). dt comes from the spot's per-step
            # dt (constant for the weekly toy). The regime cols in the market block
            # are discovered by walking the same factor-iteration order as
            # `_read_privileged_market`'s privileged path — robust to factor reorder.
            spot_proc = self._calc.stoch_factors_inner.get(self._spot_key)
            mu_t = getattr(spot_proc, 'mu_per_state', None)
            if mu_t is None:
                self._advantage_decomp = False
                logging.warning(
                    "DifferentialSolver advantage decomp requested but spot factor "
                    "has no `mu_per_state` — falling back to the no-baseline path.")
            else:
                self._mu_per_regime = mu_t.to(self.device, dtype=self.dtype)
                # dt source: outer scenario grid (`time_grid_days_cpu`) is the
                # canonical authoritative source. Compute the OUTER per-step dt in
                # years from consecutive grid days, converted by 1/365.25 (calendar
                # convention — see memory `calendar-vs-business-day`). spot_proc's
                # `dt_per_step` reflects the INNER fork's dt and is zero on the
                # one-step diff-ML path; not what we want.
                days = bundle['time_grid_days_cpu'][hist:]
                # `time_grid_days_cpu` is a python list of day indices.
                if len(days) >= 2:
                    self._dt_per_step = float(int(days[1]) - int(days[0])) / 365.25
                else:
                    self._dt_per_step = 7.0 / 365.25      # weekly toy fallback
                # Discover regime col range in the market block by replicating
                # `_read_privileged_market`'s column accounting.
                col = 0
                self._regime_cols_in_market = None
                for key in self._calc.stoch_factors_inner:
                    if key.type in utils.DimensionLessFactors:
                        continue
                    width = self._calc._extract_outer_state_at(
                        0, key, self._outer_buf, privileged=True).shape[0]
                    if key == self._spot_key and width > 0:
                        self._regime_cols_in_market = (col, col + width)
                        break
                    col += width
                if self._regime_cols_in_market is None:
                    self._advantage_decomp = False
                    logging.warning(
                        "DifferentialSolver advantage decomp: couldn't locate spot "
                        "regime cols in market block — falling back to no-baseline.")
                else:
                    logging.info(
                        "DifferentialSolver advantage decomp ENABLED: "
                        "regime_cols=%s, μ_per_regime=%s, dt_per_step=%.6f",
                        self._regime_cols_in_market,
                        self._mu_per_regime.cpu().tolist(),
                        self._dt_per_step)

    def _baseline_B(self, z):
        """Closed-form buy-and-hold baseline `B_t(z)` for advantage decomposition.

        For deep state z at time t (with `time_to_T` encoded), this is the symlog
        utility of the EXPECTED terminal wealth under the policy 'hold current q
        from t through T':
            B_t(z) = symlog(z.wealth + (Σq − 1) · S · (E[S_T/S_t] − 1), c)
                   ≈ symlog(z.wealth + (Σq − 1) · S · (exp(μ̄ · ttot) − 1), c)
        where μ̄ is the expected per-time-unit drift given the current regime
        (one-hot or belief vector in the market block).

        Action-dependent through Σq (separates long/short policies), regime-
        dependent through μ̄ (encodes long-bull / short-bear), depth-dependent
        through ttot. Bounded by symlog(|max wealth| + |max Σq · S · drift|, c).
        AAD-safe — every op is differentiable in pytorch."""
        if not self._advantage_decomp:
            return torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        n_h = self.statepack.n_hedge
        market_dim = self.statepack.market_dim
        # StatePack layout: positions | wealth | accrued | futures | market | static | time
        positions = z[..., :n_h]
        wealth = z[..., n_h]
        futures = z[..., n_h + 2 : 2 * n_h + 2]
        market_start = 2 * n_h + 2
        market = z[..., market_start : market_start + market_dim]
        time_to_T = z[..., -1]

        sum_q = positions.sum(dim=-1)                                # (..., )
        S = futures[..., 0]                                          # spot tracker
        r_lo, r_hi = self._regime_cols_in_market
        regime_block = market[..., r_lo:r_hi]                        # (..., n_states)
        # Expected per-time drift = Σ_r regime_block[r] · μ_r (one-hot → picks μ).
        expected_mu = (regime_block * self._mu_per_regime).sum(dim=-1)
        ttot_years = time_to_T * self._dt_per_step
        drift_factor = torch.expm1(expected_mu * ttot_years)         # exp(x)-1, stable
        E_wealth_T = wealth + (sum_q - 1.0) * S * drift_factor
        return torch.sign(E_wealth_T) * torch.log1p(
            E_wealth_T.abs() / self.utility_c)

    def _C_full(self, net, z):
        """Evaluate the full continuation value `C_t(z) = B_t(z) + A_t(z)` where
        `A_t = net` is the NN's residual. Under advantage decomposition, the NN
        learns only the small bounded residual; under the legacy no-baseline mode
        (`use_advantage_decomp=False`), `B_t ≡ 0` and this collapses to `net(z)`.
        Both call sites that previously did `net(z)` should route through here so
        the decomposition is transparent to the rest of the solver."""
        nn_out = net(z)
        return nn_out + self._baseline_B(z)

    # ------------------------------------------------------------------ seam
    def sample_exogenous(self, n=None, seed=None):
        """**THE ONE SWAPPABLE METHOD** between (B) solver-internal and (A) framework-
        bundle-source variants. Returns a per-t trajectory dict with the canonical
        privileged exogenous coordinates ready for endogenous-span composition.

        For Milestone 0 (current): wraps `HedgeMonteCarlo._extract_outer_state_at` over
        the already-populated `_outer_scenario_buffer` — n is fixed at the existing
        outer batch size (re-sweep with fresh seed deferred to Milestone 2 / audit).
        For Milestone 2: a fresh-seeded resweep will run the canonical forward loop
        (`calculation.py:1887–1909`) via the calc handle.

        Returns: `{t: {'market': (B, market_dim), 'tradables': (B, n_hedge)}}` for
        every outer index t. The market block is the privileged extractor output
        (regime belief for the HMM toy, plus basis/carry where present). Tradable
        prices come from `tradables_sim[h][t]`. Both are read no-grad.
        """
        # n is ignored in Milestone 0 — the buffer is sized at the existing outer
        # batch and we don't re-sweep.
        del n, seed
        traj = {}
        with torch.no_grad():
            for t in range(self.t_outer):
                market_t = self._read_privileged_market(t)
                tradables_t = torch.stack(
                    [self._tradables_sim[h][t].to(self.device) for h in self.hedges],
                    dim=-1)                                # (B, n_hedge)
                traj[t] = {"market": market_t, "tradables": tradables_t}
        return traj

    def _read_privileged_market(self, t):
        """Privileged exogenous coordinates at outer t, B-last → B-leading. Concatenates
        per-process privileged blocks in `stoch_factors_inner` iteration order — same
        ordering the inner-MC `market_t` uses, so the StatePack market block is
        layout-consistent across diff-ML training and the existing inner-MC machinery.

        **For the gate2 toy comparison: TRUE-regime one-hot, NOT belief.** gate2 is a
        full-information DP whose state vector includes the observable regime
        coordinate (gate2_exact_dp.py:14, 28). The canonical
        `_extract_outer_state_at(..., privileged=True)` at calculation.py:2167-2181
        prefers the filtered belief `(key, 'regime_belief')` when the buffer holds it
        (per the spec's filter posterior convention) — but on the toy that turns the
        diff-ML solver into a partial-information solver whose value is strictly
        ≤ gate2's, and the cell-by-cell comparison silently fails on an information
        mismatch (not on the method). Solution: for regime-switching factors, read
        the realized regime path directly from `outer_buf[(spot_key, 'regimes')]`
        (published by `MarkovHMMSpotModel.generate` — destination-regime convention
        verified against gate2_exact_dp.py:123-134) and one-hot it. All other factor
        types fall through to the canonical privileged extractor.
        """
        parts = []
        for key in self._calc.stoch_factors_inner:
            if key.type in utils.DimensionLessFactors:
                continue
            proc = self._calc.stoch_factors_inner[key]
            regimes = self._outer_buf.get((key, 'regimes'))
            if (key.type in ('CommodityPrice', 'CommodityBasis')
                    and regimes is not None
                    and getattr(proc, 'n_states', None)):
                onehot = torch.nn.functional.one_hot(
                    regimes[t].long(), num_classes=proc.n_states
                ).to(dtype=self.dtype)                  # (B, n_states)
                block = onehot.movedim(-1, 0)            # (n_states, B) — B-last
            else:
                block = self._calc._extract_outer_state_at(
                    t, key, self._outer_buf, privileged=True).reshape(-1, self._B())
            parts.append(block)
        return torch.cat(parts, dim=0).permute(1, 0).contiguous()  # (B, market_dim)

    def _B(self):
        """Outer batch size — read off any factor's snapshot."""
        return self._outer_buf[self._spot_key].shape[-1]

    # ------------------------------------------------------------ C_T cold train
    def _train_C_terminal(self, traj):
        """Cold-train `C[t_outer-1]` against the deal's closed-form terminal utility
        over the realized exogenous slice at the LAST outer index. Endogenous
        coordinates (inventory, wealth) are spanned across the designer box so the
        NN learns U as a function of all deep-state coordinates the backward sweep
        will query later. No bootstrap, no AAD inner-MC — closed-form labels."""
        t_T = self.t_outer - 1
        market_T = traj[t_T]["market"]                     # (B, market_dim)
        tradables_T = traj[t_T]["tradables"]               # (B, n_hedge)
        b_outer = market_T.shape[0]
        n_h = len(self.hedges)
        device = self.device

        # Endogenous span: q ∈ [-5, 5] uniformly per leg, wealth uniformly across a
        # designer box matched to gate2's observed range (~K ± 1500 in raw units).
        # For Milestone 0 the band is hard-wired; promote to JSON in Milestone 1 once
        # the proper bank constructor lands.
        q_min, q_max = -5, 5
        wealth_halfwidth = 1500.0
        rows_per_outer = self.b_endo
        # Span: random uniform points in (q, wealth) per outer path. The NN learns
        # a smooth function across the full (z_T, q_T, wealth_T) joint.
        q_span = torch.randint(q_min, q_max + 1,
                                (b_outer, rows_per_outer, n_h),
                                device=device).to(self.dtype)
        wealth_span = (torch.rand(b_outer, rows_per_outer, device=device) - 0.5) \
            * (2.0 * wealth_halfwidth)

        # Broadcast market/tradables to (B, rows_per_outer, ...) and flatten.
        market_b = market_T.unsqueeze(1).expand(-1, rows_per_outer, -1).reshape(
            b_outer * rows_per_outer, -1)
        tradables_b = tradables_T.unsqueeze(1).expand(-1, rows_per_outer, -1).reshape(
            b_outer * rows_per_outer, -1)
        q_flat = q_span.reshape(b_outer * rows_per_outer, n_h)
        wealth_flat = wealth_span.reshape(-1)
        zero_b = torch.zeros_like(wealth_flat)
        time_to_T = torch.zeros_like(wealth_flat)
        static_T = self._static_sim[-1].to(device).expand(
            b_outer * rows_per_outer, self._static_sim.shape[-1])

        # Deep-state vector — reuse `_assemble_deep_state` for layout consistency.
        z_T = _assemble_deep_state(
            positions=q_flat, g=wealth_flat, a=zero_b, futures=tradables_b,
            market=market_b, static=static_T, time_to_t=time_to_T)

        # Terminal utility — symlog of wealth in the gate2 convention (w_T already
        # carries cash + (q_total - 1)·S + K → fully realized terminal wealth).
        c_util = max(float(self.bundle.get("utility_scale", 100.0)), 1.0)
        wealth_idx = self.statepack.n_hedge
        if self._advantage_decomp:
            # Under advantage decomp the NN fits the RESIDUAL A_T = C_T − B_T(z).
            # At terminal (`time_to_T = 0`), `B_T(z) = symlog(wealth)` exactly
            # (drift_factor = e^0 − 1 = 0), so A_T = symlog(W_T) − symlog(z.wealth)
            # = 0 identically. Both value and gradient targets collapse to 0; the
            # NN trains to ~zero output and the full continuation is recovered as
            # `B_T(z) + A_T(z) ≈ symlog(z.wealth) + 0`.
            y_target = torch.zeros_like(wealth_flat)
            dy_dz = torch.zeros_like(z_T)
        else:
            y_target = torch.sign(wealth_flat) * torch.log1p(wealth_flat.abs() / c_util)
            # Gradient label: ∂U/∂wealth = 1/(c + |w|); zero on other coords. The
            # wealth column is at StatePack offset n_hedge (after positions block).
            dy_dz = torch.zeros_like(z_T)
            dy_dz[:, wealth_idx] = 1.0 / (c_util + wealth_flat.abs())

        # Build and fit the twin network.
        net = construct_twin_network(
            self.statepack.deep_dim, self.runtime, device=device)
        net.set_normalization(
            z_T.mean(dim=0), z_T.std(dim=0),
            y_target.mean(), y_target.std())
        opt = torch.optim.Adam(net.parameters(), lr=self.adam_lr)

        n_rows = z_T.shape[0]
        batch = min(self.train_minibatch, n_rows)
        steps = max(1, self.train_steps_per_solve)              # guard against steps=0
        last_diag = {"val_loss": float("nan"), "diff_loss": float("nan"),
                     "mask_mean": 1.0}
        for step in range(steps):
            idx = torch.randint(0, n_rows, (batch,), device=device)
            loss, last_diag = twin_loss(
                net, z_T[idx], y_target[idx], dy_dz[idx],
                w_val=1.0, w_diff=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % max(1, steps // 5) == 0 or step == steps - 1:
                logging.info(
                    "DifferentialSolver C_T cold-train step=%d L=%.5f val=%.5f diff=%.5f",
                    step, float(loss),
                    last_diag["val_loss"], last_diag["diff_loss"])

        # Holdout check — closed-form value MAE on fresh spans.
        with torch.no_grad():
            n_check = min(2048, b_outer * rows_per_outer)
            check_idx = torch.randperm(n_rows, device=device)[:n_check]
            y_pred = net(z_T[check_idx])
            mae = (y_pred - y_target[check_idx]).abs().mean()
        logging.info(
            "DifferentialSolver C_T cold-train holdout MAE=%.5f over %d rows; "
            "twin val=%.4g diff=%.4g", float(mae), n_check,
            last_diag["val_loss"], last_diag["diff_loss"])

        self.C[t_T] = net
        return {
            "terminal_twin_value_loss": last_diag["val_loss"],
            "terminal_twin_diff_loss": last_diag["diff_loss"],
            "terminal_holdout_mae": float(mae),
            "terminal_bank_rows": int(n_rows),
        }

    # ----------------------------------------------- Step D: post_state + bank
    def _assemble_z(self, market, q, wealth, F, time_to_T, static):
        """Build a deep_state in StatePack layout from raw components. All-tensor
        broadcasting: leading dims compose. Used by both the bank construction (at t)
        and the bootstrap label gen (at t+1, where market/F come from the one-step
        inner-MC fork)."""
        zero = torch.zeros_like(wealth)
        return _assemble_deep_state(
            positions=q, g=wealth, a=zero, futures=F,
            market=market, static=static, time_to_t=time_to_T)

    def _turnover(self, q_new, q_prev, F):
        """L1 turnover cost: λ_turn · Σ_h |Δq_h| · F_h · contract_size. Per spec §3 /
        gate2's `lambda_turnover · |Δq| · S` (gate2_exact_dp.py:301). Caller broadcasts
        shapes — typically (..., n_h) for q's and F, returns (...,) cost."""
        dq = (q_new - q_prev).abs()
        cs = self.contract_size                                       # (n_h,)
        return self.lambda_turn * (dq * F * cs).sum(dim=-1)

    def _build_bank(self, t, traj, outer_indices=None, rows_per_outer=None):
        """Endogenous span on top of the realized exogenous slice at outer t.

        `outer_indices`: optional 1-D long tensor of outer-path indices to include.
        Defaults to all `B_outer` paths. M2 uses this to (a) exclude the audit slices
        from training and (b) oversample flagged buckets for correction rounds.
        `rows_per_outer`: defaults to `self.b_endo`; can be raised to densify the
        endogenous span within a flagged-bucket correction sample.

        Spec §6 forward × span + the live-axis collapse for dead hedges (rows where
        `is_live[t, j] = False` force `q[..., j] = 0`)."""
        market_t = traj[t]["market"]                                  # (B, market_dim)
        tradables_t = traj[t]["tradables"]                            # (B, n_hedge)
        device = self.device
        if outer_indices is None:
            outer_indices = torch.arange(market_t.shape[0], device=device)
        B = int(outer_indices.shape[0])
        n_h = len(self.hedges)

        rows = int(rows_per_outer if rows_per_outer is not None else self.b_endo)
        q_min, q_max = -5, 5
        wealth_halfwidth = 1500.0
        q_prev = torch.randint(q_min, q_max + 1, (B, rows, n_h),
                                device=device).to(self.dtype)
        wealth_pre = (torch.rand(B, rows, device=device) - 0.5) * \
            (2.0 * wealth_halfwidth)
        live_t = self.is_live[t]
        if (~live_t).any():
            mask_dead = (~live_t).view(1, 1, n_h)
            q_prev = torch.where(mask_dead, torch.zeros_like(q_prev), q_prev)

        # Gather the selected exogenous rows then broadcast across endogenous spans.
        market_sel = market_t[outer_indices]                          # (B, market_dim)
        tradables_sel = tradables_t[outer_indices]                    # (B, n_hedge)
        market_b = market_sel.unsqueeze(1).expand(-1, rows, -1).reshape(B * rows, -1)
        tradables_b = tradables_sel.unsqueeze(1).expand(-1, rows, -1).reshape(B * rows, -1)
        q_prev_flat = q_prev.reshape(B * rows, n_h)
        wealth_pre_flat = wealth_pre.reshape(-1)
        # `outer_index` is the index INTO THE OUTER MC BATCH (not into the bank's
        # selected subset) — needed for inner-MC alignment when the label generator
        # slices `inner['F_t1'][h]` / `market_t1` per row.
        outer_index = outer_indices.unsqueeze(1).expand(-1, rows).reshape(-1)
        time_to_T = torch.full((B * rows,), float(self.t_outer - 1 - t), device=device)
        static_b = self._static_sim[t].to(device).expand(
            B * rows, self._static_sim.shape[-1])
        return {
            "market": market_b, "F_at_t": tradables_b,
            "q_prev": q_prev_flat, "wealth_pre": wealth_pre_flat,
            "time_to_T": time_to_T, "static": static_b,
            "outer_index": outer_index,
            "rows_per_outer": rows, "B_outer": B,
        }

    # ---------------------------------------------- Step E: decision operator
    def _decision_at(self, C_next, market, F, q_prev, wealth_pre, time_to_T,
                     static, t):
        """Explicit external argmax `q* = argmax_q C_next(post_state(q))` over the
        live action grid, no_grad. Per spec §4 invariants: `C_next` is a post-decision
        function with NO internal max — the Bellman max is the EXPLICIT external
        operator applied only to the frozen `C_next`. Dead-axis trades forced to 0 by
        masking action_grid columns where `is_live[t, j] = False` (Reader-C: any
        non-zero q_dead at t > T_A would silently leak `−F_dead(T_A)` into wealth via
        advance_state-style chains).

        Returns:
          q_star : (N, n_hedge) — chosen post-decision inventory at t
          v_star : (N,)        — `C_next(post(q*))` at the chosen action
          action_gap : (N,)    — best minus second-best value (used for spec §9's
                                 derivative mask `w_diff = clamp(gap / threshold, 0, 1)`)
        """
        K, n_h = self.action_grid.shape
        N = market.shape[0]
        device = self.device
        # Live-axis collapse on the action grid — broadcast and force dead axes to 0.
        live_t = self.is_live[t]                                      # (n_h,)
        grid = self.action_grid.unsqueeze(0).expand(N, K, n_h).clone()      # (N, K, n_h)
        if (~live_t).any():
            grid[..., ~live_t] = 0.0

        # Broadcast components to (N, K, ...).
        market_K = market.unsqueeze(1).expand(N, K, -1)
        F_K = F.unsqueeze(1).expand(N, K, n_h)
        q_prev_K = q_prev.unsqueeze(1).expand(N, K, n_h)
        wealth_pre_K = wealth_pre.unsqueeze(1).expand(N, K)
        time_to_T_K = time_to_T.unsqueeze(1).expand(N, K)
        static_K = static.unsqueeze(1).expand(N, K, -1)

        # Turnover cost per (N, K) — spec §3's L1 model.
        cost = self._turnover(grid, q_prev_K, F_K)                    # (N, K)
        wealth_post_K = wealth_pre_K - cost                           # (N, K)

        # Build post-decision states and score against frozen C_next. NO autograd —
        # the decision operator's only output that flows into the differential label
        # is `q*` (detached). Value is read for ranking + action_gap diagnostic.
        with torch.no_grad():
            z_post = self._assemble_z(
                market=market_K.reshape(N * K, -1),
                q=grid.reshape(N * K, n_h),
                wealth=wealth_post_K.reshape(-1),
                F=F_K.reshape(N * K, n_h),
                time_to_T=time_to_T_K.reshape(-1),
                static=static_K.reshape(N * K, -1))                   # (N*K, deep_dim)
            # Advantage decomposition: argmax the SUM B_t + A_t, NOT A_t alone.
            # The decomposition is transparent: legacy no-baseline path collapses
            # `_C_full` to `C_next(z_post)` since `_baseline_B` returns zero.
            values = self._C_full(C_next, z_post).reshape(N, K)       # (N, K)
            argmax_idx = values.argmax(dim=-1)                        # (N,)
            v_star = values.gather(-1, argmax_idx.unsqueeze(-1)).squeeze(-1)
            # Action gap = top minus second-best. Robust to ties (sort then diff).
            top2, _ = values.topk(min(2, K), dim=-1)
            action_gap = (top2[..., 0] - top2[..., -1]).abs()         # (N,)

        # Gather q*: shape (N, n_h).
        q_star = grid.gather(
            1, argmax_idx.view(N, 1, 1).expand(N, 1, n_h)).squeeze(1)
        return q_star, v_star, action_gap

    def _compute_labels_for_bank(self, t, bank, C_next, traj=None):
        """Generate `(z_t, y_target, dy_dz, action_gap)` for a bank of pre-decision
        rows via the spec §4/§12 one-step-bootstrap pipeline:
          • sample q_t spanned across the action grid (per-row anchor)
          • assemble `z_t` post-decision after applying q_t
          • call `bundle['inner_mc_grad_fn_one_step'](t)` ONCE per call
            (AAD-attached `market_{t+1}` + per-process `state_t_leaves`)
          • propagate to t+1 (gate2 MTM-invariant wealth)
          • decision at t+1 via `argmax_q C_next(post_{t+1}(q))` (q* detached)
          • `Y_boot = C_next(post_{t+1}(q*))`
          • `dy_dz` via `autograd.grad(Y_boot, leaves)` + closed-form wealth chain
        Used both by initial training-bank label gen and by per-round correction-row
        label gen (M2). The expensive inner-MC fork runs ONCE per call; M2's
        audit-correct loop re-invokes this method per round on a grown bank."""
        N = bank["wealth_pre"].shape[0]
        n_h = len(self.hedges)
        device = self.device
        live_t = self.is_live[t]
        q_t_sampled = torch.randint(-5, 6, (N, n_h), device=device).to(self.dtype)
        if (~live_t).any():
            q_t_sampled[:, ~live_t] = 0.0
        cost_t = self._turnover(q_t_sampled, bank["q_prev"], bank["F_at_t"])
        wealth_post_t = bank["wealth_pre"] - cost_t
        z_t = self._assemble_z(
            market=bank["market"], q=q_t_sampled, wealth=wealth_post_t,
            F=bank["F_at_t"], time_to_T=bank["time_to_T"],
            static=bank["static"])

        # --- One-step inner-MC fork at t (grad-enabled) ---
        with torch.enable_grad():
            inner = self.bundle["inner_mc_grad_fn_one_step"](t)
            market_t1 = inner["market_t1"]                            # (B_outer, B_inner_per, market_dim)
            # Use a single representative inner draw per outer; M=1 per spec §6/§9.
            market_t1 = market_t1[:, 0, :]                            # (B_outer, market_dim)
            # F at t+1 for live hedges — index by outer_index, single inner draw.
            F_t1_per_h = []
            for h in self.hedges:
                f = inner["F_t1"].get(h)
                if f is None:
                    # Closed hedge (past expiry): tradable freezes at expiry value.
                    F_t1_per_h.append(bank["F_at_t"][:, self.hedges.index(h)])
                else:
                    F_t1_per_h.append(f[bank["outer_index"], 0])      # (N,)
            # Stack into (N, n_h). Per-outer alignment via bank["outer_index"].
            F_t1 = torch.stack(F_t1_per_h, dim=-1)                    # (N, n_h)
            # Market at t+1 indexed by outer_index → (N, market_dim).
            market_t1_per_row = market_t1[bank["outer_index"]]

            # Wealth evolution at constant q_t: (gate2-invariant MTM)
            # w_{t+1}_pre = w_post_t + (Σq_t − 1) · dS. For multi-tradable, the
            # generalization uses dF per hedge minus the implicit liability move dS.
            # Forward-=-spot toy: dF_h = dS for live h, so:
            #   Σ_h q_h · dF_h − dS = (Σ_h q_h − 1) · dS
            # which collapses cleanly. Compute via the explicit per-hedge sum so the
            # multi-asset generalization is direct.
            spot_h_idx = 0   # first live hedge tracks spot for the toy
            dS = F_t1[:, spot_h_idx] - bank["F_at_t"][:, spot_h_idx]
            pnl_hedges = ((q_t_sampled * (F_t1 - bank["F_at_t"])).sum(dim=-1))
            wealth_pre_t1 = wealth_post_t + pnl_hedges - dS           # (N,)

            time_to_T_t1 = bank["time_to_T"] - 1.0                    # (N,)
            # Decision at t+1 — external argmax over frozen C_next. q* detached.
            q_star_t1, _, action_gap = self._decision_at(
                C_next, market_t1_per_row, F_t1, q_t_sampled,
                wealth_pre_t1, time_to_T_t1, bank["static"], min(t + 1, self.t_outer - 1))
            q_star_t1 = q_star_t1.detach()
            # Post-decision wealth at t+1 (action q* applied).
            cost_t1 = self._turnover(q_star_t1, q_t_sampled, F_t1)
            wealth_post_t1 = wealth_pre_t1 - cost_t1                  # (N,)
            # Post-decision state at t+1 — AUTOGRAD-CONNECTED to spot leaves via
            # market_t1 / F_t1 / wealth_pre_t1 / wealth_post_t1.
            z_t1 = self._assemble_z(
                market=market_t1_per_row, q=q_star_t1,
                wealth=wealth_post_t1, F=F_t1,
                time_to_T=time_to_T_t1, static=bank["static"])
            # Bootstrap value: the single grad-enabled eval of C_next. Advantage
            # decomp: `_C_full` adds the closed-form baseline at t+1, so Y_boot is
            # the FULL value (baseline + NN residual). The NN at t is then trained
            # against Y_boot − B_t(z_t) — fitting only the small bounded residual.
            Y_boot = self._C_full(C_next, z_t1)                       # (N,)

            # Differential label: ∂Y_boot/∂state_t_leaves. Single backward pass.
            leaves = inner["state_t_leaves"]
            leaf_keys = list(leaves.keys())
            leaf_list = [leaves[k] for k in leaf_keys]
            leaf_grads = torch.autograd.grad(
                Y_boot.sum(), leaf_list, retain_graph=False, allow_unused=True)

        # Project leaf gradients to deep-state F_t columns. The spot leaf (whose value
        # equals F_t for live tradables — forward=spot) maps to all live tradable_prices
        # columns; per-row alignment via outer_index. Other leaves (carry curve,
        # basis if differentiable) are skipped here for the toy; promote to a full
        # projection in M3 when richer underlyings come online.
        dy_dz = torch.zeros_like(z_t)
        F_col_start = self.statepack.n_hedge + 2          # tradable_prices block start
        # Wealth gradient: ∂Y_boot/∂wealth_post_t = ∂Y_boot/∂wealth_post_t+1 (chain
        # is wealth_post_t → wealth_pre_t+1 → wealth_post_t+1 → C_next; all linear).
        # Compute via a separate small autograd pass on a leaf-wealth construction.
        with torch.enable_grad():
            w_leaf = wealth_post_t.detach().clone().requires_grad_(True)
            wealth_pre_t1_l = w_leaf + pnl_hedges.detach() - dS.detach()
            wealth_post_t1_l = wealth_pre_t1_l - cost_t1.detach()
            z_t1_l = self._assemble_z(
                market=market_t1_per_row.detach(), q=q_star_t1,
                wealth=wealth_post_t1_l, F=F_t1.detach(),
                time_to_T=time_to_T_t1, static=bank["static"])
            Y_for_w = self._C_full(C_next, z_t1_l)
            dY_dw = torch.autograd.grad(Y_for_w.sum(), w_leaf)[0]    # (N,)
        wealth_col = self.statepack.n_hedge                          # cash slot
        dy_dz[:, wealth_col] = dY_dw

        # Spot leaf gradient → live tradable_prices cols. Locate the CommodityPrice
        # leaf via leaf_keys (factor-type 'CommodityPrice').
        for key, g in zip(leaf_keys, leaf_grads):
            if g is None:
                continue
            if key.type == "CommodityPrice":
                # g has leaf shape (B_outer,) for HMM spot. Project to all live
                # tradable_prices cols, indexed by outer_index.
                g_per_row = g[bank["outer_index"]]                   # (N,)
                for j, h in enumerate(self.hedges):
                    if live_t[j]:
                        dy_dz[:, F_col_start + j] += g_per_row

        # Advantage decomposition: subtract B_t(z_t) from the value label so the
        # NN fits the residual A_t = C_t − B_t. The gradient label is LEFT
        # UNCHANGED — it remains ∂Y_boot/∂z (= ∂C_t/∂z under the bootstrap
        # approximation), not ∂A_t/∂z. Pragmatically this gives the NN a gradient
        # target close to A_t's true gradient when B's gradient is small (the
        # baseline is bounded in symlog units, so ∂B/∂z = O(1/c)). Subtracting
        # ∂B/∂z exactly turned out to drive the standardized diff-loss target to
        # ~10× the value target and destabilize twin-loss training; the small
        # asymptotic bias from leaving it in is preferred over the instability.
        # λ-mix (when active): blend Y_boot with a no-grad rollout Y_rollout to
        # break max-of-noisy compounding on the value labels. Rollout starts at
        # the bank's post-decision state (q_t_sampled applied, wealth_post_t)
        # and uses frozen C-stack for downstream decisions. Gradient label is
        # untouched — only the value path gets mixed.
        if self._advantage_decomp:
            B_at_z_t = self._baseline_B(z_t).detach()
            Y_value = Y_boot.detach()
            if self._lambda_mix > 0.0:
                with torch.no_grad():
                    Y_rollout = self._rollout_no_grad_to_T(
                        t_start=t, outer_indices=bank["outer_index"],
                        q_post_t=q_t_sampled.detach(),
                        wealth_post_t=wealth_post_t.detach(), traj=traj)
                Y_value = ((1.0 - self._lambda_mix) * Y_value
                            + self._lambda_mix * Y_rollout)
            y_target = Y_value - B_at_z_t
        else:
            y_target = Y_boot.detach()
        return z_t.detach(), y_target, dy_dz.detach(), action_gap.detach()

    def _fit_twin_sgd(self, net, z, y, dy_dz, action_gap, steps, log_label=""):
        """One SGD pass over the supplied (z, y, dy_dz, action_gap) bank. Used by
        both the initial round and any correction-round refit. The action-gap mask
        threshold is recomputed per call against the current bank's median gap."""
        device = self.device
        opt = torch.optim.Adam(net.parameters(), lr=self.adam_lr)
        gap_threshold = max(float(action_gap.median()), 1.0e-6) * 0.5
        n_rows = z.shape[0]
        batch = min(self.train_minibatch, n_rows)
        steps = max(1, int(steps))
        last_diag = {"val_loss": float("nan"), "diff_loss": float("nan")}
        # Under advantage decomposition the gradient label `dy_dz` is for the FULL
        # `C_t` (= ∂Y_boot/∂z), not the residual `A_t = C_t − B_t` that the NN
        # actually fits. Subtracting `∂B/∂z` to make targets consistent destabilized
        # twin loss training (large standardized targets at positions cols when the
        # action grid expands to 121). Cheaper: drop `w_diff` to 0 under advantage
        # decomp — let the NN focus on the value-only fit. The differential premise
        # of diff-ML is then traded for stability; the gradient information re-enters
        # through the value labels themselves (which are derived from the AAD chain).
        w_diff = 0.0 if self._advantage_decomp else 1.0
        for step in range(steps):
            idx = torch.randint(0, n_rows, (batch,), device=device)
            loss, last_diag = twin_loss(
                net, z[idx], y[idx], dy_dz[idx],
                action_gap=action_gap[idx], gap_threshold=gap_threshold,
                w_val=1.0, w_diff=w_diff)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return last_diag

    # ------------------ M2: audit-correct-refit loop for C_t at one t step ----
    def _train_C_at_step(self, t, C_next, traj, training_outer_idx=None,
                          audit_slices=None):
        """Fit `C_t` with the spec §2 audit-correct-refit loop.

        Per round:
          1. (Re)build bank from `accumulated_outer_idx` (starts as training_outer_idx;
             grows with correction rows from prior flagged audits)
          2. Compute one-step bootstrap labels via `_compute_labels_for_bank`
          3. SGD on the twin loss (warm-started from the previous round's weights)
          4. Audit `C_t` against `audit_slices[round]` via `_audit_at`
          5. Pass → break. Fail with shrinking δ → append correction rows targeting
             flagged buckets, continue. Fail with non-shrinking δ → escalate flag
             and break (spec §14: residual is label bias, not fit error — λ-mix case;
             do NOT spin)."""
        device = self.device
        n_h = len(self.hedges)
        live_t = self.is_live[t]
        # Default: all outer paths for training (no audit slices reserved).
        if training_outer_idx is None:
            training_outer_idx = torch.arange(self._B(), device=device)
        if audit_slices is None:
            audit_slices = []
        max_rounds = max(1, len(audit_slices))

        # Net initialized once; warm-starts across correction rounds.
        net = construct_twin_network(
            self.statepack.deep_dim, self.runtime, device=device)
        # Initialize the net's standardization from the first round's bank.
        accumulated_outer_idx = training_outer_idx.clone()
        delta_history = []      # mean|delta| per round, for shrinkage check
        rounds_taken = 0
        final_diag = {"val_loss": float("nan"), "diff_loss": float("nan")}
        escalation_flag = None
        per_round_audits = []

        for round_num in range(max_rounds):
            rounds_taken = round_num + 1
            # 1. Build bank from accumulated outer indices.
            bank = self._build_bank(t, traj, outer_indices=accumulated_outer_idx)
            # 2. Compute labels.
            z_t, y_target, dy_dz, action_gap = self._compute_labels_for_bank(
                t, bank, C_next, traj=traj)
            # On round 0 (first fit), refresh the net's standardization stats.
            if round_num == 0:
                net.set_normalization(
                    z_t.mean(dim=0), z_t.std(dim=0),
                    y_target.mean(), y_target.std())
            # 3. SGD pass (per-round step budget = full budget; warm-start makes this
            #    incremental on rounds 2+).
            final_diag = self._fit_twin_sgd(
                net, z_t, y_target, dy_dz, action_gap,
                steps=self.train_steps_per_solve,
                log_label=f"C_t={t}_r={round_num}")
            self.C[t] = net      # so the audit can call `self.C[t]` via decision op
            logging.info(
                "DifferentialSolver C_t=%d round=%d fit: live_axes=%d bank=%d "
                "twin val=%.5f diff=%.5f action_gap=%.4g",
                t, round_num, int(live_t.sum()), z_t.shape[0],
                final_diag["val_loss"], final_diag["diff_loss"],
                float(action_gap.median()))

            # 4. Audit — if no slices reserved (e.g., M1 single-round usage), break.
            if round_num >= len(audit_slices):
                break
            audit_outer_idx = audit_slices[round_num].to(device)
            audit = self._audit_at(t, net, traj, audit_outer_idx, round_num)
            per_round_audits.append({
                r: {k: v for k, v in b.items() if isinstance(v, (int, float, bool))}
                for r, b in audit["buckets"].items()
            })
            delta_abs_mean = float(audit["delta"].abs().mean())
            delta_history.append(delta_abs_mean)

            if not audit["any_flagged"]:
                logging.info(
                    "DifferentialSolver C_t=%d audit round=%d PASS — accepting C_t.",
                    t, round_num)
                break

            # 5a. Delta-shrinkage gate. If round-over-round |δ| isn't shrinking by
            # at least 5%, the residual is LABEL bias (max-amplification through
            # C_{t+1}), not fit-error. Spec §14: deferred λ-mix; don't add more rows.
            if len(delta_history) >= 2:
                shrink = delta_history[-2] - delta_history[-1]
                shrink_rel = shrink / max(delta_history[-2], 1.0e-9)
                if shrink_rel < 0.05:
                    escalation_flag = (
                        f"delta not shrinking across rounds "
                        f"(|δ|: {delta_history[-2]:.4f} → {delta_history[-1]:.4f}, "
                        f"shrink {shrink_rel*100:.1f}%); residual is label-bias "
                        f"(§14 λ-mix territory), NOT fit-error. Stop appending rows.")
                    logging.warning(
                        "DifferentialSolver C_t=%d audit round=%d ESCALATE — %s",
                        t, round_num, escalation_flag)
                    break

            # 5b. Append correction rows targeting flagged buckets — oversample
            # outer paths in the existing batch that LAND in flagged regimes at t.
            flagged_regimes = [r for r, b in audit["buckets"].items()
                                if b.get("flagged", False)]
            if not flagged_regimes:
                break
            regime_at_t_all = self._outer_buf[(self._spot_key, 'regimes')][t]
            extra_idx_list = []
            for r in flagged_regimes:
                mask_r = (regime_at_t_all == r)
                idx_r = mask_r.nonzero(as_tuple=True)[0]
                if idx_r.numel() == 0:
                    logging.info(
                        "DifferentialSolver C_t=%d round=%d: regime=%d flagged but "
                        "no outer paths land there at t=%d — thin bucket, skip.",
                        t, round_num, r, t)
                    continue
                # Oversample with replacement to densify endogenous span within
                # flagged regime — same path indices, fresh (q_prev, wealth) draws
                # via _build_bank's random sampling at the next call.
                oversample_n = min(int(idx_r.numel() * 2), int(idx_r.numel()))
                pick = torch.randint(0, idx_r.numel(), (oversample_n,), device=device)
                extra_idx_list.append(idx_r[pick])
            if not extra_idx_list:
                break
            extra_idx = torch.cat(extra_idx_list)
            accumulated_outer_idx = torch.cat([accumulated_outer_idx, extra_idx])
            logging.info(
                "DifferentialSolver C_t=%d audit round=%d FLAGGED regimes=%s; "
                "appending %d correction outer paths (bank grows %d → %d).",
                t, round_num, flagged_regimes, int(extra_idx.shape[0]),
                int(accumulated_outer_idx.shape[0] - extra_idx.shape[0]),
                int(accumulated_outer_idx.shape[0]))

        return {
            "twin_value_loss": final_diag["val_loss"],
            "twin_diff_loss": final_diag["diff_loss"],
            "bank_rows": int(accumulated_outer_idx.shape[0]) * int(self.b_endo),
            "live_axes_at_t": int(live_t.sum()),
            "rounds_taken": rounds_taken,
            "delta_history_abs_mean": delta_history,
            "per_round_audits": per_round_audits,
            "escalation_flag": escalation_flag,
        }

    # --------------------------------- M2: audit rollout + correction loop
    def _rollout_no_grad_to_T(self, t_start, outer_indices, q_post_t, wealth_post_t,
                              traj):
        """Frozen-policy no-grad rollout from post-decision state at `t_start` to
        terminal `T_dec`. Returns `Y_eval` = realized terminal utility per audit row.

        At each future step u ∈ [t_start, t_outer−2]:
          • read realized (market_{u+1}, F_{u+1}) from `traj[u+1]` at the audit row's
            outer index (no fresh inner-MC needed — the audit path's outer trajectory
            IS the realized market evolution we're auditing the policy against).
          • evolve wealth: `wealth_pre_{u+1} = wealth_post_u + Σ_h q_post_u[h]·dF_h − dS`
            (gate2's MTM-invariant: forward=spot ⇒ collapses to `(Σq−1)·dS`).
          • pick `q*_{u+1} = argmax_q C[u+1](post_{u+1}(q))` over live action grid.
          • `wealth_post_{u+1} = wealth_pre_{u+1} − turnover(q*_{u+1} − q_post_u)`.
        At terminal u = t_outer−1: `Y_eval = U(w_terminal)` via symlog.

        This is the rollout the spec §7 audit uses to test whether the model's fitted
        `C_t` matches realized utility under its own policy. Disagreement = bias.
        """
        N = outer_indices.shape[0]
        n_h = len(self.hedges)
        device = self.device
        # Carry forward state — q (post-decision inventory) and wealth (post-decision).
        q_carry = q_post_t.clone()                              # (N, n_h)
        wealth = wealth_post_t.clone()                          # (N,)
        with torch.no_grad():
            for u in range(t_start, self.t_outer - 1):
                # Read the realized (market_{u+1}, F_{u+1}) for the audit rows.
                market_u1 = traj[u + 1]["market"][outer_indices]    # (N, market_dim)
                F_u1 = traj[u + 1]["tradables"][outer_indices]      # (N, n_h)
                # Wealth MTM: w_pre_{u+1} = w_post_u + Σ_h q_h·(F_{u+1,h}−F_u,h) − (S_{u+1}−S_u)
                # where S = first tradable (forward=spot toy). Generalizes per-hedge
                # for multi-asset.
                F_u = traj[u]["tradables"][outer_indices]
                dF = F_u1 - F_u                                     # (N, n_h)
                dS = dF[:, 0]                                       # (N,) — spot move
                pnl_hedges = (q_carry * dF).sum(dim=-1)             # (N,)
                wealth_pre_u1 = wealth + pnl_hedges - dS            # (N,)
                # Decision at u+1 with frozen C[u+1] (or terminal U if u+1 = T-1).
                if u + 1 >= self.t_outer - 1 or self.C[u + 1] is None:
                    # No further continuation function — apply terminal utility.
                    # At terminal there's no action; carry q_carry, wealth = wealth_pre.
                    wealth = wealth_pre_u1
                    break
                static_u1 = self._static_sim[u + 1].to(device).expand(
                    N, self._static_sim.shape[-1])
                time_to_T_u1 = torch.full((N,), float(self.t_outer - 1 - (u + 1)),
                                           device=device)
                q_star_u1, _, _ = self._decision_at(
                    self.C[u + 1], market_u1, F_u1, q_carry,
                    wealth_pre_u1, time_to_T_u1, static_u1, u + 1)
                cost_u1 = self._turnover(q_star_u1, q_carry, F_u1)
                wealth = wealth_pre_u1 - cost_u1
                q_carry = q_star_u1
            # Terminal utility under symlog at the framework's c (typically 100 for the
            # gate2 toy after the M4 reconciliation).
            c_util = self.utility_c
            Y_eval = torch.sign(wealth) * torch.log1p(wealth.abs() / c_util)
        return Y_eval

    def _audit_at(self, t, net, traj, audit_outer_indices, round_num):
        """Spec §7 audit at time t. For each audit row:
          1. Sample pre-decision (q_prev, wealth_pre) within the designer endogenous
             box (same span as training to keep the audit on the same domain `C_t` is
             supposed to model).
          2. Compute decision `q*_t = argmax_q C_t(post_t(q))`.
          3. `Y_boot = C_t(z_t)` where `z_t = post_t(q*_t)` — the **fitted** value at
             the audit state. Per spec §7's "current bootstrap value (frozen stack)";
             using C_t directly catches the TOTAL deviation (fit error + label bias),
             which is what compounds backward — `C_{t-1}`'s labels are
             `max_q C_t(post_t(q))`, so the thing `C_{t-1}` reads is the *fitted* C_t.
          4. `Y_eval` via `_rollout_no_grad_to_T` from `z_t` using frozen C-stack.
          5. `delta = Y_eval − Y_boot`. Bucket by **regime at t** (spec §7 mentions
             (time, inventory, action_gap, tail) — for the toy's regime-conditional
             bias, regime is the load-bearing axis; promote to richer bucketing only
             if p95 flags but the location is ambiguous).
          6. Per-bucket: mean δ, p95|δ|, n_in_bucket. Flag if `|mean|>MEAN_TOL` OR
             `p95>P95_TOL` (per-bucket p95 catches tail-concentrated bias the mean
             washes out — the classic max-bias signature).
        """
        N_audit = audit_outer_indices.shape[0]
        n_h = len(self.hedges)
        device = self.device
        # Audit-row endogenous sampling — same designer box as training, fresh draws.
        # `q_prev` uniform integer, `wealth_pre` uniform on ±1500.
        q_min, q_max = -5, 5
        wealth_halfwidth = 1500.0
        q_prev = torch.randint(q_min, q_max + 1, (N_audit, n_h),
                                device=device).to(self.dtype)
        wealth_pre = (torch.rand(N_audit, device=device) - 0.5) * (2.0 * wealth_halfwidth)
        live_t = self.is_live[t]
        if (~live_t).any():
            q_prev[:, ~live_t] = 0.0

        market_t_audit = traj[t]["market"][audit_outer_indices]
        F_t_audit = traj[t]["tradables"][audit_outer_indices]
        static_t_audit = self._static_sim[t].to(device).expand(
            N_audit, self._static_sim.shape[-1])
        time_to_T_t = torch.full((N_audit,), float(self.t_outer - 1 - t), device=device)

        # Decision at t: q*_t via C_t (which we're auditing).
        q_star_t, _, action_gap = self._decision_at(
            net, market_t_audit, F_t_audit, q_prev, wealth_pre, time_to_T_t,
            static_t_audit, t)
        cost_t = self._turnover(q_star_t, q_prev, F_t_audit)
        wealth_post_t = wealth_pre - cost_t

        # Y_boot via C_t at the chosen post-decision state. Routes through
        # `_C_full` so advantage decomposition's `A_t + B_t` is measured as a
        # whole — auditing `net(z)` alone under decomp would compare raw A_t to
        # Y_eval (in symlog units), making δ meaningless.
        with torch.no_grad():
            z_t_audit = self._assemble_z(
                market=market_t_audit, q=q_star_t, wealth=wealth_post_t,
                F=F_t_audit, time_to_T=time_to_T_t, static=static_t_audit)
            Y_boot = self._C_full(net, z_t_audit)                      # (N_audit,)

        # Y_eval via rollout to T.
        Y_eval = self._rollout_no_grad_to_T(
            t_start=t, outer_indices=audit_outer_indices,
            q_post_t=q_star_t, wealth_post_t=wealth_post_t, traj=traj)
        delta = Y_eval - Y_boot                                       # (N_audit,)

        # Bucket by regime at t. Regime is one-hot in the market block (spot regime
        # is the first n_states cols by `_read_privileged_market` iteration order).
        # Extract via argmax over the regime block.
        proc_spot = self._calc.stoch_factors_inner[self._spot_key]
        n_states = int(getattr(proc_spot, 'n_states', 1))
        # `outer_buf[(spot_key, 'regimes')]` is the realized regime path (T, B) long.
        # Read regime at t for the audit's outer indices.
        regime_at_t = self._outer_buf[(self._spot_key, 'regimes')][t][audit_outer_indices]

        buckets = {}
        any_flagged = False
        MEAN_TOL = float(self.runtime["solver"].get("audit_mean_tol", 0.05))
        P95_TOL = float(self.runtime["solver"].get("audit_p95_tol", 0.20))
        bucket_min_count = int(self.runtime["solver"].get("audit_bucket_min_count", 8))
        for r in range(n_states):
            mask_r = (regime_at_t == r)
            n_r = int(mask_r.sum())
            if n_r == 0:
                buckets[r] = {"n": 0, "skipped": True, "reason": "no audit rows"}
                continue
            delta_r = delta[mask_r]
            mean_r = float(delta_r.mean())
            p95_r = float(delta_r.abs().quantile(0.95)) if n_r >= 3 else float(delta_r.abs().max())
            flagged = (abs(mean_r) > MEAN_TOL) or (p95_r > P95_TOL)
            buckets[r] = {
                "n": n_r,
                "mean_delta": mean_r,
                "p95_abs_delta": p95_r,
                "flagged": flagged,
                "thin": n_r < bucket_min_count,
            }
            if flagged:
                any_flagged = True

        logging.info(
            "DifferentialSolver audit t=%d round=%d: %s",
            t, round_num,
            "; ".join(f"r={r} n={b.get('n',0)} mean={b.get('mean_delta',float('nan')):+.4f} "
                       f"p95={b.get('p95_abs_delta',float('nan')):.4f}"
                       f"{' FLAG' if b.get('flagged') else ''}"
                       f"{' (thin)' if b.get('thin') else ''}"
                       for r, b in buckets.items()))

        return {
            "buckets": buckets,
            "any_flagged": any_flagged,
            "delta": delta.detach(),
            "regime_at_t": regime_at_t.detach(),
            "q_prev": q_prev.detach(),
            "wealth_pre": wealth_pre.detach(),
            "audit_outer_indices": audit_outer_indices,
        }

    def _grade_against_oracle_policy(self, npz_path, traj):
        """Compare learned argmax policy `n*_0` per regime at t=0 against gate2's
        `initial_policy`. Per user's mandate: 'V_0 climbs' is necessary but not
        sufficient — the policy must become regime-conditional and match the DP's
        actions. Returns per-regime (q_oracle, q_learned, match) tuples."""
        try:
            np.load(npz_path, allow_pickle=True)
        except FileNotFoundError:
            return {"oracle_policy_skipped": True}
        # gate2 stores the policy summary in `artifacts/gate2_summary.json` next to
        # the npz — fixed name, not derived from the npz path.
        import os, json
        summary_path = os.path.join(os.path.dirname(npz_path), 'gate2_summary.json')
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            initial_policy = summary.get("initial_policy", {})
        except (FileNotFoundError, KeyError):
            return {"oracle_policy_skipped": True}

        if not initial_policy or self.C[0] is None:
            return {"oracle_policy_skipped": True}

        device = self.device
        n_h = len(self.hedges)
        # In-distribution market probe — pull a real market block from traj[0]
        # filtered by regime, same approach as `_depth_profile_diagnostic`. A
        # synthetic one-hot `market[0, r_idx] = 1.0` would set the wrong columns
        # (the regime one-hot is not at index 0 in the market block; see
        # `_depth_profile_diagnostic` for details).
        regimes_at_0 = self._outer_buf[(self._spot_key, 'regimes')][0]
        traj_market_0 = traj[0]["market"]
        results = {}
        for regime_key, oracle_q in initial_policy.items():
            r_idx = int(regime_key.split('=')[1]) if '=' in regime_key else int(regime_key)
            q_oracle = [int(oracle_q.get("q_A_value", 0)),
                        int(oracle_q.get("q_B_value", 0))]
            in_regime = (regimes_at_0 == r_idx).nonzero(as_tuple=True)[0]
            if in_regime.numel() == 0:
                continue
            src = int(in_regime[0])
            market_block = traj_market_0[src:src+1]
            q_prev = torch.zeros(1, n_h, device=device)
            F_t = torch.full((1, n_h), 100.0, device=device)
            wealth_pre = torch.full((1,), float(self.bundle.get("initial_wealth", 100.0)),
                                     device=device)
            time_to_T = torch.full((1,), float(self.t_outer - 1), device=device)
            static_t = self._static_sim[0].to(device).flatten().unsqueeze(0)
            q_star, _, _ = self._decision_at(
                self.C[0], market_block, F_t, q_prev, wealth_pre, time_to_T,
                static_t, t=0)
            q_learned = q_star[0].cpu().tolist()
            match = (int(round(q_learned[0])) == q_oracle[0]
                      and int(round(q_learned[1])) == q_oracle[1])
            results[r_idx] = {
                "q_oracle": q_oracle,
                "q_learned": [float(x) for x in q_learned],
                "match": match,
            }
            logging.info(
                "DifferentialSolver oracle policy r=%d: oracle=%s, learned=%s — %s",
                r_idx, q_oracle, [float(x) for x in q_learned],
                "MATCH" if match else "MISMATCH")
        return {"oracle_policy_per_regime": results,
                "oracle_policy_match_rate": sum(1 for v in results.values() if v["match"])
                                              / max(1, len(results))}

    def _action_error_vs_depth(self, npz_path, traj):
        """Per-t oracle-policy match across the full backward sweep — the structural
        validation gate per spec: 'ranking survives depth'. Distinguishes a
        compounding failure mode (action-error growing backward with depth) from
        a clean per-step bias (flat action-error across t).

        Probe is canonical-state-per-regime (q_prev=(0,0), w=K, s=mid) to keep the
        signal interpretable; per-t variation reveals depth-driven degradation.

        Requires gate2 to have saved policy tensors (`pol_pre_A_qA`, `pol_pre_A_qB`,
        `pol_post_A_qB`); silently skips if absent (gate2 npz from before the
        save-policy patch).
        """
        try:
            npz = np.load(npz_path, allow_pickle=True)
        except FileNotFoundError:
            return {"action_error_vs_depth_skipped": "npz_not_found"}
        if 'pol_pre_A_qA' not in npz.files:
            return {"action_error_vs_depth_skipped": "policy_tensors_absent"}

        pol_pre_A_qA = npz['pol_pre_A_qA']           # (T_A, R, N_S, N_Q, N_Q, N_W)
        pol_pre_A_qB = npz['pol_pre_A_qB']
        pol_post_A_qB = npz['pol_post_A_qB']         # (T_dec-T_A, R, N_S, N_Q, N_W)
        q_grid = npz['q_grid']                       # (N_Q,) values
        s_grid = npz['s_grid']
        w_grid = npz['w_grid']

        T_A_oracle = int(npz['T_A'])                 # scalars saved by gate2
        T_dec_oracle = int(npz['T_dec'])
        # Map solver outer t to oracle t. Solver's t_outer = T_dec_oracle + 2
        # (sim grid has T_dec_oracle + 1 weekly points + a base date). Probe at
        # solver t = oracle t (1:1 in the weekly toy).
        if self.t_outer - 2 != T_dec_oracle:
            logging.warning(
                "action_error_vs_depth: solver t_outer=%d implies T_dec=%d, but oracle "
                "T_dec=%d — depth mismatch, skipping.",
                self.t_outer, self.t_outer - 2, T_dec_oracle)
            return {"action_error_vs_depth_skipped": "horizon_mismatch"}

        s_mid_idx = s_grid.shape[0] // 2
        K = float(self.bundle.get("initial_wealth", 100.0))
        w_init_idx = int(np.argmin(np.abs(w_grid - K)))
        q0_idx = int(np.where(q_grid == 0)[0][0])
        n_h = len(self.hedges)

        device = self.device
        regimes_per_t = self._outer_buf[(self._spot_key, 'regimes')]      # (T, B)

        per_t = []
        for t in range(self.t_outer - 1):
            if self.C[t] is None or t >= len(traj) or t >= T_dec_oracle:
                continue                              # no decision past T_dec - 1
            traj_market_t = traj[t]["market"]
            regimes_at_t = regimes_per_t[t]
            row = {"t": t, "matches": [], "regime_results": {}}
            for r_idx in (0, 1):
                in_regime = (regimes_at_t == r_idx).nonzero(as_tuple=True)[0]
                if in_regime.numel() == 0:
                    continue
                src = int(in_regime[0])
                market_block = traj_market_t[src:src+1]
                q_prev = torch.zeros(1, n_h, device=device)
                F_t = torch.full((1, n_h), 100.0, device=device, dtype=self.dtype)
                wealth_pre = torch.full((1,), K, device=device, dtype=self.dtype)
                time_to_T = torch.full((1,), float(self.t_outer - 1 - t),
                                        device=device, dtype=self.dtype)
                static_t = self._static_sim[t].to(device).flatten().unsqueeze(0)
                q_star, _, _ = self._decision_at(
                    self.C[t], market_block, F_t, q_prev, wealth_pre,
                    time_to_T, static_t, t=t)
                q_learned = q_star[0].cpu().tolist()
                # Oracle action at this canonical state.
                if t < T_A_oracle:
                    qA_oracle_idx = int(pol_pre_A_qA[t, r_idx, s_mid_idx,
                                                       q0_idx, q0_idx, w_init_idx])
                    qB_oracle_idx = int(pol_pre_A_qB[t, r_idx, s_mid_idx,
                                                       q0_idx, q0_idx, w_init_idx])
                    q_oracle = [int(q_grid[qA_oracle_idx]), int(q_grid[qB_oracle_idx])]
                else:
                    # post_A: q_A=0 forced; oracle pol_post_A_qB indexed at (t-T_A).
                    qB_oracle_idx = int(pol_post_A_qB[t - T_A_oracle, r_idx,
                                                       s_mid_idx, q0_idx, w_init_idx])
                    q_oracle = [0, int(q_grid[qB_oracle_idx])]
                match = (int(round(q_learned[0])) == q_oracle[0]
                          and int(round(q_learned[1])) == q_oracle[1])
                row["matches"].append(match)
                row["regime_results"][r_idx] = {
                    "q_oracle": q_oracle,
                    "q_learned": [float(x) for x in q_learned],
                    "match": match,
                }
            if row["matches"]:
                row["match_rate"] = sum(row["matches"]) / len(row["matches"])
                per_t.append(row)

        # Log the depth profile as a compact table.
        logging.info(
            "DifferentialSolver ACTION-ERROR vs DEPTH — per-t oracle match "
            "at canonical (q_prev=0, w=K, s=mid). flat curve = no compounding; "
            "growing backward = compounding signature:")
        logging.info("  t    r=0 oracle vs learned       r=1 oracle vs learned       match")
        for r in per_t:
            r0 = r["regime_results"].get(0, {})
            r1 = r["regime_results"].get(1, {})
            r0_str = (f"{r0.get('q_oracle')} vs {r0.get('q_learned')}"
                       + ("✓" if r0.get('match') else "✗")) if r0 else "—"
            r1_str = (f"{r1.get('q_oracle')} vs {r1.get('q_learned')}"
                       + ("✓" if r1.get('match') else "✗")) if r1 else "—"
            logging.info("  %3d  %-29s  %-29s  %d/%d",
                          r["t"], r0_str, r1_str,
                          sum(r["matches"]), len(r["matches"]))

        # Aggregate: match rate vs depth.
        match_rates = [r["match_rate"] for r in per_t]
        overall = sum(match_rates) / max(1, len(match_rates))
        # Compounding signature: action-error trend across t. Linear regression
        # slope of (1 - match_rate) vs (t_outer - 2 - t) [depth from terminal].
        if len(match_rates) >= 4:
            depths = np.array([self.t_outer - 2 - r["t"] for r in per_t])
            errors = 1.0 - np.array(match_rates)
            slope = float(np.polyfit(depths, errors, 1)[0]) if errors.std() > 1e-6 else 0.0
        else:
            slope = float('nan')
        logging.info(
            "ACTION-ERROR vs DEPTH SUMMARY: %d t-points; overall match rate=%.2f%%; "
            "error-vs-depth slope=%+.4f (positive = compounding signature; ~0 = flat = no compounding)",
            len(match_rates), overall * 100, slope)
        return {"action_error_per_t": per_t, "action_error_overall_match_rate": overall,
                "action_error_depth_slope": slope}

    # ------------------------------- End-of-F smoke: oracle + grad sanity
    def _smoke_against_gate2(self, t, npz_path):
        """Single-contract mechanism test: compare fitted C_t's argmax + a few V cells
        at outer t against gate2's V_post_A oracle. **The smoke is intentionally
        single-contract** (t > T_A: only q_B live, 11 candidates) — the live-axis,
        bank-collapse, and dead-axis-mask code paths run but the multi-contract
        argmax is NOT exercised. That's M1.5's job."""
        try:
            npz = np.load(npz_path)
        except FileNotFoundError:
            logging.warning("Skipping oracle comparison — %s not found.", npz_path)
            return {"oracle_skipped": True}
        V_post_A = npz.get("V_post_A")
        s_grid = npz.get("s_grid")
        q_grid = npz.get("q_grid")
        w_grid = npz.get("w_grid")
        if V_post_A is None:
            logging.warning("Skipping oracle comparison — no V_post_A in %s.", npz_path)
            return {"oracle_skipped": True}
        # gate2's V_post_A has time axis spanning [T_A, T_dec]; t in our solver indexes
        # 0..T_dec-1 weekly. For T_dec=10, T_A=5: V_post_A has 6 slices; t=8 corresponds
        # to V_post_A[3]. Solver's t_outer = T_dec + 1 = 11 (sim grid includes t=0).
        # gate2's t_outer convention: indices into V_post_A are (t - T_A).
        t_dec = V_post_A.shape[0] - 1 + (q_grid is not None and 0 or 0)   # heuristic
        # We treat solver t = t_outer-1 - hist as the latest decision index.
        # Just compute the offset from the last DP time slice — at our self.t_outer-2,
        # gate2's V_post_A[-2] is the matching cell. Be defensive about index math.
        v_slice_idx = V_post_A.shape[0] - (self.t_outer - 1 - t)
        if v_slice_idx < 0 or v_slice_idx >= V_post_A.shape[0]:
            logging.warning("Oracle slice index %d out of bounds for V_post_A shape %s",
                            v_slice_idx, V_post_A.shape)
            return {"oracle_skipped": True}
        V_slice = V_post_A[v_slice_idx]                              # (n_regime, N_S, N_q_B, N_W)
        # Compare a few representative cells: regime=0/1, mid s, mid q, mid w.
        device = self.device
        rows = []
        S0 = float(self.bundle.get("initial_spot", 100.0))
        for r_idx in range(V_slice.shape[0]):
            for s_idx in [V_slice.shape[1] // 2]:                    # mid log-spot
                for qB_idx in [V_slice.shape[2] // 2, 0, V_slice.shape[2] - 1]:
                    for w_idx in [V_slice.shape[3] // 2]:
                        # Build z at this cell.
                        S_cell = float(s_grid[s_idx]) if s_grid is not None else S0
                        qB_cell = float(q_grid[qB_idx]) if q_grid is not None else 0.0
                        w_cell = float(w_grid[w_idx]) if w_grid is not None else 0.0
                        v_dp = float(V_slice[r_idx, s_idx, qB_idx, w_idx])
                        # Assemble synthetic z (single-contract: q_A=0; live regime
                        # one-hot from r_idx; market block = the regime one-hot for
                        # spot only; basis/carry default to zero/static).
                        market_block = torch.zeros(self.statepack.market_dim,
                                                    device=device)
                        market_block[r_idx] = 1.0           # spot regime one-hot
                        n_h = len(self.hedges)
                        q_cell = torch.zeros(n_h, device=device)
                        q_cell[1] = qB_cell                 # PL_B is hedge[1]
                        F_cell = torch.full((n_h,), S_cell, device=device)
                        static_cell = self._static_sim[t].to(device).flatten()
                        z = self._assemble_z(
                            market=market_block.unsqueeze(0),
                            q=q_cell.unsqueeze(0),
                            wealth=torch.tensor([w_cell], device=device),
                            F=F_cell.unsqueeze(0),
                            time_to_T=torch.tensor([float(self.t_outer - 1 - t)],
                                                    device=device),
                            static=static_cell.unsqueeze(0))
                        with torch.no_grad():
                            v_nn = float(self._C_full(self.C[t], z).item())
                        rows.append({"regime": r_idx, "s_idx": s_idx, "qB_idx": qB_idx,
                                      "w_idx": w_idx, "v_dp": v_dp, "v_nn": v_nn,
                                      "residual": v_nn - v_dp})
        if rows:
            residuals = [r["residual"] for r in rows]
            mae = sum(abs(r) for r in residuals) / len(residuals)
            logging.info(
                "DifferentialSolver oracle compare t=%d (v_slice_idx=%d, %d cells): "
                "MAE=%.4f, max|residual|=%.4f", t, v_slice_idx, len(rows),
                mae, max(abs(r) for r in residuals))
            for r in rows[:6]:
                logging.info(
                    "  r=%d s=%d qB=%d w=%d  V_dp=%+.4f  C_nn=%+.4f  Δ=%+.4f",
                    r["regime"], r["s_idx"], r["qB_idx"], r["w_idx"],
                    r["v_dp"], r["v_nn"], r["residual"])
            return {"oracle_mae": float(mae),
                    "oracle_max_abs": float(max(abs(r) for r in residuals)),
                    "oracle_cells_compared": len(rows)}
        return {"oracle_skipped": True}

    def _depth_profile_diagnostic(self, npz_path, traj):
        """Decides whether M2's residual is COMPOUNDING (gap small at terminal anchor,
        grows backward) or BOUNDARY BREAK (gap already large at the cold-trained
        anchor / one step in). The two patterns need different fixes:

        - **Compounding** (oracle Δ grows from t=T backward, regime-separation in
          fitted C_t is PRESENT but slightly compressed) → λ-mix is the right remedy:
          de-bias the optimistic max-label with a rollout target.
        - **Boundary break** (oracle Δ is already large at C_T or C_{T-1}, regime
          separation in fitted C_t is FLAT or wrong-signed) → the rot is in label-gen
          / regime conditioning, NOT compounded max-of-noisy. λ-mix won't help; need
          to look upstream at the label generator's regime sensitivity.

        Per the user's diagnostic: "Pure max-of-noisy would, if anything, preserve or
        amplify regime structure. Uniform-across-regime degradation points at the
        regime signal being washed out in label-gen." So the test is two-part:
          (a) oracle gap vs depth (grows backward = compounding)
          (b) regime separation in fitted C_t at each depth (present = compounding-OK;
              flat/uniform = label-gen problem)
        """
        try:
            npz = np.load(npz_path)
        except FileNotFoundError:
            return {"depth_profile_skipped": True}
        V_post_A = npz.get("V_post_A")
        V_pre_A = npz.get("V_pre_A")
        s_grid = npz.get("s_grid")
        q_grid = npz.get("q_grid")
        w_grid = npz.get("w_grid")
        if V_post_A is None or V_pre_A is None:
            return {"depth_profile_skipped": True}

        device = self.device
        n_h = len(self.hedges)
        # Probe at t in {T-1 (boundary), T-2, mid, T_A+1, T_A-1 (post-mid pre-T_A)}.
        T_outer_dec = self.t_outer - 1
        probe_ts = sorted({T_outer_dec - 1, T_outer_dec - 2,
                            (T_outer_dec + self.t_min) // 2,
                            self.t_min})
        probe_ts = [t for t in probe_ts if 0 <= t < self.t_outer
                     and self.C[t] is not None]

        results = {}
        for t in probe_ts:
            v_slice_idx = V_post_A.shape[0] - (T_outer_dec - t)
            use_post_a = v_slice_idx >= 0
            if use_post_a:
                if v_slice_idx >= V_post_A.shape[0]:
                    continue
                V_slice = V_post_A[v_slice_idx]            # (n_reg, N_S, N_qB, N_W)
                live_n_h = 1                                # only q_B
            else:
                # t is in the pre-T_A regime (both q_A, q_B live). V_pre_A slice.
                v_pre_idx = V_pre_A.shape[0] - (V_post_A.shape[0] - v_slice_idx)
                if v_pre_idx < 0 or v_pre_idx >= V_pre_A.shape[0]:
                    continue
                V_slice = V_pre_A[v_pre_idx]               # (n_reg, N_S, N_qA, N_qB, N_W)
                live_n_h = 2

            # Sample mid-spot, mid-wealth cells across the q_B axis (and q_A=0 in
            # the post-A regime; q_A=mid in pre-A regime).
            cells = []
            s_idx = V_slice.shape[1] // 2
            w_idx = V_slice.shape[-1] // 2
            if use_post_a:
                # V_slice indexed as [r, s, qB, w]
                for r_idx in range(V_slice.shape[0]):
                    for qB_idx in [0, V_slice.shape[2] // 2, V_slice.shape[2] - 1]:
                        v_dp = float(V_slice[r_idx, s_idx, qB_idx, w_idx])
                        if abs(v_dp) > 1e8:
                            continue   # off-grid sentinel
                        cells.append((r_idx, s_idx, 0, qB_idx, w_idx, v_dp))
            else:
                # V_slice indexed as [r, s, qA, qB, w]
                qA_mid = V_slice.shape[2] // 2
                for r_idx in range(V_slice.shape[0]):
                    for qB_idx in [0, V_slice.shape[3] // 2, V_slice.shape[3] - 1]:
                        v_dp = float(V_slice[r_idx, s_idx, qA_mid, qB_idx, w_idx])
                        if abs(v_dp) > 1e8:
                            continue
                        cells.append((r_idx, s_idx, qA_mid, qB_idx, w_idx, v_dp))

            # Evaluate C_t at each cell, plus regime/action separation probes.
            # IMPORTANT: use REAL market blocks from `traj[t]` filtered by
            # `regime_at_t`, not a synthetic one-hot. A constructed market with
            # `market[0, r_idx] = 1.0` would set the wrong columns (the regime
            # one-hot is offset by ForwardRate's 3 carry tenors at indices 0-2 in
            # the toy's market block; regime sits at cols 3-4), and feeding the NN
            # an OOD market block makes the probe meaningless — the NN was trained
            # on `traj[t]['market']` distribution, so we evaluate it there.
            time_to_T = float(T_outer_dec - t)
            static_t = self._static_sim[t].to(device).flatten()
            regimes_at_t = self._outer_buf[(self._spot_key, 'regimes')][t]
            traj_market_t = traj[t]["market"]
            per_cell = []
            for r_idx, si, qA_idx, qB_idx, wi, v_dp in cells:
                # Find an outer path landing in regime r_idx at t and reuse its
                # market block — that's the in-distribution probe.
                in_regime = (regimes_at_t == r_idx).nonzero(as_tuple=True)[0]
                if in_regime.numel() == 0:
                    continue
                src = int(in_regime[0])                # first match
                market = traj_market_t[src:src+1]      # (1, market_dim)
                # Other state coords from the oracle's grid (still synthetic — but
                # only on coords the NN can correctly interpolate over).
                qA_cell = float(q_grid[qA_idx]) if q_grid is not None else 0.0
                qB_cell = float(q_grid[qB_idx]) if q_grid is not None else 0.0
                w_cell = float(w_grid[wi]) if w_grid is not None else 0.0
                S_cell = float(s_grid[si]) if s_grid is not None else 100.0
                q = torch.tensor([[qA_cell, qB_cell]], device=device, dtype=self.dtype)
                F = torch.full((1, n_h), S_cell, device=device, dtype=self.dtype)
                w = torch.tensor([w_cell], device=device, dtype=self.dtype)
                z = self._assemble_z(market, q, w, F,
                                      torch.tensor([time_to_T], device=device),
                                      static_t.unsqueeze(0))
                with torch.no_grad():
                    v_nn = float(self._C_full(self.C[t], z).item())
                per_cell.append({"r": r_idx, "qB": int(qB_cell), "v_dp": v_dp,
                                  "v_nn": v_nn, "delta": v_nn - v_dp})
            if not per_cell:
                continue
            mae = sum(abs(c["delta"]) for c in per_cell) / len(per_cell)
            r0 = [c for c in per_cell if c["r"] == 0]
            r1 = [c for c in per_cell if c["r"] == 1]
            r0_mean = sum(c["delta"] for c in r0) / max(1, len(r0))
            r1_mean = sum(c["delta"] for c in r1) / max(1, len(r1))

            # Regime separation in fitted C_t — paired comparison at matched (qB, w):
            # |C_t(r=0,...) - C_t(r=1,...)| averaged. If FLAT → label-gen washing out
            # regime. Pure max-of-noisy preserves regime structure.
            r0_by_qB = {c["qB"]: c["v_nn"] for c in r0}
            r1_by_qB = {c["qB"]: c["v_nn"] for c in r1}
            paired_diffs = [abs(r0_by_qB[qB] - r1_by_qB[qB])
                            for qB in r0_by_qB if qB in r1_by_qB]
            regime_sep_nn = (sum(paired_diffs) / len(paired_diffs)
                              if paired_diffs else 0.0)
            # Same for the DP oracle — the right answer for "how much regime
            # separation should C_t have at this t".
            r0_dp_by_qB = {c["qB"]: c["v_dp"] for c in r0}
            r1_dp_by_qB = {c["qB"]: c["v_dp"] for c in r1}
            paired_dp = [abs(r0_dp_by_qB[qB] - r1_dp_by_qB[qB])
                          for qB in r0_dp_by_qB if qB in r1_dp_by_qB]
            regime_sep_dp = (sum(paired_dp) / len(paired_dp)
                              if paired_dp else 0.0)

            # Action separation in C_t — does it differentiate qB choices at all?
            qB_vals_r0 = [c["v_nn"] for c in r0]
            action_sep_nn = (max(qB_vals_r0) - min(qB_vals_r0)) if qB_vals_r0 else 0.0
            qB_vals_r0_dp = [c["v_dp"] for c in r0]
            action_sep_dp = ((max(qB_vals_r0_dp) - min(qB_vals_r0_dp))
                              if qB_vals_r0_dp else 0.0)

            results[t] = {
                "v_slice_kind": "post_A" if use_post_a else "pre_A",
                "v_slice_idx": v_slice_idx,
                "n_cells": len(per_cell),
                "mae": mae,
                "r0_mean_delta": r0_mean,
                "r1_mean_delta": r1_mean,
                "regime_sep_nn": regime_sep_nn,
                "regime_sep_dp": regime_sep_dp,
                "regime_sep_ratio": (regime_sep_nn / regime_sep_dp
                                      if regime_sep_dp > 1e-6 else float("nan")),
                "action_sep_nn": action_sep_nn,
                "action_sep_dp": action_sep_dp,
                "action_sep_ratio": (action_sep_nn / action_sep_dp
                                      if action_sep_dp > 1e-6 else float("nan")),
            }

        # Log the depth profile as a table — easier to scan than a JSON dump.
        logging.info(
            "DifferentialSolver DEPTH PROFILE — gap and separation vs depth "
            "(compounding ↔ growing-backward ; boundary break ↔ already flat at T-1):")
        logging.info(
            "  t     kind    MAE      Δ(r=0)   Δ(r=1)   regime-sep(NN/DP)    "
            "action-sep(NN/DP)")
        for t in sorted(results.keys(), reverse=True):
            r = results[t]
            logging.info(
                "  %3d   %-6s  %+.4f  %+.4f  %+.4f  %.4f/%.4f (%.2fx)  "
                "%.4f/%.4f (%.2fx)",
                t, r["v_slice_kind"], r["mae"],
                r["r0_mean_delta"], r["r1_mean_delta"],
                r["regime_sep_nn"], r["regime_sep_dp"], r["regime_sep_ratio"],
                r["action_sep_nn"], r["action_sep_dp"], r["action_sep_ratio"])

        # Verdict heuristic — compare MAE at T-1 (boundary) vs deepest probed t.
        ts_sorted = sorted(results.keys(), reverse=True)
        if len(ts_sorted) >= 2:
            mae_boundary = results[ts_sorted[0]]["mae"]
            mae_deep = results[ts_sorted[-1]]["mae"]
            regime_sep_ratio_avg = (
                sum(r["regime_sep_ratio"] for r in results.values()
                     if not (isinstance(r["regime_sep_ratio"], float)
                              and r["regime_sep_ratio"] != r["regime_sep_ratio"]))
                / max(1, sum(1 for r in results.values()
                              if not (isinstance(r["regime_sep_ratio"], float)
                                       and r["regime_sep_ratio"] != r["regime_sep_ratio"]))))
            if mae_boundary > 0.05 and regime_sep_ratio_avg < 0.5:
                verdict = "BOUNDARY-BREAK: gap is already large at T-1 AND regime separation < 50% of DP. λ-mix WON'T fix this; look upstream at label-gen / regime conditioning."
            elif mae_deep > 3 * mae_boundary and regime_sep_ratio_avg > 0.5:
                verdict = "COMPOUNDING: gap grows backward AND regime separation is preserved. λ-mix is the right remedy."
            else:
                verdict = ("AMBIGUOUS — gap grows backward by factor "
                            f"{mae_deep/max(mae_boundary,1e-9):.2f}, "
                            f"regime_sep_ratio_avg={regime_sep_ratio_avg:.2f}. "
                            "Inspect per-t numbers before deciding.")
            logging.info("DEPTH PROFILE VERDICT — %s", verdict)
            return {"depth_profile_per_t": results,
                    "depth_profile_verdict": verdict,
                    "mae_boundary": mae_boundary,
                    "mae_deep": mae_deep,
                    "regime_sep_ratio_avg": regime_sep_ratio_avg}
        return {"depth_profile_per_t": results}

    def _grad_sanity_check(self, t):
        """FD vs AAD on ∂C/∂z_wealth at a small sample of fit rows. Also asserts that
        q* (decision argmax) carries no grad (spec §4 invariant). Catches grad leaking
        through the rollout or detach failure, neither of which any value smoke can
        see."""
        if self.C[t] is None:
            return {"grad_sanity_skipped": True}
        device = self.device
        # Build 8 random z rows.
        N = 8
        market_block = torch.zeros(N, self.statepack.market_dim, device=device)
        market_block[torch.arange(N), torch.randint(0, 2, (N,), device=device)] = 1.0
        n_h = len(self.hedges)
        q_cell = torch.zeros(N, n_h, device=device)
        F_cell = torch.full((N, n_h), 100.0, device=device)
        wealth = (torch.rand(N, device=device) - 0.5) * 2000.0
        static_t = self._static_sim[t].to(device).expand(N, self._static_sim.shape[-1])
        time_to_T = torch.full((N,), float(self.t_outer - 1 - t), device=device)
        z = self._assemble_z(market_block, q_cell, wealth, F_cell, time_to_T, static_t)
        wealth_col = self.statepack.n_hedge

        # AAD ∂C/∂z_wealth. Under advantage decomp, C = B + A so the gradient
        # includes both contributions — `_C_full` is the right query.
        z_leaf = z.detach().clone().requires_grad_(True)
        y_pred = self._C_full(self.C[t], z_leaf)
        dy_dz_aad = torch.autograd.grad(y_pred.sum(), z_leaf)[0]
        aad_wealth = dy_dz_aad[:, wealth_col]

        # FD with eps = 0.5 (in dollars — wealth band is ~±1500 so a 0.5 perturbation
        # is well inside the fit hull).
        eps = 0.5
        with torch.no_grad():
            z_plus = z.clone(); z_plus[:, wealth_col] += eps
            z_minus = z.clone(); z_minus[:, wealth_col] -= eps
            y_plus = self._C_full(self.C[t], z_plus)
            y_minus = self._C_full(self.C[t], z_minus)
            fd_wealth = (y_plus - y_minus) / (2 * eps)
        rel_err = ((aad_wealth - fd_wealth).abs() / fd_wealth.abs().clamp_min(1.0e-8)
                  ).mean()
        logging.info(
            "DifferentialSolver grad sanity t=%d: ∂C/∂wealth AAD vs FD "
            "(eps=%.2g) mean |rel err| = %.4f%%; AAD mean = %.5g, FD mean = %.5g",
            t, eps, float(rel_err) * 100.0,
            float(aad_wealth.mean()), float(fd_wealth.mean()))
        # q* detach assert — invoke decision operator and check no grad.
        q_star, _, _ = self._decision_at(
            self.C[t], market_block, F_cell, q_cell, wealth, time_to_T, static_t, t)
        assert not q_star.requires_grad, \
            "q* must not carry autograd (spec §4 invariant)."
        return {
            "grad_sanity_rel_err_wealth_pct": float(rel_err) * 100.0,
            "grad_sanity_aad_mean": float(aad_wealth.mean()),
            "grad_sanity_fd_mean": float(fd_wealth.mean()),
            "q_star_detach_ok": True,
        }

    # --------------------------------------------------------------- solve
    def solve(self):
        """**Milestone 0**: cold-train C_T (= `C[t_outer-1]`) and exit. Validates the
        seam (`sample_exogenous` reads from the canonical buffer) and the StatePack
        layout (gate4 §5 ordering matches existing inner-MC machinery) before any
        backward-step / bootstrap / audit machinery lands.
        """
        logging.info(
            "DifferentialSolver Milestone 0: t_outer=%d, B=%d, b_endo=%d, "
            "hidden=%s, steps=%d, batch=%d",
            self.t_outer, self._B(), self.b_endo,
            self.hidden_sizes, self.train_steps_per_solve, self.train_minibatch)

        traj = self.sample_exogenous()

        # Quick invariant: under TRUE-regime conditioning, the t0 spot is shared
        # (single S_0, no perturbation — enters via the tradable_prices block, not
        # market), while regime[0] is legitimately sampled per path from π_0. So
        # `traj[0]['market']` varies across paths *only* through the regime one-hot
        # encoding. Diagnostic = the fraction of paths in each regime at t=0; the
        # canonical π_0 = (0.5, 0.5) should yield ~50/50.
        m0 = traj[0]["market"]
        regime_t0_freq = m0.mean(dim=0).cpu().tolist()    # avg over paths per market col
        logging.info(
            "DifferentialSolver t0 regime occupancy (per market-block column): %s "
            "(spec π_0 = (0.5, 0.5); spot enters via tradable_prices, not market).",
            ["%.3f" % v for v in regime_t0_freq])

        diag = self._train_C_terminal(traj)
        diag["t0_regime_occupancy"] = regime_t0_freq
        diag["seam_milestone"] = 0
        # v0_estimate diagnostics filled after the no-hedge evaluation below.

        # Milestone-0 V_0 proxy: the average terminal utility, computed by evaluating
        # the fitted C_T at the realized terminal slice with zero inventory (the
        # canonical "hold no hedge to T" baseline). This is the value of NO hedging,
        # not the optimal V_0 — that requires the full backward sweep (M3). For the
        # gate2 toy with utility_scale c (≈1000 in this framework configuration), the
        # baseline E[U(K-S_T)] sets the floor the backward sweep should improve over.
        with torch.no_grad():
            t_T = self.t_outer - 1
            B = self._B()
            market_T = traj[t_T]["market"]                       # (B, market_dim)
            tradables_T = traj[t_T]["tradables"]                 # (B, n_hedge)
            n_h = len(self.hedges)
            positions = torch.zeros(B, n_h, device=self.device)
            # Buy-and-hold wealth at T under no hedge: cash carried forward minus
            # liability cashflow. The framework's gate2-style invariant w = cash +
            # (Σq − 1)·S + K with q=0 gives w_T = cash_T − S_T + K. For Milestone 0
            # we use the bundle's initial wealth (cash carried forward, no funding
            # cost) so w_T = initial_W − S_T + K. Reading S_T off the spot factor
            # snapshot directly — bypasses the deal pricer for this sanity check.
            spot_T = self._outer_buf[self._spot_key][-1].to(self.device)         # (B,)
            initial_w = float(self.bundle.get("initial_wealth", 100.0))
            K_strike = 100.0   # toy spec — promote to JSON in M1.
            wealth_T = initial_w + K_strike - spot_T
            zero_b = torch.zeros(B, device=self.device)
            time_to_t_T = torch.zeros(B, device=self.device)     # terminal: 0
            static_T = self._static_sim[-1].to(self.device).expand(
                B, self._static_sim.shape[-1])
            z_query = _assemble_deep_state(
                positions=positions, g=wealth_T, a=zero_b,
                futures=tradables_T, market=market_T, static=static_T,
                time_to_t=time_to_t_T)
            terminal_values = self._C_full(self.C[t_T], z_query)  # (B,)
            v0_estimate = float(terminal_values.mean())
            # Compare to closed-form expected terminal utility under buy-and-hold
            # for a sanity check.
            c_util = max(float(self.bundle.get("utility_scale", 100.0)), 1.0)
            truth_per_path = torch.sign(wealth_T) * torch.log1p(wealth_T.abs() / c_util)
            truth_mean = float(truth_per_path.mean())
            v0_residual = v0_estimate - truth_mean

        logging.info(
            "DifferentialSolver Milestone 0 complete: E[U(W_T) | no-hedge] via "
            "fitted C_T = %.4f; closed-form = %.4f; residual = %.4f (NN fit error). "
            "Backward sweep deferred to M1+.",
            v0_estimate, truth_mean, v0_residual)

        # --- Backward sweep: fit C_{t} for t = T_outer-2 down to t_min with M2's
        # audit-correct loop. At sweep entry we partition the B_outer paths into one
        # training subset and MAX_ROUNDS mutually-disjoint audit slices. Per the
        # user's correction: a single reserved holdout reused across rounds leaks
        # info into the correction-row placement and drifts optimistic; disjoint
        # per-round slices preserve the §7 re-audit-fresh invariant within one
        # outer sweep (no per-round re-sim cost).
        max_rounds = int(self.runtime["solver"].get("audit_max_rounds", 3))
        B = self._B()
        # Reserve ~25% for audit (split into MAX_ROUNDS slices); the rest is training.
        n_audit_total = max(max_rounds * 8, int(B * 0.25))
        n_audit_total = min(n_audit_total, B // 2)        # cap at 50%
        n_train = B - n_audit_total
        perm = torch.randperm(B, device=self.device)
        training_outer_idx = perm[:n_train]
        audit_slice_size = n_audit_total // max_rounds
        audit_slices_global = [
            perm[n_train + r * audit_slice_size : n_train + (r + 1) * audit_slice_size]
            for r in range(max_rounds)
        ]
        logging.info(
            "DifferentialSolver M2 setup: B=%d  training=%d  audit=%d ÷ %d rounds × %d/slice  "
            "(disjoint per spec §7 re-audit-fresh invariant)",
            B, n_train, n_audit_total, max_rounds, audit_slice_size)
        per_t_diag = {}
        logging.info(
            "DifferentialSolver backward sweep: t = %d → %d (T_A live transition "
            "expected at the step where live_axes drops from 2 → 1)",
            self.t_outer - 2, self.t_min)
        for t_step in range(self.t_outer - 2, self.t_min - 1, -1):
            t_diag = self._train_C_at_step(
                t_step, self.C[t_step + 1], traj,
                training_outer_idx=training_outer_idx,
                audit_slices=audit_slices_global)
            per_t_diag[t_step] = t_diag
            logging.info(
                "DifferentialSolver C[%d] fitted: live_axes=%d  rounds=%d  "
                "twin val=%.5f diff=%.5f  bank=%d%s",
                t_step, t_diag["live_axes_at_t"], t_diag.get("rounds_taken", 1),
                t_diag["twin_value_loss"], t_diag["twin_diff_loss"],
                t_diag["bank_rows"],
                " ESCALATED" if t_diag.get("escalation_flag") else "")

        # Grad sanity at three representative t values: end-of-sweep (t_min),
        # mid-sweep, and just-past-the-T_A-transition. Catches grad propagation
        # bugs that any single-t smoke misses.
        t_sample = sorted({self.t_min, self.t_outer - 2,
                            (self.t_outer - 2 + self.t_min) // 2})
        grad_sanity = {}
        for t_s in t_sample:
            if self.C[t_s] is not None:
                grad_sanity[t_s] = self._grad_sanity_check(t_s)

        # Oracle comparison at the mid t (still single-contract for T_dec=10 / T_A=5
        # if mid > T_A; gives the c=1000 vs c=100 scale-mismatch baseline that M4 will
        # reconcile, plus a "shape matches" diagnostic in the meantime).
        mid_t = (self.t_outer - 2 + self.t_min) // 2
        oracle_diag = self._smoke_against_gate2(mid_t, 'artifacts/gate2_exact_dp.npz')

        # M2 final grade: regime-conditional `n*_0` against gate2's initial_policy.
        # Per the user's mandate: V_0 climbing is necessary but not sufficient — the
        # policy must become regime-conditional and match the DP's actions. This is
        # the smoke check that exercises whether the audit-correct loop actually
        # un-biased the policy (vs just shifting V_0 toward the right number).
        oracle_policy_diag = self._grade_against_oracle_policy(
            'artifacts/gate2_exact_dp.npz', traj)

        # V_0 estimate via the decision operator at t_min: q*_t_min = argmax_q
        # C[t_min](post_state_t_min(q)). For the M1.5 toy (t_min = T_A - 1) this is
        # the value at the first multi-contract decision step. For M3 (t_min = 0)
        # this is the true V_0.
        with torch.no_grad():
            market_tm = traj[self.t_min]["market"]
            tradables_tm = traj[self.t_min]["tradables"]
            B = market_tm.shape[0]
            zero_q = torch.zeros(B, n_h, device=self.device)
            # Initial wealth at t_min — gate2 MTM-invariant convention: w_0 = cash +
            # (Σq − 1)·S + K. With q=0, cash=initial_W (=100), S=S_0 (=100), K=100:
            # w_t_min ≈ initial_W (under no-rebalance forward sim from t=0).
            initial_w = float(self.bundle.get("initial_wealth", 100.0))
            wealth_tm = torch.full((B,), initial_w, device=self.device)
            time_to_T_tm = torch.full((B,), float(self.t_outer - 1 - self.t_min),
                                       device=self.device)
            static_tm = self._static_sim[self.t_min].to(self.device).expand(
                B, self._static_sim.shape[-1])
            if self.C[self.t_min] is not None:
                q_opt, v_opt, gap_opt = self._decision_at(
                    self.C[self.t_min], market_tm, tradables_tm, zero_q,
                    wealth_tm, time_to_T_tm, static_tm, self.t_min)
                v0_decision = float(v_opt.mean())
                q_opt_mean = q_opt.mean(dim=0).cpu().tolist()
            else:
                v0_decision = float("nan")
                q_opt_mean = None

        # Aggregate sweep diagnostics.
        sweep_summary = {
            "t_min": self.t_min,
            "t_max": self.t_outer - 2,
            "C_fitted_count": sum(1 for c in self.C if c is not None),
            "per_t_diagnostics": {
                t_step: {k: v for k, v in d.items()
                          if isinstance(v, (int, float))}
                for t_step, d in per_t_diag.items()
            },
            "grad_sanity_at_t": {t_s: g.get("grad_sanity_rel_err_wealth_pct",
                                            float("nan"))
                                  for t_s, g in grad_sanity.items()},
            "v0_decision_at_t_min": v0_decision,
            "q_opt_t_min_mean": q_opt_mean,
        }
        sweep_summary.update(oracle_diag)
        sweep_summary.update(oracle_policy_diag)
        # Depth-profile diagnostic — decisive on whether the M2 residual is
        # COMPOUNDING (λ-mix fix) or BOUNDARY BREAK (label-gen/conditioning fix).
        depth_diag = self._depth_profile_diagnostic(
            'artifacts/gate2_exact_dp.npz', traj)
        sweep_summary.update(depth_diag)
        action_depth_diag = self._action_error_vs_depth(
            'artifacts/gate2_exact_dp.npz', traj)
        sweep_summary.update(action_depth_diag)
        logging.info(
            "DifferentialSolver sweep complete: %d C_t functions fitted "
            "(t ∈ [%d, %d]); V_0_decision @ t=%d = %.4f; n*_t_min mean = %s",
            sweep_summary["C_fitted_count"], self.t_min, self.t_outer - 1,
            self.t_min, v0_decision, q_opt_mean)
        # M2 engagement trajectory — per-t rounds, escalation flag, and δ history.
        # Each value is one t step; pattern shows whether audit-correct converged
        # (rounds<MAX), fixed via correction (rounds>1 then PASS), or escalated
        # (label bias detected — spec §14 λ-mix territory).
        diff_engagement = [d["twin_diff_loss"] for _, d in sorted(per_t_diag.items())]
        rounds_per_t = [d.get("rounds_taken", 1) for _, d in sorted(per_t_diag.items())]
        escalated_t = [t_step for t_step, d in sorted(per_t_diag.items())
                        if d.get("escalation_flag")]
        logging.info(
            "DifferentialSolver M2 trajectory (t %d→%d): "
            "rounds_per_t=[%s], diff_loss=[%s], escalated_t=%s",
            self.t_outer - 2, self.t_min,
            ", ".join(str(r) for r in rounds_per_t),
            ", ".join("%.3g" % v for v in diff_engagement),
            escalated_t)
        m1_diag = sweep_summary

        # Return a SolverResult so the dispatcher in solve_hedge can package it.
        b = self._B()
        actions_stacked = torch.zeros(self.t_outer, b, n_h)
        values_stacked = torch.full((self.t_outer, b), float("nan"))
        values_stacked[self.t_outer - 1] = v0_estimate
        return SolverResult(
            solver_name="DifferentialSolver",
            actions=actions_stacked.detach().cpu(),
            values=values_stacked.detach().cpu(),
            value_fn_artifacts={},
            diagnostics={
                "V_0": v0_estimate,                # dispatcher convention
                "n_star_0": torch.zeros(n_h).tolist(),
                "v0_estimate": v0_estimate,
                "v0_closedform_no_hedge": truth_mean,
                "v0_nn_fit_residual": v0_residual,
                "milestone": 1,
                **diag,
                **{f"M1_{k}": v for k, v in m1_diag.items()},
            },
        )


_SOLVERS: Dict[str, Callable] = {
    "mpcsolver": MpcSolver,
    "lsmdpsolver": LsmDpSolver,
    "hindsightdpsolver": HindsightDpSolver,
    "differentialsolver": DifferentialSolver,
}


def _acceptance_ladder(comparison):
    """The brief's acceptance ordering — hindsight ≥ DP ≥ MPC ≥ textbook — over whatever
    tracks are present. `holds` allows a tiny tolerance for Monte-Carlo noise."""
    order = [("HindsightDpSolver", "hindsight"), ("LsmDpSolver", "DP"),
             ("MpcSolver", "MPC"), ("textbook", "textbook")]
    rungs = [(label, comparison[key]["v0_mean"])
             for key, label in order if key in comparison]
    holds = all(rungs[i][1] >= rungs[i + 1][1] - 1.0e-6
                for i in range(len(rungs) - 1))
    return {"order": rungs, "holds": holds}


def solve_hedge(bundle, runtime):
    """Dispatcher + orchestration for `Execution_Mode='solve_hedge'`. Runs the configured
    `Solver.Object`; when that is the LSM/DP deliverable it also assembles the reference
    tracks (MPC / hindsight / textbook) enabled by the `Run_*` flags into a `comparison`
    table — V_0 mean ± std per track — plus the acceptance ladder. Multi-seed repeats
    re-use the cached outer paths but advance the inner-MC Sobol stream.

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

    # Reference tracks — assembled alongside the LSM/DP deliverable.
    if obj == "lsmdpsolver":
        if solver_cfg.get("run_mpc_comparison"):
            comparison["MpcSolver"] = _multiseed_summary(
                [MpcSolver(bundle, runtime).solve() for _ in range(n_seed)])
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
