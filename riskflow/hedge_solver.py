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
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import pandas as pd
import torch
from scipy.linalg import expm as matrix_expm, logm as matrix_logm

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
    DifferentialSolver. Small MLP, **softplus** activation
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
        final = torch.nn.Linear(prev, 1)
        # Zero-init the OUTPUT layer so the residual A_t ≡ 0 at init (in standardized
        # space): C_t = B_t(z) + A_t starts at exactly the analytic baseline rather than
        # baseline + random high-frequency structure. This is the toy's design (it zeros
        # its final Linear so A starts at 0) and the natural start under advantage decomp,
        # where A is the SMALL residual to learn — a strictly easier optimization start
        # than default Kaiming init. A fresh net is built per backward step, so every C_t
        # benefits. The standardized output is then `0·y_std + y_mean = y_mean` (the bank's
        # mean residual), and the differential label starts from a zero slope.
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        layers.append(final)
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
              w_val=1.0, w_diff=1.0, col_mask=None):
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

    Column mask (`col_mask`, optional `(D,)`): restricts the gradient MSE to the
    columns where the label pipeline actually computed a pathwise derivative. The
    one-step AAD label populates ONLY the wealth column and the live tradable_prices
    columns; every other column of `dy_dz_target` is an uncomputed zero, not a
    measured slope. Fitting against those zeros is wrong for the advantage-decomp
    residual (whose positions-slope is `−∂B/∂q` away from zero), so the residual
    path passes a mask and unlabeled columns contribute nothing. `None` (default)
    reproduces the legacy all-columns mean exactly. The per-column mean is taken
    over ACTIVE columns (`sum/Σmask`) so the diff-term scale is independent of how
    many unlabeled columns the state layout carries.

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

    sq_resid = (dyn_dzn_pred - dyn_dzn_target) ** 2                 # (N, D)
    if col_mask is not None:
        cm = col_mask.to(sq_resid.dtype)
        diff_per_row = (sq_resid * cm).sum(dim=-1) / cm.sum().clamp_min(1.0)
    else:
        diff_per_row = sq_resid.mean(dim=-1)                        # (N,)
    if action_gap is not None and gap_threshold is not None and gap_threshold > 0:
        mask = (action_gap / float(gap_threshold)).clamp(0.0, 1.0)
        diff_loss = (diff_per_row * mask).mean()
        mask_mean = float(mask.mean())
    else:
        diff_loss = diff_per_row.mean()
        mask_mean = 1.0

    total = w_val * val_loss + w_diff * diff_loss
    return total, {
        "total_loss": float(total.detach()),
        "val_loss": float(val_loss.detach()),
        "diff_loss": float(diff_loss.detach()),
        "mask_mean": mask_mean,
    }


def construct_twin_network(deep_dim, runtime, device=None):
    """Factory for the diff-ML continuation function. Reads `Solver.Value_Fn.MLP_Hidden`
    for the layer widths; defaults to `(128, 128, 128)` for the diff-ML twin network.
    Device defaults to the bundle's primary device when one is reachable through
    `runtime`; callers pass it explicitly otherwise."""
    vf = runtime["solver"].get("value_fn", {})
    hidden = tuple(vf.get("mlp_hidden", (128, 128, 128)))
    return TwinNetwork(deep_dim, hidden_sizes=hidden, device=device)


def build_cumulative_regime_drift(P_calib, dt_calib_years, dt_years_outer,
                                  mu_per_state):
    """Expected cumulative drift table under regime propagation
    (validation_sandwich_spec.md §2: "propagate belief under the transition
    matrix; take expected cumulative drift" — NOT today's belief-weighted μ̄
    held constant over the horizon).

        cum[a, k, i] = E[ Σ_{u=a}^{a+k-1} μ_{r_u} · dt_u  |  r_a = i ]

    so the belief-mixture `b · expm1(cum[a, k])` is the first-order
    `E[S_{a+k}/S_a] − 1` from belief `b` at outer step `a`. Backward
    recurrence `cum[a, k] = μ·dt_a + P_a @ cum[a+1, k−1]` with the
    calibrated transition re-discretised per outer step through the CTMC
    generator (`Q = logm(P_calib)/dt_calib`, `P_a = expm(Q·dt_a)`) — the same
    route the simulator uses, so the table matches the dynamics that generate
    the paths. Steps beyond the grid extend time-homogeneously with the last
    (P, dt); production queries never reach the extension (hedge horizon ≤
    remaining steps; liability fixings inside the grid by terminal
    consistency).

    With `P_calib = I` the table collapses to `cum[a, k, i] = μ_i · Σ dt` —
    the pre-propagation constant-drift form — so a one-hot belief reproduces
    the previous `expm1(μ_r · ttot)` baseline exactly (regression anchor).

    Args: `P_calib` (n, n) numpy/list, `dt_calib_years` float, `dt_years_outer`
    1-D array of per-outer-step dt in years (length T−1 for T grid points),
    `mu_per_state` (n,) torch tensor. Returns torch (T, T+1, n) on
    `mu_per_state`'s device/dtype, where row `a` covers horizons k ∈ [0, T].
    """
    import numpy as _np
    mu = mu_per_state
    n = mu.shape[0]
    dt_arr = [float(d) for d in dt_years_outer]
    T = len(dt_arr) + 1
    P_calib = _np.asarray(P_calib, dtype=_np.float64)
    if _np.allclose(P_calib, _np.eye(n)):
        Q = _np.zeros((n, n))
    else:
        Q = _np.real(matrix_logm(P_calib)) / float(dt_calib_years)
    P_steps = [
        torch.as_tensor(
            _np.real(matrix_expm(Q * dt)) if dt > 1.0e-12 else _np.eye(n),
            device=mu.device, dtype=mu.dtype)
        for dt in dt_arr]
    # Time-homogeneous extension values for the (unreached) tail.
    dt_last = dt_arr[-1] if dt_arr else 7.0 / 365.25
    P_last = P_steps[-1] if P_steps else torch.eye(n, device=mu.device,
                                                   dtype=mu.dtype)
    cum = torch.zeros(T, T + 1, n, device=mu.device, dtype=mu.dtype)
    # Virtual row "a = T" (beyond grid): cum_ext[k] = μ·dt_last + P_last @ cum_ext[k−1].
    cum_ext = torch.zeros(T + 1, n, device=mu.device, dtype=mu.dtype)
    for k in range(1, T + 1):
        cum_ext[k] = mu * dt_last + P_last @ cum_ext[k - 1]
    for a in range(T - 1, -1, -1):
        dt_a = dt_arr[a] if a < len(dt_arr) else dt_last
        P_a = P_steps[a] if a < len(P_steps) else P_last
        nxt = cum[a + 1] if a + 1 < T else cum_ext
        for k in range(1, T + 1):
            cum[a, k] = mu * dt_a + P_a @ nxt[k - 1]
    return cum


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
                    # underlying mapping; deferred.
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
                        # Aggregate spot gradients across all CommodityPrice leaves (single-underlying
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


def _interp_wealth_grid(xs, ys, q):
    """Linear interpolation of per-path values `ys` (P, G) over the shared wealth grid
    `xs` (G,), queried at `q` (P, …). Port of `diffml_hedge_huber._interp_path`.

    Queries are CLAMPED to `[xs[0], xs[-1]]`; the caller measures the clamp (off-grid)
    fraction SEPARATELY and treats it as the discretised-DP bias gate (memory rule
    `discretised-DP-boundary-clamp`: silent off-grid clamping biases `U` and can invert
    the optimal policy direction — it must be a permanent diagnostic, not a hidden detail)."""
    P, G = ys.shape
    qf = q.reshape(P, -1)
    qc = qf.clamp(float(xs[0]), float(xs[-1]))
    idx = torch.searchsorted(xs, qc).clamp(1, G - 1)          # (P, M)
    x0 = xs[idx - 1]
    x1 = xs[idx]
    y0 = ys.gather(1, idx - 1)
    y1 = ys.gather(1, idx)
    w = (qc - x0) / (x1 - x0)
    return (y0 + w * (y1 - y0)).reshape(q.shape)


class DifferentialSolver:
    """Differential-ML dynamic-hedging solver. Casts each
    per-step value function as a smooth post-decision continuation `C_t : z_t → scalar`
    fit by supervised twin loss (value + AAD gradient labels) over a controlled bank,
    with the Bellman max moved **out** of the regression into an explicit external
    operator applied only to the already-frozen `C_{t+1}`. This removes the
    "max-inside-fit + extrapolation clamp" pathology that pins LsmDpSolver to off-
    distribution values.

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
        self.train_loss_tol = float(vf_cfg.get("mlp_loss_tol", 0.0))
        self.train_minibatch = int(vf_cfg.get("mlp_minibatch", 4096))
        self.adam_lr = float(vf_cfg.get("mlp_adam_lr", 1.0e-3))
        # Bank construction knobs (spec §6).
        bank_cfg = solver_cfg.get("bank_sampling", {})
        self.b_endo = int(bank_cfg.get("b_endo", 2))
        # Bootstrap value-label fanout. The realised-path branch below still supplies the
        # pathwise differential label, but the VALUE target can average the Bellman envelope
        # over the retained one-step inner fork instead of training on one realised next state.
        # This directly attacks the high-variance single-sample target that compounds backward.
        self._bootstrap_inner_samples = max(1, int(solver_cfg.get("bootstrap_inner_samples", 1)))
        if self._bootstrap_inner_samples > 1:
            logging.info(
                "DifferentialSolver bootstrap value labels: inner expectation ACTIVE "
                "with up to %d one-step samples per bank row.",
                self._bootstrap_inner_samples)
        # Backward sweep depth — t_min = 0 is the full sweep (Milestone 3); M1.5
        # uses t_min = T_A − 1 to validate the multi-contract path across the T_A
        # transition without committing to the full 30-step sweep. Configurable via
        # JSON `Solver.T_Min`; default 0 = sweep all the way to the initial decision.
        self.t_min = int(solver_cfg.get("t_min", 0))
        # BSS sandwich (validation_sandwich_spec §4.1): the penalty's zero-mean dual-feasibility
        # is graded on the INTERIOR; the boundary is DATA-DRIVEN, not a magic count. Near the
        # terminal the inner fork's latent-regime resimulation disperses the one-step move a few %
        # more than the outer sim (a variance, not a mean, effect — drift/dt/σ are identical),
        # which under high near-terminal continuation curvature leaves a small Δ>0 ramp at the
        # last held step(s). is_live does NOT mark this (the longest contract stays live to
        # terminal; the flatten is force_flat_at_end, not expiry). So the boundary = the trailing
        # block grown inward until the remaining interior sum is zero-mean (|mean|/stderr <
        # `Penalty_ZeroMean_Z`), capped at `Penalty_Max_Boundary`. U zeros π on exactly that
        # block (zero is the dual-feasible choice — keeps U a valid, safe-loose bound); a gross
        # whole-profile dynamics bug makes the block blow past the cap → caught.
        self._penalty_zero_mean_z = float(solver_cfg.get("penalty_zero_mean_z", 3.0))
        self._penalty_max_boundary = int(solver_cfg.get("penalty_max_boundary", 4))
        # BSS sandwich step 3 — penalized clairvoyant upper bound `U` (the wealth-grid
        # backward DP of `diffml_hedge_huber.penalized_upper`). All offline/diagnostic.
        # `U` is the envelope×wealth-grid×inner-fork DP — cost ∝ P·G·K²·I per step — so the
        # path count, grid size and inner-fork count are capped (scope shallow/few-live first;
        # raise via JSON for a tighter, more expensive bound).
        self._run_upper_bound = bool(solver_cfg.get("run_upper_bound", False))
        self._upper_max_paths = int(solver_cfg.get("upper_bound_max_paths", 128))
        self._upper_n_grid = int(solver_cfg.get("upper_bound_wealth_grid", 41))
        self._upper_n_inner = int(solver_cfg.get("upper_bound_n_inner", 4))
        self._upper_grid_pad = float(solver_cfg.get("upper_bound_grid_pad", 1.0))
        self._upper_chunk_rows = int(solver_cfg.get("upper_bound_chunk_rows", 200_000))
        # Off-grid (clamp) fraction above which `U` is flagged biased (memory:
        # discretised-DP-boundary-clamp — silent clamping biases U / can invert the policy).
        self._upper_clamp_warn = float(solver_cfg.get("upper_bound_clamp_warn_frac", 0.02))
        # C-stack persistence (OUT-OF-SAMPLE validation). `save_value_fn_path`: write the fitted
        # twin-net stack after the sweep. `load_value_fn_path`: load it, SKIP training, and run
        # the L/π/U sandwich on THIS run's (fresh-seeded) outer batch. Two JSON-config variants
        # (train → save; eval → load + new Random_Seed) — never both in one run.
        self._save_value_fn_path = solver_cfg.get("save_value_fn_path") or None
        self._load_value_fn_path = solver_cfg.get("load_value_fn_path") or None
        self._oracle_action_match_path = solver_cfg.get("oracle_action_match_path") or None
        # Label audit (opt-in, diagnostic): timesteps at which to snapshot the actual bootstrap
        # labels the net is asked to fit — Y_boot, the analytic baseline B_t, and the RESIDUAL
        # y_target = Y_boot − B_t (what the NN regresses to; should be SMALL under advantage
        # decomp). Confirms the labels are sane vs exploding (label-bias ⇒ |residual| blows up).
        self._label_audit_t_steps = set(int(x) for x in solver_cfg.get("label_audit_t_steps", []))
        self._label_audit = {}

        self.hedges = list(runtime["names"]["hedges"])
        # T_outer-1 decision points: t=0..T-2 (DP convention). C[T_outer-1] is the
        # boundary anchor — fit to closed-form terminal utility, no bootstrap.
        tradables_sim, static_sim, t_outer = _bundle_sim_views(bundle)
        self._tradables_sim = tradables_sim
        self._static_sim = static_sim
        self.t_outer = t_outer

        # State pack — reuse the existing layout. Market block carries the regime
        # belief for a regime-switching spot; dim is whatever `_run_inner_mc_at_t` would publish.
        # For Milestone 0 we don't need the inner-MC machinery; only the terminal
        # slice of the outer buffer. Market dim derived from privileged-mode
        # extraction at t=0 (any t works — same per-process widths).
        market_t0 = self._read_privileged_market(t=0)
        market_dim = market_t0.shape[-1]
        self.statepack = StatePack.from_bundle(bundle, runtime, market_dim)

        # Locate the spot factor's sub-coordinates within the market block. The privileged
        # CommodityPrice block is [belief(n_states), spot_price(1)] (calculation.py
        # `_extract_outer_state_at`): split it so the belief differential reads the regime
        # columns and the baseline / spot label read the spot-price column. Walk the same
        # factor-iteration order `_read_privileged_market` concatenates in. Both default to
        # None (factor absent / no belief / no price) and callers guard accordingly.
        self._regime_cols_in_market = None
        self._spot_cols_in_market = None
        _spot_proc0 = self._calc.stoch_factors_inner.get(self._spot_key)
        _n_states0 = int(getattr(_spot_proc0, 'n_states', 0) or 0)
        _col = 0
        for _key in self._calc.stoch_factors_inner:
            if _key.type in utils.DimensionLessFactors:
                continue
            _width = self._calc._extract_outer_state_at(
                0, _key, self._outer_buf, privileged=True).shape[0]
            if _key == self._spot_key and _width > 0:
                self._regime_cols_in_market = (_col, _col + _n_states0)
                if _width > _n_states0:
                    self._spot_cols_in_market = (_col + _n_states0, _col + _width)
                break
            _col += _width

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

        # Accounting constants pulled from the runtime (JSON) — no hardcoded defaults. The wealth
        # convention is MTM-invariant: `w_t = cash + (Σq − 1)·S_t + K`,
        # so under no rebalance `w_{t+1} = w_t + (Σq − 1)·dS`; under rebalance,
        # `w_post = w_pre − turnover`.
        self.lambda_turn = float(runtime["accounting"].get(
            "transaction_cost_per_unit", 1.0e-4))
        # Real per-tradable contract sizes (50 oz for platinum futures). Was hardcoded
        # to `ones(n_h)` which broke any deal where the
        # tradable's price wasn't already denominated in deal units.
        self.contract_size = torch.tensor(
            [float(runtime["tradables"][h]["contract_size"]) for h in self.hedges],
            device=self.device, dtype=self.dtype)
        # `bundle['liability_mtm']` is not required by the diff-ML path — we use
        # the closed form `dL = R_t · dS` instead (AAD-aware), which reduces to
        # the framework's realised MTM change by construction for linear-average
        # deals. The framework computes liability_mtm for the other solvers; we
        # don't need it on the diff-ML training/rollout path.
        self.utility_c = max(float(bundle.get("utility_scale", 100.0)), 1.0)

        # Bank-sampling ranges, derived from the runtime — no separate JSON knob.
        # `position_limits` is the same source `build_action_grid` reads (above), so
        # the sampling range is consistent with the decision search space at every t.
        # Wealth half-width sized to the symlog scale: 15 × c covers ±2.78 in symlog
        # units, the meaningful dynamic range of the activation; e.g.
        # (`utility_scale=100`) this lands on the prior ±$1500, at the production
        # deal (`vol_scaled_notional ≈ deal notional`) it scales with the deal.
        limits = runtime["accounting"].get("position_limits", {})
        if not limits:
            raise ValueError(
                "DifferentialSolver requires `Evaluator.Position_Limits` per hedge; "
                "got an empty dict. Set Min_Position / Max_Position per tradable.")
        q_lo = [int(float(limits[h]["min_position"])) for h in self.hedges]
        q_hi = [int(float(limits[h]["max_position"])) for h in self.hedges]
        self.bank_q_min = torch.tensor(q_lo, dtype=torch.long, device=self.device)
        self.bank_q_max = torch.tensor(q_hi, dtype=torch.long, device=self.device)
        self.bank_wealth_halfwidth = 15.0 * self.utility_c
        logging.info(
            "DifferentialSolver bank ranges: q_min=%s, q_max=%s, "
            "wealth_halfwidth=%.4g (= 15·utility_c, utility_c=%.4g), "
            "contract_size=%s",
            q_lo, q_hi, self.bank_wealth_halfwidth, self.utility_c,
            self.contract_size.cpu().tolist())

        # Linear average-rate liability: `R_t` (remaining-fixing-weights sum, in
        # deal units) drives the AAD-aware wealth update `dL = R_t · dS`.
        # Per-fixing lag weights `w_by_lag` feed the propagated baseline's
        # `_liab_drift_state` (built in the advantage-decomp init below;
        # supersedes the τ̄ duration approximation inside B). `τ̄_t` is retained
        # for diagnostic logs only. Weights = δ_T (single payment at terminal)
        # reduces to R_t ≡ 1 with a terminal-lag ladder — bit-for-bit.
        self._init_liability_R_tau(runtime, bundle)

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
            # dt (constant for a uniform grid). The regime cols in the market block
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
                    self._dt_per_step = 7.0 / 365.25      # degenerate single-point-grid fallback
                # Regime col range was located once in __init__ (`_regime_cols_in_market`,
                # split from the [belief, spot_price] block); reuse it here.
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
                    # --- Propagated regime drift (validation_sandwich_spec §2) ---
                    # Build cum_mu_state[a, k, i] = E[Σ_{u<k} μ·dt | r_a = i] from
                    # the spot factor's calibrated transition matrix via the same
                    # generator-rediscretise route the simulator uses. A constant
                    # belief-weighted μ̄ over a switching horizon forces A_t to
                    # absorb the regime mixing (spec §5 — the depth-growing
                    # over-optimism mechanism); the propagated table is the
                    # closed-form remedy and stays linear in the belief, so
                    # `_dB_dz` (autograd through `_baseline_B`) needs no change.
                    days_T = [int(d) for d in days[:self.t_outer]]
                    dt_years_outer = [
                        (days_T[i + 1] - days_T[i]) / 365.25
                        for i in range(len(days_T) - 1)]
                    P_calib = (spot_proc.param or {}).get('Transition_Matrix') \
                        if hasattr(spot_proc, 'param') else None
                    dt_calib = float((spot_proc.param or {}).get(
                        'Calibration_DT_Years', 1.0 / 252.0)) \
                        if hasattr(spot_proc, 'param') else 1.0 / 252.0
                    if P_calib is None:
                        n_states = int(self._mu_per_regime.shape[0])
                        P_calib = torch.eye(n_states).tolist()
                        logging.warning(
                            "DifferentialSolver advantage decomp: spot factor has "
                            "no Transition_Matrix — drift propagation degrades to "
                            "constant per-state drift (P = I).")
                    self._cum_mu_state = build_cumulative_regime_drift(
                        P_calib, dt_calib, dt_years_outer, self._mu_per_regime)
                    # Liability drift per (t, state): fixing-ladder-exact combine
                    # of the lag-bucket weights (built in `_init_liability_R_tau`)
                    # with the propagated table. Supersedes the duration-bucket
                    # τ̄ approximation inside the baseline (τ̄ is kept for logs;
                    # `R_t` is still the wealth-evolution `dL = R_t·dS` source).
                    self._liab_drift_state = (
                        self._liab_w_by_lag.unsqueeze(-1)
                        * torch.expm1(self._cum_mu_state)).sum(dim=1)   # (T, n)
                    logging.info(
                        "DifferentialSolver propagated drift tables built: "
                        "cum_mu_state=%s, liab_drift_state[0]=%s",
                        tuple(self._cum_mu_state.shape),
                        [round(float(x), 5) for x in self._liab_drift_state[0]])

    def _init_liability_R_tau(self, runtime, bundle):
        """Build `self._liability_R` (shape t_outer, deal-unit exposure remaining
        at each t) and `self._liability_tau_bar_years` (duration-weighted remaining
        time to fixing, in years) from the linear-average-rate liability cashflows.

        For a deal with `Period_Start..Period_End` averaged-rate cashflows, each
        calendar day in the period is a fixing with weight `Volume / N_days`.
        - `R_t = Σ_{u : fix_u > t} w_u`   (remaining exposure)
        - `τ̄_t = (Σ w_u · (fix_u − t)) / R_t`   (duration-weighted remaining time)

        A single terminal payment (weights = δ_T) gives R_t ≡ 1 for
        t < T_dec, τ̄_t = ttot — reproducing the prior code bit-for-bit.
        """
        import pandas as pd
        from bisect import bisect_right
        hist = int(bundle.get('initial_time_index', 0))
        days = bundle['time_grid_days_cpu'][hist:hist + self.t_outer]
        base_date = pd.Timestamp(bundle['meta']['base_date'])
        T = self.t_outer
        days_int = [int(d) for d in days]
        R_t = torch.zeros(T, device=self.device, dtype=self.dtype)
        wt_dt = torch.zeros(T, device=self.device, dtype=self.dtype)
        # Lag-bucket weights: w_by_lag[t, k] = liability weight landing k outer
        # steps after t (fractional lags lerp-split between ⌊k⌋ and ⌈k⌉). Consumed
        # by the advantage-decomp init to build the fixing-ladder-exact per-state
        # liability drift `Σ_k w_by_lag[t, k] · expm1(cum_mu_state[t, k])`.
        w_by_lag = torch.zeros(T, T + 1, device=self.device, dtype=self.dtype)

        def _frac_pos(fix_day):
            """Grid-relative fractional index of `fix_day` on `days_int`."""
            j = bisect_right(days_int, fix_day) - 1
            if j < 0:
                return 0.0
            if j >= len(days_int) - 1:
                return float(len(days_int) - 1)
            span = days_int[j + 1] - days_int[j]
            return j + (fix_day - days_int[j]) / max(span, 1)

        for name, lib in runtime.get('liabilities', {}).items():
            items = (lib.get('params', {}).get('Payments') or {}).get('Items') or []
            for cf in items:
                volume = float(cf.get('Volume', 0.0))
                ps_raw = cf.get('Period_Start')
                pe_raw = cf.get('Period_End')
                # Handle the framework's {.Timestamp: 'YYYY-MM-DD'} envelope.
                if isinstance(ps_raw, dict):
                    ps_raw = ps_raw.get('.Timestamp')
                if isinstance(pe_raw, dict):
                    pe_raw = pe_raw.get('.Timestamp')
                if ps_raw is None or pe_raw is None or volume == 0.0:
                    continue
                ps = pd.Timestamp(ps_raw)
                pe = pd.Timestamp(pe_raw)
                n_fix_days = max(int((pe - ps).days) + 1, 1)
                weight_per_day = volume / n_fix_days
                for d_offset in range(n_fix_days):
                    fix_day = int((ps + pd.Timedelta(days=d_offset) - base_date).days)
                    pos = _frac_pos(fix_day)
                    for t_idx in range(T):
                        day_t = days_int[t_idx]
                        if fix_day > day_t:
                            R_t[t_idx] = R_t[t_idx] + weight_per_day
                            wt_dt[t_idx] = wt_dt[t_idx] + weight_per_day * (fix_day - day_t)
                            lag = min(max(pos - t_idx, 0.0), float(T))
                            k0 = int(lag)
                            fr = lag - k0
                            w_by_lag[t_idx, k0] += weight_per_day * (1.0 - fr)
                            if fr > 0.0 and k0 + 1 <= T:
                                w_by_lag[t_idx, k0 + 1] += weight_per_day * fr

        tau_days = torch.where(R_t > 1e-12, wt_dt / R_t.clamp_min(1e-12),
                                torch.zeros_like(R_t))
        self._liability_R = R_t                                # (T,) deal units
        self._liability_tau_bar_years = tau_days / 365.25       # (T,) years
        self._liab_w_by_lag = w_by_lag                           # (T, T+1)
        # Validation diagnostic: a single terminal payment gives R_t ≡ 1; a multi-fixing
        # average-rate liability decays R_t over the averaging window — exercising the
        # general dL = R_t·dS that a single-payment liability never does.
        logging.info(
            "DifferentialSolver liability R_t: first=%.4g last=%.4g min=%.4g max=%.4g "
            "constant=%s (constant ⇒ single terminal payment; decaying ⇒ averaging window)",
            float(R_t[0]), float(R_t[-1]), float(R_t.min()), float(R_t.max()),
            bool(torch.allclose(R_t, R_t[0])))
        if float(R_t[-1]) > 1e-9:
            logging.warning(
                "DifferentialSolver liability: R[t_outer-1] = %.4g > 0 — fixings "
                "remain after the last decision step. The terminal anchor trains "
                "A_T ≡ 0 (C_T = B_T = symlog of fully-realised wealth), which is "
                "only consistent when the decision horizon covers the averaging "
                "period (brief: T_dec = last fixing). Check the deal/horizon "
                "configuration before trusting the sweep.", float(R_t[-1]))
        logging.info(
            "DifferentialSolver liability R/τ̄ — R[0]=%.3f, R[T_dec]=%.3f; "
            "τ̄_years[0]=%.4f, τ̄_years[T_dec]=%.4f",
            float(R_t[0]), float(R_t[-1]),
            float(self._liability_tau_bar_years[0]),
            float(self._liability_tau_bar_years[-1]))
        self._audit_liability_convention(R_t)

    def _audit_liability_convention(self, R_t):
        """Record the liability-mark gaps that justify using the framework MTM buffer.

        Both the realised label path and the inner bootstrap mark the liability from the
        framework's MTM buffer (`dL = liability_mtm[t+1] − liability_mtm[t]`, Jacobian-
        reconstructed off the realised path). This logs how far the two closed-form shortcuts
        fall from that ground truth, so nobody reverts to them: (a) `R_t·dS` is ~30–45% off
        even where R is flat — the liability is FORWARD-driven, not spot-driven; (b) the level
        form `R_{t+1}·s_{t+1} − R_t·s_t` additionally injects a spurious
        `−(R_t − R_{t+1})·s_{t+1} = −w_fix·s` shock at every fixing (≈4× the typical |dL| in
        the averaging window) that compounds the inner value off-band backward. Guards against
        silent convention drift in the bootstrap's liability mark."""
        liab_mtm = self.bundle.get("liability_mtm")
        if liab_mtm is None or bool(torch.allclose(R_t, R_t[0])):
            return                                                  # no MTM, or no fixings to exercise
        T = self.t_outer
        hist = int(self.bundle.get("initial_time_index", 0))
        spot = self._outer_buf[self._spot_key]                      # (>=T, B_outer)
        L = liab_mtm[hist:hist + T].to(self.dtype)                  # (T, B)
        s = spot[:T].to(self.dtype)                                 # (T, B)
        if L.shape != s.shape:
            return                                                  # path dims misaligned — skip
        Rt = R_t.unsqueeze(-1)                                      # (T, 1)
        dL_true = L[1:] - L[:-1]                                    # framework MTM change
        dL_design = Rt[:-1] * (s[1:] - s[:-1])                      # R_t·dS (closed form)
        dL_level = Rt[1:] * s[1:] - Rt[:-1] * s[:-1]               # R_{t+1}s_{t+1} − R_t s_t
        # The terminal settlement (last 1-2 steps) is a discontinuous MTM jump no spot-linear
        # closed form captures; it masks the bulk, so report the max over the DECISION bulk
        # (exclude the final 2 steps). The isolated `level − design` term is the spurious
        # `−(R_t − R_{t+1})·s_{t+1}` fixing shock — present wherever R decays, ZERO where R flat.
        bulk = slice(0, max(1, dL_true.shape[0] - 2))
        err_design = (dL_true - dL_design).abs().mean(dim=-1)       # (T-1,)
        err_level = (dL_true - dL_level).abs().mean(dim=-1)
        spurious = (dL_level - dL_design).abs().mean(dim=-1)        # the level-form extra term
        scale = dL_true[bulk].abs().mean().clamp_min(1e-12)
        ed_b, el_b, sp_b = err_design[bulk], err_level[bulk], spurious[bulk]
        logging.info(
            "DifferentialSolver liability dL convention check (mean |Δ| over paths, decision "
            "bulk t<%d): R_t·dS vs framework-MTM max=%.4g @t=%d; LEVEL vs framework-MTM max=%.4g "
            "@t=%d; isolated LEVEL−design (spurious −w_fix·s) max=%.4g @t=%d; typical |dL_true|"
            "=%.4g. (Inner bootstrap uses the framework MTM directly; these are the closed-form "
            "gaps it avoids — R_t·dS is forward-vs-spot biased, LEVEL adds the fixing shock.)",
            int(bulk.stop), float(ed_b.max()), int(ed_b.argmax()),
            float(el_b.max()), int(el_b.argmax()),
            float(sp_b.max()), int(sp_b.argmax()), float(scale))
        if os.environ.get("RF_DL_AUDIT_TABLE"):
            T1 = dL_true.shape[0]
            ts = sorted({0, T1 // 8, T1 // 4, T1 * 3 // 8, T1 // 2, T1 * 5 // 8,
                         T1 * 3 // 4, T1 - 5, T1 - 2, T1 - 1})
            logging.info("  t  | R_t    | mean dL_true | mean dL_design | mean dL_level | |lvl-dsn|")
            for ti in ts:
                if 0 <= ti < T1:
                    logging.info(
                        "  %3d| %7.1f| %12.4g | %14.4g | %13.4g | %9.4g", ti, float(R_t[ti]),
                        float(dL_true[ti].mean()), float(dL_design[ti].mean()),
                        float(dL_level[ti].mean()), float(spurious[ti]))

    def _baseline_B(self, z):
        """Closed-form buy-and-hold baseline `B_t(z)` for advantage decomposition.

        Symlog utility of the EXPECTED terminal wealth under 'hold current q from
        t through T', with regime drift PROPAGATED under the transition matrix
        (validation_sandwich_spec.md §2) rather than today's belief-weighted μ̄
        held constant:

            B_t(z) = symlog( w + (Σq·cs)·S·D_hedge(t, b)
                               −  S·LiabDrift(t, b),  c )
            D_hedge(t, b)   = b · expm1(cum_mu_state[t, ttot])
            LiabDrift(t, b) = b · Σ_k w_by_lag[t, k] · expm1(cum_mu_state[t, k])

        where `b` is the belief vector in the market block and
        `cum_mu_state[t, k, i] = E[Σ_{u<k} μ_{r_u}·dt | r_t = i]`. The liability
        term is fixing-ladder-exact (per-fixing lag buckets), superseding the
        duration-bucket τ̄ form. With `P = I` and a one-hot belief this reduces
        exactly to the previous `expm1(μ_r·ttot)` baseline (regression anchor);
        a single terminal fixing gives `w_by_lag[t] = δ_{ttot}` and
        recovers `(Σq·cs − 1)·S·D` bit-for-bit.

        Action-dependent through Σq·cs, belief-dependent linearly (the dots —
        so `_dB_dz` belief gradients are exact and well-scaled), depth-dependent
        through the table gather. Bounded in symlog units. AAD-safe: every op
        differentiable; the time/index gathers are piecewise-constant in the
        time coordinate (true within-step derivative is 0; the time column is
        masked out of the differential labels regardless)."""
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

        # Deal-unit exposure: q is in contracts; contract_size converts to the
        # units S is quoted in (50 oz platinum futures). Must match
        # the PnL lines in `_compute_labels_for_bank` / `_rollout_no_grad_to_T`
        # and `_turnover` — B's q-slope feeds the residual differential label
        # ∂A/∂z = ∂C/∂z − ∂B/∂z, so a units mismatch here corrupts the labels.
        # `_dB_dz` (autograd through this method) inherits the change for free.
        sum_q = (positions * self.contract_size).sum(dim=-1)         # (..., )
        # Mark against the real reference spot (the observable LME spot column in the market
        # block), basis-free — the liability marks to spot, NOT the front future (which carries
        # basis+carry). Fall back to the front future only when no spot column is present.
        if self._spot_cols_in_market is not None:
            S = market[..., self._spot_cols_in_market[0]]            # observable spot
        else:
            S = futures[..., 0]                                      # legacy front-future proxy
        r_lo, r_hi = self._regime_cols_in_market
        belief = market[..., r_lo:r_hi]                              # (..., n_states)
        # Production `time_to_T` is integer-valued (assembled as `T−1−t` at every
        # call site). Derive BOTH table indices from one floor so the pair stays
        # coherent (`t_idx + n_steps = T−1`) even for synthetic fractional probes
        # — independent truncation/rounding can otherwise query a `cum[t, k]`
        # cell with `t + k ≠ T−1` at half-steps.
        n_steps = time_to_T.long().clamp(0, self.t_outer - 1)
        t_idx = self.t_outer - 1 - n_steps
        cum_hedge = self._cum_mu_state[t_idx, n_steps]               # (..., n_states)
        D_hedges = (belief * torch.expm1(cum_hedge)).sum(dim=-1)     # (...,)
        liab_drift = (belief * self._liab_drift_state[t_idx]).sum(dim=-1)
        E_wealth_T = wealth + sum_q * S * D_hedges - S * liab_drift
        # Baseline = the configured terminal utility of the buy-and-hold expected wealth
        # (symlog / huber / cara — single source of truth). AAD-safe; A absorbs the residual.
        return _utility_wrap_signed(E_wealth_T, self.runtime)

    def _dB_dz(self, z):
        """Exact gradient `∂B_t/∂z` of the closed-form baseline, same shape as `z`.

        Computed by autograd through `_baseline_B` itself — a handful of closed-form
        ops, no network — so it is exact to machine precision and can never drift
        out of sync with the baseline definition. Validated against central finite
        differences by `fd_check_dB.py` (the live-thread guard: a sign/layout error
        here reproduces the instability that originally forced `w_diff = 0`).

        Used to convert the AAD gradient label for the FULL value,
        `∂Y_boot/∂z = ∂C_t/∂z`, into the residual's correct label
        `∂A_t/∂z = ∂C_t/∂z − ∂B_t/∂z` on the columns where `∂C_t/∂z` is actually
        computed (wealth + live tradable_prices)."""
        if not self._advantage_decomp:
            return torch.zeros_like(z)
        with torch.enable_grad():
            z_leaf = z.detach().clone().requires_grad_(True)
            B = self._baseline_B(z_leaf)
            (g,) = torch.autograd.grad(B.sum(), z_leaf, create_graph=False)
        return g.detach()

    def _C_full(self, net, z):
        """Evaluate the full continuation value `C_t(z) = B_t(z) + A_t(z)` where
        `A_t = net` is the NN's residual. Under advantage decomposition, the NN
        learns only the small bounded residual; under the legacy no-baseline mode
        (`use_advantage_decomp=False`), `B_t ≡ 0` and this collapses to `net(z)`.
        Both call sites that previously did `net(z)` should route through here so
        the decomposition is transparent to the rest of the solver.

        Row-chunked under `no_grad` (gate-5 finding): `_decision_at` scores the FULL
        action grid (levels^n_h — 1331 at 3 axes × 11 levels; live-masking zeros dead
        columns but K stays 1331), so `z` is `(N·K, deep_dim)`. With an audit-grown
        bank (N~7k) that is ~10M rows → the MLP forward alone needs ~9 GiB and OOMs at
        production batch. Chunking the forward bounds it to `_upper_chunk_rows` (the same
        MLP-eval budget the U DP / penalty envelope use). In a grad-enabled context (the
        bootstrap-label / FD-gate paths) `z` is only `(N, deep_dim)` (the single realized
        next state, no action axis) — small — so we keep the single fused call there to
        leave the autograd graph through `C_{t+1}` intact."""
        M = z.shape[0]
        chunk = self._upper_chunk_rows
        if torch.is_grad_enabled() or M <= chunk:
            return net(z) + self._baseline_B(z)
        out = torch.empty(M, dtype=z.dtype, device=z.device)
        for i in range(0, M, chunk):
            sl = slice(i, i + chunk)
            out[sl] = net(z[sl]) + self._baseline_B(z[sl])
        return out

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
        (regime belief for a regime-switching spot, plus basis/carry where present). Tradable
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
        """Participant-visible exogenous coordinates at outer t, B-last → B-leading.
        Concatenates per-process privileged blocks in `stoch_factors_inner`
        iteration order — same ordering the inner-MC `market_t` uses, so the
        StatePack market block is layout-consistent across diff-ML training and
        the existing inner-MC machinery.

        **Belief, not regime** (validation_sandwich_spec.md §1, §2): the
        canonical `_extract_outer_state_at(..., privileged=True)` returns the
        HMM filter posterior `(key, 'regime_belief')` when the buffer publishes
        it — the participant-visible coordinate. A market participant doesn't
        observe the regime label; feeding the true regime one-hot is privileged
        information and the resulting policy doesn't transfer to deployment.
        For factor types that don't publish belief (carry curves, basis), the
        canonical extractor's raw privileged state is the right input.
        """
        parts = []
        for key in self._calc.stoch_factors_inner:
            if key.type in utils.DimensionLessFactors:
                continue
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

        # Endogenous span: q per-hedge from `position_limits`, wealth band scaled
        # to the symlog c (15·c covers ±2.78 in symlog units — the meaningful
        # dynamic range). Sampled uniform per row; the NN learns a smooth function
        # across the full (z_T, q_T, wealth_T) joint at deal-realistic scales.
        rows_per_outer = self.b_endo
        q_span = torch.empty(b_outer, rows_per_outer, n_h,
                              dtype=torch.long, device=device)
        for j in range(n_h):
            q_span[..., j] = torch.randint(
                int(self.bank_q_min[j]), int(self.bank_q_max[j]) + 1,
                (b_outer, rows_per_outer), device=device)
        q_span = q_span.to(self.dtype)
        wealth_span = (torch.rand(b_outer, rows_per_outer, device=device) - 0.5) \
            * (2.0 * self.bank_wealth_halfwidth)

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

        # Terminal utility — the configured shape (symlog / huber / cara) of wealth (w_T
        # already carries cash + (q_total - 1)·S + K → fully realized terminal wealth).
        wealth_idx = self.statepack.n_hedge
        if self._advantage_decomp:
            # Under advantage decomp the NN fits the RESIDUAL A_T = C_T − B_T(z).
            # At terminal (`time_to_T = 0`), `B_T(z) = u(wealth)` exactly (drift_factor =
            # e^0 − 1 = 0 ⇒ E_wealth_T = wealth), so A_T = u(W_T) − u(z.wealth) = 0
            # identically. Both value and gradient targets collapse to 0; the NN trains to
            # ~zero output and the full continuation is recovered as `B_T(z) + A_T(z) ≈
            # u(z.wealth) + 0` — shape-agnostic (B_T routes through `_utility_wrap_signed`).
            y_target = torch.zeros_like(wealth_flat)
            dy_dz = torch.zeros_like(z_T)
        else:
            # No-baseline path: C_T = u(W_T) directly. Value + pathwise gradient label from
            # the configured utility (autograd → shape-correct ∂u/∂wealth for symlog/huber/cara,
            # zero on other coords). Wealth column at StatePack offset n_hedge.
            with torch.enable_grad():
                w_leaf = wealth_flat.detach().clone().requires_grad_(True)
                u_T = _utility_wrap_signed(w_leaf, self.runtime)
                (gw,) = torch.autograd.grad(u_T.sum(), w_leaf)
            y_target = u_T.detach()
            dy_dz = torch.zeros_like(z_T)
            dy_dz[:, wealth_idx] = gw.detach()

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
        last_diag = {"total_loss": float("nan"), "val_loss": float("nan"),
                     "diff_loss": float("nan"), "mask_mean": 1.0,
                     "steps_used": 0, "max_steps": steps,
                     "stopped_early": False}
        for step in range(steps):
            idx = torch.randint(0, n_rows, (batch,), device=device)
            loss, last_diag = twin_loss(
                net, z_T[idx], y_target[idx], dy_dz[idx],
                w_val=1.0, w_diff=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_value = float(loss.detach())
            last_diag.update(
                steps_used=step + 1, max_steps=steps,
                stopped_early=self.train_loss_tol > 0.0
                and loss_value <= self.train_loss_tol)
            if step % max(1, steps // 5) == 0 or step == steps - 1:
                logging.info(
                    "DifferentialSolver C_T cold-train step=%d L=%.5f val=%.5f diff=%.5f",
                    step, loss_value,
                    last_diag["val_loss"], last_diag["diff_loss"])
            if last_diag["stopped_early"]:
                logging.info(
                    "DifferentialSolver C_T cold-train early stop: loss %.5g <= tol %.5g "
                    "after %d/%d steps",
                    loss_value, self.train_loss_tol, step + 1, steps)
                break

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
            "terminal_train_steps_used": int(last_diag["steps_used"]),
            "terminal_train_stopped_early": bool(last_diag["stopped_early"]),
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
        """L1 turnover cost: λ_turn · Σ_h |Δq_h| · F_h · contract_size. Per spec §3.
        Caller broadcasts shapes — typically (..., n_h) for q's and F, returns (...,) cost."""
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
        # q per-hedge from position_limits, wealth band scaled to symlog c.
        q_prev = torch.empty(B, rows, n_h, dtype=torch.long, device=device)
        for j in range(n_h):
            q_prev[..., j] = torch.randint(
                int(self.bank_q_min[j]), int(self.bank_q_max[j]) + 1,
                (B, rows), device=device)
        q_prev = q_prev.to(self.dtype)
        wealth_pre = (torch.rand(B, rows, device=device) - 0.5) \
            * (2.0 * self.bank_wealth_halfwidth)
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

        Row-chunked over N: each row's argmax is independent, so the `(N·K, deep_dim)`
        scoring tensor is built + evaluated in blocks of `~_upper_chunk_rows` post-expansion
        rows. Peak memory is then bounded by the chunk, INDEPENDENT of the outer batch — so
        the outer batch can scale (the full action grid K=levels^n_h × an audit-grown bank
        otherwise reaches ~10–80M rows and OOMs; gate-5). Bit-identical to the un-chunked
        form (same per-row ops); guarded by the chunk-transparency test.
        """
        K, n_h = self.action_grid.shape
        N = market.shape[0]
        device = self.device
        # Live-axis collapse — dead axes forced to 0. Grid is row-independent, so build the
        # (K, n_h) masked grid ONCE and expand per block (was an N·K·n_h clone).
        live_t = self.is_live[t]                                      # (n_h,)
        grid_K = self.action_grid.clone()                            # (K, n_h)
        if (~live_t).any():
            grid_K[:, ~live_t] = 0.0

        chunk_n = max(1, self._upper_chunk_rows // max(1, K))         # N-rows per block
        q_parts, v_parts, gap_parts = [], [], []
        with torch.no_grad():
            for i in range(0, N, chunk_n):
                sl = slice(i, i + chunk_n)
                n_b = market[sl].shape[0]
                grid_b = grid_K.unsqueeze(0).expand(n_b, K, n_h)         # (n_b, K, n_h)
                market_b = market[sl].unsqueeze(1).expand(n_b, K, -1)
                F_b = F[sl].unsqueeze(1).expand(n_b, K, n_h)
                q_prev_b = q_prev[sl].unsqueeze(1).expand(n_b, K, n_h)
                wealth_pre_b = wealth_pre[sl].unsqueeze(1).expand(n_b, K)
                time_to_T_b = time_to_T[sl].unsqueeze(1).expand(n_b, K)
                static_b = static[sl].unsqueeze(1).expand(n_b, K, -1)
                # Turnover cost per (n_b, K) — spec §3's L1 model.
                wealth_post_b = wealth_pre_b - self._turnover(grid_b, q_prev_b, F_b)
                # Build post-decision states + score the frozen C_next. NO autograd — the
                # only decision output flowing into the differential label is q* (detached);
                # value is read for ranking + the action_gap diagnostic. Advantage decomp:
                # argmax the SUM B_t + A_t (legacy no-baseline path collapses _C_full to C_next).
                z_post = self._assemble_z(
                    market=market_b.reshape(n_b * K, -1),
                    q=grid_b.reshape(n_b * K, n_h),
                    wealth=wealth_post_b.reshape(-1),
                    F=F_b.reshape(n_b * K, n_h),
                    time_to_T=time_to_T_b.reshape(-1),
                    static=static_b.reshape(n_b * K, -1))               # (n_b·K, deep_dim)
                values = self._C_full(C_next, z_post).reshape(n_b, K)   # (n_b, K)
                amax = values.argmax(dim=-1)                            # (n_b,)
                v_parts.append(values.gather(-1, amax.unsqueeze(-1)).squeeze(-1))
                # Action gap = top minus second-best (sort then diff; robust to ties).
                top2, _ = values.topk(min(2, K), dim=-1)
                gap_parts.append((top2[..., 0] - top2[..., -1]).abs())
                q_parts.append(grid_b.gather(
                    1, amax.view(n_b, 1, 1).expand(n_b, 1, n_h)).squeeze(1))
        q_star = torch.cat(q_parts, dim=0)                              # (N, n_h)
        v_star = torch.cat(v_parts, dim=0)                              # (N,)
        action_gap = torch.cat(gap_parts, dim=0)                        # (N,)
        return q_star, v_star, action_gap

    def _inner_bootstrap_value_and_endogenous_grads(self, t, bank, q_post_t,
                                                    wealth_post_t, C_next):
        """Inner-MC Bellman label plus endogenous differential labels.

        This is the production analogue of the toy's `E[max_q C_{t+1}]` target:
        select the next-step action on the inner expectation under `no_grad`, then
        recompute that same expectation at the fixed selected actions with the
        post-decision endogenous coordinates as leaves. The returned gradients are
        therefore derivatives of the same MC estimator used for the scalar value label,
        for the columns where no simulator AAD tape is needed: live positions, wealth,
        and current tradable prices. Market spot/belief gradients are still supplied by
        the realised-path AAD branch below until the one-step grad fork is chunked for
        full production batch sizes.
        """
        if self._bootstrap_inner_samples <= 1 or "inner_mc_fn" not in self.bundle:
            return None
        t1 = min(t + 1, self.t_outer - 1)
        inner = self.bundle["inner_mc_fn"](t)
        market_t1 = inner.get("market_t1")
        if market_t1 is None:
            return None

        device = self.device
        n_h = len(self.hedges)
        oi = bank["outer_index"]
        N = q_post_t.shape[0]
        I = min(int(self._bootstrap_inner_samples), int(market_t1.shape[1]))
        if I <= 0:
            return None

        market_i = market_t1[oi, :I].to(device).detach()                    # (N,I,market_dim)
        F_t1_i = torch.stack(
            [inner["F_t1"][h].to(device)[oi, :I] for h in self.hedges],
            dim=-1).detach()                                                # (N,I,n_h)

        # spot[t] and belief[t] enter as PER-ROW leaves; each inner draw's z_{t+1} spot/belief
        # column is reconstructed FROM them through the process's OWN one-step transitions
        # (`reseed_one_step` / `forward_belief_onestep` — the same dynamics the realized-path
        # bootstrap uses), so ∂value/∂spot and ∂value/∂belief fall out of the SAME autograd pass
        # as q/wealth/F. The differential is a side effect of the state, not a backfilled label.
        # The spot reseed lands EXACTLY on the fork's inner draw (value-coherent on price); a
        # coherence check guards the belief filter. The liability is reconstructed the SAME way
        # the realised/training label path does — first-order from the framework MTM buffer and
        # the persisted spot Jacobian: L_t = liability_mtm[t] + JL_t·(s_t − s_t^real),
        # L_{t+1} = liability_mtm[t+1] + JL_{t+1}·(s_{t+1} − s_{t+1}^real), dL = L_{t+1} − L_t.
        # This makes the inner bootstrap's liability mark identical to the training labels by
        # construction (value AND ∂dL/∂spot = JL_{t+1} − JL_t), so the inner expectation is a
        # genuine variance reduction on the SAME quantity. The earlier closed forms were both
        # wrong: the level form `R_{t+1}·s_{t+1} − R_t·s_t` injects a spurious `−w_fix·s` shock
        # at every fixing (≈4× the typical |dL| in the averaging window — it compounds the value
        # off-band backward), and even `R_t·dS` is ~30–45% off the forward-driven MTM. See the
        # construction-time `_audit_liability_convention` for both gaps.
        spot_proc = self._calc.stoch_factors.get(self._spot_key)
        spot_col = (self._spot_cols_in_market[0]
                    if self._spot_cols_in_market is not None else None)
        regime_cols = self._regime_cols_in_market
        belief_buf = self._outer_buf.get((self._spot_key, 'regime_belief'))
        spot_t_real = self._spot_col(bank["market"])                        # (N,) or None
        spot_t1_inner = market_i[..., spot_col] if spot_col is not None else None  # (N,I)
        R_t = float(self._liability_R[t].item())

        # Framework liability (MTM buffer is history-prefixed → +hist; the spot Jacobian is
        # sim-grid → no offset; mirrors the realised path in `_compute_labels_for_bank`).
        liab_mtm = self.bundle.get("liability_mtm")
        liab_jac = self.bundle.get("liability_jacobian", {})
        have_liab = liab_mtm is not None and spot_t1_inner is not None
        if have_liab:
            hist = int(self.bundle.get("initial_time_index", 0))
            spot_t1_real = self._outer_buf[self._spot_key][t1][oi]           # (N,) realised t+1 spot
            L_t_real = liab_mtm[hist + t][oi]                                # (N,)
            L_t1_real = liab_mtm[hist + t1][oi]                              # (N,)
            jl = liab_jac.get(utils.check_tuple_name(self._spot_key)) if liab_jac else None
            JL_t = jl[t][oi] if jl is not None else torch.zeros_like(spot_t1_real)    # (N,)
            JL_t1 = jl[t1][oi] if jl is not None else torch.zeros_like(spot_t1_real)  # (N,)

        static = bank["static"].unsqueeze(1).expand(
            N, I, bank["static"].shape[-1]).reshape(N * I, -1)
        time_to_T = (bank["time_to_T"] - 1.0).unsqueeze(1).expand(N, I).reshape(-1)
        if have_liab:
            # d_t = 0 at the realised t spot ⇒ L_t = L_t_real; t+1 marks the inner draw.
            dL_det = (L_t1_real.unsqueeze(1)
                      + JL_t1.unsqueeze(1) * (spot_t1_inner - spot_t1_real.unsqueeze(1))
                      - L_t_real.unsqueeze(1))                              # (N,I) framework MTM
        elif spot_t1_inner is not None:
            dL_det = R_t * (spot_t1_inner - spot_t_real.unsqueeze(1))        # fallback: R_t·dS
        else:
            dL_det = R_t * (F_t1_i[..., 0] - bank["F_at_t"][:, :1])

        # Action selection: NO gradient through the Bellman argmax/envelope.
        with torch.no_grad():
            dF_sel = F_t1_i - bank["F_at_t"].unsqueeze(1)
            pnl_sel = (q_post_t.unsqueeze(1) * self.contract_size * dF_sel).sum(-1)
            wealth_pre_sel = wealth_post_t.unsqueeze(1) + pnl_sel - dL_det
            q_prev_sel = q_post_t.unsqueeze(1).expand(N, I, n_h).reshape(N * I, n_h)
            q_star, _, gap = self._decision_at(
                C_next, market_i.reshape(N * I, -1), F_t1_i.reshape(N * I, n_h),
                q_prev_sel, wealth_pre_sel.reshape(-1), time_to_T, static, t1)
            q_star = q_star.detach().reshape(N, I, n_h)
            action_gap = gap.reshape(N, I).mean(dim=1).detach()

        belief_coherence = float('nan')
        with torch.enable_grad():
            q_leaf = q_post_t.detach().clone().requires_grad_(True)
            wealth_leaf = wealth_post_t.detach().clone().requires_grad_(True)
            F_leaf = bank["F_at_t"].detach().clone().requires_grad_(True)
            leaves = [q_leaf, wealth_leaf, F_leaf]
            market_recon = market_i.clone()                                 # (N,I,market_dim)

            # spot leaf → predictive channel (reseed → z_{t+1} spot col) + liability (dL).
            spot_leaf = None
            dL = dL_det
            if spot_col is not None and spot_t1_inner is not None and spot_proc is not None:
                spot_leaf = spot_t_real.detach().clone().requires_grad_(True)   # (N,)
                spot_t1_recon = spot_proc.reseed_one_step(
                    spot_leaf.unsqueeze(1), spot_t_real.unsqueeze(1), spot_t1_inner)  # (N,I)
                market_recon[..., spot_col] = spot_t1_recon
                if have_liab:
                    # Framework MTM reconstruction with the spot leaf: ∂dL/∂spot = JL_{t+1} − JL_t.
                    d_t = (spot_leaf - spot_t_real).unsqueeze(1)            # (N,1), 0 at value
                    d_t1 = spot_t1_recon - spot_t1_real.unsqueeze(1)        # (N,I)
                    L_t = L_t_real.unsqueeze(1) + JL_t.unsqueeze(1) * d_t   # (N,1)
                    L_t1 = L_t1_real.unsqueeze(1) + JL_t1.unsqueeze(1) * d_t1  # (N,I)
                    dL = L_t1 - L_t                                          # (N,I) matches training
                else:
                    dL = R_t * (spot_t1_recon - spot_leaf.unsqueeze(1))      # fallback: R_t·dS
                leaves.append(spot_leaf)

            # belief leaf → one outer-grid filter step per inner draw → z_{t+1} belief cols.
            belief_leaf = None
            if (regime_cols is not None and belief_buf is not None
                    and spot_t1_inner is not None
                    and hasattr(spot_proc, 'forward_belief_onestep')):
                r_lo, r_hi = regime_cols
                belief_leaf = belief_buf[t][:, oi].detach().clone().requires_grad_(True)  # (ns,N)
                ns = belief_leaf.shape[0]
                bl_exp = belief_leaf.unsqueeze(-1).expand(ns, N, I).reshape(ns, N * I)
                sp_prev = spot_t_real.unsqueeze(1).expand(N, I).reshape(-1)
                belief_t1 = spot_proc.forward_belief_onestep(
                    bl_exp, sp_prev, spot_t1_inner.reshape(-1), t1)         # (ns, N*I)
                belief_recon = belief_t1.movedim(0, -1).reshape(N, I, ns)   # (N,I,ns)
                belief_coherence = float(
                    (belief_recon.detach() - market_i[..., r_lo:r_hi]).abs().max())
                market_recon[..., r_lo:r_hi] = belief_recon
                leaves.append(belief_leaf)

            dF = F_t1_i - F_leaf.unsqueeze(1)
            pnl = (q_leaf.unsqueeze(1) * self.contract_size * dF).sum(-1)
            wealth_pre_t1 = wealth_leaf.unsqueeze(1) + pnl - dL
            cost_t1 = self._turnover(q_star, q_leaf.unsqueeze(1), F_t1_i)
            wealth_post_t1 = wealth_pre_t1 - cost_t1
            z_t1 = self._assemble_z(
                market=market_recon.reshape(N * I, -1),
                q=q_star.reshape(N * I, n_h),
                wealth=wealth_post_t1.reshape(-1),
                F=F_t1_i.reshape(N * I, n_h),
                time_to_T=time_to_T, static=static)
            y_inner = self._C_full(C_next, z_t1).reshape(N, I).mean(dim=1)
            grads = torch.autograd.grad(
                y_inner.sum(), leaves, retain_graph=False, allow_unused=True)

        # Unpack by build order [q, wealth, F, (spot), (belief)].
        q_grad, wealth_grad, F_grad = grads[0], grads[1], grads[2]
        gi = 3
        spot_grad = belief_grad = None
        if spot_leaf is not None:
            spot_grad = grads[gi]; gi += 1
        if belief_leaf is not None:
            belief_grad = grads[gi]; gi += 1
        return {
            "value": y_inner.detach(),
            "q_grad": (torch.zeros_like(q_post_t) if q_grad is None else q_grad.detach()),
            "wealth_grad": (torch.zeros_like(wealth_post_t) if wealth_grad is None
                             else wealth_grad.detach()),
            "F_grad": (torch.zeros_like(bank["F_at_t"]) if F_grad is None else F_grad.detach()),
            "spot_grad": (None if spot_grad is None else spot_grad.detach()),   # (N,) per-row
            "belief_grad": (None if belief_grad is None else belief_grad.detach()),  # (ns,N) per-row
            "belief_coherence": belief_coherence,
            "action_gap": action_gap,
        }

    def _compute_labels_for_bank(self, t, bank, C_next, traj=None):
        """Generate `(z_t, y_target, dy_dz, action_gap)` for a bank of pre-decision
        rows via the spec §4/§12 one-step-bootstrap pipeline:
          • sample q_t spanned across the action grid (per-row anchor)
          • assemble `z_t` post-decision after applying q_t
          • re-price t+1 on the REALIZED outer path for AAD labels: spot[t] the sole
            pricing leaf, spot[t+1] its analytic re-seed, F/L from the persisted Jacobians,
            belief[t+1] from a one-step filter on the realized move (belief[t] a leaf)
          • decision at t+1 via `argmax_q C_next(post_{t+1}(q))` (q* detached)
          • value target = inner-averaged `E[max_q C_next(post_{t+1}(q))]` when configured,
            otherwise the realised-path `C_next(post_{t+1}(q*))` fallback
          • `dy_dz` via `autograd.grad(Y_boot, leaves)` + closed-form wealth chain
        Used both by initial training-bank label gen and by per-round correction-row
        label gen (M2); re-invoked per round on a grown bank by M2's audit-correct loop."""
        N = bank["wealth_pre"].shape[0]
        n_h = len(self.hedges)
        device = self.device
        live_t = self.is_live[t]
        # q_t sampled per-hedge from position_limits, then dead axes forced to 0.
        q_t_sampled = torch.empty(N, n_h, dtype=torch.long, device=device)
        for j in range(n_h):
            q_t_sampled[..., j] = torch.randint(
                int(self.bank_q_min[j]), int(self.bank_q_max[j]) + 1,
                (N,), device=device)
        q_t_sampled = q_t_sampled.to(self.dtype)
        if (~live_t).any():
            q_t_sampled[:, ~live_t] = 0.0
        cost_t = self._turnover(q_t_sampled, bank["q_prev"], bank["F_at_t"])
        wealth_post_t = bank["wealth_pre"] - cost_t
        z_t = self._assemble_z(
            market=bank["market"], q=q_t_sampled, wealth=wealth_post_t,
            F=bank["F_at_t"], time_to_T=bank["time_to_T"],
            static=bank["static"])

        # --- One-step pathwise bootstrap label (real t+1 MtMs, spot[t] the sole pricing leaf) ---
        # Fully reconstructed on the REALIZED outer path — no inner-MC fork. spot[t] is the sole
        # pricing leaf; spot[t+1] is its analytic one-step re-seed (the spot is a regime random
        # walk, so the increment is independent of spot_t ⇒ exact derivative, lands on the
        # realized value); F_h and L at t and t+1 are reconstructed first-order from the
        # persisted Jacobians J_F = ∂F/∂spot, J_L = ∂L/∂spot. Because J = G'(realized), the
        # linear reconstruction equals the true slice re-price AT the realized point — exact
        # value AND exact gradient — so no re-simulation is needed. Uses the OUTER spot process
        # (precalculated on the outer grid; the inner proc's grid/state is for the LSM fork).
        oi = bank["outer_index"]
        t1 = min(t + 1, self.t_outer - 1)
        spot_proc = self._calc.stoch_factors.get(self._spot_key)
        spot_name = utils.check_tuple_name(self._spot_key)
        hist = int(self.bundle.get("initial_time_index", 0))
        # Realized endpoints. The outer spot buffer + tradables_sim views are sim-grid; the
        # forward/liability Jacobians are sim-grid (NOT history-prefixed); liability_mtm IS
        # history-prefixed (→ +hist). See torchrl_hedge._prepend_history_prefix.
        spot_t_real = self._outer_buf[self._spot_key][t][oi]                  # (N,)
        spot_t1_real = self._outer_buf[self._spot_key][t1][oi]                # (N,)
        F_t1_real = torch.stack(
            [self._tradables_sim[h][t1].to(self.device)[oi] for h in self.hedges], dim=-1)  # (N,n_h)
        liab_mtm = self.bundle["liability_mtm"]                               # (hist+T, B)
        L_t_real = liab_mtm[hist + t][oi]                                     # (N,)
        L_t1_real = liab_mtm[hist + t1][oi]                                   # (N,)
        zeroN = torch.zeros_like(spot_t_real)
        fwd_jac = self.bundle.get("forward_jacobian", {})
        liab_jac = self.bundle.get("liability_jacobian", {})

        def _spot_jac(table, ti):
            d = table.get(spot_name) if table is not None else None
            return d[ti][oi] if d is not None else zeroN

        JF_t = torch.stack([_spot_jac(fwd_jac.get(h, {}), t) for h in self.hedges], dim=-1)   # (N,n_h)
        JF_t1 = torch.stack([_spot_jac(fwd_jac.get(h, {}), t1) for h in self.hedges], dim=-1) # (N,n_h)
        JL_t = _spot_jac(liab_jac, t)                                         # (N,)
        JL_t1 = _spot_jac(liab_jac, t1)                                       # (N,)
        time_to_T_t1 = bank["time_to_T"] - 1.0                               # (N,)

        # --- Fork-free t+1 market: read the realized privileged outer slice (carry, belief,
        # spot, basis at the realized t+1); no inner-MC re-simulation, so the bootstrap state IS
        # the realized outer path — every column from the same t+1 ⇒ the value is coherent.
        market_t1_realized = self._read_privileged_market(t1)[oi]            # (N, market_dim)

        # Belief differential (fork-free): seed a LEAF from the realized belief[t] and run ONE
        # outer-grid filter step on the realized move → belief[t+1]. This EQUALS the belief the
        # outer filter wrote to the buffer (value-coherent) but carries ∂belief[t+1]/∂belief[t],
        # so the twin loss supervises the belief column. Independent of the spot leaf. Gated on
        # the belief-filter flag; off ⇒ belief stays the detached realized value (no diff label).
        belief_leaf = None
        belief_t1_outer = None
        belief_buf = self._outer_buf.get((self._spot_key, 'regime_belief'))
        if (self._calc._inner_belief_filter and belief_buf is not None
                and self._regime_cols_in_market is not None):
            spot_t_outer = self._outer_buf[self._spot_key][t]               # (B_outer,)
            spot_t1_outer = self._outer_buf[self._spot_key][t1]
            belief_leaf = belief_buf[t].detach().clone().requires_grad_(True)   # (n_states, B_outer)
            with torch.enable_grad():
                belief_t1_outer = spot_proc.forward_belief_onestep(
                    belief_leaf, spot_t_outer, spot_t1_outer, t1)           # (n_states, B_outer)
            if not getattr(self, '_logged_belief_recon', False):
                self._logged_belief_recon = True
                logging.info(
                    'DifferentialSolver realized-path belief differential ACTIVE (fork-free '
                    'one-step filter; n_states=%d) — the belief column carries '
                    '∂belief_{t+1}/∂belief_t on the realized move, replacing the inner-MC '
                    'belief leaf; calibration is the outer filter (unchanged).',
                    int(belief_t1_outer.shape[0]))

        with torch.enable_grad():
            # z_{t+1}'s market: realized columns, with the belief columns overridden by the
            # grad-connected one-step filter (the spot column is overridden per-spot inside the
            # closure). carry/basis stay realized-detached → fully coherent on the realized path.
            market_t1_per_row = market_t1_realized
            if belief_t1_outer is not None:
                r_lo, r_hi = self._regime_cols_in_market
                market_t1_per_row = market_t1_per_row.clone()
                market_t1_per_row[:, r_lo:r_hi] = belief_t1_outer[:, oi].T   # (N, n_states), grad

            # F-column hedge-book probe: a per-(row, hedge) ZERO leaf added to the position's
            # entry price F_j(t). It enters Y_boot ONLY through the hedge PnL (not a coordinate
            # of z_{t+1}), so ∂Y_boot/∂f_delta_j = ∂C_t/∂F_j(t) is the PURE hedge-book partial
            # (spot held fixed) — the liability is a function of spot, not F_j, so it lives on
            # ∂C/∂spot (once, total), NEVER on the F-columns. Per-contract via the position q_j.
            q_leaf = q_t_sampled.detach().clone().requires_grad_(True)       # post-decision q coord
            f_delta = torch.zeros_like(bank["F_at_t"]).requires_grad_(True)  # (N, n_h)
            spot_col = (self._spot_cols_in_market[0]
                        if self._spot_cols_in_market is not None else None)

            def _y_boot_of_spot(spot_t, q_star, move_entry_F=True):
                """Re-price t+1 from the realized path. `spot_t` always drives the predictive
                channel — the re-seeded spot[t+1] → F_h[t+1], L[t+1], and the spot COLUMN of
                z_{t+1} — and the liability's contemporaneous mark L[t] (the liability is a
                function of spot, not a hedge coordinate). The contemporaneous ENTRY-price
                channel (∂F_h[t]/∂spot) is included only when `move_entry_F`: it is the
                F-columns' job, so the spot-column PARTIAL holds the current F_h fixed
                (`move_entry_F=False`). The total (`True`) is kept only for the
                `total = partial + Σ_h ∂C/∂F_h·∂F_h/∂spot` cross-check. q* frozen (envelope)."""
                spot_t1 = spot_proc.reseed_one_step(spot_t, spot_t_real, spot_t1_real)  # (N,)
                d_t = (spot_t - spot_t_real).unsqueeze(-1)                   # (N,1)
                d_t1 = (spot_t1 - spot_t1_real).unsqueeze(-1)               # (N,1)
                F_t = bank["F_at_t"] + f_delta                              # (N,n_h)
                if move_entry_F:
                    F_t = F_t + JF_t * d_t                                  # contemporaneous entry channel
                F_t1 = F_t1_real + JF_t1 * d_t1                             # (N,n_h) forward channel
                L_t = L_t_real + JL_t * d_t.squeeze(-1)                      # (N,) liability marks to spot
                L_t1 = L_t1_real + JL_t1 * d_t1.squeeze(-1)                  # (N,)
                pnl_hedges = (q_leaf * self.contract_size * (F_t1 - F_t)).sum(dim=-1)
                wealth_pre_t1 = wealth_post_t + pnl_hedges - (L_t1 - L_t)    # (N,)
                cost_t1 = self._turnover(q_star, q_leaf, F_t1)
                wealth_post_t1 = wealth_pre_t1 - cost_t1
                # z_{t+1}'s spot coordinate = the re-seeded spot[t+1] (grad-connected), so the
                # predictive channel ∂C_{t+1}/∂spot_{t+1}·∂spot_{t+1}/∂spot_t flows into the
                # label. The fork's market carries an inner-draw price there; override it with
                # the realized re-seed so the bootstrap state is coherent on the realized path.
                market_t1 = market_t1_per_row
                if spot_col is not None:
                    market_t1 = market_t1.clone()
                    market_t1[:, spot_col] = spot_t1
                z_t1 = self._assemble_z(
                    market=market_t1, q=q_star, wealth=wealth_post_t1,
                    F=F_t1, time_to_T=time_to_T_t1, static=bank["static"])
                return self._C_full(C_next, z_t1)

            # Decision at t+1 — external argmax over frozen C_next at the realized spot.
            # Envelope theorem: q* frozen, so ∂max_q/∂spot = ∂/∂spot at the optimizing q*.
            with torch.no_grad():
                pnl_real = (q_t_sampled * self.contract_size
                            * (F_t1_real - bank["F_at_t"])).sum(dim=-1)      # (N,)
                dL_real = L_t1_real - L_t_real                               # (N,)
                wp_t1_real = wealth_post_t + pnl_real - dL_real             # (N,)
            q_star_t1, _, action_gap = self._decision_at(
                C_next, market_t1_per_row, F_t1_real, q_t_sampled,
                wp_t1_real, time_to_T_t1, bank["static"], t1)
            q_star_t1 = q_star_t1.detach()
            cost_t1_real = self._turnover(q_star_t1, q_t_sampled, F_t1_real).detach()

            # Bootstrap value + its differential label, single backward pass. `_C_full` adds
            # the closed-form baseline at t+1 ⇒ Y_boot is the FULL value (baseline + residual).
            # The spot-column label is the PARTIAL (move_entry_F=False): holding the current
            # F_h fixed, since the F-columns carry ∂C/∂F_h. The VALUE is identical either way
            # at the realized point (d_t=0), so Y_boot is correct for y_target regardless.
            spot_t_leaf = spot_t_real.detach().clone().requires_grad_(True)  # (N,)
            Y_boot = _y_boot_of_spot(spot_t_leaf, q_star_t1, move_entry_F=False)   # (N,)
            grad_leaves = ([q_leaf, f_delta, spot_t_leaf]
                           + ([belief_leaf] if belief_leaf is not None else []))
            all_grads = torch.autograd.grad(
                Y_boot.sum(), grad_leaves, retain_graph=False, allow_unused=True)
            q_grad = all_grads[0]
            f_delta_grad = all_grads[1]                                      # (N,n_h) = ∂C/∂F_j(t)
            g_spot = all_grads[2]                                           # (N,) = ∂C/∂spot|_F (partial)
            belief_grad = all_grads[3] if belief_leaf is not None else None
            if q_grad is None:
                q_grad = torch.zeros_like(q_leaf)
            if f_delta_grad is None:
                f_delta_grad = torch.zeros_like(f_delta)
            if g_spot is None:
                g_spot = torch.zeros_like(spot_t_leaf)

        inner_label = self._inner_bootstrap_value_and_endogenous_grads(
            t, bank, q_t_sampled.detach(), wealth_post_t.detach(), C_next)
        dY_dw_inner = None
        inner_spot_grad = inner_belief_grad = None        # inner-estimator differentials (per-row)
        if inner_label is not None:
            Y_value_boot = inner_label["value"]
            q_grad = inner_label["q_grad"]
            f_delta_grad = inner_label["F_grad"]
            dY_dw_inner = inner_label["wealth_grad"]
            action_gap = inner_label["action_gap"]
            # Spot/belief differentials from the inner estimator (a side effect of the spot[t]/
            # belief[t] state leaves), consistent with the inner-averaged value. Kept SEPARATE
            # from the realised g_spot / belief_grad, which the spot FD gate + coherence checks
            # below still validate against the realised reconstruction. spot_grad is (N,);
            # belief_grad is (ns,N) per-bank-row (realised is (ns,B_outer), indexed by oi).
            inner_spot_grad = inner_label.get("spot_grad")
            inner_belief_grad = inner_label.get("belief_grad")
            # Coherence guard: forward_belief_onestep (outer-grid step) must reproduce the fork's
            # filtered belief, else overriding z_{t+1}'s belief changes the value (the P-index /
            # convention concern). Log once + warn if it drifts.
            _bc = inner_label.get("belief_coherence")
            if _bc is not None and _bc == _bc and not getattr(self, "_logged_inner_belief_coh", False):
                self._logged_inner_belief_coh = True
                (logging.warning if _bc > 1.0e-2 else logging.info)(
                    "DifferentialSolver inner-MC belief reconstruction coherence "
                    "max|forward_belief_onestep − fork belief| = %.3g (t=%d) — %s", _bc, t,
                    "OK (value-coherent)" if _bc <= 1.0e-2 else
                    "DRIFT: belief override changes the value; check P-index convention")
        else:
            Y_value_boot = Y_boot.detach()

        # Assemble the differential label dy_dz on its three supervised column groups:
        #   • tradable_prices F_j  ← the f_delta hedge-book probe (∂C_t/∂F_j, per-contract)
        #   • wealth (cash slot)   ← the closed-form ∂C/∂wealth pass below
        #   • positions q_j        ← post-decision inventory leaf, holding wealth fixed
        #   • belief (market block)← the inner-belief seed-leaf gradient
        # Every other column is an uncomputed zero (masked out of the diff loss).
        dy_dz = torch.zeros_like(z_t)
        for j in range(n_h):
            if live_t[j]:
                dy_dz[:, j] = q_grad[:, j]
        F_col_start = self.statepack.n_hedge + 2          # tradable_prices block start
        # Wealth gradient: when the inner-expected label is active, this is the derivative
        # of that SAME MC estimator. Otherwise fall back to the realised-path AAD branch.
        if dY_dw_inner is not None:
            dY_dw = dY_dw_inner
        else:
            with torch.enable_grad():
                w_leaf = wealth_post_t.detach().clone().requires_grad_(True)
                wealth_pre_t1_l = w_leaf + pnl_real - dL_real            # realized-point, detached
                wealth_post_t1_l = wealth_pre_t1_l - cost_t1_real
                # Same z_{t+1} as the main bootstrap: override the spot column with the realized
                # re-seed spot[t+1] (the fork's market carries an inner-draw price there). Without
                # this the wealth pass evaluates C_next at a different spot than `Y_boot`, breaking
                # the machine-tight hedge-book identity f_delta_grad == -dY_dw·q·cs.
                market_w = market_t1_per_row.detach()
                if spot_col is not None:
                    market_w = market_w.clone()
                    market_w[:, spot_col] = spot_t1_real
                z_t1_l = self._assemble_z(
                    market=market_w, q=q_star_t1,
                    wealth=wealth_post_t1_l, F=F_t1_real.detach(),
                    time_to_T=time_to_T_t1, static=bank["static"])
                Y_for_w = self._C_full(C_next, z_t1_l)
                dY_dw = torch.autograd.grad(Y_for_w.sum(), w_leaf)[0]    # (N,)
        wealth_col = self.statepack.n_hedge                          # cash slot
        dy_dz[:, wealth_col] = dY_dw

        # F-column labels: ∂C_t/∂F_j(t) = the hedge-book partial read straight off the
        # f_delta probe (dY_dw · ∂pnl/∂F_j = −dY_dw·q_j·cs_j). Per-contract by construction —
        # the per-row position q_j differs across contracts — so the columns are genuinely
        # distinct, NOT the spot-leaf broadcast (one number on every column). NO liability
        # term: the liability is a function of spot, not F_j, so it does not appear in the
        # partial (it lives on ∂C/∂spot, once, total).
        for j in range(n_h):
            if live_t[j]:
                dy_dz[:, F_col_start + j] = f_delta_grad[:, j]

        # Belief seed leaf → the belief columns of the market block. `belief_cols` is set
        # only when the inner belief filter populated a grad leaf, so it gates the unmask
        # below (one-hot fallback leaves the regime block detached → no slope → stay masked).
        belief_cols = None
        belief_label = inner_belief_grad if inner_belief_grad is not None else belief_grad
        if belief_label is not None and self._regime_cols_in_market is not None:
            # ∂C_t/∂belief_t (flows through the filter's predict step into z_{t+1}), projected
            # onto the belief slice of the market block. The inner-estimator grad is already
            # per-bank-row (ns,N); the realised single-path grad is per-outer-path (ns,B_outer)
            # and indexed by `oi`.
            ms = 2 * self.statepack.n_hedge + 2
            r_lo, r_hi = self._regime_cols_in_market
            dy_dz[:, ms + r_lo : ms + r_hi] = (
                belief_label.T if inner_belief_grad is not None else belief_label[:, oi].T)
            belief_cols = (ms + r_lo, ms + r_hi)

        # Spot column ← the PARTIAL ∂C/∂spot|_F (holds the current F fixed; the F-columns carry
        # the hedge-book partial, so this is the liability + predictive channel that ISN'T on
        # them). This is the off-manifold (basis-direction) sensitivity collinear value data
        # cannot identify — exactly what the differential label supplies.
        spot_market_col = None
        spot_label = inner_spot_grad if inner_spot_grad is not None else g_spot
        if self._spot_cols_in_market is not None and spot_label is not None:
            spot_market_col = 2 * self.statepack.n_hedge + 2 + self._spot_cols_in_market[0]
            dy_dz[:, spot_market_col] = spot_label

        # Price-differential gate (diagnostic): snapshot the RAW pathwise ∂Y_boot/∂F_h on the
        # live tradable_prices columns BEFORE the advantage-decomp baseline subtraction, so the
        # gate isolates the hedge-book partial itself (∂B/∂F is per-contract and would otherwise
        # mask the bug). Distinctness is measured PER ROW (mean over rows of the max−min across
        # live columns), NOT spread-of-column-means: the hedge-book label −dY_dw·q_j·cs_j differs
        # across contracts via the per-row position q_j, but q_j has ~the same MEAN across
        # contracts, so a column-mean spread would wash the signal out. The spot-broadcast makes
        # every live F-column identical PER ROW → per-row spread ≡ 0; the per-contract label makes
        # them differ per row. Keyed by t (first round wins); recorded only where ≥2 hedges are
        # live (the broadcast is invisible with a single live col).
        if not hasattr(self, "_price_diff_gate"):
            self._price_diff_gate = {}
        if t not in self._price_diff_gate:
            live_idx = [j for j in range(n_h) if live_t[j]]
            live_cols = [F_col_start + j for j in live_idx]
            if len(live_cols) >= 2:
                cols = dy_dz[:, live_cols]                             # (N, n_live)
                per_row_spread = (cols.max(dim=1).values - cols.min(dim=1).values).abs()
                scale = cols.abs().max() + 1.0e-12
                # Hedge-book correctness: ∂C_t/∂F_j MUST equal the analytic partial
                # −dY_dw·q_j·cs_j (a price gradient is the position MtM, spot held fixed; no
                # liability). f_delta_grad and dY_dw are the SAME wealth gradient, so this is
                # machine-tight — it catches the broadcast (no q_j dependence), a missing
                # dY_dw factor, or a wrong contract size, none of which the coarse distinctness
                # metric can see.
                analytic = -(dY_dw.unsqueeze(-1) * q_t_sampled * self.contract_size)  # (N, n_h)
                num = (f_delta_grad[:, live_idx] - analytic[:, live_idx]).abs()
                den = analytic[:, live_idx].abs() + 1.0e-8
                self._price_diff_gate[t] = {
                    "n_live": len(live_cols),
                    "raw_fcol_distinctness": float(per_row_spread.mean() / scale),
                    "raw_fcol_means": [float(v) for v in cols.mean(dim=0)],
                    "hedgebook_rel_err": float((num / den).max()),
                }

        # Spot-gradient FD gate (permanent diagnostic): validate the one-step pathwise
        # ∂Y_boot/∂spot stays on-graph and the re-seed/Jacobian reconstruction is numerically
        # exact. (1) bump spot[t] ±ε, recompute Y_boot through the IDENTICAL reconstruction
        # with q* frozen (envelope), compare the central difference to the autograd label —
        # a detached channel / wrong sign / mis-indexed Jacobian gives an O(1) error; a correct
        # on-graph reconstruction stays «1e-2 (float32 FD floor) on the bounded operating
        # region (R_t≠0). Near terminal (R_t=0, fixings exhausted) the C_next hull-clamp adds a
        # kink that inflates the FD difference — graded out, per the established "grade at
        # bounded operating points, not the blow-up" methodology; `Rt` is recorded so the gate
        # is never silently vacuous. (2) liability-sign anchor (theta-free): the persisted
        # autograd liability spot-delta J_L_t must SIGN-agree with the trusted closed-form
        # remaining-fixing weight R_t (both are ∂L/∂spot in deal units). FD-vs-autograd is
        # blind to a wealth-equation liability sign flip (both sides flip together); this
        # anchors it. The realized L[t+1]−L[t] also carries liability theta, so it legitimately
        # differs from R_t·dS in magnitude — only the spot-delta SIGN is the invariant.
        if not hasattr(self, "_spot_grad_fd_gate"):
            self._spot_grad_fd_gate = {}
        if t not in self._spot_grad_fd_gate and g_spot is not None:
            with torch.no_grad():
                # FD-vs-autograd on the PARTIAL (the label): bump spot ±ε through the SAME
                # move_entry_F=False closure, q* frozen.
                eps = 1.0e-3 * (spot_t_real.abs() + 1.0)                    # (N,)
                y_plus = _y_boot_of_spot(spot_t_real + eps, q_star_t1, move_entry_F=False)
                y_minus = _y_boot_of_spot(spot_t_real - eps, q_star_t1, move_entry_F=False)
                g_fd = (y_plus - y_minus) / (2.0 * eps)                     # (N,)
                fd_scale = g_spot.abs().median() + 1.0e-8
                rel = (g_fd - g_spot).abs() / (g_spot.abs() + fd_scale)
                R_t = float(self._liability_R[t].item())
                if R_t != 0.0:
                    sgn_R = torch.tensor(R_t, device=JL_t.device).sign()
                    liab_sign = float((JL_t.sign() == sgn_R).float().mean())
                    liab_ratio = float((JL_t / R_t).median())
                else:
                    liab_sign = float('nan')
                    liab_ratio = float('nan')
                # J_L-based liveness + sign, recorded at EVERY t (independent of R_t). The
                # label uses J_L_t (the framework on-day liability spot-delta), whose live
                # region extends one step past R_t's strict `fix_day > day_t` convention (the
                # last-fixing boundary: R_t=0 but J_L≠0). So the R_t sign-anchor above is
                # vacated there; the test re-anchors the sign on this J_L-live region instead,
                # asserting J_L's sign stays consistent with the R_t≠0 region across the
                # boundary (a wealth-equation liability flip would invert it everywhere).
                jl_mean_abs = float(JL_t.abs().mean())
                jl_sign_median = int(torch.sign(JL_t.median()).item())
                # Belief-coherence gate: `forward_belief_onestep` is a DERIVED replication of
                # the outer `_forward_belief` step — VERIFY it, don't trust the derivation. The
                # reconstructed (grad-connected) belief[t+1] MUST equal the outer filter's
                # published belief in the buffer; any off-by-one (P index, emission, predict-only
                # guard) would mark z[t+1] with a slightly-wrong belief, silently re-introducing
                # the value-incoherence the fork-free path removed. Expect ~1e-6.
                if belief_t1_outer is not None and belief_buf is not None:
                    belief_coherence = float(
                        (belief_t1_outer.detach() - belief_buf[t1]).abs().max())
                else:
                    belief_coherence = float('nan')
            # Decomposition cross-check: the F-moving TOTAL must equal the partial plus the
            # contemporaneous entry channel, total = partial + Σ_h ∂C/∂F_h·∂F_h/∂spot, with
            # ∂C/∂F_h = f_delta_grad and ∂F_h/∂spot = JF_t. Exact to autograd precision; proves
            # the partial/total split is the right decomposition (no double-counted F channel).
            with torch.enable_grad():
                spot_leaf_tot = spot_t_real.detach().clone().requires_grad_(True)
                Y_tot = _y_boot_of_spot(spot_leaf_tot, q_star_t1, move_entry_F=True)
                g_spot_total = torch.autograd.grad(Y_tot.sum(), spot_leaf_tot)[0]
            recon = g_spot + (f_delta_grad * JF_t).sum(dim=-1)
            decomp_scale = g_spot_total.abs().median() + 1.0e-8
            decomp_rel = ((g_spot_total - recon).abs()
                          / (g_spot_total.abs() + decomp_scale)).median()
            self._spot_grad_fd_gate[t] = {
                "n_rows": int(g_spot.shape[0]),
                "Rt": R_t,
                "fd_autograd_rel_err_median": float(rel.median()),
                "fd_autograd_rel_err_p90": float(rel.quantile(0.90)),
                "liab_delta_sign_agree_vs_Rt": liab_sign,
                "liab_delta_ratio_vs_Rt_median": liab_ratio,
                "liab_jl_mean_abs": jl_mean_abs,
                "liab_jl_sign_median": jl_sign_median,
                "belief_coherence_max_abs": belief_coherence,
                "total_eq_partial_plus_Fchannel_rel_err": float(decomp_rel),
                "eps_rel": 1.0e-3,
            }

        # Advantage decomposition: subtract B_t(z_t) from the value label so the
        # NN fits the residual A_t = C_t − B_t, and convert the gradient label to
        # the residual's slope on the columns where a slope was actually computed:
        #     ∂A_t/∂z = ∂Y_boot/∂z − ∂B_t/∂z        (∂B_t/∂z exact, closed form)
        # `dy_dz` carries a genuine pathwise ∂C_t/∂z on exactly the supervised column
        # groups below: live positions, wealth, live tradable_prices, belief when the
        # filter is differentiable, and observable spot. Every other column is an UNCOMPUTED zero, not a
        # measured slope. The earlier attempt subtracted ∂B/∂z across ALL columns,
        # which wrote `0 − ∂B/∂q` onto the positions block: a large slope target
        # of the WRONG SIGN (B carries the dominant first-order q-dependence by
        # design — that is its job), and standardization against the small
        # residual y_std inflated it ~10× → the twin-loss instability that forced
        # `w_diff = 0`. The w_diff=0 fallback in turn silently degraded the method
        # to value-only regression, whose max-of-noisy over-optimism grows with
        # backward depth — the measured horizon-scaling V_0 gap. Fix: subtract
        # ∂B/∂z where ∂C/∂z exists and MASK the unlabeled columns out of the diff
        # loss entirely (no target, no constraint) via `diff_col_mask`.
        # λ-mix (when active): blend the value label with a no-grad rollout Y_rollout to
        # break max-of-noisy compounding on the value labels. Rollout starts at
        # the bank's post-decision state (q_t_sampled applied, wealth_post_t)
        # and uses frozen C-stack for downstream decisions. Gradient label is
        # untouched — only the value path gets mixed.
        if self._advantage_decomp:
            B_at_z_t = self._baseline_B(z_t).detach()
            Y_value = Y_value_boot.detach()
            if self._lambda_mix > 0.0:
                with torch.no_grad():
                    Y_rollout = self._rollout_no_grad_to_T(
                        t_start=t, outer_indices=bank["outer_index"],
                        q_post_t=q_t_sampled.detach(),
                        wealth_post_t=wealth_post_t.detach(), traj=traj)
                Y_value = ((1.0 - self._lambda_mix) * Y_value
                            + self._lambda_mix * Y_rollout)
            y_target = Y_value - B_at_z_t
            diff_col_mask = torch.zeros(
                z_t.shape[-1], device=z_t.device, dtype=self.dtype)
            for j in range(n_h):
                if live_t[j]:
                    diff_col_mask[j] = 1.0
            diff_col_mask[wealth_col] = 1.0
            for j in range(n_h):
                if live_t[j]:
                    diff_col_mask[F_col_start + j] = 1.0
            # Belief columns: supervised only when the filter computed ∂Y_boot/∂belief_t
            # above. `∂B/∂belief` is already in `_dB_dz` (B is belief-linear), so the
            # residual label ∂A/∂belief = ∂Y_boot/∂belief − ∂B/∂belief is correct here.
            if belief_cols is not None:
                diff_col_mask[belief_cols[0]:belief_cols[1]] = 1.0
            # Spot column: the partial ∂C/∂spot|_F label. ∂B/∂spot is in `_dB_dz` (B marks the
            # liability against the real spot column), so the residual ∂A/∂spot = ∂C/∂spot −
            # ∂B/∂spot is correct here — B carries the bulk, A the residual.
            if spot_market_col is not None:
                diff_col_mask[spot_market_col] = 1.0
            if not getattr(self, "_logged_diffcols", False):
                self._logged_diffcols = True
                sup = [i for i in range(diff_col_mask.shape[0]) if float(diff_col_mask[i]) > 0]
                logging.info(
                    "DifferentialSolver diff-label supervised cols (StatePack idx)=%s "
                    "[live positions=%s, wealth=%d, live F=%s, belief=%s, spot=%s]; "
                    "belief differential %s",
                    sup, [j for j in range(n_h) if live_t[j]], wealth_col,
                    [F_col_start + j for j in range(n_h) if live_t[j]],
                    list(belief_cols) if belief_cols else None, spot_market_col,
                    "ACTIVE (filtered)" if belief_cols else "masked (one-hot/off)")
            dy_dz = (dy_dz - self._dB_dz(z_t)) * diff_col_mask
        else:
            y_target = Y_value_boot.detach()
            diff_col_mask = torch.zeros(
                z_t.shape[-1], device=z_t.device, dtype=self.dtype)
            for j in range(n_h):
                if live_t[j]:
                    diff_col_mask[j] = 1.0
                    diff_col_mask[F_col_start + j] = 1.0
            diff_col_mask[wealth_col] = 1.0
            if belief_cols is not None:
                diff_col_mask[belief_cols[0]:belief_cols[1]] = 1.0
            if spot_market_col is not None:
                diff_col_mask[spot_market_col] = 1.0

        # --- Label audit (opt-in, first round per t wins — the initial training bank): snapshot
        # the actual labels the net regresses to. Y_value_label = bootstrap value target;
        # Y_boot_realized = the realised-path AAD branch; baseline_B = analytic B_t;
        # residual = y_target = Y_value_label − B_t (what the NN fits — should be SMALL/O(1) under
        # advantage decomp). A residual that is LARGE and grows with the bank IS the label-bias
        # signature; |residual| ≫ |B| means the baseline is not capturing the bulk. Surfaces as
        # M1_label_audit_at_t. ---
        if t in self._label_audit_t_steps and t not in self._label_audit:
            with torch.no_grad():
                B_audit = self._baseline_B(z_t).detach()
                yt = y_target.detach()
                y_value = Y_value_boot.detach()
                y_realized = Y_boot.detach()
                sup_cols = [i for i in range(dy_dz.shape[-1])
                            if diff_col_mask is None or float(diff_col_mask[i]) > 0]
                grad_norm = (dy_dz[:, sup_cols].norm(dim=-1) if sup_cols
                             else torch.zeros(N, device=device))

                def _stats(x):
                    xf = x[torch.isfinite(x)]
                    nonfin = float((~torch.isfinite(x)).float().mean())
                    if xf.numel() == 0:
                        return {"min": float('nan'), "max": float('nan'), "mean": float('nan'),
                                "std": float('nan'), "median": float('nan'),
                                "abs_mean": float('nan'), "frac_nonfinite": nonfin}
                    return {"min": float(xf.min()), "max": float(xf.max()),
                            "mean": float(xf.mean()), "std": float(xf.std()),
                            "median": float(xf.median()), "abs_mean": float(xf.abs().mean()),
                            "frac_nonfinite": nonfin}

                k = min(3, N)
                self._label_audit[t] = {
                    "n_rows": int(N), "live_axes": int(live_t.sum()),
                    "Y_value_label": _stats(y_value),
                    "Y_boot_realized": _stats(y_realized),
                    # Backward-compatible key for the harness readout: now means the actual
                    # scalar value target, not necessarily the realised one-sample branch.
                    "Y_boot": _stats(y_value), "baseline_B": _stats(B_audit),
                    "residual_y_target": _stats(yt),
                    "grad_label_norm": {"median": float(grad_norm.median()),
                                        "max": float(grad_norm.max())},
                    "examples": [
                        {"Y_value_label": float(y_value[i]),
                         "Y_boot_realized": float(y_realized[i]), "B": float(B_audit[i]),
                         "residual": float(yt[i]),
                         "q_t": [float(v) for v in q_t_sampled[i]],
                         "wealth_post_t": float(wealth_post_t[i])}
                        for i in range(k)],
                }
            a = self._label_audit[t]
            logging.info(
                "DifferentialSolver LABEL AUDIT t=%d (live_axes=%d, N=%d): "
                "Y_value[med=%.4g |mean|=%.4g max=%.4g] baseline_B[med=%.4g |mean|=%.4g] "
                "RESIDUAL_target[med=%.4g |mean|=%.4g max=%.4g] (the NN's target — should be SMALL) "
                "grad_norm[med=%.4g max=%.4g] nonfinite=%.3g",
                t, a["live_axes"], a["n_rows"],
                a["Y_boot"]["median"], a["Y_boot"]["abs_mean"], a["Y_boot"]["max"],
                a["baseline_B"]["median"], a["baseline_B"]["abs_mean"],
                a["residual_y_target"]["median"], a["residual_y_target"]["abs_mean"],
                a["residual_y_target"]["max"],
                a["grad_label_norm"]["median"], a["grad_label_norm"]["max"],
                a["Y_boot"]["frac_nonfinite"])

        return (z_t.detach(), y_target, dy_dz.detach(), action_gap.detach(),
                diff_col_mask)

    def _fit_twin_sgd(self, net, z, y, dy_dz, action_gap, steps, log_label="",
                      col_mask=None):
        """One SGD pass over the supplied (z, y, dy_dz, action_gap) bank. Used by
        both the initial round and any correction-round refit. The action-gap mask
        threshold is recomputed per call against the current bank's median gap.
        `col_mask`: optional (D,) column mask restricting the differential loss to
        the labeled columns (advantage-decomp path; see `_compute_labels_for_bank`)."""
        device = self.device
        opt = torch.optim.Adam(net.parameters(), lr=self.adam_lr)
        gap_threshold = max(float(action_gap.median()), 1.0e-6) * 0.5
        n_rows = z.shape[0]
        batch = min(self.train_minibatch, n_rows)
        steps = max(1, int(steps))
        last_diag = {"total_loss": float("nan"), "val_loss": float("nan"),
                 "diff_loss": float("nan"), "steps_used": 0,
                 "max_steps": steps, "stopped_early": False}
        # Differentials are ON in both modes. Under advantage decomposition the
        # gradient label is the residual's own slope ∂A_t/∂z = ∂Y_boot/∂z − ∂B_t/∂z
        # on the labeled columns (col_mask), with ∂B_t/∂z exact closed-form — the
        # targets are now consistent with the function the NN fits, which removes
        # the instability that previously forced w_diff = 0. The w_diff = 0
        # fallback is retired: it silently degraded the method to value-only
        # regression, the exact setting where max-of-noisy over-optimism compounds
        # with backward depth (the measured horizon-scaling V_0 gap).
        w_diff = 1.0
        for step in range(steps):
            idx = torch.randint(0, n_rows, (batch,), device=device)
            loss, last_diag = twin_loss(
                net, z[idx], y[idx], dy_dz[idx],
                action_gap=action_gap[idx], gap_threshold=gap_threshold,
                w_val=1.0, w_diff=w_diff, col_mask=col_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_value = float(loss.detach())
            last_diag.update(
                steps_used=step + 1, max_steps=steps,
                stopped_early=self.train_loss_tol > 0.0
                and loss_value <= self.train_loss_tol)
            if last_diag["stopped_early"]:
                break
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
        final_diag = {"total_loss": float("nan"), "val_loss": float("nan"),
                  "diff_loss": float("nan"), "steps_used": 0,
                  "max_steps": self.train_steps_per_solve,
                  "stopped_early": False}
        escalation_flag = None
        per_round_audits = []

        for round_num in range(max_rounds):
            rounds_taken = round_num + 1
            # 1. Build bank from accumulated outer indices.
            bank = self._build_bank(t, traj, outer_indices=accumulated_outer_idx)
            # 2. Compute labels.
            z_t, y_target, dy_dz, action_gap, diff_col_mask = \
                self._compute_labels_for_bank(t, bank, C_next, traj=traj)
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
                log_label=f"C_t={t}_r={round_num}",
                col_mask=diff_col_mask)
            self.C[t] = net      # so the audit can call `self.C[t]` via decision op
            logging.info(
                "DifferentialSolver C_t=%d round=%d fit: live_axes=%d bank=%d "
                "twin val=%.5f diff=%.5f steps=%d/%d%s action_gap=%.4g",
                t, round_num, int(live_t.sum()), z_t.shape[0],
                final_diag["val_loss"], final_diag["diff_loss"],
                int(final_diag.get("steps_used", 0)),
                int(final_diag.get("max_steps", self.train_steps_per_solve)),
                " early" if final_diag.get("stopped_early") else "",
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
            "twin_total_loss": final_diag.get("total_loss", float("nan")),
            "train_steps_used": int(final_diag.get("steps_used", 0)),
            "train_max_steps": int(final_diag.get("max_steps", self.train_steps_per_solve)),
            "train_stopped_early": bool(final_diag.get("stopped_early", False)),
            "bank_rows": int(accumulated_outer_idx.shape[0]) * int(self.b_endo),
            "live_axes_at_t": int(live_t.sum()),
            "rounds_taken": rounds_taken,
            "delta_history_abs_mean": delta_history,
            "per_round_audits": per_round_audits,
            "escalation_flag": escalation_flag,
        }

    # --------------------------------- M2: audit rollout + correction loop
    def _rollout_no_grad_to_T(self, t_start, outer_indices, q_post_t, wealth_post_t,
                              traj, *, return_raw_wealth=False, decide_fn=None):
        """Frozen-policy no-grad rollout from post-decision state at `t_start` to
        terminal `T_dec`. By default returns `Y_eval = symlog(W_T)` per row (the
        audit's utility-units residual). Pass `return_raw_wealth=True` to get the
        raw realised terminal wealth in dollars — used by `_compute_dollar_floor`
        (spec §6 step 2: lower bound L = realised $/oz).

        `decide_fn=None` (default) ⇒ the learned argmax `C[u+1]` policy (the floor /
        audit path — unchanged, byte-identical). Pass a callable
        `decide_fn(u+1, market, F, q_carry, wealth_pre, static, time_to_T) -> q_target
        (N, n_h)` to roll a NON-learned frozen policy (unhedged / static) through the
        IDENTICAL wealth-MTM recursion — the apples-to-apples benchmark verdict (§7).

        At each future step u ∈ [t_start, t_outer−2]:
          • read realized (market_{u+1}, F_{u+1}) from `traj[u+1]` at the audit row's
            outer index (no fresh inner-MC needed — the audit path's outer trajectory
            IS the realized market evolution we're auditing the policy against).
          • evolve wealth: `wealth_pre_{u+1} = wealth_post_u + Σ_h q_h·cs_h·dF_h − R_u·dS`
            (MTM-invariant; linear average-rate liability closed form `dL = R_u·dS`).
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
                # Wealth MTM: hedge PnL minus liability MTM change. Linear
                # average-rate closed form `dL = R_u · dS` where dS = spot move
                # (here approximated by the first hedge's dF; near-month future
                # tracks spot tightly).
                F_u = traj[u]["tradables"][outer_indices]
                dF = F_u1 - F_u                                     # (N, n_h)
                # Liability marks to SPOT — the observable spot column, not the front
                # future (`dF[:,0]` freezes at the front contract's expiry; see _spot_col).
                spot_u = self._spot_col(traj[u]["market"][outer_indices])
                dS = (self._spot_col(market_u1) - spot_u
                      if spot_u is not None else dF[:, 0])
                pnl_hedges = (q_carry * self.contract_size
                              * dF).sum(dim=-1)                     # (N,)
                R_u_scalar = float(self._liability_R[u].item())
                dL = R_u_scalar * dS                                # (N,)
                wealth_pre_u1 = wealth + pnl_hedges - dL            # (N,)
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
                if decide_fn is None:
                    q_star_u1, _, _ = self._decision_at(
                        self.C[u + 1], market_u1, F_u1, q_carry,
                        wealth_pre_u1, time_to_T_u1, static_u1, u + 1)
                else:
                    q_star_u1 = decide_fn(u + 1, market_u1, F_u1, q_carry,
                                          wealth_pre_u1, static_u1, time_to_T_u1)
                cost_u1 = self._turnover(q_star_u1, q_carry, F_u1)
                wealth = wealth_pre_u1 - cost_u1
                q_carry = q_star_u1
            if return_raw_wealth:
                return wealth                                       # (N,) raw $
            # Terminal utility — the configured shape (symlog / huber / cara).
            Y_eval = _utility_wrap_signed(wealth, self.runtime)
        return Y_eval

    def _compute_dollar_floor(self, traj):
        """Spec §6 step 2: lower bound `L` = realised mean cost in $/oz of the
        policy on a fresh MC set.

        Rollout starts at `t_min` with `q_prev = 0`, `wealth_pre = 0`, and applies
        the policy's argmax at t=t_min, then runs `_rollout_no_grad_to_T` with
        `return_raw_wealth=True` to get realised terminal wealth in dollars per
        path. The deal denominator is `bundle['total_leg_volume']` (already used
        by `vol_scaled_notional` to size the symlog c, so the same source here is
        consistent).

        Returns mean / p95 / count + the $/oz floor for the ship-criterion
        readout (`gap = U − L`, ship when floor clears the dealer margin)."""
        device = self.device
        n_h = len(self.hedges)
        B = self._B()
        outer_idx = torch.arange(B, device=device)
        t0 = self.t_min
        market_0 = traj[t0]["market"]
        F_0 = traj[t0]["tradables"]
        static_0 = self._static_sim[t0].to(device).expand(B, self._static_sim.shape[-1])
        time_to_T_0 = torch.full((B,), float(self.t_outer - 1 - t0), device=device,
                                  dtype=self.dtype)
        q_prev = torch.zeros(B, n_h, device=device, dtype=self.dtype)
        wealth_pre = torch.zeros(B, device=device, dtype=self.dtype)
        with torch.no_grad():
            q_star_0, _, _ = self._decision_at(
                self.C[t0], market_0, F_0, q_prev, wealth_pre,
                time_to_T_0, static_0, t0)
            cost_0 = self._turnover(q_star_0, q_prev, F_0)
            wealth_post_0 = wealth_pre - cost_0
        terminal_wealth = self._rollout_no_grad_to_T(
            t_start=t0, outer_indices=outer_idx,
            q_post_t=q_star_0, wealth_post_t=wealth_post_0,
            traj=traj, return_raw_wealth=True)             # (B,) raw $
        volume_oz = float(self.bundle.get("total_leg_volume", 0.0))
        if volume_oz <= 0.0:
            logging.warning(
                "DifferentialSolver dollar floor: bundle['total_leg_volume']=%.3g, "
                "cannot compute $/oz. Reporting raw $ only.", volume_oz)
            volume_oz = 1.0
        mean_wealth = float(terminal_wealth.mean().item())
        p5_wealth = float(terminal_wealth.quantile(0.05).item())
        p95_wealth = float(terminal_wealth.quantile(0.95).item())
        # `L` = realised cost in $/oz where COST = `−wealth/volume` (positive cost =
        # we lost money, ship if cost ≤ margin). Negative cost = we made money.
        L_cost_per_oz = -mean_wealth / volume_oz
        L_p95_cost = -p5_wealth / volume_oz       # 95th-percentile worst case (cost ↑)
        logging.info(
            "DifferentialSolver L (realised $/oz): mean=%+.4f, "
            "p5_wealth=%+.4g, p95_wealth=%+.4g, volume=%.1f oz, "
            "L_cost=%.4f $/oz (positive=cost), L_p95_cost=%.4f $/oz",
            mean_wealth / volume_oz,
            p5_wealth, p95_wealth, volume_oz,
            L_cost_per_oz, L_p95_cost)
        return {
            "L_mean_wealth_usd": mean_wealth,
            "L_p5_wealth_usd": p5_wealth,
            "L_p95_wealth_usd": p95_wealth,
            "L_volume_oz": volume_oz,
            "L_mean_per_oz_usd": mean_wealth / volume_oz,
            "L_cost_per_oz_usd": L_cost_per_oz,
            "L_p95_cost_per_oz_usd": L_p95_cost,
            # Per-path learned terminal wealth (B,) — reused by the §7 benchmark verdict so
            # learned/unhedged/static are scored on the IDENTICAL batch. Underscore-prefixed:
            # the caller pops it before merging into the (JSON-scalar) diagnostics.
            "_terminal_wealth": terminal_wealth,
        }

    def _frozen_policy_terminal_wealth(self, traj, decide_fn):
        """Per-path terminal wealth (B,) in $ for a NON-learned frozen policy whose
        post-decision target inventory at each step is `decide_fn(t, market, F, q_carry,
        wealth_pre, static, time_to_T)`. Mirrors `_compute_dollar_floor`'s t_min setup
        (decide from flat, charge entry turnover) then reuses `_rollout_no_grad_to_T` with
        the SAME `decide_fn` — so the wealth-MTM recursion (hedge Σq·cs·dF minus liability
        dL=R·dS, plus per-expiry forced-unwind turnover) is IDENTICAL to the learned floor.
        Apples-to-apples with `_compute_dollar_floor`'s learned terminal wealth."""
        device = self.device
        n_h = len(self.hedges)
        B = self._B()
        outer_idx = torch.arange(B, device=device)
        t0 = self.t_min
        market_0 = traj[t0]["market"]
        F_0 = traj[t0]["tradables"]
        static_0 = self._static_sim[t0].to(device).expand(B, self._static_sim.shape[-1])
        time_to_T_0 = torch.full((B,), float(self.t_outer - 1 - t0), device=device,
                                  dtype=self.dtype)
        q_prev = torch.zeros(B, n_h, device=device, dtype=self.dtype)
        wealth_pre = torch.zeros(B, device=device, dtype=self.dtype)
        with torch.no_grad():
            q_star_0 = decide_fn(t0, market_0, F_0, q_prev, wealth_pre, static_0, time_to_T_0)
            wealth_post_0 = wealth_pre - self._turnover(q_star_0, q_prev, F_0)
        return self._rollout_no_grad_to_T(
            t_start=t0, outer_indices=outer_idx, q_post_t=q_star_0,
            wealth_post_t=wealth_post_0, traj=traj, return_raw_wealth=True,
            decide_fn=decide_fn)                                       # (B,) raw $

    def _compute_benchmark_comparison(self, traj, learned_wealth):
        """Piece §7 — the downside-protection VERDICT (port of `diffml_hedge_huber.py`'s
        final block). On the SAME outer batch (shared shocks) as the floor L, roll three
        policies and report mean / downside_sd / 5%-worst / 95%-best per path, in $ and $/oz:
          • learned   — the fitted argmax policy (terminal wealth reused from the floor).
          • unhedged  — q≡0 (bare liability; the do-nothing baseline).
          • static    — the single best constant hold (`run_textbook_benchmark`'s n*), held
                        live-masked (dead axes forced flat at expiry; the rollout charges the
                        unwind). n* is selected under the textbook objective; the realised
                        comparison here re-rolls it through the floor's recursion so all three
                        policies share one wealth convention.

        'Solved' (toy-parity criterion): learned `downside_sd` ≤ unhedged AND ≤ static — the
        policy cuts the realised downside vs both naïve baselines. `downside_sd` = RMS of the
        LOSS side only (`sqrt(mean(relu(-W)²))`) — the asymmetric-utility metric that matters;
        plain std counts upside dispersion as 'risk'."""
        device = self.device
        n_h = len(self.hedges)
        is_live = self.is_live

        def _unhedged(t, market, F, q_carry, wealth_pre, static, time_to_T):
            return torch.zeros_like(q_carry)

        tb = run_textbook_benchmark(self.bundle, self.runtime)
        n_star = torch.tensor(tb["n_star"], dtype=self.dtype, device=device)   # (n_h,)

        def _static(t, market, F, q_carry, wealth_pre, static, time_to_T):
            live = is_live[t].to(self.dtype)                                   # (n_h,)
            return (n_star * live).expand(q_carry.shape[0], n_h)

        unhedged_wealth = self._frozen_policy_terminal_wealth(traj, _unhedged)
        static_wealth = self._frozen_policy_terminal_wealth(traj, _static)
        volume_oz = float(self.bundle.get("total_leg_volume", 0.0)) or 1.0

        def _metrics(W):
            loss = torch.clamp(-W, min=0.0)
            return {
                "mean_usd": float(W.mean().item()),
                "downside_sd_usd": float((loss ** 2).mean().sqrt().item()),
                "p5_usd": float(W.quantile(0.05).item()),     # 5%-worst
                "p95_usd": float(W.quantile(0.95).item()),    # 95%-best
            }

        out = {}
        for name, W in (("learned", learned_wealth), ("unhedged", unhedged_wealth),
                        ("static", static_wealth)):
            m = _metrics(W)
            for k, v in m.items():
                out[f"bench_{name}_{k}"] = v
            out[f"bench_{name}_mean_per_oz_usd"] = m["mean_usd"] / volume_oz
            out[f"bench_{name}_downside_sd_per_oz_usd"] = m["downside_sd_usd"] / volume_oz
        out["bench_volume_oz"] = volume_oz
        out["bench_n_star_static"] = tb["n_star"]
        out["bench_learned_beats_unhedged_downside"] = bool(
            out["bench_learned_downside_sd_usd"] <= out["bench_unhedged_downside_sd_usd"])
        out["bench_learned_beats_static_downside"] = bool(
            out["bench_learned_downside_sd_usd"] <= out["bench_static_downside_sd_usd"])
        logging.info(
            "DifferentialSolver §7 VERDICT (downside protection, $/oz, B=%d paths):\n"
            "  %-9s  mean=%+9.4f  down_sd=%9.4f  5%%worst=%+9.4f  95%%best=%+9.4f\n"
            "  %-9s  mean=%+9.4f  down_sd=%9.4f  5%%worst=%+9.4f  95%%best=%+9.4f\n"
            "  %-9s  mean=%+9.4f  down_sd=%9.4f  5%%worst=%+9.4f  95%%best=%+9.4f\n"
            "  learned cuts downside vs unhedged=%s, vs static=%s  (n*_static=%s)",
            self._B(),
            "unhedged", out["bench_unhedged_mean_usd"] / volume_oz,
            out["bench_unhedged_downside_sd_usd"] / volume_oz,
            out["bench_unhedged_p5_usd"] / volume_oz, out["bench_unhedged_p95_usd"] / volume_oz,
            "static", out["bench_static_mean_usd"] / volume_oz,
            out["bench_static_downside_sd_usd"] / volume_oz,
            out["bench_static_p5_usd"] / volume_oz, out["bench_static_p95_usd"] / volume_oz,
            "learned", out["bench_learned_mean_usd"] / volume_oz,
            out["bench_learned_downside_sd_usd"] / volume_oz,
            out["bench_learned_p5_usd"] / volume_oz, out["bench_learned_p95_usd"] / volume_oz,
            out["bench_learned_beats_unhedged_downside"],
            out["bench_learned_beats_static_downside"], tb["n_star"])
        return out

    def _spot_col(self, market):
        """Observable spot price from a market block `(…, market_dim)` → `(…)`.

        The average-rate liability marks to SPOT, so the wealth-evolution `dL = R_t·dS`
        needs the SPOT increment. The historical `dS = dF[:,0]` (front-future move) proxy is
        WRONG once the front contract expires — its price freezes (realised) / is zeroed
        (inner fork), while the spot factor keeps simulating the whole horizon. The spot is
        now a first-class observable market column (`_spot_cols_in_market`); reading it is the
        only liability mark that stays valid across expiries. Returns None when no spot column
        is published (caller falls back to the front-future move)."""
        sc = self._spot_cols_in_market
        return market[..., sc[0]] if sc is not None else None

    def _continuation_value(self, C_next, market, F, q_prev, wealth_pre, t_next):
        """Continuation VALUE of the pre-decision state at `t_next` under the frozen
        `C_next` — the operator shared by the realised and inner-MC sides of the BSS
        penalty so the martingale difference is symmetric (same C, same decision rule;
        only the next-step noise differs).

          • interior t_next: the Bellman envelope `max_q C_next(post_{t_next}(q))` over
            the live action grid (`_decision_at`'s v_star), returning the argmax `q*` so
            the realised rollout can advance.
          • terminal t_next (= t_outer−1): no action — carry the inventory and read
            `C_next` at the post-state directly (mirrors `_rollout_no_grad_to_T`'s
            terminal handling). Returns `q*=None`.

        All no_grad — this is a value read, not a label.
        """
        N = market.shape[0]
        device = self.device
        static = self._static_sim[t_next].to(device).expand(
            N, self._static_sim.shape[-1])
        time_to_T = torch.full((N,), float(self.t_outer - 1 - t_next), device=device,
                               dtype=self.dtype)
        if t_next >= self.t_outer - 1:
            with torch.no_grad():
                z = self._assemble_z(market=market, q=q_prev, wealth=wealth_pre, F=F,
                                     time_to_T=time_to_T, static=static)
                v = self._C_full(C_next, z)
            return v, None
        q_star, v_star, _ = self._decision_at(
            C_next, market, F, q_prev, wealth_pre, time_to_T, static, t_next)
        return v_star, q_star

    def _compute_martingale_penalty_check(self, traj):
        """BSS validation sandwich — penalty `π`'s martingale-difference terms `Δ_t`
        and the dual-feasibility (zero-mean) HARD GATE (validation_sandwich_spec §3/§4.1;
        brief §1/§2). This is the first sandwich checkpoint: `π` must be dual-feasible or
        `U` is not a valid upper bound and the whole certificate is void.

        Along the policy rollout from `t_min` to terminal (same trajectory `L` is measured
        on — q=0/wealth=0 start, argmax policy), at each outer step `t` compute

            Δ_t  =  C_{t+1}(s_{t+1})  −  Ê[ C_{t+1}(s'_{t+1}) | s_t, a_t ]

          • `C_{t+1}(s_{t+1})` — the continuation VALUE (`_continuation_value`, the Bellman
            envelope / terminal utility) at the REALISED next state on this outer path under
            the policy action `a_t = q_carry`.
          • `Ê[·|s_t,a_t]` — the conditional expectation over the next-step noise, estimated
            on the retained inner-MC fork `bundle['inner_mc_fn'](t)` (offline; more draws than
            training are afforded via `Inner_Sub_Batch`). Ê is an unbiased mean over `B_inner`
            forks of the SAME continuation operator.

        Both terms share the (s_t,a_t)-measurable endogenous state (`q_carry`, `wealth`) and
        the SAME belief-filtered market extractor; they differ ONLY in the next-step noise
        (realised outer move vs inner-MC draws). Hence `E[Δ_t|s_t,a_t]=0` exactly and the
        sample mean of `Σ_t Δ_t` over paths is 0 up to MC error for ANY `C_t`. A non-zero
        mean ⇒ `π` is not dual-feasible (outer/inner dynamics mismatch) — STOP, do not build
        `U`.

        GUARD 2 (information sets, §4.2): both sides read the participant-visible
        belief-filtered market (`traj['market']` / inner `market_t1`, via the same
        `_extract_outer_state_at(privileged=True)`), NEVER the hidden regime one-hot.

        Returns the gate statistics + the per-`t` mean `Δ_t` profile (the §5 localisation
        hook). Pure diagnostic — never a solver dependency, never shipped (same offline
        status as the LSM / hindsight yardsticks)."""
        device = self.device
        n_h = len(self.hedges)
        B = self._B()
        t0 = self.t_min
        cs = self.contract_size

        if "inner_mc_fn" not in self.bundle:
            logging.warning(
                "DifferentialSolver penalty check: bundle has no 'inner_mc_fn' — cannot "
                "estimate Ê[C_{t+1}|s_t,a_t]; skipping the BSS zero-mean gate.")
            return {"penalty_zero_mean_skipped": True}

        # Policy rollout start at t_min (mirrors _compute_dollar_floor exactly).
        market_0 = traj[t0]["market"]
        F_0 = traj[t0]["tradables"]
        static_0 = self._static_sim[t0].to(device).expand(B, self._static_sim.shape[-1])
        time_to_T_0 = torch.full((B,), float(self.t_outer - 1 - t0), device=device,
                                  dtype=self.dtype)
        q_prev = torch.zeros(B, n_h, device=device, dtype=self.dtype)
        wealth_pre = torch.zeros(B, device=device, dtype=self.dtype)
        with torch.no_grad():
            q_star_0, _, _ = self._decision_at(
                self.C[t0], market_0, F_0, q_prev, wealth_pre, time_to_T_0, static_0, t0)
            cost_0 = self._turnover(q_star_0, q_prev, F_0)
            wealth = wealth_pre - cost_0
        q_carry = q_star_0                                       # a_{t_min} (post-decision)

        delta_sum = torch.zeros(B, device=device, dtype=self.dtype)
        deltas_per_t = []                 # [(t, Δ_t (B,))] — for the data-driven boundary
        per_t_mean = {}
        term_abs_total = 0.0
        term_count = 0
        n_inner_used = None
        with torch.no_grad():
            for t in range(t0, self.t_outer - 1):
                t1 = t + 1
                C_next = self.C[t1]
                if C_next is None:
                    break

                # --- realised next state under action a_t = q_carry (outer path) ---
                market_t = traj[t]["market"]
                F_t = traj[t]["tradables"]                       # (B, n_h)
                F_t1_real = traj[t1]["tradables"]
                market_t1_real = traj[t1]["market"]
                dF_real = F_t1_real - F_t                         # (B, n_h)
                # Liability marks to SPOT (front-future proxy freezes at expiry — see _spot_col).
                spot_t = self._spot_col(market_t)
                dS_real = (self._spot_col(market_t1_real) - spot_t
                           if spot_t is not None else dF_real[:, 0])
                pnl_real = (q_carry * cs * dF_real).sum(dim=-1)
                R_t = float(self._liability_R[t].item())
                wealth_pre_t1_real = wealth + pnl_real - R_t * dS_real
                v_real, q_star_t1 = self._continuation_value(
                    C_next, market_t1_real, F_t1_real, q_carry, wealth_pre_t1_real, t1)

                # --- Ê[C_{t+1}|s_t,a_t]: inner-MC fork at t, same operator/state, fresh noise ---
                inner = self.bundle["inner_mc_fn"](t)
                m_t1 = inner.get("market_t1")
                if m_t1 is None:
                    # Degenerate inner horizon (no t+1 slice) — Δ_t undefined; stop here.
                    break
                B_inner = m_t1.shape[1]
                n_inner_used = B_inner
                F_t1_inner = torch.stack(
                    [inner["F_t1"][h] for h in self.hedges], dim=-1)   # (B, B_inner, n_h)
                dF_inner = F_t1_inner - F_t.unsqueeze(1)                # (B, B_inner, n_h)
                # Same SPOT liability mark as the realised side (inner spot column at t+1).
                dS_inner = (self._spot_col(m_t1) - spot_t.unsqueeze(1)
                            if spot_t is not None else dF_inner[..., 0])
                pnl_inner = (q_carry.unsqueeze(1) * cs * dF_inner).sum(dim=-1)
                wealth_pre_t1_inner = wealth.unsqueeze(1) + pnl_inner - R_t * dS_inner
                BI = B * B_inner
                # Chunk the inner continuation over BI: `_continuation_value` → `_decision_at`
                # expands each row by the action grid K, so the un-chunked (B·B_inner·K) envelope
                # OOMs on GPU at large B (e.g. B=2048, B_inner=128, K=121 → ~15 GiB). Bound it to
                # ~`upper_bound_chunk_rows` post-expansion, same as the U DP's `_env`.
                mk_in = m_t1.reshape(BI, -1)
                Ff_in = F_t1_inner.reshape(BI, n_h)
                qf_in = q_carry.unsqueeze(1).expand(B, B_inner, n_h).reshape(BI, n_h)
                wf_in = wealth_pre_t1_inner.reshape(BI)
                v_inner = torch.empty(BI, device=device, dtype=self.dtype)
                K_act = self.action_grid.shape[0]
                step_bi = max(1, self._upper_chunk_rows // max(1, K_act))
                for i0 in range(0, BI, step_bi):
                    sl = slice(i0, i0 + step_bi)
                    vv, _ = self._continuation_value(
                        C_next, mk_in[sl], Ff_in[sl], qf_in[sl], wf_in[sl], t1)
                    v_inner[sl] = vv
                E_hat = v_inner.reshape(B, B_inner).mean(dim=1)        # (B,)

                delta_t = v_real - E_hat                               # (B,)
                delta_sum = delta_sum + delta_t
                deltas_per_t.append((t, delta_t))
                per_t_mean[t] = float(delta_t.mean().item())
                term_abs_total += float(delta_t.abs().sum().item())
                term_count += B

                if q_star_t1 is None:                                  # terminal reached
                    break
                cost_t1 = self._turnover(q_star_t1, q_carry, F_t1_real)
                wealth = wealth_pre_t1_real - cost_t1
                q_carry = q_star_t1

        n_steps = len(per_t_mean)
        term_abs_mean = term_abs_total / term_count if term_count else 0.0

        # Zero-mean statistic of a Σ over a set of per-t Δ vectors.
        def _zero_mean_stats(dvecs):
            if not dvecs:
                return 0.0, 0.0, 0.0, 0.0
            s = torch.stack(dvecs, dim=0).sum(dim=0)               # (B,)
            m = float(s.mean().item())
            sd = float(s.std(unbiased=True).item()) if B > 1 else 0.0
            se = sd / (B ** 0.5) if B > 0 else float("nan")
            z = abs(m) / se if se > 0 else (0.0 if m == 0 else float("inf"))
            return m, se, z, sd

        all_vecs = [d for _, d in deltas_per_t]
        mean_delta_sum, stderr, z_stat, std_delta_sum = _zero_mean_stats(all_vecs)

        # DATA-DRIVEN boundary (validation_sandwich_spec §4.1): grow a contiguous TERMINAL block
        # inward — dropping the terminal-most step each time — until the remaining INTERIOR sum is
        # zero-mean (|mean|/stderr < z-threshold), capped at `_penalty_max_boundary`. The penalty
        # is certified dual-feasible on the interior; U zeros π on exactly the boundary block (the
        # steps where Ê is over-dispersion-biased under near-terminal curvature). Self-consistent
        # with the gate, non-magic, calendar-agnostic. A gross whole-profile dynamics bug makes
        # EVERY step fail → the block hits the cap with the interior still failing → caught.
        cap = min(self._penalty_max_boundary, max(0, n_steps - 1))
        interior_vecs = list(all_vecs)
        boundary_count = 0
        while boundary_count < cap:
            _, _, z_try, _ = _zero_mean_stats(interior_vecs)
            if z_try < self._penalty_zero_mean_z:
                break
            interior_vecs.pop()                                    # drop the terminal-most step
            boundary_count += 1
        n_steps_interior = len(interior_vecs)
        mean_int, stderr_int, z_int, _ = _zero_mean_stats(interior_vecs)
        boundary_t_steps = [t for t, _ in deltas_per_t[n_steps_interior:]]
        boundary_hit_cap = boundary_count >= cap and z_int >= self._penalty_zero_mean_z

        logging.info(
            "DifferentialSolver BSS penalty zero-mean gate (§4.1): INTERIOR mean ΣΔ_t = %+.4g "
            "(stderr %.4g, z=%.2f, %d steps) — the dual-feasibility grade; FULL mean ΣΔ_t = "
            "%+.4g (z=%.2f, %d steps). Data-driven boundary = %d step(s) %s (zeroed in U). "
            "B=%d, B_inner=%s, mean|Δ_t|=%.4g. %s",
            mean_int, stderr_int, z_int, n_steps_interior,
            mean_delta_sum, z_stat, n_steps, boundary_count, boundary_t_steps or "[]",
            B, n_inner_used, term_abs_mean,
            "BOUNDARY HIT CAP — interior still NOT zero-mean, π INFEASIBLE, U void"
            if boundary_hit_cap else
            ("INTERIOR DUAL-FEASIBLE (≈0)" if z_int < self._penalty_zero_mean_z
             else "INTERIOR NOT zero-mean — π INFEASIBLE, U void"))
        return {
            # Interior (after dropping the data-driven boundary block) — the dual-feasibility grade.
            "penalty_zero_mean_delta_sum": mean_int,
            "penalty_zero_mean_stderr": stderr_int,
            "penalty_zero_mean_z": z_int,
            "penalty_n_steps_interior": n_steps_interior,
            # Data-driven boundary block (zeroed in U): count + the actual t indices.
            "penalty_boundary_count": boundary_count,
            "penalty_boundary_t_steps": boundary_t_steps,
            "penalty_boundary_hit_cap": boundary_hit_cap,
            "penalty_max_boundary": cap,
            "penalty_zero_mean_z_threshold": self._penalty_zero_mean_z,
            # Full sum (all steps incl. the near-terminal boundary ramp) — reported alongside.
            "penalty_full_delta_sum": mean_delta_sum,
            "penalty_full_z": z_stat,
            "penalty_full_delta_sum_std": std_delta_sum,
            "penalty_term_abs_mean": term_abs_mean,
            "penalty_n_paths": B,
            "penalty_n_inner": n_inner_used if n_inner_used is not None else 0,
            "penalty_n_steps": n_steps,
            "penalty_t_start": t0,
            "penalty_delta_per_t_mean": per_t_mean,
        }

    def _compute_penalized_upper_bound(self, traj, penalty):
        """BSS sandwich step 3 — the penalized clairvoyant UPPER bound `U` (validation_
        sandwich_spec §3; brief §3). Port of `diffml_hedge_huber.penalized_upper`:

            U = E[ max_{q-path} ( u(W_T) − Σ_t π_t(s_t, q_t) ) ],
            π_t(s_t, q) = C_{t+1}(s_{t+1})  −  Ê[ C_{t+1}(s'_{t+1}) | s_t, q ].

        The clairvoyant sees the whole realised price path; the martingale penalty `π`
        charges it for that foresight, so the bound tightens toward `V*` as `C → V*`.
        Solved by a per-path BACKWARD DP on a discretised WEALTH grid:

            J_T(N)  = u(N)                              (the configured terminal utility)
            J_t(N)  = max_q [ J_{t+1}(N')  −  π_t(N, q) ],  N' interpolated on the grid.

        Exactly the operator `_compute_martingale_penalty_check` already builds, but swept
        over (wealth grid × action grid) instead of along the single policy trajectory —
        `C_real`/`Ê` reuse the SAME continuation envelope (`_continuation_value`), so `U`'s
        `π` is the identical martingale difference the zero-mean gate certified.

        Per the standing directives:
          • `π ≡ 0` on the DATA-DRIVEN boundary block (`penalty['penalty_boundary_t_steps']`)
            — zero is itself dual-feasible, keeping `U` a VALID (safe-loose) bound there.
          • `U = min(U_naive, U_pen)` — the tightest valid bound. `U_naive` is the same DP
            with `π ≡ 0` everywhere (pure clairvoyant: free foresight, no penalty).
          • A WIDE gap `U − L` is AMBIGUOUS (perfect-foresight slack + the turnover/grid
            relaxations below), NOT necessarily policy suboptimality.

        Relaxations that keep `U` valid but LOOSE (each only RAISES the bound): the wealth-
        grid DP picks `q` per step with no inventory axis, so the step-`t` ENTRY turnover
        (`q_t` vs `q_{t-1}`) is not charged (the t+1 re-trade still is, inside the envelope);
        and the bound is read at the policy's `t_min` start wealth (0), matching `L`. Both
        are documented looseness, not bias — `HindsightDpSolver` is a separate, tighter
        naive clairvoyant reference (max-plus on cash with the inventory axis + turnover).

        Discretised-DP GUARDS (memory rules, mandatory and permanent): the OFF-GRID clamp
        fraction (silent clamping biases `U` upward / can invert the policy) and the
        wealth-move RESOLUTION vs grid step (`vol/Δgrid`, `|drift|/Δgrid` — too coarse and
        the kernel masks the dynamics). Both are logged and returned; a high clamp fraction
        means the grid is too narrow and `U` is suspect.

        Offline diagnostic only — never a solver dependency, never shipped."""
        device = self.device
        dtype = self.dtype
        n_h = len(self.hedges)
        cs = self.contract_size                                  # (n_h,)
        t0 = self.t_min
        runtime = self.runtime

        if penalty.get("penalty_zero_mean_skipped"):
            logging.warning(
                "DifferentialSolver U: penalty check was skipped (no inner_mc_fn) — Ê[C] is "
                "unavailable, cannot build the penalized bound. Skipping U.")
            return {"U_skipped": True, "U_skip_reason": "penalty_skipped"}
        if "inner_mc_fn" not in self.bundle:
            logging.warning("DifferentialSolver U: bundle has no 'inner_mc_fn' — skipping U.")
            return {"U_skipped": True, "U_skip_reason": "no_inner_mc_fn"}
        if penalty.get("penalty_boundary_hit_cap"):
            # Interior π is NOT zero-mean even after the boundary block — π is infeasible, so
            # the certificate is void. Per the spec: stop, do not certify U.
            logging.error(
                "DifferentialSolver U: penalty boundary HIT CAP (interior ΣΔ_t still not "
                "zero-mean, z=%.2f ≥ %.2f) — π is INFEASIBLE, U would NOT be a valid upper "
                "bound. Skipping U (certificate void).",
                penalty.get("penalty_zero_mean_z", float("nan")), self._penalty_zero_mean_z)
            return {"U_skipped": True, "U_skip_reason": "penalty_infeasible"}

        boundary_set = set(penalty.get("penalty_boundary_t_steps", []))
        B = self._B()
        P = min(B, self._upper_max_paths)
        G = self._upper_n_grid
        outer_idx = torch.arange(P, device=device)
        K = self.action_grid.shape[0]
        steps = list(range(t0, self.t_outer - 1))               # decision steps t0 … t_outer-2
        if not steps:
            logging.warning(
                "DifferentialSolver U: no decision steps in [t_min=%d, t_outer-1=%d) — "
                "degenerate horizon, skipping U.", t0, self.t_outer - 1)
            return {"U_skipped": True, "U_skip_reason": "degenerate_horizon"}

        # ---- Lower bound L (utility units) on the SAME P paths, for the gap + U≥L check ----
        # Mirror _compute_dollar_floor's t0 decision, then a frozen-policy rollout. Y_eval is
        # already the configured terminal utility u(W_T); L_util = mean. (The $/oz floor stays
        # separate — different unit system, different consumer.)
        market_0 = traj[t0]["market"][:P]
        F_0 = traj[t0]["tradables"][:P]
        static_0 = self._static_sim[t0].to(device).expand(P, self._static_sim.shape[-1])
        time_to_T_0 = torch.full((P,), float(self.t_outer - 1 - t0), device=device, dtype=dtype)
        q_prev0 = torch.zeros(P, n_h, device=device, dtype=dtype)
        wealth_pre0 = torch.zeros(P, device=device, dtype=dtype)
        with torch.no_grad():
            q_star_0, _, _ = self._decision_at(
                self.C[t0], market_0, F_0, q_prev0, wealth_pre0, time_to_T_0, static_0, t0)
            wealth_post_0 = wealth_pre0 - self._turnover(q_star_0, q_prev0, F_0)
        W_T_greedy = self._rollout_no_grad_to_T(
            t_start=t0, outer_indices=outer_idx, q_post_t=q_star_0,
            wealth_post_t=wealth_post_0, traj=traj, return_raw_wealth=True)   # (P,) raw $
        L_util_path = _utility_wrap_signed(W_T_greedy, runtime)               # (P,) utility units
        L_util = float(L_util_path.mean().item())

        # ---- Wealth grid: bracket the realised greedy terminal wealth, padded each side.
        # Off-grid fraction (below) is the quality gate; widen `upper_bound_grid_pad` if it bites.
        w_lo = float(W_T_greedy.min().item())
        w_hi = float(W_T_greedy.max().item())
        span = max(w_hi - w_lo, 2.0 * self.utility_c)
        pad = self._upper_grid_pad * span
        xs = torch.linspace(w_lo - pad, w_hi + pad, G, device=device, dtype=dtype)  # (G,)
        grid_step = float((xs[-1] - xs[0]).item()) / (G - 1)

        # Live-masked action grid (dead axes → 0, exactly as _decision_at): the DP's own max_q.
        q_grid = self.action_grid.clone()                        # (K, n_h)
        live_any = self.is_live[t0]
        if (~live_any).any():
            # Use per-step liveness inside the loop; here just note the grid shape.
            pass

        est_evals = len(steps) * P * G * K * K * (1 + self._upper_n_inner)
        logging.info(
            "DifferentialSolver U: building penalized clairvoyant bound. P=%d (of B=%d), "
            "wealth_grid=%d, action_grid K=%d, n_inner=%d, steps=%d (t %d→%d). "
            "Boundary block (π≡0) = %s. ~%.2gM continuation evals (chunk=%d rows). "
            "Grid=[%.4g, %.4g], Δgrid=%.4g $. Utilities normalised by c=%.4g.",
            P, B, G, K, self._upper_n_inner, len(steps), steps[-1], t0,
            sorted(boundary_set) or "[]", est_evals / 1e6, self._upper_chunk_rows,
            float(xs[0]), float(xs[-1]), grid_step, self.utility_c)

        # ---- chunked continuation envelope over a flattened (market,F,q,wealth) batch ----
        def _env(C_next, market_f, F_f, q_f, w_f, t1):
            M = market_f.shape[0]
            out = torch.empty(M, device=device, dtype=dtype)
            step = max(1, self._upper_chunk_rows // max(1, K))   # envelope expands by K interior
            for i in range(0, M, step):
                sl = slice(i, i + step)
                v, _ = self._continuation_value(
                    C_next, market_f[sl], F_f[sl], q_f[sl], w_f[sl], t1)
                out[sl] = v
            return out

        # ---- backward DP: two J's (penalized + naive) sharing the realised paths ----
        J_pen = _utility_wrap_signed(xs, runtime).unsqueeze(0).expand(P, G).clone()   # J_T = u(N)
        J_naive = J_pen.clone()
        off_grid_n = 0
        off_grid_tot = 0
        vol_over_step = []
        drift_over_step = []
        with torch.no_grad():
            for t in reversed(steps):
                t1 = t + 1
                C_next = self.C[t1]
                if C_next is None:
                    logging.warning("DifferentialSolver U: C[%d] is None — stopping DP early.", t1)
                    break
                live_t = self.is_live[t]
                qg = q_grid.clone()
                if (~live_t).any():
                    qg[:, ~live_t] = 0.0
                qg_cs = qg * cs                                  # (K, n_h)

                market_t = traj[t]["market"][:P]
                F_t = traj[t]["tradables"][:P]
                market_t1 = traj[t1]["market"][:P]
                F_t1 = traj[t1]["tradables"][:P]
                dim = market_t1.shape[-1]
                spot_t = self._spot_col(market_t)
                dF_real = F_t1 - F_t                             # (P, n_h)
                dS_real = (self._spot_col(market_t1) - spot_t
                           if spot_t is not None else dF_real[:, 0])         # (P,)
                R_t = self._liability_R[t]
                dL_real = R_t * dS_real                          # (P,)
                pnl_real = (qg_cs.unsqueeze(0) * dF_real.unsqueeze(1)).sum(-1)   # (P, K)
                dN_real = pnl_real - dL_real.unsqueeze(1)        # (P, K) per-step wealth move
                Nreal = (xs.view(1, G, 1) + pnl_real.unsqueeze(1)
                         - dL_real.view(P, 1, 1))                # (P, G, K)

                # off-grid + resolution diagnostics (mandatory discretised-DP guards)
                off_grid_n += int(((Nreal < xs[0]) | (Nreal > xs[-1])).sum().item())
                off_grid_tot += Nreal.numel()
                vol_over_step.append(float(dN_real.std().item()) / grid_step)
                drift_over_step.append(abs(float(dN_real.mean().item())) / grid_step)

                is_boundary = t in boundary_set
                if is_boundary:
                    pi = torch.zeros(P, G, K, device=device, dtype=dtype)
                else:
                    # C_real = continuation envelope at the REALISED next state (P,G,K)
                    mk = market_t1.view(P, 1, 1, dim).expand(P, G, K, dim).reshape(-1, dim)
                    Ff = F_t1.view(P, 1, 1, n_h).expand(P, G, K, n_h).reshape(-1, n_h)
                    qf = qg.view(1, 1, K, n_h).expand(P, G, K, n_h).reshape(-1, n_h)
                    C_real = _env(C_next, mk, Ff, qf, Nreal.reshape(-1), t1).reshape(P, G, K)

                    # Ê[C|s_t,q] via INDEPENDENT inner draws (no winner's curse), mean over forks
                    inner = self.bundle["inner_mc_fn"](t)
                    m_t1_in = inner.get("market_t1")
                    if m_t1_in is None:
                        logging.warning(
                            "DifferentialSolver U: inner fork has no market_t1 at t=%d — "
                            "treating as boundary (π≡0) for this step.", t)
                        pi = torch.zeros(P, G, K, device=device, dtype=dtype)
                    else:
                        I = min(m_t1_in.shape[1], self._upper_n_inner)
                        m_t1_in = m_t1_in[:P, :I]                          # (P, I, dim)
                        F_t1_in = torch.stack(
                            [inner["F_t1"][h] for h in self.hedges], dim=-1)[:P, :I]  # (P,I,n_h)
                        dF_in = F_t1_in - F_t.unsqueeze(1)                 # (P, I, n_h)
                        dS_in = (self._spot_col(m_t1_in) - spot_t.unsqueeze(1)
                                 if spot_t is not None else dF_in[..., 0])  # (P, I)
                        dL_in = R_t * dS_in                                # (P, I)
                        pnl_in = (qg_cs.view(1, 1, K, n_h)
                                  * dF_in.view(P, I, 1, n_h)).sum(-1)      # (P, I, K)
                        Nin = (xs.view(1, G, 1, 1)
                               + pnl_in.permute(0, 2, 1).unsqueeze(1)      # (P,1,K,I)
                               - dL_in.view(P, 1, 1, I))                   # (P, G, K, I)
                        off_grid_n += int(((Nin < xs[0]) | (Nin > xs[-1])).sum().item())
                        off_grid_tot += Nin.numel()
                        mk = m_t1_in.view(P, 1, 1, I, dim).expand(
                            P, G, K, I, dim).reshape(-1, dim)
                        Ff = F_t1_in.view(P, 1, 1, I, n_h).expand(
                            P, G, K, I, n_h).reshape(-1, n_h)
                        qf = qg.view(1, 1, K, 1, n_h).expand(
                            P, G, K, I, n_h).reshape(-1, n_h)
                        C_e = _env(C_next, mk, Ff, qf, Nin.reshape(-1), t1
                                   ).reshape(P, G, K, I).mean(-1)          # (P, G, K)
                        pi = C_real - C_e                                  # (P,G,K) mean-zero in q

                Jnext_pen = _interp_wealth_grid(xs, J_pen, Nreal)          # (P, G, K)
                Jnext_naive = _interp_wealth_grid(xs, J_naive, Nreal)
                J_pen = (Jnext_pen - pi).max(-1).values                    # (P, G)
                J_naive = Jnext_naive.max(-1).values                       # (P, G)

                if (not is_boundary) and m_t1_in is not None:
                    logging.info(
                        "DifferentialSolver U  t=%d: π mean=%+.4g abs_max=%.4g | "
                        "J_pen∈[%+.3g,%+.3g] J_naive∈[%+.3g,%+.3g] | dN_real σ/Δgrid=%.2f",
                        t, float(pi.mean()), float(pi.abs().max()),
                        float(J_pen.min()), float(J_pen.max()),
                        float(J_naive.min()), float(J_naive.max()), vol_over_step[-1])
                else:
                    logging.info(
                        "DifferentialSolver U  t=%d: BOUNDARY π≡0 | J_pen∈[%+.3g,%+.3g] | "
                        "dN_real σ/Δgrid=%.2f",
                        t, float(J_pen.min()), float(J_pen.max()), vol_over_step[-1])

            zero_q = torch.zeros(P, 1, device=device, dtype=dtype)
            U_pen_path = _interp_wealth_grid(xs, J_pen, zero_q).squeeze(1)     # (P,)
            U_naive_path = _interp_wealth_grid(xs, J_naive, zero_q).squeeze(1)
        U_pen = float(U_pen_path.mean().item())
        U_naive = float(U_naive_path.mean().item())
        # U = min of the two PATH-AVERAGED bounds (the toy's `min(U_naive, U_pen)`). Each is an
        # upper bound on V* IN EXPECTATION (E[dual] ≥ V*); the per-path min would NOT be — min and
        # E don't commute, so mean_p min(·,·) ≤ min(mean,mean) can fall below V*. Min of the means.
        U = min(U_naive, U_pen)
        gap = U - L_util

        # ---- U ≥ L verification. The pathwise check uses U_NAIVE (the pure clairvoyant must
        # dominate the deployed policy on EVERY path); U_pen ≥ L holds in MEAN only (π can
        # depress single paths). Mean U ≥ mean L is the headline certificate. ----
        tol = 1e-3 * max(1.0, abs(L_util))
        viol_path = int((U_naive_path < L_util_path - tol).sum().item())
        mean_ok = U >= L_util - tol
        off_grid_frac = off_grid_n / off_grid_tot if off_grid_tot else 0.0
        vol_res = sum(vol_over_step) / len(vol_over_step) if vol_over_step else 0.0
        drift_res = sum(drift_over_step) / len(drift_over_step) if drift_over_step else 0.0

        if off_grid_frac > self._upper_clamp_warn:
            logging.warning(
                "DifferentialSolver U: OFF-GRID clamp fraction = %.2f%% > %.2f%% — the wealth "
                "grid is too narrow; U is biased (clamping caps the bound). Widen "
                "`upper_bound_grid_pad`/`upper_bound_wealth_grid`. (memory: discretised-DP-"
                "boundary-clamp.)", 100 * off_grid_frac, 100 * self._upper_clamp_warn)
        if vol_res < 0.3:
            logging.warning(
                "DifferentialSolver U: wealth-move σ/Δgrid = %.2f < 0.3 — the grid is too "
                "COARSE relative to per-step wealth moves; the discrete kernel may mask the "
                "dynamics (memory: discretised-DP-drift-resolution). Increase "
                "`upper_bound_wealth_grid`.", vol_res)
        if viol_path > 0:
            logging.warning(
                "DifferentialSolver U: %d/%d paths have U_naive < L (pathwise clairvoyant "
                "below policy) — expected only if off-grid clamping bites (off-grid=%.2f%%).",
                viol_path, P, 100 * off_grid_frac)

        logging.info(
            "DifferentialSolver BSS UPPER bound (§3): L=%+.4f ≤ U=%+.4f  →  gap U−L = %.4f "
            "(utility units). Components: U_naive=%+.4f, U_pen=%+.4f (U=min). "
            "MEAN U≥L: %s. Off-grid clamp=%.2f%%, σ/Δgrid=%.2f, |drift|/Δgrid=%.3f, "
            "boundary steps zeroed=%d. NOTE: a wide gap is AMBIGUOUS — perfect-foresight + "
            "turnover/grid relaxation slack, NOT necessarily policy suboptimality.",
            L_util, U, gap, U_naive, U_pen, "OK" if mean_ok else "VIOLATED (U<L!)",
            100 * off_grid_frac, vol_res, drift_res, len(boundary_set))

        return {
            "U_skipped": False,
            "U_upper_bound": U,
            "U_naive": U_naive,
            "U_penalized": U_pen,
            "U_lower_bound_L_util": L_util,
            "U_gap": gap,
            "U_mean_ge_L": bool(mean_ok),
            "U_pathwise_naive_violations": viol_path,
            "U_n_paths": P,
            "U_wealth_grid": G,
            "U_n_inner": self._upper_n_inner,
            "U_grid_lo": float(xs[0].item()),
            "U_grid_hi": float(xs[-1].item()),
            "U_grid_step": grid_step,
            "U_off_grid_frac": off_grid_frac,
            "U_vol_over_gridstep": vol_res,
            "U_drift_over_gridstep": drift_res,
            "U_boundary_steps_zeroed": sorted(boundary_set),
        }

    def _compute_oracle_action_match(self, traj):
        """OFFLINE gate-4 verdict: per-DEPTH action-match of the fitted policy vs the exact-DP
        oracle (`gate2_exact_dp.npz`). For each decision t, at canonical oracle grid states,
        compare `argmax_a C[t](post(a))` to the oracle's optimal action; the regression slope of
        mean |Δq| vs backward depth is the 'does the policy ranking survive depth' gate (flat =
        depth-robust; growing-backward = max-of-noisy compounding). Gated behind
        `Oracle_Action_Match_Path`, toy-only, NEVER shipped — same offline status as the BSS / LSM
        / hindsight yardsticks.

        Alignments (verified on the gate4 toy): riskflow t == oracle t (is_live A-expiry == T_A and
        B-expiry == T_dec, so no offset); wealth = the oracle MTM `w` (same convention that made the
        historical V_0 comparison valid); true regime r → ONE-HOT belief (privileged — matches the
        oracle's true-regime conditioning); forward == spot (toy has no carry/basis) so F_A=F_B=s;
        basis/carry/static come from a real `traj` template row (zero-vol on the toy → any row)."""
        import numpy as np
        path = self._oracle_action_match_path
        if not os.path.exists(path):
            logging.warning("Oracle action-match: npz '%s' not found — skipping.", path)
            return {"oracle_match_skipped": True, "oracle_match_reason": "npz_missing"}
        d = np.load(path, allow_pickle=True)
        q_grid = d["q_grid"]; s_grid = d["s_grid"]; w_grid = d["w_grid"]
        T_A = int(d["T_A"]); T_dec = int(d["T_dec"])
        pol_preA_qA = d["pol_pre_A_qA"]; pol_preA_qB = d["pol_pre_A_qB"]
        pol_postA_qB = d["pol_post_A_qB"]
        device, dtype = self.device, self.dtype
        n_h = len(self.hedges)
        qcen = int(np.argmin(np.abs(q_grid)))            # index of q=0 in q_grid
        # Hedge A = expires first (is_live drops earliest), B = last.
        first_dead = [next((t for t in range(self.t_outer) if not bool(self.is_live[t, j])),
                            self.t_outer) for j in range(n_h)]
        A = int(np.argmin(first_dead)); B = int(np.argmax(first_dead))
        rlo, rhi = self._regime_cols_in_market
        n_states = rhi - rlo
        spot_col = (self._spot_cols_in_market[0]
                    if self._spot_cols_in_market is not None else None)
        # In-domain wealth subsample (C trained on |w| ≤ bank_wealth_halfwidth) + canonical w≈100.
        w_targets = [w for w in (100.0, 0.0, -400.0, 600.0)
                     if abs(w) <= self.bank_wealth_halfwidth]
        w_idx = [int(np.argmin(np.abs(w_grid - w))) for w in w_targets]
        s_idx = list(range(len(s_grid)))
        logging.info(
            "Oracle action-match: A=%s(hedge %d, dies t=%d) B=%s(hedge %d, dies t=%d); "
            "T_A=%d T_dec=%d; regime_cols=%s spot_col=%s; grading t=0..%d at q_prev=0, "
            "%d spots × %d wealth × %d regimes.",
            self.hedges[A], A, first_dead[A], self.hedges[B], B, first_dead[B],
            T_A, T_dec, (rlo, rhi), spot_col, min(T_dec, self.t_outer - 1) - 1,
            len(s_idx), len(w_idx), n_states)

        per_t_err = {}; per_t_match = {}
        with torch.no_grad():
            for t in range(0, min(T_dec, self.t_outer - 1)):
                Ct = self.C[t]
                if Ct is None:
                    continue
                postA = t >= T_A
                m_tmpl = traj[t]["market"][0].to(device)               # (market_dim,)
                static = self._static_sim[t].to(device)
                rows = [(r, si, wi) for r in range(n_states)
                        for si in s_idx for wi in w_idx]
                R = len(rows)
                market = m_tmpl.unsqueeze(0).expand(R, -1).clone()
                belief = torch.zeros(R, n_states, device=device, dtype=dtype)
                spot = torch.empty(R, device=device, dtype=dtype)
                wealth = torch.empty(R, device=device, dtype=dtype)
                for i, (r, si, wi) in enumerate(rows):
                    belief[i, r] = 1.0
                    spot[i] = float(s_grid[si])
                    wealth[i] = float(w_grid[wi])
                market[:, rlo:rhi] = belief
                if spot_col is not None:
                    market[:, spot_col] = spot
                F = spot.unsqueeze(1).expand(R, n_h).clone()            # forward = spot
                q_prev = torch.zeros(R, n_h, device=device, dtype=dtype)  # canonical flat start
                ttT = torch.full((R,), float(self.t_outer - 1 - t), device=device, dtype=dtype)
                static_b = static.expand(R, static.shape[-1])
                q_star, _, _ = self._decision_at(
                    Ct, market, F, q_prev, wealth, ttT, static_b, t)    # (R, n_h)
                qsA = q_star[:, A].cpu().numpy(); qsB = q_star[:, B].cpu().numpy()
                errs = []; matches = []
                for i, (r, si, wi) in enumerate(rows):
                    if postA:
                        o_qB = float(q_grid[pol_postA_qB[t - T_A, r, qcen, si, wi]])
                        e = abs(qsB[i] - o_qB)              # A is dead (forced 0 by is_live)
                        m = (round(qsB[i]) == round(o_qB))
                    else:
                        o_qA = float(q_grid[pol_preA_qA[t, r, qcen, qcen, si, wi]])
                        o_qB = float(q_grid[pol_preA_qB[t, r, qcen, qcen, si, wi]])
                        e = 0.5 * (abs(qsA[i] - o_qA) + abs(qsB[i] - o_qB))
                        m = (round(qsA[i]) == round(o_qA) and round(qsB[i]) == round(o_qB))
                    errs.append(e); matches.append(1.0 if m else 0.0)
                per_t_err[t] = float(np.mean(errs))
                per_t_match[t] = float(np.mean(matches))

        ts = sorted(per_t_err)
        if not ts:
            return {"oracle_match_skipped": True, "oracle_match_reason": "no_C"}
        depth = np.array([T_dec - 1 - t for t in ts], dtype=np.float64)   # 0 = nearest terminal
        err = np.array([per_t_err[t] for t in ts])
        match = np.array([per_t_match[t] for t in ts])
        # slope of mean|Δq| vs backward depth: >0 ⇒ error grows backward (compounding).
        slope = float(np.polyfit(depth, err, 1)[0]) if len(ts) > 1 else 0.0
        mean_err = float(err.mean()); mean_match = float(match.mean())
        # split match by region for the headline
        pre_match = float(np.mean([per_t_match[t] for t in ts if t < T_A])) if any(t < T_A for t in ts) else float("nan")
        post_match = float(np.mean([per_t_match[t] for t in ts if t >= T_A])) if any(t >= T_A for t in ts) else float("nan")
        logging.info(
            "Oracle action-match VERDICT: mean|Δq|=%.3f contracts, exact-match=%.1f%% "
            "(pre-A=%.1f%%, post-A=%.1f%%); error-vs-depth slope=%+.4f /step "
            "(>0 ⇒ compounding backward, ≈0 ⇒ depth-robust). Oracle V_0=%.4f.",
            mean_err, 100 * mean_match, 100 * pre_match, 100 * post_match, slope,
            float(d["V_0"]))
        return {
            "oracle_match_skipped": False,
            "oracle_match_mean_abs_dq": mean_err,
            "oracle_match_exact_frac": mean_match,
            "oracle_match_exact_frac_preA": pre_match,
            "oracle_match_exact_frac_postA": post_match,
            "oracle_match_err_vs_depth_slope": slope,
            "oracle_match_per_t_err": per_t_err,
            "oracle_match_per_t_match": per_t_match,
            "oracle_V0": float(d["V_0"]),
        }

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
             (time, inventory, action_gap, tail) — for regime-conditional
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
        # q per-hedge from position_limits, wealth band scaled to symlog c.
        q_prev = torch.empty(N_audit, n_h, dtype=torch.long, device=device)
        for j in range(n_h):
            q_prev[..., j] = torch.randint(
                int(self.bank_q_min[j]), int(self.bank_q_max[j]) + 1,
                (N_audit,), device=device)
        q_prev = q_prev.to(self.dtype)
        wealth_pre = (torch.rand(N_audit, device=device) - 0.5) \
            * (2.0 * self.bank_wealth_halfwidth)
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

        # FD eps scaled to the sampled wealth band (was hardcoded 0.5 against the
        # old ±1500): 1/3000 of the band reproduces 0.5 there exactly and
        # stays well inside the fit hull at deal scales.
        eps = self.bank_wealth_halfwidth / 3000.0
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

    # ----------------------------------------------- C-stack persistence (OOS)
    def _save_value_fn(self, path):
        """Serialise the fitted C-stack for out-of-sample reuse: per-t `TwinNetwork`
        `state_dict`s (weights AND the standardization buffers z_mean/z_std/y_mean/y_std,
        which are registered buffers) plus a header for load-time compatibility checks.

        Only the nets are persisted — everything else the sandwich needs (is_live, action
        grid, liability R/τ tables, baseline drift tables, StatePack layout) is rebuilt
        deterministically from the same deal JSON when the eval run constructs its solver.
        So an OOS run = the SAME config with a fresh `Random_Seed` + `Load_Value_Fn_Path`."""
        payload = {
            "header": {
                "format": 1,
                "deep_dim": int(self.statepack.deep_dim),
                "t_outer": int(self.t_outer),
                "t_min": int(self.t_min),
                "n_hedge": len(self.hedges),
                "hidden_sizes": list(self.hidden_sizes),
                "advantage_decomp": bool(self._advantage_decomp),
            },
            "C_state": [None if c is None
                        else {k: v.detach().cpu() for k, v in c.state_dict().items()}
                        for c in self.C],
        }
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        torch.save(payload, path)
        n_saved = sum(1 for c in self.C if c is not None)
        logging.info(
            "DifferentialSolver: SAVED C-stack (%d/%d nets, deep_dim=%d) → %s",
            n_saved, self.t_outer, self.statepack.deep_dim, path)

    def _load_value_fn(self, path):
        """Load a C-stack saved by `_save_value_fn` into `self.C` and SKIP all training.
        Fails LOUD if the artifact is from a different deal/config (deep_dim, t_outer,
        n_hedge, or advantage_decomp mismatch) — a stale artifact would silently corrupt the
        baseline (the net is only the residual `A`; `_C_full` adds the config's `B`)."""
        payload = torch.load(path, map_location=self.device)
        h = payload["header"]
        dd = int(self.statepack.deep_dim)
        mism = [f"{k}: artifact={h.get(k)} vs solver={v}" for k, v in
                (("deep_dim", dd), ("t_outer", self.t_outer), ("n_hedge", len(self.hedges)),
                 ("advantage_decomp", bool(self._advantage_decomp))) if h.get(k) != v]
        if mism:
            raise ValueError(
                f"value-fn artifact {path} is incompatible with this deal/config "
                f"({'; '.join(mism)}). Re-train the C-stack for this configuration.")
        n_loaded = 0
        for t, sd in enumerate(payload["C_state"]):
            if sd is None:
                self.C[t] = None
                continue
            net = construct_twin_network(dd, self.runtime, self.device)
            net.load_state_dict({k: v.to(self.device) for k, v in sd.items()})
            net.eval()
            self.C[t] = net
            n_loaded += 1
        logging.info(
            "DifferentialSolver: LOADED C-stack (%d/%d nets) ← %s — OOS eval (no training; "
            "sandwich runs on this run's fresh-seeded outer batch).",
            n_loaded, self.t_outer, path)
        return {"loaded_value_fn": True, "loaded_value_fn_path": path,
                "C_loaded_count": n_loaded, "seam_milestone": "oos_eval"}

    # --------------------------------------------------------------- solve
    def solve(self):
        """**Milestone 0**: cold-train C_T (= `C[t_outer-1]`) and exit. Validates the
        seam (`sample_exogenous` reads from the canonical buffer) and the StatePack
        layout before any backward-step / bootstrap / audit machinery lands.

        OOS-eval mode (`Load_Value_Fn_Path` set): load a pretrained C-stack, SKIP the
        terminal train + backward sweep, and run the V_0 read + L/π/U sandwich on this
        run's (fresh-seeded) outer batch. Training mode optionally `Save_Value_Fn_Path`s
        the fitted stack after the sweep.
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

        n_h = len(self.hedges)
        per_t_diag = {}
        grad_sanity = {}
        conv_summary = {}          # convergence assumption-check (training branch only)
        v0_over_bound = False      # V_0 utility-bound assumption-check

        if self._load_value_fn_path:
            # OOS-EVAL: load the pretrained C-stack, skip training entirely, and evaluate the
            # sandwich on THIS run's fresh-seeded outer batch (`traj` above). The deal machinery
            # (is_live, action grid, liability/baseline tables, StatePack) was already rebuilt
            # in __init__ from the same config, so the loaded residual nets compose correctly.
            diag = self._load_value_fn(self._load_value_fn_path)
            diag["t0_regime_occupancy"] = regime_t0_freq
        else:
            diag = self._train_C_terminal(traj)
            diag["t0_regime_occupancy"] = regime_t0_freq
            diag["seam_milestone"] = 0

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

            # --- CONVERGENCE assumption-check: did every C_t actually fit? The verdict-quality
            # gate. Flags (a) UNDER-FIT outliers — value loss ≫ the sweep median (an under-trained
            # step whose value/action estimate is unreliable), (b) ESCALATED steps (label-bias,
            # §14), and (c) budget-exhausted steps when a loss tol is set (ran out of SGD steps
            # before reaching tol). Threshold-free where possible (outlier vs the sweep itself). ---
            vls = {t: float(d["twin_value_loss"]) for t, d in per_t_diag.items()}
            if vls:
                vals = sorted(vls.values())
                med = vals[len(vals) // 2]
                vmax_t = max(vls, key=vls.get)
                budget_hit = (sorted(t for t, d in per_t_diag.items()
                                     if not d.get("train_stopped_early")
                                     and d.get("train_steps_used", 0) >= d.get("train_max_steps", 1))
                              if self.train_loss_tol > 0 else [])
                escalated = sorted(t for t, d in per_t_diag.items() if d.get("escalation_flag"))
                # UNDER-FIT = twin value loss above an absolute floor (0.1). All utility families
                # are O(1)-normalised by c, so the value target is O(1) and a converged residual
                # fit sits «0.1 (RMS «0.3 utility units); a t above the floor is unreliable. This
                # catches BOTH outlier steps and a UNIFORMLY under-fit sweep (where a "× median"
                # rule sees no outlier because everything is equally bad).
                FLOOR = 0.1
                underfit = sorted(t for t, v in vls.items() if v > FLOOR)
                conv_summary = {
                    "conv_val_loss_median": float(med),
                    "conv_val_loss_max": float(vls[vmax_t]), "conv_val_loss_max_t": int(vmax_t),
                    "conv_budget_exhausted_t": budget_hit, "conv_escalated_t": escalated,
                    "conv_underfit_t": underfit, "conv_n_fitted": len(vls),
                }
                (logging.warning if (underfit or escalated) else logging.info)(
                    "DifferentialSolver CONVERGENCE check: %d C_t fitted; val_loss median=%.4f, "
                    "max=%.4f @t=%d. Budget-exhausted t=%s; escalated (label-bias) t=%s; "
                    "UNDER-FIT (val>%.2f) t=%s. %s",
                    len(vls), med, vls[vmax_t], vmax_t, budget_hit or "[]", escalated or "[]",
                    FLOOR, ("%d steps" % len(underfit)) if len(underfit) > 8 else (underfit or "[]"),
                    "⚠ some C_t did NOT converge — value/action estimates at those t are unreliable "
                    "(the depth-compounding signature); raise B_exo / train steps, or check §14."
                    if (underfit or escalated) else "all C_t fits healthy (no outliers).")

            # Persist the fitted stack for out-of-sample reuse (before the sandwich, so the
            # nets are saved even if the optional U DP is heavy).
            if self._save_value_fn_path:
                self._save_value_fn(self._save_value_fn_path)

            # Grad sanity at three representative t values: end-of-sweep (t_min),
            # mid-sweep, and just-past-the-T_A-transition. Catches grad propagation
            # bugs that any single-t smoke misses.
            t_sample = sorted({self.t_min, self.t_outer - 2,
                                (self.t_outer - 2 + self.t_min) // 2})
            for t_s in t_sample:
                if self.C[t_s] is not None:
                    grad_sanity[t_s] = self._grad_sanity_check(t_s)

        # V_0 estimate via the decision operator at t_min: q*_t_min = argmax_q
        # C[t_min](post_state_t_min(q)). For the M1.5 path (t_min = T_A - 1) this is
        # the value at the first multi-contract decision step. For M3 (t_min = 0)
        # this is the true V_0.
        with torch.no_grad():
            market_tm = traj[self.t_min]["market"]
            tradables_tm = traj[self.t_min]["tradables"]
            B = market_tm.shape[0]
            zero_q = torch.zeros(B, n_h, device=self.device)
            # Initial wealth at t_min — MTM-invariant convention: w_0 = cash +
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

        # --- V_0 BOUND assumption-check: a symlog/CARA value-to-go is bounded by the utility of
        # reachable wealth. C is only TRAINED on |w| ≤ bank_wealth_halfwidth, so a |V_0| beyond
        # |u(±halfwidth)| means C extrapolates ABOVE achievable utility — over-optimism / depth
        # compounding (the verdict's V_0=3.63 > symlog bound ~2.77 was exactly this). nan compares
        # False, so a missing C_t never trips it. ---
        util_bound = float(_utility_wrap_signed(
            torch.tensor([self.bank_wealth_halfwidth], device=self.device, dtype=self.dtype),
            self.runtime).abs().item())
        v0_over_bound = bool(abs(v0_decision) > util_bound + 1.0e-6)
        if v0_over_bound:
            logging.warning(
                "DifferentialSolver V_0 BOUND check: |V_0|=%.4f EXCEEDS the utility bound %.4f "
                "(=|u(±%.3g)| at the bank wealth span) — the value function is EXTRAPOLATING above "
                "achievable utility (over-optimism / depth compounding); V_0 unreliable.",
                v0_decision, util_bound, self.bank_wealth_halfwidth)

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
            # Price-differential gate: per-t raw ∂Y_boot/∂F_h distinctness across live
            # contracts (recorded only where ≥2 are live). Surfaces as
            # M1_price_diff_gate_at_t. 0 ⇒ the broadcast bug; >0 ⇒ per-contract slopes.
            "price_diff_gate_at_t": dict(getattr(self, "_price_diff_gate", {})),
            # Spot-gradient FD gate: per-t FD-vs-autograd rel-err on ∂Y_boot/∂spot + the
            # realized-dL vs R_t·dS sign cross-check. Surfaces as M1_spot_grad_fd_gate_at_t.
            "spot_grad_fd_gate_at_t": dict(getattr(self, "_spot_grad_fd_gate", {})),
            # Label audit (opt-in): per-t snapshot of the bootstrap labels the net fits
            # (Y_boot / baseline_B / residual_y_target distributions + examples). Surfaces
            # as M1_label_audit_at_t. Empty unless Label_Audit_T_Steps is set.
            "label_audit_at_t": dict(getattr(self, "_label_audit", {})),
        }
        # Production validation: BSS sandwich lower bound L (realised $/oz of the
        # policy on a fresh MC set). Runs unconditionally; computes mean, p5/p95,
        # and the cost-per-oz that the dealer-margin ship criterion reads.
        floor = self._compute_dollar_floor(traj)
        learned_wealth = floor.pop("_terminal_wealth")     # (B,) — reused by the §7 verdict
        sweep_summary.update(floor)
        # §7 downside-protection VERDICT: learned vs unhedged vs best-static on the SAME
        # batch (mean / downside_sd / 5%-worst / 95%-best). The practical "did it hedge"
        # check — for the platinum deal there is no exact-DP oracle, so this IS the verdict.
        # Cheap (two extra no-grad rollouts + the static grid search); runs unconditionally.
        sweep_summary.update(self._compute_benchmark_comparison(traj, learned_wealth))
        # BSS sandwich step 1+2: penalty π's martingale-difference terms + the
        # dual-feasibility (zero-mean) HARD GATE (§4.1). Must be green before U is a
        # valid upper bound. Surfaces as M1_penalty_*; the per-t mean Δ_t is the §5
        # localisation hook. Offline yardstick — diagnostic only, never shipped.
        penalty = self._compute_martingale_penalty_check(traj)
        sweep_summary.update(penalty)
        # BSS sandwich step 3: penalized clairvoyant UPPER bound U + the gap U−L (utility
        # units). Reuses the SAME penalty machinery (so U's π is the gate-certified
        # martingale difference); π≡0 on the data-driven boundary block; U=min(naive,pen);
        # U≥L verified. Surfaces as M1_U_*. Skips itself (loud) if π is infeasible/unavailable.
        # OPT-IN (Run_Upper_Bound): the wealth-grid DP is O(P·G·K²·I), too costly to run on
        # every solve — the cheap penalty gate above always runs as the dual-feasibility check.
        if self._run_upper_bound:
            sweep_summary.update(self._compute_penalized_upper_bound(traj, penalty))
        else:
            logging.info(
                "DifferentialSolver U: Run_Upper_Bound=False — skipping the penalized "
                "clairvoyant upper bound (penalty zero-mean gate ran; enable for gap=U−L).")
        # Offline gate-4 verdict (opt-in): per-depth action-match vs the exact-DP oracle.
        if self._oracle_action_match_path:
            sweep_summary.update(self._compute_oracle_action_match(traj))
        sweep_summary.update(conv_summary)
        sweep_summary["v0_over_utility_bound"] = v0_over_bound

        # --- CONSOLIDATED ASSUMPTIONS readout: one line confirming the run is trustworthy. Every
        # check the downstream V_0 / sandwich / action-match rely on. A FAIL here means estimates
        # below it are suspect (see the specific warning above). ---
        _pz = sweep_summary.get("penalty_zero_mean_z")
        _pen_ok = (_pz is not None and _pz < self._penalty_zero_mean_z
                   and not sweep_summary.get("penalty_boundary_hit_cap", False))
        _conv_ok = not (conv_summary.get("conv_underfit_t") or conv_summary.get("conv_escalated_t"))
        _u_ok = sweep_summary.get("U_mean_ge_L")
        _all_ok = (_conv_ok if conv_summary else True) and (_pen_ok if _pz is not None else True) \
            and not v0_over_bound and (_u_ok if "U_mean_ge_L" in sweep_summary else True)
        (logging.info if _all_ok else logging.warning)(
            "DifferentialSolver ASSUMPTIONS CHECK: C_t-converged=%s (underfit t=%s, escalated=%s) | "
            "penalty dual-feasible=%s (z=%s, thr=%.1f) | V_0 within utility bound=%s | U≥L=%s. %s",
            ("OK" if _conv_ok else "FAIL") if conv_summary else "n/a (loaded)",
            conv_summary.get("conv_underfit_t", []), conv_summary.get("conv_escalated_t", []),
            ("OK" if _pen_ok else "FAIL") if _pz is not None else "n/a",
            None if _pz is None else round(_pz, 2), self._penalty_zero_mean_z,
            "OK" if not v0_over_bound else "FAIL",
            ("OK" if _u_ok else "FAIL") if "U_mean_ge_L" in sweep_summary else "n/a (Run_Upper_Bound off)",
            "✓ all key assumptions hold." if _all_ok
            else "⚠ ASSUMPTION(S) FAILED — downstream V_0 / action-match estimates may be unreliable.")

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
        values_stacked[self.t_outer - 1] = v0_decision
        return SolverResult(
            solver_name="DifferentialSolver",
            actions=actions_stacked.detach().cpu(),
            values=values_stacked.detach().cpu(),
            value_fn_artifacts={},
            diagnostics={
                # Headline V_0 = the backward-sweep decision value at t_min (the actual
                # optimal-policy value). Dispatcher convention: every solver exposes its
                # headline under "V_0" (read by `_result_v0` → `v0_mean`).
                "V_0": v0_decision,
                "n_star_0": q_opt_mean if q_opt_mean is not None else torch.zeros(n_h).tolist(),
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
