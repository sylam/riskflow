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

import torch

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

    def __init__(self, statepack, mlp_hidden, device):
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

    def fit(self, z, v_target, train_steps, lr):
        """Fit OLS β (per-column ridge — robust to basis columns of disparate norm) then
        the MLP on the OLS residual. Records the training hull + standardization stats.
        Returns the OLS R²."""
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
        phi = self._basis(zn)                                             # (N, P)
        gram = phi.transpose(-1, -2) @ phi
        # Per-column ridge ∝ each column's own Gram diagonal — regularizes every basis
        # column equally in relative terms regardless of its norm.
        ridge = torch.diag(1.0e-6 * gram.diagonal().clamp_min(1.0e-12))
        self.beta = torch.linalg.solve(gram + ridge, phi.transpose(-1, -2) @ v_target)
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

        tradables_sim, static_sim, t_outer = _bundle_sim_views(bundle)
        statepack = None        # built lazily — market_dim is known only once inner MC runs
        action_grid = build_action_grid(
            runtime, solver_cfg["training_action_grid_levels_per_axis"], device)
        n_actions = action_grid.shape[0]
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
                pnl_step = float((action_grid.abs().amax(dim=0) * contract_size
                                  * dstep.abs().amax(dim=(1, 2))).sum())
                g_halfwidth = max(4.0 * pnl_step, c)

                # Fit rows: sample per-outer-path pre-decision inventory + cash. At t=0 the
                # true decision state is flat (n_prev=0, G_0=0) — evaluate it exactly.
                if t == 0:
                    n_prev = torch.zeros(b_outer, len(hedges), device=device)
                    g_t = torch.zeros(b_outer, device=device)
                else:
                    n_prev = action_grid[torch.randint(0, n_actions, (b_outer,), device=device)]
                    g_t = (torch.rand(b_outer, device=device) - 0.5) * (2.0 * g_halfwidth)
                zero_b = torch.zeros(b_outer, device=device)

                is_terminal_next = (t + 1 == t_outer - 1)
                vfa_next = value_fns.get(t + 1)        # None at terminal-next, else (head_A, head_B)

                # Cross-fit split: half the inner paths select the argmax action, the
                # disjoint other half scores it (`_split_inner_axis` slices by shape).
                b_inner = L_T.shape[1]
                half = b_inner // 2
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
                        flat = v.detach().reshape(-1)
                        terminal_u_pool.append(flat[::max(1, flat.numel() // 200_000)])
                    else:
                        # Asymmetric Double-V (variant 2): a single V̂_{t+1} head scores
                        # the whole batch — the caller passes the selecting head on the
                        # argmax pass and the *other*, disjoint head on the evaluation
                        # pass, so the fitted target is never maxed over its own noise.
                        # V̂ standardizes its input internally — pass the raw deep state.
                        v = vhead(deep_next)
                        vhat_query_abs_max = torch.maximum(vhat_query_abs_max, v.abs().max())
                        # Range projection: V̂ provably lies in [u_min, u_max]; clamp before
                        # the max_a so it cannot exploit off-hull extrapolation upward.
                        if project:
                            v = v.clamp(u_min_proj, u_max_proj)
                        # Hull/query diagnostic: per-component min/max of the deep state
                        # actually queried against V̂_{t+1} during this step's search.
                        flat_dn = deep_next.reshape(-1, deep_next.shape[-1])
                        query_lo = torch.minimum(query_lo, flat_dn.amin(dim=0))
                        query_hi = torch.maximum(query_hi, flat_dn.amax(dim=0))
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
                v_t_AB = objective_fn(n_star_AB.unsqueeze(0), inner_B, head_B).squeeze(0)
                n_star_BA, v_sel_BA = search_action_grid(
                    action_grid, lambda a: objective_fn(a.unsqueeze(1), inner_B, head_B),
                    chunk_size=chunk_size, total_abs_limit=total_abs_limit)
                v_t_BA = objective_fn(n_star_BA.unsqueeze(0), inner_A, head_A).squeeze(0)
                # Report the mean of the two cross-fit estimates.
                n_star = 0.5 * (n_star_AB + n_star_BA)
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
                vfa_A = ValueFunctionApproximator(statepack, vf_cfg["mlp_hidden"], device)
                vfa_B = ValueFunctionApproximator(statepack, vf_cfg["mlp_hidden"], device)
                r2_A = vfa_A.fit(state_t, v_t_AB,
                                 vf_cfg["mlp_train_steps_per_solve"], vf_cfg["mlp_adam_lr"])
                r2_B = vfa_B.fit(state_t, v_t_BA,
                                 vf_cfg["mlp_train_steps_per_solve"], vf_cfg["mlp_adam_lr"])
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
                # beta index of the (standardized) cash term: basis = [ones(1), z(deep_dim),
                # ...], and cash sits at z-index n_hedge (after the n_hedge positions).
                diag_by_t[t]["dVhat_dG_normalized"] = float(vfa_A.beta[1 + statepack.n_hedge])
                diag_by_t[t]["ols_r2"] = min(r2_A, r2_B)
                diag_by_t[t]["resid_corr"] = resid_corr
                diag_by_t[t]["resid_corr_worst"] = (dim_names[worst], resid_corr[worst])
                d = diag_by_t[t]
                logging.info(
                    'LsmDpSolver t=%d: |V_t|max=%.3g V_t.mean=%.3g asym_gap=%.3g '
                    'head_disagree=%.3g Vhat_query|max=%.3g resid_worst=%s:%.3f R2=%.4f',
                    t, d["v_t_abs_max"], d["v_t_mean"],
                    0.5 * (d["asym_gap_AB_mean"] + d["asym_gap_BA_mean"]),
                    d["head_disagree_mean"], d["vhat_query_abs_max"],
                    dim_names[worst], resid_corr[worst], min(r2_A, r2_B))
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


_SOLVERS: Dict[str, Callable] = {
    "mpcsolver": MpcSolver,
    "lsmdpsolver": LsmDpSolver,
    "hindsightdpsolver": HindsightDpSolver,
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
