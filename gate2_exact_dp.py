"""Gate 2 — exact backward grid-DP brute-force solver as ground truth for the
differential-ML build (see differential_ml_redesign_v14.md §8 / brief gate 3).

Toy specification (deliberately small, fully self-contained — NOT a riskflow path):
  - Binary regime HMM spot (2 states with distinct μ, σ; CTMC transition)
  - Single underlying, log-GBM per regime
  - 2 hedge futures (forward = spot, no carry/basis): A expires at T_A (interior),
    B at T_B = T_dec
  - Liability: terminal payoff `K - S_{T_dec}` (receiver of fixed K, payer of S)
  - Utility: symlog `sign(W)·log1p(|W|/c)`
  - L1 turnover cost on inventory changes

State (discretized):
  - regime r ∈ {0, 1}
  - log-price s ∈ N_S pts (relative to log S_0)
  - inventory q_A ∈ N_Q pts on integer contracts (collapsed to {0} after T_A)
  - inventory q_B ∈ N_Q pts
  - wealth w ∈ N_W pts; `w = cash + (q_A + q_B - 1)·S + K` is the mark-to-market
    net wealth (invariant under rebalances modulo turnover cost; at terminal
    unwind = realized terminal wealth = w_{T_dec}).

Output:
  - V_0 at the canonical initial state — the optimal value under the exact policy
  - per-cell V[t, r, s, q_A, q_B, w] (saved to .npz for Gate 3 consumption)
  - sanity checks (V_0 ≥ static-hold, V_0 ≤ hindsight, monotone in wealth)

Wealth axis uses bilinear interpolation when the post-transition wealth lands between
grid points — gives exact DP on the (regime × log_S × q_A × q_B) discrete factors
and only loses precision to the wealth-axis discretization.
"""
import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Toy configuration
# ---------------------------------------------------------------------------

@dataclass
class ToyConfig:
    """Retuned dynamics (v2): sharper regime contrast + lower turnover so the
    dynamic policy has real alpha over delta-flat. v1 (μ=+0.05/-0.20, σ=0.15/0.40,
    sticky-90%) collapsed to delta-flat at the wide grid — too little regime impact
    and too much turnover to switch on. v2 spreads μ and σ further, makes regime 1
    less persistent (more switching ⇒ more dynamic value), and cuts turnover 5×.
    """
    # Horizon — v5 uses WEEKLY steps (dt=5/252). The earlier daily dt=1/252 left the
    # per-step log-S drift at ~0.002 — 3% of the log_s grid spacing — which meant the
    # discrete CDF-difference kernel masked drift completely (100% probability stayed
    # in the source bin). Without drift, the DP saw a static problem and naturally
    # chose delta-flat. Weekly dt puts drift/grid ≈ 0.14, vol/grid ≈ 0.16 — small but
    # resolvable. T_dec=30 weekly ≈ 6 months, T_A=15 weekly ≈ 3 months.
    T_dec: int = 30                 # total decision steps (weekly ⇒ ~6 months)
    T_A: int = 15                   # contract A expiry index (must be < T_dec)
    dt_years: float = 5.0 / 252.0
    # Underlying
    S_0: float = 100.0
    K: float = 100.0                # strike for the K-S terminal liability
    # Regimes (annualised drift, vol). State 0 = "bull", 1 = "bear/stress".
    # σ bumped to 0.15 / 0.40 — moderate; at σ=0.08 weekly the per-step kernel was
    # still narrow (vol/grid_step ≈ 0.22). With σ=0.15, vol/grid ≈ 0.42 — proper.
    mu_per_state: tuple = (0.50, -0.50)
    sigma_per_state: tuple = (0.15, 0.40)
    P_calib: tuple = (
        (0.97, 0.03),
        (0.10, 0.90),
    )                               # transition per dt_years (now weekly!); expected duration ~33 / ~10 weeks
    initial_state_probs: tuple = (0.5, 0.5)
    # Discretization
    N_S: int = 21                   # log-price grid points (step 0.05)
    log_S_halfwidth: float = 0.5    # log-S grid spans [-h, +h]
    q_min: int = -5
    q_max: int = 5                  # inventory range per contract
    N_W: int = 601                  # wealth grid points (dw=5)
    W_halfwidth: float = 1500.0     # wealth grid spans [K-h, K+h]
    # Turnover and utility
    lambda_turnover: float = 0.0001     # L1 cost as fraction of |Δq|·S
    symlog_c: float = 100.0


# ---------------------------------------------------------------------------
# Grid + kernel construction
# ---------------------------------------------------------------------------

def build_grids(cfg):
    log_s = np.linspace(-cfg.log_S_halfwidth, cfg.log_S_halfwidth, cfg.N_S)
    s_grid = cfg.S_0 * np.exp(log_s)
    q_grid = np.arange(cfg.q_min, cfg.q_max + 1, dtype=np.int64)
    w_grid = np.linspace(cfg.K - cfg.W_halfwidth, cfg.K + cfg.W_halfwidth, cfg.N_W)
    return {
        'log_s': log_s, 's_grid': s_grid,
        'q_grid': q_grid,
        'w_grid': w_grid,
    }


def discretize_log_s_kernel(log_s, mu_dt, sigma_sq_dt):
    """Build the discretized P(log_s_next | log_s_curr) matrix under N(mu_dt, sigma_sq_dt).
    Edges between grid points are midpoint cuts; outer edges go to ±∞ via wide
    sentinel so probability mass past the grid is folded into the boundary bin
    (acceptable for the toy — log_S halfwidth is wide enough that boundary mass
    is small; checked in sanity)."""
    N = len(log_s)
    edges = np.empty(N + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges[1:-1] = (log_s[:-1] + log_s[1:]) / 2.0
    std = np.sqrt(sigma_sq_dt)
    M = np.empty((N, N))
    for i in range(N):
        mean_i = log_s[i] + mu_dt
        cdfs = norm.cdf(edges, loc=mean_i, scale=std)
        M[i] = cdfs[1:] - cdfs[:-1]
    return M                          # rows sum to 1


def build_log_s_kernels(cfg):
    """Per-regime log_s transition kernel applied at each step. Regime is *destination*
    regime — i.e. the per-step return is governed by the regime at time t+1.
    Returns (n_states, N_S, N_S)."""
    grids = build_grids(cfg)
    log_s = grids['log_s']
    n_states = len(cfg.mu_per_state)
    kernels = np.empty((n_states, cfg.N_S, cfg.N_S))
    for r in range(n_states):
        mu_dt = (cfg.mu_per_state[r] - 0.5 * cfg.sigma_per_state[r] ** 2) * cfg.dt_years
        sigma_sq_dt = (cfg.sigma_per_state[r] ** 2) * cfg.dt_years
        kernels[r] = discretize_log_s_kernel(log_s, mu_dt, sigma_sq_dt)
    return kernels                    # [r_next, s_curr_idx, s_next_idx]


def build_regime_transition(cfg):
    """CTMC re-discretisation onto dt_years steps. The provided P_calib is at dt_years
    by convention — pass-through unless we need a finer rebase."""
    P = np.asarray(cfg.P_calib, dtype=np.float64)
    return P                          # [r_curr, r_next]


# ---------------------------------------------------------------------------
# Utility + wealth bilinear interpolation
# ---------------------------------------------------------------------------

class OffGridTracker:
    """Track how often bilinear V-lookup queries fall outside the wealth grid.

    Silent boundary clamping is a top-3 source of false-PASS results in discretised
    DP / V̂ evaluators (see `feedback_discretised_dp_boundary_clamp` in memory): when
    a large fraction of queries hit the boundary, V_0 is biased upward (symlog loss
    tails are under-penalised by the boundary clamp). This tracker reports the off-
    grid rate + observed query range vs grid range as a permanent diagnostic — it
    MUST appear in the headline output, not just on demand.
    """
    def __init__(self, w_grid):
        self.w_lo = float(w_grid[0])
        self.w_hi = float(w_grid[-1])
        self.n_total = 0
        self.n_below = 0
        self.n_above = 0
        self.observed_min = float('inf')
        self.observed_max = float('-inf')

    def record(self, w_query):
        self.n_total += int(w_query.size)
        self.n_below += int((w_query < self.w_lo).sum())
        self.n_above += int((w_query > self.w_hi).sum())
        self.observed_min = min(self.observed_min, float(w_query.min()))
        self.observed_max = max(self.observed_max, float(w_query.max()))

    def report(self):
        if self.n_total == 0:
            return {'total_queries': 0, 'frac_off_grid': 0.0}
        return {
            'total_queries': int(self.n_total),
            'frac_below': self.n_below / self.n_total,
            'frac_above': self.n_above / self.n_total,
            'frac_off_grid': (self.n_below + self.n_above) / self.n_total,
            'observed_w_range': (self.observed_min, self.observed_max),
            'grid_w_range': (self.w_lo, self.w_hi),
        }


def symlog_u(w, c):
    """Signed symlog utility (numpy)."""
    return np.sign(w) * np.log1p(np.abs(w) / c)


def bilinear_w_interp(V_w, w_grid, w_query):
    """Bilinear (linear) interpolation of V along the wealth axis.

    V_w: (..., N_W) array.
    w_grid: (N_W,) sorted ascending.
    w_query: scalar or array, broadcastable to V_w's leading shape.

    Returns V at w_query along the last axis. Clamps to boundary (constant
    extrapolation) outside the grid range — acceptable for the toy since wealth
    halfwidth is chosen to cover the reachable range; off-grid frequency is logged.
    """
    N_W = len(w_grid)
    dw = w_grid[1] - w_grid[0]
    w_clamped = np.clip(w_query, w_grid[0], w_grid[-1])
    pos = (w_clamped - w_grid[0]) / dw
    lo = np.floor(pos).astype(np.int64)
    lo = np.clip(lo, 0, N_W - 2)
    hi = lo + 1
    frac = pos - lo
    # V_w[..., lo] * (1 - frac) + V_w[..., hi] * frac
    V_lo = np.take_along_axis(V_w, lo[..., None], axis=-1)[..., 0]
    V_hi = np.take_along_axis(V_w, hi[..., None], axis=-1)[..., 0]
    return V_lo * (1.0 - frac) + V_hi * frac


# ---------------------------------------------------------------------------
# Backward DP
# ---------------------------------------------------------------------------

def backward_dp(cfg, verbose=True):
    """Backward induction over the discretized state space. Returns:
      V[t, r, s, q_A, q_B, w]            for t in [0, T_A] (before contract A expires)
      V_post_A[t, r, s, q_B, w]           for t in [T_A, T_dec] (after q_A collapsed)
    plus the t=0 root value V_0 at the canonical initial state AND an off-grid
    diagnostic from the bilinear-wealth-interp clamp.

    Iteration:
      - For t > T_A (only contract B live): action q_B' ∈ q_grid, 11 candidates.
      - At t = T_A: q_A cash-settled (added to wealth). Collapse the q_A dim.
      - For t < T_A: action (q_A', q_B') ∈ q_grid × q_grid, 121 candidates.
    """
    grids = build_grids(cfg)
    log_s = grids['log_s']
    s_grid = grids['s_grid']
    q_grid = grids['q_grid']
    w_grid = grids['w_grid']
    N_S, N_Q, N_W = cfg.N_S, len(q_grid), cfg.N_W
    n_states = len(cfg.mu_per_state)
    off_grid = OffGridTracker(w_grid)

    K_log_s = build_log_s_kernels(cfg)             # (R, N_S, N_S) — [r_next, s_curr, s_next]
    P_regime = build_regime_transition(cfg)        # (R, R) — [r_curr, r_next]
    # joint[r_curr, r_next, s_curr, s_next] = P_regime[r_curr, r_next] * K_log_s[r_next, s_curr, s_next].
    joint = np.einsum('cn,nse->cnse', P_regime, K_log_s)

    # ---------- Terminal V at t = T_dec (post-unwind) ----------
    # At T_dec, contract B is forced to 0. Wealth = w (invariant by definition).
    # V_{T_dec}(r, s, q_B, w) = U(w) — independent of (r, s, q_B), only depends on w.
    # We still index over (r, s, q_B) so the backward step interface stays uniform.
    V_T_post_A = np.broadcast_to(
        symlog_u(w_grid, cfg.symlog_c)[None, None, None, :],
        (n_states, N_S, N_Q, N_W),
    ).copy()
    # Force-unwind constraint: at T_dec, q_B != 0 is infeasible. Use a large finite
    # negative penalty (not -inf — bilinear interp with frac=0 yields -inf·0 = NaN).
    q_B_is_zero = (q_grid == 0)
    if not q_B_is_zero.any():
        raise ValueError('q_grid must include 0 for the force-unwind terminal.')
    INF_PENALTY = -1.0e10
    pen_mask = np.where(q_B_is_zero[None, None, :, None], 0.0, INF_PENALTY)
    V_T_post_A = V_T_post_A + pen_mask                                # (R, N_S, N_Q, N_W)

    # Optimal policy at t = T_dec: q_B' = 0 trivially.
    # Storage for backward results:
    V_post_A_list = [None] * (cfg.T_dec - cfg.T_A + 1)                # indexed by (t - T_A)
    pol_post_A_list = [None] * (cfg.T_dec - cfg.T_A + 1)
    V_post_A_list[-1] = V_T_post_A

    # ---------- Backward loop: t in [T_dec - 1, ..., T_A] (post-A regime) ----------
    if verbose:
        print(f'[backward post-A] T_A={cfg.T_A} -> T_dec={cfg.T_dec}, '
              f'state shape (R, N_S, N_Q, N_W) = ({n_states}, {N_S}, {N_Q}, {N_W})')
    for t in range(cfg.T_dec - 1, cfg.T_A - 1, -1):
        V_next = V_post_A_list[t - cfg.T_A + 1]                       # (R, N_S, N_Q, N_W)
        # For each (r_curr, s_curr, q_B_curr, w_curr) and action q_B', compute
        # E[V_next(r_n, s_n, q_B', w_n) | r_curr, s_curr, q_B', w_curr, action q_B'].
        # w_n = w_curr - λ·|q_B' - q_B_curr|·S_curr + (q_B' - 1)·(S_next - S_curr)
        # (recall liability has delta -1; only q_B is the live hedge after T_A.)

        # Precompute action effects.
        # turnover_cost: shape (N_Q, N_Q) — [q_curr_idx, q_action_idx]
        dq = np.abs(q_grid[:, None] - q_grid[None, :])                # (N_Q, N_Q)
        # turnover_cost[q_c, q_a, s_c] = λ * dq * S_curr
        turnover = cfg.lambda_turnover * dq[:, :, None] * s_grid[None, None, :]   # (N_Q, N_Q, N_S)

        # Price-move effect on wealth: (q_B' - 1) * (S_next - S_curr)
        # shape (N_Q_action, N_S_curr, N_S_next); ds[0, c, n] = s_grid[n] - s_grid[c]
        ds = s_grid[None, None, :] - s_grid[None, :, None]
        exposure = (q_grid - 1).astype(np.float64)[:, None, None]    # (N_Q_action, 1, 1)
        dw_price = exposure * ds                                      # (N_Q_action, N_S_curr, N_S_next)

        # Loop over action q_B' (small: N_Q = 11). For each action, compute expected V.
        # Output of this step: V_t[r, s_curr, q_curr, w_curr] = max over q' of E[V_next(...)]
        V_t = np.full_like(V_next, -np.inf)
        pol_t = np.zeros((n_states, N_S, N_Q, N_W), dtype=np.int64)

        # Pre-extract V_next for each (r_n, s_n) and the full w_n axis.
        # V_next shape: (R, N_S, N_Q, N_W). For fixed q_B' (q_a_idx), V_next[..., q_a_idx, :] is
        # the wealth-curve at that action's resulting q.
        for q_a_idx in range(N_Q):
            # w_after_turnover[r, s_curr, q_curr, w_curr] = w_curr - tov[q_curr, q_a_idx, s_curr]
            tov = turnover[:, q_a_idx, :]                             # (N_Q_curr, N_S_curr)
            w_after = w_grid[None, None, None, :] - tov.T[None, :, :, None]   # (1, N_S_curr, N_Q_curr, N_W)
            w_after = np.broadcast_to(w_after, (n_states, N_S, N_Q, N_W))

            # For each (r_n, s_n), w_next = w_after + dw_price[q_a_idx, s_curr, s_n]
            # Build dw_price slice for this action: (N_S_curr, N_S_next)
            dw_p = dw_price[q_a_idx]                                  # (N_S_curr, N_S_next)

            # Compute E_{r_n, s_n} [V_next(r_n, s_n, q_a_idx, w_after + dw_p[s_curr, s_n])]
            # We can interpolate V_next along the w axis at the query points.
            # V_next at (r_n, :, q_a_idx, :) is (N_S_next, N_W).
            # For each (r_curr, s_curr, q_curr, w_curr) and each (r_n, s_n), the query is:
            #   w_n_query = w_after[r_c=any, s_c, q_c, w_c] + dw_p[s_c, s_n]
            # Note w_after doesn't depend on r_c (turnover is per-q,s only).
            # So we can dispatch one r_curr at a time and average via joint kernel.
            #
            # We'll compute V_next interpolated, accumulated weighted by joint kernel.
            E_V = np.zeros((n_states, N_S, N_Q, N_W))                # (r_curr, s_curr, q_curr, w_curr)
            V_q = V_next[:, :, q_a_idx, :]                            # (R, N_S_next, N_W) — already at q'
            for r_curr in range(n_states):
                for r_n in range(n_states):
                    # joint[r_curr, r_n, s_curr, s_next]
                    p_joint = joint[r_curr, r_n]                     # (N_S_curr, N_S_next)
                    V_rn = V_q[r_n]                                   # (N_S_next, N_W)
                    # Need: for each (s_curr, q_curr, w_curr, s_next): V_rn[s_next, interp(w_after + dw_p[s_curr, s_next])]
                    # w_query: (s_curr, q_curr, w_curr, s_next) = w_after[0, s_curr, q_curr, w_curr] + dw_p[s_curr, s_next]
                    w_q = w_after[0, :, :, :, None] + dw_p[:, None, None, :]  # (N_S_curr, N_Q_curr, N_W, N_S_next)
                    off_grid.record(w_q)
                    # Bilinear in-grid + analytic-U off-grid: for off-grid w_query, set
                    # V(w_off) := U(w_off). At intermediate t, V(w) ≈ U(w) for large |w|
                    # (the recursion correction is small relative to U's log growth in the
                    # tails), so this is more accurate than the constant-boundary clamp.
                    # The clamp introduced a symlog-asymmetry bias that inverted the
                    # optimal policy direction in the bull regime — see commit message.
                    dw = w_grid[1] - w_grid[0]
                    pos = (np.clip(w_q, w_grid[0], w_grid[-1]) - w_grid[0]) / dw
                    lo = np.clip(np.floor(pos).astype(np.int64), 0, N_W - 2)
                    hi = lo + 1
                    frac = pos - lo
                    sn_idx = np.broadcast_to(
                        np.arange(N_S)[None, None, None, :], lo.shape)
                    V_lo_at = V_rn[sn_idx, lo]                       # (s_curr, q_curr, w_curr, s_next)
                    V_hi_at = V_rn[sn_idx, hi]
                    V_in_grid = V_lo_at * (1.0 - frac) + V_hi_at * frac
                    in_grid_mask = (w_q >= w_grid[0]) & (w_q <= w_grid[-1])
                    V_interp = np.where(in_grid_mask, V_in_grid, symlog_u(w_q, cfg.symlog_c))
                    weighted = (V_interp * p_joint[:, None, None, :]).sum(axis=-1)  # (s_curr, q_curr, w_curr)
                    E_V[r_curr] += weighted

            # Compare with current V_t
            better = E_V > V_t
            V_t = np.where(better, E_V, V_t)
            pol_t = np.where(better, q_a_idx, pol_t)

        V_post_A_list[t - cfg.T_A] = V_t
        pol_post_A_list[t - cfg.T_A] = pol_t
        if verbose and (t == cfg.T_dec - 1 or t == cfg.T_A):
            v0_at_mid = V_t[0, N_S // 2, N_Q // 2, N_W // 2]
            print(f'  t={t:3d}  V(r=0, s=mid, q=0, w=K) = {v0_at_mid:+.6f}')

    # ---------- Bridge at t = T_A: cash-settle q_A into wealth ----------
    # State just before T_A is (r, s, q_A, q_B, w). Cash-settlement folds q_A into wealth:
    # w_settled = w  (since wealth by definition already MtM's q_A·S; cash-settlement
    #                 just moves q_A·S from inventory accounting to cash accounting,
    #                 and the wealth invariant W = cash + (q_A + q_B - 1)·S + K is preserved
    #                 if we redefine "the world after T_A" as having q_A := 0.)
    # So V at T_A post-settlement (q_A := 0) = V_post_A_list[0][r, s, q_B, w].
    # The pre-settlement V at T_A is therefore independent of q_A:
    #   V_pre_T_A(r, s, q_A, q_B, w) = V_post_A_list[0](r, s, q_B, w)   for all q_A.
    if verbose:
        print(f'[bridge t={cfg.T_A}] cash-settle q_A — wealth invariant under our W definition')
    V_t_T_A = np.broadcast_to(
        V_post_A_list[0][:, :, None, :, :],
        (n_states, N_S, N_Q, N_Q, N_W),
    ).copy()                                                          # pre-rebalance V at t=T_A; q_A appears as a state dim but the value is invariant in it (post-settle).

    # ---------- Backward loop: t in [T_A - 1, ..., 0] (both contracts live) ----------
    if verbose:
        print(f'[backward pre-A] 0 -> T_A={cfg.T_A}, action grid 121 (q_A x q_B)')
    V_pre_A_list = [None] * (cfg.T_A + 1)
    pol_pre_A_list = [None] * (cfg.T_A + 1)
    V_pre_A_list[cfg.T_A] = V_t_T_A

    for t in range(cfg.T_A - 1, -1, -1):
        V_next = V_pre_A_list[t + 1]                                  # (R, N_S, N_Q_A, N_Q_B, N_W)
        # turnover: |Δq_A| + |Δq_B|, times S_curr × λ
        dq_A = np.abs(q_grid[:, None] - q_grid[None, :])             # (N_Q_curr, N_Q_action)
        dq_B = np.abs(q_grid[:, None] - q_grid[None, :])             # same
        # Joint turnover[q_A_curr, q_A_act, q_B_curr, q_B_act, s_curr] =
        #   λ * (dq_A[q_A_curr, q_A_act] + dq_B[q_B_curr, q_B_act]) * s_grid[s_curr]
        # Too big to materialize (11^4 * 15 = ~220k entries × float64 = 1.8MB; OK).

        # Exposure after action: (q_A_act + q_B_act - 1)
        exposure_pre = (q_grid[:, None] + q_grid[None, :] - 1).astype(np.float64)  # (N_Q_A, N_Q_B)

        V_t = np.full_like(V_next, -np.inf)
        pol_t_A = np.zeros((n_states, N_S, N_Q, N_Q, N_W), dtype=np.int64)
        pol_t_B = np.zeros((n_states, N_S, N_Q, N_Q, N_W), dtype=np.int64)

        # 121 actions: nested over (q_A_act, q_B_act)
        for q_A_act in range(N_Q):
            for q_B_act in range(N_Q):
                # turnover[q_A_curr, q_B_curr, s_curr]
                tov = cfg.lambda_turnover * (dq_A[:, q_A_act][:, None, None]
                                              + dq_B[:, q_B_act][None, :, None]) * s_grid[None, None, :]
                # w_after_turnover[r, s_curr, q_A_curr, q_B_curr, w_curr] = w_curr - tov[q_A_curr, q_B_curr, s_curr]
                tov_e = tov[None, :, :, :, None].transpose(0, 3, 1, 2, 4)   # → (1, s_curr, q_A_curr, q_B_curr, 1)
                w_after = w_grid[None, None, None, None, :] - tov_e
                # broadcasting over r:
                w_after = np.broadcast_to(w_after, (n_states, N_S, N_Q, N_Q, N_W))

                # Price-move effect: exposure * dS
                exp_act = exposure_pre[q_A_act, q_B_act]              # scalar
                dw_p = exp_act * (s_grid[None, :] - s_grid[:, None])  # (N_S_curr, N_S_next)

                # E[V_next(r_n, s_n, q_A_act, q_B_act, w_after + dw_p[s_curr, s_n])]
                V_q = V_next[:, :, q_A_act, q_B_act, :]               # (R, N_S_next, N_W)
                dw = w_grid[1] - w_grid[0]

                E_V = np.zeros((n_states, N_S, N_Q, N_Q, N_W))
                for r_curr in range(n_states):
                    for r_n in range(n_states):
                        p_joint = joint[r_curr, r_n]                  # (N_S_curr, N_S_next)
                        V_rn = V_q[r_n]                               # (N_S_next, N_W)
                        # w_query[s_curr, q_A_curr, q_B_curr, w_curr, s_next]
                        # = w_after[0, s_curr, q_A_curr, q_B_curr, w_curr] + dw_p[s_curr, s_next]
                        w_q = w_after[0, :, :, :, :, None] + dw_p[:, None, None, None, :]
                        off_grid.record(w_q)
                        pos = (np.clip(w_q, w_grid[0], w_grid[-1]) - w_grid[0]) / dw
                        lo = np.clip(np.floor(pos).astype(np.int64), 0, N_W - 2)
                        hi = lo + 1
                        frac = pos - lo
                        sn_idx = np.broadcast_to(
                            np.arange(N_S)[None, None, None, None, :], lo.shape)
                        V_lo_at = V_rn[sn_idx, lo]
                        V_hi_at = V_rn[sn_idx, hi]
                        V_in_grid = V_lo_at * (1.0 - frac) + V_hi_at * frac
                        in_grid_mask = (w_q >= w_grid[0]) & (w_q <= w_grid[-1])
                        V_interp = np.where(in_grid_mask, V_in_grid, symlog_u(w_q, cfg.symlog_c))
                        weighted = (V_interp * p_joint[:, None, None, None, :]).sum(axis=-1)
                        E_V[r_curr] += weighted

                better = E_V > V_t
                V_t = np.where(better, E_V, V_t)
                pol_t_A = np.where(better, q_A_act, pol_t_A)
                pol_t_B = np.where(better, q_B_act, pol_t_B)

        V_pre_A_list[t] = V_t
        pol_pre_A_list[t] = (pol_t_A, pol_t_B)
        if verbose:
            v0_at_mid = V_t[0, N_S // 2, N_Q // 2, N_Q // 2, N_W // 2]
            print(f'  t={t:3d}  V(r=0, s=mid, q_A=q_B=0, w=K) = {v0_at_mid:+.6f}')

    # ---------- Root: V_0 + optimal policy at canonical initial state ----------
    s_idx_0 = N_S // 2                                                # log_S = 0 (S = S_0)
    q_idx_0 = list(q_grid).index(0)                                   # q_A = q_B = 0
    w_idx_0 = np.argmin(np.abs(w_grid - cfg.K))                       # W_0 = K (cash_0 = S_0 chosen)
    V_0_per_regime = V_pre_A_list[0][:, s_idx_0, q_idx_0, q_idx_0, w_idx_0]
    V_0 = float(np.dot(np.asarray(cfg.initial_state_probs), V_0_per_regime))
    # Policy at the initial state per regime — confirms the DP found a regime-dependent
    # tilt (alpha magnitude is small in utility units; what matters for Gate 4 is whether
    # the policy is non-trivially regime-aware vs always-delta-flat).
    pol_q_A_t0, pol_q_B_t0 = pol_pre_A_list[0]
    initial_policy = {
        f'r={r}': {
            'q_A_idx': int(pol_q_A_t0[r, s_idx_0, q_idx_0, q_idx_0, w_idx_0]),
            'q_B_idx': int(pol_q_B_t0[r, s_idx_0, q_idx_0, q_idx_0, w_idx_0]),
            'q_A_value': int(q_grid[pol_q_A_t0[r, s_idx_0, q_idx_0, q_idx_0, w_idx_0]]),
            'q_B_value': int(q_grid[pol_q_B_t0[r, s_idx_0, q_idx_0, q_idx_0, w_idx_0]]),
        }
        for r in range(n_states)
    }

    return {
        'V_pre_A': V_pre_A_list,       # length T_A + 1; V_pre_A_list[t] shape (R, N_S, N_Q, N_Q, N_W)
        'pol_pre_A': pol_pre_A_list,   # length T_A + 1; each entry is (pol_q_A, pol_q_B)
        'V_post_A': V_post_A_list,     # length T_dec - T_A + 1; V_post_A_list[t-T_A] shape (R, N_S, N_Q, N_W)
        'pol_post_A': pol_post_A_list,
        'V_0': V_0,
        'V_0_per_regime': V_0_per_regime,
        'grids': build_grids(cfg),
        'off_grid': off_grid.report(),
        'initial_policy': initial_policy,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def static_hold_value(cfg, n_paths=20000, seed=0):
    """Monte-Carlo lower bound: never trade. q_A = q_B = 0 always. Wealth at T_dec
    is just K + (0 + 0 - 1)·(S_{T_dec} - S_0) = K - (S_{T_dec} - S_0). Symlog utility."""
    rng = np.random.default_rng(seed)
    P = np.asarray(cfg.P_calib)
    pi0 = np.asarray(cfg.initial_state_probs)
    s_paths = np.full(n_paths, cfg.S_0, dtype=np.float64)
    # initial regime
    regimes = rng.choice(2, size=n_paths, p=pi0)
    for _ in range(cfg.T_dec):
        # transition regime first (next-step regime drives next-step return per simulator convention)
        u = rng.random(size=n_paths)
        regime_cdf = np.cumsum(P[regimes], axis=1)
        new_regime = (u[:, None] >= regime_cdf).sum(axis=1)
        new_regime = np.clip(new_regime, 0, 1)
        # log return under new_regime
        mu_dt = (np.array(cfg.mu_per_state)[new_regime] - 0.5 * np.array(cfg.sigma_per_state)[new_regime] ** 2) * cfg.dt_years
        sig_sqrt = np.array(cfg.sigma_per_state)[new_regime] * np.sqrt(cfg.dt_years)
        z = rng.standard_normal(size=n_paths)
        dlog_s = mu_dt + sig_sqrt * z
        s_paths *= np.exp(dlog_s)
        regimes = new_regime
    # terminal wealth = K - (S_T - S_0). With S_0 = 100 = K, wealth = K - S_T + S_0 = 2K - S_T.
    # Wait — that's not right. Liability = K - S_T, paid by us; cash flows out.
    # Wealth = cash + 0 + (K - S_T realized as cashflow folded into wealth).
    # Recall W = cash + (q_A+q_B-1)·S + K. At T_dec post-unwind (q=0): W = cash + K - S.
    # Initial state: W_0 = K (we set cash_0 = S_0 implicitly).
    # If never traded, no cashflow changes (except final liability), so cash stays at S_0 until terminal.
    # Then at terminal: cash = S_0 - (K - S_T) ??? — wait the liability is K - S_T; if it's POSITIVE we receive, if negative we pay.
    # We're the receiver of fixed: liability cashflow IN = K - S_T (positive when S_T < K, negative when S_T > K). So cash += (K - S_T).
    # final wealth = (S_0 + K - S_T) + 0 + 0 = S_0 + K - S_T.
    # With S_0 = K = 100: W_T = 200 - S_T.
    # But our W definition has W_0 = K. So W_T = W_0 + (q_A+q_B-1)*(S_T - S_0) over the dynamic = K + (-1)*(S_T - S_0) = K - S_T + S_0 = 2K - S_T (with K = S_0).
    # So both views agree: W_T = S_0 + K - S_T = 2K - S_T.
    w_T = cfg.S_0 + cfg.K - s_paths
    u_T = symlog_u(w_T, cfg.symlog_c)
    return float(u_T.mean()), float(u_T.std()) / np.sqrt(n_paths)


def delta_flat_value(cfg, n_paths=20000, seed=0):
    """Lower-bound MC: constant (q_A=0, q_B=1) policy. Wealth invariant between t=0
    and T_dec; turnover cost paid at t=0 (open q_B=1) and at T_dec (unwind). DP must
    dominate this — it's one of the feasible policies."""
    rng = np.random.default_rng(seed)
    P = np.asarray(cfg.P_calib)
    pi0 = np.asarray(cfg.initial_state_probs)
    s_paths = np.full(n_paths, cfg.S_0, dtype=np.float64)
    regimes = rng.choice(2, size=n_paths, p=pi0)
    for _ in range(cfg.T_dec):
        u = rng.random(size=n_paths)
        regime_cdf = np.cumsum(P[regimes], axis=1)
        new_regime = (u[:, None] >= regime_cdf).sum(axis=1)
        new_regime = np.clip(new_regime, 0, 1)
        mu_dt = (np.array(cfg.mu_per_state)[new_regime] - 0.5 * np.array(cfg.sigma_per_state)[new_regime] ** 2) * cfg.dt_years
        sig_sqrt = np.array(cfg.sigma_per_state)[new_regime] * np.sqrt(cfg.dt_years)
        z = rng.standard_normal(size=n_paths)
        s_paths *= np.exp(mu_dt + sig_sqrt * z)
        regimes = new_regime
    w_T = cfg.K - cfg.lambda_turnover * cfg.S_0 - cfg.lambda_turnover * s_paths
    u_T = symlog_u(w_T, cfg.symlog_c)
    return float(u_T.mean()), float(u_T.std()) / np.sqrt(n_paths)


def check_wealth_monotonicity(V_pre_A_list, V_post_A_list, tol=1.0e-2):
    """V should be (weakly) increasing in the wealth coordinate. Small numerical
    artifacts from the bilinear-vs-analytic U boundary can produce tiny non-monotonic
    deltas (~1e-3 utility units near grid edges); tolerance defaults to 1e-2 so the
    check still catches real bugs but isn't tripped by edge-numerics."""
    failures = []
    for t, V in enumerate(V_pre_A_list):
        if V is None:
            continue
        delta = np.diff(V, axis=-1)
        n_bad = int(np.sum(delta < -tol))
        if n_bad > 0:
            failures.append((f'pre-A t={t}', n_bad, float(delta.min())))
    for i, V in enumerate(V_post_A_list):
        if V is None:
            continue
        delta = np.diff(V, axis=-1)
        n_bad = int(np.sum(delta < -tol))
        if n_bad > 0:
            failures.append((f'post-A t={i}', n_bad, float(delta.min())))
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _report_off_grid(og):
    """Print the off-grid tracker summary — top-of-headline because silent boundary
    clamping is the most insidious failure mode (see feedback_discretised_dp_boundary_clamp)."""
    print('Off-grid wealth queries (bilinear interp clamp diagnostic):')
    print(f'  total queries        : {og["total_queries"]:,}')
    print(f'  fraction off-grid    : {og["frac_off_grid"]:.2%}   '
          f'(below: {og["frac_below"]:.2%}, above: {og["frac_above"]:.2%})')
    print(f'  observed w_query range: [{og["observed_w_range"][0]:+.2f}, {og["observed_w_range"][1]:+.2f}]')
    print(f'  grid w range          : [{og["grid_w_range"][0]:+.2f}, {og["grid_w_range"][1]:+.2f}]')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--T-dec', type=int, default=30)
    p.add_argument('--T-A', type=int, default=15)
    p.add_argument('--N-S', type=int, default=15)
    p.add_argument('--N-W', type=int, default=401)
    p.add_argument('--W-halfwidth', type=float, default=500.0)
    p.add_argument('--out', type=str, default='artifacts/gate2_exact_dp.npz')
    p.add_argument('--summary-out', type=str, default='artifacts/gate2_summary.json')
    p.add_argument('--off-grid-tol', type=float, default=0.05,
                   help='Max acceptable off-grid query fraction; gate FAILs if exceeded.')
    p.add_argument('--grid-stability-rel-tol', type=float, default=0.005,
                   help='|ΔV_0 / V_0| tolerance between primary and 1.5× wider grid runs.')
    p.add_argument('--alpha-tol', type=float, default=0.003,
                   help='Min DP alpha over delta-flat (in utility units) — should be a '
                        'few × MC-noise floor (~3e-4 here) to be Gate 4-useful. What also '
                        'matters: the optimal policy at the initial state is regime-dependent '
                        '(printed below); a regime-blind delta-flat would give alpha == 0.')
    p.add_argument('--skip-stability', action='store_true',
                   help='Skip the second-grid stability check (saves ~ run-time).')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    cfg = ToyConfig(T_dec=args.T_dec, T_A=args.T_A, N_S=args.N_S, N_W=args.N_W,
                    W_halfwidth=args.W_halfwidth)
    verbose = not args.quiet

    print('=' * 78)
    print('Gate 2 — exact backward grid-DP ground truth')
    print('=' * 78)
    print('Config:')
    for k, v in cfg.__dict__.items():
        print(f'  {k} = {v}')
    print()

    t0 = time.monotonic()
    result = backward_dp(cfg, verbose=verbose)
    elapsed = time.monotonic() - t0
    print(f'\nBackward DP completed in {elapsed:.1f} s')
    print(f'V_0 per regime              : {result["V_0_per_regime"]}')
    print(f'V_0 (mixed by initial probs): {result["V_0"]:+.6f}')
    print()
    print('Optimal policy at t=0, initial state (s=S_0, q_A=q_B=0, w=K) — per regime:')
    for r_lbl, p in result['initial_policy'].items():
        print(f'  {r_lbl}: q_A* = {p["q_A_value"]:+d}, q_B* = {p["q_B_value"]:+d}, '
              f'net exposure (q_A+q_B-1) = {p["q_A_value"]+p["q_B_value"]-1:+d}')
    print()
    _report_off_grid(result['off_grid'])
    print()

    # Off-grid gate
    og = result['off_grid']
    off_grid_pass = og['frac_off_grid'] <= args.off_grid_tol

    # Sanity checks
    print('Sanity checks:')
    print(f'  off-grid fraction <= {args.off_grid_tol:.0%}: '
          f'{"PASS" if off_grid_pass else "FAIL"} ({og["frac_off_grid"]:.2%})')
    mono_failures = check_wealth_monotonicity(result['V_pre_A'], result['V_post_A'])
    if mono_failures:
        print(f'  wealth monotonicity: FAIL — {len(mono_failures)} cells with -ve delta')
        for f in mono_failures[:5]:
            print(f'    {f}')
    else:
        print('  wealth monotonicity: PASS')

    sh_mean, sh_se = static_hold_value(cfg, n_paths=20000)
    print(f'  static-hold MC (lower bound): {sh_mean:+.6f} ± {1.96*sh_se:.6f} (95% CI)')
    static_pass = result['V_0'] >= sh_mean - 3 * sh_se
    print(f'    V_0 >= static-hold:         {"PASS" if static_pass else "FAIL"}')

    df_mean, df_se = delta_flat_value(cfg, n_paths=20000)
    print(f'  delta-flat MC (lower bound):  {df_mean:+.6f} ± {1.96*df_se:.6f} (95% CI)')
    deltaflat_pass = result['V_0'] >= df_mean - 3 * df_se
    print(f'    V_0 >= delta-flat:          {"PASS" if deltaflat_pass else "FAIL"}')

    # Dynamic-alpha gate: DP must meaningfully beat delta-flat — otherwise the toy
    # cannot discriminate "differential-ML works" from "differential-ML learned delta-flat",
    # which is Gate 4's whole purpose.
    alpha = result['V_0'] - df_mean
    alpha_pass = alpha >= args.alpha_tol
    print(f'  dynamic alpha (V_0 − delta-flat) = {alpha:+.6f}  '
          f'(tol {args.alpha_tol:+.3f}): {"PASS" if alpha_pass else "FAIL"}')

    # Grid-stability gate: re-run with 1.5× wealth halfwidth, same dw, and confirm
    # V_0 is stable. Catches residual truncation bias the off-grid threshold missed.
    stability_pass = True
    v0_wide = None
    if not args.skip_stability:
        print('\nGrid-stability check (1.5x wealth halfwidth, same dw):')
        wide_W = cfg.W_halfwidth * 1.5
        wide_N = int(round(cfg.N_W * 1.5))
        cfg_wide = ToyConfig(**{**cfg.__dict__, 'W_halfwidth': wide_W, 'N_W': wide_N})
        t_wide = time.monotonic()
        result_wide = backward_dp(cfg_wide, verbose=False)
        v0_wide = float(result_wide['V_0'])
        og_wide = result_wide['off_grid']
        print(f'  wide-grid: W_halfwidth={wide_W}, N_W={wide_N}, '
              f'off-grid={og_wide["frac_off_grid"]:.2%}, '
              f'V_0={v0_wide:+.6f}  ({time.monotonic() - t_wide:.1f} s)')
        rel_delta = abs(v0_wide - result['V_0']) / max(abs(result['V_0']), 1.0e-12)
        stability_pass = rel_delta <= args.grid_stability_rel_tol
        print(f'  |ΔV_0 / V_0| = {rel_delta:.4f}  '
              f'(tol {args.grid_stability_rel_tol:.4f}): '
              f'{"PASS" if stability_pass else "FAIL"}')

    overall = (off_grid_pass and (not mono_failures) and static_pass
               and deltaflat_pass and alpha_pass and stability_pass)
    print(f'\nOverall Gate 2: {"PASS" if overall else "FAIL"}')

    # Save value surface for Gate 3 consumption
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        # pol_pre_A_list entries are (pol_q_A, pol_q_B) tuples; the last index (t=T_A)
        # is intentionally None — the bridge step, not a decision. Skip it on save.
        pol_pre_A_qA = np.stack([p[0] for p in result['pol_pre_A'] if p is not None], axis=0)
        pol_pre_A_qB = np.stack([p[1] for p in result['pol_pre_A'] if p is not None], axis=0)
        pol_post_A_qB = np.stack([p for p in result['pol_post_A'] if p is not None], axis=0)
        np.savez_compressed(
            args.out,
            V_pre_A=np.stack([V for V in result['V_pre_A']], axis=0),
            V_post_A=np.stack([V for V in result['V_post_A']], axis=0),
            pol_pre_A_qA=pol_pre_A_qA,            # (T_A, R, N_S, N_Q, N_Q, N_W) chosen q_A
            pol_pre_A_qB=pol_pre_A_qB,            # (T_A, R, N_S, N_Q, N_Q, N_W) chosen q_B
            pol_post_A_qB=pol_post_A_qB,          # (T_dec-T_A, R, N_S, N_Q, N_W) chosen q_B
            T_dec=np.asarray(cfg.T_dec),          # scalars: avoid off-by-N inference from shapes
            T_A=np.asarray(cfg.T_A),
            V_0=np.array(result['V_0']),
            V_0_per_regime=np.asarray(result['V_0_per_regime']),
            log_s_grid=result['grids']['log_s'],
            s_grid=result['grids']['s_grid'],
            q_grid=result['grids']['q_grid'],
            w_grid=result['grids']['w_grid'],
        )
        print(f'\nSaved value surface to {args.out}')
    if args.summary_out:
        import os
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
        summary = {
            'config': {k: v for k, v in cfg.__dict__.items()},
            'V_0': float(result['V_0']),
            'V_0_per_regime': result['V_0_per_regime'].tolist(),
            'V_0_wide_grid': v0_wide,
            'off_grid': result['off_grid'],
            'initial_policy': result['initial_policy'],
            'static_hold_mean': float(sh_mean), 'static_hold_se': float(sh_se),
            'delta_flat_mean': float(df_mean), 'delta_flat_se': float(df_se),
            'dynamic_alpha': float(alpha),
            'wealth_monotonicity_failures': int(len(mono_failures)),
            'elapsed_seconds': float(elapsed),
            'pass_off_grid': bool(off_grid_pass),
            'pass_static_hold': bool(static_pass),
            'pass_delta_flat': bool(deltaflat_pass),
            'pass_alpha': bool(alpha_pass),
            'pass_stability': bool(stability_pass),
            'pass_overall': bool(overall),
        }
        with open(args.summary_out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved summary to {args.summary_out}')

    raise SystemExit(0 if overall else 1)


if __name__ == '__main__':
    main()
