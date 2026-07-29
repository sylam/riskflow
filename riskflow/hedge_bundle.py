"""Hedging `Bundle` + environment simulator + objective/utility stack.

`Bundle` is the simulated hedging world: `Bundle.from_batch` assembles the per-batch tensor
blocks the HedgeMonteCarlo simulator produces into one object (four construction stages —
tradables/time-axis, liability, factors, frame), and every downstream consumer reads it by
attribute. It owns the LOCKED frame (`utility_scale`, the per-step vol series), the sim-grid
views the solver indexes by `t`, and the inner-MC fork closures the calc attaches.

`BundleStepper` is the environment: the real futures/cash accounting (variation margin,
financing, transaction cost, roll rebate, IM funding, expiry flatten) advanced one day at a
time under any explicit policy. `run_hedge_execution` dispatches by Execution_Mode:
'solve_hedge' → the differential-ML solver (hedge_solver), 'simulate_only' → the no-trade
baseline. Module level keeps only the pure math both sides share: the wealth law, the utility
shapes, the friction debits, and the state-based portfolio reads.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import torch

from . import utils
from .hedge_runtime import assemble_privileged_factors, per_contract_kappa, privileged_block


# Rolling window (business days) for the realized-vol series feeding the symlog utility scale
# and the state-dependent bid/offer spread.
PRICE_ZSCORE_WINDOW = 20


def wealth_step(W, q, contract_size, dF, dL):
    """The ONE analytic hedged-wealth transition — frictionless telescoping core.

        W_{t+1} = W_t + Σ_i q_i·cs_i·dF_i + dL,   W_t = cumulative hedge P&L + marked liability L_t

    `q` (…,n_hedge) is the per-instrument position, `dF` (…,n_hedge) the per-instrument price
    move F_{t+1}−F_t, `dL` (…) the marked-liability change L_{t+1}−L_t. This is the frictionless
    law DiffSolverV2 rolls (bank/verdict) and, crucially, the one the twin loss DIFFERENTIATES:
    u(W_{t+1}) is taped back to a wealth leaf + the state-at-t market leaves, so this MUST stay a
    pure tensor op — no .item()/.detach()/.cpu()/.to(); callers own the grad context.

    `BundleStepper` is the DEPLOYMENT discretization of this same law: it books the same
    Σ q·cs·dF as per-instrument variation margin `pos·(price−settlement)·cs`, then layers the
    deployment extras this frictionless form deliberately omits — overnight financing (growth on
    cash/margin), transaction cost, per-instrument settlement/expiry, and terminal forced-flat.
    `_tracking_error_value` is its state-based read. That gap is an intentional fidelity
    difference (the value is position-free / freely-repositioning), not a bug — the faithful
    walk-forward path is hedge_solver `_rollout_on_stepper`, which rolls the real stepper.
    """
    return W + (q * contract_size * dF).sum(dim=-1) + dL


# The terminal-utility SHAPES (all map dollars→O(1) utility via the deal scale c: x = W/c).
# symlog is odd/symmetric; huber and cara are asymmetric (downside-averse). Selected by
# `Objective.Object`; any non-utility Object takes the identity (no-op) path.
# All consume the same scale `c` and live in "utility space" (the DP / value-fn recursion).
_UTILITY_OBJECTS = (
    "asymmetricutility_symlog", "asymmetricutility_huber", "asymmetricutility_cara")


def _is_symlog_objective(runtime):
    # TRUE symlog only — for symlog-SPECIFIC diagnostics (e.g. the −45 saturation tripwire,
    # the log1p-floor penalty-bite report), which don't transfer to huber/cara shapes.
    # `objective["object"]` is canonical-lowercased at normalization time
    # (hedge_runtime.normalize_objective), so plain equality is sufficient here.
    return (runtime.get("objective") or {}).get("object") == "asymmetricutility_symlog"


def _is_utility_objective(runtime):
    """True iff the objective transforms wealth through a utility shape (symlog/huber/cara) —
    so the DP/value-fn live in utility space and a scale `c` is required. False for the legacy
    identity objective."""
    return (runtime.get("objective") or {}).get("object") in _UTILITY_OBJECTS


def _utility_wrap_signed(x_dollars, runtime):
    """The terminal UTILITY u(W) applied to signed wealth — the single source of truth for
    the objective, the DP recursion, and the solver's value labels. Dispatches on the
    configured shape (all in normalised wealth x = W / c), identity for the legacy objective:

      symlog : sign(x)·log1p(|x|)                  — odd, tail-compressing (variance aversion)
      huber  : x − [a·loss² | a·δ²+2aδ(loss−δ)]    — linear gains; quadratic small losses;
               (loss = max(−x,0), knee δ)             linear deep tail (bounded scale, live grad)
      cara   : (1 − exp(−γ·x)) / γ                  — bounded gains, exponentially-penalised loss

    Shape params (huber a/δ, cara γ) are DIMENSIONLESS in c-units. Differentiable in `w`
    (AAD path: twin-loss labels, DP penalty, baseline B) — huber's knee is C¹, cara is smooth.

    `c` is `Bundle.utility_scale`, mirrored onto the runtime objective when the bundle is built.
    Missing it is fail-loud: the silent fallback (c = 1.0) produces plausible-looking-but-wrong
    rewards (log1p($1M / 1.0) ≈ 14) with no error at the call site."""
    if not _is_utility_objective(runtime):
        return x_dollars
    obj = runtime["objective"]
    c = obj.get("utility_scale")
    if c is None:
        raise ValueError(
            "utility objective active but runtime['objective']['utility_scale'] is not set — "
            "Bundle.from_batch mirrors it from the resolved utility_scale; a hand-built runtime "
            "must set it explicitly.")
    x = x_dollars / float(c)
    shape = obj["object"]
    if shape == "asymmetricutility_symlog":
        return torch.sign(x) * torch.log1p(x.abs())
    if shape == "asymmetricutility_huber":
        a = float(obj.get("huber_aversion", 2.5))
        d = float(obj.get("huber_delta", 1.0))
        loss = (-x).clamp(min=0.0)
        quad = a * loss * loss
        lin = a * d * d + 2.0 * a * d * (loss - d)
        return x - torch.where(loss <= d, quad, lin)
    # asymmetricutility_cara
    g = float(obj.get("cara_gamma", 1.0))
    return (1.0 - torch.exp(-g * x)) / g


def _realized_vol_series(spot, window=PRICE_ZSCORE_WINDOW):
    """Per-step annualized realized log-vol of a `(T, B)` mark path, batch-reduced to `(T,)`.
    The trailing-window (rolling std of log-returns × √252) proxy the state-dependent bid/offer
    spread falls back to when a world exposes no revealed conditional-vol state. Floored strictly
    positive so the warm-up window (before the window fills) never zeros the spread."""
    log_ret = spot.clamp_min(1e-9).log().diff(dim=0)                                  # (T-1, B)
    ret_sq = torch.cat([torch.zeros_like(log_ret[:1]), log_ret * log_ret], dim=0)     # (T, B)
    cum = torch.cat([torch.zeros_like(ret_sq[:1]), ret_sq.cumsum(dim=0)], dim=0)      # (T+1, B)
    T = ret_sq.shape[0]
    idx = torch.arange(T + 1, device=spot.device)
    lo = (idx - window).clamp_min(0)
    count = (idx - lo).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
    rv = (252.0 * (cum[idx[1:]] - cum[lo[1:]]) / count[1:]).clamp_min(0.0).sqrt()     # (T, B)
    return rv.mean(dim=-1).clamp_min(1e-6)                                            # (T,)


def _roll_rebate(deltas, prices, runtime, vol=None):
    """Calendar-spread rebate for a step's turnover (Evaluator.Roll_As_Calendar_Spread='Yes').
    A rebalance that REDUCES one contract month and INCREASES an adjacent month is a roll: the
    matched quantity should pay a single calendar-spread half-cost, not two independent outright
    half-spreads. Greedily match offsetting Δq (opposite signs) across CONSECUTIVE maturities
    (`names['hedges']` is maturity-ordered): the matched x on each pair earns
    `rebate = outright(x on both legs) − calendar(x)`; the unmatched residual keeps paying the
    per-instrument outright debit (untouched). Returns the per-path rebate `(B,)` to CREDIT back.

    Default calendar rate = half the sum of the two outright kappas (so it composes with the
    per-instrument / Vol_Scale spread automatically); `Evaluator.Calendar_Spread_Bps` overrides
    the matched leg with an absolute half-spread bps on the average leg notional (still charging
    the flat per-unit fee on both legs). `deltas`/`prices` key by hedge name; `vol` is the step's
    scalar annualized vol."""
    acc = runtime["accounting"]
    hedges = list(runtime["names"]["hedges"])
    rem = [deltas[h].to(torch.float32) for h in hedges]                 # remaining unmatched Δq
    rebate = torch.zeros_like(rem[0])
    cal_bps = acc["calendar_spread_bps"]
    for i in range(len(hedges) - 1):
        di, dj = rem[i], rem[i + 1]
        x = torch.minimum(di.abs(), dj.abs()) * ((di * dj) < 0)         # (B,) matched contracts
        rem[i] = di - di.sign() * x
        rem[i + 1] = dj - dj.sign() * x
        ni, nj = hedges[i], hedges[i + 1]
        pi, pj = prices[ni].abs(), prices[nj].abs()
        k_i = per_contract_kappa(runtime, pi, ni, vol)
        k_j = per_contract_kappa(runtime, pj, nj, vol)
        outright = x * (k_i + k_j)
        if cal_bps is None:
            cal = x * 0.5 * (k_i + k_j)
        else:
            cs_i = float(runtime["tradables"][ni]["contract_size"])
            cs_j = float(runtime["tradables"][nj]["contract_size"])
            n_avg = 0.5 * (pi * cs_i + pj * cs_j)
            cal = x * (2.0 * acc["transaction_cost_per_unit"] + 0.5 * cal_bps * 1.0e-4 * n_avg)
        rebate = rebate + (outright - cal)
    return rebate


def _im_funding_charge(positions, prices, runtime, vol, dt):
    """Per-hedge-leg initial-margin funding debit `{name: (B,)}` on a step's POST-trade book. The
    desk posts vol-linked IM on GROSS per-leg |q| — conservative: it over-margins calendar spreads,
    since a −1/+1 roll posts two legs' IM, not a netted spread margin —
        IM_i = IM_Vol_Multiplier · (σ_t/IM_Ref_Vol) · F_i · |q_i^post| · cs_i
    and pays `IM_i · IM_Funding_Spread_Bps · 1e-4 · dt` to FUND it over the calendar step. This is
    the spread the desk pays ABOVE the risk-free the margin ledger already earns, so it is a pure
    debit. σ_t is the SAME shared per-step vol that drives the Vol_Scale bid/offer spread, so
    funding rises with vol — a documented coupling. Anchoring: with IM_Ref_Vol and
    IM_Vol_Multiplier chosen so IM_i at the base σ_0/F_0 equals today's flat Initial_Margin.Amount
    per contract, the level is pinned to reality (calibration sets the knobs). Only called when
    `im_funding_spread_bps` is truthy — at which point the bundle's vol series exists so `vol` is
    a scalar, never None (per `Bundle._resolve_step_vol` gating)."""
    acc = runtime["accounting"]
    # mult · (σ/ref) · spread_bps · 1e-4 · dt — the per-leg scalar; ×(F_i·|q_i|·cs_i) gives funding_i.
    factor = (acc["im_vol_multiplier"] * (vol / acc["im_ref_vol"])
              * acc["im_funding_spread_bps"] * 1.0e-4 * dt)
    return {n: factor * prices[n].abs() * positions[n].abs()
               * float(runtime["tradables"][n]["contract_size"])
            for n in runtime["names"]["hedges"]}


def _portfolio_value(state, runtime):
    """Absolute total wealth (cash + margin + unrealized VM + position value where applicable).
    Use `_pnl_excess` to get wealth change since inception — that's what the asymmetric utility
    needs so the floor at zero correctly discriminates loss from gain.

    Cash and margin balances are stored as raw dollars and compounded daily (each step multiplies
    by cash_tv(next)/cash_tv(curr) ≈ 1 + SOFR·dt, then adds today's flows), so each dollar earns
    interest only from the day it landed."""
    hedges = runtime["names"]["hedges"]
    tradables = runtime["tradables"]
    positions, prices = state["positions"], state["tradable_values"]
    total = torch.zeros_like(state["done"], dtype=torch.float32)
    if runtime["accounting_mode"] == "cash_account":
        for name in hedges:
            total = total + (positions[name].to(dtype=torch.float32)
                             * prices[name].to(dtype=torch.float32)
                             * float(tradables[name]["contract_size"]))
        cash = None
        for balance in state["cash_accounts"].values():
            v = balance.to(dtype=torch.float32)
            cash = v if cash is None else cash + v
        return total if cash is None else total + cash
    # Futures mode: cash = starting capital (frozen); margin = all VM and trade-cost flows since
    # inception; the unrealized term position × (current_price - last_settlement_price) × cs
    # captures the gap between the most recent settlement and the current observable price
    # (typically zero within an episode, but non-zero at t=0 when the book starts with an
    # overnight position whose prior settlement is yesterday's close).
    for accounts in (state["margin_accounts"], state["cash_accounts"]):
        balances = None
        for balance in accounts.values():
            v = balance.to(dtype=torch.float32)
            balances = v if balances is None else balances + v
        if balances is not None:
            total = total + balances
    settlements = state["settlement_prices"]
    for name in hedges:
        if name not in settlements or name not in prices:
            continue
        total = total + (positions[name].to(dtype=torch.float32)
                         * (prices[name].to(dtype=torch.float32)
                            - settlements[name].to(dtype=torch.float32))
                         * float(tradables[name]["contract_size"]))
    return total


def _pnl_excess(state, runtime):
    """Wealth change since inception: portfolio_value - initial_portfolio_value. Used by the
    asymmetric utility and reported metrics — anywhere we want net P&L rather than absolute
    wealth so the floor at zero is meaningful. The initial baseline is snapshotted when the
    stepper builds its opening state and threaded through every transition."""
    pv = _portfolio_value(state, runtime)
    initial = state.get("initial_portfolio_value")
    if initial is None:
        return pv
    return pv - initial.to(device=pv.device, dtype=pv.dtype)


def _tracking_error_value(state, runtime):
    # Optimal hedge keeps pnl_excess + (liability_mtm + cumulative_liability_value) ≈ 0 at every
    # step. liability_mtm alone drops by the cashflow amount on payment dates (the cashflow moves
    # to realized cash, summed into cumulative_liability_value), which would otherwise inject a
    # ±cf shock unrelated to any action. The sum across the two channels is continuous across the
    # payment boundary, matching the terminal invariant (`pnl_excess + cumulative_liability_value`
    # once everything has been paid).
    pnl_excess = _pnl_excess(state, runtime).to(dtype=torch.float32)
    liability_mtm = state["liability_mtm_value"].to(dtype=torch.float32, device=pnl_excess.device)
    cumulative = state["cumulative_liability_value"].to(dtype=torch.float32, device=pnl_excess.device)
    return pnl_excess + liability_mtm + cumulative


def _align_time_axis(tensor, steps):
    """Truncate/pad a time-major tensor to `steps` rows (padding repeats the last row) — the
    liability MTM grid is the reference axis every other series is aligned to."""
    current = int(tensor.shape[0])
    if current >= steps:
        return tensor[:steps]
    pad_shape = (steps - current,) + tuple(tensor.shape[1:])
    return torch.cat([tensor, tensor[-1:].expand(*pad_shape)], dim=0)


class Bundle:
    """The simulated hedging world: every time-indexed tensor the solver and the environment
    read, plus the frame locked at build time.

    Built ONCE per simulation by `from_batch` (four stages — `_resolve_tradables`,
    `_resolve_liability`, `_resolve_factors`, `_resolve_frame`), then read-only. Time tensors
    carry a `History_Lookback_Business_Days` prefix of realized rows in front of the simulated
    grid, so full-grid indexing is `initial_time_index + t`; the `*_sim` views strip that prefix
    for solver code that indexes by simulation-grid `t`.

    The frame — `utility_scale` (the symlog/Huber/CARA scale c, also mirrored onto the runtime
    objective) and `step_annual_vol` (the per-step vol driving the state-dependent spread and IM
    funding) — is resolved here and never recomputed: a per-rollout c silently rescales every
    reward. `inner_mc` / `inner_mc_grad` are attached by `HedgeMonteCarlo.execute` in solve_hedge
    mode; they fork the simulator at a decision step and price the {t, t+1} window."""

    def __init__(self, base_date=None, business_day=None, num_batches=1):
        self.base_date = base_date
        self.business_day = business_day
        self.num_batches = int(num_batches)
        # Time axis (history prefix + simulated grid).
        self.time_grid_days = None          # (T,) day offsets from base_date
        self.time_grid_days_cpu = []        # the same as python ints (sync-free step loop)
        self.scenario_dates = None          # DatetimeIndex over the full grid
        self.business_indices = ()          # decision steps: business days in the sim window
        self.initial_time_index = 0         # history/sim boundary
        self.history_rows = 0               # History_Lookback_Business_Days prefix rows
        # Simulated series.
        self.tradables = {}                 # {instrument: (T, B)} marks
        self.factors = {}                   # {factor name: (T, …, B)} scenario paths
        self.privileged_factors = {}        # {name: (T, B, dim)} process-revealed state
        self.liability_mtm = None           # (T, B) marked liability, or None
        self.realized_cashflows = {}        # {currency: (T, B)} paid cashflows
        self.spot_price_history = {}        # {commodity: (H, B)} realized history prefix
        # Frame + liability descriptors.
        self.spot_realized_vol = {}         # {commodity: (T, B)} rolling realized vol
        self.step_annual_vol = None         # (T,) scalar vol per step, or None
        self.calibrated_utility_inputs = None
        self.utility_scale = None
        self.total_leg_volume = 0.0
        self.last_settlement_index = None
        self.last_live_mtm_index = 0
        # Inner-MC forks (attached by HedgeMonteCarlo.execute for Execution_Mode='solve_hedge').
        # Single-pass: peak is a function of Batch_Size x Inner_Sub_Batch, both JSON.
        self.inner_mc = None
        self.inner_mc_grad = None

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_batch(cls, base_date, business_day, time_grid_days, tradable_blocks,
                   factor_tensor_blocks, hedge_profile_blocks, num_batches, stoch_factors,
                   runtime, privileged_factor_blocks=None, total_leg_volume=None,
                   last_payment_day=None):
        """Assemble the per-batch tensor blocks the HedgeMonteCarlo simulator produced into one
        bundle: join them along the batch axis, then run the four resolve stages.

        `stoch_factors` is the simulator's factor map (factor_key → process); the processes are
        asked for their privileged surfaces, calibrated vols and revealed conditional vol."""
        self = cls(base_date, business_day, num_batches)
        tradables = {n: torch.cat(blocks, dim=1) for n, blocks in tradable_blocks.items()}
        factors = {n: torch.cat(blocks, dim=-1) for n, blocks in factor_tensor_blocks.items()}
        mtm = (torch.cat(hedge_profile_blocks['mtm'], dim=1)
               if hedge_profile_blocks.get('mtm') else None)
        cashflows = {currency: torch.cat(blocks, dim=1) for currency, blocks
                     in (hedge_profile_blocks.get('realized_cashflows') or {}).items()}
        # The liability MTM sets the time axis (its native length is the mtm grid); everything
        # else pads/truncates to it. No liability ⇒ the shortest simulated series wins.
        steps = int(mtm.shape[0]) if mtm is not None else min(
            [int(time_grid_days.shape[0])]
            + [int(t.shape[0]) for t in tradables.values()]
            + [int(t.shape[0]) for t in factors.values()])
        self._resolve_tradables(tradables, time_grid_days, steps, runtime)
        self._resolve_liability(mtm, cashflows, steps, total_leg_volume, last_payment_day)
        self._resolve_factors(factors, privileged_factor_blocks or {}, stoch_factors, steps)
        self._resolve_frame(runtime, stoch_factors)
        return self

    def _resolve_tradables(self, tradables, time_grid_days, steps, runtime):
        """Stage 1 — the time axis and the per-instrument mark series, both carrying the realized
        history prefix. `History_Lookback_Business_Days` rows of realized spot are prepended so a
        rolling-window feature at sim-day-0 already has its lookback: a tradable whose `Commodity`
        has a history takes that series, every other series broadcasts its first row. Also derives
        the CPU day mirror, the scenario dates, the sim origin `initial_time_index` and the
        business-day decision indices."""
        grid = _align_time_axis(time_grid_days, steps)
        tradables = {n: _align_time_axis(t, steps) for n, t in tradables.items()}
        history = runtime['portfolio_state']['spot_price_history']
        self.history_rows = H = int(runtime['history_lookback_business_days'])
        base_ts = pd.Timestamp(self.base_date)
        if H > 0 and history:
            batch_size = int(next(iter(tradables.values())).shape[1])
            ref_dates = history[next(iter(history))]['dates'][-H:]
            prefix_days = torch.tensor([int((d - base_ts).days) for d in ref_dates],
                                       dtype=grid.dtype, device=grid.device)
            grid = torch.cat([prefix_days, grid], dim=0)
            self.spot_price_history = {
                commodity: torch.tensor(payload['prices'][-H:], dtype=torch.float32,
                                        device=grid.device).unsqueeze(1)
                                       .expand(-1, batch_size).contiguous()
                for commodity, payload in history.items()}
            tradables = {
                n: torch.cat([self._history_prefix(
                    t, self.spot_price_history.get(
                        runtime['tradables'][n]['params'].get('Commodity'))), t], dim=0)
                for n, t in tradables.items()}
        self.time_grid_days = grid
        self.tradables = tradables
        self.time_grid_days_cpu = grid.detach().cpu().to(dtype=torch.int64).tolist()
        days = self.time_grid_days_cpu
        self.scenario_dates = pd.DatetimeIndex(
            [base_ts + pd.Timedelta(days=int(d)) for d in days])
        # Index where the history prefix ends and the simulation grid begins (history rows carry
        # negative day offsets). Solvers strip this offset to index time tensors by sim-grid t.
        self.initial_time_index = next(
            (i for i, d in enumerate(days) if int(d) >= 0), len(days))
        self.business_indices = tuple(
            i for i in range(max(len(self.scenario_dates) - 1, 0))
            if i >= self.initial_time_index and self.business_day.is_on_offset(self.scenario_dates[i]))

    def _history_prefix(self, tensor, realized=None):
        """The `history_rows` prepended to a simulated series: the realized history broadcast
        across the batch when the series has one, else the series' own first row repeated."""
        if realized is not None:
            return realized.to(dtype=tensor.dtype)
        return tensor[:1].expand((self.history_rows,) + tuple(tensor.shape[1:])).contiguous()

    def _resolve_liability(self, mtm, cashflows, steps, total_leg_volume, last_payment_day):
        """Stage 2 — the marked liability, the realized cashflow ledger, and the two schedule
        scalars the utility scale needs. `last_live_mtm_index` is the structural pre-settlement
        terminal: the grid appends one clean-exit row where the liability settles to zero, so the
        last LIVE mtm row is `steps - 2` — the single source for the DP depth (DiffSolverV2.T_dec)
        and the realized-path L_T read (no magnitude heuristic)."""
        prefixed = bool(self.spot_price_history)          # the stage-1 prefix gate
        if mtm is not None:
            mtm = _align_time_axis(mtm, steps)
            if prefixed:
                mtm = torch.cat([self._history_prefix(mtm), mtm], dim=0)
        self.liability_mtm = mtm
        self.realized_cashflows = {}
        for currency, tensor in cashflows.items():
            tensor = _align_time_axis(tensor, steps)
            if prefixed:
                # Nothing was paid before the sim starts — the prefix is zeros, not a repeat.
                tensor = torch.cat([torch.zeros((self.history_rows,) + tuple(tensor.shape[1:]),
                                                dtype=tensor.dtype, device=tensor.device),
                                    tensor], dim=0)
            self.realized_cashflows[currency] = tensor
        self.last_live_mtm_index = int(steps - 2)
        if total_leg_volume:
            self.total_leg_volume = float(total_leg_volume)
        if last_payment_day is not None:
            # Last (history-prefixed) grid step still strictly before the final payment.
            pending = [i for i, d in enumerate(self.time_grid_days_cpu) if d < last_payment_day]
            if pending:
                self.last_settlement_index = pending[-1]
        logging.info('liability scalars: total_leg_volume=%s last_settlement_index=%s',
                     self.total_leg_volume or None, self.last_settlement_index)

    def _resolve_factors(self, factors, privileged_factor_blocks, stoch_factors, steps):
        """Stage 3 — the simulated factor paths and the per-process privileged surfaces (the
        market state the value function consumes), both prefixed onto the history rows."""
        self.factors = {n: _align_time_axis(t, steps) for n, t in factors.items()}
        if self.spot_price_history:
            self.factors = {n: torch.cat([self._history_prefix(t), t], dim=0)
                            for n, t in self.factors.items()}
        privileged = assemble_privileged_factors(privileged_factor_blocks, stoch_factors)
        # NOTE: the privileged prefix is gated on the lookback ALONE (the other series need a
        # realized history to prefix onto); with a lookback but no Spot_Price_History these rows
        # are prepended while nothing else is, which offsets the surface by `history_rows`.
        if self.history_rows > 0 and privileged:
            privileged = {n: torch.cat([self._history_prefix(t), t], dim=0)
                          for n, t in privileged.items()}
        self.privileged_factors = privileged

    def _resolve_frame(self, runtime, stoch_factors):
        """Stage 4 — the LOCKED frame: the realized-vol surface, the utility scale c (mirrored
        onto the runtime objective so every reward/penalty reads one number), and the per-step
        vol series. Nothing here may be recomputed per rollout: a drifting c silently rescales
        the whole objective, and the vol series drives realized transaction cost."""
        self.spot_realized_vol = self._spot_realized_vol()
        if not self.spot_price_history:
            # No Spot_Price_History ⇒ source (spot, σ) from the CALIBRATED market data instead.
            self.calibrated_utility_inputs = self._calibrated_utility_inputs(runtime, stoch_factors)
        self.utility_scale = self._resolve_utility_scale(runtime)
        logging.info('utility_scale (symlog c) resolved to {0:.2f}'.format(self.utility_scale))
        self.mirror_utility_scale(runtime)
        self.step_annual_vol = self._resolve_step_vol(runtime, stoch_factors)

    def mirror_utility_scale(self, runtime):
        """Cache the resolved `c` on `runtime['objective']` so every reward, penalty and utility
        transform reads one number without taking the bundle in its signature. Invariant:
        `runtime['objective']['utility_scale'] == bundle.utility_scale` for any rollout that
        computes rewards against this bundle."""
        if runtime['objective'] is not None:
            runtime['objective']['utility_scale'] = float(self.utility_scale)

    def _spot_timeline(self, commodity):
        """The (H+T_sim, B) spot tensor for `commodity`: JSON history rows 0..H-1 concatenated
        with the simulator's CommodityPrice factor from row H. `commodity` is the canonical
        factor name (`utils.check_tuple_name`), which both dicts key by."""
        hist = self.spot_price_history.get(commodity)
        sim = self.factors.get(commodity)
        if hist is None or sim is None:
            return None
        sim_full = sim.to(dtype=torch.float32)
        hist_t = hist.to(dtype=torch.float32, device=sim_full.device)
        return torch.cat([hist_t, sim_full[int(hist_t.shape[0]):]], dim=0)

    def _spot_realized_vol(self, window=PRICE_ZSCORE_WINDOW, min_periods=5):
        """Annualized rolling realized log-vol of each underlying spot, `{commodity: (T, B)}`.
        In MR regimes with σ scaled to keep the stationary std fixed, realized vol increases with
        kappa — making this a regime signal as well as the utility-scale σ source."""
        out = {}
        for commodity in self.spot_price_history:
            S = self._spot_timeline(commodity)
            if S is None:
                continue
            log_S = S.clamp_min(1e-9).log()
            log_ret = log_S[1:] - log_S[:-1]                                    # (T_full - 1, B)
            ret_sq = torch.cat([torch.zeros_like(log_ret[:1]), log_ret * log_ret], dim=0)
            cum = torch.cat([torch.zeros_like(ret_sq[:1]), ret_sq.cumsum(dim=0)], dim=0)
            idx = torch.arange(int(ret_sq.shape[0]) + 1, device=S.device)
            lo = (idx - window).clamp_min(0)
            count = (idx - lo).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
            rv = (252.0 * (cum[idx[1:]] - cum[lo[1:]]) / count[1:]).clamp_min(0.0).sqrt()
            rv[:min_periods] = 0.0
            out[commodity] = rv
        return out

    def _calibrated_utility_inputs(self, runtime, stoch_factors):
        """Utility-scale (commodity, spot, σ) sourced from CALIBRATED market data — the
        Spot_Price_History-absent fallback. Spot is the sim-day-0 CommodityPrice level (factor
        row 0 = the price-factor Spot); σ is the process's calibrated annualized vol. With several
        CommodityPrice factors (cross-market strips) the sufficient-statistic-owning primary is
        preferred — the martingale the tradeable futures reference. None when no referenced
        underlying reports a calibrated vol."""
        referenced = set(runtime['referenced_commodities'])
        candidates = []
        for key, proc in (stoch_factors or {}).items():
            name = utils.check_tuple_name(key)
            if name not in referenced or name not in self.factors:
                continue
            sigma = proc.calibrated_annual_vol()
            if sigma is None or sigma <= 0.0:
                continue
            # Spots exposing a revealed sufficient statistic (HMM belief, GARCH log-variance) sort
            # first — the martingale primary of a cross-market strip.
            candidates.append((not bool(proc.privileged_layout(proc.param)), name, float(sigma)))
        if not candidates:
            return None
        candidates.sort()
        _, commodity, sigma = candidates[0]
        return (commodity, float(self.factors[commodity][0].median().item()), sigma)

    def _resolve_utility_scale(self, runtime):
        """The dollar scale `c` mapping dollars to utility (u(x; c) = sign(x)·log1p(|x|/c) for
        symlog). Single source of truth — every reward and penalty reads `Bundle.utility_scale`.

        Modes (`Objective.Utility_Scale_Mode`): `vol_scaled_notional` (default) gives
        c = total_leg_volume × initial_spot × σ_annual × √τ; `Objective.Utility_Scale_Explicit`
        overrides with a literal dollar value.

        Under a utility objective every degenerate path RAISES: a floor-c symlog silently breaks
        tail compression (log1p($1M/$1k) ≈ 7 ≈ log1p($100M/$1k) ≈ 11.5 — a 100× dollar gap
        becomes a 1.6× utility gap), which defeats the whole point. The legacy identity objective
        doesn't consume c, so it gets the harmless $1k floor."""
        objective = runtime['objective'] or {}
        needs_scale = _is_utility_objective(runtime)   # symlog / huber / cara all consume c

        def _degenerate(reason):
            if needs_scale:
                raise ValueError(
                    f"utility scale: cannot compute a meaningful c — {reason}. "
                    "A floor-c symlog silently compresses tails and defeats the reward shape. "
                    "Fix the bundle/config, or set Objective.Utility_Scale_Explicit to a "
                    "literal dollar value.")
            return 1.0e3

        mode = str(objective.get('utility_scale_mode', 'vol_scaled_notional')).lower()
        if mode != 'vol_scaled_notional':
            raise ValueError(
                f"Unsupported Objective.Utility_Scale_Mode: {mode!r}. "
                "Supported modes: 'vol_scaled_notional'. Set Utility_Scale_Explicit to "
                "override the formula with a literal dollar value.")
        explicit = objective.get('utility_scale_explicit')
        if explicit is not None:
            # An EXPLICIT override is honored exactly, including below the $1k production floor:
            # silently clamping would make a cell-by-cell oracle comparison fail for a reason
            # unrelated to the method. The floor only guards the formula path.
            c_explicit = float(explicit)
            if c_explicit < 1.0e3:
                logging.info(
                    'utility_scale Explicit override: c=%.4g (below the $1k '
                    'production floor — test mode; trust mode active)', c_explicit)
            return c_explicit
        # τ measures the SIM horizon, so anchor it at the sim-grid origin (== the history
        # lookback H when Spot_Price_History is present, 0 when it is absent).
        H = self.initial_time_index
        if self.last_settlement_index is None:
            return _degenerate("last_settlement_index missing from the bundle")
        tau_years = max(float(int(self.last_settlement_index) - H) / 252.0, 1.0 / 252.0)
        if not self.spot_price_history and self.calibrated_utility_inputs is None:
            return _degenerate(
                "spot_price_history is empty and no calibrated underlying vol is available "
                "(no referenced CommodityPrice process reports calibrated_annual_vol)")
        if not self.total_leg_volume:
            return _degenerate("total_leg_volume is zero")
        if self.spot_price_history:
            # History path (bit-anchored): spot + realized σ read at the history/sim boundary H.
            commodity = next(iter(self.spot_price_history))
            full = self._spot_timeline(commodity)
            if full is None or H >= int(full.shape[0]):
                return _degenerate(
                    f"spot timeline for {commodity!r} has length "
                    f"{0 if full is None else int(full.shape[0])} ≤ history_lookback H={H}")
            # Batch-median at index H rather than slot [H, 0]: the rolling-vol window at H spans
            # broadcast history rows so all batch entries are equal in well-behaved cases — but
            # `full[H]` is the FIRST sim step, and any process emitting a stochastic initial draw
            # would silently make c path-dependent off slot 0. Negligible cost (one (B,) reduce).
            initial_spot = float(full[H].median().item())
            rv = self.spot_realized_vol.get(commodity)
            sigma = float(rv[H].median().item()) if rv is not None and H < int(rv.shape[0]) else 0.0
        else:
            commodity, initial_spot, sigma = self.calibrated_utility_inputs
        if sigma <= 0.0:
            return _degenerate(
                f"realized vol for {commodity!r} is non-positive (σ={sigma}) — "
                "likely early-calibration window or upstream calc bug")
        c = self.total_leg_volume * initial_spot * sigma * (tau_years ** 0.5)
        if c < 1.0e3:
            return _degenerate(
                f"formula produced c=${c:,.2f} < $1k floor "
                f"(volume={self.total_leg_volume}, spot={initial_spot:.2f}, σ={sigma:.4f}, "
                f"√τ={tau_years ** 0.5:.3f})")
        return c

    def _resolve_step_vol(self, runtime, stoch_factors):
        """Per-step scalar annualized-vol series `(T,)` driving BOTH the state-dependent bid/offer
        half-spread (`per_contract_kappa` Vol_Scale) and the vol-linked IM funding charge. Built
        ONLY when a Vol_Scale spec OR IM funding is active — None otherwise, so `per_contract_kappa`
        ignores vol and no funding accrues. World-agnostic source: PREFER a process-revealed
        conditional vol (GARCH publishes log h_t → σ_t = √(exp(log h_t)/dt_c)); else the trailing
        realized-vol proxy off the primary spot factor path. Batch-reduced to one scalar per step —
        the cost model charges a single spread per step, matching the solver's mean-mark kappa. The
        chosen source is logged EXACTLY once per build: a silent proxy fallback under a GARCH
        retrain would invalidate the whole vol coupling, so it must be observable + test-pinned."""
        acc = runtime['accounting']
        spec = acc['bid_offer_spread_spec']
        vol_scale_active = bool(spec and spec['vol_scale'])
        im_funding_active = bool(acc['im_funding_spread_bps'])
        if not (vol_scale_active or im_funding_active):
            return None
        factor, proc, block = privileged_block(self.privileged_factors, stoch_factors, 'log_h')
        if block is not None:
            logging.info('step_annual_vol source: revealed %s log_h (Vol_Scale=%s IM_funding=%s)',
                         factor.name[0], vol_scale_active, im_funding_active)
            return proc.revealed_annual_vol(block[..., 0]).mean(dim=-1)                 # (T,)
        commodity = next((c for c in runtime['referenced_commodities'] if c in self.factors), None)
        if commodity is None:
            return None
        logging.info('step_annual_vol source: realized-vol proxy on %s (Vol_Scale=%s IM_funding=%s)',
                     commodity, vol_scale_active, im_funding_active)
        return _realized_vol_series(self.factors[commodity])

    # ---- accessors -----------------------------------------------------------
    @property
    def device(self):
        return self.time_grid_days.device

    @property
    def batch_size(self):
        return int(next(iter(self.tradables.values())).shape[-1])

    @property
    def last_index(self):
        """Final grid row — the terminal the stepper stops at."""
        return max(int(self.time_grid_days.shape[0]) - 1, 0)

    @property
    def tradables_sim(self):
        """History-stripped mark series `{instrument: (t_outer, B)}`, indexed by sim-grid `t`
        exactly like the inner-MC forks."""
        return {k: v[self.initial_time_index:] for k, v in self.tradables.items()}

    @property
    def liability_sim(self):
        """History-stripped liability MTM `(t_outer, B)`."""
        return self.liability_mtm[self.initial_time_index:]

    @property
    def n_outer_steps(self):
        return int(self.liability_sim.shape[0])

    @property
    def vol_sim(self):
        """Sim-grid slice of the per-step vol series (or None when no Vol_Scale/IM funding)."""
        v = self.step_annual_vol
        return None if v is None else v[self.initial_time_index:]

    def realized_paths(self, runtime):
        """The realized outer-path data the no-inner-MC tracks (hindsight, textbook) consume:
        `F` `(n_hedge, t_outer, B)` hedge prices, `L_T` `(B,)` the liability terminal MTM at the
        pre-settlement `last_live_mtm_index`, and `t_outer`."""
        tradables_sim = self.tradables_sim
        F = torch.stack([tradables_sim[h] for h in runtime['names']['hedges']], dim=0)
        return F, self.liability_sim[self.last_live_mtm_index], self.n_outer_steps

    def liability_at(self, time_index):
        """Marked liability `(B,)` at full-grid step `time_index` (zeros when there is none)."""
        if self.liability_mtm is None:
            return torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        return self.liability_mtm[time_index].to(dtype=torch.float32, device=self.device)

    def cashflow_at(self, time_index):
        """Realized cashflow `(B,)` paid at full-grid step `time_index`, summed over currencies."""
        total = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        for tensor in self.realized_cashflows.values():
            total = total + tensor[time_index].to(dtype=torch.float32, device=self.device)
        return total

    def vol_at(self, time_index):
        """Scalar annualized vol at full-grid step `time_index`, or None when no vol series was
        built (`per_contract_kappa` then ignores vol)."""
        return None if self.step_annual_vol is None else self.step_annual_vol[int(time_index)]

    def calendar_dt(self, current, next_index):
        """CALENDAR-clock step-year fraction `(days[next]-days[cur])/365.25` — the sim dt
        convention (dt=1/365.25) over which posted initial margin is funded. Scalar python float
        off the CPU day mirror (no CUDA sync)."""
        days = self.time_grid_days_cpu
        return (days[int(next_index)] - days[int(current)]) / 365.25


class BundleStepper:
    """The hedging ENVIRONMENT: advance the bundle one day at a time under arbitrary actions.

    Owns the realized accounting the frictionless `wealth_step` law abstracts away — overnight
    financing on cash/margin, per-instrument variation margin and settlement, transaction cost
    (with the optional calendar-spread rebate and vol-linked IM funding), position rounding to
    integer contracts, and the terminal forced flatten. Each step returns a state dict the caller
    inspects; the caller chooses the next action. Supports `copy.deepcopy(stepper)` to fork into
    counterfactual branches, and records its own trajectory for `write_diagnostic_csvs`.

    Vectorized over the bundle's full batch (B paths advance in lockstep). Action values can be
    scalars (broadcast) or per-path `(B,)` tensors. `runtime` is a PARAMETER of the replay, not
    the bundle's: passing a variant (e.g. one accounting switch flipped) is how the cost
    decomposition isolates each friction on an unchanged world. `mirror_scale=False` keeps the
    runtime's utility scale as the caller set it (see `__init__`)."""

    def __init__(self, bundle, runtime, mirror_scale=True):
        self.bundle = bundle
        self.runtime = runtime
        self._accounting = runtime['accounting']
        self._account_of = self._accounting['instrument_to_cash_account']
        self._hedges = tuple(runtime['names']['hedges'])
        self._instrument_order = tuple(runtime['names']['action_instruments'])
        self._cash_names = tuple(runtime['names']['cash_accounts'])
        self._tradable_names = tuple(runtime['names']['tradables'])
        self._device = bundle.device
        self._batch_size = bundle.batch_size
        self._last_idx = bundle.last_index
        self._decision_set = set(int(i) for i in bundle.business_indices)
        # The replay's rewards are marked against THIS bundle, so by default its scale is
        # re-mirrored onto the (possibly variant) runtime. TRAP: under a frozen-policy run
        # (DiffV2_Load_Value_Fn) the solver has already restored the CHECKPOINT's scale — the
        # value function's own frame — and this overwrites it with the eval world's, so a policy
        # rollout decides under a different c than the solver's own verdict did. `mirror_scale`
        # =False leaves the runtime's scale alone, which is what a rollout of a FROZEN value
        # function wants (hedge_solver passes it in streaming mode). The default stays True
        # because every walk-forward anchor to date was measured through the re-mirror.
        if mirror_scale:
            bundle.mirror_utility_scale(runtime)
        self._state = self._initial_state()
        # Per-decision recording for post-hoc diagnostic CSV writing. Cheap (a few (B,) tensors
        # per decision step); always-on so write_diagnostic_csvs has data to use.
        self._times = []
        self._position_history = {n: [] for n in self._instrument_order}
        self._trade_history = {n: [] for n in self._instrument_order}
        self._price_history = {n: [] for n in self._instrument_order}
        self._terminal_transition = None

    # ---- public surface ------------------------------------------------------
    @property
    def time_index(self):
        return int(self._state['time_index'])

    @property
    def is_decision_step(self):
        return self.time_index in self._decision_set

    @property
    def done(self):
        return self.time_index >= self._last_idx

    def observe(self):
        """Snapshot of the current pre-step state. Tensors are returned as-is (caller can
        `.cpu().numpy()` if needed); positions are post-last-step."""
        return {
            'time_index': self.time_index,
            'is_decision_step': self.is_decision_step,
            'done': self.done,
            'positions': dict(self._state['positions']),
            'tradable_values': dict(self._state['tradable_values']),
            'cumulative_liability_value': self._state['cumulative_liability_value'],
        }

    def step(self, action=None):
        """Advance one time step. `action` is `{instrument_name: scalar_or_(B,)_tensor}` applied
        as trade deltas (only meaningful at decision steps; ignored otherwise). Pass `None` for
        zero trades. Returns the post-step observation plus this transition's per-path
        pnl_excess + liability_value."""
        was_decision_step = self.is_decision_step
        if was_decision_step:
            # Record pre-step position + price for the diagnostic CSV.
            self._times.append(self.time_index)
            for n in self._instrument_order:
                self._position_history[n].append(self._state['positions'][n].detach().cpu().clone())
                self._price_history[n].append(self._state['tradable_values'][n].detach().cpu().clone())
        structured = (self._structured_action(action)
                      if (action is not None and was_decision_step) else None)
        next_state = self._step_state(self._state, structured)
        transition = self._payoff(next_state)
        if was_decision_step:
            # Realized trade = post-step position − pre-step position (handles env clips/forces).
            for n in self._instrument_order:
                self._trade_history[n].append(
                    next_state['positions'][n].detach().cpu() - self._position_history[n][-1])
        self._state = next_state
        self._terminal_transition = transition
        return {
            **self.observe(),
            'transition_pnl_excess': transition['pnl_excess'],
            'transition_liability_value': transition['liability_value'],
        }

    def evaluation_output(self, timing=None):
        """Terminal P&L summary in the `evaluation_summary` shape `HedgeMonteCarlo.execute`
        publishes. `simulate_only` rolls the no-trade policy, so the policy and the reference
        baseline are the same trajectory and the difference block is identically zero."""
        hedge_pnl = self._terminal_transition['pnl_excess'].detach().to(dtype=torch.float32)
        liability = self._terminal_transition['liability_value'].detach().to(dtype=torch.float32)
        net_pnl = hedge_pnl + liability
        metrics = {
            'average_net_pnl': float(net_pnl.mean().item()),
            'median_net_pnl': float(torch.quantile(net_pnl.to(dtype=torch.float64), 0.5).item()),
            'worst_net_pnl': float(net_pnl.min().item()),
            'average_hedge_pnl': float(hedge_pnl.mean().item()),
            'average_liability': float(liability.mean().item()),
        }
        cpu = lambda values: {str(n): t.detach().to(dtype=torch.float32).cpu()
                              for n, t in values.items()}
        return {
            'metrics': metrics,
            'final_state': {
                'positions': cpu(self._state['positions']),
                'cash_accounts': cpu(self._state['cash_accounts']),
                'margin_accounts': cpu(self._state['margin_accounts']),
                'pnl_excess': hedge_pnl.cpu(),
                'liability_value': liability.cpu(),
                'net_pnl': net_pnl.cpu(),
            },
            'diagnostics': {'num_episodes': int(net_pnl.shape[0]),
                            'num_batches': self.bundle.num_batches,
                            'trainer_type': 'simulate'},
            'timing': dict(timing or {}),
            'reference': {'no_trade': {'metrics': metrics},
                          'policy_minus_no_trade': {k: 0.0 for k in metrics}},
        }

    def write_diagnostic_csvs(self, output_dir: str, label: str = 'custom') -> None:
        """Write the per-day per-instrument breakdown + terminal P&L summary for the trajectory
        this stepper accumulated, driven by whatever policy the caller chose. Files:
          <output_dir>/<label>_paths.csv   — 5 representative paths (worst/p5/mean/p95/best)
          <output_dir>/<label>_summary.csv — terminal P&L stats

        Must be called after the rollout has reached `done`."""
        if self._terminal_transition is None:
            raise ValueError("Stepper has no recorded trajectory yet — call step() until done first.")
        rollout = self._rollout()
        fields = self._diag_fields(rollout)
        os.makedirs(output_dir, exist_ok=True)
        self._diag_write_paths(fields, label, os.path.join(output_dir, f'{label}_paths.csv'))
        net = rollout['net_pnl'].numpy()
        total = fields['total_ex_funding_discount'][-1].numpy()
        pd.DataFrame([
            {'policy': label, 'metric': 'mean', 'net_pnl': float(net.mean()),
             'total_ex_funding': float(total.mean())},
            {'policy': label, 'metric': 'std', 'net_pnl': float(net.std()),
             'total_ex_funding': float(total.std())},
            {'policy': label, 'metric': 'min', 'net_pnl': float(net.min()),
             'total_ex_funding': float(total.min())},
            {'policy': label, 'metric': 'p5', 'net_pnl': float(np.percentile(net, 5)),
             'total_ex_funding': float(np.percentile(total, 5))},
            {'policy': label, 'metric': 'p95', 'net_pnl': float(np.percentile(net, 95)),
             'total_ex_funding': float(np.percentile(total, 95))},
            {'policy': label, 'metric': 'max', 'net_pnl': float(net.max()),
             'total_ex_funding': float(total.max())},
        ]).to_csv(os.path.join(output_dir, f'{label}_summary.csv'), index=False,
                  float_format='%.2f')

    # ---- state construction + transition -------------------------------------
    def _zeros_by_name(self, names):
        return {str(n): torch.zeros(self._batch_size, dtype=torch.float32, device=self._device)
                for n in names}

    def _seed_by_name(self, seed_values, names):
        return {str(n): torch.full((self._batch_size,), float(seed_values.get(str(n), 0.0)),
                                   dtype=torch.float32, device=self._device) for n in names}

    def _values_at(self, time_index):
        """Current mark `{instrument: (B,)}` at full-grid `time_index`."""
        tradables = self.bundle.tradables
        return {n: tradables[n][time_index].to(dtype=torch.float32)
                for n in self._tradable_names if n in tradables}

    def _initial_state(self):
        """The opening state at sim-day-0 (bundle row `initial_time_index`) — the history prefix
        feeds features only, never the simulator. JSON-supplied positions are "today's overnight
        book", not "H days ago"."""
        portfolio_state = self.runtime['portfolio_state']
        initial_time_index = self.bundle.initial_time_index
        fallback = self._values_at(initial_time_index)
        seeded_settlement = portfolio_state['settlement_prices']
        state = {
            'done': torch.zeros(self._batch_size, dtype=torch.bool, device=self._device),
            'positions': self._seed_by_name(portfolio_state['positions'], self._hedges),
            'cash_accounts': self._seed_by_name(portfolio_state['cash_balances'], self._cash_names),
            'margin_accounts': self._seed_by_name(portfolio_state['margin_balances'], self._cash_names),
            'realized_pnl': self._zeros_by_name(self._hedges),
            'variation_margin': self._zeros_by_name(self._hedges),
            'cumulative_pnl': self._zeros_by_name(self._hedges),
            'time_held': self._zeros_by_name(self._hedges),
            'cumulative_liability_value': torch.zeros(self._batch_size, dtype=torch.float32,
                                                      device=self._device),
            'settlement_prices': {
                name: torch.full((self._batch_size,), float(seeded_settlement[name]),
                                 dtype=torch.float32, device=self._device)
                if name in seeded_settlement else fallback[name].clone()
                for name in self._hedges if name in fallback},
        }
        state = self._refresh(state, initial_time_index)
        # Snapshot the inception baseline (cash + margin + initial unrealized VM if positions
        # started non-zero against a stale settlement) so `_pnl_excess` returns the change.
        state['initial_portfolio_value'] = _portfolio_value(state, self.runtime).detach().clone()
        # Re-seat settlement_prices to the simulator's price at sim-day-0: the seed (yesterday's
        # close) vs sim-day-0 forward gap was just absorbed into initial_portfolio_value via the
        # unrealized-VM term, so subsequent steps' VM is clean step-over-step P&L. Without this
        # the first trade carries a seed-gap noise of (price_H − seed) × delta × cs.
        for name, current_price in state['tradable_values'].items():
            if name in state['settlement_prices']:
                state['settlement_prices'][name] = current_price.detach().clone()
        return state

    def _refresh(self, state, time_index):
        """Cheap per-step refresh: re-read the price/liability views at `time_index` and carry
        the simulator state forward. `cumulative_liability_value` absorbs the step's realized
        cashflow so the liability channel stays continuous across payment dates."""
        refreshed = {
            'time_index': int(time_index),
            'done': state['done'],
            'positions': state['positions'],
            'cash_accounts': state['cash_accounts'],
            'margin_accounts': state['margin_accounts'],
            'realized_pnl': state['realized_pnl'],
            'variation_margin': state['variation_margin'],
            'cumulative_pnl': state['cumulative_pnl'],
            'time_held': state['time_held'],
            'settlement_prices': state['settlement_prices'],
            'tradable_values': self._values_at(time_index),
            'liability_mtm_value': self.bundle.liability_at(time_index),
            'realized_cashflow_value': self.bundle.cashflow_at(time_index),
            'initial_portfolio_value': state.get('initial_portfolio_value'),
        }
        refreshed['cumulative_liability_value'] = (
            state['cumulative_liability_value'].to(dtype=torch.float32)
            + refreshed['realized_cashflow_value'])
        return refreshed

    def _step_state(self, state, action):
        if self.runtime['accounting_mode'] == 'cash_account':
            return self._cash_account_step(state, action)
        return self._futures_account_step(state, action)

    def _trade_deltas(self, action):
        """Per-hedge trade deltas `(B,)` from a structured action (zeros for no action)."""
        resolved = self._zeros_by_name(self._hedges)
        if action is None:
            return resolved
        for name, value in action['trade_deltas'].items():
            if name in resolved:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=self._device)
                resolved[name] = tensor.repeat(self._batch_size) if tensor.ndim == 0 else tensor
        return resolved

    def _structured_action(self, action_dict):
        """Client action → the integer-contract trade deltas the env books. Position limits are
        enforced reward-side, not clipped here."""
        deltas = {}
        for name in self._instrument_order:
            v = action_dict.get(name, 0)
            deltas[name] = (v.to(device=self._device, dtype=torch.float32)
                            if isinstance(v, torch.Tensor)
                            else torch.full((self._batch_size,), float(v), dtype=torch.float32,
                                            device=self._device))
        ordered = torch.stack([deltas[n] for n in self._instrument_order],
                              dim=1).round().to(dtype=torch.int64)
        return {'trade_deltas': {n: ordered[:, i] for i, n in enumerate(self._instrument_order)}}

    def _cost(self, trade_delta, price, name, vol):
        # Realized debit on |Δq| contracts at the turnover-cost rule (per_contract_kappa).
        return trade_delta.abs() * per_contract_kappa(self.runtime, price.abs(), name, vol)

    def _credit(self, accounts, account_name, amount):
        if account_name is not None:
            accounts[account_name] = accounts[account_name] + amount

    def _growth_factors(self, current, next_idx):
        """Per-cash-account one-day growth factor for compounding balances overnight. The cash
        tradable value is `Units / D(t) × fx_rep`, so `tv(next)/tv(cur) = D(cur)/D(next)` is the
        growth factor (≈ 1 + SOFR·dt). Empty when next_idx == current (terminal pass-through)."""
        if current >= next_idx:
            return {}
        tv_curr = self._values_at(current)
        tv_next = self._values_at(next_idx)
        return {n: tv_next[n] / tv_curr[n] for n in self._cash_names
                if n in tv_curr and n in tv_next}

    def _compound(self, accounts, factors):
        """Multiply each balance by its growth factor (pass-through by reference when absent —
        safe because balances are always rebound to new tensors, never mutated in place)."""
        return {n: (bal * factors[n] if n in factors else bal) for n, bal in accounts.items()}

    def _step_time_held(self, time_held, next_positions):
        # Increment where the position is non-zero, reset where it returned to flat.
        return {name: torch.where(next_positions[name].abs() > 0, prev + 1.0, torch.zeros_like(prev))
                for name, prev in time_held.items()}

    def _step_cumulative_pnl(self, cumulative_pnl, variation_margin, next_positions):
        """Per-trade running P&L: accumulate VM while a position is open, reset to 0 when flat —
        the same lifetime-of-current-trade semantics as `_step_time_held`."""
        return {name: torch.where(next_positions[name].abs() > 0,
                                  prev + variation_margin[name], torch.zeros_like(prev))
                for name, prev in cumulative_pnl.items()}

    def _flatten_cash(self, positions, cash_accounts, terminal_values, vol):
        for n in self._hedges:
            delta = -positions[n]
            cost = self._cost(delta, terminal_values[n], n, vol)
            cs = float(self.runtime['tradables'][n]['contract_size'])
            account = self._account_of[n]
            if account is not None:
                cash_accounts[account] = (cash_accounts[account]
                                          - (delta * terminal_values[n] * cs + cost))
            positions[n] = positions[n] + delta

    def _flatten_futures(self, positions, margin_accounts, settlement_prices, terminal_values, vol):
        """Close hedge positions at `terminal_values` and capture any residual variation margin
        (`position × (terminal − settlement) × cs`) so no P&L leaks between the last settlement and
        the close. When the caller has already advanced settlement to terminal (the normal path)
        the residual is 0 and only the trade cost is debited. `vol` scales the forced-flat turnover
        exactly like every other decision-step debit."""
        for n in self._hedges:
            cs = float(self.runtime['tradables'][n]['contract_size'])
            residual_vm = positions[n] * (terminal_values[n] - settlement_prices[n]) * cs
            delta = -positions[n]
            cost = self._cost(delta, terminal_values[n], n, vol)
            account = self._account_of[n]
            self._credit(margin_accounts, account, residual_vm)
            self._credit(margin_accounts, account, -cost)
            positions[n] = positions[n] + delta

    def _cash_account_step(self, state, action):
        # `done` is purely a function of time_index vs the last bundle index; checking the python
        # int avoids a CUDA-CPU sync that would fire on every step.
        current = int(state['time_index'])
        last = self._last_idx
        if current >= last:
            return state
        next_idx = min(current + 1, last)
        acc = self._accounting
        # Shallow dict copy: tensors are replaced in-slot, never mutated in place.
        next_positions = dict(state['positions'])
        next_cash = self._compound(state['cash_accounts'], self._growth_factors(current, next_idx))
        deltas = self._trade_deltas(action)
        vol_t = self.bundle.vol_at(current)
        for n in self._instrument_order:
            delta = deltas[n]
            price = state['tradable_values'][n]
            cost = self._cost(delta, price, n, vol_t)
            cs = float(self.runtime['tradables'][n]['contract_size'])
            account = self._account_of[n]
            if account is not None:
                # Notional debit is `delta × price × contract_size` — each contract is
                # `contract_size` units of the underlying.
                next_cash[account] = next_cash[account] - (delta * price * cs + cost)
            next_positions[n] = (next_positions[n] + delta).round()
        if acc['roll_as_calendar_spread']:
            rebate = _roll_rebate(deltas, {h: state['tradable_values'][h] for h in self._hedges},
                                  self.runtime, vol_t)
            self._credit(next_cash, self._account_of[self._hedges[0]], rebate)
        if acc['im_funding_spread_bps']:
            # Vol-linked IM funding on the post-trade book over the calendar step — a pure debit
            # into the same realized cash P&L path as transaction cost (per-leg, routed by
            # currency), so it flows through _portfolio_value → _pnl_excess → the utility.
            dt = self.bundle.calendar_dt(current, next_idx)
            for n, funding in _im_funding_charge(next_positions, state['tradable_values'],
                                                 self.runtime, vol_t, dt).items():
                self._credit(next_cash, self._account_of[n], -funding)
        if acc['force_flat_at_end'] and current >= last - 1:
            self._flatten_cash(next_positions, next_cash, self._values_at(last), vol_t)
        # Cash mode tracks no daily VM (realized_pnl / variation_margin stay zero), but
        # cumulative_pnl still resets at flat for the same per-trade-lifetime semantics.
        zero_vm = self._zeros_by_name(self._hedges)
        next_state = {
            'done': torch.full_like(state['done'], next_idx >= last, dtype=torch.bool),
            'positions': next_positions,
            'cash_accounts': next_cash,
            'margin_accounts': state['margin_accounts'],
            'realized_pnl': self._zeros_by_name(self._hedges),
            'variation_margin': zero_vm,
            'cumulative_pnl': self._step_cumulative_pnl(state['cumulative_pnl'], zero_vm,
                                                        next_positions),
            'time_held': self._step_time_held(state['time_held'], next_positions),
            'cumulative_liability_value': state['cumulative_liability_value'],
            'settlement_prices': state['settlement_prices'],
            'initial_portfolio_value': state['initial_portfolio_value'],
        }
        return self._refresh(next_state, next_idx)

    def _futures_account_step(self, state, action):
        current = int(state['time_index'])
        last = self._last_idx
        if current >= last:
            return state
        settlement_idx = min(current + 1, last)
        acc = self._accounting
        next_positions = dict(state['positions'])
        growth = self._growth_factors(current, settlement_idx)
        next_cash = self._compound(state['cash_accounts'], growth)
        next_margin = self._compound(state['margin_accounts'], growth)
        next_settlement = dict(state['settlement_prices'])
        realized_pnl = self._zeros_by_name(self._hedges)
        variation_margin = self._zeros_by_name(self._hedges)
        deltas = self._trade_deltas(action)
        next_values = self._values_at(settlement_idx)
        # Futures: cash_accounts is frozen at starting capital; only margin tracks VM and trade
        # cost. Apply the trade BEFORE computing VM so the new delta participates in the
        # price(t)→price(t+1) accrual: the agent transacted at the decision-time price
        # (= settlement_old in steady state) and the whole post-trade position is then marked at
        # price(t+1). Positions round to integer contracts so float drift can't accumulate.
        for n in self._hedges:
            cs = float(self.runtime['tradables'][n]['contract_size'])
            next_positions[n] = (next_positions[n] + deltas[n]).round()
            vm = next_positions[n] * (next_values[n] - state['settlement_prices'][n]) * cs
            realized_pnl[n] = vm
            variation_margin[n] = vm
            next_settlement[n] = next_values[n].clone()
            self._credit(next_margin, self._account_of[n], vm)
        vol_t = self.bundle.vol_at(current)
        for n in self._instrument_order:
            # Trade cost references the price the agent actually saw and acted on (decision-time):
            # using next_values would let unrelated overnight mid moves distort the spread cost.
            self._credit(next_margin, self._account_of[n],
                         -self._cost(deltas[n], state['tradable_values'][n], n, vol_t))
        if acc['roll_as_calendar_spread']:
            rebate = _roll_rebate(deltas, {h: state['tradable_values'][h] for h in self._hedges},
                                  self.runtime, vol_t)
            self._credit(next_margin, self._account_of[self._hedges[0]], rebate)
        if acc['im_funding_spread_bps']:
            dt = self.bundle.calendar_dt(current, settlement_idx)
            for n, funding in _im_funding_charge(next_positions, state['tradable_values'],
                                                 self.runtime, vol_t, dt).items():
                self._credit(next_margin, self._account_of[n], -funding)
        if acc['force_flat_at_end'] and current >= last - 1:
            self._flatten_futures(next_positions, next_margin, next_settlement, next_values, vol_t)
        next_state = {
            'done': torch.full_like(state['done'], settlement_idx >= last, dtype=torch.bool),
            'positions': next_positions,
            'cash_accounts': next_cash,
            'settlement_prices': next_settlement,
            'margin_accounts': next_margin,
            'realized_pnl': realized_pnl,
            'variation_margin': variation_margin,
            'cumulative_pnl': self._step_cumulative_pnl(state['cumulative_pnl'], variation_margin,
                                                        next_positions),
            'time_held': self._step_time_held(state['time_held'], next_positions),
            'cumulative_liability_value': state['cumulative_liability_value'],
            'initial_portfolio_value': state['initial_portfolio_value'],
        }
        return self._refresh(next_state, settlement_idx)

    def _payoff(self, state):
        """This transition's terminal read: `pnl_excess` is the wealth change since inception and
        is only marked at the terminal step (a sync-free check on time_index — `done` is uniform
        across the batch since every scenario advances together); `liability_value` is the
        cumulative realized liability, always carried."""
        pnl_excess = (_pnl_excess(state, self.runtime).to(device=self._device, dtype=torch.float32)
                      if int(state['time_index']) >= self._last_idx
                      else torch.zeros(self._batch_size, dtype=torch.float32, device=self._device))
        return {'pnl_excess': pnl_excess,
                'liability_value': state['cumulative_liability_value'].to(
                    device=self._device, dtype=torch.float32)}

    # ---- diagnostic CSV assembly ---------------------------------------------
    def _rollout(self):
        """The recorded trajectory, stacked per instrument over decision steps."""
        return {
            'times': self._times,
            'position': {n: torch.stack(self._position_history[n], dim=0) for n in self._instrument_order},
            'trade': {n: torch.stack(self._trade_history[n], dim=0) for n in self._instrument_order},
            'price': {n: torch.stack(self._price_history[n], dim=0) for n in self._instrument_order},
            'net_pnl': (self._terminal_transition['pnl_excess']
                        + self._terminal_transition['liability_value']).detach().cpu(),
        }

    def _diag_fields(self, rollout=None):
        """Per-instrument (T, B) per-day cashflow tensors plus portfolio totals, reconstructed on
        CPU from the recorded trajectory. Multi-commodity bundles raise — diagnostic CSVs for
        multi-commodity are out of scope."""
        rollout = self._rollout() if rollout is None else rollout
        bundle, runtime = self.bundle, self.runtime
        spot_keys = [k for k in bundle.factors if k in runtime['referenced_commodities']]
        if len(spot_keys) != 1:
            raise ValueError(
                f"Diagnostic CSV writer expects exactly one commodity-spot factor in the bundle; "
                f"got {spot_keys}. Multi-commodity diagnostic output is not yet implemented.")
        spot = bundle.factors[spot_keys[0]].detach().cpu().float()
        mtm_running = bundle.liability_mtm.detach().cpu().float()
        T, B = mtm_running.shape

        nonzero = (mtm_running != 0)
        last_nz = (nonzero * torch.arange(T).unsqueeze(1)).max(dim=0).values
        fill_mask = torch.arange(T).unsqueeze(1).expand(T, B) > last_nz.unsqueeze(0)
        realised = mtm_running.gather(0, last_nz.unsqueeze(0)).expand(T, B)
        mtm = torch.where(fill_mask, realised, mtm_running)

        # Per-step vol `(T, 1)` for the vol-scaled kappa — index-aligned with the full-grid marks.
        # None when no vol series was built; inert on the scalar-spread fast-path, so threading it
        # reconciles the reconstructed cost with the realized vol-scaled debit when Vol_Scale is on.
        vol_series = bundle.step_annual_vol
        diag_vol = None if vol_series is None else vol_series.detach().cpu().float().unsqueeze(-1)
        times = [int(t) for t in rollout['times']]
        per_instr = {}
        for name in self._instrument_order:
            cs = float(runtime['tradables'][name]['contract_size'])
            fut = bundle.tradables[name].detach().cpu().float()
            pos = torch.zeros((T, B)); trd = torch.zeros((T, B))
            cur = torch.zeros(B); j = 0
            rollout_pos = rollout['position'][name]
            rollout_trd = rollout['trade'][name]
            for t in range(T):
                if j < len(times) and t == times[j]:
                    trd[t] = rollout_trd[j].cpu().float()
                    cur = rollout_pos[j].cpu().float() + rollout_trd[j].cpu().float()
                    j += 1
                pos[t] = cur
            trade_cash = -trd * fut * cs
            trade_cost = trd.abs() * per_contract_kappa(runtime, fut, name, diag_vol)
            per_instr[name] = {
                'fut': fut, 'cs': cs, 'pos': pos, 'trd': trd,
                'trade_cash': trade_cash, 'trade_cost': trade_cost,
                'position_mtm': pos * fut * cs,
                'cum_cash': trade_cash.cumsum(0), 'cum_cost': trade_cost.cumsum(0),
            }
        portfolio_pos_mtm = sum(p['position_mtm'] for p in per_instr.values())
        portfolio_cum_cash = sum(p['cum_cash'] for p in per_instr.values())
        portfolio_cum_cost = sum(p['cum_cost'] for p in per_instr.values())
        # Roll rebate threaded with the SAME per-step vol as the cost above, so the matched-roll
        # credit reconstructs the realized vol-scaled rebate.
        if runtime['accounting']['roll_as_calendar_spread']:
            rebate = torch.stack([
                _roll_rebate({h: per_instr[h]['trd'][t] for h in self._hedges},
                             {h: per_instr[h]['fut'][t] for h in self._hedges}, runtime,
                             None if diag_vol is None else diag_vol[t])
                for t in range(T)])
            portfolio_cum_cost = portfolio_cum_cost - rebate.cumsum(0)
        hp = portfolio_cum_cash + portfolio_pos_mtm - portfolio_cum_cost
        return {
            'spot': spot, 'spread_bps': float(runtime['accounting']['bid_offer_spread_bps']),
            'per_instr': per_instr, 'mtm': mtm,
            'portfolio_position_mtm': portfolio_pos_mtm,
            'portfolio_cum_trade_cash': portfolio_cum_cash,
            'portfolio_cum_trade_cost': portfolio_cum_cost,
            'hedge_portfolio_ex_funding': hp,
            'total_ex_funding_discount': mtm + hp,
        }

    def _diag_write_paths(self, fields, label, csv_path):
        """Per-day per-instrument breakdown for 5 representative cases (worst/p5/mean/p95/best),
        selected by terminal `total_ex_funding_discount`."""
        total = fields['total_ex_funding_discount']
        T, B = total.shape
        sidx = torch.argsort(total[-1])
        cases = {'worst': int(sidx[0]), 'p5': int(sidx[round(0.05 * (B - 1))]),
                 'p95': int(sidx[round(0.95 * (B - 1))]), 'best': int(sidx[-1])}
        day_strs = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in self.bundle.scenario_dates]
        rows = []

        def _row(t, case, path_idx, getter):
            row = {'policy': label, 'case': case, 'path_idx': int(path_idx), 'day': day_strs[t],
                   'spot': getter(fields['spot'][t]).item(),
                   'spread_bps': fields['spread_bps']}
            for name in self._instrument_order:
                p = fields['per_instr'][name]
                row[f'{name}_futures'] = getter(p['fut'][t]).item()
                row[f'{name}_contract_size'] = p['cs']
                row[f'{name}_position'] = float(getter(p['pos'][t]).item())
                row[f'{name}_trade'] = float(getter(p['trd'][t]).item())
                row[f'{name}_trade_cash'] = float(getter(p['trade_cash'][t]).item())
                row[f'{name}_trade_cost'] = float(getter(p['trade_cost'][t]).item())
                row[f'{name}_position_mtm'] = float(getter(p['position_mtm'][t]).item())
            row['portfolio_position_mtm'] = float(getter(fields['portfolio_position_mtm'][t]).item())
            row['portfolio_cum_trade_cash'] = float(getter(fields['portfolio_cum_trade_cash'][t]).item())
            row['portfolio_cum_trade_cost'] = float(getter(fields['portfolio_cum_trade_cost'][t]).item())
            row['hedge_portfolio_ex_funding'] = float(getter(fields['hedge_portfolio_ex_funding'][t]).item())
            row['mtm_ex_post_settle_discount'] = float(getter(fields['mtm'][t]).item())
            row['total_ex_funding_discount'] = float(getter(fields['total_ex_funding_discount'][t]).item())
            return row

        sim_start = self.bundle.initial_time_index
        for case_name, idx in cases.items():
            for t in range(sim_start, T):
                rows.append(_row(t, case_name, idx, lambda x, idx=idx: x[idx]))
        for t in range(sim_start, T):
            rows.append(_row(t, 'mean', -1, lambda x: x.mean()))
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')


class HedgeRuntimeExecutionResult:
    """High-level result for HedgeMonteCarlo's hedge-bundle handoff.

    Carries the `Bundle` + normalized runtime + evaluation summary + the solver artifact
    (`policy_artifact` = DiffSolverV2's saved value-function nets, JSON-serializable) so
    downstream consumers (post-hoc analysis, streaming-service handlers) can do their own
    work without touching framework internals. `create_stepper()` spawns a `BundleStepper`
    to drive the simulator day-by-day with any explicit policy (e.g. the textbook hedge).
    """
    def __init__(self, *, bundle=None, runtime=None, evaluation_summary=None,
                 optimizer_diagnostics=None, policy_artifact=None, metadata=None):
        self.bundle = bundle
        self.runtime = runtime
        self.evaluation_summary = evaluation_summary
        self.optimizer_diagnostics = optimizer_diagnostics
        self.policy_artifact = policy_artifact
        self.metadata = metadata or {}

    def create_stepper(self) -> 'BundleStepper':
        """Spawn an interactive `BundleStepper` for the bundle. Lets client code drive
        the simulator one step at a time with arbitrary actions — useful for textbook
        hedges, custom policies, debugging, counterfactual what-ifs (deep-copy the
        stepper to fork branches)."""
        return BundleStepper(self.bundle, self.runtime)


def run_hedge_execution(bundle, runtime):
    """Roll the env forward with zero trades and report the terminal P&L summary — the unhedged
    baseline for `Execution_Mode='simulate_only'`. Callers drive an explicit policy on top via
    `HedgeRuntimeExecutionResult.create_stepper()`. `solve_hedge` does not come through here:
    `HedgeMonteCarlo.execute` drives `StreamingSolve` a batch at a time, and the mode itself is
    validated at the JSON boundary in `construct_hedge_runtime`."""
    started = time.perf_counter()
    stepper = BundleStepper(bundle, runtime)
    while not stepper.done:
        stepper.step(None)
    return {'policy': None, 'policy_artifact': None, 'optimizer_diagnostics': None,
            'evaluation_output': stepper.evaluation_output(
                timing={'evaluation_time_seconds': float(time.perf_counter() - started)})}
