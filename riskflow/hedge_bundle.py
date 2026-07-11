"""Hedging bundle + environment simulator + objective/utility stack.

Builds the simulated scenario bundle the solver consumes (`build_hedge_bundle`), the
mark-to-market env-step machinery (`build_shared_state` / `step_runtime_state` / the
cash & futures account steps / `reward_and_terminal_payoff`), the asymmetric-utility
objective stack (`_utility_wrap_signed`, `resolve_utility_scale`, …), and the
interactive `BundleStepper` for day-by-day replay of any explicit policy. `run_hedge_execution`
dispatches by Execution_Mode: 'solve_hedge' → the differential-ML solver (hedge_solver),
'simulate_only' → the no-trade baseline. (The PPO/policy-gradient track was removed.)
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import torch

from .hedge_runtime import assemble_privileged_factors, per_contract_kappa


# --- spot realized-vol + timeline (relocated from the deleted hedge_features.py); feed the
#     symlog utility scale c in resolve_utility_scale ------------------------------------------ #
PRICE_ZSCORE_WINDOW = 20


def _full_spot_timeline(bundle, commodity):
    """Return the (H+T_sim, B) spot tensor for `commodity`, made of JSON history (rows 0..H-1)
    concatenated with the simulator's CommodityPrice factor (rows H..). `commodity` is the
    canonical factor name produced by `utils.check_tuple_name` (e.g. 'CommodityPrice.PLATINUM');
    the bundle's `spot_price_history` and `factors` dicts both key by that same form."""
    spot_history = bundle.get('spot_price_history') or {}
    factors = bundle.get('factors') or {}
    hist = spot_history.get(commodity)
    sim = factors.get(commodity)
    if hist is None or sim is None:
        return None
    sim_full = sim.to(dtype=torch.float32)
    hist_t = hist.to(dtype=torch.float32, device=sim_full.device)
    H = int(hist_t.shape[0])
    return torch.cat([hist_t, sim_full[H:]], dim=0)


def compute_spot_realized_vol(bundle, window=PRICE_ZSCORE_WINDOW, min_periods=5, eps=1e-6):
    """Annualized rolling realized log-vol of underlying spot. In MR regimes with σ scaled to
    keep stationary std fixed, realized vol increases with kappa — making this a regime signal."""
    spot_history = bundle.get('spot_price_history') or {}
    if not spot_history:
        return {}
    out = {}
    for commodity in spot_history.keys():
        S = _full_spot_timeline(bundle, commodity)
        if S is None:
            continue
        log_S = S.clamp_min(1e-9).log()
        log_ret = log_S[1:] - log_S[:-1]  # (T_full - 1, B)
        ret_sq = torch.cat([torch.zeros_like(log_ret[:1]), log_ret * log_ret], dim=0)  # (T_full, B)
        cum = torch.cat([torch.zeros_like(ret_sq[:1]), ret_sq.cumsum(dim=0)], dim=0)
        T = ret_sq.shape[0]
        idx = torch.arange(T + 1, device=S.device)
        lo = (idx - window).clamp_min(0)
        count = (idx - lo).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
        sum_w = cum[idx[1:]] - cum[lo[1:]]
        rv = (252.0 * sum_w / count[1:]).clamp_min(0.0).sqrt()
        rv[:min_periods] = 0.0
        out[commodity] = rv
    return out


def _batch_size_from_bundle(bundle):
    tradables = bundle.get("tradables", {})
    for tensor in tradables.values():
        return int(tensor.shape[-1])
    factors = bundle.get("factors", {})
    for tensor in factors.values():
        return int(tensor.shape[-1])
    return 0


def _zeros_by_name(names, batch_size, *, device, dtype=torch.float32):
    return {str(name): torch.zeros(batch_size, dtype=dtype, device=device) for name in names}


def _seed_by_name(seed_values, names, batch_size, *, device, dtype=torch.float32):
    return {str(name): torch.full((batch_size,), float(seed_values.get(str(name), 0.0)), dtype=dtype, device=device) for name in names}


def _runtime_names(runtime, key):
    # Returns the underlying tuple from runtime["names"] — read-only by all callers.
    return runtime.get("names", {}).get(key, ())


def _runtime_tradable(runtime, name):
    # Returns the underlying dict from runtime["tradables"][name] — read-only by all callers.
    return runtime.get("tradables", {}).get(str(name), {})


def _current_liability_mtm(bundle, time_index):
    liability_mtm = bundle.get("liability_mtm")
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    if liability_mtm is None:
        return torch.zeros(batch_size, dtype=torch.float32, device=device)
    return liability_mtm[time_index].to(dtype=torch.float32, device=device)


def _current_realized_cashflow_total(bundle, time_index):
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    total = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for tensor in (bundle.get("realized_cashflows") or {}).values():
        total = total + tensor[time_index].to(dtype=torch.float32, device=device)
    return total


def _position_value_total(state, runtime):
    total = torch.zeros_like(state["done"], dtype=torch.float32)
    tradable_values = state.get("tradable_values", {})
    for name in _runtime_names(runtime, "hedges"):
        position = state["positions"][name].to(dtype=torch.float32)
        price = tradable_values[name].to(dtype=torch.float32)
        contract_size = float(_runtime_tradable(runtime, name).get("contract_size", 1.0))
        total = total + position * price * contract_size
    return total


def _current_tradable_values(bundle, runtime, time_index):
    tradables = bundle.get("tradables", {})
    return {n: tradables[str(n)][time_index].to(dtype=torch.float32) for n in _runtime_names(runtime, "tradables") if n in tradables}


def _refresh_state_views(state, bundle, runtime, time_index):
    """Cheap per-step refresh: refreshes price/liability views and carries forward simulator state."""
    tradable_values = _current_tradable_values(bundle, runtime, time_index)
    sample = next(iter(state["positions"].values()))
    batch_size = int(sample.shape[0])
    device = sample.device
    return {
        "time_index": int(time_index),
        "done": state["done"],
        "positions": state["positions"],
        "cash_accounts": state["cash_accounts"],
        "margin_accounts": state["margin_accounts"],
        "realized_pnl": state["realized_pnl"],
        "variation_margin": state["variation_margin"],
        "cumulative_pnl": state.get("cumulative_pnl", _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)),
        "time_held": state.get("time_held", _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)),
        "settlement_prices": state["settlement_prices"],
        "tradable_values": tradable_values,
        "liability_mtm_value": _current_liability_mtm(bundle, time_index),
        "realized_cashflow_value": _current_realized_cashflow_total(bundle, time_index),
        "cumulative_liability_value": state.get(
            "cumulative_liability_value",
            torch.zeros(batch_size, dtype=torch.float32, device=device),
        ).to(dtype=torch.float32),
        "initial_portfolio_value": state.get("initial_portfolio_value"),
    }


def _sum_account_balances(accounts):
    # Cash and margin balances are stored as raw dollars and compounded daily (each step we
    # multiply by cash_tv(next)/cash_tv(curr) ≈ 1 + SOFR·dt, then add today's flows). That gives
    # each dollar interest only from the day it landed — not from t=0 the way a single multiply
    # by tradable_value(t) = 1/D(t) at portfolio-value time would.
    total = None
    for tensor in accounts.values():
        v = tensor.to(dtype=torch.float32)
        total = v if total is None else total + v
    return total


def _daily_growth_factors(bundle, runtime, current, next_idx):
    """Per-cash-account one-day growth factor for compounding cash/margin balances overnight.

    The cash tradable value is `Units / D(t) × fx_rep`, so the ratio
    `tradable_value(next_idx) / tradable_value(current) = D(current) / D(next_idx)` is the one-day
    growth factor (≈ 1 + SOFR · dt). Returns {} when next_idx == current so terminal steps pass
    balances through unchanged."""
    if current >= next_idx:
        return {}
    tv_curr = _current_tradable_values(bundle, runtime, current)
    tv_next = _current_tradable_values(bundle, runtime, next_idx)
    out = {}
    for name in _runtime_names(runtime, "cash_accounts"):
        n = str(name)
        if n in tv_curr and n in tv_next:
            out[n] = tv_next[n] / tv_curr[n]
    return out


def _compound_accounts(accounts, factors):
    """Multiply each balance by its one-day growth factor (or pass through reference if no
    factor present). Pass-through is safe: callers never mutate balances in place — they
    rebind via `next_state[key] = new_tensor` or via `_apply_account_flow` (which writes a
    new tensor into the dict, not in-place into the existing tensor)."""
    return {n: (bal * factors[n] if n in factors else bal) for n, bal in accounts.items()}


def _portfolio_value(state, runtime):
    """Absolute total wealth (cash + margin + unrealized VM + position value where applicable).
    Use `_pnl_excess` to get wealth change since inception — that's what the asymmetric utility
    and dense tracking reward need so the floor at zero correctly discriminates loss from gain."""
    accounting_mode = str(runtime.get("accounting_mode", "futures"))
    position_total = _position_value_total(state, runtime)
    if accounting_mode == "cash_account":
        cash_total = _sum_account_balances(state.get("cash_accounts", {}))
        if cash_total is None:
            cash_total = torch.zeros_like(position_total)
        return position_total + cash_total
    # Futures mode: cash = starting capital (frozen); margin = all VM and trade-cost flows since
    # inception; unrealized_vm = position × (current_price - last_settlement_price) × cs captures
    # the gap between the most recent settlement and the current observable price (typically zero
    # within an episode, but non-zero at t=0 when the book starts with an overnight position whose
    # prior settlement is yesterday's close).
    total = torch.zeros_like(position_total)
    margin_total = _sum_account_balances(state.get("margin_accounts", {}))
    if margin_total is not None:
        total = total + margin_total
    cash_total = _sum_account_balances(state.get("cash_accounts", {}))
    if cash_total is not None:
        total = total + cash_total
    settlement_prices = state.get("settlement_prices", {})
    tradable_values = state.get("tradable_values", {})
    for name in _runtime_names(runtime, "hedges"):
        n = str(name)
        if n not in settlement_prices or n not in tradable_values:
            continue
        pos = state["positions"][n].to(dtype=torch.float32)
        price = tradable_values[n].to(dtype=torch.float32)
        settlement = settlement_prices[n].to(dtype=torch.float32)
        contract_size = float(_runtime_tradable(runtime, n).get("contract_size", 1.0))
        total = total + pos * (price - settlement) * contract_size
    return total


def _pnl_excess(state, runtime):
    """Wealth change since inception: portfolio_value - initial_portfolio_value. Used by the
    asymmetric utility, dense tracking reward, and reported metrics — anywhere we want net P&L
    rather than absolute wealth so the floor at zero is meaningful. The initial baseline is
    snapshotted in build_shared_state and threaded through state across all step transitions."""
    pv = _portfolio_value(state, runtime)
    initial = state.get("initial_portfolio_value")
    if initial is None:
        return pv
    return pv - initial.to(device=pv.device, dtype=pv.dtype)


def _tracking_error_value(state, runtime):
    # Optimal hedge keeps pnl_excess + (liability_mtm + cumulative_liability_value) ≈ 0 at every
    # step. liability_mtm alone drops by the cashflow amount on payment dates (the cashflow moves
    # to realized cash, summed into cumulative_liability_value), which would otherwise inject a
    # ±cf shock into the dense reward unrelated to any action. The sum across the two channels
    # is continuous across the payment boundary, matching the terminal-reward invariant
    # (`pnl_excess + cumulative_liability_value` once everything has been paid).
    pnl_excess = _pnl_excess(state, runtime).to(dtype=torch.float32)
    liability_mtm = state.get("liability_mtm_value", torch.zeros_like(pnl_excess)).to(dtype=torch.float32, device=pnl_excess.device)
    cumulative_liability = state.get("cumulative_liability_value", torch.zeros_like(pnl_excess)).to(dtype=torch.float32, device=pnl_excess.device)
    return pnl_excess + liability_mtm + cumulative_liability


def wealth_step(W, q, contract_size, dF, dL):
    """The ONE analytic hedged-wealth transition — frictionless telescoping core.

        W_{t+1} = W_t + Σ_i q_i·cs_i·dF_i + dL,   W_t = cumulative hedge P&L + marked liability L_t

    `q` (…,n_hedge) is the per-instrument position, `dF` (…,n_hedge) the per-instrument price
    move F_{t+1}−F_t, `dL` (…) the marked-liability change L_{t+1}−L_t. This is the frictionless
    law DiffSolverV2 rolls (bank/verdict) and, crucially, the one the twin loss DIFFERENTIATES:
    u(W_{t+1}) is taped back to a wealth leaf + the state-at-t market leaves, so this MUST stay a
    pure tensor op — no .item()/.detach()/.cpu()/.to(); callers own the grad context.

    `futures_account_step` is the DEPLOYMENT discretization of this same law: it books the same
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
# `Objective.Object`; the legacy `TerminalFloorThenSurplusUtility` is the identity path.
# All consume the same scale `c` and live in "utility space" (the DP / value-fn recursion).
_UTILITY_OBJECTS = (
    "asymmetricutility_symlog", "asymmetricutility_huber", "asymmetricutility_cara")


def _is_symlog_objective(runtime):
    # TRUE symlog only — for symlog-SPECIFIC diagnostics (e.g. the −45 saturation tripwire,
    # the log1p-floor penalty-bite report), which don't transfer to huber/cara shapes.
    # `objective["object"]` is canonical-lowercased at normalization time
    # (hedge_runtime._normalize_objective_config), so plain equality is sufficient here.
    obj = (runtime.get("objective") or {})
    return obj.get("object") == "asymmetricutility_symlog"


def _is_utility_objective(runtime):
    """True iff the objective transforms wealth through a utility shape (symlog/huber/cara) —
    so the DP/value-fn live in utility space and a scale `c` is required. False for the legacy
    identity objective."""
    return (runtime.get("objective") or {}).get("object") in _UTILITY_OBJECTS


def _mirror_utility_scale_to_runtime(bundle, runtime):
    """Cache `bundle["utility_scale"]` on `runtime["objective"]["utility_scale"]` so the
    reward / penalty functions can read `c` without taking `bundle` in their signatures.
    Invariant: `runtime["objective"]["utility_scale"] == bundle["utility_scale"]` for the
    duration of any rollout that computes rewards (PPO, no-trade baseline)."""
    if "utility_scale" in bundle and runtime.get("objective") is not None:
        runtime["objective"]["utility_scale"] = float(bundle["utility_scale"])


def _require_utility_scale(runtime):
    """Pull the symlog scale `c` from runtime. Fails loud if missing — the silent fallback
    (`c = 1.0`) would produce plausible-looking-but-wrong rewards (e.g. log1p($1M / 1.0) ≈ 14)
    with no error at the call site, masking any caller that bypassed the mirror.

    Mirroring `bundle["utility_scale"]` is the responsibility of
    `_mirror_utility_scale_to_runtime`, invoked at the start of every rollout that computes
    rewards (PPO + no-trade). Custom rollout callers must do the same."""
    c = (runtime.get("objective") or {}).get("utility_scale")
    if c is None:
        raise ValueError(
            "symlog objective active but runtime['objective']['utility_scale'] is not set — "
            "_mirror_utility_scale_to_runtime(bundle, runtime) must run before any reward "
            "computation. Either call it in your custom rollout, or set utility_scale "
            "explicitly on the runtime."
        )
    return float(c)


def _utility_wrap_signed(x_dollars, runtime):
    """The terminal UTILITY u(W) applied to signed wealth — the single source of truth for
    the objective, the DP recursion, and the solver's value labels. Dispatches on the
    configured shape (all in normalised wealth x = W / c), identity for the legacy objective:

      symlog : sign(x)·log1p(|x|)                  — odd, tail-compressing (variance aversion)
      huber  : x − [a·loss² | a·δ²+2aδ(loss−δ)]    — linear gains; quadratic small losses;
               (loss = max(−x,0), knee δ)             linear deep tail (bounded scale, live grad)
      cara   : (1 − exp(−γ·x)) / γ                  — bounded gains, exponentially-penalised loss

    Shape params (huber a/δ, cara γ) are DIMENSIONLESS in c-units. Differentiable in `w`
    (AAD path: twin-loss labels, DP penalty, baseline B) — huber's knee is C¹, cara is smooth."""
    if not _is_utility_objective(runtime):
        return x_dollars
    obj = runtime.get("objective") or {}
    shape = obj.get("object")
    x = x_dollars / _require_utility_scale(runtime)
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


def resolve_utility_scale(bundle, runtime):
    """Compute the dollar scale `c` used by `AsymmetricUtility_Symlog` to map dollars to
    log-utility: u(x; c) = sign(x) · log1p(|x| / c). Single source of truth — every reward
    and penalty reads `bundle["utility_scale"]`. Called once at bundle build.

    Modes (from `Objective.Utility_Scale_Mode`):
      - `"vol_scaled_notional"` (default): c = total_leg_volume × initial_spot × σ_annual × √τ
      - explicit override via `Objective.Utility_Scale_Explicit` (dollars).

    Behavior under degeneracies:
      - Symlog active (`Object == AsymmetricUtility_Symlog`): raises on any path that would
        return the $1k floor. A floor-c symlog silently breaks tail compression
        (log1p($1M/$1k) ≈ 7 ≈ log1p($100M/$1k) ≈ 11.5 — 100× dollar gap → 1.6× utility gap),
        which defeats the whole point. Fail-loud is consistent with `_require_utility_scale`.
      - Legacy objective (doesn't consume c): returns $1k harmlessly.

    Single-commodity scope: uses the unique declared underlying.
    """
    objective = (runtime.get('objective') or {})
    needs_scale = _is_utility_objective(runtime)   # symlog / huber / cara all consume c

    def _degenerate(reason):
        """Either raise (a utility shape — degraded c silently breaks the wealth scaling) or
        return the $1k floor (legacy — c isn't consumed)."""
        if needs_scale:
            raise ValueError(
                f"resolve_utility_scale: cannot compute a meaningful c — {reason}. "
                "A floor-c symlog silently compresses tails and defeats the reward shape. "
                "Fix the bundle/config, or set Objective.Utility_Scale_Explicit to a "
                "literal dollar value."
            )
        return 1.0e3

    mode = str(objective.get('utility_scale_mode', 'vol_scaled_notional')).lower()
    if mode != 'vol_scaled_notional':
        raise ValueError(
            f"Unsupported Objective.Utility_Scale_Mode: {mode!r}. "
            "Supported modes: 'vol_scaled_notional'. Set Utility_Scale_Explicit to "
            "override the formula with a literal dollar value."
        )
    explicit = objective.get('utility_scale_explicit')
    if explicit is not None:
        # `Utility_Scale_Explicit` is an EXPLICIT override — honor it exactly,
        # including values below the $1k production-default floor. Silently
        # clamping to $1k would make a cell-by-cell oracle comparison fail
        # for a reason unrelated to the method. The $1k floor only applies to
        # the formula path (`vol_scaled_notional`) where a small c would indicate
        # a degenerate calc; explicit user input gets trust.
        c_explicit = float(explicit)
        if c_explicit < 1.0e3:
            logging.info(
                'utility_scale Explicit override: c=%.4g (below the $1k '
                'production floor — test mode; trust mode active)', c_explicit)
        return c_explicit
    H = int(runtime.get('history_lookback_business_days', 0))
    last_idx = bundle.get('last_settlement_index')
    if last_idx is None:
        return _degenerate("last_settlement_index missing from bundle")
    tau_years = max(float(int(last_idx) - H) / 252.0, 1.0 / 252.0)
    total_volume = float(bundle.get('total_leg_volume', 0.0))
    spot_history = bundle.get('spot_price_history') or {}
    rv_by_commodity = bundle.get('spot_realized_vol') or {}
    if not spot_history:
        return _degenerate("bundle['spot_price_history'] is empty")
    if not total_volume:
        return _degenerate("bundle['total_leg_volume'] is zero")
    commodity = next(iter(spot_history))
    full = _full_spot_timeline(bundle, commodity)
    if full is None or H >= int(full.shape[0]):
        return _degenerate(
            f"spot timeline for {commodity!r} has length "
            f"{0 if full is None else int(full.shape[0])} ≤ history_lookback H={H}"
        )
    # Take batch-median at index H (history/sim boundary) rather than slot [H, 0]. The
    # rolling-vol window at H spans broadcast history rows, so all batch entries are equal
    # in well-behaved cases — but `full[H]` is the FIRST sim step, and any process emitting
    # a stochastic initial draw (some HMM variants, jump-diffusion with state-sampling)
    # would silently make c path-dependent off slot 0. Median is defensive: deterministic
    # when batch is uniform, robust when it's not. Negligible cost (one (B,) reduce).
    initial_spot = float(full[H].median().item())
    rv = rv_by_commodity.get(commodity)
    sigma = float(rv[H].median().item()) if rv is not None and H < int(rv.shape[0]) else 0.0
    if sigma <= 0.0:
        return _degenerate(
            f"realized vol for {commodity!r} is non-positive (σ={sigma}) — "
            "likely early-calibration window or upstream calc bug"
        )
    c = total_volume * initial_spot * sigma * (tau_years ** 0.5)
    if c < 1.0e3:
        return _degenerate(
            f"formula produced c=${c:,.2f} < $1k floor "
            f"(volume={total_volume}, spot={initial_spot:.2f}, σ={sigma:.4f}, √τ={tau_years**0.5:.3f})"
        )
    return c


def _cash_account_for_instrument(name, runtime):
    # Precomputed at runtime construction (hedge_runtime._build_instrument_cash_account_map).
    return runtime.get("accounting", {}).get("instrument_to_cash_account", {}).get(str(name))


def _coerce_batch_tensor(value, batch_size, *, device):
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        return tensor.repeat(batch_size)
    return tensor


def _resolve_trade_deltas(action, runtime, *, batch_size, device):
    if action is None:
        return _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    resolved = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    for name, value in action["trade_deltas"].items():
        n = str(name)
        if n in resolved:
            resolved[n] = _coerce_batch_tensor(value, batch_size, device=device)
    return resolved


def _realized_structured_action(action, current_positions, runtime, *, batch_size, device):
    if action is None:
        return None
    executed = _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device)
    instrument_order = tuple(str(n) for n in _runtime_names(runtime, "action_instruments"))
    ordered = torch.stack([executed[n] for n in instrument_order], dim=1).round().to(dtype=torch.int64)
    # Trades are integer contracts; position limits are enforced reward-side, not clipped here.
    return {
        **action,
        "trade_deltas": {n: ordered[:, i] for i, n in enumerate(instrument_order)},
    }


def _transaction_costs(trade_delta, price, runtime, name):
    # Realized debit on |Δq| contracts at the turnover-cost rule (hedge_runtime.per_contract_kappa).
    return trade_delta.abs() * per_contract_kappa(runtime, price.abs(), name)


def _apply_cash_trade(cash_accounts, account_name, trade_delta, price, transaction_cost, contract_size):
    if account_name is None:
        return
    # Notional debit is `delta × price × contract_size` — each contract is `contract_size` units of
    # the underlying. Without contract_size the cash leg was off by exactly that factor (50 for
    # platinum at $2050: $2050 instead of $102,500 per contract).
    cash_accounts[account_name] = cash_accounts[account_name] - (trade_delta * price * contract_size + transaction_cost)


def _apply_account_flow(accounts, account_name, amount):
    if account_name is None:
        return
    accounts[account_name] = accounts[account_name] + amount


def _last_time_index(bundle):
    return max(int(bundle["time_grid_days"].shape[0]) - 1, 0)


def _should_terminal_flatten(runtime, current, last):
    return bool(runtime.get("accounting", {}).get("force_flat_at_end", False)) and current >= last - 1


def _step_time_held(time_held, next_positions):
    # Increment time-held counter where position is non-zero, reset where it returned to zero.
    out = {}
    for name, prev in time_held.items():
        active = next_positions[name].abs() > 0
        out[name] = torch.where(active, prev + 1.0, torch.zeros_like(prev))
    return out


def _step_cumulative_pnl(cumulative_pnl, variation_margin, next_positions):
    """Per-trade running P&L: accumulates VM while a position is open, resets to 0 when flat.
    Mirrors the lifetime-of-current-trade semantics of `_step_time_held` so the policy sees
    consistent features describing the *current* open position rather than mixed
    since-inception (cum_pnl) / per-trade (time_held) scopes."""
    out = {}
    for name, prev in cumulative_pnl.items():
        active = next_positions[name].abs() > 0
        out[name] = torch.where(active, prev + variation_margin[name], torch.zeros_like(prev))
    return out


def _flatten_cash_inventory(positions, cash_accounts, terminal_values, runtime):
    for name in _runtime_names(runtime, "hedges"):
        n = str(name)
        delta = -positions[n]
        cost = _transaction_costs(delta, terminal_values[n], runtime, n)
        cs = float(_runtime_tradable(runtime, n).get("contract_size", 1.0))
        _apply_cash_trade(cash_accounts, _cash_account_for_instrument(n, runtime), delta, terminal_values[n], cost, cs)
        positions[n] = positions[n] + delta


def _flatten_futures_inventory(positions, margin_accounts, settlement_prices, terminal_values, runtime):
    """Close hedge positions at `terminal_values` and capture any residual variation margin.

    `settlement_prices` is the price each position last settled at; `terminal_values` is the
    price at which we're closing. The residual VM = position × (terminal - settlement) × cs is
    applied to margin to ensure no PnL leaks between the last settlement and close. When the
    caller has already advanced settlement to terminal (current futures_account_step path), the
    residual is 0 and only the trade cost is debited; otherwise the residual is captured here.
    Trade cost is always computed at `terminal_values` (the actual close price)."""
    for name in _runtime_names(runtime, "hedges"):
        n = str(name)
        cs = float(_runtime_tradable(runtime, n).get("contract_size", 1.0))
        residual_vm = positions[n] * (terminal_values[n] - settlement_prices[n]) * cs
        delta = -positions[n]
        cost = _transaction_costs(delta, terminal_values[n], runtime, n)
        account = _cash_account_for_instrument(n, runtime)
        _apply_account_flow(margin_accounts, account, residual_vm)
        _apply_account_flow(margin_accounts, account, -cost)
        positions[n] = positions[n] + delta


def build_shared_state(bundle, runtime):
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    portfolio_state = runtime.get("portfolio_state") or {}
    positions = _seed_by_name(portfolio_state.get("positions", {}), _runtime_names(runtime, "hedges"), batch_size, device=device)
    cash_accounts = _seed_by_name(portfolio_state.get("cash_balances", {}), _runtime_names(runtime, "cash_accounts"), batch_size, device=device)
    margin_accounts = _seed_by_name(portfolio_state.get("margin_balances", {}), _runtime_names(runtime, "cash_accounts"), batch_size, device=device)
    # Start simulation state at sim-day-0 (= bundle row H) so the prefix-injected historical rows
    # are only used by features (z-score, cross-delta, momentum) — never visited by the simulator
    # itself. JSON-supplied positions are "today's overnight book", not "30 days ago".
    initial_time_index = int(bundle["initial_time_index"])
    seeded_settlement = portfolio_state.get("settlement_prices", {})
    fallback = _current_tradable_values(bundle, runtime, initial_time_index)
    settlement_prices = {
        name: torch.full((batch_size,), float(seeded_settlement[name]), dtype=torch.float32, device=device)
        if name in seeded_settlement
        else fallback[name].clone()
        for name in _runtime_names(runtime, "hedges")
        if name in fallback
    }
    state = {
        "done": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "positions": positions,
        "cash_accounts": cash_accounts,
        "margin_accounts": margin_accounts,
        "realized_pnl": _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device),
        "variation_margin": _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device),
        "cumulative_pnl": _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device),
        "time_held": _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device),
        "cumulative_liability_value": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "settlement_prices": settlement_prices,
    }
    refreshed = _refresh_state_views(state, bundle, runtime, initial_time_index)
    refreshed["cumulative_liability_value"] = refreshed["cumulative_liability_value"] + refreshed["realized_cashflow_value"]
    # Snapshot the inception baseline (cash + margin + initial unrealized VM if positions started
    # non-zero with a stale settlement). Threaded through state so _pnl_excess returns the change.
    refreshed["initial_portfolio_value"] = _portfolio_value(refreshed, runtime).detach().clone()
    # Re-seat settlement_prices to the simulator's price at sim-day-0. The seed (yesterday's
    # close) vs sim-day-0 forward gap was just absorbed into initial_portfolio_value via the
    # unrealized-VM term in _portfolio_value, so subsequent steps' VM is clean step-over-step
    # P&L. Without this the first trade carries a seed-gap noise of (price_H − seed) × delta × cs
    # — typically a few hundred dollars per contract, raising the variance floor on advantages
    # at every first decision.
    for name, current_price in refreshed["tradable_values"].items():
        if name in refreshed["settlement_prices"]:
            refreshed["settlement_prices"][name] = current_price.detach().clone()
    return refreshed


def cash_account_step(state, action, bundle, runtime):
    # `done` is purely a function of time_index vs. last bundle index; checking the Python int
    # avoids a CUDA-CPU sync that would fire on every step.
    if int(state["time_index"]) >= _last_time_index(bundle):
        return state
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    current = int(state["time_index"])
    last = _last_time_index(bundle)
    next_idx = min(current + 1, last)
    # Shallow dict copy: we'll replace tensors in-slot via `next_positions[n] = ...`. Cloning the
    # tensors themselves is wasted since each is immediately overwritten with a new tensor.
    next_positions = dict(state["positions"])
    growth = _daily_growth_factors(bundle, runtime, current, next_idx)
    next_cash = _compound_accounts(state["cash_accounts"], growth)
    deltas = _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device)
    for name in _runtime_names(runtime, "action_instruments"):
        n = str(name)
        delta = deltas.get(n)
        price = state["tradable_values"][n]
        cost = _transaction_costs(delta, price, runtime, n)
        cs = float(_runtime_tradable(runtime, n).get("contract_size", 1.0))
        _apply_cash_trade(next_cash, _cash_account_for_instrument(n, runtime), delta, price, cost, cs)
        next_positions[n] = (next_positions[n] + delta).round()
    if _should_terminal_flatten(runtime, current, last):
        _flatten_cash_inventory(next_positions, next_cash, _current_tradable_values(bundle, runtime, last), runtime)
    # cash-account mode tracks no daily VM — cumulative_pnl can only be added to in futures
    # mode. Still reset at flat for the same per-trade-lifetime semantics as time_held.
    zero_vm = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    next_cumulative_pnl = _step_cumulative_pnl(state["cumulative_pnl"], zero_vm, next_positions)
    next_time_held = _step_time_held(state["time_held"], next_positions)
    # Cash mode: no daily VM accrued (realized_pnl / variation_margin always zero).
    # margin_accounts / settlement_prices are not modified — pass through references.
    hedge_names = _runtime_names(runtime, "hedges")
    next_state = {
        "done": torch.full_like(state["done"], next_idx >= last, dtype=torch.bool),
        "positions": next_positions,
        "cash_accounts": next_cash,
        "margin_accounts": state["margin_accounts"],
        "realized_pnl": _zeros_by_name(hedge_names, batch_size, device=device),
        "variation_margin": _zeros_by_name(hedge_names, batch_size, device=device),
        "cumulative_pnl": next_cumulative_pnl,
        "time_held": next_time_held,
        "cumulative_liability_value": state["cumulative_liability_value"],
        "settlement_prices": state["settlement_prices"],
        "initial_portfolio_value": state["initial_portfolio_value"],
    }
    refreshed = _refresh_state_views(next_state, bundle, runtime, next_idx)
    refreshed["cumulative_liability_value"] = refreshed["cumulative_liability_value"] + refreshed["realized_cashflow_value"]
    return refreshed


def futures_account_step(state, action, bundle, runtime):
    # See cash_account_step — sync-free guard via time_index.
    if int(state["time_index"]) >= _last_time_index(bundle):
        return state
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    current = int(state["time_index"])
    last = _last_time_index(bundle)
    settlement_idx = min(current + 1, last)
    # Shallow dict copies — we replace tensors in-slot via `next_positions[n] = ...` and
    # `next_settlement[n] = ...`. The original tensors are never mutated in place anywhere in
    # this codebase, so reusing references is safe; cloning every step would just burn memory
    # bandwidth.
    next_positions = dict(state["positions"])
    growth = _daily_growth_factors(bundle, runtime, current, settlement_idx)
    next_cash = _compound_accounts(state["cash_accounts"], growth)
    next_margin = _compound_accounts(state["margin_accounts"], growth)
    next_settlement = dict(state["settlement_prices"])
    realized_pnl = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    variation_margin = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    deltas = _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device)
    next_values = _current_tradable_values(bundle, runtime, settlement_idx)
    # Futures: cash_accounts is frozen at starting capital; only margin tracks VM and trade cost.
    # Apply the trade BEFORE computing VM so the new delta participates in the price(t)→price(t+1)
    # accrual: the agent transacted at decision-time price (= settlement_old in steady state, since
    # the previous step set settlement to that step's tradable_values), and after this step the
    # whole post-trade position is marked at price(t+1). Pre-fix, VM was computed on pre-trade
    # position only, so `delta × (next_value − decision_price) × cs` was silently dropped — mean
    # zero but ~$25k std/path noise on advantages.
    for name in _runtime_names(runtime, "hedges"):
        n = str(name)
        cs = float(_runtime_tradable(runtime, n).get("contract_size", 1.0))
        # Round to integer contracts to prevent float drift from accumulating across steps —
        # otherwise the policy's int64-cast feasibility mask can disagree with the env's
        # float-clamped position limits and decouple recorded action_bins from realized trades.
        next_positions[n] = (next_positions[n] + deltas[n]).round()
        vm = next_positions[n] * (next_values[n] - state["settlement_prices"][n]) * cs
        realized_pnl[n] = vm
        variation_margin[n] = vm
        next_settlement[n] = next_values[n].clone()
        account = _cash_account_for_instrument(n, runtime)
        _apply_account_flow(next_margin, account, vm)
    for name in _runtime_names(runtime, "action_instruments"):
        n = str(name)
        delta = deltas.get(n)
        # Trade cost references the price the agent actually saw and acted on (decision-time).
        # Using next_values[n] would let unrelated overnight mid moves distort the spread cost.
        cost = _transaction_costs(delta, state["tradable_values"][n], runtime, n)
        account = _cash_account_for_instrument(n, runtime)
        _apply_account_flow(next_margin, account, -cost)
    if _should_terminal_flatten(runtime, current, last):
        _flatten_futures_inventory(next_positions, next_margin, next_settlement, next_values, runtime)
    next_cumulative_pnl = _step_cumulative_pnl(state["cumulative_pnl"], variation_margin, next_positions)
    next_time_held = _step_time_held(state["time_held"], next_positions)
    next_state = {
        "done": torch.full_like(state["done"], settlement_idx >= last, dtype=torch.bool),
        "positions": next_positions,
        "cash_accounts": next_cash,
        "settlement_prices": next_settlement,
        "margin_accounts": next_margin,
        "realized_pnl": realized_pnl,
        "variation_margin": variation_margin,
        "cumulative_pnl": next_cumulative_pnl,
        "time_held": next_time_held,
        # cumulative_liability_value gets `= old + realized_cashflow_value` on the line below —
        # new tensor, no in-place. initial_portfolio_value is read-only after build_shared_state.
        "cumulative_liability_value": state["cumulative_liability_value"],
        "initial_portfolio_value": state["initial_portfolio_value"],
    }
    refreshed = _refresh_state_views(next_state, bundle, runtime, settlement_idx)
    refreshed["cumulative_liability_value"] = refreshed["cumulative_liability_value"] + refreshed["realized_cashflow_value"]
    return refreshed


def step_runtime_state(state, action, bundle, runtime):
    if str(runtime.get("accounting_mode", "futures")) == "cash_account":
        return cash_account_step(state, action, bundle, runtime)
    return futures_account_step(state, action, bundle, runtime)


def reward_and_terminal_payoff(prev_state, state, bundle, runtime):
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    is_terminal = int(state["time_index"]) >= _last_time_index(bundle)
    pnl_excess = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liability_value = state.get("cumulative_liability_value", torch.zeros(batch_size, dtype=torch.float32, device=device)).to(device=device, dtype=torch.float32)
    # Sync-free terminality check: done becomes True iff time_index hits last_idx, and is uniform
    # across the batch (the simulator advances all scenarios together). Avoids a `.item()` sync on
    # every rollout step.
    if is_terminal:
        # pnl_excess = portfolio change since inception (NOT absolute portfolio value, so the seed
        # cash baseline doesn't bias the terminal P&L read the solver/stepper consume).
        pnl_excess = _pnl_excess(state, runtime).to(device=device, dtype=torch.float32)
    return {"pnl_excess": pnl_excess, "liability_value": liability_value}


def _bundle_scenario_dates(bundle):
    # Cached at bundle build; falls back to recompute only if a caller passes a bundle that
    # never went through `build_hedge_bundle` (e.g. handcrafted test fixtures).
    cached = bundle.get("scenario_dates")
    if cached is not None:
        return cached
    base_date = pd.Timestamp(bundle.get("meta", {}).get("base_date"))
    days = torch.as_tensor(bundle["time_grid_days"]).detach().cpu().to(dtype=torch.int64).tolist()
    return pd.DatetimeIndex([base_date + pd.Timedelta(days=int(d)) for d in days])


def _decision_time_indices(bundle):
    """Business-day rollout indices where the stepper accepts trades — every business day in
    the live-sim window (history rows excluded). Precomputed at bundle build."""
    return tuple(bundle.get("business_indices", ()))


def _collect_no_trade_rollout(bundle, runtime):
    state = build_shared_state(bundle, runtime)
    _mirror_utility_scale_to_runtime(bundle, runtime)
    last_idx = _last_time_index(bundle)
    terminal = None
    while int(state["time_index"]) < last_idx:
        next_state = step_runtime_state(state, None, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        terminal = {"state": next_state, **transition}
        state = next_state
    return terminal


def _cpu_tensor_dict(values):
    return {str(n): torch.as_tensor(t).detach().to(dtype=torch.float32).cpu() for n, t in values.items()}


def _terminal_summary(state, terminal_transition, bundle):
    # `pnl_excess` is hedge gain since inception (cleaner Sortino interpretation than absolute
    # portfolio_value, which would otherwise have the seed cash baked into every metric).
    hedge_pnl = terminal_transition["pnl_excess"].detach().to(dtype=torch.float32)
    liability = terminal_transition["liability_value"].detach().to(dtype=torch.float32)
    net_pnl = hedge_pnl + liability
    return {
        "metrics": {
            "average_net_pnl": float(net_pnl.mean().item()),
            "median_net_pnl": float(torch.quantile(net_pnl.to(dtype=torch.float64), 0.5).item()),
            "worst_net_pnl": float(net_pnl.min().item()),
            "average_hedge_pnl": float(hedge_pnl.mean().item()),
            "average_liability": float(liability.mean().item()),
        },
        "final_state": {
            "positions": _cpu_tensor_dict(state.get("positions", {})),
            "cash_accounts": _cpu_tensor_dict(state.get("cash_accounts", {})),
            "margin_accounts": _cpu_tensor_dict(state.get("margin_accounts", {})),
            "pnl_excess": hedge_pnl.cpu(),
            "liability_value": liability.cpu(),
            "net_pnl": net_pnl.cpu(),
        },
    }


def build_hedge_evaluation_output(state, terminal_transition, bundle, *, no_trade_terminal=None, timing=None):
    summary = _terminal_summary(state, terminal_transition, bundle)
    out = {
        "metrics": summary["metrics"],
        "final_state": summary["final_state"],
        "diagnostics": {
            "num_episodes": int(summary["final_state"]["net_pnl"].shape[0]),
            "num_batches": int(bundle.get("meta", {}).get("num_batches", 1)),
            "trainer_type": "simulate",
        },
        "timing": dict(timing or {}),
    }
    if no_trade_terminal is not None:
        baseline = _terminal_summary(no_trade_terminal["state"], no_trade_terminal, bundle)
        out["reference"] = {
            "no_trade": {"metrics": baseline["metrics"]},
            "policy_minus_no_trade": {k: float(out["metrics"][k] - baseline["metrics"][k]) for k in baseline["metrics"]},
        }
    return out


def run_simulation(bundle, runtime):
    """Execution_Mode='simulate_only': roll the env forward with zero trades and report the
    terminal P&L summary (the unhedged baseline). Callers drive an explicit policy on top via
    `HedgeRuntimeExecutionResult.create_stepper()`."""
    _mirror_utility_scale_to_runtime(bundle, runtime)
    started = time.perf_counter()
    no_trade_terminal = _collect_no_trade_rollout(bundle, runtime)
    output = build_hedge_evaluation_output(
        no_trade_terminal["state"], no_trade_terminal, bundle,
        no_trade_terminal=no_trade_terminal,
        timing={"evaluation_time_seconds": float(time.perf_counter() - started)},
    )
    return {"policy": None, "policy_artifact": None,
            "evaluation_output": output, "optimizer_diagnostics": None}


def run_hedge_execution(bundle, runtime):
    """Dispatch the hedging run by Execution_Mode. `solve_hedge` runs the differential-ML
    solver (DiffSolverV2 + benchmark tracks); `simulate_only` runs the no-trade baseline."""
    if runtime is None or bundle is None:
        return None
    mode = str(runtime.get("execution_mode", ""))
    if mode == "solve_hedge":
        from .hedge_solver import solve_hedge
        return solve_hedge(bundle, runtime)
    if mode == "simulate_only":
        return run_simulation(bundle, runtime)
    raise ValueError(
        f"Unknown Execution_Mode {mode!r}; supported: 'solve_hedge' | 'simulate_only'.")


def _prepend_history_prefix(bundle, runtime, base_date):
    """Prepend H rows of realized history to all time-axis tensors in the bundle so that
    rolling-window features at sim-day-0 already have a populated lookback. Historical rows
    are broadcast across the batch dim (one realized series per commodity).

    Adds bundle['spot_price_history'][commodity] of shape (H, B) — broadcast historical spot.
    Adjusts bundle['time_grid_days'], bundle['tradables'][name], bundle['liability_mtm'],
    bundle['realized_cashflows'][currency], bundle['factors'][name].
    """
    H = int(runtime.get('history_lookback_business_days', 0))
    spot_history = (runtime.get('portfolio_state') or {}).get('spot_price_history') or {}
    if H <= 0 or not spot_history:
        return
    device = bundle['time_grid_days'].device
    dtype = bundle['time_grid_days'].dtype
    base_ts = pd.Timestamp(base_date)
    any_tradable = next(iter(bundle['tradables'].values())) if bundle.get('tradables') else None
    batch_size = int(any_tradable.shape[1]) if any_tradable is not None else 1
    ref_commodity = next(iter(spot_history))
    ref_dates = spot_history[ref_commodity]['dates'][-H:]
    prefix_days = torch.tensor([int((d - base_ts).days) for d in ref_dates], dtype=dtype, device=device)
    bundle['time_grid_days'] = torch.cat([prefix_days, bundle['time_grid_days']], dim=0)
    commodity_history_tensors = {}
    for commodity, payload in spot_history.items():
        prices = torch.tensor(payload['prices'][-H:], dtype=torch.float32, device=device)
        commodity_history_tensors[commodity] = prices.unsqueeze(1).expand(-1, batch_size).contiguous()
    bundle['spot_price_history'] = commodity_history_tensors
    runtime_tradables = runtime.get('tradables') or {}
    new_tradables = {}
    for name, tensor in bundle['tradables'].items():
        commodity = (runtime_tradables.get(name, {}).get('params') or {}).get('Commodity')
        if commodity is not None and commodity in commodity_history_tensors:
            prefix = commodity_history_tensors[commodity].to(dtype=tensor.dtype)
        else:
            prefix = tensor[:1].expand((H,) + tuple(tensor.shape[1:])).contiguous()
        new_tradables[name] = torch.cat([prefix, tensor], dim=0)
    bundle['tradables'] = new_tradables
    if bundle.get('liability_mtm') is not None:
        mtm = bundle['liability_mtm']
        mtm_prefix = mtm[:1].expand(H, -1).contiguous()
        bundle['liability_mtm'] = torch.cat([mtm_prefix, mtm], dim=0)
    if bundle.get('realized_cashflows'):
        new_cf = {}
        for currency, tensor in bundle['realized_cashflows'].items():
            cf_prefix = torch.zeros((H,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            new_cf[currency] = torch.cat([cf_prefix, tensor], dim=0)
        bundle['realized_cashflows'] = new_cf
    if bundle.get('factors'):
        new_factors = {}
        for factor_name, tensor in bundle['factors'].items():
            f_prefix = tensor[:1].expand((H,) + tuple(tensor.shape[1:])).contiguous()
            new_factors[factor_name] = torch.cat([f_prefix, tensor], dim=0)
        bundle['factors'] = new_factors


def build_hedge_bundle(base_date, business_day, time_grid_days, tradable_blocks,
                          factor_tensor_blocks, hedge_profile_blocks, num_batches,
                          stoch_factors, runtime=None, privileged_factor_blocks=None,
                          total_leg_volume=None, last_payment_day=None):
    """Assemble the per-batch tensor blocks (produced by the HedgeMonteCarlo simulator) into
    a single hedge bundle: concatenates blocks along the batch axis, prepends
    history-prefix rows, derives feature surfaces (realized vol, trend, stretch, privileged
    factors, utility scale), and caches CPU mirrors of the time grid + business-day decision
    indices.

    `stoch_factors` is the simulator's stochastic-factor map (factor_key → process). Used
    by `assemble_privileged_factors` to wire up per-process privileged surfaces.
    """
    def pad_time_axis(tensor, target_steps):
        current_steps = int(tensor.shape[0])
        if current_steps >= target_steps:
            return tensor[:target_steps]
        if current_steps == 0:
            raise ValueError('Cannot pad an empty tensor along the time axis')
        pad_shape = (target_steps - current_steps,) + tuple(tensor.shape[1:])
        pad_block = tensor[-1:].expand(*pad_shape)
        return torch.cat([tensor, pad_block], dim=0)

    tradables = {n: torch.cat(blocks, dim=1) for n, blocks in tradable_blocks.items()}
    factors = {n: torch.cat(blocks, dim=-1) for n, blocks in factor_tensor_blocks.items()}
    liability_mtm = torch.cat(hedge_profile_blocks['mtm'], dim=1) if hedge_profile_blocks.get('mtm') else None
    realized_cashflows = {
        currency: torch.cat(blocks, dim=1)
        for currency, blocks in (hedge_profile_blocks.get('realized_cashflows') or {}).items()
    }
    if liability_mtm is not None:
        # The liability MTM sets the mtm/hedge time axis (its native length is the mtm grid —
        # the same axis the old leg-feature tensor defined); everything else pads/truncates to it.
        aligned_time_steps = int(liability_mtm.shape[0])
    else:
        candidate_lengths = [int(time_grid_days.shape[0])]
        candidate_lengths.extend(int(tensor.shape[0]) for tensor in tradables.values())
        candidate_lengths.extend(int(tensor.shape[0]) for tensor in factors.values())
        aligned_time_steps = min(candidate_lengths) if candidate_lengths else 0
    bundle = {
        'time_grid_days': pad_time_axis(time_grid_days, aligned_time_steps),
        'tradables': {n: pad_time_axis(t, aligned_time_steps) for n, t in tradables.items()},
        'meta': {'base_date': base_date, 'business_day': business_day, 'num_batches': int(num_batches)},
    }
    if factors:
        bundle['factors'] = {n: pad_time_axis(t, aligned_time_steps) for n, t in factors.items()}
    if liability_mtm is not None:
        bundle['liability_mtm'] = pad_time_axis(liability_mtm, aligned_time_steps)
    if realized_cashflows:
        bundle['realized_cashflows'] = {
            currency: pad_time_axis(tensor, aligned_time_steps)
            for currency, tensor in realized_cashflows.items()
        }
    if runtime is not None:
        _prepend_history_prefix(bundle, runtime, base_date)
    bundle['time_grid_days_cpu'] = bundle['time_grid_days'].detach().cpu().to(dtype=torch.int64).tolist()
    days_cpu = bundle['time_grid_days_cpu']
    base_ts = pd.Timestamp(base_date)
    scenario_dates = pd.DatetimeIndex([base_ts + pd.Timedelta(days=int(d)) for d in days_cpu])
    bundle['scenario_dates'] = scenario_dates
    bday = (bundle.get('meta') or {}).get('business_day')
    if bday is not None and hasattr(bday, 'is_on_offset'):
        bd_mask = [bool(bday.is_on_offset(d)) for d in scenario_dates]
    else:
        bd_mask = [d.weekday() < 5 for d in scenario_dates]
    initial_time_index = next((i for i, d in enumerate(days_cpu) if int(d) >= 0), len(days_cpu))
    # Index where the history prefix ends and the simulation grid begins. Solvers strip
    # this offset so they can index time tensors by simulation-grid t (inner_mc_fn coords).
    bundle['initial_time_index'] = int(initial_time_index)
    # Pre-settlement terminal on the (history-stripped) sim grid: the time grid appends one
    # clean-exit row where the liability settles to zero, so the last LIVE mtm row is
    # `aligned_time_steps - 2` (the structural `[-2]`). Single source for the DP terminal
    # depth (DiffSolverV2.T_dec) and the realized-path L_T read — no magnitude heuristic.
    bundle['last_live_mtm_index'] = int(aligned_time_steps - 2)
    last = max(len(scenario_dates) - 1, 0)
    bundle['business_indices'] = tuple(
        i for i in range(min(last, len(bd_mask))) if bd_mask[i] and i >= initial_time_index
    )
    # Liability descriptors the symlog utility-scale needs, from the cashflow schedule (no leg
    # tensor). `total_leg_volume` = Σ|notional|; `last_settlement_index` = last (history-prefixed)
    # grid step still strictly before the final payment — the same value the old leg-feature
    # `time_to_payment > 0` reduction produced (time_to_payment>0 ⟺ grid_day < payment_day).
    if total_leg_volume:
        bundle['total_leg_volume'] = float(total_leg_volume)
    if last_payment_day is not None:
        pending = [i for i, d in enumerate(days_cpu) if d < last_payment_day]
        if pending:
            bundle['last_settlement_index'] = pending[-1]
    logging.info('liability scalars: total_leg_volume=%s last_settlement_index=%s',
                 bundle.get('total_leg_volume'), bundle.get('last_settlement_index'))
    bundle['spot_realized_vol'] = compute_spot_realized_vol(bundle)
    bundle['utility_scale'] = resolve_utility_scale(bundle, runtime or {})
    logging.info('utility_scale (symlog c) resolved to {0:.2f}'.format(bundle['utility_scale']))
    bundle['privileged_factors'] = assemble_privileged_factors(privileged_factor_blocks or {}, stoch_factors)
    H = int(runtime.get('history_lookback_business_days', 0)) if runtime else 0
    if H > 0 and bundle['privileged_factors']:
        bundle['privileged_factors'] = {
            name: torch.cat(
                [tensor[:1].expand((H,) + tuple(tensor.shape[1:])).contiguous(), tensor], dim=0,
            )
            for name, tensor in bundle['privileged_factors'].items()
        }
    return bundle


# --- Diagnostic-CSV post-processing helpers + result class ------------------------------
# Single consumer (HedgeRuntimeExecutionResult.write_diagnostic_csvs). Multi-commodity
# bundles raise — diagnostic CSVs for multi-commodity are out of scope.

def _diag_expand_per_day(rollout, bundle, runtime):
    """Build per-instrument (T, B) per-day cashflow tensors plus portfolio totals."""
    factors = bundle['factors']
    spot_keys = [k for k in factors if k in runtime['referenced_commodities']]
    if len(spot_keys) != 1:
        raise ValueError(
            f"Diagnostic CSV writer expects exactly one commodity-spot factor in the bundle; "
            f"got {spot_keys}. Multi-commodity diagnostic output is not yet implemented."
        )
    spot = factors[spot_keys[0]].detach().cpu().float()
    mtm_running = bundle['liability_mtm'].detach().cpu().float()
    instrument_order = tuple(runtime['names']['action_instruments'])
    spread_bps = float((runtime.get('accounting') or {}).get('bid_offer_spread_bps', 0.0))
    T, B = mtm_running.shape

    nonzero = (mtm_running != 0)
    last_nz = (nonzero * torch.arange(T).unsqueeze(1)).max(dim=0).values
    rows = torch.arange(T).unsqueeze(1).expand(T, B)
    fill_mask = rows > last_nz.unsqueeze(0)
    realised = mtm_running.gather(0, last_nz.unsqueeze(0)).expand(T, B)
    mtm = torch.where(fill_mask, realised, mtm_running)

    times = [int(t) for t in rollout['times']]
    per_instr = {}
    for name in instrument_order:
        cs = float(runtime['tradables'][name]['contract_size'])
        fut = bundle['tradables'][name].detach().cpu().float()
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
        trade_cost = trd.abs() * per_contract_kappa(runtime, fut, name)
        position_mtm = pos * fut * cs
        cum_cash = trade_cash.cumsum(0)
        cum_cost = trade_cost.cumsum(0)
        per_instr[name] = {
            'fut': fut, 'cs': cs, 'pos': pos, 'trd': trd,
            'trade_cash': trade_cash, 'trade_cost': trade_cost, 'position_mtm': position_mtm,
            'cum_cash': cum_cash, 'cum_cost': cum_cost,
        }
    portfolio_pos_mtm = sum(p['position_mtm'] for p in per_instr.values())
    portfolio_cum_cash = sum(p['cum_cash'] for p in per_instr.values())
    portfolio_cum_cost = sum(p['cum_cost'] for p in per_instr.values())
    hp = portfolio_cum_cash + portfolio_pos_mtm - portfolio_cum_cost
    total = mtm + hp
    return {
        'spot': spot, 'spread_bps': spread_bps, 'per_instr': per_instr, 'mtm': mtm,
        'portfolio_position_mtm': portfolio_pos_mtm,
        'portfolio_cum_trade_cash': portfolio_cum_cash,
        'portfolio_cum_trade_cost': portfolio_cum_cost,
        'hedge_portfolio_ex_funding': hp,
        'total_ex_funding_discount': total,
    }


def _diag_write_paths_csv(fields, day_strs, label, bundle, runtime, csv_path):
    """Per-day per-instrument breakdown for 5 representative cases (worst/p5/mean/p95/best),
    selected by terminal `total_ex_funding_discount`."""
    total = fields['total_ex_funding_discount']
    T, B = total.shape
    sidx = torch.argsort(total[-1])
    cases = {
        'worst': int(sidx[0]),
        'p5':    int(sidx[round(0.05 * (B - 1))]),
        'p95':   int(sidx[round(0.95 * (B - 1))]),
        'best':  int(sidx[-1]),
    }
    sim_start = int(bundle['initial_time_index'])
    instrument_order = tuple(runtime['names']['action_instruments'])
    rows = []

    def _row(t, case, path_idx, getter):
        row = {
            'policy': label, 'case': case, 'path_idx': int(path_idx), 'day': day_strs[t],
            'spot': getter(fields['spot'][t]).item(),
            'spread_bps': fields['spread_bps'],
        }
        for name in instrument_order:
            p = fields['per_instr'][name]
            row[f'{name}_futures']      = getter(p['fut'][t]).item()
            row[f'{name}_contract_size'] = p['cs']
            row[f'{name}_position']     = float(getter(p['pos'][t]).item())
            row[f'{name}_trade']        = float(getter(p['trd'][t]).item())
            row[f'{name}_trade_cash']   = float(getter(p['trade_cash'][t]).item())
            row[f'{name}_trade_cost']   = float(getter(p['trade_cost'][t]).item())
            row[f'{name}_position_mtm'] = float(getter(p['position_mtm'][t]).item())
        row['portfolio_position_mtm']      = float(getter(fields['portfolio_position_mtm'][t]).item())
        row['portfolio_cum_trade_cash']    = float(getter(fields['portfolio_cum_trade_cash'][t]).item())
        row['portfolio_cum_trade_cost']    = float(getter(fields['portfolio_cum_trade_cost'][t]).item())
        row['hedge_portfolio_ex_funding']  = float(getter(fields['hedge_portfolio_ex_funding'][t]).item())
        row['mtm_ex_post_settle_discount'] = float(getter(fields['mtm'][t]).item())
        row['total_ex_funding_discount']   = float(getter(fields['total_ex_funding_discount'][t]).item())
        return row

    for case_name, idx in cases.items():
        for t in range(sim_start, T):
            rows.append(_row(t, case_name, idx, lambda x, idx=idx: x[idx]))
    for t in range(sim_start, T):
        rows.append(_row(t, 'mean', -1, lambda x: x.mean()))

    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')


def _diag_summary_rows(label, rollout, fields):
    """Terminal P&L summary stats for one rollout."""
    net = rollout['net_pnl'].numpy()
    total = fields['total_ex_funding_discount'][-1].numpy()
    return [
        {'policy': label, 'metric': 'mean',  'net_pnl': float(net.mean()),  'total_ex_funding': float(total.mean())},
        {'policy': label, 'metric': 'std',   'net_pnl': float(net.std()),   'total_ex_funding': float(total.std())},
        {'policy': label, 'metric': 'min',   'net_pnl': float(net.min()),   'total_ex_funding': float(total.min())},
        {'policy': label, 'metric': 'p5',    'net_pnl': float(np.percentile(net, 5)),
         'total_ex_funding': float(np.percentile(total, 5))},
        {'policy': label, 'metric': 'p95',   'net_pnl': float(np.percentile(net, 95)),
         'total_ex_funding': float(np.percentile(total, 95))},
        {'policy': label, 'metric': 'max',   'net_pnl': float(net.max()),   'total_ex_funding': float(total.max())},
    ]


class BundleStepper:
    """Interactive simulator: advance the env one step at a time with arbitrary actions.

    Lets client code run any policy — textbook hedge, custom rule, perfect-foresight,
    counterfactual what-ifs — without reaching into framework internals. Each step
    returns a state dict the caller can inspect; the caller chooses an action and calls
    `step(action)`. Supports `copy.deepcopy(stepper)` to fork into counterfactual branches.

    Vectorized over the bundle's full batch (B paths advance in lockstep). Action values
    can be scalars (broadcast across batch) or per-path tensors of shape (B,).
    """

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime
        _mirror_utility_scale_to_runtime(bundle, runtime)
        self._state = build_shared_state(bundle, runtime)
        self._last_idx = _last_time_index(bundle)
        self._decision_set = set(int(i) for i in _decision_time_indices(bundle))
        self._instrument_order = tuple(runtime['names']['action_instruments'])
        self._device = bundle['time_grid_days'].device
        self._batch_size = int(next(iter(self._state['positions'].values())).shape[0])
        # Per-decision recording for post-hoc diagnostic CSV writing. Cheap (a few (B,)
        # tensors per decision step); always-on so write_diagnostic_csvs has data to use.
        self._times = []
        self._position_history = {n: [] for n in self._instrument_order}
        self._trade_history = {n: [] for n in self._instrument_order}
        self._price_history = {n: [] for n in self._instrument_order}
        self._terminal_transition = None

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
            'tradable_values': dict(self._state.get('tradable_values', {})),
            'cumulative_liability_value': self._state.get('cumulative_liability_value'),
            'pnl_excess': self._state.get('pnl_excess'),
        }

    def step(self, action=None):
        """Advance one time step. `action` is `{instrument_name: scalar_or_(B,)_tensor}`
        applied as trade deltas (only meaningful at decision steps; ignored otherwise).
        Pass `None` for zero trades. Returns a dict combining the post-step observation
        with the per-path pnl_excess + liability_value for this transition.
        """
        was_decision_step = self.is_decision_step
        t_pre = self.time_index
        if was_decision_step:
            # Record pre-step position + price for diagnostic CSV.
            self._times.append(t_pre)
            for n in self._instrument_order:
                self._position_history[n].append(self._state['positions'][n].detach().cpu().clone())
                self._price_history[n].append(self._state['tradable_values'][n].detach().cpu().clone())
        structured = self._build_action(action) if (action is not None and was_decision_step) else None
        next_state = step_runtime_state(self._state, structured, self.bundle, self.runtime)
        transition = reward_and_terminal_payoff(self._state, next_state, self.bundle, self.runtime)
        if was_decision_step:
            # Realized trade = post-step position - pre-step position (handles env clips/forces).
            for n in self._instrument_order:
                pre_pos = self._position_history[n][-1]
                self._trade_history[n].append(next_state['positions'][n].detach().cpu() - pre_pos)
        self._state = next_state
        self._terminal_transition = transition
        return {
            **self.observe(),
            'transition_pnl_excess': transition['pnl_excess'],
            'transition_liability_value': transition['liability_value'],
        }

    def write_diagnostic_csvs(self, output_dir: str, label: str = 'custom') -> None:
        """Write per-day per-instrument breakdown + terminal P&L summary for the trajectory
        this stepper has accumulated. Same output shape as
        `HedgeRuntimeExecutionResult.write_diagnostic_csvs`, but driven by whatever policy
        the caller chose (textbook hedge, custom rule, etc.). Files written:
          <output_dir>/<label>_paths.csv
          <output_dir>/<label>_summary.csv

        Must be called after the stepper has been advanced through the rollout (i.e. when
        `done` is True). Cases reported: worst / p5 / mean / p95 / best, ranked by terminal P&L.
        """
        if self._terminal_transition is None:
            raise ValueError("Stepper has no recorded trajectory yet — call step() until done first.")
        rollout = {
            'times': self._times,
            'position': {n: torch.stack(self._position_history[n], dim=0) for n in self._instrument_order},
            'trade':    {n: torch.stack(self._trade_history[n], dim=0)    for n in self._instrument_order},
            'price':    {n: torch.stack(self._price_history[n], dim=0)    for n in self._instrument_order},
            'pnl_excess': self._terminal_transition['pnl_excess'].detach().cpu(),
            'liability':  self._terminal_transition['liability_value'].detach().cpu(),
            'net_pnl':    (self._terminal_transition['pnl_excess']
                            + self._terminal_transition['liability_value']).detach().cpu(),
        }
        os.makedirs(output_dir, exist_ok=True)
        fields = _diag_expand_per_day(rollout, self.bundle, self.runtime)
        day_strs = [pd.Timestamp(d).strftime('%Y-%m-%d')
                    for d in _bundle_scenario_dates(self.bundle)]
        _diag_write_paths_csv(fields, day_strs, label, self.bundle, self.runtime,
                              os.path.join(output_dir, f'{label}_paths.csv'))
        pd.DataFrame(_diag_summary_rows(label, rollout, fields)).to_csv(
            os.path.join(output_dir, f'{label}_summary.csv'), index=False, float_format='%.2f',
        )

    def _build_action(self, action_dict):
        trade_deltas = {}
        for name in self._instrument_order:
            v = action_dict.get(name, 0)
            if isinstance(v, torch.Tensor):
                trade_deltas[name] = v.to(device=self._device, dtype=torch.float32)
            else:
                trade_deltas[name] = torch.full((self._batch_size,), float(v),
                                                 dtype=torch.float32, device=self._device)
        return _realized_structured_action(
            {'trade_deltas': trade_deltas}, self._state['positions'], self.runtime,
            batch_size=self._batch_size, device=self._device,
        )


class HedgeRuntimeExecutionResult:
    """High-level result for HedgeMonteCarlo's hedge-bundle handoff.

    Carries the bundle + normalized runtime + evaluation summary + the solver artifact
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
        if self.bundle is None or self.runtime is None:
            raise ValueError("create_stepper needs bundle and runtime on the result.")
        return BundleStepper(self.bundle, self.runtime)


if __name__ == '__main__':
    pass
