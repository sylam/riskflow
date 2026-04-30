from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tensordict import TensorDict

from .hedge_features import build_entity_state
from .structured_policy import StructuredActionSpace, StructuredRebalancePolicy


def _batch_size_from_bundle(bundle):
    tradables = bundle.get("tradables", {})
    for tensor in tradables.values():
        return int(tensor.shape[-1])
    factors = bundle.get("factors", {})
    for tensor in factors.values():
        return int(tensor.shape[-1])
    return 0


def _slice_episode_tensor(tensor, episode_indices, batch_size):
    tensor_indices = episode_indices.to(device=tensor.device, dtype=torch.int64)
    if tensor.ndim >= 2 and int(tensor.shape[1]) == int(batch_size):
        return tensor.index_select(1, tensor_indices)
    if tensor.ndim >= 1 and int(tensor.shape[-1]) == int(batch_size):
        return tensor.index_select(tensor.ndim - 1, tensor_indices)
    return tensor


def _slice_bundle_episodes(bundle, episode_indices):
    batch_size = _batch_size_from_bundle(bundle)
    if batch_size <= 0:
        return bundle
    sliced = {
        "time_grid_days": bundle["time_grid_days"],
        "tradables": {n: _slice_episode_tensor(t, episode_indices, batch_size) for n, t in bundle.get("tradables", {}).items()},
        "meta": dict(bundle.get("meta", {})),
    }
    if bundle.get("factors"):
        sliced["factors"] = {n: _slice_episode_tensor(t, episode_indices, batch_size) for n, t in bundle["factors"].items()}
    if bundle.get("legs") is not None:
        legs = bundle["legs"]
        sliced["legs"] = {
            "features": _slice_episode_tensor(legs["features"], episode_indices, batch_size),
            "ids": list(legs.get("ids", ())),
            "feature_names": tuple(legs.get("feature_names", ())),
            "id_names": tuple(legs.get("id_names", ())),
        }
    if bundle.get("liability_mtm") is not None:
        sliced["liability_mtm"] = _slice_episode_tensor(bundle["liability_mtm"], episode_indices, batch_size)
    if bundle.get("realized_cashflows") is not None:
        sliced["realized_cashflows"] = {
            ccy: _slice_episode_tensor(t, episode_indices, batch_size) for ccy, t in bundle["realized_cashflows"].items()
        }
    sliced["meta"]["validation_episode_count"] = int(episode_indices.numel())
    return sliced


def _split_episode_indices(total, num_splits):
    if total <= 0:
        return tuple()
    n = max(min(int(num_splits), total), 1)
    full = torch.arange(total, dtype=torch.int64)
    return tuple(chunk for chunk in torch.tensor_split(full, n) if int(chunk.numel()) > 0)


def _zeros_by_name(names, batch_size, *, device, dtype=torch.float32):
    return {str(name): torch.zeros(batch_size, dtype=dtype, device=device) for name in names}


def _seed_by_name(seed_values, names, batch_size, *, device, dtype=torch.float32):
    return {str(name): torch.full((batch_size,), float(seed_values.get(str(name), 0.0)), dtype=dtype, device=device) for name in names}


def _runtime_names(runtime, key):
    return tuple(name for name in runtime.get("names", {}).get(key, ()))


def _runtime_tradable(runtime, name):
    return dict(runtime.get("tradables", {}).get(str(name), {}))


def _clone_tensor_dict(values):
    return {str(n): t.clone() for n, t in values.items()}


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
    """Cheap per-step refresh: refreshes price/liability views and carries forward simulator state.
    Entity-state construction (the expensive transformer-shaped feature stack) is deferred to
    `_ensure_decision_views`, which only runs on decision steps where the policy actually reads
    it. With a 10-day decision interval this saves ~9 entity rebuilds per decision."""
    refreshed = dict(state)
    tradable_values = _current_tradable_values(bundle, runtime, time_index)
    sample = next(iter(refreshed["positions"].values()))
    batch_size = int(sample.shape[0])
    device = sample.device
    return {
        "time_index": int(time_index),
        "done": refreshed["done"],
        "positions": refreshed["positions"],
        "cash_accounts": refreshed["cash_accounts"],
        "margin_accounts": refreshed["margin_accounts"],
        "realized_pnl": refreshed["realized_pnl"],
        "variation_margin": refreshed["variation_margin"],
        "cumulative_pnl": refreshed.get("cumulative_pnl", _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)),
        "time_held": refreshed.get("time_held", _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)),
        "settlement_prices": refreshed["settlement_prices"],
        "tradable_values": tradable_values,
        "liability_mtm_value": _current_liability_mtm(bundle, time_index),
        "realized_cashflow_value": _current_realized_cashflow_total(bundle, time_index),
        "cumulative_liability_value": refreshed.get(
            "cumulative_liability_value",
            torch.zeros(batch_size, dtype=torch.float32, device=device),
        ).to(dtype=torch.float32),
        "initial_portfolio_value": refreshed.get("initial_portfolio_value"),
    }


def _ensure_decision_views(state, bundle, runtime):
    """Materialize `entity_state` on a state dict that came out of the cheap refresh path.
    Idempotent — returns the input unchanged when entity_state is already present."""
    if state.get("entity_state") is not None:
        return state
    entity_state = build_entity_state(bundle, state, int(state["time_index"]), runtime["entity_layout"])
    return {**state, "entity_state": entity_state}


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
    """Multiply each balance by its one-day growth factor (or pass through if no factor present)."""
    return {n: (bal * factors[n] if n in factors else bal.clone()) for n, bal in accounts.items()}


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


def _dense_tracking_reward(prev_state, next_state, runtime):
    """Per-step shaping in dollars: |prev_err| - |next_err| where err = pnl_excess + liability_mtm.
    Positive when the step closes the tracking gap. Telescopes to |initial_err| - |final_err|
    over an episode, so the dense channel adds per-step gradient without distorting the asymptote
    — the optimum is still "minimize final |error|".

    Living in dollars (same units as the terminal asymmetric utility) means a single reward_scale
    calibrates both consistently; the prior relative-error formulation was 4 orders of magnitude
    below terminal under reward_scale=1e-6 and contributed effectively no gradient.

    `dense_tracking_reward_clip` (dollars, optional, default 0=off) caps per-step magnitude in
    case a price jump produces a single-step spike that would dominate the update."""
    settings = dict(runtime.get("optimizer") or {})
    rs = float(settings.get("dense_tracking_reward_scale", 0.0))
    prev_err = _tracking_error_value(prev_state, runtime)
    next_err = _tracking_error_value(next_state, runtime)
    shaping = prev_err.abs() - next_err.abs()
    rc = float(settings.get("dense_tracking_reward_clip", 0.0))
    if rc > 0.0:
        shaping = torch.clamp(shaping, min=-rc, max=rc)
    return rs * shaping


def _evaluate_objective(pnl_excess, liability, runtime):
    objective = runtime.get("objective")
    # net_pnl = hedge gain since inception + leg cashflows received. Perfect hedge → pnl_excess
    # offsets liability → net_pnl ≈ 0, sitting exactly at the floor where Floor_Penalty kicks in.
    # Using ABSOLUTE portfolio_value here (cash baseline included) would silently bias every
    # scenario into the surplus regime and the floor penalty would never fire.
    net_pnl = pnl_excess + liability
    if objective is None:
        return net_pnl
    fp = float(objective.get("floor_penalty", 1.0))
    sr = float(objective.get("surplus_reward", 1.0))
    p = float(objective.get("power", 1.0))
    surplus = sr * torch.pow(torch.clamp(net_pnl, min=0.0), p)
    shortfall = -fp * torch.pow(torch.clamp(-net_pnl, min=0.0), p)
    return torch.where(net_pnl >= 0.0, surplus, shortfall)


def _cash_account_currency_map(runtime):
    return {str(n): str(d.get("currency")) for n, d in runtime.get("accounting", {}).get("cash_accounts", {}).items()}


def _cash_account_for_instrument(name, runtime):
    instrument_currency = str(_runtime_tradable(runtime, str(name)).get("currency"))
    cmap = _cash_account_currency_map(runtime)
    for account in _runtime_names(runtime, "cash_accounts"):
        if cmap.get(str(account)) == instrument_currency:
            return str(account)
    accounts = _runtime_names(runtime, "cash_accounts")
    return None if not accounts else accounts[0]


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


def _enforce_position_limits(positions, trade_deltas, runtime):
    adjusted = dict(trade_deltas)
    limits = dict(runtime.get("accounting", {}).get("position_limits", {}))
    for name in _runtime_names(runtime, "action_instruments"):
        n = str(name)
        if n not in positions or n not in adjusted:
            continue
        lim = dict(limits.get(n, {}))
        min_p = float(lim.get("min_position", float("-inf")))
        max_p = float(lim.get("max_position", float("inf")))
        proposed = positions[n] + adjusted[n]
        bounded = torch.clamp(proposed, min=min_p, max=max_p)
        adjusted[n] = bounded - positions[n]
    return adjusted


def _realized_structured_action(action, current_positions, runtime, *, batch_size, device):
    if action is None:
        return None
    executed = _enforce_position_limits(current_positions, _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device), runtime)
    instrument_order = tuple(str(n) for n in _runtime_names(runtime, "action_instruments"))
    ordered = torch.stack([executed[n] for n in instrument_order], dim=1).round().to(dtype=torch.int64)
    return {
        **action,
        "ordered_trade_deltas": ordered,
        "trade_deltas": {n: ordered[:, i] for i, n in enumerate(instrument_order)},
    }


def _transaction_costs(trade_delta, price, runtime, name):
    unit_cost = float(runtime.get("accounting", {}).get("transaction_cost_per_unit", 0.0))
    spread_bps = float(runtime.get("accounting", {}).get("bid_offer_spread_bps", 0.0))
    contract_size = float(_runtime_tradable(runtime, name).get("contract_size", 1.0))
    cost = trade_delta.abs() * unit_cost
    if spread_bps > 0.0:
        cost = cost + trade_delta.abs() * price.abs() * contract_size * 0.5 * spread_bps * 1.0e-4
    return cost


def _apply_cash_trade(cash_accounts, account_name, trade_delta, price, transaction_cost):
    if account_name is None:
        return
    cash_accounts[account_name] = cash_accounts[account_name] - (trade_delta * price + transaction_cost)


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


def _zero_futures_flows(state, runtime):
    batch_size = int(state["done"].shape[0])
    device = state["done"].device
    state["realized_pnl"] = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    state["variation_margin"] = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)


def _flatten_cash_inventory(positions, cash_accounts, terminal_values, runtime):
    for name in _runtime_names(runtime, "hedges"):
        n = str(name)
        delta = -positions[n]
        cost = _transaction_costs(delta, terminal_values[n], runtime, n)
        _apply_cash_trade(cash_accounts, _cash_account_for_instrument(n, runtime), delta, terminal_values[n], cost)
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
    initial_time_index = int(runtime.get("history_lookback_business_days", 0) or 0)
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
    next_positions = _clone_tensor_dict(state["positions"])
    growth = _daily_growth_factors(bundle, runtime, current, next_idx)
    next_cash = _compound_accounts(state["cash_accounts"], growth)
    deltas = _enforce_position_limits(state["positions"], _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device), runtime)
    for name in _runtime_names(runtime, "action_instruments"):
        n = str(name)
        delta = deltas.get(n)
        price = state["tradable_values"][n]
        cost = _transaction_costs(delta, price, runtime, n)
        _apply_cash_trade(next_cash, _cash_account_for_instrument(n, runtime), delta, price, cost)
        next_positions[n] = next_positions[n] + delta
    if _should_terminal_flatten(runtime, current, last):
        _flatten_cash_inventory(next_positions, next_cash, _current_tradable_values(bundle, runtime, last), runtime)
    # cash-account mode: no daily VM. cumulative_pnl unchanged here.
    next_cumulative_pnl = _clone_tensor_dict(state["cumulative_pnl"])
    next_time_held = _step_time_held(state["time_held"], next_positions)
    next_state = {
        "done": state["done"],
        "positions": next_positions,
        "cash_accounts": next_cash,
        "margin_accounts": _clone_tensor_dict(state["margin_accounts"]),
        "realized_pnl": _clone_tensor_dict(state["realized_pnl"]),
        "variation_margin": _clone_tensor_dict(state["variation_margin"]),
        "cumulative_pnl": next_cumulative_pnl,
        "time_held": next_time_held,
        "cumulative_liability_value": state["cumulative_liability_value"].clone(),
        "settlement_prices": _clone_tensor_dict(state["settlement_prices"]),
        "initial_portfolio_value": state["initial_portfolio_value"].clone(),
    }
    _zero_futures_flows(next_state, runtime)
    next_state["done"] = torch.full_like(state["done"], next_idx >= last, dtype=torch.bool)
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
    next_positions = _clone_tensor_dict(state["positions"])
    growth = _daily_growth_factors(bundle, runtime, current, settlement_idx)
    next_cash = _compound_accounts(state["cash_accounts"], growth)
    next_margin = _compound_accounts(state["margin_accounts"], growth)
    next_settlement = _clone_tensor_dict(state["settlement_prices"])
    realized_pnl = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    variation_margin = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    deltas = _enforce_position_limits(state["positions"], _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device), runtime)
    next_values = _current_tradable_values(bundle, runtime, settlement_idx)
    # Futures: cash_accounts is frozen at starting capital; only margin tracks VM and trade cost.
    for name in _runtime_names(runtime, "hedges"):
        n = str(name)
        vm = state["positions"][n] * (next_values[n] - state["settlement_prices"][n]) * float(_runtime_tradable(runtime, n).get("contract_size", 1.0))
        realized_pnl[n] = vm
        variation_margin[n] = vm
        next_settlement[n] = next_values[n].clone()
        account = _cash_account_for_instrument(n, runtime)
        _apply_account_flow(next_margin, account, vm)
    for name in _runtime_names(runtime, "action_instruments"):
        n = str(name)
        delta = deltas.get(n)
        # Trade is executed at decision-time price (state["tradable_values"]), not next-step
        # settlement. Bid/offer spread and per-unit cost should both reference the price the agent
        # actually saw and acted on; using next_values[n] would let unrelated overnight mid moves
        # distort cost.
        cost = _transaction_costs(delta, state["tradable_values"][n], runtime, n)
        account = _cash_account_for_instrument(n, runtime)
        next_positions[n] = next_positions[n] + delta
        _apply_account_flow(next_margin, account, -cost)
    if _should_terminal_flatten(runtime, current, last):
        _flatten_futures_inventory(next_positions, next_margin, next_settlement, next_values, runtime)
    next_cumulative_pnl = {n: state["cumulative_pnl"][n] + variation_margin[n] for n in state["cumulative_pnl"]}
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
        "cumulative_liability_value": state["cumulative_liability_value"].clone(),
        "initial_portfolio_value": state["initial_portfolio_value"].clone(),
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
    reward = _dense_tracking_reward(prev_state, state, runtime).to(device=device, dtype=torch.float32)
    terminal_payoff = torch.zeros(batch_size, dtype=torch.float32, device=device)
    pnl_excess = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liability_value = state.get("cumulative_liability_value", torch.zeros(batch_size, dtype=torch.float32, device=device)).to(device=device, dtype=torch.float32)
    # Sync-free terminality check: done becomes True iff time_index hits last_idx, and is uniform
    # across the batch (the simulator advances all scenarios together). Avoids a `.item()` sync on
    # every rollout step.
    if int(state["time_index"]) >= _last_time_index(bundle):
        done_mask = state["done"].to(dtype=torch.bool)
        # pnl_excess = portfolio change since inception; the asymmetric utility evaluates against
        # this (NOT absolute portfolio value) so the seed cash baseline doesn't bias the floor.
        pnl_excess = _pnl_excess(state, runtime).to(device=device, dtype=torch.float32)
        terminal_reward = _evaluate_objective(pnl_excess, liability_value, runtime).to(device=device, dtype=torch.float32)
        reward = reward + torch.where(done_mask, terminal_reward, torch.zeros_like(reward))
        terminal_payoff = torch.where(done_mask, liability_value, terminal_payoff)
    return {"reward": reward, "terminal_payoff": terminal_payoff, "pnl_excess": pnl_excess, "liability_value": liability_value}


def _bundle_scenario_dates(bundle):
    base_date = pd.Timestamp(bundle.get("meta", {}).get("base_date"))
    days = torch.as_tensor(bundle["time_grid_days"]).detach().cpu().to(dtype=torch.int64).tolist()
    return pd.DatetimeIndex([base_date + pd.Timedelta(days=int(d)) for d in days])


def _is_business_date(date, bundle):
    bday = bundle.get("meta", {}).get("business_day")
    if bday is not None and hasattr(bday, "is_on_offset"):
        return bool(bday.is_on_offset(pd.Timestamp(date)))
    return pd.Timestamp(date).weekday() < 5


def _decision_interval_business_days_for_epoch(settings, epoch, *, evaluation):
    curriculum = tuple(dict(s) for s in settings.get("decision_interval_curriculum", ()))
    if not curriculum:
        return 1
    if evaluation or epoch is None:
        return int(curriculum[-1]["interval_business_days"])
    epoch_n = int(epoch) + 1
    for stage in curriculum:
        start = int(stage.get("start_epoch", 1))
        end = stage.get("end_epoch")
        if epoch_n < start:
            continue
        if end is None or epoch_n <= int(end):
            return int(stage["interval_business_days"])
    return int(curriculum[-1]["interval_business_days"])


def _decision_time_indices(bundle, settings, *, epoch, evaluation):
    dates = _bundle_scenario_dates(bundle)
    if dates.empty:
        return tuple()
    last = max(len(dates) - 1, 0)
    days = bundle["time_grid_days"].detach().cpu().to(dtype=torch.int64).tolist()
    business = [
        i for i, d in enumerate(dates[:last])
        if _is_business_date(d, bundle) and int(days[i]) >= 0
    ]
    if not business:
        return tuple()
    interval = max(_decision_interval_business_days_for_epoch(settings, epoch, evaluation=evaluation), 1)
    return tuple(b for j, b in enumerate(business) if j % interval == 0)


def _optimizer_settings(runtime):
    p = runtime.get("optimizer") or {}
    return {
        "epochs": int(p.get("epochs", 20)),
        "ppo_epochs": int(p.get("ppo_epochs", 4)),
        "minibatch_size": int(p.get("minibatch_size", 4096)),
        "gamma": float(p.get("gamma", 0.99)),
        "gae_lambda": float(p.get("gae_lambda", 0.95)),
        "learning_rate": float(p.get("learning_rate", 3.0e-4)),
        "clip_eps": float(p.get("clip_eps", 0.2)),
        "value_coef": float(p.get("value_coef", 0.5)),
        "entropy_coef": float(p.get("entropy_coef", 0.01)),
        "max_grad_norm": float(p.get("max_grad_norm", 0.5)),
        "reward_scale": float(p.get("reward_scale", 1.0)),
        "action_sparsity_coef": float(p.get("action_sparsity_coef", 0.0)),
        "validation_fraction": float(p.get("validation_fraction", 0.25)),
        "validation_min_batch": int(p.get("validation_min_batch", 256)),
        "decision_interval_curriculum": tuple(dict(s) for s in p.get("decision_interval_curriculum", ())),
        "seed": p.get("seed"),
    }


def _make_structured_policy(runtime, *, device):
    policy_cfg = runtime.get("policy", {})
    model = policy_cfg.get("model", {})
    action_space_cfg = policy_cfg.get("action_space", {})
    instrument_order = tuple(action_space_cfg.get("instrument_order", ()))
    return StructuredRebalancePolicy(
        action_space=StructuredActionSpace(
            instrument_order,
            tuple(action_space_cfg.get("min_trade_delta", {}).get(n, 0) for n in instrument_order),
            tuple(action_space_cfg.get("max_trade_delta", {}).get(n, 0) for n in instrument_order),
        ),
        entity_layout=runtime["entity_layout"],
        privileged_layout=runtime.get("privileged_layout") or {},
        position_limits=runtime.get("accounting", {}).get("position_limits", {}),
        token_dim=int(model.get("token_dim", 64)),
        emb_dim=int(model.get("emb_dim", 8)),
        n_heads=int(model.get("n_heads", 4)),
        n_layers=int(model.get("n_layers", 2)),
        device=device,
    )


def _positions_tensor(state, instrument_order, *, device):
    """Stack a state's per-instrument position dict into a (B, I) int64 tensor in policy order.
    Used to feed the policy's feasibility mask at sample / evaluate time."""
    return torch.stack(
        [state["positions"][name].to(dtype=torch.int64, device=device) for name in instrument_order],
        dim=-1,
    )


def _privileged_state_at(bundle, time_index, batch_size, device):
    """Slice bundle['privileged_factors'] at the given time index, returning a per-name dict of
    tensors shaped (B, dim). Empty dict if no privileged factors are available."""
    factors = (bundle.get("privileged_factors") or {}) if bundle else {}
    if not factors:
        return {}
    out = {}
    for name, t in factors.items():
        slice_t = t[time_index].to(dtype=torch.float32, device=device)
        if slice_t.shape[0] == 1 and slice_t.shape[0] != batch_size:
            slice_t = slice_t.expand(batch_size, -1).contiguous()
        out[name] = slice_t
    return out


def _entity_state_snapshot(entity_state):
    return TensorDict({k: v.detach().clone() for k, v in entity_state.items()}, batch_size=entity_state.batch_size)


def _stack_entity_states(entity_states):
    keys = entity_states[0].keys()
    out = {k: torch.stack([es[k] for es in entity_states], dim=0) for k in keys}
    D, B = out[next(iter(keys))].shape[:2]
    return TensorDict(out, batch_size=[D, B])


def _flatten_for_minibatch(stacked_td):
    D, B = stacked_td.batch_size[0], stacked_td.batch_size[1]
    return stacked_td.reshape(D * B), D, B


def _collect_ppo_rollout(policy, bundle, runtime, decision_indices, *, deterministic=False, reward_scale=1.0):
    state = build_shared_state(bundle, runtime)
    decision_set = set(int(i) for i in decision_indices)
    batch_size = int(next(iter(state["positions"].values())).shape[0])
    device = policy.device
    instrument_order = policy.action_space.instrument_order
    entity_snapshots = []
    privileged_snapshots = []
    position_snapshots = []
    action_bins_list = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    pending_reward = None
    pending_index = -1
    last_state = state
    # Loop bounded by Python-side time_index — `done` becomes True iff time_index hits the bundle's
    # last index, so the tensor.all().item() sync was equivalent and just forced a CUDA-CPU sync
    # every step. With B=4096 and ~200 sim steps × ppo_epochs the syncs become measurable on GPU.
    last_idx = _last_time_index(bundle)
    while int(state["time_index"]) < last_idx:
        t = int(state["time_index"])
        # Per-iteration step_action: applied at exactly one step (the decision step). On
        # non-decision steps we pass None → step_runtime_state computes zero trade deltas.
        # Previously a persistent `mapped` was reused across non-decision steps, so the chosen
        # delta was applied every day until the next decision — multiplying effective trade size
        # by the decision interval and biasing early-curriculum policies toward the floor.
        step_action = None
        if t in decision_set:
            if pending_reward is not None:
                rewards.append(pending_reward)
                dones.append(state["done"].to(dtype=torch.float32, device=device))
            # Materialize entity_state lazily on the decision step. The cheap refresh path skips
            # this on non-decision steps (interval=10 → 9× saved per decision).
            state = _ensure_decision_views(state, bundle, runtime)
            priv = _privileged_state_at(bundle, t, batch_size, device)
            positions_t = _positions_tensor(state, instrument_order, device=device)
            output = policy.sample(
                state["entity_state"], deterministic=deterministic,
                privileged_state=priv, positions=positions_t,
            )
            step_action = _realized_structured_action(policy.map_actions(output), state["positions"], runtime, batch_size=batch_size, device=device)
            entity_snapshots.append(_entity_state_snapshot(state["entity_state"]))
            privileged_snapshots.append({k: v.detach().clone() for k, v in priv.items()})
            position_snapshots.append(positions_t.detach().clone())
            action_bins_list.append(output["action_bins"].detach())
            log_probs.append(output["log_prob"].detach())
            values.append(output["value"].detach())
            pending_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
            pending_index = len(entity_snapshots) - 1
        next_state = step_runtime_state(state, step_action, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        if pending_reward is not None:
            pending_reward = pending_reward + transition["reward"].to(device=device, dtype=torch.float32) * reward_scale
        last_state = next_state
        state = next_state
    if pending_reward is not None and pending_index == len(entity_snapshots) - 1:
        rewards.append(pending_reward)
        dones.append(state["done"].to(dtype=torch.float32, device=device))
    if not entity_snapshots:
        return None
    stacked_states = _stack_entity_states(entity_snapshots)
    stacked_privileged = {}
    if privileged_snapshots and privileged_snapshots[0]:
        for name in privileged_snapshots[0].keys():
            stacked_privileged[name] = torch.stack([s[name] for s in privileged_snapshots], dim=0)
    stacked_positions = torch.stack(position_snapshots, dim=0) if position_snapshots else None
    return {
        "states": stacked_states,
        "privileged_states": stacked_privileged,
        "positions": stacked_positions,
        "action_bins": torch.stack(action_bins_list, dim=0),
        "log_probs": torch.stack(log_probs, dim=0),
        "values": torch.stack(values, dim=0),
        "rewards": torch.stack(rewards, dim=0),
        "dones": torch.stack(dones, dim=0),
        "terminal_state": last_state,
        "terminal_transition": {**transition, "state": last_state},
    }


def _compute_gae(rewards, values, dones, gamma, lam, *, last_value=None, last_done=None):
    """SB3-style GAE: at iter t, mask uses dones[t] (whether the transition out of segment t ended
    the episode), not dones[t+1]. The previous implementation lagged the mask by one step, which
    silently dropped the bootstrap from V[t+1] at the second-to-last decision.

    For the final segment, `last_value` is the value estimate at the post-rollout state (0 if the
    rollout terminated naturally — the default — or V(s_{D}) if you cut off mid-episode), and
    `last_done` indicates whether the rollout boundary IS terminal (1 = terminal, 0 = cutoff)."""
    D = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    last_value_t = torch.zeros_like(values[0]) if last_value is None else last_value
    last_done_t = torch.ones_like(rewards[0]) if last_done is None else last_done
    for t in reversed(range(D)):
        if t == D - 1:
            next_value = last_value_t
            next_nonterminal = 1.0 - last_done_t
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _ppo_update(policy, optimizer, rollout, settings, *, decision_interval=1):
    # Gamma and gae_lambda are interpreted per-business-day so a curriculum that ramps the
    # decision interval (10 → 5 → 2 → 1) doesn't change the implicit per-day discount and shift
    # value-target / advantage scales at every stage boundary. gamma_stage = gamma_per_day ** N
    # gives the effective per-decision discount over N days; same logic for lambda keeps the
    # effective horizon stable in calendar time across stages.
    interval = max(int(decision_interval), 1)
    gamma_stage = float(settings["gamma"]) ** interval
    lam_stage = float(settings["gae_lambda"]) ** interval
    advantages, returns = _compute_gae(rollout["rewards"], rollout["values"], rollout["dones"], gamma_stage, lam_stage)
    flat_states, D, B = _flatten_for_minibatch(rollout["states"])
    flat_actions = rollout["action_bins"].reshape(D * B, -1)
    flat_old_log_probs = rollout["log_probs"].reshape(D * B)
    flat_old_values = rollout["values"].reshape(D * B)
    flat_advantages = advantages.reshape(D * B)
    flat_returns = returns.reshape(D * B)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
    flat_privileged = {}
    for name, t in (rollout.get("privileged_states") or {}).items():
        flat_privileged[name] = t.reshape(D * B, -1)
    rollout_positions = rollout.get("positions")
    flat_positions = rollout_positions.reshape(D * B, -1) if rollout_positions is not None else None
    N = D * B
    mb = max(int(settings["minibatch_size"]), 1)
    # Accumulate per-minibatch losses as device-side tensors and sync once at the end of the
    # update — saves ~3 .item() syncs × ~60 minibatches per update.
    device = flat_actions.device
    policy_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    value_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
    minibatch_count = 0
    clip_eps = float(settings["clip_eps"])
    for _ in range(int(settings["ppo_epochs"])):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, mb):
            idx = perm[start:start + mb]
            mb_state = flat_states[idx]
            mb_actions = flat_actions[idx]
            mb_old_lp = flat_old_log_probs[idx]
            mb_old_v = flat_old_values[idx]
            mb_adv = flat_advantages[idx]
            mb_ret = flat_returns[idx]
            mb_privileged = {name: t[idx] for name, t in flat_privileged.items()} if flat_privileged else None
            mb_positions = flat_positions[idx] if flat_positions is not None else None
            new_lp, entropy, value, logits = policy.evaluate_action(mb_state, mb_actions, privileged_state=mb_privileged, positions=mb_positions)
            ratio = (new_lp - mb_old_lp).exp()
            clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
            v_clipped = mb_old_v + (value - mb_old_v).clamp(-clip_eps, clip_eps)
            v_loss_unclipped = (value - mb_ret).pow(2)
            v_loss_clipped = (v_clipped - mb_ret).pow(2)
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            entropy_bonus = entropy.mean()
            # Action-sparsity penalty: |trade_delta| normalized by action range. Discourages
            # over-trading when the gradient signal is weak (e.g., noisy regime features).
            trade_deltas = (mb_actions + policy._min_trade_delta).to(dtype=torch.float32)
            action_sparsity_loss = (trade_deltas.abs() / policy._action_feature_scale).mean()
            loss = (
                policy_loss
                + settings["value_coef"] * value_loss
                - settings["entropy_coef"] * entropy_bonus
                + settings["action_sparsity_coef"] * action_sparsity_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["max_grad_norm"])
            optimizer.step()
            policy_loss_sum = policy_loss_sum + policy_loss.detach()
            value_loss_sum = value_loss_sum + value_loss.detach()
            entropy_sum = entropy_sum + entropy_bonus.detach()
            minibatch_count += 1
    denom = max(minibatch_count, 1)
    return {
        "policy_loss": float((policy_loss_sum / denom).item()),
        "value_loss": float((value_loss_sum / denom).item()),
        "entropy": float((entropy_sum / denom).item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std().item()),
    }


def _collect_no_trade_rollout(bundle, runtime, decision_indices):
    state = build_shared_state(bundle, runtime)
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


def build_torchrl_evaluation_output(state, terminal_transition, bundle, *, no_trade_terminal=None, timing=None, optimizer_diagnostics=None):
    summary = _terminal_summary(state, terminal_transition, bundle)
    out = {
        "metrics": summary["metrics"],
        "final_state": summary["final_state"],
        "diagnostics": {
            "num_episodes": int(summary["final_state"]["net_pnl"].shape[0]),
            "num_batches": int(bundle.get("meta", {}).get("num_batches", 1)),
            "trainer_type": "ppo",
            "optimizer_diagnostics": optimizer_diagnostics or {},
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


def evaluate_torchrl_policy(bundle, runtime):
    settings = _optimizer_settings(runtime)
    device = bundle["time_grid_days"].device
    policy = _make_structured_policy(runtime, device=device)
    decision_indices = _decision_time_indices(bundle, settings, epoch=None, evaluation=True)
    started = time.perf_counter()
    rollout = _collect_ppo_rollout(policy, bundle, runtime, decision_indices, deterministic=True)
    no_trade_terminal = _collect_no_trade_rollout(bundle, runtime, decision_indices)
    output = build_torchrl_evaluation_output(
        rollout["terminal_state"], rollout["terminal_transition"], bundle,
        no_trade_terminal=no_trade_terminal,
        timing={"evaluation_time_seconds": float(time.perf_counter() - started)},
    )
    return {"policy": policy, "policy_artifact": policy.to_artifact(), "evaluation_output": output, "optimizer_diagnostics": None}


def train_torchrl_policy(bundle, runtime):
    settings = _optimizer_settings(runtime)
    if settings["seed"] is not None:
        seed = int(settings["seed"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = bundle["time_grid_days"].device
    policy = _make_structured_policy(runtime, device=device)
    optimizer = optim.Adam(policy.parameters(), lr=settings["learning_rate"])

    total = _batch_size_from_bundle(bundle)
    val_fraction = min(max(settings["validation_fraction"], 0.0), 0.5)
    val_min = max(settings["validation_min_batch"], 1)
    if total > 2 * val_min and val_fraction > 0.0:
        val_count = min(max(int(total * val_fraction), val_min), total - val_min)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(settings["seed"] or 0) + 101)
        perm = torch.randperm(total, generator=gen)
        train_bundle = _slice_bundle_episodes(bundle, perm[val_count:].to(dtype=torch.int64))
        val_bundle = _slice_bundle_episodes(bundle, perm[:val_count].to(dtype=torch.int64))
    else:
        train_bundle = bundle
        val_bundle = bundle

    epoch_diag = []
    started = time.perf_counter()
    for epoch in range(settings["epochs"]):
        decision_indices = _decision_time_indices(train_bundle, settings, epoch=epoch, evaluation=False)
        decision_interval = _decision_interval_business_days_for_epoch(settings, epoch, evaluation=False)
        rollout_started = time.perf_counter()
        rollout = _collect_ppo_rollout(policy, train_bundle, runtime, decision_indices, deterministic=False, reward_scale=settings["reward_scale"])
        rollout_time = float(time.perf_counter() - rollout_started)
        if rollout is None:
            continue
        update_started = time.perf_counter()
        diag = _ppo_update(policy, optimizer, rollout, settings, decision_interval=decision_interval)
        update_time = float(time.perf_counter() - update_started)
        terminal_reward = float(rollout["terminal_transition"]["reward"].mean().item())
        net_pnl = float((rollout["terminal_transition"]["pnl_excess"] + rollout["terminal_transition"]["liability_value"]).mean().item())
        # action stats: action_bins are integer [0, n_bins). trade_delta = bin + min_trade.
        trade_deltas_seen = rollout["action_bins"] + policy._min_trade_delta
        action_abs_mean = float(trade_deltas_seen.float().abs().mean().item())
        epoch_diag.append({
            "epoch": epoch,
            "policy_loss": diag["policy_loss"],
            "value_loss": diag["value_loss"],
            "entropy": diag["entropy"],
            "advantage_mean": diag["advantage_mean"],
            "advantage_std": diag["advantage_std"],
            "average_reward": terminal_reward,
            "average_net_pnl": net_pnl,
            "action_abs_mean": action_abs_mean,
            "rollout_time_seconds": rollout_time,
            "update_time_seconds": update_time,
            "decision_interval_business_days": decision_interval,
        })
        print(
            f"epoch={epoch:>2} interval={decision_interval}d "
            f"net_pnl={net_pnl:>+12.0f} "
            f"reward={terminal_reward:>+10.2f} "
            f"|trade|={action_abs_mean:.2f} "
            f"pol_l={diag['policy_loss']:+.4f} val_l={diag['value_loss']:.2e} ent={diag['entropy']:.3f} "
            f"t={rollout_time + update_time:.2f}s",
            flush=True,
        )

    total_time = float(time.perf_counter() - started)

    eval_indices = _decision_time_indices(val_bundle, settings, epoch=None, evaluation=True)
    eval_rollout = _collect_ppo_rollout(policy, val_bundle, runtime, eval_indices, deterministic=True)
    no_trade_terminal = _collect_no_trade_rollout(val_bundle, runtime, eval_indices)
    output = build_torchrl_evaluation_output(
        eval_rollout["terminal_state"], eval_rollout["terminal_transition"], val_bundle,
        no_trade_terminal=no_trade_terminal,
        timing={"total_fit_time_seconds": total_time},
        optimizer_diagnostics={"epochs": epoch_diag},
    )
    return {"policy": policy, "policy_artifact": policy.to_artifact(), "evaluation_output": output, "optimizer_diagnostics": {"epochs": epoch_diag, "torchrl_rollout_time_seconds": total_time}}


def run_torchrl_execution(bundle, runtime):
    if runtime is None or bundle is None:
        return None
    if str(runtime.get("execution_mode", "")) == "optimize_policy":
        return train_torchrl_policy(bundle, runtime)
    return evaluate_torchrl_policy(bundle, runtime)


if __name__ == '__main__':
    pass
