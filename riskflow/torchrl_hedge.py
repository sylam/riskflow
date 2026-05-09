from __future__ import annotations

import math
import random
import time

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
    for key in ("privileged_factors", "spot_realized_vol",
                "spot_trend_20", "spot_trend_60", "spot_stretch_20",
                "hawkes_h_plus", "hawkes_h_minus", "hawkes_ratio",
                "spot_price_history"):
        store = bundle.get(key)
        if store:
            sliced[key] = {n: _slice_episode_tensor(t, episode_indices, batch_size) for n, t in store.items()}
    # hedge_ratios is a 2-deep dict: {tradable: {factor: (T, B)}} — slice the inner tensors only.
    if bundle.get("hedge_ratios"):
        sliced["hedge_ratios"] = {
            tradable: {
                factor: _slice_episode_tensor(t, episode_indices, batch_size)
                for factor, t in by_factor.items()
            }
            for tradable, by_factor in bundle["hedge_ratios"].items()
        }
    for scalar_key in ("last_settlement_index", "time_grid_days_cpu", "total_leg_volume",
                        "scenario_dates", "business_indices"):
        if scalar_key in bundle:
            sliced[scalar_key] = bundle[scalar_key]
    sliced["meta"]["validation_episode_count"] = int(episode_indices.numel())
    return sliced


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
    """Cheap per-step refresh: refreshes price/liability views and carries forward simulator state.
    Entity-state construction (the expensive transformer-shaped feature stack) is deferred to
    `_ensure_decision_views`, which only runs on decision steps where the policy actually reads
    it. With a 10-day decision interval this saves ~9 entity rebuilds per decision."""
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


def _dense_tracking_reward(prev_state, next_state, runtime):
    """Per-step shaping reward on the downside part of the tracking error. Two modes:

    `Dense_Reward_Mode = "asymmetric"` (default, current behavior):
        shaping_t = rs · fp · (max(−err_t, 0) − max(−err_{t+1}, 0))   for non-terminal t
        Terminal transition's shaping is FORCIBLY ZEROED in `reward_and_terminal_payoff`
        (to avoid penalizing the unavoidable force-flat close cost as fresh downside),
        so the telescoping sum is rs·fp·(|downside_0| − |downside_{T-1}|), NOT
        |downside_T|. Combined with the terminal asymmetric utility (−fp·|downside_T|
        on losses), the EFFECTIVE floor penalty is approximately (1 + rs)·fp — exact
        only when |downside_{T-1}| ≈ |downside_T|, i.e. when the close-cost segment
        (bid-offer × position × contract_size) is small relative to total downside.
        For non-trivial close costs the actual coefficient sits between fp and (1+rs)·fp.
        With rs=2, fp=10, sr=1 the agent is optimized as if effective fp ≈ 30 vs sr=1.
        Use rs=1 for terminal-parity (effective fp ≈ 2·fp), rs=0 to disable shaping.

    `Dense_Reward_Mode = "potential_based"` (Ng et al. 1999 PBRS):
        Φ(s) = −fp · max(−err(s), 0); shaping_t = γ·Φ(s_{t+1}) − Φ(s_t).
        Same terminal-zero applies; the PBRS invariance theorem holds for the
        kept-transitions sum γ^{T-1}·Φ(s_{T-1}) − Φ(s_0). Use this mode when you
        want densified credit assignment WITHOUT shifting the asymmetry of the
        terminal utility. `Dense_Tracking_Reward_Scale` is ignored in this mode
        (scaling Φ would break the invariance theorem)."""
    settings = dict(runtime.get("optimizer") or {})
    fp = float((runtime.get("objective") or {}).get("floor_penalty", 1.0))
    prev_err = _tracking_error_value(prev_state, runtime)
    next_err = _tracking_error_value(next_state, runtime)
    rc = float(settings.get("dense_tracking_reward_clip", 0.0))
    mode = str(settings.get("dense_reward_mode", "asymmetric")).lower()
    if mode == "potential_based":
        gamma = float(settings.get("gamma", 1.0))
        prev_pot = -fp * torch.clamp(-prev_err, min=0.0)
        next_pot = -fp * torch.clamp(-next_err, min=0.0)
        shaping = gamma * next_pot - prev_pot
        if rc > 0.0:
            shaping = torch.clamp(shaping, min=-rc, max=rc)
        return shaping
    # Asymmetric mode preserves original clip-before-scale semantics: rc is in dollar units of
    # downside change, then `rs · fp` scales the clipped shaping into reward-pre-Reward_Scale.
    rs = float(settings.get("dense_tracking_reward_scale", 0.0))
    shaping = torch.clamp(-prev_err, min=0.0) - torch.clamp(-next_err, min=0.0)
    if rc > 0.0:
        shaping = torch.clamp(shaping, min=-rc, max=rc)
    return rs * fp * shaping


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
    # Action bins from realized deltas — searchsorted reverse-lookup against the per-instrument
    # sorted delta list. Since limits are now reward-side rather than clipped, executed ≡ proposed.
    deltas_map = runtime.get("policy", {}).get("action_space", {}).get("trade_deltas", {})
    if deltas_map:
        per_instrument = [tuple(deltas_map[n]) for n in instrument_order]
        max_bins = max(len(d) for d in per_instrument) if per_instrument else 1
        padded = [list(d) + [d[-1]] * (max_bins - len(d)) for d in per_instrument]
        deltas_tensor = torch.tensor(padded, dtype=torch.int64, device=ordered.device)  # (I, max_bins)
        # searchsorted(sorted=(I, max_bins), values=(I, B)) → (I, B)
        bins_T = torch.searchsorted(deltas_tensor, ordered.transpose(0, 1).contiguous())
        bins = bins_T.transpose(0, 1).contiguous()
    else:
        # Fallback: legacy unit-step ranges. bin = delta - min.
        min_map = runtime.get("policy", {}).get("action_space", {}).get("min_trade_delta", {})
        min_trade_delta = torch.tensor(
            [int(min_map.get(n, 0)) for n in instrument_order],
            dtype=torch.int64, device=ordered.device,
        )
        bins = ordered - min_trade_delta
    return {
        **action,
        "action_bins": bins,
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


def _expiry_holding_penalty(state, bundle, runtime):
    """Per-step penalty for non-zero positions near or past contract expiry.
    Pre-expiry: smooth exp ramp inside the warning window. Post-expiry: large
    linear-in-days-past hammer. Plateauing past expiry leaves the policy with
    no marginal incentive to *exit* an already-expired contract — so the
    multiplier must keep growing once we cross zero.

    Coef from `Expiry_Penalty`; threshold from `Expiry_Threshold_Days` (default 4).
    Disabled when objective.expiry_penalty <= 0.
    """
    objective = runtime.get("objective") or {}
    coef = float(objective.get("expiry_penalty", 0.0))
    if coef <= 0.0:
        return None
    threshold = float(objective.get("expiry_threshold_days", 4.0))
    layout_instruments = (runtime.get("entity_layout", {})
                          .get("instruments", {}).get("meta", ()))
    if not layout_instruments:
        return None

    base_date = pd.Timestamp(bundle["meta"]["base_date"])
    # Use the CPU-cached day grid (Python list) — avoids a per-step GPU→CPU sync
    # that fires on every reward computation across the rollout.
    current_day_offset = int(bundle["time_grid_days_cpu"][int(state["time_index"])])

    device = bundle["time_grid_days"].device
    batch_size = _batch_size_from_bundle(bundle)
    penalty = torch.zeros(batch_size, dtype=torch.float32, device=device)
    any_active = False
    for entry in layout_instruments:
        name = entry["name"]
        if name not in state["positions"]:
            continue
        last_trade_day = (entry["last_trade_date"] - base_date).days
        days_to_expiry = last_trade_day - current_day_offset
        # Outside the warning window (still plenty of time) → no penalty.
        if days_to_expiry > threshold:
            continue
        if days_to_expiry >= 0:
            mult = math.exp(threshold - days_to_expiry)
        else:
            # Past expiry: continuous with the at-expiry value, then grows
            # linearly with days-past so each extra day of holding hurts more
            # than the last. Linear (not exp) avoids float32 overflow over a
            # multi-week post-expiry tail.
            mult = math.exp(threshold) * (1.0 + abs(days_to_expiry))
        position = state["positions"][name].to(device=device, dtype=torch.float32)
        price = state["tradable_values"][name].to(device=device, dtype=torch.float32)
        cs = float(_runtime_tradable(runtime, name).get("contract_size", 1.0))
        penalty = penalty - coef * mult * position.abs() * price * cs
        any_active = True
    return penalty if any_active else None


def _post_deal_trade_penalty(prev_state, state, bundle, runtime):
    """Per-step holding penalty on ANY non-zero position past the original
    liability's last settlement. Mirrors the expiry penalty's post-expiry form
    (linear-in-days-past × |position notional|) but applied at portfolio level
    rather than per-contract: once the deal is gone, no instrument should be
    held, regardless of which contract it is.

    Coefficient `Post_Deal_Trade_Penalty` × `(1 + days_past_settlement)` × Σ_i
    |position_i| × price_i × contract_size_i. Disabled when coef <= 0 or
    `last_settlement_index` missing.
    """
    objective = runtime.get("objective") or {}
    coef = float(objective.get("post_deal_trade_penalty", 0.0))
    if coef <= 0.0:
        return None
    last_settle = bundle.get("last_settlement_index")
    if last_settle is None:
        return None
    current_idx = int(state["time_index"])
    if current_idx <= int(last_settle):
        return None
    days_past = float(current_idx - int(last_settle))
    device = bundle["time_grid_days"].device
    batch_size = _batch_size_from_bundle(bundle)
    penalty = torch.zeros(batch_size, dtype=torch.float32, device=device)
    any_active = False
    for name, pos in state["positions"].items():
        price = state["tradable_values"][name].to(device=device, dtype=torch.float32)
        cs = float(_runtime_tradable(runtime, name).get("contract_size", 1.0))
        pos_abs = pos.to(device=device, dtype=torch.float32).abs()
        penalty = penalty - coef * (1.0 + days_past) * pos_abs * price * cs
        any_active = True
    return penalty if any_active else None


def _exp_linear_ramp(violation, threshold):
    """f(0)=0; exp(v)−1 for 0≤v<T; exp(T)−1 + exp(T)·(v−T) for v≥T. C¹ at v=T."""
    v = violation.clamp_min(0.0)
    eT = math.exp(threshold)
    return torch.exp(v.clamp_max(threshold)) - 1.0 + eT * (v - threshold).clamp_min(0.0)


def _position_bounds_penalty(state, bundle, runtime):
    """exp→linear penalty on the portfolio Σ_i|pos_i| total-position constraint:
      v = max(0, Σ_i|pos_i| − total_position_abs_limit)
    Scaled by held-position weighted average notional (dollar units, consistent with
    the rest of the reward stack). Disabled when coef <= 0 or limit <= 0.

    Per-instrument [min_position, max_position] bounds are enforced upstream by
    `StructuredRebalancePolicy._feasible_mask`, which sets logits to -inf for any
    bin whose resulting position would breach the per-instrument range — those
    violations are unreachable so a per-instrument soft term would always be zero."""
    objective = runtime.get("objective") or {}
    coef = float(objective.get("position_bounds_penalty", 0.0))
    threshold = float(objective.get("position_bounds_threshold", 5.0))
    total_limit = float((runtime.get("accounting") or {}).get("total_position_abs_limit", 0.0))
    if coef <= 0.0 or total_limit <= 0.0:
        return None
    device = bundle["time_grid_days"].device
    batch_size = _batch_size_from_bundle(bundle)
    abs_count = torch.zeros(batch_size, dtype=torch.float32, device=device)
    abs_notional = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for name in _runtime_names(runtime, "action_instruments"):
        if name not in state["positions"]:
            continue
        p = state["positions"][name].to(device=device, dtype=torch.float32).abs()
        price = state["tradable_values"][name].to(device=device, dtype=torch.float32)
        cs = float(_runtime_tradable(runtime, name).get("contract_size", 1.0))
        abs_count = abs_count + p
        abs_notional = abs_notional + p * price * cs
    avg_notional = abs_notional / abs_count.clamp_min(1.0e-9)
    return -coef * _exp_linear_ramp(abs_count - total_limit, threshold) * avg_notional


def reward_and_terminal_payoff(prev_state, state, bundle, runtime):
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    is_terminal = int(state["time_index"]) >= _last_time_index(bundle)
    # Skip dense shaping on the terminal transition: Force_Flat_At_End closes positions inside
    # this step's env update, debiting transaction cost from margin → tracking error drops by
    # the close cost → dense shaping reads it as fresh downside and penalizes the agent for an
    # unavoidable cost. Sacrifices one step of legitimate price-move signal (out of ~250) to
    # keep per-step gradients clean. Terminal asymmetric utility captures the closing P&L anyway.
    if is_terminal:
        reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
    else:
        reward = _dense_tracking_reward(prev_state, state, runtime).to(device=device, dtype=torch.float32)
    expiry = _expiry_holding_penalty(state, bundle, runtime)
    if expiry is not None:
        reward = reward + expiry.to(device=device, dtype=torch.float32)
    post_deal = _post_deal_trade_penalty(prev_state, state, bundle, runtime)
    if post_deal is not None:
        reward = reward + post_deal.to(device=device, dtype=torch.float32)
    bounds = _position_bounds_penalty(state, bundle, runtime)
    if bounds is not None:
        reward = reward + bounds.to(device=device, dtype=torch.float32)
    terminal_payoff = torch.zeros(batch_size, dtype=torch.float32, device=device)
    pnl_excess = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liability_value = state.get("cumulative_liability_value", torch.zeros(batch_size, dtype=torch.float32, device=device)).to(device=device, dtype=torch.float32)
    # Sync-free terminality check: done becomes True iff time_index hits last_idx, and is uniform
    # across the batch (the simulator advances all scenarios together). Avoids a `.item()` sync on
    # every rollout step.
    if is_terminal:
        # pnl_excess = portfolio change since inception; the asymmetric utility evaluates against
        # this (NOT absolute portfolio value) so the seed cash baseline doesn't bias the floor.
        pnl_excess = _pnl_excess(state, runtime).to(device=device, dtype=torch.float32)
        terminal_reward = _evaluate_objective(pnl_excess, liability_value, runtime).to(device=device, dtype=torch.float32)
        reward = reward + terminal_reward
        terminal_payoff = liability_value
    return {"reward": reward, "terminal_payoff": terminal_payoff, "pnl_excess": pnl_excess, "liability_value": liability_value}


def _bundle_scenario_dates(bundle):
    # Cached at bundle build; falls back to recompute only if a caller passes a bundle that
    # never went through `_build_torchrl_bundle` (e.g. handcrafted test fixtures).
    cached = bundle.get("scenario_dates")
    if cached is not None:
        return cached
    base_date = pd.Timestamp(bundle.get("meta", {}).get("base_date"))
    days = torch.as_tensor(bundle["time_grid_days"]).detach().cpu().to(dtype=torch.int64).tolist()
    return pd.DatetimeIndex([base_date + pd.Timedelta(days=int(d)) for d in days])


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
    # `business_indices` is precomputed at bundle build (cached list of business-day
    # rollout indices in the live-sim window, history rows excluded). Per-epoch only
    # the curriculum-driven stride changes — no GPU sync, no date re-walk per call.
    business = bundle.get("business_indices", ())
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
        "gamma": float(p.get("gamma", 1.0)),
        "gae_lambda": float(p.get("gae_lambda", 0.95)),
        "learning_rate": float(p.get("learning_rate", 3.0e-4)),
        "lr_schedule": str(p.get("lr_schedule", "constant")).lower(),
        "lr_min": float(p.get("lr_min", 0.0)),
        "lr_warmup_epochs": int(p.get("lr_warmup_epochs", 0)),
        "clip_eps": float(p.get("clip_eps", 0.2)),
        "value_coef": float(p.get("value_coef", 0.5)),
        "entropy_coef": float(p.get("entropy_coef", 0.01)),
        "entropy_schedule": str(p.get("entropy_schedule", "constant")).lower(),
        "entropy_coef_min": float(p.get("entropy_coef_min", 0.0)),
        "warm_start_epochs": int(p.get("warm_start_epochs", 0)),
        "max_grad_norm": float(p.get("max_grad_norm", 0.5)),
        "reward_scale": float(p.get("reward_scale", 1.0)),
        "validation_fraction": float(p.get("validation_fraction", 0.25)),
        "validation_min_batch": int(p.get("validation_min_batch", 256)),
        "decision_interval_curriculum": tuple(dict(s) for s in p.get("decision_interval_curriculum", ())),
        "anchor_beta": float(p.get("anchor_beta", 0.0)),
        "anchor_beta_floor": float(p.get("anchor_beta_floor", 0.0)),
        "anchor_anneal_epochs": int(p.get("anchor_anneal_epochs", 0)),
        "anchor_bin_sharpness": float(p.get("anchor_bin_sharpness", 2.0)),
        "anchor_target": str(p.get("anchor_target", "delta1_jul")).lower(),
        "cvar_alpha": float(p.get("cvar_alpha", 0.0)),
        "cvar_lambda": float(p.get("cvar_lambda", 0.0)),
        "value_loss_asym_weight": float(p.get("value_loss_asym_weight", 1.0)),
        "entropy_floor_h_min": float(p.get("entropy_floor_h_min", 0.0)),
        "entropy_floor_coef": float(p.get("entropy_floor_coef", 0.0)),
        "seed": p.get("seed"),
    }


def _make_structured_policy(runtime, *, device):
    policy_cfg = runtime.get("policy", {})
    model = policy_cfg.get("model", {})
    action_space_cfg = policy_cfg.get("action_space", {})
    instrument_order = tuple(action_space_cfg.get("instrument_order", ()))
    hedge_order = tuple(_runtime_names(runtime, "hedges"))
    if hedge_order != instrument_order:
        raise ValueError(
            "Action_Space.Instrument_Order must match Position_Limits/hedges order; "
            f"got Instrument_Order={instrument_order} but hedges={hedge_order}. "
            "The transformer outputs ctx['instruments'] in hedge order while the policy's "
            "feasibility masks are built in Instrument_Order — divergence silently miswires "
            "per-instrument position constraints onto the wrong logit row."
        )
    deltas_map = action_space_cfg.get("trade_deltas") or {}
    if deltas_map:
        per_instrument_deltas = tuple(tuple(deltas_map[n]) for n in instrument_order)
        action_space = StructuredActionSpace(
            instrument_order,
            trade_deltas=per_instrument_deltas,
        )
    else:
        action_space = StructuredActionSpace(
            instrument_order,
            min_trade_delta=tuple(action_space_cfg.get("min_trade_delta", {}).get(n, 0) for n in instrument_order),
            max_trade_delta=tuple(action_space_cfg.get("max_trade_delta", {}).get(n, 0) for n in instrument_order),
        )
    return StructuredRebalancePolicy(
        action_space=action_space,
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


def _textbook_targets_per_step(bundle, runtime, *, buyback_days=21):
    """Deterministic per-step textbook hedge target per instrument: pre-period_start fully short
    (Min_Position), linear buyback over `buyback_days` from the latest leg's period_start, zero
    after. Uses unclamped `time_to_period_start` to locate tn. Instruments with Min_Position >= 0
    (no short allowed) get zero throughout — matches diagnose_vs_textbook."""
    legs = bundle.get("legs")
    if not legs or legs.get("features") is None:
        return None
    feature_names = list(legs["feature_names"])
    if "time_to_period_start" not in feature_names:
        return None
    ttps_idx = feature_names.index("time_to_period_start")
    ttp = legs["features"][..., ttps_idx]
    while ttp.ndim > 2:
        ttp = ttp[:, 0]
    if ttp.ndim == 1:
        ttp = ttp.unsqueeze(-1)
    nonpos = ttp <= 0
    started = nonpos.any(dim=0)
    first_idx = nonpos.int().argmax(dim=0)
    tn_per_leg = torch.where(started, first_idx, torch.full_like(first_idx, ttp.shape[0] - 1))
    tn_idx = int(tn_per_leg.max().item())
    T = ttp.shape[0]
    position_limits = runtime.get("accounting", {}).get("position_limits", {})
    targets = {}
    for name in runtime["names"]["hedges"]:
        min_pos = float((position_limits.get(name) or {}).get("min_position", 0.0))
        target = torch.zeros(T, dtype=torch.float32)
        if min_pos >= 0.0:
            targets[name] = target
            continue
        for t in range(T):
            if t < tn_idx:
                target[t] = min_pos
            elif t < tn_idx + buyback_days:
                # Front-loaded ramp: u = (i+1)/N so day-1 of the averaging window already
                # steps off -short instead of wasting it as a no-op.
                u = (t - tn_idx + 1) / buyback_days
                target[t] = min_pos * (1.0 - u)
            else:
                target[t] = 0.0
        targets[name] = target
    return targets


def _collect_textbook_rollout(policy, bundle, runtime, decision_indices, *, reward_scale=1.0):
    """Roll out the textbook hedge through the env, recording (state, action_bin) pairs at every
    decision step for behavior-cloning warm-start. State trajectory is what the env sees under the
    textbook policy — not under the (still-random) network. Returns the same dict shape as
    `_collect_ppo_rollout` minus value/log_prob (BC doesn't need them)."""
    state = build_shared_state(bundle, runtime)
    decision_set = set(int(i) for i in decision_indices)
    batch_size = int(next(iter(state["positions"].values())).shape[0])
    device = policy.device
    instrument_order = list(policy.action_space.instrument_order)
    n_bins = policy._action_bins
    targets_per_step = _textbook_targets_per_step(bundle, runtime)
    if targets_per_step is None:
        return None
    # Host-side caches: avoid a per-step-per-instrument .item() sync (was ~3 syncs per decision).
    target_table = {name: targets_per_step[name].tolist() for name in instrument_order}
    # Per-instrument valid delta tables (handles non-contiguous spaces with gaps; padded slots
    # in `policy._trade_deltas` are excluded by slicing to n_bins[i]).
    valid_deltas_per_inst = [policy._trade_deltas[i, :n_bins[i]].to(device=device, dtype=torch.float32)
                              for i in range(len(instrument_order))]
    entity_snapshots = []
    privileged_snapshots = []
    position_snapshots = []
    action_bins_list = []
    last_idx = _last_time_index(bundle)
    while int(state["time_index"]) < last_idx:
        t = int(state["time_index"])
        step_action = None
        if t in decision_set:
            state = _ensure_decision_views(state, bundle, runtime)
            priv = _privileged_state_at(bundle, t, batch_size, device)
            positions_t = _positions_tensor(state, instrument_order, device=device)
            trade_per_instr = {}
            for i, name in enumerate(instrument_order):
                target_scalar = target_table[name][t]
                cur_pos = state["positions"][name].to(device=device, dtype=torch.float32)
                raw = (torch.full_like(cur_pos, target_scalar) - cur_pos).round()
                # Snap to nearest valid discrete delta. searchsorted-then-pick-listed-value
                # would silently mismatch against off-list values (e.g. raw=-48 → searchsorted
                # bin=-45, but env executes the raw -48 — recorded action_bin would not match
                # the actual trade). Argmin against the valid list is correct + straight-line.
                vd = valid_deltas_per_inst[i]
                best = (vd.unsqueeze(0) - raw.unsqueeze(-1)).abs().argmin(dim=-1)
                trade_per_instr[name] = vd[best]
            step_action = _realized_structured_action(
                {"trade_deltas": trade_per_instr}, state["positions"], runtime,
                batch_size=batch_size, device=device,
            )
            entity_snapshots.append(_entity_state_snapshot(state["entity_state"]))
            privileged_snapshots.append({k: v.detach().clone() for k, v in priv.items()})
            position_snapshots.append(positions_t.detach().clone())
            # Use the post-clip bins from `_realized_structured_action` so recorded actions
            # match what the env actually executes AND remain feasible under the policy's
            # mask. Pre-clip target bins can be infeasible if position_limits clip further
            # than the action range, producing -inf log_probs and NaN BC loss.
            action_bins_list.append(step_action["action_bins"].detach())
        next_state = step_runtime_state(state, step_action, bundle, runtime)
        state = next_state
    if not entity_snapshots:
        return None
    stacked_states = _stack_entity_states(entity_snapshots)
    stacked_privileged = {}
    if privileged_snapshots and privileged_snapshots[0]:
        for name in privileged_snapshots[0].keys():
            stacked_privileged[name] = torch.stack([s[name] for s in privileged_snapshots], dim=0)
    stacked_positions = torch.stack(position_snapshots, dim=0)
    return {
        "states": stacked_states,
        "privileged_states": stacked_privileged,
        "positions": stacked_positions,
        "action_bins": torch.stack(action_bins_list, dim=0),
    }


def _bc_update(policy, optimizer, rollout, settings):
    """Cross-entropy imitation loss: train policy logits to match recorded textbook action bins.
    Uses the same minibatching as _ppo_update (no GAE/values needed)."""
    flat_states, D, B = _flatten_for_minibatch(rollout["states"])
    flat_actions = rollout["action_bins"].reshape(D * B, -1)
    flat_privileged = {n: t.reshape(D * B, -1) for n, t in (rollout.get("privileged_states") or {}).items()}
    rollout_positions = rollout.get("positions")
    flat_positions = rollout_positions.reshape(D * B, -1) if rollout_positions is not None else None
    N = D * B
    mb = max(int(settings["minibatch_size"]), 1)
    device = flat_actions.device
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    minibatch_count = 0
    for _ in range(int(settings["ppo_epochs"])):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, mb):
            idx = perm[start:start + mb]
            mb_state = flat_states[idx]
            mb_actions = flat_actions[idx]
            mb_priv = {n: t[idx] for n, t in flat_privileged.items()} if flat_privileged else None
            mb_positions = flat_positions[idx] if flat_positions is not None else None
            new_lp, _entropy, _value, _logits = policy.evaluate_action(mb_state, mb_actions, privileged_state=mb_priv, positions=mb_positions)
            loss = -new_lp.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["max_grad_norm"])
            optimizer.step()
            loss_sum = loss_sum + loss.detach()
            minibatch_count += 1
    return {"bc_loss": float((loss_sum / max(minibatch_count, 1)).item())}


def _collect_ppo_rollout(policy, bundle, runtime, decision_indices, *, deterministic=False, reward_scale=1.0, anchor_targets=False):
    state = build_shared_state(bundle, runtime)
    decision_set = set(int(i) for i in decision_indices)
    batch_size = int(next(iter(state["positions"].values())).shape[0])
    device = policy.device
    instrument_order = policy.action_space.instrument_order
    debug_strict_bins = bool((runtime.get("optimizer") or {}).get("debug_strict_bins", False))
    # Anchor targets (textbook delta1) — recorded only when KL-anchor is active. Stored as the
    # *unclipped* target trade delta in real bin-space (target_position - current_position), so
    # the KL prior can build a Gaussian centered at the true textbook target and renormalize within
    # the feasible bin range, putting extra mass on the boundary bin when the target is out of range
    # rather than artificially flattening near the boundary.
    anchor_target_table = None
    if anchor_targets:
        targets_per_step = _textbook_targets_per_step(bundle, runtime)
        if targets_per_step is not None:
            anchor_target_table = {name: targets_per_step[name].tolist() for name in instrument_order}
    entity_snapshots = []
    privileged_snapshots = []
    position_snapshots = []
    action_bins_list = []
    target_deltas_list = []
    entropy_list = []
    n_feasible_list = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    # Always-on reward accumulator: pre-first-decision transitions (e.g. when sim-day-0 is a
    # holiday and the first business day is t>0) accumulate here and get rolled into the first
    # decision's reward bucket. Previously pending_reward was None until the first decision,
    # silently dropping any pre-decision transition rewards.
    pending_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
    seen_first_decision = False
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
            if seen_first_decision:
                rewards.append(pending_reward)
                dones.append(state["done"].to(dtype=torch.float32, device=device))
                pending_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
            seen_first_decision = True
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
            # Optional invariant check: the sampled bins (under the policy's int64-cast feasibility
            # mask) must equal the env's post-clip bins (under float-clamped position limits). True
            # in our setup because positions are always integer-valued; would silently corrupt the
            # PPO ratio if ever decoupled (e.g. fractional positions, lot-size != 1, mask drift).
            # Off by default — adds a CPU sync per decision step.
            if debug_strict_bins and not torch.equal(output["action_bins"], step_action["action_bins"]):
                raise AssertionError(
                    "PPO ratio decoupled: policy-sampled bins != env-clipped bins. "
                    "Check fractional positions, lot-size, or position-limit mask divergence."
                )
            entity_snapshots.append(_entity_state_snapshot(state["entity_state"]))
            privileged_snapshots.append({k: v.detach().clone() for k, v in priv.items()})
            position_snapshots.append(positions_t.detach().clone())
            # Record the SAMPLED bins (feasible by construction under the policy's mask) and their
            # log_prob.
            action_bins_list.append(output["action_bins"].detach())
            if anchor_target_table is not None:
                # Unclipped target_delta[b, i] = target_pos[t] - cur_pos[b]. May exceed the action
                # range — Gaussian prior in PPO update will renormalize within feasible bins, which
                # correctly puts extra mass on the boundary bin when target is out-of-range.
                target_deltas_step = torch.empty((batch_size, len(instrument_order)), dtype=torch.float32, device=device)
                for i, name in enumerate(instrument_order):
                    target_scalar = anchor_target_table[name][t]
                    cur_pos = state["positions"][name].to(device=device, dtype=torch.float32)
                    target_deltas_step[:, i] = torch.full_like(cur_pos, target_scalar) - cur_pos
                target_deltas_list.append(target_deltas_step)
            log_probs.append(output["log_prob"].detach())
            values.append(output["value"].detach())
            # Diagnostic-only: per-decision per-path entropy (summed over instruments) and the
            # count of feasible bins. Cheap (D × B floats per rollout) and lets post-hoc analysis
            # bin paths by win/lose and look for entropy-collapse / mask-pinning at specific time
            # indices. Computed from the same logits the policy sampled from, so they reflect
            # the actual decision distribution post-feasibility-mask.
            with torch.no_grad():
                logits_det = output["logits"].detach()
                log_probs_per = torch.log_softmax(logits_det, dim=-1)
                probs_per = log_probs_per.exp()
                safe_log = log_probs_per.masked_fill(log_probs_per == float('-inf'), 0.0)
                entropy_per_instr = -(probs_per * safe_log).sum(dim=-1)
                entropy_list.append(entropy_per_instr.sum(dim=-1))
                n_feasible_list.append(torch.isfinite(logits_det).sum(dim=-1).sum(dim=-1).to(dtype=torch.int32))
        next_state = step_runtime_state(state, step_action, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        pending_reward = pending_reward + transition["reward"].to(device=device, dtype=torch.float32) * reward_scale
        state = next_state
    if seen_first_decision:
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
        "target_deltas": torch.stack(target_deltas_list, dim=0) if target_deltas_list else None,
        "entropy_per_decision": torch.stack(entropy_list, dim=0),  # (D, B), sum over instruments
        "n_feasible_per_decision": torch.stack(n_feasible_list, dim=0),  # (D, B), sum over instruments
        "log_probs": torch.stack(log_probs, dim=0),
        "values": torch.stack(values, dim=0),
        "rewards": torch.stack(rewards, dim=0),
        "dones": torch.stack(dones, dim=0),
        "terminal_state": state,
        "terminal_transition": {**transition, "state": state},
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


def _ppo_update(policy, optimizer, rollout, settings, *, decision_interval=1, epoch=0):
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
    # CVaR-α weighting: rare crash paths are too dilute to drive the policy gradient under PPO's
    # mean-aggregating loss. Identify the bottom α-fraction by path-realized return, multiply
    # those paths' advantages by (1 + λ). λ=0 ⇒ no-op; λ>0 amplifies tail gradient. Path-level
    # (not per-decision): all decisions on a tail path carry the same upweight, propagating the
    # crash signal back to early decisions. When CVaR is active we skip advantage normalization —
    # the standardization was a variance-reduction trick that partly undoes intentional re-weighting,
    # and CVaR runs may need lr/clip_eps re-tuning to compensate for the new advantage scale.
    cvar_alpha = float(settings.get("cvar_alpha", 0.0))
    cvar_lambda = float(settings.get("cvar_lambda", 0.0))
    cvar_active = cvar_alpha > 0.0 and cvar_lambda > 0.0
    if cvar_active:
        path_realized = rollout["rewards"].sum(dim=0)
        tail_threshold = torch.quantile(path_realized.double(), cvar_alpha).to(dtype=path_realized.dtype)
        tail_mask = (path_realized <= tail_threshold).to(dtype=torch.float32)
        path_weight = 1.0 + cvar_lambda * tail_mask  # (B,)
        flat_weight = path_weight.unsqueeze(0).expand(D, -1).reshape(D * B)
        flat_advantages = flat_advantages * flat_weight
    else:
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
    flat_privileged = {}
    for name, t in (rollout.get("privileged_states") or {}).items():
        flat_privileged[name] = t.reshape(D * B, -1)
    rollout_positions = rollout.get("positions")
    flat_positions = rollout_positions.reshape(D * B, -1) if rollout_positions is not None else None
    # KL-anchor: textbook prior on the action distribution. β anneals linearly from anchor_beta
    # to anchor_beta_floor over anchor_anneal_epochs (anneal-to-zero ⇒ warm-start interpretation;
    # nonzero floor ⇒ persistent prior). Target is a Gaussian over bin space centered on the
    # unclipped textbook target delta, masked to feasible bins, then renormalized — preserves
    # ordinal action structure (close-to-textbook bins > far-from-textbook bins) and naturally
    # puts extra mass on the boundary bin when the textbook target is outside the action range.
    anchor_beta_init = float(settings.get("anchor_beta", 0.0))
    anchor_beta_floor = float(settings.get("anchor_beta_floor", 0.0))
    anneal_epochs = int(settings.get("anchor_anneal_epochs", 0))
    if anchor_beta_init > 0.0 and anneal_epochs > 0:
        frac = max(0.0, 1.0 - epoch / anneal_epochs)
        anchor_beta = anchor_beta_floor + (anchor_beta_init - anchor_beta_floor) * frac
    else:
        anchor_beta = anchor_beta_init
    anchor_sharpness = float(settings.get("anchor_bin_sharpness", 2.0))
    target_deltas = rollout.get("target_deltas")
    flat_target_deltas = target_deltas.reshape(D * B, -1) if (target_deltas is not None and anchor_beta > 0.0) else None
    N = D * B
    mb = max(int(settings["minibatch_size"]), 1)
    # Accumulate per-minibatch losses as device-side tensors and sync once at the end of the
    # update — saves ~3 .item() syncs × ~60 minibatches per update.
    device = flat_actions.device
    policy_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    value_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
    anchor_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
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
            # Per-element entropy floor: hinge loss penalizing any minibatch element whose entropy
            # falls below H_min. Standard `-coef × entropy.mean()` regularization can leave a
            # low-entropy *minimum* coexisting with a healthy mean — the failing-paths signature
            # we observe in Diagnostic F. The floor activates exactly when the bug manifests
            # (entropy < H_min on some path) and is silent when the policy is healthy.
            entropy_floor_h_min = float(settings.get("entropy_floor_h_min", 0.0))
            entropy_floor_coef = float(settings.get("entropy_floor_coef", 0.0))
            if entropy_floor_coef > 0.0:
                floor_violation = (entropy_floor_h_min - entropy).clamp_min(0.0)
                entropy_floor_loss = floor_violation.mean()
            else:
                entropy_floor_loss = torch.zeros((), dtype=torch.float32, device=device)
            ratio = (new_lp - mb_old_lp).exp()
            clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
            v_clipped = mb_old_v + (value - mb_old_v).clamp(-clip_eps, clip_eps)
            # Optional asymmetric weighting: penalize over-prediction (V > G, i.e. V optimistic
            # about a realized outcome) more than under-prediction. Counters the median-bias of
            # MSE on heavy-tailed-downside distributions. asym_weight=1.0 ⇒ symmetric MSE.
            sq_unclipped = (value - mb_ret).pow(2)
            sq_clipped = (v_clipped - mb_ret).pow(2)
            asym_weight = float(settings.get("value_loss_asym_weight", 1.0))
            if asym_weight != 1.0:
                over_unclipped = (value > mb_ret).to(dtype=sq_unclipped.dtype)
                over_clipped = (v_clipped > mb_ret).to(dtype=sq_clipped.dtype)
                w_unclipped = 1.0 + (asym_weight - 1.0) * over_unclipped
                w_clipped = 1.0 + (asym_weight - 1.0) * over_clipped
                sq_unclipped = w_unclipped * sq_unclipped
                sq_clipped = w_clipped * sq_clipped
            value_loss = 0.5 * torch.max(sq_unclipped, sq_clipped).mean()
            entropy_bonus = entropy.mean()
            # KL-anchor: forward KL(π_textbook || π_current) — pulls policy mass toward textbook
            # modes (right form when we want π to *cover* the prior rather than match marginals).
            # π_textbook is a Gaussian over bin-deltas centered on the unclipped textbook target,
            # masked to bins the policy considers feasible, then renormalized.
            anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
            if flat_target_deltas is not None:
                mb_target_deltas = flat_target_deltas[idx]  # (mb, n_inst)
                # Per-instrument delta lookup table — handles non-contiguous deltas
                # (e.g. [-50,-45,...,-10,-9,...,9,10,15,...,50]). Shape (1, n_inst, max_bins).
                trade_per_bin = policy._trade_deltas.unsqueeze(0).to(dtype=torch.float32)
                # Clip textbook target to feasible bin-delta range before centering the Gaussian.
                # When target is far out-of-range, the unclipped Gaussian underflows to zero across
                # all feasible bins in float32 and the prior provides no signal. Clipping puts the
                # Gaussian center on the boundary bin — "as far in the textbook direction as feasible".
                min_trade_per_inst = policy._min_trade_delta.to(dtype=torch.float32)  # (1, n_inst)
                max_trade_per_inst = policy._max_trade_delta.to(dtype=torch.float32)  # (1, n_inst)
                mb_target_clipped = mb_target_deltas.clamp(min_trade_per_inst, max_trade_per_inst)
                sq_dev = (trade_per_bin - mb_target_clipped.unsqueeze(-1)).pow(2)
                target_unnorm = torch.exp(-anchor_sharpness * sq_dev)
                # Feasibility mask from the policy's logits (already includes static + position
                # mask). Bins whose logits are -inf get probability zero in the prior too.
                feasible = torch.isfinite(logits)
                target_unnorm = target_unnorm * feasible.to(target_unnorm.dtype)
                target_dist = target_unnorm / target_unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                log_probs_dist = torch.log_softmax(logits, dim=-1)
                log_target = torch.log(target_dist.clamp_min(1e-12))
                # Infeasible bins have logits=-inf → log_probs_dist=-inf and target_dist=0.
                # 0 × -inf = NaN; replace -inf with 0 (safe because target_dist=0 there too,
                # so the contribution is identically zero).
                log_probs_safe = torch.where(torch.isfinite(log_probs_dist), log_probs_dist, torch.zeros_like(log_probs_dist))
                kl_per = (target_dist * (log_target - log_probs_safe)).sum(dim=-1)
                anchor_loss = kl_per.mean()
            loss = (
                policy_loss
                + settings["value_coef"] * value_loss
                - settings["entropy_coef"] * entropy_bonus
                + anchor_beta * anchor_loss
                + entropy_floor_coef * entropy_floor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["max_grad_norm"])
            optimizer.step()
            policy_loss_sum = policy_loss_sum + policy_loss.detach()
            value_loss_sum = value_loss_sum + value_loss.detach()
            entropy_sum = entropy_sum + entropy_bonus.detach()
            anchor_loss_sum = anchor_loss_sum + anchor_loss.detach()
            minibatch_count += 1
    denom = max(minibatch_count, 1)
    return {
        "policy_loss": float((policy_loss_sum / denom).item()),
        "value_loss": float((value_loss_sum / denom).item()),
        "entropy": float((entropy_sum / denom).item()),
        "anchor_loss": float((anchor_loss_sum / denom).item()),
        "anchor_beta": float(anchor_beta),
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
    """Run a deterministic-argmax rollout of the policy on `bundle`. If
    `runtime['policy']['artifact_path']` is set, loads weights from that artifact; otherwise
    falls back to a freshly-initialised policy (useful as a "policy-untrained" baseline)."""
    from .structured_policy import load_policy_artifact
    settings = _optimizer_settings(runtime)
    device = bundle["time_grid_days"].device
    artifact_path = (runtime.get("policy") or {}).get("artifact_path")
    if artifact_path:
        policy = load_policy_artifact(artifact_path, device=device)
        # Verify the eval runtime's privileged_layout matches the layout the policy was
        # trained against. If the runtime was built without `stoch_factors` (or with a
        # different set), the encoder's input dim won't match what the policy expects and
        # the value head will see a wrong-shaped tensor downstream — better to fail loud
        # here with a clear cause than to chase a cryptic shape error.
        train_layout = dict(policy.privileged_layout or {})
        eval_layout = dict(runtime.get("privileged_layout") or {})
        if train_layout.keys() != eval_layout.keys():
            raise ValueError(
                "Privileged-layout mismatch between loaded policy and current runtime. "
                f"Policy expects {sorted(train_layout.keys())}; runtime provides "
                f"{sorted(eval_layout.keys())}. Typical cause: "
                "construct_torchrl_runtime was called with stoch_factors=None or a "
                "different stoch_factors set than at training. Rebuild the runtime with "
                "the same stoch_factors used during training."
            )
    else:
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
    if settings["lr_schedule"] == "cosine":
        warmup_epochs = max(int(settings["lr_warmup_epochs"]), 0)
        total_epochs = max(int(settings["epochs"]), 1)
        cosine_epochs = max(total_epochs - warmup_epochs, 1)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=settings["lr_min"],
        )
        if warmup_epochs > 0 and settings["learning_rate"] > 0:
            # Linear warmup from lr_min → learning_rate over warmup_epochs, then cosine to lr_min.
            # Standard PPO recipe — peak LR can be set 3-10x higher than constant baseline since
            # warmup avoids the early-training instability that high LR alone would cause.
            start_factor = max(settings["lr_min"] / settings["learning_rate"], 1.0e-8)
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine
    else:
        scheduler = None

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
    entropy_start = settings["entropy_coef"]
    entropy_end = settings["entropy_coef_min"]
    epochs_total = max(int(settings["epochs"]), 1)
    warm_start_epochs = int(settings.get("warm_start_epochs", 0))
    if warm_start_epochs > 0:
        warm_indices = _decision_time_indices(train_bundle, settings, epoch=0, evaluation=False)
        tb_rollout = _collect_textbook_rollout(policy, train_bundle, runtime, warm_indices)
        if tb_rollout is not None:
            for warm_epoch in range(warm_start_epochs):
                bc_diag = _bc_update(policy, optimizer, tb_rollout, settings)
                print(f"warm={warm_epoch:>2} bc_loss={bc_diag['bc_loss']:+.4f}", flush=True)
    for epoch in range(settings["epochs"]):
        if settings["entropy_schedule"] == "linear" and epochs_total > 1:
            frac = epoch / (epochs_total - 1)
            settings["entropy_coef"] = entropy_start + (entropy_end - entropy_start) * frac
        decision_indices = _decision_time_indices(train_bundle, settings, epoch=epoch, evaluation=False)
        decision_interval = _decision_interval_business_days_for_epoch(settings, epoch, evaluation=False)
        rollout_started = time.perf_counter()
        rollout = _collect_ppo_rollout(policy, train_bundle, runtime, decision_indices, deterministic=False, reward_scale=settings["reward_scale"], anchor_targets=settings.get("anchor_beta", 0.0) > 0.0)
        rollout_time = float(time.perf_counter() - rollout_started)
        if rollout is None:
            continue
        update_started = time.perf_counter()
        diag = _ppo_update(policy, optimizer, rollout, settings, decision_interval=decision_interval, epoch=epoch)
        update_time = float(time.perf_counter() - update_started)
        if scheduler is not None:
            scheduler.step()
        terminal_reward = float(rollout["terminal_transition"]["reward"].mean().item())
        net_pnl_t = (rollout["terminal_transition"]["pnl_excess"] + rollout["terminal_transition"]["liability_value"]).to(dtype=torch.float64)
        net_pnl = float(net_pnl_t.mean().item())
        std_net_pnl = float(net_pnl_t.std().item())
        worst_net_pnl = float(net_pnl_t.min().item())
        p5_net_pnl = float(torch.quantile(net_pnl_t, 0.05).item())
        # action_bins are integer [0, n_bins); deltas may be non-contiguous (e.g.
        # [-50,-45,...,-10,-9,...,9,10,15,...,50]) so look up the actual delta from
        # `policy._trade_deltas[i, bin]` instead of assuming bin + min.
        bins = rollout["action_bins"]  # (D, B, I)
        inst_idx = torch.arange(policy._trade_deltas.shape[0], device=policy._trade_deltas.device)
        trade_deltas_seen = policy._trade_deltas[inst_idx, bins]
        action_abs_mean = float(trade_deltas_seen.float().abs().mean().item())
        # Per-decision entropy / feasibility — summary across the rollout's decision steps.
        # Surfaces entropy-collapse and over-restricted feasibility before they show up in
        # the rollout-averaged `entropy` from PPO update diag (which can hide a single
        # collapsed decision step inside a healthy rollout-mean).
        ent_per_dec = rollout["entropy_per_decision"].to(dtype=torch.float32)        # (D, B)
        nfe_per_dec = rollout["n_feasible_per_decision"].to(dtype=torch.float32)     # (D, B)
        min_entropy_decision = float(ent_per_dec.mean(dim=-1).min().item())
        min_n_feasible_decision = float(nfe_per_dec.mean(dim=-1).min().item())
        epoch_diag.append({
            "epoch": epoch,
            "policy_loss": diag["policy_loss"],
            "value_loss": diag["value_loss"],
            "entropy": diag["entropy"],
            "anchor_loss": diag["anchor_loss"],
            "anchor_beta": diag["anchor_beta"],
            "advantage_mean": diag["advantage_mean"],
            "advantage_std": diag["advantage_std"],
            "average_reward": terminal_reward,
            "average_net_pnl": net_pnl,
            "std_net_pnl": std_net_pnl,
            "worst_net_pnl": worst_net_pnl,
            "p5_net_pnl": p5_net_pnl,
            "action_abs_mean": action_abs_mean,
            "min_entropy_decision": min_entropy_decision,
            "min_n_feasible_decision": min_n_feasible_decision,
            "rollout_time_seconds": rollout_time,
            "update_time_seconds": update_time,
            "decision_interval_business_days": decision_interval,
        })
        # Live per-epoch dump for buffered-stdout-blind monitoring. Atomic via temp+rename
        # so a reader never sees a partial write. Skipped when no path is set in runtime.
        live_path = runtime.get("live_diag_path")
        if live_path:
            import json as _jsonlib, os as _os
            tmp = live_path + ".tmp"
            with open(tmp, "w") as _f:
                _jsonlib.dump({"epochs": epoch_diag, "current_epoch": int(epoch),
                               "total_epochs": int(settings["epochs"])}, _f)
            _os.replace(tmp, live_path)
        anchor_str = f" anc_kl={diag['anchor_loss']:.3f} β={diag['anchor_beta']:.3f}" if diag["anchor_beta"] > 0.0 else ""
        print(
            f"epoch={epoch:>2} interval={decision_interval}d "
            f"net_pnl={net_pnl:>+12.0f} std={std_net_pnl:>10.0f} worst={worst_net_pnl:>+12.0f} "
            f"reward={terminal_reward:>+10.2f} "
            f"|trade|={action_abs_mean:.2f} "
            f"pol_l={diag['policy_loss']:+.4f} val_l={diag['value_loss']:.2e} ent={diag['entropy']:.3f}{anchor_str} "
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
