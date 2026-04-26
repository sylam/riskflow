from __future__ import annotations

from copy import deepcopy
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer
from .hedge_runtime import build_state_feature_groups, flatten_state_feature_groups
from .structured_policy import StructuredActionSpace, StructuredRebalancePolicy, TorchRLStateFeatureExtractor


def _clone_policy_state(policy: StructuredRebalancePolicy) -> Dict[str, torch.Tensor]:
    return deepcopy(policy.state_dict())


OPTIMIZER_DEFAULTS = {
    "epochs": 20,
    "replay_capacity": 10000,
    "batch_size": 64,
    "gamma": 0.99,
    "learning_rate": 1.0e-3,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_epochs": 10,
    "target_update_interval": 5,
    "gradient_steps_per_epoch": 10,
    "decision_interval_curriculum": (),
    "dense_tracking_reward_scale": 0.05,
    "dense_tracking_reward_clip": 0.15,
    "validation_fraction": 0.25,
    "validation_min_batch": 256,
    "validation_shards": 4,
    "top_validation_checkpoints": 3,
    "performance_gated_curriculum": True,
    "curriculum_advance_patience": 2,
    "curriculum_min_improvement": 3.0,
    "curriculum_min_trade_rate": 0.01,
    "curriculum_max_trade_rate": 0.70,
    "turnover_penalty_scale": 0.75,
    "no_trade_reference_scale": 0.0,
    "replay_imitation_weight": 0.01,
    "action_sparsity_penalty_weight": 0.0,
}


def _checkpoint_selection_metrics(
    greedy_rollout: Dict[str, Any],
    greedy_terminal_transition: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, float]:
    average_reward = float(greedy_terminal_transition["reward"].mean().item())
    average_net_pnl = float(
        (
            greedy_terminal_transition["portfolio_value"]
            - greedy_terminal_transition["liability_value"]
        ).mean().item()
    )
    average_liability = float(greedy_terminal_transition["liability_value"].abs().mean().item())
    rollout_stats = _scalar_rollout_diagnostics(greedy_rollout.get("rollout_diagnostics"))
    mean_abs_trade_delta = float(rollout_stats.get("mean_abs_trade_delta", 0.0))
    nonzero_trade_rate = float(rollout_stats.get("nonzero_trade_rate", 0.0))
    objective = dict(runtime.get("objective") or {})
    floor_penalty = max(float(objective.get("floor_penalty", 1.0)), 1.0)
    inactivity_scale = max(average_liability, 1.0) * floor_penalty
    action_space = dict(runtime.get("policy", {}).get("action_space", {}))
    max_trade_delta = tuple(abs(float(value)) for value in action_space.get("max_trade_delta", {}).values())
    mean_trade_limit = max(float(sum(max_trade_delta) / len(max_trade_delta)), 1.0) if max_trade_delta else 1.0
    mean_trade_utilization = mean_abs_trade_delta / mean_trade_limit
    inactivity_penalty = inactivity_scale * max(0.05 - nonzero_trade_rate, 0.0) * 0.02
    selection_score = average_net_pnl - inactivity_penalty
    return {
        "average_reward": average_reward,
        "average_net_pnl": average_net_pnl,
        "average_liability": average_liability,
        "nonzero_trade_rate": nonzero_trade_rate,
        "mean_abs_trade_delta": mean_abs_trade_delta,
        "mean_trade_utilization": float(mean_trade_utilization),
        "selection_score": float(selection_score),
    }


def _turnover_penalty(ordered_trade_deltas: torch.Tensor, runtime: Dict[str, Any]) -> torch.Tensor:
    action_space = dict(runtime.get("policy", {}).get("action_space", {}))
    max_trade_delta = tuple(abs(float(value)) for value in action_space.get("max_trade_delta", {}).values())
    mean_trade_limit = max(float(sum(max_trade_delta) / len(max_trade_delta)), 1.0) if max_trade_delta else 1.0
    penalty_scale = float((runtime.get("optimizer") or {}).get("turnover_penalty_scale", OPTIMIZER_DEFAULTS["turnover_penalty_scale"]))
    if penalty_scale <= 0.0:
        return torch.zeros(int(ordered_trade_deltas.shape[0]), dtype=torch.float32, device=ordered_trade_deltas.device)
    normalized_turnover = ordered_trade_deltas.abs().to(dtype=torch.float32).mean(dim=1) / mean_trade_limit
    return penalty_scale * normalized_turnover


def _is_better_checkpoint(candidate: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    candidate_score = float(candidate.get("selection_score", float("-inf")))
    incumbent_score = float(incumbent.get("selection_score", float("-inf")))
    if candidate_score != incumbent_score:
        return candidate_score > incumbent_score
    candidate_reward = float(candidate.get("average_reward", float("-inf")))
    incumbent_reward = float(incumbent.get("average_reward", float("-inf")))
    if candidate_reward != incumbent_reward:
        return candidate_reward > incumbent_reward
    candidate_net_pnl = float(candidate.get("average_net_pnl", float("-inf")))
    incumbent_net_pnl = float(incumbent.get("average_net_pnl", float("-inf")))
    if candidate_net_pnl != incumbent_net_pnl:
        return candidate_net_pnl > incumbent_net_pnl
    candidate_trade_rate = float(candidate.get("nonzero_trade_rate", float("-inf")))
    incumbent_trade_rate = float(incumbent.get("nonzero_trade_rate", float("-inf")))
    return candidate_trade_rate > incumbent_trade_rate


def _checkpoint_sort_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        float(metrics.get("selection_score", float("-inf"))),
        float(metrics.get("average_reward", float("-inf"))),
        float(metrics.get("average_net_pnl", float("-inf"))),
        float(metrics.get("nonzero_trade_rate", float("-inf"))),
    )


def _batch_size_from_bundle(bundle: Dict[str, Any]) -> int:
    tradables = bundle.get("tradables", {})
    for tensor in tradables.values():
        return tensor.shape[-1]
    factors = bundle.get("factors", {})
    for tensor in factors.values():
        return int(tensor.shape[-1])
    return 0


def _slice_episode_tensor(tensor: torch.Tensor, episode_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    tensor_indices = episode_indices.to(device=tensor.device, dtype=torch.int64)
    if tensor.ndim >= 2 and int(tensor.shape[1]) == int(batch_size):
        return tensor.index_select(1, tensor_indices)
    if tensor.ndim >= 1 and int(tensor.shape[-1]) == int(batch_size):
        return tensor.index_select(tensor.ndim - 1, tensor_indices)
    return tensor


def _slice_bundle_episodes(bundle: Dict[str, Any], episode_indices: torch.Tensor) -> Dict[str, Any]:
    batch_size = _batch_size_from_bundle(bundle)
    if batch_size <= 0:
        return bundle
    sliced_bundle = {
        "time_grid_days": bundle["time_grid_days"],
        "tradables": {
            name: _slice_episode_tensor(tensor, episode_indices, batch_size)
            for name, tensor in bundle.get("tradables", {}).items()
        },
        "meta": dict(bundle.get("meta", {})),
    }
    if bundle.get("factors"):
        sliced_bundle["factors"] = {
            name: _slice_episode_tensor(tensor, episode_indices, batch_size)
            for name, tensor in bundle.get("factors", {}).items()
        }
    if bundle.get("hedge_profile") is not None:
        hedge_profile = dict(bundle.get("hedge_profile", {}))
        sliced_bundle["hedge_profile"] = {
            "features": _slice_episode_tensor(hedge_profile["features"], episode_indices, batch_size),
            "feature_names": tuple(hedge_profile.get("feature_names", ())),
            "liability_cashflows": {
                currency: _slice_episode_tensor(tensor, episode_indices, batch_size)
                for currency, tensor in hedge_profile.get("liability_cashflows", {}).items()
            },
            "liability_mtm": _slice_episode_tensor(hedge_profile["liability_mtm"], episode_indices, batch_size),
        }
    sliced_bundle["meta"]["validation_episode_count"] = int(episode_indices.numel())
    return sliced_bundle


def _split_episode_indices(total_episode_count: int, num_splits: int) -> Tuple[torch.Tensor, ...]:
    if total_episode_count <= 0:
        return tuple()
    normalized_splits = max(min(int(num_splits), total_episode_count), 1)
    full_indices = torch.arange(total_episode_count, dtype=torch.int64)
    return tuple(chunk for chunk in torch.tensor_split(full_indices, normalized_splits) if int(chunk.numel()) > 0)


def _aggregate_checkpoint_metrics(metrics_list: Tuple[Dict[str, float], ...]) -> Dict[str, float]:
    if not metrics_list:
        return {
            "average_reward": float("-inf"),
            "average_net_pnl": float("-inf"),
            "average_liability": 0.0,
            "nonzero_trade_rate": 0.0,
            "mean_abs_trade_delta": 0.0,
            "mean_trade_utilization": 0.0,
            "selection_score": float("-inf"),
        }
    keys = metrics_list[0].keys()
    return {
        key: float(sum(float(metrics[key]) for metrics in metrics_list) / len(metrics_list))
        for key in keys
    }


def _zeros_by_name(names, batch_size, *, device, dtype=torch.float32):
    return {
        str(name): torch.zeros(batch_size, dtype=dtype, device=device)
        for name in names
    }


def _runtime_names(runtime: Dict[str, Any], key: str) -> tuple:
    return tuple(name for name in runtime.get("names", {}).get(key, ()))


def _runtime_tradable(runtime: Dict[str, Any], instrument_name: str) -> Dict[str, Any]:
    return dict(runtime.get("tradables", {}).get(str(instrument_name), {}))

def _clone_tensor_dict(values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        str(name): tensor.clone()
        for name, tensor in values.items()
    }


def _state_layout_flat(runtime: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(runtime.get("state_layout", {}).get("flat", ()))


def _bundle_hedge_profile(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return dict(bundle.get("hedge_profile", {}))


def _hedge_profile_feature_dim(bundle: Dict[str, Any]) -> int:
    hedge_profile = _bundle_hedge_profile(bundle)
    feature_tensor = hedge_profile.get("features")
    return 0 if feature_tensor is None else int(feature_tensor.shape[-1])


def _current_hedge_features(bundle: Dict[str, Any], time_index: int) -> torch.Tensor:
    hedge_profile = _bundle_hedge_profile(bundle)
    feature_tensor = hedge_profile.get("features")
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    if feature_tensor is None:
        return torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
    return feature_tensor[time_index].to(dtype=torch.float32)


def _cash_account_for_currency(currency: str, runtime: Dict[str, Any]) -> Optional[str]:
    currency_map = _cash_account_currency_map(runtime)
    for account_name in _runtime_names(runtime, "cash_accounts"):
        if currency_map.get(str(account_name)) == str(currency):
            return str(account_name)
    return None


def _current_liability_cashflow_value(
    tradable_values: Dict[str, torch.Tensor],
    bundle: Dict[str, Any],
    runtime: Dict[str, Any],
    time_index: int,
) -> torch.Tensor:
    batch_size = _batch_size_from_bundle(bundle)
    template = next(iter(tradable_values.values()), None)
    device = bundle["time_grid_days"].device if template is None else template.device
    liability_value = torch.zeros(batch_size, dtype=torch.float32, device=device)
    hedge_profile = _bundle_hedge_profile(bundle)
    for currency, cashflow_tensor in hedge_profile.get("liability_cashflows", {}).items():
        currency_cashflow = cashflow_tensor[time_index].to(dtype=torch.float32, device=device)
        cash_account_name = _cash_account_for_currency(str(currency), runtime)
        if cash_account_name is None:
            liability_value = liability_value + currency_cashflow
            continue
        cash_account_value = tradable_values.get(cash_account_name)
        if cash_account_value is None:
            liability_value = liability_value + currency_cashflow
            continue
        liability_value = liability_value + currency_cashflow * cash_account_value.to(dtype=torch.float32)
    return liability_value


def _current_liability_mtm_value(bundle: Dict[str, Any], time_index: int) -> torch.Tensor:
    hedge_profile = _bundle_hedge_profile(bundle)
    liability_mtm = hedge_profile.get("liability_mtm")
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    if liability_mtm is None:
        return torch.zeros(batch_size, dtype=torch.float32, device=device)
    return liability_mtm[time_index].to(dtype=torch.float32, device=device)


def _current_tradable_values(bundle: Dict[str, Any], runtime: Dict[str, Any], time_index: int) -> Dict[str, torch.Tensor]:
    tradables = bundle.get("tradables", {})
    return {
        name: tradables[str(name)][time_index].to(dtype=torch.float32)
        for name in _runtime_names(runtime, "tradables")
        if name in tradables
    }


def _sanitize_policy_features(policy_features: torch.Tensor) -> torch.Tensor:
    sanitized = torch.nan_to_num(
        torch.as_tensor(policy_features, dtype=torch.float32),
        nan=0.0,
        posinf=1.0e6,
        neginf=-1.0e6,
    )
    compressed = torch.sign(sanitized) * torch.log1p(sanitized.abs())
    return torch.clamp(compressed, min=-16.0, max=16.0)


def _refresh_state_views(state: Dict[str, Any], bundle: Dict[str, Any], runtime: Dict[str, Any], time_index: int) -> Dict[str, Any]:
    refreshed_state = dict(state)
    tradable_values = _current_tradable_values(bundle, runtime, time_index)
    feature_groups = build_state_feature_groups(
        runtime,
        tradable_values=tradable_values,
        positions=refreshed_state["positions"],
        cash_accounts=refreshed_state["cash_accounts"],
    )
    base_policy_features = flatten_state_feature_groups(feature_groups, _state_layout_flat(runtime))
    hedge_policy_features = _current_hedge_features(bundle, time_index)
    policy_features = _sanitize_policy_features(torch.cat([base_policy_features, hedge_policy_features], dim=1))
    return {
        "time_index": int(time_index),
        "done": refreshed_state["done"],
        "positions": refreshed_state["positions"],
        "cash_accounts": refreshed_state["cash_accounts"],
        "margin_accounts": refreshed_state["margin_accounts"],
        "realized_pnl": refreshed_state["realized_pnl"],
        "variation_margin": refreshed_state["variation_margin"],
        "settlement_prices": refreshed_state["settlement_prices"],
        "tradable_values": tradable_values,
        "feature_groups": feature_groups,
        "policy_features": policy_features,
        "liability_cashflow_value": _current_liability_cashflow_value(tradable_values, bundle, runtime, time_index),
        "liability_mtm_value": _current_liability_mtm_value(bundle, time_index),
        "feature_names": tuple(_bundle_hedge_profile(bundle).get("feature_names", ())),
        "cumulative_liability_value": refreshed_state.get(
            "cumulative_liability_value",
            torch.zeros(int(policy_features.shape[0]), dtype=torch.float32, device=policy_features.device),
        ).to(dtype=torch.float32),
    }


def _account_market_values(account_values: Dict[str, torch.Tensor], tradable_values: Dict[str, torch.Tensor], account_names) -> Dict[str, torch.Tensor]:
    account_market_values = {}
    for account_name in account_names:
        name = str(account_name)
        if name in tradable_values:
            account_market_values[name] = account_values[name].to(dtype=torch.float32) * tradable_values[name]
        else:
            account_market_values[name] = account_values[name].to(dtype=torch.float32)
    return account_market_values


def _portfolio_value(state: Dict[str, Any], runtime: Dict[str, Any]) -> torch.Tensor:
    accounting_mode = str(runtime.get("accounting_mode", "futures"))
    position_total = torch.zeros_like(state["done"], dtype=torch.float32)
    for tensor in state.get("feature_groups", {}).get("position_values", {}).values():
        position_total = position_total + tensor.to(dtype=torch.float32)

    if accounting_mode == "cash_account":
        cash_total = torch.zeros_like(position_total)
        cash_market_values = _account_market_values(
            state.get("cash_accounts", {}),
            state.get("tradable_values", {}),
            _runtime_names(runtime, "cash_accounts"),
        )
        for tensor in cash_market_values.values():
            cash_total = cash_total + tensor.to(dtype=torch.float32)
        return position_total + cash_total

    margin_total = torch.zeros_like(position_total)
    margin_market_values = _account_market_values(
        state.get("margin_accounts", {}),
        state.get("tradable_values", {}),
        _runtime_names(runtime, "cash_accounts"),
    )
    for tensor in margin_market_values.values():
        margin_total = margin_total + tensor.to(dtype=torch.float32)
    return margin_total


def _terminal_portfolio_value(state: Dict[str, Any], runtime: Dict[str, Any]) -> torch.Tensor:
    return _portfolio_value(state, runtime)


def _tracking_error_value(state: Dict[str, Any], runtime: Dict[str, Any]) -> torch.Tensor:
    portfolio_value = _portfolio_value(state, runtime).to(dtype=torch.float32)
    liability_mtm_value = torch.as_tensor(
        state.get("liability_mtm_value", torch.zeros_like(portfolio_value)),
        dtype=torch.float32,
        device=portfolio_value.device,
    )
    return portfolio_value - liability_mtm_value


def _dense_tracking_reward(previous_state: Dict[str, Any], next_state: Dict[str, Any], runtime: Dict[str, Any]) -> torch.Tensor:
    previous_error = _tracking_error_value(previous_state, runtime)
    next_error = _tracking_error_value(next_state, runtime)
    scale = torch.maximum(
        torch.as_tensor(
            next_state.get("liability_mtm_value", torch.zeros_like(next_error)),
            dtype=torch.float32,
            device=next_error.device,
        ).abs(),
        torch.ones_like(next_error),
    )
    previous_distance = previous_error.abs() / scale
    next_distance = next_error.abs() / scale
    optimizer_settings = dict(runtime.get("optimizer") or {})
    reward_scale = float(
        optimizer_settings.get(
            "dense_tracking_reward_scale",
            OPTIMIZER_DEFAULTS["dense_tracking_reward_scale"],
        )
    )
    reward_clip = float(
        optimizer_settings.get(
            "dense_tracking_reward_clip",
            OPTIMIZER_DEFAULTS["dense_tracking_reward_clip"],
        )
    )
    shaping_reward = previous_distance - next_distance
    if reward_clip > 0.0:
        shaping_reward = torch.clamp(shaping_reward, min=-reward_clip, max=reward_clip)
    return reward_scale * shaping_reward


def _evaluate_objective(hedge_pnl: torch.Tensor, liability: torch.Tensor, runtime: Dict[str, Any]) -> torch.Tensor:
    objective = runtime.get("objective")
    net_pnl = hedge_pnl - liability
    if objective is None:
        return net_pnl
    object_name = str(objective.get("object"))
    if object_name != "TerminalFloorThenSurplusUtility":
        raise ValueError("Unsupported objective object for TorchRL reward path")
    floor_penalty = float(objective.get("floor_penalty", 1.0))
    surplus_reward = float(objective.get("surplus_reward", 1.0))
    power = float(objective.get("power", 1.0))
    non_negative = net_pnl >= 0.0
    surplus_value = surplus_reward * torch.pow(torch.clamp(net_pnl, min=0.0), power)
    shortfall_value = -floor_penalty * torch.pow(torch.clamp(-net_pnl, min=0.0), power)
    return torch.where(non_negative, surplus_value, shortfall_value)


def _cash_account_currency_map(runtime: Dict[str, Any]) -> Dict[str, str]:
    cash_accounts = runtime.get("accounting", {}).get("cash_accounts", {})
    return {
        str(account_name): str(account_data.get("currency"))
        for account_name, account_data in cash_accounts.items()
    }

def _cash_account_for_instrument(instrument_name: str, runtime: Dict[str, Any]) -> Optional[str]:
    instrument_currency = str(_runtime_tradable(runtime, str(instrument_name)).get("currency"))
    currency_map = _cash_account_currency_map(runtime)
    for account_name in _runtime_names(runtime, "cash_accounts"):
        if currency_map.get(str(account_name)) == instrument_currency:
            return str(account_name)
    cash_account_names = _runtime_names(runtime, "cash_accounts")
    return None if not cash_account_names else cash_account_names[0]

def _coerce_batch_tensor(value: Any, batch_size: int, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        return tensor.repeat(batch_size)
    if tensor.ndim == 1 and int(tensor.shape[0]) == batch_size:
        return tensor
    raise ValueError("Action tensor shape must be scalar or [batch]")


def _resolve_trade_deltas(
    action: Optional[Dict[str, Any]],
    runtime: Dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if action is None:
        return _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    if "trade_deltas" not in action:
        raise ValueError("Structured TorchRL actions must include a trade_deltas mapping")
    raw_trade_deltas = action["trade_deltas"]
    if not isinstance(raw_trade_deltas, dict):
        raise ValueError("Structured TorchRL trade_deltas must be a mapping keyed by instrument name")
    resolved = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    for instrument_name, value in raw_trade_deltas.items():
        name = str(instrument_name)
        if name in resolved:
            resolved[name] = _coerce_batch_tensor(value, batch_size, device=device)
    return resolved


def _realized_structured_action(
    action: Optional[Dict[str, Any]],
    current_positions: Dict[str, torch.Tensor],
    runtime: Dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    if action is None:
        return None
    executed_trade_deltas = _enforce_position_limits(
        current_positions,
        _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device),
        runtime,
    )
    instrument_order = tuple(str(name) for name in _runtime_names(runtime, "action_instruments"))
    ordered_trade_deltas = torch.stack(
        [executed_trade_deltas[name] for name in instrument_order],
        dim=1,
    ).round().to(dtype=torch.int64)
    return {
        **action,
        "ordered_trade_deltas": ordered_trade_deltas,
        "trade_deltas": {
            instrument_name: ordered_trade_deltas[:, index]
            for index, instrument_name in enumerate(instrument_order)
        },
    }


def _enforce_position_limits(
    current_positions: Dict[str, torch.Tensor],
    trade_deltas: Dict[str, torch.Tensor],
    runtime: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    adjusted_trade_deltas = dict(trade_deltas)
    position_limits = dict(runtime.get("accounting", {}).get("position_limits", {}))
    fail_on_unhedgeable = bool(runtime.get("accounting", {}).get("fail_on_unhedgeable_intent", False))
    for instrument_name in _runtime_names(runtime, "action_instruments"):
        name = str(instrument_name)
        if name not in current_positions or name not in adjusted_trade_deltas:
            continue
        limits = dict(position_limits.get(name, {}))
        min_position = float(limits.get("min_position", float("-inf")))
        max_position = float(limits.get("max_position", float("inf")))
        requested_trade_delta = adjusted_trade_deltas[name]
        proposed_position = current_positions[name] + requested_trade_delta
        bounded_position = torch.clamp(proposed_position, min=min_position, max=max_position)
        if fail_on_unhedgeable and bool((bounded_position != proposed_position).any().item()):
            raise ValueError(f"Action for {name} breaches position limits")
        adjusted_trade_deltas[name] = bounded_position - current_positions[name]
    return adjusted_trade_deltas


def _transaction_costs(
    trade_delta: torch.Tensor,
    price: torch.Tensor,
    runtime: Dict[str, Any],
    instrument_name: str,
) -> torch.Tensor:
    unit_cost = float(runtime.get("accounting", {}).get("transaction_cost_per_unit", 0.0))
    spread_bps = float(runtime.get("accounting", {}).get("bid_offer_spread_bps", 0.0))
    contract_size = float(_runtime_tradable(runtime, instrument_name).get("contract_size", 1.0))
    per_unit_cost = trade_delta.abs() * unit_cost
    if spread_bps <= 0.0:
        return per_unit_cost
    half_spread = 0.5 * spread_bps * 1.0e-4
    spread_cost = trade_delta.abs() * price.abs() * contract_size * half_spread
    return per_unit_cost + spread_cost


def _apply_cash_trade(
    cash_accounts: Dict[str, torch.Tensor],
    account_name: Optional[str],
    trade_delta: torch.Tensor,
    price: torch.Tensor,
    transaction_cost: torch.Tensor,
) -> None:
    if account_name is None:
        return
    cash_accounts[account_name] = cash_accounts[account_name] - (trade_delta * price + transaction_cost)


def _apply_account_flow(accounts: Dict[str, torch.Tensor], account_name: Optional[str], amount: torch.Tensor) -> None:
    if account_name is None:
        return
    accounts[account_name] = accounts[account_name] + amount


def _last_time_index(bundle: Dict[str, Any]) -> int:
    return max(int(bundle["time_grid_days"].shape[0]) - 1, 0)


def _should_terminal_flatten(runtime: Dict[str, Any], current_time_index: int, last_time_index: int) -> bool:
    return bool(runtime.get("accounting", {}).get("force_flat_at_end", False)) and current_time_index >= last_time_index - 1


def _zero_futures_flows(state: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    batch_size = int(state["done"].shape[0])
    device = state["done"].device
    state["realized_pnl"] = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    state["variation_margin"] = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)


def _flatten_cash_inventory(
    positions: Dict[str, torch.Tensor],
    cash_accounts: Dict[str, torch.Tensor],
    terminal_tradable_values: Dict[str, torch.Tensor],
    runtime: Dict[str, Any],
) -> None:
    for instrument_name in _runtime_names(runtime, "hedges"):
        name = str(instrument_name)
        trade_delta = -positions[name]
        transaction_cost = _transaction_costs(trade_delta, terminal_tradable_values[name], runtime, name)
        _apply_cash_trade(
            cash_accounts,
            _cash_account_for_instrument(name, runtime),
            trade_delta,
            terminal_tradable_values[name],
            transaction_cost,
        )
        positions[name] = positions[name] + trade_delta


def _flatten_futures_inventory(
    positions: Dict[str, torch.Tensor],
    cash_accounts: Dict[str, torch.Tensor],
    margin_accounts: Dict[str, torch.Tensor],
    settlement_prices: Dict[str, torch.Tensor],
    runtime: Dict[str, Any],
) -> None:
    for instrument_name in _runtime_names(runtime, "hedges"):
        name = str(instrument_name)
        trade_delta = -positions[name]
        transaction_cost = _transaction_costs(trade_delta, settlement_prices[name], runtime, name)
        cash_account_name = _cash_account_for_instrument(name, runtime)
        _apply_account_flow(cash_accounts, cash_account_name, -transaction_cost)
        _apply_account_flow(margin_accounts, cash_account_name, -transaction_cost)
        positions[name] = positions[name] + trade_delta


def build_shared_state(bundle: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    batch_size = _batch_size_from_bundle(bundle)
    time_grid_days = bundle["time_grid_days"]
    device = time_grid_days.device
    time_index = 0
    positions = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    cash_accounts = _zeros_by_name(_runtime_names(runtime, "cash_accounts"), batch_size, device=device)
    settlement_prices = _current_tradable_values(bundle, runtime, time_index)
    state = {
        "done": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "positions": positions,
        "cash_accounts": cash_accounts,
        "margin_accounts": _zeros_by_name(_runtime_names(runtime, "cash_accounts"), batch_size, device=device),
        "realized_pnl": _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device),
        "variation_margin": _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device),
        "cumulative_liability_value": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "settlement_prices": {
            name: settlement_prices[name].clone()
            for name in _runtime_names(runtime, "hedges")
            if name in settlement_prices
        },
    }
    refreshed_state = _refresh_state_views(state, bundle, runtime, time_index)
    refreshed_state["cumulative_liability_value"] = refreshed_state["cumulative_liability_value"] + refreshed_state["liability_cashflow_value"]
    return refreshed_state


def _clone_runtime_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "done": state["done"].clone(),
        "positions": _clone_tensor_dict(state["positions"]),
        "cash_accounts": _clone_tensor_dict(state["cash_accounts"]),
        "margin_accounts": _clone_tensor_dict(state["margin_accounts"]),
        "realized_pnl": _clone_tensor_dict(state["realized_pnl"]),
        "variation_margin": _clone_tensor_dict(state["variation_margin"]),
        "cumulative_liability_value": state["cumulative_liability_value"].clone(),
        "settlement_prices": _clone_tensor_dict(state["settlement_prices"]),
        "time_index": int(state["time_index"]),
        "policy_features": state["policy_features"].clone(),
        "liability_cashflow_value": state["liability_cashflow_value"].clone(),
        "liability_mtm_value": state["liability_mtm_value"].clone(),
        "tradable_values": _clone_tensor_dict(state["tradable_values"]),
    }


def cash_account_step(
    state: Dict[str, Any],
    action: Optional[Dict[str, Any]],
    bundle: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    if bool(state["done"].all().item()):
        return state
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    current_time_index = int(state["time_index"])
    last_time_index = _last_time_index(bundle)
    next_time_index = min(current_time_index + 1, last_time_index)
    current_tradable_values = state["tradable_values"]
    next_positions = _clone_tensor_dict(state["positions"])
    next_cash_accounts = _clone_tensor_dict(state["cash_accounts"])
    resolved_trade_deltas = _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device)
    resolved_trade_deltas = _enforce_position_limits(state["positions"], resolved_trade_deltas, runtime)

    for instrument_name in _runtime_names(runtime, "action_instruments"):
        name = str(instrument_name)
        trade_delta = resolved_trade_deltas.get(name)
        price = current_tradable_values[name]
        transaction_cost = _transaction_costs(trade_delta, price, runtime, name)
        cash_account_name = _cash_account_for_instrument(name, runtime)
        _apply_cash_trade(next_cash_accounts, cash_account_name, trade_delta, price, transaction_cost)
        next_positions[name] = next_positions[name] + trade_delta

    if _should_terminal_flatten(runtime, current_time_index, last_time_index):
        _flatten_cash_inventory(next_positions, next_cash_accounts, _current_tradable_values(bundle, runtime, last_time_index), runtime)

    next_state = {
        "done": state["done"],
        "positions": next_positions,
        "cash_accounts": next_cash_accounts,
        "margin_accounts": _clone_tensor_dict(state["margin_accounts"]),
        "realized_pnl": _clone_tensor_dict(state["realized_pnl"]),
        "variation_margin": _clone_tensor_dict(state["variation_margin"]),
        "cumulative_liability_value": state["cumulative_liability_value"].clone(),
        "settlement_prices": _clone_tensor_dict(state["settlement_prices"]),
    }
    _zero_futures_flows(next_state, runtime)
    next_state["done"] = torch.full_like(state["done"], next_time_index >= last_time_index, dtype=torch.bool)
    refreshed_state = _refresh_state_views(next_state, bundle, runtime, next_time_index)
    refreshed_state["cumulative_liability_value"] = refreshed_state["cumulative_liability_value"] + refreshed_state["liability_cashflow_value"]
    return refreshed_state


def futures_account_step(
    state: Dict[str, Any],
    action: Optional[Dict[str, Any]],
    bundle: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    if bool(state["done"].all().item()):
        return state
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    current_time_index = int(state["time_index"])
    last_time_index = _last_time_index(bundle)
    settlement_time_index = min(current_time_index + 1, last_time_index)
    next_positions = _clone_tensor_dict(state["positions"])
    next_cash_accounts = _clone_tensor_dict(state["cash_accounts"])
    next_margin_accounts = _clone_tensor_dict(state["margin_accounts"])
    next_settlement_prices = _clone_tensor_dict(state["settlement_prices"])
    realized_pnl = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    variation_margin = _zeros_by_name(_runtime_names(runtime, "hedges"), batch_size, device=device)
    resolved_trade_deltas = _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device)
    resolved_trade_deltas = _enforce_position_limits(state["positions"], resolved_trade_deltas, runtime)
    next_tradable_values = _current_tradable_values(bundle, runtime, settlement_time_index)

    for instrument_name in _runtime_names(runtime, "hedges"):
        name = str(instrument_name)
        vm = state["positions"][name] * (next_tradable_values[name] - state["settlement_prices"][name]) * float(
            _runtime_tradable(runtime, name).get("contract_size", 1.0)
        )
        realized_pnl[name] = vm
        variation_margin[name] = vm
        next_settlement_prices[name] = next_tradable_values[name].clone()
        cash_account_name = _cash_account_for_instrument(name, runtime)
        _apply_account_flow(next_margin_accounts, cash_account_name, vm)
        _apply_account_flow(next_cash_accounts, cash_account_name, vm)

    for instrument_name in _runtime_names(runtime, "action_instruments"):
        name = str(instrument_name)
        trade_delta = resolved_trade_deltas.get(name)
        transaction_cost = _transaction_costs(trade_delta, next_tradable_values[name], runtime, name)
        cash_account_name = _cash_account_for_instrument(name, runtime)
        next_positions[name] = next_positions[name] + trade_delta
        _apply_account_flow(next_cash_accounts, cash_account_name, -transaction_cost)
        _apply_account_flow(next_margin_accounts, cash_account_name, -transaction_cost)

    if _should_terminal_flatten(runtime, current_time_index, last_time_index):
        _flatten_futures_inventory(next_positions, next_cash_accounts, next_margin_accounts, next_settlement_prices, runtime)

    next_state = {
        "done": torch.full_like(state["done"], settlement_time_index >= last_time_index, dtype=torch.bool),
        "positions": next_positions,
        "cash_accounts": next_cash_accounts,
        "settlement_prices": next_settlement_prices,
        "margin_accounts": next_margin_accounts,
        "realized_pnl": realized_pnl,
        "variation_margin": variation_margin,
        "cumulative_liability_value": state["cumulative_liability_value"].clone(),
    }
    refreshed_state = _refresh_state_views(next_state, bundle, runtime, settlement_time_index)
    refreshed_state["cumulative_liability_value"] = refreshed_state["cumulative_liability_value"] + refreshed_state["liability_cashflow_value"]
    return refreshed_state


def step_runtime_state(
    state: Dict[str, Any],
    action: Optional[Dict[str, Any]],
    bundle: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    accounting_mode = str(runtime.get("accounting_mode", "futures"))
    if accounting_mode == "cash_account":
        return cash_account_step(state, action, bundle, runtime)
    return futures_account_step(state, action, bundle, runtime)


def reward_and_terminal_payoff(
    previous_state: Dict[str, Any],
    state: Dict[str, Any],
    bundle: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    batch_size = _batch_size_from_bundle(bundle)
    device = bundle["time_grid_days"].device
    reward = _dense_tracking_reward(previous_state, state, runtime).to(device=device, dtype=torch.float32)
    terminal_payoff = torch.zeros(batch_size, dtype=torch.float32, device=device)
    portfolio_value = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liability_value = state.get("cumulative_liability_value", torch.zeros(batch_size, dtype=torch.float32, device=device)).to(device=device, dtype=torch.float32)
    done_mask = state["done"].to(dtype=torch.bool)
    if bool(done_mask.any().item()):
        portfolio_value = _terminal_portfolio_value(state, runtime).to(device=device, dtype=torch.float32)
        terminal_reward = _evaluate_objective(portfolio_value, liability_value, runtime).to(device=device, dtype=torch.float32)
        reward = reward + torch.where(done_mask, terminal_reward, torch.zeros_like(reward))
        terminal_payoff = torch.where(done_mask, liability_value, terminal_payoff)
    return {
        "reward": reward,
        "terminal_payoff": terminal_payoff,
        "portfolio_value": portfolio_value,
        "liability_value": liability_value,
    }


def _make_replay_buffer(capacity: int) -> Any:
    if int(capacity) <= 0:
        raise ValueError("Replay buffer capacity must be positive")
    single_capacity = max(int(capacity) // 2, 1)
    return {
        "active": ReplayBuffer(storage=LazyTensorStorage(max_size=single_capacity)),
        "inactive": ReplayBuffer(storage=LazyTensorStorage(max_size=single_capacity)),
    }


def _replay_buffer_total_size(replay_buffer: Dict[str, ReplayBuffer]) -> int:
    return int(len(replay_buffer["active"])) + int(len(replay_buffer["inactive"]))


def _extend_replay_buffer(replay_buffer: Dict[str, ReplayBuffer], replay_batch: TensorDict) -> None:
    if int(replay_batch.batch_size[0]) == 0:
        return
    gate_feature = replay_batch.get("action_features")[:, 0] > 0.5
    for pool_name, mask in (("active", gate_feature), ("inactive", ~gate_feature)):
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if int(indices.numel()) == 0:
            continue
        replay_buffer[pool_name].extend(replay_batch[indices])


def _sample_replay_batch(
    replay_buffer: Dict[str, ReplayBuffer],
    batch_size: int,
    device: torch.device,
) -> Optional[TensorDict]:
    total_size = _replay_buffer_total_size(replay_buffer)
    if total_size < int(batch_size):
        return None
    active_size = int(len(replay_buffer["active"]))
    inactive_size = int(len(replay_buffer["inactive"]))
    active_target = min(active_size, max(int(batch_size) // 2, 1))
    inactive_target = min(inactive_size, int(batch_size) - active_target)
    remaining = int(batch_size) - active_target - inactive_target
    if remaining > 0:
        extra_active = min(remaining, active_size - active_target)
        active_target += extra_active
        remaining -= extra_active
    if remaining > 0:
        inactive_target += min(remaining, inactive_size - inactive_target)
    samples = []
    if active_target > 0:
        samples.append(replay_buffer["active"].sample(active_target).to(device))
    if inactive_target > 0:
        samples.append(replay_buffer["inactive"].sample(inactive_target).to(device))
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    return TensorDict(
        {
            key: torch.cat([sample.get(key) for sample in samples], dim=0)
            for key in samples[0].keys()
        },
        batch_size=[sum(int(sample.batch_size[0]) for sample in samples)],
    )


class StructuredValueOperator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_features: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        if state_features.ndim == 1:
            state_features = state_features.unsqueeze(0)
        if action_features.ndim == 1:
            action_features = action_features.unsqueeze(0)
        return self.net(torch.cat([state_features, action_features], dim=-1)).squeeze(-1)


def _optimizer_settings(runtime: Dict[str, Any]) -> Dict[str, Any]:
    optimizer_params = runtime.get("optimizer", {})
    return {
        "epochs": int(optimizer_params.get("epochs", OPTIMIZER_DEFAULTS["epochs"])),
        "replay_capacity": int(optimizer_params.get("replay_capacity", OPTIMIZER_DEFAULTS["replay_capacity"])),
        "batch_size": int(optimizer_params.get("batch_size", OPTIMIZER_DEFAULTS["batch_size"])),
        "gamma": float(optimizer_params.get("gamma", OPTIMIZER_DEFAULTS["gamma"])),
        "learning_rate": float(optimizer_params.get("learning_rate", OPTIMIZER_DEFAULTS["learning_rate"])),
        "epsilon_start": float(optimizer_params.get("epsilon_start", OPTIMIZER_DEFAULTS["epsilon_start"])),
        "epsilon_end": float(optimizer_params.get("epsilon_end", OPTIMIZER_DEFAULTS["epsilon_end"])),
        "epsilon_decay_epochs": int(optimizer_params.get("epsilon_decay_epochs", OPTIMIZER_DEFAULTS["epsilon_decay_epochs"])),
        "target_update_interval": int(optimizer_params.get("target_update_interval", OPTIMIZER_DEFAULTS["target_update_interval"])),
        "gradient_steps_per_epoch": int(optimizer_params.get("gradient_steps_per_epoch", OPTIMIZER_DEFAULTS["gradient_steps_per_epoch"])),
        "dense_tracking_reward_scale": float(optimizer_params.get("dense_tracking_reward_scale", OPTIMIZER_DEFAULTS["dense_tracking_reward_scale"])),
        "dense_tracking_reward_clip": float(optimizer_params.get("dense_tracking_reward_clip", OPTIMIZER_DEFAULTS["dense_tracking_reward_clip"])),
        "validation_fraction": float(optimizer_params.get("validation_fraction", OPTIMIZER_DEFAULTS["validation_fraction"])),
        "validation_min_batch": int(optimizer_params.get("validation_min_batch", OPTIMIZER_DEFAULTS["validation_min_batch"])),
        "validation_shards": int(optimizer_params.get("validation_shards", OPTIMIZER_DEFAULTS["validation_shards"])),
        "top_validation_checkpoints": int(optimizer_params.get("top_validation_checkpoints", OPTIMIZER_DEFAULTS["top_validation_checkpoints"])),
        "performance_gated_curriculum": bool(optimizer_params.get("performance_gated_curriculum", OPTIMIZER_DEFAULTS["performance_gated_curriculum"])),
        "curriculum_advance_patience": int(optimizer_params.get("curriculum_advance_patience", OPTIMIZER_DEFAULTS["curriculum_advance_patience"])),
        "curriculum_min_improvement": float(optimizer_params.get("curriculum_min_improvement", OPTIMIZER_DEFAULTS["curriculum_min_improvement"])),
        "curriculum_min_trade_rate": float(optimizer_params.get("curriculum_min_trade_rate", OPTIMIZER_DEFAULTS["curriculum_min_trade_rate"])),
        "curriculum_max_trade_rate": float(optimizer_params.get("curriculum_max_trade_rate", OPTIMIZER_DEFAULTS["curriculum_max_trade_rate"])),
        "turnover_penalty_scale": float(optimizer_params.get("turnover_penalty_scale", OPTIMIZER_DEFAULTS["turnover_penalty_scale"])),
        "no_trade_reference_scale": float(optimizer_params.get("no_trade_reference_scale", OPTIMIZER_DEFAULTS["no_trade_reference_scale"])),
        "replay_imitation_weight": float(optimizer_params.get("replay_imitation_weight", OPTIMIZER_DEFAULTS["replay_imitation_weight"])),
        "action_sparsity_penalty_weight": float(optimizer_params.get("action_sparsity_penalty_weight", OPTIMIZER_DEFAULTS["action_sparsity_penalty_weight"])),
        "decision_interval_curriculum": tuple(
            dict(stage) for stage in optimizer_params.get("decision_interval_curriculum", OPTIMIZER_DEFAULTS["decision_interval_curriculum"])
        ),
        "seed": optimizer_params.get("seed"),
    }


def _epsilon_for_epoch(settings: Dict[str, Any], epoch: int) -> float:
    decay_epochs = max(int(settings["epsilon_decay_epochs"]), 1)
    if epoch >= decay_epochs:
        return float(settings["epsilon_end"])
    progress = float(epoch) / float(decay_epochs)
    return float(settings["epsilon_start"] + progress * (settings["epsilon_end"] - settings["epsilon_start"]))


def _bundle_scenario_dates(bundle: Dict[str, Any]) -> pd.DatetimeIndex:
    base_date = pd.Timestamp(bundle.get("meta", {}).get("base_date"))
    day_offsets = torch.as_tensor(bundle["time_grid_days"]).detach().cpu().to(dtype=torch.int64).tolist()
    return pd.DatetimeIndex([base_date + pd.Timedelta(days=int(day_offset)) for day_offset in day_offsets])


def _is_business_date(date: pd.Timestamp, bundle: Dict[str, Any]) -> bool:
    business_day = bundle.get("meta", {}).get("business_day")
    if business_day is not None and hasattr(business_day, "is_on_offset"):
        return bool(business_day.is_on_offset(pd.Timestamp(date)))
    return pd.Timestamp(date).weekday() < 5


def _decision_interval_business_days_for_epoch(settings: Dict[str, Any], epoch: Optional[int], *, evaluation: bool) -> int:
    curriculum = tuple(dict(stage) for stage in settings.get("decision_interval_curriculum", ()))
    if not curriculum:
        return 1
    if evaluation or epoch is None:
        return int(curriculum[-1]["interval_business_days"])
    epoch_number = int(epoch) + 1
    for stage in curriculum:
        start_epoch = int(stage.get("start_epoch", 1))
        end_epoch = stage.get("end_epoch")
        if epoch_number < start_epoch:
            continue
        if end_epoch is None or epoch_number <= int(end_epoch):
            return int(stage["interval_business_days"])
    return int(curriculum[-1]["interval_business_days"])


def _decision_time_indices(
    bundle: Dict[str, Any],
    settings: Dict[str, Any],
    *,
    epoch: Optional[int],
    evaluation: bool,
) -> Tuple[int, ...]:
    scenario_dates = _bundle_scenario_dates(bundle)
    if scenario_dates.empty:
        return tuple()
    last_time_index = max(len(scenario_dates) - 1, 0)
    business_indices = [
        index
        for index, date in enumerate(scenario_dates[:last_time_index])
        if _is_business_date(date, bundle)
    ]
    if not business_indices:
        return tuple()
    interval_business_days = max(_decision_interval_business_days_for_epoch(settings, epoch, evaluation=evaluation), 1)
    selected_indices = [
        business_index
        for selection_index, business_index in enumerate(business_indices)
        if selection_index % interval_business_days == 0
    ]
    return tuple(selected_indices)


def _decision_time_indices_for_interval(bundle: Dict[str, Any], interval_business_days: int) -> Tuple[int, ...]:
    scenario_dates = _bundle_scenario_dates(bundle)
    if scenario_dates.empty:
        return tuple()
    last_time_index = max(len(scenario_dates) - 1, 0)
    business_indices = [
        index
        for index, date in enumerate(scenario_dates[:last_time_index])
        if _is_business_date(date, bundle)
    ]
    if not business_indices:
        return tuple()
    normalized_interval = max(int(interval_business_days), 1)
    return tuple(
        business_index
        for selection_index, business_index in enumerate(business_indices)
        if selection_index % normalized_interval == 0
    )


def _empty_replay_batch(policy: StructuredRebalancePolicy, state_feature_dim: int) -> TensorDict:
    action_feature_dim = int(policy.action_space.dimension)
    return TensorDict(
        {
            "state_features": torch.zeros((0, state_feature_dim), dtype=torch.float32, device=policy.device),
            "action_features": torch.zeros((0, action_feature_dim), dtype=torch.float32, device=policy.device),
            "rewards": torch.zeros((0,), dtype=torch.float32, device=policy.device),
            "baseline_rewards": torch.zeros((0,), dtype=torch.float32, device=policy.device),
            "next_state_features": torch.zeros((0, state_feature_dim), dtype=torch.float32, device=policy.device),
            "dones": torch.zeros((0,), dtype=torch.float32, device=policy.device),
        },
        batch_size=[0],
    )


def _make_structured_policy(bundle: Dict[str, Any], runtime: Dict[str, Any], *, device: torch.device) -> StructuredRebalancePolicy:
    policy_config = runtime.get("policy", {})
    model_config = policy_config.get("model", {})
    feature_extractor = TorchRLStateFeatureExtractor(
        feature_dim=int(runtime.get("state_layout", {}).get("dimension", 0)) + _hedge_profile_feature_dim(bundle),
    )
    action_space = policy_config.get("action_space", {})
    instrument_order = tuple(name for name in action_space.get("instrument_order", ()))
    min_trade_delta = tuple(action_space.get("min_trade_delta", {}).get(name, 0) for name in instrument_order)
    max_trade_delta = tuple(action_space.get("max_trade_delta", {}).get(name, 0) for name in instrument_order)
    return StructuredRebalancePolicy(
        action_space=StructuredActionSpace(instrument_order, min_trade_delta, max_trade_delta),
        feature_extractor=feature_extractor,
        hidden_layers=tuple(value for value in model_config.get("hidden_layers", (64, 64))),
        activation=model_config.get("activation", "ReLU"),
        device=device,
    )


def _initialize_policy_model(policy: StructuredRebalancePolicy, bundle: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    sample_state = build_shared_state(bundle, runtime)
    _ = policy(sample_state["policy_features"][:1].to(policy.device))


def _collect_structured_rollout(
    policy: StructuredRebalancePolicy,
    bundle: Dict[str, Any],
    runtime: Dict[str, Any],
    *,
    greedy: bool,
    epsilon: float,
    decision_time_indices: Optional[Tuple[int, ...]] = None,
    decision_interval_business_days: int = 1,
) -> Dict[str, Any]:
    state = build_shared_state(bundle, runtime)
    batch_size = int(state["policy_features"].shape[0])
    state_feature_dim = int(state["policy_features"].shape[1])
    transition_rows = []
    action_counts = torch.zeros(batch_size, dtype=torch.float32, device=policy.device)
    no_trade_steps = 0
    total_action_steps = 0
    terminal_transition = None
    trade_abs_sum = 0.0
    trade_elem_count = 0
    trade_abs_max = 0.0
    rebalance_abs_sum = 0.0
    rebalance_elem_count = 0
    rebalance_abs_max = 0.0
    initial_feature_mean_abs = float(state["policy_features"].detach().abs().mean().item()) if state["policy_features"].numel() else 0.0
    decision_time_index_set = set(int(index) for index in (decision_time_indices or ()))
    decision_steps = 0
    rollout_steps = 0
    pending_transition = None
    pending_baseline_state = None

    while True:
        current_time_index = int(state["time_index"])
        should_decide = current_time_index in decision_time_index_set
        mapped = None
        if should_decide:
            decision_steps += 1
            state_features = state["policy_features"].to(policy.device)
            output = policy.sample(state_features, epsilon=epsilon, greedy=greedy)
            mapped = _realized_structured_action(
                policy.map_actions(output),
                state["positions"],
                runtime,
                batch_size=batch_size,
                device=policy.device,
            )
            if mapped is None:
                raise RuntimeError("Structured policy decision unexpectedly produced no action")
            ordered_trade_deltas = torch.as_tensor(mapped["ordered_trade_deltas"], dtype=torch.float32, device=policy.device)
            trade_abs = ordered_trade_deltas.abs()
            trade_abs_sum += float(trade_abs.sum().item())
            trade_elem_count += int(trade_abs.numel())
            trade_abs_max = max(trade_abs_max, float(trade_abs.max().item()) if trade_abs.numel() else 0.0)
            rebalance_abs = torch.as_tensor(output["rebalance_vector"], dtype=torch.float32, device=policy.device).abs()
            rebalance_abs_sum += float(rebalance_abs.sum().item())
            rebalance_elem_count += int(rebalance_abs.numel())
            rebalance_abs_max = max(rebalance_abs_max, float(rebalance_abs.max().item()) if rebalance_abs.numel() else 0.0)
            trade_mask = mapped["ordered_trade_deltas"].abs().sum(dim=1) > 0
            action_counts = action_counts + trade_mask.to(dtype=torch.float32)
            no_trade_steps += int((~trade_mask).sum().item())
            total_action_steps += int(trade_mask.numel())
            pending_transition = {
                "state_features": state_features.detach(),
                "action_features": policy.action_features(mapped).detach(),
                "rewards": -_turnover_penalty(ordered_trade_deltas, runtime).to(dtype=torch.float32, device=policy.device),
                "baseline_rewards": torch.zeros(batch_size, dtype=torch.float32, device=policy.device),
            }
            pending_baseline_state = _clone_runtime_state(state)
        next_state = step_runtime_state(state, mapped, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        if pending_transition is not None:
            pending_transition["rewards"] = pending_transition["rewards"] + transition["reward"].to(
                policy.device, dtype=torch.float32
            ).detach()
            baseline_state = pending_baseline_state
            if baseline_state is None:
                raise RuntimeError("Pending baseline state missing for TorchRL rollout transition")
            next_baseline_state = step_runtime_state(baseline_state, None, bundle, runtime)
            baseline_transition = reward_and_terminal_payoff(baseline_state, next_baseline_state, bundle, runtime)
            pending_transition["baseline_rewards"] = pending_transition["baseline_rewards"] + baseline_transition["reward"].to(
                policy.device, dtype=torch.float32
            ).detach()
            pending_baseline_state = next_baseline_state
        terminal_transition = {"state": next_state, **transition}
        state = next_state
        rollout_steps += 1
        next_time_index = int(state["time_index"])
        if pending_transition is not None and (
            bool(state["done"].all().item()) or next_time_index in decision_time_index_set
        ):
            transition_rows.append(
                TensorDict(
                    {
                        "state_features": pending_transition["state_features"],
                        "action_features": pending_transition["action_features"],
                        "rewards": pending_transition["rewards"],
                        "baseline_rewards": pending_transition["baseline_rewards"],
                        "next_state_features": state["policy_features"].to(policy.device).detach(),
                        "dones": state["done"].to(policy.device, dtype=torch.float32).detach(),
                    },
                    batch_size=[batch_size],
                )
            )
            pending_transition = None
            pending_baseline_state = None
        if bool(next_state["done"].all().item()):
            break

    if transition_rows:
        replay_batch = TensorDict(
            {
                key: torch.cat([row.get(key) for row in transition_rows], dim=0)
                for key in transition_rows[0].keys()
            },
            batch_size=[sum(int(row.batch_size[0]) for row in transition_rows)],
        )
    else:
        replay_batch = _empty_replay_batch(policy, state_feature_dim)
    final_state = terminal_transition["state"] if terminal_transition is not None else state
    final_policy_features = final_state["policy_features"].detach().to(dtype=torch.float32)
    final_inventory_fraction = final_state.get("feature_groups", {}).get("inventory", {}).get("inventory_fraction")
    if final_inventory_fraction is None:
        final_inventory_fraction_mean_abs = 0.0
        final_inventory_fraction_max_abs = 0.0
    else:
        final_inventory_fraction = torch.as_tensor(final_inventory_fraction, dtype=torch.float32)
        final_inventory_fraction_mean_abs = float(final_inventory_fraction.abs().mean().item()) if final_inventory_fraction.numel() else 0.0
        final_inventory_fraction_max_abs = float(final_inventory_fraction.abs().max().item()) if final_inventory_fraction.numel() else 0.0
    return {
        "replay_batch": replay_batch,
        "terminal_transition": terminal_transition,
        "action_counts": action_counts.detach(),
        "no_trade_rate": 0.0 if total_action_steps == 0 else float(no_trade_steps / total_action_steps),
        "num_steps": rollout_steps,
        "rollout_diagnostics": {
            "num_steps": float(rollout_steps),
            "num_decision_steps": float(decision_steps),
            "decision_interval_business_days": float(max(int(decision_interval_business_days), 1)),
            "initial_feature_mean_abs": float(initial_feature_mean_abs),
            "final_feature_mean_abs": float(final_policy_features.abs().mean().item()) if final_policy_features.numel() else 0.0,
            "nonzero_trade_rate": 0.0 if total_action_steps == 0 else float(1.0 - (no_trade_steps / total_action_steps)),
            "mean_abs_trade_delta": 0.0 if trade_elem_count == 0 else float(trade_abs_sum / trade_elem_count),
            "max_abs_trade_delta": float(trade_abs_max),
            "mean_abs_rebalance_signal": 0.0 if rebalance_elem_count == 0 else float(rebalance_abs_sum / rebalance_elem_count),
            "max_abs_rebalance_signal": float(rebalance_abs_max),
            "final_inventory_fraction_mean_abs": float(final_inventory_fraction_mean_abs),
            "final_inventory_fraction_max_abs": float(final_inventory_fraction_max_abs),
        },
    }


def _collect_no_trade_rollout(
    bundle: Dict[str, Any],
    runtime: Dict[str, Any],
    *,
    decision_time_indices: Optional[Tuple[int, ...]] = None,
    decision_interval_business_days: int = 1,
) -> Dict[str, Any]:
    state = build_shared_state(bundle, runtime)
    batch_size = int(state["policy_features"].shape[0])
    initial_feature_mean_abs = float(state["policy_features"].detach().abs().mean().item()) if state["policy_features"].numel() else 0.0
    decision_time_index_set = set(int(index) for index in (decision_time_indices or ()))
    decision_steps = 0
    rollout_steps = 0
    terminal_transition = None

    while True:
        current_time_index = int(state["time_index"])
        if current_time_index in decision_time_index_set:
            decision_steps += 1
        next_state = step_runtime_state(state, None, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        terminal_transition = {"state": next_state, **transition}
        state = next_state
        rollout_steps += 1
        if bool(next_state["done"].all().item()):
            break

    return {
        "terminal_transition": terminal_transition,
        "action_counts": torch.zeros(batch_size, dtype=torch.float32, device=state["policy_features"].device),
        "no_trade_rate": 1.0,
        "rollout_diagnostics": {
            "num_steps": float(rollout_steps),
            "num_decision_steps": float(decision_steps),
            "decision_interval_business_days": float(decision_interval_business_days),
            "initial_feature_mean_abs": initial_feature_mean_abs,
            "final_feature_mean_abs": float(state["policy_features"].detach().abs().mean().item()) if state["policy_features"].numel() else 0.0,
            "nonzero_trade_rate": 0.0,
            "mean_abs_trade_delta": 0.0,
            "max_abs_trade_delta": 0.0,
            "mean_abs_rebalance_signal": 0.0,
            "max_abs_rebalance_signal": 0.0,
            "final_inventory_fraction_mean_abs": 0.0,
            "final_inventory_fraction_max_abs": 0.0,
        },
    }


def _update_structured_policy(
    policy: StructuredRebalancePolicy,
    critic: StructuredValueOperator,
    target_critic: StructuredValueOperator,
    critic_optimizer: optim.Optimizer,
    actor_optimizer: optim.Optimizer,
    replay_buffer: Dict[str, ReplayBuffer],
    settings: Dict[str, Any],
) -> Dict[str, list]:
    if _replay_buffer_total_size(replay_buffer) < int(settings["batch_size"]):
        return {"critic_losses": [], "actor_losses": []}
    critic_losses = []
    actor_losses = []
    rebalance_l2_penalty_weight = 5.0e-3
    replay_imitation_weight = float(settings.get("replay_imitation_weight", OPTIMIZER_DEFAULTS["replay_imitation_weight"]))
    action_sparsity_penalty_weight = float(settings.get("action_sparsity_penalty_weight", OPTIMIZER_DEFAULTS["action_sparsity_penalty_weight"]))
    target_critic.eval()
    for _ in range(int(settings["gradient_steps_per_epoch"])):
        sample = _sample_replay_batch(replay_buffer, int(settings["batch_size"]), policy.device)
        if sample is None:
            break
        states = sample.get("state_features").to(dtype=torch.float32)
        actions = sample.get("action_features").to(dtype=torch.float32).reshape(states.shape[0], -1)
        no_trade_actions = torch.zeros_like(actions)
        rewards = sample.get("rewards").to(dtype=torch.float32)
        baseline_rewards = sample.get("baseline_rewards").to(dtype=torch.float32)
        advantage_rewards = rewards - baseline_rewards
        next_states = sample.get("next_state_features").to(dtype=torch.float32)
        dones = sample.get("dones").to(dtype=torch.float32)

        q_values = critic(states, actions)
        with torch.no_grad():
            next_output = policy(next_states)
            next_actions = policy.action_features(next_output)
            next_q_values = target_critic(next_states, next_actions)
            target_values = advantage_rewards + float(settings["gamma"]) * (1.0 - dones) * next_q_values

        critic_loss = nn.functional.mse_loss(q_values, target_values)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        critic_losses.append(float(critic_loss.item()))

        actor_optimizer.zero_grad()
        for param in critic.parameters():
            param.requires_grad_(False)
        policy_output = policy(states)
        policy_actions = policy.action_features(policy_output)
        rebalance_penalty = torch.as_tensor(policy_output.get("rebalance_vector"), dtype=torch.float32).pow(2).mean()
        action_sparsity_penalty = policy_actions.abs().mean()
        sampled_action_values = critic(states, actions).detach()
        policy_action_values = critic(states, policy_actions).detach()
        no_trade_action_values = critic(states, no_trade_actions).detach()
        no_trade_reference_scale = float(settings.get("no_trade_reference_scale", OPTIMIZER_DEFAULTS["no_trade_reference_scale"]))
        scaled_no_trade_action_values = no_trade_reference_scale * no_trade_action_values
        imitation_baseline = torch.maximum(policy_action_values, scaled_no_trade_action_values)
        imitation_advantage = torch.relu(sampled_action_values - imitation_baseline)
        imitation_weight = (imitation_advantage / (imitation_advantage.mean() + 1.0e-6)).reshape(-1, 1)
        rebalance_targets = actions
        rebalance_imitation_loss = (
            (policy_actions - rebalance_targets).pow(2)
            * imitation_weight
        ).mean()
        policy_advantage = critic(states, policy_actions) - scaled_no_trade_action_values
        actor_loss = (
            -policy_advantage.mean()
            + rebalance_l2_penalty_weight * rebalance_penalty
            + action_sparsity_penalty_weight * action_sparsity_penalty
            + replay_imitation_weight * rebalance_imitation_loss
        )
        actor_loss.backward()
        actor_optimizer.step()
        for param in critic.parameters():
            param.requires_grad_(True)
        actor_losses.append(float(actor_loss.item()))
    return {"critic_losses": critic_losses, "actor_losses": actor_losses}


def _cpu_tensor_dict(values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        str(name): torch.as_tensor(tensor).detach().to(dtype=torch.float32).cpu()
        for name, tensor in values.items()
    }


def _scalar_rollout_diagnostics(rollout_diagnostics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    return {
        str(key): float(value)
        for key, value in dict(rollout_diagnostics or {}).items()
    }


def _terminal_evaluation_summary(
    state: Dict[str, Any],
    terminal_transition: Dict[str, Any],
    action_counts: torch.Tensor,
    no_trade_rate: float,
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    hedge_pnl = terminal_transition["portfolio_value"].detach().to(dtype=torch.float32)
    liability = terminal_transition["liability_value"].detach().to(dtype=torch.float32)
    net_pnl = hedge_pnl - liability
    hedge_profile = dict(bundle.get("hedge_profile", {}))
    liability_mtm = hedge_profile.get("liability_mtm")
    if liability_mtm is None:
        initial_liability_mtm = torch.zeros_like(net_pnl)
    else:
        initial_liability_mtm = liability_mtm[0].detach().to(dtype=torch.float32, device=net_pnl.device)
    net_pnl_plus_initial_liability_mtm = net_pnl + initial_liability_mtm
    action_counts = action_counts.detach().to(dtype=torch.float32)
    return {
        "metrics": {
            "average_net_pnl": float(net_pnl.mean().item()),
            "average_net_pnl_plus_initial_liability_mtm": float(net_pnl_plus_initial_liability_mtm.mean().item()),
            "median_net_pnl": float(torch.quantile(net_pnl.to(dtype=torch.float64), 0.5).item()),
            "worst_net_pnl": float(net_pnl.min().item()),
            "average_hedge_pnl": float(hedge_pnl.mean().item()),
            "average_liability": float(liability.mean().item()),
            "average_initial_liability_mtm": float(initial_liability_mtm.mean().item()),
            "avg_actions": float(action_counts.mean().item()),
            "final_no_trade_rate": float(no_trade_rate),
        },
        "final_state": {
            "positions": _cpu_tensor_dict(state.get("positions", {})),
            "cash_accounts": _cpu_tensor_dict(state.get("cash_accounts", {})),
            "margin_accounts": _cpu_tensor_dict(state.get("margin_accounts", {})),
            "portfolio_value": hedge_pnl.cpu(),
            "liability_value": liability.cpu(),
            "net_pnl": net_pnl.cpu(),
            "action_counts": action_counts.cpu(),
        },
    }


def build_torchrl_evaluation_output(
    state: Dict[str, Any],
    terminal_transition: Dict[str, Any],
    action_counts: torch.Tensor,
    no_trade_rate: float,
    bundle: Dict[str, Any],
    *,
    trainer_type: str,
    rollout_diagnostics: Optional[Dict[str, Any]] = None,
    no_trade_reference: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    policy_summary = _terminal_evaluation_summary(state, terminal_transition, action_counts, no_trade_rate, bundle)
    num_paths = int(policy_summary["final_state"]["net_pnl"].shape[0])
    evaluation_output = {
        "metrics": policy_summary["metrics"],
        "final_state": policy_summary["final_state"],
        "diagnostics": {
            "num_episodes": num_paths,
            "num_batches": int(bundle.get("meta", {}).get("num_batches", 1)),
            "trainer_type": trainer_type,
            "evaluation_output": "tensor_summary_only",
            "rollout": _scalar_rollout_diagnostics(rollout_diagnostics),
        },
        "timing": dict(timing or {}),
    }
    if no_trade_reference is not None:
        baseline_summary = _terminal_evaluation_summary(
            no_trade_reference["state"],
            no_trade_reference["terminal_transition"],
            no_trade_reference["action_counts"],
            no_trade_reference["no_trade_rate"],
            bundle,
        )
        evaluation_output["reference"] = {
            "no_trade": {
                "metrics": baseline_summary["metrics"],
                "diagnostics": {
                    "rollout": _scalar_rollout_diagnostics(no_trade_reference.get("rollout_diagnostics")),
                },
            },
            "policy_minus_no_trade": {
                metric_name: float(policy_summary["metrics"][metric_name] - baseline_summary["metrics"][metric_name])
                for metric_name in (
                    "average_net_pnl",
                    "average_net_pnl_plus_initial_liability_mtm",
                    "median_net_pnl",
                    "worst_net_pnl",
                    "average_hedge_pnl",
                    "avg_actions",
                    "final_no_trade_rate",
                )
            },
        }
    return evaluation_output


def evaluate_torchrl_policy(bundle: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    settings = _optimizer_settings(runtime)
    device = bundle["time_grid_days"].device
    policy = _make_structured_policy(bundle, runtime, device=device)
    _initialize_policy_model(policy, bundle, runtime)
    evaluation_decision_interval_business_days = _decision_interval_business_days_for_epoch(
        settings,
        None,
        evaluation=True,
    )
    evaluation_decision_time_indices = _decision_time_indices(
        bundle,
        settings,
        epoch=None,
        evaluation=True,
    )
    evaluation_started = time.perf_counter()
    evaluation_rollout = _collect_structured_rollout(
        policy,
        bundle,
        runtime,
        greedy=True,
        epsilon=0.0,
        decision_time_indices=evaluation_decision_time_indices,
        decision_interval_business_days=evaluation_decision_interval_business_days,
    )
    no_trade_rollout = _collect_no_trade_rollout(
        bundle,
        runtime,
        decision_time_indices=evaluation_decision_time_indices,
        decision_interval_business_days=evaluation_decision_interval_business_days,
    )
    evaluation_time_seconds = float(time.perf_counter() - evaluation_started)
    evaluation_output = build_torchrl_evaluation_output(
        evaluation_rollout["terminal_transition"]["state"],
        evaluation_rollout["terminal_transition"],
        evaluation_rollout["action_counts"],
        evaluation_rollout["no_trade_rate"],
        bundle,
        trainer_type="torchrl_tensor_evaluation",
        rollout_diagnostics=evaluation_rollout.get("rollout_diagnostics"),
        no_trade_reference={
            "state": no_trade_rollout["terminal_transition"]["state"],
            "terminal_transition": no_trade_rollout["terminal_transition"],
            "action_counts": no_trade_rollout["action_counts"],
            "no_trade_rate": no_trade_rollout["no_trade_rate"],
            "rollout_diagnostics": no_trade_rollout.get("rollout_diagnostics"),
        },
        timing={
            "evaluation_time_seconds": evaluation_time_seconds,
        },
    )
    return {
        "policy": policy,
        "policy_artifact": policy.to_artifact(),
        "evaluation_output": evaluation_output,
        "optimizer_diagnostics": None,
    }


def train_torchrl_policy(bundle: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    settings = _optimizer_settings(runtime)
    if settings["seed"] is not None:
        seed = int(settings["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    device = bundle["time_grid_days"].device
    policy = _make_structured_policy(bundle, runtime, device=device)
    _initialize_policy_model(policy, bundle, runtime)

    sample_state = build_shared_state(bundle, runtime)["policy_features"][:1].to(policy.device)
    sample_output = policy(sample_state)
    action_features = policy.action_features(sample_output)
    critic = StructuredValueOperator(
        state_dim=int(sample_state.shape[1]),
        action_dim=int(action_features.shape[1]),
    ).to(policy.device)
    target_critic = deepcopy(critic).to(policy.device)
    actor_optimizer = optim.Adam(policy.parameters(), lr=float(settings["learning_rate"]))
    critic_optimizer = optim.Adam(critic.parameters(), lr=float(settings["learning_rate"]))
    replay_buffer = _make_replay_buffer(int(settings["replay_capacity"]))
    total_episode_count = _batch_size_from_bundle(bundle)
    train_bundle = bundle
    validation_bundle = bundle
    validation_episode_count = 0
    validation_shards = (bundle,)
    validation_fraction = min(max(float(settings["validation_fraction"]), 0.0), 0.5)
    validation_min_batch = max(int(settings["validation_min_batch"]), 1)
    if total_episode_count > (2 * validation_min_batch) and validation_fraction > 0.0:
        validation_episode_count = min(
            max(int(total_episode_count * validation_fraction), validation_min_batch),
            total_episode_count - validation_min_batch,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(settings.get("seed") or 0) + 101)
        shuffled_indices = torch.randperm(total_episode_count, generator=generator)
        validation_indices = shuffled_indices[:validation_episode_count]
        train_indices = shuffled_indices[validation_episode_count:]
        train_bundle = _slice_bundle_episodes(bundle, train_indices.to(dtype=torch.int64))
        validation_bundle = _slice_bundle_episodes(bundle, validation_indices.to(dtype=torch.int64))
        shard_count = max(int(settings["validation_shards"]), 1)
        validation_shards = tuple(
            _slice_bundle_episodes(validation_bundle, shard_indices)
            for shard_indices in _split_episode_indices(validation_episode_count, shard_count)
        )
    else:
        shard_count = max(int(settings["validation_shards"]), 1)
        validation_shards = tuple(
            _slice_bundle_episodes(validation_bundle, shard_indices)
            for shard_indices in _split_episode_indices(_batch_size_from_bundle(validation_bundle), shard_count)
        ) or (validation_bundle,)
    training_episode_count = _batch_size_from_bundle(train_bundle)
    greedy_validation_time_indices = tuple(
        _decision_time_indices(
            validation_shard,
            settings,
            epoch=None,
            evaluation=True,
        )
        for validation_shard in validation_shards
    )
    stage_validation_time_indices_cache: Dict[int, Tuple[Tuple[int, ...], ...]] = {}

    epoch_rewards = []
    epoch_critic_losses = []
    epoch_actor_losses = []
    epoch_average_net_pnls = []
    epoch_replay_sizes = []
    epoch_no_trade_rates = []
    epoch_mean_abs_trade_deltas = []
    epoch_mean_abs_rebalance_signals = []
    epoch_rollout_times_seconds = []
    epoch_replay_storage_times_seconds = []
    epoch_gradient_update_times_seconds = []
    epoch_greedy_eval_times_seconds = []
    epoch_decision_interval_business_days = []
    epoch_greedy_selection_scores = []
    epoch_greedy_average_rewards = []
    epoch_greedy_average_net_pnls = []
    epoch_greedy_nonzero_trade_rates = []
    epoch_greedy_mean_abs_trade_deltas = []
    epoch_stage_validation_net_pnls = []
    epoch_stage_validation_trade_rates = []
    epoch_stage_validation_shard_success_rates = []
    epoch_curriculum_stage_indices = []
    fit_started = time.perf_counter()
    best_epoch = -1
    best_checkpoint_metrics = {
        "selection_score": float("-inf"),
        "average_reward": float("-inf"),
        "average_net_pnl": float("-inf"),
        "nonzero_trade_rate": float("-inf"),
    }
    best_no_trade_epoch = -1
    best_no_trade_metrics = {
        "selection_score": float("-inf"),
        "average_reward": float("-inf"),
        "average_net_pnl": float("-inf"),
        "nonzero_trade_rate": float("-inf"),
    }
    initial_policy_state = _clone_policy_state(policy)
    best_no_trade_policy_state = initial_policy_state
    best_active_epoch = -1
    best_active_metrics = {
        "selection_score": float("-inf"),
        "average_reward": float("-inf"),
        "average_net_pnl": float("-inf"),
        "nonzero_trade_rate": float("-inf"),
    }
    best_active_policy_state = initial_policy_state
    top_checkpoint_candidates = []
    best_policy_state = initial_policy_state
    evaluation_decision_interval_business_days = _decision_interval_business_days_for_epoch(
        settings,
        None,
        evaluation=True,
    )
    evaluation_decision_time_indices = _decision_time_indices(
        bundle,
        settings,
        epoch=None,
        evaluation=True,
    )
    curriculum_stages = tuple(dict(stage) for stage in settings.get("decision_interval_curriculum", ()))
    current_stage_index = 0
    curriculum_success_streak = 0
    no_trade_validation_rollout = _collect_no_trade_rollout(
        validation_bundle,
        runtime,
        decision_time_indices=evaluation_decision_time_indices,
        decision_interval_business_days=evaluation_decision_interval_business_days,
    )
    no_trade_validation_metrics = _checkpoint_selection_metrics(
        no_trade_validation_rollout,
        no_trade_validation_rollout["terminal_transition"],
        runtime,
    )
    no_trade_validation_net_pnl = float(
        (
            no_trade_validation_rollout["terminal_transition"]["portfolio_value"]
            - no_trade_validation_rollout["terminal_transition"]["liability_value"]
        ).mean().item()
    )

    for epoch in range(int(settings["epochs"])):
        epsilon = _epsilon_for_epoch(settings, epoch)
        if curriculum_stages and bool(settings["performance_gated_curriculum"]):
            train_decision_interval_business_days = int(curriculum_stages[current_stage_index]["interval_business_days"])
            train_decision_time_indices = _decision_time_indices_for_interval(train_bundle, train_decision_interval_business_days)
        else:
            train_decision_interval_business_days = _decision_interval_business_days_for_epoch(
                settings,
                epoch,
                evaluation=False,
            )
            train_decision_time_indices = _decision_time_indices(
                train_bundle,
                settings,
                epoch=epoch,
                evaluation=False,
            )
        rollout_started = time.perf_counter()
        rollout = _collect_structured_rollout(
            policy,
            train_bundle,
            runtime,
            greedy=False,
            epsilon=epsilon,
            decision_time_indices=train_decision_time_indices,
            decision_interval_business_days=train_decision_interval_business_days,
        )
        epoch_rollout_times_seconds.append(float(time.perf_counter() - rollout_started))

        replay_storage_started = time.perf_counter()
        if int(rollout["replay_batch"].batch_size[0]) > 0:
            _extend_replay_buffer(replay_buffer, rollout["replay_batch"])
        epoch_replay_storage_times_seconds.append(float(time.perf_counter() - replay_storage_started))
        gradient_update_started = time.perf_counter()
        losses = _update_structured_policy(
            policy,
            critic,
            target_critic,
            critic_optimizer,
            actor_optimizer,
            replay_buffer,
            settings,
        )
        epoch_gradient_update_times_seconds.append(float(time.perf_counter() - gradient_update_started))
        if (epoch + 1) % max(int(settings["target_update_interval"]), 1) == 0:
            target_critic.load_state_dict(critic.state_dict())
        terminal_transition = rollout["terminal_transition"]
        epoch_rewards.append(float(terminal_transition["reward"].mean().item()))
        epoch_average_net_pnls.append(float((terminal_transition["portfolio_value"] - terminal_transition["liability_value"]).mean().item()))
        epoch_critic_losses.append(
            float(sum(losses["critic_losses"]) / len(losses["critic_losses"])) if losses["critic_losses"] else 0.0
        )
        epoch_actor_losses.append(
            float(sum(losses["actor_losses"]) / len(losses["actor_losses"])) if losses["actor_losses"] else 0.0
        )
        epoch_replay_sizes.append(_replay_buffer_total_size(replay_buffer))
        epoch_no_trade_rates.append(float(rollout["no_trade_rate"]))
        rollout_stats = _scalar_rollout_diagnostics(rollout.get("rollout_diagnostics"))
        epoch_mean_abs_trade_deltas.append(float(rollout_stats.get("mean_abs_trade_delta", 0.0)))
        epoch_mean_abs_rebalance_signals.append(float(rollout_stats.get("mean_abs_rebalance_signal", 0.0)))
        epoch_decision_interval_business_days.append(float(train_decision_interval_business_days))
        epoch_curriculum_stage_indices.append(float(current_stage_index))

        shard_stage_net_pnls = []
        shard_stage_trade_rates = []
        shard_success_count = 0
        trade_rate_low = float(settings["curriculum_min_trade_rate"])
        trade_rate_high = float(settings["curriculum_max_trade_rate"])
        stage_validation_time_indices_by_shard = stage_validation_time_indices_cache.get(train_decision_interval_business_days)
        if stage_validation_time_indices_by_shard is None:
            stage_validation_time_indices_by_shard = tuple(
                _decision_time_indices_for_interval(validation_shard, train_decision_interval_business_days)
                for validation_shard in validation_shards
            )
            stage_validation_time_indices_cache[train_decision_interval_business_days] = stage_validation_time_indices_by_shard
        for validation_shard, stage_validation_time_indices in zip(validation_shards, stage_validation_time_indices_by_shard):
            stage_validation_rollout = _collect_structured_rollout(
                policy,
                validation_shard,
                runtime,
                greedy=True,
                epsilon=0.0,
                decision_time_indices=stage_validation_time_indices,
                decision_interval_business_days=train_decision_interval_business_days,
            )
            stage_validation_terminal = stage_validation_rollout["terminal_transition"]
            shard_net_pnl = float(
                (
                    stage_validation_terminal["portfolio_value"]
                    - stage_validation_terminal["liability_value"]
                ).mean().item()
            )
            shard_trade_rate = float(stage_validation_rollout["rollout_diagnostics"].get("nonzero_trade_rate", 0.0))
            shard_stage_net_pnls.append(shard_net_pnl)
            shard_stage_trade_rates.append(shard_trade_rate)
            if shard_net_pnl > no_trade_validation_net_pnl and trade_rate_low <= shard_trade_rate <= trade_rate_high:
                shard_success_count += 1
        stage_validation_net_pnl = float(sum(shard_stage_net_pnls) / len(shard_stage_net_pnls)) if shard_stage_net_pnls else no_trade_validation_net_pnl
        stage_validation_trade_rate = float(sum(shard_stage_trade_rates) / len(shard_stage_trade_rates)) if shard_stage_trade_rates else 0.0
        shard_success_rate = float(shard_success_count / len(shard_stage_net_pnls)) if shard_stage_net_pnls else 0.0
        epoch_stage_validation_net_pnls.append(stage_validation_net_pnl)
        epoch_stage_validation_trade_rates.append(stage_validation_trade_rate)
        epoch_stage_validation_shard_success_rates.append(shard_success_rate)
        if curriculum_stages and bool(settings["performance_gated_curriculum"]) and current_stage_index < len(curriculum_stages) - 1:
            improvement_threshold = no_trade_validation_net_pnl + (0.25 * float(settings["curriculum_min_improvement"]))
            if (
                stage_validation_net_pnl > improvement_threshold
                and trade_rate_low <= stage_validation_trade_rate <= trade_rate_high
                and shard_success_rate >= 0.5
            ):
                curriculum_success_streak += 1
            else:
                curriculum_success_streak = 0
            if curriculum_success_streak >= max(int(settings["curriculum_advance_patience"]), 1):
                current_stage_index += 1
                curriculum_success_streak = 0

        greedy_eval_started = time.perf_counter()
        shard_metrics = []
        current_policy_state = None
        for validation_shard, shard_time_indices in zip(validation_shards, greedy_validation_time_indices):
            greedy_rollout = _collect_structured_rollout(
                policy,
                validation_shard,
                runtime,
                greedy=True,
                epsilon=0.0,
                decision_time_indices=shard_time_indices,
                decision_interval_business_days=evaluation_decision_interval_business_days,
            )
            greedy_terminal_transition = greedy_rollout["terminal_transition"]
            shard_metrics.append(
                _checkpoint_selection_metrics(
                    greedy_rollout,
                    greedy_terminal_transition,
                    runtime,
                )
            )
        epoch_greedy_eval_times_seconds.append(float(time.perf_counter() - greedy_eval_started))
        greedy_checkpoint_metrics = _aggregate_checkpoint_metrics(tuple(shard_metrics))
        epoch_greedy_selection_scores.append(float(greedy_checkpoint_metrics["selection_score"]))
        epoch_greedy_average_rewards.append(float(greedy_checkpoint_metrics["average_reward"]))
        epoch_greedy_average_net_pnls.append(float(greedy_checkpoint_metrics["average_net_pnl"]))
        epoch_greedy_nonzero_trade_rates.append(float(greedy_checkpoint_metrics["nonzero_trade_rate"]))
        epoch_greedy_mean_abs_trade_deltas.append(float(greedy_checkpoint_metrics["mean_abs_trade_delta"]))
        if float(greedy_checkpoint_metrics["nonzero_trade_rate"]) <= 0.01:
            if float(greedy_checkpoint_metrics["average_net_pnl"]) > float(best_no_trade_metrics["average_net_pnl"]):
                best_no_trade_metrics = dict(greedy_checkpoint_metrics)
                best_no_trade_epoch = int(epoch)
                if current_policy_state is None:
                    current_policy_state = _clone_policy_state(policy)
                best_no_trade_policy_state = current_policy_state
        elif _is_better_checkpoint(greedy_checkpoint_metrics, best_active_metrics):
            best_active_metrics = dict(greedy_checkpoint_metrics)
            best_active_epoch = int(epoch)
            if current_policy_state is None:
                current_policy_state = _clone_policy_state(policy)
            best_active_policy_state = current_policy_state
        if float(greedy_checkpoint_metrics["nonzero_trade_rate"]) > 0.01:
            if current_policy_state is None:
                current_policy_state = _clone_policy_state(policy)
            top_checkpoint_candidates.append(
                {
                    "epoch": int(epoch),
                    "metrics": dict(greedy_checkpoint_metrics),
                    "policy_state": current_policy_state,
                }
            )
            top_checkpoint_candidates.sort(
                key=lambda candidate: _checkpoint_sort_key(candidate["metrics"]),
                reverse=True,
            )
            top_checkpoint_candidates = top_checkpoint_candidates[: max(int(settings["top_validation_checkpoints"]), 1)]
    shortlisted_candidates = list(top_checkpoint_candidates)

    if shortlisted_candidates:
        shortlisted_candidates.sort(
            key=lambda candidate: _checkpoint_sort_key(candidate["metrics"]),
            reverse=True,
        )
        top_candidate = shortlisted_candidates[0]
        net_pnl_tolerance = max(4.0 * float(settings["curriculum_min_improvement"]), 1.0)
        candidate_materially_worse_than_no_trade = float(top_candidate["metrics"]["average_net_pnl"]) < (
            float(no_trade_validation_metrics["average_net_pnl"]) - net_pnl_tolerance
        )
        candidate_trade_rate = float(top_candidate["metrics"].get("nonzero_trade_rate", 0.0))
        candidate_is_active_enough = candidate_trade_rate >= float(settings["curriculum_min_trade_rate"])
        if candidate_is_active_enough and not candidate_materially_worse_than_no_trade:
            best_checkpoint_metrics = dict(top_candidate["metrics"])
            best_epoch = int(top_candidate["epoch"])
            best_policy_state = deepcopy(top_candidate["policy_state"])
        else:
            best_checkpoint_metrics = dict(no_trade_validation_metrics)
            best_epoch = int(best_no_trade_epoch)
            best_policy_state = deepcopy(best_no_trade_policy_state)
    elif float(best_active_metrics["average_net_pnl"]) > float(best_no_trade_metrics["average_net_pnl"]):
        best_checkpoint_metrics = dict(best_active_metrics)
        best_epoch = int(best_active_epoch)
        best_policy_state = deepcopy(best_active_policy_state)
    else:
        best_checkpoint_metrics = dict(no_trade_validation_metrics)
        best_epoch = int(best_no_trade_epoch)
        best_policy_state = deepcopy(best_no_trade_policy_state)

    policy.load_state_dict(best_policy_state)

    final_evaluation_started = time.perf_counter()
    evaluation_rollout = _collect_structured_rollout(
        policy,
        bundle,
        runtime,
        greedy=True,
        epsilon=0.0,
        decision_time_indices=evaluation_decision_time_indices,
        decision_interval_business_days=evaluation_decision_interval_business_days,
    )
    no_trade_rollout = _collect_no_trade_rollout(
        bundle,
        runtime,
        decision_time_indices=evaluation_decision_time_indices,
        decision_interval_business_days=evaluation_decision_interval_business_days,
    )
    final_evaluation_time_seconds = float(time.perf_counter() - final_evaluation_started)
    total_fit_time_seconds = float(time.perf_counter() - fit_started)
    evaluation_output = build_torchrl_evaluation_output(
        evaluation_rollout["terminal_transition"]["state"],
        evaluation_rollout["terminal_transition"],
        evaluation_rollout["action_counts"],
        evaluation_rollout["no_trade_rate"],
        bundle,
        trainer_type="minimal_tensor_actor_critic",
        rollout_diagnostics=evaluation_rollout.get("rollout_diagnostics"),
        no_trade_reference={
            "state": no_trade_rollout["terminal_transition"]["state"],
            "terminal_transition": no_trade_rollout["terminal_transition"],
            "action_counts": no_trade_rollout["action_counts"],
            "no_trade_rate": no_trade_rollout["no_trade_rate"],
            "rollout_diagnostics": no_trade_rollout.get("rollout_diagnostics"),
        },
        timing={
            "rollout_time_seconds": float(sum(epoch_rollout_times_seconds)),
            "replay_storage_time_seconds": float(sum(epoch_replay_storage_times_seconds)),
            "gradient_update_time_seconds": float(sum(epoch_gradient_update_times_seconds)),
            "greedy_selection_time_seconds": float(sum(epoch_greedy_eval_times_seconds)),
            "evaluation_time_seconds": final_evaluation_time_seconds,
            "total_fit_time_seconds": total_fit_time_seconds,
        },
    )
    return {
        "policy": policy,
        "policy_artifact": policy.to_artifact(),
        "evaluation_output": evaluation_output,
        "optimizer_diagnostics": {
            "learner_type": "torchrl_tensor_policy",
            "trainer_type": "minimal_torchrl_training_entrypoint",
            "adapter_type": "dict_tensor_state",
            "epoch_rewards": epoch_rewards,
            "epoch_critic_losses": epoch_critic_losses,
            "epoch_actor_losses": epoch_actor_losses,
            "epoch_average_net_pnls": epoch_average_net_pnls,
            "epoch_no_trade_rates": epoch_no_trade_rates,
            "epoch_mean_abs_trade_deltas": epoch_mean_abs_trade_deltas,
            "epoch_mean_abs_rebalance_signals": epoch_mean_abs_rebalance_signals,
            "epoch_decision_interval_business_days": epoch_decision_interval_business_days,
            "epoch_greedy_selection_scores": epoch_greedy_selection_scores,
            "epoch_greedy_average_rewards": epoch_greedy_average_rewards,
            "epoch_greedy_average_net_pnls": epoch_greedy_average_net_pnls,
            "epoch_greedy_nonzero_trade_rates": epoch_greedy_nonzero_trade_rates,
            "epoch_greedy_mean_abs_trade_deltas": epoch_greedy_mean_abs_trade_deltas,
            "epoch_stage_validation_net_pnls": epoch_stage_validation_net_pnls,
            "epoch_stage_validation_trade_rates": epoch_stage_validation_trade_rates,
            "epoch_stage_validation_shard_success_rates": epoch_stage_validation_shard_success_rates,
            "epoch_curriculum_stage_indices": epoch_curriculum_stage_indices,
            "epoch_replay_sizes": epoch_replay_sizes,
            "epoch_rollout_times_seconds": epoch_rollout_times_seconds,
            "epoch_replay_storage_times_seconds": epoch_replay_storage_times_seconds,
            "epoch_gradient_update_times_seconds": epoch_gradient_update_times_seconds,
            "epoch_greedy_eval_times_seconds": epoch_greedy_eval_times_seconds,
            "torchrl_rollout_time_seconds": float(sum(epoch_rollout_times_seconds)),
            "replay_storage_time_seconds": float(sum(epoch_replay_storage_times_seconds)),
            "gradient_update_time_seconds": float(sum(epoch_gradient_update_times_seconds)),
            "greedy_selection_time_seconds": float(sum(epoch_greedy_eval_times_seconds)),
            "final_evaluation_time_seconds": final_evaluation_time_seconds,
            "total_fit_time_seconds": total_fit_time_seconds,
            "final_average_net_pnl": float(evaluation_output["metrics"]["average_net_pnl"]),
            "final_average_liability": float(evaluation_output["metrics"]["average_liability"]),
            "final_average_hedge_pnl": float(evaluation_output["metrics"]["average_hedge_pnl"]),
            "final_no_trade_rate": float(evaluation_rollout["no_trade_rate"]),
            "final_rollout_diagnostics": _scalar_rollout_diagnostics(evaluation_rollout.get("rollout_diagnostics")),
            "best_epoch": best_epoch,
            "best_selection_score": float(best_checkpoint_metrics["selection_score"]),
            "best_average_reward": float(best_checkpoint_metrics["average_reward"]),
            "best_average_net_pnl": float(best_checkpoint_metrics["average_net_pnl"]),
            "best_nonzero_trade_rate": float(best_checkpoint_metrics["nonzero_trade_rate"]),
            "best_mean_abs_trade_delta": float(best_checkpoint_metrics.get("mean_abs_trade_delta", 0.0)),
            "replay_size": _replay_buffer_total_size(replay_buffer),
            "active_replay_size": int(len(replay_buffer["active"])),
            "inactive_replay_size": int(len(replay_buffer["inactive"])),
            "training_episode_count": int(training_episode_count),
            "validation_episode_count": int(validation_episode_count),
            "validation_shard_count": int(len(validation_shards)),
            "validation_no_trade_net_pnl": float(no_trade_validation_net_pnl),
            "top_validation_candidate_epochs": [int(candidate["epoch"]) for candidate in shortlisted_candidates],
            "num_episodes": int(evaluation_output["diagnostics"].get("num_episodes", 0)),
            "optimizer_params": dict(runtime.get("optimizer", {})),
        },
    }


def run_torchrl_execution(bundle: Dict[str, Any], runtime: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    execution_mode = runtime.get("execution_mode", "simulate_only")
    if execution_mode not in {"evaluate_policy", "optimize_policy"}:
        return None
    if runtime.get("policy", {}).get("object") != "StructuredRebalancePolicy":
        return None
    if execution_mode == "optimize_policy":
        return train_torchrl_policy(bundle, runtime)
    return evaluate_torchrl_policy(bundle, runtime)