"""Normalized TorchRL hedging runtime construction.

This module owns the canonical TorchRL runtime contracts:

- runtime dict normalization from JSON/job spec
- canonical state flattening layout

Execution, rollout, and optimization live in torchrl_hedge.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import torch

from .hedge_features import build_entity_layout


class TerminalFloorThenSurplusUtility:
    def __init__(self, params: Optional[Mapping[str, Any]] = None, **kwargs):
        config = dict(params or {})
        config.update(kwargs)

        self.floor_penalty = float(config['Floor_Penalty'] if 'Floor_Penalty' in config else config['floor_penalty'])
        self.surplus_reward = float(config['Surplus_Reward'] if 'Surplus_Reward' in config else config['surplus_reward'])
        self.power = float(config['Power'] if 'Power' in config else config['power'])

        if self.floor_penalty < 0.0:
            raise ValueError("Objective.Floor_Penalty must be non-negative")
        if self.surplus_reward < 0.0:
            raise ValueError("Objective.Surplus_Reward must be non-negative")
        if self.power <= 0.0:
            raise ValueError("Objective.Power must be positive")

    def evaluate_terminal_outcome(self, *, hedge_pnl, liability, net_pnl):
        del hedge_pnl, liability
        if net_pnl >= 0.0:
            return self.surplus_reward * (net_pnl ** self.power)
        shortfall = -net_pnl
        return -self.floor_penalty * (shortfall ** self.power)

    def evaluate_episode(self, episode):
        return self.evaluate_terminal_outcome(
            hedge_pnl=float(episode.hedge_pnl),
            liability=float(episode.liability),
            net_pnl=float(episode.net_pnl),
        )


def construct_objective(config: Mapping[str, Any]):
    if str(config["Object"]) != "TerminalFloorThenSurplusUtility":
        raise ValueError(f"Unsupported objective: {config['Object']}")
    return TerminalFloorThenSurplusUtility(config)


def _as_timestamp(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, dict) and ".Timestamp" in value:
        return pd.Timestamp(value[".Timestamp"])
    return pd.Timestamp(value)


def _unwrap_calc_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "Hedging_Problem" in config:
        return config
    return config["Calc"]["Calculation"]


def _normalize_execution_mode(config: Mapping[str, Any]) -> str:
    return str(config.get("Execution_Mode", "simulate_only")).lower()


def _normalize_accounting_mode(evaluator_config: Mapping[str, Any]) -> str:
    return str(evaluator_config.get("Accounting_Mode", "futures")).lower()


def _normalize_cash_account_names(evaluator_config: Mapping[str, Any]) -> tuple:
    if evaluator_config.get("Cash_Instruments") is not None:
        return tuple(str(name) for name in evaluator_config.get("Cash_Instruments", ()))
    if evaluator_config.get("Cash_Accounts") is not None:
        return tuple(str(name) for name in evaluator_config.get("Cash_Accounts", ()))
    cash_name = evaluator_config.get("Cash_Instrument")
    if cash_name is None:
        return tuple()
    return (str(cash_name),)


def _flatten_tradable_entries(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    flattened: Dict[str, Dict[str, Any]] = {}
    for deal_type, deal_mapping in config.items():
        for instrument_name, instrument_config in deal_mapping.items():
            flattened[str(instrument_name)] = {
                "deal_type": str(deal_type),
                "params": deepcopy(dict(instrument_config)),
            }
    return flattened


def _normalize_liabilities(hedging_problem: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if hedging_problem.get("Liabilities") is not None:
        liability_entries = _flatten_tradable_entries(hedging_problem.get("Liabilities", {}))
        normalized = {}
        for liability_name, liability_entry in liability_entries.items():
            params = liability_entry["params"]
            normalized[str(liability_name)] = {
                "reference": str(liability_name),
                "object": liability_entry["deal_type"],
                "deal_type": liability_entry["deal_type"],
                "underlying": params.get("Underlying"),
                "currency": params.get("Currency"),
                "strike": float(params.get("Strike", params.get("Strike_Price", 0.0))),
                "quantity": float(params.get("Quantity", params.get("Units", 0.0))),
                "expiry_date": params.get("Expiry_Date"),
                "params": deepcopy(dict(params)),
            }
        return normalized
    return {}


def _normalize_policy_config(policy_config: Mapping[str, Any]) -> Dict[str, Any]:
    action_space = _normalize_action_space(policy_config)
    model_config = dict(policy_config.get("Model", {}))
    return {
        "object": str(policy_config["Object"]),
        "action_space": action_space,
        "model": {
            "object": str(model_config.get("Object", "EntityTransformer")),
            "token_dim": int(model_config.get("Token_Dim", 64)),
            "emb_dim": int(model_config.get("Emb_Dim", 8)),
            "n_heads": int(model_config.get("N_Heads", 4)),
            "n_layers": int(model_config.get("N_Layers", 2)),
            "log_std_init": float(model_config.get("Log_Std_Init", -0.5)),
        },
    }


def _normalize_optimizer_config(optimizer_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if optimizer_config is None:
        return None
    decision_interval_curriculum = []
    for stage in optimizer_config.get("Decision_Interval_Curriculum", ()):
        decision_interval_curriculum.append({
            "start_epoch": int(stage.get("Start_Epoch", 1)),
            "end_epoch": None if stage.get("End_Epoch") is None else int(stage.get("End_Epoch")),
            "interval_business_days": int(stage.get("Interval_Business_Days", 1)),
        })
    return {
        "object": str(optimizer_config["Object"]),
        "epochs": int(optimizer_config.get("Epochs", 20)),
        "ppo_epochs": int(optimizer_config.get("PPO_Epochs", 4)),
        "minibatch_size": int(optimizer_config.get("Minibatch_Size", 4096)),
        "gamma": float(optimizer_config.get("Gamma", 0.99)),
        "gae_lambda": float(optimizer_config.get("GAE_Lambda", 0.95)),
        "learning_rate": float(optimizer_config.get("Learning_Rate", 3.0e-4)),
        "clip_eps": float(optimizer_config.get("Clip_Eps", 0.2)),
        "value_coef": float(optimizer_config.get("Value_Coef", 0.5)),
        "entropy_coef": float(optimizer_config.get("Entropy_Coef", 0.01)),
        "max_grad_norm": float(optimizer_config.get("Max_Grad_Norm", 0.5)),
        "reward_scale": float(optimizer_config.get("Reward_Scale", 1.0)),
        "action_sparsity_coef": float(optimizer_config.get("Action_Sparsity_Coef", 0.0)),
        "dense_tracking_reward_scale": float(optimizer_config.get("Dense_Tracking_Reward_Scale", 0.0)),
        "dense_tracking_reward_clip": float(optimizer_config.get("Dense_Tracking_Reward_Clip", 0.15)),
        "validation_fraction": float(optimizer_config.get("Validation_Fraction", 0.25)),
        "validation_min_batch": int(optimizer_config.get("Validation_Min_Batch", 256)),
        "validation_shards": int(optimizer_config.get("Validation_Shards", 4)),
        "decision_interval_curriculum": tuple(decision_interval_curriculum),
        "seed": optimizer_config.get("Seed"),
    }


def _normalize_objective_config(objective_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if objective_config is None:
        return None
    return {
        "object": str(objective_config["Object"]),
        "floor_penalty": float(objective_config.get("Floor_Penalty", 1.0)),
        "surplus_reward": float(objective_config.get("Surplus_Reward", 1.0)),
        "power": float(objective_config.get("Power", 1.0)),
    }


def _normalize_action_space(policy_config: Mapping[str, Any]) -> Dict[str, Any]:
    action_space = dict(policy_config.get("Action_Space", {}))
    instrument_order = tuple(str(name) for name in action_space.get("Instrument_Order", ()))
    min_trade_delta = tuple(int(value) for value in action_space.get("Min_Trade_Delta", ()))
    max_trade_delta = tuple(int(value) for value in action_space.get("Max_Trade_Delta", ()))
    if instrument_order and not (
        len(instrument_order) == len(min_trade_delta) == len(max_trade_delta)
    ):
        raise ValueError("Structured action space entries must have matching lengths")
    return {
        "instrument_order": instrument_order,
        "min_trade_delta": {
            instrument_name: min_value
            for instrument_name, min_value in zip(instrument_order, min_trade_delta)
        },
        "max_trade_delta": {
            instrument_name: max_value
            for instrument_name, max_value in zip(instrument_order, max_trade_delta)
        },
    }


def _normalize_position_limits(evaluator_config: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    position_limits = {}
    for instrument_name, limit_config in evaluator_config.get("Position_Limits", {}).items():
        position_limits[str(instrument_name)] = {
            "min_position": int(limit_config["Min_Position"]),
            "max_position": int(limit_config["Max_Position"]),
        }
    return position_limits


def _normalize_spot_price_history(
    hedging_problem: Mapping[str, Any],
    lookback: int,
    referenced_commodities: tuple,
) -> Dict[str, Dict[str, Any]]:
    state = hedging_problem.get("Portfolio_State") or {}
    raw_history = state.get("Spot_Price_History") or {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for commodity, payload in raw_history.items():
        commodity_name = str(commodity)
        dates_raw = payload.get("Dates", ())
        prices_raw = payload.get("Prices", ())
        if len(dates_raw) != len(prices_raw):
            raise ValueError(
                f"Spot_Price_History['{commodity_name}']: Dates and Prices must have equal length "
                f"({len(dates_raw)} vs {len(prices_raw)})"
            )
        if len(dates_raw) < lookback:
            raise ValueError(
                f"Spot_Price_History['{commodity_name}']: needs at least "
                f"History_Lookback_Business_Days={lookback} entries, got {len(dates_raw)}"
            )
        dates = tuple(_as_timestamp(d) for d in dates_raw)
        for i in range(1, len(dates)):
            if dates[i] <= dates[i - 1]:
                raise ValueError(
                    f"Spot_Price_History['{commodity_name}']: Dates must be strictly ascending; "
                    f"found {dates[i - 1]} >= {dates[i]} at index {i}"
                )
        prices = tuple(float(p) for p in prices_raw)
        normalized[commodity_name] = {"dates": dates, "prices": prices}
    missing = tuple(c for c in referenced_commodities if c not in normalized)
    if missing:
        raise ValueError(
            f"Spot_Price_History missing entries for referenced commodities: {missing}"
        )
    return normalized


def _collect_referenced_commodities(
    liabilities: Mapping[str, Mapping[str, Any]],
    tradables: Mapping[str, Mapping[str, Any]],
) -> tuple:
    referenced: list = []
    for entry in liabilities.values():
        params = entry.get("params") or {}
        for key in ("Commodity", "Reference_Type"):
            value = params.get(key)
            if value is not None and str(value) not in referenced:
                referenced.append(str(value))
    for entry in tradables.values():
        params = entry.get("params") or {}
        value = params.get("Commodity")
        if value is not None and str(value) not in referenced:
            referenced.append(str(value))
    return tuple(referenced)


def _normalize_portfolio_state(
    hedging_problem: Mapping[str, Any],
    *,
    lookback: int,
    referenced_commodities: tuple,
) -> Dict[str, Any]:
    state = hedging_problem.get("Portfolio_State") or {}
    spot_history = _normalize_spot_price_history(hedging_problem, lookback, referenced_commodities)
    return {
        "positions": {str(name): float(value) for name, value in state.get("Positions", {}).items()},
        "cash_balances": {str(name): float(value) for name, value in state.get("Cash_Balances", {}).items()},
        "settlement_prices": {str(name): float(value) for name, value in state.get("Settlement_Prices", {}).items()},
        "margin_balances": {str(name): float(value) for name, value in state.get("Margin_Balances", {}).items()},
        "initial_margin": {
            str(name): {"method": str(spec["Method"]), "amount": float(spec["Amount"])}
            for name, spec in state.get("Initial_Margin", {}).items()
        },
        "spot_price_history": spot_history,
    }


def _normalize_instrument_metadata(
    instrument_name: str,
    tradable_entry: Mapping[str, Any],
    *,
    hedge_names: tuple,
    cash_account_names: tuple,
    liability_expiry: Any,
) -> Dict[str, Any]:
    params = tradable_entry["params"]
    investment_horizon = params.get("Investment_Horizon")
    expiry_date = params.get("Expiry_Date", investment_horizon if investment_horizon is not None else liability_expiry)
    last_trade_date = params.get("Last_Trade_Date", investment_horizon if investment_horizon is not None else liability_expiry)
    return {
        "name": str(instrument_name),
        "deal_type": tradable_entry["deal_type"],
        "is_hedge": instrument_name in hedge_names,
        "is_cash_account": instrument_name in cash_account_names,
        "currency": params.get("Currency"),
        "last_trade_date": last_trade_date,
        "expiry_date": expiry_date,
        "first_notice_date": params.get("First_Notice_Date"),
        "auto_close_days_before_last_trade": int(params.get("Auto_Close_Days_Before_Last_Trade", 0)),
        "allow_new_positions_until_last_trade": bool(params.get("Allow_New_Positions_Until_Last_Trade", True)),
        "allow_holding_past_last_trade": bool(params.get("Allow_Holding_Past_Last_Trade", False)),
        "contract_size": float(params.get("Contract_Size", 1.0)),
        "params": deepcopy(dict(params)),
    }


def construct_torchrl_runtime(config: Mapping[str, Any]) -> Dict[str, Any]:
    config = _unwrap_calc_config(config)
    hedging_problem = config["Hedging_Problem"]
    evaluator_config = hedging_problem["Evaluator"]
    liabilities = _normalize_liabilities(hedging_problem)
    objective_config = hedging_problem.get("Objective")
    policy_config = hedging_problem["Policy"]
    optimizer_config = hedging_problem.get("Optimizer")
    tradables = _flatten_tradable_entries(hedging_problem["Tradable_Instruments"])
    execution_mode = _normalize_execution_mode(config)
    accounting_mode = _normalize_accounting_mode(evaluator_config)
    cash_account_names = _normalize_cash_account_names(evaluator_config)
    for account_name in cash_account_names:
        if str(account_name) not in tradables:
            raise ValueError(f"Evaluator cash account '{account_name}' is not in Tradable_Instruments")
    position_limits = _normalize_position_limits(evaluator_config)
    allow_short = bool(evaluator_config.get("Allow_Short", True))
    if not allow_short:
        invalid_shorts = [
            instrument_name
            for instrument_name, limit in position_limits.items()
            if int(limit["min_position"]) < 0
        ]
        if invalid_shorts:
            raise ValueError(
                "Evaluator.Allow_Short is False but Position_Limits allow short positions for "
                f"{invalid_shorts}"
            )
    hedge_names = tuple(
        instrument_name
        for instrument_name in position_limits.keys()
        if instrument_name not in cash_account_names
    )
    liability_expiry = None
    for liability in liabilities.values():
        expiry_date = liability.get("expiry_date")
        if expiry_date is None:
            continue
        expiry_timestamp = _as_timestamp(expiry_date)
        if liability_expiry is None or expiry_timestamp > _as_timestamp(liability_expiry):
            liability_expiry = expiry_date
    normalized_tradables = {
        instrument_name: _normalize_instrument_metadata(
            instrument_name,
            tradable_entry,
            hedge_names=hedge_names,
            cash_account_names=cash_account_names,
            liability_expiry=liability_expiry,
        )
        for instrument_name, tradable_entry in tradables.items()
    }
    if execution_mode == "optimize_policy" and optimizer_config is None:
        raise ValueError("Execution_Mode 'optimize_policy' requires Hedging_Problem['Optimizer']")
    policy = _normalize_policy_config(policy_config)
    names = {
        "tradables": tuple(normalized_tradables.keys()),
        "hedges": hedge_names,
        "cash_accounts": cash_account_names,
        "action_instruments": tuple(policy["action_space"]["instrument_order"]),
        "liabilities": tuple(liabilities.keys()),
    }
    history_lookback = int(hedging_problem.get("History_Lookback_Business_Days", 30))
    if history_lookback < 0:
        raise ValueError("Hedging_Problem.History_Lookback_Business_Days must be non-negative")
    referenced_commodities = _collect_referenced_commodities(liabilities, normalized_tradables)
    runtime = {
        "execution_mode": execution_mode,
        "accounting_mode": accounting_mode,
        "names": names,
        "tradables": normalized_tradables,
        "liabilities": liabilities,
        "objective": _normalize_objective_config(objective_config),
        "policy": policy,
        "optimizer": _normalize_optimizer_config(optimizer_config),
        "history_lookback_business_days": history_lookback,
        "portfolio_state": _normalize_portfolio_state(
            hedging_problem,
            lookback=history_lookback,
            referenced_commodities=referenced_commodities,
        ),
        "accounting": {
            "position_limits": position_limits,
            "cash_accounts": {
                account_name: {
                    "currency": normalized_tradables[account_name]["currency"],
                }
                for account_name in cash_account_names
            },
            "transaction_cost_per_unit": float(evaluator_config.get("Transaction_Cost_Per_Unit", 0.0)),
            "bid_offer_spread_bps": float(evaluator_config.get("Bid_Offer_Spread_Bps", 0.0)),
            "force_flat_at_end": bool(evaluator_config.get("Force_Flat_At_End", True)),
            "allow_short": allow_short,
            "fail_on_unhedgeable_intent": bool(evaluator_config.get("Fail_On_Unhedgeable_Intent", False)),
        },
    }
    runtime["entity_layout"] = build_entity_layout(runtime)
    return runtime


def construct_hedging_runtime(config: Mapping[str, Any]) -> Dict[str, Any]:
    return construct_torchrl_runtime(config)
