"""Normalized hedging-runtime construction.

This module owns the canonical hedging runtime contract:

- runtime dict normalization from JSON/job spec (Evaluator, Objective, Solver, tradables)
- canonical state/entity layout

The bundle builder, env simulator, and Execution_Mode dispatch live in hedge_bundle.py;
the differential-ML solver lives in hedge_solver.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import torch

from . import utils


def _privileged_name(factor_name, attr_name, multi):
    """Multi-commodity runs prefix factor attribute names with `<factor>_` to disambiguate."""
    return f'{factor_name.lower()}_{attr_name}' if multi else attr_name


def _privileged_multi(stoch_factors):
    """True iff there are multiple distinct primary factor names — drives the prefix decision."""
    return len({f.name[0] for f in (stoch_factors or {})}) > 1


def derive_privileged_layout(stoch_factors):
    """Build the {name: dim} schema by asking each live stoch-factor process what it emits.
    Polymorphic via `type(process).privileged_layout(process.param)` — adding a new
    StochasticProcess subclass with its own privileged surface flows through automatically."""
    multi = _privileged_multi(stoch_factors)
    layout = {}
    for factor, process in (stoch_factors or {}).items():
        for attr_name, dim in type(process).privileged_layout(process.param).items():
            layout[_privileged_name(factor.name[0], attr_name, multi)] = int(dim)
    return layout


def assemble_privileged_factors(privileged_factor_blocks, stoch_factors):
    """Concatenate per-batch privileged-factor tensors collected during the simulation loop into
    a single dict ready for the bundle. Input keyed by (factor_name, attr_name); output keys match
    the schema produced by `derive_privileged_layout`."""
    multi = _privileged_multi(stoch_factors)
    return {
        _privileged_name(factor_name, attr_name, multi): torch.cat(blocks, dim=1)
        for (factor_name, attr_name), blocks in privileged_factor_blocks.items()
    }


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


def _normalize_solver_config(solver_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize the `Solver` config block (Execution_Mode='solve_hedge'). Accepts None
    (non-solve modes); requires `Object` — one of 'diffsolverv2' | 'hindsightdpsolver'."""
    if solver_config is None:
        return None
    if "Object" not in solver_config:
        raise ValueError("Hedging_Problem['Solver'] requires an 'Object' field")
    return {
        "object": str(solver_config["Object"]).lower(),
        "multi_seed_count": int(solver_config.get("Multi_Seed_Count", 1)),
        # Backward-sweep depth: fit C_t for t in [t_outer-2 .. t_min]. 0 = full sweep to the
        # initial decision; t_min near t_outer-1 = a shallow (bounded) sweep.
        "t_min": int(solver_config.get("T_Min", 0)),
        # Greedy-decision action grid (levels per hedge axis) + batched-argmax chunk size.
        "training_action_grid_levels_per_axis":
            int(solver_config.get("Training_Action_Grid_Levels_Per_Axis", 11)),
        "training_action_chunk_size": int(solver_config.get("Training_Action_Chunk_Size", 64)),
        # Advantage decomposition: fit A = C - u(W) (NN residual over the bounded-utility anchor).
        "use_advantage_decomp": bool(solver_config.get("Use_Advantage_Decomp", True)),
        # Differential-ML twin-loss weight (stored; the active twin weight is diffv2_lambda_grad).
        "lambda_diff": float(solver_config.get("Lambda_Diff", 0.0)),
        # --- DiffSolverV2 (clean-room differential-ML solver) knobs ---
        # Per-t residual-net Adam iters / lr; bank q-exploration noise as a fraction of each
        # instrument's [Min,Max] range; the subset of hedge instruments whose action axis VARIES
        # in the grid (others pinned to 0). None = all vary.
        "diffv2_fit_iters": int(solver_config.get("DiffV2_Fit_Iters", 150)),
        "diffv2_lr": float(solver_config.get("DiffV2_LR", 2.0e-3)),
        "diffv2_bank_noise_frac": float(solver_config.get("DiffV2_Bank_Noise_Frac", 0.15)),
        # Out-of-sample split: fraction of OUTER paths held out from the bank/fit, used ONLY for
        # the verdict rollout (honest OOS policy eval). 0 = in-sample only.
        "diffv2_oos_frac": float(solver_config.get("DiffV2_OOS_Frac", 0.5)),
        # Residual-net regularization. The PRINCIPLED regularizer is the twin-loss pathwise-
        # gradient match (diffv2_lambda_grad), applied in STANDARDIZED space; weight decay is an
        # optional crutch for outer-path-starved (tiny-batch) problems.
        "diffv2_weight_decay": float(solver_config.get("DiffV2_Weight_Decay", 0.0)),
        "diffv2_hidden": int(solver_config.get("DiffV2_Hidden", 32)),
        "diffv2_lambda_grad": float(solver_config.get("DiffV2_Lambda_Grad", 1.0)),
        # Downside-aware action SELECTION: at the argmax, score each action by
        # mean(C) - DiffV2_Risk_Kappa * downside-semidev(C) over the inner-MC, de-risking ONLY the
        # bad-tail actions (keeps upside). 0 = off (plain E[C] argmax, bit-identical). Tune ~0.5;
        # scale with regime-drift magnitude. (Toy: RISK_KAPPA beat the uniform min-var blend.)
        "diffv2_risk_kappa": float(solver_config.get("DiffV2_Risk_Kappa", 0.0)),
        "active_hedge_indices":
            (list(solver_config["Active_Hedge_Indices"])
             if solver_config.get("Active_Hedge_Indices") is not None else None),
        # Benchmark tracks assembled alongside the DiffSolverV2 deliverable (hindsight upper
        # bound / textbook lower bound).
        "run_hindsight_diagnostic": bool(solver_config.get("Run_Hindsight_Diagnostic", False)),
        "run_textbook_benchmark": bool(solver_config.get("Run_Textbook_Benchmark", False)),
    }


def _normalize_objective_config(objective_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if objective_config is None:
        return None
    explicit = objective_config.get("Utility_Scale_Explicit")
    return {
        # Canonical lowercased form — every dispatch site compares against the lowercase
        # literal (e.g. "asymmetricutility_symlog"), so normalize once at the boundary
        # rather than re-lowercasing on every reward call.
        "object": str(objective_config["Object"]).lower(),
        "floor_penalty": float(objective_config.get("Floor_Penalty", 1.0)),
        "surplus_reward": float(objective_config.get("Surplus_Reward", 1.0)),
        "power": float(objective_config.get("Power", 1.0)),
        "expiry_penalty": float(objective_config.get("Expiry_Penalty", 0.0)),
        "expiry_threshold_days": float(objective_config.get("Expiry_Threshold_Days", 4.0)),
        "post_deal_trade_penalty": float(objective_config.get("Post_Deal_Trade_Penalty", 0.0)),
        # NOTE: despite the name, `Position_Bounds_Penalty` controls ONLY the
        # portfolio-total Σ|pos_i| ramp (against `Evaluator.Total_Position_Abs_Limit`).
        # Per-instrument [Min_Position, Max_Position] bounds are handled separately by the
        # soft, reward-side `Per_Instrument_Bounds_Penalty` below, so this term is
        # portfolio-only. The naming pre-dates that split; rename to
        # `Total_Position_Abs_Penalty` if it becomes worth the JSON / CSV / docs churn.
        "position_bounds_penalty": float(objective_config.get("Position_Bounds_Penalty", 0.0)),
        "position_bounds_threshold": float(objective_config.get("Position_Bounds_Threshold", 5.0)),
        # Per-instrument [Min_Position, Max_Position] enforcement (soft, reward-side).
        "per_instrument_bounds_penalty": float(objective_config.get("Per_Instrument_Bounds_Penalty", 0.0)),
        "per_instrument_bounds_threshold": float(objective_config.get("Per_Instrument_Bounds_Threshold", 5.0)),
        # Utility-transform scale. Consumed by any utility Object (Symlog / Huber / CARA);
        # legacy "TerminalFloorThenSurplusUtility" path ignores it. `utility_scale` is mirrored
        # from `bundle["utility_scale"]` (see hedge_bundle._mirror_utility_scale_to_runtime).
        "utility_scale_mode": str(objective_config.get("Utility_Scale_Mode", "vol_scaled_notional")).lower(),
        "utility_scale_explicit": None if explicit is None else float(explicit),
        # Utility SHAPE params (DIMENSIONLESS, in units of the scale c — applied to x = W/c).
        # Huber (AsymmetricUtility_Huber): linear gains, quadratic small losses with curvature
        # `huber_aversion`, linear deep tail beyond the knee `huber_delta`. CARA
        # (AsymmetricUtility_CARA): u = (1−e^{−γx})/γ with risk aversion `cara_gamma`. Symlog
        # ignores all three. See hedge_bundle._utility_wrap_signed for the exact forms.
        "huber_aversion": float(objective_config.get("Huber_Aversion", 2.5)),
        "huber_delta": float(objective_config.get("Huber_Delta", 1.0)),
        "cara_gamma": float(objective_config.get("CARA_Gamma", 1.0)),
    }


def _build_instrument_cash_account_map(normalized_tradables, cash_account_names):
    """Static instrument→cash_account routing by currency. First cash account whose
    currency matches the instrument's currency wins; falls back to the first cash account
    if none match. Computed once at runtime construction; consumed by the env step loop."""
    accounts = list(cash_account_names)
    fallback = accounts[0] if accounts else None
    by_currency = {}
    for account in accounts:
        ccy = normalized_tradables.get(account, {}).get("currency")
        by_currency.setdefault(ccy, account)
    return {
        name: by_currency.get(meta.get("currency"), fallback)
        for name, meta in normalized_tradables.items()
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
        dates = tuple(dates_raw)
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
    if len(normalized) > 1:
        commodity_names = list(normalized.keys())
        ref_dates = normalized[commodity_names[0]]["dates"]
        for other in commodity_names[1:]:
            if normalized[other]["dates"] != ref_dates:
                raise ValueError(
                    f"Spot_Price_History['{other}'].Dates must match "
                    f"Spot_Price_History['{commodity_names[0]}'].Dates exactly"
                )
    return normalized


def _collect_referenced_commodities(stoch_factors: Optional[Mapping[Any, Any]]) -> tuple:
    """Pull commodity names from the live CommodityPrice factors. Instruments populate
    `stoch_factors` at construction (in `Calculation.calc_dependencies`); downstream
    consumers read from there rather than re-parsing instrument JSON params."""
    return tuple(dict.fromkeys(
        utils.check_tuple_name(factor) for factor in (stoch_factors or {})
        if factor.type == 'CommodityPrice'
    ))


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
    maturity_date = params.get("Maturity_Date")
    fallback = investment_horizon if investment_horizon is not None else liability_expiry
    expiry_date = params.get("Expiry_Date", maturity_date if maturity_date is not None else fallback)
    last_trade_date = params.get("Last_Trade_Date", maturity_date if maturity_date is not None else fallback)
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


def construct_hedge_runtime(
    config: Mapping[str, Any],
    stoch_factors: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    config = _unwrap_calc_config(config)
    hedging_problem = config["Hedging_Problem"]
    evaluator_config = hedging_problem["Evaluator"]
    liabilities = _normalize_liabilities(hedging_problem)
    objective_config = hedging_problem.get("Objective")
    solver_config = hedging_problem.get("Solver")
    tradables = _flatten_tradable_entries(hedging_problem["Tradable_Instruments"])
    execution_mode = _normalize_execution_mode(config)
    accounting_mode = _normalize_accounting_mode(evaluator_config)
    cash_account_names = _normalize_cash_account_names(evaluator_config)
    for account_name in cash_account_names:
        if str(account_name) not in tradables:
            raise ValueError(f"Evaluator cash account '{account_name}' is not in Tradable_Instruments")
    position_limits = _normalize_position_limits(evaluator_config)
    hedge_names = tuple(
        instrument_name
        for instrument_name in tradables.keys()
        if instrument_name not in cash_account_names
    )
    liability_expiry = None
    for liability in liabilities.values():
        expiry_date = liability.get("expiry_date")
        if expiry_date is None:
            continue
        if liability_expiry is None or expiry_date > liability_expiry:
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
    if execution_mode == "solve_hedge":
        if solver_config is None:
            raise ValueError("Execution_Mode 'solve_hedge' requires Hedging_Problem['Solver']")
        if str(config.get("Inner_MC_Enabled", "No")) != "Yes":
            raise ValueError("Execution_Mode 'solve_hedge' requires Inner_MC_Enabled='Yes'")
        solver_object = str(solver_config.get("Object", "")).lower()
        min_inner = 2 if solver_object == "diffsolverv2" else 128
        if int(config.get("Inner_Sub_Batch", 0)) < min_inner:
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Inner_Sub_Batch >= "
                f"{min_inner} for Solver.Object={solver_config.get('Object')!r}")
        _utility_objects = ("asymmetricutility_symlog", "asymmetricutility_huber",
                            "asymmetricutility_cara")
        if str((objective_config or {}).get("Object", "")).lower() not in _utility_objects:
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires a utility Objective.Object — one of "
                "'AsymmetricUtility_Symlog' | 'AsymmetricUtility_Huber' | 'AsymmetricUtility_CARA'. "
                "The DP recursion lives in utility space: an identity (legacy) objective leaves "
                "V-hat unbounded in dollars and the backward sweep blows up multiplicatively.")
    names = {
        "tradables": tuple(normalized_tradables.keys()),
        "hedges": hedge_names,
        "cash_accounts": cash_account_names,
        # The tradable hedge instruments (cash accounts excluded) are the action set; the
        # differential-ML solver builds its own action grid over them.
        "action_instruments": hedge_names,
        "liabilities": tuple(liabilities.keys()),
    }
    history_lookback = int(hedging_problem.get("History_Lookback_Business_Days", 30))
    if history_lookback < 0:
        raise ValueError("Hedging_Problem.History_Lookback_Business_Days must be non-negative")
    referenced_commodities = _collect_referenced_commodities(stoch_factors)
    runtime = {
        "execution_mode": execution_mode,
        "accounting_mode": accounting_mode,
        "names": names,
        "referenced_commodities": referenced_commodities,
        "tradables": normalized_tradables,
        "liabilities": liabilities,
        "objective": _normalize_objective_config(objective_config),
        "policy": None,
        "optimizer": None,
        "solver": _normalize_solver_config(solver_config),
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
            # Precomputed instrument→cash_account routing by currency match. Avoids
            # per-step rebuild in env loops; static for the lifetime of the runtime.
            "instrument_to_cash_account": _build_instrument_cash_account_map(
                normalized_tradables, cash_account_names),
            "transaction_cost_per_unit": float(evaluator_config.get("Transaction_Cost_Per_Unit", 0.0)),
            "bid_offer_spread_bps": float(evaluator_config.get("Bid_Offer_Spread_Bps", 0.0)),
            "force_flat_at_end": bool(evaluator_config.get("Force_Flat_At_End", True)),
            "total_position_abs_limit": float(evaluator_config.get("Total_Position_Abs_Limit", 0.0)),
        },
    }
    # Sanity check: Position_Bounds_Penalty is silently no-op'd when Total_Position_Abs_Limit
    # is unset, since the penalty's only active branch is the portfolio-total ramp. (Per-instrument
    # enforcement is a separate coefficient: Per_Instrument_Bounds_Penalty.)
    if (runtime["objective"]["position_bounds_penalty"] > 0.0
            and runtime["accounting"]["total_position_abs_limit"] <= 0.0):
        raise ValueError(
            "Objective.Position_Bounds_Penalty > 0 requires "
            "Evaluator.Total_Position_Abs_Limit > 0 — otherwise the penalty is silently disabled."
        )
    # Sanity check: Per_Instrument_Bounds_Penalty needs per-instrument Position_Limits; without
    # them the per-instrument penalty is silently no-op'd (every instrument has [-∞, +∞] bounds
    # so violations are never registered).
    if (runtime["objective"]["per_instrument_bounds_penalty"] > 0.0
            and not runtime["accounting"]["position_limits"]):
        raise ValueError(
            "Objective.Per_Instrument_Bounds_Penalty > 0 requires Evaluator.Position_Limits "
            "to be set on at least one instrument — otherwise the per-instrument penalty is "
            "silently disabled."
        )
    # Slim instrument metadata for the expiry-holding penalty: {name, last_trade_date} per hedge.
    runtime["instrument_meta"] = tuple(
        {"name": str(name), "last_trade_date": runtime["tradables"][name]["last_trade_date"]}
        for name in runtime["names"]["hedges"]
    )
    runtime["privileged_layout"] = derive_privileged_layout(stoch_factors)
    return runtime
