"""Normalized TorchRL hedging runtime construction.

This module owns the canonical TorchRL runtime contracts:

- runtime dict normalization from JSON/job spec
- canonical state flattening layout

Execution, rollout, and optimization live in torchrl_hedge.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from . import utils
from .hedge_features import build_entity_layout, derive_privileged_layout


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
    artifact_path = policy_config.get("Artifact_Path")
    return {
        "object": str(policy_config["Object"]),
        "action_space": action_space,
        "model": {
            "object": str(model_config.get("Object", "EntityTransformer")),
            "token_dim": int(model_config.get("Token_Dim", 64)),
            "emb_dim": int(model_config.get("Emb_Dim", 8)),
            "n_heads": int(model_config.get("N_Heads", 4)),
            "n_layers": int(model_config.get("N_Layers", 2)),
        },
        "artifact_path": str(artifact_path) if artifact_path else None,
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
        # Per-business-day discounts; effective per-decision values are gamma**interval, lam**interval
        "gamma": float(optimizer_config.get("Gamma", 1.0)),
        "gae_lambda": float(optimizer_config.get("GAE_Lambda", 0.995)),
        "learning_rate": float(optimizer_config.get("Learning_Rate", 3.0e-4)),
        "lr_schedule": str(optimizer_config.get("LR_Schedule", "constant")).lower(),
        "lr_min": float(optimizer_config.get("LR_Min", 0.0)),
        "lr_warmup_epochs": int(optimizer_config.get("LR_Warmup_Epochs", 0)),
        "clip_eps": float(optimizer_config.get("Clip_Eps", 0.2)),
        "value_coef": float(optimizer_config.get("Value_Coef", 0.5)),
        "entropy_coef": float(optimizer_config.get("Entropy_Coef", 0.01)),
        "entropy_schedule": str(optimizer_config.get("Entropy_Schedule", "constant")).lower(),
        "entropy_coef_min": float(optimizer_config.get("Entropy_Coef_Min", 0.0)),
        "max_grad_norm": float(optimizer_config.get("Max_Grad_Norm", 0.5)),
        "reward_scale": float(optimizer_config.get("Reward_Scale", 1.0)),
        "debug_strict_bins": bool(optimizer_config.get("Debug_Strict_Bins", False)),
        "dense_tracking_reward_scale": float(optimizer_config.get("Dense_Tracking_Reward_Scale", 0.0)),
        "dense_tracking_reward_clip": float(optimizer_config.get("Dense_Tracking_Reward_Clip", 0.0)),
        "dense_reward_mode": str(optimizer_config.get("Dense_Reward_Mode", "asymmetric")).lower(),
        "validation_fraction": float(optimizer_config.get("Validation_Fraction", 0.25)),
        "validation_min_batch": int(optimizer_config.get("Validation_Min_Batch", 256)),
        "decision_interval_curriculum": tuple(decision_interval_curriculum),
        "cvar_alpha": float(optimizer_config.get("CVaR_Alpha", 0.0)),
        "cvar_lambda": float(optimizer_config.get("CVaR_Lambda", 0.0)),
        "value_loss_asym_weight": float(optimizer_config.get("Value_Loss_Asym_Weight", 1.0)),
        "entropy_floor_h_min": float(optimizer_config.get("Entropy_Floor_H_Min", 0.0)),
        "entropy_floor_coef": float(optimizer_config.get("Entropy_Floor_Coef", 0.0)),
        "seed": optimizer_config.get("Seed"),
        # Optional path for atomic per-epoch diag dump. None disables. Reader (a separate
        # script) can tail this file for live progress without parsing buffered stdout.
        "live_diag_path": optimizer_config.get("Live_Diag_Path"),
    }


def _normalize_solver_config(solver_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize the `Solver` config block (Execution_Mode='solve_hedge'). Mirrors
    `_normalize_optimizer_config`: accepts None (non-solve modes), requires `Object`."""
    if solver_config is None:
        return None
    if "Object" not in solver_config:
        raise ValueError("Hedging_Problem['Solver'] requires an 'Object' field")
    value_fn = dict(solver_config.get("Value_Fn", {}))
    return {
        "object": str(solver_config["Object"]).lower(),
        "training_action_grid_levels_per_axis":
            int(solver_config.get("Training_Action_Grid_Levels_Per_Axis", 11)),
        "training_action_chunk_size": int(solver_config.get("Training_Action_Chunk_Size", 64)),
        "decision_action_search": str(solver_config.get("Decision_Action_Search", "fine_grid")).lower(),
        "decision_action_grid_levels_per_axis":
            int(solver_config.get("Decision_Action_Grid_Levels_Per_Axis", 51)),
        "turnover_cost_padding_multiplier":
            float(solver_config.get("Turnover_Cost_Padding_Multiplier", 1.0)),
        "range_projection_alpha":
            float(solver_config.get("Range_Projection_Alpha", 1.0e-3)),
        # λ-return blend between one-step bootstrap (λ=0, the legacy LSM-DP target)
        # and a buy-and-hold value-only rollout to T_dec (λ=1, removes V̂ from the
        # target entirely). See differential_ml_redesign_v14.md §2.2.
        "lambda_return": float(solver_config.get("Lambda_Return", 0.0)),
        # Differential-ML twin-loss weight (Huge–Savine): stacked least-squares
        # `‖v - X·β‖² + λ_diff · ‖v_grad - X_grad·β‖²` in V̂.fit, with X_grad the
        # Jacobian of the OLS basis w.r.t. the differentiable deep-state columns and
        # v_grad the pathwise gradient of the cross-fit target through the AAD inner-MC.
        # 0 disables the differential branch (bit-exact baseline). See
        # differential_ml_redesign_v14.md §2.3.
        "lambda_diff": float(solver_config.get("Lambda_Diff", 0.0)),
        # λ-mix (DifferentialSolver advantage decomp): blend
        # `(1-λ)·Y_boot + λ·Y_rollout` for the value label. Spec §14 deferred lever,
        # indicated when a horizon-stable bounded residual V_0 gap survives advantage
        # decomposition. Default 0 = pure bootstrap (banked behavior).
        "lambda_mix": float(solver_config.get("Lambda_Mix", 0.0)),
        "use_advantage_decomp": bool(solver_config.get("Use_Advantage_Decomp", True)),
        # Backward-sweep depth: fit C_t for t in [t_outer-2 .. t_min]. 0 = full sweep
        # to the initial decision; t_min near t_outer-1 = a shallow (bounded) sweep.
        "t_min": int(solver_config.get("T_Min", 0)),
        "audit_max_rounds": int(solver_config.get("Audit_Max_Rounds", 3)),
        # --- BSS validation sandwich (DifferentialSolver offline diagnostics) ---
        # Penalty π zero-mean GATE (step 1+2): interior |mean ΣΔ|/stderr threshold for
        # dual feasibility, and the cap on the data-driven terminal boundary block U zeros π on.
        "penalty_zero_mean_z": float(solver_config.get("Penalty_ZeroMean_Z", 3.0)),
        "penalty_max_boundary": int(solver_config.get("Penalty_Max_Boundary", 4)),
        # Penalized clairvoyant UPPER bound U (step 3) — the wealth-grid backward DP. Cost
        # ∝ P·G·K²·I per step, so it is OPT-IN (like Run_Hindsight_Diagnostic): off by default,
        # enable for the validation readout. The cheap single-trajectory penalty gate above
        # always runs. The caps below scope it (shallow/few-live first; raise for tightness).
        "run_upper_bound": bool(solver_config.get("Run_Upper_Bound", False)),
        "upper_bound_max_paths": int(solver_config.get("Upper_Bound_Max_Paths", 128)),
        "upper_bound_wealth_grid": int(solver_config.get("Upper_Bound_Wealth_Grid", 41)),
        "upper_bound_n_inner": int(solver_config.get("Upper_Bound_N_Inner", 4)),
        "upper_bound_grid_pad": float(solver_config.get("Upper_Bound_Grid_Pad", 1.0)),
        "upper_bound_chunk_rows": int(solver_config.get("Upper_Bound_Chunk_Rows", 200_000)),
        "upper_bound_clamp_warn_frac":
            float(solver_config.get("Upper_Bound_Clamp_Warn_Frac", 0.02)),
        # C-stack persistence for OUT-OF-SAMPLE validation (DifferentialSolver). Save_* writes
        # the fitted twin-net stack after the sweep; Load_* loads it, skips training, and runs
        # the L/π/U sandwich on a fresh-seeded batch. Two config variants (train→save, eval→load
        # + new Random_Seed), never both in one run.
        "save_value_fn_path": solver_config.get("Save_Value_Fn_Path") or None,
        "load_value_fn_path": solver_config.get("Load_Value_Fn_Path") or None,
        # Offline gate4 diagnostic: per-depth action-match of the fitted policy vs an exact-DP
        # oracle (gate2_exact_dp.npz). Gated path; toy-only; never shipped.
        "oracle_action_match_path": solver_config.get("Oracle_Action_Match_Path") or None,
        # Opt-in label audit: timesteps at which to snapshot the bootstrap labels the net fits
        # (Y_boot / baseline_B / residual). Diagnostic; empty list = off.
        "label_audit_t_steps": list(solver_config.get("Label_Audit_T_Steps", []) or []),
        # Endogenous-span bank knob (DifferentialSolver): inventory/wealth replicas
        # layered on each exogenous slice.
        "bank_sampling": {
            "b_endo": int((solver_config.get("Bank_Sampling") or {}).get("B_Endo", 2)),
        },
        "include_dynamic_features_in_value_inputs":
            bool(solver_config.get("Include_Dynamic_Features_In_Value_Inputs", False)),
        "multi_seed_count": int(solver_config.get("Multi_Seed_Count", 1)),
        "run_hindsight_diagnostic": bool(solver_config.get("Run_Hindsight_Diagnostic", False)),
        "run_mpc_comparison": bool(solver_config.get("Run_Mpc_Comparison", False)),
        "run_textbook_benchmark": bool(solver_config.get("Run_Textbook_Benchmark", False)),
        "textbook_allocation": str(solver_config.get("Textbook_Allocation", "maturity_matched")).lower(),
        "value_fn": {
            "ols_refit_cadence_daily_solves": int(value_fn.get("OLS_Refit_Cadence_Daily_Solves", 5)),
            "buffer_max_age_daily_solves": int(value_fn.get("Buffer_Max_Age_Daily_Solves", 60)),
            "mlp_hidden": list(value_fn.get("MLP_Hidden", [64, 64, 64])),
            "mlp_activation": str(value_fn.get("MLP_Activation", "gelu")).lower(),
            "mlp_train_steps_per_solve": int(value_fn.get("MLP_Train_Steps_Per_Solve", 200)),
            "mlp_loss_tol": float(value_fn.get("MLP_Loss_Tol", 0.0)),
            "mlp_adam_lr": float(value_fn.get("MLP_Adam_LR", 1.0e-3)),
            "mlp_final_init_scale": float(value_fn.get("MLP_Final_Init_Scale", 0.0)),
            # Tail-saturating columns: deep-state indices to push through tanh after
            # standardization, before the OLS basis. Bounds the V̂ input on load-bearing
            # breach columns (outer-MC training covers ~σ, inner-MC queries explore tails
            # to many σ where the linear basis has no support). Empty = off.
            "tail_saturating_columns": [int(c) for c in value_fn.get("Tail_Saturating_Columns", [])],
            # Steepness multiplier on the standardized input before tanh. scale > 1
            # tightens the saturation knee inward in raw units; scale = 1 keeps the knee
            # at the 1σ point of the training distribution.
            "tail_saturation_scale": float(value_fn.get("Tail_Saturation_Scale", 1.0)),
        },
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
        # Per-instrument [Min_Position, Max_Position] bounds are HARD-enforced upstream
        # by `StructuredRebalancePolicy._feasible_mask` (logit -∞ on out-of-range bins),
        # so a per-instrument soft term would always be zero. The naming pre-dates that
        # clarification; behavior is portfolio-only. Rename to `Total_Position_Abs_Penalty`
        # if it becomes worth the JSON / CSV / docs churn.
        "position_bounds_penalty": float(objective_config.get("Position_Bounds_Penalty", 0.0)),
        "position_bounds_threshold": float(objective_config.get("Position_Bounds_Threshold", 5.0)),
        # Per-instrument [Min_Position, Max_Position] enforcement (soft, reward-side).
        # Replaces the prior hard mask in `StructuredRebalancePolicy._feasible_mask`.
        "per_instrument_bounds_penalty": float(objective_config.get("Per_Instrument_Bounds_Penalty", 0.0)),
        "per_instrument_bounds_threshold": float(objective_config.get("Per_Instrument_Bounds_Threshold", 5.0)),
        # Utility-transform scale. Consumed by any utility Object (Symlog / Huber / CARA);
        # legacy "TerminalFloorThenSurplusUtility" path ignores it. `utility_scale` is mirrored
        # from `bundle["utility_scale"]` at rollout start (see _collect_ppo_rollout).
        "utility_scale_mode": str(objective_config.get("Utility_Scale_Mode", "vol_scaled_notional")).lower(),
        "utility_scale_explicit": None if explicit is None else float(explicit),
        # Utility SHAPE params (DIMENSIONLESS, in units of the scale c — applied to x = W/c).
        # Huber (AsymmetricUtility_Huber): linear gains, quadratic small losses with curvature
        # `huber_aversion`, linear deep tail beyond the knee `huber_delta`. CARA
        # (AsymmetricUtility_CARA): u = (1−e^{−γx})/γ with risk aversion `cara_gamma`. Symlog
        # ignores all three. See torchrl_hedge._utility_wrap_signed for the exact forms.
        "huber_aversion": float(objective_config.get("Huber_Aversion", 2.5)),
        "huber_delta": float(objective_config.get("Huber_Delta", 1.0)),
        "cara_gamma": float(objective_config.get("CARA_Gamma", 1.0)),
    }


def _normalize_action_space(policy_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize the JSON action-space config into a runtime dict. Accepts either:

      - `Trade_Deltas`: list-of-list of allowed integer deltas per instrument (preferred).
        Allows non-uniform spacings (e.g. fine bins near 0 + coarse bins at extremes).
      - `Min_Trade_Delta` + `Max_Trade_Delta`: legacy unit-step integer ranges; expanded
        to range(min, max+1) for each instrument.

    The runtime dict carries both `trade_deltas` (per-instrument tuple of ints) and the
    derived `min_trade_delta` / `max_trade_delta` (per-instrument int) for backward compat
    with downstream consumers that only need the bounds.
    """
    action_space = dict(policy_config.get("Action_Space", {}))
    instrument_order = tuple(str(name) for name in action_space.get("Instrument_Order", ()))
    explicit_deltas = action_space.get("Trade_Deltas")

    if explicit_deltas is not None:
        per_instrument = tuple(
            tuple(sorted(set(int(d) for d in deltas)))
            for deltas in explicit_deltas
        )
        if instrument_order and len(instrument_order) != len(per_instrument):
            raise ValueError(
                "Trade_Deltas length must match Instrument_Order length; "
                f"got {len(per_instrument)} delta-lists for {len(instrument_order)} instruments")
    else:
        import warnings
        warnings.warn(
            "Action_Space.Min_Trade_Delta/Max_Trade_Delta is deprecated; use "
            "Action_Space.Trade_Deltas (per-instrument list of allowed integer deltas). "
            "The min/max form expands to a uniform unit-step range and is scheduled for "
            "removal — non-uniform spacings (e.g. fine bins near 0, coarse at extremes) "
            "are now standard.",
            DeprecationWarning, stacklevel=2,
        )
        min_trade_delta = tuple(int(value) for value in action_space.get("Min_Trade_Delta", ()))
        max_trade_delta = tuple(int(value) for value in action_space.get("Max_Trade_Delta", ()))
        if instrument_order and not (
            len(instrument_order) == len(min_trade_delta) == len(max_trade_delta)
        ):
            raise ValueError("Structured action space entries must have matching lengths")
        per_instrument = tuple(
            tuple(range(int(lo), int(hi) + 1))
            for lo, hi in zip(min_trade_delta, max_trade_delta))

    return {
        "instrument_order": instrument_order,
        "trade_deltas": {
            instrument_name: deltas
            for instrument_name, deltas in zip(instrument_order, per_instrument)
        },
        "min_trade_delta": {
            instrument_name: min(deltas)
            for instrument_name, deltas in zip(instrument_order, per_instrument)
        },
        "max_trade_delta": {
            instrument_name: max(deltas)
            for instrument_name, deltas in zip(instrument_order, per_instrument)
        },
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


def construct_torchrl_runtime(
    config: Mapping[str, Any],
    stoch_factors: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    config = _unwrap_calc_config(config)
    hedging_problem = config["Hedging_Problem"]
    evaluator_config = hedging_problem["Evaluator"]
    liabilities = _normalize_liabilities(hedging_problem)
    objective_config = hedging_problem.get("Objective")
    policy_config = hedging_problem.get("Policy")
    optimizer_config = hedging_problem.get("Optimizer")
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
    if execution_mode == "optimize_policy" and optimizer_config is None:
        raise ValueError("Execution_Mode 'optimize_policy' requires Hedging_Problem['Optimizer']")
    if execution_mode == "solve_hedge":
        if solver_config is None:
            raise ValueError("Execution_Mode 'solve_hedge' requires Hedging_Problem['Solver']")
        if str(config.get("Inner_MC_Enabled", "No")) != "Yes":
            raise ValueError("Execution_Mode 'solve_hedge' requires Inner_MC_Enabled='Yes'")
        solver_object = str(solver_config.get("Object", "")).lower()
        min_inner = 2 if solver_object == "differentialsolver" else 128
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
    # Policy is optional under solve_hedge — the DP/MPC solver replaces the RL policy track.
    policy = _normalize_policy_config(policy_config) if policy_config is not None else None
    names = {
        "tradables": tuple(normalized_tradables.keys()),
        "hedges": hedge_names,
        "cash_accounts": cash_account_names,
        "action_instruments": (tuple(policy["action_space"]["instrument_order"])
                               if policy is not None else hedge_names),
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
        "policy": policy,
        "optimizer": _normalize_optimizer_config(optimizer_config),
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
    runtime["entity_layout"] = build_entity_layout(runtime)
    runtime["privileged_layout"] = derive_privileged_layout(stoch_factors)
    return runtime


def construct_hedging_runtime(config: Mapping[str, Any]) -> Dict[str, Any]:
    return construct_torchrl_runtime(config)
