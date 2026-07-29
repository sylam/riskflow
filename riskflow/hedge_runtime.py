"""The ONE JSON → hedging-runtime boundary.

`construct_hedge_runtime` reads the `Hedging_Problem` JSON block and returns the normalized
runtime dict every hedging consumer indexes by key: canonical lowercased modes, the instrument /
cash-account / hedge name sets, per-instrument metadata, the accounting rules (position limits,
turnover cost, spreads, margin funding, corridor), the objective, the solver config and the
portfolio state. Everything is validated HERE and nowhere else — past this boundary the runtime is
the contract, so downstream code indexes it directly rather than re-checking it.

Also owns the privileged-factor naming convention (what each stochastic process publishes as
market state) and `per_contract_kappa`, the single turnover-cost rule the solver, the environment
and the diagnostic CSV writer all price frictions through.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import torch

from . import utils


def _privileged_name(factor_name, attr_name, stoch_factors):
    """How a process's published state coordinate is keyed. Multi-commodity runs (more than one
    distinct primary factor name) prefix the attribute with `<factor>_` to disambiguate."""
    multi = len({f.name[0] for f in (stoch_factors or {})}) > 1
    return f'{factor_name.lower()}_{attr_name}' if multi else attr_name


def derive_privileged_layout(stoch_factors):
    """Build the {name: dim} schema by asking each live stoch-factor process what it emits.
    Polymorphic via `type(process).privileged_layout(process.param)` — adding a new
    StochasticProcess subclass with its own privileged surface flows through automatically."""
    layout = {}
    for factor, process in (stoch_factors or {}).items():
        for attr_name, dim in type(process).privileged_layout(process.param).items():
            layout[_privileged_name(factor.name[0], attr_name, stoch_factors)] = int(dim)
    return layout


def assemble_privileged_factors(privileged_factor_blocks, stoch_factors):
    """Concatenate per-batch privileged-factor tensors collected during the simulation loop into
    a single dict ready for the bundle. Input keyed by (factor_name, attr_name); output keys match
    the schema produced by `derive_privileged_layout`."""
    return {
        _privileged_name(factor_name, attr_name, stoch_factors): torch.cat(blocks, dim=1)
        for (factor_name, attr_name), blocks in privileged_factor_blocks.items()
    }


def privileged_block(privileged_factors, stoch_factors, attr_name):
    """`(factor, process, block)` for the first live factor that PUBLISHES `attr_name` in its
    privileged layout and has a matching assembled block — the read side of the naming rule above
    (e.g. GARCH's revealed `log_h`). `(None, None, None)` when no process exposes it."""
    for factor, process in (stoch_factors or {}).items():
        if attr_name not in type(process).privileged_layout(process.param):
            continue
        block = privileged_factors.get(_privileged_name(factor.name[0], attr_name, stoch_factors))
        if block is not None:
            return factor, process, block
    return None, None, None


def _flatten_deals(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """`{DealType: {name: params}}` (the JSON object-map form) → `{name: {deal_type, params}}`."""
    return {str(name): {"deal_type": str(deal_type), "params": deepcopy(dict(params))}
            for deal_type, deals in config.items() for name, params in deals.items()}


def _instrument_metadata(name, entry, *, hedge_names, cash_account_names, liability_expiry):
    """Per-tradable metadata: the expiry / last-trade dates the pricers and the expiry mask read,
    the routing flags, and the contract size. Dates fall back Expiry_Date → Maturity_Date →
    Investment_Horizon → the latest liability expiry."""
    params = entry["params"]
    investment_horizon = params.get("Investment_Horizon")
    maturity_date = params.get("Maturity_Date")
    fallback = investment_horizon if investment_horizon is not None else liability_expiry
    default = maturity_date if maturity_date is not None else fallback
    return {
        "name": str(name),
        "deal_type": entry["deal_type"],
        "is_hedge": name in hedge_names,
        "is_cash_account": name in cash_account_names,
        "currency": params.get("Currency"),
        "last_trade_date": params.get("Last_Trade_Date", default),
        "expiry_date": params.get("Expiry_Date", default),
        "first_notice_date": params.get("First_Notice_Date"),
        "auto_close_days_before_last_trade": int(params.get("Auto_Close_Days_Before_Last_Trade", 0)),
        "allow_new_positions_until_last_trade":
            params.get("Allow_New_Positions_Until_Last_Trade", "Yes") == "Yes",
        "allow_holding_past_last_trade": params.get("Allow_Holding_Past_Last_Trade", "No") == "Yes",
        "contract_size": float(params.get("Contract_Size", 1.0)),
        "params": deepcopy(dict(params)),
    }


def _bid_offer_spread(evaluator_config: Mapping[str, Any]):
    """`Evaluator.Bid_Offer_Spread_Bps` is EITHER a scalar half-spread bps applied to every
    instrument (the fast path) OR a spec dict for maturity/liquidity- and volatility-dependent
    spreads:

        {"Default_Bps": d,
         "Per_Instrument": {name: base_bps, ...},
         "Vol_Scale": {"Ref_Vol": r, "Beta": b}}

    The effective half-spread for instrument `name` at annualized vol σ_t is
    `base_bps[name] · (σ_t/Ref_Vol)**Beta`, where `base_bps[name]` falls back to `Default_Bps`
    (Per_Instrument absent ⇒ Default_Bps for all) and the vol factor is 1 when Vol_Scale is
    absent, Beta==0, or σ_t is unknown. Returns `(scalar_bps, spec)`; `spec` is None in the
    scalar case, and `scalar_bps` (the Default_Bps in the dict case) also feeds the diagnostic
    CSV display column + the scalar fast-path in `per_contract_kappa`."""
    raw = evaluator_config.get("Bid_Offer_Spread_Bps", 0.0)
    if not isinstance(raw, Mapping):
        return float(raw), None
    default_bps = float(raw.get("Default_Bps", 0.0))
    vs = raw.get("Vol_Scale") or {}
    return default_bps, {
        "default_bps": default_bps,
        "per_instrument": {str(k): float(v) for k, v in (raw.get("Per_Instrument") or {}).items()},
        "vol_scale": ({"ref_vol": float(vs["Ref_Vol"]), "beta": float(vs.get("Beta", 0.0))}
                      if vs else None)}


def _position_schedule(evaluator_config: Mapping[str, Any]):
    """Optional per-decision-step corridor on the SIGNED total position Σq_i. A list of
    `{Step, Min_Total, Max_Total}` knots (piecewise-constant between knots): at sim-grid
    decision step t the signed book total must lie within [Min_Total, Max_Total] of the
    rightmost knot with `Step <= t`. Absent → None (no corridor). Returns a sorted tuple of
    `(step, min_total, max_total)` with strictly ascending, non-negative steps and
    Min_Total <= Max_Total per knot."""
    raw = evaluator_config.get("Total_Position_Schedule")
    if not raw:
        return None
    knots = sorted(
        (int(k["Step"]), float(k["Min_Total"]), float(k["Max_Total"])) for k in raw)
    if knots[0][0] < 0:
        raise ValueError(
            f"Total_Position_Schedule Step must be >= 0; got {knots[0][0]}")
    for (a, _, _), (b, _, _) in zip(knots, knots[1:]):
        if b <= a:
            raise ValueError(
                f"Total_Position_Schedule Steps must be strictly ascending; got {a} >= {b}")
    for step, lo, hi in knots:
        if lo > hi:
            raise ValueError(
                f"Total_Position_Schedule knot at Step {step}: Min_Total {lo} > Max_Total {hi}")
    return tuple(knots)


def _spot_price_history(hedging_problem: Mapping[str, Any], lookback: int,
                        referenced_commodities: tuple) -> Dict[str, Dict[str, Any]]:
    """Realized spot history per commodity — the rolling-feature lookback the bundle prepends.
    OPTIONAL: absent it the utility scale falls back to the calibrated market data and the prefix
    no-ops, so an empty history is returned rather than demanding an entry per commodity. A
    PARTIAL history (some but not all referenced commodities) IS an error, as are ragged
    dates/prices, a series shorter than the lookback, non-ascending dates, and two commodities
    whose date axes disagree."""
    raw_history = (hedging_problem.get("Portfolio_State") or {}).get("Spot_Price_History") or {}
    if not raw_history:
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for commodity, payload in raw_history.items():
        name = str(commodity)
        dates_raw = payload.get("Dates", ())
        prices_raw = payload.get("Prices", ())
        if len(dates_raw) != len(prices_raw):
            raise ValueError(
                f"Spot_Price_History['{name}']: Dates and Prices must have equal length "
                f"({len(dates_raw)} vs {len(prices_raw)})")
        if len(dates_raw) < lookback:
            raise ValueError(
                f"Spot_Price_History['{name}']: needs at least "
                f"History_Lookback_Business_Days={lookback} entries, got {len(dates_raw)}")
        dates = tuple(dates_raw)
        for i in range(1, len(dates)):
            if dates[i] <= dates[i - 1]:
                raise ValueError(
                    f"Spot_Price_History['{name}']: Dates must be strictly ascending; "
                    f"found {dates[i - 1]} >= {dates[i]} at index {i}")
        normalized[name] = {"dates": dates, "prices": tuple(float(p) for p in prices_raw)}
    missing = tuple(c for c in referenced_commodities if c not in normalized)
    if missing:
        raise ValueError(
            f"Spot_Price_History missing entries for referenced commodities: {missing}")
    names = list(normalized)
    for other in names[1:]:
        if normalized[other]["dates"] != normalized[names[0]]["dates"]:
            raise ValueError(
                f"Spot_Price_History['{other}'].Dates must match "
                f"Spot_Price_History['{names[0]}'].Dates exactly")
    return normalized


def _solver_config(solver_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize the `Solver` block (Execution_Mode='solve_hedge'). Accepts None (non-solve
    modes); requires `Object` — one of 'diffsolverv2' | 'hindsightdpsolver'."""
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
        "use_advantage_decomp": solver_config.get("Use_Advantage_Decomp", "Yes") == "Yes",
        # --- DiffSolverV2 (clean-room differential-ML solver) knobs ---
        # Per-t residual-net Adam iters / lr; bank q-exploration noise as a fraction of each
        # instrument's [Min,Max] range; the subset of hedge instruments whose action axis VARIES
        # in the grid (others pinned to 0). None = all vary.
        "diffv2_fit_iters": int(solver_config.get("DiffV2_Fit_Iters", 150)),
        "diffv2_lr": float(solver_config.get("DiffV2_LR", 2.0e-3)),
        "diffv2_bank_noise_frac": float(solver_config.get("DiffV2_Bank_Noise_Frac", 0.15)),
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
        # Cost-aware EXECUTION: the verdict rollout charges the L1 repositioning cost
        # (Transaction_Cost_Per_Unit + half Bid_Offer_Spread_Bps) at the argmax, trading
        # expected value against the cost of getting there. Training stays cost-free.
        "diffv2_cost_aware_argmax":
            solver_config.get("DiffV2_Cost_Aware_Argmax", "No") == "Yes",
        # Deployment-faithful backtest: with a frozen policy loaded, roll it day-by-day on the
        # observed path via BundleStepper (real futures accounting; decisions off the stepper's
        # own wealth). Exposes diagnostics['stepper_verdict']. 'No' = only the fast _verdict.
        "diffv2_stepper_rollout":
            solver_config.get("DiffV2_Stepper_Rollout", "No") == "Yes",
        # Twin-loss differential normalization: Huge-Savine's official implementation
        # normalizes greeks PER INPUT COLUMN (lambda_j vector) — validated +0.01-0.017 u
        # on every 8k seed vs the pooled scalar. 'No' = legacy pooled variance (one
        # fat-tailed column deflates the constraint for all columns).
        "diffv2_per_column_grad_norm":
            solver_config.get("DiffV2_Per_Column_Grad_Norm", "Yes") == "Yes",
        # Value-function persistence: save the fitted nets (+ standardization stats + utility
        # scale) after the backward sweep, or load them and SKIP training — a frozen-policy
        # eval, e.g. OOD stress gates (train under the calibrated world, evaluate the frozen
        # policy under a stressed one). Load accepts a LIST of checkpoint paths for an
        # ENSEMBLE-argmax eval: each member evaluated in its own standardization frame, the
        # continuations averaged before the argmax (cross-fit winner's-curse reduction).
        # Train and evaluate are SEPARATE runs: loading skips every fit step under streaming too
        # (`DiffSolverV2.step` no-ops), so a frozen policy stays frozen batch after batch, and
        # setting both keys at once raises rather than silently discarding a retrained net.
        "diffv2_save_value_fn": str(solver_config.get("DiffV2_Save_Value_Fn", "") or ""),
        "diffv2_load_value_fn":
            ([str(p) for p in solver_config["DiffV2_Load_Value_Fn"]]
             if isinstance(solver_config.get("DiffV2_Load_Value_Fn"), (list, tuple))
             else str(solver_config.get("DiffV2_Load_Value_Fn", "") or "")),
        "active_hedge_indices":
            (list(solver_config["Active_Hedge_Indices"])
             if solver_config.get("Active_Hedge_Indices") is not None else None),
        # Benchmark tracks assembled alongside the DiffSolverV2 deliverable (hindsight upper
        # bound / textbook lower bound).
        "run_hindsight_diagnostic": solver_config.get("Run_Hindsight_Diagnostic", "No") == "Yes",
        "run_textbook_benchmark": solver_config.get("Run_Textbook_Benchmark", "No") == "Yes",
    }


# Utility Objectives — the DP / value function lives in utility space and needs the scale c.
_UTILITY_OBJECTS = ("asymmetricutility_symlog", "asymmetricutility_huber", "asymmetricutility_cara")


def construct_hedge_runtime(
    config: Mapping[str, Any],
    stoch_factors: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """The JSON → runtime boundary: read `Hedging_Problem`, validate it, and return the runtime
    dict every consumer indexes directly. Nothing downstream re-validates."""
    config = config if "Hedging_Problem" in config else config["Calc"]["Calculation"]
    hedging_problem = config["Hedging_Problem"]
    evaluator_config = hedging_problem["Evaluator"]
    objective_config = hedging_problem.get("Objective")
    solver_config = hedging_problem.get("Solver")
    execution_mode = str(config.get("Execution_Mode", "simulate_only")).lower()
    if execution_mode not in ("solve_hedge", "simulate_only"):
        raise ValueError(
            f"Unknown Execution_Mode {config.get('Execution_Mode')!r}; supported: 'solve_hedge' | "
            "'simulate_only'.")

    # --- instruments: the tradable universe splits into cash accounts and hedge legs ---
    tradables = _flatten_deals(hedging_problem["Tradable_Instruments"])
    if evaluator_config.get("Cash_Instruments") is not None:
        cash_account_names = tuple(str(n) for n in evaluator_config["Cash_Instruments"])
    elif evaluator_config.get("Cash_Accounts") is not None:
        cash_account_names = tuple(str(n) for n in evaluator_config["Cash_Accounts"])
    elif evaluator_config.get("Cash_Instrument") is not None:
        cash_account_names = (str(evaluator_config["Cash_Instrument"]),)
    else:
        cash_account_names = ()
    for account_name in cash_account_names:
        if account_name not in tradables:
            raise ValueError(
                f"Evaluator cash account '{account_name}' is not in Tradable_Instruments")
    hedge_names = tuple(n for n in tradables if n not in cash_account_names)
    if not hedge_names:
        raise ValueError("no hedge instruments: Tradable_Instruments has only cash accounts")

    # --- liabilities: the book being hedged; its latest expiry dates the hedge instruments ---
    liabilities = {}
    for name, entry in _flatten_deals(hedging_problem.get("Liabilities") or {}).items():
        params = entry["params"]
        liabilities[name] = {
            "reference": name, "object": entry["deal_type"], "deal_type": entry["deal_type"],
            "underlying": params.get("Underlying"), "currency": params.get("Currency"),
            "strike": float(params.get("Strike", params.get("Strike_Price", 0.0))),
            "quantity": float(params.get("Quantity", params.get("Units", 0.0))),
            "expiry_date": params.get("Expiry_Date"), "params": deepcopy(dict(params))}
    liability_expiry = None
    for liability in liabilities.values():
        expiry_date = liability["expiry_date"]
        if expiry_date is not None and (liability_expiry is None or expiry_date > liability_expiry):
            liability_expiry = expiry_date
    normalized_tradables = {
        name: _instrument_metadata(name, entry, hedge_names=hedge_names,
                                   cash_account_names=cash_account_names,
                                   liability_expiry=liability_expiry)
        for name, entry in tradables.items()}

    if execution_mode == "solve_hedge":
        if solver_config is None:
            raise ValueError("Execution_Mode 'solve_hedge' requires Hedging_Problem['Solver']")
        if str(config.get("Inner_MC_Enabled", "No")) != "Yes":
            raise ValueError("Execution_Mode 'solve_hedge' requires Inner_MC_Enabled='Yes'")
        min_inner = 2 if str(solver_config.get("Object", "")).lower() == "diffsolverv2" else 128
        if int(config.get("Inner_Sub_Batch", 0)) < min_inner:
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Inner_Sub_Batch >= "
                f"{min_inner} for Solver.Object={solver_config.get('Object')!r}")
        if str(solver_config.get("Object", "")).lower() != "diffsolverv2":
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Solver.Object='DiffSolverV2' (the "
                f"incremental warmup/step/finish API); got {solver_config.get('Object')!r}. "
                "HindsightDpSolver remains available as the Run_Hindsight_Diagnostic track.")
        # A solve is a STREAM: Simulation_Batches - 1 fit batches, then a held-out batch no fit
        # step saw. Two is the shortest honest stream. A loaded checkpoint fits nothing, so it is
        # a stream of one — its single batch is the held-out world.
        n_batches = int(config.get("Simulation_Batches", 1))
        if solver_config.get("DiffV2_Load_Value_Fn"):
            if n_batches != 1:
                raise ValueError(
                    "Execution_Mode 'solve_hedge' with DiffV2_Load_Value_Fn requires "
                    "Simulation_Batches == 1: a frozen policy fits nothing, so its one batch IS "
                    f"the held-out world; got {n_batches}.")
        elif n_batches < 2:
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Simulation_Batches >= 2 (fit batches, then "
                f"a held-out batch no fit step saw); got {n_batches}. Simulation_Batches is a path "
                "MULTIPLIER under 'simulate_only' and a STREAM LENGTH here, and riskflow_batch "
                "divides it by the job count before this check.")
        if solver_config.get("DiffV2_Load_Value_Fn") and solver_config.get("DiffV2_Save_Value_Fn"):
            raise ValueError(
                "Solver.DiffV2_Save_Value_Fn is set alongside DiffV2_Load_Value_Fn: a loaded "
                "checkpoint is a frozen-policy EVALUATION and fits nothing, so there is no new "
                "value fn to write. Train (save) and evaluate (load) are separate runs.")
        if str((objective_config or {}).get("Object", "")).lower() not in _UTILITY_OBJECTS:
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires a utility Objective.Object — one of "
                "'AsymmetricUtility_Symlog' | 'AsymmetricUtility_Huber' | 'AsymmetricUtility_CARA'. "
                "The DP recursion lives in utility space: an identity (legacy) objective leaves "
                "V-hat unbounded in dollars and the backward sweep blows up multiplicatively.")

    history_lookback = int(hedging_problem.get("History_Lookback_Business_Days", 30))
    if history_lookback < 0:
        raise ValueError("Hedging_Problem.History_Lookback_Business_Days must be non-negative")
    # Commodity names come from the live CommodityPrice factors the instruments created at
    # calc-dependency time — never re-parsed out of instrument JSON params.
    referenced_commodities = tuple(dict.fromkeys(
        utils.check_tuple_name(factor) for factor in (stoch_factors or {})
        if factor.type == 'CommodityPrice'))
    portfolio_state = hedging_problem.get("Portfolio_State") or {}
    scalar_spread_bps, spread_spec = _bid_offer_spread(evaluator_config)
    # Static instrument→cash_account routing by currency: the first cash account whose currency
    # matches wins, else the first account. Computed once; the env step loop reads it per step.
    account_by_currency = {}
    for account_name in cash_account_names:
        account_by_currency.setdefault(
            normalized_tradables[account_name]["currency"], account_name)
    fallback_account = cash_account_names[0] if cash_account_names else None

    return {
        "execution_mode": execution_mode,
        "accounting_mode": str(evaluator_config.get("Accounting_Mode", "futures")).lower(),
        "names": {
            "tradables": tuple(normalized_tradables),
            "hedges": hedge_names,
            "cash_accounts": cash_account_names,
            # The hedge legs ARE the action set; the solver builds its action grid over them.
            "action_instruments": hedge_names,
            "liabilities": tuple(liabilities),
        },
        "referenced_commodities": referenced_commodities,
        "tradables": normalized_tradables,
        "liabilities": liabilities,
        "objective": None if objective_config is None else {
            # Canonical lowercased form — every dispatch site compares against the lowercase
            # literal, so normalize once here rather than re-lowercasing on every reward call.
            "object": str(objective_config["Object"]).lower(),
            # Utility-transform scale. Consumed by any utility Object (Symlog / Huber / CARA);
            # the identity path ignores it. `utility_scale` is mirrored in from the bundle's
            # resolved c (hedge_bundle.Bundle.mirror_utility_scale).
            "utility_scale_mode":
                str(objective_config.get("Utility_Scale_Mode", "vol_scaled_notional")).lower(),
            "utility_scale_explicit":
                (None if objective_config.get("Utility_Scale_Explicit") is None
                 else float(objective_config["Utility_Scale_Explicit"])),
            # Utility SHAPE params (DIMENSIONLESS, in units of c — applied to x = W/c). Huber:
            # linear gains, quadratic small losses with curvature `huber_aversion`, linear deep
            # tail beyond the knee `huber_delta`. CARA: u = (1−e^{−γx})/γ. Symlog ignores all
            # three. See hedge_bundle._utility_wrap_signed for the exact forms.
            "huber_aversion": float(objective_config.get("Huber_Aversion", 2.5)),
            "huber_delta": float(objective_config.get("Huber_Delta", 1.0)),
            "cara_gamma": float(objective_config.get("CARA_Gamma", 1.0)),
        },
        "policy": None,
        "optimizer": None,
        "solver": _solver_config(solver_config),
        "history_lookback_business_days": history_lookback,
        "portfolio_state": {
            "positions": {str(n): float(v)
                          for n, v in portfolio_state.get("Positions", {}).items()},
            "cash_balances": {str(n): float(v)
                              for n, v in portfolio_state.get("Cash_Balances", {}).items()},
            "settlement_prices": {str(n): float(v)
                                  for n, v in portfolio_state.get("Settlement_Prices", {}).items()},
            "margin_balances": {str(n): float(v)
                                for n, v in portfolio_state.get("Margin_Balances", {}).items()},
            "initial_margin": {
                str(n): {"method": str(spec["Method"]), "amount": float(spec["Amount"])}
                for n, spec in portfolio_state.get("Initial_Margin", {}).items()},
            "spot_price_history": _spot_price_history(
                hedging_problem, history_lookback, referenced_commodities),
        },
        "accounting": {
            "position_limits": {
                str(n): {"min_position": int(limit["Min_Position"]),
                         "max_position": int(limit["Max_Position"])}
                for n, limit in evaluator_config.get("Position_Limits", {}).items()},
            "cash_accounts": {n: {"currency": normalized_tradables[n]["currency"]}
                              for n in cash_account_names},
            "instrument_to_cash_account": {
                n: account_by_currency.get(meta["currency"], fallback_account)
                for n, meta in normalized_tradables.items()},
            "transaction_cost_per_unit":
                float(evaluator_config.get("Transaction_Cost_Per_Unit", 0.0)),
            # Scalar half-spread bps (fast-path + diagnostic display); `spec` is None for a scalar
            # Bid_Offer_Spread_Bps and a normalized per-instrument/vol-scale dict otherwise —
            # resolved in `per_contract_kappa`.
            "bid_offer_spread_bps": scalar_spread_bps,
            "bid_offer_spread_spec": spread_spec,
            # Roll-as-calendar-spread: when a rebalance offsets Δq across adjacent maturities, the
            # matched quantity pays a single calendar half-cost instead of two outright half-spreads
            # (realized accounting only — see hedge_bundle._roll_rebate). Default off.
            "roll_as_calendar_spread":
                evaluator_config.get("Roll_As_Calendar_Spread", "No") == "Yes",
            "calendar_spread_bps": (float(evaluator_config["Calendar_Spread_Bps"])
                                    if evaluator_config.get("Calendar_Spread_Bps") is not None
                                    else None),
            # Vol-linked initial-margin FUNDING charge on the post-trade book (realized accounting
            # only — see hedge_bundle._im_funding_charge). Per hedge leg i at step t the desk posts
            # IM_i = IM_Vol_Multiplier·(σ_t/IM_Ref_Vol)·F_i·|q_i^post|·cs_i and pays
            # IM_Funding_Spread_Bps·1e-4·dt to FUND it over the calendar step (above the risk-free
            # the margin ledger already earns). Spread default 0.0 ⇒ the term is exactly 0 and never
            # executes; IM_Ref_Vol default 1.0 is inert (only divided when the spread is on).
            "im_funding_spread_bps": float(evaluator_config.get("IM_Funding_Spread_Bps", 0.0)),
            "im_vol_multiplier": float(evaluator_config.get("IM_Vol_Multiplier", 0.0)),
            "im_ref_vol": float(evaluator_config.get("IM_Ref_Vol", 1.0)),
            "force_flat_at_end": evaluator_config.get("Force_Flat_At_End", "Yes") == "Yes",
            "total_position_abs_limit":
                float(evaluator_config.get("Total_Position_Abs_Limit", 0.0)),
            "total_position_schedule": _position_schedule(evaluator_config),
        },
        "privileged_layout": derive_privileged_layout(stoch_factors),
    }


def per_contract_kappa(runtime, price, name, vol=None):
    """Per-contract turnover cost for tradable `name` at mark `price`: a flat
    Transaction_Cost_Per_Unit plus a half-spread charge on notional
    (`0.5 · half_bps · 1e-4 · price · contract_size`). `price` is a scalar or tensor mark.
    Single source for the solver's decision-time kappa, the env's realized debit, and the
    diagnostic CSV writer — any change (asymmetric bid/offer, tiered spread) lives here alone.

    `half_bps` is the scalar `Bid_Offer_Spread_Bps` (fast-path) unless a spread SPEC is
    configured, in which case it is the instrument's `Per_Instrument` base (falling back to
    `Default_Bps`) scaled by `(vol/Ref_Vol)**Beta` when the spec declares a `Vol_Scale` and a
    world-agnostic annualized `vol` is supplied. `vol=None` (or scalar spread / no Vol_Scale) ⇒
    vol-independent, bit-identical to the scalar behaviour."""
    acc = runtime["accounting"]
    contract_size = float(runtime["tradables"][name]["contract_size"])
    spec = acc["bid_offer_spread_spec"]
    if spec is None:
        half_bps = acc["bid_offer_spread_bps"]
    else:
        half_bps = spec["per_instrument"].get(str(name), spec["default_bps"])
        vscale = spec["vol_scale"]
        if vscale is not None and vol is not None:
            half_bps = half_bps * (vol / vscale["ref_vol"]) ** vscale["beta"]
    return (acc["transaction_cost_per_unit"]
            + 0.5 * half_bps * 1.0e-4 * price * contract_size)


def initial_q_from_runtime(runtime, batch, device):
    """Per-hedge initial contract book `q0` `(batch, n_hedge)` from the normalized
    `Portfolio_State` positions, in `runtime['names']['hedges']` order (hedge legs only,
    cash accounts excluded). The seed the stepper already applies to its opening positions —
    exposed here so the solver's frictionless bank/verdict/benchmark tracks measure their
    FIRST-step turnover from the real opening book rather than from flat.

    The differential-ML value function is POSITION-FREE: `q0` affects only first-step
    turnover diagnostics + the rolled P&L, never the fitted value. If turnover cost ever
    becomes material to the objective, the incoming position becomes a genuine state
    variable and `q_prev` must move into the value-function state (V(market, W, q))."""
    positions = runtime["portfolio_state"]["positions"]
    hedges = runtime["names"]["hedges"]
    q0 = torch.tensor([float(positions.get(str(h), 0.0)) for h in hedges], device=device)
    return q0.unsqueeze(0).expand(batch, len(hedges)).contiguous()
