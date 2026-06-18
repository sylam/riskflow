from __future__ import annotations

import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tensordict import TensorDict

from .hedge_features import (
    build_entity_state, _full_spot_timeline, LEG_FEATURE_NAMES,
    assemble_privileged_factors,
    compute_spot_realized_vol, compute_spot_trend, compute_spot_stretch,
    compute_basis_zscore,
)
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
                "basis_zscore_20", "spot_price_history"):
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
    # forward_jacobian: same 2-deep {tradable: {factor: (T, [n_tenors,] B)}} shape as
    # hedge_ratios; _slice_episode_tensor slices the trailing batch axis for 2D and 3D alike.
    if bundle.get("forward_jacobian"):
        sliced["forward_jacobian"] = {
            tradable: {
                factor: _slice_episode_tensor(t, episode_indices, batch_size)
                for factor, t in by_factor.items()
            }
            for tradable, by_factor in bundle["forward_jacobian"].items()
        }
    # liability_jacobian: 1-deep {factor: (T, [n_tenors,] B)} — slice the trailing batch axis.
    if bundle.get("liability_jacobian"):
        sliced["liability_jacobian"] = {
            factor: _slice_episode_tensor(t, episode_indices, batch_size)
            for factor, t in bundle["liability_jacobian"].items()
        }
    for scalar_key in ("last_settlement_index", "time_grid_days_cpu", "total_leg_volume",
                        "scenario_dates", "business_indices", "utility_scale"):
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


def _is_symlog_objective(runtime):
    # `objective["object"]` is canonical-lowercased at normalization time
    # (hedge_runtime._normalize_objective_config), so plain equality is sufficient here.
    obj = (runtime.get("objective") or {})
    return obj.get("object") == "asymmetricutility_symlog"


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


def _utility_wrap(x_dollars, runtime):
    """Wrap a non-negative dollar quantity through the configured utility transform.
    Returns x_dollars unchanged for non-symlog objectives. Caller must ensure x_dollars >= 0
    (log1p is invalid on negative inputs)."""
    if not _is_symlog_objective(runtime):
        return x_dollars
    c = _require_utility_scale(runtime)
    return torch.log1p(x_dollars / c)


def _utility_wrap_signed(x_dollars, runtime):
    """Signed-input variant of `_utility_wrap`: u(x; c) = sign(x) · log1p(|x| / c) for
    symlog, identity for legacy. Use this when the input can be either sign (e.g. net_pnl
    in `_evaluate_objective`); `_utility_wrap` rejects negatives by docstring."""
    if not _is_symlog_objective(runtime):
        return x_dollars
    c = _require_utility_scale(runtime)
    return torch.sign(x_dollars) * torch.log1p(x_dollars.abs() / c)


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

    Multi-commodity TODO: use the deal's reference commodity; current scope is single-commodity.
    """
    objective = (runtime.get('objective') or {})
    is_symlog = objective.get('object') == 'asymmetricutility_symlog'

    def _degenerate(reason):
        """Either raise (symlog — degraded c silently breaks tail compression) or return
        the $1k floor (legacy — c isn't consumed)."""
        if is_symlog:
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


def log_symlog_penalty_calibration(bundle, runtime):
    """Under symlog, the dollar notional factor in every per-step penalty is compressed
    through log1p(notional / c) — a coefficient previously tuned against dollar-scale
    notional ($M) now multiplies a value of order 1, i.e. ~6 orders of magnitude weaker.
    Print the per-step magnitude for each active penalty at notional = c (the natural
    scale, where log1p(1) ≈ 0.693) so users migrating from a legacy config can see at a
    glance whether their penalties still bite vs the dense-tracking effective floor.

    No-op for legacy objectives. Reports active (coef > 0) penalties only.
    """
    objective = runtime.get('objective') or {}
    if objective.get('object') != 'asymmetricutility_symlog':
        return
    # Direct lookup, no default — `resolve_utility_scale` is called immediately before this
    # in `build_torchrl_bundle` and either succeeds (c ≥ $1k) or raises. If the key is
    # missing here, that's a refactor regression worth surfacing as KeyError, not a silent
    # divide-by-zero in the bite math below.
    c = float(bundle['utility_scale'])
    fp = float(objective.get('floor_penalty', 1.0))
    rs = float((runtime.get('optimizer') or {}).get('dense_tracking_reward_scale', 0.0))
    log1p_at_c = float(np.log1p(1.0))  # ≈ 0.693
    expiry_coef = float(objective.get('expiry_penalty', 0.0))
    expiry_thr = float(objective.get('expiry_threshold_days', 4.0))
    post_deal_coef = float(objective.get('post_deal_trade_penalty', 0.0))
    bounds_coef = float(objective.get('position_bounds_penalty', 0.0))
    bounds_thr = float(objective.get('position_bounds_threshold', 5.0))
    per_inst_coef = float(objective.get('per_instrument_bounds_penalty', 0.0))
    per_inst_thr = float(objective.get('per_instrument_bounds_threshold', 5.0))
    dense_step = rs * fp * log1p_at_c
    lines = [f'symlog penalty bite-check (notional = c = ${c:,.0f}, log1p(1) = {log1p_at_c:.3f}):']
    lines.append(f'  reference: dense_tracking ≈ rs·fp·log1p ≈ {dense_step:.2f} utility/step (per-step asymmetric shaping)')
    if expiry_coef > 0.0:
        m = float(np.exp(expiry_thr))
        bite = expiry_coef * m * log1p_at_c
        lines.append(f'  expiry      = {expiry_coef:g} × exp({expiry_thr:g})={m:.1f} × {log1p_at_c:.2f} = {bite:.2f} utility/step at expiry')
    if post_deal_coef > 0.0:
        m = 2.0
        bite = post_deal_coef * m * log1p_at_c
        lines.append(f'  post_deal   = {post_deal_coef:g} × {m:.0f} (1 day past) × {log1p_at_c:.2f} = {bite:.2f} utility/step')
    if bounds_coef > 0.0:
        # Both bounds penalties are pure-count: bite = coef × ramp(violation, threshold) — no
        # dollar wrap. Reported bite at violation=threshold is the unitful answer (utility/step).
        m = float(np.exp(bounds_thr) - 1.0)
        lines.append(f'  bounds      = {bounds_coef:g} × (exp({bounds_thr:g})-1)={m:.1f} = {bounds_coef * m:.2f} utility/step at violation=threshold (portfolio Σ|pos| past limit)')
    if per_inst_coef > 0.0:
        m = float(np.exp(per_inst_thr) - 1.0)
        lines.append(f'  per_instr   = {per_inst_coef:g} × (exp({per_inst_thr:g})-1)={m:.1f} = {per_inst_coef * m:.2f} utility/step at violation=threshold (per-instrument [Min, Max])')
    rc = float((runtime.get('optimizer') or {}).get('dense_tracking_reward_clip', 0.0))
    if rc > 0.0:
        # `rc` clips `shaping = prev_down - next_down` BEFORE the rs·fp multiplication in
        # asymmetric mode (the spec §8 default), so the relevant scale is the pre-scale
        # log1p-saturation bound (~5) — NOT the post-rs·fp number. A clip much larger than
        # ~5 is a no-op (the shaping never reaches it).
        dense_pre_scale_bound = 5.0
        flag = ' (no-op: clip > pre-scale shaping bound — likely a stale dollar value)' if rc > 10.0 * dense_pre_scale_bound else ''
        lines.append(f'  dense_clip  = {rc:g} pre-scale utility (post rs·fp max ≈ {rs * fp * dense_pre_scale_bound:.0f}){flag}')
    # Reward_Scale × symlog footgun: legacy configs used Reward_Scale ≪ 1 to compress
    # dollar rewards into a tractable range; under symlog rewards are already in utility
    # space (≈ [-100, 100]) so a small Reward_Scale crushes the gradient signal flat.
    reward_scale = float((runtime.get('optimizer') or {}).get('reward_scale', 1.0))
    if reward_scale != 1.0:
        lines.append(f'  reward_scale = {reward_scale:g} (canonical for symlog is 1.0)')
    lines.append('  if any bite << dense_tracking, that penalty is unlikely to influence the policy — coef may need 10-1000x scaling vs the legacy dollar-tuned value')
    logging.info('\n'.join(lines))

    # Standalone warnings (logging.WARNING level so they're visible in noisy training logs)
    # for two symlog footguns that survive from legacy dollar-tuned configs:
    #
    #   1. Reward_Scale ≪ 1: was needed under legacy ($1M-$100M reward range); under
    #      symlog (utility units ~[-100, +100]) it just crushes the gradient signal.
    #
    #   2. Value_Loss_Asym_Weight > 1: justified under legacy heavy-tailed downside
    #      (8-order-of-magnitude V-target span); under symlog the target distribution
    #      has bounded tails, so asymmetric V-loss is unmotivated.
    if reward_scale < 0.1 or reward_scale > 10.0:
        logging.warning(
            f"Reward_Scale={reward_scale:g} under AsymmetricUtility_Symlog: canonical "
            f"value is 1.0. Symlog rewards are utility-units (~[-100, +100]); a small "
            f"Reward_Scale crushes the gradient signal flat. If this is a stale legacy "
            f"dollar-tuned config, set Optimizer.Reward_Scale=1.0."
        )
    asym_weight = float((runtime.get('optimizer') or {}).get('value_loss_asym_weight', 1.0))
    if asym_weight != 1.0:
        logging.warning(
            f"Value_Loss_Asym_Weight={asym_weight:g} under AsymmetricUtility_Symlog: "
            f"canonical is 1.0 (symmetric V-loss). The asymmetric V-loss countered "
            f"median-bias on heavy-tailed dollar targets; symlog bounds the tails so "
            f"the motivation evaporates. Consider Optimizer.Value_Loss_Asym_Weight=1.0."
        )
    power = float(objective.get('power', 1.0))
    if power != 1.0:
        logging.warning(
            f"Power={power:g} under AsymmetricUtility_Symlog: surplus/shortfall = "
            f"sr·log1p(net_pnl/c)^p — a *compound* concave-then-power compression "
            f"that differs in shape from legacy sr·net_pnl^p (dollar-domain "
            f"convexification). Legacy `Power>1` amplified large wins/losses; under "
            f"symlog the same `Power>1` double-compresses tails. Coefficients tuned "
            f"for legacy Power!=1 may not transfer meaningfully. Canonical for symlog is 1.0."
        )


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

    `Dense_Reward_Mode = "potential_based"` (Ng et al. 1999 PBRS-shaped):
        Φ(s) = −fp · max(−err(s), 0); shaping_t = γ·Φ(s_{t+1}) − Φ(s_t).
        Same terminal-zero applies. **Note**: strict PBRS invariance requires Φ at
        boundaries to be policy-independent (Φ(terminal) = 0 by convention OR boundary
        states reached deterministically). Our terminal-zero implementation effectively
        sets Φ(terminal) = Φ(s_{T-1}), which IS policy-dependent (tracking error at
        T−1 depends on action history). The kept-transitions sum is therefore
        γ^{T-1}·Φ(s_{T-1}) − Φ(s_0), with a small policy-dependent residual
        γ^{T-1}·(−fp·log1p(down_{T-1}/c)) on top of the terminal asymmetric utility.
        Practically the two modes are scalar-multiples of each other modulo rs:
        asymmetric at rs=1 equals fp·(prev_down − next_down) which is exactly PBRS
        shaping with Φ = −fp·down. So the *behavior* is roughly equivalent; the
        "invariance" claim is approximate, not strict. `Dense_Tracking_Reward_Scale`
        is ignored in this mode (scaling Φ would multiply the residual too)."""
    settings = dict(runtime.get("optimizer") or {})
    fp = float((runtime.get("objective") or {}).get("floor_penalty", 1.0))
    prev_err = _tracking_error_value(prev_state, runtime)
    next_err = _tracking_error_value(next_state, runtime)
    rc = float(settings.get("dense_tracking_reward_clip", 0.0))
    mode = str(settings.get("dense_reward_mode", "asymmetric")).lower()
    # Symlog: replace dollar-valued downside `clamp(-err, 0)` with `log1p(clamp(-err, 0)/c)`.
    # The PBRS theorem (Ng-Harada-Russell 1999) requires only that Φ be a state function — it is
    # silent on Φ's shape — so utility-wrapping doesn't change the invariance status (which is
    # already approximate, not strict, per the docstring above).
    prev_down = torch.clamp(-prev_err, min=0.0)
    next_down = torch.clamp(-next_err, min=0.0)
    prev_down = _utility_wrap(prev_down, runtime)
    next_down = _utility_wrap(next_down, runtime)
    if mode == "potential_based":
        gamma = float(settings.get("gamma", 1.0))
        prev_pot = -fp * prev_down
        next_pot = -fp * next_down
        shaping = gamma * next_pot - prev_pot
        if rc > 0.0:
            # Match asymmetric mode's documented semantics: `rc` is the pre-fp-scale
            # clip on the down-change (`prev_down - next_down`, or its PBRS-discounted
            # analog `prev_down - gamma·next_down`). Asymmetric mode clips
            # `prev_down - next_down` at ±rc BEFORE multiplying by `rs · fp` → effective
            # post-scale clip = `rs · fp · rc`. The PBRS shaping expands to
            # `fp · (prev_down - gamma · next_down)` — its `fp` factor is baked in
            # before the clamp, so to apply the same pre-scale clip we multiply the
            # clamp bound by `fp` (the PBRS analog of asymmetric's rs=1 post-scale clip).
            clip_bound = rc * fp
            shaping = torch.clamp(shaping, min=-clip_bound, max=clip_bound)
        return shaping
    # Asymmetric mode preserves original clip-before-scale semantics: rc is in (utility or dollar)
    # units of downside change depending on objective; `rs · fp` scales the clipped shaping.
    rs = float(settings.get("dense_tracking_reward_scale", 0.0))
    shaping = prev_down - next_down
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
    # `_utility_wrap_signed` returns sign(x)·log1p(|x|/c) for symlog, identity for legacy.
    # The clamps below are sign-disjoint (one is zero where the other isn't), so adding
    # surplus + shortfall is exact — no `torch.where` needed. Clamps are still load-bearing:
    # `torch.pow` of a negative base with non-integer p produces NaN, so we clamp the base
    # to non-negative before pow.
    u_pnl = _utility_wrap_signed(net_pnl, runtime)
    surplus = sr * torch.pow(torch.clamp(u_pnl, min=0.0), p)
    shortfall = -fp * torch.pow(torch.clamp(-u_pnl, min=0.0), p)
    return surplus + shortfall


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


def _action_space_padded_deltas(runtime, *, device):
    """Return the (I, max_bins) int64 padded-deltas tensor for the action space. Built
    once per runtime/device and cached on `runtime["policy"]["action_space"]["_padded_deltas"]`
    — avoids the per-decision-step Python-list-to-tensor allocation in the rollout hot loop."""
    action_space = runtime["policy"]["action_space"]
    cache_key = f"_padded_deltas_dev_{device}"
    cached = action_space.get(cache_key)
    if cached is not None:
        return cached
    deltas_map = action_space.get("trade_deltas")
    if not deltas_map:
        return None
    instrument_order = tuple(str(n) for n in _runtime_names(runtime, "action_instruments"))
    per_instrument = [tuple(deltas_map[n]) for n in instrument_order]
    max_bins = max(len(d) for d in per_instrument) if per_instrument else 1
    padded = [list(d) + [d[-1]] * (max_bins - len(d)) for d in per_instrument]
    tensor = torch.tensor(padded, dtype=torch.int64, device=device)
    action_space[cache_key] = tensor
    return tensor


def _action_space_min_deltas(runtime, *, device):
    """Cached per-runtime/device min-trade-delta tensor for the legacy unit-step fallback path."""
    action_space = runtime["policy"]["action_space"]
    cache_key = f"_min_deltas_dev_{device}"
    cached = action_space.get(cache_key)
    if cached is not None:
        return cached
    min_map = action_space.get("min_trade_delta", {})
    instrument_order = tuple(str(n) for n in _runtime_names(runtime, "action_instruments"))
    tensor = torch.tensor([int(min_map.get(n, 0)) for n in instrument_order],
                          dtype=torch.int64, device=device)
    action_space[cache_key] = tensor
    return tensor


def _realized_structured_action(action, current_positions, runtime, *, batch_size, device):
    if action is None:
        return None
    executed = _resolve_trade_deltas(action, runtime, batch_size=batch_size, device=device)
    instrument_order = tuple(str(n) for n in _runtime_names(runtime, "action_instruments"))
    ordered = torch.stack([executed[n] for n in instrument_order], dim=1).round().to(dtype=torch.int64)
    # Action bins from realized deltas — searchsorted reverse-lookup against the per-instrument
    # sorted delta list. Since limits are now reward-side rather than clipped, executed ≡ proposed.
    deltas_tensor = _action_space_padded_deltas(runtime, device=ordered.device)
    if deltas_tensor is not None:
        # searchsorted(sorted=(I, max_bins), values=(I, B)) → (I, B)
        bins_T = torch.searchsorted(deltas_tensor, ordered.transpose(0, 1).contiguous())
        bins = bins_T.transpose(0, 1).contiguous()
    else:
        # Fallback: legacy unit-step ranges. bin = delta - min.
        bins = ordered - _action_space_min_deltas(runtime, device=ordered.device)
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

    Gated on `bundle["last_settlement_index"]`: penalty only fires AFTER the deal
    has fully settled. During the deal's active life, holding contracts close to
    their own expiry is legitimate hedging (e.g. an averaging deal whose window
    ends ~contract expiry — the textbook hedge holds JUL contracts up to JUL's
    last trade date, and the framework must not punish that). Once the deal is
    over (`current > last_settle`), any remaining position is non-hedge exposure
    and contract-expiry physical-delivery risk applies; the penalty fires then.

    Coef from `Expiry_Penalty`; threshold from `Expiry_Threshold_Days` (default 4).
    Disabled when objective.expiry_penalty <= 0 or `last_settlement_index` missing.
    """
    objective = runtime.get("objective") or {}
    coef = float(objective.get("expiry_penalty", 0.0))
    if coef <= 0.0:
        return None
    last_settle = bundle.get("last_settlement_index")
    if last_settle is None or int(state["time_index"]) <= int(last_settle):
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
            mult = np.exp(threshold - days_to_expiry)
        else:
            # Past expiry: continuous with the at-expiry value, then grows
            # linearly with days-past so each extra day of holding hurts more
            # than the last. Linear (not exp) avoids float32 overflow over a
            # multi-week post-expiry tail.
            mult = np.exp(threshold) * (1.0 + abs(days_to_expiry))
        position = state["positions"][name]
        price = state["tradable_values"][name]
        cs = float(_runtime_tradable(runtime, name).get("contract_size", 1.0))
        notional = position.abs() * price * cs
        penalty = penalty - coef * mult * _utility_wrap(notional, runtime)
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
        price = state["tradable_values"][name]
        cs = float(_runtime_tradable(runtime, name).get("contract_size", 1.0))
        notional = pos.abs() * price * cs
        penalty = penalty - coef * (1.0 + days_past) * _utility_wrap(notional, runtime)
        any_active = True
    return penalty if any_active else None


def _exp_linear_ramp(violation, threshold):
    """f(0)=0; exp(v)−1 for 0≤v<T; exp(T)−1 + exp(T)·(v−T) for v≥T. C¹ at v=T."""
    v = violation.clamp_min(0.0)
    eT = np.exp(threshold)
    return torch.exp(v.clamp_max(threshold)) - 1.0 + eT * (v - threshold).clamp_min(0.0)


def _per_instrument_bounds_penalty(state, bundle, runtime):
    """exp→linear penalty on each instrument's [Min_Position, Max_Position] constraint:
      v_i = relu(pos_i − max_position_i) + relu(min_position_i − pos_i)
      penalty_i = coef × ramp(v_i, threshold)
    Pure count-based: violation is dimensionless (integer contracts past the bound), and
    `ramp` is dimensionless, so `coef` carries the utility-unit conversion directly. No
    dollar wrap — that would only add a constant scaling factor for a single-commodity
    hedger anyway, while making the design appear inconsistent with the portfolio-total
    penalty (which has the same structure). Symmetric in long/short violations of equal
    count."""
    objective = runtime.get("objective") or {}
    coef = float(objective.get("per_instrument_bounds_penalty", 0.0))
    threshold = float(objective.get("per_instrument_bounds_threshold", 5.0))
    position_limits = (runtime.get("accounting") or {}).get("position_limits") or {}
    if coef <= 0.0 or not position_limits:
        return None
    device = bundle["time_grid_days"].device
    batch_size = _batch_size_from_bundle(bundle)
    penalty = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for name in _runtime_names(runtime, "action_instruments"):
        if name not in state["positions"] or name not in position_limits:
            continue
        limits = position_limits[name]
        min_pos = float(limits["min_position"])
        max_pos = float(limits["max_position"])
        pos = state["positions"][name]
        violation = (pos - max_pos).clamp_min(0.0) + (min_pos - pos).clamp_min(0.0)
        # `_exp_linear_ramp(0, threshold) == 0`, so summing inactive instruments adds
        # only zeros. Skipping was a `.any()` sync per (instrument, step) — 3·250=750
        # CUDA-CPU syncs per epoch — for negligible compute savings.
        penalty = penalty - coef * _exp_linear_ramp(violation, threshold)
    return penalty


def _position_bounds_penalty(state, bundle, runtime):
    """exp→linear penalty on the portfolio Σ_i|pos_i| total-position constraint:
      v = max(0, Σ_i|pos_i| − total_position_abs_limit)
      penalty = coef × ramp(v, threshold)
    Same pure count-based formulation as `_per_instrument_bounds_penalty` — see that
    function's docstring for rationale. Identical math, just fed a different violation."""
    objective = runtime.get("objective") or {}
    coef = float(objective.get("position_bounds_penalty", 0.0))
    threshold = float(objective.get("position_bounds_threshold", 5.0))
    total_limit = float((runtime.get("accounting") or {}).get("total_position_abs_limit", 0.0))
    if coef <= 0.0 or total_limit <= 0.0:
        return None
    device = bundle["time_grid_days"].device
    batch_size = _batch_size_from_bundle(bundle)
    abs_count = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for name in _runtime_names(runtime, "action_instruments"):
        if name not in state["positions"]:
            continue
        abs_count = abs_count + state["positions"][name].abs()
    return -coef * _exp_linear_ramp(abs_count - total_limit, threshold)


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
    per_instrument_bounds = _per_instrument_bounds_penalty(state, bundle, runtime)
    if per_instrument_bounds is not None:
        reward = reward + per_instrument_bounds.to(device=device, dtype=torch.float32)
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
        "max_grad_norm": float(p.get("max_grad_norm", 0.5)),
        "reward_scale": float(p.get("reward_scale", 1.0)),
        "validation_fraction": float(p.get("validation_fraction", 0.25)),
        "validation_min_batch": int(p.get("validation_min_batch", 256)),
        "decision_interval_curriculum": tuple(dict(s) for s in p.get("decision_interval_curriculum", ())),
        "cvar_alpha": float(p.get("cvar_alpha", 0.0)),
        "cvar_lambda": float(p.get("cvar_lambda", 0.0)),
        "value_loss_asym_weight": float(p.get("value_loss_asym_weight", 1.0)),
        "entropy_floor_h_min": float(p.get("entropy_floor_h_min", 0.0)),
        "entropy_floor_coef": float(p.get("entropy_floor_coef", 0.0)),
        "seed": p.get("seed"),
        "live_diag_path": p.get("live_diag_path"),
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


def _collect_ppo_rollout(policy, bundle, runtime, decision_indices, *, deterministic=False, reward_scale=1.0):
    state = build_shared_state(bundle, runtime)
    _mirror_utility_scale_to_runtime(bundle, runtime)
    decision_set = set(int(i) for i in decision_indices)
    batch_size = int(next(iter(state["positions"].values())).shape[0])
    device = policy.device
    instrument_order = policy.action_space.instrument_order
    debug_strict_bins = bool((runtime.get("optimizer") or {}).get("debug_strict_bins", False))
    entity_snapshots = []
    privileged_snapshots = []
    position_snapshots = []
    action_bins_list = []
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
                # Snapshot the accumulated reward, then reset the accumulator in place.
                # Without the clone, the in-place .zero_() below would also zero the
                # tensor we just appended to `rewards`. Cheaper than allocating a
                # fresh zeros tensor every decision step.
                rewards.append(pending_reward.clone())
                dones.append(state["done"].to(dtype=torch.float32, device=device))
                pending_reward.zero_()
            seen_first_decision = True
            # Materialize entity_state lazily on the decision step. The cheap refresh path skips
            # this on non-decision steps (interval=10 → 9× saved per decision).
            state = _ensure_decision_views(state, bundle, runtime)
            priv = _privileged_state_at(bundle, t, batch_size, device)
            positions_t = _positions_tensor(state, instrument_order, device=device)
            output = policy.sample(
                state["entity_state"], deterministic=deterministic,
                privileged_state=priv,
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
        pending_reward.add_(transition["reward"].to(device=device, dtype=torch.float32), alpha=reward_scale)
        state = next_state
    if seen_first_decision:
        rewards.append(pending_reward)  # loop exited — no further mutation, no need to clone
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
    # Note: rollout["positions"] is recorded only for `_post_settle_holding_metrics`
    # diagnostics. The policy no longer consumes per-decision positions (per-instrument
    # limits are reward-side, not mask-side), so we don't reshape it into minibatches.
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
            new_lp, entropy, value, logits = policy.evaluate_action(mb_state, mb_actions, privileged_state=mb_privileged)
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
            loss = (
                policy_loss
                + settings["value_coef"] * value_loss
                - settings["entropy_coef"] * entropy_bonus
                + entropy_floor_coef * entropy_floor_loss
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


def _post_settle_holding_metrics(rollout_positions, decision_indices, last_settle):
    """Diagnostic: fraction of paths holding any non-zero position past `last_settle`,
    plus a position-limit audit (max single-instrument long, min single-instrument short,
    max Σ|pos| across instruments). Returns dict with all four. No-op if last_settle is
    None or rollout has no decision steps."""
    if last_settle is None or rollout_positions is None or rollout_positions.shape[0] == 0:
        return {"frac_post_settle_holding": 0.0,
                "max_per_instrument_position": 0.0,
                "min_per_instrument_position": 0.0,
                "max_total_abs_position": 0.0}
    di_t = torch.as_tensor(decision_indices, device=rollout_positions.device)
    post_mask = (di_t > int(last_settle))                                       # (D,)
    if bool(post_mask.any()):
        post_pos = rollout_positions[post_mask]                                 # (D_post, B, I)
        per_path_held = (post_pos.abs().sum(dim=(0, -1)) > 0).float()           # (B,)
        frac = float(per_path_held.mean().item())
    else:
        frac = 0.0
    pos_f = rollout_positions.float()
    max_per_instr = float(pos_f.max().item())
    min_per_instr = float(pos_f.min().item())
    max_total_abs = float(pos_f.abs().sum(dim=-1).max().item())
    return {"frac_post_settle_holding": frac,
            "max_per_instrument_position": max_per_instr,
            "min_per_instrument_position": min_per_instr,
            "max_total_abs_position": max_total_abs}


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


def build_torchrl_evaluation_output(state, terminal_transition, bundle, *, no_trade_terminal=None, timing=None, optimizer_diagnostics=None, post_settle_metrics=None):
    summary = _terminal_summary(state, terminal_transition, bundle)
    out = {
        "metrics": summary["metrics"],
        "final_state": summary["final_state"],
        "diagnostics": {
            "num_episodes": int(summary["final_state"]["net_pnl"].shape[0]),
            "num_batches": int(bundle.get("meta", {}).get("num_batches", 1)),
            "trainer_type": "ppo",
            "optimizer_diagnostics": optimizer_diagnostics or {},
            "post_settle": post_settle_metrics or {},
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
    post_settle = _post_settle_holding_metrics(
        rollout.get("positions"), decision_indices, bundle.get("last_settlement_index"),
    )
    output = build_torchrl_evaluation_output(
        rollout["terminal_state"], rollout["terminal_transition"], bundle,
        no_trade_terminal=no_trade_terminal,
        timing={"evaluation_time_seconds": float(time.perf_counter() - started)},
        post_settle_metrics=post_settle,
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
        # No-split fallback is silent + dangerous: train_bundle == val_bundle means the
        # "evaluation_summary" is the training-set metric, not held-out. Raise loud so
        # any caller running training MUST configure Batch_Size and Validation_Min_Batch
        # to admit a real hold-out, or explicitly accept by setting Validation_Fraction=0.
        if val_fraction > 0.0:
            raise ValueError(
                f"Validation hold-out impossible: Batch_Size={total} but "
                f"Validation_Min_Batch={val_min} (need Batch_Size > 2*Validation_Min_Batch). "
                f"Either raise Batch_Size, lower Validation_Min_Batch, or set "
                f"Validation_Fraction=0 to explicitly disable hold-out (evaluation will be "
                f"on the training set — overfitting risk)."
            )
        train_bundle = bundle
        val_bundle = bundle

    epoch_diag = []
    started = time.perf_counter()
    entropy_start = settings["entropy_coef"]
    entropy_end = settings["entropy_coef_min"]
    epochs_total = max(int(settings["epochs"]), 1)
    for epoch in range(settings["epochs"]):
        if settings["entropy_schedule"] == "linear" and epochs_total > 1:
            frac = epoch / (epochs_total - 1)
            settings["entropy_coef"] = entropy_start + (entropy_end - entropy_start) * frac
        decision_indices = _decision_time_indices(train_bundle, settings, epoch=epoch, evaluation=False)
        decision_interval = _decision_interval_business_days_for_epoch(settings, epoch, evaluation=False)
        rollout_started = time.perf_counter()
        rollout = _collect_ppo_rollout(policy, train_bundle, runtime, decision_indices, deterministic=False, reward_scale=settings["reward_scale"])
        rollout_time = float(time.perf_counter() - rollout_started)
        if rollout is None:
            continue
        update_started = time.perf_counter()
        diag = _ppo_update(policy, optimizer, rollout, settings, decision_interval=decision_interval, epoch=epoch)
        update_time = float(time.perf_counter() - update_started)
        if scheduler is not None:
            scheduler.step()
        terminal_reward = float(rollout["terminal_transition"]["reward"].mean().item())
        # V-saturation tripwire (symlog only): worst-batch terminal utility. Approaches the
        # symlog asymptote at -fp · log1p(big/c); crossing ~-45 means observed tails are
        # close to compressing all extreme losses to the same number → V loses gradient
        # signal where it matters. Per project memory: restart with --utility_scale_explicit,
        # don't try to fix mid-flight.
        terminal_reward_min = float(rollout["terminal_transition"]["reward"].min().item())
        if _is_symlog_objective(runtime) and terminal_reward_min < -45.0:
            logging.warning(
                f"epoch={epoch} min(util_terminal)={terminal_reward_min:.2f} approaching symlog "
                f"saturation at -fp·log1p(huge/c) — rollout tails growing OOD vs c=${(runtime.get('objective') or {}).get('utility_scale', 0.0):,.0f}; "
                "consider restarting with a larger --utility_scale_explicit"
            )
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
        # Direct measure of whether `Post_Deal_Trade_Penalty` is biting: fraction of paths
        # that hold ANY non-zero position at ANY decision step strictly past the deal's last
        # settlement. Resolves "did the post_deal coef change actually fix the speculative-
        # holding behavior" without needing to read worst_net_pnl trends.
        ps = _post_settle_holding_metrics(rollout.get("positions"), decision_indices, train_bundle.get("last_settlement_index"))
        frac_post_settle_holding = ps["frac_post_settle_holding"]
        epoch_diag.append({
            "epoch": epoch,
            "policy_loss": diag["policy_loss"],
            "value_loss": diag["value_loss"],
            "entropy": diag["entropy"],
            "advantage_mean": diag["advantage_mean"],
            "advantage_std": diag["advantage_std"],
            "average_reward": terminal_reward,
            "min_util_terminal": terminal_reward_min,
            "average_net_pnl": net_pnl,
            "std_net_pnl": std_net_pnl,
            "worst_net_pnl": worst_net_pnl,
            "p5_net_pnl": p5_net_pnl,
            "action_abs_mean": action_abs_mean,
            "min_entropy_decision": min_entropy_decision,
            "min_n_feasible_decision": min_n_feasible_decision,
            "frac_post_settle_holding": frac_post_settle_holding,
            "rollout_time_seconds": rollout_time,
            "update_time_seconds": update_time,
            "decision_interval_business_days": decision_interval,
        })
        # Validation-set rollout every 10 epochs (and on the final epoch). Deterministic /
        # no-grad; surfaces the train/val gap during training so overfitting is visible in
        # the live log instead of only at the final evaluation_summary.
        val_log_suffix = ""
        if val_bundle is not train_bundle and ((epoch % 10 == 0) or (epoch == settings["epochs"] - 1)):
            with torch.no_grad():
                val_decision_indices = _decision_time_indices(val_bundle, settings, epoch=epoch, evaluation=True)
                val_rollout = _collect_ppo_rollout(policy, val_bundle, runtime, val_decision_indices, deterministic=True)
            if val_rollout is not None:
                v_net = (val_rollout["terminal_transition"]["pnl_excess"] + val_rollout["terminal_transition"]["liability_value"]).to(dtype=torch.float64)
                val_mean = float(v_net.mean().item())
                val_std = float(v_net.std().item())
                val_worst = float(v_net.min().item())
                val_p5 = float(torch.quantile(v_net, 0.05).item())
                val_sortino = val_mean - max(0.0, -val_p5)
                epoch_diag[-1]["val_average_net_pnl"] = val_mean
                epoch_diag[-1]["val_std_net_pnl"] = val_std
                epoch_diag[-1]["val_worst_net_pnl"] = val_worst
                epoch_diag[-1]["val_p5_net_pnl"] = val_p5
                val_log_suffix = (
                    f"\n  [val] mean={val_mean:>+12.0f} std={val_std:>10.0f} "
                    f"worst={val_worst:>+12.0f} p5={val_p5:>+12.0f} sortino={val_sortino:>+12.0f}"
                )
        # Live per-epoch dump for buffered-stdout-blind monitoring. Atomic via temp+rename
        # so a reader never sees a partial write. Skipped when no path is set in JSON config
        # (Optimizer.Live_Diag_Path) or directly on runtime (legacy).
        live_path = settings.get("live_diag_path") or runtime.get("live_diag_path")
        if live_path:
            import json as _jsonlib, os as _os
            tmp = live_path + ".tmp"
            with open(tmp, "w") as _f:
                _jsonlib.dump({"epochs": epoch_diag, "current_epoch": int(epoch),
                               "total_epochs": int(settings["epochs"])}, _f)
            _os.replace(tmp, live_path)
        print(
            f"epoch={epoch:>2} interval={decision_interval}d "
            f"net_pnl={net_pnl:>+12.0f} std={std_net_pnl:>10.0f} worst={worst_net_pnl:>+12.0f} "
            f"reward={terminal_reward:>+10.2f} "
            f"|trade|={action_abs_mean:.2f} "
            f"pol_l={diag['policy_loss']:+.4f} val_l={diag['value_loss']:.2e} ent={diag['entropy']:.3f} "
            f"t={rollout_time + update_time:.2f}s"
            f"{val_log_suffix}",
            flush=True,
        )

    total_time = float(time.perf_counter() - started)

    eval_indices = _decision_time_indices(val_bundle, settings, epoch=None, evaluation=True)
    eval_rollout = _collect_ppo_rollout(policy, val_bundle, runtime, eval_indices, deterministic=True)
    no_trade_terminal = _collect_no_trade_rollout(val_bundle, runtime, eval_indices)
    post_settle = _post_settle_holding_metrics(
        eval_rollout.get("positions"), eval_indices, val_bundle.get("last_settlement_index"),
    )
    output = build_torchrl_evaluation_output(
        eval_rollout["terminal_state"], eval_rollout["terminal_transition"], val_bundle,
        no_trade_terminal=no_trade_terminal,
        timing={"total_fit_time_seconds": total_time},
        optimizer_diagnostics={"epochs": epoch_diag},
        post_settle_metrics=post_settle,
    )
    return {"policy": policy, "policy_artifact": policy.to_artifact(), "evaluation_output": output, "optimizer_diagnostics": {"epochs": epoch_diag, "torchrl_rollout_time_seconds": total_time}}


def run_torchrl_execution(bundle, runtime):
    if runtime is None or bundle is None:
        return None
    mode = str(runtime.get("execution_mode", ""))
    if mode == "solve_hedge":
        from .hedge_solver import solve_hedge
        return solve_hedge(bundle, runtime)
    if mode == "optimize_policy":
        return train_torchrl_policy(bundle, runtime)
    return evaluate_torchrl_policy(bundle, runtime)


def _prepend_history_prefix(bundle, runtime, base_date):
    """Prepend H rows of realized history to all time-axis tensors in the bundle so that
    rolling-window features at sim-day-0 already have a populated lookback. Historical rows
    are broadcast across the batch dim (one realized series per commodity).

    Adds bundle['spot_price_history'][commodity] of shape (H, B) — broadcast historical spot.
    Adjusts bundle['time_grid_days'], bundle['tradables'][name], bundle['liability_mtm'],
    bundle['realized_cashflows'][currency], bundle['legs']['features'], bundle['factors'][name].
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
    if bundle.get('hedge_ratios'):
        new_ratios = {}
        for tradable_name, by_factor in bundle['hedge_ratios'].items():
            new_ratios[tradable_name] = {}
            for factor_name, tensor in by_factor.items():
                r_prefix = tensor[:1].expand((H,) + tuple(tensor.shape[1:])).contiguous()
                new_ratios[tradable_name][factor_name] = torch.cat([r_prefix, tensor], dim=0)
        bundle['hedge_ratios'] = new_ratios
    if bundle.get('realized_cashflows'):
        new_cf = {}
        for currency, tensor in bundle['realized_cashflows'].items():
            cf_prefix = torch.zeros((H,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            new_cf[currency] = torch.cat([cf_prefix, tensor], dim=0)
        bundle['realized_cashflows'] = new_cf
    if bundle.get('legs') is not None and bundle['legs'].get('features') is not None:
        feats = bundle['legs']['features']
        feats_prefix = feats[:1].expand((H,) + tuple(feats.shape[1:])).contiguous()
        bundle['legs']['features'] = torch.cat([feats_prefix, feats], dim=0)
    if bundle.get('factors'):
        new_factors = {}
        for factor_name, tensor in bundle['factors'].items():
            f_prefix = tensor[:1].expand((H,) + tuple(tensor.shape[1:])).contiguous()
            new_factors[factor_name] = torch.cat([f_prefix, tensor], dim=0)
        bundle['factors'] = new_factors


def _static_portfolio_descriptors(legs_features, feat_names, ref_price):
    """Deterministic per-outer-t portfolio descriptors from the leg-feature tensor —
    identical for every batch element (book composition is deterministic). Returns
    `(T, 7)`: aggregate_notional, weighted_avg_strike, weighted_avg_tte,
    weighted_moneyness, expiry_dispersion, strike_dispersion, fraction_in_window.

    `|fixed_basis|` is used as the strike level — the energy-Asian payoff references
    `Fixed_Basis`, and LEG_FEATURE_NAMES carries no separate strike field. Volume is the
    notional weight. For a single-leg book the dispersions are identically zero."""
    names = tuple(feat_names)
    lf = legs_features
    while lf.ndim > 3:                      # (T, B, L, F) -> (T, L, F)
        lf = lf[:, 0]
    fixed_basis = lf[..., names.index('fixed_basis')]              # (T, L)
    volume = lf[..., names.index('volume')]
    ttps = lf[..., names.index('time_to_period_start')]
    ttp = lf[..., names.index('time_to_payment')]
    w = volume.abs()                                                # (T, L)
    w_sum = w.sum(dim=-1).clamp_min(1.0e-8)                          # (T,)
    strike = fixed_basis.abs()
    aggregate_notional = w.sum(dim=-1)
    weighted_avg_strike = (w * strike).sum(-1) / w_sum
    weighted_avg_tte = (w * ttp).sum(-1) / w_sum
    weighted_moneyness = weighted_avg_strike / max(float(ref_price), 1.0e-8)
    expiry_dispersion = torch.sqrt(
        ((w * (ttp - weighted_avg_tte.unsqueeze(-1)) ** 2).sum(-1) / w_sum).clamp_min(0.0))
    strike_dispersion = torch.sqrt(
        ((w * (strike - weighted_avg_strike.unsqueeze(-1)) ** 2).sum(-1) / w_sum).clamp_min(0.0))
    in_window = ((ttps <= 0.0) & (ttp > 0.0)).to(w.dtype)           # period started, unpaid
    fraction_in_window = (w * in_window).sum(-1) / w_sum
    return torch.stack([
        aggregate_notional, weighted_avg_strike, weighted_avg_tte, weighted_moneyness,
        expiry_dispersion, strike_dispersion, fraction_in_window], dim=-1)               # (T, 7)


def build_torchrl_bundle(base_date, business_day, time_grid_days, tradable_blocks,
                          factor_tensor_blocks, hedge_profile_blocks, num_batches,
                          stoch_factors, runtime=None, privileged_factor_blocks=None):
    """Assemble the per-batch tensor blocks (produced by the HedgeMonteCarlo simulator) into
    a single torchrl-consumable bundle: concatenates blocks along the batch axis, prepends
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
    legs_bundle = None
    if hedge_profile_blocks and hedge_profile_blocks.get('legs_features'):
        legs_bundle = {
            'features': torch.cat(hedge_profile_blocks['legs_features'], dim=1),
            'ids': list(hedge_profile_blocks['legs_ids'] or ()),
            'feature_names': tuple(hedge_profile_blocks.get('legs_feature_names') or ()),
            'id_names': tuple(hedge_profile_blocks.get('legs_id_names') or ()),
        }
    liability_mtm = torch.cat(hedge_profile_blocks['mtm'], dim=1) if hedge_profile_blocks.get('mtm') else None
    hedge_ratio_blocks = hedge_profile_blocks.get('hedge_ratios') or {}
    hedge_ratios_pre_pad = {}
    for tradable_name, blocks in hedge_ratio_blocks.items():
        if not blocks:
            continue
        factor_names = list(blocks[0].keys())
        hedge_ratios_pre_pad[tradable_name] = {
            name: torch.cat([block[name] for block in blocks], dim=1) for name in factor_names
        }
    fwd_jac_blocks = hedge_profile_blocks.get('forward_jacobian') or {}
    fwd_jac_pre_pad = {}
    for tradable_name, blocks in fwd_jac_blocks.items():
        if not blocks:
            continue
        factor_names = list(blocks[0].keys())
        fwd_jac_pre_pad[tradable_name] = {
            # Batch is the trailing axis for both scalar (T, B) and curve (T, n_tenors, B) deltas.
            name: torch.cat([block[name] for block in blocks], dim=-1) for name in factor_names
        }
    # Liability Jacobian ∂L/∂factor — a single {factor: (T,[n_tenors,]B)} dict per batch
    # (not per-tradable, since the liability is one deal). Batch trailing, like fwd_jac.
    liab_jac_blocks = hedge_profile_blocks.get('liability_jacobian') or []
    liab_jac_pre_pad = {}
    if liab_jac_blocks:
        for name in list(liab_jac_blocks[0].keys()):
            liab_jac_pre_pad[name] = torch.cat([blk[name] for blk in liab_jac_blocks], dim=-1)
    realized_cashflows = {
        currency: torch.cat(blocks, dim=1)
        for currency, blocks in (hedge_profile_blocks.get('realized_cashflows') or {}).items()
    }
    if legs_bundle is not None:
        aligned_time_steps = int(legs_bundle['features'].shape[0])
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
    if legs_bundle is not None:
        bundle['legs'] = {
            'features': pad_time_axis(legs_bundle['features'], aligned_time_steps),
            'ids': legs_bundle['ids'],
            'feature_names': legs_bundle['feature_names'],
            'id_names': legs_bundle['id_names'],
        }
    if liability_mtm is not None:
        bundle['liability_mtm'] = pad_time_axis(liability_mtm, aligned_time_steps)
    # Per-(tradable, factor) hedge ratios — zero-pad post-expiry rows; don't repeat last row.
    if hedge_ratios_pre_pad:
        hedge_ratios = {}
        for tradable_name, by_factor in hedge_ratios_pre_pad.items():
            hedge_ratios[tradable_name] = {}
            for factor_name, tensor in by_factor.items():
                cur_T = int(tensor.shape[0])
                if cur_T >= aligned_time_steps:
                    padded = tensor[:aligned_time_steps]
                else:
                    zeros = tensor.new_zeros((aligned_time_steps - cur_T,) + tuple(tensor.shape[1:]))
                    padded = torch.cat([tensor, zeros], dim=0)
                hedge_ratios[tradable_name][factor_name] = padded
        bundle['hedge_ratios'] = hedge_ratios
    # Per-(tradable, factor) forward Jacobian ∂F_h/∂θ — same zero-pad-post-expiry as ratios;
    # shape[1:] preserves the trailing curve dims for multi-tenor factors.
    if fwd_jac_pre_pad:
        forward_jacobian = {}
        for tradable_name, by_factor in fwd_jac_pre_pad.items():
            forward_jacobian[tradable_name] = {}
            for factor_name, tensor in by_factor.items():
                cur_T = int(tensor.shape[0])
                if cur_T >= aligned_time_steps:
                    padded = tensor[:aligned_time_steps]
                else:
                    zeros = tensor.new_zeros((aligned_time_steps - cur_T,) + tuple(tensor.shape[1:]))
                    padded = torch.cat([tensor, zeros], dim=0)
                forward_jacobian[tradable_name][factor_name] = padded
        bundle['forward_jacobian'] = forward_jacobian
    # Liability Jacobian — same zero-pad-post-expiry; single {factor: (T,[n_tenors,]B)} dict.
    if liab_jac_pre_pad:
        liability_jacobian = {}
        for factor_name, tensor in liab_jac_pre_pad.items():
            cur_T = int(tensor.shape[0])
            if cur_T >= aligned_time_steps:
                padded = tensor[:aligned_time_steps]
            else:
                zeros = tensor.new_zeros((aligned_time_steps - cur_T,) + tuple(tensor.shape[1:]))
                padded = torch.cat([tensor, zeros], dim=0)
            liability_jacobian[factor_name] = padded
        bundle['liability_jacobian'] = liability_jacobian
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
    last = max(len(scenario_dates) - 1, 0)
    bundle['business_indices'] = tuple(
        i for i in range(min(last, len(bd_mask))) if bd_mask[i] and i >= initial_time_index
    )
    if bundle.get('legs') is not None and bundle['legs'].get('features') is not None:
        feat_names = tuple(bundle['legs'].get('feature_names', LEG_FEATURE_NAMES))
        feats = bundle['legs']['features']
        if 'volume' in feat_names:
            leg_vol = feats[..., feat_names.index('volume')]
            while leg_vol.ndim > 1:
                leg_vol = leg_vol[0]
            bundle['total_leg_volume'] = float(leg_vol.abs().sum().item())
        if 'time_to_payment' in feat_names:
            ttp = feats[..., feat_names.index('time_to_payment')]
            while ttp.ndim > 2:
                ttp = ttp[:, 0]
            active = (ttp > 0).any(dim=-1) if ttp.ndim == 2 else (ttp > 0)
            if bool(active.any()):
                bundle['last_settlement_index'] = int(active.nonzero(as_tuple=False).max().item())
        # Static (batch-independent) portfolio descriptors for the solver state vector.
        ref_price = 1.0
        if bundle.get('tradables'):
            first_trad = next(iter(bundle['tradables'].values()))
            ref_price = float(first_trad[min(initial_time_index, first_trad.shape[0] - 1)].mean())
        bundle['static_portfolio_descriptors'] = _static_portfolio_descriptors(
            feats, feat_names, ref_price)
    bundle['spot_realized_vol'] = compute_spot_realized_vol(bundle)
    bundle['utility_scale'] = resolve_utility_scale(bundle, runtime or {})
    logging.info('utility_scale (symlog c) resolved to {0:.2f}'.format(bundle['utility_scale']))
    log_symlog_penalty_calibration(bundle, runtime or {})
    bundle['spot_trend_20'] = compute_spot_trend(bundle, window=20)
    bundle['spot_trend_60'] = compute_spot_trend(bundle, window=60)
    bundle['spot_stretch_20'] = compute_spot_stretch(bundle, window=20)
    bundle['basis_zscore_20'] = compute_basis_zscore(bundle, window=20)
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

def _diag_rollout_policy(policy, bundle, runtime):
    """Deterministic argmax rollout capturing position/trade/price for each decision step.
    Passes positions + privileged_state to policy.sample so the per-instrument feasibility
    mask is honored (Position_Limits)."""
    _mirror_utility_scale_to_runtime(bundle, runtime)
    settings = _optimizer_settings(runtime)
    decision_indices = _decision_time_indices(bundle, settings, epoch=None, evaluation=True)
    decision_set = set(int(i) for i in decision_indices)
    instrument_order = tuple(runtime['names']['action_instruments'])
    state = build_shared_state(bundle, runtime)
    last_idx = _last_time_index(bundle)
    batch_size = int(next(iter(state['positions'].values())).shape[0])
    device = bundle['time_grid_days'].device

    times = []
    positions = {n: [] for n in instrument_order}
    trades = {n: [] for n in instrument_order}
    prices = {n: [] for n in instrument_order}
    transition = None

    while int(state['time_index']) < last_idx:
        t = int(state['time_index'])
        if t in decision_set:
            state = _ensure_decision_views(state, bundle, runtime)
            priv = _privileged_state_at(bundle, t, batch_size, device)
            output = policy.sample(state['entity_state'], deterministic=True,
                                   privileged_state=priv)
            mapped = _realized_structured_action(policy.map_actions(output), state['positions'],
                                                 runtime, batch_size=batch_size, device=device)
            times.append(t)
            pre = {name: state['positions'][name].detach().cpu() for name in instrument_order}
            for name in instrument_order:
                positions[name].append(pre[name])
                prices[name].append(state['tradable_values'][name].detach().cpu())
        else:
            mapped = None
        next_state = step_runtime_state(state, mapped, bundle, runtime)
        if t in decision_set:
            for name in instrument_order:
                trades[name].append((next_state['positions'][name].detach().cpu() - pre[name]))
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        state = next_state

    return {
        'times': times,
        'position': {n: torch.stack(positions[n], dim=0) for n in instrument_order},
        'trade':    {n: torch.stack(trades[n], dim=0)    for n in instrument_order},
        'price':    {n: torch.stack(prices[n], dim=0)    for n in instrument_order},
        'pnl_excess': transition['pnl_excess'].detach().cpu(),
        'liability':  transition['liability_value'].detach().cpu(),
        'net_pnl':    (transition['pnl_excess'] + transition['liability_value']).detach().cpu(),
    }


def _diag_expand_per_day(rollout, bundle, runtime):
    """Build per-instrument (T, B) per-day cashflow tensors plus portfolio totals."""
    factors = bundle['factors']
    spot_keys = [k for k in factors if k.startswith('CommodityPrice.')]
    if len(spot_keys) != 1:
        raise ValueError(
            f"Diagnostic CSV writer expects exactly one CommodityPrice factor in the bundle; "
            f"got {spot_keys}. Multi-commodity diagnostic output is not yet implemented."
        )
    spot = factors[spot_keys[0]].detach().cpu().float()
    mtm_running = bundle['liability_mtm'].detach().cpu().float()
    instrument_order = tuple(runtime['names']['action_instruments'])
    spread_bps = float((runtime.get('accounting') or {}).get('bid_offer_spread_bps', 0.0))
    unit_cost = float((runtime.get('accounting') or {}).get('transaction_cost_per_unit', 0.0))
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
        trade_cost = trd.abs() * (unit_cost + fut * cs * 0.5 * spread_bps * 1.0e-4)
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


def _diag_write_paths_csv(fields, day_strs, label, runtime, csv_path):
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
    sim_start = int(runtime.get('history_lookback_business_days', 0) or 0)
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
        settings = _optimizer_settings(runtime)
        self._decision_set = set(int(i) for i in _decision_time_indices(
            bundle, settings, epoch=None, evaluation=True))
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
        with the per-path reward + pnl_excess + liability_value for this transition.
        """
        was_decision_step = self.is_decision_step
        t_pre = self.time_index
        if was_decision_step:
            self._state = _ensure_decision_views(self._state, self.bundle, self.runtime)
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
            'reward': transition['reward'],
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
        _diag_write_paths_csv(fields, day_strs, label, self.runtime,
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
    """High-level result for HedgeMonteCarlo's TorchRL bundle handoff.

    Carries the bundle + normalized runtime + evaluation summary + saved policy artifact
    so downstream consumers (post-hoc analysis, streaming-service handlers) can do their
    own work without touching framework internals. `policy_artifact` is JSON-serializable;
    reload it via `riskflow.structured_policy.policy_from_artifact`.
    """
    def __init__(self, *, torchrl_bundle=None, runtime=None, evaluation_summary=None,
                 optimizer_diagnostics=None, policy_artifact=None, metadata=None):
        self.torchrl_bundle = torchrl_bundle
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
        if self.torchrl_bundle is None or self.runtime is None:
            raise ValueError("create_stepper needs torchrl_bundle and runtime on the result.")
        return BundleStepper(self.torchrl_bundle, self.runtime)

    def write_diagnostic_csvs(self, output_dir: str) -> None:
        """Post-hoc per-day per-instrument breakdown for the trained policy plus a terminal
        P&L summary. Writes `ml_paths.csv` and `summary.csv` to `output_dir`.

        Pure post-processing — does not mutate the result. Caller decides when (or whether)
        to invoke. Designed for the streaming-service model where compute side effects
        belong to downstream consumers, not the run_job call itself.
        """
        if self.torchrl_bundle is None or self.runtime is None or self.policy_artifact is None:
            raise ValueError(
                "write_diagnostic_csvs needs torchrl_bundle, runtime, and policy_artifact on "
                "the result. Was this an `optimize_policy` / `simulate_only` run?"
            )
        from .structured_policy import policy_from_artifact
        device = self.torchrl_bundle['time_grid_days'].device
        policy = policy_from_artifact(self.policy_artifact, device=device)
        os.makedirs(output_dir, exist_ok=True)
        rollout = _diag_rollout_policy(policy, self.torchrl_bundle, self.runtime)
        fields = _diag_expand_per_day(rollout, self.torchrl_bundle, self.runtime)
        day_strs = [pd.Timestamp(d).strftime('%Y-%m-%d')
                    for d in _bundle_scenario_dates(self.torchrl_bundle)]
        _diag_write_paths_csv(fields, day_strs, 'ml', self.runtime,
                              os.path.join(output_dir, 'ml_paths.csv'))
        pd.DataFrame(_diag_summary_rows('ml', rollout, fields)).to_csv(
            os.path.join(output_dir, 'summary.csv'), index=False, float_format='%.2f',
        )


if __name__ == '__main__':
    pass
