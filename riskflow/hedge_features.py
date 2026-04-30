import pandas as pd
import torch
from tensordict import TensorDict

from . import utils


LEG_FEATURE_NAMES = (
    'fixed_basis',
    'volume',
    'time_to_period_start',
    'time_to_period_end',
    'time_to_payment',
    'accumulation_fraction',
)
LEG_ID_NAMES = ('currency', 'underlying', 'leg_type')

INSTRUMENT_FEATURE_NAMES = (
    'price',
    'price_change_1d',
    'inventory',
    'position_value',
    'time_to_expiry',
    'contract_size',
    'is_tradable_now',
    'cumulative_pnl',
    'time_held',
    'cross_delta',
)

PRICE_MOMENTUM_LOOKBACK = 1
PRICE_ZSCORE_WINDOW = 20

INSTRUMENT_ID_NAMES = ('currency', 'underlying', 'instrument_type')

CASH_FEATURE_NAMES = ('balance', 'margin_balance', 'realized_cashflow', 'realized_cashflow_prev')
CASH_ID_NAMES = ('currency',)

GLOBAL_BASE_FEATURE_NAMES = (
    'time_to_horizon',
    'liability_mtm',
    'liability_mtm_prev',
    'total_position_value',     # sum of position_value across all instruments (negative if net short)
    'residual_exposure',         # portfolio + liability_mtm (zero = fully hedged)
    'coverage_ratio',            # -portfolio / |liability_mtm| (1.0 = fully hedged short)
    'near_position_value',       # sum of position_value for instruments with ttf < 60d
    'mid_position_value',        # 60d <= ttf < 180d
    'far_position_value',        # ttf >= 180d
)

# Per-commodity globals appended after the base block, suffixed with the commodity name.
# (bundle_key, name_prefix) pairs — order is preserved in feature names.
PER_COMMODITY_GLOBAL_FEATURES = (
    ('spot_zscore', 'spot_zscore_20d'),
    ('spot_realized_vol', 'spot_realized_vol_20d'),
    ('implied_atm_vol', 'implied_atm_vol_30d'),
)


def build_global_feature_names(referenced_commodities):
    names = list(GLOBAL_BASE_FEATURE_NAMES)
    for commodity in referenced_commodities:
        for _bundle_key, prefix in PER_COMMODITY_GLOBAL_FEATURES:
            names.append(f'{prefix}_{commodity}')
    return tuple(names)


# Backwards-compat alias for callers that imported the flat list before per-commodity globals
# (none remain in-repo, but exporting keeps third-party consumers from breaking silently).
GLOBAL_FEATURE_NAMES = build_global_feature_names(())


def _intern(registry, vocab, name):
    table = registry.setdefault(vocab, {})
    if name not in table:
        table[name] = len(table)
    return table[name]


def _to_timestamp(value):
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, dict) and '.Timestamp' in value:
        return pd.Timestamp(value['.Timestamp'])
    return pd.Timestamp(value)


def build_entity_layout(runtime):
    registry = {'currency': {}, 'underlying': {}, 'leg_type': {}, 'instrument_type': {}}

    hedge_names = runtime['names']['hedges']
    cash_names = runtime['names']['cash_accounts']

    instrument_meta = []
    for name in hedge_names:
        spec = runtime['tradables'][name]
        params = spec['params']
        currency = str(spec['currency'])
        underlying = str(params.get('Commodity', params.get('Equity', currency)))
        instrument_type = str(spec['deal_type'])
        instrument_meta.append({
            'name': str(name),
            'currency_id': _intern(registry, 'currency', currency),
            'underlying_id': _intern(registry, 'underlying', underlying),
            'instrument_type_id': _intern(registry, 'instrument_type', instrument_type),
            'contract_size': float(spec['contract_size']),
            'last_trade_date': _to_timestamp(spec['last_trade_date']),
        })

    cash_meta = []
    for name in cash_names:
        spec = runtime['tradables'][name]
        currency = str(spec['currency'])
        cash_meta.append({
            'name': str(name),
            'currency': currency,
            'currency_id': _intern(registry, 'currency', currency),
        })

    leg_type_lookup = {
        'FloatingEnergyDeal': 'energy_floating',
        'FixedEnergyDeal': 'energy_fixed',
    }
    leg_count = 0
    for liability in runtime['liabilities'].values():
        deal_type = str(liability.get('deal_type', ''))
        leg_type = leg_type_lookup.get(deal_type, deal_type)
        _intern(registry, 'leg_type', leg_type)
        params = liability.get('params', {})
        for key in ('Currency', 'Payoff_Currency'):
            value = params.get(key)
            if value is not None:
                _intern(registry, 'currency', str(value))
        for key in ('Commodity', 'Reference_Type', 'Equity'):
            value = params.get(key)
            if value is not None:
                _intern(registry, 'underlying', str(value))
        items = params.get('Payments', {}).get('Items', ())
        leg_count += len(items)

    referenced_commodities = tuple(
        ((runtime.get('portfolio_state') or {}).get('spot_price_history') or {}).keys()
    )
    global_feature_names = build_global_feature_names(referenced_commodities)

    return {
        'registry': registry,
        'instruments': {
            'meta': tuple(instrument_meta),
            'feature_names': INSTRUMENT_FEATURE_NAMES,
            'id_names': INSTRUMENT_ID_NAMES,
            'feature_dim': len(INSTRUMENT_FEATURE_NAMES),
            'id_dim': len(INSTRUMENT_ID_NAMES),
            'max_cardinality': len(instrument_meta),
        },
        'cash_accounts': {
            'meta': tuple(cash_meta),
            'feature_names': CASH_FEATURE_NAMES,
            'id_names': CASH_ID_NAMES,
            'feature_dim': len(CASH_FEATURE_NAMES),
            'id_dim': len(CASH_ID_NAMES),
            'max_cardinality': len(cash_meta),
        },
        'legs': {
            'feature_names': LEG_FEATURE_NAMES,
            'id_names': LEG_ID_NAMES,
            'feature_dim': len(LEG_FEATURE_NAMES),
            'id_dim': len(LEG_ID_NAMES),
            'max_cardinality': leg_count,
        },
        'globals': {
            'feature_names': global_feature_names,
            'feature_dim': len(global_feature_names),
            'referenced_commodities': referenced_commodities,
        },
    }


def _id_lookup(registry, vocab, name):
    return registry[vocab].get(str(name), 0)


def derive_privileged_layout(price_models, referenced_commodities, *, factor_module='LogOUSpotModel'):
    """Static schema of privileged factors the simulator will expose, derived from price-model
    config alone. Used by the policy at construction time to size the privileged encoder, before
    the simulation has run. Mirrors the keys that compute_privileged_factors emits at sim time."""
    layout = {}
    multi = len(referenced_commodities) > 1
    for commodity in referenced_commodities:
        if f'{factor_module}.{commodity}' not in (price_models or {}):
            continue
        prefix = f'{commodity.lower()}_' if multi else ''
        layout[f'{prefix}ou_log_deviation'] = 1
        layout[f'{prefix}ou_kappa'] = 1
        layout[f'{prefix}ou_sigma'] = 1
    return layout


def compute_privileged_factors(bundle, price_models, *, factor_module='LogOUSpotModel'):
    """Privileged factors the asymmetric critic sees but the actor does not. For each commodity
    referenced by Spot_Price_History, looks up its LogOUSpotModel params and produces:
      - ou_log_deviation: log(spot_t) - theta  (time-varying per-scenario)
      - ou_kappa, ou_sigma: scenario constants (broadcast)
    All tensors are shape (T_full, B, 1). Multi-commodity prefixes the factor name."""
    factors = bundle.get('factors') or {}
    spot_history = bundle.get('spot_price_history') or {}
    if not factors or not spot_history or not price_models:
        return {}
    out = {}
    multi = len(spot_history) > 1
    for commodity in spot_history.keys():
        spot_key = f'CommodityPrice:{commodity}'
        model_key = f'{factor_module}.{commodity}'
        if spot_key not in factors or model_key not in price_models:
            continue
        spot = factors[spot_key].to(dtype=torch.float32)
        params = price_models[model_key]
        theta = float(params['Theta'])
        kappa = float(params['Kappa'])
        sigma = float(params['Sigma'])
        log_dev = (spot.clamp_min(1e-9).log() - theta).unsqueeze(-1)
        const_shape = log_dev.shape
        kappa_t = torch.full(const_shape, kappa, dtype=torch.float32, device=spot.device)
        sigma_t = torch.full(const_shape, sigma, dtype=torch.float32, device=spot.device)
        prefix = f'{commodity.lower()}_' if multi else ''
        out[f'{prefix}ou_log_deviation'] = log_dev
        out[f'{prefix}ou_kappa'] = kappa_t
        out[f'{prefix}ou_sigma'] = sigma_t
    return out


def _full_spot_timeline(bundle, commodity):
    """Return the (H+T_sim, B) spot tensor for `commodity`, made of JSON history (rows 0..H-1)
    concatenated with the simulator's CommodityPrice factor (rows H..)."""
    spot_history = bundle.get('spot_price_history') or {}
    factors = bundle.get('factors') or {}
    hist = spot_history.get(commodity)
    sim = factors.get(f'CommodityPrice:{commodity}')
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


def compute_implied_atm_vol(bundle, price_models, *, tenor_days=30, factor_module='LogOUSpotModel'):
    """Per-commodity implied ATM volatility for a `tenor_days` option (calendar days), derived
    from LogOU process parameters. Closed-form: integrated log-variance V(T) = σ²(1-e^(-2κT))/(2κ),
    annualized IV = √(V(T)/T). Constant across (t, b) in synthetic mode (one set of OU params per
    bundle); in production this would be replaced by a JSON `Implied_ATM_Vol_History` block."""
    spot_history = bundle.get('spot_price_history') or {}
    factors = bundle.get('factors') or {}
    if not spot_history or not price_models:
        return {}
    import math as _math
    T = float(tenor_days) / utils.DAYS_IN_YEAR
    out = {}
    for commodity in spot_history.keys():
        spot_key = f'CommodityPrice:{commodity}'
        model_key = f'{factor_module}.{commodity}'
        if spot_key not in factors or model_key not in price_models:
            continue
        params = price_models[model_key]
        kappa = max(float(params['Kappa']), 1e-9)
        sigma = float(params['Sigma'])
        integrated_var = sigma * sigma * (1.0 - _math.exp(-2.0 * kappa * T)) / (2.0 * kappa)
        iv = _math.sqrt(max(integrated_var / max(T, 1e-9), 0.0))
        sim = factors[spot_key]
        T_full, B = sim.shape
        out[commodity] = torch.full((T_full, B), float(iv), dtype=torch.float32, device=sim.device)
    return out


def compute_spot_zscore(bundle, window=PRICE_ZSCORE_WINDOW, min_periods=5, eps=1e-6):
    """Per-commodity rolling z-score of underlying spot: (S - SMA_w(S)) / STD_w(S), per scenario.
    Captures overbought/oversold regime in the underlying.
    Returns {commodity: tensor (H+T, B)}."""
    spot_history = bundle.get('spot_price_history') or {}
    if not spot_history:
        return {}
    out = {}
    for commodity in spot_history.keys():
        S = _full_spot_timeline(bundle, commodity)
        if S is None:
            continue
        T = S.shape[0]
        cum = torch.cat([torch.zeros_like(S[:1]), S.cumsum(dim=0)], dim=0)
        sq_cum = torch.cat([torch.zeros_like(S[:1]), (S * S).cumsum(dim=0)], dim=0)
        idx = torch.arange(T + 1, device=S.device)
        lo = (idx - window).clamp_min(0)
        count = (idx - lo).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
        sum_w = cum[idx[1:]] - cum[lo[1:]]
        sum_sq_w = sq_cum[idx[1:]] - sq_cum[lo[1:]]
        n = count[1:]
        mean = sum_w / n
        var = (sum_sq_w / n) - mean * mean
        std = var.clamp_min(0.0).sqrt()
        z = (S - mean) / (std + eps)
        z[:min_periods] = 0.0
        out[commodity] = z
    return out


def compute_cross_delta(bundle, window=PRICE_ZSCORE_WINDOW, eps=1e-6):
    """Per-instrument PATH-LOCAL BACKWARD-LOOKING rolling hedge ratio. For each path b at time t:
        beta[t, b, c] = Σ_{s=max(0,t-window)..t-1} dP_c[s,b] * dL[s,b]
                        / Σ_{s=max(0,t-window)..t-1} dP_c[s,b]^2 + eps
    Uses only data observable on path b at or before time t — deployable in production where the
    agent has only its own realized history. No look-ahead (window ends at t-1, not t+1) and no
    cross-section privilege (regression is along the time axis of path b, not across paths)."""
    liability_mtm = bundle.get('liability_mtm')
    tradables = bundle.get('tradables') or {}
    if liability_mtm is None or not tradables:
        return {}
    L = liability_mtm.to(dtype=torch.float32)
    dL = L[1:] - L[:-1]
    out = {}
    for name, price in tradables.items():
        P = price.to(dtype=torch.float32)
        dP = P[1:] - P[:-1]
        T = P.shape[0]
        zeros = torch.zeros_like(dP[:1])
        # cumulative sums prepended with zero so cum_sum[t] = Σ_{s=0..t-1} (dP·dL)[s]
        num_cum = torch.cat([zeros, (dP * dL).cumsum(dim=0)], dim=0)
        den_cum = torch.cat([zeros, (dP * dP).cumsum(dim=0)], dim=0)
        idx = torch.arange(T, device=P.device)
        lo = (idx - window).clamp_min(0)
        num = num_cum[idx] - num_cum[lo]
        den = den_cum[idx] - den_cum[lo]
        out[name] = num / (den + eps)
    return out


def _legs_state(bundle, time_index, layout, batch_size, device):
    leg_dim = layout['legs']['feature_dim']
    id_dim = layout['legs']['id_dim']
    legs = bundle.get('legs') if bundle else None
    if not legs or not legs.get('ids'):
        return (
            torch.zeros((batch_size, 0, leg_dim), dtype=torch.float32, device=device),
            torch.zeros((batch_size, 0, id_dim), dtype=torch.long, device=device),
            torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
        )
    features_t = legs['features'][time_index].to(dtype=torch.float32, device=device)
    # Bundle legs may be (T, B, L, F) → indexed (B, L, F) — or (T, L, F) → indexed (L, F).
    # Normalize to (batch_size, L, F) by inserting a singleton batch dim when missing, then
    # broadcasting if the present batch dim is 1.
    if features_t.ndim == 2:
        features_t = features_t.unsqueeze(0)
    if features_t.shape[0] == 1 and features_t.shape[0] != batch_size:
        features_t = features_t.expand(batch_size, -1, -1).contiguous()
    registry = layout['registry']
    id_rows = [
        [_id_lookup(registry, vocab, code) for vocab, code in zip(LEG_ID_NAMES, ids)]
        for ids in legs['ids']
    ]
    ids_t = torch.tensor(id_rows, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    time_to_payment_idx = LEG_FEATURE_NAMES.index('time_to_payment')
    mask_t = features_t[..., time_to_payment_idx] > 0
    return features_t, ids_t, mask_t


def _instruments_state(bundle, state, time_index, layout, batch_size, device):
    meta = layout['instruments']['meta']
    feature_dim = layout['instruments']['feature_dim']
    id_dim = layout['instruments']['id_dim']
    if not meta:
        return (
            torch.zeros((batch_size, 0, feature_dim), dtype=torch.float32, device=device),
            torch.zeros((batch_size, 0, id_dim), dtype=torch.long, device=device),
            torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
        )
    base_date = pd.Timestamp(bundle['meta']['base_date'])
    current_day = float(bundle['time_grid_days'][time_index].item())
    columns = []
    ids_rows = []
    mask_rows = []
    cumulative_pnl_state = state.get('cumulative_pnl', {})
    time_held_state = state.get('time_held', {})
    cross_delta_bundle = bundle.get('cross_delta') or {}
    tradables_bundle = bundle.get('tradables') or {}
    prev_index = max(time_index - PRICE_MOMENTUM_LOOKBACK, 0)
    for entry in meta:
        name = entry['name']
        price = state['tradable_values'][name].to(device=device, dtype=torch.float32)
        position = state['positions'][name].to(device=device, dtype=torch.float32)
        contract_size = float(entry['contract_size'])
        position_value = position * price * contract_size
        last_trade_day = float((entry['last_trade_date'] - base_date).days)
        time_to_expiry = max(0.0, (last_trade_day - current_day)) / utils.DAYS_IN_YEAR
        tradable = 1.0 if current_day < last_trade_day else 0.0
        cs = torch.full_like(price, contract_size)
        tte = torch.full_like(price, time_to_expiry)
        flag = torch.full_like(price, tradable)
        cum_pnl = cumulative_pnl_state.get(name, torch.zeros_like(price)).to(device=device, dtype=torch.float32)
        held_steps = time_held_state.get(name, torch.zeros_like(price)).to(device=device, dtype=torch.float32)
        held_years = held_steps / utils.DAYS_IN_YEAR
        cd_tensor = cross_delta_bundle.get(name)
        if cd_tensor is None:
            cd = torch.zeros_like(price)
        else:
            cd = cd_tensor[time_index].to(device=device, dtype=torch.float32)
            if cd.shape[0] == 1 and cd.shape[0] != batch_size:
                cd = cd.expand(batch_size).contiguous()
        prev_tensor = tradables_bundle.get(name)
        if prev_tensor is None:
            price_change = torch.zeros_like(price)
        else:
            price_prev = prev_tensor[prev_index].to(device=device, dtype=torch.float32)
            if price_prev.shape[0] == 1 and price_prev.shape[0] != batch_size:
                price_prev = price_prev.expand(batch_size).contiguous()
            price_change = price - price_prev
        columns.append(torch.stack([price, price_change, position, position_value, tte, cs, flag, cum_pnl, held_years, cd], dim=-1))
        ids_rows.append([entry['currency_id'], entry['underlying_id'], entry['instrument_type_id']])
        mask_rows.append(True)
    features = torch.stack(columns, dim=1)
    ids = torch.tensor(ids_rows, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    mask = torch.tensor(mask_rows, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    return features, ids, mask


def _cash_state(bundle, state, layout, time_index, batch_size, device):
    meta = layout['cash_accounts']['meta']
    feature_dim = layout['cash_accounts']['feature_dim']
    id_dim = layout['cash_accounts']['id_dim']
    if not meta:
        return (
            torch.zeros((batch_size, 0, feature_dim), dtype=torch.float32, device=device),
            torch.zeros((batch_size, 0, id_dim), dtype=torch.long, device=device),
            torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
        )
    realized_cashflows = (bundle.get('realized_cashflows') or {}) if bundle else {}
    prev_index = max(time_index - 1, 0)
    columns = []
    ids_rows = []
    for entry in meta:
        name = entry['name']
        currency = entry['currency']
        balance = state['cash_accounts'][name].to(device=device, dtype=torch.float32)
        margin = state['margin_accounts'].get(name, torch.zeros_like(balance)).to(device=device, dtype=torch.float32)
        cf_tensor = realized_cashflows.get(currency)
        if cf_tensor is None:
            cf_now = torch.zeros_like(balance)
            cf_prev = torch.zeros_like(balance)
        else:
            cf_now = cf_tensor[time_index].to(device=device, dtype=torch.float32)
            cf_prev = cf_tensor[prev_index].to(device=device, dtype=torch.float32)
            if cf_now.shape[0] == 1 and cf_now.shape[0] != batch_size:
                cf_now = cf_now.expand(batch_size).contiguous()
                cf_prev = cf_prev.expand(batch_size).contiguous()
        columns.append(torch.stack([balance, margin, cf_now, cf_prev], dim=-1))
        ids_rows.append([entry['currency_id']])
    features = torch.stack(columns, dim=1)
    ids = torch.tensor(ids_rows, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    mask = torch.ones((batch_size, len(meta)), dtype=torch.bool, device=device)
    return features, ids, mask


def _globals_state(bundle, state, layout, time_index, batch_size, device):
    time_grid_days = bundle['time_grid_days'].to(device=device, dtype=torch.float32)
    horizon_days = (time_grid_days[-1] - time_grid_days[time_index]).clamp_min(0.0)
    time_to_horizon = (horizon_days / utils.DAYS_IN_YEAR).expand(batch_size).reshape(batch_size, 1)
    liability_mtm_tensor = bundle.get('liability_mtm') if bundle else None
    if liability_mtm_tensor is None:
        mtm_now = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        mtm_prev = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    else:
        prev_index = max(time_index - 1, 0)
        mtm_now = liability_mtm_tensor[time_index].to(device=device, dtype=torch.float32).reshape(-1, 1)
        mtm_prev = liability_mtm_tensor[prev_index].to(device=device, dtype=torch.float32).reshape(-1, 1)
        if mtm_now.shape[0] == 1 and mtm_now.shape[0] != batch_size:
            mtm_now = mtm_now.expand(batch_size, 1).contiguous()
            mtm_prev = mtm_prev.expand(batch_size, 1).contiguous()
    # Portfolio-level aggregates over instruments, bucketed by time-to-expiry.
    base_date = pd.Timestamp(bundle['meta']['base_date'])
    current_day = float(time_grid_days[time_index].item())
    total_pv = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    near_pv = torch.zeros_like(total_pv)
    mid_pv = torch.zeros_like(total_pv)
    far_pv = torch.zeros_like(total_pv)
    for entry in layout['instruments']['meta']:
        name = entry['name']
        price = state['tradable_values'][name].to(device=device, dtype=torch.float32)
        position = state['positions'][name].to(device=device, dtype=torch.float32)
        contract_size = float(entry['contract_size'])
        pv = (position * price * contract_size).reshape(-1, 1)
        total_pv = total_pv + pv
        ttf_days = float((entry['last_trade_date'] - base_date).days) - current_day
        if ttf_days < 60.0:
            near_pv = near_pv + pv
        elif ttf_days < 180.0:
            mid_pv = mid_pv + pv
        else:
            far_pv = far_pv + pv
    residual_exposure = total_pv + mtm_now
    coverage_ratio = -total_pv / mtm_now.abs().clamp_min(1.0)

    def _commodity_scalar(bundle_key, commodity):
        store = bundle.get(bundle_key) or {}
        if commodity not in store:
            return torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        slice_t = store[commodity][time_index].to(device=device, dtype=torch.float32).reshape(-1, 1)
        if slice_t.shape[0] == 1 and slice_t.shape[0] != batch_size:
            slice_t = slice_t.expand(batch_size, 1).contiguous()
        return slice_t

    referenced = layout['globals'].get('referenced_commodities', ())
    extras = []
    for commodity in referenced:
        for bundle_key, _prefix in PER_COMMODITY_GLOBAL_FEATURES:
            extras.append(_commodity_scalar(bundle_key, commodity))
    return torch.cat(
        [time_to_horizon, mtm_now, mtm_prev, total_pv, residual_exposure, coverage_ratio,
         near_pv, mid_pv, far_pv, *extras],
        dim=1,
    )


def build_entity_state(bundle, state, time_index, layout):
    positions = state.get('positions') or {}
    if positions:
        first = next(iter(positions.values()))
        batch_size = int(first.shape[0])
        device = first.device
    else:
        batch_size = int(bundle['time_grid_days'].shape[0])
        device = bundle['time_grid_days'].device

    legs_features, legs_ids, legs_mask = _legs_state(bundle, time_index, layout, batch_size, device)
    instr_features, instr_ids, instr_mask = _instruments_state(bundle, state, time_index, layout, batch_size, device)
    cash_features, cash_ids, cash_mask = _cash_state(bundle, state, layout, time_index, batch_size, device)
    globals_features = _globals_state(bundle, state, layout, time_index, batch_size, device)

    return TensorDict({
        'legs': legs_features,
        'legs_ids': legs_ids,
        'legs_mask': legs_mask,
        'instruments': instr_features,
        'instruments_ids': instr_ids,
        'instruments_mask': instr_mask,
        'cash_accounts': cash_features,
        'cash_accounts_ids': cash_ids,
        'cash_accounts_mask': cash_mask,
        'globals': globals_features,
    }, batch_size=[batch_size])


def flatten_entity_features(entity_state):
    parts = []
    for name in ('legs', 'instruments', 'cash_accounts'):
        features = entity_state[name]
        if features.shape[1] == 0:
            continue
        mask = entity_state[name + '_mask'].unsqueeze(-1).to(features.dtype)
        parts.append((features * mask).flatten(start_dim=1))
    parts.append(entity_state['globals'])
    return torch.cat(parts, dim=1)


def entity_feature_dim(layout):
    return (
        layout['legs']['feature_dim'] * layout['legs']['max_cardinality']
        + layout['instruments']['feature_dim'] * layout['instruments']['max_cardinality']
        + layout['cash_accounts']['feature_dim'] * layout['cash_accounts']['max_cardinality']
        + layout['globals']['feature_dim']
    )


if __name__ == '__main__':
    pass
