import pandas as pd
import torch
from tensordict import TensorDict

from . import utils


LEG_FEATURE_NAMES = (
    'fixed_basis',
    'volume',
    'time_to_period_start',
    # time_to_period_end pruned — for our deal types it differs from `time_to_payment`
    # by a fixed settlement-lag offset and is therefore highly correlated. Reintroduce
    # for deal structures with non-trivial period-end-to-payment dynamics (deferred
    # settlement, conditional payments).
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
    # contract_size pruned — constant per instrument and already implicit via the
    # categorical `instrument_type_id` embedding; never varies during a rollout.
    'is_tradable_now',
    'cumulative_pnl',
    'time_held',
    # cross_delta pruned — was a per-path rolling-regression proxy for d(MtM)/d(spot)
    # over the per-instrument tradable; the AAD-derived `hedge_ratio_*` globals now
    # provide the exact derivative, making cross_delta the noisy version of a feature
    # we have cleanly.
    'basis_zscore_20',  # rolling z-score of (F_i - S) over 20 bd — tenor-distinct
                        # carry/richness signal attached to each future's own token.
)

PRICE_MOMENTUM_LOOKBACK = 1
PRICE_ZSCORE_WINDOW = 20

INSTRUMENT_ID_NAMES = ('currency', 'underlying', 'instrument_type')

CASH_FEATURE_NAMES = ('balance', 'margin_balance', 'realized_cashflow_today')
CASH_ID_NAMES = ('currency',)

GLOBAL_BASE_FEATURE_NAMES = (
    'time_to_horizon',
    'liability_mtm',
    # liability_mtm_prev intentionally pruned — `liability_mtm_change_1d` lookback already
    # carries the delta (the only informative quantity); the lagged absolute level is
    # redundant.
    'total_position_value',     # sum of position_value across all instruments (negative if net short)
    # residual_exposure intentionally pruned — derivable from `liability_mtm + total_position_value`
    # in one linear unit; carrying it is a free hint that costs a correlated input.
    'coverage_ratio',            # -(Σ position × contract_size) / Σ |leg.Volume| — unitless delta coverage; 1.0 = fully hedged short
    'near_position_value',       # sum of position_value for instruments with ttf < 60d
    'mid_position_value',        # 60d <= ttf < 180d
    'far_position_value',        # ttf >= 180d
)

# Per-commodity globals appended after the base block, suffixed with the commodity name.
# (bundle_key, name_prefix) pairs — order is preserved in feature names.
PER_COMMODITY_GLOBAL_FEATURES = (
    # spot_zscore_20d intentionally pruned — vol-normalised `spot_stretch_20d` below
    # supersedes it (both are mean-reversion proximity; stretch's RV-normalised denominator
    # adapts to vol regime more cleanly than zscore's price-STD denominator).
    ('spot_realized_vol', 'spot_realized_vol_20d'),
    # Direction-aware trend stack — signed log-return at two horizons + vol-normalised
    # stretch against the 20d moving average. The realised-vol feature above is
    # direction-blind; these tell the policy which way the underlying is moving and
    # whether current price is stretched relative to its recent path.
    ('spot_trend_20', 'spot_trend_20d'),
    ('spot_trend_60', 'spot_trend_60d'),
    ('spot_stretch_20', 'spot_stretch_20d'),
)

# Trajectory lookbacks (business days) on liability_mtm and per-commodity spot. Generic
# regime/momentum stack — deliberately not tuned to any deal calendar. Encodes how rapidly
# the underlying / liability has been moving so the policy can distinguish "stable at $X"
# from "rapidly evolving toward $X".
LOOKBACK_WINDOWS = (1, 5, 20, 60)
# Spot-only lookback windows. The 20d and 60d horizons are omitted because the
# direction-aware `spot_trend_20d`/`spot_trend_60d` features carry the same information
# annualised. The 5d horizon is also omitted — for monotone-MtM deals (which ours is)
# `liability_mtm_change_5d` overlaps heavily with it. Only the 1d short-momentum signal
# remains, since no other feature covers it.
SPOT_LOOKBACK_WINDOWS = (1,)

# Temporal-encoder window: number of business days of history fed into the 1D-conv
# trajectory token. The conv learns its own temporal weighting (vs the hand-picked
# LOOKBACK_WINDOWS above). The minimum 20-day window covers enough lookback for the
# 60-day signal to also fit at run time once the H-row history prefix is included.
TEMPORAL_WINDOW = 20


def build_global_feature_names(referenced_commodities, hedge_ratio_pairs=()):
    names = list(GLOBAL_BASE_FEATURE_NAMES)
    for commodity in referenced_commodities:
        for _bundle_key, prefix in PER_COMMODITY_GLOBAL_FEATURES:
            names.append(f'{prefix}_{commodity}')
    for tradable, commodity in hedge_ratio_pairs:
        names.append(f'hedge_ratio_{tradable}_{commodity}')
    # Lookback diagnostics: liability MtM change at all windows (no redundancy elsewhere)
    # + per-commodity short-horizon spot log-return (long-horizon spot covered by trend stack).
    for w in LOOKBACK_WINDOWS:
        names.append(f'liability_mtm_change_{w}d')
    for w in SPOT_LOOKBACK_WINDOWS:
        for commodity in referenced_commodities:
            names.append(f'spot_logret_{w}d_{commodity}')
    return tuple(names)


def _intern(registry, vocab, name):
    table = registry.setdefault(vocab, {})
    if name not in table:
        table[name] = len(table)
    return table[name]


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
            'last_trade_date': spec['last_trade_date'],
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

    referenced_commodities = tuple(runtime.get('referenced_commodities', ()))
    # Hedge-ratio globals: one scalar per (hedge tradable, factor) pair. `factor_name` is the
    # full dotted factor name produced by `_collect_referenced_commodities` (e.g.
    # 'CommodityPrice.PLATINUM'); it keys directly into bundle['hedge_ratios'][tradable].
    # Pair order is deterministic: hedges × commodities, fixed by hedge_names iteration order.
    hedge_ratio_pairs = tuple(
        (tradable, factor_name)
        for tradable in hedge_names
        for factor_name in referenced_commodities
    )
    global_feature_names = build_global_feature_names(referenced_commodities, hedge_ratio_pairs)

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
            'hedge_ratio_pairs': hedge_ratio_pairs,
        },
        # Temporal trajectory token: 1D-conv-encoded summary of the last K business days of
        # (per-commodity log-spot, liability MtM). Channel ordering matches the iteration
        # order of referenced_commodities + the global liability_mtm appended last.
        'temporal': {
            'window': TEMPORAL_WINDOW,
            'n_channels': len(referenced_commodities) + (1 if 'liability_mtm' in GLOBAL_BASE_FEATURE_NAMES else 0),
            'channel_names': tuple(
                [f'log_spot_{c}' for c in referenced_commodities] + ['liability_mtm']
            ),
        },
    }


def _id_lookup(registry, vocab, name):
    return registry[vocab].get(str(name), 0)


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


def _temporal_window(bundle, time_index, batch_size, device, *, window=TEMPORAL_WINDOW, commodities=None):
    """Build a (B, K, n_features) tensor of the last `window` business days of trajectory
    signals at the current decision step. Features per timepoint: per-commodity spot prices
    (using the full history+sim timeline) followed by the global liability MtM. The history
    prefix added at bundle build (`_prepend_history_prefix`) ensures the lookback is always
    in-bounds for any sim-time decision step.

    Channel ordering: caller passes `commodities` (typically
    `layout['globals']['referenced_commodities']`) so the runtime channel order matches the
    metadata `channel_names` exactly. Falls back to `bundle['spot_price_history']` keys when
    not supplied — equivalent only when the two iteration orders coincide (single-commodity
    or accidental dict-order alignment)."""
    spot_history = bundle.get('spot_price_history') or {}
    liability_mtm = bundle.get('liability_mtm')
    K = int(window)
    t_end = int(time_index) + 1
    t_start = max(0, t_end - K)
    channels = []
    iter_order = tuple(commodities) if commodities is not None else tuple(spot_history.keys())
    for commodity in iter_order:
        S = _full_spot_timeline(bundle, commodity)
        if S is None:
            continue
        # Log-prices: stationary scale, robust to spot levels across commodities.
        slice_t = S[t_start:t_end].clamp_min(1e-9).log().to(device=device, dtype=torch.float32)
        if slice_t.shape[0] < K:
            pad = slice_t[:1].expand(K - slice_t.shape[0], -1)
            slice_t = torch.cat([pad, slice_t], dim=0)
        channels.append(slice_t)
    if liability_mtm is not None:
        slice_t = liability_mtm[t_start:t_end].to(device=device, dtype=torch.float32)
        if slice_t.shape[0] < K:
            pad = slice_t[:1].expand(K - slice_t.shape[0], -1)
            slice_t = torch.cat([pad, slice_t], dim=0)
        channels.append(slice_t)
    if not channels:
        return torch.zeros((batch_size, K, 0), dtype=torch.float32, device=device)
    # Stack to (n_features, K, B) then permute to (B, K, n_features).
    stacked = torch.stack(channels, dim=0).permute(2, 1, 0).contiguous()
    return stacked


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


def compute_spot_trend(bundle, *, window):
    """Annualised log-return over `window` business days, per commodity.

        trend_W(t) = log(S(t) / S(t-W)) * (252 / W)

    Signed direction-and-magnitude trend signal — positive for uptrend, negative for
    downtrend, magnitude comparable across windows because of the 252/W scaling. Filled
    with zero for `t < window` (no full lookback yet)."""
    spot_history = bundle.get('spot_price_history') or {}
    if not spot_history:
        return {}
    out = {}
    for commodity in spot_history.keys():
        S = _full_spot_timeline(bundle, commodity)
        if S is None:
            continue
        log_S = S.clamp_min(1e-9).log()
        T = log_S.shape[0]
        trend = torch.zeros_like(log_S)
        if T > window:
            trend[window:] = (log_S[window:] - log_S[:-window]) * (252.0 / float(window))
        out[commodity] = trend
    return out


def compute_spot_stretch(bundle, *, window):
    """Vol-normalised stretch from the rolling moving average:

        stretch_W(t) = (S(t) - MA_W(t)) / (RV_W(t) * S(t))

    Tells the policy how far current price is from its recent path, normalised by the
    typical move size in that path — values around ±0.5 are noise, ±2 is meaningful
    stretch (mean-reversion candidate). The denominator normalises across vol regimes:
    during high-vol periods, larger absolute deviations are normal.

    Reuses `bundle['spot_realized_vol']` if available (which the calculation already
    computes at the same default 20d window); otherwise falls back to recomputing.
    """
    spot_history = bundle.get('spot_price_history') or {}
    if not spot_history:
        return {}
    rv_dict = bundle.get('spot_realized_vol') or {}
    out = {}
    for commodity in spot_history.keys():
        S = _full_spot_timeline(bundle, commodity)
        if S is None:
            continue
        T = S.shape[0]
        # Vectorised rolling mean via cumulative sum: MA_W(t) = (cum[t+1] - cum[max(0, t-W+1)]) / count
        cum = torch.cat([torch.zeros_like(S[:1]), S.cumsum(dim=0)], dim=0)  # (T+1, B)
        idx = torch.arange(T, device=S.device)
        lo = (idx - window + 1).clamp_min(0)
        count = (idx - lo + 1).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
        ma = (cum[idx + 1] - cum[lo]) / count
        rv = rv_dict.get(commodity)
        if rv is None:
            # No RV available; fall back to price-std normalisation (≈ what spot_zscore uses).
            sq = (S - ma).pow(2)
            cum_sq = torch.cat([torch.zeros_like(sq[:1]), sq.cumsum(dim=0)], dim=0)
            window_sq = cum_sq[idx + 1] - cum_sq[lo]
            std = (window_sq / count).clamp_min(1e-12).sqrt()
            denom = std
        else:
            denom = (rv * S).clamp_min(1e-9)
        out[commodity] = (S - ma) / denom
    return out


def compute_basis_zscore(bundle, *, window=20, eps=1e-9):
    """Per-future basis z-score over a rolling window:

        basis_i(t)   = F_i(t) - S(t)
        zscore_i(t)  = (basis_i(t) - MA_W(basis_i)) / std_W(basis_i)

    Returns dict {instrument_name → (T, B) z-score tensor}.

    Tells the policy which contract is rich/cheap vs its own recent history — a
    tenor-distinct carry signal attached to each future's own token (the global
    per-commodity carry slots were dropped once this was in place).

    Single-commodity assumption: all instruments are taken to track the first
    spot factor in `bundle['spot_price_history']`. Multi-commodity support would
    require threading a per-instrument underlying mapping through the bundle.

    Note on the history prefix: for the first `H` rows the bundle replaces F_i
    with the spot history, so `basis ≡ 0` there. The 20-bd rolling stats are
    therefore mildly biased for the first ~20 sim days; clean from sim-day 20
    onwards.
    """
    spot_history = bundle.get('spot_price_history') or {}
    tradables = bundle.get('tradables') or {}
    if not spot_history or not tradables:
        return {}
    commodity = next(iter(spot_history))
    S = _full_spot_timeline(bundle, commodity)
    if S is None:
        return {}
    out = {}
    for name, F_i in tradables.items():
        F = F_i.to(dtype=torch.float32)
        S_t = S.to(dtype=torch.float32, device=F.device)
        basis = F - S_t                                                 # (T, B)
        T = basis.shape[0]
        cum = torch.cat([torch.zeros_like(basis[:1]), basis.cumsum(dim=0)], dim=0)
        idx = torch.arange(T, device=basis.device)
        lo = (idx - window + 1).clamp_min(0)
        count = (idx - lo + 1).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
        ma = (cum[idx + 1] - cum[lo]) / count
        sq = (basis - ma).pow(2)
        cum_sq = torch.cat([torch.zeros_like(sq[:1]), sq.cumsum(dim=0)], dim=0)
        window_sq = cum_sq[idx + 1] - cum_sq[lo]
        std = (window_sq / count).clamp_min(eps).sqrt()
        out[name] = (basis - ma) / std
    return out


def _legs_state(bundle, time_index, layout, batch_size, device):
    leg_dim = layout['legs']['feature_dim']
    id_dim = layout['legs']['id_dim']
    legs = bundle.get('legs')
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
    mask_t = torch.ones(features_t.shape[:2], dtype=torch.bool, device=device)
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
    current_day = float(bundle['time_grid_days_cpu'][time_index])
    columns = []
    ids_rows = []
    cumulative_pnl_state = state.get('cumulative_pnl', {})
    time_held_state = state.get('time_held', {})
    tradables_bundle = bundle.get('tradables') or {}
    basis_zscore_bundle = bundle.get('basis_zscore_20') or {}
    prev_index = max(time_index - PRICE_MOMENTUM_LOOKBACK, 0)
    for entry in meta:
        name = entry['name']
        price = state['tradable_values'][name].to(device=device, dtype=torch.float32)
        position = state['positions'][name].to(device=device, dtype=torch.float32)
        contract_size = float(entry['contract_size'])
        position_value = position * price * contract_size
        last_trade_day = float((entry['last_trade_date'] - base_date).days)
        time_to_expiry = (last_trade_day - current_day) / utils.DAYS_IN_YEAR
        tradable = 1.0 if current_day < last_trade_day else 0.0
        tte = torch.full_like(price, time_to_expiry)
        flag = torch.full_like(price, tradable)
        cum_pnl = cumulative_pnl_state.get(name, torch.zeros_like(price)).to(device=device, dtype=torch.float32)
        held_steps = time_held_state.get(name, torch.zeros_like(price)).to(device=device, dtype=torch.float32)
        held_years = held_steps / utils.DAYS_IN_YEAR
        prev_tensor = tradables_bundle.get(name)
        if prev_tensor is None:
            price_change = torch.zeros_like(price)
        else:
            price_prev = prev_tensor[prev_index].to(device=device, dtype=torch.float32)
            if price_prev.shape[0] == 1 and price_prev.shape[0] != batch_size:
                price_prev = price_prev.expand(batch_size).contiguous()
            price_change = price - price_prev
        bz_tensor = basis_zscore_bundle.get(name)
        if bz_tensor is None:
            basis_z = torch.zeros_like(price)
        else:
            basis_z = bz_tensor[time_index].to(device=device, dtype=torch.float32)
            if basis_z.shape[0] == 1 and basis_z.shape[0] != batch_size:
                basis_z = basis_z.expand(batch_size).contiguous()
        columns.append(torch.stack([price, price_change, position, position_value, tte, flag, cum_pnl, held_years, basis_z], dim=-1))
        ids_rows.append([entry['currency_id'], entry['underlying_id'], entry['instrument_type_id']])
    features = torch.stack(columns, dim=1)
    ids = torch.tensor(ids_rows, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    mask = torch.ones((batch_size, len(meta)), dtype=torch.bool, device=device)
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
    realized_cashflows = bundle.get('realized_cashflows') or {}
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
            cf_today = torch.zeros_like(balance)
        else:
            cf_now = cf_tensor[time_index].to(device=device, dtype=torch.float32)
            cf_prev = cf_tensor[prev_index].to(device=device, dtype=torch.float32)
            if cf_now.shape[0] == 1 and cf_now.shape[0] != batch_size:
                cf_now = cf_now.expand(batch_size).contiguous()
                cf_prev = cf_prev.expand(batch_size).contiguous()
            # `cf_tensor` is the cumulative-since-inception cashflow per currency. The agent
            # only needs today's flow (cumulative levels are recoverable from balance + margin
            # already in this token); replacing two correlated cumulative scalars with one
            # delta drops 1 feature and gives a sharper signal.
            cf_today = cf_now - cf_prev
        columns.append(torch.stack([balance, margin, cf_today], dim=-1))
        ids_rows.append([entry['currency_id']])
    features = torch.stack(columns, dim=1)
    ids = torch.tensor(ids_rows, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    mask = torch.ones((batch_size, len(meta)), dtype=torch.bool, device=device)
    return features, ids, mask


def _globals_state(bundle, state, layout, time_index, batch_size, device):
    time_grid_days = bundle['time_grid_days'].to(device=device, dtype=torch.float32)
    horizon_days = (time_grid_days[-1] - time_grid_days[time_index]).clamp_min(0.0)
    time_to_horizon = (horizon_days / utils.DAYS_IN_YEAR).expand(batch_size).reshape(batch_size, 1)
    liability_mtm_tensor = bundle.get('liability_mtm')
    if liability_mtm_tensor is None:
        mtm_now = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    else:
        mtm_now = liability_mtm_tensor[time_index].to(device=device, dtype=torch.float32).reshape(-1, 1)
        if mtm_now.shape[0] == 1 and mtm_now.shape[0] != batch_size:
            mtm_now = mtm_now.expand(batch_size, 1).contiguous()
    # Portfolio-level aggregates over instruments, bucketed by time-to-expiry.
    base_date = pd.Timestamp(bundle['meta']['base_date'])
    current_day = float(bundle['time_grid_days_cpu'][time_index])
    total_pv = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    near_pv = torch.zeros_like(total_pv)
    mid_pv = torch.zeros_like(total_pv)
    far_pv = torch.zeros_like(total_pv)
    total_oz = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    for entry in layout['instruments']['meta']:
        name = entry['name']
        price = state['tradable_values'][name].to(device=device, dtype=torch.float32)
        position = state['positions'][name].to(device=device, dtype=torch.float32)
        contract_size = float(entry['contract_size'])
        pv = (position * price * contract_size).reshape(-1, 1)
        total_pv = total_pv + pv
        total_oz = total_oz + (position * contract_size).reshape(-1, 1)
        ttf_days = float((entry['last_trade_date'] - base_date).days) - current_day
        if ttf_days < 60.0:
            near_pv = near_pv + pv
        elif ttf_days < 180.0:
            mid_pv = mid_pv + pv
        else:
            far_pv = far_pv + pv
    # Delta coverage: fraction of leg notional (in underlying units) covered by the hedge book.
    # Unitless and bounded around [-1, 0] for short hedges in the regime we care about; 1.0 means
    # the short book exactly offsets the leg's volume. Cached on the bundle at build time —
    # constant across the episode, so no per-step .sum().item() sync.
    coverage_ratio = -total_oz / max(float(bundle.get('total_leg_volume', 0.0)), 1.0)

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
    # Hedge-ratio scalars per (tradable, factor). `factor_name` here is the full dotted
    # factor name (e.g. 'CommodityPrice.PLATINUM') matching bundle['hedge_ratios'] keys.
    # Missing entries contribute zero and the policy learns to ignore them.
    hedge_ratios = bundle.get('hedge_ratios') or {}
    for tradable, factor_name in layout['globals'].get('hedge_ratio_pairs', ()):
        by_factor = hedge_ratios.get(tradable) or {}
        tensor = by_factor.get(factor_name)
        if tensor is None:
            extras.append(torch.zeros((batch_size, 1), dtype=torch.float32, device=device))
        else:
            slice_t = tensor[time_index].to(device=device, dtype=torch.float32).reshape(-1, 1)
            if slice_t.shape[0] == 1 and slice_t.shape[0] != batch_size:
                slice_t = slice_t.expand(batch_size, 1).contiguous()
            extras.append(slice_t)
    # Lookback features: liability MtM change at all windows; per-commodity spot
    # log-return only at short horizons (long-horizon spot returns are redundant with the
    # annualised `spot_trend_20d`/`spot_trend_60d` features handled via PER_COMMODITY_GLOBAL_FEATURES).
    # The bundle already has `H` rows of history-prefix prepended, so all windows are
    # in-bounds; the 60-day MtM window can underflow for the first ~30 sim decisions
    # (clamp to row 0 → change collapses to zero, "no info yet").
    liability_mtm_full = bundle.get('liability_mtm')
    for w in LOOKBACK_WINDOWS:
        prev_t = max(0, int(time_index) - int(w))
        if liability_mtm_full is not None:
            mtm_then = liability_mtm_full[prev_t].to(device=device, dtype=torch.float32).reshape(-1, 1)
            if mtm_then.shape[0] == 1 and mtm_then.shape[0] != batch_size:
                mtm_then = mtm_then.expand(batch_size, 1).contiguous()
            extras.append(mtm_now - mtm_then)
        else:
            extras.append(torch.zeros((batch_size, 1), dtype=torch.float32, device=device))
    for w in SPOT_LOOKBACK_WINDOWS:
        prev_t = max(0, int(time_index) - int(w))
        for commodity in referenced:
            S = _full_spot_timeline(bundle, commodity)
            if S is None:
                extras.append(torch.zeros((batch_size, 1), dtype=torch.float32, device=device))
                continue
            S_now = S[int(time_index)].to(device=device, dtype=torch.float32).reshape(-1, 1)
            S_then = S[prev_t].to(device=device, dtype=torch.float32).reshape(-1, 1)
            extras.append(S_now.clamp_min(1e-9).log() - S_then.clamp_min(1e-9).log())
    return torch.cat(
        [time_to_horizon, mtm_now, total_pv, coverage_ratio,
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
    temporal = _temporal_window(
        bundle, time_index, batch_size, device,
        window=int(layout.get('temporal', {}).get('window', TEMPORAL_WINDOW)),
        commodities=layout.get('globals', {}).get('referenced_commodities'),
    )

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
        'temporal': temporal,
    }, batch_size=[batch_size])


if __name__ == '__main__':
    pass
