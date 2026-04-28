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
    'inventory',
    'position_value',
    'time_to_expiry',
    'contract_size',
    'is_tradable_now',
)
INSTRUMENT_ID_NAMES = ('currency', 'underlying', 'instrument_type')

CASH_FEATURE_NAMES = ('balance', 'margin_balance', 'realized_cashflow', 'realized_cashflow_prev')
CASH_ID_NAMES = ('currency',)

GLOBAL_FEATURE_NAMES = ('time_to_horizon', 'liability_mtm', 'liability_mtm_prev')


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
            'feature_names': GLOBAL_FEATURE_NAMES,
            'feature_dim': len(GLOBAL_FEATURE_NAMES),
        },
    }


def _id_lookup(registry, vocab, name):
    return registry[vocab].get(str(name), 0)


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
        columns.append(torch.stack([price, position, position_value, tte, cs, flag], dim=-1))
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


def _globals_state(bundle, time_index, batch_size, device):
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
    return torch.cat([time_to_horizon, mtm_now, mtm_prev], dim=1)


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
    globals_features = _globals_state(bundle, time_index, batch_size, device)

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
