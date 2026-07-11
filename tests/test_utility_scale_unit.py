"""Utility-scale plumbing tests — re-homed live-code coverage from the deleted RL-era
test_symlog_legacy_identical.py (the band tests died with the band; these guard code that
survives: _mirror_utility_scale_to_runtime, resolve_utility_scale, _utility_wrap_signed's
fail-loud missing-scale contract)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from riskflow.hedge_bundle import (
    _mirror_utility_scale_to_runtime, _utility_wrap_signed, resolve_utility_scale)


def make_legacy_runtime():
    return {"objective": {"object": "terminalfloorthensurplusutility", "utility_scale": 1.0e6}}


def test_mirror_utility_scale_legacy_safe():
    """Mirroring utility_scale onto a legacy runtime caches the value but the legacy
    objective stays identity — the utility gate is False regardless of the cached scale."""
    runtime = make_legacy_runtime()
    bundle = {"utility_scale": 9.99e9}
    _mirror_utility_scale_to_runtime(bundle, runtime)
    assert runtime["objective"]["utility_scale"] == 9.99e9
    x = torch.tensor([1.0e6])
    assert _utility_wrap_signed(x, runtime).item() == 1.0e6  # identity for legacy


def test_mirror_runtime_no_objective():
    """Mirror gracefully no-ops when runtime has no objective dict."""
    runtime = {"objective": None}
    _mirror_utility_scale_to_runtime({"utility_scale": 1.0e6}, runtime)
    assert runtime["objective"] is None


def test_mirror_bundle_no_utility_scale():
    """Mirror gracefully no-ops when bundle has no utility_scale (cached pre-change bundle)."""
    runtime = make_legacy_runtime()
    saved = runtime["objective"]["utility_scale"]
    _mirror_utility_scale_to_runtime({}, runtime)
    assert runtime["objective"]["utility_scale"] == saved


def test_resolve_utility_scale_fails_loud_on_symlog_degeneracies():
    """Each silent-degrade path in resolve_utility_scale must raise under a utility
    objective (a $1k floor silently breaks tail compression); legacy returns the floor."""
    def runtime_with_object(obj_name, **objective_extras):
        return {'objective': {'object': obj_name, **objective_extras},
                'history_lookback_business_days': 0}

    sym = lambda **kw: runtime_with_object('asymmetricutility_symlog', **kw)
    legacy = lambda **kw: runtime_with_object('terminalfloorthensurplusutility', **kw)

    # Path 1: last_settlement_index missing
    bundle = {'total_leg_volume': 2500.0, 'spot_price_history': {}, 'spot_realized_vol': {}}
    try:
        resolve_utility_scale(bundle, sym())
        raise AssertionError("symlog should raise on missing last_settlement_index")
    except ValueError as e:
        assert 'last_settlement_index' in str(e), e
    assert resolve_utility_scale(bundle, legacy()) == 1.0e3

    # Path 2: empty spot_price_history
    bundle = {'last_settlement_index': 200, 'total_leg_volume': 2500.0,
              'spot_price_history': {}, 'spot_realized_vol': {}}
    try:
        resolve_utility_scale(bundle, sym())
        raise AssertionError("symlog should raise on empty spot_price_history")
    except ValueError as e:
        assert 'spot_price_history' in str(e), e
    assert resolve_utility_scale(bundle, legacy()) == 1.0e3

    # Path 3: zero total_leg_volume
    bundle = {'last_settlement_index': 200, 'total_leg_volume': 0.0,
              'spot_price_history': {'CommodityPrice.X': torch.zeros(1)},
              'spot_realized_vol': {}}
    try:
        resolve_utility_scale(bundle, sym())
        raise AssertionError("symlog should raise on zero total_leg_volume")
    except ValueError as e:
        assert 'total_leg_volume' in str(e), e
    assert resolve_utility_scale(bundle, legacy()) == 1.0e3

    # Explicit override always honored (no degeneracy check fires).
    assert resolve_utility_scale({}, sym(utility_scale_explicit=5_000_000.0)) == 5_000_000.0
    assert resolve_utility_scale({}, legacy(utility_scale_explicit=5_000_000.0)) == 5_000_000.0


def test_unknown_utility_scale_mode_fails_loud():
    """Typo in Utility_Scale_Mode raises with a message naming the typo'd value."""
    bundle = {'last_settlement_index': 200, 'total_leg_volume': 2500.0,
              'spot_price_history': {}, 'spot_realized_vol': {}}
    runtime = {'objective': {'utility_scale_mode': 'vol_scled_notional'},  # typo
               'history_lookback_business_days': 0}
    try:
        resolve_utility_scale(bundle, runtime)
    except ValueError as e:
        assert 'vol_scled_notional' in str(e) and 'Supported modes' in str(e), e
    else:
        raise AssertionError("resolve_utility_scale should have raised on unknown mode")
    runtime['objective']['utility_scale_mode'] = 'vol_scaled_notional'
    resolve_utility_scale(bundle, runtime)
    del runtime['objective']['utility_scale_mode']
    resolve_utility_scale(bundle, runtime)


def test_utility_missing_scale_fails_loud():
    """A utility objective without utility_scale must fail loud in _utility_wrap_signed —
    a silent default would give wrong-but-plausible utilities."""
    runtime = {"objective": {"object": "asymmetricutility_symlog"}}  # NO utility_scale
    try:
        _utility_wrap_signed(torch.tensor([1.0e6]), runtime)
    except ValueError as e:
        assert "utility_scale" in str(e), e
    else:
        raise AssertionError("_utility_wrap_signed should have raised on missing utility_scale")
    runtime["objective"]["utility_scale"] = 1.0e6
    _utility_wrap_signed(torch.tensor([1.0e6]), runtime)  # now OK
