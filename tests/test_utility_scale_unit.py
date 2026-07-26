"""Utility-scale plumbing tests: the frame stage that resolves + LOCKS c onto the runtime
(`Bundle._resolve_frame`), the fail-loud degeneracies of `Bundle._resolve_utility_scale`, and
`_utility_wrap_signed`'s fail-loud missing-scale contract."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from riskflow.hedge_bundle import Bundle, _utility_wrap_signed


def frame_runtime(objective):
    """The minimal runtime `Bundle._resolve_frame` reads: an objective, the accounting switches
    that gate the vol series, and the referenced-commodity set."""
    return {'objective': objective,
            'accounting': {'bid_offer_spread_spec': None, 'im_funding_spread_bps': 0.0},
            'referenced_commodities': ()}


def test_frame_locks_and_mirrors_the_utility_scale():
    """The frame stage resolves c ONCE and mirrors it onto the runtime objective — that mirrored
    number is what every reward/penalty divides by, so the transform must agree with the bundle."""
    runtime = frame_runtime({'object': 'asymmetricutility_symlog',
                             'utility_scale_explicit': 5.0e6})
    bundle = Bundle()
    bundle._resolve_frame(runtime, {})
    assert bundle.utility_scale == 5.0e6
    assert runtime['objective']['utility_scale'] == 5.0e6
    u = float(_utility_wrap_signed(torch.tensor([5.0e6]), runtime).item())
    assert abs(u - math.log1p(1.0)) < 1e-6, u        # x = W/c = 1 → log1p(1)


def test_frame_mirror_identity_objective_stays_identity():
    """Mirroring onto a non-utility (legacy) objective caches the harmless $1k floor but the
    identity path stays identity regardless of the cached scale."""
    runtime = frame_runtime({'object': 'terminalvalue'})
    bundle = Bundle()
    bundle._resolve_frame(runtime, {})
    assert bundle.utility_scale == 1.0e3
    assert runtime['objective']['utility_scale'] == 1.0e3
    assert _utility_wrap_signed(torch.tensor([1.0e6]), runtime).item() == 1.0e6


def test_frame_mirror_no_objective():
    """A runtime with no Objective block (simulate_only) resolves the floor and skips the
    mirror rather than blowing up."""
    runtime = frame_runtime(None)
    bundle = Bundle()
    bundle._resolve_frame(runtime, {})
    assert bundle.utility_scale == 1.0e3
    assert runtime['objective'] is None
    assert _utility_wrap_signed(torch.tensor([1.0e6]), runtime).item() == 1.0e6


def test_resolve_utility_scale_fails_loud_on_symlog_degeneracies():
    """Each silent-degrade path must raise under a utility objective (a $1k floor silently
    breaks tail compression); the identity path returns the floor."""
    sym = lambda **kw: {'objective': {'object': 'asymmetricutility_symlog', **kw}}
    identity = lambda **kw: {'objective': {'object': 'terminalvalue', **kw}}

    # Path 1: last_settlement_index missing
    bundle = Bundle()
    bundle.total_leg_volume = 2500.0
    try:
        bundle._resolve_utility_scale(sym())
        raise AssertionError("symlog should raise on missing last_settlement_index")
    except ValueError as e:
        assert 'last_settlement_index' in str(e), e
    assert bundle._resolve_utility_scale(identity()) == 1.0e3

    # Path 2: empty spot_price_history (and no calibrated fallback)
    bundle.last_settlement_index = 200
    try:
        bundle._resolve_utility_scale(sym())
        raise AssertionError("symlog should raise on empty spot_price_history")
    except ValueError as e:
        assert 'spot_price_history' in str(e), e
    assert bundle._resolve_utility_scale(identity()) == 1.0e3

    # Path 3: zero total_leg_volume
    bundle.spot_price_history = {'CommodityPrice.X': torch.zeros(1)}
    bundle.total_leg_volume = 0.0
    try:
        bundle._resolve_utility_scale(sym())
        raise AssertionError("symlog should raise on zero total_leg_volume")
    except ValueError as e:
        assert 'total_leg_volume' in str(e), e
    assert bundle._resolve_utility_scale(identity()) == 1.0e3

    # Explicit override always honored (no degeneracy check fires).
    assert Bundle()._resolve_utility_scale(sym(utility_scale_explicit=5_000_000.0)) == 5_000_000.0
    assert Bundle()._resolve_utility_scale(identity(utility_scale_explicit=5_000_000.0)) == 5_000_000.0


def test_unknown_utility_scale_mode_fails_loud():
    """Typo in Utility_Scale_Mode raises with a message naming the typo'd value."""
    bundle = Bundle()
    bundle.last_settlement_index = 200
    bundle.total_leg_volume = 2500.0
    runtime = {'objective': {'utility_scale_mode': 'vol_scled_notional'}}      # typo
    try:
        bundle._resolve_utility_scale(runtime)
    except ValueError as e:
        assert 'vol_scled_notional' in str(e) and 'Supported modes' in str(e), e
    else:
        raise AssertionError("_resolve_utility_scale should have raised on unknown mode")
    runtime['objective']['utility_scale_mode'] = 'vol_scaled_notional'
    bundle._resolve_utility_scale(runtime)
    del runtime['objective']['utility_scale_mode']
    bundle._resolve_utility_scale(runtime)


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
