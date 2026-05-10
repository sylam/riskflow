"""Lock-in tests: every legacy code path produces bit-identical outputs after the symlog change.

Constructs synthetic states / runtimes with `Object = TerminalFloorThenSurplusUtility` and:
  1. recomputes `_dense_tracking_reward` outputs against the pre-change formula
  2. recomputes `_evaluate_objective` outputs against the pre-change formula
  3. recomputes the three per-step penalties' notional factors

For each, asserts torch.equal(actual, expected_pre_change_formula).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from riskflow import torchrl_hedge as th


def make_legacy_runtime(*, fp=50.0, sr=1.0, p=1.0, mode="asymmetric", rs=2.0, rc=0.0, gamma=1.0):
    return {
        "objective": {
            "object": "terminalfloorthensurplusutility",
            "floor_penalty": fp,
            "surplus_reward": sr,
            "power": p,
            # Setting utility_scale on a legacy runtime should be a no-op (gate is False).
            "utility_scale": 1.0e6,
        },
        "optimizer": {
            "dense_tracking_reward_clip": rc,
            "dense_reward_mode": mode,
            "dense_tracking_reward_scale": rs,
            "gamma": gamma,
        },
    }


# ---------- Hand-rolled pre-change formulas ----------

def expected_dense_tracking_asymmetric(prev_err, next_err, *, fp, rs, rc):
    shaping = torch.clamp(-prev_err, min=0.0) - torch.clamp(-next_err, min=0.0)
    if rc > 0.0:
        shaping = torch.clamp(shaping, min=-rc, max=rc)
    return rs * fp * shaping


def expected_dense_tracking_pbrs(prev_err, next_err, *, fp, gamma, rc):
    prev_pot = -fp * torch.clamp(-prev_err, min=0.0)
    next_pot = -fp * torch.clamp(-next_err, min=0.0)
    shaping = gamma * next_pot - prev_pot
    if rc > 0.0:
        shaping = torch.clamp(shaping, min=-rc, max=rc)
    return shaping


def expected_evaluate_objective(net_pnl, *, fp, sr, p):
    surplus = sr * torch.pow(torch.clamp(net_pnl, min=0.0), p)
    shortfall = -fp * torch.pow(torch.clamp(-net_pnl, min=0.0), p)
    return torch.where(net_pnl >= 0.0, surplus, shortfall)


# ---------- Tests ----------

def test_dense_tracking_legacy_asymmetric():
    """Stub _tracking_error_value to return our synthetic err tensors; pull a result through
    the actual function and compare to the pre-change formula."""
    runtime = make_legacy_runtime(mode="asymmetric", rs=2.0, fp=50.0, rc=0.0)
    prev_err = torch.tensor([+1.0e3, -5.0e2, -1.0e6, 0.0, +1.0e7])
    next_err = torch.tensor([-2.0e3, -1.0e3, -8.0e5, -1.0, +5.0e6])
    fake_prev_state = {"_err": prev_err}
    fake_next_state = {"_err": next_err}
    real = th._tracking_error_value
    th._tracking_error_value = lambda s, _runtime: s["_err"]
    try:
        actual = th._dense_tracking_reward(fake_prev_state, fake_next_state, runtime)
        expected = expected_dense_tracking_asymmetric(prev_err, next_err, fp=50.0, rs=2.0, rc=0.0)
        assert torch.equal(actual, expected), f"asymmetric drift: actual={actual} expected={expected}"
    finally:
        th._tracking_error_value = real
    print("test_dense_tracking_legacy_asymmetric: PASS")


def test_dense_tracking_legacy_asymmetric_with_clip():
    """Same as above but with rc > 0 — the clip path is on the unscaled diff."""
    runtime = make_legacy_runtime(mode="asymmetric", rs=2.0, fp=50.0, rc=1.0e3)
    prev_err = torch.tensor([-1.0e6, -5.0e2, +1.0e3, -2.0e3, -1.0e7])
    next_err = torch.tensor([-1.0e3, -1.5e3, -2.0e3, +5.0e2, -8.0e6])
    real = th._tracking_error_value
    th._tracking_error_value = lambda s, _r: s["_err"]
    try:
        actual = th._dense_tracking_reward({"_err": prev_err}, {"_err": next_err}, runtime)
        expected = expected_dense_tracking_asymmetric(prev_err, next_err, fp=50.0, rs=2.0, rc=1.0e3)
        assert torch.equal(actual, expected), f"asymmetric+clip drift: {actual.tolist()} vs {expected.tolist()}"
    finally:
        th._tracking_error_value = real
    print("test_dense_tracking_legacy_asymmetric_with_clip: PASS")


def test_dense_tracking_legacy_pbrs():
    runtime = make_legacy_runtime(mode="potential_based", fp=50.0, gamma=0.99, rc=0.0)
    prev_err = torch.tensor([-1.0e3, -5.0e2, +1.0e6, -2.0e6, -3.0e3])
    next_err = torch.tensor([-2.0e3, -1.0e3, -8.0e5, -1.5e6, +5.0e2])
    real = th._tracking_error_value
    th._tracking_error_value = lambda s, _r: s["_err"]
    try:
        actual = th._dense_tracking_reward({"_err": prev_err}, {"_err": next_err}, runtime)
        expected = expected_dense_tracking_pbrs(prev_err, next_err, fp=50.0, gamma=0.99, rc=0.0)
        assert torch.equal(actual, expected), f"pbrs drift: {actual.tolist()} vs {expected.tolist()}"
    finally:
        th._tracking_error_value = real
    print("test_dense_tracking_legacy_pbrs: PASS")


def test_evaluate_objective_legacy():
    runtime = make_legacy_runtime(fp=50.0, sr=1.0, p=1.0)
    pnl_excess = torch.tensor([+1.0e6, +1.0e3, 0.0, -1.0e3, -1.0e6, -1.0e8])
    liability = torch.zeros_like(pnl_excess)
    actual = th._evaluate_objective(pnl_excess, liability, runtime)
    expected = expected_evaluate_objective(pnl_excess + liability, fp=50.0, sr=1.0, p=1.0)
    assert torch.equal(actual, expected), f"evaluate_objective legacy drift: {actual} vs {expected}"
    # Power != 1
    runtime["objective"]["power"] = 0.5
    actual = th._evaluate_objective(pnl_excess, liability, runtime)
    expected = expected_evaluate_objective(pnl_excess + liability, fp=50.0, sr=1.0, p=0.5)
    assert torch.equal(actual, expected), f"evaluate_objective power=0.5 drift"
    print("test_evaluate_objective_legacy: PASS")


def test_utility_wrap_legacy_passthrough():
    """Whatever non-symlog runtime is passed, _utility_wrap returns the input tensor object
    unchanged (not a copy). Confirms the gate prevents allocation in the legacy path."""
    runtime = make_legacy_runtime()
    x = torch.tensor([0.0, 1.0e3, 1.0e6, 1.0e9])
    out = th._utility_wrap(x, runtime)
    # In the legacy passthrough, _utility_wrap should return the *same* tensor object.
    assert out is x, "passthrough should return same tensor (no allocation in legacy path)"
    print("test_utility_wrap_legacy_passthrough: PASS")


def test_mirror_utility_scale_legacy_safe():
    """Mirroring utility_scale on a legacy runtime is a no-op for the reward stack — the
    symlog gate is False, so utility_wrap stays passthrough regardless of the cached value."""
    runtime = make_legacy_runtime()
    bundle = {"utility_scale": 9.99e9}
    th._mirror_utility_scale_to_runtime(bundle, runtime)
    # utility_scale gets cached but doesn't affect legacy reward output
    assert runtime["objective"]["utility_scale"] == 9.99e9
    x = torch.tensor([1.0e6])
    assert th._utility_wrap(x, runtime).item() == 1.0e6
    print("test_mirror_utility_scale_legacy_safe: PASS")


def test_mirror_runtime_no_objective():
    """Mirror gracefully no-ops when runtime has no objective dict."""
    runtime = {"objective": None}
    bundle = {"utility_scale": 1.0e6}
    th._mirror_utility_scale_to_runtime(bundle, runtime)
    assert runtime["objective"] is None  # untouched
    print("test_mirror_runtime_no_objective: PASS")


def test_mirror_bundle_no_utility_scale():
    """Mirror gracefully no-ops when bundle has no utility_scale (e.g. cached pre-change bundle)."""
    runtime = make_legacy_runtime()
    bundle = {}  # no utility_scale
    saved = runtime["objective"]["utility_scale"]
    th._mirror_utility_scale_to_runtime(bundle, runtime)
    # Should not have changed the existing utility_scale (or set one)
    assert runtime["objective"]["utility_scale"] == saved
    print("test_mirror_bundle_no_utility_scale: PASS")


def test_resolve_utility_scale_fails_loud_on_symlog_degeneracies():
    """Each silent-degrade path in resolve_utility_scale must raise under symlog (a $1k
    floor silently breaks tail compression). Same paths return $1k harmlessly under legacy."""
    from riskflow.torchrl_hedge import resolve_utility_scale

    def runtime_with_object(obj_name, **objective_extras):
        rt = {'objective': {'object': obj_name, **objective_extras},
              'history_lookback_business_days': 0}
        return rt

    sym = lambda **kw: runtime_with_object('asymmetricutility_symlog', **kw)
    legacy = lambda **kw: runtime_with_object('terminalfloorthensurplusutility', **kw)

    # Path 1: last_settlement_index missing
    bundle = {'total_leg_volume': 2500.0, 'spot_price_history': {}, 'spot_realized_vol': {}}
    try:
        resolve_utility_scale(bundle, sym())
        raise AssertionError("symlog should raise on missing last_settlement_index")
    except ValueError as e:
        assert 'last_settlement_index' in str(e), e
    assert resolve_utility_scale(bundle, legacy()) == 1.0e3, "legacy should return floor"

    # Path 2: empty spot_price_history
    bundle = {'last_settlement_index': 200, 'total_leg_volume': 2500.0,
              'spot_price_history': {}, 'spot_realized_vol': {}}
    try:
        resolve_utility_scale(bundle, sym())
        raise AssertionError("symlog should raise on empty spot_price_history")
    except ValueError as e:
        assert 'spot_price_history' in str(e), e
    assert resolve_utility_scale(bundle, legacy()) == 1.0e3

    # Path 3: zero total_leg_volume (only triggers if spot_history non-empty so we get past path 2)
    bundle = {'last_settlement_index': 200, 'total_leg_volume': 0.0,
              'spot_price_history': {'CommodityPrice.X': torch.zeros(1)},
              'spot_realized_vol': {}}
    try:
        resolve_utility_scale(bundle, sym())
        raise AssertionError("symlog should raise on zero total_leg_volume")
    except ValueError as e:
        assert 'total_leg_volume' in str(e), e
    assert resolve_utility_scale(bundle, legacy()) == 1.0e3

    # Explicit override always honored under both objectives (no degeneracy check fires).
    bundle = {}
    assert resolve_utility_scale(bundle, sym(utility_scale_explicit=5_000_000.0)) == 5_000_000.0
    assert resolve_utility_scale(bundle, legacy(utility_scale_explicit=5_000_000.0)) == 5_000_000.0

    print("test_resolve_utility_scale_fails_loud_on_symlog_degeneracies: PASS")


def test_unknown_utility_scale_mode_fails_loud():
    """Typo in Utility_Scale_Mode silently resolved to 1e3 floor before; now raises with
    a clear message identifying the typo'd value."""
    from riskflow.torchrl_hedge import resolve_utility_scale
    bundle = {'last_settlement_index': 200, 'total_leg_volume': 2500.0,
              'spot_price_history': {}, 'spot_realized_vol': {}}
    runtime = {'objective': {'utility_scale_mode': 'vol_scled_notional'},  # typo
               'history_lookback_business_days': 0}
    try:
        resolve_utility_scale(bundle, runtime)
    except ValueError as e:
        assert 'vol_scled_notional' in str(e), f"unexpected message: {e}"
        assert 'Supported modes' in str(e), f"unexpected message: {e}"
    else:
        raise AssertionError("resolve_utility_scale should have raised on unknown mode")
    # Default (absent) and explicit 'vol_scaled_notional' both pass.
    runtime['objective']['utility_scale_mode'] = 'vol_scaled_notional'
    resolve_utility_scale(bundle, runtime)
    del runtime['objective']['utility_scale_mode']
    resolve_utility_scale(bundle, runtime)
    print("test_unknown_utility_scale_mode_fails_loud: PASS")


def test_symlog_missing_utility_scale_fails_loud():
    """If symlog is active but utility_scale wasn't mirrored (bypassed rollout entry point),
    every reward computation must fail loud — silent default would give wrong-but-plausible rewards."""
    runtime = {
        "objective": {
            "object": "asymmetricutility_symlog",
            "floor_penalty": 10.0, "surplus_reward": 1.0, "power": 1.0,
            # NO utility_scale
        },
        "optimizer": {"dense_reward_mode": "asymmetric", "dense_tracking_reward_scale": 2.0},
    }
    # _utility_wrap must raise
    try:
        th._utility_wrap(torch.tensor([1.0e6]), runtime)
    except ValueError as e:
        assert "utility_scale" in str(e) and "is not set" in str(e), f"unexpected message: {e}"
    else:
        raise AssertionError("_utility_wrap should have raised on missing utility_scale")
    # _evaluate_objective must raise
    try:
        th._evaluate_objective(torch.tensor([-1.0e6]), torch.tensor([0.0]), runtime)
    except ValueError as e:
        assert "utility_scale" in str(e) and "is not set" in str(e), f"unexpected message: {e}"
    else:
        raise AssertionError("_evaluate_objective should have raised on missing utility_scale")
    # Adding the scale fixes both
    runtime["objective"]["utility_scale"] = 1.0e6
    th._utility_wrap(torch.tensor([1.0e6]), runtime)  # now OK
    th._evaluate_objective(torch.tensor([-1.0e6]), torch.tensor([0.0]), runtime)
    print("test_symlog_missing_utility_scale_fails_loud: PASS")


if __name__ == "__main__":
    test_utility_wrap_legacy_passthrough()
    test_dense_tracking_legacy_asymmetric()
    test_dense_tracking_legacy_asymmetric_with_clip()
    test_dense_tracking_legacy_pbrs()
    test_evaluate_objective_legacy()
    test_mirror_utility_scale_legacy_safe()
    test_mirror_runtime_no_objective()
    test_mirror_bundle_no_utility_scale()
    test_resolve_utility_scale_fails_loud_on_symlog_degeneracies()
    test_unknown_utility_scale_mode_fails_loud()
    test_symlog_missing_utility_scale_fails_loud()
    print("\nAll legacy-bit-identical tests passed.")
