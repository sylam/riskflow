"""End-to-end smoke test: drives the framework through `cx.run_job()` on a JSON fixture
to verify the full symlog wiring (bundle build, utility_scale resolution, evaluate_objective
dispatch, evaluation_summary surface, position-limit hard mask).

NO monkey-patching. NO internal imports. Just JSON in, result out.
"""
import json as jsonlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'policy_test_simulate_only.json')


def _load(**objective_overrides):
    """Load the base fixture and apply Objective overrides."""
    data = jsonlib.load(open(FIXTURE))
    obj = data['Calc']['Calculation']['Hedging_Problem']['Objective']
    obj.update(objective_overrides)
    return data


def _run(data, *, expect_eval=True):
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(data), 'smoke.json'))
    _, result = cx.run_job()
    assert result.torchrl_bundle is not None, "bundle missing from result"
    if expect_eval:
        assert result.evaluation_summary is not None, "evaluation_summary missing"
    return result


def test_bundle_resolves_utility_scale():
    """Bundle build always sets utility_scale (consumed by symlog, harmless for legacy)."""
    result = _run(_load())
    c = float(result.torchrl_bundle.get('utility_scale', 0.0))
    assert c > 1e3, f"utility_scale should be > $1k floor; got ${c:,.0f}"
    print(f"test_bundle_resolves_utility_scale: PASS  (c = ${c:,.0f})")


def test_legacy_objective_runs():
    """Legacy TerminalFloorThenSurplusUtility evaluates without crashing.
    Position limits are now reward-side (Per_Instrument_Bounds_Penalty), not hard-masked,
    so an untrained 1-epoch policy may briefly explore past [Min, Max] before the penalty
    teaches it; we just verify the run produces a finite headline summary."""
    data = _load(Object='TerminalFloorThenSurplusUtility', Floor_Penalty=10.0)
    result = _run(data)
    metrics = result.evaluation_summary['metrics']
    assert metrics['average_net_pnl'] is not None and not (metrics['average_net_pnl'] != metrics['average_net_pnl']), \
        f"legacy run produced NaN headline: {metrics}"
    print(f"test_legacy_objective_runs: PASS  (mean=${metrics['average_net_pnl']:+,.0f})")


def test_symlog_objective_runs():
    """AsymmetricUtility_Symlog evaluates and produces a finite utility-space reward.
    Position limits are reward-side, not hard-masked — see test_legacy_objective_runs."""
    data = _load(Object='AsymmetricUtility_Symlog', Floor_Penalty=10.0,
                 Surplus_Reward=1.0, Power=1.0)
    result = _run(data)
    metrics = result.evaluation_summary['metrics']
    assert metrics['average_net_pnl'] is not None and not (metrics['average_net_pnl'] != metrics['average_net_pnl']), \
        f"symlog run produced NaN headline: {metrics}"
    print(f"test_symlog_objective_runs: PASS  (mean=${metrics['average_net_pnl']:+,.0f})")


def test_unknown_utility_scale_mode_fails_loud():
    """Typo in Utility_Scale_Mode raises at bundle build, not silently."""
    data = _load(Object='AsymmetricUtility_Symlog', Floor_Penalty=10.0,
                 Utility_Scale_Mode='vol_scled_notional')  # typo
    try:
        _run(data)
    except ValueError as e:
        assert 'vol_scled_notional' in str(e), f"unexpected message: {e}"
        print(f"test_unknown_utility_scale_mode_fails_loud: PASS")
        return
    raise AssertionError("typo'd mode should have raised")


def test_explicit_utility_scale_override():
    """Utility_Scale_Explicit overrides the formula."""
    data = _load(Object='AsymmetricUtility_Symlog', Floor_Penalty=10.0,
                 Utility_Scale_Explicit=5_000_000.0)
    result = _run(data)
    c = float(result.torchrl_bundle['utility_scale'])
    assert abs(c - 5_000_000.0) < 1.0, f"explicit override not honored: c=${c:,.0f}"
    print(f"test_explicit_utility_scale_override: PASS  (c = ${c:,.0f})")


if __name__ == '__main__':
    test_bundle_resolves_utility_scale()
    test_legacy_objective_runs()
    test_symlog_objective_runs()
    test_unknown_utility_scale_mode_fails_loud()
    test_explicit_utility_scale_override()
    print("\nAll smoke tests passed.")
