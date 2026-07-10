"""Unit tests for the symlog utility transform — pure math, no bundle/runtime needed.

Covers the inverse round-trip and PBRS telescoping identities.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from riskflow.hedge_bundle import (
    _utility_wrap, _utility_wrap_signed, _is_symlog_objective, _is_utility_objective)


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """Some tests here flip the global default dtype to float64 for an exact FD gate.
    Restore it afterwards so the float32 default doesn't leak into other test modules in
    the same process (a bit-exact regression elsewhere would otherwise fail by ordering)."""
    prev = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def make_runtime(symlog: bool, c: float = 1.0e6):
    return {
        "objective": {
            "object": "asymmetricutility_symlog" if symlog else "terminalfloorthensurplusutility",
            "utility_scale": c,
            "floor_penalty": 10.0,
            "surplus_reward": 1.0,
        }
    }


def make_util_runtime(shape: str, c: float = 1.0e6, **params):
    """Runtime for a terminal-utility shape: 'symlog' | 'huber' | 'cara' (or 'legacy')."""
    obj = {
        "symlog": "asymmetricutility_symlog",
        "huber": "asymmetricutility_huber",
        "cara": "asymmetricutility_cara",
        "legacy": "terminalfloorthensurplusutility",
    }[shape]
    cfg = {"object": obj, "utility_scale": c}
    cfg.update(params)
    return {"objective": cfg}


def test_is_symlog():
    assert _is_symlog_objective(make_runtime(True))
    assert not _is_symlog_objective(make_runtime(False))
    print("test_is_symlog: PASS")


def test_is_utility_objective():
    """All three shapes are utility objectives (need c); symlog check is symlog-only."""
    for shape in ("symlog", "huber", "cara"):
        assert _is_utility_objective(make_util_runtime(shape)), shape
    assert not _is_utility_objective(make_util_runtime("legacy"))
    # _is_symlog_objective stays symlog-specific (for the saturation tripwire etc.)
    assert _is_symlog_objective(make_util_runtime("symlog"))
    assert not _is_symlog_objective(make_util_runtime("huber"))
    assert not _is_symlog_objective(make_util_runtime("cara"))
    print("test_is_utility_objective: PASS")


def test_utility_shapes_match_reference():
    """_utility_wrap_signed matches the reference formula for each shape (on x = W/c)."""
    torch.set_default_dtype(torch.float64)
    c = 5.0e5
    W = torch.tensor([-3.0e6, -1.0e6, -2.0e5, 0.0, 2.0e5, 1.0e6, 3.0e6])
    x = W / c

    # symlog: sign(x)·log1p(|x|)
    got = _utility_wrap_signed(W, make_util_runtime("symlog", c))
    assert torch.allclose(got, torch.sign(x) * torch.log1p(x.abs()), atol=1e-12)

    # huber: x − [a·loss² (loss≤δ) | a·δ²+2aδ(loss−δ)];  loss = max(−x,0)
    a, d = 3.0, 0.5
    got = _utility_wrap_signed(W, make_util_runtime("huber", c, huber_aversion=a, huber_delta=d))
    loss = (-x).clamp(min=0.0)
    ref = x - torch.where(loss <= d, a * loss ** 2, a * d * d + 2 * a * d * (loss - d))
    assert torch.allclose(got, ref, atol=1e-12)
    # downside-only: gains (x>0) are untouched (linear, no penalty)
    pos = W > 0
    assert torch.allclose(got[pos], x[pos], atol=1e-12)

    # cara: (1 − exp(−γx)) / γ
    g = 1.5
    got = _utility_wrap_signed(W, make_util_runtime("cara", c, cara_gamma=g))
    assert torch.allclose(got, (1.0 - torch.exp(-g * x)) / g, atol=1e-12)
    print("test_utility_shapes_match_reference: PASS  (symlog / huber / cara)")


def test_utility_shapes_differentiable():
    """FD-vs-autograd on du/dW for each shape, incl. ACROSS the huber knee (C¹ — the linear
    tail keeps a constant, non-vanishing gradient). A broken knee or exp would show here."""
    torch.set_default_dtype(torch.float64)
    c, a, d, g = 4.0e5, 2.5, 0.4, 1.2
    # W range straddles the huber knee at x=−d ⇒ W=−d·c, and 0.
    W0 = torch.linspace(-3.0 * d * c, 2.0 * d * c, 64)
    for rt in (make_util_runtime("symlog", c),
               make_util_runtime("huber", c, huber_aversion=a, huber_delta=d),
               make_util_runtime("cara", c, cara_gamma=g)):
        W = W0.clone().requires_grad_(True)
        u = _utility_wrap_signed(W, rt)
        (auto,) = torch.autograd.grad(u.sum(), W)
        h = 1.0  # dollars; double precision → tight central difference
        with torch.no_grad():
            fd = (_utility_wrap_signed(W0 + h, rt) - _utility_wrap_signed(W0 - h, rt)) / (2 * h)
        err = float((auto.detach() - fd).abs().max())
        assert err < 1e-6, f"{rt['objective']['object']}: FD-vs-autograd {err:.2e}"
    print("test_utility_shapes_differentiable: PASS  (incl. huber knee C¹)")


def test_utility_signed_legacy_identity():
    """Legacy objective: _utility_wrap_signed returns the dollar value unchanged."""
    W = torch.tensor([-1.0e6, 0.0, 1.0e6])
    assert torch.equal(_utility_wrap_signed(W, make_util_runtime("legacy")), W)
    print("test_utility_signed_legacy_identity: PASS")


def test_utility_wrap_passes_through_for_legacy():
    """For non-symlog objectives, _utility_wrap returns input unchanged."""
    runtime = make_runtime(symlog=False)
    x = torch.tensor([0.0, 1.0e3, 1.0e6, 1.0e9])
    out = _utility_wrap(x, runtime)
    assert torch.equal(out, x), "legacy path must not modify the dollar value"
    print("test_utility_wrap_passes_through_for_legacy: PASS")


def test_pbrs_telescoping():
    """On a 5-step synthetic rollout with known err_t, kept-transition shaping sum
    must equal Φ(s_0) − Φ(s_{T-1}) for Φ(s) = −fp · log1p(max(−err(s), 0) / c).

    Telescoping: shaping_t = γ Φ(s_{t+1}) − Φ(s_t). With γ=1, sum over t=0..T-2 = Φ(s_{T-1}) − Φ(s_0).
    But asymmetric mode in our impl has a sign convention from the dense-shaping comment:
        shaping_t = rs · fp · (max(−err_t, 0) − max(−err_{t+1}, 0))
                  = -1 · rs · (Φ(s_{t+1}) − Φ(s_t)) / 1   (with Φ = −fp · |down|)
    so kept-sum = Φ(s_0) − Φ(s_{T-1}) — opposite sign from PBRS but identical PBRS-invariance proof.
    Test the asymmetric form which is what we use."""
    fp = 10.0
    rs = 2.0
    c = 1.0e6
    # Synthetic err sequence: deepening losses then partial recovery
    err = torch.tensor([100.0, -500.0, -2.0e6, -1.5e6, -3.0e5], dtype=torch.float32)
    down = torch.clamp(-err, min=0.0)  # [0, 500, 2e6, 1.5e6, 3e5]
    u_down = torch.log1p(down / c)
    # Asymmetric shaping per step: rs · fp · (u_down[t] − u_down[t+1])
    shaping_sum = (rs * fp * (u_down[:-1] - u_down[1:])).sum()
    expected = rs * fp * (u_down[0] - u_down[-1])
    err_abs = float((shaping_sum - expected).abs().item())
    assert err_abs < 1e-5, f"telescoping err {err_abs}"
    print(f"test_pbrs_telescoping: PASS  (asymmetric mode, kept-sum = rs·fp·(Φ(0)-Φ(T-1))) err={err_abs:.2e}")


def test_magnitude_sanity():
    """Per-step shaping under realistic err magnitudes should sit in O(10–100), not O(1M+)."""
    fp = 10.0
    rs = 2.0
    c = 8.0e5
    # Realistic err range: $0 to $50M downside per step
    down = torch.tensor([0.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 5.0e7], dtype=torch.float32)
    u = torch.log1p(down / c)
    # Per-step: bounded by rs·fp·u_max where u_max ≈ log(50e6/8e5) ≈ 4.13
    # → rs·fp·u_max = 2·10·4.13 ≈ 82
    bound = rs * fp * float(u.max().item())
    assert 10.0 < bound < 200.0, f"bound {bound} outside expected O(10-100) range"
    print(f"test_magnitude_sanity: PASS  (max per-step shaping magnitude ≈ {bound:.1f})")


if __name__ == "__main__":
    test_is_symlog()
    test_is_utility_objective()
    test_utility_shapes_match_reference()
    test_utility_shapes_differentiable()
    test_utility_signed_legacy_identity()
    test_utility_wrap_passes_through_for_legacy()
    test_pbrs_telescoping()
    test_magnitude_sanity()
    print("\nAll symlog unit tests passed.")
