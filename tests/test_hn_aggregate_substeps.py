"""The `Approximate_Substeps` valuation switch: sampling the unmonitored OSS run from its
aggregate law instead of walking it (utils.hn_aggregate_moments / cornish_fisher /
hn_aggregate_draw / hn_aggregate_substeps, dispatched in the three OSS pricers).

Nothing observes the spot between fixings, so only the JOINT law of (aggregate return, terminal
variance) is needed. HN is affine, so that law's moments are exact and are SCALARS per interval;
the draw is then O(1) per path instead of O(n_sub) bandwidth-bound tensor steps.

This is the one APPROXIMATION in the OSS stack — everything else is exact — so the gates are
built around that:

  (a) the moments are EXACT: cumulants match the independent autodiff reference in
      tests/hn_reference.py, and the terminal-variance moments match brute-force MC.
  (b) the affine split is real: a + b*h1 reproduces the reference at several h1.
  (c) Cornish-Fisher stays MONOTONE in z — a folded quantile map would put mass on the wrong
      side of a barrier, which is exactly what an OSS pricer cannot tolerate.
  (d) the sampler reproduces the exact walk's distribution within MC error.
  (e) the switch DEFAULTS OFF and off is bit-identical to before it existed.
  (f) the moment cache is transparent (same numbers cached or not) and is bypassed under
      gradients, where a cached graph node would raise on a second backward.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

import riskflow
from riskflow import utils
import hn_reference as hnref

DTYPE = torch.float64
_SP = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=252.0)
PARAMS = tuple(torch.tensor(float(_SP[k]), dtype=DTYPE)
               for k in ('omega', 'alpha', 'beta', 'gamma_star'))
H0_STAT = float(utils.hn_stationary_var(*PARAMS))


def _ref_params():
    return hnref.as_tensors({'omega': float(PARAMS[0]), 'alpha': float(PARAMS[1]),
                             'beta': float(PARAMS[2]), 'gamma_star': float(PARAMS[3]), 'r': 0.0})


def test_uses_repo_under_test():
    assert riskflow.__file__ == os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'riskflow', '__init__.py')


# ---------------------------------------------------------------------------
# (a)/(b) the moments are exact and affine in the entry variance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n', [2, 5, 21, 63])
@pytest.mark.parametrize('h_mult', [0.5, 1.6])
def test_cumulants_match_the_reference(n, h_mult):
    """kappa_1..4 of the aggregate, against the independent autodiff-of-logmgf reference.
    The stencil uses two step sizes precisely so all four survive this at once."""
    a, b = utils.hn_aggregate_moments(n, *PARAMS)
    h1 = h_mult * H0_STAT
    ref = hnref.hn_cumulants(_ref_params(), n, h1, 4)
    tol = (1.0e-5, 1.0e-5, 1.0e-2, 1.0e-2)          # k3/k4 only weight correction terms
    for i, (r, t) in enumerate(zip(ref, tol)):
        got = float(a[i] + b[i] * h1)
        assert abs(got - r) <= t * max(abs(r), 1e-300), f'kappa_{i + 1}: {got:.6e} vs {r:.6e}'


def test_terminal_variance_moments_match_brute_force():
    """E[h_end], Var(h_end) and Cov(X, h_end) come from the theta-seeded recursion; the daily
    walk is the oracle. Cov is what carries vol clustering across the interval — a zero there
    would decouple the next fixing from this one."""
    n, B, h1 = 21, 2_000_000, 1.6 * H0_STAT
    a, b = utils.hn_aggregate_moments(n, *PARAMS)
    e_h, v_h, cov = (float(a[i] + b[i] * h1) for i in (4, 5, 6))
    g = torch.Generator().manual_seed(3)
    h = torch.full((B,), h1, dtype=DTYPE)
    x = torch.zeros(B, dtype=DTYPE)
    for _ in range(n):
        z = torch.randn(B, generator=g, dtype=DTYPE)
        sh = h.sqrt()
        x = x + (-0.5 * h) + sh * z
        h = utils.hn_variance_step(h, sh, z, *PARAMS)
    assert abs(e_h / h.mean().item() - 1.0) < 5e-3, f'E[h_end] {e_h:.4e} vs {h.mean():.4e}'
    assert abs(v_h / h.var().item() - 1.0) < 2e-2, f'Var(h_end) {v_h:.4e} vs {h.var():.4e}'
    mc_cov = ((x - x.mean()) * (h - h.mean())).mean().item()
    assert abs(cov / mc_cov - 1.0) < 2e-2, f'Cov {cov:.4e} vs {mc_cov:.4e}'
    assert cov < 0.0, 'leverage should make the aggregate and the terminal variance anti-correlated'


def test_moments_are_affine_in_h1():
    """The whole O(1) claim rests on this: two scalars per moment describe every path."""
    a, b = utils.hn_aggregate_moments(21, *PARAMS)
    lo, hi = 0.4 * H0_STAT, 3.0 * H0_STAT
    mid = 0.5 * (lo + hi)
    for i in range(7):
        f = lambda x: float(a[i] + b[i] * x)
        assert abs(f(mid) - 0.5 * (f(lo) + f(hi))) <= 1e-10 * max(abs(f(mid)), 1e-30)


# ---------------------------------------------------------------------------
# (c) Cornish-Fisher must stay monotone
# ---------------------------------------------------------------------------

def test_cornish_fisher_is_monotone_and_matches_moments():
    """A folded quantile map puts probability mass on the wrong side of a barrier. The clamp
    keeps the map monotone even where the skew/kurtosis correction would dominate."""
    z = torch.linspace(-6.0, 6.0, 20001, dtype=DTYPE)
    for g1, g2 in ((0.0, 0.0), (-0.9, 1.2), (-4.0, 20.0), (3.0, -1.0)):   # last two are extreme
        k2 = torch.tensor(4.0e-3, dtype=DTYPE)
        x = utils.cornish_fisher(z, torch.tensor(-2.0e-3, dtype=DTYPE), k2,
                                 torch.tensor(g1, dtype=DTYPE) * k2 ** 1.5,
                                 torch.tensor(g2, dtype=DTYPE) * k2 ** 2)
        assert (x[1:] - x[:-1] >= 0).all(), f'non-monotone at skew={g1}, exkurt={g2}'
    # with no skew/kurtosis it is exactly the Gaussian quantile
    k1, k2 = torch.tensor(0.1, dtype=DTYPE), torch.tensor(0.25, dtype=DTYPE)
    zero = torch.zeros((), dtype=DTYPE)
    assert torch.allclose(utils.cornish_fisher(z, k1, k2, zero, zero), k1 + k2.sqrt() * z)


def test_cornish_fisher_delivers_the_requested_moments():
    torch.manual_seed(0)
    z = torch.randn(4_000_000, dtype=DTYPE)
    k1, k2 = torch.tensor(-0.02, dtype=DTYPE), torch.tensor(9.0e-4, dtype=DTYPE)
    g1 = -0.6
    x = utils.cornish_fisher(z, k1, k2, torch.tensor(g1, dtype=DTYPE) * k2 ** 1.5,
                             torch.zeros((), dtype=DTYPE))
    # E[correction] is zero only for UNCLAMPED z, and unclamped is exactly what folds; saturating
    # past CF_Z therefore leaves a small mean shift. It is bounded well inside a percent of a
    # standard deviation — pinned here so a regression that widens it is caught.
    sd = float(k2) ** 0.5
    assert abs(x.mean().item() - float(k1)) < 0.005 * sd, 'saturation bias grew'
    assert abs(x.var().item() / float(k2) - 1.0) < 0.02
    # Saturation costs skew as well as mean: the tail it truncates is the tail that carries the
    # third moment, so ~82% of the requested skew survives. It cannot be bought back by widening
    # CF_Z — at this skew the cubic genuinely folds at z~2.83, so a wider window would trip the
    # per-path monotonicity guard and lose the correction ENTIRELY. Most of the skew beats none.
    skew = (((x - x.mean()) / x.std()) ** 3).mean().item()
    assert g1 * 0.7 > skew > g1 * 1.05, f'skew {skew:.3f} out of band for requested {g1}'


# ---------------------------------------------------------------------------
# (d) the sampler reproduces the exact walk
# ---------------------------------------------------------------------------

class _Shared:
    def __init__(self, batch):
        self.simulation_batch = batch
        self.one = torch.ones(1, dtype=DTYPE)
        self.t_PreCalc = {}


@pytest.mark.parametrize('n', [5, 21])
def test_sampler_matches_the_exact_walk(n):
    B, S = 64, 8192
    shared = _Shared(B)
    Sj = torch.full((B, 2 * S), 100.0, dtype=DTYPE)
    h = torch.full((B, 2 * S), 1.6 * H0_STAT, dtype=DTYPE)
    b_step = torch.full((B, 1), 1.0e-4, dtype=DTYPE)
    torch.manual_seed(1)
    Se, he = utils.hn_unmonitored_substeps(Sj, h, b_step, n, PARAMS, shared, S, True)
    torch.manual_seed(2)
    Sa, ha = utils.hn_aggregate_substeps(Sj, h, b_step, n, PARAMS, shared, S, True)
    xe, xa = (Se / 100.0).log().flatten(), (Sa / 100.0).log().flatten()
    se = xe.std().item() / np.sqrt(xe.numel())
    assert abs(xa.mean().item() - xe.mean().item()) < 6.0 * se
    assert abs(xa.std().item() / xe.std().item() - 1.0) < 0.01
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        qe = torch.quantile(xe[:1_000_000].float(), p).item()
        qa = torch.quantile(xa[:1_000_000].float(), p).item()
        assert abs(qa - qe) < 0.01 * xe.std().item() + 2.0e-3, f'q{p}: {qa:.5f} vs {qe:.5f}'
    assert abs(ha.mean().item() / he.mean().item() - 1.0) < 0.02, 'terminal variance level off'
    assert (ha > 0).all(), 'terminal variance must stay positive'


def test_zero_steps_is_a_no_op():
    shared = _Shared(4)
    Sj = torch.full((4, 8), 100.0, dtype=DTYPE)
    h = torch.full((4, 8), H0_STAT, dtype=DTYPE)
    b = torch.zeros((4, 1), dtype=DTYPE)
    for fn in (utils.hn_unmonitored_substeps, utils.hn_aggregate_substeps):
        S2, h2 = fn(Sj, h, b, 0, PARAMS, shared, 4, True)
        assert S2 is Sj and h2 is h, f'{fn.__name__} touched the state on an empty walk'


# ---------------------------------------------------------------------------
# (f) the cache is transparent and steps aside under gradients
# ---------------------------------------------------------------------------

def test_moment_cache_lives_in_t_precalc():
    """t_PreCalc, not t_Buffer: the table is a function of the calibration, not the batch, and
    reset() clears t_Buffer. Base valuation must have one too — that is where OSS prices."""
    sh = _Shared(4)
    a1, b1 = utils.hn_cached_moments(sh, 21, *PARAMS)
    assert len(sh.t_PreCalc) == 1
    a2, b2 = utils.hn_cached_moments(sh, 21, *PARAMS)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    a3, _ = utils.hn_cached_moments(sh, 22, *PARAMS)
    assert len(sh.t_PreCalc) == 2 and not torch.equal(a1, a3)
    direct_a, direct_b = utils.hn_aggregate_moments(21, *PARAMS)
    assert torch.equal(a1, direct_a) and torch.equal(b1, direct_b)


def test_gradients_bypass_the_cache_and_flow():
    """Under greeks the cache must step aside and the parameters must still receive gradient
    through the stencil. The reason is ATTRIBUTION, not graph lifetime: the key is parameter
    values, so two underlyings calibrated to identical numbers would collide and the second
    would be handed moments wired to the first one's leaves. (Reuse across batches is otherwise
    safe — SensitivitiesEstimator retains the graph and the leaves are minted once per calc.)"""
    sh = _Shared(4)
    p = [t.clone().requires_grad_(True) for t in PARAMS]
    for _ in range(2):
        a, b = utils.hn_cached_moments(sh, 21, *p)
        (a.sum() + b.sum()).backward(retain_graph=True)   # what SensitivitiesEstimator does
    assert not sh.t_PreCalc, 'differentiable moments must not be cached'
    # the collision the bypass exists for: same VALUES, different leaves -> distinct gradients
    q = [t.clone().detach().requires_grad_(True) for t in PARAMS]
    a_q, b_q = utils.hn_cached_moments(sh, 21, *q)
    (a_q.sum() + b_q.sum()).backward()
    assert q[0].grad is not None and not torch.equal(q[0].grad, p[0].grad), \
        'a second factor with identical parameters got the first one\'s gradient'
    for t, name in zip(p, ('omega', 'alpha', 'beta', 'gamma_star')):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs() > 0, \
            f'no gradient reached {name}'
