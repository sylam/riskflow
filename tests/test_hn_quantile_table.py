"""Drawing the unmonitored OSS run from its EXACT tabulated law instead of walking it
(utils.hn_aggregate_moments / hn_quantile_table / interp2d_lookup / hn_table_substeps).

Nothing observes the spot between fixings, so only the joint law of (aggregate return, terminal
variance) is needed. HN is affine, so that law is a one-parameter family in the entry variance:
tabulating the exact Fourier-inverted CDF over (u, h1) captures the whole family, and a path's
draw becomes a 2-D lookup — O(1) instead of O(n_sub) bandwidth-bound tensor steps, with NO
distributional approximation and so no validity ceiling in the tail where a barrier reads.

  (a) the moments are EXACT: cumulants match the independent autodiff reference in
      tests/hn_reference.py, and the terminal-variance moments match brute-force MC.
  (b) the affine split is real: a + b*h1 reproduces the reference at several h1.
  (c) the table inverts the CDF: reading it at u returns the x where F(x) == u, and it is
      monotone in u — a folded quantile map would put mass on the wrong side of a barrier.
  (d) the drawn interval reproduces the exact walk's distribution, tails included.
  (e) the lookup is exact on its own grid nodes and bilinear off them.
  (f) the moment cache is transparent and is bypassed under gradients, where a value-keyed
      entry would misattribute one factor's greeks to another.

The table is engaged structurally — only where the calculation owns a t_PreCalc, i.e. is
exposure-based and amortises a precalculation. A single valuation walks the interval exactly.
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
# (c)/(e) the table really is the inverse CDF, and the lookup reads it faithfully
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n', [5, 21, 63])
def test_table_inverts_the_exact_cdf(n):
    """Reading the table at probability u must return the x with F(x) == u, where F is the
    Fourier-inverted CDF itself. This is the whole claim: no moment expansion, no tail ceiling."""
    h_grid = torch.logspace(-0.5, 0.5, 12, dtype=DTYPE) * H0_STAT
    z_grid, x_table = utils.hn_quantile_table(n, *PARAMS, h_grid)
    assert (x_table[:, 1:] >= x_table[:, :-1]).all(), 'quantile table not monotone in z'
    zero = torch.zeros((), dtype=DTYPE)
    for i in (0, 6, 11):
        for j in (2, len(z_grid) // 2, len(z_grid) - 3):
            want = float(utils.norm_cdf(z_grid[j]))
            back = float(utils.hn_cdf_logret(x_table[i, j], n, h_grid[i], *PARAMS, zero))
            assert abs(back - want) < 2e-4, f'n={n} h[{i}] u={want:.4f}: F(Q(u)) = {back:.6f}'


def test_table_covers_the_tail_a_moment_expansion_cannot():
    """The reason this replaced Cornish-Fisher: at 1% and 99% the table is still the exact CDF,
    where a 4-moment expansion is past its validity range and had to be saturated."""
    n = 63
    h_grid = torch.logspace(-0.3, 0.3, 8, dtype=DTYPE) * H0_STAT
    z_grid, x_table = utils.hn_quantile_table(n, *PARAMS, h_grid)
    zero = torch.zeros((), dtype=DTYPE)
    for target in (0.01, 0.99):
        j = int(torch.argmin((utils.norm_cdf(z_grid) - target).abs()))
        want = float(utils.norm_cdf(z_grid[j]))
        back = float(utils.hn_cdf_logret(x_table[4, j], n, h_grid[4], *PARAMS, zero))
        assert abs(back - want) < 2e-4, f'tail u={target}: F(Q(u)) = {back:.6f}'


def test_interp2d_lookup_is_exact_on_nodes():
    y_grid = torch.logspace(-1.0, 1.0, 9, dtype=DTYPE)
    u_grid = torch.linspace(0.05, 0.95, 11, dtype=DTYPE)
    table = (torch.arange(9, dtype=DTYPE).reshape(-1, 1) * 10.0
             + torch.arange(11, dtype=DTYPE).reshape(1, -1))
    yy, uu = torch.meshgrid(y_grid[1:], u_grid[1:], indexing='ij')
    got = utils.interp2d_lookup(uu, yy, u_grid, y_grid, table)
    assert torch.allclose(got, table[1:, 1:]), 'lookup is not exact on its own grid nodes'
    # and interpolates, not steps, in between
    mid_u = 0.5 * (u_grid[3] + u_grid[4])
    mid = utils.interp2d_lookup(mid_u.reshape(1, 1), y_grid[5].reshape(1, 1), u_grid, y_grid, table)
    assert abs(float(mid) - 0.5 * float(table[5, 3] + table[5, 4])) < 1e-12


# ---------------------------------------------------------------------------
# (d) the drawn interval reproduces the exact walk
# ---------------------------------------------------------------------------

class _Shared:
    """The pricer-side contract the substep functions read: batch width, a unit tensor carrying
    dtype/device, and the per-calculation precalc memo whose PRESENCE is what marks a calculation
    as one that amortises a table."""

    def __init__(self, batch):
        self.simulation_batch = batch
        self.one = torch.ones(1, dtype=DTYPE)
        self.t_PreCalc = {}


@pytest.mark.parametrize('n', [5, 21])
def test_table_draw_matches_the_exact_walk(n):
    """The daily walk is the oracle. Both are Monte Carlo, so the body is compared on MC error
    and the tails on an absolute band — the tails are the point of the table."""
    B, S = 64, 8192
    shared = _Shared(B)
    Sj = torch.full((B, 2 * S), 100.0, dtype=DTYPE)
    h = torch.full((B, 2 * S), 1.6 * H0_STAT, dtype=DTYPE)
    b_step = torch.full((B, 1), 1.0e-4, dtype=DTYPE)
    torch.manual_seed(1)
    Se, he = utils.hn_unmonitored_substeps(Sj, h, b_step, n, PARAMS, shared, S, True)
    torch.manual_seed(2)
    St, ht = utils.hn_table_substeps(Sj, h, b_step, n, PARAMS, shared, S, True)
    xe, xt = (Se / 100.0).log().flatten(), (St / 100.0).log().flatten()
    se = xe.std().item() / np.sqrt(xe.numel())
    assert abs(xt.mean().item() - xe.mean().item()) < 6.0 * se
    assert abs(xt.std().item() / xe.std().item() - 1.0) < 0.01
    for p in (0.01, 0.05, 0.5, 0.95, 0.99):
        qe = torch.quantile(xe[:1_000_000].float(), p).item()
        qt = torch.quantile(xt[:1_000_000].float(), p).item()
        assert abs(qt - qe) < 0.02 * xe.std().item(), f'q{p}: {qt:.5f} vs {qe:.5f}'
    assert (ht > 0).all(), 'terminal variance must stay positive'
    assert abs(ht.mean().item() / he.mean().item() - 1.0) < 0.02, 'terminal variance level off'


def test_zero_steps_is_a_no_op():
    shared = _Shared(4)
    Sj = torch.full((4, 8), 100.0, dtype=DTYPE)
    h = torch.full((4, 8), H0_STAT, dtype=DTYPE)
    b = torch.zeros((4, 1), dtype=DTYPE)
    for fn in (utils.hn_unmonitored_substeps, utils.hn_table_substeps):
        S2, h2 = fn(Sj, h, b, 0, PARAMS, shared, 4, True)
        assert S2 is Sj and h2 is h, f'{fn.__name__} touched the state on an empty walk'


# ---------------------------------------------------------------------------
# (f) the cache is transparent and steps aside under gradients
# ---------------------------------------------------------------------------

def test_moment_cache_lives_in_t_precalc():
    """t_PreCalc, not t_Buffer: these are functions of the calibration, not of the batch, and
    reset() clears t_Buffer every batch. Owning one is also the marker the pricers dispatch on."""
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
