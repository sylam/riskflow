"""Validation suite for ``hn_garch`` -- the Heston-Nandi closed-form GARCH pricer.

This module is the ORACLE that DELIVERABLE 2 (the OSS aggregation-bias experiment) is
measured against, so the tests below are deliberately over-determined: the price is
pinned against Black-Scholes in a degenerate limit (analytic, machine precision), the
convention is pinned against the martingale identity (analytic, machine precision), and
everything is cross-checked against an independent daily Monte Carlo of the recursion.

Run: ``pytest tests/test_hn_garch.py -q``  (~40s on CPU, ~15s with CUDA).
"""

import math
import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskflow import hn_garch as hg                                     # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
S0 = 100.0
R_STEP = 0.03 / 252.0


def _p(ann_vol=0.20, psi=0.98, gamma=400.0, lev=0.12, r=R_STEP):
    return hg.hn_params_from_targets(ann_vol, psi, gamma, lev, r=r).as_tensors(DT)


def _t(x):
    return torch.tensor(float(x), dtype=DT)


# ======================================================================================
# convention / martingale -- this is what pins the indexing and the -phi/2 term
# ======================================================================================

@pytest.mark.parametrize('n', [1, 5, 21, 63, 252])
def test_martingale_phi_one(n):
    """f_t(1) = S_t * exp(r*n)  <=>  B_t(1) == 0 and A_t(1) == r*n, EXACTLY.

    This is the single identity that pins the LRNVR convention:
      * ``h`` predictable (h_{t+1} drives the return over (t, t+1]),
      * lambda* = -1/2 in the drift,
      * and therefore the linear term in the B-recursion is phi*(gamma*-1/2), NOT
        phi*gamma*.
    Any of those three wrong and B(1) != 0.
    """
    p = _p()
    A, B = hg.hn_ab(_t(1.0), p, n)
    assert abs(float(B)) < 1e-14
    assert abs(float(A) - float(p.r) * n) < 1e-14


def test_dropping_the_half_phi_breaks_the_martingale():
    """Guard against the common mis-statement B_t = phi*gamma - gamma^2/2 + ... .

    With that (wrong) linear term the one-step B(phi=1) is exactly +1/2, so the model
    forward is S*exp(r + h/2) -- a harvestable Jensen drift on the tradeable.  Pinned
    here so nobody 'simplifies' the recursion back to it.
    """
    p = _p()
    ga = p.gamma
    phi = _t(1.0)
    B_wrong = phi * ga - 0.5 * ga ** 2 + 0.5 * (phi - ga) ** 2       # one step, B_T = 0
    assert abs(float(B_wrong) - 0.5) < 1e-12


def test_forward_is_martingale_in_mc():
    """The simulator and the CF agree on the forward -- independent check of the drift."""
    p = _p()
    h1 = float(p.stationary_var)
    r = hg.hn_simulate(p, 63, h1, 2_000_000, seed=3, device=DEV)
    st = S0 * torch.exp(r)
    se = float(st.std() / math.sqrt(len(st)))
    assert abs(float(st.mean()) - S0 * math.exp(float(p.r) * 63)) < 4.0 * se


# ======================================================================================
# (b) Black-Scholes degenerate limit -- the high-precision analytic anchor
# ======================================================================================

@pytest.mark.parametrize('gamma', [0.0, 250.0])
@pytest.mark.parametrize('n', [5, 63, 252])
@pytest.mark.parametrize('k', [0.80, 0.95, 1.00, 1.05, 1.25])
def test_alpha_zero_collapses_to_black_scholes(gamma, n, k):
    """alpha -> 0 makes h deterministic; started AT its fixed point omega/(1-beta) the
    model is exactly Black-Scholes with per-step variance omega/(1-beta).

    Done for gamma* = 0 AND gamma* = 250: with alpha = 0 the gamma* terms cancel
    algebraically out of the B recursion (B_t = beta*B_{t+1} + (phi^2-phi)/2), so
    gamma*-independence is itself part of the assertion.
    """
    beta = 0.7
    p = hg.HNParams(omega=0.04 / 252 * (1 - beta), alpha=0.0, beta=beta,
                    gamma=gamma, r=R_STEP).as_tensors(DT)
    h = float(p.omega / (1.0 - p.beta))
    got = float(hg.hn_call(S0, S0 * k, p, n, h))
    want = hg.bs_call_np(S0, S0 * k, float(p.r), n, h * n)
    assert abs(got - want) < 1e-11 + 1e-12 * abs(want)


def test_alpha_zero_cdf_matches_normal_in_the_tails():
    """The INVERSION itself (not just the price) is accurate in the far tail: in the
    alpha=0 limit Q(R_n <= b) is an exact normal CDF.  This is the property DELIVERABLE 2
    leans on -- survival probabilities out to 3.5 sigma."""
    beta = 0.7
    p = hg.HNParams(omega=0.04 / 252 * (1 - beta), alpha=0.0, beta=beta,
                    gamma=0.0, r=R_STEP).as_tensors(DT)
    h = float(p.omega / (1.0 - p.beta))
    for n in (5, 21, 63):
        v = h * n
        mu = float(p.r) * n - 0.5 * v
        for z in (-3.5, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5):
            b = mu + z * math.sqrt(v)
            got = float(hg.hn_cdf_logret(b, p, n, h))
            want = 0.5 * math.erfc(-z / math.sqrt(2.0))
            assert abs(got - want) < 1e-12, (n, z, got, want)


# ======================================================================================
# (c) put-call parity and the probability interpretation of P1/P2
# ======================================================================================

@pytest.mark.parametrize('n', [5, 63, 252])
@pytest.mark.parametrize('k', [0.8, 1.0, 1.2])
def test_put_call_parity(n, k):
    p = _p()
    h1 = float(p.stationary_var)
    c = float(hg.hn_call(S0, S0 * k, p, n, h1))
    pu = float(hg.hn_put(S0, S0 * k, p, n, h1))
    assert abs((c - pu) - (S0 - S0 * k * math.exp(-float(p.r) * n))) < 1e-10


@pytest.mark.parametrize('n', [21, 63])
def test_p1_p2_are_probabilities_and_match_mc(n):
    """P2 = Q(S_T > K) and P1 = E[S_T 1{S_T>K}] / (S e^{rn}).  Both checked against the
    daily simulator -- an INDEPENDENT confirmation of the inversion, not just parity
    algebra (which is automatic once the put is defined by parity)."""
    p = _p()
    h1 = float(p.stationary_var)
    ks = torch.tensor([0.85, 0.95, 1.0, 1.05, 1.15], dtype=DT) * S0
    p1, p2 = hg._p1_p2(torch.log(ks / S0), p, n, torch.tensor(h1, dtype=DT),
                       None, None, 8, True)
    assert bool(((p1 >= 0) & (p1 <= 1) & (p2 >= 0) & (p2 <= 1)).all())
    st = S0 * torch.exp(hg.hn_simulate(p, n, h1, 4_000_000, seed=5, device=DEV))
    fwd = S0 * math.exp(float(p.r) * n)
    for i, k in enumerate(ks):
        ind = (st > float(k)).to(DT)
        m2, s2 = float(ind.mean()), float(ind.std() / math.sqrt(len(ind)))
        w = st * ind / fwd
        m1, s1 = float(w.mean()), float(w.std() / math.sqrt(len(w)))
        assert abs(float(p2[i]) - m2) < 4.5 * s2, ('P2', k, float(p2[i]), m2, s2)
        assert abs(float(p1[i]) - m1) < 4.5 * s1, ('P1', k, float(p1[i]), m1, s1)


# ======================================================================================
# (a) independent price validation against Monte Carlo
# ======================================================================================

@pytest.mark.parametrize('n', [21, 63])
def test_price_matches_mc(n):
    """Pooled over 4 seeds so the common-random-driver correlation between strikes does
    not masquerade as bias (single-seed z-scores are all the same sign by construction).
    """
    p = _p()
    h1 = float(p.stationary_var)
    ks = [85.0, 95.0, 100.0, 105.0, 115.0]
    cf = [float(hg.hn_call(S0, k, p, n, h1)) for k in ks]
    zs = []
    for seed in range(4):
        st = S0 * torch.exp(hg.hn_simulate(p, n, h1, 2_000_000, seed=200 + seed, device=DEV))
        for k, c in zip(ks, cf):
            pay = torch.clamp(st - k, min=0) * math.exp(-float(p.r) * n)
            se = float(pay.std() / math.sqrt(len(pay)))
            zs.append((c - float(pay.mean())) / se)
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.6, zs.mean()
    assert np.abs(zs).max() < 4.5, zs


# ---- (a) PUBLISHED reference values -------------------------------------------------
#
# TWO INCOMPATIBLE CALLING CONVENTIONS EXIST IN THE WILD.  Getting this wrong silently
# shifts gamma* by lambda + 1/2:
#   * R ``fOptions`` / python ``finoptions``: you pass PHYSICAL (lambda, gamma); the code
#     derives gamma* = gamma + lambda + 1/2, lambda* = -1/2, and FORCES h(0) to the
#     stationary variance (you cannot supply h0).
#   * Rouah-Vainberg VBA / PyPI ``hngoption`` / Christoffersen-Jacobs-Ornthanalai: you
#     pass RISK-NEUTRAL (lambda* = -1/2, gamma*) and supply h0 yourself.
# THIS MODULE uses the second convention (``HNParams.gamma`` IS gamma*).  Both reference
# sets below are translated into it explicitly.

FOPTIONS = dict(omega=2.3e-6, alpha=2.9e-6, beta=0.85,
                gamma=184.25 + (-0.5) + 0.5,      # physical lambda = -0.5 => gamma* = gamma
                r=0.05 / 252.0)


def test_published_foptions_hngoption_example():
    """PUBLISHED BENCHMARK 1 -- the documented example of R ``fOptions::HNGOption``.

    Source: CRAN fOptions manual page ``HestonNandiOptions``
    (https://rdrr.io/cran/fOptions/man/HestonNandiOptions.html), example block
        model = list(lambda=-0.5, omega=2.3e-6, alpha=2.9e-6, beta=0.85, gamma=184.25)
        S=100, X=100, Time.inDays=252, r.daily=0.05/252
    documented call price 8.992100, and independently reproduced to full double
    precision by the python ``finoptions`` package's own test-suite
    (https://github.com/bbcho/finoptions-dev/blob/main/pytest/test_heston_options.py):
        call 8.9920997701416, put 4.115042220213013.

    fOptions forces h(0) to the stationary variance, so this ALSO pins
    ``HNParams.stationary_var`` = (omega+alpha)/(1-beta-alpha*gamma*^2) = 1.008717e-4.
    """
    p = hg.HNParams(**FOPTIONS).as_tensors(DT)
    h0 = float(p.stationary_var)
    assert abs(h0 - 1.008717e-4) < 1e-10
    assert abs(float(hg.hn_call(S0, 100.0, p, 252, h0)) - 8.9920997701416) < 1e-11
    assert abs(float(hg.hn_put(S0, 100.0, p, 252, h0)) - 4.115042220213013) < 1e-11
    assert abs(float(hg.hn_call(S0, 90.0, p, 252, h0)) - 15.854473) < 5e-7


def test_published_foptions_integrand_at_phi_20():
    """PUBLISHED BENCHMARK 1b -- fOptions' INTEGRAND ``fstarHN`` at phi = 20, same
    parameters.  This isolates the A/B recursion from the quadrature: if the price is
    off but this matches, the bug is in the integration, and vice versa.

    fOptions reports    const=1 -> 0.01201464798465922 ,  const=0 -> 0.00015242037661681416
    Its ``fstarHN`` carries a 1/pi factor, and the const=1 (share-measure) contour
    additionally carries the spot; with S = X the log-moneyness shift is 1, so the
    comparable quantity here is  (S or 1)/pi * Re[ exp(A + B h0) / (i phi) ].
    """
    p = hg.HNParams(**FOPTIONS).as_tensors(DT)
    h0 = float(p.stationary_var)
    iphi = torch.tensor([20.0], dtype=DT) * 1j
    for const, target, fac in ((1.0, 0.01201464798465922, S0 / math.pi),
                               (0.0, 0.00015242037661681416, 1.0 / math.pi)):
        a, b = hg.hn_ab(iphi + const, p, 252)
        got = float((torch.exp(a + b * h0) / iphi).real) * fac
        assert abs(got - target) < 1e-14 * target, (const, got, target)


def test_published_rouah_vainberg_chapter6():
    """PUBLISHED BENCHMARK 2 -- Rouah & Vainberg, *Option Pricing Models and Volatility
    Using Excel-VBA*, ch. 6 pp. 178-179.  RISK-NEUTRAL parameters (lambda* = -1/2, so
    the quoted gamma IS gamma*) and a USER-SUPPLIED h0, which is the other half of the
    interface that BENCHMARK 1 cannot reach (fOptions always uses stationary h0).

        alpha=1.32e-6, beta=0.589, gamma*=421.39, omega=5.02e-6
        S=K=100, T=100 days, h0 = 0.15^2/252, r=0   ->  book quotes $2.4767

    The book ALSO quotes $3.4735 "with a yearly interest rate of 5 percent".  We do not
    reproduce that number under ANY day-count (see ``test_rouah_five_percent_figure_is_
    not_reproducible``); $3.4735 implies a daily r of 1.787e-4, which is 4.50%/252 --
    not 5% on any convention.  The r=0 figure is the usable benchmark.
    """
    p = hg.HNParams(5.02e-6, 1.32e-6, 0.589, 421.39, 0.0).as_tensors(DT)
    got = float(hg.hn_call(S0, 100.0, p, 100, 0.15 ** 2 / 252.0))
    assert abs(got - 2.476704) < 1e-6, got


def test_rouah_five_percent_figure_is_not_reproducible():
    """Documents the ONE published number we could not match, and pins what we DO get,
    so a future reader does not re-open the question.  An independently written
    third-party implementation produced exactly these same three values, which locates
    the discrepancy in the book, not here."""
    h0 = 0.15 ** 2 / 252.0
    for basis, want in ((252.0, 3.594107), (365.0, 3.224850), (360.0, 3.235984)):
        p = hg.HNParams(5.02e-6, 1.32e-6, 0.589, 421.39, 0.05 / basis).as_tensors(DT)
        assert abs(float(hg.hn_call(S0, 100.0, p, 100, h0)) - want) < 1e-6
    assert not any(abs(w - 3.4735) < 1e-3 for w in (3.594107, 3.224850, 3.235984))


@pytest.mark.parametrize('n,k,want', [
    (30, 90, 10.633858), (30, 100, 2.489571), (30, 110, 0.049684),
    (90, 90, 12.161075), (90, 100, 4.724929), (90, 110, 0.936175),
    (252, 90, 15.854473), (252, 100, 8.992100), (252, 110, 4.279543),
    (504, 90, 20.733100), (504, 100, 14.221243), (504, 110, 9.132086)])
def test_third_party_implementation_grid(n, k, want):
    """CROSS-IMPLEMENTATION check on the fOptions parameter set: a separately written
    implementation of the same recursion, over a 3-strike x 4-maturity grid.  Not a
    published table (only the (252, 100) and (252, 90) cells are), but it exercises
    short and long maturities and deep OTM where quadrature error would show first."""
    p = hg.HNParams(**FOPTIONS).as_tensors(DT)
    got = float(hg.hn_call(S0, float(k), p, n, float(p.stationary_var)))
    assert abs(got - want) < 1e-6, (n, k, got, want)


def test_heston_nandi_2000_mle_set_regression():
    """REGRESSION LOCK on the Heston-Nandi S&P500 MLE parameter set, with its known
    provenance wrinkle.

    Heston & Nandi (1997) FRB Atlanta WP 97-9 Table 1(a) -- the working-paper version of
    the RFS (2000) paper -- prints omega=5.02e-6, beta=0.589, gamma=421.39, lambda=0.205
    and **alpha=1.0e-6**, with a stated long-run annualised vol of 8.02%.  Every
    downstream quotation of the PUBLISHED table (Rouah-Vainberg; Christoffersen, Jacobs &
    Ornthanalai CREATES RP 2012-50 appendix code, commented "mostly from Heston and Nandi
    (2000) Table 1(a)") instead uses **alpha=1.32e-6**.  The RFS version is paywalled and
    could not be read, so WHICH alpha the published table carries is UNRESOLVED.

    Both are locked below.  alpha=1.0e-6 reproduces the working paper's own printed
    8.02% long-run vol (we get 8.07%), which is evidence that 1.0e-6 is the WP value and
    1.32e-6 is a later revision or a transcription error propagated through the ecosystem.

    The prices here are OUR output (no published price exists for this set); the pricer
    itself is validated by the two published benchmarks above.
    """
    for alpha, want_psi, want_vol, want in (
            (1.0e-6, 0.767164, 0.080719, [11.1237046772, 2.5984016955, 0.0614696912]),
            (1.32e-6, 0.824177, 0.095325, [11.1718940770, 2.9366492383, 0.1317679989])):
        p = hg.HNParams(5.02e-6, alpha, 0.589, 421.39 + 0.205 + 0.5,
                        0.05 / 365.0).as_tensors(DT)
        assert abs(float(p.persistence) - want_psi) < 1e-5
        assert abs(float(p.ann_vol(252.0)) - want_vol) < 1e-5
        h1 = float(p.stationary_var)
        got = [float(hg.hn_call(S0, k, p, 90, h1)) for k in (90.0, 100.0, 110.0)]
        for g, w in zip(got, want):
            assert abs(g - w) < 5e-8, (alpha, got, want)
        fine = [float(hg.hn_call(S0, k, p, 90, h1, phi_max=4000.0, panels=2048, order=12))
                for k in (90.0, 100.0, 110.0)]
        for g, f in zip(got, fine):
            assert abs(g - f) < 1e-9, (alpha, got, fine)


# ======================================================================================
# (d) monotonicity / smile shape
# ======================================================================================

def test_price_increasing_in_h1_and_in_maturity():
    p = _p()
    m = float(p.stationary_var)
    prev = -1.0
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        c = float(hg.hn_call(S0, S0, p, 63, m * mult))
        assert c > prev
        prev = c
    prev = -1.0
    for n in (5, 21, 63, 126, 252):
        c = float(hg.hn_call(S0, S0, p, n, m))
        assert c > prev
        prev = c


def test_gamma_star_sign_controls_skew():
    """gamma* > 0 (leverage) => NEGATIVE implied-vol skew; gamma* < 0 => positive; the
    smile is symmetric at gamma* = 0.  Skew measured as IV(90%) - IV(110%)."""
    n = 63

    def skew(gamma):
        # hold persistence and stationary variance fixed so only the leverage sign moves
        alpha, beta = 0.12 * 0.98 / gamma ** 2, 0.98 * 0.88
        omega = (0.04 / 252) * 0.02 - alpha
        p = hg.HNParams(omega, alpha, beta, gamma, R_STEP).as_tensors(DT)
        h1 = float(p.stationary_var)
        return (hg.hn_implied_vol(S0, 0.90 * S0, p, n, h1)
                - hg.hn_implied_vol(S0, 1.10 * S0, p, n, h1))

    assert skew(400.0) > 0.01           # negative skew: low strikes richer
    assert skew(-400.0) < -0.01         # mirror image
    assert abs(skew(400.0) + skew(-400.0)) < 5e-3
    assert abs(skew(1e-9)) < 1e-4       # gamma* -> 0: symmetric


def test_smile_is_curved():
    """HN must produce a genuine smile, not a flat BS vol."""
    p = _p()
    h1 = float(p.stationary_var)
    ivs = [hg.hn_implied_vol(S0, k * S0, p, 63, h1) for k in (0.85, 1.0, 1.15)]
    assert ivs[0] > ivs[1] > ivs[2]                       # monotone skew
    assert ivs[0] - ivs[2] > 0.02                         # material


# ======================================================================================
# numerics: branch cuts, quadrature convergence, gradients
# ======================================================================================

@pytest.mark.parametrize('cfg', [(0.20, 0.98, 400.0, 0.12), (0.35, 0.99, 500.0, 0.55),
                                 (1.20, 0.995, 2000.0, 0.95)])
@pytest.mark.parametrize('n', [252, 2520])
def test_no_branch_winding(cfg, n):
    """1 - 2*alpha*B never leaves the right half plane on either inversion contour, out
    to u = 20000 -- far past any phi_max the quadrature uses.  This is WHY the discrete
    'Heston trap' does not bite here; the unwrap in ``_clog`` is a no-op guard."""
    p = _p(*cfg)
    nodes, _ = hg.gauss_legendre(0.0, 20000.0, 512, 8, DT)
    for contour in (0.0, 1.0):
        z = nodes * 1j + contour
        B = torch.zeros_like(z)
        lin = z * (p.gamma - 0.5) - 0.5 * p.gamma ** 2
        hs = 0.5 * (z - p.gamma) ** 2
        for _ in range(n):
            w = 1.0 - 2.0 * p.alpha * B
            assert float(w.real.min()) > 0.999999
            assert float(torch.angle(w).abs().max()) < 0.5 * math.pi
            B = lin + p.beta * B + hs / w


@pytest.mark.parametrize('n', [63, 252, 1260])
def test_unwrap_is_a_noop_but_kept(n):
    p = _p(0.35, 0.99, 500.0, 0.55)
    h1 = float(p.stationary_var)
    a = float(hg.hn_call(S0, S0, p, n, h1, unwrap=True))
    b = float(hg.hn_call(S0, S0, p, n, h1, unwrap=False))
    assert abs(a - b) < 1e-11


@pytest.mark.parametrize('n', [1, 5, 63, 252])
def test_quadrature_is_converged(n):
    """Default (auto phi_max, 256 panels x 8-pt GL) vs 4x the range and 8x the nodes."""
    p = _p()
    h1 = float(p.stationary_var)
    pm = hg.auto_phi_max(p, n, torch.tensor(h1, dtype=DT))
    for k in (0.8, 1.0, 1.2):
        a = float(hg.hn_call(S0, S0 * k, p, n, h1))
        b = float(hg.hn_call(S0, S0 * k, p, n, h1, phi_max=4 * pm, panels=2048, order=12))
        assert abs(a - b) < 1e-9, (n, k, a, b)


def test_gradients_match_finite_difference():
    """Differentiable w.r.t. (omega, alpha, beta, gamma*, r, h1) -- the AAD requirement.
    Quadrature is PINNED so the auto-phi_max scan cannot step between FD bumps."""
    kw = dict(phi_max=512.0, panels=256)
    base = [2.4396e-06, 7.35e-07, 0.8624, 400.0, R_STEP]
    h1b = 1.5873e-4

    def price(vals, h1v):
        return hg.hn_call(_t(S0), _t(105.0), hg.HNParams(*vals), 63, h1v, **kw)

    vals = [torch.tensor(v, dtype=DT, requires_grad=True) for v in base]
    h1 = torch.tensor(h1b, dtype=DT, requires_grad=True)
    price(vals, h1).backward()
    for i in range(6):
        x0 = (base + [h1b])[i]
        eps = abs(x0) * 1e-6

        def bump(d):
            v = [torch.tensor(u, dtype=DT) for u in base]
            hh = torch.tensor(h1b, dtype=DT)
            if i < 5:
                v[i] = v[i] + d
            else:
                hh = hh + d
            return float(price(v, hh).detach())

        fd = (bump(eps) - bump(-eps)) / (2 * eps)
        ad = float((vals + [h1])[i].grad)
        assert abs(ad - fd) < 1e-6 * abs(fd), (i, ad, fd)


# ======================================================================================
# DELIVERABLE 2 primitives: E[Sum h] and the exact cumulants
# ======================================================================================

@pytest.mark.parametrize('h_mult', [0.5, 1.0, 2.0])
def test_expected_sum_h_exact_at_n1_and_matches_mc(h_mult):
    p = _p()
    h1 = h_mult * float(p.stationary_var)
    assert abs(float(hg.hn_expected_sum_h(p, 1, _t(h1))) - h1) < 1e-18
    for n in (5, 21, 63):
        cf = float(hg.hn_expected_sum_h(p, n, _t(h1)))
        mc, se = hg.hn_simulate_sum_h(p, n, h1, 2_000_000, seed=7, device=DEV)
        assert abs(cf - mc) < 4.0 * se, (n, cf, mc, se)
    # per-k path must sum to the closed form
    for n in (5, 21, 63):
        tot = float(hg.hn_expected_sum_h(p, n, _t(h1)))
        assert abs(float(hg.hn_expected_h_path(p, n, _t(h1)).sum()) - tot) < 1e-14 * tot


def test_cumulant_one_is_minus_half_expected_sum_h():
    """LRNVR: E[R_n] = n*r - V/2 with V = E[Sum h].  Ties the cumulant machinery to the
    closed-form variance forecast."""
    p = _p()
    for h_mult in (0.5, 1.0, 2.0):
        h1 = h_mult * float(p.stationary_var)
        for n in (1, 5, 21, 63):
            k1 = hg.hn_cumulants(p, n, _t(h1), 1)[0]
            v = float(hg.hn_expected_sum_h(p, n, _t(h1)))
            assert abs(k1 - (float(p.r) * n - 0.5 * v)) < 1e-15


def test_n1_aggregate_is_exactly_gaussian():
    p = _p()
    h1 = float(p.stationary_var)
    k1, k2, sk, ek = hg.hn_moments(p, 1, _t(h1))
    assert abs(k2 - h1) < 1e-18 and abs(sk) < 1e-12 and abs(ek) < 1e-12


@pytest.mark.parametrize('n', [5, 21, 63])
def test_cumulants_match_mc(n):
    p = _p()
    h1 = float(p.stationary_var)
    _, k2, sk, ek = hg.hn_moments(p, n, _t(h1))
    r = hg.hn_simulate(p, n, h1, 8_000_000, seed=11, device=DEV)
    d = r - r.mean()
    v = float((d ** 2).mean())
    assert abs(k2 / v - 1.0) < 2e-3
    assert abs(sk - float((d ** 3).mean()) / v ** 1.5) < 5e-3
    assert abs(ek - (float((d ** 4).mean()) / v ** 2 - 3.0)) < 1e-2


def test_true_variance_exceeds_expected_sum_h():
    """THE POINT OF DELIVERABLE 2, in one assertion: the 'mean-matched normal bridge'
    N(nr - V/2, V) with V = E[Sum h] is NOT even variance-matched.  Leverage makes
    Cov(Sum sqrt(h) z, Sum h) < 0, and R_n = nr - (1/2)Sum h + Sum sqrt(h) z, so

        Var(R_n) = V + (1/4)Var(Sum h) - Cov(Sum sqrt(h) z, Sum h)  >  V

    and the excess grows with n."""
    p = _p()
    h1 = float(p.stationary_var)
    prev = 1.0
    for n in (1, 5, 21, 63):
        v = float(hg.hn_expected_sum_h(p, n, _t(h1)))
        k2 = hg.hn_cumulants(p, n, _t(h1), 2)[1]
        assert k2 >= v * (1.0 - 1e-15)
        assert k2 / v >= prev - 1e-15
        prev = k2 / v
    assert prev > 1.005                     # >0.5% variance shortfall by n=63


@pytest.mark.parametrize('n', [5, 21, 63])
def test_cdf_matches_mc_in_the_tails(n):
    """The exact survival probability used by DELIVERABLE 2, vs brute force."""
    p = _p()
    h1 = float(p.stationary_var)
    v = float(hg.hn_expected_sum_h(p, n, _t(h1)))
    mu = float(p.r) * n - 0.5 * v
    r = hg.hn_simulate(p, n, h1, 8_000_000, seed=13, device=DEV)
    for z in (-2.5, -1.0, 0.0, 1.0, 2.5):
        b = mu + z * math.sqrt(v)
        cf = float(hg.hn_cdf_logret(b, p, n, h1))
        ind = (r <= b).to(DT)
        mc, se = float(ind.mean()), float(ind.std() / math.sqrt(len(ind)))
        assert abs(cf - mc) < 4.5 * se, (n, z, cf, mc, se)


def test_cdf_is_a_valid_distribution_function():
    p = _p()
    h1 = float(p.stationary_var)
    for n in (5, 21, 63):
        bs = torch.linspace(-1.0, 1.0, 41, dtype=DT)
        f = hg.hn_cdf_logret(bs, p, n, h1)
        assert bool((f.diff() > -1e-13).all())
        assert float(f[0]) < 1e-10 and float(f[-1]) > 1.0 - 1e-10
