"""Test-only Heston-Nandi reference helpers -- NOT part of the shipping framework.

The CORE HN model (the A/B recursion, the closed-form pricers, the daily-step recursion
``hn_variance_step``, the BS reference) lives in ``riskflow.utils`` because the framework
itself consumes it (bootstrappers.py, pricing.py, stochasticprocess.py).  Everything HERE is
consumed ONLY by the HN test suite: the brute-force daily Monte Carlo cross-check, the exact
aggregate cumulants / variance-forecast, and the target-hitting parameter builder the tests
use to construct example laws.  They route through ``utils.hn_variance_step`` /
``utils.hn_logmgf`` so the h-recursion stays a single source of truth.
"""

import numpy as np
import torch

from riskflow.utils import HNParams, hn_variance_step, hn_logmgf


def hn_params_from_targets(ann_vol, persistence, gamma, leverage_share, r=0.0,
                           steps_per_year=252.0):
    """Build an HNParams hitting a target annualised vol / persistence / gamma*.

    ``leverage_share`` = alpha*gamma*^2 / psi, i.e. the fraction of the persistence that
    comes from the ARCH-with-leverage channel rather than from beta.  Solving:

        alpha = leverage_share * psi / gamma^2
        beta  = psi - alpha * gamma^2 = psi * (1 - leverage_share)
        omega = m * (1 - psi) - alpha,     m = ann_vol^2 / steps_per_year

    Raises if that omega is non-positive (the target triple is infeasible: very high
    persistence leaves almost no room for alpha).
    """
    m = ann_vol ** 2 / steps_per_year
    alpha = leverage_share * persistence / gamma ** 2
    beta = persistence * (1.0 - leverage_share)
    omega = m * (1.0 - persistence) - alpha
    if omega <= 0.0:
        raise ValueError(
            'infeasible: omega=%.3e <= 0 (m*(1-psi)=%.3e < alpha=%.3e); '
            'raise gamma, lower leverage_share, or lower persistence'
            % (omega, m * (1.0 - persistence), alpha))
    return HNParams(omega, alpha, beta, gamma, r)


# ======================================================================================
# affine variance forecast and exact cumulants
# ======================================================================================

def hn_expected_sum_h(p, n_steps, h1):
    """E_t[ Sum_{k=1..n} h_{t+k} ] in closed form.

    From E_t[h_{t+k+1}] = omega + alpha + psi * E_t[h_{t+k}] (because
    E[(z - gamma sqrt(h))^2 | h] = 1 + gamma^2 h), so with m = (omega+alpha)/(1-psi)

        E_t[h_{t+k}] = m + psi^(k-1) * (h1 - m)          k = 1..n
        E_t[Sum]     = n*m + (h1 - m) * (1 - psi^n)/(1 - psi)

    EXACT at n=1 (returns h1 identically), which ``test_expected_sum_h`` pins.
    """
    psi = p.persistence
    m = (p.omega + p.alpha) / (1.0 - psi)
    n = int(n_steps)
    return n * m + (h1 - m) * (1.0 - psi ** n) / (1.0 - psi)


def hn_expected_h_path(p, n_steps, h1):
    """E_t[h_{t+k}] for k = 1..n as a tensor (same closed form, per-k)."""
    psi = p.persistence
    m = (p.omega + p.alpha) / (1.0 - psi)
    k = torch.arange(int(n_steps), dtype=p.omega.dtype, device=p.omega.device)
    return m + psi ** k * (h1 - m)


def hn_cumulants(p, n_steps, h1, order=4):
    """Exact cumulants kappa_1..kappa_order of R_n = log(S_{t+n}/S_t), by autodiff of
    the real-phi log-MGF at phi = 0.  Machine precision, no Monte Carlo.

    kappa_1 must equal n*r - V/2 with V = E[Sum h] (LRNVR); kappa_2 is the TRUE variance
    of the aggregate, which is NOT V (see the module report).
    """
    phi = torch.zeros((), dtype=p.omega.dtype, device=p.omega.device, requires_grad=True)
    h1t = torch.as_tensor(h1, dtype=p.omega.dtype, device=p.omega.device)
    y = hn_logmgf(phi, p, n_steps, h1t)
    out = []
    for _ in range(order):
        # n = 1 is EXACTLY Gaussian, so the graph legitimately terminates at kappa_2 and
        # every higher cumulant is identically zero -- not an error condition.
        if not y.requires_grad:
            out.append(torch.zeros_like(phi))
            continue
        y = torch.autograd.grad(y, phi, create_graph=True, allow_unused=True)[0]
        if y is None:
            y = torch.zeros_like(phi)
        out.append(y)
    return [float(c.detach()) for c in out]


def hn_moments(p, n_steps, h1):
    """(mean, var, skew, excess_kurtosis) of the n-step aggregate log-return, exact."""
    k1, k2, k3, k4 = hn_cumulants(p, n_steps, h1, 4)
    return k1, k2, k3 / k2 ** 1.5, k4 / k2 ** 2


# ======================================================================================
# Monte Carlo (the independent cross-check, and the brute-force answer for D2)
# ======================================================================================

@torch.no_grad()
def hn_simulate(p, n_steps, h1, n_paths, seed=0, device='cpu', dtype=torch.float64,
                chunk=2 ** 21, track_extrema=False):
    """Daily-stepped exact simulation of the risk-neutral HN recursion.

    Returns ``R`` (aggregate log-return, shape (n_paths,)) and, when ``track_extrema``,
    also ``(Rmax, Rmin)`` -- the running max/min of the log-return over the n DAILY
    observations (used to price the daily-monitored variant, which is a DIFFERENT
    product from the discrete fixing).
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    om, al, be, ga, r = (float(x) for x in (p.omega, p.alpha, p.beta, p.gamma, p.r))
    outs, mx, mn = [], [], []
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        h = torch.full((m,), float(h1), dtype=dtype, device=device)
        x = torch.zeros((m,), dtype=dtype, device=device)
        hi = torch.full((m,), -1e300, dtype=dtype, device=device)
        lo = torch.full((m,), 1e300, dtype=dtype, device=device)
        for _ in range(int(n_steps)):
            z = torch.randn((m,), generator=g, dtype=dtype, device=device)
            sh = h.sqrt()
            x = x + (r - 0.5 * h + sh * z)
            h = hn_variance_step(h, sh, z, om, al, be, ga)
            if track_extrema:
                hi = torch.maximum(hi, x)
                lo = torch.minimum(lo, x)
        outs.append(x)
        if track_extrema:
            mx.append(hi)
            mn.append(lo)
        done += m
    R = torch.cat(outs)
    return (R, torch.cat(mx), torch.cat(mn)) if track_extrema else R


@torch.no_grad()
def hn_simulate_sum_h(p, n_steps, h1, n_paths, seed=0, device='cpu',
                      dtype=torch.float64, chunk=2 ** 21):
    """Brute-force E_t[Sum_{k=1..n} h_{t+k}] -- the check on :func:`hn_expected_sum_h`.
    Returns (mean, standard_error)."""
    g = torch.Generator(device=device).manual_seed(int(seed))
    om, al, be, ga = (float(x) for x in (p.omega, p.alpha, p.beta, p.gamma))
    tot, done = [], 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        h = torch.full((m,), float(h1), dtype=dtype, device=device)
        s = torch.zeros((m,), dtype=dtype, device=device)
        for _ in range(int(n_steps)):
            s = s + h
            z = torch.randn((m,), generator=g, dtype=dtype, device=device)
            h = hn_variance_step(h, h.sqrt(), z, om, al, be, ga)
        tot.append(s)
        done += m
    s = torch.cat(tot)
    return float(s.mean()), float(s.std() / np.sqrt(len(s)))
