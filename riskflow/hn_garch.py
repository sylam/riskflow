"""
Heston-Nandi GARCH(1,1) -- semi-analytic European option pricing via the recursive
characteristic function, plus the affine variance-forecast machinery.

THE MODEL MODULE.  This holds the Heston-Nandi model itself: the parameters, the A/B
backward recursion, the daily-step simulation recursion, and the exact aggregate cumulants.
The MODEL-AGNOSTIC Fourier-inversion machinery (composite Gauss-Legendre quadrature, the
adaptive ``phi_max`` scan, the branch-continuity complex log, and the P1/P2 primitive) lives
once in ``riskflow.utils`` next to the other pricing primitives; ``hn_call`` / ``hn_put`` /
``hn_cdf_logret`` here are thin delegations that pass the HN log-CF into it (Heston / Bates /
VG would reuse the same primitive with their own log-CF).  It IS the pricer behind the
``HestonNandiModelParameters`` bootstrapper (``riskflow/bootstrappers.py``), it supplies the
daily-step recursion shared by the one-step-survival (OSS) Monte Carlo pricers in
``riskflow/pricing.py``, and it quantifies the multi-step OSS aggregation bias there.

--------------------------------------------------------------------------------------
INDEXING CONVENTION (stated once, pinned by ``tests/test_hn_garch.py``)
--------------------------------------------------------------------------------------
We use the ORIGINAL Heston & Nandi (2000) convention, in which the conditional variance
carries the index of the END of the period it governs, and is therefore PREDICTABLE
(known one period ahead).  Writing one step as Delta and dropping it (Delta = 1 step):

    RETURN     log(S_{t+1} / S_t) = r + lambda * h_{t+1} + sqrt(h_{t+1}) * z_{t+1}
    VARIANCE   h_{t+1} = omega + beta * h_t + alpha * (z_t - gamma * sqrt(h_t))**2

so ``h_{t+1}`` -- the variance of the NEXT return -- is F_t-measurable.  Under the
risk-neutral measure Q (Heston-Nandi's LRNVR) the same functional form holds with

    lambda -> lambda* = -1/2        gamma -> gamma* = gamma + lambda + 1/2

which is the ONLY measure this module works in, so we parameterise directly in gamma*
(field ``gamma``) and hard-wire lambda* = -1/2:

    log(S_{t+1} / S_t) = r - 1/2 * h_{t+1} + sqrt(h_{t+1}) * z_{t+1}     z ~ N(0,1) iid
    h_{t+1}            = omega + beta * h_t + alpha * (z_t - gamma* sqrt(h_t))**2

    persistence         psi   = beta + alpha * gamma*^2
    stationary variance E[h]  = (omega + alpha) / (1 - psi)

Because ``h`` is predictable, the natural state handed to every function here is
``h1 = h_{t+1}`` (the variance of the FIRST simulated step), NOT ``h_t``.  Every
argument named ``h1`` means exactly that.  ``r`` is the PER-STEP risk-free rate.

--------------------------------------------------------------------------------------
GENERATING FUNCTION
--------------------------------------------------------------------------------------
    f_t(phi) = E_t[ S_T^phi ] = S_t^phi * exp( A_t + B_t * h_{t+1} )

with terminal conditions A_T = B_T = 0 and the backward recursions

    A_t = A_{t+1} + phi*r + B_{t+1}*omega - 1/2 * ln(1 - 2*alpha*B_{t+1})
    B_t = phi*(lambda + gamma) - 1/2*gamma^2 + beta*B_{t+1}
              + (1/2 * (phi - gamma)^2) / (1 - 2*alpha*B_{t+1})

Under Q (lambda = -1/2, gamma = gamma*) the linear-in-phi term is

    phi * (gamma* - 1/2) - 1/2 * gamma*^2                       <-- NOTE THE -phi/2

The ``-phi/2`` is NOT optional: it is what makes f_t(1) = S_t * exp(r*(T-t)), i.e. the
discounted spot a Q-martingale.  Dropping it (as some restatements of the recursion do)
gives B_{T-1} = 1/2 at phi = 1 and a non-martingale forward.  ``test_martingale_phi_one``
pins this.  See the derivation note at the bottom of this docstring block.

DERIVATION of the phi*(gamma-1/2) term, for the record.  With b = B_{t+1}, g = sqrt(h),
u = 1 - 2*alpha*b, and E[exp(c z + d z^2)] = (1-2d)^(-1/2) exp(c^2/(2(1-2d))):

    E_t[ exp(-phi/2 h + phi g z + alpha b (z - gamma g)^2) ]
        = u^(-1/2) exp( h * [ -phi/2 + (phi - 2 alpha b gamma)^2/(2u) + alpha b gamma^2 ] )
        = u^(-1/2) exp( h * [ (phi-gamma)^2/(2u) + phi*gamma - gamma^2/2 - phi/2 ] )

--------------------------------------------------------------------------------------
INVERSION
--------------------------------------------------------------------------------------
    C = S * P1 - K * exp(-r*n) * P2

    P1 = 1/2 + (1/pi) * Int_0^inf Re[ K^(-i phi) f(i phi + 1) / (i phi * f(1)) ] d phi
    P2 = 1/2 + (1/pi) * Int_0^inf Re[ K^(-i phi) f(i phi)     / (i phi)        ] d phi

P2 = Q(S_T > K) and P1 is the same probability under the share measure.  Because
K^(-i phi) f(i phi) = exp(-i phi * ln(K/S)) * exp(A + B h1), BOTH integrands depend on
(S, K) only through the log-moneyness ln(K/S) -- the spot scales out.  That is why
``hn_cdf_logret`` (the object DELIVERABLE 2 needs) shares all of this machinery.

BRANCH CUTS -- MEASURED RESULT: the discrete "Heston trap" DOES NOT ARISE HERE.
``ln(1 - 2 alpha B_{t+1})`` is evaluated at a complex argument, and taking the principal
branch independently at each backward step would be wrong if that argument wound around
the origin.  It does not.  Along BOTH inversion contours (phi = i*u and phi = i*u + 1,
u >= 0) the recursion keeps Re(B) <= 0, hence for alpha > 0

    Re(1 - 2*alpha*B) >= 1        and     |arg(1 - 2*alpha*B)| < pi/2

i.e. the argument never leaves the right half-plane, so the principal branch IS the
continuous branch.  Verified numerically to u = 20000 (far past any phi_max the
quadrature uses), n up to 2520 steps, over annualised vols 20%-120%, persistence
0.98-0.995, gamma* 400-2000: min Re(1-2 alpha B) = 1.0000 and max |arg| = 1.52 rad in
every case (``test_no_branch_winding``).  Prices with and without the unwrap agree to
1e-12 (``test_unwrap_is_a_noop_but_kept``).

The correct branch is anyway fixed by continuity in phi along the contour, anchored at
u = 0 where B == 0 exactly (for P2) and, for P1, at phi = 1 where B == 0 exactly by the
martingale property -- in both cases ln(1) = 0.  We evaluate the recursion VECTORISED
over an ascending grid of phi and unwrap along that grid (``utils.complex_log_unwrap``,
called from ``hn_ab``) as a zero-risk guard that costs ~nothing and would still be correct
if this recursion were ever re-used with alpha < 0 or with a non-LRNVR lambda.
``unwrap=False`` selects the naive principal branch.

QUADRATURE (in ``riskflow.utils``).  Composite Gauss-Legendre on [0, phi_max]: panel
endpoints are excluded so the removable 1/(i phi) singularity at phi = 0 is never evaluated
(``utils.gauss_legendre`` / ``utils.cf_european_probabilities``).  ``phi_max`` defaults to
the automatic scan ``utils.cf_adaptive_phi_max`` that doubles phi until the integrand
modulus ``Re(A + B h1) - ln(phi)`` drops below ``exp(-40)``, evaluated at the extreme h1 in
the batch (small h1 decays slowest).  The integrand's envelope is roughly
E[exp(-phi^2 Sum h / 2)], which by Jensen decays SLOWER than the pure-Gaussian
exp(-phi^2 V/2) -- hence the scan rather than a closed-form phi_max.

No overflow guard is needed in ``exp(A + B*h1)``: on the P2 contour |f| <= 1 so
Re(A + B*h1) <= 0, and on the P1 contour it is bounded by r*n; the scan only ever walks
phi_max further into the decaying region.

Everything is torch and differentiable w.r.t. (omega, alpha, beta, gamma, h1, r):
``torch.round`` inside the unwrap carries zero gradient, which is correct because the
unwrap correction is a locally-constant integer winding number.

--------------------------------------------------------------------------------------
VALIDATION (tests/test_hn_garch.py, 98 tests)
--------------------------------------------------------------------------------------
  * R ``fOptions::HNGOption`` documented example: call 8.9920997701416 and put
    4.115042220213013 reproduced to 3e-14 absolute; its integrand ``fstarHN`` at phi=20
    reproduced to 1e-15 relative (isolates the recursion from the quadrature).
  * Rouah & Vainberg ch.6: $2.476704 (r=0, user-supplied h0) reproduced to 5e-8.  Their
    r=5% figure of $3.4735 does NOT reproduce on any day-count -- see
    ``test_rouah_five_percent_figure_is_not_reproducible``.
  * alpha -> 0: exact Black-Scholes to 1e-14, and the CDF to 1e-12 out to 3.5 sigma.
  * f_t(1) = S_t e^{r n}: B(1) = 0 to 1e-14 at n = 1..252.
  * Daily Monte Carlo: prices, P1, P2, cumulants and tail CDFs all within MC error.
  * Gradients vs central differences: 1e-9 relative on all six inputs.

INTERFACE WARNING.  Two incompatible conventions circulate.  fOptions/finoptions take
PHYSICAL (lambda, gamma) and force h0 to the stationary variance; Rouah-Vainberg,
PyPI ``hngoption`` and Christoffersen-Jacobs-Ornthanalai take RISK-NEUTRAL (gamma*) with
a user-supplied h0.  THIS MODULE IS THE SECOND: ``HNParams.gamma`` is gamma*, and h1 is
always supplied by the caller.
"""

from dataclasses import dataclass

import numpy as np
import torch

from . import utils
from .utils import gauss_legendre   # re-exported: the composite Gauss-Legendre grid now
# lives in utils (model-agnostic quadrature); kept importable as ``hn_garch.gauss_legendre``.


# ======================================================================================
# parameters
# ======================================================================================

@dataclass
class HNParams:
    """Risk-neutral Heston-Nandi(1,1) parameters.  ``gamma`` IS gamma* (lambda* = -1/2).

    All fields may be python floats or 0-dim torch tensors; make them
    ``requires_grad=True`` tensors to differentiate prices w.r.t. them.
    ``r`` is the PER-STEP risk-free rate (per Delta, not annualised).
    """
    omega: object
    alpha: object
    beta: object
    gamma: object
    r: object = 0.0

    def as_tensors(self, dtype=torch.float64, device='cpu'):
        def t(x):
            return x if torch.is_tensor(x) else torch.tensor(x, dtype=dtype, device=device)
        return HNParams(t(self.omega), t(self.alpha), t(self.beta), t(self.gamma), t(self.r))

    @property
    def persistence(self):
        """psi = beta + alpha * gamma*^2."""
        return self.beta + self.alpha * self.gamma ** 2

    @property
    def stationary_var(self):
        """E[h] = (omega + alpha) / (1 - psi), the per-step stationary variance."""
        return (self.omega + self.alpha) / (1.0 - self.persistence)

    def ann_vol(self, steps_per_year=252.0):
        v = self.stationary_var * steps_per_year
        return float(v) ** 0.5 if not torch.is_tensor(v) else v.sqrt()


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
# the A/B recursion
# ======================================================================================

def hn_ab(phi, p, n_steps, unwrap=True, phi_dim=-1):
    """Backward A/B recursion for ``n_steps`` steps.  Returns ``(A, B)``.

    ``phi``     : real OR complex tensor.  If complex it is assumed to vary smoothly and
                  ascending along ``phi_dim`` (needed for the branch unwrap).
    ``p``       : HNParams (tensors).
    Result satisfies E_t[S_{t+n}^phi] = S_t^phi * exp(A + B * h_{t+1}); i.e. the HN affine
    log-CF of the aggregate log-return is ``A + B * h1`` (the closure handed to the
    model-agnostic inversion primitive ``utils.cf_european_probabilities``).
    """
    A = torch.zeros_like(phi)
    B = torch.zeros_like(phi)
    om, al, be, ga, r = p.omega, p.alpha, p.beta, p.gamma, p.r
    lin = phi * (ga - 0.5) - 0.5 * ga ** 2          # <-- the -phi/2 is the LRNVR drift
    half_sq = 0.5 * (phi - ga) ** 2
    phir = phi * r
    for _ in range(int(n_steps)):
        w = 1.0 - 2.0 * al * B
        logw = utils.complex_log_unwrap(w, dim=phi_dim) if (unwrap and w.is_complex()) else torch.log(w)
        A = A + phir + B * om - 0.5 * logw
        B = lin + be * B + half_sq / w
    return A, B


def hn_logmgf(phi, p, n_steps, h1, **kw):
    """log E_t[exp(phi * R_n)] where R_n = log(S_{t+n}/S_t).  = A + B*h1."""
    A, B = hn_ab(phi, p, n_steps, **kw)
    return A + B * h1


# ======================================================================================
# quadrature
# ======================================================================================

def auto_phi_max(p, n_steps, h1, log_tol=-40.0, start=8.0, cap=2.0 ** 24):
    """Smallest power-of-two phi_max with Re(A + B*h1) - ln(phi) < log_tol.

    The HN glue for the model-agnostic scan ``utils.cf_adaptive_phi_max``: it reduces the
    batch to the extreme h1 (the smallest, whose integrand decays slowest) so the scan runs
    on a 2-element phi and checks BOTH inversion contours (i*phi and i*phi+1, the latter
    normalised by the log forward-growth r*n).
    """
    h1t = torch.as_tensor(h1).detach()
    hs = torch.stack([h1t.min(), h1t.max()]).to(p.omega.dtype).reshape(-1, 1)
    carry = torch.as_tensor(p.r).detach() * int(n_steps)
    return utils.cf_adaptive_phi_max(
        lambda z: hn_logmgf(z, p, n_steps, hs), carry,
        p.omega.dtype, p.omega.device, log_tol, start, cap)


# ======================================================================================
# DELIVERABLE 1 -- prices and the exact aggregate CDF
# ======================================================================================

def _p1_p2(logm, p, n_steps, h1, phi_max, panels, order, unwrap, want=3):
    """P1, P2 for log-moneyness ``logm`` = ln(K/S).  ``logm``/``h1`` broadcast together.

    Thin HN glue over the model-agnostic Fourier-inversion primitive
    ``utils.cf_european_probabilities``: it hands over the HN affine log-CF ``A + B*h1``
    (from :func:`hn_ab`) as the ``logcf`` closure and the log forward-growth ``r*n`` as the
    P1-contour normalisation.  ``want`` is a bit mask: 1 = P1, 2 = P2, 3 = both.
    """
    logm = torch.as_tensor(logm, dtype=p.omega.dtype, device=p.omega.device)
    h1 = torch.as_tensor(h1, dtype=p.omega.dtype, device=p.omega.device)
    logm, h1 = torch.broadcast_tensors(logm, h1)
    if phi_max is None:
        phi_max = auto_phi_max(p, n_steps, h1)
    if panels is None:
        panels = 256
    hh = h1.unsqueeze(-1)

    def logcf(phi):
        A, B = hn_ab(phi, p, n_steps, unwrap=unwrap)
        return A + B * hh

    return utils.cf_european_probabilities(
        logcf, logm, p.r * n_steps, phi_max, panels, order,
        p.omega.dtype, p.omega.device, want)


def hn_call(S, K, p, n_steps, h1, phi_max=None, panels=None, order=8, unwrap=True):
    """European CALL, ``n_steps`` steps to expiry, spot ``S``, strike ``K``.

    ``h1`` is the (predictable) variance of the FIRST step.  Differentiable w.r.t.
    (omega, alpha, beta, gamma, r, h1, S, K).
    """
    S = torch.as_tensor(S, dtype=p.omega.dtype, device=p.omega.device)
    K = torch.as_tensor(K, dtype=p.omega.dtype, device=p.omega.device)
    P1, P2 = _p1_p2(torch.log(K / S), p, n_steps, h1, phi_max, panels, order, unwrap)
    return S * P1 - K * torch.exp(-p.r * n_steps) * P2


def hn_put(S, K, p, n_steps, h1, **kw):
    """European PUT.  By put-call parity off :func:`hn_call` (the parity residual of the
    inversion itself is tested separately via the phi=1 martingale identity)."""
    S = torch.as_tensor(S, dtype=p.omega.dtype, device=p.omega.device)
    K = torch.as_tensor(K, dtype=p.omega.dtype, device=p.omega.device)
    return hn_call(S, K, p, n_steps, h1, **kw) - S + K * torch.exp(-p.r * n_steps)


def hn_cdf_logret(b, p, n_steps, h1, phi_max=None, panels=None, order=8, unwrap=True):
    """EXACT  Q( R_n <= b )  where R_n = log(S_{t+n}/S_t), by Fourier inversion.

    This is the quantity the one-step-survival loop needs for an UP barrier at
    S*exp(b) (survival = stay below).  Spot-free by construction.  ``b`` and ``h1``
    broadcast against each other.
    """
    _, P2 = _p1_p2(b, p, n_steps, h1, phi_max, panels, order, unwrap, want=2)
    return 1.0 - P2


# ======================================================================================
# DELIVERABLE 2 -- affine variance forecast and exact cumulants
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
# The daily-step recursion -- ONE SOURCE OF TRUTH for the HN step
# ======================================================================================
#
# The predictable-variance recursion h_{t+1} = omega + beta*h_t + alpha*(z_t - gamma*sqrt(h_t))^2
# lives ONLY in ``hn_variance_step``.  Every consumer routes through it: the standalone daily
# simulator (``hn_simulate`` / ``hn_simulate_sum_h``) AND the one-step-survival (OSS) Monte
# Carlo pricers in ``riskflow/pricing.py`` (which import ``hn_daily_advance`` /
# ``hn_unmonitored_substeps`` from here).  So a single mutation-kill matrix on the step covers
# every pricer, and the simulator and the pricers can never drift apart.

def hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star):
    """The HN predictable-variance recursion h_{t+1} = omega + beta*h + alpha*(z - gamma*sqrt(h))^2.

    ``sh`` = sqrt(h) is passed in (the caller already needs it for the log-spot step), so the
    square root is computed exactly once.  All args broadcast on the simulation axis.
    """
    return omega + beta * h + alpha * (z - gamma_star * sh) ** 2


def hn_daily_advance(Sj, h, b_step, z, omega, alpha, beta, gamma_star):
    """One daily Heston-Nandi step under the risk-neutral (LRNVR) measure. Returns (Sj, h).

    Advances the log-spot by ``(b_step - 0.5*h) + sqrt(h)*z`` and recurses the predictable
    variance (via :func:`hn_variance_step`). ``z`` is either a fresh unconditional normal (an
    unmonitored sub-step) or the survival-truncated final draw of a monitored interval; in
    BOTH cases the recursion is fed the REALISED z (the survival-conditioned law -
    leverage-asymmetric under truncation, DO NOT 'fix' back to an unconditional draw, see the
    pv_MC_Tarf note). ``b_step`` is the per-step cost-of-carry (r-q). All args broadcast on
    the trailing simulation axis.
    """
    sh = torch.sqrt(h)
    Sj = Sj * torch.exp((b_step - 0.5 * h) + sh * z)
    h = hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star)
    return Sj, h


def hn_unmonitored_substeps(Sj, h, b_step, n_steps, hn_params, shared, num_sims, antithetic):
    """Advance (Sj, h) through ``n_steps`` UNCONDITIONAL (unmonitored) daily HN steps. These
    carry no barrier - the OSS truncation applies only on the monitored final step (done by
    the caller). A monitored interval of n_sub days passes ``n_steps = n_sub - 1`` here; a
    non-monitored interval (e.g. the run from the last barrier date to expiry) passes the full
    ``n_steps = n_sub``. Fresh regular-stream normals per step (Sobol/antithetic variance
    reduction is reserved for the truncated final draw); with ``antithetic`` the normal is
    negated on the paired half (z, -z) to align with the u<->1-u halves of the final draw
    (TARF/barrier), otherwise a plain num_sims-wide normal (autocall, whose final draw is not
    antithetic). ``hn_params`` = (Omega, Alpha, Beta, Gamma_Star).
    """
    for _ in range(n_steps):
        zc = torch.randn([shared.simulation_batch, num_sims],
                         dtype=shared.one.dtype, device=shared.one.device)
        z = torch.cat([zc, -zc], dim=-1) if antithetic else zc
        Sj, h = hn_daily_advance(Sj, h, b_step, z, *hn_params)
    return Sj, h


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


# ======================================================================================
# Black-Scholes reference + implied vol
# ======================================================================================

def _ncdf(x):
    return 0.5 * torch.erfc(-x / np.sqrt(2.0))


def bs_call(S, K, total_var, disc, fwd_growth):
    """BS call from TOTAL variance.  ``disc`` = exp(-r*n), ``fwd_growth`` = exp(r*n)."""
    S, K = torch.as_tensor(S), torch.as_tensor(K)
    F = S * fwd_growth
    sd = torch.as_tensor(total_var).sqrt()
    d1 = (torch.log(F / K) + 0.5 * total_var) / sd
    return disc * (F * _ncdf(d1) - K * _ncdf(d1 - sd))


def bs_call_np(S, K, r, n, total_var):
    d = np.exp(-r * n)
    return float(bs_call(torch.as_tensor(float(S), dtype=torch.float64),
                         torch.as_tensor(float(K), dtype=torch.float64),
                         torch.as_tensor(float(total_var), dtype=torch.float64),
                         d, 1.0 / d))


def bs_implied_total_var(price, S, K, r, n, lo=1e-12, hi=25.0, tol=1e-14, iters=200):
    """Bisection on TOTAL variance (no time units, so this is convention-free)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bs_call_np(S, K, r, n, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def hn_implied_vol(S, K, p, n_steps, h1, steps_per_year=252.0, **kw):
    """Annualised BS implied vol of the HN price (for the smile/skew diagnostics)."""
    c = float(hn_call(S, K, p, n_steps, h1, **kw))
    tv = bs_implied_total_var(c, float(S), float(K), float(p.r), int(n_steps))
    return np.sqrt(tv / (int(n_steps) / steps_per_year))
