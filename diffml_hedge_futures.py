"""
Differential-ML downside hedge of an average-rate (Asian) forward, hedged with FUTURES
on a DIFFERENT exchange than the averaging index -- so the hedge carries basis and carry
risk the liability does not have.  Standalone extension of the spot-hedge toy.

What changed vs the spot toy
----------------------------
Short an average-rate forward on the spot index S_t (an AR(1), mean-reverting -- no
Black-Scholes delta).  We may no longer trade S_t; instead we hedge with THREE exchange
futures whose price is

        F_i(t) = (S_t + B_t) * exp( c_t(tau_i) * tau_i ),   tau_i = T_exp_i - t,

i.e. the future references (index + cross-exchange BASIS) grossed up by CARRY to its
own expiry.  Three independent risk drivers move each future:

   dF_i = e^{c tau_i} dS            <- SPOT: shared with the liability
        + e^{c tau_i} dB            <- cross-exchange BASIS: the future has it, the deal does NOT
        + (S+B) e^{c tau_i} tau_i dc_i   <- CARRY move at this tenor

Crucially the LIABILITY is now marked at carry too: its remaining fixings are projected at
the index carry forward  S_t * exp(c_t(tau_u) tau_u)  (same NS curve, no basis).  So:
   * SPOT  is shared  -> hedgeable.
   * CARRY is shared  -> hedgeable, and because each tau_i loads on (level,slope,curvature)
            differently, the three futures are genuinely non-collinear: >=2 of them are
            needed to match the liability's spot AND carry exposure (not redundant).
   * BASIS is in the future only -> UNHEDGEABLE (pure tracking error).
DEAL SPREAD (the realistic bit): the deal is NOT struck at the money -- it is struck ~$5-8/oz
ABOVE fair, so the desk books a margin.  The strike level cancels in the mark-to-market wealth
dynamics, so the spread enters as a day-1 wealth CUSHION  N_INIT = NU*MARGIN  (see MARGIN).  That
turns the objective from "minimise variance" into "do not lose MORE than the spread, keep upside":
the spread is a downside BUDGET.  Consequences (shown in the downside table): the best STATIC hedge
UNDER-hedges (alpha~0.6, not 1.0); full min-var OVER-hedges (it spends upside to buy tail safety
the cushion already gives); the dynamic learned policy spends the budget on reversion-timing and
keeps far more upside at a comparable spread-loss probability.  AVERSION/DELTA set how much of the
spread to put at risk.  (Carry is still zero-mean -> no carry premium; the spread is structural.)
The value baseline U(N) is only approximate; the NN residual absorbs the rest.

Two new stochastic blocks supply B and c:
  * BASIS  : mean-reverting OU      dB = -alpha B dt + sigma_B dZ   (here AR(1) to 0).
  * CARRY  : a Nelson-Siegel curve  c(tau) = L0 + L1*g1(tau) + L2*g2(tau)  whose three
             factors (LEVEL L0, SLOPE L1, CURVATURE L2) are each a mean-reverting AR(1).
             Different tau_i load on (L0,L1,L2) differently -> the three futures are
             genuinely DIFFERENT carry instruments (not collinear), so the curve can be
             bucket-hedged.

EXPIRY / ROLL: the contracts expire DURING the averaging window (T_EXP inside [1,T]) so the
tradeable set shrinks -- early on all of {Apr,Jul,Oct} trade, then Apr expires, then Jul, and
near the payoff only Oct is left.  A contract is tradeable at t iff T_EXP_i > t; over its final
step tau->0 and the future converges to S+B (its basis is realised at settlement).  Holdings in
expired contracts are masked to zero everywhere (see `alive`).

OBJECTIVE is ONE-SIDED (asymmetric Huber): keep the upside, penalise the downside.  So the
right STATIC benchmark is the best CONSTANT-SCALE of the min-var shape under the utility, not
full min-var.  min-var is reported only as the DOWNSIDE (tail) benchmark -- a symmetric hedge
gives up the upside.  The dynamic learned policy should beat it by hedging MORE when losing and
LESS when winning (synthetic convexity a static hedge cannot produce).

Consequences the toy now exhibits
---------------------------------
  * Basis is UNHEDGEABLE: dL has no dB term, so no futures position removes basis P&L -- it caps
    how much variance any hedge can remove (var_reduc < 1 in the benchmark print).
  * The per-contract hedge is the multivariate projection h* = Sigma_F^{-1} Cov(dF, dL) over the
    ALIVE contracts, NOT three univariate ratios (the live futures share S+B, so Sigma_F is
    ill-conditioned -> ridge).  Computed by MC regression of dL on the live dF, printed per step.
  * Everything from the spot toy carries over: analytic BASELINE U(N) + NN RESIDUAL;
    TWIN (value + pathwise-gradient) loss via AAD; EXTERNAL argmax over a discrete action
    grid (now 3-D, one axis per contract) under COMMON RANDOM NUMBERS; the explicit,
    hyperparameter-free lambda de-bias; the FD gate (now across spot/basis/carry); and the
    BSS sandwich  L <= V* <= U  with the martingale-penalty zero-mean guard.

FINDINGS (what makes the one-sided hedge work)
----------------------------------------------
Result: the learned dynamic policy E[util] ~ +1.1 vs min-var -11.1; 5%-tail -4.8 vs -6.5 AND
~5.7x the upside (+36 vs +6.4).  It protects the downside BETTER than min-var while keeping the
upside -- robust across seeds.  Mechanism (see policy_response): it TIMES the AR(1) reversion --
net-LONGER when spot is below MU, shorter when above -- a synthetic convexity a static hedge can't
make.  Lever hierarchy, in order of impact:
  1. N_INNER (Bellman-argmax inner samples) -- DECISIVE.  Too few -> winner's-curse over-leverage
     and a blown tail (ni8 lost to min-var); ni32 wins.  Knee ~32.
  2. Standardised differential loss -- weight the gradient-MSE by the net's input scales, else the
     ~800-magnitude carry gradients swamp the value fit (val_loss 20x worse).  Free win.
  3. KAPPA (spot reversion) is the EDGE SOURCE: learned-vs-min-var gap scales +1.5 -> +12 -> +26
     as reversion strengthens.  No reversion -> no edge -> learned ~ min-var.
  4. SIGMA_B (basis) is the UNHEDGEABLE floor: degrades both; learned's edge widens at high basis.
  5. Action grid resolution / Huber aversion -- minor (grid finer = +0.2 at 2.7x cost).
Diagnostics must use the deployment N_INNER: a low-n_inner rollout fabricates apparent optimism.

Run:  python3 diffml_hedge_futures.py
"""
import math
import logging
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)          # double precision -> exact FD gate

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("diffml_futures").info   # diagnostics at INFO (filterable), never env-gated

# --------------------------------------------------------------------------- #
# Model / problem constants                                                   #
# --------------------------------------------------------------------------- #
T      = 8            # decision dates 0..T-1 ; averaging dates 1..T ; payoff at T
MU     = 100.0       # AR(1) long-run mean (the spot INDEX, e.g. LME average leg)
KAPPA  = 0.25        # spot mean-reversion speed
PHI    = 1.0 - KAPPA # AR(1) persistence
SIGMA  = 4.0         # spot per-step shock std (price level)
NU     = 3.0         # liability notional
S_INIT = MU
MARGIN = 6.5         # USD/oz the deal is struck ABOVE fair (the desk's spread; real deals are NOT
#   at-the-money -- we lock in $5-8/oz).  The strike level CANCELS in the mark-to-market wealth
#   dynamics (we track dL), so the spread enters as a day-1 wealth CUSHION N_INIT = NU*MARGIN.
#   Economically it is a downside BUDGET: the job is to not lose MORE than the spread, not to avoid
#   any loss -- so the policy can under-hedge and keep upside, spending the spread on tail risk.
N_INIT = NU * MARGIN
# STRIKE is set below to fair (E_0[avg]); the spread lives in N_INIT, not in STRIKE (which cancels).

# --- BASIS: cross-exchange spread (CME future index - LME averaging index), OU to 0 --- #
ALPHA   = 0.30       # basis mean-reversion speed
SIGMA_B = 1.5        # basis per-step shock std -> MATERIAL, unhedgeable basis risk
B_INIT  = 0.0

# --- CARRY: Nelson-Siegel curve c(tau)=L0+L1*g1+L2*g2, factors are mean-reverting AR(1) --- #
LAM     = 5.0                     # NS decay (in step units) -> shape of slope/curvature loadings
L0_BAR, L1_BAR, L2_BAR = 0.0, 0.0, 0.0                 # ZERO-MEAN carry: keeps carry a hedgeable
RHO0, RHO1, RHO2       = 0.90, 0.85, 0.80              #   risk (via its VOLS) WITHOUT a drift
SIG0, SIG1, SIG2       = 0.0015, 0.0020, 0.0025        #   premium.  Nonzero means would mark the
#   forward above the mean-reverting spot -> a carry risk premium that makes "don't hedge" +EV and
#   muddies the one-sided (downside) demonstration.  The VOLS are what make the three contracts load
#   DIFFERENTLY on (level,slope,curvature), so carry is genuinely hedgeable; the MEANS only set the
#   premium.  (Basis, SIGMA_B, is the material UNHEDGEABLE risk -- it is in the future, not the deal.)


def _ns_carry_scalar(tau):
    """Nelson-Siegel carry rate at a single tenor (scalar floats) -- the long-run curve.
    tau->0 (a contract at expiry) gives c=L0 (g1->1, g2->0)."""
    x = tau / LAM
    if abs(x) < 1e-8:
        return L0_BAR
    ex = math.exp(-x); g1 = (1.0 - ex) / x; g2 = g1 - ex
    return L0_BAR + L1_BAR * g1 + L2_BAR * g2


# fair strike: E_0[avg] with every remaining fixing projected at the long-run carry forward, so
# the deal starts at zero MTM (L_0 = 0).  With zero-mean carry this is just MU; nonzero carry means
# would lift it above MU and create the premium discussed above.
STRIKE = sum(MU * math.exp(_ns_carry_scalar(u) * u) for u in range(1, T + 1)) / T

# --- the three hedge futures (Apr/Jul/Oct analogues): expiries INSIDE the averaging window, so
#     contracts EXPIRE during the hedge and the tradeable set shrinks (roll).  At t the contract is
#     tradeable iff T_EXP_i > t; over its final step tau->0 and the future converges to S+B (the
#     basis is realised at settlement).  With T=8: Apr(3) trades t=0..2, Jul(6) t=0..5, Oct(9) all
#     t -- so t=0..2 has 3 contracts, t=3..5 has {Jul,Oct}, t=6..7 only {Oct}. --- #
T_EXP = torch.tensor([3.0, 6.0, 9.0])
NF    = T_EXP.numel()
DIM_M = 5            # market-state factors packed as M = [S, B, L0, L1, L2]


def alive(t):
    """1.0 for each contract still trading over the step starting at t (T_EXP_i > t), else 0.
    Multiplying holdings by this zeros out expired contracts everywhere (0 is on the QLEV grid,
    so a masked gridded action is still a valid grid action)."""
    return (T_EXP > t).double()                         # (NF,)

AVERSION = 1.5       # downside curvature near W=0
DELTA    = 8.0       # Huber knee: loss beyond which penalty goes LINEAR (set near a typical loss,
#   so the steep linear tail (slope 1+2*AVERSION*DELTA) only bites on genuine extremes)

# 3-D action grid: holdings (q_Apr, q_Jul, q_Oct).  Coarse per-contract grid kept small so
# the external argmax over the PRODUCT (NA = 5^3 = 125) stays cheap; wide enough to span
# the min-variance hedge and some leverage.
QLEV = torch.linspace(-1.5, 1.5, 5)
QG   = torch.cartesian_prod(QLEV, QLEV, QLEV)          # (NA, 3)
NA   = QG.shape[0]

# inner-MC samples for the external Bellman argmax -- the SINGLE most important knob (see decide).
# Knee sweep (E[util], 5%-tail vs min-var -11.07 / -6.49):  ni16 -4.5/-6.98 (loses tail) ->
# ni24 -0.3/-5.3 -> ni32 +0.1/-4.8 -> ni48 +0.3/-4.2 (best).  32 captures ~all the benefit at 2/3
# the cost of 48.  Used for deployment AND every diagnostic rollout (consistency: a low-n_inner
# rollout fabricates apparent over-optimism).
N_INNER = 32


# --------------------------------------------------------------------------- #
# Market dynamics, futures pricing, liability                                 #
# --------------------------------------------------------------------------- #
def init_market(n):
    """All factors at their initial / long-run values."""
    M = torch.empty(n, DIM_M)
    M[:, 0] = S_INIT
    M[:, 1] = B_INIT
    M[:, 2] = L0_BAR
    M[:, 3] = L1_BAR
    M[:, 4] = L2_BAR
    return M


def init_wealth(n):
    """Day-1 wealth = the booked spread cushion N_INIT = NU*MARGIN (deal struck above fair)."""
    return torch.full((n,), N_INIT)


def market_step(M, eps):
    """One-step joint evolution of M=[S,B,L0,L1,L2].  eps has matching trailing dim 5.
    Spot AR(1) to MU; basis OU to 0; each carry factor AR(1) to its mean."""
    S, B, L0, L1, L2 = M.unbind(-1)
    eS, eB, e0, e1, e2 = eps.unbind(-1)
    S1  = S + KAPPA * (MU - S) + SIGMA * eS
    B1  = (1.0 - ALPHA) * B + SIGMA_B * eB
    L0p = L0_BAR + RHO0 * (L0 - L0_BAR) + SIG0 * e0
    L1p = L1_BAR + RHO1 * (L1 - L1_BAR) + SIG1 * e1
    L2p = L2_BAR + RHO2 * (L2 - L2_BAR) + SIG2 * e2
    return torch.stack([S1, B1, L0p, L1p, L2p], dim=-1)


def carry_curve(taus, L0, L1, L2):
    """Nelson-Siegel carry at tenors `taus` (..., NF) given factors L0,L1,L2 (..., 1).
    g1 = (1-e^{-x})/x  (slope loading),  g2 = g1 - e^{-x}  (curvature loading), x=tau/LAM."""
    x  = taus / LAM
    ex = torch.exp(-x)
    xs = torch.where(x.abs() > 1e-8, x, torch.ones_like(x))   # avoid 0/0 at expiry (tau->0)
    g1 = torch.where(x.abs() > 1e-8, (1.0 - ex) / xs, torch.ones_like(x))   # g1 -> 1 as x -> 0
    g2 = g1 - ex                                              # g2 -> 0 as x -> 0  => c(0)=L0, F=S+B
    return L0 + L1 * g1 + L2 * g2                       # (..., NF)


def futures(M, t):
    """Futures prices F_i = (S+B) exp(c(tau_i) tau_i) for the NF contracts.  Returns (...,NF)."""
    S, B, L0, L1, L2 = M.unbind(-1)
    taus = (T_EXP - t)                                  # (NF,)  broadcasts on last axis
    c = carry_curve(taus, L0.unsqueeze(-1), L1.unsqueeze(-1), L2.unsqueeze(-1))
    return (S + B).unsqueeze(-1) * torch.exp(c * taus)  # (..., NF)


def utility(W):
    """Huber-style downside penalty: linear (uncapped) GAINS keep the upside; QUADRATIC
    small losses (tail/variance aversion); LINEAR deep tail beyond DELTA (bounds the value
    scale so the residual stays fittable; keeps a live gradient so deep losses still steer)."""
    loss = torch.clamp(-W, min=0.0)
    quad = AVERSION * loss ** 2
    lin  = AVERSION * DELTA ** 2 + 2.0 * AVERSION * DELTA * (loss - DELTA)
    return W - torch.where(loss <= DELTA, quad, lin)


def liab(M, pe, t):
    """MTM of the average-rate forward on the spot INDEX, L_t = NU*(E_t[avg]-STRIKE).
    Fixings already set are in `pe`; the current date t is observed at S (t>=1); every
    REMAINING fixing u is projected at the index CARRY forward  S * exp(c(tau_u) tau_u),
    tau_u = u - t, using the SAME Nelson-Siegel curve as the futures.  So the liability is a
    function of S AND the carry factors (L0,L1,L2) -- carry is now a SHARED, hedgeable risk.
    It is NOT a function of the basis B (the deal references the index, not the basis-laden
    future), so basis stays unhedgeable.  Marking at carry while spot mean-reverts means L is
    no longer a martingale -- a small carry risk premium the hedge cannot remove."""
    S, _B, L0, L1, L2 = M.unbind(-1)
    Et = pe + (S if t >= 1 else torch.zeros_like(S))            # realised + current fixing
    for u in range(t + 1, T + 1):                              # remaining fixings, carry-projected
        tau = u - t
        x = tau / LAM; ex = math.exp(-x); g1 = (1.0 - ex) / x; g2 = g1 - ex
        c = L0 + L1 * g1 + L2 * g2                              # NS carry rate at tenor tau
        Et = Et + S * torch.exp(c * tau)
    return NU * (Et / T - STRIKE)


# --------------------------------------------------------------------------- #
# Continuation value  C_t = U(N) + A_t(M, N)                                  #
# --------------------------------------------------------------------------- #
class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(DIM_M + 1, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        for p in self.body[-1].parameters():            # init A ~ 0  =>  C starts at baseline
            nn.init.zeros_(p)

    def forward(self, M, N):
        S, B, L0, L1, L2 = M.unbind(-1)
        x = torch.stack([(S - MU) / 10.0, B / 2.0,
                         (L0 - L0_BAR) / 0.01, (L1 - L1_BAR) / 0.01, (L2 - L2_BAR) / 0.01,
                         (N - N_INIT) / 10.0], dim=-1)            # centre on the cushion
        return self.body(x).squeeze(-1)


def continuation(nets, M, N, t, chunk=200_000):
    """C_t(M,N) = U(N) + A_t(M,N).  Terminal C_T = U(N).  Net eval is row-chunked."""
    base = utility(N)
    if t >= T:
        return base
    net = nets[t]
    if M.shape[0] <= chunk:
        return base + net(M, N)
    res = torch.empty_like(N)
    for i in range(0, M.shape[0], chunk):
        sl = slice(i, i + chunk)
        res[sl] = net(M[sl], N[sl])
    return base + res


# --------------------------------------------------------------------------- #
# External argmax over the 3-D action grid (Bellman max OUTSIDE the value)     #
# --------------------------------------------------------------------------- #
def action_values(nets, M, N, pe, t, n_inner, gen):
    """E[C_{t+1}] for every action in QG via inner MC over the next MARKET state.
    Common random numbers: the market shocks are action-independent and broadcast across
    all actions, so the argmax compares actions on identical draws.  Returns (n, NA)."""
    n = M.shape[0]
    eps = torch.randn(n, n_inner, DIM_M, generator=gen)         # market shocks (CRN)
    M1  = market_step(M[:, None, :], eps)                       # (n, n_inner, 5)
    F0  = futures(M, t)                                         # (n, NF)
    F1  = futures(M1, t + 1)                                    # (n, n_inner, NF)
    dF  = F1 - F0[:, None, :]                                   # (n, n_inner, NF)
    pe1 = pe + (M[:, 0] if t >= 1 else torch.zeros_like(M[:, 0]))
    dL  = liab(M1, pe1[:, None], t + 1) - liab(M, pe, t)[:, None]   # (n, n_inner)
    Qa  = QG * alive(t)                                         # zero expired contracts (NA, NF)
    hedge = torch.einsum('ad,nid->nai', Qa, dF)                 # (n, NA, n_inner)
    N1  = N[:, None, None] + hedge - dL[:, None, :]             # (n, NA, n_inner)
    M1e = M1[:, None, :, :].expand(n, NA, n_inner, DIM_M).reshape(-1, DIM_M)
    C1  = continuation(nets, M1e, N1.reshape(-1), t + 1).reshape(n, NA, n_inner)
    return C1.mean(-1)


def decide(nets, M, N, pe, t, n_inner=N_INNER, gen=None):
    """External Bellman argmax.  n_inner is deliberately LARGE: the max over NA noisy inner-MC
    action-values is upward-biased (winner's curse ~ sd*sqrt(2 ln NA)); with NA=125 and few
    inner draws the argmax locks onto the luckiest-noise (most leveraged) action -> over-optimism
    and a blown tail.  Raising n_inner cuts the selection noise (~1/sqrt(n_inner)), which is what
    lets the learned policy hedge correctly (more when losing, less when winning) and beat min-var
    on the downside while keeping the upside.  This is the single highest-leverage knob in the toy
    (see the N_INNER knee sweep); grid resolution and Huber aversion barely move the result."""
    with torch.no_grad():
        idx = action_values(nets, M, N, pe, t, n_inner, gen).argmax(-1)
        return QG[idx] * alive(t)                               # (n, NF), expired contracts -> 0


# --------------------------------------------------------------------------- #
# Greedy realised rollout (lambda label + lower bound)                         #
# --------------------------------------------------------------------------- #
def greedy_rollout(nets, M, N, pe, t_start, gen, n_inner=N_INNER):
    with torch.no_grad():
        M, N, pe = M.clone(), N.clone(), pe.clone()
        for t in range(t_start, T):
            q   = decide(nets, M, N, pe, t, n_inner=n_inner, gen=gen)       # (n, NF)
            eps = torch.randn(M.shape[0], DIM_M, generator=gen)
            M1  = market_step(M, eps)
            pe1 = pe + (M[:, 0] if t >= 1 else 0.0)
            dF  = futures(M1, t + 1) - futures(M, t)
            dL  = liab(M1, pe1, t + 1) - liab(M, pe, t)
            N   = N + (q * dF).sum(-1) - dL
            M, pe = M1, pe1
        return utility(N)


# --------------------------------------------------------------------------- #
# Exploratory bank (random actions around a rough futures hedge)               #
# --------------------------------------------------------------------------- #
def simulate_bank(n_paths, gen):
    M  = init_market(n_paths)
    N  = init_wealth(n_paths)
    pe = torch.zeros(n_paths)
    bank = {"M": [], "N": [], "pe": []}
    for t in range(T):
        bank["M"].append(M.clone()); bank["N"].append(N.clone()); bank["pe"].append(pe.clone())
        # centre exploration on a crude split of the liability spot-delta across contracts
        cur     = 1.0 if t >= 1 else 0.0
        delta_L = NU / T * (cur + sum(math.exp(_ns_carry_scalar(u - t) * (u - t))
                                      for u in range(t + 1, T + 1)))   # dL/dS, carry-projected
        a_i     = torch.tensor([math.exp(_ns_carry_scalar(float(E - t)) * float(E - t))
                                for E in T_EXP])               # ~ spot loading e^{c tau_i} of each future
        q_c     = (delta_L / (NF * a_i))                       # (NF,)
        q = (q_c[None, :] + 0.35 * torch.randn(n_paths, NF, generator=gen)).clamp(
            float(QLEV[0]), float(QLEV[-1])) * alive(t)         # cannot hold expired contracts
        eps = torch.randn(n_paths, DIM_M, generator=gen)
        M1  = market_step(M, eps)
        pe1 = pe + (M[:, 0] if t >= 1 else 0.0)
        dF  = futures(M1, t + 1) - futures(M, t)
        dL  = liab(M1, pe1, t + 1) - liab(M, pe, t)
        N   = N + (q * dF).sum(-1) - dL
        M, pe = M1, pe1
    return bank


# --------------------------------------------------------------------------- #
# One backward step: twin labels, explicit lambda, residual fit                #
# --------------------------------------------------------------------------- #
# STANDARDISE the differential (gradient) loss by the net's input divisors (Residual.forward):
# raw carry gradients are ~800 (dL/dL0) vs ~few for spot/wealth, so an unweighted grad-MSE is
# ~100x the value-MSE and is dominated by carry -> the value barely fits.  Matching gradients in
# the net's NORMALISED input space rebalances it (lifted E[util] +0.13 -> +1.06).  See
# project_diffml_argmax_winners_curse / the standardized-Jacobian rule.
_GRAD_SCALE_M = torch.tensor([10.0, 2.0, 0.01, 0.01, 0.01])     # = [S, B, L0, L1, L2] divisors
_GRAD_SCALE_N = 10.0                                            # = N divisor


def fit_step(nets, bank, t, n_iter=160, n_boot=24, gen=None, lr=2e-3):
    M0, N0, pe = bank["M"][t], bank["N"][t], bank["pe"][t]
    n = M0.shape[0]

    q_star = decide(nets, M0, N0, pe, t, gen=gen)              # external argmax (no grad)

    # --- bootstrap value + pathwise gradient w.r.t. (M, N), averaged over n_boot (CRN) ---
    M = M0.clone().requires_grad_(True)
    N = N0.clone().requires_grad_(True)
    eps = torch.randn(n, n_boot, DIM_M, generator=gen)
    M1  = market_step(M[:, None, :], eps)                      # (n, n_boot, 5)
    F0  = futures(M, t)
    dF  = futures(M1, t + 1) - F0[:, None, :]
    pe1 = pe + (M[:, 0] if t >= 1 else torch.zeros_like(M[:, 0]))
    dL  = liab(M1, pe1[:, None], t + 1) - liab(M, pe, t)[:, None]
    N1  = N[:, None] + (q_star[:, None, :] * dF).sum(-1) - dL  # (n, n_boot)
    Ybar = continuation(nets, M1.reshape(-1, DIM_M), N1.reshape(-1), t + 1).reshape(n, n_boot).mean(1)
    gM, gN = torch.autograd.grad(Ybar.sum(), [M, N])
    Y_boot, gM, gN = Ybar.detach(), gM.detach(), gN.detach()

    # --- EXPLICIT lambda (non-circular: only the already-fit downstream stack) ---
    # diagnostic scalar -> a single rollout on a subsample is enough.  Use the SAME argmax
    # quality as deployment (N_INNER): a noisy (low n_inner) rollout understates the policy
    # and fabricates apparent over-optimism, which is exactly the artifact we just removed.
    sub = min(n, 256)
    Y_roll = greedy_rollout(nets, M0[:sub], N0[:sub], pe[:sub], t, gen, n_inner=N_INNER)
    g_t = (Y_boot[:sub] - Y_roll).mean()
    s_t = Y_roll.std() / math.sqrt(sub)
    lam = float(torch.clamp(g_t / (g_t + s_t), 0.0, 1.0)) if g_t > 0 else 0.0
    debias = lam * float(g_t)

    # --- residual targets: value and gradients minus the analytic baseline U(N) ---
    Nb = N0.clone().requires_grad_(True)
    (dB_dN,) = torch.autograd.grad(utility(Nb).sum(), Nb)
    a_val = Y_boot - utility(N0)
    a_gM  = gM                                                 # dU/dM = 0
    a_gN  = gN - dB_dN.detach()

    net = nets[t]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(n_iter):
        Mg = M0.clone().requires_grad_(True)
        Ng = N0.clone().requires_grad_(True)
        a  = net(Mg, Ng)
        daM, daN = torch.autograd.grad(a.sum(), [Mg, Ng], create_graph=True)
        loss = ((a - a_val) ** 2).mean() \
             + (((daM - a_gM) * _GRAD_SCALE_M) ** 2).mean() \
             + (((daN - a_gN) * _GRAD_SCALE_N) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        val_loss = float(((net(M0, N0) - a_val) ** 2).mean())
    return dict(t=t, q=q_star.float().mean(0).tolist(), lam=lam,
                g=float(g_t), s=float(s_t), debias=debias, val_loss=val_loss)


# --------------------------------------------------------------------------- #
# FD gate on the differential label (autograd dY/dM vs central difference)     #
# --------------------------------------------------------------------------- #
def fd_gate(nets, bank, t, gen):
    M0, N0, pe = bank["M"][t][:256], bank["N"][t][:256], bank["pe"][t][:256]
    q_star = decide(nets, M0, N0, pe, t, gen=gen)
    eps = torch.randn(M0.shape[0], DIM_M, generator=gen)

    def Y_of_M(Min):
        M1  = market_step(Min, eps)
        pe1 = pe + (Min[:, 0] if t >= 1 else 0.0)
        dF  = futures(M1, t + 1) - futures(Min, t)
        dL  = liab(M1, pe1, t + 1) - liab(Min, pe, t)
        N1  = N0 + (q_star * dF).sum(-1) - dL
        return continuation(nets, M1, N1, t + 1)

    M = M0.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(Y_of_M(M).sum(), [M])        # (256, 5)
    hvec = torch.tensor([1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    errs = []
    with torch.no_grad():
        for k in range(DIM_M):
            e = torch.zeros_like(M0); e[:, k] = hvec[k]
            fd = (Y_of_M(M0 + e) - Y_of_M(M0 - e)) / (2 * hvec[k])
            errs.append(float((auto[:, k] - fd).abs().max()))
    return errs                                                # per-factor max |auto - FD|


# --------------------------------------------------------------------------- #
# Analytic minimum-variance futures hedge  h* = Sigma_F^{-1} Cov(dF, dL)        #
# --------------------------------------------------------------------------- #
def minvar_hedge(M_state, pe, t, n_mc, gen, ridge=0.05):
    """MC regression of dL on the three futures' one-step moves at a representative state.
    Returns the joint (multivariate) hedge h* = (Sigma_F + gamma I)^{-1} Cov(dF,dL) -- NOT
    three univariate ratios.  The three futures share the S+B factor (Sigma_F is severely
    ill-conditioned), so a desk-style RIDGE gamma = ridge*mean(diag Sigma_F) is applied: the
    unregularised solve chases the near-null carry direction into huge offsetting positions
    for a negligible variance gain.  Also returns the variance-reduction achieved and the
    ceiling (1 - basis share) the hedge cannot beat because basis is unhedgeable."""
    al  = alive(t).bool()                                      # solve only over tradeable contracts
    na  = int(al.sum())
    Me  = M_state[None, :].expand(n_mc, DIM_M)
    eps = torch.randn(n_mc, DIM_M, generator=gen)
    M1  = market_step(Me, eps)
    dF  = (futures(M1, t + 1) - futures(Me, t))[:, al]         # (n_mc, na)
    dL  = liab(M1, pe, t + 1) - liab(Me, pe, t)                # (n_mc,)
    dFc = dF - dF.mean(0)
    dLc = dL - dL.mean(0)
    Sig = dFc.t() @ dFc / n_mc                                 # Cov(dF,dF) over alive contracts
    cov = (dFc * dLc[:, None]).mean(0)                         # Cov(dF,dL)
    gamma = ridge * float(torch.diag(Sig).mean())
    ha = torch.linalg.solve(Sig + gamma * torch.eye(na), cov)
    h = torch.zeros(NF); h[al] = ha                            # scatter back; expired contracts -> 0
    var_reduc = float(1.0 - (dLc - dFc @ ha).var() / dLc.var())
    return h, var_reduc


def mean_path():
    """Deterministic (shock-free) path of the market state and past-average accumulator."""
    M = init_market(1)[0]
    pe = 0.0
    Ms, pes = [], []
    for t in range(T):
        Ms.append(M.clone()); pes.append(pe)
        M = market_step(M[None, :], torch.zeros(1, DIM_M))[0]
        pe = pe + (Ms[-1][0].item() if t >= 1 else 0.0)
    return Ms, pes


# --------------------------------------------------------------------------- #
# BSS sandwich:  L <= V* <= U,  + martingale-penalty zero-mean guard            #
# --------------------------------------------------------------------------- #
def sandwich(nets, n_paths, gen):
    # lower bound = deployed greedy policy value -- MUST use the same argmax quality the policy
    # is deployed with (N_INNER), else L understates the policy and looks artificially loose.
    L = float(greedy_rollout(nets, init_market(n_paths), init_wealth(n_paths),
                             torch.zeros(n_paths), 0, gen, n_inner=N_INNER).mean())

    # naive clairvoyant upper bound: utility is monotone in wealth, so perfect-foresight
    # maximises terminal N by picking, each step, the gridded action best aligned with dF.
    M = init_market(n_paths); pe = torch.zeros(n_paths); N = init_wealth(n_paths)
    for t in range(T):
        eps = torch.randn(n_paths, DIM_M, generator=gen)
        M1  = market_step(M, eps)
        pe1 = pe + (M[:, 0] if t >= 1 else 0.0)
        dF  = futures(M1, t + 1) - futures(M, t)
        dL  = liab(M1, pe1, t + 1) - liab(M, pe, t)
        best = torch.einsum('pd,ad->pa', dF, QG * alive(t)).max(-1).values  # best q.dF (alive only)
        N   = N + best - dL
        M, pe = M1, pe1
    U_naive = float(utility(N).mean())

    # martingale-penalty zero-mean guard (independent selection vs estimate draws).
    # This is the meaningful dual-feasibility check: if the fitted C is a good value, the
    # one-step penalty pi_t = C_{t+1}(realised) - E[C_{t+1}|s_t,q*] is mean-zero.
    M = init_market(n_paths); N = init_wealth(n_paths); pe = torch.zeros(n_paths)
    pis = []
    with torch.no_grad():
        for t in range(T):
            q    = decide(nets, M, N, pe, t, gen=gen)
            qi   = (QG[None, :, :] == q[:, None, :]).all(-1).double().argmax(-1)
            ehat = action_values(nets, M, N, pe, t, n_inner=16, gen=gen)
            ehat_q = ehat.gather(1, qi[:, None]).squeeze(1)
            eps  = torch.randn(n_paths, DIM_M, generator=gen)
            M1   = market_step(M, eps)
            pe1  = pe + (M[:, 0] if t >= 1 else 0.0)
            dF   = futures(M1, t + 1) - futures(M, t)
            dL   = liab(M1, pe1, t + 1) - liab(M, pe, t)
            N1   = N + (q * dF).sum(-1) - dL
            pis.append(continuation(nets, M1, N1, t + 1) - ehat_q)
            M, N, pe = M1, N1, pe1
    interior = torch.stack(pis, 1)[:, :-1]
    z = float(interior.mean() / (interior.std() / math.sqrt(interior.numel())))
    return L, U_naive, float(interior.mean()), z


def oos_optimism(nets, n_paths, gen):
    bank = simulate_bank(n_paths, gen)
    out = []
    for t in range(T):
        M0, N0, pe = bank["M"][t], bank["N"][t], bank["pe"][t]
        with torch.no_grad():
            C  = continuation(nets, M0, N0, t)
            Yr = greedy_rollout(nets, M0, N0, pe, t, gen, n_inner=N_INNER)  # deployment-quality rollout
        out.append(float((C - Yr).mean()))
    return out


def policy_response(nets, gen):
    """Mechanism, made visible: the chosen hedge as a SPOT-DELTA-EQUIVALENT
    (sum_i q_i * dF_i/dS = sum_i q_i e^{c tau_i}) vs the SPOT S, at fixed wealth (N=0) and mean
    basis/carry.  The liability's own spot-delta is ~flat in S (carry~0), so any TILT of the hedge
    vs S is the policy timing the AR(1) mean-reversion -- net-LONGER when S is below MU (reversion
    up is +EV), net-shorter when S is above MU.  That directional state-dependence is the edge a
    static min-var hedge (constant ratio) cannot capture, and the source of the retained upside."""
    Ms, pes = mean_path()
    log("\npolicy response  q* spot-delta-equiv vs SPOT (S-MU)  (tilt vs S = mean-reversion timing):")
    Sgrid = MU + torch.linspace(-3.0, 3.0, 7) * SIGMA          # +/- 3 sigma around the mean
    for t in (1, 4, 6):
        M0 = Ms[t][None, :].repeat(Sgrid.numel(), 1); M0[:, 0] = Sgrid
        pe = torch.full((Sgrid.numel(),), float(pes[t]))
        q  = decide(nets, M0, init_wealth(Sgrid.numel()), pe, t, gen=gen)
        S, B, L0, L1, L2 = M0.unbind(-1); taus = T_EXP - t
        dFdS = torch.exp(carry_curve(taus, L0[:, None], L1[:, None], L2[:, None]) * taus)
        sd = (q * dFdS).sum(-1)                                     # portfolio spot-delta-equivalent
        dLdS = NU / T * (float(t >= 1) + sum(math.exp(_ns_carry_scalar(u - t) * (u - t))
                                             for u in range(t + 1, T + 1)))   # liability delta (~flat)
        live = "".join("AJO"[i] if alive(t)[i] else "." for i in range(NF))
        row = "  ".join(f"{float(s):+.2f}" for s in sd)
        log(f"  t={t} [{live}]  S-MU=[{', '.join(f'{int(s - MU):+d}' for s in Sgrid)}]:"
            f"  dEq=[{row}]   (liab dL/dS={dLdS:+.2f})")


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    gen  = torch.Generator().manual_seed(1)
    nets = [Residual() for _ in range(T)]

    # ---- config / parameter summary (logged so each run is self-documenting) ----
    log("=" * 78)
    log("CONFIG  T=%d  NU=%.1f  STRIKE=%.3f (MU=%.1f)  AVERSION=%.2f  DELTA=%.1f" %
        (T, NU, STRIKE, MU, AVERSION, DELTA))
    log("  deal spread: $%.1f/oz above fair -> day-1 cushion N_INIT=%.2f (downside budget)" %
        (MARGIN, N_INIT))
    log("  spot   : KAPPA=%.2f  SIGMA=%.2f  (per-step sd; AR(1) to MU)" % (KAPPA, SIGMA))
    log("  basis  : ALPHA=%.2f  SIGMA_B=%.2f  (OU to 0; UNHEDGEABLE -- not in the deal)" % (ALPHA, SIGMA_B))
    log("  carry  : means=(%.3f,%.3f,%.3f) rho=(%.2f,%.2f,%.2f) sig=(%.4f,%.4f,%.4f) LAM=%.1f" %
        (L0_BAR, L1_BAR, L2_BAR, RHO0, RHO1, RHO2, SIG0, SIG1, SIG2, LAM))
    log("  futures: expiries=%s  action grid=%d^%d=%d over [%.1f,%.1f]" %
        (T_EXP.tolist(), QLEV.numel(), NF, NA, float(QLEV[0]), float(QLEV[-1])))
    log("  contract availability by t (1=tradeable):")
    for t in range(T):
        log("    t=%d  %s   (%d live)" % (t, alive(t).int().tolist(), int(alive(t).sum())))
    # rough per-step risk budget (sd of one-step P&L contributions at the initial state)
    M0 = init_market(1)[0]
    dLdS = NU / T * (0.0 + sum(math.exp(_ns_carry_scalar(u) * u) for u in range(1, T + 1)))
    dLdc = NU / T * sum(M0[0].item() * u * math.exp(_ns_carry_scalar(u) * u) for u in range(1, T + 1))
    log("  risk budget @t0 (1-step sd): spot~%.2f  carry(L0)~%.2f  basis(per unit q)~%.2f" %
        (dLdS * SIGMA, dLdc * SIG0, math.exp(0.0) * SIGMA_B))
    log("=" * 78)

    log("simulating exploratory bank ...")
    bank = simulate_bank(512, gen)

    print("\nbackward differential fit (twin loss, external 3-D argmax, CRN):")
    rows = [fit_step(nets, bank, t, n_iter=120, gen=gen) for t in reversed(range(T))]
    for r in sorted(rows, key=lambda r: r["t"]):
        qa, qj, qo = r["q"]
        print(f"  t={r['t']}  q*~=({qa:+.2f},{qj:+.2f},{qo:+.2f})  val_loss={r['val_loss']:8.3f}")

    print("\nFD gate  max|autograd - FD| of dY/d[ S, B, L0, L1, L2 ]:")
    for t in (0, T // 2, T - 1):
        e = fd_gate(nets, bank, t, gen)
        print(f"  t={t}:  S={e[0]:.1e}  B={e[1]:.1e}  L0={e[2]:.1e}  L1={e[3]:.1e}  L2={e[4]:.1e}")

    print("\nexplicit lambda = clip( g_t/(g_t+s_t), 0, 1 )  (non-circular, hyperparameter-free):")
    for r in sorted(rows, key=lambda r: r["t"]):
        tag = "ON  (de-bias %+.3f)" % r["debias"] if r["lam"] > 0.05 \
              else "off (optimism within MC noise)"
        print(f"  t={r['t']}:  g_t={r['g']:+.4f}  s_t={r['s']:.4f}  lambda={r['lam']:.3f}  {tag}")

    opt = oos_optimism(nets, 256, gen)
    lam_by_t = {r["t"]: r["lam"] for r in rows}
    print("\nout-of-sample bootstrap over-optimism  E[C_t - Y_rollout]  and lambda's de-bias:")
    for t in range(T):
        print(f"  t={t} : {opt[t]:+8.3f}  ->  {(1-lam_by_t[t])*opt[t]:+8.3f}   (lam {lam_by_t[t]:.2f})")

    print("\nlearned q*  vs  ridged minimum-variance futures hedge  h*=(Sigma_F+gI)^-1 Cov(dF,dL):")
    print("  (basis is unhedgeable, so var-reduction is capped well below 1; the three futures")
    print("   share S+B -> Sigma_F is near-singular, hence the ridge and the modest gross hedge)")
    Ms, pes = mean_path()
    M = init_market(256); N = init_wealth(256); pe = torch.zeros(256)
    with torch.no_grad():
        for t in range(T):
            q = decide(nets, M, N, pe, t, gen=gen).float().mean(0)
            h, vr = minvar_hedge(Ms[t], pes[t], t, 100_000, gen)
            live = "".join("AJO"[i] if alive(t)[i] else "." for i in range(NF))
            print(f"  t={t} [{live}]:  q*=({q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f})   "
                  f"h*=({h[0]:+.2f},{h[1]:+.2f},{h[2]:+.2f})   var_reduc={vr:.3f}")
            qd  = QG[(QG - q).pow(2).sum(-1).argmin()]
            eps = torch.randn(256, DIM_M, generator=gen)
            M1  = market_step(M, eps); pe1 = pe + (M[:, 0] if t >= 1 else 0.0)
            dF  = futures(M1, t + 1) - futures(M, t)
            dL  = liab(M1, pe1, t + 1) - liab(M, pe, t)
            N   = N + (qd * dF).sum(-1) - dL; M, pe = M1, pe1

    policy_response(nets, gen)

    print("\nBSS sandwich (fresh MC):")
    L, U_naive, mean_pi, z = sandwich(nets, 512, gen)
    print(f"  lower bound L (deployed greedy policy)  = {L:+.4f}")
    print(f"  upper bound U (clairvoyant, monotone)   = {U_naive:+.4f}   gap U-L = {U_naive - L:.4f}")
    print(f"    (Huber gains are linear/unbounded, so perfect foresight harvests upside -> the")
    print(f"     clairvoyant bound is loose; the gap is foresight slack, not policy suboptimality.)")
    print(f"  penalty zero-mean guard  mean(pi)={mean_pi:+.5f}  z={z:+.2f}  (|z|<~2 => dual-feasible)")

    print("\ndownside protection (paired on identical market paths):")
    P = 1536
    g2 = torch.Generator().manual_seed(7)
    eps_all = torch.randn(P, T, DIM_M, generator=g2)
    Ms, pes = mean_path()
    hstar = [minvar_hedge(Ms[t], pes[t], t, 100_000, g2)[0] for t in range(T)]

    def run_policy(policy):
        M = init_market(P); N = init_wealth(P); pe = torch.zeros(P)
        with torch.no_grad():
            for t in range(T):
                q  = policy(t, M, N, pe) * alive(t)             # cannot hold expired contracts
                M1 = market_step(M, eps_all[:, t, :]); pe1 = pe + (M[:, 0] if t >= 1 else 0.0)
                dF = futures(M1, t + 1) - futures(M, t)
                dL = liab(M1, pe1, t + 1) - liab(M, pe, t)
                N  = N + (q * dF).sum(-1) - dL; M, pe = M1, pe1
        return N

    # ONE-SIDED benchmark: the objective is asymmetric (keep upside, penalise downside), so the
    # right static comparator is NOT full min-var (it kills upside symmetrically) but the best
    # CONSTANT SCALE of the min-var shape under the utility -- typically alpha<1 (under-hedge to
    # keep upside).  min-var (alpha=1) is shown only as the downside (tail) benchmark.
    def scaled(a):
        return lambda t, M, N, pe: a * hstar[t][None, :].expand(M.shape[0], NF)
    best_a, best_u = 0.0, -1e18
    for a in [i * 0.1 for i in range(0, 16)]:
        u = float(utility(run_policy(scaled(a))).mean())
        log(f"  static-scale scan: alpha={a:.1f}  E[util]={u:+.3f}")
        if u > best_u: best_a, best_u = a, u

    pols = [
        ("unhedged (q=0)",                        lambda t, M, N, pe: torch.zeros(M.shape[0], NF)),
        (f"best static ({best_a:.1f}x h*, 1-sided)", scaled(best_a)),
        ("min-var (1.0x h*, downside bench)",     lambda t, M, N, pe: hstar[t][None, :].expand(M.shape[0], NF)),
        ("learned policy",                        lambda t, M, N, pe: decide(nets, M, N, pe, t, gen=g2)),
    ]
    print(f"  terminal wealth W_T = spread cushion ({N_INIT:+.1f} = NU*${MARGIN:.1f}/oz) + hedge P&L - deal P&L")
    print(f"  {'policy':30s}  {'mean':>6s} {'5%worst':>8s} {'95%best':>8s} {'P(lose$)':>8s} {'E[util]':>8s}")
    for name, pol in pols:
        N = run_policy(pol)
        ploss = float((N < 0.0).double().mean()) * 100.0          # ate through the WHOLE spread
        print(f"  {name:30s}  {float(N.mean()):+6.2f} "
              f"{float(torch.quantile(N, 0.05)):+8.2f} {float(torch.quantile(N, 0.95)):+8.2f} "
              f"{ploss:7.1f}% {float(utility(N).mean()):+8.2f}")
    print(f"  (W_T<0 = we lost the ENTIRE spread and more.  The spread is a downside BUDGET: with it,")
    print(f"   min-var over-hedges -- it spends upside to buy tail safety the cushion already gives.  The")
    print(f"   learned policy under-hedges, keeps the upside, and still rarely loses the spread.)")


if __name__ == "__main__":
    main()
