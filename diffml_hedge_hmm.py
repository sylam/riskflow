"""
Differential-ML downside hedge of an average-rate (Asian) forward under a HIDDEN-REGIME spot,
hedged with expiring FUTURES on another exchange (basis + carry).  POMDP extension of
diffml_hedge_futures.py: the spot drift is driven by a 3-state hidden Markov chain
(BEAR / SIDE / BULL) that we DO NOT observe at production.

How partial observation is handled  (POMDP -> belief MDP)
--------------------------------------------------------
We never see the regime z_t.  The sufficient statistic for optimal control is the posterior
BELIEF  b_t = P(z_t | prices so far) in the 3-simplex (Astrom).  So the policy/value state is the
OBSERVABLES + b_t -- never z_t.  b_t is produced online by the HMM forward filter:

    predict:  bar = P^T b_t                         (regime can switch)
    correct:  b_{t+1}[k] ∝ bar[k] * N(innov; mu_k, sigma^2),   innov = dS - kappa(MU - S)
              (the known mean-reversion is removed; only the regime drift bias is inferred)

Two consistency rules (the privileged-info trap):
  * TRAIN on the belief, never the true regime.  We simulate z_t to make prices, but the policy,
    value and bootstrap only ever consume b_t.
  * The inner-MC / bootstrap must propagate b_t through the SAME filter, with NO regime leakage.
    Here we MARGINALISE the 3 regimes analytically (not sample them): E[.|b] = sum_z' bar[z'] E[.|z'].
    That makes b a SMOOTH (differentiable, FD-consistent) coordinate -- belief joins {price,
    wealth, belief} in the AAD twin-loss, and there is no discrete-sample gradient leak.

PRIVILEGED gap: set PRIVILEGED=True and the "belief" becomes the true one-hot regime (perfect
foresight of the *current* state).  Train+eval both; the deployable(belief) - privileged(one-hot)
E[util] gap is the honest cost of not observing the regime.

Everything else carries over from the futures toy: deal struck $MARGIN/oz above fair (a day-1
wealth cushion = downside budget); expiring futures with an availability mask; analytic
baseline U(N) + NN residual; external argmax (large N_INNER -> no winner's curse); standardised
differential loss; min-var as the downside (tail) benchmark.

KEY FINDINGS / KNOBS (see project_diffml_hmm_belief_toy memory for the full evidence)
-----------------------------------------------------------------------------------
* ANTITHETIC (default ON): sign-flip the inner-MC emission draws.  ~halves label+argmax variance ->
  doubles effective N_INNER for free.  +~4 E[util], worst-case flips positive, collapses the winner's
  curse (N_INNER trims to 12), AND buys OOD generalisation (kills the noise the argmax would overfit).
* RISK_KAPPA (downside-aware SELECTION, the best deployable): at argmax, score actions by
  mean(C) - RISK_KAPPA*downside-semidev(C) over the inner-MC.  De-risks ONLY bad-tail actions ->
  beats the uniform min-var blend (more upside at equal worst-case).  k~=0.5 is the sweet spot.
  It is the ONE lever that protects the downside (AVERSION / grid / volMult / bank-size all just feed
  upside speculation, because they act through the expectation-smoothed value fn; RISK_KAPPA acts on
  the per-action outcome DISTRIBUTION).
* The residual downside is structural POMDP lag (the filter is reactive); min-var (model-free) is the
  floor, privileged one-hot is the ceiling.  RISK_KAPPA's one weakness is DRIFT under-specification
  (its penalty trusts the assumed model) -> robustify by conservatively inflating the assumed
  REGIME_MU (dominates a uniform floor) or add a small min-var blend.

Run:  python3 diffml_hedge_hmm.py   (prints the full deployable ladder)
"""
import math
import logging
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("diffml_hmm").info

# --------------------------------------------------------------------------- #
# Model / problem constants                                                   #
# --------------------------------------------------------------------------- #
T      = 8
MU     = 100.0
KAPPA  = 0.25        # spot mean-reversion (the KNOWN part of the drift; removed before filtering)
SIGMA  = 4.0         # spot per-step shock std (also the regime EMISSION std)
NU     = 3.0
S_INIT = MU
MARGIN = 6.5         # USD/oz struck above fair -> day-1 cushion (downside budget)
N_INIT = NU * MARGIN

# --- HIDDEN REGIME: 3-state Markov chain driving an extra spot DRIFT BIAS --- #
REGIME_MU  = torch.tensor([-3.0, 0.0, 3.0])            # BEAR / SIDE / BULL per-step drift bias
P_TRANS    = torch.tensor([[0.80, 0.15, 0.05],         # persistent transition matrix
                           [0.10, 0.80, 0.10],
                           [0.05, 0.15, 0.80]])
B0_BELIEF  = torch.tensor([1.0, 1.0, 1.0]) / 3.0       # initial belief (and initial regime dist)
NR         = 3
PRIVILEGED = False   # False: deployable (filtered belief).  True: privileged (true-regime one-hot).

# --- BASIS: cross-exchange spread, OU to 0 (UNHEDGEABLE -- not in the deal) --- #
ALPHA   = 0.30
SIGMA_B = 1.5
B_INIT  = 0.0

# --- CARRY: Nelson-Siegel curve, zero-mean factors (hedgeable via vols, no premium) --- #
LAM     = 5.0
L0_BAR, L1_BAR, L2_BAR = 0.0, 0.0, 0.0
RHO0, RHO1, RHO2       = 0.90, 0.85, 0.80
SIG0, SIG1, SIG2       = 0.0015, 0.0020, 0.0025


def _ns_carry_scalar(tau):
    x = tau / LAM
    if abs(x) < 1e-8:
        return L0_BAR
    ex = math.exp(-x); g1 = (1.0 - ex) / x; g2 = g1 - ex
    return L0_BAR + L1_BAR * g1 + L2_BAR * g2


STRIKE = sum(MU * math.exp(_ns_carry_scalar(u) * u) for u in range(1, T + 1)) / T

T_EXP = torch.tensor([3.0, 6.0, 9.0])                  # expiries inside the window -> roll
NF    = T_EXP.numel()
# market state M = [S, B, L0, L1, L2, b0, b1, b2]  (belief in cols 5:8); regime z carried apart
DIM_M = 8

AVERSION = 1.5
DELTA    = 8.0
RISK_KAPPA = 0.0     # downside-aware action SELECTION (0 = off, bit-identical). At argmax time, score each
                     # action by mean(C) - RISK_KAPPA*downside-semidev(C) over the inner-MC scenarios.
                     # Unlike AVERSION (averaged into the value fn) or the uniform blend (gives up upside
                     # everywhere), this de-risks ONLY actions whose tail is bad — keeps upside dispersion.
BLEND_ALPHA = 0.5    # deployable hedge = a*belief + (1-a)*min-var: belief upside + a min-var floor.
                     # Frontier (in-dist E[util] / OOD-drift±6 E[util] / max P(lose) across worlds):
                     #   a=0.75 -> +23.7 / +14.4 / 11%   (in-dist optimum, OOD-fragile)
                     #   a=0.50 -> +23.2 / +19.5 /  4%   (robust all-weather default; beats both pures)
                     #   a=0.25 -> +21.5 / +20.7 /  1%   (most defensive, gives up upside)
                     # Both 0.5 and 0.75 dominate pure belief AND pure min-var in-distribution; 0.5 is
                     # the default because it stays robust when the world's drift exceeds the model.
                     # Lean lower at thin spreads / high model uncertainty, higher if model is trusted.

QLEV = torch.linspace(-1.5, 1.5, 5)
QG   = torch.cartesian_prod(QLEV, QLEV, QLEV)
NA   = QG.shape[0]
N_INNER = 24         # emission-noise inner draws for the argmax (regimes are marginalised, not sampled)

# gradient-loss standardisation = net input divisors  [S,B,L0,L1,L2,b0,b1,b2] then N
_GRAD_SCALE_M = torch.tensor([10.0, 2.0, 0.01, 0.01, 0.01, 1.0, 1.0, 1.0])
_GRAD_SCALE_N = 10.0

# --- Huge-Savine recipe knobs (ablated 2026-06-27; see DifferentialML.ipynb) -- #
ANTITHETIC     = True       # ON: pair each inner-MC emission draw with its sign-flip. The clean win
                            #   (belief E[util] +24.7->+27.8, worst-case flips +, P(lose) 6.8->2.7%):
                            #   ~halves label+argmax variance -> doubles effective N_INNER for free,
                            #   directly damping the winner's curse [[project_diffml_argmax_winners_curse]].
VOL_MULT       = 1.0        # OFF: inflating the bank's tail coverage does NOT protect the downside -- the
                            #   policy spends it on UPSIDE speculation (best +60->+101 but P(lose) 2.7->15%),
                            #   the same expressiveness->speculation law as grid-width/aversion. Only pays off
                            #   WITH the min-var floor (lifts the BLEND's E[util]); set 1.5 for blend deploys.
DIFF_WEIGHTING = "fixed"    # "fixed": hand-tuned _GRAD_SCALE (default); "adaptive": H-S normalised combined
                            #   cost = per-column λ_j=1/RMS(∂Y/∂x_j) + value-std + α/β value-vs-diff balance.
                            #   Adaptive is a mild downside stabiliser + self-calibrates (no _GRAD_SCALE
                            #   re-tuning across regime/spread sweeps) -- prefer it under sweeps / VOL_MULT>1.
DIFF_LAMBDA    = 1.0        # H-S λ in α=1/(1+λ·n); paper says minor effect, left at 1


def alive(t):
    return (T_EXP > t).double()


# --------------------------------------------------------------------------- #
# Regime sampling, filter, market dynamics                                    #
# --------------------------------------------------------------------------- #
def sample_cat(probs, u):
    """Sample a categorical index from probs (...,K) using uniform u (...,)."""
    return (u[..., None] > probs.cumsum(-1)).sum(-1).clamp(max=probs.shape[-1] - 1)


def belief_filter(b, S, S1):
    """One forward-filter step: b (...,3), S/S1 (...). Returns posterior (...,3).  Differentiable."""
    bar = b @ P_TRANS                                  # predict
    innov = (S1 - S) - KAPPA * (MU - S)                # remove known reversion
    ll = torch.exp(-0.5 * ((innov[..., None] - REGIME_MU) / SIGMA) ** 2)
    post = bar * ll
    return post / post.sum(-1, keepdim=True)


def _carry_step(L0, L1, L2, e0, e1, e2):
    return (L0_BAR + RHO0 * (L0 - L0_BAR) + SIG0 * e0,
            L1_BAR + RHO1 * (L1 - L1_BAR) + SIG1 * e1,
            L2_BAR + RHO2 * (L2 - L2_BAR) + SIG2 * e2)


def _new_belief(b, S, S1, z1):
    """Posterior belief (filter) OR privileged one-hot of the realised regime z1."""
    if PRIVILEGED:
        return torch.nn.functional.one_hot(z1, NR).to(b.dtype)
    return belief_filter(b, S, S1)


def market_step_gen(M, z, gauss, uni, true_model=None):
    """GENERATIVE step on a realised path: advance the TRUE regime z, emit, evolve B/carry, and
    update the belief column (filter, or one-hot if PRIVILEGED).  gauss (...,5), uni (...,).

    `true_model` (None = in-distribution) decouples the WORLD's regime dynamics from the model the
    filter assumes — an OOD/mis-specification test: the realised regime jump, drift and emission use
    `true_model`'s {P_TRANS, REGIME_MU, SIGMA}, while `_new_belief` (the deployed filter) keeps using
    the ASSUMED globals.  Default None reproduces the in-distribution path bit-for-bit."""
    tm = true_model or {}
    P, Rmu, Sig = tm.get("P_TRANS", P_TRANS), tm.get("REGIME_MU", REGIME_MU), tm.get("SIGMA", SIGMA)
    S, B = M[..., 0], M[..., 1]
    L0, L1, L2 = M[..., 2], M[..., 3], M[..., 4]
    b = M[..., 5:8]
    eS, eB, e0, e1, e2 = gauss.unbind(-1)
    z1 = sample_cat(P[z], uni)
    S1 = S + KAPPA * (MU - S) + Rmu[z1] + Sig * eS
    B1 = (1.0 - ALPHA) * B + SIGMA_B * eB
    L0p, L1p, L2p = _carry_step(L0, L1, L2, e0, e1, e2)
    b1 = _new_belief(b, S, S1, z1)            # filter uses the ASSUMED globals (mis-specified under OOD)
    M1 = torch.stack([S1, B1, L0p, L1p, L2p, b1[..., 0], b1[..., 1], b1[..., 2]], -1)
    return M1, z1


def next_states_bel(M, gauss):
    """BELIEF-MDP one step with the 3 next-regimes MARGINALISED analytically (differentiable in M
    incl. belief).  gauss (...,5) shared emission/basis/carry noise.  Returns:
      M1 : (..., NR, DIM_M)   the next market state for each next-regime z'
      w  : (..., NR)          predictive weight bar[z'] = (b @ P)[z']  (the belief MDP transition)."""
    S, B = M[..., 0], M[..., 1]
    L0, L1, L2 = M[..., 2], M[..., 3], M[..., 4]
    b = M[..., 5:8]
    eS, eB, e0, e1, e2 = gauss.unbind(-1)
    w = b @ P_TRANS                                                  # (..., NR)
    S1 = (S + KAPPA * (MU - S))[..., None] + REGIME_MU + SIGMA * eS[..., None]   # (..., NR)
    B1 = (1.0 - ALPHA) * B + SIGMA_B * eB                           # (...,)
    L0p, L1p, L2p = _carry_step(L0, L1, L2, e0, e1, e2)             # (...,)
    # belief posterior for each realised z' :  filter(b, S, S1[z'])  -> (..., NR_z', 3)
    bar = b @ P_TRANS
    innov = S1 - (S + KAPPA * (MU - S))[..., None]                  # (..., NR) = REGIME_MU + sigma eS
    ll = torch.exp(-0.5 * ((innov[..., None] - REGIME_MU) / SIGMA) ** 2)   # (..., NR_z', 3)
    post = bar[..., None, :] * ll
    b1 = post / post.sum(-1, keepdim=True)                          # (..., NR_z', 3)
    if PRIVILEGED:
        b1 = torch.eye(NR, dtype=M.dtype).expand(*b1.shape[:-2], NR, NR)   # one-hot per z'
    nlead = S1.shape                                               # (..., NR)
    cols = [S1,
            B1[..., None].expand(nlead),
            L0p[..., None].expand(nlead),
            L1p[..., None].expand(nlead),
            L2p[..., None].expand(nlead)]
    M1 = torch.cat([torch.stack(cols, -1), b1], -1)                # (..., NR, DIM_M)
    return M1, w


def utility(W):
    loss = torch.clamp(-W, min=0.0)
    quad = AVERSION * loss ** 2
    lin = AVERSION * DELTA ** 2 + 2.0 * AVERSION * DELTA * (loss - DELTA)
    return W - torch.where(loss <= DELTA, quad, lin)


def carry_curve(taus, L0, L1, L2):
    x = taus / LAM
    ex = torch.exp(-x)
    xs = torch.where(x.abs() > 1e-8, x, torch.ones_like(x))
    g1 = torch.where(x.abs() > 1e-8, (1.0 - ex) / xs, torch.ones_like(x))
    g2 = g1 - ex
    return L0 + L1 * g1 + L2 * g2


def futures(M, t):
    S, B = M[..., 0], M[..., 1]
    L0, L1, L2 = M[..., 2], M[..., 3], M[..., 4]
    taus = (T_EXP - t)
    c = carry_curve(taus, L0.unsqueeze(-1), L1.unsqueeze(-1), L2.unsqueeze(-1))
    return (S + B).unsqueeze(-1) * torch.exp(c * taus)


def liab(M, pe, t):
    """Asian-forward MTM on the index; remaining fixings projected at the carry forward.  Depends
    on S and carry, NOT on basis or regime (the deal references the index)."""
    S = M[..., 0]
    L0, L1, L2 = M[..., 2], M[..., 3], M[..., 4]
    Et = pe + (S if t >= 1 else torch.zeros_like(S))
    for u in range(t + 1, T + 1):
        tau = u - t
        x = tau / LAM; ex = math.exp(-x); g1 = (1.0 - ex) / x; g2 = g1 - ex
        c = L0 + L1 * g1 + L2 * g2
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
        for p in self.body[-1].parameters():
            nn.init.zeros_(p)

    def forward(self, M, N):
        S, B = M[..., 0], M[..., 1]
        L0, L1, L2 = M[..., 2], M[..., 3], M[..., 4]
        b0, b1, b2 = M[..., 5], M[..., 6], M[..., 7]
        x = torch.stack([(S - MU) / 10.0, B / 2.0,
                         (L0 - L0_BAR) / 0.01, (L1 - L1_BAR) / 0.01, (L2 - L2_BAR) / 0.01,
                         b0 - 1 / 3, b1 - 1 / 3, b2 - 1 / 3,
                         (N - N_INIT) / 10.0], dim=-1)
        return self.body(x).squeeze(-1)


def continuation(nets, M, N, t, chunk=200_000):
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
# External argmax over the action grid (regimes marginalised in the inner MC)  #
# --------------------------------------------------------------------------- #
def _inner_gauss(n, n_inner, gen):
    """Inner-MC emission draws (n,n_inner,5); ANTITHETIC pairs the 2nd half = -(1st half) for an
    unbiased, lower-variance estimate of the value label AND its pathwise gradient.  n_inner even."""
    if ANTITHETIC:
        h = torch.randn(n, n_inner // 2, 5, generator=gen)
        return torch.cat([h, -h], 1)
    return torch.randn(n, n_inner, 5, generator=gen)


def action_values(nets, M, N, pe, t, n_inner, gen):
    """E[C_{t+1} | belief, action] for every action.  Inner MC over EMISSION noise; the 3 next-
    regimes are marginalised analytically with weights bar = b@P (no regime sampling)."""
    n = M.shape[0]
    gauss = _inner_gauss(n, n_inner, gen)
    Mexp = M[:, None, :].expand(n, n_inner, DIM_M)
    M1, w = next_states_bel(Mexp, gauss)                       # (n,ni,NR,DIM_M), (n,ni,NR)
    F0 = futures(M, t)                                         # (n, NF)
    F1 = futures(M1, t + 1)                                    # (n,ni,NR,NF)
    dF = F1 - F0[:, None, None, :]                             # (n,ni,NR,NF)
    pe1 = pe + (M[:, 0] if t >= 1 else torch.zeros_like(M[:, 0]))
    dL = liab(M1, pe1[:, None, None], t + 1) - liab(M, pe, t)[:, None, None]   # (n,ni,NR)
    Qa = QG * alive(t)                                         # (NA,NF)
    hedge = torch.einsum('af,nirf->niar', Qa, dF)             # (n,ni,NA,NR)
    N1 = N[:, None, None, None] + hedge - dL[:, :, None, :]   # (n,ni,NA,NR)
    M1e = M1[:, :, None, :, :].expand(n, n_inner, NA, NR, DIM_M).reshape(-1, DIM_M)
    C = continuation(nets, M1e, N1.reshape(-1), t + 1).reshape(n, n_inner, NA, NR)
    EC = (C * w[:, :, None, :]).sum(-1)                        # marginalise z' -> (n,ni,NA)
    mean = EC.mean(1)                                          # E[C|a] over emission -> (n,NA)
    if RISK_KAPPA > 0.0:                                       # downside-aware selection: penalise ONLY the
        dev = (C - mean[:, None, :, None]).clamp(max=0.0)      #   downside dispersion of the continuation per
        semidev = ((dev ** 2 * w[:, :, None, :]).sum(-1).mean(1)).sqrt()   # action (keeps upside) -> (n,NA)
        return mean - RISK_KAPPA * semidev
    return mean


def decide(nets, M, N, pe, t, n_inner=N_INNER, gen=None):
    with torch.no_grad():
        idx = action_values(nets, M, N, pe, t, n_inner, gen).argmax(-1)
        return QG[idx] * alive(t)


# --------------------------------------------------------------------------- #
# Realised (generative) rollout and exploratory bank                          #
# --------------------------------------------------------------------------- #
def init_market(n):
    M = torch.empty(n, DIM_M)
    M[:, 0] = S_INIT; M[:, 1] = B_INIT
    M[:, 2] = L0_BAR; M[:, 3] = L1_BAR; M[:, 4] = L2_BAR
    M[:, 5:8] = B0_BELIEF
    return M


def init_wealth(n):
    return torch.full((n,), N_INIT)


def init_regime(n, gen):
    return sample_cat(B0_BELIEF.expand(n, NR), torch.rand(n, generator=gen))


def gen_shocks(shape, gen):
    return torch.randn(*shape, 5, generator=gen), torch.rand(*shape, generator=gen)


def greedy_rollout(nets, M, z, N, pe, t_start, gen, n_inner=N_INNER):
    with torch.no_grad():
        M, z, N, pe = M.clone(), z.clone(), N.clone(), pe.clone()
        for t in range(t_start, T):
            q = decide(nets, M, N, pe, t, n_inner=n_inner, gen=gen)
            g, u = gen_shocks((M.shape[0],), gen)
            M1, z1 = market_step_gen(M, z, g, u)
            pe1 = pe + (M[:, 0] if t >= 1 else 0.0)
            dF = futures(M1, t + 1) - futures(M, t)
            dL = liab(M1, pe1, t + 1) - liab(M, pe, t)
            N = N + (q * dF).sum(-1) - dL
            M, z, pe = M1, z1, pe1
        return utility(N)


def simulate_bank(n_paths, gen):
    M = init_market(n_paths); z = init_regime(n_paths, gen)
    N = init_wealth(n_paths); pe = torch.zeros(n_paths)
    bank = {"M": [], "N": [], "pe": []}
    for t in range(T):
        bank["M"].append(M.clone()); bank["N"].append(N.clone()); bank["pe"].append(pe.clone())
        cur = 1.0 if t >= 1 else 0.0
        delta_L = NU / T * (cur + sum(math.exp(_ns_carry_scalar(u - t) * (u - t))
                                      for u in range(t + 1, T + 1)))
        a_i = torch.tensor([math.exp(_ns_carry_scalar(float(E - t)) * float(E - t)) for E in T_EXP])
        q = ((delta_L / (NF * a_i))[None, :] + 0.35 * torch.randn(n_paths, NF, generator=gen)).clamp(
            float(QLEV[0]), float(QLEV[-1])) * alive(t)
        g, u = gen_shocks((n_paths,), gen)
        M1, z1 = market_step_gen(M, z, g * VOL_MULT, u)   # inflate ONLY the bank states (tail coverage);
        pe1 = pe + (M[:, 0] if t >= 1 else 0.0)            # the value labels (fit_step inner-MC) stay natural
        dF = futures(M1, t + 1) - futures(M, t)
        dL = liab(M1, pe1, t + 1) - liab(M, pe, t)
        N = N + (q * dF).sum(-1) - dL
        M, z, pe = M1, z1, pe1
    return bank


# --------------------------------------------------------------------------- #
# Backward differential fit (twin loss, marginalised belief MDP)              #
# --------------------------------------------------------------------------- #
def fit_step(nets, bank, t, n_iter=160, n_boot=12, gen=None, lr=2e-3, n_inner_decide=N_INNER):
    M0, N0, pe = bank["M"][t], bank["N"][t], bank["pe"][t]
    n = M0.shape[0]
    q_star = decide(nets, M0, N0, pe, t, n_inner=n_inner_decide, gen=gen)

    M = M0.clone().requires_grad_(True)
    N = N0.clone().requires_grad_(True)
    gauss = _inner_gauss(n, n_boot, gen)
    M1, w = next_states_bel(M[:, None, :].expand(n, n_boot, DIM_M), gauss)     # (n,nb,NR,DIM_M)
    dF = futures(M1, t + 1) - futures(M, t)[:, None, None, :]
    pe1 = pe + (M[:, 0] if t >= 1 else torch.zeros_like(M[:, 0]))
    dL = liab(M1, pe1[:, None, None], t + 1) - liab(M, pe, t)[:, None, None]
    N1 = N[:, None, None] + (q_star[:, None, None, :] * dF).sum(-1) - dL        # (n,nb,NR)
    M1f = M1.reshape(-1, DIM_M)
    C = continuation(nets, M1f, N1.reshape(-1), t + 1).reshape(n, n_boot, NR)
    Ybar = (C * w).sum(-1).mean(1)                                             # marginalise z', mean emission
    gM, gN = torch.autograd.grad(Ybar.sum(), [M, N])
    Y_boot, gM, gN = Ybar.detach(), gM.detach(), gN.detach()

    Nb = N0.clone().requires_grad_(True)
    (dB_dN,) = torch.autograd.grad(utility(Nb).sum(), Nb)
    a_val = Y_boot - utility(N0)
    a_gM = gM
    a_gN = gN - dB_dN.detach()

    if DIFF_WEIGHTING == "adaptive":           # Huge-Savine normalised combined cost
        wM = 1.0 / (a_gM.pow(2).mean(0).sqrt() + 1e-8)     # per-column λ_j = 1/RMS(∂Y/∂M_j)
        wN = 1.0 / (a_gN.pow(2).mean().sqrt() + 1e-8)
        wV = 1.0 / (a_val.std() + 1e-8)                    # standardise the value label too
        alpha = 1.0 / (1.0 + DIFF_LAMBDA * (DIM_M + 1))    # so one value error ≈ one differential error
        beta = 1.0 - alpha
    else:                                      # hand-tuned fixed scales (default; bit-identical to before)
        wM, wN, wV, alpha, beta = _GRAD_SCALE_M, _GRAD_SCALE_N, 1.0, 1.0, 1.0

    net = nets[t]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(n_iter):
        Mg = M0.clone().requires_grad_(True)
        Ng = N0.clone().requires_grad_(True)
        a = net(Mg, Ng)
        daM, daN = torch.autograd.grad(a.sum(), [Mg, Ng], create_graph=True)
        v_loss = (((a - a_val) * wV) ** 2).mean()
        g_loss = (((daM - a_gM) * wM) ** 2).mean() + (((daN - a_gN) * wN) ** 2).mean()
        loss = alpha * v_loss + beta * g_loss
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        val_loss = float(((net(M0, N0) - a_val) ** 2).mean())
    return dict(t=t, q=q_star.float().mean(0).tolist(), val_loss=val_loss)


# --------------------------------------------------------------------------- #
# FD gate on the differential label, incl. the BELIEF columns                  #
# --------------------------------------------------------------------------- #
def fd_gate(nets, bank, t, gen):
    M0, N0, pe = bank["M"][t][:256], bank["N"][t][:256], bank["pe"][t][:256]
    gauss = _inner_gauss(M0.shape[0], 4, gen)

    def Y_of_M(Min):
        M1, w = next_states_bel(Min[:, None, :].expand(Min.shape[0], 4, DIM_M), gauss)
        dF = futures(M1, t + 1) - futures(Min, t)[:, None, None, :]
        pe1 = pe + (Min[:, 0] if t >= 1 else 0.0)
        dL = liab(M1, pe1[:, None, None], t + 1) - liab(Min, pe, t)[:, None, None]
        N1 = N0[:, None, None] + (decide_q[:, None, None, :] * dF).sum(-1) - dL
        C = continuation(nets, M1.reshape(-1, DIM_M), N1.reshape(-1), t + 1).reshape(Min.shape[0], 4, NR)
        return (C * w).sum(-1).mean(1)

    decide_q = decide(nets, M0, N0, pe, t, gen=gen)
    M = M0.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(Y_of_M(M).sum(), [M])
    hvec = torch.tensor([1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5])
    errs = []
    with torch.no_grad():
        for k in range(DIM_M):
            e = torch.zeros_like(M0); e[:, k] = hvec[k]
            fd = (Y_of_M(M0 + e) - Y_of_M(M0 - e)) / (2 * hvec[k])
            errs.append(float((auto[:, k] - fd).abs().max()))
    return errs


# --------------------------------------------------------------------------- #
# Min-variance futures hedge (downside/tail benchmark) over the marginal step  #
# --------------------------------------------------------------------------- #
def minvar_hedge(M_state, pe, t, n_mc, gen, ridge=0.05):
    al = alive(t).bool(); na = int(al.sum())
    Me = M_state[None, :].expand(n_mc, DIM_M)
    gauss = torch.randn(n_mc, 5, generator=gen)
    g_unif = torch.rand(n_mc, generator=gen)
    z = sample_cat(Me[:, 5:8], g_unif)                         # sample regime from belief
    z1 = sample_cat(P_TRANS[z], torch.rand(n_mc, generator=gen))
    eS, eB, e0, e1, e2 = gauss.unbind(-1)
    S1 = Me[:, 0] + KAPPA * (MU - Me[:, 0]) + REGIME_MU[z1] + SIGMA * eS
    B1 = (1 - ALPHA) * Me[:, 1] + SIGMA_B * eB
    L0p, L1p, L2p = _carry_step(Me[:, 2], Me[:, 3], Me[:, 4], e0, e1, e2)
    b1 = belief_filter(Me[:, 5:8], Me[:, 0], S1)
    M1 = torch.stack([S1, B1, L0p, L1p, L2p, b1[:, 0], b1[:, 1], b1[:, 2]], -1)
    dF = (futures(M1, t + 1) - futures(Me, t))[:, al]
    dL = liab(M1, pe, t + 1) - liab(Me, pe, t)
    dFc = dF - dF.mean(0); dLc = dL - dL.mean(0)
    Sig = dFc.t() @ dFc / n_mc
    cov = (dFc * dLc[:, None]).mean(0)
    gamma = ridge * float(torch.diag(Sig).mean())
    ha = torch.linalg.solve(Sig + gamma * torch.eye(na), cov)
    h = torch.zeros(NF); h[al] = ha
    var_reduc = float(1.0 - (dLc - dFc @ ha).var() / dLc.var())
    return h, var_reduc


# --------------------------------------------------------------------------- #
# Belief-filter quality diagnostic (drives the privileged-vs-belief gap)        #
# --------------------------------------------------------------------------- #
def belief_quality(gen, n_paths=4096):
    """Simulate true regimes + run the forward filter; report the separability ratio
    (Δμ/σ, the SNR that governs how learnable the regime is), the MAP regime accuracy, and the
    mean posterior entropy. Sharper separability -> sharper belief -> smaller privileged gap."""
    sep = float((REGIME_MU.max() - REGIME_MU.min()) / SIGMA)
    M = init_market(n_paths)
    z = init_regime(n_paths, gen)
    acc, ent = [], []
    for t in range(T):
        g, u = gen_shocks((n_paths,), gen)
        M1, z1 = market_step_gen(M, z, g, u)          # PRIVILEGED=False here -> b1 is the filter post
        b1 = M1[:, 5:8]
        acc.append(float((b1.argmax(-1) == z1).double().mean()))
        ent.append(float((-(b1 * b1.clamp_min(1e-9).log()).sum(-1)).mean()))
        M, z = M1, z1
    map_acc = sum(acc) / len(acc)
    log("  belief filter:  separability Δμ/σ=%.2f   MAP acc=%.2f (chance=%.2f)   mean entropy=%.2f/%.2f"
        % (sep, map_acc, 1.0 / NR, sum(ent) / len(ent), math.log(NR)))
    return sep, map_acc


# --------------------------------------------------------------------------- #
# Policy response vs belief (the new mechanism: hedge by regime belief)        #
# --------------------------------------------------------------------------- #
def policy_response(nets, gen):
    Ms = init_market(1)[0]
    log("\npolicy response  q* spot-delta-equiv vs BELIEF (bear..bull), fixed S=MU, N=cushion:")
    beliefs = [("bear ", [0.8, 0.15, 0.05]), ("side ", [0.15, 0.7, 0.15]),
               ("bull ", [0.05, 0.15, 0.8]), ("unsure", [1 / 3, 1 / 3, 1 / 3])]
    for t in (1, 4, 6):
        live = "".join("AJO"[i] if alive(t)[i] else "." for i in range(NF))
        row = []
        for name, bvec in beliefs:
            M0 = Ms.clone()[None, :]; M0[:, 5:8] = torch.tensor(bvec)
            q = decide(nets, M0, init_wealth(1), torch.zeros(1), t, gen=gen)
            taus = T_EXP - t
            dFdS = torch.exp(carry_curve(taus, M0[:, 2:3], M0[:, 3:4], M0[:, 4:5]) * taus)
            row.append(f"{name}={float((q * dFdS).sum(-1)):+.2f}")
        log(f"  t={t} [{live}]:  " + "  ".join(row))


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def train_and_eval(tag, gen_seed, eval_paths=4096, n_inner_eval=N_INNER, minvar_mc=100_000,
                   fit_iter=160, fit_inner=N_INNER, bank_paths=640, diag=True, ood_true=None,
                   reuse=None):
    gen = torch.Generator().manual_seed(gen_seed)
    bq = None
    if reuse is not None:                                  # eval-only: reuse a trained policy + hedges
        nets, hstar_reuse = reuse
    else:
        hstar_reuse = None
        nets = [Residual() for _ in range(T)]
        bank = simulate_bank(bank_paths, gen)
        rows = [fit_step(nets, bank, t, n_iter=fit_iter, gen=gen, n_inner_decide=fit_inner)
                for t in reversed(range(T))]
        log(f"\n[{tag}]  backward fit (PRIVILEGED={PRIVILEGED}):")
        for r in sorted(rows, key=lambda r: r["t"]):
            qa, qj, qo = r["q"]
            log(f"  t={r['t']}  q*~=({qa:+.2f},{qj:+.2f},{qo:+.2f})  val_loss={r['val_loss']:8.3f}")
        if not PRIVILEGED:
            bq = belief_quality(gen)
            if diag:
                log(f"[{tag}]  FD gate max|auto-FD| dY/d[S,B,L0,L1,L2,b0,b1,b2]:")
                for t in (0, T // 2, T - 1):
                    e = fd_gate(nets, bank, t, gen)
                    log("  t=%d: " % t + " ".join(f"{v:.1e}" for v in e))
                policy_response(nets, gen)
    # paired evaluation on FRESH realised paths (true regime drives prices; ood_true = world model)
    g2 = torch.Generator().manual_seed(20260626)
    M0 = init_market(eval_paths); z0 = init_regime(eval_paths, g2)
    gshocks = [gen_shocks((eval_paths,), g2) for _ in range(T)]
    Ms_mean = init_market(1)[0]
    hstar = hstar_reuse if hstar_reuse is not None else \
        [minvar_hedge(Ms_mean, 0.0, t, minvar_mc, g2)[0] for t in range(T)]
    g_dec = torch.Generator().manual_seed(20260626)   # eval-argmax RNG, DECOUPLED from minvar_mc/paths

    def run(policy):
        g_dec.manual_seed(20260626)                   # reset -> eval reproducible & policy-order-independent
        M, z, N, pe = M0.clone(), z0.clone(), init_wealth(eval_paths), torch.zeros(eval_paths)
        with torch.no_grad():
            for t in range(T):
                q = policy(t, M, N, pe) * alive(t)
                g, u = gshocks[t]
                M1, z1 = market_step_gen(M, z, g, u, true_model=ood_true)
                pe1 = pe + (M[:, 0] if t >= 1 else 0.0)
                dF = futures(M1, t + 1) - futures(M, t)
                dL = liab(M1, pe1, t + 1) - liab(M, pe, t)
                N = N + (q * dF).sum(-1) - dL; M, z, pe = M1, z1, pe1
        return N

    pols = [("unhedged", lambda t, M, N, pe: torch.zeros(M.shape[0], NF)),
            ("min-var (downside bench)", lambda t, M, N, pe: hstar[t][None, :].expand(M.shape[0], NF)),
            (f"learned [{tag}]", lambda t, M, N, pe: decide(nets, M, N, pe, t, n_inner=n_inner_eval, gen=g_dec))]
    if not PRIVILEGED:                                     # deployable blend: belief overlay + min-var floor
        pols.append((f"blend {BLEND_ALPHA:.2f} (bel+mv)",
                     lambda t, M, N, pe: BLEND_ALPHA * decide(nets, M, N, pe, t, n_inner=n_inner_eval, gen=g_dec)
                     + (1.0 - BLEND_ALPHA) * hstar[t][None, :].expand(M.shape[0], NF)))
    out = {}
    log(f"\n[{tag}]  W_T = cushion ({N_INIT:+.1f}) + hedge - deal P&L   (true regime drives prices):")
    log(f"  {'policy':26s} {'mean':>7s} {'5%wst':>7s} {'95%bst':>7s} {'P(lose$)':>8s} {'E[util]':>8s}")
    for name, pol in pols:
        N = run(pol)
        eu = float(utility(N).mean())
        out[name] = dict(mean=float(N.mean()), wst=float(torch.quantile(N, 0.05)),
                         best=float(torch.quantile(N, 0.95)),
                         plose=float((N < 0).double().mean()) * 100, eu=eu)
        log(f"  {name:26s} {out[name]['mean']:+7.2f} {out[name]['wst']:+7.2f} "
            f"{out[name]['best']:+7.2f} {out[name]['plose']:7.1f}% {eu:+8.2f}")
    if bq is not None:
        out["_belief"] = dict(separability=bq[0], map_acc=bq[1])
    out["_nets"], out["_hstar"] = nets, hstar          # for eval-only reuse (e.g. OOD sweeps)
    return out


def main():
    global PRIVILEGED, RISK_KAPPA
    log("=" * 78)
    log("HMM regime hedge  T=%d NU=%.1f MARGIN=$%.1f/oz cushion=%.1f  AVERSION=%.1f DELTA=%.1f" %
        (T, NU, MARGIN, N_INIT, AVERSION, DELTA))
    log("  regimes BEAR/SIDE/BULL drift=%s  sigma=%.1f  P_diag=%.2f  N_INNER=%d (regimes marginalised)" %
        (REGIME_MU.tolist(), SIGMA, float(P_TRANS[0, 0]), N_INNER))
    log("  ANTITHETIC=%s  risk-aware row k=0.5  blend α=%.2f" % (ANTITHETIC, BLEND_ALPHA))
    log("=" * 78)

    PRIVILEGED = False; RISK_KAPPA = 0.0
    dep = train_and_eval("belief", gen_seed=1)
    RISK_KAPPA = 0.5                                  # downside-aware selection (recommended deployable)
    risk = train_and_eval("risk-aware k0.5", gen_seed=1)
    RISK_KAPPA = 0.0
    PRIVILEGED = True
    priv = train_and_eval("privileged", gen_seed=1)
    PRIVILEGED = False

    bel_k = next(k for k in dep if k.startswith('learned'))
    bld_k = next(k for k in dep if k.startswith('blend'))
    rsk_k = next(k for k in risk if k.startswith('learned'))
    prv_k = next(k for k in priv if k.startswith('learned'))

    def _row(d, key, label):
        r = d[key]
        return "  %-30s eu=%+7.2f   5%%wst=%+7.2f   P(lose)=%5.1f%%" % (label, r["eu"], r["wst"], r["plose"])

    log("\n" + "=" * 78)
    log("DEPLOYABLE LADDER  (min-var = model-free floor; privileged one-hot = upper bound):")
    log(_row(dep, "unhedged", "unhedged"))
    log(_row(dep, "min-var (downside bench)", "min-var (downside floor)"))
    log(_row(dep, bel_k, "belief (risk-neutral)"))
    log(_row(dep, bld_k, "belief + min-var blend"))
    log(_row(risk, rsk_k, "risk-aware select k=0.5  *"))
    log(_row(priv, prv_k, "privileged one-hot (ceiling)"))
    log("  * recommended deployable; for drift-OOD robustness inflate the assumed REGIME_MU")
    log("    (dominates a uniform floor) or add a small min-var blend.")
    log("=" * 78)


if __name__ == "__main__":
    main()
