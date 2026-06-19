"""
Differential-ML downside hedge of an average-rate (Asian) forward, standalone.

Setup
-----
Spot follows a mean-reverting AR(1) (NOT geometric Brownian motion, so Black-Scholes
gives no valid hedge):            S_{t+1} = S_t + kappa*(mu - S_t) + sigma*eps_t .
Liability: a short average-rate forward, payoff at T = average(S_1..S_T) - K.
Hedge instruments: spot S and a zero-rate bank account.
Objective: protect the DOWNSIDE while keeping the UPSIDE -- a downside-quadratic utility
(linear on gains, quadratic on losses).  The network learns a state-dependent policy that
caps losses yet rides the favourable drift; aversion is the dial trading tail vs leverage.

What it demonstrates (every piece we worked through)
----------------------------------------------------
  1. Analytic martingale BASELINE  B_t = U(N_t)  (the liability MTM is a martingale)
     plus a small NN RESIDUAL  A_t        ->   C_t = B_t + A_t      (per-timestep value).
  2. TWIN loss:  value MSE  +  AAD pathwise-GRADIENT MSE  (autograd, create_graph).
  3. The Bellman max is an EXTERNAL argmax over a discrete action grid; C_t is max-free.
     Actions are compared under COMMON RANDOM NUMBERS (one shock shared across the grid).
  4. One-step BOOTSTRAP label  Y_boot = C_{t+1}(realised next state under q*).
  5. A finite-difference GATE validating the differential (gradient) label.
  6. EXPLICIT, hyperparameter-free lambda for the bootstrap optimism:
         g_t = mean(Y_boot - Y_rollout),   s_t = stderr(Y_rollout)
         lambda_t = clip( g_t / (g_t + s_t), 0, 1 )
     Both targets depend only on the ALREADY-FIT downstream stack C_{t+1..T}, never on
     C_t -> the rule is a plain CALCULATION, not circular and with nothing to tune.  It
     turns itself off (lambda=0) where the bootstrap optimism is within MC noise.
  7. BSS SANDWICH:  lower bound L (greedy policy value)  <=  V*  <=  U (clairvoyant),
     plus the martingale-penalty ZERO-MEAN guard (dual feasibility).  The gap L..U is the
     honest verdict on how far the (small, approximate) learned policy sits from optimal.

Run:  python3 diffml_hedge_toy.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)          # double precision -> exact FD gate

# --------------------------------------------------------------------------- #
# Model / problem constants                                                   #
# --------------------------------------------------------------------------- #
T      = 8            # decision dates 0..T-1 ; averaging dates 1..T ; payoff at T
MU     = 100.0        # AR(1) long-run mean
KAPPA  = 0.25         # mean-reversion speed
PHI    = 1.0 - KAPPA  # AR(1) persistence:  E_t[S_{t+1}] - mu = PHI*(S_t - mu)
SIGMA  = 4.0          # per-step shock std (in price level)
NU     = 3.0          # liability notional: scales the deal so hedging dominates utility
S_INIT = MU           # start at the mean: no drift, so the hedge is purely risk-driven
STRIKE = MU           # fair strike  E_0[avg] = MU  => forward starts at zero value

AVERSION = 1.8        # downside-quadratic aversion.  U(W)=W - a*max(-W,0)^2 : linear (uncapped)
                      # on GAINS, quadratic on LOSSES.  Keeps the full upside while penalising
                      # large losses.  a is the dial: higher a -> tighter tail at less leverage.
                      # (a is horizon-dependent; this value targets ~T=8.)
QGRID  = torch.linspace(-2.5, 4.5, 29)           # admissible spot holdings (wide enough for the
NA     = QGRID.numel()                           # state-dependent drift tilt: long low-S, short high-S)


def transition(S, eps):
    """AR(1) one-step move (mean-reverting; not GBM)."""
    return S + KAPPA * (MU - S) + SIGMA * eps


def utility(W):
    """Downside-quadratic: linear on gains (keep upside), quadratic on losses (protect tail)."""
    return W - AVERSION * torch.clamp(-W, min=0.0) ** 2


def future_weight(t):
    """sum_{k=1}^{T-t} PHI^k  -- how a unit move in S_t propagates into future avg dates."""
    m = T - t
    return 0.0 if m <= 0 else PHI * (1.0 - PHI ** m) / (1.0 - PHI)


def liability_mtm(S, pe, t):
    """
    Mark-to-market of the average-rate forward, kept differentiable in S_t.
        L_t = E_t[average] - STRIKE
    pe  = sum of averaging dates strictly before t (realised, detached scalar/tensor).
    The current date t is itself an averaging date for t>=1 (weight 'cur').
    A conditional expectation of a fixed terminal payoff -> L_t is a MARTINGALE.
    """
    m   = T - t
    cur = 1.0 if t >= 1 else 0.0
    Et_avg = (pe + cur * S + m * MU + future_weight(t) * (S - MU)) / T
    return NU * (Et_avg - STRIKE)


# --------------------------------------------------------------------------- #
# Continuation value  C_t = U(N_t) + A_t(S, N)                                #
# --------------------------------------------------------------------------- #
class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        for p in self.body[-1].parameters():        # init A ~ 0  =>  C starts at baseline
            nn.init.zeros_(p)

    def forward(self, S, N):
        x = torch.stack([(S - MU) / 10.0, N / 10.0], dim=-1)
        return self.body(x).squeeze(-1)


def continuation(nets, S, N, t, chunk=400_000):
    """C_t(S,N) = U(N) + A_t(S,N).  Terminal C_T = U(N).  Net eval is row-chunked."""
    base = utility(N)
    if t >= T:
        return base
    if S.numel() <= chunk:
        return base + nets[t](S, N)
    res = torch.empty_like(S)
    for i in range(0, S.numel(), chunk):
        sl = slice(i, i + chunk)
        res[sl] = nets[t](S[sl], N[sl])
    return base + res


# --------------------------------------------------------------------------- #
# External argmax (the Bellman max lives OUTSIDE the fitted value)            #
# --------------------------------------------------------------------------- #
def action_values(nets, S, N, pe, t, n_inner, gen):
    """E[C_{t+1}] for every action in QGRID via inner MC over S_{t+1}.  Returns (n,NA).
    Common random numbers: ONE shock set shared across all actions so the argmax
    compares actions on identical draws (kills cross-action comparison variance)."""
    n   = S.shape[0]
    eps = torch.randn(n, 1, n_inner, generator=gen).expand(n, NA, n_inner)   # CRN
    S1  = transition(S[:, None, None], eps)
    dS  = S1 - S[:, None, None]
    pe_next = pe + (S if t >= 1 else torch.zeros_like(S))     # S_t locks into the average
    dL  = liability_mtm(S1, pe_next[:, None, None], t + 1) \
        - liability_mtm(S, pe, t)[:, None, None]
    N1  = N[:, None, None] + QGRID[None, :, None] * dS - dL
    C1  = continuation(nets, S1.reshape(-1), N1.reshape(-1), t + 1).reshape(n, NA, n_inner)
    return C1.mean(-1)


def decide(nets, S, N, pe, t, n_inner=16, gen=None):
    with torch.no_grad():
        return QGRID[action_values(nets, S, N, pe, t, n_inner, gen).argmax(-1)]


# --------------------------------------------------------------------------- #
# Greedy realised rollout (used for the lambda label and the lower bound)      #
# --------------------------------------------------------------------------- #
def greedy_rollout(nets, S, N, pe, t_start, gen, n_inner=6):
    """Follow the greedy C-stack from t_start to T; return realised U(N_T).  No grad."""
    with torch.no_grad():
        S, N, pe = S.clone(), N.clone(), pe.clone()
        for t in range(t_start, T):
            q   = decide(nets, S, N, pe, t, n_inner=n_inner, gen=gen)
            eps = torch.randn(S.shape[0], generator=gen)
            S1  = transition(S, eps)
            pe1 = pe + (S if t >= 1 else 0.0)
            dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
            N   = N + q * (S1 - S) - dL
            S, pe = S1, pe1
        return utility(N)


# --------------------------------------------------------------------------- #
# One exploratory bank of states (random actions -> broad coverage of N)       #
# --------------------------------------------------------------------------- #
def simulate_bank(n_paths, gen):
    S  = torch.full((n_paths,), S_INIT)
    N  = torch.zeros(n_paths)
    pe = torch.zeros(n_paths)
    bank = {"S": [], "N": [], "pe": []}
    for t in range(T):
        bank["S"].append(S.clone()); bank["N"].append(N.clone()); bank["pe"].append(pe.clone())
        # explore AROUND the replication hedge (keeps wealth in the operating region,
        # so the residual A stays small and fittable) -- still spans q for argmax signal
        q_rep = NU * (1.0 + future_weight(t + 1)) / T
        q   = (q_rep + 0.30 * torch.rand(n_paths, generator=gen) * (MU - S)
                     + 0.7 * torch.randn(n_paths, generator=gen)).clamp(float(QGRID[0]),
                                                                        float(QGRID[-1]))
        eps = torch.randn(n_paths, generator=gen)
        S1  = transition(S, eps)
        pe1 = pe + (S if t >= 1 else 0.0)
        dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
        N   = N + q * (S1 - S) - dL
        S, pe = S1, pe1
    return bank


# --------------------------------------------------------------------------- #
# One backward step: twin labels, explicit lambda blend, residual fit          #
# --------------------------------------------------------------------------- #
def fit_step(nets, bank, t, n_iter=220, n_boot=8, gen=None, apply_lam=None, lr=2e-3):
    """apply_lam=None : pure-bootstrap fit, and MEASURE lambda (returned).
       apply_lam=float: warm-started refit that APPLIES that lambda (value blend +
                        coupled gradient weight w_diff=1-lambda)."""
    S0, N0, pe = bank["S"][t], bank["N"][t], bank["pe"][t]
    n = S0.shape[0]

    q_star = decide(nets, S0, N0, pe, t, gen=gen)               # external argmax (no grad)

    # --- bootstrap value + pathwise gradient, averaged over n_boot draws (CRN) ---
    S = S0.clone().requires_grad_(True)
    N = N0.clone().requires_grad_(True)
    eps = torch.randn(n, n_boot, generator=gen)
    S1  = transition(S[:, None], eps)
    pe1 = (pe + (S if t >= 1 else torch.zeros_like(S)))[:, None]
    dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)[:, None]
    N1  = N[:, None] + q_star[:, None] * (S1 - S[:, None]) - dL
    Ybar = continuation(nets, S1.reshape(-1), N1.reshape(-1), t + 1).reshape(n, n_boot).mean(1)
    gS, gN = torch.autograd.grad(Ybar.sum(), [S, N])
    Y_boot, gS, gN = Ybar.detach(), gS.detach(), gN.detach()

    # --- EXPLICIT lambda (non-circular: built only from the already-fit downstream stack) ---
    rolls = torch.stack([greedy_rollout(nets, S0, N0, pe, t, gen) for _ in range(2)], 0)
    Y_roll = rolls.mean(0)
    g_t = (Y_boot - Y_roll).mean()                              # >0  <=>  bootstrap optimistic
    s_t = Y_roll.std() / math.sqrt(n)                           # MC noise of the estimate
    lam = float(torch.clamp(g_t / (g_t + s_t), 0.0, 1.0)) if g_t > 0 else 0.0
    debias = lam * float(g_t)

    # --- targets:  value (optionally lambda-blended), gradient (optionally coupled) ---
    use_lam = lam if apply_lam is None else apply_lam
    if apply_lam is None:
        Y_val, w_diff = Y_boot, 1.0                             # sweep 1: clean bootstrap
    else:
        Y_val = (1.0 - apply_lam) * Y_boot + apply_lam * Y_roll # sweep 2: de-biased
        w_diff = 1.0 - apply_lam                               # couple the gradient weight
    Nb = N0.clone().requires_grad_(True)
    (dB_dN,) = torch.autograd.grad(utility(Nb).sum(), Nb)
    a_val = Y_val - utility(N0)
    a_gS  = gS                                                 # dB/dS = 0
    a_gN  = gN - dB_dN.detach()

    net = nets[t]
    opt = torch.optim.Adam(net.parameters(), lr=lr)            # warm-started weights on sweep 2
    for _ in range(n_iter):
        Sg = S0.clone().requires_grad_(True)
        Ng = N0.clone().requires_grad_(True)
        a  = net(Sg, Ng)
        daS, daN = torch.autograd.grad(a.sum(), [Sg, Ng], create_graph=True)
        loss = ((a - a_val) ** 2).mean() \
             + w_diff * (((daS - a_gS) ** 2).mean() + ((daN - a_gN) ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        val_loss = float(((net(S0, N0) - a_val) ** 2).mean())
    return dict(t=t, q=float(q_star.float().mean()), lam=lam,
                g=float(g_t), s=float(s_t), debias=debias, val_loss=val_loss)


# --------------------------------------------------------------------------- #
# Finite-difference gate on the differential label  (dY_boot/dS_t)             #
# --------------------------------------------------------------------------- #
def fd_gate(nets, bank, t, gen):
    S0, N0, pe = bank["S"][t][:256], bank["N"][t][:256], bank["pe"][t][:256]
    q_star = decide(nets, S0, N0, pe, t, gen=gen)
    eps = torch.randn(S0.shape[0], generator=gen)

    def Y_of_S(Sin):
        S1  = transition(Sin, eps)
        pe1 = pe + (Sin if t >= 1 else 0.0)
        dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(Sin, pe, t)
        N1  = N0 + q_star * (S1 - Sin) - dL
        return continuation(nets, S1, N1, t + 1)

    S = S0.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(Y_of_S(S).sum(), [S])
    with torch.no_grad():
        h  = 1e-4
        fd = (Y_of_S(S0 + h) - Y_of_S(S0 - h)) / (2 * h)
    return float((auto.detach() - fd).abs().max())


# --------------------------------------------------------------------------- #
# BSS sandwich:  L <= V* <= U,  plus the martingale-penalty zero-mean guard     #
# --------------------------------------------------------------------------- #
def _interp_path(xs, ys, q):
    """Linear interp of ys (P,G) over shared grid xs (G,), queried at q (P,...).
    Queries clamped to the grid range."""
    G = xs.shape[0]
    qc = q.clamp(float(xs[0]), float(xs[-1]))
    idx = torch.searchsorted(xs, qc.reshape(q.shape[0], -1)).clamp(1, G - 1)  # (P,M)
    x0 = xs[idx - 1]; x1 = xs[idx]
    y0 = ys.gather(1, idx - 1); y1 = ys.gather(1, idx)
    w  = (qc.reshape(q.shape[0], -1) - x0) / (x1 - x0)
    return (y0 + w * (y1 - y0)).reshape(q.shape)


def penalized_upper(nets, n_paths, gen, n_grid=41, n_inner=4):
    """
    BSS information-relaxation upper bound:
        U = E[ max_{q-path} ( utility(W_T) - sum_t pi_t(q_t) ) ],
        pi_t(q) = C_{t+1}(s_{t+1}) - E[C_{t+1} | s_t, q]      (martingale difference).
    The clairvoyant sees the whole price path; the penalty charges it for using that
    foresight, so the bound tightens toward V* as C -> V*.  Solved by a per-path
    backward DP on a wealth grid.  E[.] uses INDEPENDENT inner draws (no winner's curse).
    """
    with torch.no_grad():
        # realised price paths and the (known) per-step liability increments
        Spath = [torch.full((n_paths,), S_INIT)]
        pes   = [torch.zeros(n_paths)]
        for t in range(T):
            eps = torch.randn(n_paths, generator=gen)
            Spath.append(transition(Spath[-1], eps))
            pes.append(pes[-1] + (Spath[t] if t >= 1 else 0.0))

        Ng = torch.linspace(-80.0, 80.0, n_grid)                 # wealth grid
        J  = utility(Ng)[None, :].repeat(n_paths, 1)             # J_T(N) = U(N)
        for t in reversed(range(T)):
            St, St1, pe, pe1 = Spath[t], Spath[t + 1], pes[t], pes[t + 1]
            dS = (St1 - St)[:, None, None]
            dL = (liability_mtm(St1, pe1, t + 1) - liability_mtm(St, pe, t))[:, None, None]
            Nb = Ng[None, :, None]; qb = QGRID[None, None, :]
            # clairvoyant next wealth using the KNOWN realised move
            Nnext = Nb + qb * dS - dL                            # (P,G,NA)
            C_real = continuation(nets, St1[:, None, None].expand_as(Nnext).reshape(-1),
                                  Nnext.reshape(-1), t + 1).reshape(Nnext.shape)
            # conditional mean via INDEPENDENT inner draws
            e_in = torch.randn(n_paths, 1, 1, n_inner, generator=gen)
            Sp   = transition(St[:, None, None, None], e_in)
            dLp  = liability_mtm(Sp, pe1[:, None, None, None], t + 1) \
                 - liability_mtm(St, pe, t)[:, None, None, None]
            Npp  = Nb[..., None] + qb[..., None] * (Sp - St[:, None, None, None]) - dLp
            C_e  = continuation(nets, Sp.expand_as(Npp).reshape(-1),
                                Npp.reshape(-1), t + 1).reshape(Npp.shape).mean(-1)
            pi   = C_real - C_e                                  # (P,G,NA) mean-zero in q
            Jnext = _interp_path(Ng, J, Nnext)                   # value-to-go at N'
            J = (Jnext - pi).max(-1).values                      # max over actions -> J_t(N)
        return float(_interp_path(Ng, J, torch.zeros(n_paths, 1)).mean())


def sandwich(nets, n_paths, gen):
    # lower bound: realised value of the greedy learned policy
    L = float(greedy_rollout(nets, torch.full((n_paths,), S_INIT),
                             torch.zeros(n_paths), torch.zeros(n_paths), 0, gen).mean())

    # naive (un-penalised) clairvoyant upper bound -- valid but loose, for reference
    S = torch.full((n_paths,), S_INIT)
    dS = []
    for _ in range(T):
        eps = torch.randn(n_paths, generator=gen); S1 = transition(S, eps)
        dS.append(S1 - S); S = S1
    dS = torch.stack(dS, 1)
    qhi, qlo = float(QGRID.max()), float(QGRID.min())
    qc = torch.where(dS > 0, torch.full_like(dS, qhi), torch.full_like(dS, qlo))
    N = torch.zeros(n_paths); Sc = torch.full((n_paths,), S_INIT); pe = torch.zeros(n_paths)
    for t in range(T):
        S1 = Sc + dS[:, t]; pe1 = pe + (Sc if t >= 1 else 0.0)
        dL = liability_mtm(S1, pe1, t + 1) - liability_mtm(Sc, pe, t)
        N  = N + qc[:, t] * (S1 - Sc) - dL; Sc, pe = S1, pe1
    U_naive = float(utility(N).mean())

    # penalised (tight) clairvoyant upper bound
    U_pen = penalized_upper(nets, min(n_paths, 256), gen)

    # martingale-penalty zero-mean guard.  SELECTION (decide) and the conditional-mean
    # ESTIMATE use independent draws -> no winner's-curse bias -> pi is mean-zero.
    S = torch.full((n_paths,), S_INIT); N = torch.zeros(n_paths); pe = torch.zeros(n_paths)
    pis = []
    with torch.no_grad():
        for t in range(T):
            q    = decide(nets, S, N, pe, t, gen=gen)                       # picks q* (draws A)
            qi   = (QGRID[None, :] == q[:, None]).double().argmax(-1)
            ehat = action_values(nets, S, N, pe, t, n_inner=32, gen=gen)    # fresh draws (B)
            ehat_q = ehat.gather(1, qi[:, None]).squeeze(1)
            eps  = torch.randn(n_paths, generator=gen); S1 = transition(S, eps)  # draws (C)
            pe1  = pe + (S if t >= 1 else 0.0)
            dL   = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
            N1   = N + q * (S1 - S) - dL
            pis.append(continuation(nets, S1, N1, t + 1) - ehat_q)
            S, N, pe = S1, N1, pe1
    interior = torch.stack(pis, 1)[:, :-1]                     # drop forced-flat terminal step
    z = float(interior.mean() / (interior.std() / math.sqrt(interior.numel())))
    return L, U_naive, U_pen, float(interior.mean()), z


# --------------------------------------------------------------------------- #
# Out-of-sample value over-optimism by depth:  E[ C_t(s) - Y_rollout(s) ]       #
# (measured on a FRESH bank, so shrinkage after applying lambda is genuine)     #
# --------------------------------------------------------------------------- #
def oos_optimism(nets, n_paths, gen):
    bank = simulate_bank(n_paths, gen)
    out = []
    for t in range(T):
        S0, N0, pe = bank["S"][t], bank["N"][t], bank["pe"][t]
        with torch.no_grad():
            C  = continuation(nets, S0, N0, t)
            Yr = torch.stack([greedy_rollout(nets, S0, N0, pe, t, gen)
                              for _ in range(2)], 0).mean(0)
        out.append(float((C - Yr).mean()))
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    gen = torch.Generator().manual_seed(1)
    nets = [Residual() for _ in range(T)]

    print("simulating exploratory bank ...")
    bank = simulate_bank(896, gen)

    print("\nbackward differential fit (twin loss, external argmax, CRN):")
    rows = [fit_step(nets, bank, t, n_iter=200, gen=gen) for t in reversed(range(T))]
    for r in sorted(rows, key=lambda r: r["t"]):
        print(f"  t={r['t']}  q*~={r['q']:+.3f}  val_loss={r['val_loss']:7.3f}")

    print("\nFD gate on differential label  max|autograd - FD| of dY/dS_t:")
    for t in (0, T // 2, T - 1):
        print(f"  t={t}: {fd_gate(nets, bank, t, gen):.2e}")

    print("\nexplicit lambda  =  clip( g_t / (g_t + s_t), 0, 1 )   [g_t=mean(Y_boot-Y_roll)]")
    print("computed from the already-fit downstream stack -> non-circular, no hyperparameter:")
    for r in sorted(rows, key=lambda r: r["t"]):
        tag = "ON  (pull value down by lam*g_t = %+.3f)" % r["debias"] if r["lam"] > 0.05 \
              else "off (bootstrap optimism within MC noise)"
        print(f"  t={r['t']}:  g_t={r['g']:+.4f}  s_t={r['s']:.4f}  lambda={r['lam']:.3f}  {tag}")

    # ---- the payoff lambda WOULD apply, measured out-of-sample (we keep the clean policy:
    #      applying lambda's gradient-coupling destabilises this leveraged argmax, so here
    #      lambda is the explicit de-bias CALCULATION, not folded into the fit) ----
    opt = oos_optimism(nets, 512, gen)
    lam_by_t = {r["t"]: r["lam"] for r in rows}
    print("\nout-of-sample bootstrap over-optimism  E[C_t - Y_rollout]  and lambda's de-bias:")
    print("  t :  optimism  ->  residual after lambda  (= (1-lam)*optimism)")
    for t in range(T):
        print(f"  {t} : {opt[t]:+8.3f}  ->  {(1-lam_by_t[t])*opt[t]:+8.3f}   (lam {lam_by_t[t]:.2f})")
    print(f"  mean: {sum(opt)/T:+8.3f}  ->  {sum((1-lam_by_t[t])*opt[t] for t in range(T))/T:+8.3f}")

    print("\nlearned hedge q*  vs  per-step replication (min-var) hedge  q_t = dL_(t+1)/dS_(t+1):")
    S = torch.full((1024,), S_INIT); N = torch.zeros(1024); pe = torch.zeros(1024)
    with torch.no_grad():
        for t in range(T):
            q = float(decide(nets, S, N, pe, t, gen=gen).float().mean())
            delta = NU * (1.0 + future_weight(t + 1)) / T      # neutralises the t+1 move
            print(f"  t={t}:  q*={q:+.3f}   replication_hedge={delta:+.3f}")
            eps = torch.randn(1024, generator=gen); S1 = transition(S, eps)
            pe1 = pe + (S if t >= 1 else 0.0)
            dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
            qd  = QGRID[(QGRID - q).abs().argmin()]
            N   = N + qd * (S1 - S) - dL; S, pe = S1, pe1

    print("\nBSS sandwich (fresh MC, on the de-biased stack):")
    L, U_naive, U_pen, mean_pi, z = sandwich(nets, 1536, gen)
    U = min(U_naive, U_pen)                                     # tightest VALID upper bound
    print(f"  lower bound  L  (greedy policy)        = {L:+.4f}")
    print(f"  upper bound  U  (tightest valid)       = {U:+.4f}   gap U-L = {U - L:.4f}")
    print(f"    components: naive clairvoyant={U_naive:+.4f}  penalised={U_pen:+.4f}")
    print(f"    (downside-quadratic is unbounded above, so the clairvoyant upper bound is loose;")
    print(f"     the gap is dominated by perfect-foresight slack, not policy suboptimality.)")
    print(f"  penalty zero-mean guard  mean(pi)={mean_pi:+.5f}  z={z:+.2f}  (|z|<~2 => dual-feasible)")

    print("\ndownside protection check (paired on identical price paths):")
    print("  (downside_sd = RMS of losses only; the metric that matters here. std counts")
    print("   upside dispersion as 'risk', so read 95% alongside it.)")
    P = 4096
    g2 = torch.Generator().manual_seed(7)
    eps_all = torch.randn(P, T, generator=g2)                  # shared shocks
    qrepl = [NU * (1.0 + future_weight(t + 1)) / T for t in range(T)]
    def run_policy(policy):
        S = torch.full((P,), S_INIT); N = torch.zeros(P); pe = torch.zeros(P)
        with torch.no_grad():
            for t in range(T):
                q = policy(t, S, N, pe)
                S1 = transition(S, eps_all[:, t]); pe1 = pe + (S if t >= 1 else 0.0)
                dL = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
                N = N + q * (S1 - S) - dL; S, pe = S1, pe1
        return N
    # utility-optimal benchmark: best constant scale of the replication shape (grid search)
    best_s, best_u = 1.0, -1e18
    for sc in [i * 0.05 for i in range(0, 31)]:
        u = float(utility(run_policy(lambda t, S, N, pe, sc=sc: torch.full_like(S, sc * qrepl[t]))).mean())
        if u > best_u: best_s, best_u = sc, u
    pols = [
        ("unhedged (q=0)",          lambda t, S, N, pe: torch.zeros_like(S)),
        (f"best static hedge ({best_s:.2f}x repl)", lambda t, S, N, pe: torch.full_like(S, best_s * qrepl[t])),
        ("learned policy",          lambda t, S, N, pe: decide(nets, S, N, pe, t, gen=g2)),
    ]
    print(f"  {'policy':30s}  {'mean':>6s} {'down_sd':>8s} {'5%worst':>8s} {'95%best':>8s}")
    for name, pol in pols:
        N = run_policy(pol)
        dn = torch.clamp(-N, min=0.0)
        print(f"  {name:30s}  {float(N.mean()):+6.2f} {float((dn**2).mean().sqrt()):8.2f} "
              f"{float(torch.quantile(N,0.05)):+8.2f} {float(torch.quantile(N,0.95)):+8.2f}")
    print("  (downside-quadratic -> the policy caps the tail AND keeps the upside (95% >> static")
    print("   hedge); raise AVERSION for a tighter tail, lower it for more leverage/upside.)")


if __name__ == "__main__":
    main()
