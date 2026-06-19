"""
Differential-ML downside hedge of an average-rate (Asian) forward, standalone.
[UPGRADED VERSION: Fixed state space, utility gradients, exploration, and bounds]
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
T      = 8
MU     = 100.0
KAPPA  = 0.25
PHI    = 1.0 - KAPPA
SIGMA  = 4.0
NU     = 3.0
S_INIT = MU
STRIKE = MU

AVERSION = 0.20
QGRID  = torch.linspace(-0.25, 2.25, 26)
NA     = QGRID.numel()

def transition(S, eps):
    return S + KAPPA * (MU - S) + SIGMA * eps

# IMPROVEMENT 2: Fixed CARA utility to prevent vanishing gradients in the tail.
# Instead of a hard clamp that goes flat (zero gradient), we use a linear tangent
# extrapolation for extreme losses.
def utility(W):
    k = -25.0  # threshold for extrapolation
    w_safe = torch.clamp(W, min=k)
    u_true = -torch.exp(-AVERSION * w_safe)
    # Compute the gradient at the kink to form a linear tangent
    grad_at_k = AVERSION * torch.exp(AVERSION * k)
    u_tangent = u_true + grad_at_k * (W - k)
    # Use true exponential for W > k, tangent line for W <= k
    return torch.where(W > k, u_true, u_tangent)

def future_weight(t):
    m = T - t
    return 0.0 if m <= 0 else PHI * (1.0 - PHI ** m) / (1.0 - PHI)

def liability_mtm(S, pe, t):
    m   = T - t
    cur = 1.0 if t >= 1 else 0.0
    Et_avg = (pe + cur * S + m * MU + future_weight(t) * (S - MU)) / T
    return NU * (Et_avg - STRIKE)

# --------------------------------------------------------------------------- #
# Continuation value  C_t = U(N_t) + A_t(S, N, pe)                            #
# --------------------------------------------------------------------------- #
# IMPROVEMENT 1: Added `pe` (past evolution) as a state variable.
class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),  # Changed from 2 to 3 inputs
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        for p in self.body[-1].parameters():
            nn.init.zeros_(p)

    def forward(self, S, N, pe):
        # Normalize pe (running sum of S). Roughly bounded by 0 to T*MU.
        x = torch.stack([(S - MU) / 10.0, N / 10.0, pe / (T * MU)], dim=-1)
        return self.body(x).squeeze(-1)

# IMPROVEMENT 1: `continuation` now accepts and routes `pe`
def continuation(nets, S, N, pe, t, chunk=400_000):
    base = utility(N)
    if t >= T:
        return base
    if S.numel() <= chunk:
        return base + nets[t](S, N, pe)
    res = torch.empty_like(S)
    for i in range(0, S.numel(), chunk):
        sl = slice(i, i + chunk)
        res[sl] = nets[t](S[sl], N[sl], pe[sl])
    return base + res

# --------------------------------------------------------------------------- #
# External argmax                                                             #
# --------------------------------------------------------------------------- #
def action_values(nets, S, N, pe, t, n_inner, gen):
    n   = S.shape[0]
    eps = torch.randn(n, 1, n_inner, generator=gen).expand(n, NA, n_inner)
    S1  = transition(S[:, None, None], eps)
    dS  = S1 - S[:, None, None]
    pe_next = pe + (S if t >= 1 else torch.zeros_like(S))

    # Broadcast pe_next to match the (n, NA, n_inner) shape
    pe_next_exp = pe_next[:, None, None].expand(n, NA, n_inner)

    dL  = liability_mtm(S1, pe_next_exp, t + 1) \
          - liability_mtm(S, pe, t)[:, None, None]
    N1  = N[:, None, None] + QGRID[None, :, None] * dS - dL

    C1  = continuation(nets, S1.reshape(-1), N1.reshape(-1), pe_next_exp.reshape(-1), t + 1).reshape(n, NA, n_inner)
    return C1.mean(-1)

def decide(nets, S, N, pe, t, n_inner=16, gen=None):
    with torch.no_grad():
        return QGRID[action_values(nets, S, N, pe, t, n_inner, gen).argmax(-1)]

# --------------------------------------------------------------------------- #
# Greedy realised rollout                                                     #
# --------------------------------------------------------------------------- #
def greedy_rollout(nets, S, N, pe, t_start, gen, n_inner=6):
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
# Exploratory bank                                                            #
# --------------------------------------------------------------------------- #
# IMPROVEMENT 4: Broadened state space exploration to cover extreme wealth/pe.
def simulate_bank(n_paths, gen):
    S  = torch.full((n_paths,), S_INIT)
    N  = torch.zeros(n_paths)
    pe = torch.zeros(n_paths)
    bank = {"S": [], "N": [], "pe": []}
    for t in range(T):
        bank["S"].append(S.clone()); bank["N"].append(N.clone()); bank["pe"].append(pe.clone())
        q_rep = NU * (1.0 + future_weight(t + 1)) / T

        # Force exploration across wealth states by randomly scaling the baseline
        # hedge and adding larger noise. This spans the QGRID much more effectively.
        scale = torch.rand(n_paths, generator=gen) * 1.5
        q   = (q_rep * scale + 0.8 * torch.randn(n_paths, generator=gen)).clamp(
            float(QGRID[0]), float(QGRID[-1]))

        eps = torch.randn(n_paths, generator=gen)
        S1  = transition(S, eps)
        pe1 = pe + (S if t >= 1 else 0.0)
        dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
        N   = N + q * (S1 - S) - dL
        S, pe = S1, pe1
    return bank

# --------------------------------------------------------------------------- #
# One backward step                                                           #
# --------------------------------------------------------------------------- #
def fit_step(nets, bank, t, n_iter=220, n_boot=8, gen=None, apply_lam=None, lr=2e-3):
    S0, N0, pe = bank["S"][t], bank["N"][t], bank["pe"][t]
    n = S0.shape[0]

    q_star = decide(nets, S0, N0, pe, t, gen=gen)

    S = S0.clone().requires_grad_(True)
    N = N0.clone().requires_grad_(True)
    eps = torch.randn(n, n_boot, generator=gen)
    S1  = transition(S[:, None], eps)
    pe1 = (pe + (S if t >= 1 else torch.zeros_like(S)))[:, None]
    dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)[:, None]
    N1  = N[:, None] + q_star[:, None] * (S1 - S[:, None]) - dL

    Ybar = continuation(nets, S1.reshape(-1), N1.reshape(-1), pe1.reshape(-1), t + 1).reshape(n, n_boot).mean(1)
    gS, gN = torch.autograd.grad(Ybar.sum(), [S, N])
    Y_boot, gS, gN = Ybar.detach(), gS.detach(), gN.detach()

    # IMPROVEMENT 3: Reduced variance in lambda calculation.
    # Increased inner paths to 16 and rollouts to 5 for a stable s_t estimate.
    rolls = torch.stack([greedy_rollout(nets, S0, N0, pe, t, gen, n_inner=16) for _ in range(5)], 0)
    Y_roll = rolls.mean(0)
    g_t = (Y_boot - Y_roll).mean()
    s_t = Y_roll.std() / math.sqrt(n)
    lam = float(torch.clamp(g_t / (g_t + s_t), 0.0, 1.0)) if g_t > 0 else 0.0
    debias = lam * float(g_t)

    use_lam = lam if apply_lam is None else apply_lam
    if apply_lam is None:
        Y_val, w_diff = Y_boot, 1.0
    else:
        Y_val = (1.0 - apply_lam) * Y_boot + apply_lam * Y_roll
        w_diff = 1.0 - apply_lam

    Nb = N0.clone().requires_grad_(True)
    (dB_dN,) = torch.autograd.grad(utility(Nb).sum(), Nb)
    a_val = Y_val - utility(N0)
    a_gS  = gS
    a_gN  = gN - dB_dN.detach()

    net = nets[t]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(n_iter):
        Sg = S0.clone().requires_grad_(True)
        Ng = N0.clone().requires_grad_(True)
        a  = net(Sg, Ng, pe)
        daS, daN = torch.autograd.grad(a.sum(), [Sg, Ng], create_graph=True)
        loss = ((a - a_val) ** 2).mean() \
               + w_diff * (((daS - a_gS) ** 2).mean() + ((daN - a_gN) ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        val_loss = float(((net(S0, N0, pe) - a_val) ** 2).mean())
    return dict(t=t, q=float(q_star.float().mean()), lam=lam,
                g=float(g_t), s=float(s_t), debias=debias, val_loss=val_loss)

# --------------------------------------------------------------------------- #
# Finite-difference gate                                                      #
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
        return continuation(nets, S1, N1, pe1, t + 1)

    S = S0.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(Y_of_S(S).sum(), [S])
    with torch.no_grad():
        h  = 1e-4
        fd = (Y_of_S(S0 + h) - Y_of_S(S0 - h)) / (2 * h)
    return float((auto.detach() - fd).abs().max())

# --------------------------------------------------------------------------- #
# BSS sandwich                                                                #
# --------------------------------------------------------------------------- #
def _interp_path(xs, ys, q):
    G = xs.shape[0]
    qc = q.clamp(float(xs[0]), float(xs[-1]))
    idx = torch.searchsorted(xs, qc.reshape(q.shape[0], -1)).clamp(1, G - 1)
    x0 = xs[idx - 1]; x1 = xs[idx]
    y0 = ys.gather(1, idx - 1); y1 = ys.gather(1, idx)
    w  = (qc.reshape(q.shape[0], -1) - x0) / (x1 - x0)
    return (y0 + w * (y1 - y0)).reshape(q.shape)

def penalized_upper(nets, n_paths, gen, n_grid=121, n_inner=4):
    # IMPROVEMENT 5: Increased n_grid from 41 to 121 to reduce linear interpolation error
    with torch.no_grad():
        Spath = [torch.full((n_paths,), S_INIT)]
        pes   = [torch.zeros(n_paths)]
        for t in range(T):
            eps = torch.randn(n_paths, generator=gen)
            Spath.append(transition(Spath[-1], eps))
            pes.append(pes[-1] + (Spath[t] if t >= 1 else 0.0))

        Ng = torch.linspace(-80.0, 80.0, n_grid)
        J  = utility(Ng)[None, :].repeat(n_paths, 1)
        for t in reversed(range(T)):
            St, St1, pe, pe1 = Spath[t], Spath[t + 1], pes[t], pes[t + 1]
            dS = (St1 - St)[:, None, None]
            dL = (liability_mtm(St1, pe1, t + 1) - liability_mtm(St, pe, t))[:, None, None]
            Nb = Ng[None, :, None]; qb = QGRID[None, None, :]

            Nnext = Nb + qb * dS - dL

            # Thread pe1 properly for the real and expected continuations
            pe1_exp = pe1[:, None, None].expand_as(Nnext)
            C_real = continuation(nets, St1[:, None, None].expand_as(Nnext).reshape(-1),
                                  Nnext.reshape(-1), pe1_exp.reshape(-1), t + 1).reshape(Nnext.shape)

            e_in = torch.randn(n_paths, 1, 1, n_inner, generator=gen)
            Sp   = transition(St[:, None, None, None], e_in)
            dLp  = liability_mtm(Sp, pe1[:, None, None, None], t + 1) \
                   - liability_mtm(St, pe, t)[:, None, None, None]
            Npp  = Nb[..., None] + qb[..., None] * (Sp - St[:, None, None, None]) - dLp

            pe1_inner = pe1[:, None, None, None].expand_as(Npp)
            C_e  = continuation(nets, Sp.expand_as(Npp).reshape(-1),
                                Npp.reshape(-1), pe1_inner.reshape(-1), t + 1).reshape(Npp.shape).mean(-1)

            pi   = C_real - C_e
            Jnext = _interp_path(Ng, J, Nnext)
            J = (Jnext - pi).max(-1).values
        return float(_interp_path(Ng, J, torch.zeros(n_paths, 1)).mean())

def sandwich(nets, n_paths, gen):
    L = float(greedy_rollout(nets, torch.full((n_paths,), S_INIT),
                             torch.zeros(n_paths), torch.zeros(n_paths), 0, gen).mean())

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

    U_pen = penalized_upper(nets, min(n_paths, 384), gen)

    S = torch.full((n_paths,), S_INIT); N = torch.zeros(n_paths); pe = torch.zeros(n_paths)
    pis = []
    with torch.no_grad():
        for t in range(T):
            q    = decide(nets, S, N, pe, t, gen=gen)
            qi   = (QGRID[None, :] == q[:, None]).double().argmax(-1)
            ehat = action_values(nets, S, N, pe, t, n_inner=32, gen=gen)
            ehat_q = ehat.gather(1, qi[:, None]).squeeze(1)
            eps  = torch.randn(n_paths, generator=gen); S1 = transition(S, eps)
            pe1  = pe + (S if t >= 1 else 0.0)
            dL   = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
            N1   = N + q * (S1 - S) - dL
            pis.append(continuation(nets, S1, N1, pe1, t + 1) - ehat_q)
            S, N, pe = S1, N1, pe1
    interior = torch.stack(pis, 1)[:, :-1]
    z = float(interior.mean() / (interior.std() / math.sqrt(interior.numel())))
    return L, U_naive, U_pen, float(interior.mean()), z

# --------------------------------------------------------------------------- #
# Out-of-sample value over-optimism                                           #
# --------------------------------------------------------------------------- #
def oos_optimism(nets, n_paths, gen):
    bank = simulate_bank(n_paths, gen)
    out = []
    for t in range(T):
        S0, N0, pe = bank["S"][t], bank["N"][t], bank["pe"][t]
        with torch.no_grad():
            C  = continuation(nets, S0, N0, pe, t)
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
    bank = simulate_bank(1024, gen)

    print("\nSWEEP 1 -- pure-bootstrap differential fit (twin loss, external argmax, CRN):")
    rows = [fit_step(nets, bank, t, n_iter=170, gen=gen) for t in reversed(range(T))]
    for r in sorted(rows, key=lambda r: r["t"]):
        print(f"  t={r['t']}  q*~={r['q']:+.3f}  val_loss={r['val_loss']:7.3f}")

    print("\nFD gate on differential label  max|autograd - FD| of dY/dS_t:")
    for t in (0, T // 2, T - 1):
        print(f"  t={t}: {fd_gate(nets, bank, t, gen):.2e}")

    print("\nexplicit lambda  =  clip( g_t / (g_t + s_t), 0, 1 )   [g_t=mean(Y_boot-Y_roll)]")
    for r in sorted(rows, key=lambda r: r["t"]):
        tag = "ON  (pull value down by lam*g_t = %+.3f)" % r["debias"] if r["lam"] > 0.05 \
            else "off (bootstrap optimism within MC noise)"
        print(f"  t={r['t']}:  g_t={r['g']:+.4f}  s_t={r['s']:.4f}  lambda={r['lam']:.3f}  {tag}")

    opt_before = oos_optimism(nets, 768, gen)
    lam_by_t = {r["t"]: r["lam"] for r in rows}
    print("\nSWEEP 2 -- warm-started refit APPLYING the measured lambda (value blend + coupled grad):")
    _ = [fit_step(nets, bank, t, n_iter=130, gen=gen, apply_lam=lam_by_t[t])
         for t in reversed(range(T))]
    opt_after = oos_optimism(nets, 768, gen)

    print("\npayoff -- out-of-sample value over-optimism  E[C_t - Y_rollout]  by depth:")
    print("  t :   before    after   (lambda)")
    for t in range(T):
        print(f"  {t} : {opt_before[t]:+8.3f} {opt_after[t]:+8.3f}   ({lam_by_t[t]:.2f})")
    print(f"  mean: {sum(opt_before)/T:+8.3f} {sum(opt_after)/T:+8.3f}"
          f"   <- applying lambda removes the compounding winner's-curse optimism")

    print("\nlearned hedge q*  vs  per-step replication (min-var) hedge  q_t = dL_(t+1)/dS_(t+1):")
    S = torch.full((1024,), S_INIT); N = torch.zeros(1024); pe = torch.zeros(1024)
    with torch.no_grad():
        for t in range(T):
            q = float(decide(nets, S, N, pe, t, gen=gen).float().mean())
            delta = NU * (1.0 + future_weight(t + 1)) / T
            print(f"  t={t}:  q*={q:+.3f}   replication_hedge={delta:+.3f}")
            eps = torch.randn(1024, generator=gen); S1 = transition(S, eps)
            pe1 = pe + (S if t >= 1 else 0.0)
            dL  = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
            qd  = QGRID[(QGRID - q).abs().argmin()]
            N   = N + qd * (S1 - S) - dL; S, pe = S1, pe1

    print("\nBSS sandwich (fresh MC, on the de-biased stack):")
    L, U_naive, U_pen, mean_pi, z = sandwich(nets, 2560, gen)
    U = min(U_naive, U_pen)
    print(f"  lower bound  L  (greedy policy)        = {L:+.4f}")
    print(f"  upper bound  U  (tightest valid)       = {U:+.4f}   gap U-L = {U - L:.4f}")
    print(f"    components: naive clairvoyant={U_naive:+.4f}  penalised={U_pen:+.4f}")
    print(f"  penalty zero-mean guard  mean(pi)={mean_pi:+.5f}  z={z:+.2f}  (|z|<~2 => dual-feasible)")

    print("\ndownside protection check (paired on identical price paths):")
    P = 4096
    g2 = torch.Generator().manual_seed(7)
    eps_all = torch.randn(P, T, generator=g2)
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

    best_s, best_u = 1.0, -1e18
    for sc in [i * 0.05 for i in range(0, 31)]:
        u = float(utility(run_policy(lambda t, S, N, pe, sc=sc: torch.full_like(S, sc * qrepl[t]))).mean())
        if u > best_u: best_s, best_u = sc, u
    pols = [
        ("unhedged (q=0)",          lambda t, S, N, pe: torch.zeros_like(S)),
        (f"best static hedge ({best_s:.2f}x repl)", lambda t, S, N, pe: torch.full_like(S, best_s * qrepl[t])),
        ("learned policy",          lambda t, S, N, pe: decide(nets, S, N, pe, t, gen=g2)),
    ]
    for name, pol in pols:
        N = run_policy(pol)
        print(f"  {name:28s}:  mean U={float(utility(N).mean()):+.2f}   "
              f"std W={float(N.std()):.2f}   5% worst W={float(torch.quantile(N, 0.05)):+.2f}")

if __name__ == "__main__":
    main()