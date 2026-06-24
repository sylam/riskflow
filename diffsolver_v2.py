"""DiffSolverV2 — clean-room differential-ML hedging solver, built FROM THE TOY.

Reference: diffml_hedge_huber.py (the standalone toy that WORKS — verified bounded at
T=119). This is a from-scratch rewrite that does NOT read the existing DifferentialSolver
or other methods in hedge_solver.py; it bakes in the lessons of that investigation:

  * The toy's pure-bootstrap backward DP is BOUNDED at T=119 (val_loss O(1e3), bias goes
    negative). The production solver's 1e8 blow-up was therefore a BUG in its bootstrap,
    not fundamental depth-compounding. Build clean to avoid it.
  * Continuation is `C = bounded_baseline(U) + residual_A`; the bounded utility anchors the
    value so the unbounded residual net can't run away off-support.
  * External argmax (the Bellman max lives OUTSIDE the fitted value), advantage decomposition
    (fit A = C − B with the value AND the pathwise gradient — Huge-Savine twin loss),
    operating-region bank (explore AROUND the replication hedge so A stays small/on-support).
  * MULTI-INSTRUMENT from the start: the state carries the full position vector q ∈ R^{n_inst}
    so the net always knows there are n_inst instruments. The action grid is over all of them;
    a "single-future" test activates one axis and pins the others to 0 (e.g. [0,0,-50]…[0,0,0]).

Stage 1 (this file, validated first): the toy's analytic single-underlying dynamics,
generalised to n_inst instruments, run standalone to confirm the clean core reproduces the
toy's BOUNDED behaviour at depth. Stage 2 swaps the analytic dynamics for the riskflow bundle's
`inner_mc_grad_fn` (resolve_structure/resolve_hedge_structure) — same solver logic, real pricing.
"""
import math

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Problem constants (toy values; Stage 2 reads these from the riskflow runtime)
# --------------------------------------------------------------------------- #
T = 8                 # decision dates 0..T-1; averaging dates 1..T; payoff at T
MU = 100.0            # mean / start (no drift)
KAPPA = 0.25          # AR(1) mean-reversion speed
PHI = 1.0 - KAPPA
SIGMA = 4.0           # per-step shock std (price level)
NU = 3.0              # liability notional
S_INIT = MU
STRIKE = MU
AVERSION = 1.0        # Huber downside aversion
DELTA = 5.0           # Huber quadratic→linear knee (loss units)

N_INST = 1            # number of hedge instruments (1 = toy; 3 = platinum)
LEVELS = 29           # action levels per instrument
Q_LO, Q_HI = -2.5, 4.5    # per-instrument position range (toy scale)
DEVICE = torch.device("cpu")


def _qgrid_1d():
    return torch.linspace(Q_LO, Q_HI, LEVELS, device=DEVICE)


def action_grid(active=None):
    """Full action grid over n_inst instruments as (K, n_inst). `active` is an optional list
    of instrument indices that VARY; the rest are pinned to 0 (e.g. active=[2] on 3 instruments
    gives [0,0,-2.5]…[0,0,4.5] — the single-future-of-three test the production deal needs).
    With active=None all instruments vary (full product grid)."""
    g = _qgrid_1d()
    axes = []
    active = list(range(N_INST)) if active is None else list(active)
    for i in range(N_INST):
        axes.append(g if i in active else torch.zeros(1, device=DEVICE))
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)            # (K, n_inst)


# --------------------------------------------------------------------------- #
# Dynamics — analytic single underlying (toy). Stage 2 replaces with the bundle.
# All instruments hedge the SAME underlying S; F_i differ only by carry (here 1 ⇒ F_i=S).
# --------------------------------------------------------------------------- #
CARRY = None          # (n_inst,) per-instrument F/S ratio; None ⇒ all 1.0 (toy)


def _carry():
    return torch.ones(N_INST, device=DEVICE) if CARRY is None else CARRY


def transition(S, eps):
    """AR(1) one-step (mean-reverting)."""
    return S + KAPPA * (MU - S) + SIGMA * eps


def utility(W):
    """Huber downside utility (toy): linear gains, quadratic small losses, linear deep tail —
    BOUNDED-growth so it anchors the continuation."""
    loss = torch.clamp(-W, min=0.0)
    quad = AVERSION * loss ** 2
    lin = AVERSION * DELTA ** 2 + 2.0 * AVERSION * DELTA * (loss - DELTA)
    return W - torch.where(loss <= DELTA, quad, lin)


def future_weight(t):
    m = T - t
    return 0.0 if m <= 0 else PHI * (1.0 - PHI ** m) / (1.0 - PHI)


def liability_mtm(S, pe, t):
    """MTM of the average-rate forward, differentiable in S (a martingale in t)."""
    m = T - t
    cur = 1.0 if t >= 1 else 0.0
    Et_avg = (pe + cur * S + m * MU + future_weight(t) * (S - MU)) / T
    return NU * (Et_avg - STRIKE)


# --------------------------------------------------------------------------- #
# Continuation  C_t(S, q_vec, W) = U(W)  +  A_t(S, q_vec, W)
#   baseline U(W): the bounded terminal utility of CURRENT wealth (toy's anchor).
#   residual A_t : a small NN correction, the only learned part.
# --------------------------------------------------------------------------- #
class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(1 + N_INST + 1, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        for p in self.body[-1].parameters():        # init A ≈ 0 ⇒ C starts at baseline U
            nn.init.zeros_(p)

    def forward(self, S, q_vec, W):
        x = torch.cat([((S - MU) / 10.0).unsqueeze(-1),
                       q_vec / 10.0,
                       (W / 10.0).unsqueeze(-1)], dim=-1)
        return self.body(x).squeeze(-1)


def continuation(nets, S, q_vec, W, t, chunk=400_000):
    """C_t = U(W) + A_t(S, q_vec, W). Terminal C_T = U(W). Net eval row-chunked."""
    base = utility(W)
    if t >= T:
        return base
    if S.numel() <= chunk:
        return base + nets[t](S, q_vec, W)
    res = torch.empty_like(S)
    for i in range(0, S.numel(), chunk):
        sl = slice(i, i + chunk)
        res[sl] = nets[t](S[sl], q_vec[sl], W[sl])
    return base + res


# --------------------------------------------------------------------------- #
# One-step wealth move for a held position vector q (carry: F_i = carry_i·S).
# --------------------------------------------------------------------------- #
def wealth_step(W, q_vec, S, S1, pe, t):
    """W_{t+1} = W + Σ_i q_i·(F_i(S1) − F_i(S)) − dL. F_i = carry_i·S."""
    c = _carry()
    dF = (S1.unsqueeze(-1) - S.unsqueeze(-1)) * c                       # (..., n_inst)
    pnl = (q_vec * dF).sum(dim=-1)
    pe1 = pe + (S if t >= 1 else torch.zeros_like(S))
    dL = liability_mtm(S1, pe1, t + 1) - liability_mtm(S, pe, t)
    return W + pnl - dL, pe1


# --------------------------------------------------------------------------- #
# External argmax over the action grid (the Bellman max OUTSIDE the fitted value)
# --------------------------------------------------------------------------- #
def action_values(nets, S, q_prev, W, pe, t, grid, n_inner, gen):
    """E[C_{t+1}] for every grid action via inner MC over S_{t+1}. CRN across actions.
    Returns (n, K)."""
    n, K = S.shape[0], grid.shape[0]
    eps = torch.randn(n, 1, n_inner, generator=gen, device=DEVICE).expand(n, K, n_inner)
    S1 = transition(S[:, None, None], eps)                              # (n,K,inner)
    q = grid[None, :, None, :].expand(n, K, n_inner, N_INST)            # held position = action
    W1, _ = wealth_step(W[:, None, None], q, S[:, None, None], S1, pe[:, None, None], t)
    C1 = continuation(nets, S1.reshape(-1), q.reshape(-1, N_INST),
                      W1.reshape(-1), t + 1).reshape(n, K, n_inner)
    return C1.mean(-1)


def decide(nets, S, q_prev, W, pe, t, grid, n_inner=16, gen=None):
    with torch.no_grad():
        return grid[action_values(nets, S, q_prev, W, pe, t, grid, n_inner, gen).argmax(-1)]


# --------------------------------------------------------------------------- #
# Operating-region bank: roll q AROUND the replication hedge so wealth stays in-band.
# --------------------------------------------------------------------------- #
def replication_hedge(t):
    """Per-instrument min-var hedge: short the liability's spot delta, split across instruments.
    dL/dS = NU·(cur + future_weight(t))/T (from liability_mtm). Hedge offsets it via Σ q_i·carry_i."""
    cur = 1.0 if t >= 1 else 0.0
    dLdS = NU * (cur + future_weight(t)) / T
    c = _carry()
    return -(dLdS / N_INST) / c                                         # (n_inst,)


def simulate_bank(n_paths, gen):
    S = torch.full((n_paths,), S_INIT, device=DEVICE)
    W = torch.zeros(n_paths, device=DEVICE)
    pe = torch.zeros(n_paths, device=DEVICE)
    q_prev = torch.zeros(n_paths, N_INST, device=DEVICE)
    bank = {"S": [], "q_prev": [], "W": [], "pe": []}
    for t in range(T):
        bank["S"].append(S.clone()); bank["q_prev"].append(q_prev.clone())
        bank["W"].append(W.clone()); bank["pe"].append(pe.clone())
        q_rep = replication_hedge(t)                                    # (n_inst,)
        # explore around the hedge, spanning the grid for argmax signal
        q = (q_rep[None, :] + 0.7 * torch.randn(n_paths, N_INST, generator=gen, device=DEVICE)
             ).clamp(Q_LO, Q_HI)
        eps = torch.randn(n_paths, generator=gen, device=DEVICE)
        S1 = transition(S, eps)
        W, pe = wealth_step(W, q, S, S1, pe, t)
        S, q_prev = S1, q
    return bank


# --------------------------------------------------------------------------- #
# One backward step: bootstrap value + pathwise gradient, advantage decomp, twin fit.
# --------------------------------------------------------------------------- #
def fit_step(nets, bank, t, grid, n_iter=200, n_boot=8, gen=None, lr=2e-3):
    S0, q0, W0, pe = bank["S"][t], bank["q_prev"][t], bank["W"][t], bank["pe"][t]
    n = S0.shape[0]
    q_star = decide(nets, S0, q0, W0, pe, t, grid, gen=gen)             # external argmax (no grad)

    # bootstrap value + pathwise grad over n_boot inner draws (S0, W0 leaves)
    S = S0.clone().requires_grad_(True)
    W = W0.clone().requires_grad_(True)
    eps = torch.randn(n, n_boot, generator=gen, device=DEVICE)
    S1 = transition(S[:, None], eps)
    q = q_star[:, None, :].expand(n, n_boot, N_INST)
    W1, _ = wealth_step(W[:, None], q, S[:, None], S1, pe[:, None], t)
    Ybar = continuation(nets, S1.reshape(-1), q.reshape(-1, N_INST),
                        W1.reshape(-1), t + 1).reshape(n, n_boot).mean(1)
    gS, gW = torch.autograd.grad(Ybar.sum(), [S, W])
    Y_boot, gS, gW = Ybar.detach(), gS.detach(), gW.detach()

    # advantage decomposition: fit A = C − U(W0); subtract baseline's grad on W
    Wb = W0.clone().requires_grad_(True)
    (dB_dW,) = torch.autograd.grad(utility(Wb).sum(), Wb)
    a_val = Y_boot - utility(W0)
    a_gS = gS                                                          # dU/dS = 0
    a_gW = gW - dB_dW.detach()

    net = nets[t]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(n_iter):
        Sg = S0.clone().requires_grad_(True)
        Wg = W0.clone().requires_grad_(True)
        a = net(Sg, q_star, Wg)
        daS, daW = torch.autograd.grad(a.sum(), [Sg, Wg], create_graph=True)
        loss = ((a - a_val) ** 2).mean() + ((daS - a_gS) ** 2).mean() + ((daW - a_gW) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        val_loss = float(((net(S0, q_star, W0) - a_val) ** 2).mean())
    # bias vs an unbiased greedy rollout (diagnostic only; NOT applied — toy stays clean)
    return dict(t=t, val_loss=val_loss, Yboot_absmean=float(Y_boot.abs().mean()),
                q=float(q_star.abs().mean()))


def run(T_=8, n_inst=1, n_paths=512, n_iter=120, active=None, seed=1, verbose=True):
    """Backward sweep; prints per-t val_loss / |Y_boot| to check BOUNDEDNESS at depth."""
    global T, N_INST
    T, N_INST = T_, n_inst
    gen = torch.Generator().manual_seed(seed)
    nets = [Residual() for _ in range(T)]
    bank = simulate_bank(n_paths, gen)
    grid = action_grid(active)
    rows = []
    for t in reversed(range(T)):
        r = fit_step(nets, bank, t, grid, n_iter=n_iter, gen=gen)
        rows.append(r)
        if verbose:
            print(f"  t={r['t']:3d}  val_loss={r['val_loss']:.4g}  "
                  f"|Y_boot|={r['Yboot_absmean']:.4g}  |q*|={r['q']:.3f}", flush=True)
    worst = max(r["Yboot_absmean"] for r in rows)
    print(f"DiffSolverV2 T={T} n_inst={N_INST} active={active} grid_K={grid.shape[0]}: "
          f"max|Y_boot| over sweep = {worst:.4g}  ({'BOUNDED' if worst < 1e4 else 'EXPLODED'})",
          flush=True)
    return rows


if __name__ == "__main__":
    import sys
    T_ = int(sys.argv[1]) if len(sys.argv) > 1 else 119
    n_inst = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    active = [n_inst - 1] if n_inst > 1 else None     # single-future-of-n test: last axis varies
    run(T_=T_, n_inst=n_inst, n_paths=384, n_iter=80, active=active)
