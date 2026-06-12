"""Guard v2 for the propagated-belief baseline (validation_sandwich_spec §2).

Runs against the SHIPPED `DifferentialSolver._baseline_B` / `_dB_dz`,
`build_cumulative_regime_drift`, and `twin_loss` from the patched
hedge_solver.py via a duck-typed instance.

Checks:
  1. Table: `build_cumulative_regime_drift` vs the direct definition
     `cum[a,k] = Σ_{u<k} (P^u μ)·dt` (constant P, dt), and vs an independent
     2-state hand recurrence with heterogeneous dt.
  2. Regression anchor: P = I + one-hot belief + single-terminal-fixing ladder
     reproduces the OLD constant-μ̄ closed form `symlog(w + (Σq·cs − R)·S·
     expm1(μ_r·ttot))` exactly.
  3. Central-FD vs `_dB_dz` per column (float64), one-hot and soft beliefs,
     contract_size ∈ {1, 50}, mixing P. Time probes at mid-step (gathers make
     B piecewise-constant in time; production ttot is integer).
  4. Structural zeros: accrued / static / futures[1:] — and now time (gather-
     indexed ⇒ within-step slope 0).
  5. Terminal identity: ttot=0, empty remaining ladder ⇒ dB/dw = 1/(c+|w|),
     all else 0.
  6. Sign table: bull/bear × long/short under a mixing P.
  7. twin_loss col_mask invariants + masked twin fit stability (unchanged
     code path — re-asserted).
"""
import math
import sys
import types

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

from pkg.hedge_solver import (  # noqa: E402
    DifferentialSolver, TwinNetwork, twin_loss, _assemble_deep_state,
    build_cumulative_regime_drift)

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)


WEEK = 7.0 / 365.25


def make_solver(n_h, market_dim, regime_cols, mu, T, P_calib, *, cs=None,
                ladder="terminal", c=100.0):
    s = object.__new__(DifferentialSolver)
    s._advantage_decomp = True
    s.statepack = types.SimpleNamespace(n_hedge=n_h, market_dim=market_dim)
    s._regime_cols_in_market = regime_cols
    s._mu_per_regime = torch.tensor(mu)
    s._dt_per_step = WEEK
    s.utility_c = c
    s.t_outer = T
    s.contract_size = (torch.ones(n_h) if cs is None
                       else torch.as_tensor(cs, dtype=torch.get_default_dtype()))
    s._cum_mu_state = build_cumulative_regime_drift(
        P_calib, WEEK, [WEEK] * (T - 1), s._mu_per_regime)
    n = len(mu)
    w = torch.zeros(T, T + 1)
    if ladder == "terminal":          # toy: one fixing at the terminal step
        for t in range(T - 1):
            w[t, T - 1 - t] = 1.0     # weight 1, integer lag to terminal
    elif ladder == "spread":          # daily fixings over the last 3 weeks
        fix_lags_days = range(1, 22)
        for t in range(T):
            for d in fix_lags_days:
                lag = d / 7.0 + (0)   # day-lag in outer (weekly) steps
                # only fixings landing before terminal-from-t
                if lag <= 0 or t + lag > T - 1 + 1e-9:
                    continue
                k0, fr = int(lag), lag - int(lag)
                w[t, k0] += (1.0 / 21.0) * (1 - fr)
                w[t, k0 + 1] += (1.0 / 21.0) * fr
    s._liab_w_by_lag = w
    s._liability_R = w.sum(dim=1)
    s._liab_drift_state = (w.unsqueeze(-1)
                           * torch.expm1(s._cum_mu_state)).sum(dim=1)
    return s


def make_z(s, n_static, N, soft=False, ttot=None):
    n_h, md = s.statepack.n_hedge, s.statepack.market_dim
    r_lo, r_hi = s._regime_cols_in_market
    n_r = r_hi - r_lo
    q = torch.randint(-5, 6, (N, n_h)).double()
    w = (torch.rand(N) - 0.5) * 3000.0
    a = torch.randn(N) * 7.0
    F = 100.0 * torch.exp(0.3 * torch.randn(N, n_h))
    market = torch.randn(N, md) * 0.5
    if soft:
        b = torch.rand(N, n_r); b = b / b.sum(-1, keepdim=True)
    else:
        b = torch.zeros(N, n_r)
        b[torch.arange(N), torch.randint(0, n_r, (N,))] = 1.0
    market[:, r_lo:r_hi] = b
    static = torch.randn(N, n_static)
    tt = (torch.randint(1, s.t_outer, (N,)).double() if ttot is None
          else torch.full((N,), float(ttot)))
    return _assemble_deep_state(positions=q, g=w, a=a, futures=F,
                                market=market, static=static, time_to_t=tt)


def fd_grad(fn, z, eps_base=1e-5):
    g = torch.zeros_like(z)
    for j in range(z.shape[-1]):
        e = eps_base * max(1.0, float(z[:, j].abs().max()))
        zp, zm = z.clone(), z.clone()
        zp[:, j] += e; zm[:, j] -= e
        g[:, j] = (fn(zp) - fn(zm)) / (2.0 * e)
    return g


print("== 1. cumulative-drift table ==")
mu2 = torch.tensor([0.30, -0.25])
P = [[0.92, 0.08], [0.15, 0.85]]
T = 12
cum = build_cumulative_regime_drift(P, WEEK, [WEEK] * (T - 1), mu2)
# Direct definition (constant P, dt): cum[a,k] = Σ_{u<k} (P^u μ)·dt — a-invariant.
Pt = torch.tensor(P)
direct = torch.zeros(T + 1, 2)
acc, Pu = torch.zeros(2), torch.eye(2)
for k in range(1, T + 1):
    acc = acc + (Pu @ mu2) * WEEK
    Pu = Pt @ Pu
    direct[k] = acc
err = max(float((cum[a] - direct).abs().max()) for a in range(T))
check("matches direct Σ (P^u μ)·dt (constant P)", err < 1e-12, f"max_err={err:.1e}")
# Heterogeneous dt vs independent hand recurrence (P re-discretised per dt).
dts = [WEEK, 2 * WEEK, WEEK / 2, WEEK, 3 * WEEK]
cum_h = build_cumulative_regime_drift(P, WEEK, dts, mu2)
import numpy as np
from scipy.linalg import logm, expm
Q = np.real(logm(np.array(P))) / WEEK
Ps = [torch.tensor(np.real(expm(Q * d))) for d in dts]
ref = torch.zeros(len(dts) + 1, len(dts) + 2, 2)
for a in range(len(dts) - 1, -1, -1):
    for k in range(1, len(dts) + 2):
        nxt = ref[a + 1][k - 1] if a + 1 <= len(dts) else ref[len(dts)][k - 1]
        ref[a, k] = mu2 * dts[a] + Ps[a] @ nxt
# Compare on the production-valid wedge k ≤ T−1−a only (beyond it the
# builder applies its documented time-homogeneous extension; the hand
# reference has a zero tail there — and production never queries past it).
err_h = max(float((cum_h[a, :len(dts) - a + 1] - ref[a, :len(dts) - a + 1])
                  .abs().max()) for a in range(len(dts)))
check("heterogeneous-dt recurrence (valid wedge)", err_h < 1e-12,
      f"max_err={err_h:.1e}")

print("\n== 2. regression anchor: P=I, one-hot, terminal ladder == old closed form ==")
mu3 = [0.25, 0.02, -0.30]
sI = make_solver(2, 5, (1, 4), mu3, T=14, P_calib=torch.eye(3).tolist())
zI = make_z(sI, n_static=3, N=128)
B_new = sI._baseline_B(zI)
# Old form: symlog(w + (Σq·cs − R)·S·expm1(μ_r·ttot)); R=1 below terminal.
n_h = 2
qcs = (zI[:, :n_h] * sI.contract_size).sum(-1)
wlt, S = zI[:, n_h], zI[:, n_h + 2]
b = zI[:, 2 * n_h + 2 + 1: 2 * n_h + 2 + 4]
mu_r = (b * sI._mu_per_regime).sum(-1)
tt = zI[:, -1]
u_old = wlt + (qcs - 1.0) * S * torch.expm1(mu_r * tt * WEEK)
B_old = torch.sign(u_old) * torch.log1p(u_old.abs() / sI.utility_c)
err_a = float((B_new - B_old).abs().max())
check("bit-level match to pre-propagation baseline", err_a < 1e-12,
      f"max|ΔB|={err_a:.1e}")

print("\n== 3/4. FD vs _dB_dz (mixing P), structural zeros ==")
P3 = [[0.90, 0.07, 0.03], [0.10, 0.80, 0.10], [0.04, 0.06, 0.90]]


def run_fd(label, s, z, n_static):
    with torch.no_grad():
        B = s._baseline_B(z)
        u = torch.sign(B) * (torch.expm1(B.abs()) * s.utility_c)
    keep = u.abs() > 5.0
    z = z[keep]
    ana = s._dB_dz(z)
    num = fd_grad(s._baseline_B, z)
    abs_err = (ana - num).abs()
    sig = num.abs() > 1e-8
    max_rel = float((abs_err / num.abs().clamp_min(1e-8))[sig].max()) if sig.any() else 0.0
    check(f"{label}: FD vs _dB_dz", max_rel < 1e-5 and float(abs_err.max()) < 1e-6,
          f"max_rel={max_rel:.2e} rows={int(keep.sum())}")
    n_h = s.statepack.n_hedge
    zero_cols = ([n_h + 1] + [n_h + 2 + j for j in range(1, n_h)]
                 + [z.shape[-1] - 1 - n_static + k for k in range(n_static)]
                 + [z.shape[-1] - 1])                      # + time (gathers)
    zmax = max(float(ana[:, c].abs().max()) for c in zero_cols)
    zmax_fd = max(float(num[:, c].abs().max()) for c in zero_cols)
    check(f"{label}: structural zeros (accr/static/F[1:]/time)",
          zmax < 1e-12 and zmax_fd < 1e-6, f"ana={zmax:.1e} fd={zmax_fd:.1e}")


s3 = make_solver(2, 5, (1, 4), mu3, T=14, P_calib=P3, ladder="spread")
z3 = make_z(s3, 3, 96); z3[:, -1] += 0.4          # mid-step time probes
run_fd("one-hot, cs=1, spread ladder", s3, z3, 3)
z3s = make_z(s3, 3, 96, soft=True); z3s[:, -1] += 0.4
run_fd("soft belief", s3, z3s, 3)
s50 = make_solver(3, 7, (0, 4), [0.2, 0.05, -0.1, -0.35], T=10,
                  P_calib=torch.eye(4).tolist(), cs=[50.0, 50.0, 1.0],
                  ladder="terminal")
z50 = make_z(s50, 1, 96); z50[:, -1] += 0.4
run_fd("n_h=3, cs=50, P=I", s50, z50, 1)

print("\n== 5. terminal identity ==")
zt = make_z(s3, 3, 64, ttot=0)
g_t = s3._dB_dz(zt)
w_col = 2
sp = 1.0 / (s3.utility_c + zt[:, w_col].abs())
others = g_t.clone(); others[:, w_col] = 0.0
check("ttot=0: dB/dw = 1/(c+|w|), all else 0 (R[T-1]=0 ladder)",
      float((g_t[:, w_col] - sp).abs().max()) < 1e-12
      and float(others.abs().max()) < 1e-12)

print("\n== 6. sign table (mixing P, ttot=10wk, spread ladder) ==")
rows = []
for regime, r_idx in (("bull", 0), ("bear", 2)):
    for pos, qv in (("long", 3.0), ("short", -3.0)):
        z1 = make_z(s3, 3, 1, ttot=10)
        z1[:, :2] = qv; z1[:, 2] = 50.0
        m = torch.zeros(3); m[r_idx] = 1.0
        z1[:, 2 * 2 + 2 + 1: 2 * 2 + 2 + 4] = m
        g = s3._dB_dz(z1)[0]
        rows.append((float(g[0]), float(g[4])))
        print(f"  {regime:4s} {pos:6s}  dB/dq={g[0]:+.5f}  dB/dS={g[4]:+.5f}")
check("bull ⇒ dB/dq>0, bear ⇒ dB/dq<0 (mixing damps, not flips, at 10wk)",
      rows[0][0] > 0 and rows[1][0] > 0 and rows[2][0] < 0 and rows[3][0] < 0)

print("\n== 7. twin_loss col_mask invariants (re-asserted) ==")
D = 12
net = TwinNetwork(D, hidden_sizes=(32, 32), device="cpu")
zr = torch.randn(256, D)
net.set_normalization(zr.mean(0), zr.std(0), torch.tensor(0.3), torch.tensor(0.7))
y = torch.randn(256); dy = torch.randn(256, D)
l_none, _ = twin_loss(net, zr, y, dy)
l_ones, _ = twin_loss(net, zr, y, dy, col_mask=torch.ones(D))
check("all-ones mask ≡ legacy",
      abs(float((l_none - l_ones).detach())) < 1e-12)
mask = torch.zeros(D); mask[2] = 1.0; mask[5] = 1.0
dy_p = dy.clone(); dy_p[:, 0] += 1e6
la, _ = twin_loss(net, zr, y, dy, col_mask=mask)
lb, _ = twin_loss(net, zr, y, dy_p, col_mask=mask)
check("masked columns unconstrained", abs(float((la - lb).detach())) < 1e-9)
teacher = TwinNetwork(D, hidden_sizes=(32, 32), device="cpu")
teacher.set_normalization(zr.mean(0), zr.std(0), torch.tensor(0.0), torch.tensor(1.0))
zl = zr.detach().clone().requires_grad_(True)
yt = teacher(zl)
dyt = (torch.autograd.grad(yt.sum(), zl)[0].detach()) * mask
yt = yt.detach()
student = TwinNetwork(D, hidden_sizes=(32, 32), device="cpu")
student.set_normalization(zr.mean(0), zr.std(0), yt.mean(), yt.std())
opt = torch.optim.Adam(student.parameters(), lr=3e-3)
h0 = h1 = None
for it in range(400):
    loss, diag = twin_loss(student, zr, yt, dyt, col_mask=mask)
    opt.zero_grad(); loss.backward(); opt.step()
    if it == 0: h0 = (diag["val_loss"], diag["diff_loss"])
    h1 = (diag["val_loss"], diag["diff_loss"])
check("masked twin fit trains stably",
      math.isfinite(h1[0]) and h1[0] < 0.2 * h0[0] and h1[1] < 0.2 * h0[1],
      f"val {h0[0]:.3f}→{h1[0]:.4f}, diff {h0[1]:.3f}→{h1[1]:.4f}")

print()
if FAIL:
    print("RESULT: FAIL —", FAIL); sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
