"""How wrong is the endpoint-only touch flag, and does a Brownian bridge fix it?

FINDINGS (2026-08-01, S0=100 vol=0.25 T=1, 200k paths, DOWN barrier 90):

    closed form (truth)                 0.29169
    bridge on a 4-point QUARTERLY grid  0.29104   -0.0007
    bridge on 2520 steps                0.29165   -0.00004
    endpoint-only, daily                0.31636   +0.025
    endpoint-only, quarterly            0.47057   +0.179     <- 61% overstated

The bridge on FOUR points beats a 2520-step brute-force simulation without it by ~30x. The
naive fine-grid estimate is itself biased high - it still misses crossings between its own
steps - which is why it is not used as the reference here.

riskflow's Reiner-Rubinstein closed form is separately verified against Haug to 1e-15 on both
the K>H and K<H branches, so the REMAINING life is priced exactly. The defect is entirely in
the historical state.

pv_barrier_option prices the REMAINING life with Reiner-Rubinstein, which assumes CONTINUOUS
monitoring and is right. The historical state is `touched = (prev + (s_t > barrier)).clip(max=1)`
- it only asks whether the spot sits beyond the barrier AT an exposure date, so any path that
crossed and came back in between is recorded as never having touched.

This measures that, with a finely simulated path as the truth, and checks that the
bridge probability recovers it. Everything here is plain GBM and independent of riskflow's
pricers, so it is a reference rather than a self-consistency check.
"""
import numpy as np
import torch

torch.manual_seed(0)
D = torch.float64
S0, VOL, DRIFT, T = 100.0, 0.25, 0.0, 1.0
PATHS = 200_000
FINE = 2520                      # ~daily x10, the "continuous" truth


def simulate(steps, paths=PATHS, seed=0):
    """GBM log-path on `steps` equal intervals, shape (steps+1, paths)."""
    g = torch.Generator().manual_seed(seed)
    dt = T / steps
    z = torch.randn(steps, paths, generator=g, dtype=D)
    incr = (DRIFT - 0.5 * VOL ** 2) * dt + VOL * np.sqrt(dt) * z
    return S0 * torch.cat([torch.zeros(1, paths, dtype=D), incr.cumsum(0)], 0).exp()


def bridge_survival(s0, s1, barrier, var, down=True):
    """P(no crossing in the interval | both endpoints safe), the Brownian-bridge probability."""
    if down:
        d0, d1 = torch.log(s0 / barrier), torch.log(s1 / barrier)
    else:
        d0, d1 = torch.log(barrier / s0), torch.log(barrier / s1)
    safe = (d0 > 0) & (d1 > 0)
    p_cross = torch.exp((-2.0 * d0 * d1 / var).clamp(max=0.0))
    return torch.where(safe, 1.0 - p_cross, torch.zeros_like(p_cross))


print(f'GBM S0={S0} vol={VOL} T={T} paths={PATHS:,}\n')
fine = simulate(FINE)

for barrier, down in ((90.0, True), (80.0, True), (110.0, False)):
    label = f'{"DOWN" if down else "UP"} barrier {barrier}'
    # truth: continuous monitoring, approximated by the fine path
    hit_fine = (fine <= barrier).any(0) if down else (fine >= barrier).any(0)
    truth = 1.0 - hit_fine.double().mean()

    print(f'{label}   true survival (2520 steps) {truth:.5f}')
    print(f'   {"grid":>10} {"endpoint-only":>15} {"error":>9} {"bridge":>10} {"error":>9}')
    for steps in (4, 12, 52, 252):
        idx = torch.linspace(0, FINE, steps + 1, dtype=torch.long)
        coarse = fine[idx]
        # what riskflow does today: look only at the grid points
        hit_end = (coarse <= barrier).any(0) if down else (coarse >= barrier).any(0)
        endpoint = 1.0 - hit_end.double().mean()
        # the bridge: survival over each interval, conditional on both endpoints
        var = (VOL ** 2) * (T / steps)
        q = bridge_survival(coarse[:-1], coarse[1:], barrier,
                            torch.full_like(coarse[:-1], var), down=down)
        bridge = q.prod(0).mean()
        print(f'   {steps:10d} {endpoint:15.5f} {endpoint - truth:+9.5f} '
              f'{bridge:10.5f} {bridge - truth:+9.5f}')
    print()
