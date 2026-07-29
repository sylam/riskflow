"""ARM B md builder + price-space drift diagnostic (throwaway, tb_ prefix, no commit).

(1) Builds the 8 DRIFTLESS GARCH md files: byte-identical copies of the exact garch48 md the
    ARM A checkpoints were trained on, with ONLY Convexity_Correction=Yes spliced onto the
    GARCHSpotModel.PLATINUM_CME block. This isolates the DRIFT term (no re-calibration => every
    other param identical), and matches what production garchify_md now stamps.
(2) Price-space drift diagnostic per month: replicates the EXACT _simulate_returns recursion
    (fractional trading clock f, n_sub=1 on the calendar-daily grid, convexity term) in numpy and
    reports annualized E[dS/S] under convexity OFF (ARM A, with-drift) vs ON (ARM B, driftless).
    Confirms ARM A carries the +1/2 h Jensen drift and ARM B is a price-martingale (~0).
"""
import os, sys, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
MDDIR = os.path.join(WF, 'driftless_md')

MONTHS = ['202105', '202202', '202106', '202001', '202204', '202011', '202102', '202112']

CAL_DT = 1.0 / 252.0          # calibration clock (business day)
GRID_DT = 1.0 / 365.25        # sim grid step (calendar daily "0d 1d(1d)")
F = GRID_DT / CAL_DT          # trading-time step length ~0.69 (n_sub=1)
H_STEPS = 90                  # ~ roll horizon in calendar days
N_PATHS = 200_000
SEED = 12345


def month_md_map():
    from tb_garch_corridor import build_month_map
    mmap = build_month_map()
    return {m: mmap[m][1] for m in MONTHS}


def build_driftless(src_md, out_md):
    md = json.load(open(src_md))
    blk = md['MarketData']['Price Models']['GARCHSpotModel.PLATINUM_CME']
    assert blk.get('Convexity_Correction', 'No') == 'No', f'{src_md} already convexity-on'
    blk['Convexity_Correction'] = 'Yes'
    tmp = out_md + '.tmp'
    json.dump(md, open(tmp, 'w'), indent=1)
    os.replace(tmp, out_md)
    return blk


def params_identical(src_md, out_md):
    a = json.load(open(src_md))['MarketData']['Price Models']['GARCHSpotModel.PLATINUM_CME']
    b = json.load(open(out_md))['MarketData']['Price Models']['GARCHSpotModel.PLATINUM_CME']
    keys = set(a) | set(b)
    diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
    return diff  # should be exactly {'Convexity_Correction': ('No'/absent, 'Yes')}


def drift_annual(block, convexity, rng):
    """Annualized E[dS/S] over H_STEPS calendar days for one GARCH block, matching
    _simulate_returns (n_sub=1, ds = r - 0.5*var if convexity else r; Mu=0)."""
    omega = float(block['Omega']); alpha = float(block['Alpha']); beta = float(block['Beta'])
    nu = float(block['Nu']); h0 = float(block['H0'])
    # standardized-t innovations, exactly as generate(): eps = Z*sqrt(nu/W)*sqrt((nu-2)/nu)
    Z = rng.standard_normal((H_STEPS, N_PATHS))
    W = rng.gamma(nu / 2.0, 2.0, (H_STEPS, N_PATHS)).clip(1e-6)   # Gamma(nu/2, scale=2)=chi2_nu
    eps = Z * np.sqrt(nu / W) * np.sqrt((nu - 2.0) / nu)
    h = np.full(N_PATHS, h0)
    log_ret = np.zeros(N_PATHS)
    for t in range(H_STEPS):
        var_step = h * F
        r = np.sqrt(var_step) * eps[t]
        ds = (r - 0.5 * var_step) if convexity else r
        log_ret += ds
        h = h + F * (omega - (1.0 - beta) * h) + alpha * r * r
    price_ret = np.exp(log_ret) - 1.0                      # S_T/S_0 - 1
    t_years = H_STEPS * GRID_DT
    return dict(
        E_dS_over_S_ann=float(price_ret.mean() / t_years),          # simple annualized price drift
        E_dlogS_ann=float(log_ret.mean() / t_years),                # log-space drift (should be ~0 both)
        lr_vol=float(np.sqrt(omega / (1.0 - alpha - beta) / CAL_DT)),
    )


def main():
    os.makedirs(MDDIR, exist_ok=True)
    mmap = month_md_map()
    rows = []
    seen = {}
    for m in MONTHS:
        src = mmap[m]
        base = os.path.basename(src).replace('.json', '_driftless.json')
        out = os.path.join(MDDIR, base)
        if out not in seen:
            blk = build_driftless(src, out)
            diff = params_identical(src, out)
            assert set(diff) == {'Convexity_Correction'}, f'unexpected md diff {m}: {diff}'
            seen[out] = blk
            print(f'BUILT {base}: params identical except {diff}')
        block = seen[out]
        rng = np.random.default_rng(SEED)
        off = drift_annual(block, convexity=False, rng=rng)   # ARM A world
        rng = np.random.default_rng(SEED)                      # same innovations => paired
        on = drift_annual(block, convexity=True, rng=rng)      # ARM B world
        rows.append(dict(month=m, md=base, lr_vol=round(off['lr_vol'], 4),
                         A_price_drift_pct_yr=round(off['E_dS_over_S_ann'] * 100, 3),
                         B_price_drift_pct_yr=round(on['E_dS_over_S_ann'] * 100, 3),
                         A_logdrift_pct_yr=round(off['E_dlogS_ann'] * 100, 3),
                         B_logdrift_pct_yr=round(on['E_dlogS_ann'] * 100, 3)))
    df = pd.DataFrame(rows)
    out_csv = os.path.join(WF, 'driftless_drift_diag.csv')
    df.to_csv(out_csv, index=False)
    pd.set_option('display.width', 200)
    print('\n=== PRICE-SPACE DRIFT DIAGNOSTIC (annualized E[dS/S], %/yr; paired innovations) ===')
    print(df.to_string(index=False))
    print(f'\nARM A (convexity OFF) mean price drift {df.A_price_drift_pct_yr.mean():+.3f}%/yr  '
          f'| ARM B (ON) mean {df.B_price_drift_pct_yr.mean():+.3f}%/yr')
    print('WROTE', out_csv, 'and', len(seen), 'driftless md files to', MDDIR)


if __name__ == '__main__':
    main()
