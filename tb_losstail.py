"""Loss-tail forensics across BOTH worlds (throwaway, tb_ prefix, no commit).

Exact per-month decomposition of realized hedge P&L against the causal delta-ramp benchmark,
using the realized CME forward strip rebuilt exactly as pf_bound does. No riskflow internals
are touched: everything is read back from the committed run artifacts (diag_*.json, obs_*.npz)
and the corrected archive.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, riskflow.__file__

from production_walk_forward import (build_corrected_archive, CME_COL, LME_COL, BASIS_COL,
                                     CARRY_COL, SOFR_PREFIX, CARRY_TENORS)

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = f'{ROOT}/artifacts/walk_forward'
VOL = 2500.0
CS = 50.0

WORLDS = {
    'garch': ([(f'{WF}/garch48_gpu0', '2020-01'), (f'{WF}/garch48_gpu1', '2022-01')], '_garch'),
    'hmm':   ([(f'{WF}/full48_gpu0', '2020-01'), (f'{WF}/full48_gpu1', '2022-01')], ''),
}


def month_list(start, n):
    return [(pd.Timestamp(start + '-01') + pd.offsets.MonthBegin(m)).strftime('%Y%m') for m in range(n)]


def trade_date_of(m):
    return (pd.Timestamp(f'{m[:4]}-{m[4:]}-01') + pd.offsets.BDay(0)).normalize()


def fixings_of(td):
    a = (td + pd.offsets.MonthBegin(3)).normalize()
    return pd.bdate_range(a, (a + pd.offsets.MonthEnd(0)).normalize())


def forward_strip(arch, td, dates):
    """F_i(t) on the realized daily calendar grid `dates`, same construction as pf_bound."""
    row = arch.loc[:td].iloc[-1]
    taus = [float(row[f'Tenor.{t}']) for t in CARRY_TENORS]
    mats = [td + pd.Timedelta(days=round(t * 360)) for t in taus]
    sub = arch.reindex(arch.index.union(dates)).ffill().loc[dates]
    sofr = sorted((float(c.split(',')[1]), c) for c in arch.columns if c.startswith(SOFR_PREFIX))
    F = []
    for mat in mats:
        tau_t = np.array([(mat - d).days for d in dates]) / 365.25
        ct = np.array([np.interp(tt, [sub[f'Tenor.PLATINUM_TAU{j}'].iloc[k] for j in (1, 2, 3)],
                                 [sub[f'{CARRY_COL},PLATINUM_TAU{j}'].iloc[k] for j in (1, 2, 3)])
                       for k, tt in enumerate(tau_t)])
        rt = np.array([np.interp(tt, [t for t, _ in sofr], [sub[c].iloc[k] for _, c in sofr])
                       for k, tt in enumerate(tau_t)])
        F.append(sub[CME_COL].to_numpy() * np.exp((ct + rt) * np.clip(tau_t, 0, None)))
    return np.array(F), mats


def analyse(world, arch, rows):
    lanes, sfx = WORLDS[world]
    out = []
    for src, start in lanes:
        for m in month_list(start, 24):
            dg = f'{src}/diag_{m}{sfx}.json'
            ob = f'{src}/obs_{m}{sfx}.npz'
            if not (os.path.exists(dg) and os.path.exists(ob)):
                print(f'  MISSING {world} {m}')
                continue
            d = json.load(open(dg))
            sv = d.get('stepper_verdict') or {}
            q = np.array(sv.get('greedy_q_traj') or [])
            t = np.array(sv.get('greedy_q_t') or [])
            if q.ndim != 2:
                print(f'  BAD traj {world} {m}')
                continue
            td = trade_date_of(m)
            fx = fixings_of(td)
            f0 = (fx[0] - td).days
            fN = (fx[-1] - td).days
            dates = pd.DatetimeIndex([td + pd.Timedelta(days=i) for i in range(220)])
            F, mats = forward_strip(arch, td, dates)

            u = q.sum(1) / CS                      # total cover in [-1, 0]
            ramp = np.clip((fN - t) / max(fN - f0, 1), 0.0, 1.0)
            ramp = np.where(t < f0, 1.0, ramp)
            u_ramp = -ramp
            dev = u - u_ramp                        # >0 under-covered, <0 over-short

            # exact realized hedge P&L per oz, and the per-unit-cover effective forward return
            H = 0.0
            dFbar = np.zeros(len(t))
            for i in range(len(t) - 1):
                dF = F[:, t[i + 1]] - F[:, t[i]]
                leg = float(q[i] @ dF) / CS         # $/oz contribution
                H += leg
                s = q[i].sum() / CS
                dFbar[i] = (leg / s) if abs(s) > 1e-9 else float(dF[2])
            H_ramp = float((u_ramp * dFbar).sum())
            gap = float((dev * dFbar).sum())        # == H - H_ramp exactly

            pre = t < f0
            avg = (t >= f0) & (t <= fN)
            post = t > fN
            contrib = dev * dFbar

            b = np.load(ob)['CommodityBasis.LME_CME']
            fx_off = np.array([(f - td).days for f in fx])
            fx_off = fx_off[fx_off < len(b)]

            r = rows[m]
            out.append(dict(
                world=world, tag=m, greedy=r['greedy'], nohedge=r['nohedge'], bound=r['bound'],
                churn=r['churn'], H_exact=round(r['greedy'] - r['nohedge'], 2), H_fwd=round(H, 2),
                recon=round((r['greedy'] - r['nohedge']) - H, 2),
                H_ramp=round(H_ramp, 2), ramp_pnl=round(r['nohedge'] + H_ramp, 2),
                gap_vs_ramp=round(gap, 2),
                gap_pre=round(float(contrib[pre].sum()), 2),
                gap_avg=round(float(contrib[avg].sum()), 2),
                gap_post=round(float(contrib[post].sum()), 2),
                gap_overshort=round(float(contrib[dev < 0].sum()), 2),
                gap_undercover=round(float(contrib[dev > 0].sum()), 2),
                u_post_max=round(float(np.abs(u[post]).max()) if post.any() else 0.0, 3),
                u_post_mean=round(float(np.abs(u[post]).mean()) if post.any() else 0.0, 3),
                u_pre_mean=round(float(np.abs(u[pre]).mean()) if pre.any() else 0.0, 3),
                u_mean=round(float(np.abs(u).mean()), 3),
                dev_mean=round(float(dev.mean()), 3),
                basis0=round(float(b[0]), 2), basis_avg=round(float(b[fx_off].mean()), 2),
                basis_move=round(float(b[fx_off].mean() - b[0]), 2),
                P0=round(float(np.load(ob)[CME_COL][0]), 1),
                dP_pct=round(float(np.load(ob)[CME_COL][fx_off].mean() / np.load(ob)[CME_COL][0] - 1) * 100, 2),
            ))
    return pd.DataFrame(out)


def main():
    raw = pd.read_csv(f'{ROOT}/data/pl_exp.csv', index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    frames = []
    for world, base in [('garch', f'{WF}/garch48_final.csv'), ('hmm', f'{WF}/full48_baseline.csv')]:
        df = pd.read_csv(base)
        df['tag'] = df['tag'].astype(str)
        rows = {r.tag: {'greedy': r.greedy_usd_oz, 'nohedge': r.nohedge_usd_oz,
                        'bound': r.pf_bound, 'churn': r.churn} for r in df.itertuples()}
        print(f'--- {world} ---')
        frames.append(analyse(world, arch, rows))
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(f'{ROOT}/tb_losstail_panel.csv', index=False)
    print('wrote tb_losstail_panel.csv', res.shape)
    print('recon |residual|: mean=%.2f p95=%.2f max=%.2f' % (
        res.recon.abs().mean(), res.recon.abs().quantile(.95), res.recon.abs().max()))


if __name__ == '__main__':
    main()
