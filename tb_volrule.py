"""Step 2+3: (a) is the trade-date vol state predictive of loss? (b) LOO'd vol-conditioned band
rule vs fixed bands. (c) priced structural levers (post-fixing flat rule, basis).

Every rule that touches the 48 months is LEAVE-ONE-OUT: the threshold AND the band mapping are
re-fit on the other 47 months for each evaluated month. Fixed-band comparators need no fitting.
"""
import os, sys, glob, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = f'{ROOT}/artifacts/walk_forward'
RNG = np.random.default_rng(20260720)


def cv(x, q=.05):
    x = np.sort(np.asarray(x, float))
    return x[:int(np.ceil(q * len(x)))].mean()


def stat(name, x, base=None):
    x = np.asarray(x, float)
    m, s, c = x.mean(), x.std(ddof=1), cv(x)
    gb = '' if base is None else f' giveback={m - np.mean(base):+6.2f}'
    return (f'{name:34s} mean={m:+7.2f} std={s:6.2f} CVaR5={c:+7.2f} min={x.min():+7.1f} '
            f'm/std={m / s:5.3f} m/|CVaR|={m / abs(c):5.3f}{gb}')


def load_bands(world):
    """band -> Series(tag -> greedy), from the committed CSVs plus the tb_ re-roll rows."""
    out = {}
    if world == 'garch':
        base = pd.read_csv(f'{WF}/garch48_final.csv')
        base['tag'] = base.tag.astype(str)
        out['free'] = base.set_index('tag').greedy_usd_oz
        rows = [json.load(open(p)) for p in glob.glob(f'{WF}/garch48_corridor/tb_row_*.json')]
    else:
        base = pd.read_csv(f'{WF}/full48_baseline.csv')
        base['tag'] = base.tag.astype(str)
        out['free'] = base.set_index('tag').greedy_usd_oz
        c = pd.read_csv(f'{WF}/full48_corridor_rolls.csv')
        c['tag'] = c.tag.astype(str)
        for b, g in c.groupby('band'):
            out[round(float(b), 2)] = g.set_index('tag').greedy
        rows = [json.load(open(p)) for p in glob.glob(f'{WF}/hmm48_corridor/tb_row_*.json')]
    if rows:
        d = pd.DataFrame(rows)
        d['tag'] = d.tag.astype(str)
        for b, g in d.groupby('band'):
            b = round(float(b), 2)
            if len(g) == 48:
                out[b] = g.set_index('tag').greedy
    return out


def loo_rule(vol, bands, band_choices, feature):
    """LOO'd 2-band threshold rule: tighten when feature is high.
    For each held-out month, the threshold (a quantile of the feature) and the (tight, loose)
    band assignment are chosen on the OTHER 47 months only, maximising mean/std there."""
    tags = list(vol.index)
    qgrid = np.arange(0.30, 0.91, 0.05)
    picked, sel = [], []
    for i, t in enumerate(tags):
        tr = [u for u in tags if u != t]
        best, bq, bpair = -np.inf, None, None
        for q in qgrid:
            thr = vol.loc[tr, feature].quantile(q)
            hi = vol.loc[tr, feature] > thr
            for tight, loose in itertools.permutations(band_choices, 2):
                x = np.where(hi, bands[tight].loc[tr], bands[loose].loc[tr])
                s = x.std(ddof=1)
                sc = x.mean() / s if s > 0 else -np.inf
                if sc > best:
                    best, bq, bpair = sc, (thr, q), (tight, loose)
        thr = bq[0]
        b = bpair[0] if vol.loc[t, feature] > thr else bpair[1]
        picked.append(bands[b].loc[t])
        sel.append(dict(tag=t, band=b, thr=round(float(thr), 4), q=round(float(bq[1]), 2),
                        pair=str(bpair), feat=round(float(vol.loc[t, feature]), 4)))
    return np.array(picked), pd.DataFrame(sel)


def paired_boot(a, b, n=20000):
    """P(stat(a) > stat(b)) under a paired month bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = RNG.integers(0, len(a), size=(n, len(a)))
    A, B = a[idx], b[idx]
    return (dict(mean=float((A.mean(1) > B.mean(1)).mean()),
                 std_lower=float((A.std(1, ddof=1) < B.std(1, ddof=1)).mean()),
                 cvar_higher=float((np.sort(A, 1)[:, :3].mean(1) > np.sort(B, 1)[:, :3].mean(1)).mean()),
                 ratio=float(((A.mean(1) / A.std(1, ddof=1)) > (B.mean(1) / B.std(1, ddof=1))).mean())))


def main():
    vol = pd.read_csv(f'{ROOT}/tb_volstate.csv')
    vol['tag'] = vol.tag.astype(str)
    vol = vol.set_index('tag')
    panel = pd.read_csv(f'{ROOT}/tb_losstail_panel.csv')
    panel['tag'] = panel.tag.astype(str)

    for world in ('garch', 'hmm'):
        bands = load_bands(world)
        avail = sorted([b for b in bands if b != 'free'])
        print(f'\n{"=" * 100}\n{world.upper()} WORLD — bands available: {avail}\n{"=" * 100}')
        p = panel[panel.world == world].set_index('tag')
        vv = vol.loc[bands['free'].index]

        # ---------- 2a. is trade-date vol state predictive? ----------
        print('\n--- 2a. Spearman(trade-date vol state, realized P&L) — external observable ---')
        for f in ['h_trade', 'h_ratio', 'sigma_trade', 'rv20', 'rv60', 'h_ratio_logged']:
            for tgt, lbl in [(bands['free'], 'free'), (bands[max(avail)], f'b{max(avail)}')]:
                r = stats.spearmanr(vv[f], tgt.loc[vv.index])
                print(f'   {f:16s} vs {lbl:8s} rho={r.statistic:+.3f} p={r.pvalue:.3f}', end='')
            print()
        print('   (loss MAGNITUDE, losing months only)')
        for f in ['h_ratio', 'rv20']:
            g = bands['free']
            lo = g[g < 0]
            r = stats.spearmanr(vv.loc[lo.index, f], -lo)
            print(f'   {f:16s} vs |loss| n={len(lo)} rho={r.statistic:+.3f} p={r.pvalue:.3f}')

        print('\n--- 2b. Tercile table on h_ratio (trade-date filtered GARCH state) ---')
        for f in ['h_ratio', 'rv20']:
            vv2 = vv.copy()
            vv2['ter'] = pd.qcut(vv2[f], 3, labels=['low', 'mid', 'high'])
            print(f'   feature={f}')
            for t, g in vv2.groupby('ter', observed=True):
                x = bands['free'].loc[g.index]
                y = bands[max(avail)].loc[g.index]
                print(f'     {t:5s} n={len(g):2d} {f}∈[{g[f].min():.2f},{g[f].max():.2f}] '
                      f'| free mean={x.mean():+7.2f} std={x.std(ddof=1):6.2f} min={x.min():+7.1f} '
                      f'| b{max(avail)} mean={y.mean():+7.2f} std={y.std(ddof=1):6.2f}')

        # ---------- 2c. LOO'd vol-conditioned band rule ----------
        print('\n--- 2c. LOO vol-conditioned band rule vs FIXED bands ---')
        free = bands['free'].to_numpy()
        print('  ' + stat('fixed: corridor-free', free))
        for b in avail:
            print('  ' + stat(f'fixed: band {b}', bands[b].to_numpy(), free))
        best_fixed_b = max(avail, key=lambda b: bands[b].mean() / bands[b].std(ddof=1))
        best_fixed = bands[best_fixed_b].to_numpy()
        print(f'  -> best fixed by mean/std: band {best_fixed_b}')
        for feature in ['h_ratio', 'rv20', 'sigma_trade']:
            for choices in [(min(avail), max(avail)), (0.25, 0.60) if 0.25 in avail else (min(avail), max(avail))]:
                choices = tuple(sorted(set(choices)))
                if len(choices) < 2:
                    continue
                x, sel = loo_rule(vv, bands, list(choices), feature)
                print('  ' + stat(f'LOO {feature} -> {choices}', x, best_fixed))
                pb = paired_boot(x, best_fixed)
                print(f'      vs fixed b{best_fixed_b}: P(mean higher)={pb["mean"]:.2f} '
                      f'P(std lower)={pb["std_lower"]:.2f} P(CVaR better)={pb["cvar_higher"]:.2f} '
                      f'P(mean/std better)={pb["ratio"]:.2f} | tight picked {int((sel.band == min(choices)).sum())}/48')

        # ---------- 3c. structural lever: flat after last fixing ----------
        print('\n--- 3c. STRUCTURAL: force flat after the last fixing (causal, unfitted) ---')
        gp = p.gap_post.reindex(bands['free'].index).fillna(0).to_numpy()
        print('  ' + stat('free', free))
        print('  ' + stat('free MINUS post-fixing P&L', free - gp, free))
        print(f'      post-fixing P&L: mean={gp.mean():+.2f} std={gp.std(ddof=1):.2f} '
              f'min={gp.min():+.1f} max={gp.max():+.1f} | months with any={int((np.abs(gp) > 1e-6).sum())}/48')
        pb = paired_boot(free - gp, free)
        print(f'      P(mean higher)={pb["mean"]:.2f} P(std lower)={pb["std_lower"]:.2f} '
              f'P(CVaR better)={pb["cvar_higher"]:.2f} P(mean/std better)={pb["ratio"]:.2f}')
        for b in avail:
            gpb = p.gap_post.reindex(bands[b].index).fillna(0).to_numpy()  # NOTE: post from the FREE roll
            _ = gpb

        # ---------- 3b. basis ----------
        print('\n--- 3b. BASIS contribution ---')
        bm = p.basis_move.reindex(bands['free'].index).to_numpy()
        r = stats.spearmanr(bm, free)
        print(f'   basis_move (avg-window basis - trade-date basis): mean={bm.mean():+.2f} '
              f'std={bm.std(ddof=1):.2f} range=[{bm.min():+.1f},{bm.max():+.1f}]')
        print(f'   Spearman(basis_move, free P&L) rho={r.statistic:+.3f} p={r.pvalue:.3f}')
        lo3 = bands['free'].nsmallest(3).index
        print(f'   basis_move on the CVaR5 cohort {list(lo3)}: {np.round(p.basis_move.loc[lo3].to_numpy(), 1)} '
              f'(mean {p.basis_move.loc[lo3].mean():+.1f} vs all-48 {bm.mean():+.1f})')


if __name__ == '__main__':
    main()
