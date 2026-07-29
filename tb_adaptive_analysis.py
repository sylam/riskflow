"""ADAPTIVE-CORRIDOR selector analysis (pure pandas on the sweep + the realized menu).

1. Aggregates the per-(month,fence,seed) sidecars into adaptive_corridor_sweep.csv.
2. Noise: pooled across the 8 spot months x 3 fence-pairs, the SD across eval seeds of the
   paired u-margin — the a-priori deviation threshold for the guarded selector (1 sigma; the
   selector consumes ONE seed's margin, so no /sqrt(n)).
3. RAW selector: per month argmax_fence u_mean (seed 101) -> that fence's REALIZED roll.
   GUARDED: default band 0.60, deviate only when the u-margin over 0.60 exceeds the threshold.
4. Reports portfolio mean/std/min/p10 vs fixed 0.25/0.40/0.60/free and the oracle, hit rate vs
   oracle, error taxonomy, u-margin vs realized-margin rank correlation, and the verdict vs the
   success bar (mean > +23.10 AND std < 60.4). Writes adaptive_corridor_verdict.md.
"""
import os, sys, json, glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
SP = os.environ['TB_SP']                      # scratchpad dir
SWEEP_DIR = os.path.join(SP, 'adaptive_sweep')
FENCES = ['b040', 'b060', 'free']
PRIMARY_SEED = 101
SPOT = ['202008', '202011', '202102', '202105', '202202', '202207', '202301', '202312']
BAR_MEAN, BAR_STD = 23.10, 60.4


def portfolio(s):
    return {'mean': s.mean(), 'std': s.std(ddof=1), 'min': s.min(), 'p10': s.quantile(0.10)}


def main():
    # ---- 1) aggregate sidecars -> sweep CSV --------------------------------------------------
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(SWEEP_DIR, 'res_*.json')))]
    sw = pd.DataFrame(rows)[['month', 'fence', 'band', 'seed', 'u_mean', 'ew', 'p5', 'cvar5']]
    sw = sw.sort_values(['month', 'fence', 'seed']).reset_index(drop=True)
    sweep_csv = os.path.join(SP, 'adaptive_corridor_sweep.csv')
    sw.to_csv(sweep_csv, index=False)
    print(f'wrote {sweep_csv} ({len(sw)} rows)')

    prim = (sw[sw.seed == PRIMARY_SEED].pivot(index='month', columns='fence', values='u_mean')
            [FENCES])
    assert prim.notna().all().all() and len(prim) == 48, 'incomplete primary sweep'

    # ---- realized menu ------------------------------------------------------------------------
    base = pd.read_csv(os.path.join(WF, 'full48_baseline.csv'))
    cr = pd.read_csv(os.path.join(WF, 'full48_corridor_rolls.csv'))
    base['month'] = base['trade'].astype(str)
    cr['month'] = cr['tag'].astype(str)
    menu = pd.DataFrame({
        'free': base.set_index('month')['greedy_usd_oz'],
        'b040': cr[cr.band == 0.40].set_index('month')['greedy'],
        'b060': cr[cr.band == 0.60].set_index('month')['greedy'],
        'b025': cr[cr.band == 0.25].set_index('month')['greedy']}).loc[prim.index]
    assert menu.notna().all().all()

    # ---- 2) noise estimate (spot months, 3 eval seeds) ----------------------------------------
    noise = sw[sw.month.isin(SPOT)].pivot_table(index=['month', 'seed'], columns='fence',
                                                values='u_mean')
    margins = pd.DataFrame({'b040-b060': noise['b040'] - noise['b060'],
                            'free-b060': noise['free'] - noise['b060'],
                            'free-b040': noise['free'] - noise['b040']})
    # SD across seeds within (month, pair), pooled
    sd_by = margins.groupby('month').std(ddof=1)
    sigma = float(np.sqrt((sd_by ** 2).mean().mean()))
    n_seeds = noise.groupby('month').size()
    print(f'\nNOISE (paired u-margins, {len(sd_by)} spot months x 3 pairs, '
          f'{int(n_seeds.iloc[0])} eval seeds): pooled SD = {sigma:.5f}')
    print('per-month margin SDs:\n', sd_by.round(5).to_string())

    # ---- 3) selectors --------------------------------------------------------------------------
    def realized_of(picks):
        return pd.Series([menu.loc[m, p] for m, p in picks.items()], index=picks.index)

    raw_pick = prim.idxmax(axis=1)
    raw_real = realized_of(raw_pick)

    def guarded(th):
        picks = {}
        for m in prim.index:
            u = prim.loc[m]
            best = u.idxmax()
            picks[m] = best if (best != 'b060' and u[best] - u['b060'] > th) else 'b060'
        return pd.Series(picks)

    g_pick = guarded(sigma)                                   # a-priori 1-sigma headline
    g_real = realized_of(g_pick)

    oracle3 = menu[FENCES].max(axis=1)
    oracle_pick = menu[FENCES].idxmax(axis=1)

    # ---- report --------------------------------------------------------------------------------
    tbl = pd.DataFrame({
        'fixed 0.25': portfolio(menu['b025']), 'fixed 0.40': portfolio(menu['b040']),
        'fixed 0.60': portfolio(menu['b060']), 'free': portfolio(menu['free']),
        'RAW selector': portfolio(raw_real), f'GUARDED (1sig={sigma:.4f})': portfolio(g_real),
        'oracle {040,060,free}': portfolio(oracle3)}).T.round(2)

    sens = {}
    for k, th in [('0.0 (=RAW w/ 060 ties)', 0.0), ('0.5 sigma', 0.5 * sigma),
                  ('1.0 sigma', sigma), ('2.0 sigma', 2 * sigma),
                  ('inf (=fixed 0.60)', np.inf)]:
        r = realized_of(guarded(th))
        p = portfolio(r)
        ndev = int((guarded(th) != 'b060').sum())
        sens[k] = {**p, 'n_deviations': ndev}
    sens = pd.DataFrame(sens).T.round(2)

    hit_raw = float((raw_pick == oracle_pick).mean())
    hit_g = float((g_pick == oracle_pick).mean())

    # error taxonomy: months where the deviation from b060 lost money vs staying
    def errors(picks, real):
        e = []
        for m in prim.index:
            if picks[m] != 'b060' and real[m] < menu.loc[m, 'b060']:
                e.append((m, picks[m], round(real[m] - menu.loc[m, 'b060'], 1)))
        return e

    # causal-skill diagnostic: rank corr of u-margin vs realized margin (free vs b060, b040 vs b060)
    skl = {}
    for a, b in (('free', 'b060'), ('b040', 'b060'), ('free', 'b040')):
        um = prim[a] - prim[b]
        rm = menu[a] - menu[b]
        skl[f'{a} vs {b}'] = {'spearman': round(um.corr(rm, method='spearman'), 3),
                              'pearson': round(um.corr(rm), 3)}
    skl = pd.DataFrame(skl).T

    per_month = pd.DataFrame({
        'u_b040': prim['b040'].round(4), 'u_b060': prim['b060'].round(4),
        'u_free': prim['free'].round(4),
        'raw_pick': raw_pick, 'guarded_pick': g_pick, 'oracle_pick': oracle_pick,
        'real_b040': menu['b040'], 'real_b060': menu['b060'], 'real_free': menu['free'],
        'raw_real': raw_real.round(1), 'guarded_real': g_real.round(1)})

    raw_p, g_p = portfolio(raw_real), portfolio(g_real)
    def bar(p):
        ok = p['mean'] > BAR_MEAN and p['std'] < BAR_STD
        return f"mean {p['mean']:+.2f} (> {BAR_MEAN:+.2f}? {p['mean'] > BAR_MEAN}) and std {p['std']:.1f} (< {BAR_STD:.1f}? {p['std'] < BAR_STD}) -> {'PASS' if ok else 'FAIL'}"

    md = []
    md.append('# Adaptive corridor — eval sweep + selector verdict\n')
    md.append(f'48 months x 3 fences (band 0.40 / 0.60 / free), eval-only ensemble verdicts '
              f'(frozen corridor-free full48 checkpoints, seeds 7/42/314), batch 2048, common '
              f'random numbers per month (eval seed {PRIMARY_SEED}; fences differ only by '
              f'Evaluator.Total_Position_Schedule). Selector consumes fenced in-model u_mean — '
              f'fully causal at trade date. Realized outcomes from full48_baseline.csv + '
              f'full48_corridor_rolls.csv.\n')
    md.append(f'## Noise\nPaired u-margin SD across eval seeds (101/202/303), pooled over 8 spot '
              f'months x 3 fence pairs: **sigma = {sigma:.5f}** (a-priori guarded threshold = 1 '
              f'sigma, chosen before looking at realized outcomes).\n')
    md.append('## Portfolio comparison (realized $/oz, 48 trades)\n')
    md.append(tbl.to_markdown() + '\n')
    md.append(f'## Success bar (mean > {BAR_MEAN:+.2f} AND std < {BAR_STD:.1f})\n'
              f'- RAW: {bar(raw_p)}\n- GUARDED (1 sigma): {bar(g_p)}\n')
    md.append('## Guarded-threshold sensitivity (honest; headline is the a-priori 1 sigma)\n')
    md.append(sens.to_markdown() + '\n')
    md.append(f'## Hit rate vs oracle picks\n- RAW: {hit_raw:.1%}\n- GUARDED: {hit_g:.1%}\n')
    md.append('## Causal-skill diagnostic (u-margin vs realized-margin correlation, 48 months)\n')
    md.append(skl.to_markdown() + '\n')
    md.append('## Deviations that LOST vs staying at 0.60\n')
    md.append(f'- RAW: {errors(raw_pick, raw_real)}\n- GUARDED: {errors(g_pick, g_real)}\n')
    md.append('## Per-month table\n')
    md.append(per_month.to_markdown() + '\n')
    out_md = os.path.join(SP, 'adaptive_corridor_verdict.md')
    open(out_md, 'w').write('\n'.join(md))
    print(f'\nwrote {out_md}')
    print('\n' + tbl.to_string())
    print(f'\nRAW: {bar(raw_p)}\nGUARDED: {bar(g_p)}')
    print(f'hit rate raw={hit_raw:.1%} guarded={hit_g:.1%}')
    print(skl.to_string())
    print(sens.to_string())


if __name__ == '__main__':
    main()
