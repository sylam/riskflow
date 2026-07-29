"""Loss taxonomy for the union of each world's worst months + CVaR attribution."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 250)


def cv(x, q=.05):
    x = np.sort(np.asarray(x, float))
    return x[:int(np.ceil(q * len(x)))].mean()


def classify(r):
    """Mechanism from the exact-sign/phase split of the gap vs the causal delta ramp."""
    tags = []
    if r.u_post_max > 0.15:
        tags.append('OVER-HOLD(post-fixing)')
    if r.gap_overshort < -15:
        tags.append('over-short')
    if r.gap_undercover < -15:
        tags.append('under-cover')
    if abs(r.basis_move) > 12:
        tags.append('basis')
    if not tags:
        tags.append('market (little deviation)')
    return '+'.join(tags)


def main():
    p = pd.read_csv(f'{ROOT}/tb_losstail_panel.csv')
    p['tag'] = p['tag'].astype(str)
    p['vs_nohedge'] = (p.greedy - p.nohedge).round(2)
    p['vs_ramp'] = (p.greedy - p.ramp_pnl).round(2)
    p['mech'] = p.apply(classify, axis=1)

    worst = {}
    for w in ('garch', 'hmm'):
        s = p[p.world == w].nsmallest(8, 'greedy')
        worst[w] = set(s.tag)
        print(f'\n=== {w.upper()} worst 8 ===')
        print(s[['tag', 'greedy', 'nohedge', 'vs_nohedge', 'ramp_pnl', 'vs_ramp', 'gap_pre',
                 'gap_avg', 'gap_post', 'gap_overshort', 'gap_undercover', 'u_post_max',
                 'basis_move', 'dP_pct', 'churn', 'mech']].to_string(index=False))

    union = sorted(worst['garch'] | worst['hmm'])
    print(f'\n=== UNION of worst-8 across worlds: {len(union)} months ===')
    print(' '.join(union))
    sub = p[p.tag.isin(union)].sort_values(['tag', 'world'])
    print(sub[['world', 'tag', 'greedy', 'nohedge', 'vs_nohedge', 'ramp_pnl', 'vs_ramp',
               'gap_post', 'u_post_max', 'u_pre_mean', 'basis_move', 'dP_pct', 'mech']].to_string(index=False))

    print('\n=== POST-LAST-FIXING NAKED EXPOSURE (all 48, per world) ===')
    for w in ('garch', 'hmm'):
        s = p[p.world == w]
        print(f'{w}: months with |u| post-last-fixing > 0.15: {int((s.u_post_max > .15).sum())}/48 '
              f'| mean gap_post = {s.gap_post.mean():+.2f} $/oz | sum {s.gap_post.sum():+.1f} '
              f'| gap_post on the 3 CVaR months = {s.nsmallest(3, "greedy").gap_post.mean():+.2f}')
        bad = s[s.u_post_max > .15]
        print(f'   those months mean P&L {bad.greedy.mean():+.2f} vs rest {s[s.u_post_max <= .15].greedy.mean():+.2f}'
              f' | their mean gap_post {bad.gap_post.mean():+.2f}')

    print('\n=== BUCKET SHARE OF THE 48-TRADE LOSS TAIL (worst 3 = CVaR5 cohort) ===')
    for w in ('garch', 'hmm'):
        s = p[p.world == w]
        t = s.nsmallest(3, 'greedy')
        print(f'{w}: CVaR5={cv(s.greedy):+.2f}  cohort={list(t.tag)}')
        print(f'   mean nohedge on cohort {t.nohedge.mean():+.2f} | mean ramp {t.ramp_pnl.mean():+.2f} '
              f'| mean greedy {t.greedy.mean():+.2f} | mean gap_vs_ramp {t.vs_ramp.mean():+.2f}')
        print(f'   of which pre {t.gap_pre.mean():+.2f} / avg-window {t.gap_avg.mean():+.2f} / post {t.gap_post.mean():+.2f}'
              f' ; over-short {t.gap_overshort.mean():+.2f} / under-cover {t.gap_undercover.mean():+.2f}')

    print('\n=== AVOIDABLE vs UNAVOIDABLE ===')
    for w in ('garch', 'hmm'):
        s = p[p.world == w]
        lost = s[s.greedy < 0]
        own = s[s.greedy < s.nohedge]
        print(f'{w}: {len(lost)}/48 losing months (sum {lost.greedy.sum():+.1f}); '
              f'{len(own)}/48 WORSE THAN NOHEDGE (own-goals): {sorted(own.tag)}')
        print(f'   own-goal months: mean greedy {own.greedy.mean():+.2f} vs nohedge {own.nohedge.mean():+.2f} '
              f'vs ramp {own.ramp_pnl.mean():+.2f} | mean u_post_max {own.u_post_max.mean():.2f} '
              f'| mean gap_post {own.gap_post.mean():+.2f}')

    print('\n=== RAMP BENCHMARK OVERALL (reconstructed, corr .99 w/ framework) ===')
    for w in ('garch', 'hmm'):
        s = p[p.world == w]
        for nm, col in [('greedy', s.greedy), ('ramp(recon)', s.ramp_pnl), ('nohedge', s.nohedge)]:
            x = col.to_numpy()
            print(f'  {w} {nm:12s} mean={x.mean():+7.2f} std={x.std(ddof=1):6.2f} CVaR5={cv(x):+7.2f} m/s={x.mean()/x.std(ddof=1):.3f}')

    p.to_csv(f'{ROOT}/tb_losstail_panel.csv', index=False)


if __name__ == '__main__':
    main()
