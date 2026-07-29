"""Final ranking: frontier in both worlds, per-year stability, paired-bootstrap significance,
and the loss-month rescue table. Reads only committed artifacts + the tb_ re-roll rows."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from tb_volrule import load_bands, cv, paired_boot

ROOT = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 250)


def main():
    panel = pd.read_csv(f'{ROOT}/tb_losstail_panel.csv'); panel['tag'] = panel.tag.astype(str)
    for world in ('garch', 'hmm'):
        b = load_bands(world)
        avail = sorted([k for k in b if k != 'free'])
        free = b['free']
        print(f'\n{"=" * 118}\n{world.upper()}: FRONTIER  (n=48, eval-only re-roll on corridor-free checkpoints)\n{"=" * 118}')
        print(f'{"band":>6} {"mean":>7} {"giveback":>9} {"std":>7} {"CVaR5":>8} {"min":>7} '
              f'{"m/std":>6} {"m/|CVaR|":>9} {"neg":>4} {"P(std<free)":>11} {"P(CVaR>free)":>12}')
        x0 = free.to_numpy()
        print(f'{"free":>6} {x0.mean():>+7.2f} {"-":>9} {x0.std(ddof=1):>7.2f} {cv(x0):>+8.2f} '
              f'{x0.min():>+7.1f} {x0.mean() / x0.std(ddof=1):>6.3f} {x0.mean() / abs(cv(x0)):>9.3f} '
              f'{int((x0 < 0).sum()):>4} {"-":>11} {"-":>12}')
        for k in avail:
            x = b[k].loc[free.index].to_numpy()
            pb = paired_boot(x, x0)
            print(f'{k:>6.2f} {x.mean():>+7.2f} {x.mean() - x0.mean():>+9.2f} {x.std(ddof=1):>7.2f} '
                  f'{cv(x):>+8.2f} {x.min():>+7.1f} {x.mean() / x.std(ddof=1):>6.3f} '
                  f'{x.mean() / abs(cv(x)):>9.3f} {int((x < 0).sum()):>4} '
                  f'{pb["std_lower"]:>11.2f} {pb["cvar_higher"]:>12.2f}')

        tight = min(avail)
        print(f'\n  --- band {tight} vs band 0.60 (paired bootstrap, 20k) ---')
        pb = paired_boot(b[tight].loc[free.index].to_numpy(), b[0.6].loc[free.index].to_numpy())
        print(f'   P(mean higher)={pb["mean"]:.2f}  P(std lower)={pb["std_lower"]:.2f}  '
              f'P(CVaR better)={pb["cvar_higher"]:.2f}  P(mean/std better)={pb["ratio"]:.2f}')

        print(f'\n  --- per-year stability (mean / std) ---')
        yrs = sorted({t[:4] for t in free.index})
        hdr = ''.join(f'{y:>18}' for y in yrs)
        print(f'   {"band":>6}{hdr}')
        for k in ['free'] + avail:
            s = b[k].loc[free.index]
            cells = ''
            for y in yrs:
                v = s[[t for t in s.index if t.startswith(y)]].to_numpy()
                cells += f'{v.mean():>+9.1f}/{v.std(ddof=1):<8.1f}'
            print(f'   {str(k):>6}{cells}')

        print(f'\n  --- the free-world loss tail, rescued by the tight band ---')
        lo = free.nsmallest(8)
        p = panel[panel.world == world].set_index('tag')
        t = pd.DataFrame({'free': lo, f'b{tight}': b[tight].loc[lo.index],
                          'b0.6': b[0.6].loc[lo.index],
                          'ramp_recon': p.ramp_pnl.loc[lo.index],
                          'nohedge': p.nohedge.loc[lo.index],
                          'u_pre_mean': p.u_pre_mean.loc[lo.index],
                          'u_post_max': p.u_post_max.loc[lo.index],
                          'dP_pct': p.dP_pct.loc[lo.index]})
        t['rescue'] = (t[f'b{tight}'] - t.free).round(2)
        print(t.round(2).to_string())


if __name__ == '__main__':
    main()
