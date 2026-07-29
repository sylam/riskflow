"""PHASE B analysis: build tb_results.csv, the per-month comparison table (baseline | roll-clip |
train-in), paired per-month deltas + sign count, subset means/stds, and the verdict."""
import os, sys, json, glob, csv
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
BASE = os.environ['TB_BASE']

MONTHS = ['202105', '202202', '202008', '202106', '202102', '202112', '202011', '202207']

# reference tables
base = pd.read_csv(f'{WF}/full48_baseline.csv', dtype={'tag': str, 'trade': str})
base['month'] = base['trade'].astype(str)
BASELINE = dict(zip(base['month'], base['greedy_usd_oz']))

clip = pd.read_csv(f'{WF}/full48_corridor_rolls.csv', dtype={'tag': str})
clip['month'] = clip['tag'].astype(str)
CLIP = {(r['month'], round(float(r['band']), 2)): r for _, r in clip.iterrows()}

# train-in results
rows = []
for f in sorted(glob.glob(f'{BASE}/tb_row_*.json')):
    rows.append(json.load(open(f)))
TI = {(r['month'], round(float(r['band']), 2)): r for r in rows}


def fmt(x, s='%+.2f'):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else s % x


# ---- tb_results.csv ----
out_csv = os.path.join(BASE, 'tb_results.csv')
with open(out_csv, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['month', 'band', 'mode', 'greedy', 'nohedge', 'bound', 'PASS', 'churn',
                'breaches', 'train_u_s7', 'train_u_s42', 'train_u_s314'])
    for (m, b), r in sorted(TI.items()):
        tu = (r.get('train_u_seeds') or []) + [None, None, None]
        w.writerow([m, b, r['mode'], r['greedy'], r['nohedge'], r['bound'], r['PASS'],
                    r['churn'], r['breaches'], tu[0], tu[1], tu[2]])
print('wrote', out_csv)


def table_for_band(band, months):
    print(f'\n===== COMPARISON @ band {band:.2f}  ($/oz greedy) =====')
    print(f"{'month':>7} | {'baseline':>9} | {'roll-clip':>9} | {'train-in':>9} | "
          f"{'Δ(TI-clip)':>10} | {'PASS':>4} | {'breach':>6} | {'churn TI/clip':>13}")
    b_col, c_col, t_col, d_col = [], [], [], []
    for m in months:
        bl = BASELINE.get(m)
        cl = CLIP.get((m, band))
        ti = TI.get((m, band))
        clg = None if cl is None else float(cl['greedy'])
        tig = None if ti is None else float(ti['greedy'])
        d = None if (clg is None or tig is None) else tig - clg
        b_col.append(bl); c_col.append(clg); t_col.append(tig)
        if d is not None:
            d_col.append(d)
        chn = '' if (ti is None or cl is None) else f"{ti['churn']:.0f}/{float(cl['churn']):.0f}"
        print(f"{m:>7} | {fmt(bl):>9} | {fmt(clg):>9} | {fmt(tig):>9} | {fmt(d):>10} | "
              f"{str(None if ti is None else ti['PASS']):>4} | "
              f"{str(None if ti is None else ti['breaches']):>6} | {chn:>13}")

    def ms(col):
        a = np.array([x for x in col if x is not None], float)
        return (a.mean(), a.std(ddof=0), a.std(ddof=1) if len(a) > 1 else float('nan'), len(a))

    bm = ms(b_col); cm = ms(c_col); tm = ms(t_col)
    print(f"{'MEAN':>7} | {fmt(bm[0]):>9} | {fmt(cm[0]):>9} | {fmt(tm[0]):>9} | "
          f"{fmt(tm[0]-cm[0]):>10} |")
    print(f"{'STD(pop)':>7} | {fmt(bm[1],'%.2f'):>9} | {fmt(cm[1],'%.2f'):>9} | {fmt(tm[1],'%.2f'):>9} |")
    print(f"{'STD(sam)':>7} | {fmt(bm[2],'%.2f'):>9} | {fmt(cm[2],'%.2f'):>9} | {fmt(tm[2],'%.2f'):>9} |")
    # paired deltas
    if d_col:
        d = np.array(d_col, float)
        pos = int((d > 0).sum()); neg = int((d < 0).sum())
        print(f"\nPaired per-month Δ (train-in − roll-clip), n={len(d)}: "
              f"mean {d.mean():+.2f}  median {np.median(d):+.2f}  std(sam) "
              f"{d.std(ddof=1) if len(d) > 1 else float('nan'):.2f}")
        print(f"  sign count: {pos} up / {neg} down / {len(d)-pos-neg} tie | values "
              f"{['%+.1f' % x for x in d]}")
        # crude paired t
        if len(d) > 1 and d.std(ddof=1) > 0:
            tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"  paired t-stat (H0: Δ=0): {tstat:+.2f} (|t|>2.36 ~ p<0.05 at df=7)")
    return cm, tm, d_col


print('\n' + '#' * 70)
print('# BAND 0.40 (all 8 months)')
c40, t40, d40 = table_for_band(0.40, MONTHS)

print('\n' + '#' * 70)
print('# BAND 0.60 (spot-checks: 202105, 202102)')
c60, t60, d60 = table_for_band(0.60, ['202105', '202102'])

# gates
print('\n===== GATES =====')
allpass = all(r['PASS'] for r in TI.values())
allbreach0 = all((r['breaches'] == 0) for r in TI.values())
print(f"bound-PASS all rolls: {allpass}  ({sum(1 for r in TI.values() if r['PASS'])}/{len(TI)})")
print(f"zero corridor breaches all rolls: {allbreach0}  "
      f"(max breaches over rolls = {max((r['breaches'] or 0) for r in TI.values())})")

# verdict
print('\n===== VERDICT (band 0.40 subset, n=8) =====')
dm = np.mean(d40); ds = np.std(d40, ddof=1)
up = sum(1 for x in d40 if x > 0)
print(f"train-in mean {t40[0]:+.2f} (std {t40[2]:.2f}) vs roll-clip mean {c40[0]:+.2f} (std {c40[2]:.2f})")
print(f"paired Δ mean {dm:+.2f}, {up}/{len(d40)} months up, variance "
      f"{'DOWN' if t40[2] < c40[2] else 'UP'} ({t40[2]:.2f} vs {c40[2]:.2f})")
