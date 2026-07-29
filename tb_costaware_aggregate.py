"""Paired ARM A (cost-aware retrain) vs ARM B (frozen cost-blind, cost-aware roll) aggregator.

Reads ARM A tb_rows from the two retrain lane dirs and ARM B from the existing net_of_cost
base/band-015 rows; both greedy P&Ls are net of the SAME realistic BASE cost at band 0.15.
Writes artifacts/walk_forward/costaware_subset.csv and prints the paired report. n=8 is a
SCOPING sample, not a verdict-grade estimate.
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
NOC = os.path.join(WF, 'net_of_cost')

LOSERS = {'202105', '202202', '202106', '202001', '202204'}
DEFAULT_MONTHS = ['202105', '202202', '202106', '202001', '202204', '202011', '202102', '202112']


def load_arm_a(month, lane_dirs):
    for d in lane_dirs:
        p = os.path.join(d, f'tb_row_costaware_{month}.json')
        if os.path.exists(p):
            return json.load(open(p))
    return None


def load_arm_b(month):
    p = os.path.join(NOC, f'tb_row_base_015_{month}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def stats(x):
    x = np.asarray(x, float)
    p5 = np.quantile(x, 0.05)
    cvar5 = float(x[x <= p5].mean()) if (x <= p5).any() else float(p5)
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)), min=float(x.min()),
                max=float(x.max()), cvar5=cvar5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', nargs='+', default=DEFAULT_MONTHS)
    ap.add_argument('--lanes', nargs='+',
                    default=[os.path.join(WF, 'costaware_retrain_gpu0'),
                             os.path.join(WF, 'costaware_retrain_gpu1')])
    ap.add_argument('--out', default=os.path.join(WF, 'costaware_subset.csv'))
    args = ap.parse_args()

    rows, missing = [], []
    for m in args.months:
        a, b = load_arm_a(m, args.lanes), load_arm_b(m)
        if a is None or b is None:
            missing.append((m, a is None, b is None))
            continue
        rows.append({
            'tag': m, 'is_loser': m in LOSERS,
            'A_greedy': a['greedy'], 'B_greedy': b['greedy'],
            'delta': round(a['greedy'] - b['greedy'], 4),
            'A_nohedge': a['nohedge'], 'B_nohedge': b['nohedge'],
            'A_pass': a['pass'], 'A_breaches': a['breaches'], 'A_churn': a['churn'],
            'B_churn': b['churn'], 'A_bound': a['bound'],
            'A_train_u_seeds': a.get('train_u_seeds'), 'md': a['md']})
    if missing:
        print('MISSING (month, A_missing, B_missing):', missing)
    if not rows:
        print('no complete pairs yet'); return

    df = pd.DataFrame(rows).sort_values('A_greedy').reset_index(drop=True)
    df.to_csv(args.out, index=False)

    A, B, D = df['A_greedy'].values, df['B_greedy'].values, df['delta'].values
    n = len(df)
    sa, sb = stats(A), stats(B)
    md, sd = float(D.mean()), float(D.std(ddof=1))
    se = sd / np.sqrt(n)
    t = md / se if se > 0 else float('nan')
    n_pass = int(df['A_pass'].sum())
    n_breach = int((df['A_breaches'].fillna(0) > 0).sum())
    tot_breach = int(df['A_breaches'].fillna(0).sum())

    pd.set_option('display.width', 200)
    print('\n================ COST-AWARE RETRAIN SUBSET (band 0.15, net of realistic BASE cost) ================')
    print('n =', n, '(SCOPING sample, not a verdict-grade estimate)\n')
    print(df[['tag', 'is_loser', 'A_greedy', 'B_greedy', 'delta', 'A_pass', 'A_breaches',
              'A_churn', 'B_churn']].to_string(index=False))
    print('\n--- ARM A = cost-aware RETRAIN (train WITH cost+corridor, roll@0.15 net of cost) ---')
    print(f"  mean {sa['mean']:+.3f}  std {sa['std']:.3f}  min {sa['min']:+.3f}  "
          f"max {sa['max']:+.3f}  CVaR5 {sa['cvar5']:+.3f}")
    print('--- ARM B = frozen cost-BLIND checkpoints, cost-aware roll@0.15 net of cost ---')
    print(f"  mean {sb['mean']:+.3f}  std {sb['std']:.3f}  min {sb['min']:+.3f}  "
          f"max {sb['max']:+.3f}  CVaR5 {sb['cvar5']:+.3f}")
    print('\n--- PAIRED delta (A - B), $/oz ---')
    print(f"  mean {md:+.4f}  SE {se:.4f}  t {t:+.3f}  (n={n})")
    print(f"  delta min {D.min():+.4f}  max {D.max():+.4f}  |delta| max {np.abs(D).max():.4f}")
    los = df[df['is_loser']]
    if len(los):
        print(f"  losers-only ({len(los)}): A mean {los['A_greedy'].mean():+.3f}  "
              f"B mean {los['B_greedy'].mean():+.3f}  delta mean {los['delta'].mean():+.4f}  "
              f"delta min {los['delta'].min():+.4f} max {los['delta'].max():+.4f}")
    print('\n--- ARM A validity ---')
    print(f"  bound-PASS {n_pass}/{n}   months-with-breaches {n_breach}/{n}   total breaches {tot_breach}")
    print('\nWROTE', args.out)


if __name__ == '__main__':
    main()
