"""Aggregate net-of-cost frontier tb_row sidecars into net_of_cost_frontier.csv (throwaway)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

WF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts', 'walk_forward')
BASE = os.path.join(WF, 'net_of_cost')
BAND_ORDER = {'free': 0, '015': 1, '025': 2, '040': 3, '060': 4}
COST_ORDER = {'flat10': 0, 'base': 1, 'high': 2, 'zero': 3}


def agg():
    groups = {}
    for p in glob.glob(os.path.join(BASE, 'tb_row_*.json')):
        r = json.load(open(p))
        if r.get('cost') is None or r.get('cost') == 'zero':
            continue
        groups.setdefault((r['cost'], r['band']), []).append(r)
    rows = []
    for (cost, band), recs in groups.items():
        g = np.array([x['greedy'] for x in recs], float)
        churn = np.array([x['churn'] for x in recs], float)
        npass = sum(1 for x in recs if x['pass'])
        breaches = sum(int(x['breaches']) for x in recs)
        worst3 = np.sort(g)[:3]
        cvar5 = float(worst3.mean())
        mean, std = float(g.mean()), float(g.std(ddof=0))
        rows.append({
            'band': band, 'cost_config': cost, 'n': len(recs),
            'mean': round(mean, 3), 'std': round(std, 3), 'min': round(float(g.min()), 3),
            'p10': round(float(np.percentile(g, 10)), 3), 'cvar5': round(cvar5, 3),
            'm_std': round(mean / std, 4) if std else None,
            'm_cvar': round(mean / abs(cvar5), 4) if cvar5 else None,
            'bound_pass': f'{npass}/{len(recs)}', 'breaches': breaches,
            'churn': round(float(churn.mean()), 1),
        })
    rows.sort(key=lambda r: (COST_ORDER.get(r['cost_config'], 9), BAND_ORDER.get(r['band'], 9)))
    df = pd.DataFrame(rows)
    out = os.path.join(WF, 'net_of_cost_frontier.csv')
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print('\nWROTE', out, f'({len(df)} rows)')


if __name__ == '__main__':
    agg()
