"""Paired ARM A (with-drift, frozen garch48) vs ARM B (driftless retrain) aggregator + report.

n=8 SCOPING sample. Isolates the Ito convexity (driftless) effect on the walk-forward.
  COVERAGE (hypothesis): natural (unfenced) u_pre A vs B -- does driftless raise pre-fixing cover
    toward the ramp? Also reports the band-0.15 (fence-compressed) u_pre.
  TAIL/PNL: greedy $/oz @ band 0.15 (operating point), paired delta+-SE, t, CVaR5, losers, std.
  INTEGRITY: ARM B 8/8 bound-PASS, 0 breaches @0.15.
  SANITY: realized rolls must DIFFER A vs B (drift touches the inner-MC) -- NOT bit-identical.
Writes artifacts/walk_forward/driftless_subset.csv.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
CS = 50.0
MONTHS = ['202105', '202202', '202106', '202001', '202204', '202011', '202102', '202112']
LOSERS = {'202105', '202202', '202106', '202001', '202204'}


def trade_date_of(m):
    return (pd.Timestamp(f'{m[:4]}-{m[4:]}-01') + pd.offsets.BDay(0)).normalize()


def f0_of(m):
    td = trade_date_of(m)
    a = (td + pd.offsets.MonthBegin(3)).normalize()
    return (pd.bdate_range(a, (a + pd.offsets.MonthEnd(0)).normalize())[0] - td).days


def load_diag(path):
    d = json.load(open(path))
    sv = d.get('stepper_verdict') or {}
    q = np.array(sv.get('greedy_q_traj') or [])
    t = np.array(sv.get('greedy_q_t') or [])
    return q, t


def u_pre(q, t, m):
    if q.ndim != 2 or not len(t):
        return None
    u = q.sum(1) / CS
    pre = t < f0_of(m)
    return round(float(np.abs(u[pre]).mean()), 4) if pre.any() else None


def find(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def armA_free_diag(m):
    return find([f'{WF}/garch48_gpu0/diag_{m}_garch.json', f'{WF}/garch48_gpu1/diag_{m}_garch.json'])


def armB_row(m):
    # the lane script's run-dir join doubled the artifacts/walk_forward prefix on the first run;
    # accept BOTH the doubled and the fixed location so this works before and after the fix.
    hits = sorted(glob.glob(f'{WF}/driftless_gpu*/tb_row_driftless_{m}.json') +
                  glob.glob(f'{WF}/artifacts/walk_forward/driftless_gpu*/tb_row_driftless_{m}.json'))
    return hits[0] if hits else None


def armB_lane(m):
    r = armB_row(m)
    return os.path.dirname(r) if r else None


def stats(x):
    x = np.asarray(x, float)
    p5 = np.quantile(x, 0.05)
    cvar5 = float(x[x <= p5].mean()) if (x <= p5).any() else float(p5)
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)), min=float(x.min()),
                max=float(x.max()), cvar5=cvar5)


def q_diff(qa, qb):
    """element-wise realized-roll difference; None if shapes mismatch."""
    if qa.shape != qb.shape or qa.size == 0:
        return None
    return dict(l1=float(np.abs(qa - qb).sum()), maxabs=float(np.abs(qa - qb).max()),
                identical=bool(np.array_equal(qa, qb)))


def main():
    gf = pd.read_csv(f'{WF}/garch48_final.csv'); gf['tag'] = gf['tag'].astype(str)
    gf = gf.set_index('tag')
    rows, sanity = [], []
    for m in MONTHS:
        # ARM A
        a015 = json.load(open(f'{WF}/driftless_armA_b015/tb_row_{m}_b015.json'))
        qa_free, ta_free = load_diag(armA_free_diag(m))
        qa_015, ta_015 = load_diag(f'{WF}/driftless_armA_b015/run_b015/diag_{m}_garch.json')
        A_greedy_free = float(gf.loc[m, 'greedy_usd_oz'])
        A_up_free = u_pre(qa_free, ta_free, m)
        A_up_015 = u_pre(qa_015, ta_015, m)
        # ARM B
        b = json.load(open(armB_row(m)))
        lane = armB_lane(m)
        qb_free, tb_free = load_diag(f'{lane}/diag_{m}_garch_free.json')
        qb_015, tb_015 = load_diag(f'{lane}/b015_{m}/diag_{m}_garch.json')

        rows.append(dict(
            tag=m, is_loser=m in LOSERS,
            A_greedy_015=a015['greedy'], B_greedy_015=b['B_greedy_015'],
            d_greedy_015=round(b['B_greedy_015'] - a015['greedy'], 3),
            A_greedy_free=round(A_greedy_free, 2), B_greedy_free=b['B_greedy_free'],
            d_greedy_free=round(b['B_greedy_free'] - A_greedy_free, 3),
            A_u_pre_free=A_up_free, B_u_pre_free=b['B_u_pre_free'],
            d_u_pre_free=round((b['B_u_pre_free'] or 0) - (A_up_free or 0), 4),
            A_u_pre_015=A_up_015, B_u_pre_015=b['B_u_pre_015'],
            d_u_pre_015=round((b['B_u_pre_015'] or 0) - (A_up_015 or 0), 4),
            A_pass_015=a015['pass'], B_pass_015=b['B_pass_015'],
            A_breaches_015=a015['breaches'], B_breaches_015=b['B_breaches_015'],
            A_churn_015=a015['churn'], B_churn_015=b['B_churn_015'], bound=a015['bound']))
        sanity.append(dict(tag=m, free=q_diff(qa_free, qb_free), b015=q_diff(qa_015, qb_015)))

    df = pd.DataFrame(rows).sort_values('A_greedy_015').reset_index(drop=True)
    df.to_csv(f'{WF}/driftless_subset_paired.csv', index=False)
    # deliverable (long form): one row per (tag, arm, band)
    long = []
    for r in df.to_dict('records'):
        for arm, band, g, up, ps, br in [
                ('A_with_drift', 'free', r['A_greedy_free'], r['A_u_pre_free'], True, None),
                ('A_with_drift', '0.15', r['A_greedy_015'], r['A_u_pre_015'], r['A_pass_015'], r['A_breaches_015']),
                ('B_driftless', 'free', r['B_greedy_free'], r['B_u_pre_free'], True, None),
                ('B_driftless', '0.15', r['B_greedy_015'], r['B_u_pre_015'], r['B_pass_015'], r['B_breaches_015'])]:
            long.append(dict(tag=r['tag'], arm=arm, band=band, greedy=g, u_pre=up,
                             **{'pass': ps}, breaches=br, is_loser=r['is_loser']))
    pd.DataFrame(long).to_csv(f'{WF}/driftless_subset.csv', index=False)

    pd.set_option('display.width', 260)
    n = len(df)
    print('\n================= DRIFTLESS (Ito convexity) SUBSET: ARM A with-drift vs ARM B driftless =================')
    print(f'n = {n}  (SCOPING sample, not a verdict-grade estimate)\n')

    # ---- 1. COVERAGE (the hypothesis: driftless raises pre-fixing cover toward the ramp=1.0) ----
    print('--- 1. COVERAGE: mean |u_pre| (pre-first-fixing net cover; ramp target = 1.000) ---')
    print(df[['tag', 'is_loser', 'A_u_pre_free', 'B_u_pre_free', 'd_u_pre_free',
              'A_u_pre_015', 'B_u_pre_015', 'd_u_pre_015']].to_string(index=False))
    for lbl, ca, cb, cd in [('NATURAL (unfenced) u_pre', 'A_u_pre_free', 'B_u_pre_free', 'd_u_pre_free'),
                            ('BAND-0.15 u_pre', 'A_u_pre_015', 'B_u_pre_015', 'd_u_pre_015')]:
        A, B, D = df[ca].astype(float).values, df[cb].astype(float).values, df[cd].astype(float).values
        se = D.std(ddof=1) / np.sqrt(n)
        from math import comb
        k = int((D > 0).sum())
        p_sign = sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n   # one-sided sign test
        print(f'  {lbl:26s}: A mean {A.mean():.3f}  B mean {B.mean():.3f}  '
              f'delta(B-A) mean {D.mean():+.4f} SE {se:.4f} t {D.mean()/se if se>0 else float("nan"):+.2f}  '
              f'| median {np.median(D):+.4f} | B>A {k}/{n} (sign p={p_sign:.3f}) '
              f'| gap-to-ramp A {1-A.mean():.3f} -> B {1-B.mean():.3f}')

    # ---- 2. TAIL/PNL @ band 0.15 (operating point) ----
    for lbl, ca, cb, cd in [('BAND 0.15 (operating point)', 'A_greedy_015', 'B_greedy_015', 'd_greedy_015'),
                            ('FREE (unfenced)', 'A_greedy_free', 'B_greedy_free', 'd_greedy_free')]:
        A, B, D = df[ca].values, df[cb].values, df[cd].values
        sa, sb = stats(A), stats(B)
        md, sd = float(D.mean()), float(D.std(ddof=1)); se = sd/np.sqrt(n)
        print(f'\n--- 2. TAIL/PNL greedy $/oz -- {lbl} ---')
        print(df[['tag', 'is_loser', ca, cb, cd]].to_string(index=False))
        print(f'  ARM A: mean {sa["mean"]:+.3f} std {sa["std"]:.3f} min {sa["min"]:+.3f} max {sa["max"]:+.3f} '
              f'CVaR5 {sa["cvar5"]:+.3f} mean/std {sa["mean"]/sa["std"]:+.3f}')
        print(f'  ARM B: mean {sb["mean"]:+.3f} std {sb["std"]:.3f} min {sb["min"]:+.3f} max {sb["max"]:+.3f} '
              f'CVaR5 {sb["cvar5"]:+.3f} mean/std {sb["mean"]/sb["std"]:+.3f}')
        print(f'  PAIRED delta(B-A): mean {md:+.4f} SE {se:.4f} t {md/se if se>0 else float("nan"):+.3f} '
              f'| delta min {D.min():+.3f} max {D.max():+.3f} | B>A {int((D>0).sum())}/{n}')
        los = df[df['is_loser']]
        print(f'  losers-only({len(los)}): A {los[ca].mean():+.3f} B {los[cb].mean():+.3f} '
              f'delta {los[cd].mean():+.4f}')

    # ---- 3. INTEGRITY ----
    nb_pass = int(df['B_pass_015'].sum()); nb_breach = int((df['B_breaches_015'].fillna(0) > 0).sum())
    tot_breach = int(df['B_breaches_015'].fillna(0).sum())
    print(f'\n--- 3. INTEGRITY (ARM B @0.15) --- bound-PASS {nb_pass}/{n}  months-with-breaches {nb_breach}/{n}  total breaches {tot_breach}')

    # ---- 4. SANITY: realized rolls must DIFFER A vs B ----
    print('\n--- 4. SANITY: realized-roll A-vs-B difference (must be NON-zero: drift touches inner-MC) ---')
    n_ident_free = n_ident_015 = 0
    for s in sanity:
        ff, bb = s['free'], s['b015']
        if ff and ff['identical']: n_ident_free += 1
        if bb and bb['identical']: n_ident_015 += 1
        print(f"  {s['tag']}: free L1={ff['l1'] if ff else 'NA':>8}  maxΔq={ff['maxabs'] if ff else 'NA':>6}  identical={ff['identical'] if ff else 'NA'}"
              f"  || b015 L1={bb['l1'] if bb else 'NA':>8}  maxΔq={bb['maxabs'] if bb else 'NA':>6}  identical={bb['identical'] if bb else 'NA'}")
    print(f'  identical rolls: free {n_ident_free}/{n}  band0.15 {n_ident_015}/{n}  '
          f'(expect 0/{n}; any identical => investigate)')
    print('\nWROTE', f'{WF}/driftless_subset.csv')


if __name__ == '__main__':
    main()
