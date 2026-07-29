"""PHASE B: train-in-corridor experiment driver (throwaway; tb_ prefix, untracked, no commit).

Reuses the validated production_walk_forward.one_trade machinery to TRAIN each seed IN the
delta-corridor (build_deal_config(delta_corridor=band) stamps Evaluator.Total_Position_Schedule
on BOTH the train and roll cfgs) and roll the seed-ensemble under the SAME corridor. md-parity:
we point calibrated_md at the EXACT quarterly md json full48 used, so the training world matches
the roll-clip reference apples-to-apples. Per-(month,band) run dirs; seed-level + job-level
idempotency for restart safety. Verifies: bank-breach diagnostic ~0.000 in training, silent roll
provenance, zero realized corridor breaches, bound-PASS.
"""
import os, sys, json, argparse, logging, copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root first (shadow-import trap)
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

from production_walk_forward import build_corrected_archive, one_trade, delta_corridor_schedule

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, 'data', 'pl_exp.csv')
TEMPLATE = os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
BASE = os.environ['TB_BASE']  # scratchpad results/run base

# Exact quarterly md json each month used in full48 (deterministic from the 2020-01/recal-3 schedule)
MD_MAP = {
    '202008': f'{WF}/full48_gpu0/md_202007.json',
    '202011': f'{WF}/full48_gpu0/md_202010.json',
    '202102': f'{WF}/full48_gpu0/md_202101.json',
    '202105': f'{WF}/full48_gpu0/md_202104.json',
    '202106': f'{WF}/full48_gpu0/md_202104.json',
    '202112': f'{WF}/full48_gpu0/md_202110.json',
    '202202': f'{WF}/full48_gpu1/md_202201.json',
    '202207': f'{WF}/full48_gpu1/md_202207.json',
}

# (month, band) jobs per lane
LANES = {
    'A': [('202105', 0.40), ('202202', 0.40), ('202008', 0.40), ('202106', 0.40), ('202105', 0.60)],
    'B': [('202102', 0.40), ('202112', 0.40), ('202011', 0.40), ('202207', 0.40), ('202102', 0.60)],
}

SEEDS = [7, 42, 314]


def trade_date_of(month):
    d = pd.Timestamp(f'{month[:4]}-{month[4:]}-01') + pd.offsets.BDay(0)
    return d.normalize()


def fixings_of(trade_date):
    avg_start = (trade_date + pd.offsets.MonthBegin(3)).normalize()
    avg_end = (avg_start + pd.offsets.MonthEnd(0)).normalize()
    return pd.bdate_range(avg_start, avg_end)


def corridor_at(schedule, t):
    lo, hi = schedule[0]['Min_Total'], schedule[0]['Max_Total']
    for k in schedule:
        if k['Step'] > t:
            break
        lo, hi = k['Min_Total'], k['Max_Total']
    return lo, hi


def count_breaches(diag, schedule):
    """Realized corridor breaches on the stepper roll: total signed position (row-sum of
    greedy_q_traj) outside the corridor at each decision step (keyed by greedy_q_t calendar step)."""
    sv = diag.get('stepper_verdict') or {}
    q = np.array(sv.get('greedy_q_traj') or [])
    steps = sv.get('greedy_q_t') or list(range(len(q)))
    if q.ndim != 2:
        return None, None
    tol = 1e-3
    n = 0
    worst = 0.0
    for i in range(len(q)):
        lo, hi = corridor_at(schedule, int(steps[i]))
        tot = float(q[i].sum())
        d = max(lo - tol - tot, tot - (hi + tol), 0.0)
        if d > 0:
            n += 1
            worst = max(worst, d)
    return n, worst


def run_job(month, band, arch, template):
    tag = month
    bstr = f'{int(round(band * 100)):03d}'
    row_path = os.path.join(BASE, f'tb_row_{month}_b{bstr}.json')
    if os.path.exists(row_path):
        logging.info('JOB %s b%.2f: SKIP (tb_row exists)', month, band)
        return json.load(open(row_path))

    run_dir = os.path.join(BASE, f'run_{month}_b{bstr}')
    os.makedirs(run_dir, exist_ok=True)
    md = MD_MAP[month]
    trade_date = trade_date_of(month)

    args = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40,
                              seeds=list(SEEDS), roll_inner=512, delta_corridor=band)
    logging.info('=== JOB %s band=%.2f md=%s trade_date=%s ===', month, band,
                 os.path.basename(md), trade_date.date())
    rec = one_trade(template, arch, trade_date, md, args, run_dir, tag)

    # realized breach check against the same causal schedule
    fixings = fixings_of(trade_date)
    schedule = delta_corridor_schedule(trade_date, fixings, band)
    diag = json.load(open(os.path.join(run_dir, f'diag_{tag}.json')))
    breaches, worst = count_breaches(diag, schedule)

    out = {'month': month, 'band': band, 'mode': 'train-in',
           'greedy': rec['greedy_usd_oz'], 'nohedge': rec['nohedge_usd_oz'],
           'bound': rec['pf_bound'], 'PASS': rec['bound_pass'], 'churn': rec['churn'],
           'breaches': breaches, 'breach_worst': round(worst, 6) if worst is not None else None,
           'train_u_seeds': rec['train_u_seeds'], 'V_0': rec['V_0'],
           'fair': rec['fair'], 'strike': rec['strike']}
    tmp = row_path + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, row_path)
    logging.info('JOB %s b%.2f DONE: greedy=%s nohedge=%s bound=%s PASS=%s churn=%s breaches=%s(worst=%.4g)',
                 month, band, out['greedy'], out['nohedge'], out['bound'], out['PASS'],
                 out['churn'], breaches, worst or 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', required=True, choices=['A', 'B'])
    ap.add_argument('--smoke', action='store_true', help='fit_iters=2, seeds=[7], one job only')
    args = ap.parse_args()
    os.makedirs(BASE, exist_ok=True)

    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    template = json.load(open(TEMPLATE))

    jobs = LANES[args.lane]
    if args.smoke:
        global SEEDS
        SEEDS = [7]
        jobs = jobs[:1]

    logging.info('LANE %s starting: %d jobs, CUDA_VISIBLE_DEVICES=%s riskflow=%s',
                 args.lane, len(jobs), os.environ.get('CUDA_VISIBLE_DEVICES'), riskflow.__file__)
    for month, band in jobs:
        if args.smoke:
            # tiny config into a throwaway dir
            run_dir = os.path.join(BASE, f'smoke_{month}_b{int(round(band*100)):03d}')
            os.makedirs(run_dir, exist_ok=True)
            a = argparse.Namespace(margin=8.0, volume=2500.0, batch=256, fit_iters=2,
                                   seeds=[7], roll_inner=64, delta_corridor=band)
            rec = one_trade(template, arch, trade_date_of(month), MD_MAP[month], a, run_dir, month)
            fx = fixings_of(trade_date_of(month))
            sch = delta_corridor_schedule(trade_date_of(month), fx, band)
            dg = json.load(open(os.path.join(run_dir, f'diag_{month}.json')))
            br, w = count_breaches(dg, sch)
            logging.info('SMOKE %s b%.2f: greedy=%s PASS=%s breaches=%s worst=%.4g',
                         month, band, rec['greedy_usd_oz'], rec['bound_pass'], br, w or 0.0)
            return
        run_job(month, band, arch, template)
    logging.info('LANE %s COMPLETE', args.lane)


if __name__ == '__main__':
    main()
