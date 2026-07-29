"""HMM-world 48-trade corridor RE-ROLL (throwaway; tb_ prefix, untracked, no commit).

EVAL-ONLY: reuses production_walk_forward.one_trade, but every month's seed checkpoints are
symlinked in from the completed garch48_gpu{0,1} run dirs, so one_trade's seed-level idempotency
skips training entirely and only the stepper ROLL executes -- under the causal delta corridor
stamped on Evaluator.Total_Position_Schedule (the layer where 100% of the corridor's effect lives,
per f8c067b). md is the EXACT per-month calibrated GARCH md the training run used (quarterly, or
the _fresh_garch fallback for the months that took the stale-md retry path).

Fresh per-band run dirs; job-level idempotency (tb_row_*.json) for restart safety. Asserts no
training happened (checkpoint mtimes unchanged) and that the source rows are spot_model='hmm'.
"""
import os, sys, json, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root first (shadow-import trap)
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

from production_walk_forward import build_corrected_archive, one_trade, delta_corridor_schedule
os.environ.setdefault('TB_BASE', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'artifacts', 'walk_forward', 'hmm48_corridor'))
from tb_train_in_corridor import count_breaches, trade_date_of, fixings_of  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, 'data', 'pl_exp.csv')
TEMPLATE = os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')

SEEDS = [7, 42, 314]
SFX = ''
# the two completed training lanes: (run dir, first trade month) -- 24 months each, recal every 3
LANES_SRC = [(f'{WF}/full48_gpu0', '2020-01'), (f'{WF}/full48_gpu1', '2022-01')]


def month_list(start, n):
    return [(pd.Timestamp(start + '-01') + pd.offsets.MonthBegin(m)).strftime('%Y%m') for m in range(n)]


def build_month_map():
    """month -> (src_dir, md_path). md is the quarterly calibrated GARCH md, or the _fresh_garch
    one for months whose row sidecar recorded the stale-md fallback."""
    out = {}
    for src, start in LANES_SRC:
        months = month_list(start, 24)
        base = pd.read_csv(f'{WF}/full48_baseline.csv')
        base['tag'] = base['tag'].astype(str)
        fresh = set(base.loc[base.fresh_md.notna(), 'tag'])
        for i, m in enumerate(months):
            assert os.path.exists(f'{src}/row_{m}.json'), m
            md = (f'{src}/md_{m}_fresh.json' if m in fresh
                  else f'{src}/md_{months[(i // 3) * 3]}.json')
            assert os.path.exists(md), md
            out[m] = (src, os.path.abspath(md))
    assert len(out) == 48, len(out)
    return out


def link_checkpoints(src, month, run_dir):
    """Symlink the month's trained seed checkpoints into run_dir under the names one_trade expects,
    so its seed-level idempotency skips training. Returns their (path, mtime) for the no-train gate."""
    stamps = []
    for s in SEEDS:
        name = f'value_fn_{month}{SFX}_s{s}.pt'
        srcp = os.path.abspath(os.path.join(src, name))
        assert os.path.exists(srcp), f'missing checkpoint {srcp}'
        dstp = os.path.join(run_dir, name)
        if not os.path.exists(dstp):
            os.symlink(srcp, dstp)
        stamps.append((srcp, os.path.getmtime(srcp)))
    return stamps


def run_job(month, band, arch, template, mmap, base):
    bstr = f'{int(round(band * 100)):03d}'
    row_path = os.path.join(base, f'tb_row_{month}_b{bstr}.json')
    if os.path.exists(row_path):
        logging.info('JOB %s b%.2f: SKIP (tb_row exists)', month, band)
        return json.load(open(row_path))

    src, md = mmap[month]
    run_dir = os.path.join(base, f'run_b{bstr}')
    os.makedirs(run_dir, exist_ok=True)
    stamps = link_checkpoints(src, month, run_dir)
    trade_date = trade_date_of(month)

    args = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40,
                              seeds=list(SEEDS), roll_inner=512, delta_corridor=band,
                              spot_model='hmm')
    logging.info('=== JOB %s band=%.2f md=%s trade_date=%s (EVAL-ONLY re-roll) ===',
                 month, band, os.path.basename(md), trade_date.date())
    rec = one_trade(template, arch, trade_date, md, args, run_dir, month)

    for p, mt in stamps:                      # no-train gate: training would rewrite the checkpoint
        assert os.path.getmtime(p) == mt, f'checkpoint {p} was REWRITTEN -- this was not eval-only'

    schedule = delta_corridor_schedule(trade_date, fixings_of(trade_date), band)
    diag = json.load(open(os.path.join(run_dir, f'diag_{month}{SFX}.json')))
    breaches, worst = count_breaches(diag, schedule)

    out = {'tag': month, 'band': band, 'greedy': rec['greedy_usd_oz'], 'nohedge': rec['nohedge_usd_oz'],
           'bound': rec['pf_bound'], 'pass': rec['bound_pass'], 'churn': rec['churn'],
           'breaches': breaches, 'breach_worst': round(worst, 6) if worst is not None else None,
           'md': os.path.basename(md), 'V_0': rec['V_0']}
    tmp = row_path + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, row_path)
    logging.info('JOB %s b%.2f DONE: greedy=%s nohedge=%s bound=%s PASS=%s churn=%s breaches=%s(worst=%.4g)',
                 month, band, out['greedy'], out['nohedge'], out['bound'], out['pass'],
                 out['churn'], breaches, worst or 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', type=float, required=True)
    ap.add_argument('--months', nargs='*', default=None, help='default: all 48')
    ap.add_argument('--base', default=os.path.join(WF, 'hmm48_corridor'))
    args = ap.parse_args()
    os.makedirs(args.base, exist_ok=True)

    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    template = json.load(open(TEMPLATE))
    mmap = build_month_map()
    months = args.months or (month_list('2020-01', 24) + month_list('2022-01', 24))

    logging.info('BAND %.2f: %d months, CUDA_VISIBLE_DEVICES=%s riskflow=%s',
                 args.band, len(months), os.environ.get('CUDA_VISIBLE_DEVICES'), riskflow.__file__)
    for m in months:
        run_job(m, args.band, arch, template, mmap, args.base)
    logging.info('BAND %.2f COMPLETE', args.band)


if __name__ == '__main__':
    main()
