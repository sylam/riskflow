"""ARM B: DRIFTLESS GARCH retrain subset (throwaway, tb_ prefix, untracked, no commit).

Mirrors the train-in-corridor Phase-B / cost-aware-subset methodology. Isolates the DRIFT term:
ARM A = frozen garch48 checkpoints (trained WITH the +1/2 h Jensen drift) rolled at band 0.15;
ARM B = RETRAIN from scratch on the DRIFTLESS md (Convexity_Correction=Yes, byte-identical params
otherwise -- built by tb_driftless_md.py), corridor-FREE, same seeds/batch/iters/roll-inner.

Corridor-FREE training (delta_corridor=None) mirrors ARM A's construction (the garch48 checkpoints
are corridor-free, re-rolled at bands eval-only) and is REQUIRED so the same checkpoints can be
rolled BOTH unfenced (natural-coverage u_pre = the hypothesis test; at band 0.15 the fence pins
u_pre near the ramp and masks the drift's coverage signal) AND at the operating band 0.15 (tail).
Per the corridor verdict train-in-corridor is a roll no-op, so corridor-free vs 0.15-trained give
the same 0.15 roll -- corridor-free is strictly more informative here.

Per month: (1) train 3 corridor-free checkpoints on the driftless md + FREE roll (one_trade,
delta_corridor=None); (2) eval-only roll @ band 0.15 (symlink the checkpoints into a b015 subdir,
one_trade delta_corridor=0.15, training skipped). Records greedy/u_pre/churn/breaches/bound-PASS
for BOTH rolls. Restart-safe: per-month sidecar (skip if present) + one_trade seed-idempotency.
JSON-is-the-contract: only the md file differs from ARM A (no framework edit, no monkey-patch).
"""
import os, sys, json, copy, shutil, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root first (shadow-import trap)
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

from production_walk_forward import build_corrected_archive, one_trade, delta_corridor_schedule
from tb_garch_corridor import ROOT, ARCHIVE, TEMPLATE, WF, SEEDS, SFX, build_month_map, link_checkpoints
from tb_train_in_corridor import count_breaches, trade_date_of, fixings_of

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

BAND = 0.15
MDDIR = os.path.join(WF, 'driftless_md')
CS = 50.0


def driftless_md_for(month, mmap):
    src = mmap[month][1]
    p = os.path.join(MDDIR, os.path.basename(src).replace('.json', '_driftless.json'))
    assert os.path.exists(p), f'driftless md missing (run tb_driftless_md.py first): {p}'
    blk = json.load(open(p))['MarketData']['Price Models']['GARCHSpotModel.PLATINUM_CME']
    assert blk.get('Convexity_Correction') == 'Yes', f'{p} is NOT driftless'
    return p, blk


def u_pre_of(diag, trade_date):
    """mean |total net position / 50| over the pre-first-fixing window (the coverage metric)."""
    sv = diag.get('stepper_verdict') or {}
    q = np.array(sv.get('greedy_q_traj') or [])
    t = np.array(sv.get('greedy_q_t') or [])
    if q.ndim != 2 or not len(t):
        return None
    f0 = (fixings_of(trade_date)[0] - trade_date).days
    u = q.sum(1) / CS
    pre = t < f0
    return round(float(np.abs(u[pre]).mean()), 4) if pre.any() else None


def q_traj_of(diag):
    sv = diag.get('stepper_verdict') or {}
    return np.array(sv.get('greedy_q_traj') or [])


def run_job(month, arch, template, mmap, run_dir):
    row_path = os.path.join(run_dir, f'tb_row_driftless_{month}.json')
    if os.path.exists(row_path):
        logging.info('JOB %s: SKIP (tb_row exists)', month)
        return json.load(open(row_path))

    md, blk = driftless_md_for(month, mmap)
    trade_date = trade_date_of(month)
    logging.info('=== ARM B %s DRIFTLESS retrain: md=%s Convexity=%s (omega=%.3e alpha=%.4f beta=%.4f nu=%.2f H0=%.3e) ===',
                 month, os.path.basename(md), blk['Convexity_Correction'],
                 blk['Omega'], blk['Alpha'], blk['Beta'], blk['Nu'], blk['H0'])

    # (1) corridor-FREE train + FREE roll
    args_free = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40,
                                   seeds=list(SEEDS), roll_inner=512, delta_corridor=None,
                                   spot_model='garch')
    rec_free = one_trade(template, arch, trade_date, md, args_free, run_dir, month)
    for s in SEEDS:
        ck = os.path.join(run_dir, f'value_fn_{month}{SFX}_s{s}.pt')
        assert os.path.exists(ck), f'missing retrained checkpoint {ck}'
    diag_free = json.load(open(os.path.join(run_dir, f'diag_{month}{SFX}.json')))
    shutil.copyfile(os.path.join(run_dir, f'diag_{month}{SFX}.json'),
                    os.path.join(run_dir, f'diag_{month}{SFX}_free.json'))   # preserve before b015 clobbers
    up_free = u_pre_of(diag_free, trade_date)
    q_free = q_traj_of(diag_free)

    # (2) eval-only roll @ band 0.15 (symlink corridor-free checkpoints; training skipped)
    b015 = os.path.join(run_dir, f'b015_{month}')
    os.makedirs(b015, exist_ok=True)
    stamps = link_checkpoints(run_dir, month, b015)
    args_015 = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40,
                                  seeds=list(SEEDS), roll_inner=512, delta_corridor=BAND,
                                  spot_model='garch')
    rec_015 = one_trade(template, arch, trade_date, md, args_015, b015, month)
    for p, mt in stamps:
        assert os.path.getmtime(p) == mt, f'checkpoint {p} REWRITTEN -- b015 roll was not eval-only'
    diag_015 = json.load(open(os.path.join(b015, f'diag_{month}{SFX}.json')))
    schedule = delta_corridor_schedule(trade_date, fixings_of(trade_date), BAND)
    breaches, worst = count_breaches(diag_015, schedule)
    up_015 = u_pre_of(diag_015, trade_date)
    q_015 = q_traj_of(diag_015)

    out = {'tag': month, 'arm': 'B_driftless', 'md': os.path.basename(md),
           'convexity': blk['Convexity_Correction'],
           'B_greedy_free': rec_free['greedy_usd_oz'], 'B_nohedge': rec_free['nohedge_usd_oz'],
           'B_churn_free': rec_free['churn'], 'B_u_pre_free': up_free,
           'B_greedy_015': rec_015['greedy_usd_oz'], 'B_churn_015': rec_015['churn'],
           'B_u_pre_015': up_015, 'B_pass_015': rec_015['bound_pass'],
           'B_breaches_015': breaches, 'B_breach_worst': round(worst, 6) if worst is not None else None,
           'bound': rec_015['pf_bound'], 'V_0': rec_free['V_0'],
           'train_u_seeds': rec_free['train_u_seeds'], 'fair': rec_free['fair'],
           'q_free_l1_last': float(q_free[-1].sum()) if len(q_free) else None,
           'q_015_l1_last': float(q_015[-1].sum()) if len(q_015) else None}
    tmp = row_path + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, row_path)
    logging.info('JOB %s DONE: greedy_free=%s greedy_015=%s u_pre_free=%s u_pre_015=%s '
                 'PASS=%s breaches=%s churn_015=%s train_u=%s',
                 month, out['B_greedy_free'], out['B_greedy_015'], up_free, up_015,
                 out['B_pass_015'], breaches, out['B_churn_015'], out['train_u_seeds'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', nargs='+', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    # Path bug fix: only join WF for a BARE lane name; a caller-supplied relative path that
    # already points under artifacts/walk_forward must not be re-joined (that doubled the prefix).
    run_dir = (args.run_dir if (os.path.isabs(args.run_dir) or os.sep in args.run_dir)
               else os.path.join(WF, args.run_dir))
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    template = json.load(open(TEMPLATE))
    mmap = build_month_map()

    logging.info('ARM B driftless: months=%s run_dir=%s CUDA=%s riskflow=%s',
                 args.months, run_dir, os.environ.get('CUDA_VISIBLE_DEVICES'), riskflow.__file__)

    if args.smoke:
        m = args.months[0]
        md, blk = driftless_md_for(m, mmap)
        a = argparse.Namespace(margin=8.0, volume=2500.0, batch=256, fit_iters=2, seeds=[7],
                               roll_inner=64, delta_corridor=None, spot_model='garch')
        sd = os.path.join(run_dir, f'smoke_{m}')
        os.makedirs(sd, exist_ok=True)
        rec = one_trade(template, arch, trade_date_of(m), md, a, sd, m)
        logging.info('SMOKE %s: greedy=%s nohedge=%s PASS=%s churn=%s train_u=%s (Convexity=%s)',
                     m, rec['greedy_usd_oz'], rec['nohedge_usd_oz'], rec['bound_pass'],
                     rec['churn'], rec['train_u_seeds'], blk['Convexity_Correction'])
        return

    for m in args.months:
        run_job(m, arch, template, mmap, run_dir)
    logging.info('ARM B LANE COMPLETE (run_dir=%s, %d months)', run_dir, len(args.months))


if __name__ == '__main__':
    main()
