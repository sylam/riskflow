"""ARM A: cost-aware RETRAIN subset (throwaway; tb_ prefix, untracked, no commit).

Scoped confirmatory retrain (mirrors the train-in-corridor Phase-B methodology): TRAIN the
DiffSolverV2 value function WITH the realistic BASE Evaluator cost block overlaid AND the delta
corridor band 0.15 active, then ROLL the seed-ensemble on the realized path net of the same
realistic cost. Fresh per-lane run dir => one_trade TRAINS from scratch (no symlinked
checkpoints), so the saved value fn is produced under the cost+corridor Evaluator.

Paired comparator (ARM B) is the already-computed net_of_cost eval:
  artifacts/walk_forward/net_of_cost/tb_row_base_015_<month>.json
  = frozen cost-BLIND garch48 checkpoints rolled at band 0.15 with the realistic cost-aware
  argmax + realistic realized cost. Same months, same band, same BASE cost, same md.

md-parity: calibrated_md is the EXACT per-month GARCH md the garch48 training run used
(build_month_map), so the ONLY difference from ARM B is train-WITH-cost/corridor vs frozen.

Restart-safe: per-month tb_row sidecar (skip if present) + one_trade seed-level idempotency
(a seed whose checkpoint already exists in the run dir is skipped). Records greedy/nohedge (net
of realistic cost, $/oz), bound/PASS, churn, realized corridor breaches, per-seed cost-aware
train_u, V_0.

JSON-is-the-contract: the cost block is spliced into the in-memory Evaluator before build; no
framework edit, no monkey-patching. (NOTE: the framework's DiffV2_Cost_Aware_Argmax is applied
at the VERDICT + stepper ROLL argmax only; the backward-DP fit + exploration bank are cost-free
by design -- so the SAVED nets are cost-insensitive; the corridor DOES enter the fit via bank
projection + grid filter. This subset settles whether that combined training difference moves the
net-of-cost tail vs merely rolling the frozen cost-blind checkpoints.)
"""
import os, sys, json, copy, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root first (shadow-import trap)
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

from production_walk_forward import build_corrected_archive, one_trade, delta_corridor_schedule
from tb_garch_corridor import ROOT, ARCHIVE, TEMPLATE, WF, SEEDS, SFX, build_month_map
from tb_cost_frontier import cost_blocks
from tb_train_in_corridor import count_breaches, trade_date_of, fixings_of  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

BAND = 0.15


def run_job(month, arch, mmap, run_dir):
    row_path = os.path.join(run_dir, f'tb_row_costaware_{month}.json')
    if os.path.exists(row_path):
        logging.info('JOB %s: SKIP (tb_row exists)', month)
        return json.load(open(row_path))

    src, md = mmap[month]
    trade_date = trade_date_of(month)

    # Overlay the realistic BASE cost block onto a FRESH template Evaluator; build_deal_config
    # deep-copies again and only rewrites Position_Limits/Total_Position_Schedule, so the cost
    # keys survive into BOTH the train and roll cfgs. This is the same overlay tb_cost_frontier
    # uses, but here the run dir is fresh => one_trade TRAINS under the cost+corridor Evaluator.
    template = json.load(open(TEMPLATE))
    ev = template['Calc']['Calculation']['Hedging_Problem']['Evaluator']
    ev.update(copy.deepcopy(cost_blocks()['base']))

    args = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40,
                              seeds=list(SEEDS), roll_inner=512, delta_corridor=BAND,
                              spot_model='garch')
    logging.info('=== RETRAIN %s band=%.2f md=%s trade_date=%s (train WITH cost+corridor) ===',
                 month, BAND, os.path.basename(md), trade_date.date())
    rec = one_trade(template, arch, trade_date, md, args, run_dir, month)

    # Sanity: this was a REAL retrain -- the cost-aware-trained checkpoints exist in THIS dir.
    for s in SEEDS:
        ck = os.path.join(run_dir, f'value_fn_{month}{SFX}_s{s}.pt')
        assert os.path.exists(ck), f'missing retrained checkpoint {ck}'

    schedule = delta_corridor_schedule(trade_date, fixings_of(trade_date), BAND)
    diag = json.load(open(os.path.join(run_dir, f'diag_{month}{SFX}.json')))
    breaches, worst = count_breaches(diag, schedule)

    out = {'tag': month, 'arm': 'A_costaware_retrain', 'band': f'{int(round(BAND*100)):03d}',
           'greedy': rec['greedy_usd_oz'], 'nohedge': rec['nohedge_usd_oz'],
           'bound': rec['pf_bound'], 'pass': rec['bound_pass'], 'churn': rec['churn'],
           'breaches': breaches, 'breach_worst': round(worst, 6) if worst is not None else None,
           'md': os.path.basename(md), 'V_0': rec['V_0'],
           'train_u': rec['train_u'], 'train_u_seeds': rec['train_u_seeds'],
           'fair': rec['fair'], 'strike': rec['strike']}
    tmp = row_path + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, row_path)
    logging.info('JOB %s DONE: greedy=%s nohedge=%s bound=%s PASS=%s churn=%s breaches=%s(worst=%.4g) train_u=%s',
                 month, out['greedy'], out['nohedge'], out['bound'], out['pass'],
                 out['churn'], breaches, worst or 0.0, out['train_u_seeds'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', nargs='+', required=True)
    ap.add_argument('--run-dir', required=True, help='fresh lane dir under artifacts/walk_forward')
    ap.add_argument('--smoke', action='store_true', help='batch=256 fit_iters=2 seeds=[7] one month')
    args = ap.parse_args()
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(WF, args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    mmap = build_month_map()

    logging.info('ARM A retrain: months=%s run_dir=%s CUDA=%s riskflow=%s',
                 args.months, run_dir, os.environ.get('CUDA_VISIBLE_DEVICES'), riskflow.__file__)

    if args.smoke:
        m = args.months[0]
        src, md = mmap[m]
        template = json.load(open(TEMPLATE))
        template['Calc']['Calculation']['Hedging_Problem']['Evaluator'].update(
            copy.deepcopy(cost_blocks()['base']))
        a = argparse.Namespace(margin=8.0, volume=2500.0, batch=256, fit_iters=2, seeds=[7],
                               roll_inner=64, delta_corridor=BAND, spot_model='garch')
        sd = os.path.join(run_dir, f'smoke_{m}')
        os.makedirs(sd, exist_ok=True)
        rec = one_trade(template, arch, trade_date_of(m), md, a, sd, m)
        logging.info('SMOKE %s: greedy=%s nohedge=%s PASS=%s churn=%s train_u=%s',
                     m, rec['greedy_usd_oz'], rec['nohedge_usd_oz'], rec['bound_pass'],
                     rec['churn'], rec['train_u_seeds'])
        return

    for m in args.months:
        run_job(m, arch, mmap, run_dir)
    logging.info('ARM A LANE COMPLETE (run_dir=%s, %d months)', run_dir, len(args.months))


if __name__ == '__main__':
    main()
