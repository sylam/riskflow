"""NET-OF-COST delta-corridor band frontier (throwaway; tb_ prefix, untracked, no commit).

EVAL-ONLY: reuses tb_garch_corridor's frozen-checkpoint symlink machinery + production
one_trade, but OVERLAYS one of several Evaluator cost blocks before the roll. Training is
skipped (frozen checkpoints), only the stepper roll executes, under the causal delta corridor
stamped on Evaluator.Total_Position_Schedule AND the chosen cost model.

Cost configs (overlaid onto Calc.Calculation.Hedging_Problem.Evaluator):
  flat10 : current baseline -> Bid_Offer_Spread_Bps=10.0 scalar, no calendar, no IM.
  zero   : Bid_Offer_Spread_Bps=0.0 scalar (STEP 0 gross reference).
  base   : Evaluator_Cost_Block_BASE  from artifacts/cost_model_realistic.json.
  high   : Evaluator_Cost_Block_HIGH  from artifacts/cost_model_realistic.json.

Bands: 'free' (no Total_Position_Schedule) or a float half-width fraction.
Idempotent per (cost, band, month) via tb_row sidecars; asserts no training happened.
"""
import os, sys, json, copy, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root first (shadow-import trap)
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

from production_walk_forward import build_corrected_archive, one_trade, delta_corridor_schedule
from tb_garch_corridor import (ROOT, ARCHIVE, TEMPLATE, WF, SEEDS, SFX,
                               build_month_map, link_checkpoints, month_list)
from tb_train_in_corridor import count_breaches, trade_date_of, fixings_of  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

COST_JSON = os.path.join(ROOT, 'artifacts', 'cost_model_realistic.json')


def cost_blocks():
    cm = json.load(open(COST_JSON))
    return {
        'flat10': {'Transaction_Cost_Per_Unit': 0.0, 'Bid_Offer_Spread_Bps': 10.0},
        'zero':   {'Transaction_Cost_Per_Unit': 0.0, 'Bid_Offer_Spread_Bps': 0.0},
        'base':   cm['Evaluator_Cost_Block_BASE'],
        'high':   cm['Evaluator_Cost_Block_HIGH'],
    }


def band_str(band):
    return 'free' if band is None else f'{int(round(band * 100)):03d}'


def run_job(month, band, cost_name, block, arch, base, mmap):
    bstr = band_str(band)
    row_path = os.path.join(base, f'tb_row_{cost_name}_{bstr}_{month}.json')
    if os.path.exists(row_path):
        logging.info('JOB %s %s b%s: SKIP (tb_row exists)', month, cost_name, bstr)
        return json.load(open(row_path))

    src, md = mmap[month]
    run_dir = os.path.join(base, f'run_{cost_name}_{bstr}')
    os.makedirs(run_dir, exist_ok=True)
    stamps = link_checkpoints(src, month, run_dir)
    trade_date = trade_date_of(month)

    # Overlay the cost block onto a fresh template Evaluator (build_deal_config deep-copies again).
    template = json.load(open(TEMPLATE))
    ev = template['Calc']['Calculation']['Hedging_Problem']['Evaluator']
    ev.update(copy.deepcopy(block))

    args = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40,
                              seeds=list(SEEDS), roll_inner=512,
                              delta_corridor=band, spot_model='garch')
    logging.info('=== JOB %s cost=%s band=%s md=%s (EVAL-ONLY) ===',
                 month, cost_name, bstr, os.path.basename(md))
    rec = one_trade(template, arch, trade_date, md, args, run_dir, month)

    for p, mt in stamps:
        assert os.path.getmtime(p) == mt, f'checkpoint {p} was REWRITTEN -- not eval-only'

    if band is None:
        breaches, worst = 0, None
    else:
        schedule = delta_corridor_schedule(trade_date, fixings_of(trade_date), band)
        diag = json.load(open(os.path.join(run_dir, f'diag_{month}{SFX}.json')))
        breaches, worst = count_breaches(diag, schedule)

    diag = json.load(open(os.path.join(run_dir, f'diag_{month}{SFX}.json')))
    sv = diag.get('stepper_verdict') or {}
    out = {'tag': month, 'cost': cost_name, 'band': bstr,
           'greedy': rec['greedy_usd_oz'], 'nohedge': rec['nohedge_usd_oz'],
           'bound': rec['pf_bound'], 'pass': rec['bound_pass'], 'churn': rec['churn'],
           'breaches': breaches, 'breach_worst': round(worst, 6) if worst is not None else None,
           'md': os.path.basename(md), 'V_0': rec['V_0'],
           'greedy_q_traj': sv.get('greedy_q_traj'), 'greedy_q_t': sv.get('greedy_q_t')}
    tmp = row_path + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, row_path)
    logging.info('JOB %s cost=%s b%s DONE: greedy=%s nohedge=%s PASS=%s churn=%s breaches=%s',
                 month, cost_name, bstr, out['greedy'], out['nohedge'], out['pass'],
                 out['churn'], breaches)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost', required=True, choices=['flat10', 'zero', 'base', 'high'])
    ap.add_argument('--band', required=True, help="'free' or a float e.g. 0.15")
    ap.add_argument('--months', nargs='*', default=None, help='default: all 48')
    ap.add_argument('--base', default=os.path.join(WF, 'net_of_cost'))
    args = ap.parse_args()
    os.makedirs(args.base, exist_ok=True)

    band = None if args.band == 'free' else float(args.band)
    block = cost_blocks()[args.cost]

    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    mmap = build_month_map()
    months = args.months or (month_list('2020-01', 24) + month_list('2022-01', 24))

    logging.info('COST %s BAND %s: %d months, CUDA=%s riskflow=%s',
                 args.cost, args.band, len(months), os.environ.get('CUDA_VISIBLE_DEVICES'),
                 riskflow.__file__)
    for m in months:
        run_job(m, band, args.cost, block, arch, args.base, mmap)
    logging.info('COST %s BAND %s COMPLETE', args.cost, args.band)


if __name__ == '__main__':
    main()
