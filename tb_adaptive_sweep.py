"""ADAPTIVE-CORRIDOR eval sweep (eval only, no training; tb_ prefix, untracked, no commit).

For each of the 48 walk-forward months and each fence in {band 0.40, band 0.60, free (no
schedule)}: run the frozen 3-seed ensemble batch verdict (batch 2048) under that fence, reusing
the validated production machinery. The corridor is enforced at eval time via
Evaluator.Total_Position_Schedule (build_deal_config(delta_corridor=band)) which HedgeActionSpace
clamps in grid_at(t), so the batch-verdict argmax obeys the fence. COMMON RANDOM NUMBERS: the
three fences of a month share the eval seed => identical outer worlds, only the fence differs
(paired). Records verdict.greedy u_mean / E[W_T] / p5 / cvar5.

Restart-safe: one result sidecar per (month, fence, seed); an existing sidecar is skipped. Two
GPU lanes split the months (CUDA_VISIBLE_DEVICES=0/1). Noise estimate: extra eval seeds on 8
spot months (the verdict path exposes only aggregate stats, no per-path W_T through the JSON
contract, so between-seed dispersion of the paired margin stands in for the paired SE).
"""
import os, sys, json, argparse, logging, copy, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root first (shadow trap)
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

from production_walk_forward import build_corrected_archive, build_deal_config
from production_solver import apply_config, run

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
OUT = os.environ.get('TB_OUT', os.path.join(ROOT, 'scratchpad_adaptive'))
SEEDS = [7, 42, 314]                 # ensemble members (frozen checkpoints)
PRIMARY_SEED = 101                   # eval outer-world seed (distinct from training seeds)
NOISE_SEEDS = [202, 303]             # extra eval seeds for the spot-month noise estimate
FENCES = [('b040', 0.40), ('b060', 0.60), ('free', None)]
FRESH = {'202212', '202303', '202306', '202309', '202312'}   # stale-md fallback months (fresh md)
SPOT_MONTHS = ['202008', '202011', '202102', '202105',       # lane 0 (2020-21)
               '202202', '202207', '202301', '202312']       # lane 1 (2022-23)

ALL_MONTHS = [f'{y}{mo:02d}' for y in (2020, 2021, 2022, 2023) for mo in range(1, 13)]


def gpu_dir(m):
    return os.path.join(WF, 'full48_gpu0' if m < '2022' else 'full48_gpu1')


def md_of(m):
    if m in FRESH:
        return os.path.join(gpu_dir(m), f'md_{m}_fresh.json')
    y, mo = int(m[:4]), int(m[4:])
    qmo = ((mo - 1) // 3) * 3 + 1                              # most-recent quarterly recal
    return os.path.join(gpu_dir(m), f'md_{y}{qmo:02d}.json')


def ckpts_of(m):
    return [os.path.join(gpu_dir(m), f'value_fn_{m}_s{s}.pt') for s in SEEDS]


def trade_date_of(month):
    d = pd.Timestamp(f'{month[:4]}-{month[4:]}-01') + pd.offsets.BDay(0)
    return d.normalize()


def eval_one(template, arch, month, fence_lbl, band, seed):
    """Eval-only ensemble batch verdict under `band` (None = free). Returns greedy stats."""
    trade_date = trade_date_of(month)
    cfg, _ = build_deal_config(template, arch, trade_date, md_of(month), 8.0, 2500.0,
                               delta_corridor=band)
    ev = apply_config(copy.deepcopy(cfg), batch=2048, seed=seed, load=ckpts_of(month),
                      randomize_initial_state=False)
    diag = run(ev, f'sweep_{month}_{fence_lbl}_s{seed}')
    v = (diag.get('verdict') or {}).get('greedy') or {}
    return {'u_mean': v.get('u_mean'), 'ew': v.get('wT_mean'),
            'p5': v.get('wT_p5'), 'cvar5': v.get('wT_cvar5')}


def run_cell(template, arch, month, fence_lbl, band, seed):
    path = os.path.join(OUT, f'res_{month}_{fence_lbl}_s{seed}.json')
    if os.path.exists(path):
        logging.info('SKIP %s %s s%d (exists)', month, fence_lbl, seed)
        return
    t0 = time.time()
    stats = eval_one(template, arch, month, fence_lbl, band, seed)
    rec = {'month': month, 'fence': fence_lbl, 'band': band, 'seed': seed, **stats}
    tmp = path + '.tmp'
    json.dump(rec, open(tmp, 'w'), default=str)
    os.replace(tmp, path)
    logging.info('DONE %s %s s%d: u=%+.4f ew=%+.0f p5=%+.0f cvar5=%+.0f (%.1fs)',
                 month, fence_lbl, seed, stats['u_mean'], stats['ew'], stats['p5'],
                 stats['cvar5'], time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', required=True, choices=['2020', '2021', '2022', '2023'],
                    help='year lane (two lanes share each GPU; months are disjoint so no claim race)')
    ap.add_argument('--smoke', action='store_true', help='one eval only, timing check')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    raw = pd.read_csv(os.path.join(ROOT, 'data', 'pl_exp.csv'), index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    template = json.load(open(os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')))

    months = [m for m in ALL_MONTHS if m.startswith(args.lane)]
    logging.info('LANE %s: %d months, CUDA_VISIBLE_DEVICES=%s riskflow=%s',
                 args.lane, len(months), os.environ.get('CUDA_VISIBLE_DEVICES'), riskflow.__file__)

    if args.smoke:
        run_cell(template, arch, months[0], *FENCES[0], PRIMARY_SEED)
        return

    # main sweep: every month x every fence at the primary seed
    for month in months:
        for lbl, band in FENCES:
            run_cell(template, arch, month, lbl, band, PRIMARY_SEED)
    # noise: extra eval seeds on this lane's spot months x every fence
    for month in [m for m in SPOT_MONTHS if m in months]:
        for seed in NOISE_SEEDS:
            for lbl, band in FENCES:
                run_cell(template, arch, month, lbl, band, seed)
    logging.info('LANE %s COMPLETE', args.lane)


if __name__ == '__main__':
    main()
