"""Pick the streaming shipping recipe for the platinum book (throwaway; tb_ prefix).

The retired shape was `Batch_Size 8192, Simulation_Batches 1` with `DiffV2_OOS_Frac 0.5`: 4096
paths fitted, 4096 sibling rows of the SAME draw held out. A solve is now a stream, so the two
axes are `Batch_Size` (paths per batch, and the only thing fork memory depends on) and
`Simulation_Batches` (stream length: N-1 fit batches, then a held-out batch that is an
INDEPENDENT draw).

Comparability. Arms differ in both fitted and held-out path counts, so the raw greedy number is
not comparable across them — each arm's verdict is measured on its own held-out world. The
comparable quantity is the EDGE over that arm's own textbook and no-hedge tracks, which are rolled
on the same paths. Both are reported.

    python tb_shipping_recipe.py <out.csv> [seed ...]
"""
import copy
import csv
import json
import logging
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch

from production_solver import apply_config, run

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

CONFIG = 'artifacts/platinum_hedge_shipping.json'
# (Batch_Size, Simulation_Batches) -> (fitted, held-out) paths.
#   4096x2  = 4096 fitted, 4096 held out — the retired shape's arithmetic, honest held-out world
#   8192x2  = 8192 / 8192 — the old Batch_Size, now all of it fitted
#   4096x5  = 16384 / 4096 — four fit batches, fresh paths each
#   8192x5  = 32768 / 8192 — both levers up
ARMS = [(4096, 2), (8192, 2), (4096, 5), (8192, 5)]
FIELDS = ('batch', 'batches', 'fitted', 'held_out', 'seed', 'V_0', 'bounded', 'wall_s',
          'peak_GiB', 'greedy_u', 'textbook_u', 'nohedge_u', 'edge_u', 'greedy_wT',
          'greedy_p5', 'greedy_cvar5', 'textbook_p5', 'textbook_cvar5', 'edge_p5', 'edge_cvar5')


def arm(template, batch, batches, seed):
    cfg = apply_config(copy.deepcopy(template), batch=batch, batches=batches, seed=seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    diag = run(cfg, f'ship_b{batch}_n{batches}_s{seed}')
    torch.cuda.synchronize()
    v = diag.get('verdict') or {}
    g, tb, nh = (v.get(k) or {} for k in ('greedy', 'textbook', 'nohedge'))
    return {
        'batch': batch, 'batches': batches, 'fitted': batch * (batches - 1), 'held_out': batch,
        'seed': seed, 'V_0': diag.get('V_0'), 'bounded': diag.get('bounded'),
        'wall_s': round(time.perf_counter() - t0, 1),
        'peak_GiB': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        'greedy_u': g.get('u_mean'), 'textbook_u': tb.get('u_mean'), 'nohedge_u': nh.get('u_mean'),
        'edge_u': (None if g.get('u_mean') is None or tb.get('u_mean') is None
                   else round(g['u_mean'] - tb['u_mean'], 5)),
        'greedy_wT': g.get('wT_mean'), 'greedy_p5': g.get('wT_p5'),
        'greedy_cvar5': g.get('wT_cvar5'), 'textbook_p5': tb.get('wT_p5'),
        'textbook_cvar5': tb.get('wT_cvar5'),
        'edge_p5': (None if g.get('wT_p5') is None or tb.get('wT_p5') is None
                    else round(g['wT_p5'] - tb['wT_p5'], 1)),
        'edge_cvar5': (None if g.get('wT_cvar5') is None or tb.get('wT_cvar5') is None
                       else round(g['wT_cvar5'] - tb['wT_cvar5'], 1)),
    }


def main():
    out = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]] or [7]
    template = json.load(open(CONFIG))
    rows = []
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for batch, batches in ARMS:
            for seed in seeds:
                row = arm(template, batch, batches, seed)
                rows.append(row)
                w.writerow(row)
                fh.flush()
                logging.info('ARM %dx%d seed=%d: edge_u=%s edge_cvar5=%s wall=%ss peak=%sGiB',
                             batch, batches, seed, row['edge_u'], row['edge_cvar5'],
                             row['wall_s'], row['peak_GiB'])
    print(f'\n{"arm":>12} {"fitted":>8} {"edge_u":>9} {"edge_p5":>11} {"edge_cvar5":>11} '
          f'{"wall_s":>8} {"peak":>6}')
    for r in rows:
        print(f'{r["batch"]:>6}x{r["batches"]:<5} {r["fitted"]:>8} {r["edge_u"]!s:>9} '
              f'{r["edge_p5"]!s:>11} {r["edge_cvar5"]!s:>11} {r["wall_s"]:>8} {r["peak_GiB"]:>6}')
    print('->', out)


if __name__ == '__main__':
    main()
