"""STEP 3b: pure value of cost-aware roll-time relocation (throwaway; no commit).

Same frozen checkpoints, same world (realistic-BASE bundle: identical prices + vol series for a
given month/band). Replay TWO decision schedules on that one world and price both under the SAME
realistic realized cost:
  net_reloc = replay(realistic greedy schedule)   [decisions made cost-aware @ realistic spreads]
  net_naive = replay(flat10   greedy schedule)     [decisions made cost-aware @ flat 10bps]
reloc_value = net_reloc - net_naive  ($/oz) = pure P&L from the frozen policy relocating off the
expensive M3 once it is charged real per-maturity spreads at decision time (cost model held fixed,
so this is NOT the cost-accounting change — it is the decision change alone).
"""
import os, sys, json, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import riskflow as rf
assert 'PycharmProjects' in rf.__file__

from production_walk_forward import build_corrected_archive
from tb_garch_corridor import ARCHIVE, WF, month_list, build_month_map
from tb_cost_frontier import cost_blocks
from tb_cost_decomp import run_full, build_roll_cfg, replay_net, VOL

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
FBASE = os.path.join(WF, 'net_of_cost')


def flat10_qmap(month, band_str):
    r = json.load(open(os.path.join(FBASE, f'tb_row_flat10_{band_str}_{month}.json')))
    return {int(t): q for t, q in zip(r['greedy_q_t'], r['greedy_q_traj'])}, r['greedy']


def relocation(month, band, arch, mmap, base):
    bstr = 'free' if band is None else f'{int(round(band*100)):03d}'
    run_dir = os.path.join(base, f'run_reloc_{bstr}')
    os.makedirs(run_dir, exist_ok=True)
    roll = build_roll_cfg(month, band, cost_blocks()['base'], arch, mmap, run_dir)
    res = run_full(roll, f'reloc_{month}')
    bundle, runtime = res.bundle, res.runtime
    sv = ((res.evaluation_summary or {}).get('diagnostics') or {}).get('stepper_verdict') or {}
    hedges = list(runtime['names']['hedges'])
    q_reloc = {int(t): q for t, q in zip(sv['greedy_q_t'], sv['greedy_q_traj'])}
    q_naive, flat_net = flat10_qmap(month, bstr)

    net_reloc = replay_net(bundle, runtime, q_reloc, hedges)     # == diag greedy (realistic)
    net_naive = replay_net(bundle, runtime, q_naive, hedges)     # flat10 decisions @ realistic cost
    row = {'tag': month, 'band': bstr,
           'net_reloc_oz': round(net_reloc / VOL, 4),
           'net_naive_at_realistic_oz': round(net_naive / VOL, 4),
           'reloc_value_oz': round((net_reloc - net_naive) / VOL, 4),
           'flat10_at_flat10_oz': flat_net}
    logging.info('RELOC %s b%s: reloc=%.3f naive@realistic=%.3f -> value=%.3f/oz (flat10@flat10=%.2f)',
                 month, bstr, row['net_reloc_oz'], row['net_naive_at_realistic_oz'],
                 row['reloc_value_oz'], flat_net)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', nargs='*', default=['0.15', '0.60'])
    ap.add_argument('--months', nargs='*', default=None)
    ap.add_argument('--base', default=FBASE)
    ap.add_argument('--out', default=os.path.join(FBASE, 'relocation.csv'))
    args = ap.parse_args()
    raw = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    mmap = build_month_map()
    months = args.months or (month_list('2020-01', 24) + month_list('2022-01', 24))
    rows = []
    for band in args.bands:
        b = None if band == 'free' else float(band)
        for m in months:
            rows.append(relocation(m, b, arch, mmap, args.base))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    logging.info('WROTE %s (%d rows)', args.out, len(rows))


if __name__ == '__main__':
    main()
