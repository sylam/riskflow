"""NaN-poison sufficiency + aliasing gate for the derived Hermite window (throwaway; tb_ prefix).

The runtime guard only catches gathers that fall OUTSIDE the windowed block's index range. It
cannot catch ALIASING: a wrong offset that maps an out-of-window absolute row onto a valid
in-window index returns a wrong number silently. This closes that, on the real walk-forward book:

  run A (clean)    window ON  — the shipping path.
  run B (poison)   window OFF so the FULL g,c block is built, then every row OUTSIDE the derived
                   window is set to NaN before any gather can reach it.

Then every fork output of A must be bitwise equal to B. If anything reads outside the window, B
carries NaN where A has a number. If A aliases onto the wrong row, B (reading the right, unpoisoned
row) disagrees. Only the g,c coefficients are poisoned — the curve tensor itself is never windowed,
so poisoning it would fail legitimate reads and prove nothing.

    python tb_hermite_poison.py clean|poison <out.json>
    python tb_hermite_poison.py cmp <clean.json> <poison.json>
"""
import hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

MODE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
import time as _time
# Unique per invocation. The walk-forward driver is idempotent per seed checkpoint, so a
# reused run dir SKIPS training entirely — 0 forks, nothing poisoned, and a gate that
# 'passes' without testing anything. Caught exactly that way once.
RUN = f'/tmp/hermite_poison_{int(_time.time())}'


def h(t):
    if t is None:
        return None
    a = t.detach().cpu().contiguous()
    return hashlib.sha1(a.numpy().tobytes()).hexdigest()[:16]


def cmp(f1, f2):
    a, b = json.load(open(f1)), json.load(open(f2))
    if len(a) != len(b):
        print(f'DIRTY: fork counts differ {len(a)} vs {len(b)}')
        sys.exit(1)
    bad = [(x['t'], x['grad'], k) for x, y in zip(a, b) for k in x
           if k not in ('t', 'grad') and x[k] != y[k]]
    print(f'compared {len(a)} forks x {len(a[0]) - 2} output tensors')
    if bad:
        print(f'DIRTY ({len(bad)} mismatches):')
        for t, g, k in bad[:15]:
            print(f'  t={t} grad={g} field={k}')
        sys.exit(1)
    print('POISON GATE CLEAN — every fork output bitwise equal; nothing reads outside the '
          'derived window and the windowed gather does not alias')


if MODE == 'cmp':
    cmp(sys.argv[2], sys.argv[3])
    sys.exit(0)

from riskflow import utils
from riskflow.calculation import HedgeMonteCarlo

DUMP = []
CUR = {'window': None}
STATS = {'entries': 0, 'rows_poisoned': 0, 'rows_total': 0}

_orig_window = HedgeMonteCarlo._hermite_window
_orig_interp_init = utils.Interpolation.__init__
_orig_fork = HedgeMonteCarlo._run_inner_mc_at_t


def window(self, cutoff_idx, n_inner_rows):
    w = _orig_window(self, cutoff_idx, n_inner_rows)
    CUR['window'] = w
    # poison mode disables the window so the FULL block is built, then poisons it below
    return None if MODE == 'poison' else w


def interp_init(self, tensor, interp_params, row_offset=0):
    """Poison at the coefficients themselves. `make_curve_tensor` returns a gathered CurveTensor,
    not the Interpolation, so the g,c live only on the object built here — poisoning the return
    value silently did nothing, which the accounting caught."""
    _orig_interp_init(self, tensor, interp_params, row_offset=row_offset)
    w = CUR['window']
    if MODE not in ('poison', 'control') or w is None or not self.interp_params:
        return
    lo, hi = w
    n_tenors = tensor.shape[1]
    rows = self.interp_params[0].shape[0]
    lo_flat, hi_flat = max(0, lo * n_tenors), min(rows, (hi + 1) * n_tenors)
    if MODE == 'control':
        # POSITIVE CONTROL: poison the last scenario row INSIDE the window — a row the fork does
        # read — so the comparison MUST go dirty. Without this, a clean 'poison' proves nothing.
        lo_flat, hi_flat = max(0, (hi - 1) * n_tenors), rows
    for q in self.interp_params:
        q[:lo_flat] = float('nan')
        q[hi_flat:] = float('nan')
    STATS['entries'] += 1
    STATS['rows_poisoned'] += lo_flat + max(0, rows - hi_flat)
    STATS['rows_total'] += rows


def fork(self, t, *a, **kw):
    r = _orig_fork(self, t, *a, **kw)
    rec = {'t': int(t), 'grad': bool(kw.get('with_grad', False))}
    for k in ('L_t', 'L_t1', 'L_T', 'market_t', 'market_t1'):
        rec[k] = h(r.get(k))
    for ref, v in sorted((r.get('F_t1') or {}).items()):
        rec[f'F_t1[{ref}]'] = h(v)
    DUMP.append(rec)
    return r


HedgeMonteCarlo._hermite_window = window
HedgeMonteCarlo._run_inner_mc_at_t = fork
utils.Interpolation.__init__ = interp_init

import runpy
sys.argv = ['production_walk_forward.py', '--spot-model', 'garch', '--start', '2020-01',
            '--months', '1', '--seeds', '7', '--batch', '512', '--streaming-batches', '5',
            '--fit-iters', '5', '--run-dir', f'{RUN}_{MODE}']
runpy.run_path('production_walk_forward.py', run_name='__main__')
if not DUMP:
    raise SystemExit(f'{MODE}: NO FORKS RECORDED — the run trained nothing (stale run dir?); '
                     f'refusing to write a vacuous dump')
if MODE in ('poison', 'control') and not STATS['entries']:
    raise SystemExit(f'{MODE}: NOTHING WAS POISONED — the gate would be vacuous; refusing')
json.dump(DUMP, open(OUT, 'w'))
print(f'{MODE}: {len(DUMP)} forks dumped -> {OUT}')
if STATS['entries']:
    print(f"  poisoned {STATS['rows_poisoned']} of {STATS['rows_total']} g,c rows across "
          f"{STATS['entries']} curve entries "
          f"({100.0 * STATS['rows_poisoned'] / STATS['rows_total']:.1f}%)")
