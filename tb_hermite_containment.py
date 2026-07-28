"""Direct window-coverage check on the REAL walk-forward book (throwaway; tb_ prefix).

Asks the poison gate's question without the poison: for every Hermite gather inside a fork, is the
row it touches inside the window the derivation produced? Records the derived window and the
actual min/max row per gather, then reports any read outside it plus the tightest slack.

No NaN propagation, no disabled-window reference run, nothing to get subtly wrong — the poison
harness needed three fixes before it stopped reporting vacuous passes, so this is the measurement
that stands on its own.

    python tb_hermite_containment.py
"""
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from riskflow import utils
from riskflow.calculation import HedgeMonteCarlo

CUR = {'w': None, 'in_fork': False}
REC = []

_ow = HedgeMonteCarlo._hermite_window
_of = HedgeMonteCarlo._run_inner_mc_at_t
_oe = utils.Interpolation.eval


def w(self, cutoff, n):
    r = _ow(self, cutoff, n)
    CUR['w'] = r
    return r


def fork(self, t, *a, **kw):
    CUR['in_fork'] = True
    try:
        return _of(self, t, *a, **kw)
    finally:
        CUR['in_fork'], CUR['w'] = False, None


def ev(self, td, i1, i2, t_index, t_next, w2, tnr, alpha=None, time_factor=1.0):
    if td[0].startswith('Hermite') and CUR['in_fork'] and CUR['w'] is not None:
        nt = self.tensor.shape[1]
        idx = (t_index + i1).reshape(-1)
        if t_next is not None:
            idx = torch.cat([idx, (t_next + i1).reshape(-1)])
        if idx.numel():
            lo, hi = CUR['w']
            REC.append((int(lo), int(hi), int(idx.min()) // nt, int(idx.max()) // nt))
    return _oe(self, td, i1, i2, t_index, t_next, w2, tnr, alpha=alpha, time_factor=time_factor)


HedgeMonteCarlo._hermite_window = w
HedgeMonteCarlo._run_inner_mc_at_t = fork
utils.Interpolation.eval = ev

import runpy
sys.argv = ['production_walk_forward.py', '--spot-model', 'garch', '--start', '2020-01',
            '--months', '1', '--seeds', '7', '--batch', '512', '--streaming-batches', '5',
            '--fit-iters', '5', '--run-dir', '/tmp/containment_run']
runpy.run_path('production_walk_forward.py', run_name='__main__')

below = [r for r in REC if r[2] < r[0]]
above = [r for r in REC if r[3] > r[1]]
print('CONTAINMENT ' + json.dumps({
    'gathers_checked': len(REC),
    'reads_below_window': len(below),
    'reads_above_window': len(above),
    'tightest_slack_below_rows': min((r[2] - r[0]) for r in REC) if REC else None,
    'tightest_slack_above_rows': min((r[1] - r[3]) for r in REC) if REC else None,
    'VERDICT': 'CONTAINED' if not below and not above else 'HOLE'}))
