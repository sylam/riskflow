"""Peak memory + wall for one walk-forward trade (throwaway; tb_ prefix).

The Hermite-row wave's operating point is a MEASUREMENT on the real book, not on the fixture, so
this runs `production_walk_forward` in-process and reports the CUDA peak counters the driver does
not record. Baseline to beat (commit e194893, 1280x64, trade 202001):
peak alloc 6.30 GiB, peak reserved 7.54 GiB, wall 115.6 s, greedy -100.52.

    python tb_hermite_measure.py <run-dir> [batch] [extra production_walk_forward args...]
"""
import json, os, runpy, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

if os.environ.get('HERMITE_EAGER') == '1':
    # Price the revert: build g,c for the WHOLE scenario block on the first gather, which is what
    # the pre-optimisation code did and what dropping row restriction entirely would cost.
    from riskflow import utils

    def full_block(self, i00, i10):
        if self.rows is None:
            g, c = utils.hermite_interpolation_tensor(self.hermite_tenor, self.tensor)
            self.interp_params = [p.reshape(-1, p.shape[-1]) for p in (g, c)]
            self.rows = (0, self.tensor.shape[0] - 1)
        return self.interp_params[0], self.interp_params[1], 0

    utils.Interpolation.hermite_params = full_block

RUN = sys.argv[1]
BATCH = sys.argv[2] if len(sys.argv) > 2 else '1280'
EXTRA = sys.argv[3:]

sys.argv = ['production_walk_forward.py', '--spot-model', 'garch', '--start', '2020-01',
            '--months', '1', '--seeds', '7', '--batch', BATCH, '--run-dir', RUN] + EXTRA
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
runpy.run_path('production_walk_forward.py', run_name='__main__')
wall = time.perf_counter() - t0
row = json.load(open(os.path.join(RUN, 'row_202001.json')))
print('MEASURE ' + json.dumps({
    'batch': int(BATCH), 'extra': EXTRA, 'wall_s': round(wall, 1),
    'peak_alloc_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 2),
    'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2**30, 2),
    'greedy_usd_oz': row.get('greedy_usd_oz'), 'V_0': row.get('V_0'),
    'train_u': row.get('train_u'), 'churn': row.get('churn'),
    'nohedge_usd_oz': row.get('nohedge_usd_oz'), 'pf_bound': row.get('pf_bound')}))
