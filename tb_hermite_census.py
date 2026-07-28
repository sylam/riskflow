"""Hermite coefficient census: what would lazy / row-restricted g,c population actually save?

The design (§4) says g,c are materialised for the whole (scen x n_tenors x flat) block at cache
population time, before any consumer says which rows it will gather — 907 MiB at 32768 flat. Two
candidate fixes save very different amounts, so measure which:

  DEFERRAL  (build on first gather)      -> saves only the entries NEVER gathered.
  ROW-RESTRICTION (build gathered rows)  -> saves the un-gathered SCENARIO ROWS of every entry.

Instrumentation is runtime-only (wrap utils.hermite_interpolation_tensor + Interpolation.eval),
nothing in riskflow/ is modified.

    python tb_hermite_census.py [batch] [inner]
"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import riskflow as rf
from riskflow import utils

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
INNER = int(sys.argv[2]) if len(sys.argv) > 2 else 64
FIX = 'tests/fixtures/policy_test_simulate_only.json'

built = []                                   # one record per hermite_interpolation_tensor call
gathered = collections.Counter()             # id(Interpolation) -> gather calls
rows_seen = collections.defaultdict(set)     # id(Interpolation) -> flat rows actually indexed

_orig_hermite = utils.hermite_interpolation_tensor
_orig_init = utils.Interpolation.__init__
_orig_eval = utils.Interpolation.eval


def hermite(t, rate_tensor):
    g, c = _orig_hermite(t, rate_tensor)
    built.append({'shape': tuple(rate_tensor.shape),
                  'bytes': g.numel() * g.element_size() + c.numel() * c.element_size()})
    return g, c


def init(self, tensor, interp_params):
    _orig_init(self, tensor, interp_params)
    self._census_id = len(built)             # the record this object's params came from (1-based)


def ev(self, tenor_data, i1, i2, t_index, t_index_next, w2, tnr, alpha=None, time_factor=1.0):
    if tenor_data[0].startswith('Hermite'):
        cid = getattr(self, '_census_id', 0)
        gathered[cid] += 1
        idx = (t_index + i1)
        rows_seen[cid].update(torch.unique(idx.reshape(-1)).tolist())
    return _orig_eval(self, tenor_data, i1, i2, t_index, t_index_next, w2, tnr,
                      alpha=alpha, time_factor=time_factor)


utils.hermite_interpolation_tensor = hermite
utils.Interpolation.__init__ = init
utils.Interpolation.eval = ev
# the pricing module imported the symbol directly
import riskflow.pricing as pricing
if hasattr(pricing, 'hermite_interpolation_tensor'):
    pricing.hermite_interpolation_tensor = hermite

cfg = json.load(open(FIX))
c = cfg['Calc']['Calculation']
c.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': BATCH, 'Inner_Sub_Batch': INNER,
          'Inner_MC_Enabled': 'Yes', 'Inner_Antithetic': 'Yes', 'Random_Seed': 7})
c['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
c['Hedging_Problem']['Solver'] = {
    'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 9,
    'Training_Action_Chunk_Size': 64, 'DiffV2_Fit_Iters': 60, 'DiffV2_Hidden': 32,
    'DiffV2_OOS_Frac': 0.5, 'DiffV2_Cost_Aware_Argmax': 'Yes', 'DiffV2_One_Step_Fork': 'Yes',
    'DiffV2_Per_Column_Grad_Norm': 'Yes', 'T_Min': 114}
torch.cuda.reset_peak_memory_stats()
cx = rf.Context()
cx.load_json((json.dumps(cfg), f'census_{BATCH}x{INNER}.json'))
cx.run_job()

total = sum(b['bytes'] for b in built)
never = [i for i in range(1, len(built) + 1) if gathered[i] == 0]
never_bytes = sum(built[i - 1]['bytes'] for i in never)
row_frac = []
for i in range(1, len(built) + 1):
    if gathered[i]:
        scen, n_tenors = built[i - 1]['shape'][0], built[i - 1]['shape'][1]
        row_frac.append(len(rows_seen[i]) / max(1, scen * n_tenors))
print(json.dumps({
    'batch': BATCH, 'inner': INNER,
    'hermite_entries_built': len(built),
    'total_gc_GiB': round(total / 2**30, 3),
    'entries_never_gathered': len(never),
    'DEFERRAL_saves_GiB': round(never_bytes / 2**30, 3),
    'mean_gathered_row_fraction': round(sum(row_frac) / len(row_frac), 4) if row_frac else None,
    'ROW_RESTRICTION_saves_GiB': (
        round(total * (1 - (sum(row_frac) / len(row_frac))) / 2**30, 3) if row_frac else None),
    'peak_alloc_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 2),
    'largest_entry_MiB': round(max((b['bytes'] for b in built), default=0) / 2**20, 1),
}, indent=1))
