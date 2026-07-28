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

MODE = os.environ.get('CENSUS_MODE', 'bytes')       # 'bytes' | 'pattern'

# --- phase 0: WHICH rows does each gather touch? ------------------------------------------
# g/c are indexed ONLY by i00 (and i10 on the alpha branch) inside Interpolation.eval; the curve
# `tensor` itself is indexed by i00/i01/i10/i11 but is NOT a restriction candidate. So the window
# that matters is the union of the i00/i10 rows, decomposed as row = scen_index * n_tenors + tenor.
CURRENT = {'t': None, 'in_fork': False, 'key': None}
entries = {}                                        # id(Interpolation) -> record

_orig_hermite = utils.hermite_interpolation_tensor
_orig_init = utils.Interpolation.__init__
_orig_eval = utils.Interpolation.eval


def hermite(t, rate_tensor):
    g, c = _orig_hermite(t, rate_tensor)
    built.append({'shape': tuple(rate_tensor.shape),
                  'bytes': g.numel() * g.element_size() + c.numel() * c.element_size()})
    return g, c


def init(self, tensor, interp_params, row_offset=0):
    _orig_init(self, tensor, interp_params, row_offset=row_offset)
    self._census_id = len(built)             # the record this object's params came from (1-based)
    if interp_params:                        # hermite entries carry g,c
        entries[id(self)] = {
            'key': str(CURRENT['key']), 't': CURRENT['t'], 'in_fork': CURRENT['in_fork'],
            'scen': int(tensor.shape[0]), 'n_tenors': int(tensor.shape[1]),
            'param_rows': int(interp_params[0].shape[0]), 'row_offset': int(row_offset),
            'batch': int(tensor.shape[-1]), 'gc_rows': set(), 'gathers': 0}


def ev(self, tenor_data, i1, i2, t_index, t_index_next, w2, tnr, alpha=None, time_factor=1.0):
    if tenor_data[0].startswith('Hermite'):
        cid = getattr(self, '_census_id', 0)
        gathered[cid] += 1
        idx = (t_index + i1)
        rows_seen[cid].update(torch.unique(idx.reshape(-1)).tolist())
        rec = entries.get(id(self))
        if rec is not None:
            rec['gathers'] += 1
            rec['gc_rows'].update(torch.unique(idx.reshape(-1)).tolist())
            if t_index_next is not None:                  # the alpha branch also reads g/c at i10
                rec['gc_rows'].update(
                    torch.unique((t_index_next + i1).reshape(-1)).tolist())
    return _orig_eval(self, tenor_data, i1, i2, t_index, t_index_next, w2, tnr,
                      alpha=alpha, time_factor=time_factor)


_orig_mct = utils.make_curve_tensor


def mct(tensor, curve_component, time_grid, shared, n_batch_dims=1):
    prev = CURRENT['key']
    CURRENT['key'] = curve_component[1] if len(curve_component) > 1 else '?'
    try:
        return _orig_mct(tensor, curve_component, time_grid, shared, n_batch_dims)
    finally:
        CURRENT['key'] = prev


utils.hermite_interpolation_tensor = hermite
utils.make_curve_tensor = mct
utils.Interpolation.__init__ = init
utils.Interpolation.eval = ev
# the pricing module imported the symbol directly
import riskflow.pricing as pricing
for _mod in (pricing, utils):
    if getattr(_mod, 'hermite_interpolation_tensor', None) is not _orig_hermite:
        continue
    _mod.hermite_interpolation_tensor = hermite
if getattr(pricing, 'make_curve_tensor', None) is _orig_mct:
    pricing.make_curve_tensor = mct

# mark forks so a window can be scoped to them alone
from riskflow.calculation import HedgeMonteCarlo
_orig_fork = HedgeMonteCarlo._run_inner_mc_at_t


def fork(self, t, *a, **kw):
    prev_t, prev_f = CURRENT['t'], CURRENT['in_fork']
    CURRENT['t'], CURRENT['in_fork'] = int(t), True
    try:
        return _orig_fork(self, t, *a, **kw)
    finally:
        CURRENT['t'], CURRENT['in_fork'] = prev_t, prev_f


HedgeMonteCarlo._run_inner_mc_at_t = fork

WORLD = os.environ.get('CENSUS_WORLD', 'prod')     # 'goldenYes' | 'goldenNo' | 'prod'
cfg = json.load(open(FIX))
c = cfg['Calc']['Calculation']
if WORLD.startswith('golden'):                      # the bit-identity worlds: 48 x 8, shallow sweep
    BATCH, INNER = 48, 8
c.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': BATCH, 'Inner_Sub_Batch': INNER,
          'Inner_MC_Enabled': 'Yes', 'Inner_Antithetic': 'Yes', 'Random_Seed': 7})
c['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
c['Hedging_Problem']['Solver'] = {
    'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 9,
    'Training_Action_Chunk_Size': 64, 'DiffV2_Fit_Iters': 60, 'DiffV2_Hidden': 32,
    'DiffV2_OOS_Frac': 0.5, 'DiffV2_Cost_Aware_Argmax': 'Yes', 'DiffV2_One_Step_Fork': 'Yes',
    'DiffV2_Per_Column_Grad_Norm': 'Yes', 'T_Min': 114}
if WORLD.startswith('golden'):
    c['Hedging_Problem']['Solver'].update({
        'Training_Action_Grid_Levels_Per_Axis': 5, 'DiffV2_Fit_Iters': 5, 'T_Min': 108,
        'DiffV2_One_Step_Fork': 'Yes' if WORLD.endswith('Yes') else 'No',
        'Run_Hindsight_Diagnostic': 'Yes', 'Run_Textbook_Benchmark': 'Yes'})
    c['Random_Seed'] = 1234
torch.cuda.reset_peak_memory_stats()
cx = rf.Context()
cx.load_json((json.dumps(cfg), f'census_{BATCH}x{INNER}.json'))
cx.run_job()

if MODE == 'pattern':
    rows = []
    for rec in entries.values():
        if not rec['gathers']:
            continue
        nt, scen = rec['n_tenors'], rec['scen']
        gc = sorted(rec['gc_rows'])
        sc = sorted({r // nt for r in gc})
        tn = sorted({r % nt for r in gc})
        product = {s_ * nt + k for s_ in sc for k in tn}
        rows.append({
            'curve': rec['key'][:44], 't': rec['t'], 'in_fork': rec['in_fork'],
            'scen': scen, 'n_tenors': nt,
            'scen_lo': sc[0], 'scen_hi': sc[-1], 'n_scen_used': len(sc),
            'scen_contiguous': (sc[-1] - sc[0] + 1) == len(sc),
            'all_tenors': tn == list(range(nt)),
            'is_product': set(gc) == product,
            'rows_used': len(gc), 'rows_total': scen * nt,
            'frac': round(len(gc) / (scen * nt), 4),
            'window_lo_rel_t': (sc[0] - rec['t']) if rec['t'] is not None else None,
            'window_hi_rel_t': (sc[-1] - rec['t']) if rec['t'] is not None else None,
        })
    rows.sort(key=lambda r: (not r['in_fork'], r['curve'], r['t'] if r['t'] is not None else -1))
    import csv as _csv
    out = f'artifacts/walk_forward/hermite_pattern_{WORLD}.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    fork_rows = [r for r in rows if r['in_fork']]
    outer_rows = [r for r in rows if not r['in_fork']]
    print(json.dumps({
        'world': WORLD, 'batch': BATCH, 'inner': INNER,
        'gathered_entries': len(rows), 'in_fork': len(fork_rows), 'outer': len(outer_rows),
        'FORK scen_contiguous_all': all(r['scen_contiguous'] for r in fork_rows),
        'FORK all_tenors_all': all(r['all_tenors'] for r in fork_rows),
        'FORK is_product_all': all(r['is_product'] for r in fork_rows),
        'FORK window_lo_rel_t': sorted({r['window_lo_rel_t'] for r in fork_rows}),
        'FORK window_hi_rel_t': sorted({r['window_hi_rel_t'] for r in fork_rows}),
        'FORK mean_frac': (round(sum(r['frac'] for r in fork_rows) / len(fork_rows), 4)
                           if fork_rows else None),
        'OUTER mean_frac': (round(sum(r['frac'] for r in outer_rows) / len(outer_rows), 4)
                            if outer_rows else None),
        'OUTER scen_contiguous_all': all(r['scen_contiguous'] for r in outer_rows),
        'csv': out}, indent=1))
    sys.exit(0)

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
