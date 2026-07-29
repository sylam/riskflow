"""Hermite row census on the shipped lazy build (throwaway; tb_ prefix).

What the deferred `Interpolation.hermite_params` actually does at runtime, per coefficient object:
how many scenario rows it ends up covering against the full block, how many times it (re)built to
get there, and how many gathers each build serves. The inner-MC fork is separated from everything
else, because the fork is the sparse consumer and base valuation / credit MC / the outer loop are
the full-block ones that must not regress.

    python tb_hermite_lazy_probe.py fixture [batch] [inner]      # the solve_hedge fixture
    CENSUS_WORLD=goldenYes python tb_hermite_lazy_probe.py fixture
    python tb_hermite_lazy_probe.py wf [batch]                   # the real walk-forward book
"""
import collections, json, os, runpy, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import riskflow as rf
from riskflow import utils
from riskflow.calculation import HedgeMonteCarlo

MODE = sys.argv[1] if len(sys.argv) > 1 else 'fixture'
CURRENT = {'in_fork': False}
state = {}

_orig_params = utils.Interpolation.hermite_params
_orig_fork = HedgeMonteCarlo._run_inner_mc_at_t


def params(self, i00, i10):
    before = self.rows
    out = _orig_params(self, i00, i10)
    rec = state.setdefault(id(self), {
        'in_fork': CURRENT['in_fork'], 'scen': int(self.tensor.shape[0]),
        'batch': int(self.tensor.shape[-1]), 'builds': 0, 'rows_built': 0, 'evals': 0})
    rec['evals'] += 1
    if self.rows != before:
        rec['builds'] += 1
        rec['rows_built'] += self.rows[1] - self.rows[0] + 1
    rec['rows'] = self.rows
    return out


def fork(self, t, *a, **kw):
    prev = CURRENT['in_fork']
    CURRENT['in_fork'] = True
    try:
        return _orig_fork(self, t, *a, **kw)
    finally:
        CURRENT['in_fork'] = prev


utils.Interpolation.hermite_params = params
HedgeMonteCarlo._run_inner_mc_at_t = fork


def run_fixture(batch, inner):
    world = os.environ.get('CENSUS_WORLD', 'prod')
    cfg = json.load(open('tests/fixtures/policy_test_simulate_only.json'))
    c = cfg['Calc']['Calculation']
    if world.startswith('golden'):
        batch, inner = 48, 8
    # Batch_Size IS the measured axis here, so it is preserved and the stream is the shortest
    c.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': batch, 'Simulation_Batches': 2,
              'Inner_Sub_Batch': inner,
              'Inner_MC_Enabled': 'Yes', 'Inner_Antithetic': 'Yes', 'Random_Seed': 7})
    c['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
    c['Hedging_Problem']['Solver'] = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 9,
        'Training_Action_Chunk_Size': 64, 'DiffV2_Fit_Iters': 60, 'DiffV2_Hidden': 32,
        'DiffV2_Cost_Aware_Argmax': 'Yes', 'DiffV2_One_Step_Fork': 'Yes',
        'DiffV2_Per_Column_Grad_Norm': 'Yes', 'T_Min': 114}
    if world.startswith('golden'):
        c['Hedging_Problem']['Solver'].update({
            'Training_Action_Grid_Levels_Per_Axis': 5, 'DiffV2_Fit_Iters': 5, 'T_Min': 108,
            'DiffV2_One_Step_Fork': 'Yes' if world.endswith('Yes') else 'No',
            'Run_Hindsight_Diagnostic': 'Yes', 'Run_Textbook_Benchmark': 'Yes'})
        c['Random_Seed'] = 1234
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), f'census_{batch}x{inner}.json'))
    cx.run_job()
    return {'world': world, 'batch': batch, 'inner': inner}


def run_wf(batch):
    sys.argv = ['production_walk_forward.py', '--spot-model', 'garch', '--start', '2020-01',
                '--months', '1', '--seeds', '7', '--batch', str(batch),
                '--run-dir', '/tmp/hermite_census_wf']
    runpy.run_path('production_walk_forward.py', run_name='__main__')
    return {'world': 'walk_forward', 'batch': batch}


def summarise(rows, label):
    if not rows:
        return {f'{label}_objects': 0}
    span = [r['rows'][1] - r['rows'][0] + 1 for r in rows]
    return {
        f'{label}_objects': len(rows),
        f'{label}_builds': dict(collections.Counter(r['builds'] for r in rows)),
        f'{label}_evals': dict(collections.Counter(r['evals'] for r in rows)),
        f'{label}_mean_rows_covered': round(sum(span) / len(rows), 2),
        f'{label}_mean_scen': round(sum(r['scen'] for r in rows) / len(rows), 2),
        f'{label}_rows_built': sum(r['rows_built'] for r in rows),
        f'{label}_rows_if_eager': sum(r['scen'] for r in rows),
        f'{label}_row_frac': round(sum(span) / sum(r['scen'] for r in rows), 4),
    }


head = run_wf(int(sys.argv[2]) if len(sys.argv) > 2 else 1280) if MODE == 'wf' else run_fixture(
    int(sys.argv[2]) if len(sys.argv) > 2 else 1280, int(sys.argv[3]) if len(sys.argv) > 3 else 64)
out = dict(head, coefficient_objects=len(state))
out.update(summarise([r for r in state.values() if r['in_fork']], 'FORK'))
out.update(summarise([r for r in state.values() if not r['in_fork']], 'OUTER'))
print('CENSUS ' + json.dumps(out))
