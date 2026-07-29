"""Golden-world bit-identity harness for the streaming wave (throwaway; tb_ prefix).

Dumps every tensor a hedge run produces — bundle series, every inner-MC fork output, the solver
diagnostics/verdict/artifact frame, and for simulate_only the stepper trajectory + diagnostic CSVs
— so a refactor can be proved bitwise against a baseline built from a pristine worktree.

`GOLDEN_REPO` selects which checkout to import riskflow from, so the baseline and the candidate
each run in their own process against their own code:

    GOLDEN_REPO=<repo-or-worktree> python tb_golden_worlds.py <world:Yes|No|sim> <out.pt>
    python tb_golden_worlds.py cmp <before.pt> <after.pt>

Lives in the repo rather than /tmp because two reboots wiped the scratchpad mid-wave and the
baselines had to be rebuilt from scratch each time.
"""
import json, logging, os, sys

REPO = os.environ.get('GOLDEN_REPO', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

FIX = os.path.join(REPO, 'tests/fixtures/policy_test_simulate_only.json')
BUNDLE_KEYS = (
    'time_grid_days', 'time_grid_days_cpu', 'tradables', 'factors', 'liability_mtm',
    'realized_cashflows', 'spot_price_history', 'spot_realized_vol', 'privileged_factors',
    'step_annual_vol', 'utility_scale', 'initial_time_index', 'last_live_mtm_index',
    'business_indices', 'total_leg_volume', 'last_settlement_index', 'scenario_dates',
    'calibrated_utility_inputs',
)


def snap(v):
    if torch.is_tensor(v):
        return v.detach().cpu().clone()
    if isinstance(v, dict):
        return {str(k): snap(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [snap(x) for x in v]
    return v


def bundle_snapshot(bundle):
    out = {k: snap(getattr(bundle, k, None)) for k in BUNDLE_KEYS}
    out['num_batches'] = bundle.num_batches
    out['base_date'] = str(bundle.base_date)
    out['scenario_dates'] = [str(d) for d in (bundle.scenario_dates if bundle.scenario_dates
                                              is not None else [])]
    return out


# The checkpoint the `eval` world loads. The BASELINE writes it (running `stream`) and BOTH trees
# then load that one file, so the frozen-eval comparison is about the eval path and not about
# whether training moved — `stream` is the golden that answers that.
CKPT = os.environ.get('GOLDEN_CKPT', '/tmp/golden_ckpt.pt')


def solve_cfg(world):
    """`stream` is a training stream; `eval` is a frozen stream of length one. Both emit keys a
    PRE-change tree still needs and a post-change tree ignores (unknown solver keys are dropped at
    the boundary), so one config drives either side of a comparison."""
    cfg = json.load(open(FIX))
    c = cfg['Calc']['Calculation']
    c.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': 48, 'Inner_Sub_Batch': 8,
              'Inner_MC_Enabled': 'Yes', 'Inner_Antithetic': 'Yes', 'Random_Seed': 1234})
    c['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
    c['Hedging_Problem']['Solver'] = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64, 'T_Min': 108, 'DiffV2_Fit_Iters': 5,
        'DiffV2_OOS_Frac': 0.5, 'DiffV2_One_Step_Fork': 'Yes',
        'Run_Hindsight_Diagnostic': 'Yes', 'Run_Textbook_Benchmark': 'Yes'}
    if world == 'stream':
        c['Batch_Size'], c['Simulation_Batches'] = 24, 3
        c['Hedging_Problem']['Solver'].update(
            {'DiffV2_Streaming_Batches': 'Yes', 'DiffV2_Save_Value_Fn': CKPT})
    elif world == 'eval':
        c['Batch_Size'], c['Simulation_Batches'] = 24, 1
        c['Hedging_Problem']['Solver']['DiffV2_Load_Value_Fn'] = CKPT
    return cfg


def run_solve(world, out):
    import riskflow as rf
    from riskflow.calculation import HedgeMonteCarlo
    dump = []
    orig = HedgeMonteCarlo._run_inner_mc_at_t

    def at(self, t, buf, shared, base, refs, *a, **kw):
        r = orig(self, t, buf, shared, base, refs, *a, **kw)
        rec = {'t': int(t), 'grad': bool(kw.get('with_grad', False)),
               'rows': kw.get('outer_rows')}
        for k in ('L_t', 'L_t1', 'L_T', 'market_t', 'market_t1'):
            rec[k] = snap(r.get(k))
        rec['F_t1'] = snap(r.get('F_t1') or {})
        dump.append(rec)
        return r

    HedgeMonteCarlo._run_inner_mc_at_t = at
    cx = rf.Context()
    cx.load_json((json.dumps(solve_cfg(world)), f'golden_{world}.json'))
    _, result = cx.run_job()
    ev = result.evaluation_summary or {}
    diag = ev.get('diagnostics') or {}
    keep = {k: snap(diag.get(k)) for k in
            ('V_0', 'n_star_0', 'bounded', 'max_abs_Y_boot', 'market_dim', 'root_t',
             'action_grid_size', 'per_t', 'verdict')}
    keep['comparison'] = snap(ev.get('comparison'))
    keep['ladder'] = snap(ev.get('ladder'))
    keep['actions'] = snap(ev.get('actions'))
    keep['values'] = snap(ev.get('values'))
    art = result.policy_artifact or {}
    keep['artifact'] = {k: snap(art.get(k)) for k in
                        ('m_mean', 'm_std', 'w_mean', 'w_std', 'utility_scale', 'a_bounds',
                         'V_0', 'n_star_0', 'max_abs_Y_boot', 'T_dec', 'md')}
    torch.save({'bundle': bundle_snapshot(result.bundle), 'forks': dump, 'diag': keep}, out)
    print(f'{world}: {len(dump)} forks, V_0={keep["V_0"]!r} -> {out}')


def run_sim(out):
    import riskflow as rf
    cfg = json.load(open(FIX))
    cfg['Calc']['Calculation']['Random_Seed'] = 1234
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), 'golden_sim.json'))
    _, result = cx.run_job()
    rec = {'bundle': bundle_snapshot(result.bundle), 'summary': snap(result.evaluation_summary)}
    stepper = result.create_stepper()
    hedges = list(result.runtime['names']['hedges'])
    traj, k = [], 0
    while not stepper.done:
        if stepper.is_decision_step:
            k += 1
            obs = stepper.step({hedges[k % len(hedges)]: -3.0} if k % 3 == 0 else None)
        else:
            obs = stepper.step(None)
        traj.append({'t': obs['time_index'], 'pos': snap(obs['positions']),
                     'pnl': snap(obs['transition_pnl_excess']),
                     'liab': snap(obs['transition_liability_value'])})
    rec['traj'] = traj
    rec['state'] = {k: snap(v) for k, v in stepper._state.items()}
    d = os.path.join('/tmp', '_golden_csv')
    stepper.write_diagnostic_csvs(d, label='golden')
    rec['csv'] = {f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d))}
    torch.save(rec, out)
    print(f'sim: {len(traj)} steps, {len(rec["csv"])} csvs -> {out}')


def walk(a, b, path, bad):
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        bad.append(f'{path}: type {type(a).__name__} != {type(b).__name__}')
        return
    if torch.is_tensor(a):
        if a.shape != b.shape:
            bad.append(f'{path}: shape {tuple(a.shape)} != {tuple(b.shape)}')
        elif not torch.equal(a, b):
            d = ((a.to(torch.float64) - b.to(torch.float64)).abs().max()
                 if a.is_floating_point() else -1)
            bad.append(f'{path}: NOT BITWISE (max|Δ|={float(d):.3g})')
    elif isinstance(a, dict):
        if set(a) != set(b):
            bad.append(f'{path}: keys {sorted(set(a) ^ set(b))}')
        for k in sorted(set(a) & set(b)):
            walk(a[k], b[k], f'{path}.{k}', bad)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            bad.append(f'{path}: len {len(a)} != {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}[{i}]', bad)
    elif isinstance(a, float):
        if repr(a) != repr(b):
            bad.append(f'{path}: {a!r} != {b!r} (Δ={b - a:+.3g})')
    elif a != b:
        bad.append(f'{path}: {a!r} != {b!r}')


IGNORE = ('.summary.timing.evaluation_time_seconds',)


def _leaves(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _leaves(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _leaves(x)
    else:
        yield v


def cmp(f1, f2):
    a, b = torch.load(f1, weights_only=False), torch.load(f2, weights_only=False)
    bad = []
    walk(a, b, '', bad)
    bad = [m for m in bad if not m.startswith(IGNORE)]
    n = sum(1 for _ in _leaves(a))
    if bad:
        print(f'MISMATCH ({len(bad)}) over {n} leaves:')
        for m in bad[:40]:
            print('  ' + m)
        sys.exit(1)
    print(f'BITWISE IDENTICAL over {n} leaves: {os.path.basename(f1)} == {os.path.basename(f2)}')


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    if sys.argv[1] == 'cmp':
        cmp(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'sim':
        run_sim(sys.argv[2])
    else:
        run_solve(sys.argv[1], sys.argv[2])
