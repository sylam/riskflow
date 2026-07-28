"""Where does ~20 GiB land? Operating-point ladder for the single-pass inner-MC fork.

With chunking deleted, fork peak is a function of exactly two JSON fields — `Batch_Size` and
`Inner_Sub_Batch` — so the recommended production shape is a MEASUREMENT, not a default. Each
rung runs in its OWN process (peak counters start clean) at the production solver recipe, times
every `_fit_step`, and reports peak allocated/reserved GiB plus GPU utilization.

    python tb_operating_point.py ladder            # spawn one process per rung, write the CSV
    python tb_operating_point.py rung <B> <I>      # one rung (used by `ladder`)

Outputs: artifacts/walk_forward/operating_point.csv
"""
import csv, json, os, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

FIX = 'tests/fixtures/policy_test_simulate_only.json'
OUT = 'artifacts/walk_forward/operating_point.csv'
# Fork width x inner draws around the 24 GiB card. The first nine are the historic ladder, kept
# so rows stay comparable; the last three probe the headroom the block-routed fork opened up —
# 2048x64 no longer stresses the card, and `Inner_Sub_Batch` is the lever selection quality moves on.
RUNGS = [(512, 64), (768, 64), (1024, 64), (1280, 64), (1536, 64), (2048, 64),
         (1024, 96), (1024, 128), (768, 128),
         (3072, 64), (4096, 64), (1280, 256)]
N_FIT = 3                      # fit steps per rung: T_Min = T_dec - N_FIT (fixture T_dec=117)


def rung(batch, inner):
    import torch
    import riskflow as rf
    from riskflow.hedge_solver import DiffSolverV2

    times = []
    orig = DiffSolverV2._fit_step

    def timed(self, *a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig(self, *a, **kw)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        return r

    DiffSolverV2._fit_step = timed

    utils_seen = []
    stop = threading.Event()

    def sample():                                     # GPU utilization while the rung runs
        while not stop.is_set():
            try:
                u = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                    '--format=csv,noheader,nounits', '-i', '0'],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
                utils_seen.append(int(u))
            except Exception:
                pass
            stop.wait(1.0)

    cfg = json.load(open(FIX))
    c = cfg['Calc']['Calculation']
    c.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': batch, 'Inner_Sub_Batch': inner,
              'Inner_MC_Enabled': 'Yes', 'Inner_Antithetic': 'Yes', 'Random_Seed': 7})
    c['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
    c['Hedging_Problem']['Solver'] = {          # the production recipe (production_solver.BEST_*)
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 9,
        'Training_Action_Chunk_Size': 64, 'DiffV2_Fit_Iters': 60, 'DiffV2_Hidden': 32,
        'DiffV2_LR': 0.002, 'DiffV2_OOS_Frac': 0.5, 'DiffV2_Cost_Aware_Argmax': 'Yes',
        'DiffV2_One_Step_Fork': 'Yes', 'DiffV2_Per_Column_Grad_Norm': 'Yes',
        'T_Min': 117 - N_FIT}                   # fixture T_dec = 117 (measured)
    torch.cuda.reset_peak_memory_stats()
    t = threading.Thread(target=sample, daemon=True); t.start()
    t0 = time.perf_counter()
    ok, err = True, ''
    try:
        cx = rf.Context()
        cx.load_json((json.dumps(cfg), f'op_{batch}x{inner}.json'))
        cx.run_job()
    except Exception as e:                       # a too-wide config is expected to OOM loudly
        ok, err = False, f'{type(e).__name__}: {str(e)[:120]}'
    wall = time.perf_counter() - t0
    stop.set(); t.join(timeout=3)
    row = {
        'batch': batch, 'inner': inner, 'flat': batch * inner, 'ok': ok,
        'peak_alloc_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 2),
        'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2**30, 2),
        's_per_fit_step': round(sum(times) / len(times), 3) if times else None,
        'fit_steps': len(times), 'wall_s': round(wall, 1),
        'gpu_util_mean': round(sum(utils_seen) / len(utils_seen), 1) if utils_seen else None,
        'gpu_util_peak': max(utils_seen) if utils_seen else None,
        'paths_per_s': (round(batch / (sum(times) / len(times)), 1) if times else None),
        'error': err,
    }
    print('RUNG ' + json.dumps(row), flush=True)
    return row


def ladder():
    rows = []
    for b, i in RUNGS:
        p = subprocess.run([sys.executable, __file__, 'rung', str(b), str(i)],
                           capture_output=True, text=True,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
        line = next((l for l in p.stdout.splitlines() if l.startswith('RUNG ')), None)
        if line:
            rows.append(json.loads(line[5:]))
        else:                                    # a hard OOM can take the process down with it
            rows.append({'batch': b, 'inner': i, 'flat': b * i, 'ok': False,
                         'error': (p.stderr.strip().splitlines() or ['killed'])[-1][:120]})
        print(f'{b}x{i}: {rows[-1]}', flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keys = ['batch', 'inner', 'flat', 'ok', 'peak_alloc_GiB', 'peak_reserved_GiB',
            's_per_fit_step', 'fit_steps', 'wall_s', 'gpu_util_mean', 'gpu_util_peak',
            'paths_per_s', 'error']
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f'-> {OUT}')


if __name__ == '__main__':
    if sys.argv[1] == 'rung':
        rung(int(sys.argv[2]), int(sys.argv[3]))
    else:
        ladder()
