"""Where a fork's wall and bytes go: phase split + slab census for the inner-MC fork.

The fork runs ONE pass at Batch_Size x Inner_Sub_Batch — no chunk loop — and publishes each
factor's grid as a sequence of scenario-row blocks (the outer-realized past at B_outer, then the
forked rows at B_flat) rather than a joined grid. This profile is how you see whether a slab you
expect to be block-width is: the census names every live tensor by buffer and by the batch width
it scales with.

WHAT IT MEASURES
  phases  (a) per-fork wall time (cuda.synchronize-bracketed) + live-memory deltas, split
              rng / generation / publication / pricing / extraction, aggregated over the run
              and split grad vs no-grad. Also (d): the slab census (below).
  trace   (b) one torch.profiler trace (CPU+CUDA) over a few forks -> chrome trace +
              key_averages + a GPU-busy fraction and kernel-launch count, to attribute idle
              time (launch-bound vs CPU-side publication vs H2D).
  census  (d) at the publication boundary and at the pricing peak, enumerate live device
              tensors: first the NAMED framework buffers (t_Scenario_Buffer per factor key,
              t_Buffer, cashflows, RNG), then a gc sweep of every live tensor deduped by
              STORAGE, so whatever is left is reported as genuinely unattributed.

The (Batch_Size, Inner_Sub_Batch) operating-point ladder lives in `tb_operating_point.py`; this
script does not duplicate it.

READ-ONLY: nothing in riskflow/ is modified. Instrumentation is runtime-only and confined to
this throwaway: one class-level hook (_run_inner_mc_at_t) restored in a finally, and
instance-level shadowing of bound methods on the LIVE calc / shared_mem / process /
DealStructure objects (an instance attribute shadows the class method; the class is untouched).
The phase boundaries are all observable at object call boundaries:
    fork enter -> reset_inner exit -> last generate/reseed exit -> first _restricted_struct
    enter -> last resolve_*/tensor_marks exit -> fork exit
which brackets the publication loop (pure torch between the last generate and the first
_restricted_struct) without needing a hook inside the function body.

ACCURACY NOTE: `phases` syncs at every marker, so its ABSOLUTE wall is inflated -- read it
for the SPLIT.

Deterministic: fixed seeds, fixed config, no adaptive anything.

Usage (see --help):
    python tb_profile_fork.py phases                     # (a) + (d)
    python tb_profile_fork.py trace                      # (b)
    python tb_profile_fork.py all
Outputs land in artifacts/walk_forward/fork_profile/.
"""
import argparse
import csv
import datetime
import gc
import json
import logging
import os
import subprocess
import threading
import time

# Match production_solver.py: parsed once at first allocator use, so it must precede torch.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch

import riskflow as rf
from riskflow.calculation import HedgeMonteCarlo

ROOT = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')
OUTDIR = os.path.join(ROOT, 'artifacts', 'walk_forward', 'fork_profile')
CUDA = torch.cuda.is_available()
MIB = 1024.0 ** 2
GIB = 1024.0 ** 3
PHASES = ('rng', 'generation', 'publication', 'pricing', 'extraction')


def sync():
    if CUDA:
        torch.cuda.synchronize()


def live_bytes():
    return torch.cuda.memory_allocated() if CUDA else 0


def peak_bytes():
    return torch.cuda.max_memory_allocated() if CUDA else 0


def reset_peak():
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def stamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


# ---- config ---------------------------------------------------------------------------------
def build_cfg(batch, inner, t_min, fit_iters, seed=1234, dual_strip=False):
    """The canonical fixture world, overridden in code (fixture is a TEMPLATE, never edited).
    `dual_strip` adds the LME leg via validate_cross_market.add_lme_leg -- OPT-IN because that
    helper still emits CommodityFutureDeal.Implied_Basis, which was removed from the schema."""
    cfg = json.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    # Batch_Size IS the measured axis here, so it is preserved and the stream is the shortest
    calc['Batch_Size'], calc['Simulation_Batches'] = int(batch), 2
    calc['Inner_Sub_Batch'] = int(inner)
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = 'Yes'
    calc['Random_Seed'] = int(seed)
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 9,
        'Training_Action_Chunk_Size': 64,
        'T_Min': int(t_min),
        'DiffV2_Fit_Iters': int(fit_iters),
        'DiffV2_Hidden': 32,
        'DiffV2_LR': 0.002,
    }
    if dual_strip:
        from validate_cross_market import add_lme_leg
        add_lme_leg(cfg)
    return cfg


def run_job(cfg, name):
    cx = rf.Context()
    cx.load_json((json.dumps(cfg, default=str), name + '.json'))
    _, result = cx.run_job()
    return result


# ---- tensor walking / census ----------------------------------------------------------------
def iter_tensors(obj, path='', depth=0, seen=None):
    """Yield (path, tensor) from nested dicts / sequences / simple objects. Depth-limited and
    cycle-guarded -- enough to reach t_Buffer's CurveTensor-ish payloads."""
    if seen is None:
        seen = set()
    if depth > 4 or id(obj) in seen:
        return
    seen.add(id(obj))
    if torch.is_tensor(obj):
        yield path, obj
        return
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(obj, '__dict__'):
        items = vars(obj).items()
    elif hasattr(obj, '_fields'):                                   # namedtuple without __dict__
        items = ((f, getattr(obj, f)) for f in obj._fields)
    else:
        return
    for k, v in items:
        yield from iter_tensors(v, f'{path}.{k}' if path else str(k), depth + 1, seen)


def _key_name(k):
    try:
        from riskflow import utils
        if hasattr(k, 'type') and hasattr(k, 'name'):
            return utils.check_tuple_name(k)
    except Exception:
        pass
    return str(k)


def _storage_id(t):
    try:
        s = t.untyped_storage()
        return (s.data_ptr(), s.nbytes())
    except Exception:
        return (t.data_ptr(), t.element_size() * t.numel())


def census(shared_mem, tag, t_index, scales=None):
    """Named-buffer census + gc sweep. Returns (rows, summary). Device = cuda when present,
    else cpu, so the harness is exercisable end-to-end without a GPU.

    `scales` classifies each slab by its TRAILING dim: 'flat' (B_outer x B_inner, the fork's
    own width) vs 'outer' (B_outer — retained outer state, and the width a past row block keeps)
    vs 'other'. That split is the whole point: a slab still 'flat' that holds only past rows is
    one the block routing did not reach."""
    sync()
    dev = 'cuda' if CUDA else 'cpu'
    scales = scales or {}

    def scale_of(t):
        n = t.shape[-1] if t.dim() else 0
        for nm, v in scales.items():
            if v and n == v:
                return nm
        return 'other'
    rows, named_ids = [], {}
    buckets = [
        ('t_Scenario_Buffer', getattr(shared_mem, 't_Scenario_Buffer', {})),
        ('t_Buffer', getattr(shared_mem, 't_Buffer', {})),
        ('t_Cashflows', getattr(shared_mem, 't_Cashflows', {})),
        ('t_quasi_rng', getattr(shared_mem, 't_quasi_rng', {})),
        ('t_random_numbers', {'z': getattr(shared_mem, 't_random_numbers', None)}),
        ('t_PreCalc', getattr(shared_mem, 't_PreCalc', {})),
    ]
    for bucket, container in buckets:
        for key, val in (container.items() if isinstance(container, dict) else []):
            for sub, t in iter_tensors(val):
                if t.device.type != dev:
                    continue
                sid = _storage_id(t)
                nbytes = sid[1]
                if sid in named_ids:                                # view of an already-named slab
                    continue
                named_ids[sid] = (bucket, _key_name(key))
                rows.append({'tag': tag, 't': t_index, 'bucket': bucket,
                             'scale': scale_of(t),
                             'key': _key_name(key) + (f'/{sub}' if sub else ''),
                             'shape': 'x'.join(str(d) for d in t.shape),
                             'dtype': str(t.dtype).replace('torch.', ''),
                             'MiB': round(nbytes / MIB, 3)})
    named_total = sum(sid[1] for sid in named_ids)

    gc_ids, shapes = {}, {}
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or obj.device.type != dev:
                continue
            sid = _storage_id(obj)
        except Exception:
            continue
        if sid in gc_ids:
            continue
        gc_ids[sid] = obj
        if sid not in named_ids:
            k = ('x'.join(str(d) for d in obj.shape), str(obj.dtype).replace('torch.', ''),
                 scale_of(obj))
            shapes[k] = shapes.get(k, [0, 0])
            shapes[k][0] += 1
            shapes[k][1] += sid[1]
    gc_total = sum(sid[1] for sid in gc_ids)
    for (shape, dtype, scale), (n, nbytes) in sorted(shapes.items(), key=lambda kv: -kv[1][1])[:25]:
        rows.append({'tag': tag, 't': t_index, 'bucket': 'UNATTRIBUTED', 'scale': scale,
                     'key': f'x{n}', 'shape': shape, 'dtype': dtype,
                     'MiB': round(nbytes / MIB, 3)})
    flat_MiB = sum(r['MiB'] for r in rows if r.get('scale') == 'flat')
    summary = {'tag': tag, 't': t_index, 'flat_scaled_MiB': round(flat_MiB, 1),
               'named_MiB': round(named_total / MIB, 1),
               'gc_live_MiB': round(gc_total / MIB, 1),
               'unattributed_MiB': round(max(0, gc_total - named_total) / MIB, 1),
               'allocator_live_MiB': round(live_bytes() / MIB, 1),
               'allocator_peak_MiB': round(peak_bytes() / MIB, 1)}
    return rows, summary


# ---- the probe ------------------------------------------------------------------------------
class ForkProbe:
    """Runtime-only instrumentation. install() patches two HedgeMonteCarlo methods and shadows
    bound methods on live instances; close() restores the class."""

    def __init__(self, census_stride=5, do_census=True, profiler=None, flat=0):
        self.events = []                       # (name, t_wall, live_bytes) within one fork
        self.rows = []                         # per-fork phase rows
        self.census_rows, self.census_summaries = [], []
        self.census_stride, self.do_census = census_stride, do_census
        self.flat = flat                       # fork flat samples, for kB/flat normalization
        self.scales = {}                       # trailing-dim -> scale name, set by the driver
        self.last_census_t = -10 ** 9
        self.profiler = profiler
        self.fork_calls = 0
        self.fork_log = []                     # (t, with_grad, wall_s)
        self._cur = None
        self._hooked = set()
        self._orig = {}
        self._global_peak = 0

    # -- event plumbing --
    def mark(self, name):
        sync()
        self.events.append((name, time.perf_counter(), live_bytes()))

    def _wrap(self, owner, attr, name, census_hook=None):
        """Shadow a bound method on an INSTANCE (class untouched)."""
        fn = getattr(owner, attr, None)
        if fn is None or (id(owner), attr) in self._hooked:
            return
        self._hooked.add((id(owner), attr))
        probe = self

        def wrapper(*a, **k):
            probe.mark(f'{name}:enter')
            try:
                with torch.profiler.record_function(f'fork/{name}'):
                    return fn(*a, **k)
            finally:
                probe.mark(f'{name}:exit')
                if census_hook:
                    census_hook()
        setattr(owner, attr, wrapper)

    def hook_instances(self, calc, shared_mem):
        self._wrap(shared_mem, 'reset_inner', 'reset_inner')
        self._wrap(shared_mem, 'reset_cashflows', 'reset_cashflows')
        for key, proc in getattr(calc, 'stoch_factors_inner', {}).items():
            nm = _key_name(key)
            self._wrap(proc, 'precalculate', f'gen/precalc/{nm}')
            self._wrap(proc, 'generate', f'gen/generate/{nm}')
            self._wrap(proc, 'reseed_inner_state', f'gen/reseed/{nm}')
        if (id(calc), '_restricted_struct') not in self._hooked:
            self._hooked.add((id(calc), '_restricted_struct'))
            orig_rs, probe = calc._restricted_struct, self

            def restricted(*a, **k):
                probe.mark('restricted_struct:enter')          # == end of the publication loop
                if probe.do_census and probe._cur and not probe._cur['grad'] \
                        and probe._cur['census'] and not probe._cur.get('did_stuff_census'):
                    probe._cur['did_stuff_census'] = True
                    probe._census(a, k, 'after_publication')
                struct = orig_rs(*a, **k)
                probe.mark('restricted_struct:exit')
                probe._wrap(struct, 'resolve_structure', 'price/resolve_structure')
                probe._wrap(struct, 'resolve_hedge_structure', 'price/resolve_hedge',
                            census_hook=probe._price_census)
                probe._wrap(struct, 'tensor_marks', 'price/tensor_marks')
                return struct
            calc._restricted_struct = restricted
        self._cur_shared = shared_mem

    def _census(self, a, k, tag):
        rows, summary = census(self._cur_shared, tag, self._cur['t'], self.scales)
        self.census_rows += rows
        self.census_summaries.append(summary)

    def _price_census(self):
        if self.do_census and self._cur and not self._cur['grad'] and self._cur['census'] \
                and not self._cur.get('did_price_census'):
            self._cur['did_price_census'] = True
            self._census(None, None, 'after_pricing')

    # -- phase derivation --
    def _derive(self, ev, mode, t_index, fork_peak):
        def first(pred):
            return next((e for e in ev if pred(e[0])), None)

        def last(pred):
            return next((e for e in reversed(ev) if pred(e[0])), None)
        enter, exit_ = ev[0], ev[-1]
        reset = last(lambda n: n == 'reset_inner:exit')
        price0 = first(lambda n: n == 'restricted_struct:enter')
        gen_end = None
        for e in ev:
            if e[0].startswith('gen/') and e[0].endswith(':exit'):
                if price0 is None or e[1] <= price0[1]:
                    gen_end = e
        price_end = last(lambda n: n.startswith('price/') and n.endswith(':exit'))
        pts = {'enter': enter, 'rng_end': reset or enter, 'gen_end': gen_end or reset or enter,
               'price_start': price0, 'price_end': price_end, 'exit': exit_}
        if price0 is None or price_end is None:                     # degenerate/terminal fork
            return None
        spans = [('rng', pts['enter'], pts['rng_end']),
                 ('generation', pts['rng_end'], pts['gen_end']),
                 ('publication', pts['gen_end'], pts['price_start']),
                 ('pricing', pts['price_start'], pts['price_end']),
                 ('extraction', pts['price_end'], pts['exit'])]
        row = {'t': t_index, 'mode': mode, 'fork_peak_MiB': round(fork_peak / MIB, 1)}
        for nm, s, e in spans:
            row[f'{nm}_ms'] = round((e[1] - s[1]) * 1e3, 3)
            row[f'{nm}_dMiB'] = round((e[2] - s[2]) / MIB, 2)
        return row

    # -- class hooks --
    def install(self):
        """One hook, because there is one pass. The chunk loop this used to bracket separately is
        gone: `_run_inner_mc_at_t` generates, publishes its block sequence and prices once, at
        Batch_Size x Inner_Sub_Batch."""
        probe = self
        self._orig['at'] = HedgeMonteCarlo._run_inner_mc_at_t
        orig_at = self._orig['at']

        def at(calc, t, outer_buf, shared_mem, base_date, refs, want_raw_samples=True,
               with_grad=False, max_inner_steps=None, outer_rows=None):
            probe.hook_instances(calc, shared_mem)
            probe.fork_calls += 1
            do_census = (not with_grad) and (t - probe.last_census_t >= probe.census_stride)
            if do_census:
                probe.last_census_t = t
            probe._cur = {'t': int(t), 'grad': bool(with_grad), 'census': do_census}
            probe._global_peak = max(probe._global_peak, peak_bytes())
            reset_peak()
            probe.events = []
            probe.mark('fork:enter')
            t0 = time.perf_counter()
            try:
                with torch.profiler.record_function(
                        f'fork/{"grad" if with_grad else "nograd"}/t{int(t)}'):
                    return orig_at(calc, t, outer_buf, shared_mem, base_date, refs,
                                   want_raw_samples, with_grad, max_inner_steps, outer_rows)
            finally:
                probe.mark('fork:exit')
                row = probe._derive(probe.events, 'grad' if with_grad else 'nograd',
                                    int(t), peak_bytes())
                if row:
                    probe.rows.append(row)
                probe._global_peak = max(probe._global_peak, peak_bytes())
                probe.fork_log.append((int(t), bool(with_grad),
                                       round(time.perf_counter() - t0, 4)))
                probe._cur = None
                if probe.profiler is not None:
                    probe.profiler.step()
        HedgeMonteCarlo._run_inner_mc_at_t = at
        return self

    def close(self):
        HedgeMonteCarlo._run_inner_mc_at_t = self._orig['at']

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- reporting --
    def aggregate(self):
        out = {}
        for mode in ('nograd', 'grad'):
            sel = [r for r in self.rows if r['mode'] == mode]
            if not sel:
                continue
            agg = {'forks': len(sel),
                   'peak_MiB_max': max(r['fork_peak_MiB'] for r in sel)}
            for ph in PHASES:
                agg[f'{ph}_ms_mean'] = round(sum(r[f'{ph}_ms'] for r in sel) / len(sel), 3)
                agg[f'{ph}_ms_total'] = round(sum(r[f'{ph}_ms'] for r in sel), 1)
                agg[f'{ph}_dMiB_mean'] = round(sum(r[f'{ph}_dMiB'] for r in sel) / len(sel), 2)
            tot = sum(agg[f'{ph}_ms_total'] for ph in PHASES) or 1.0
            for ph in PHASES:
                agg[f'{ph}_pct'] = round(100.0 * agg[f'{ph}_ms_total'] / tot, 1)
            out[mode] = agg
        return out


# ---- nvidia-smi sampling --------------------------------------------------------------------
class SmiSampler(threading.Thread):
    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        self.gpu = vis.split(',')[0] if vis else '0'
        self.interval, self.samples, self._halt = interval, [], threading.Event()
        self.ok = CUDA

    def run(self):
        while self.ok and not self._halt.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits', '-i', str(self.gpu)],
                    capture_output=True, text=True, timeout=5)
                util, mem = (x.strip() for x in out.stdout.strip().split(','))
                self.samples.append((float(util), float(mem)))
            except Exception:
                self.ok = False
                return
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()   # NOT _stop: that name shadows Thread._stop()
        self.join(timeout=3)
        if not self.samples:
            return {}
        u = sorted(s[0] for s in self.samples)
        return {'gpu_util_mean': round(sum(u) / len(u), 1), 'gpu_util_p50': u[len(u) // 2],
                'gpu_util_max': u[-1], 'smi_mem_max_MiB': max(s[1] for s in self.samples),
                'n_samples': len(u)}


# ---- subcommands ----------------------------------------------------------------------------
def cmd_phases(args):
    tag = stamp()
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = build_cfg(args.batch, args.inner, args.t_min, args.fit_iters,
                    dual_strip=args.dual_strip)
    fork_flat = args.batch * args.inner                 # one pass, so this IS the fork width
    reset_peak()
    t0 = time.perf_counter()
    probe = ForkProbe(census_stride=args.census_stride, do_census=not args.no_census,
                      flat=fork_flat)
    probe.scales = {'flat': fork_flat, 'outer': args.batch, 'inner': args.inner}
    with probe:
        run_job(cfg, 'tb_profile_phases')
        wall = time.perf_counter() - t0
        agg = probe.aggregate()
    peak_gib = max(probe._global_peak, peak_bytes()) / GIB

    base = os.path.join(OUTDIR, f'fork_phases_{tag}')
    with open(base + '.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(probe.rows[0].keys()) if probe.rows else ['t'])
        w.writeheader()
        w.writerows(probe.rows)
    lines = [f'inner-MC fork phase profile  ({tag})',
             f'world=fixture dual_strip={args.dual_strip} batch={args.batch} inner={args.inner} '
             f't_min={args.t_min} fit_iters={args.fit_iters}',
             f'device={"cuda" if CUDA else "cpu"}  wall={wall:.1f}s  peak={peak_gib:.2f} GiB',
             f'fork calls={probe.fork_calls}',
             '', 'NOTE: every marker is cuda.synchronize-bracketed, so absolute wall is',
             '      inflated -- read the SPLIT. `ladder` gives clean throughput.', '']
    for mode, a in agg.items():
        lines.append(f'--- {mode} ({a["forks"]} forks, max fork peak '
                     f'{a["peak_MiB_max"]:.0f} MiB) ---')
        lines.append(f'{"phase":<12}{"mean ms":>10}{"total ms":>11}{"% time":>8}{"mean dMiB":>11}')
        for ph in PHASES:
            lines.append(f'{ph:<12}{a[f"{ph}_ms_mean"]:>10.2f}{a[f"{ph}_ms_total"]:>11.1f}'
                         f'{a[f"{ph}_pct"]:>8.1f}{a[f"{ph}_dMiB_mean"]:>11.2f}')
        lines.append('')
    if probe.census_summaries:
        lines.append('--- slab census summary (MiB) ---')
        for s in probe.census_summaries:
            lines.append(f'  t={s["t"]:<4} {s["tag"]:<16} named={s["named_MiB"]:>9.1f}  '
                         f'gc_live={s["gc_live_MiB"]:>9.1f}  unattributed='
                         f'{s["unattributed_MiB"]:>9.1f}  allocator_live={s["allocator_live_MiB"]:>9.1f}')
        cb = os.path.join(OUTDIR, f'fork_census_{tag}.csv')
        with open(cb, 'w', newline='') as fh:
            w = csv.DictWriter(
                fh, fieldnames=['tag', 't', 'bucket', 'scale', 'key', 'shape', 'dtype', 'MiB'])
            w.writeheader()
            w.writerows(probe.census_rows)
        # Deepest-t pricing census only -- mixing censuses makes the top-N unreadable.
        priced = [s for s in probe.census_summaries if s['tag'] == 'after_pricing']
        if priced:
            deep = max(priced, key=lambda s: s['t'])
            sel = [r for r in probe.census_rows
                   if r['tag'] == 'after_pricing' and r['t'] == deep['t']]
            flat = probe.flat or 1
            buckets, by_scale = {}, {}
            for r in sel:
                buckets[r['bucket']] = buckets.get(r['bucket'], 0.0) + r['MiB']
                by_scale[r.get('scale', 'other')] = by_scale.get(r.get('scale', 'other'), 0.0) + r['MiB']
            lines += ['', f'--- slab census at the PRICING peak, deepest fork t={deep["t"]} '
                          f'(fork flat={flat}) ---',
                      f'{"bucket":<20}{"MiB":>10}{"kB/flat":>10}',
                      *[f'{b:<20}{m:>10.1f}{m * MIB / flat / 1024:>10.2f}'
                        for b, m in sorted(buckets.items(), key=lambda kv: -kv[1])],
                      f'{"TOTAL(gc live)":<20}{deep["gc_live_MiB"]:>10.1f}'
                      f'{deep["gc_live_MiB"] * MIB / flat / 1024:>10.2f}',
                      '', '  by SCALE (flat = the fork\'s own width; outer = a past row block):',
                      *[f'    {k:<8}{v:>10.1f} MiB{v * MIB / flat / 1024:>10.2f} kB/flat'
                        for k, v in sorted(by_scale.items(), key=lambda kv: -kv[1])],
                      '', '  top 20 individual slabs:']
            for r in sorted(sel, key=lambda r: -r['MiB'])[:20]:
                lines.append(f'  {r["MiB"]:>10.1f} MiB {r["MiB"] * MIB / flat / 1024:>8.2f} kB/flat '
                             f' [{r.get("scale", "?"):<5}] {r["bucket"]:<18} {r["shape"]:<20} '
                             f'{r["key"][:46]}')
            lines += ['', '  READ IT AS: a t_Scenario_Buffer curve sized (t+2)xn_tenorsxflat',
                      '  would mean a joined grid — the block routing publishes the realized past',
                      '  at OUTER width and only {t,t+1} at flat width, and the Hermite g,c pair',
                      '  in t_Buffer follows the block it was built from.']
        lines.append(f'\ncensus csv: {cb}')
    txt = '\n'.join(lines)
    open(base + '.txt', 'w').write(txt + '\n')
    print(txt)
    print(f'\nwrote {base}.csv / {base}.txt')


def cmd_trace(args):
    tag = stamp()
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = build_cfg(args.batch, args.inner, args.t_min, args.fit_iters,
                    dual_strip=args.dual_strip)
    acts = [torch.profiler.ProfilerActivity.CPU]
    if CUDA:
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    sched = torch.profiler.schedule(wait=0, warmup=1, active=args.active_forks, repeat=1)
    trace_path = os.path.join(OUTDIR, f'fork_trace_{tag}.json')
    with torch.profiler.profile(activities=acts, schedule=sched, record_shapes=True,
                                profile_memory=True, with_stack=False) as prof:
        with ForkProbe(do_census=False, profiler=prof):
            t0 = time.perf_counter()
            run_job(cfg, 'tb_profile_trace')
            wall = time.perf_counter() - t0
    prof.export_chrome_trace(trace_path)

    def dev_us(e):
        for a in ('self_device_time_total', 'self_cuda_time_total'):
            if hasattr(e, a):
                return getattr(e, a)
        return 0
    ka = prof.key_averages()
    gpu_us = sum(max(0, dev_us(e)) for e in ka)
    launches = sum(e.count for e in ka if dev_us(e) > 0)
    body = ka.table(sort_by='self_device_time_total' if CUDA else 'self_cpu_time_total',
                    row_limit=40)
    lines = [f'inner-MC fork trace ({tag})  device={"cuda" if CUDA else "cpu"}',
             f'batch={args.batch} inner={args.inner} t_min={args.t_min} '
             f'active_forks={args.active_forks}  job wall={wall:.1f}s',
             f'kernel time total={gpu_us / 1e6:.2f}s over {launches} device-op invocations',
             'IDLE ATTRIBUTION: compare kernel time against the profiled window; look for',
             '  fork/* record_function spans (phases) and gaps between them in the chrome trace.',
             '', body]
    txt = '\n'.join(lines)
    open(os.path.join(OUTDIR, f'fork_trace_{tag}.txt'), 'w').write(txt + '\n')
    print(txt[:4000])
    print(f'\nwrote {trace_path}\nwrote {os.path.join(OUTDIR, f"fork_trace_{tag}.txt")}')


def cmd_all(args):
    cmd_phases(args)
    cmd_trace(args)


def main():
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['phases', 'trace', 'all'])
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--inner', type=int, default=64)
    p.add_argument('--t-min', type=int, default=95, dest='t_min',
                   help='higher = fewer forks (T_dec~117 on the fixture, so 95 => ~22 forks)')
    p.add_argument('--fit-iters', type=int, default=10, dest='fit_iters')
    p.add_argument('--dual-strip', action='store_true',
                   help='add the LME leg (validate_cross_market helper still emits the removed '
                        'Implied_Basis field -- verify before using)')
    p.add_argument('--census-stride', type=int, default=5, dest='census_stride')
    p.add_argument('--no-census', action='store_true')
    p.add_argument('--active-forks', type=int, default=3, dest='active_forks')
    args = p.parse_args()
    if args.cmd == 'trace' and args.t_min == 95:
        args.t_min = 110    # the profiler records `active_forks`, but the JOB runs the whole
                            # sweep -- keep the sweep short or the trace run costs a full solve
    globals()[f'cmd_{args.cmd}'](args)


if __name__ == '__main__':
    main()
