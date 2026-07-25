"""DiffSolverV2 (clean-room framework-native diff-ML hedger) — bounded value + OUT-OF-SAMPLE
hedging gate on the platinum deal.

Locks the two things the build hinged on:
  • the value stays BOUNDED at depth — `max|Y_boot|` small, V_0 finite/small;
  • the greedy policy HEDGES out-of-sample — on held-out paths it does not underperform
    no-hedge (greedy ≫ textbook OOS at full depth).
The verdict rolls on paths the value function never saw (`DiffV2_OOS_Frac`), so a policy
that merely overfits the fitted paths fails this.

JSON-is-the-contract: the end-to-end tests here go through load_json + run_job only (the
inner-MC chunking is driven by `Calculation.Inner_MC_Flat_Limit`, not by patching). The one
direct import is `_concat_inner_chunks` — a pure function whose reassembly contract is pinned
by a unit test at the bottom.
"""
import json as jsonlib
import math
import os

import pytest
import torch

import riskflow as rf
from riskflow.calculation import _concat_inner_chunks

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg(inner_antithetic='No', one_step_fork='Yes', flat_limit=None):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'] = 48                 # 24 train / 24 OOS at the 0.5 split
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = inner_antithetic
    if flat_limit is not None:
        calc['Inner_MC_Flat_Limit'] = flat_limit
    calc['Random_Seed'] = 1234
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': 100,                       # ~17-step bounded sweep (fast); full depth in build notes
        'DiffV2_Fit_Iters': 30,
        'DiffV2_OOS_Frac': 0.5,
        'DiffV2_One_Step_Fork': one_step_fork,
        # defaults apply implicitly: DiffV2_Weight_Decay=0 (the twin-loss gradient match is
        # the regularizer, not weight decay), DiffV2_Lambda_Grad=1, DiffV2_Hidden=32. This
        # tiny-batch smoke gates "bounded + hedges OOS"; full multi-seed wd=0 robustness is
        # validated at B_outer=4095 (see project_differential_ml_build_state).
    }
    return cfg


@pytest.mark.parametrize('inner_antithetic,one_step_fork', [
    ('No', 'Yes'), ('Yes', 'Yes'),
    # legacy full-horizon forks — statistically-equivalent labels at shallow windows
    # (the mode still ships as the DiffV2_One_Step_Fork='No' fallback)
    ('Yes', 'No'),
])
def test_diffsolverv2_bounded_and_hedges_oos(inner_antithetic, one_step_fork):
    """Both inner-draw modes must clear the same gates: plain Sobol and the antithetic
    fold (Inner_Antithetic='Yes' — mirrored (z, -z) pairs on the inner axis), plus the
    legacy full-horizon fork mode."""
    cfg = _cfg(inner_antithetic, one_step_fork)
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'diffml_v2_oos.json'))
    _, result = cx.run_job()
    diag = (result.evaluation_summary or {}).get('diagnostics') or {}

    # --- value bounded & finite at depth (catches the expired-dF inflation regressing in) ---
    assert 'V_0' in diag, 'solver must expose a headline V_0'
    v0 = float(diag['V_0'])
    assert math.isfinite(v0) and abs(v0) < 50.0, f'V_0 not bounded/finite: {v0}'
    assert diag.get('bounded') is True, 'sweep flagged not-bounded'
    assert float(diag['max_abs_Y_boot']) < 100.0, \
        f"max|Y_boot|={diag['max_abs_Y_boot']} — value inflating (expired-dF guard regressed?)"

    # --- the verdict is OUT-OF-SAMPLE and the greedy policy hedges (≥ no-hedge OOS) ---
    assert diag.get('verdict_is_oos') is True, 'verdict must be on held-out paths'
    v = diag['verdict']
    g_u, nh_u = v['greedy']['u_mean'], v['nohedge']['u_mean']
    assert g_u >= nh_u - 0.05, \
        f'greedy underperforms no-hedge OOS (u greedy={g_u:.4f} vs no-hedge={nh_u:.4f})'
    assert diag['verdict_beats_nohedge_on_utility'] in (True, False)  # key present

    # --- expired contracts carry ZERO position (the live-mask correctness) ---
    # near terminal at least one of the 3 futures has expired; its rolled |q| must be 0.
    mean_abs_q = v['greedy_mean_abs_q']
    assert min(mean_abs_q) < 1e-6, \
        f'no expired contract zeroed — live-mask not applied? mean|q|={mean_abs_q}'


def _corridor_at(sched, t):
    lo, hi = sched[0]['Min_Total'], sched[0]['Max_Total']
    for k in sched:
        if k['Step'] > t:
            break
        lo, hi = k['Min_Total'], k['Max_Total']
    return lo, hi


def test_corridor_train_smoke_sign_crossing():
    """BOOK-STYLE generalization end-to-end: train DiffSolverV2 INSIDE a SIGN-CROSSING
    Total_Position_Schedule (short early, long late) on symmetric [-50, 50] limits, and assert
    (a) the backward sweep stays bounded and (b) the greedy verdict rolls INSIDE the fence at
    every step — the corridor flips the mandated total's sign mid-window and the argmax obeys it.
    Exercises grid_at long rows + the bank/textbook corridor projection on a superposable book."""
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'] = 48
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = 'Yes'
    calc['Random_Seed'] = 1234
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    ev = hp['Evaluator']
    # symmetric (book-style) limits + gross cap off so the signed corridor is the only total bound
    for lim in ev['Position_Limits'].values():
        lim['Min_Position'], lim['Max_Position'] = -50, 50
    ev['Total_Position_Abs_Limit'] = 0.0
    # short early, long late — the sign flips at Step 107, inside the T_Min=100 sweep window
    sched = [{'Step': 0, 'Min_Total': -50, 'Max_Total': -25},
             {'Step': 107, 'Min_Total': 25, 'Max_Total': 50}]
    ev['Total_Position_Schedule'] = sched
    hp['Solver'] = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64, 'T_Min': 100, 'DiffV2_Fit_Iters': 15,
        'DiffV2_OOS_Frac': 0.5}

    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'corridor_sign_crossing.json'))
    _, result = cx.run_job()
    diag = (result.evaluation_summary or {}).get('diagnostics') or {}

    # (a) bounded backward sweep under the sign-crossing corridor
    v0 = float(diag['V_0'])
    assert math.isfinite(v0) and abs(v0) < 50.0, f'V_0 not bounded under corridor: {v0}'
    assert diag.get('bounded') is True, 'sweep flagged not-bounded under corridor'

    # (b) the greedy verdict rolls INSIDE the fence at every step (short early, long late)
    v = diag['verdict']
    traj = v['greedy_q_traj']
    t0 = int(diag['root_t'])
    saw_short = saw_long = False
    for i, book in enumerate(traj):
        t = t0 + i
        lo, hi = _corridor_at(sched, t)
        tot = float(sum(book))
        assert lo - 1e-3 <= tot <= hi + 1e-3, \
            f'greedy breached corridor at t={t}: Σq={tot:.3f} vs [{lo}, {hi}]'
        saw_short = saw_short or hi < 0
        saw_long = saw_long or lo > 0
    assert saw_short and saw_long, 'sweep window did not cover both corridor signs'

    # provenance: the trained artifact stamps the corridor it was trained inside
    art = result.policy_artifact
    assert art['total_position_schedule'] is not None, 'artifact must stamp the training corridor'
    assert len(art['total_position_schedule']) == len(sched)


# --- inner-MC chunking (Calculation.Inner_MC_Flat_Limit) ------------------------------------
# The dispatcher runs a fork in outer-path sub-chunks when B_outer*Inner_Sub_Batch exceeds the
# flat limit. Before the limit became a JSON field it was an env var, so the chunked path could
# not be driven from a config and had NO coverage at any batch size the suite runs.

def _fork_slabs(one_step, b_outer=6, b_inner=4, md=5, refs=('PL_A', 'PL_B')):
    """A `_run_inner_mc_chunk` result dict, values distinct per element so a mis-ordered or
    mis-sliced concat is caught. `one_step=True` mirrors the {t, t+1} window: the horizon
    fields are empty (`L_T` None, `dF_T`/`dF_min` {}) because a 2-row window prices no terminal."""
    n = [0]

    def slab(*shape):
        n[0] += 1
        return torch.arange(math.prod(shape), dtype=torch.float32).reshape(*shape) + 1000 * n[0]

    whole = {'features': slab(b_outer, 3 + 2 * len(refs)),
             't': 7, 'cutoff_idx': 7,
             'L_T': None if one_step else slab(b_outer, b_inner),
             'L_t': slab(b_outer, b_inner), 'L_t1': slab(b_outer, b_inner),
             'F_t1': {r: slab(b_outer, b_inner) for r in refs},
             'dF_T': {} if one_step else {r: slab(b_outer, b_inner) for r in refs},
             'dF_min': {} if one_step else {r: slab(b_outer, b_inner) for r in refs},
             'market_t': slab(b_outer, md),
             'market_t1': slab(b_outer, b_inner, md)}

    def cut(v, lo, hi):
        if isinstance(v, torch.Tensor):
            return v[lo:hi]
        if isinstance(v, dict):
            return {r: x[lo:hi] for r, x in v.items()}
        return v                                             # scalars (t, cutoff_idx) and None
    return whole, [{k: cut(v, lo, hi) for k, v in whole.items()}
                   for lo, hi in ((0, 2), (2, 4), (4, 6))]


@pytest.mark.parametrize('one_step', [True, False])
def test_concat_inner_chunks_structural_identity(one_step):
    """A chunked fork's result must be STRUCTURALLY identical to the single-pass result — same
    key set, same shapes, same None-ness. The pure concat is also value-exact (only the
    SIMULATION is statistically-not-bitwise equivalent across partitions), so assert that too.

    Regression: an empty field used to be DROPPED rather than carried through as None, so
    `result['L_T']` was a KeyError in chunked mode and None in single-chunk mode — a divergence
    that only appears once the batch is big enough to chunk."""
    whole, chunks = _fork_slabs(one_step)
    out = _concat_inner_chunks(chunks, want_raw_samples=True)

    assert set(out) == set(whole), (
        f'key set depends on the chunk count: only-chunked={set(out) - set(whole)}, '
        f'only-single={set(whole) - set(out)}')
    for key, ref in whole.items():
        got = out[key]
        assert (ref is None) == (got is None), f'{key}: None-ness differs (single={ref}, chunked={got})'
        if isinstance(ref, torch.Tensor):
            assert got.shape == ref.shape, f'{key}: shape {tuple(got.shape)} != {tuple(ref.shape)}'
            assert torch.equal(got, ref), f'{key}: concat is not a faithful reassembly'
        elif isinstance(ref, dict):
            assert set(got) == set(ref), f'{key}: per-tradable keys differ'
            for r in ref:
                assert got[r].shape == ref[r].shape and torch.equal(got[r], ref[r]), f'{key}[{r}]'
        else:
            assert got == ref, f'{key}: scalar {got} != {ref}'
    if one_step:
        assert out['L_T'] is None, 'one-step window must report L_T as an explicit None'


@pytest.mark.parametrize('one_step_fork', ['Yes', 'No'])
def test_chunked_inner_mc_clears_the_same_gates(one_step_fork):
    """Drive the dispatcher's chunk loop from the JSON: at 48 x 8 = 384 flat an
    Inner_MC_Flat_Limit of 64 gives chunk = 64 // 8 = 8 outer paths, so every no-grad fork runs
    6 chunk passes (measured; one_step='No' additionally exercises the row-aware cells cap on a
    ~27-row inner grid). The GRAD forks stay single-pass — the solver slices them to its own
    cell budget (here 4 outer paths, half the no-grad chunk), which is why the with_grad
    single-chunk guard never fires.

    The chunked run must clear the same bounded/hedges-OOS gates and expose the same diagnostic
    shape as the single-pass run. NOT asserted: bit parity — each pass draws its own Sobol
    stream, so partitions are statistically, not bitwise, equivalent by construction."""
    diags = {}
    for label, flat_limit in (('single', None), ('chunked', 64)):
        cx = rf.Context()
        cx.load_json((jsonlib.dumps(_cfg('Yes', one_step_fork, flat_limit)),
                      f'diffml_v2_chunked_{label}.json'))
        _, result = cx.run_job()
        diags[label] = (result.evaluation_summary or {}).get('diagnostics') or {}

    chunked, single = diags['chunked'], diags['single']
    # the chunked run is a real run: same gates as the single-pass path
    v0 = float(chunked['V_0'])
    assert math.isfinite(v0) and abs(v0) < 50.0, f'chunked V_0 not bounded/finite: {v0}'
    assert chunked.get('bounded') is True, 'chunked sweep flagged not-bounded'
    assert float(chunked['max_abs_Y_boot']) < 100.0, 'chunked labels inflating'
    assert chunked.get('verdict_is_oos') is True
    cv = chunked['verdict']
    assert cv['greedy']['u_mean'] >= cv['nohedge']['u_mean'] - 0.05, \
        'chunked greedy underperforms no-hedge OOS'

    # structural identity vs single-pass: keys, None-ness, verdict layout
    assert set(chunked) == set(single), (
        f'diagnostic keys depend on chunking: only-chunked={set(chunked) - set(single)}, '
        f'only-single={set(single) - set(chunked)}')
    for key, ref in single.items():
        assert (ref is None) == (chunked[key] is None), f'{key}: None-ness differs under chunking'
    assert set(cv) == set(single['verdict']), 'verdict layout differs under chunking'
    for pol in ('greedy', 'textbook', 'nohedge'):
        assert set(cv[pol]) == set(single['verdict'][pol]), f'{pol} stat keys differ under chunking'
