"""Gates for the three solver-consistency fixes:

  1. ACTION UNIVERSE — one shared `HedgeActionSpace` drives DiffSolverV2, the textbook /
     hindsight benchmarks AND the stepper rollout, so an `Active_Hedge_Indices` restriction
     pins the inactive legs to exactly zero across every track (the old benchmarks used the
     generic all-hedges grid and could trade a leg the greedy run had frozen).
  2. INITIAL INVENTORY — first-step turnover is measured from the OPENING book `q0`
     (normalized `Portfolio_State` positions), not from flat, in the frictionless value
     tracks' net-of-cost diagnostics and in the stepper rollout.
  3. POLICY ARTIFACT — `DiffSolverV2.solve()` returns ONE `value_fn_artifacts` dict (also the
     bytes `DiffV2_Save_Value_Fn` persists); the in-memory artifact evaluates identically to
     the file checkpoint saved in the same run.

Configs are built in code from the canonical fixture TEMPLATE (never edited) — small batch /
shallow sweep so the gates run fast. JSON-is-the-contract for the run itself; the artifact
round-trip reaches into `DiffSolverV2` directly (it exercises framework internals a JSON
end-user never touches).
"""
import copy
import json as jsonlib
import os

import pytest
import torch

import riskflow as rf
from riskflow.hedge_runtime import per_contract_kappa
from riskflow.hedge_solver import DiffSolverV2

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')

# Fixture hedge order (names['hedges']): PL_APR_2026, PL_JUL_2026, PL_OCT_2026.
_HEDGES = ('PL_APR_2026', 'PL_JUL_2026', 'PL_OCT_2026')

_LISTED_ARTIFACT_KEYS = {
    'state_dicts', 'm_mean', 'm_std', 'w_mean', 'w_std', 'utility_scale', 'a_bounds',
    'hedges', 'active_hedge_indices', 'total_position_schedule', 'T_dec', 't_min', 'hidden',
    'solver_version', 'config_hash',
}


def _cfg(*, batch=48, inner=8, seed=7, positions=None, active=None,
         benchmarks=False, save=None):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'] = batch
    calc['Inner_Sub_Batch'] = inner
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = 'Yes'
    calc['Random_Seed'] = seed
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    if positions is not None:
        hp.setdefault('Portfolio_State', {})['Positions'] = positions
    solver = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': 100,                       # shallow ~16-step sweep (fast)
        'DiffV2_Fit_Iters': 5,
        'DiffV2_OOS_Frac': 0.5,
    }
    if active is not None:
        solver['Active_Hedge_Indices'] = list(active)
    if benchmarks:
        solver['Run_Textbook_Benchmark'] = 'Yes'
        solver['Run_Hindsight_Diagnostic'] = 'Yes'
    if save is not None:
        solver['DiffV2_Save_Value_Fn'] = save
    hp['Solver'] = solver
    return cfg


def _run(cfg):
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'gate.json'))
    _, result = cx.run_job()
    return result


# --------------------------------------------------------------------------------------------
# (i) active-mask consistency — inactive legs hold EXACTLY zero on every track
# --------------------------------------------------------------------------------------------
def test_active_mask_zeroes_inactive_legs_on_all_tracks():
    active = [2]                              # only PL_OCT_2026 varies; legs 0,1 pinned to 0
    result = _run(_cfg(active=active, benchmarks=True))
    es = result.evaluation_summary
    comp = es['comparison']
    diag = es['diagnostics']

    textbook_n = comp['textbook']['n_star']
    hindsight_n = comp['HindsightDpSolver']['n_star']
    greedy_absq = diag['verdict']['greedy_mean_abs_q']
    assert len(textbook_n) == len(hindsight_n) == len(greedy_absq) == 3

    for i in (0, 1):                          # the two INACTIVE legs
        assert textbook_n[i] == 0.0, f'textbook traded inactive leg {i}: {textbook_n}'
        assert hindsight_n[i] == 0.0, f'hindsight traded inactive leg {i}: {hindsight_n}'
        assert greedy_absq[i] == 0.0, f'greedy traded inactive leg {i}: {greedy_absq}'

    # The artifact records which axes were active — the frozen policy carries its mask.
    assert result.policy_artifact['active_hedge_indices'] == active


# --------------------------------------------------------------------------------------------
# (ii) initial inventory — first-step turnover measured from the OPENING book q0, not flat
# --------------------------------------------------------------------------------------------
def test_first_step_turnover_measured_from_opening_book():
    positions = {'PL_APR_2026': -1, 'PL_JUL_2026': -4, 'PL_OCT_2026': 0}
    flat = _run(_cfg(seed=11, positions=None, benchmarks=True))
    book = _run(_cfg(seed=11, positions=positions, benchmarks=True))

    fb, bb = flat.evaluation_summary, book.evaluation_summary

    # The value function is POSITION-FREE: the opening book must NOT move any V_0/u track.
    assert bb['comparison']['textbook']['v0_mean'] == fb['comparison']['textbook']['v0_mean']
    assert bb['comparison']['HindsightDpSolver']['v0_mean'] \
        == fb['comparison']['HindsightDpSolver']['v0_mean']
    for k in ('u_mean', 'wT_mean', 'wT_p5', 'wT_cvar5'):
        assert bb['diagnostics']['verdict']['greedy'][k] \
            == fb['diagnostics']['verdict']['greedy'][k], f'greedy {k} moved with the book'

    # ...but the NET-OF-COST turnover DOES shift, because entry is now |q_target − q0|.
    tb_flat = fb['comparison']['textbook']['turnover_cost_mean']
    tb_book = bb['comparison']['textbook']['turnover_cost_mean']
    g_flat = fb['diagnostics']['verdict']['greedy']['turnover_cost_mean']
    g_book = bb['diagnostics']['verdict']['greedy']['turnover_cost_mean']
    assert tb_book != tb_flat, 'textbook entry turnover ignored the opening book'
    assert g_book != g_flat, '_verdict entry turnover ignored the opening book'

    # The frictionless argmax is position-independent, so both runs pick the SAME constant
    # hold n_star and the SAME kappa0 (identical bundle). The ONLY thing that changed is the
    # ENTRY term: |n_star − q0| (book) vs |n_star − 0| (flat). So the whole turnover
    # difference must equal Σ_i (|n_star_i − q0_i| − |n_star_i|) · kappa0_i — the crisp
    # signature of "first-step turnover measured from q0, not from flat".
    n_star = bb['comparison']['textbook']['n_star']
    assert fb['comparison']['textbook']['n_star'] == n_star     # position-free selection
    runtime, bundle = book.runtime, book.bundle
    hist = int(bundle['initial_time_index'])
    expected_diff = sum(
        (abs(n_star[i] - float(positions[h])) - abs(n_star[i]))
        * float(per_contract_kappa(runtime, bundle['tradables'][h][hist:][0].mean(), h))
        for i, h in enumerate(_HEDGES))
    assert abs((tb_book - tb_flat) - expected_diff) < 1e-3 * (abs(expected_diff) + 1.0), \
        f'textbook turnover shift {tb_book - tb_flat} != Σ(|n*−q0|−|n*|)·kappa0 {expected_diff}'

    # The stepper OPENS from q0 (build_shared_state seeds Portfolio_State), so its realized
    # first-step trade is measured from the same book — the deployment convention agrees.
    stepper = book.create_stepper()
    opening = stepper.observe()['positions']
    for h in _HEDGES:
        assert float(opening[h][0]) == float(positions[h]), \
            f'stepper did not open from the book at {h}'
    while not stepper.is_decision_step:
        stepper.step(None)
    q_target = -10.0
    pre = float(stepper.observe()['positions']['PL_JUL_2026'][0])
    assert pre == float(positions['PL_JUL_2026'])          # still opening from the book at t0
    stepper.step({'PL_JUL_2026': q_target - pre})          # trade the verdict delta q_target − q0
    post = float(stepper.observe()['positions']['PL_JUL_2026'][0])
    assert round(post) == round(q_target)                  # reached the target
    # the realized first-step trade the stepper charged turnover on is q_target − q0 (the
    # verdict convention), NOT q_target − 0.
    assert round(post - pre) == round(q_target - float(positions['PL_JUL_2026']))


# --------------------------------------------------------------------------------------------
# (iii) artifact contract — solve() returns the artifact; in-memory == file checkpoint
# --------------------------------------------------------------------------------------------
def test_policy_artifact_contract_and_eval_from_memory(tmp_path):
    ckpt = str(tmp_path / 'value_fn.pt')
    train = _run(_cfg(seed=7, save=ckpt))
    artifact = train.policy_artifact

    # (a) the returned artifact carries EVERY listed key.
    assert isinstance(artifact, dict)
    missing = _LISTED_ARTIFACT_KEYS - set(artifact)
    assert not missing, f'artifact missing listed keys: {missing}'
    assert artifact['solver_version'] and isinstance(artifact['config_hash'], str)

    # (b) the persisted file is the SAME dict (torch.save of the identical object).
    assert os.path.exists(ckpt)
    file_dict = torch.load(ckpt, map_location='cpu')
    assert set(file_dict) == set(artifact)
    for k in ('V_0', 'config_hash', 'solver_version', 'hedges',
              'active_hedge_indices', 'T_dec', 't_min', 'md', 'hidden'):
        assert file_dict[k] == artifact[k], f'file/artifact disagree on {k!r}'

    # (c) eval-from-artifact (in-memory dict) == eval-from-file: a loaded eval echoes the
    # checkpoint's stored V_0, so both must equal artifact['V_0'] exactly (Sobol-independent).
    def _eval(load_member):
        rt = dict(train.runtime)
        rt['solver'] = dict(train.runtime['solver'])
        rt['solver']['diffv2_load_value_fn'] = [load_member]
        return DiffSolverV2(train.bundle, rt).solve().values

    v_art = _eval(artifact)                   # in-memory artifact as an ensemble member
    v_file = _eval(ckpt)                      # the loaded file
    assert v_art == v_file == artifact['V_0'], \
        f'in-memory {v_art} vs file {v_file} vs saved {artifact["V_0"]}'


# --------------------------------------------------------------------------------------------
# (iv) corridor provenance — a policy trained inside a Total_Position_Schedule is valid ONLY
#      when rolled in the SAME corridor; a different (or absent) one fails loud.
# --------------------------------------------------------------------------------------------
def test_corridor_provenance_stamped_and_load_mismatch_fails_loud(tmp_path):
    ckpt = str(tmp_path / 'value_fn_corr.pt')
    cfg = _cfg(seed=7, save=ckpt)
    S1 = [{'Step': 0, 'Min_Total': -50, 'Max_Total': -30},
          {'Step': 50, 'Min_Total': -30, 'Max_Total': -10}]
    cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']['Total_Position_Schedule'] = S1
    train = _run(cfg)
    art = train.policy_artifact
    # the trained corridor is STAMPED into the artifact (provenance)
    assert art['total_position_schedule'] is not None
    assert len(art['total_position_schedule']) == 2

    def _eval(load_member, sched):
        rt = dict(train.runtime)
        rt['solver'] = dict(train.runtime['solver'])
        rt['accounting'] = dict(train.runtime['accounting'])
        rt['accounting']['total_position_schedule'] = sched
        rt['solver']['diffv2_load_value_fn'] = [load_member]
        return DiffSolverV2(train.bundle, rt).solve().values

    same = train.runtime['accounting']['total_position_schedule']   # normalized training corridor
    assert _eval(ckpt, same) == art['V_0'], 'reload in the SAME corridor must echo the trained V_0'

    # a DIFFERENT corridor is invalid — the frozen value fn learned other reachable wealth states
    other = ((0, -40.0, -20.0),)
    with pytest.raises(ValueError, match='corridor mismatch'):
        _eval(ckpt, other)
    # dropping the corridor entirely is also a mismatch (trained fenced, rolled unfenced)
    with pytest.raises(ValueError, match='corridor mismatch'):
        _eval(ckpt, None)


def test_corridor_free_policy_rolls_in_any_corridor(tmp_path):
    """A policy trained corridor-FREE has the widest wealth support, so rolling it INSIDE a
    Total_Position_Schedule only restricts to a learned subset — allowed, not a mismatch. This is
    the roll-only-on-corridor-free validation path the delta-corridor work was validated with."""
    ckpt = str(tmp_path / 'value_fn_free.pt')
    train = _run(_cfg(seed=7, save=ckpt))                          # no schedule → trained free
    assert train.policy_artifact['total_position_schedule'] is None

    rt = dict(train.runtime)
    rt['solver'] = dict(train.runtime['solver'])
    rt['accounting'] = dict(train.runtime['accounting'])
    rt['accounting']['total_position_schedule'] = ((0, -50.0, -20.0),)   # roll inside a corridor
    rt['solver']['diffv2_load_value_fn'] = [ckpt]
    v = DiffSolverV2(train.bundle, rt).solve().values             # must NOT raise
    assert v == train.policy_artifact['V_0']


if __name__ == '__main__':
    test_active_mask_zeroes_inactive_legs_on_all_tracks()
    test_first_step_turnover_measured_from_opening_book()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        class _P:
            def __truediv__(self, n):
                return os.path.join(d, n)
        test_policy_artifact_contract_and_eval_from_memory(_P())
    print('all action-space + artifact gates passed')
