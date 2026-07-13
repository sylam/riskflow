"""DiffSolverV2 (clean-room framework-native diff-ML hedger) — bounded value + OUT-OF-SAMPLE
hedging gate on the platinum deal.

Locks the two things the build hinged on:
  • the value stays BOUNDED at depth (the expired-contract fake-dF bug inflated it to 1e3+;
    pre-rebuild the legacy solver blew to 1e8) — `max|Y_boot|` small, V_0 finite/small;
  • the greedy policy HEDGES out-of-sample — on held-out paths it does not underperform
    no-hedge (the full result is greedy ≫ textbook OOS at full depth; see the build notes).
The verdict rolls on paths the value function never saw (`DiffV2_OOS_Frac`), so a policy
that merely overfits the fitted paths fails this.

JSON-is-the-contract: load_json + run_job, no internal imports / monkey-patching.
"""
import json as jsonlib
import math
import os

import pytest

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg(inner_antithetic='No', one_step_fork='Yes'):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'] = 48                 # 24 train / 24 OOS at the 0.5 split
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = inner_antithetic
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
