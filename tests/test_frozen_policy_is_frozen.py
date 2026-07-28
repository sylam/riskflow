"""`DiffV2_Load_Value_Fn` means EVALUATION, and it has to mean that under streaming too.

`DiffSolverV2.warmup` honours a loaded checkpoint by skipping its backward sweep, but under
`DiffV2_Streaming_Batches='Yes'` the calc drives `warmup / step / finish` across batches and
`step` swept unconditionally — so batches 2..N-1 fine-tuned the "frozen" nets on the evaluation
world. `finish` then gated the artifact on `loaded is None`, so the retrained weights were
reported (V_0, verdict) and never written anywhere: unreproducible by construction.

The gate is exact rather than statistical. A frozen eval must report the CHECKPOINT's V_0 to full
precision and fit zero rows, which is what the non-streaming load path already does — so the two
load paths must agree, and both must agree with the file.

Streaming needs `Simulation_Batches >= 3` (warmup + a step + the held-out batch), so these runs
are deliberately tiny: the point is which code paths execute, not the numbers themselves.
"""
import json as jsonlib
import os

import pytest
import torch

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg(batches, save=None, load=None, streaming=True):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': 16, 'Inner_Sub_Batch': 4,
                 'Inner_MC_Enabled': 'Yes', 'Random_Seed': 1234,
                 'Simulation_Batches': batches})
    calc['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
    solver = {'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 3,
              'Training_Action_Chunk_Size': 64, 'T_Min': 113, 'DiffV2_Fit_Iters': 2,
              'DiffV2_OOS_Frac': 0.5, 'DiffV2_One_Step_Fork': 'Yes',
              'DiffV2_Streaming_Batches': 'Yes' if streaming else 'No'}
    if save:
        solver['DiffV2_Save_Value_Fn'] = save
    if load:
        solver['DiffV2_Load_Value_Fn'] = [load]
    calc['Hedging_Problem']['Solver'] = solver
    return cfg


def _run(cfg, name):
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), f'{name}.json'))
    _, result = cx.run_job()
    return ((result.evaluation_summary or {}).get('diagnostics') or {}), result


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    """One streaming TRAINING run, saved. Streaming train is the production recipe."""
    path = str(tmp_path_factory.mktemp('vf') / 'value_fn.pt')
    diag, result = _run(_cfg(batches=3, save=path), 'train_streaming')
    assert os.path.exists(path)
    assert result.policy_artifact is not None
    return path, torch.load(path, weights_only=False)


def test_streaming_load_does_not_retrain_the_checkpoint(checkpoint):
    """The defect, exactly: with streaming on, the intermediate batches ran a full backward sweep
    with `opt.step()` over the loaded nets. A frozen run reports the checkpoint's own V_0 and
    fits nothing."""
    path, ck = checkpoint
    diag, result = _run(_cfg(batches=3, load=path), 'eval_streaming')
    assert diag['per_t'] == [], 'a fit step ran on a loaded checkpoint — it is not frozen'
    assert diag['V_0'] == ck['V_0'], 'V_0 is not the checkpoint\'s — the nets moved'
    assert result.policy_artifact is None, 'a frozen eval has no new value fn to emit'


def test_both_load_paths_agree(checkpoint):
    """Streaming is only a batching strategy. Evaluating the same frozen policy through it must
    give the same answer as the non-streaming load, to full precision."""
    path, ck = checkpoint
    streaming, _ = _run(_cfg(batches=3, load=path), 'eval_streaming_cmp')
    fixed, _ = _run(_cfg(batches=3, load=path, streaming=False), 'eval_fixed_cmp')
    assert fixed['per_t'] == [] and streaming['per_t'] == []
    assert streaming['V_0'] == fixed['V_0'] == ck['V_0']


def test_the_checkpoint_file_is_untouched_by_an_eval(checkpoint):
    """The retrained nets used to be discarded silently — reported, then thrown away. Nothing
    should write to the file either."""
    path, ck = checkpoint
    before = open(path, 'rb').read()
    _run(_cfg(batches=3, load=path), 'eval_no_write')
    assert open(path, 'rb').read() == before


def test_saving_while_loading_is_a_contradiction(checkpoint):
    """Train (save) and evaluate (load) are separate runs. Setting both used to silently drop the
    save, which is how the retrained-and-discarded nets stayed invisible."""
    path, _ck = checkpoint
    with pytest.raises(ValueError, match='DiffV2_Save_Value_Fn is set alongside'):
        _run(_cfg(batches=3, save=path + '.2', load=path), 'save_and_load')
