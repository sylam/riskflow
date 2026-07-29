"""`DiffV2_Load_Value_Fn` means EVALUATION, and a solve is a stream, so it has to mean that here.

A solve runs `warmup / step / finish` across simulation batches, and `step` once swept
unconditionally — so batches 2..N-1 fine-tuned the "frozen" nets on the evaluation world. `finish`
then gated the artifact on `loaded is None`, so the retrained weights were reported (V_0, verdict)
and never written anywhere: unreproducible by construction.

Two things now stop that, and both are gated here. Structurally, a frozen policy fits nothing, so
its run is a stream of length ONE — the contract refuses anything else, and there are no step
batches to sweep. Defensively, `step` still refuses to sweep a loaded net.

The gate is exact rather than statistical: a frozen eval reports the CHECKPOINT's V_0 to full
precision and fits zero rows. The runs are deliberately tiny — the point is which code paths
execute, not the numbers.
"""
import json as jsonlib
import os

import pytest
import torch

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg(batches, save=None, load=None):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': 16, 'Inner_Sub_Batch': 4,
                 'Inner_MC_Enabled': 'Yes', 'Random_Seed': 1234,
                 'Simulation_Batches': batches})
    calc['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
    solver = {'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 3,
              'Training_Action_Chunk_Size': 64, 'T_Min': 113, 'DiffV2_Fit_Iters': 2,
              'DiffV2_One_Step_Fork': 'Yes'}
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
    """One TRAINING run, saved: two fit batches and a held-out one."""
    path = str(tmp_path_factory.mktemp('vf') / 'value_fn.pt')
    diag, result = _run(_cfg(batches=3, save=path), 'train_streaming')
    assert os.path.exists(path)
    assert result.policy_artifact is not None
    return path, torch.load(path, weights_only=False)


def test_a_loaded_checkpoint_is_not_retrained(checkpoint):
    """The defect, exactly: the intermediate batches ran a full backward sweep with `opt.step()`
    over the loaded nets. A frozen run reports the checkpoint's own V_0 and fits nothing. Its
    single batch is both the warmup bundle and the held-out world."""
    path, ck = checkpoint
    diag, result = _run(_cfg(batches=1, load=path), 'eval_frozen')
    assert diag['per_t'] == [], 'a fit step ran on a loaded checkpoint — it is not frozen'
    assert diag['V_0'] == ck['V_0'], 'V_0 is not the checkpoint\'s — the nets moved'
    assert result.policy_artifact is None, 'a frozen eval has no new value fn to emit'
    assert diag['verdict_is_oos'] is True, 'frozen nets saw none of these paths'


def test_an_eval_may_not_ask_for_fit_batches(checkpoint):
    """The structural half of the guarantee: a frozen policy fits nothing, so a multi-batch stream
    is not an evaluation — it is a request to keep training, and the contract refuses it at the
    JSON boundary rather than silently consuming the extra batches."""
    path, _ck = checkpoint
    with pytest.raises(ValueError, match='requires Simulation_Batches == 1'):
        _run(_cfg(batches=3, load=path), 'eval_too_many_batches')


def test_a_solve_needs_a_held_out_batch():
    """The other half of the contract: training on every batch would leave no unfitted world to
    report the verdict on."""
    with pytest.raises(ValueError, match='requires Simulation_Batches >= 2'):
        _run(_cfg(batches=1), 'train_no_held_out')


def test_the_checkpoint_file_is_untouched_by_an_eval(checkpoint):
    """The retrained nets used to be discarded silently — reported, then thrown away. Nothing
    should write to the file either."""
    path, ck = checkpoint
    before = open(path, 'rb').read()
    _run(_cfg(batches=1, load=path), 'eval_no_write')
    assert open(path, 'rb').read() == before


def test_saving_while_loading_is_a_contradiction(checkpoint):
    """Train (save) and evaluate (load) are separate runs. Setting both used to silently drop the
    save, which is how the retrained-and-discarded nets stayed invisible."""
    path, _ck = checkpoint
    with pytest.raises(ValueError, match='DiffV2_Save_Value_Fn is set alongside'):
        _run(_cfg(batches=1, save=path + '.2', load=path), 'save_and_load')
