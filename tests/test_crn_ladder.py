"""The measuring instrument has to be checked before the things it measures.

`crn_ladder` is what decides whether a reported gradient is real, so it is tested here on synthetic
functions whose true derivatives are known - deliberately NOT on riskflow deals, because the deals
it currently flags are about to be fixed and a gate that has to be edited afterwards was pinning
the defect rather than the instrument.

The property that matters is not "agrees at some bump size" - anything agrees somewhere. It is
that a JUMP makes the ladder scatter instead of converge, because shrinking h then changes how many
samples sit on the wrong side of the discontinuity rather than refining a limit.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from crn_ladder import ladder


def test_smooth_function_gives_a_flat_ladder_and_agrees():
    """f(x) = x^3 at x=2 has f'(2) = 12 exactly. Central differences are O(h^2) accurate, so the
    ladder must be flat and land on the analytic derivative."""
    r = ladder(price=lambda x: x ** 3, aad=12.0, base=2.0)
    assert r.flatness < 1e-4, f'a polynomial should difference cleanly\n{r}'
    assert r.agrees(), str(r)
    assert r.best == pytest.approx(12.0, rel=1e-6)


def test_a_jump_makes_the_ladder_scatter():
    """The failure this instrument exists to catch: a value with a step in it. The pathwise
    derivative is 1 everywhere except the jump, so a gradient of 1.0 looks perfectly sane in
    isolation - it is only the CRN ladder that reveals the missing flux, and it reveals it as
    SCATTER rather than as a consistent offset."""
    jump = lambda x: x + (5.0 if x > 2.0 else 0.0)
    r = ladder(price=jump, aad=1.0, base=2.0)
    assert r.flatness > 1.0, f'a step must not produce a converging ladder\n{r}'
    assert not r.agrees(), f'the instrument accepted a gradient that is missing a jump\n{r}'
    # and the readings blow up as h shrinks - the signature, not a constant bias
    assert r.crn[0] > r.crn[-1] * 5.0, f'expected 5/(2h) growth as h shrinks\n{r}'


def test_a_genuinely_wrong_gradient_is_rejected_even_on_a_flat_ladder():
    """Flatness alone is not the test. A smooth function differenced against the WRONG reported
    gradient gives a perfectly flat ladder that simply does not land on it - both conditions have
    to hold, which is why `agrees` checks them separately."""
    r = ladder(price=lambda x: x ** 3, aad=9.0, base=2.0)
    assert r.flatness < 1e-4, str(r)
    assert not r.agrees(), f'12 was reported as 9 and the instrument accepted it\n{r}'


def test_absolute_rungs_for_inputs_that_can_be_zero():
    """A relative bump is meaningless about zero - a rate, a drift, a spread. f(x)=sin(x) has
    f'(0)=1."""
    r = ladder(price=np.sin, aad=1.0, base=0.0, rungs=(1e-3, 2e-3, 5e-3), absolute=True)
    assert r.agrees(), str(r)


def test_report_grad_hands_out_a_copy_not_a_live_view():
    """`.numpy()` on a CPU tensor returns a VIEW of the live .grad buffer, and torch keeps
    ACCUMULATING into that buffer - so a report already handed to the caller silently rewrites
    itself when the next measure's backward runs. riskflow builds three estimators per batch
    (collva, fva, cva) over the same leaves, so this is reachable rather than theoretical.

    On CUDA `.cpu()` copies and hides it; on the default device it does not, so the two disagree.
    Verified pre-fix: a report of 3.0 read back as 8.0 after a second, unrelated backward."""
    import torch
    from riskflow.pricing import SensitivitiesEstimator
    from riskflow import utils

    x = torch.tensor([2.0], requires_grad=True, dtype=torch.float64)
    params = [(utils.Factor('Leaf', ('X',)), x)]
    first = SensitivitiesEstimator((3.0 * x).sum(), params).report_grad()
    SensitivitiesEstimator((5.0 * x).sum(), params).report_grad()

    assert float(x.grad) == 8.0, 'expected torch to have accumulated 3 + 5 into the leaf'
    assert [v.tolist() for v in first.values()] == [[3.0]], (
        f'the first report changed when a later backward ran: {[v.tolist() for v in first.values()]}')
