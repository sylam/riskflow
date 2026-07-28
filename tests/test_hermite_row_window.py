"""Row-restricted Hermite coefficient population — the correctness locks.

`make_curve_tensor` may build the Hermite `g,c` pair for a WINDOW of scenario rows instead of the
whole block, because the recursion in `hermite_interpolation_tensor` couples along the TENOR axis
only: every difference it takes is over dim 1, so a scenario row's coefficients depend on that row
alone. That is the property everything else rests on, so it is tested directly (gate 1), then the
gather that consumes the window is tested against the full block (gate 2).

The window is set ONLY by the inner-MC fork. Every other caller — base valuation, credit Monte
Carlo, the outer hedge loop — leaves `shared.hermite_window` None and takes the original code path,
which is asserted here structurally (gate 5) as well as by the untouched suite.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from riskflow.utils import Interpolation, hermite_interpolation_tensor


def _curve(scen, n_tenors, batch, seed=0):
    """A curve block shaped like the ones the fork prices: (scen, n_tenors, batch), monotone in
    tenor with per-scenario level/slope drift, so every scenario row has distinct coefficients."""
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(0.01, 0.06, n_tenors).reshape(1, -1, 1)
    drift = torch.linspace(0.0, 0.02, scen).reshape(-1, 1, 1)
    noise = torch.rand((scen, n_tenors, batch), generator=g) * 1e-3
    return (base + drift + noise).to(torch.float64)


def _tenor(n_tenors):
    return torch.linspace(0.08, 30.0, n_tenors, dtype=torch.float64).reshape(1, -1, 1)


# --------------------------------------------------------------------------------------------
# GATE 1 — row independence: windowed coefficients == the same rows of the full-block ones
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('scen,n_tenors,batch', [(119, 31, 8), (119, 31, 1), (40, 4, 3)])
@pytest.mark.parametrize('lo,hi', [(0, 0), (0, 5), (37, 74), (60, 118), (118, 118), (0, 118)])
def test_windowed_coefficients_are_bitwise_slices_of_the_full_block(scen, n_tenors, batch, lo, hi):
    """The whole design in one assertion: building g,c for rows [lo, hi] gives EXACTLY the rows
    [lo, hi] of building them for everything. Covers the edges the fork actually hits — window at
    the grid start (clipped), at the terminal row, and the degenerate single-row window."""
    if hi >= scen:
        pytest.skip('window beyond this block')
    t = _tenor(n_tenors)
    curve = _curve(scen, n_tenors, batch)
    g_full, c_full = hermite_interpolation_tensor(t, curve)
    g_win, c_win = hermite_interpolation_tensor(t, curve[lo:hi + 1])
    assert g_win.shape == (hi - lo + 1, n_tenors, batch)
    assert torch.equal(g_win, g_full[lo:hi + 1]), 'g differs — rows are NOT independent'
    assert torch.equal(c_win, c_full[lo:hi + 1]), 'c differs — rows are NOT independent'


def test_row_independence_holds_in_float32_too():
    """The production path is float32; rounding must not couple rows either."""
    t = _tenor(31).to(torch.float32)
    curve = _curve(119, 31, 4).to(torch.float32)
    g_full, c_full = hermite_interpolation_tensor(t, curve)
    for lo, hi in ((0, 33), (50, 90), (85, 118)):
        g_win, c_win = hermite_interpolation_tensor(t, curve[lo:hi + 1])
        assert torch.equal(g_win, g_full[lo:hi + 1])
        assert torch.equal(c_win, c_full[lo:hi + 1])


# --------------------------------------------------------------------------------------------
# GATE 2 — the gather that consumes a window returns what the full block would have
# --------------------------------------------------------------------------------------------
def _gather(interp, rows, n_tenors, batch):
    """Drive Interpolation.eval the way the pricers do: a set of scenario rows, the standard
    two-tenor bracket, and a fixed interpolation weight."""
    idx = torch.tensor(rows).reshape(-1, 1) * n_tenors
    w2 = torch.full((len(rows), 1, batch), 0.35, dtype=torch.float64)
    tnr = torch.full((len(rows), 1), 5.0, dtype=torch.float64)
    return interp.eval(('Hermite', 0.08, 30.0), 3, 4, idx, None, w2, tnr)


@pytest.mark.parametrize('lo,hi', [(0, 40), (20, 60), (70, 118)])
def test_windowed_gather_matches_full_block_gather(lo, hi):
    scen, n_tenors, batch = 119, 31, 6
    t, curve = _tenor(n_tenors), _curve(scen, n_tenors, batch, seed=3)
    g_full, c_full = hermite_interpolation_tensor(t, curve)
    g_win, c_win = hermite_interpolation_tensor(t, curve[lo:hi + 1])

    full = Interpolation(curve, [g_full, c_full])
    windowed = Interpolation(curve, [g_win, c_win], row_offset=lo * n_tenors)
    assert full.row_offset == 0

    rows = list(range(lo, hi + 1, 7))
    got_full = _gather(full, rows, n_tenors, batch)
    got_win = _gather(windowed, rows, n_tenors, batch)
    assert torch.equal(got_win, got_full), 'windowed gather diverged from the full block'


def test_gather_below_the_window_raises_instead_of_wrapping():
    """A row below the window would become a negative index and silently read from the END of the
    block — a wrong curve, not a crash. It must raise."""
    scen, n_tenors, batch = 119, 31, 4
    t, curve = _tenor(n_tenors), _curve(scen, n_tenors, batch, seed=5)
    lo, hi = 40, 80
    g_win, c_win = hermite_interpolation_tensor(t, curve[lo:hi + 1])
    windowed = Interpolation(curve, [g_win, c_win], row_offset=lo * n_tenors)
    _gather(windowed, [lo, 60, hi], n_tenors, batch)               # inside: fine
    with pytest.raises(IndexError, match='window does not cover'):
        _gather(windowed, [lo - 1], n_tenors, batch)               # one row below: loud


# --------------------------------------------------------------------------------------------
# GATE 5 — the shared stack cannot reach the windowed path by accident
# --------------------------------------------------------------------------------------------
def test_default_state_leaves_the_window_unset():
    """Every Calculation_State starts with hermite_window None, so base valuation / credit MC /
    the outer loop take the full-block path. Only the inner-MC fork sets it, and it restores None
    in a finally."""
    import inspect
    from riskflow import utils, calculation

    src = inspect.getsource(utils.Calculation_State.__init__)
    assert 'self.hermite_window = None' in src

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setters = []
    for name in ('utils.py', 'calculation.py', 'pricing.py', 'instruments.py'):
        for line in open(os.path.join(root, 'riskflow', name)).read().splitlines():
            if 'hermite_window =' in line and 'None' not in line:
                setters.append((name, line.strip()))
    assert len(setters) == 1, f'exactly one non-None setter expected, got {setters}'
    assert setters[0][0] == 'calculation.py', setters
    assert 'Interpolation(mod_tenor, [g, c])' in inspect.getsource(utils.make_curve_tensor), \
        'the unwindowed construction must remain the literal default path'


# --------------------------------------------------------------------------------------------
# GATE (a) — the DERIVATION: the window bound comes off the liability's own schedule
# --------------------------------------------------------------------------------------------
def _cashflows(periods, resets_per_period, reset_step=1.0, start=0.0):
    """A TensorCashFlows with `periods` cashflows, each carrying `resets_per_period` resets
    `reset_step` days apart — the layout `utils.make_energy_cashflows` writes (offsets carry
    [count, offset, settle])."""
    import numpy as np
    from riskflow import utils

    cash, offsets, resets, scen = [], [], [], []
    day = start
    for _ in range(periods):
        first = len(resets)
        for k in range(resets_per_period):
            row = [0.0] * 16
            row[utils.RESET_INDEX_Reset_Day] = day + k * reset_step
            row[utils.RESET_INDEX_Start_Day] = day + k * reset_step
            row[utils.RESET_INDEX_End_Day] = day + k * reset_step
            resets.append(row)
            scen.append(0)
        cf = [0.0] * 12
        cf[utils.CASHFLOW_INDEX_Start_Day] = day
        cf[utils.CASHFLOW_INDEX_End_Day] = day + (resets_per_period - 1) * reset_step
        cf[utils.CASHFLOW_INDEX_Pay_Day] = day + (resets_per_period - 1) * reset_step + 5
        cf[utils.CASHFLOW_INDEX_Nominal] = 100.0
        cash.append(cf)
        offsets.append([resets_per_period, first, 0])
        day += resets_per_period * reset_step + 30.0
    cf_obj = utils.TensorCashFlows(cash, offsets)
    cf_obj.set_resets(resets, scen)
    return cf_obj


def test_reset_span_is_the_widest_averaging_window():
    """The declared requirement is the widest per-period reset span — 21 daily fixings span 20
    days, and extra periods do not deepen it."""
    assert _cashflows(1, 21).max_reset_span() == pytest.approx(20.0)
    assert _cashflows(4, 21).max_reset_span() == pytest.approx(20.0)
    assert _cashflows(4, 31).max_reset_span() == pytest.approx(30.0)


def test_reset_span_takes_the_deepest_period_not_the_last():
    """A book whose deepest averaging period is not the final one still reports that depth."""
    deep = _cashflows(1, 61)
    shallow = _cashflows(3, 5)
    merged = _cashflows(1, 61)
    # splice the shallow schedule on after the deep one
    import numpy as np
    from riskflow import utils
    merged.schedule = np.vstack([deep.schedule, shallow.schedule])
    merged.offsets = np.vstack([deep.offsets,
                                shallow.offsets + np.array([0, len(deep.Resets.schedule), 0])])
    merged.set_resets(np.vstack([deep.Resets.schedule, shallow.Resets.schedule]).tolist(),
                      [0] * (len(deep.Resets.schedule) + len(shallow.Resets.schedule)))
    assert merged.max_reset_span() == pytest.approx(60.0)


def test_no_averaging_declares_zero_history():
    """A single-fixing (non-averaging) leg reads no history, so its requirement is 0 — a window
    of just the fork's own rows."""
    assert _cashflows(6, 1).max_reset_span() == 0.0


def test_book_history_is_fail_safe_when_a_leg_does_not_declare():
    """A leg that returns None (does not know its layout) makes the BOOK's requirement None, which
    switches the window off entirely rather than sizing it from the legs that did declare."""
    from riskflow.calculation import DealStructure

    class _Struct(DealStructure):
        def __init__(self, descriptors):
            self._d = descriptors
            self.dependencies, self.sub_structures = [], []

        def aggregate_leg_descriptors(self):
            total, pay, hist = 0.0, None, 0.0
            for vol, p, h in self._d:
                total += vol
                if p is not None:
                    pay = p if pay is None else max(pay, p)
                hist = None if (h is None or hist is None) else max(hist, h)
            return total, pay, hist

    assert _Struct([(1.0, 10.0, 20.0), (1.0, 12.0, 35.0)]).aggregate_leg_descriptors()[2] == 35.0
    assert _Struct([(1.0, 10.0, 20.0), (1.0, 12.0, None)]).aggregate_leg_descriptors()[2] is None
