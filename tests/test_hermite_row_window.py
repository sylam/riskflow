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
# GATE (a) — the DERIVATION: the bound comes off the rows the pricer's own selection can reach,
# over every deal the fork prices (tradables AND liabilities)
# --------------------------------------------------------------------------------------------
def _cashflows(periods, resets_per_period, reset_step=1.0, start=0.0, first_row=10, pay_lag=5.0):
    """A TensorCashFlows with `periods` cashflows, each carrying `resets_per_period` resets
    `reset_step` days apart and paying `pay_lag` days after its last fixing — the layout
    `utils.make_energy_cashflows` writes (offsets carry [count, offset, settle]). Each reset gets
    its own scenario row, counting up from `first_row`, exactly as `get_scenario_offset` assigns
    them; that column is what the pricer's gather is indexed by."""
    import numpy as np
    from riskflow import utils

    cash, offsets, resets, scen = [], [], [], []
    day, row = start, first_row
    for _ in range(periods):
        first = len(resets)
        for k in range(resets_per_period):
            r = [0.0] * 16
            r[utils.RESET_INDEX_Reset_Day] = day + k * reset_step
            r[utils.RESET_INDEX_Start_Day] = day + k * reset_step
            r[utils.RESET_INDEX_End_Day] = day + k * reset_step
            resets.append(r)
            scen.append(row)
            row += 1
        cf = [0.0] * 12
        cf[utils.CASHFLOW_INDEX_Start_Day] = day
        cf[utils.CASHFLOW_INDEX_End_Day] = day + (resets_per_period - 1) * reset_step
        cf[utils.CASHFLOW_INDEX_Pay_Day] = day + (resets_per_period - 1) * reset_step + pay_lag
        cf[utils.CASHFLOW_INDEX_Nominal] = 100.0
        cash.append(cf)
        offsets.append([resets_per_period, first, 0])
        day += resets_per_period * reset_step + 30.0
        row += 30
    cf_obj = utils.TensorCashFlows(cash, offsets)
    cf_obj.set_resets(resets, scen)
    return cf_obj


def test_declared_bound_is_the_deepest_row_the_pricer_can_select():
    """`pv_energy_cashflows` selects `sim_resets` with NO lower bound and gathers each at its own
    scenario row, so the declared bound is the deepest such row — NOT the widest per-period reset
    span. A 21-fixing period starting at row 10 declares 10 whatever its span implies, and the
    settlement lag (pay = last fixing + 5d) is covered because the row is absolute rather than a
    distance from the cutoff."""
    assert _cashflows(1, 21, first_row=10).deepest_reset_row() == 10.0
    assert _cashflows(1, 21, first_row=10, pay_lag=90.0).deepest_reset_row() == 10.0


def test_declared_bound_spans_the_whole_elapsed_schedule():
    """`sim_resets` is computed before `get_cashflow_start_index` is applied, so a multi-period leg
    re-gathers the fixings of periods that have already PAID. The bound is therefore the FIRST
    period's first row, not the deepest single period's span."""
    assert _cashflows(4, 21, first_row=10).deepest_reset_row() == 10.0
    assert _cashflows(4, 21, first_row=200).deepest_reset_row() == 200.0


def test_bullet_sampling_declares_its_single_fixing():
    """A `ForwardPriceSampleBullet` period carries ONE reset (count == 1). It still sits a
    settlement lag below the cutoff while the cashflow is unpaid, so it must declare that row —
    the `count > 1` filter that made it declare "no history" is the bug."""
    import numpy as np
    bullet = _cashflows(6, 1, first_row=10)
    assert bullet.deepest_reset_row() == 10.0
    assert bullet.deepest_reset_row() != np.inf


def test_a_leg_with_no_simulated_resets_declares_no_history():
    """Nothing simulated (a fixed leg, or a schedule whose fixings are all already known) reads no
    row below the step it is priced at."""
    import numpy as np
    from riskflow import utils

    cf = _cashflows(1, 5)
    cf.set_resets(cf.Resets.schedule.tolist(), [-1] * len(cf.Resets.schedule))
    assert cf.deepest_reset_row() == np.inf
    empty = utils.TensorCashFlows([[0.0] * 12], [[0, 0, 0]])
    assert empty.deepest_reset_row() == np.inf


def test_book_bound_is_fail_safe_when_a_deal_does_not_declare():
    """A deal that returns None (does not know what it reads) makes the BOOK's bound None, which
    switches the window off entirely rather than sizing it from the deals that did declare."""
    import numpy as np
    from riskflow.calculation import HedgeMonteCarlo

    class _Struct:
        def __init__(self, declared):
            self._d = declared

        def history_declarations(self):
            return self._d

    def floor(tradables, liabilities):
        calc = HedgeMonteCarlo.__new__(HedgeMonteCarlo)
        calc.netting_sets, calc.liabilities = _Struct(tradables), _Struct(liabilities)
        return calc._hermite_floor_row()

    assert floor({'FUT': np.inf}, {'AVG': 82.0}) == 82.0
    assert floor({'FUT': np.inf}, {'FIX': np.inf}) == np.inf
    assert floor({'FUT': None}, {'AVG': 82.0}) is None
    assert floor({'FUT': np.inf}, {'AVG': None}) is None


def test_the_bound_covers_the_tradables_not_only_the_liability():
    """One window is in force while the fork prices BOTH structures, so an averaging TRADABLE that
    reads deeper than the liability must move the bound. Deriving it from the liability alone is
    what silently retired such a tradable from the hedge set."""
    import numpy as np
    from riskflow.calculation import HedgeMonteCarlo

    class _Struct:
        def __init__(self, declared):
            self._d = declared

        def history_declarations(self):
            return self._d

    calc = HedgeMonteCarlo.__new__(HedgeMonteCarlo)
    calc.netting_sets = _Struct({'AVG_SWAP': 5.0, 'FUT': np.inf})
    calc.liabilities = _Struct({'AVG_OFFTAKE': 82.0})
    calc._hermite_floor = calc._hermite_floor_row()
    assert calc._hermite_floor == 5.0
    assert calc._hermite_window(100, 2) == (5, 102)


def test_window_is_the_forks_own_rows_when_nothing_reads_history():
    """Every deal declaring `inf` means no row below the fork's own step is read, so the window
    starts AT the cutoff — the tightest correct window, not a disabled one."""
    import numpy as np
    from riskflow.calculation import HedgeMonteCarlo

    calc = HedgeMonteCarlo.__new__(HedgeMonteCarlo)
    calc._hermite_floor = np.inf
    assert calc._hermite_window(100, 2) == (100, 102)
    calc._hermite_floor = None
    assert calc._hermite_window(100, 2) is None
