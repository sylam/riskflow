"""Lazily populated Hermite coefficients — the correctness locks.

`Interpolation` builds its Hermite `g,c` pair on the FIRST gather, for the scenario rows that
gather indexes, because the recursion in `hermite_interpolation_tensor` couples along the TENOR
axis only: every difference it takes is over dim 1, so a scenario row's coefficients depend on
that row alone. That is the property everything else rests on, so it is tested directly (gate 1),
then the deferred build is tested against the eager full block it replaces (gate 2), then the
growth behaviour a shared, long-lived `Interpolation` depends on (gate 3).

Nothing predicts which rows will be read: an inner-MC fork, base valuation and credit Monte Carlo
all take the same path and differ only in what their gathers ask for.
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
# GATE 1 — row independence: a row slice's coefficients == the same rows of the full block
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('scen,n_tenors,batch', [(119, 31, 8), (119, 31, 1), (40, 4, 3)])
@pytest.mark.parametrize('lo,hi', [(0, 0), (0, 5), (37, 74), (60, 118), (118, 118), (0, 118)])
def test_sliced_coefficients_are_bitwise_slices_of_the_full_block(scen, n_tenors, batch, lo, hi):
    """The whole design in one assertion: building g,c for rows [lo, hi] gives EXACTLY the rows
    [lo, hi] of building them for everything. Covers the spans a gather actually produces — at
    the grid start, at the terminal row, and the degenerate single-row span."""
    if hi >= scen:
        pytest.skip('span beyond this block')
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
# GATE 2 — the deferred build returns what the eager full block would have
# --------------------------------------------------------------------------------------------
SCEN, N_TENORS, BATCH = 119, 31, 6


def _gather(interp, rows, alpha=None):
    """Drive Interpolation.eval the way the pricers do: a set of scenario rows, the standard
    two-tenor bracket, and a fixed interpolation weight."""
    idx = torch.tensor(rows, dtype=torch.int64).reshape(-1, 1) * N_TENORS
    nxt = (torch.tensor(rows, dtype=torch.int64).clamp(max=SCEN - 2) + 1).reshape(-1, 1) * N_TENORS \
        if alpha is not None else None
    w2 = torch.full((len(rows), 1, BATCH), 0.35, dtype=torch.float64)
    tnr = torch.full((len(rows), 1), 5.0, dtype=torch.float64)
    return interp.eval(('Hermite', 0.08, 30.0), 3, 4, idx, nxt, w2, tnr, alpha=alpha)


def _eager(curve, t):
    """The pre-feature reference: an Interpolation whose coefficients cover the whole block."""
    interp = Interpolation(curve, hermite_tenor=t)
    interp.hermite_params(torch.tensor([[0], [(SCEN - 1) * N_TENORS]]), None)
    assert interp.rows == (0, SCEN - 1)
    return interp


@pytest.mark.parametrize('rows', [
    list(range(0, 41, 7)), list(range(20, 61, 7)), list(range(70, 119, 7)), [83], [0], [118]])
def test_a_lazily_built_gather_matches_the_full_block(rows):
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=3)
    lazy = Interpolation(curve, hermite_tenor=t)
    assert lazy.interp_params == [] and lazy.rows is None, 'coefficients built before any gather'
    got = _gather(lazy, rows)
    assert lazy.rows == (min(rows), max(rows)), 'built rows are not the rows the gather named'
    assert torch.equal(got, _gather(_eager(curve, t), rows)), 'lazy gather diverged'


def test_the_alpha_branch_covers_the_next_row_too():
    """With time interpolation the gather also reads g,c at `t_index_next`, one scenario row up —
    the span must include it or the second `calc_hermite_curve` reads the wrong row."""
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=11)
    rows = [40, 55, 70]
    alpha = torch.full((len(rows), 1, 1), 0.4, dtype=torch.float64)
    lazy = Interpolation(curve, hermite_tenor=t)
    got = _gather(lazy, rows, alpha=alpha)
    assert lazy.rows == (40, 71)
    assert torch.equal(got, _gather(_eager(curve, t), rows, alpha=alpha))


# --------------------------------------------------------------------------------------------
# GATE 3 — growth: one Interpolation serves many gathers over its life
# --------------------------------------------------------------------------------------------
def test_a_later_gather_outside_the_built_span_extends_it():
    """An `Interpolation` is cached per curve factor and gathered by every deal in the book, so a
    later, deeper read is the normal case — not an error. It must widen the span and still answer
    with the full-block value. This is what the declared window used to have to predict."""
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=5)
    lazy, eager = Interpolation(curve, hermite_tenor=t), _eager(curve, t)
    _gather(lazy, [80, 81])
    assert lazy.rows == (80, 81)
    for rows in ([40], [90, 100], [0, 118]):                       # below, above, everything
        assert torch.equal(_gather(lazy, rows), _gather(eager, rows))
    assert lazy.rows == (0, 118)


def test_a_gather_inside_the_built_span_does_not_rebuild():
    """The amortisation the design depends on: rows already covered are served from what is built,
    so an object gathered many times pays for its coefficients once."""
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=7)
    lazy = Interpolation(curve, hermite_tenor=t)
    _gather(lazy, [30, 90])
    built = lazy.interp_params
    for rows in ([30], [60], [90], [31, 89]):
        _gather(lazy, rows)
        assert lazy.interp_params is built, f'rebuilt for {rows}, which is already covered'
        assert lazy.rows == (30, 90)


def test_an_empty_gather_names_no_rows():
    """A step with no resets in range gathers an EMPTY index set. It names no rows, so it must not
    disturb the span (nor take a min over an empty tensor)."""
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=9)
    lazy = Interpolation(curve, hermite_tenor=t)
    assert _gather(lazy, []).shape == (0, 1, BATCH)
    assert lazy.rows == (0, 0), 'an empty gather should build the degenerate span, not the block'
    _gather(lazy, [50, 60])
    after = lazy.interp_params
    _gather(lazy, [])
    assert lazy.interp_params is after and lazy.rows == (0, 60)


def test_the_full_grid_consumer_builds_the_whole_block():
    """Base valuation, credit Monte Carlo and the outer hedge loop gather every scenario row, so
    they get the whole block — the same coefficients as before, built on their first read."""
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=13)
    lazy = Interpolation(curve, hermite_tenor=t)
    got = _gather(lazy, list(range(SCEN)))
    assert lazy.rows == (0, SCEN - 1) and lazy.row_offset == 0
    g_full, c_full = hermite_interpolation_tensor(t, curve)
    assert torch.equal(lazy.interp_params[0], g_full.reshape(-1, BATCH))
    assert torch.equal(lazy.interp_params[1], c_full.reshape(-1, BATCH))
    assert torch.equal(got, _gather(_eager(curve, t), list(range(SCEN))))


# --------------------------------------------------------------------------------------------
# GATE 4 — the two-segment curve (`Near_Interpolation` + `Near_Date` on the FACTOR)
# --------------------------------------------------------------------------------------------
def test_a_two_segment_curve_defers_per_segment():
    """`Factor1D` splits its `interpolation` at the near index when the factor declares
    `Near_Interpolation`; `update_tenors` then carries a ((start, end, kind), ...) tuple instead of
    a single kind, and `make_curve_tensor` builds ONE Interpolation per segment over a tenor SLICE
    of the same scenario rows. Each segment therefore has its own n_tenors, so each must resolve
    its own row span from its own flat indices — no in-repo factor declares the split today, which
    is exactly why it is locked here."""
    from riskflow.utils import SegmentedInterpolation
    n_tenors, split, batch = 8, 3, 5
    curve = _curve(60, n_tenors, batch, seed=17)
    tenor = torch.linspace(0.08, 30.0, n_tenors, dtype=torch.float64)
    spec = ((0, split, 'Hermite'), (split, n_tenors - 1, 'Hermite'))

    def segments():
        return [Interpolation(curve[:, s:e + 1, :],
                              hermite_tenor=tenor[s:e + 1].reshape(1, -1, 1)) for s, e, _ in spec]

    lazy, eager = segments(), segments()
    for seg in eager:                                          # pre-build the whole block
        seg.hermite_params(torch.tensor([[0], [59 * seg.tensor.shape[1]]]), None)

    rows = [12, 33, 44]
    i1 = torch.tensor([[0], [2], [5]])
    tenor_data = (spec, (0.08, float(tenor[split])), (float(tenor[split]), 30.0))
    w2 = torch.full((len(rows), 1, batch), 0.35, dtype=torch.float64)
    tnr = torch.full((len(rows), 1), 5.0, dtype=torch.float64)

    def run(segs):
        obj = SegmentedInterpolation(segs, curve)
        _a, _n, t_index, t_next = obj.get_time_index(torch.tensor(rows), None)
        return obj.eval(tenor_data, i1, i1 + 1, t_index, t_next, w2, tnr)

    assert torch.equal(run(lazy), run(eager)), 'segmented lazy build diverged from the full block'
    assert [s.rows for s in lazy] == [(12, 44), (12, 44)], 'a segment resolved the wrong rows'
    assert [s.row_offset for s in lazy] == [12 * 4, 12 * 5], 'offset ignores the segment n_tenors'
