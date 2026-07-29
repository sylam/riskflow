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

import numpy as np
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


def _tenor_np(n_tenors):
    """The factor's tenor grid, as `CurveTenor` carries it."""
    return np.linspace(0.08, 30.0, n_tenors)


def _tenor(n_tenors):
    """The same grid in the shape `hermite_interpolation_tensor` wants. Derived from the NUMPY
    tenor, because that is what the interpolation kind builds from — `torch.linspace` and
    `np.linspace` disagree in the last bits, and a reference built the other way is not the
    production grid."""
    return torch.from_numpy(_tenor_np(n_tenors)).reshape(1, -1, 1)


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


def _hermite(curve, _t):
    return Interpolation.build(curve, 'Hermite', _tenor_np(curve.shape[1]))


def _gather(interp, rows, alpha=None):
    """Drive the leaf's eval the way the pricers do: a set of scenario ROWS (the leaf flattens
    them against its own tenor stride), the standard two-tenor bracket, and a fixed weight."""
    idx = torch.tensor(rows, dtype=torch.int64)
    nxt = idx.clamp(max=SCEN - 2) + 1 if alpha is not None else None
    w2 = torch.full((len(rows), 1, BATCH), 0.35, dtype=torch.float64)
    tnr = torch.full((len(rows), 1), 5.0, dtype=torch.float64)
    return interp.eval(('Hermite', 0.08, 30.0), idx, nxt, alpha, 3, 4, w2, tnr, 1.0)


def _eager(curve, t):
    """The pre-feature reference: an Interpolation whose coefficients cover the whole block."""
    interp = _hermite(curve, t)
    interp.params(torch.tensor([[0], [(SCEN - 1) * N_TENORS]]), None)
    assert interp.rows == (0, SCEN - 1)
    return interp


def _counted(interp, tally):
    """`interp`, with the scenario-row count of every coefficient build appended to `tally`."""
    build = interp.build_rows
    interp.build_rows = lambda lo, hi: (tally.append(hi - lo + 1), build(lo, hi))[1]
    return interp


@pytest.mark.parametrize('rows', [
    list(range(0, 41, 7)), list(range(20, 61, 7)), list(range(70, 119, 7)), [83], [0], [118]])
def test_a_lazily_built_gather_matches_the_full_block(rows):
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=3)
    lazy = _hermite(curve, t)
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
    lazy = _hermite(curve, t)
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
    lazy, eager = _hermite(curve, t), _eager(curve, t)
    _gather(lazy, [80, 81])
    assert lazy.rows == (80, 81)
    for rows in ([40], [90, 100], [0, 118]):                       # below, above, everything
        assert torch.equal(_gather(lazy, rows), _gather(eager, rows))
    assert lazy.rows == (0, 118)


def test_a_widening_builds_only_the_rows_it_adds():
    """One `Interpolation` is cached per curve factor and gathered by every deal, so a book priced
    in ascending maturity widens the span one row at a time. Re-deriving the union each time is
    quadratic — measured on credit MC, 7501 rows built for a 121-row block. Only the new rows may
    be built, and the spliced pair must still equal the eager one row for row."""
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=11)
    lazy, eager = _hermite(curve, t), _eager(curve, t)
    built = []
    _counted(lazy, built)
    for row in range(20, 40):                                      # ascending, one row at a time
        _gather(lazy, [row])
    _gather(lazy, [5])                                             # then widen DOWNWARDS
    assert built == [1] * 20 + [15], f'rebuilt the union rather than splicing: {built}'
    assert lazy.rows == (5, 39)
    assert torch.equal(_gather(lazy, list(range(5, 40))), _gather(eager, list(range(5, 40))))


def test_a_gather_inside_the_built_span_does_not_rebuild():
    """The amortisation the design depends on: rows already covered are served from what is built,
    so an object gathered many times pays for its coefficients once."""
    t, curve = _tenor(N_TENORS), _curve(SCEN, N_TENORS, BATCH, seed=7)
    lazy = _hermite(curve, t)
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
    lazy = _hermite(curve, t)
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
    lazy = _hermite(curve, t)
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
    tenor = np.linspace(0.08, 30.0, n_tenors)
    spec = ((0, split, 'Hermite'), (split, n_tenors - 1, 'Hermite'))

    rows = torch.tensor([12, 33, 44])
    i1 = torch.tensor([[0], [2], [5]])
    w2 = torch.full((len(rows), 1, batch), 0.35, dtype=torch.float64)
    tnr = torch.full((len(rows), 1), 5.0, dtype=torch.float64)
    split_tenor = tenor[split]
    tenor_data = (spec, (0.08, split_tenor), (split_tenor, 30.0))

    def run(eager):
        obj = SegmentedInterpolation(curve, spec, tenor)
        if eager:                                              # pre-build the whole block
            for seg in obj.segments:
                seg.params(torch.tensor([[0], [59 * seg.shape[1]]]), None)
        return obj, obj.eval(tenor_data, rows, None, None, i1, i1 + 1, w2, tnr, 1.0)

    lazy, got = run(False)
    _eagerly, want = run(True)
    assert torch.equal(got, want), 'segmented lazy build diverged from the full block'
    assert [s.rows for s in lazy.segments] == [(12, 44), (12, 44)], 'a segment resolved wrong rows'
    assert [s.row_offset for s in lazy.segments] == [12 * 4, 12 * 5], \
        'offset ignores the segment n_tenors'


# --------------------------------------------------------------------------------------------
# GATE 5 — a scenario source is a SEQUENCE of row blocks, and one block is the ordinary case
#
# An inner-MC fork publishes two blocks: the outer-realized past at B_outer, then the forked rows
# at B_flat. Every other caller publishes one, at the simulation width. `Interpolation` reads
# through the sequence either way, so these gates assert the two-block answer against the JOINED
# grid — the tensor the fork used to build with a `cat` of an expanded past — and assert that the
# one-block source is that mechanism's trivial instance, not a path beside it.
# --------------------------------------------------------------------------------------------
from riskflow import utils
from riskflow.utils import (CurveTensor, Factor, Interpolation, RoutedInterpolation,
                            ScenarioBlock, ScenarioSource, SegmentedInterpolation,
                            build_interpolation, make_curve_tensor)

CUTOFF, T_INNER, B_OUTER, FAN = 100, 2, 5, 4          # B_flat = B_OUTER * FAN


class Shared:
    """`make_curve_tensor` reaches for exactly one thing on `shared`. A FRESH memo per call: the
    key is `(interpolation, factor, logical shape)`, so a block sequence and the grid it joins to
    hash the same — correct inside one fork (`t_Buffer` is cleared before pricing and again in the
    `finally`), and a vacuous comparison here if the two ever shared one."""

    def __init__(self):
        self.t_Buffer = {}


def _past_columns(fan=FAN, width=B_OUTER):
    """The map a fork hands over: logical flat column -> the outer column that supplies it."""
    return torch.arange(width * fan) // fan


def _source(past, inner, fan=FAN):
    return ScenarioSource(ScenarioBlock(past, batch_index=_past_columns(fan, past.shape[-1])),
                          ScenarioBlock(inner, first_row=past.shape[0]))


def _join(past, inner, fan=FAN):
    """The reference the routed answer is measured against: the tensor the fork used to build,
    a `cat` of the past written out `fan` times. Constructed HERE rather than asked of production
    code, so the gate cannot be satisfied by the thing it is testing."""
    wide = past.unsqueeze(-1).expand(*past.shape, fan).reshape(*past.shape[:-1], -1)
    return torch.cat([wide, inner], dim=0)


def _blocks(n_tenors=N_TENORS, seed=21):
    """A past block at B_OUTER and a forked block at B_flat, related exactly as a fork relates
    them: every past row repeated FAN times across the flat batch."""
    past = _curve(CUTOFF, n_tenors, B_OUTER, seed=seed)
    inner = _curve(T_INNER, n_tenors, B_OUTER * FAN, seed=seed + 1)
    return past, inner


def _time_grid(rows, alpha):
    """The three columns `gather_scenario_interp` reads: prior-scenario delta, mtm, prior index."""
    grid = np.zeros((len(rows), 3))
    grid[:, utils.TIME_GRID_ScenarioPriorIndex] = rows
    grid[:, utils.TIME_GRID_PriorScenarioDelta] = alpha
    grid[:, utils.TIME_GRID_MTM] = np.arange(len(rows))
    return grid


def _component(n_tenors, kind):
    tenor = np.linspace(0.08, 30.0, n_tenors)
    return (True, Factor('InterestRate', ('T',)), None, utils.CurveTenor(tenor, kind),
            lambda points: points / 365.0)


def _gathered(source, rows, alpha, kind, n_tenors=N_TENORS, points=(30.0, 400.0, 4000.0)):
    """Drive the WHOLE read path — make_curve_tensor -> CurveTensor (which decides the routing)
    -> interpolate_curve -> the strategy's eval — exactly as the pricer does."""
    component = _component(n_tenors, kind)
    curve_tensor = make_curve_tensor(source, component, _time_grid(rows, alpha), Shared())
    query = np.tile(np.array(points), (len(rows), 1))
    return curve_tensor, curve_tensor.interpolate_curve(component, query, 1.0)


ROW_SETS = {
    'all past': [3, 40, 97],
    'all forked': [CUTOFF, CUTOFF + 1],
    'mixed': [0, 50, 98, CUTOFF, CUTOFF + 1],
    'straddling the cut': [CUTOFF - 1],           # with alpha this reads BOTH blocks
    'deep history then the cut': list(range(0, CUTOFF + 2, 7)) + [CUTOFF - 1],
    'empty': [],
}


@pytest.mark.parametrize('kind', ['Hermite', 'Linear', 'HermiteRT', 'LinearRT'])
@pytest.mark.parametrize('alpha', [0.0, 0.4])
@pytest.mark.parametrize('label', list(ROW_SETS))
def test_a_two_block_gather_equals_the_joined_grid(label, alpha, kind):
    """The whole change in one assertion, over every interpolation kind the factor channel can
    declare and both time-interpolation states. `alpha != 0` is what makes a row at `cutoff - 1`
    read ACROSS the cut, which is the case the routing has to get right."""
    past, inner = _blocks()
    rows = ROW_SETS[label]
    split_tensor, split = _gathered(_source(past, inner), rows, alpha, kind)
    joined_tensor, joined = _gathered(_join(past, inner), rows, alpha, kind)
    assert isinstance(split_tensor.interp_obj, RoutedInterpolation), 'the split arm did not route'
    assert isinstance(joined_tensor.interp_obj, Interpolation), 'the joined arm is not a leaf'
    assert split.shape == joined.shape == (len(rows), 3, B_OUTER * FAN)
    assert torch.equal(split, joined), f'{label} / alpha={alpha} / {kind} diverged from the join'


def test_a_leaf_knows_nothing_about_blocks():
    """The separation, asserted directly. An ordinary grid builds a LEAF — no cuts, no routing, no
    block offsets, no batch map — and a fork builds a composite over leaves that are themselves
    just as ignorant. Base valuation and credit Monte Carlo only ever get the leaf."""
    tenor = utils.tenor_diff(np.linspace(0.08, 30.0, N_TENORS))
    whole = build_interpolation(_curve(CUTOFF + T_INNER, N_TENORS, B_OUTER * FAN), tenor)
    assert isinstance(whole, Interpolation)
    for attr in ('blocks', 'cuts', 'first_row', 'batch_index', 'rebase', 'broadcast'):
        assert not hasattr(whole, attr), f'a leaf grew a composite concern: {attr}'
    assert whole.interp_params == [] and whole.hermite_tenor is None, 'a Linear leaf has no params'
    for has_alpha in (False, True):
        assert whole.route(np.array([0, 7, 99]), has_alpha) is None, 'a leaf has nothing to route'

    past, inner = _blocks()
    forked = build_interpolation(_source(past, inner), tenor)
    assert isinstance(forked, RoutedInterpolation)
    assert all(isinstance(s, Interpolation) for s in forked.strategies)
    assert list(forked.cuts) == [CUTOFF]
    assert forked.shape == (CUTOFF + T_INNER, N_TENORS, B_OUTER * FAN)
    # the routing names blocks, never "was this a fork"
    assert forked.route(np.array([0, 5]), False) == ((None, 0, 0),)
    assert forked.route(np.array([CUTOFF, CUTOFF + 1]), False) == ((None, 1, 1),)
    assert [g[1:] for g in forked.route(np.array([CUTOFF - 1]), True)] == [(0, 1)]


def test_the_hermite_pair_is_built_per_block_at_that_block_s_own_width():
    """The second half of the prize: a past row's coefficients are identical across the forked
    draws, so they are built ONCE per outer path, not once per flat sample."""
    past, inner = _blocks()
    curve_tensor, _ = _gathered(_source(past, inner), [10, 20, 30], 0.0, 'Hermite')
    past_leaf, forked_leaf = curve_tensor.interp_obj.strategies
    assert past_leaf.interp_params[0].shape[-1] == B_OUTER, 'past coefficients are flat-width'
    assert forked_leaf.interp_params == [], 'the forked block was built for a gather below it'
    g_full, c_full = hermite_interpolation_tensor(          # the block's own tenor grid, exactly
        past_leaf.hermite_tenor, past[past_leaf.rows[0]:past_leaf.rows[1] + 1])
    assert torch.equal(past_leaf.interp_params[0], g_full.reshape(-1, B_OUTER))
    assert torch.equal(past_leaf.interp_params[1], c_full.reshape(-1, B_OUTER))


@pytest.mark.parametrize('alpha', [0.0, 0.4])
@pytest.mark.parametrize('label', list(ROW_SETS))
def test_the_spot_path_routes_the_same_way(label, alpha):
    """`calc_time_grid_spot_rate` gathers whole rows off a 0D factor (2-D buffer), which is the
    other read surface — same routing, on the row axis instead of the flat (row, tenor) one."""
    past = _curve(CUTOFF, 1, B_OUTER, seed=31)[:, 0, :]
    inner = _curve(T_INNER, 1, B_OUTER * FAN, seed=32)[:, 0, :]
    rows = ROW_SETS[label]
    grid = _time_grid(rows, alpha)
    flat = utils.tenor_diff(np.zeros(1))
    split = utils.gather_scenario_interp(
        build_interpolation(_source(past, inner), flat), grid, None, as_curve_tensor=False)
    joined = utils.gather_scenario_interp(
        build_interpolation(_join(past, inner), flat), grid, None, as_curve_tensor=False)
    assert split.shape == joined.shape == (len(rows), B_OUTER * FAN)
    assert torch.equal(split, joined)


def test_a_gradient_reaches_both_blocks():
    """The forked block carries the tape and the realized past is detached, so a mixed gather
    combines a grad and a no-grad term — the result must still require grad, and the gradient must
    be the one the joined grid gives. The FORWARD is bitwise (above); the backward is compared to a
    tolerance because the joined arm sums through two extra graph nodes (the expand and the cat) in
    a different order, which no production path does — the fork reads the blocks directly, and the
    grad forks themselves are bitwise-gated end to end by `tb_golden_worlds.py`."""
    past, inner = _blocks()
    rows, out = [0, 44, CUTOFF - 1, CUTOFF, CUTOFF + 1], {}
    for name in ('split', 'joined'):
        leaf = inner.clone().requires_grad_(True)
        value = _source(past, leaf) if name == 'split' else _join(past, leaf)
        _, val = _gathered(value, rows, 0.4, 'Hermite')
        assert val.requires_grad, f'{name}: the tape did not survive the gather'
        val.sum().backward()
        out[name] = leaf.grad
    assert (out['split'] - out['joined']).abs().max() < 1e-12 * out['joined'].abs().max()
    assert out['split'].abs().max() > 0, 'no gradient reached the forked block at all'


@pytest.mark.parametrize('alpha', [0.0, 0.4])
def test_a_segmented_curve_inside_a_fork_composes(alpha):
    """Scenario routing and tenor segmentation are ORTHOGONAL, so a `Near_Interpolation` curve in
    a fork is a RoutedInterpolation of SegmentedInterpolations and needs no special case — the
    saving is kept rather than given up to a join. Each physical block segments its own tenor axis
    at its own stride, which is exactly why `CurveTensor` hands out scenario ROWS."""
    n_tenors = 8
    past, inner = _blocks(n_tenors=n_tenors, seed=41)
    spec = ((0, 3, 'Hermite'), (3, n_tenors - 1, 'LinearRT'))
    rows = [5, 60, CUTOFF - 1, CUTOFF, CUTOFF + 1]
    curve_tensor, split = _gathered(_source(past, inner), rows, alpha, spec, n_tenors=n_tenors)
    _, joined = _gathered(_join(past, inner), rows, alpha, spec, n_tenors=n_tenors)
    assert isinstance(curve_tensor.interp_obj, RoutedInterpolation)
    assert all(isinstance(s, SegmentedInterpolation) for s in curve_tensor.interp_obj.strategies)
    # the past block's segments stay at the OUTER width — the whole point of not joining
    assert all(seg.shape[-1] == B_OUTER
               for seg in curve_tensor.interp_obj.strategies[0].segments)
    assert curve_tensor.interp_obj.strategies[0].segments[0].hermite_tenor is not None
    assert torch.equal(split, joined)


def test_a_source_carries_only_what_the_pricer_does_to_a_buffer_value():
    """Write-once and read-only. A source is built after every process's `generate` has published,
    and it answers `shape`/`new`/the RT tenor rescale — the operations `make_curve_tensor` performs
    on a raw buffer value. Anything else, including any attempt to write into it, fails loud rather
    than silently materializing the grid it exists to avoid."""
    past, inner = _blocks()
    source = _source(past, inner)
    assert source.shape == (CUTOFF + T_INNER, N_TENORS, B_OUTER * FAN)
    scale = torch.linspace(1.0, 2.0, N_TENORS, dtype=torch.float64).reshape(1, -1, 1)
    scaled = source * scale
    assert torch.equal(scaled.blocks[0].tensor, past * scale)
    assert torch.equal(scaled.blocks[1].tensor, inner * scale)
    for write in (lambda: source.__iadd__(1.0), lambda: source + 1.0, lambda: source.reshape(-1),
                  lambda: source.detach(), lambda: source[0]):
        with pytest.raises((TypeError, AttributeError)):
            write()


def test_projecting_a_block_reproduces_writing_it_out():
    """`ScenarioBlock.project` is the only place a narrow block widens. It must put the same values
    in the same column order the fork's `cat` of an expanded past put them in — that identity is
    what makes the whole change bitwise, and it is now a map carried as DATA rather than a fan
    inferred from two widths."""
    past = _curve(7, N_TENORS, B_OUTER, seed=51)
    written_out = past.unsqueeze(-1).expand(*past.shape, FAN).reshape(*past.shape[:-1], -1)
    block = ScenarioBlock(past, batch_index=_past_columns())
    assert torch.equal(block.project(past), written_out)
    assert torch.equal(ScenarioBlock(past).project(past), past), 'no map = already at width'
