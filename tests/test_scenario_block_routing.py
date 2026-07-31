"""A scenario source is a SEQUENCE of row blocks, and a bare tensor is the ordinary case.

An inner-MC fork publishes two blocks: the outer-realized past at `Batch_Size`, then the forked
rows at `Batch_Size x Inner_Sub_Batch`. Every past row is identical across the inner draws, so
joining them into one tensor writes the past out `Inner_Sub_Batch` times. `RoutedInterpolation`
reads through the blocks instead, so these gates assert the two-block answer against the JOINED
grid — the tensor the fork used to build — and assert that an ordinary grid never meets any of it.

Scenario routing and tenor segmentation are orthogonal: rows route by physical block, tenor points
route by interpolation segment, and a `Near_Interpolation` curve inside a fork composes the two.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from riskflow import utils
from riskflow.utils import (CurveTensor, Factor, Interpolation, RoutedInterpolation,
                            ScenarioBlock, ScenarioSource, SegmentedInterpolation,
                            build_interpolation, hermite_interpolation_tensor, make_curve_tensor)

SCEN, N_TENORS, BATCH = 119, 31, 6


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
    assert whole.interp_params == [], 'a Linear interpolation has no parameters'
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
    """The second half of the prize, and what made deferring the build unnecessary: a past row's
    coefficients are identical across the forked draws, so a block builds them at ITS width — the
    outer one — not once per flat sample."""
    past, inner = _blocks()
    curve_tensor, _ = _gathered(_source(past, inner), [10, 20, 30], 0.0, 'Hermite')
    past_leaf, forked_leaf = curve_tensor.interp_obj.strategies
    assert past_leaf.interp_params[0].shape[-1] == B_OUTER, 'past coefficients are flat-width'
    assert forked_leaf.interp_params[0].shape[-1] == B_OUTER * FAN
    t = past.new(_tenor_np(N_TENORS)).reshape(1, -1, 1)
    g_full, c_full = hermite_interpolation_tensor(t, past)
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
    grad forks themselves are bitwise-gated end to end by `gates/golden_worlds.py`."""
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
    past_segments = curve_tensor.interp_obj.strategies[0].segments
    assert all(seg.shape[-1] == B_OUTER for seg in past_segments)
    # and each carries its OWN tenor stride, which is why rows stay unflattened until here
    assert [seg.shape[1] for seg in past_segments] == [4, n_tenors - 3]
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
