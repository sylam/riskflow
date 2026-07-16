"""Total_Position_Schedule corridor gates: the optional per-decision-step bound on the SIGNED
total position Σq_i. Covers (1) hedge_runtime normalization (sort / monotone-step / Min<=Max
validation; absent => None), (2) HedgeActionSpace.grid_at(t) masking — the base grid filtered to
the corridor at t, piecewise-constant knot lookup, and bit-identical to grid() when no schedule
is configured, and (3) the fail-loud when a corridor filters every action row.

CPU-only and framework-free (no simulation): grid_at is the single per-t filter site every track
(DiffSolverV2 argmax, hindsight, textbook) shares, so masking correctness is provable in isolation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from riskflow.hedge_runtime import _normalize_total_position_schedule as _norm
from riskflow.hedge_solver import HedgeActionSpace


def _aspace(schedule=None, levels=9, lo=-50, hi=0, abs_limit=50.0, active=None):
    """Minimal HedgeActionSpace: 3 hedges on `[lo, hi]`, total |Σq| cap `abs_limit`, `levels`
    per axis. Defaults to the short-only platinum book ([-50, 0]); pass `lo`/`hi`/`abs_limit`
    for long-side or symmetric (book-style) limits."""
    hedges = ["A", "B", "C"]
    runtime = {
        "names": {"hedges": hedges},
        "tradables": {r: {"contract_size": 1.0} for r in hedges},
        "solver": {"training_action_grid_levels_per_axis": levels,
                   "active_hedge_indices": active},
        "accounting": {
            "position_limits": {r: {"min_position": lo, "max_position": hi} for r in hedges},
            "total_position_abs_limit": abs_limit,
            "total_position_schedule": schedule,
        },
    }
    return HedgeActionSpace(runtime, torch.device("cpu"))


# ---- (1) normalization -----------------------------------------------------------------------
def test_schedule_absent_is_none():
    assert _norm({}) is None
    assert _norm({"Total_Position_Schedule": []}) is None


def test_schedule_sorted_and_typed():
    s = _norm({"Total_Position_Schedule": [
        {"Step": 10, "Min_Total": -20, "Max_Total": 0},
        {"Step": 0, "Min_Total": -50, "Max_Total": -30},
        {"Step": 5, "Min_Total": -40, "Max_Total": -10}]})
    assert s == ((0, -50.0, -30.0), (5, -40.0, -10.0), (10, -20.0, 0.0))
    assert all(isinstance(st, int) for st, _, _ in s)
    assert all(isinstance(lo, float) and isinstance(hi, float) for _, lo, hi in s)


def test_schedule_rejects_non_monotone_steps():
    for bad in ([{"Step": 0, "Min_Total": -1, "Max_Total": 0},
                 {"Step": 0, "Min_Total": -1, "Max_Total": 0}],           # duplicate
                [{"Step": 5, "Min_Total": -1, "Max_Total": 0},
                 {"Step": 3, "Min_Total": -1, "Max_Total": 0},
                 {"Step": 5, "Min_Total": -1, "Max_Total": 0}]):          # dup after sort
        try:
            _norm({"Total_Position_Schedule": bad})
            assert False, "non-monotone steps must raise"
        except ValueError as e:
            assert "strictly ascending" in str(e)


def test_schedule_rejects_min_gt_max_and_negative_step():
    try:
        _norm({"Total_Position_Schedule": [{"Step": 0, "Min_Total": 0.0, "Max_Total": -5.0}]})
        assert False
    except ValueError as e:
        assert "Min_Total" in str(e)
    try:
        _norm({"Total_Position_Schedule": [{"Step": -1, "Min_Total": -5.0, "Max_Total": 0.0}]})
        assert False
    except ValueError as e:
        assert ">= 0" in str(e)


# ---- (2) grid_at masking ---------------------------------------------------------------------
def test_grid_at_no_schedule_is_base_grid():
    """No schedule ⇒ grid_at(t) returns the cached base grid itself, unchanged at every t."""
    a = _aspace(None)
    g = a.grid()
    for t in (0, 3, 99):
        assert a.grid_at(t) is g


def test_grid_at_filters_to_signed_corridor():
    sched = _norm({"Total_Position_Schedule": [
        {"Step": 0, "Min_Total": -50, "Max_Total": -30},
        {"Step": 5, "Min_Total": -40, "Max_Total": -10},
        {"Step": 10, "Min_Total": -20, "Max_Total": 0}]})
    a = _aspace(sched)
    base_rows = a.grid().shape[0]
    for t, (lo, hi) in ((0, (-50.0, -30.0)), (5, (-40.0, -10.0)), (10, (-20.0, 0.0))):
        gt = a.grid_at(t)
        tot = gt.sum(-1)
        assert (tot >= lo - 1e-9).all() and (tot <= hi + 1e-9).all()
        assert 0 < gt.shape[0] <= base_rows
    # the tight late-window corridor must admit strictly fewer rows than the wide entry corridor
    assert a.grid_at(10).shape[0] < a.grid_at(0).shape[0]


def test_grid_at_piecewise_constant_between_knots():
    """Between knots the corridor holds the last knot's value; before the first knot it clamps
    to the first knot (t=3 uses knot@0, t=6 uses knot@5)."""
    sched = _norm({"Total_Position_Schedule": [
        {"Step": 0, "Min_Total": -50, "Max_Total": -30},
        {"Step": 5, "Min_Total": -40, "Max_Total": -10}]})
    a = _aspace(sched)
    assert a._corridor_at(0) == (-50.0, -30.0)
    assert a._corridor_at(3) == (-50.0, -30.0)
    assert a._corridor_at(5) == (-40.0, -10.0)
    assert a._corridor_at(6) == (-40.0, -10.0)
    assert a._corridor_at(1000) == (-40.0, -10.0)


# ---- (2b) live-mask: corridor bounds the REALIZED (post-expiry) total ------------------------
def test_grid_at_live_mask_bounds_realized_position():
    """The corridor must bound the REALIZED signed total Σ(q_i·live_i) — the book that survives
    expiry masking (callers apply q*live after the argmax) — not the raw target Σq_i. Passing the
    step-t `live` leg mask (expired leg zeroed) filters on the post-mask total, so a corridor-
    satisfying short PARKED on an expired leg (a dF=0 wealth-neutral tie, then masked to 0 → a
    silent under-hedge) is excluded instead of chosen."""
    sched = _norm({"Total_Position_Schedule": [{"Step": 0, "Min_Total": -50, "Max_Total": -30}]})
    a = _aspace(sched)
    live = torch.tensor([0.0, 1.0, 1.0])                       # leg A expired inside the window
    g_live = a.grid_at(0, live)
    # every surviving row's REALIZED total (live legs only) is inside the corridor
    tot_live = (g_live * live).sum(-1)
    assert (tot_live >= -50 - 1e-9).all() and (tot_live <= -30 + 1e-9).all()
    # rows whose whole short sits on the expired leg (raw Σq in corridor, realized 0) exist under
    # the live-blind filter and are exactly what the old code could park on — they must be dropped
    g_raw = a.grid_at(0)
    parked = g_raw[(g_raw[:, 0] <= -30 - 1e-9) & (g_raw[:, 1:].abs().sum(-1) < 1e-9)]
    assert parked.shape[0] > 0
    assert not bool(((g_live - parked[0]).abs().sum(-1) < 1e-9).any())
    # live all-ones (nothing expired) ⇒ bit-identical to the live-blind corridor filter
    assert torch.equal(a.grid_at(0, torch.ones(3)), a.grid_at(0))


def test_grid_at_live_mask_infeasible_fails_loud():
    """When no LIVE leg can carry the mandated short (here all legs expired ⇒ realized total pinned
    at 0, outside the corridor), fail loud rather than return rows that expiry-masking would gut to
    a corridor-violating 0 — with the 'on live legs' message."""
    sched = _norm({"Total_Position_Schedule": [{"Step": 0, "Min_Total": -50, "Max_Total": -40}]})
    a = _aspace(sched)
    try:
        a.grid_at(0, torch.zeros(3))
        assert False, "corridor unreachable on live legs must raise"
    except ValueError as e:
        assert "action grid empty at step 0" in str(e) and "on live legs" in str(e)


# ---- (2c) BOOK-STYLE generalization: long-side + sign-crossing corridors ---------------------
def test_grid_at_long_only_schedule():
    """Positive-Max limits build LONG action rows and grid_at filters a positive corridor: a
    long-only book ([0, 50] per leg) under a Σq∈[30, 50] corridor returns only long totals."""
    a = _aspace(lo=0, hi=50, abs_limit=0.0, levels=5)          # abs cap off: pure signed corridor
    assert (a.grid() >= -1e-9).all(), "long-only limits must not build short rows"
    sched = _norm({"Total_Position_Schedule": [{"Step": 0, "Min_Total": 30, "Max_Total": 50}]})
    a = _aspace(sched, lo=0, hi=50, abs_limit=0.0, levels=5)
    gt = a.grid_at(0)
    tot = gt.sum(-1)
    assert gt.shape[0] > 0
    assert (tot >= 30 - 1e-9).all() and (tot <= 50 + 1e-9).all()


def test_grid_at_sign_crossing_schedule():
    """A book that is SHORT early and LONG late: symmetric limits [-50, 50] under a schedule that
    flips the corridor sign across t. grid_at returns short rows in the early phase and long rows
    in the late phase — the same base grid, resliced per t."""
    sched = _norm({"Total_Position_Schedule": [
        {"Step": 0, "Min_Total": -50, "Max_Total": -25},       # short early
        {"Step": 10, "Min_Total": 25, "Max_Total": 50}]})      # long late
    a = _aspace(sched, lo=-50, hi=50, abs_limit=0.0, levels=5)
    early = a.grid_at(3).sum(-1)                                # piecewise-constant: knot@0
    late = a.grid_at(12).sum(-1)                                # knot@10
    assert early.shape[0] > 0 and late.shape[0] > 0
    assert (early >= -50 - 1e-9).all() and (early <= -25 + 1e-9).all(), "early phase must be short"
    assert (late >= 25 - 1e-9).all() and (late <= 50 + 1e-9).all(), "late phase must be long"
    # the two phases occupy disjoint signed-total half-lines
    assert float(early.max()) < 0 < float(late.min())


# ---- (2d) project_to_corridor: continuous in-corridor nudge (bank + textbook) ----------------
def test_project_to_corridor_no_schedule_is_identity():
    a = _aspace(None, lo=-50, hi=50)
    q = torch.tensor([[-40.0, 10.0, 5.0], [3.0, 3.0, 3.0]])
    assert torch.equal(a.project_to_corridor(q, 0), q)


def test_project_to_corridor_lands_on_corridor_and_stays_feasible():
    """Projecting an out-of-corridor position lands Σq exactly on the nearest corridor edge while
    every leg stays inside [lo, hi] — for short, long, and sign-crossing corridors."""
    for lo_c, hi_c, q_raw in (
        (-50.0, -30.0, [[-5.0, -5.0, -5.0]]),     # too SHALLOW short (Σ=-15) → push to -30
        (-50.0, -30.0, [[-40.0, -40.0, 0.0]]),    # too DEEP short (Σ=-80) → pull to -50
        (30.0, 50.0, [[5.0, 5.0, 5.0]]),          # too small long (Σ=15) → push to 30
        (30.0, 50.0, [[40.0, 40.0, 40.0]]),       # too big long (Σ=120) → pull to 50
        (-10.0, 10.0, [[-40.0, 5.0, 5.0]]),       # sign-crossing corridor around 0 (Σ=-30) → -10
    ):
        sched = _norm({"Total_Position_Schedule": [
            {"Step": 0, "Min_Total": lo_c, "Max_Total": hi_c}]})
        a = _aspace(sched, lo=-50, hi=50, abs_limit=0.0)
        q = torch.tensor(q_raw)
        p = a.project_to_corridor(q, 0)
        tot = float(p.sum(-1))
        assert lo_c - 1e-4 <= tot <= hi_c + 1e-4, f"Σq {tot} not in [{lo_c}, {hi_c}]"
        assert (p >= -50 - 1e-4).all() and (p <= 50 + 1e-4).all(), f"leg left its box: {p}"


def test_project_to_corridor_in_corridor_is_untouched():
    """An already-in-corridor position is returned unchanged (deficit d=0)."""
    sched = _norm({"Total_Position_Schedule": [{"Step": 0, "Min_Total": -50, "Max_Total": -30}]})
    a = _aspace(sched, lo=-50, hi=0)
    q = torch.tensor([[-15.0, -15.0, -10.0]])                  # Σ=-40 ∈ [-50, -30]
    assert torch.allclose(a.project_to_corridor(q, 0), q)


# ---- (3) infeasible corridor fails loud ------------------------------------------------------
def test_grid_at_infeasible_corridor_fails_loud():
    """A corridor no grid row's Σq can satisfy raises with the empty-grid message pattern."""
    sched = _norm({"Total_Position_Schedule": [{"Step": 0, "Min_Total": -3.0, "Max_Total": -1.0}]})
    a = _aspace(sched, levels=5)                          # Σq lands on multiples of 12.5 — misses [-3,-1]
    try:
        a.grid_at(0)
        assert False, "infeasible corridor must raise"
    except ValueError as e:
        assert "action grid empty at step 0" in str(e)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all Total_Position_Schedule gates passed")
