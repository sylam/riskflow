"""A pricing failure inside an inner-MC fork must be LOUD, not a zero.

`Deal.calculate`'s canonical guard turns any exception into `0.0 * shared.one`, which is right for
"this deal cannot price on this grid" and wrong for everything else: no `Calc_res['tensor']` is
written, so the deal vanishes from `DealStructure.tensor_marks()`, so the fork reports
`F_t1 = 0` for it, so the solver's expired-contract mask retires it from the hedge set. The run
finishes and reports a verdict for a hedge book it silently shrank.

One class is therefore distinguished and re-raised (`utils.is_fatal_pricing_error`): running out
of memory — the failure mode the single-pass fork documents as its contract. Everything else keeps
the skip, which is asserted here too, because base valuation and credit Monte Carlo depend on it.

The end-to-end gates rebuild the deal shapes a fork's curve reads used to be predicted wrong for.
Each asserts the ANSWER, not just the absence of a crash: the lazily built run must equal the one
whose Hermite coefficients cover the whole block.
"""
import copy
import json as jsonlib
import logging
import os

import pytest
import torch

import riskflow as rf
from riskflow import instruments, utils
from riskflow.calculation import HedgeMonteCarlo

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')
TS = lambda s: {'.Timestamp': s}


# --------------------------------------------------------------------------------------------
# The distinguished classes, at the guard itself
# --------------------------------------------------------------------------------------------
class _Boom(instruments.Deal):
    """A deal whose pricing raises whatever it was constructed with."""

    def __init__(self, exc):
        super().__init__({'Reference': 'BOOM'}, {})
        self.exc = exc

    def generate(self, shared, time_grid, deal_data):
        raise self.exc


class _Shared:
    one = torch.zeros(1, 1)
    keep_tensor = False
    gamma = False


def _calculate(exc):
    deal = _Boom(exc)
    return deal.calculate(_Shared(), None, utils.DealDataType(
        Instrument=deal, Factor_dep={}, Time_dep=None, Calc_res=None))


@pytest.mark.parametrize('exc', [
    torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.67 GiB'),
    RuntimeError('CUDA out of memory. Tried to allocate 2.67 GiB'),
    MemoryError('host allocation failed'),
])
def test_running_out_of_memory_is_re_raised(exc):
    """The single-pass fork's documented contract is that a config too wide for the card raises
    CUDA OOM naming the fork. That contract only held for the liability while a tradable's OOM
    became `F_t1 = 0` — a silently smaller hedge set, or a fake one-step move when only the
    heavier grad fork failed."""
    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError, MemoryError),
                       match='out of memory|allocation failed'):
        _calculate(exc)


@pytest.mark.parametrize('exc', [
    ValueError('this deal cannot price on this grid'),
    IndexError('index 7 is out of bounds'),
    KeyError('Discount'),
])
def test_an_ordinary_pricing_failure_still_skips(exc):
    """The canonical skip is load-bearing for base valuation / credit Monte Carlo — a portfolio of
    thousands must not die on one unpriceable deal. Only the two distinguished classes re-raise;
    note a plain IndexError still skips, so the distinction is the type, not the base class."""
    assert torch.equal(_calculate(exc), torch.zeros(1, 1))


def test_build_features_follows_the_same_rule():
    """`build_features` has its own copy of the guard, and a leg swallowed there drops out of the
    liability accumulation, where a second leg's mark broadcasts over the (1,1) gap and the fork's
    shape check cannot see it."""
    deal = _Boom(MemoryError('host allocation failed'))
    data = utils.DealDataType(Instrument=deal, Factor_dep={}, Time_dep=None, Calc_res=None)
    with pytest.raises(MemoryError):
        deal.build_features(_Shared(), None, data)
    deal.exc = ValueError('unpriceable')
    assert torch.equal(deal.build_features(_Shared(), None, data)['mtm'], torch.zeros(1, 1))


# --------------------------------------------------------------------------------------------
# End to end: the deal shapes the derived bound was wrong for
# --------------------------------------------------------------------------------------------
def _energy_leg(name, start, end, pay):
    return {name: {
        'Currency': 'USD', 'Sampling_Type': 'USD', 'FX_Sampling_Type': 'USD',
        'Discount_Rate': 'USD-SOFR', 'Commodity': 'PLATINUM_LME', 'Reference_Type': 'PLATINUM',
        'Payer_Receiver': 'Receiver',
        'Payments': {'Items': [{
            'Payment_Date': TS(pay), 'Period_Start': TS(start), 'Period_End': TS(end),
            'Volume': 2500.0, 'Fixed_Basis': -2045.0, 'Price_Multiplier': 1.0,
            'FX_Period_Start': TS(start), 'FX_Period_End': TS(end),
            'Realized_Average': 0.0, 'FX_Realized_Average': 0.0}]}}}


def _cfg(t_min):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': 16, 'Inner_Sub_Batch': 4,
                 'Inner_MC_Enabled': 'Yes', 'Random_Seed': 1234})
    calc['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
    calc['Hedging_Problem']['Solver'] = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 3,
        'Training_Action_Chunk_Size': 64, 'T_Min': t_min, 'DiffV2_Fit_Iters': 2,
        'DiffV2_OOS_Frac': 0.5, 'DiffV2_One_Step_Fork': 'Yes'}
    return cfg


def _full_block(self, i00, i10):
    """The pre-feature `hermite_params`: build g,c for the WHOLE block on the first gather,
    ignoring the rows it names."""
    if self.rows is None:
        g, c = utils.hermite_interpolation_tensor(self.hermite_tenor, self.tensor)
        self.interp_params = [p.reshape(-1, p.shape[-1]) for p in (g, c)]
        self.rows = (0, self.tensor.shape[0] - 1)
    return self.interp_params[0], self.interp_params[1], 0


def _run(cfg, name, lazy=True, forks=None):
    """One JSON-only run. `lazy=False` forces the pre-feature full-block build so the lazily
    built answer can be compared against it."""
    original = utils.Interpolation.hermite_params
    if not lazy:
        utils.Interpolation.hermite_params = _full_block
    original_fork = HedgeMonteCarlo._run_inner_mc_at_t
    if forks is not None:
        def record(self, t, *a, **kw):
            r = original_fork(self, t, *a, **kw)
            forks.append({k: float(v.detach().abs().max())
                          for k, v in (r.get('F_t1') or {}).items()})
            return r
        HedgeMonteCarlo._run_inner_mc_at_t = record
    try:
        cx = rf.Context()
        cx.load_json((jsonlib.dumps(cfg), f'{name}.json'))
        _, result = cx.run_job()
        return ((result.evaluation_summary or {}).get('diagnostics') or {}).get('V_0')
    finally:
        utils.Interpolation.hermite_params = original
        HedgeMonteCarlo._run_inner_mc_at_t = original_fork


def test_an_averaging_tradable_is_priced_not_retired():
    """An averaging swap hedging an averaging offtake — the obvious hedge for this book — reads
    further back than the liability does. The window was derived from the liability alone, so the
    swap's gather fell below it, its mark was swallowed, and it read downstream as an expired
    contract: the policy trained against a hedge set silently reduced from two instruments to one,
    with a full verdict returned."""
    cfg = _cfg(t_min=113)
    cfg['Calc']['Calculation']['Hedging_Problem']['Tradable_Instruments']['FloatingEnergyDeal'] = \
        _energy_leg('PL_AVG_SWAP', '2026-04-15', '2026-07-31', '2026-08-05')
    forks = []
    built = _run(copy.deepcopy(cfg), 'avg_tradable', forks=forks)
    assert forks, 'no inner-MC forks ran'
    assert all(f['PL_AVG_SWAP'] > 0.0 for f in forks), \
        'the averaging tradable is zero in some fork — it was retired from the hedge set'
    assert built == _run(cfg, 'avg_tradable_full', lazy=False), \
        'the lazily built answer differs from the full-block one'


def test_a_settlement_lagged_liability_solves():
    """Deferred settlement (pay well after the averaging period ends) keeps the cashflow unpaid, so
    `sim_resets` keeps re-reading its fixings. A bound measured as the within-period reset span is
    short by exactly that lag, and the liability could not be solved at all."""
    cfg = _cfg(t_min=150)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Payments']['Items'][0][
        'Payment_Date'] = TS('2026-10-30')
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH'][
        'Investment_Horizon'] = TS('2026-10-30')
    assert _run(copy.deepcopy(cfg), 'lagged') == _run(cfg, 'lagged_full', lazy=False)


def test_a_bullet_sampled_leg_solves():
    """`ForwardPriceSampleBullet` writes ONE fixing per period, at the period END. The `count > 1`
    filter made such a leg declare that it reads no history while its fixing still sat a
    settlement lag below the cutoff."""
    cfg = _cfg(t_min=113)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Payments']['Items'][0][
        'Payment_Date'] = TS('2026-08-20')
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH'][
        'Investment_Horizon'] = TS('2026-08-20')
    cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
        'ForwardPriceSample.USD']['Sampling_Convention'] = 'ForwardPriceSampleBullet'
    assert _run(copy.deepcopy(cfg), 'bullet') == _run(cfg, 'bullet_full', lazy=False)


def test_a_two_leg_liability_book_solves():
    """Two averaging legs reading DIFFERENT depths of history. A single bound over the book had to
    be the min of both, and the second leg's own reads had to already be inside it; the coefficients
    now follow whichever leg gathers first and widen for the other."""
    cfg = _cfg(t_min=113)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal'].update(
        _energy_leg('PLAT_AUG29', '2026-05-01', '2026-08-31', '2026-09-04'))
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH'][
        'Investment_Horizon'] = TS('2026-09-04')
    assert _run(copy.deepcopy(cfg), 'two_leg') == _run(cfg, 'two_leg_full', lazy=False)


def test_a_failed_tradable_inside_a_fork_stops_the_run():
    """The OOM the chunk-loop deletion designated as the expected failure mode, injected into a
    hedge instrument. It must reach the caller instead of becoming `F_t1 = 0` — and because the
    `live` mask comes from the cheaper no-grad fork, a grad-only failure would leave the leg live
    while its one-step move read as -F_t of fake P&L in the training labels."""
    original = instruments.CommodityFutureDeal.generate
    original_fork = HedgeMonteCarlo._run_inner_mc_at_t
    in_fork = {'v': False}

    def fork(self, t, *a, **kw):
        in_fork['v'] = True
        try:
            return original_fork(self, t, *a, **kw)
        finally:
            in_fork['v'] = False

    def generate(self, shared, time_grid, deal_data):
        if in_fork['v'] and self.field.get('Reference') == 'PL_OCT_2026':
            raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.67 GiB')
        return original(self, shared, time_grid, deal_data)

    instruments.CommodityFutureDeal.generate = generate
    HedgeMonteCarlo._run_inner_mc_at_t = fork
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError, match='out of memory'):
            _run(_cfg(t_min=115), 'oom_tradable')
    finally:
        instruments.CommodityFutureDeal.generate = original
        HedgeMonteCarlo._run_inner_mc_at_t = original_fork


def test_a_skipped_tradable_leaves_a_loud_hole_in_the_fork():
    """The tradable half of the fork's degenerate-pricing guard. A non-distinguished failure still
    skips (correctly), but inside a fork the missing mark is indistinguishable from an expired
    contract — so the fork checks that every tradable still live in its dependency list produced
    one, mirroring the liability's shape check."""
    original = instruments.CommodityFutureDeal.generate
    original_fork = HedgeMonteCarlo._run_inner_mc_at_t
    in_fork = {'v': False}

    def fork(self, t, *a, **kw):
        in_fork['v'] = True
        try:
            return original_fork(self, t, *a, **kw)
        finally:
            in_fork['v'] = False

    def generate(self, shared, time_grid, deal_data):
        if in_fork['v'] and self.field.get('Reference') == 'PL_OCT_2026':
            raise ValueError('some ordinary pricing failure')
        return original(self, shared, time_grid, deal_data)

    instruments.CommodityFutureDeal.generate = generate
    HedgeMonteCarlo._run_inner_mc_at_t = fork
    try:
        with pytest.raises(RuntimeError, match="tradable pricing failed for \\['PL_OCT_2026'\\]"):
            _run(_cfg(t_min=115), 'skipped_tradable')
    finally:
        instruments.CommodityFutureDeal.generate = original
        HedgeMonteCarlo._run_inner_mc_at_t = original_fork
