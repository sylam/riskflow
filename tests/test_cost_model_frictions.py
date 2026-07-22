"""Realistic-friction upgrades to the hedging cost model, both behind JSON switches and all
bit-identical when the switch is off (verified here + by the rest of the suite staying green):

  Task 1 — maturity/liquidity- and volatility-dependent bid/offer half-spread. The single
           chokepoint `hedge_runtime.per_contract_kappa` grows a spec form (Per_Instrument +
           Vol_Scale) and a world-agnostic `vol` argument; the scalar path is untouched.

  Task 2 — vol-linked initial-margin FUNDING charge on the post-trade book. Per hedge leg the desk
           posts IM = IM_Vol_Multiplier·(σ_t/IM_Ref_Vol)·F·|q^post|·cs and pays
           IM_Funding_Spread_Bps·1e-4·dt to fund it over the calendar step — a pure realized debit
           into the SAME account-step P&L path as transaction cost (`_im_funding_charge`).
           IM_Funding_Spread_Bps default 0 ⇒ term is exactly 0 ⇒ bit-identical.

  Task 3 — a roll (reduce one contract month, increase an adjacent month) charges the matched
           quantity a single calendar-spread half-cost, not two outright half-spreads. Realized
           accounting only (`hedge_bundle._roll_rebate` credited into the env debit); the
           decision-time argmax keeps the conservative per-instrument outright kappa.

Audit-hardening (protect a GARCH retrain's integrity): the per-step vol SOURCE is logged + pinned
(revealed log_h vs realized proxy), and the diagnostic CSV reconstruction threads the same per-step
vol so its cost reconciles with the realized vol-scaled debit.

Runtimes are built through the public JSON contract (`construct_hedge_runtime`) so the tests
exercise the real Evaluator → accounting normalization.
"""

import contextlib
import copy
import json as jsonlib
import logging
import os
import types

import pytest
import torch


@contextlib.contextmanager
def _capture_root_info():
    """Collect INFO messages off the REAL root logger, forcing it to INFO and undoing any prior
    `logging.disable`/basicConfig-at-WARNING a preceding run_job left behind (which would silently
    drop `logging.info` — and defeat `caplog`). Yields the captured message list."""
    root = logging.getLogger()
    recs = []

    class _H(logging.Handler):
        def emit(self, r):
            recs.append(r.getMessage())

    prev_level, prev_disable = root.level, logging.root.manager.disable
    logging.disable(logging.NOTSET)
    root.addHandler(_H())
    root.setLevel(logging.INFO)
    try:
        yield recs
    finally:
        root.handlers = [h for h in root.handlers if not isinstance(h, _H)]
        root.setLevel(prev_level)
        logging.disable(prev_disable)

import riskflow as rf
from riskflow import utils
from riskflow.hedge_runtime import construct_hedge_runtime, per_contract_kappa
from riskflow.hedge_bundle import (
    BundleStepper, _roll_rebate, _realized_vol_series, _build_step_annual_vol,
    _im_funding_charge, _calendar_dt, _diag_expand_per_day, _daily_growth_factors,
)
from riskflow.stochasticprocess import GARCHSpotModel, StochasticProcess

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')
_LEGS = ('M1', 'M2', 'M3')                       # maturity-ordered synthetic hedge legs
CS = 50.0                                        # contract size (platinum)


def _runtime(evaluator):
    """Minimal simulate_only runtime built through the JSON contract, with a 3-leg futures book
    + one cash account and the given Evaluator overrides."""
    ev = {
        'Cash_Instrument': 'USD',
        'Position_Limits': {n: {'Min_Position': -50, 'Max_Position': 0} for n in _LEGS},
    }
    ev.update(evaluator)
    cfg = {'Execution_Mode': 'simulate_only', 'Hedging_Problem': {
        'Evaluator': ev,
        'Tradable_Instruments': {
            'CommodityFutureDeal': {n: {'Currency': 'USD', 'Contract_Size': CS} for n in _LEGS},
            'CashAccountDeal': {'USD': {'Currency': 'USD'}},
        },
    }}
    return construct_hedge_runtime(cfg)


# =========================== Task 1 — bid/offer spread =====================================

def test_scalar_spread_is_back_compat_bit_identical():
    """A scalar Bid_Offer_Spread_Bps (every current config) keeps spec=None and returns EXACTLY
    the historical formula `0.5·bps·1e-4·price·cs` — passing a vol is inert on this path."""
    rt = _runtime({'Bid_Offer_Spread_Bps': 10.0, 'Transaction_Cost_Per_Unit': 0.0})
    assert rt['accounting']['bid_offer_spread_spec'] is None
    assert rt['accounting']['roll_as_calendar_spread'] is False
    assert rt['accounting']['calendar_spread_bps'] is None
    price = torch.tensor([2050.0, 2100.0])
    expected = 0.5 * 10.0 * 1.0e-4 * price * CS
    assert torch.equal(per_contract_kappa(rt, price, 'M1'), expected)
    # vol is ignored on the scalar path — bit-identical.
    assert torch.equal(per_contract_kappa(rt, price, 'M1', vol=torch.tensor(0.9)), expected)


def test_per_instrument_spread_differentiates_cost_by_leg():
    """Per_Instrument bps price each maturity/liquidity bucket independently; a leg absent from
    the map falls back to Default_Bps."""
    rt = _runtime({'Bid_Offer_Spread_Bps': {
        'Default_Bps': 12.0, 'Per_Instrument': {'M1': 8.0, 'M3': 35.0}}})
    p = torch.tensor([2000.0])
    k1 = float(per_contract_kappa(rt, p, 'M1'))
    k2 = float(per_contract_kappa(rt, p, 'M2'))     # -> Default_Bps 12
    k3 = float(per_contract_kappa(rt, p, 'M3'))
    assert k1 == pytest.approx(0.5 * 8.0 * 1.0e-4 * 2000.0 * CS)
    assert k2 == pytest.approx(0.5 * 12.0 * 1.0e-4 * 2000.0 * CS)
    assert k3 == pytest.approx(0.5 * 35.0 * 1.0e-4 * 2000.0 * CS)
    assert k1 < k2 < k3                              # nearer/liquid cheaper than far/illiquid


def test_vol_scale_unit_at_ref_and_rises_above_ref():
    """`Vol_Scale`: factor is exactly 1 at σ_t==Ref_Vol, rises for σ_t>Ref_Vol (Beta=1 ⇒ linear),
    and vol is inert when Beta==0 / Vol_Scale absent / vol is None."""
    rt = _runtime({'Bid_Offer_Spread_Bps': {
        'Default_Bps': 10.0, 'Per_Instrument': {'M1': 8.0},
        'Vol_Scale': {'Ref_Vol': 0.25, 'Beta': 1.0}}})
    p = torch.tensor([2000.0])
    base = 0.5 * 8.0 * 1.0e-4 * 2000.0 * CS
    # σ_t == Ref_Vol -> factor 1.0
    assert float(per_contract_kappa(rt, p, 'M1', vol=torch.tensor(0.25))) == pytest.approx(base)
    # σ_t == 2·Ref_Vol, Beta=1 -> exactly 2x
    assert float(per_contract_kappa(rt, p, 'M1', vol=torch.tensor(0.50))) == pytest.approx(2 * base)
    # σ_t below ref -> cheaper
    assert float(per_contract_kappa(rt, p, 'M1', vol=torch.tensor(0.10))) < base
    # vol=None -> Vol_Scale ignored (bit-identical to the base bps)
    assert float(per_contract_kappa(rt, p, 'M1', vol=None)) == pytest.approx(base)

    # Beta=0 (or absent) -> vol-independent even when vol is supplied.
    rt0 = _runtime({'Bid_Offer_Spread_Bps': {
        'Default_Bps': 10.0, 'Per_Instrument': {'M1': 8.0},
        'Vol_Scale': {'Ref_Vol': 0.25, 'Beta': 0.0}}})
    assert float(per_contract_kappa(rt0, p, 'M1', vol=torch.tensor(3.0))) == pytest.approx(base)


def test_garch_revealed_annual_vol_and_base_default():
    """GARCH exposes σ_t=√(exp(log h_t)/dt_c) as the revealed vol driver; the base process
    exposes none (None)."""
    param = dict(Omega=0.0004, Alpha=0.05, Beta=0.90, Nu=6.0, H0=0.0004,
                 Mu=0.0, Log_Price=True, Calibration_DT_Years=1.0 / 252.0)
    proc = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(param))
    log_h = torch.log(torch.tensor([[0.0004, 0.0016], [0.0009, 0.0004]]))
    sigma = proc.revealed_annual_vol(log_h)
    assert torch.allclose(sigma, (log_h.exp() / (1.0 / 252.0)).sqrt())
    # √(0.0004·252) ≈ 0.3175 annual
    assert float(sigma[0, 0]) == pytest.approx((0.0004 * 252.0) ** 0.5, rel=1e-5)
    assert StochasticProcess.revealed_annual_vol(proc, log_h) is None


def test_realized_vol_proxy_series_annualizes_mark_path():
    """The trailing realized-vol proxy (fallback when no revealed vol) annualizes a mark path's
    rolling log-return std; a constant per-step move recovers √252·|move|."""
    move = 0.01
    steps = torch.arange(60.0)
    path = (100.0 * torch.exp(move * steps)).unsqueeze(-1).expand(-1, 8)   # (T, B) steady log-drift
    rv = _realized_vol_series(path)
    assert rv.shape == (60,)
    assert float(rv[-1]) == pytest.approx((252.0 ** 0.5) * move, rel=1e-3)


def test_build_step_annual_vol_gated_on_vol_scale():
    """The bundle vol series is built ONLY when a Vol_Scale spec is configured (None otherwise,
    so per_contract_kappa stays vol-independent); the realized-proxy branch fires off the primary
    spot factor when no process reveals a log-variance."""
    factors = {'CommodityPrice.X': (100.0 * torch.exp(0.01 * torch.arange(40.0))).unsqueeze(-1).expand(-1, 4)}
    bundle = {'factors': factors}
    # No Vol_Scale -> None.
    rt_off = _runtime({'Bid_Offer_Spread_Bps': {'Default_Bps': 10.0}})
    rt_off = {**rt_off, 'referenced_commodities': ('CommodityPrice.X',)}
    assert _build_step_annual_vol(bundle, rt_off, {}) is None
    # Vol_Scale + no revealed log_h -> realized proxy off the factor path.
    rt_on = _runtime({'Bid_Offer_Spread_Bps': {
        'Default_Bps': 10.0, 'Vol_Scale': {'Ref_Vol': 0.25, 'Beta': 1.0}}})
    rt_on = {**rt_on, 'referenced_commodities': ('CommodityPrice.X',)}
    series = _build_step_annual_vol(bundle, rt_on, {})
    assert series is not None and series.shape == (40,)
    assert (series > 0).all()


# =========================== Task 2 — vol-linked IM funding ================================

def _im_rt(spread_bps=25.0, mult=0.085, ref_vol=0.25, extra=None):
    ev = {'Bid_Offer_Spread_Bps': 10.0, 'Transaction_Cost_Per_Unit': 0.0,
          'IM_Funding_Spread_Bps': spread_bps, 'IM_Vol_Multiplier': mult, 'IM_Ref_Vol': ref_vol}
    if extra:
        ev.update(extra)
    return _runtime(ev)


def _im_pos(m1=10.0):
    p = {n: torch.tensor([0.0]) for n in _LEGS}
    p['M1'] = torch.tensor([m1])
    return p


def test_im_funding_defaults_are_inert():
    """Absent IM_* keys ⇒ spread 0 (gate off, bit-identical), multiplier 0, ref_vol default 1.0."""
    rt = _runtime({'Bid_Offer_Spread_Bps': 10.0})
    acc = rt['accounting']
    assert acc['im_funding_spread_bps'] == 0.0
    assert acc['im_vol_multiplier'] == 0.0
    assert acc['im_ref_vol'] == 1.0


def test_im_funding_zero_when_spread_zero():
    """IM_Funding_Spread_Bps==0 ⇒ every leg's funding is exactly 0 (no debit)."""
    rt = _im_rt(spread_bps=0.0)
    charges = _im_funding_charge(_im_pos(), _prices(), rt, torch.tensor(0.30), dt=1.0 / 365.25)
    assert all(float(v) == 0.0 for v in charges.values())


def test_im_funding_closed_form_and_linear_scaling():
    """Funding = IM·spread·1e-4·dt with IM = mult·(σ/ref)·F·|q|·cs — linear in spread, σ_t and
    |q^post|, and on GROSS |q| (sign-agnostic)."""
    px, dt = _prices(2000.0), 3.0 / 365.25
    f = lambda rt, vol, pos: float(_im_funding_charge(pos, px, rt, vol, dt)['M1'])
    base = _im_rt(spread_bps=25.0)
    f0 = f(base, torch.tensor(0.30), _im_pos(10.0))
    # explicit closed form
    im = 0.085 * (0.30 / 0.25) * 2000.0 * 10.0 * CS
    assert f0 == pytest.approx(im * 25.0 * 1.0e-4 * dt)
    # 2x spread, 2x vol, 2x |q| each double the funding (all linear couplings)
    assert f(_im_rt(spread_bps=50.0), torch.tensor(0.30), _im_pos(10.0)) == pytest.approx(2 * f0)
    assert f(base, torch.tensor(0.60), _im_pos(10.0)) == pytest.approx(2 * f0)
    assert f(base, torch.tensor(0.30), _im_pos(20.0)) == pytest.approx(2 * f0)
    # gross |q|: a short book funds the same as the mirror long book
    assert f(base, torch.tensor(0.30), _im_pos(-10.0)) == pytest.approx(f0)


def test_im_funding_anchoring_reproduces_flat_initial_margin():
    """Anchoring identity: pick IM_Ref_Vol=σ_0 and IM_Vol_Multiplier so IM at the base (σ_0, F_0)
    equals today's flat per-contract Initial_Margin.Amount. The posted IM level (funding with the
    spread·1e-4·dt wrapper divided out) then reproduces that Amount exactly."""
    F0, sigma0, target_im = 2050.0, 0.28, 8500.0            # PL_APR_2026-style flat per-contract IM
    mult = target_im / (F0 * CS)                            # so mult·(σ0/σ0)·F0·cs·1 == target_im
    rt = _im_rt(spread_bps=30.0, mult=mult, ref_vol=sigma0)
    dt = 5.0 / 365.25
    funding = float(_im_funding_charge(_im_pos(1.0), _prices(F0), rt, torch.tensor(sigma0), dt)['M1'])
    im_level = funding / (30.0 * 1.0e-4 * dt)               # back out the posted IM
    assert im_level == pytest.approx(target_im)


def _im_fixture(spread_bps, mult=0.1, ref_vol=0.25, force_flat='Yes'):
    """Build the real fixture bundle+runtime with IM funding active (so `step_annual_vol` is
    built and the funding term is live)."""
    cfg = jsonlib.load(open(FIXTURE))
    ev = cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']
    ev['IM_Funding_Spread_Bps'] = spread_bps
    ev['IM_Vol_Multiplier'] = mult
    ev['IM_Ref_Vol'] = ref_vol
    ev['Force_Flat_At_End'] = force_flat
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'im_funding.json'))
    _, result = cx.run_job()
    return result.bundle, result.runtime


def test_im_funding_debits_realized_wealth_by_exactly_the_step_charge():
    """At a single (non-terminal) step, IM-funding-ON vs OFF steppers share compounding/VM/txn-cost,
    so the margin-account delta gap is −Σ_i funding_i EXACTLY — the same realized channel
    transaction cost debits through, and `_im_funding_charge` reproduces it to the cent."""
    bundle, runtime = _im_fixture(spread_bps=250.0)
    assert bundle['step_annual_vol'] is not None
    hedges = list(runtime['names']['hedges'])
    acct = runtime['accounting']['instrument_to_cash_account'][hedges[0]]
    days, last = bundle['time_grid_days_cpu'], len(bundle['time_grid_days_cpu']) - 1

    rt_off = copy.deepcopy(runtime)
    rt_off['accounting']['im_funding_spread_bps'] = 0.0    # in-code switch OFF (JSON is the contract)
    sA, sB = BundleStepper(bundle, runtime), BundleStepper(bundle, rt_off)
    checked = False
    while not sA.done:
        if sA.is_decision_step and not checked:
            t = sA.time_index
            prices = {h: sA.observe()['tradable_values'][h].clone() for h in hedges}
            act = {hedges[0]: -25.0, hedges[1]: -20.0}
            before_a = sA._state['margin_accounts'][acct].clone()
            before_b = sB._state['margin_accounts'][acct].clone()
            sA.step(act)
            sB.step(act)
            pos_post = {h: sA._state['positions'][h].clone() for h in hedges}
            vol_t = bundle['step_annual_vol'][t]
            dt = (days[min(t + 1, last)] - days[t]) / 365.25
            expected = sum(_im_funding_charge(pos_post, prices, runtime, vol_t, dt).values())
            gap = ((sA._state['margin_accounts'][acct] - before_a)
                   - (sB._state['margin_accounts'][acct] - before_b))
            assert (expected > 0).all()
            assert torch.allclose(gap, -expected, atol=1e-2)   # A poorer by exactly the funding
            checked = True
        else:
            sA.step(None)
            sB.step(None)
    assert checked


def test_im_funding_reduces_terminal_wealth_by_compounded_sum_funding():
    """Full rollout: IM funding is a pure daily debit into the margin ledger, so ON-vs-OFF terminal
    P&L differs by exactly the per-step funding compounded forward at the SAME risk-free growth the
    ledger carries (Force_Flat='No' so post-step positions are the held book). Reconciled to the
    cent via the env's own growth factors."""
    bundle, runtime = _im_fixture(spread_bps=250.0, force_flat='No')
    hedges = list(runtime['names']['hedges'])
    acct = runtime['accounting']['instrument_to_cash_account'][hedges[0]]
    days, last = bundle['time_grid_days_cpu'], len(bundle['time_grid_days_cpu']) - 1

    rt_off = copy.deepcopy(runtime)
    rt_off['accounting']['im_funding_spread_bps'] = 0.0
    sA, sB = BundleStepper(bundle, runtime), BundleStepper(bundle, rt_off)
    acc = torch.zeros(sA._batch_size, device=sA._device)   # A's margin funding contribution, cmpd fwd
    traded = False
    while not sA.done:
        t = sA.time_index
        prices = {h: sA.observe()['tradable_values'][h].clone() for h in hedges}
        act = {hedges[0]: -25.0, hedges[1]: -20.0} if (sA.is_decision_step and not traded) else None
        traded = traded or act is not None
        sA.step(act)
        sB.step(act)
        pos_post = {h: sA._state['positions'][h].clone() for h in hedges}
        vol_t = bundle['step_annual_vol'][t]
        dt = (days[min(t + 1, last)] - days[t]) / 365.25
        funding_t = sum(_im_funding_charge(pos_post, prices, runtime, vol_t, dt).values())
        # env compounds the whole margin (incl. prior funding) at step start, THEN debits funding_t.
        g = _daily_growth_factors(bundle, runtime, t, min(t + 1, last)).get(acct)
        acc = acc * (g if g is not None else 1.0) - funding_t
    pnl_a = sA._terminal_transition['pnl_excess']
    pnl_b = sB._terminal_transition['pnl_excess']
    assert (pnl_b - pnl_a > 0).all()                    # funding strictly reduces terminal wealth
    assert torch.allclose(pnl_b - pnl_a, -acc, rtol=1e-4, atol=2.0)


# =========================== Task 3 — roll as calendar spread ==============================

def _roll_rt(cal_bps=None, spread=10.0):
    ev = {'Bid_Offer_Spread_Bps': spread, 'Transaction_Cost_Per_Unit': 0.0,
          'Roll_As_Calendar_Spread': 'Yes'}
    if cal_bps is not None:
        ev['Calendar_Spread_Bps'] = cal_bps
    return _runtime(ev)


def _prices(v=2000.0):
    return {n: torch.tensor([v]) for n in _LEGS}


def test_pure_roll_pays_calendar_not_two_outrights():
    """A pure roll (−x on M1, +x on M2) has all x matched; the default calendar rate is half the
    sum of the two outrights, so the matched turnover pays HALF what two independent outrights
    would (rebate = 0.5·x·(k1+k2))."""
    rt = _roll_rt()
    k = float(per_contract_kappa(rt, torch.tensor([2000.0]), 'M1'))
    d = {'M1': torch.tensor([-5.0]), 'M2': torch.tensor([5.0]), 'M3': torch.tensor([0.0])}
    rebate = _roll_rebate(d, _prices(), rt)
    assert float(rebate) == pytest.approx(0.5 * 5 * (k + k))     # realized 5k vs outright 10k


def test_mixed_trade_splits_matched_and_residual():
    """−8 on M1, +5 on M2: only the matched 5 rolls (rebated); the residual 3 on M1 keeps paying
    the per-instrument outright (untouched by the rebate)."""
    rt = _roll_rt()
    k = float(per_contract_kappa(rt, torch.tensor([2000.0]), 'M1'))
    d = {'M1': torch.tensor([-8.0]), 'M2': torch.tensor([5.0]), 'M3': torch.tensor([0.0])}
    assert float(_roll_rebate(d, _prices(), rt)) == pytest.approx(0.5 * 5 * (k + k))


def test_chained_rolls_across_three_legs():
    """+5 M1, −8 M2, +3 M3 rolls 5 (M1↔M2) then the M2 residual 3 (M2↔M3): 8 contracts matched."""
    rt = _roll_rt()
    k = float(per_contract_kappa(rt, torch.tensor([2000.0]), 'M1'))
    d = {'M1': torch.tensor([5.0]), 'M2': torch.tensor([-8.0]), 'M3': torch.tensor([3.0])}
    assert float(_roll_rebate(d, _prices(), rt)) == pytest.approx(0.5 * (5 + 3) * (k + k))


def test_non_roll_trades_get_zero_rebate():
    """Same-sign adjacent deltas are NOT a roll (no offsetting), and an isolated single-leg trade
    has nothing to match — zero rebate either way (0 effect on non-roll turnover)."""
    rt = _roll_rt()
    same_sign = {'M1': torch.tensor([-5.0]), 'M2': torch.tensor([-3.0]), 'M3': torch.tensor([0.0])}
    assert float(_roll_rebate(same_sign, _prices(), rt)) == 0.0
    single = {'M1': torch.tensor([-7.0]), 'M2': torch.tensor([0.0]), 'M3': torch.tensor([0.0])}
    assert float(_roll_rebate(single, _prices(), rt)) == 0.0


def test_explicit_calendar_spread_bps_override():
    """`Calendar_Spread_Bps` overrides the matched leg: a single half-spread at cal_bps on the
    average leg notional (+ flat fee on both legs), rebating outright − calendar."""
    rt = _roll_rt(cal_bps=5.0)
    k = float(per_contract_kappa(rt, torch.tensor([2000.0]), 'M1'))
    d = {'M1': torch.tensor([-5.0]), 'M2': torch.tensor([5.0]), 'M3': torch.tensor([0.0])}
    cal = 5 * (0.5 * 5.0 * 1.0e-4 * 2000.0 * CS)      # matched 5 @ 5bps half-spread, equal notionals
    assert float(_roll_rebate(d, _prices(), rt)) == pytest.approx(5 * (k + k) - cal)


def test_roll_rebate_flows_through_realized_futures_debit():
    """End-to-end: the same crafted roll, applied under switch-ON vs switch-OFF runtimes on the
    real fixture bundle, leaves positions/VM/outright-cost identical — so the switch-ON margin
    account is richer by EXACTLY the calendar rebate (the rebate credited into the env debit)."""
    cfg = jsonlib.load(open(FIXTURE))
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'cost_frictions.json'))
    _, result = cx.run_job()
    bundle, runtime = result.bundle, result.runtime
    hedges = list(runtime['names']['hedges'])                 # maturity-ordered real legs
    acct = runtime['accounting']['instrument_to_cash_account'][hedges[0]]

    rt_on = copy.deepcopy(runtime)
    rt_on['accounting']['roll_as_calendar_spread'] = True     # in-code switch (JSON is the contract)
    sA = BundleStepper(bundle, runtime)                        # switch OFF (fixture default)
    sB = BundleStepper(bundle, rt_on)                          # switch ON

    q = 5.0
    roll = {hedges[0]: q, hedges[1]: -q}                       # offsetting adjacent-maturity roll
    checked = False
    while not sA.done:
        if sA.is_decision_step and not checked:
            prices = {h: sA.observe()['tradable_values'][h].clone() for h in hedges}
            deltas = {h: torch.full_like(prices[h], float(roll.get(h, 0.0))) for h in hedges}
            expected = _roll_rebate(deltas, prices, rt_on, None)
            before_a = sA._state['margin_accounts'][acct].clone()
            before_b = sB._state['margin_accounts'][acct].clone()
            sA.step(roll)
            sB.step(roll)
            # Compounding + VM + per-instrument cost are identical between A and B; the ONLY
            # difference at this step is B's calendar rebate credit.
            delta_a = sA._state['margin_accounts'][acct] - before_a
            delta_b = sB._state['margin_accounts'][acct] - before_b
            assert (expected > 0).all()
            assert torch.allclose(delta_b - delta_a, expected, atol=1e-2)
            checked = True
        else:
            sA.step(None)
            sB.step(None)
    assert checked


# =============== Audit-hardening — vol source is observable + diagnostic reconciles ===========

def _garch(param=None):
    p = dict(Omega=8.028e-07, Alpha=0.0328, Beta=0.9639, Nu=7.50,
             Mu=0.0, H0=7.671e-04, Log_Price=True, Calibration_DT_Years=1.0 / 252.0)
    if param:
        p.update(param)
    proc = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=p)
    proc.factor_key = utils.Factor('CommodityPrice', ('PLAT',))
    return proc


def test_build_step_annual_vol_uses_revealed_garch_log_h_and_logs_it():
    """A GARCH world publishes a revealed log h_t: `_build_step_annual_vol` MUST take the revealed
    branch (σ_t=√(exp(log h_t)/dt_c)), NOT the realized-vol proxy — a silent proxy fallback under a
    GARCH retrain would invalidate the whole vol coupling. Pinned: the series equals the revealed
    conditional vol, differs from the realized proxy, is index-aligned with the price grid, and the
    chosen source is logged."""
    T, B = 12, 5
    proc = _garch()
    factor = proc.factor_key
    log_var = torch.linspace(5.0e-4, 1.5e-3, T).view(T, 1, 1).expand(T, B, 1).contiguous()
    log_h = log_var.log()                                       # revealed log-variance surface (T,B,1)
    spot = (100.0 * torch.exp(0.02 * torch.arange(T, dtype=torch.float32))).unsqueeze(-1).expand(T, B)
    bundle = {'privileged_factors': {'log_h': log_h},
              'factors': {'CommodityPrice.PLAT': spot.contiguous()}}
    rt = _runtime({'Bid_Offer_Spread_Bps': {
        'Default_Bps': 10.0, 'Vol_Scale': {'Ref_Vol': 0.25, 'Beta': 1.0}}})
    rt = {**rt, 'referenced_commodities': ('CommodityPrice.PLAT',)}

    with _capture_root_info() as recs:
        series = _build_step_annual_vol(bundle, rt, {factor: proc})

    expected = proc.revealed_annual_vol(log_h[..., 0]).mean(dim=-1)         # (T,) revealed branch
    assert series.shape == (T,)                                             # index-aligned w/ grid
    assert series.shape[0] == spot.shape[0]
    assert torch.allclose(series, expected)
    # It is NOT the realized-vol proxy off the spot factor (proves the revealed branch fired).
    assert not torch.allclose(series, _realized_vol_series(spot))
    msg = ' '.join(recs)
    assert 'step_annual_vol source: revealed' in msg and 'log_h' in msg and 'PLAT' in msg


def test_build_step_annual_vol_is_built_for_im_funding_without_vol_scale():
    """IM funding alone (no Vol_Scale spread spec) still needs σ_t, so the series is built and the
    source logged — the realized-vol proxy here, off the primary spot factor."""
    factors = {'CommodityPrice.X': (100.0 * torch.exp(0.01 * torch.arange(30.0))).unsqueeze(-1).expand(-1, 4)}
    bundle = {'factors': factors}
    rt = _im_rt(spread_bps=40.0)                                # scalar spread ⇒ spec None, no Vol_Scale
    rt = {**rt, 'referenced_commodities': ('CommodityPrice.X',)}
    assert rt['accounting']['bid_offer_spread_spec'] is None
    with _capture_root_info() as recs:
        series = _build_step_annual_vol(bundle, rt, {})
    assert series is not None and series.shape == (30,)
    assert 'realized-vol proxy on CommodityPrice.X' in ' '.join(recs)


def _vol_scale_fixture(default_bps=12.0, ref_vol=0.25, beta=1.0, m1_bps=8.0):
    """Real fixture bundle+runtime with a vol-scaled bid/offer spread active (so `step_annual_vol`
    is built and per_contract_kappa scales with σ_t)."""
    cfg = jsonlib.load(open(FIXTURE))
    ev = cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']
    hedges0 = list(ev['Position_Limits'].keys())
    ev['Bid_Offer_Spread_Bps'] = {'Default_Bps': default_bps,
                                  'Per_Instrument': {hedges0[0]: m1_bps},
                                  'Vol_Scale': {'Ref_Vol': ref_vol, 'Beta': beta}}
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'vol_scale.json'))
    _, result = cx.run_job()
    return result.bundle, result.runtime


def _rollout_of(stepper):
    """The recorded-trajectory dict `_diag_expand_per_day` consumes (mirrors
    BundleStepper.write_diagnostic_csvs' assembly)."""
    order = stepper._instrument_order
    tt = stepper._terminal_transition
    return {
        'times': stepper._times,
        'position': {n: torch.stack(stepper._position_history[n], dim=0) for n in order},
        'trade': {n: torch.stack(stepper._trade_history[n], dim=0) for n in order},
        'price': {n: torch.stack(stepper._price_history[n], dim=0) for n in order},
        'pnl_excess': tt['pnl_excess'].detach().cpu(),
        'liability': tt['liability_value'].detach().cpu(),
        'net_pnl': (tt['pnl_excess'] + tt['liability_value']).detach().cpu(),
    }


def test_diagnostic_reconstructed_cost_reconciles_with_realized_vol_scaled_debit():
    """Under an active Vol_Scale, the diagnostic CSV reconstruction (`_diag_expand_per_day`) must
    charge the SAME vol-scaled kappa as the realized env debit. At a decision step: (a) the
    per-instrument reconstructed `trade_cost` equals the vol-scaled per-step formula and DIFFERS
    from the old vol=None reconstruction; (b) their sum equals the realized margin cost isolated by
    a zero-spread twin stepper."""
    bundle, runtime = _vol_scale_fixture()
    assert bundle['step_annual_vol'] is not None
    hedges = list(runtime['names']['hedges'])
    acct = runtime['accounting']['instrument_to_cash_account'][hedges[0]]

    rt_zero = copy.deepcopy(runtime)                       # zero out the spread -> zero realized cost
    rt_zero['accounting']['bid_offer_spread_spec']['default_bps'] = 0.0
    rt_zero['accounting']['bid_offer_spread_spec']['per_instrument'] = {}
    sA, sB = BundleStepper(bundle, runtime), BundleStepper(bundle, rt_zero)

    # Reconcile at a WARMED-UP decision step (σ_t > 0.1): the realized-vol proxy is at its floor over
    # the constant history prefix, so an early step would give a ~0 cost swamped by float noise; a
    # warmed-up step gives a meaningful vol-scaled cost that reconciles cleanly.
    trade = {hedges[0]: -6.0, hedges[1]: -4.0}
    t0 = None
    realized_cost = None
    while not sA.done:
        warm = sA.is_decision_step and float(bundle['step_annual_vol'][sA.time_index]) > 0.1
        if warm and t0 is None:
            t0 = sA.time_index
            before_a = sA._state['margin_accounts'][acct].clone()
            before_b = sB._state['margin_accounts'][acct].clone()
            sA.step(trade)
            sB.step(trade)
            # VM + compounding identical; the only gap is A's vol-scaled trade cost (B has none).
            realized_cost = ((sB._state['margin_accounts'][acct] - before_b)
                             - (sA._state['margin_accounts'][acct] - before_a)).cpu()   # = cost_A
        else:
            sA.step(None)
            sB.step(None)
    assert t0 is not None
    fields = _diag_expand_per_day(_rollout_of(sA), bundle, runtime)     # per_instr fields are on CPU

    vol_t = bundle['step_annual_vol'][t0].detach().cpu()
    recon_sum = torch.zeros_like(realized_cost)
    for h in hedges:
        p = fields['per_instr'][h]
        price_t0, trd_t0, recon = p['fut'][t0], p['trd'][t0], p['trade_cost'][t0]
        # (a) diagnostic threads the per-step vol -> equals the vol-scaled formula
        assert torch.allclose(recon, trd_t0.abs() * per_contract_kappa(runtime, price_t0, h, vol_t))
        # ... and DIFFERS from the pre-fix vol=None reconstruction for the traded legs
        if float(trd_t0.abs().max()) > 0:
            none_cost = trd_t0.abs() * per_contract_kappa(runtime, price_t0, h, None)
            assert not torch.allclose(recon, none_cost)
        recon_sum = recon_sum + recon.to(realized_cost.dtype)
    # (b) reconstructed cost == realized vol-scaled debit (margin-diff float precision -> atol 0.1)
    assert (realized_cost > 0).all()
    assert torch.allclose(recon_sum, realized_cost, atol=0.1)
