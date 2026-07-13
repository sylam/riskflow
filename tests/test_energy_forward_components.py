"""FloatingEnergyDeal ForwardCurve valuation option — t0 liability mark in both curve modes,
reconciled against hand-computed values straight from the fixture's market data.

  Components: F(t,T) = S(t) exp(c(T)(T-t) + r(t,T)(T-t)) — spot x carry x repo, priced via
              utils.DerivedForwardCurve (factor_dep['ForwardPrice'] is None switches the path).
  Direct:     F(t,T) read off the Clewlow-Strickland ForwardPrice curve (legacy path).

JSON-is-the-contract: load_json + run_job, no internal imports / monkey-patching.
"""
import json as jsonlib
import os

import numpy as np
import pandas as pd
import pytest

import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')

EXCEL = pd.Timestamp('1899-12-30')


def _t0_mark(forward_curve):
    cfg = jsonlib.load(open(FIXTURE))
    cfg['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Valuation Configuration']['FloatingEnergyDeal']['ForwardCurve'] = forward_curve
    if forward_curve == 'Direct':
        # The canonical fixture unlinks the CS curve (ForwardPrice: null) so the Components
        # world doesn't simulate it; Direct mode needs the link back (in-code override only).
        cfg['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factors']['ReferencePrice.PLATINUM']['ForwardPrice'] = 'PLATINUM'
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'simulate_only'
    calc['Batch_Size'] = 16
    calc['Random_Seed'] = 1234
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), f'components_{forward_curve}.json'))
    _, out = cx.run_job()
    t0 = out.bundle['liability_mtm'][0]
    assert t0.std().item() == 0.0, 't0 mark must be deterministic across paths'
    return t0.mean().item(), cfg


def _hand_mark(cfg, forward_curve):
    pf = cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    deal = cfg['Calc']['Calculation']['Hedging_Problem']['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']
    cf = deal['Payments']['Items'][0]
    base = pd.Timestamp(cfg['Calc']['Calculation']['Base_Date']['.Timestamp'])
    sofr = np.array(pf['InterestRate.USD-SOFR']['Curve']['.Curve']['data'])

    fixings = pd.bdate_range(cf['Period_Start']['.Timestamp'], cf['Period_End']['.Timestamp'])
    fix_excel = np.array([(d - EXCEL).days for d in fixings], dtype=float)

    if forward_curve == 'Components':
        carry = np.array(pf['ForwardRate.PLATINUM_CARRY']['Curve']['.Curve']['data'])
        tau_d = fix_excel - (base - EXCEL).days
        c = np.interp(fix_excel, carry[:, 0], carry[:, 1])
        r = np.interp(tau_d / 365.0, sofr[:, 0], sofr[:, 1])
        F = pf['CommodityPrice.PLATINUM_LME']['Spot'] * np.exp(c * tau_d / 365.25 + r * tau_d / 365.0)
    else:
        cs = np.array(pf['ForwardPrice.PLATINUM']['Curve']['.Curve']['data'])
        F = np.interp(fix_excel, cs[:, 0], cs[:, 1])

    pay_tau = (pd.Timestamp(cf['Payment_Date']['.Timestamp']) - base).days / 365.0
    disc = np.exp(-np.interp(pay_tau, sofr[:, 0], sofr[:, 1]) * pay_tau)
    return disc * cf['Volume'] * (F.mean() + cf['Fixed_Basis'])


@pytest.mark.parametrize('forward_curve', ['Components', 'Direct'])
def test_t0_mark_matches_hand_computation(forward_curve):
    mark, cfg = _t0_mark(forward_curve)
    expected = _hand_mark(cfg, forward_curve)
    # float32 sim vs float64 hand-calc; Components also carries interp-detail rounding
    assert mark == pytest.approx(expected, rel=2e-4), \
        f'{forward_curve}: model {mark:,.2f} vs hand {expected:,.2f}'
