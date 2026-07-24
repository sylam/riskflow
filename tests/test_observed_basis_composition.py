"""Multi-element commodity spot: a composed spot (primary + ObservedBasis) is declared by the
EXPLICIT deal fields Commodity + Implied_Basis, resolved by the get_* layer into a multi-element
CODE, and summed by utils.calc_time_grid_spot_rate. No name parsing — composition lives in the
fields, names stay atomic.

Two layers of coverage:
  * unit — calc_time_grid_spot_rate on a hand-built code/buffer: single-element bit-path,
    multi-element sum (stoch + static), cache-hit reuse.
  * deal — FloatingEnergyDeal (Components) with an explicit Implied_Basis produces the same
    priced liability as the BasisComposedSpotModel composed factor (bit-identical), and an
    Observed_Factor/Commodity mismatch raises loudly.
"""
import json
import os
import types

import numpy as np
import pytest
import torch

import riskflow as rf
from riskflow import utils

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPING = os.path.join(REPO, 'artifacts', 'platinum_hedge_shipping.json')

# --- unit: calc_time_grid_spot_rate --------------------------------------------------------

def _shared():
    return types.SimpleNamespace(t_Buffer={}, t_Scenario_Buffer={}, t_Static_Buffer={})


def _grid(prior_indices):
    # (T_mtm, 3): col0 = PriorScenarioDelta (0 => alpha None, pure selection),
    # col1 = MTM day, col2 = ScenarioPriorIndex.
    return np.array([[0.0, float(10 * i), idx] for i, idx in enumerate(prior_indices)], dtype=np.float64)


def test_spot_rate_single_element_bit_path():
    torch.manual_seed(0)
    buf = torch.randn(5, 4)
    shared = _shared()
    shared.t_Scenario_Buffer['P'] = buf
    tg = _grid([0, 2, 4])
    out = utils.calc_time_grid_spot_rate([(True, 'P')], tg, shared)
    # single element == plain gather of the prior-index rows (the legacy behaviour, exact)
    assert torch.equal(out, buf[torch.tensor([0, 2, 4])])


def test_spot_rate_multi_element_sum_stoch():
    torch.manual_seed(1)
    prim, basis = torch.randn(5, 4), torch.randn(5, 4)
    shared = _shared()
    shared.t_Scenario_Buffer.update({'PRIM': prim, 'BAS': basis})
    tg = _grid([0, 2, 4])
    out = utils.calc_time_grid_spot_rate([(True, 'PRIM'), (True, 'BAS')], tg, shared)
    idx = torch.tensor([0, 2, 4])
    # element 0 is the primary; the basis is added onto it -> composed = primary + basis.
    assert torch.equal(out, prim[idx] + basis[idx])


def test_spot_rate_multi_element_sum_static_and_mixed():
    torch.manual_seed(2)
    stoch = torch.randn(5, 4)
    stat = torch.randn(4)                       # static spot broadcast row
    shared = _shared()
    shared.t_Scenario_Buffer['S'] = stoch
    shared.t_Static_Buffer['C'] = stat
    tg = _grid([1, 3])
    idx = torch.tensor([1, 3])
    # static only
    out_static = utils.calc_time_grid_spot_rate([(False, 'C')], tg, _static_shared(stat))
    assert torch.equal(out_static, stat.reshape(1, -1))
    # mixed static + stoch: (1,B) + (T,B) broadcast
    out_mixed = utils.calc_time_grid_spot_rate([(False, 'C'), (True, 'S')], tg, shared)
    assert torch.equal(out_mixed, stat.reshape(1, -1) + stoch[idx])


def _static_shared(stat):
    s = _shared()
    s.t_Static_Buffer['C'] = stat
    return s


def test_spot_rate_cache_hit_reuse():
    torch.manual_seed(3)
    prim, basis = torch.randn(5, 4), torch.randn(5, 4)
    shared = _shared()
    shared.t_Scenario_Buffer.update({'PRIM': prim, 'BAS': basis})
    tg = _grid([0, 1, 2])
    code = [(True, 'PRIM'), (True, 'BAS')]
    first = utils.calc_time_grid_spot_rate(code, tg, shared)
    second = utils.calc_time_grid_spot_rate(code, tg, shared)
    assert first is second                                       # served from t_Buffer
    assert len([k for k in shared.t_Buffer if k[0] == 'spot']) == 1
    # a different code (single element) is a distinct cache entry
    _ = utils.calc_time_grid_spot_rate([(True, 'PRIM')], tg, shared)
    assert len([k for k in shared.t_Buffer if k[0] == 'spot']) == 2


# --- deal-level: explicit Implied_Basis on FloatingEnergyDeal ------------------------------

def _ship_cfg(mutate=None):
    cfg = json.load(open(SHIPPING))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'simulate_only'
    calc['Batch_Size'] = 64
    calc['Simulation_Batches'] = 1
    calc['Random_Seed'] = 1
    calc['Hedging_Problem'].pop('Solver', None)
    if mutate:
        mutate(calc['Hedging_Problem']['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29'])
    return cfg


def _liability_mtm(cfg):
    cx = rf.Context()
    cx.load_json((json.dumps(cfg, default=str), 'basis.json'))
    _, out = cx.run_job()
    return out.bundle['liability_mtm']


def test_explicit_basis_matches_composed_factor():
    # Composed: deal references PLATINUM_LME (BasisComposedSpotModel = PLATINUM_CME + LME_CME).
    composed = _liability_mtm(_ship_cfg())

    # Explicit: deal references the primary PLATINUM_CME plus Implied_Basis=LME_CME. The multi-
    # element deal code sums the SAME two buffers; the composed factor is now unreferenced.
    def _explicit(deal):
        deal['Commodity'] = 'PLATINUM_CME'
        deal['Implied_Basis'] = 'LME_CME'
    explicit = _liability_mtm(_ship_cfg(_explicit))

    assert composed.shape == explicit.shape
    assert torch.equal(composed, explicit), \
        f'max|diff|={ (composed - explicit).abs().max().item() }'


def test_observed_factor_mismatch_raises(caplog):
    # Deal Commodity=PLATINUM_LME but Implied_Basis=LME_CME observes PLATINUM_CME — incoherent.
    # The ValueError fires in calc_dependencies (loud, names both); the framework's dependency
    # walker logs it and skips the deal (its standard bad-config contract), so the calc then
    # fails downstream on the missing liability — assert the validation message was emitted.
    def _mismatch(deal):
        deal['Commodity'] = 'PLATINUM_LME'
        deal['Implied_Basis'] = 'LME_CME'
    with caplog.at_level('ERROR'):
        with pytest.raises(Exception):
            _liability_mtm(_ship_cfg(_mismatch))
    joined = ' '.join(r.getMessage() for r in caplog.records)
    assert 'FloatingEnergyDeal' in joined and 'PLATINUM_LME' in joined \
        and 'PLATINUM_CME' in joined and 'LME_CME' in joined, joined
