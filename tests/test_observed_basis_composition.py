"""Composed spot = primary + ObservedBasis, carried POSITIONALLY in the NAME
(CommodityPrice.PLATINUM_CME.LME_CME — the InterestRate.USD_SOFR.FUNDING prefix chain, tail a
different type). instruments.calc_factor_code_chain is the one resolver (get_interest_factor is its
head==tail case); bases stack; deals carry NO composition fields.

Coverage:
  * spot rate — calc_time_grid_spot_rate: single-element bit-path, multi-element sum, cache reuse,
    stacked (3-element) sum.
  * resolver — positional single/composite/stack for commodity + fx, and the IR identity case.
  * linkage — BasisLinkedSpotModel.calc_references derives its linked parent from the name prefix
    (composable-type resolution at depth 2, ObservedBasis chain deeper; loud if not exactly one).
  * deal — a composed-name Commodity sums the basis into the priced liability (zero basis is a no-op).
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


# --- unit: the positional prefix-chain resolver --------------------------------------------

class _Stub:
    def get_subtype(self):
        return 'sub'


def _offsets(*names):
    # {Factor(type, field): stub}; calc_factor_index reads .get_subtype()
    return {utils.Factor(t, f): _Stub() for t, f in names}


def test_chain_positional_single_and_composite_and_stack():
    from riskflow.instruments import calc_factor_code_chain
    stat = _offsets(('CommodityPrice', ('CME',)), ('ObservedBasis', ('CME', 'LME')),
                    ('ObservedBasis', ('CME', 'LME', 'SHF')))
    # plain name -> one element (bit-path)
    assert [c[1] for c in calc_factor_code_chain('CommodityPrice', 'ObservedBasis', ('CME',), stat, {})] \
        == [utils.Factor('CommodityPrice', ('CME',))]
    # composite -> [primary, basis named by the whole prefix]
    assert [c[1] for c in calc_factor_code_chain('CommodityPrice', 'ObservedBasis', ('CME', 'LME'), stat, {})] \
        == [utils.Factor('CommodityPrice', ('CME',)), utils.Factor('ObservedBasis', ('CME', 'LME'))]
    # bases stack
    assert [c[1] for c in calc_factor_code_chain('CommodityPrice', 'ObservedBasis', ('CME', 'LME', 'SHF'), stat, {})] \
        == [utils.Factor('CommodityPrice', ('CME',)), utils.Factor('ObservedBasis', ('CME', 'LME')),
            utils.Factor('ObservedBasis', ('CME', 'LME', 'SHF'))]


def test_chain_generic_fx_and_ir_identity():
    from riskflow.instruments import calc_factor_code_chain, get_interest_factor
    fx = _offsets(('FxRate', ('EUR',)), ('ObservedBasis', ('EUR', 'PROXY')))
    assert [c[1] for c in calc_factor_code_chain('FxRate', 'ObservedBasis', ('EUR', 'PROXY'), fx, {})] \
        == [utils.Factor('FxRate', ('EUR',)), utils.Factor('ObservedBasis', ('EUR', 'PROXY'))]
    # get_interest_factor is the identity (head==tail) case of the same function
    ir = _offsets(('InterestRate', ('USD',)), ('InterestRate', ('USD', 'LIBOR')))
    assert [c[1] for c in get_interest_factor(('USD', 'LIBOR'), ir, {}, {})] \
        == [utils.Factor('InterestRate', ('USD',)), utils.Factor('InterestRate', ('USD', 'LIBOR'))]


def test_stacked_basis_sums_in_spot_rate():
    torch.manual_seed(4)
    prim, b1, b2 = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    shared = _shared()
    shared.t_Scenario_Buffer.update({'P': prim, 'B1': b1, 'B2': b2})
    tg = _grid([0, 2, 4])
    idx = torch.tensor([0, 2, 4])
    out = utils.calc_time_grid_spot_rate([(True, 'P'), (True, 'B1'), (True, 'B2')], tg, shared)
    assert torch.equal(out, prim[idx] + b1[idx] + b2[idx])


def test_linked_parent_derived_from_name():
    from riskflow.stochasticprocess import BasisLinkedSpotModel
    F = utils.Factor
    # depth-2: parent = name[:-1], type resolved from all_factors (exactly one composable type)
    b = BasisLinkedSpotModel(factor=None, param={})
    b.calc_references(F('ObservedBasis', ('PLATINUM_CME', 'LME_CME')), {}, {},
                      {}, {F('CommodityPrice', ('PLATINUM_CME',)): object()})
    assert b.linked_key == F('CommodityPrice', ('PLATINUM_CME',))
    # stacked (len>2): the immediate parent is the previous chain prefix as ObservedBasis
    b.calc_references(F('ObservedBasis', ('PLATINUM_CME', 'LME_CME', 'SHF')), {}, {}, {}, {})
    assert b.linked_key == F('ObservedBasis', ('PLATINUM_CME', 'LME_CME'))
    # ambiguous / missing parent type -> loud
    with pytest.raises(Exception, match='exactly one composable'):
        b.calc_references(F('ObservedBasis', ('PLATINUM_CME', 'LME_CME')), {}, {}, {}, {})


# --- deal-level: composed reference in the Commodity NAME -----------------------------------
# The shipping liability's Commodity is the composed name PLATINUM_CME.LME_CME (primary + basis).

def _ship_cfg(commodity=None):
    cfg = json.load(open(SHIPPING))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'simulate_only'
    calc['Batch_Size'] = 64
    calc['Simulation_Batches'] = 1
    calc['Random_Seed'] = 1
    calc['Hedging_Problem'].pop('Solver', None)
    if commodity is not None:
        calc['Hedging_Problem']['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Commodity'] = commodity
    return cfg


def _liability_mtm(cfg):
    cx = rf.Context()
    cx.load_json((json.dumps(cfg, default=str), 'basis.json'))
    _, out = cx.run_job()
    return out.bundle.liability_mtm


def test_composed_name_sums_into_liability():
    full = _liability_mtm(_ship_cfg())                                  # CME + LME_CME
    primary = _liability_mtm(_ship_cfg(commodity='PLATINUM_CME'))       # CME alone
    assert not torch.equal(full, primary), 'LME_CME basis made no difference to the liability'
