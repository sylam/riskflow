"""Heston-Nandi GARCH(1,1) price factor: bootstrapper (riskflow/bootstrappers.py) and risk
factor (riskflow/riskfactors.py).

The KEY test is the round trip: synthesise European option quotes FROM a known set of
risk-neutral HN parameters with the validated pricer (riskflow/hn_garch.py, 98 tests), hand them
to the bootstrapper as market prices, and recover the parameters. It runs for THREE asset
classes - an FX rate (FXVol, no yield), an equity (EquityPriceVol + DividendRate) and a
commodity (CommodityPriceVol + a carry InterestRate) - because the factor is asset class
agnostic and nothing else pins that.

Everything else pins the plumbing: dispatch by class name, the tenor-index/current-value key
parity that utils.get_tenors + make_factor_index rely on, the end-to-end Config.bootstrap()
path, warm-start idempotency and the stationarity constraint.

Run: ``pytest tests/test_hn_bootstrapper.py -q``
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest
import torch

from riskflow import bootstrappers, hn_garch, riskfactors, utils
from riskflow.config import Config, ModelParams

DT = torch.float64
DEV = torch.device('cpu')
BASE_DATE = pd.Timestamp('2024-06-28')
RATE = 0.03
SPY = 252.0
PANELS = 32
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'data', 'MarketDataRF_heston_nandi.json')

# the parameters the round trip has to recover
TRUTH = hn_garch.hn_params_from_targets(
    ann_vol=0.28, persistence=0.96, gamma=250.0, leverage_share=0.5, steps_per_year=SPY)
H0_TRUTH = 1.3 * TRUTH.stationary_var

# (name, spot, underlying type + params, vol type, yield type + params) - the factor is asset
# class agnostic, so the round trip runs for all of these
ASSET_CLASSES = [
    ('AUD', 0.65, 'FxRate', {'Domestic_Currency': 'USD', 'Interest_Rate': 'AUD', 'Priority': 1},
     'FXVol', None, {}),
    ('TEST_EQ', 1000.0, 'EquityPrice', {'Currency': 'USD', 'Interest_Rate': 'USD'},
     'EquityPriceVol', 'DividendRate', {'Currency': 'USD'}),
    ('TEST_COM', 950.0, 'CommodityPrice', {'Currency': 'USD', 'Interest_Rate': 'USD'},
     'CommodityPriceVol', 'InterestRate', {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None}),
]
YIELD = 0.02


def price_factors(asset):
    name, spot, spot_type, spot_param, vol_type, yield_type, yield_param = asset
    factors = {
        '{}.{}'.format(spot_type, name): dict(spot_param, Spot=spot),
        '{}.{}'.format(vol_type, name): {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness', 'Currency': 'USD',
            'Surface': utils.Curve([], [[m, t, 0.28] for m in (0.9, 1.0, 1.1) for t in (0.02, 0.25)])},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])}}
    if yield_type:
        factors['{}.{}'.format(yield_type, name)] = dict(
            yield_param, Curve=utils.Curve([], [[0.01, YIELD], [5.0, YIELD]]))
    return factors


def synthetic_quotes(asset, days, moneyness, option_type):
    """Exact HN premia at TRUTH for one expiry - the test replicates the bootstrapper's own
    (t, n, carry, yield discount) convention, so a change to it breaks the round trip."""
    spot, q = asset[1], YIELD if asset[5] else 0.0
    t = days / 365.0
    n = max(int(round(t * SPY)), 1)
    strikes = np.array(moneyness) * spot * np.exp((RATE - q) * t)
    p = hn_garch.HNParams(TRUTH.omega, TRUTH.alpha, TRUTH.beta, TRUTH.gamma,
                          (RATE - q) * t / n).as_tensors(DT)
    k = torch.tensor(strikes, dtype=DT)
    call = hn_garch.hn_call(spot, k, p, n, H0_TRUTH, panels=PANELS)
    value = np.exp(-q * t) * (call if option_type == 'Call' else call - spot + k * torch.exp(-p.r * n))
    return [{'Expiry_Date': BASE_DATE + pd.Timedelta(days=days), 'Strike': float(strike),
             'Option_Type': option_type, 'Units': 1.0, 'Weight': 1.0, 'Quoted_Market_Value': float(v)}
            for strike, v in zip(strikes, value)]


def market_prices(asset):
    name = asset[0]
    return {'HestonNandiModelPrices.' + name: {'instrument': {
        'Underlying': name, 'Volatility': name, 'Discount_Rate': 'USD', 'Yield': name if asset[5] else '',
        'Quote_Type': 'Premium', 'Steps_Per_Year': SPY, 'Quadrature_Panels': PANELS,
        'European_Options': synthetic_quotes(asset, 30, (0.85, 0.925, 1.0), 'Call') +
                            synthetic_quotes(asset, 30, (1.075, 1.15), 'Put') +
                            synthetic_quotes(asset, 91, (0.85, 0.925, 1.0), 'Call') +
                            synthetic_quotes(asset, 91, (1.075, 1.15), 'Put')}}}


def run(bootstrapper, factors, prices, name):
    bootstrapper.bootstrap({'Base_Date': BASE_DATE}, {}, factors, ModelParams(), prices, {})
    return factors['HestonNandiModelParameters.' + name]


@pytest.fixture(scope='module', params=ASSET_CLASSES, ids=lambda x: x[2])
def round_trip(request):
    """(fitted price factor, price factors, market prices, bootstrapper, asset) - one converged
    fit per asset class, shared by every test that needs an optimum."""
    asset = request.param
    factors, prices = price_factors(asset), market_prices(asset)
    boot = bootstrappers.HestonNandiModelParameters({}, DEV, DT)
    return run(boot, factors, prices, asset[0]), factors, prices, boot, asset


# ======================================================================================
# the round trip - synthetic quotes from known parameters, recovered by the bootstrapper
# ======================================================================================

@pytest.mark.parametrize('field,truth,rtol', [
    ('Omega', float(TRUTH.omega), 5e-2),
    ('Alpha', float(TRUTH.alpha), 5e-2),
    ('Beta', float(TRUTH.beta), 2e-2),
    ('Gamma_Star', float(TRUTH.gamma), 2e-2),
    ('H0', float(H0_TRUTH), 1e-2)])
def test_round_trip_recovers_parameters(round_trip, field, truth, rtol):
    assert round_trip[0][field] == pytest.approx(truth, rel=rtol)


def test_round_trip_recovers_persistence_and_long_run_vol(round_trip):
    p = hn_garch.HNParams(*[round_trip[0][x] for x in ('Omega', 'Alpha', 'Beta', 'Gamma_Star')])
    assert p.persistence == pytest.approx(float(TRUTH.persistence), abs=1e-3)
    assert p.ann_vol(SPY) == pytest.approx(TRUTH.ann_vol(SPY), rel=1e-2)


def test_round_trip_reprices_the_quotes(round_trip):
    """The premium residual, not just the parameters - the optimum must actually fit, and it must
    do so through the r-q carry / exp(-qt) discount when the asset class has a yield."""
    param, _, prices, boot, asset = round_trip
    quotes = prices['HestonNandiModelPrices.' + asset[0]]['instrument']['European_Options']
    for option in quotes:
        p = hn_garch.HNParams(*[torch.tensor(param[k], dtype=DT) for k in
                                ('Omega', 'Alpha', 'Beta', 'Gamma_Star')],
                              torch.tensor((option['r'] - option['q']) * option['T'] / option['n'], dtype=DT))
        fitted = boot.price(asset[1], torch.tensor(option['Strike'], dtype=DT),
                            1.0 if option['Option_Type'] == 'Call' else 0.0, 1.0, p, option['n'],
                            torch.tensor(param['H0'], dtype=DT), PANELS,
                            np.exp(-option['q'] * option['T']))
        assert float(fitted) == pytest.approx(option['Premium'], rel=1e-4, abs=1e-6)
    assert option['q'] == (YIELD if asset[5] else 0.0)


# ======================================================================================
# asset class genericity - the input resolution must not be commodity specific
# ======================================================================================

def test_resolve_picks_the_factor_type_that_exists(round_trip):
    _, factors, prices, boot, asset = round_trip
    instrument = prices['HestonNandiModelPrices.' + asset[0]]['instrument']
    assert boot.resolve(instrument, 'Underlying', factors).type == asset[2]
    assert boot.resolve(instrument, 'Volatility', factors).type == asset[4]
    assert boot.resolve(instrument, 'Discount_Rate', factors).type == 'InterestRate'
    yield_factor = boot.resolve(instrument, 'Yield', factors)
    assert (yield_factor.type if yield_factor else None) == asset[5]


def test_resolve_honours_an_explicit_type_and_an_unset_field():
    boot = bootstrappers.HestonNandiModelParameters({}, DEV, DT)
    factors = {'FxRate.AUD': {}, 'CommodityPrice.AUD': {}}
    # the name exists under two types - the explicit type disambiguates, otherwise first wins
    assert boot.resolve({'Underlying': 'AUD'}, 'Underlying', factors).type == 'FxRate'
    assert boot.resolve({'Underlying': 'AUD', 'Underlying_Type': 'CommodityPrice'},
                        'Underlying', factors).type == 'CommodityPrice'
    assert boot.resolve({'Underlying': 'AUD'}, 'Yield', factors) is None


def test_written_factor_name_is_asset_class_agnostic(round_trip):
    _, factors, _, _, asset = round_trip
    assert 'HestonNandiModelParameters.' + asset[0] in factors


# ======================================================================================
# the SURFACE SYNTHESIS path - the round trip above quotes premiums directly, which bypasses
# the vol lookup entirely, so the moneyness convention needs its own coverage. HN is the first
# bootstrapper to query a surface AWAY FROM THE MONEY, where the framework's five conventions
# stop coinciding.
# ======================================================================================

SKEW_SLOPE = 0.10
ATM_VOL = 0.30


def skewed_factors(surface_type, nodes, centre):
    """A surface that is LINEAR in its own moneyness coordinate, so the (bilinear) interpolated
    vol at any lookup point is exactly ATM_VOL + SKEW_SLOPE*(m - centre) and can be hand computed.
    The yield makes the forward differ from the spot, so Use_Forward is observable."""
    return {'CommodityPrice.SKEW': {'Currency': 'USD', 'Interest_Rate': 'USD', 'Spot': 1000.0},
            'CommodityPriceVol.SKEW': {
                'Surface_Type': surface_type, 'Moneyness_Rule': 'Sticky_Moneyness', 'Currency': 'USD',
                'Surface': utils.Curve([], [[m, t, ATM_VOL + SKEW_SLOPE * (m - centre)]
                                            for m in nodes for t in (0.02, 0.25)])},
            'InterestRate.SKEW': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                  'Curve': utils.Curve([], [[0.0, YIELD], [5.0, YIELD]])},
            'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                 'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])}}


def skewed_prices(**flags):
    return {'HestonNandiModelPrices.SKEW': {'instrument': dict(
        {'Underlying': 'SKEW', 'Volatility': 'SKEW', 'Discount_Rate': 'USD', 'Yield': 'SKEW',
         'Quote_Type': 'Implied_Volatility', 'Steps_Per_Year': SPY, 'Quadrature_Panels': 16,
         'European_Options': [
             {'Expiry_Date': BASE_DATE + pd.Timedelta(days=7), 'Strike': strike,
              'Option_Type': 'Call', 'Units': 1.0, 'Weight': 1.0, 'Quoted_Market_Value': 0.0}
             for strike in (900.0, 1000.0, 1100.0)]}, **flags)}}


@pytest.mark.parametrize('flags,convention', [
    ({}, lambda k, s, f: s / k),
    ({'Use_Forward': 'Yes'}, lambda k, s, f: f / k),
    ({'Invert_Moneyness': 'Yes'}, lambda k, s, f: k / s),
    ({'Use_Forward': 'Yes', 'Invert_Moneyness': 'Yes'}, lambda k, s, f: k / f)])
def test_surface_synthesis_honours_the_moneyness_convention(flags, convention):
    """The lookup point, the vol it returns and the Black premium built from it - all three pinned
    against a hand computed lookup under the DECLARED convention (defaults = the pricing path's:
    spot/strike). A silently mis-looked-up vol gives a wrong-but-converged calibration."""
    factors, prices = skewed_factors('Explicit', (0.8, 0.9, 1.0, 1.1, 1.2), 1.0), skewed_prices(**flags)
    run(bootstrappers.HestonNandiModelParameters({}, DEV, DT), factors, prices, 'SKEW')

    spot = factors['CommodityPrice.SKEW']['Spot']
    for option in prices['HestonNandiModelPrices.SKEW']['instrument']['European_Options']:
        t, k = option['T'], option['Strike']
        forward = spot * np.exp((RATE - YIELD) * t)
        assert option['q'] == YIELD and forward != pytest.approx(spot)
        moneyness = convention(k, spot, forward)
        assert option['Moneyness'] == pytest.approx(moneyness, rel=1e-12)
        assert option['sigma'] == pytest.approx(ATM_VOL + SKEW_SLOPE * (moneyness - 1.0), rel=1e-10)
        assert option['Premium'] == pytest.approx(utils.black_european_option_price(
            forward, k, option['r'], option['sigma'], t, 1.0, 1.0), rel=1e-12)


def test_the_four_conventions_actually_disagree():
    """Guards the test above from being vacuous - away from the money the conventions differ."""
    vols = []
    for flags in ({}, {'Use_Forward': 'Yes'}, {'Invert_Moneyness': 'Yes'},
                  {'Use_Forward': 'Yes', 'Invert_Moneyness': 'Yes'}):
        factors, prices = skewed_factors('Explicit', (0.8, 0.9, 1.0, 1.1, 1.2), 1.0), skewed_prices(**flags)
        run(bootstrappers.HestonNandiModelParameters({}, DEV, DT), factors, prices, 'SKEW')
        vols.append(tuple(round(x['sigma'], 10) for x in
                          prices['HestonNandiModelPrices.SKEW']['instrument']['European_Options']))
    assert len(set(vols)) == 4


def test_relative_forward_surface_is_looked_up_at_its_own_coordinate():
    """A different SubType, a different coordinate - (K-F)/F, not S/K."""
    factors = skewed_factors('Relative_Forward', (-0.2, -0.1, 0.0, 0.1, 0.2), 0.0)
    prices = skewed_prices()
    run(bootstrappers.HestonNandiModelParameters({}, DEV, DT), factors, prices, 'SKEW')

    spot = factors['CommodityPrice.SKEW']['Spot']
    for option in prices['HestonNandiModelPrices.SKEW']['instrument']['European_Options']:
        forward = spot * np.exp((RATE - YIELD) * option['T'])
        moneyness = (option['Strike'] - forward) / forward
        assert option['Moneyness'] == pytest.approx(moneyness, rel=1e-12)
        assert option['sigma'] == pytest.approx(ATM_VOL + SKEW_SLOPE * moneyness, rel=1e-10)


def test_moneyness_delegates_to_pricing_calc_moneyness():
    """Not reimplemented here - the SubType dispatch is pricing.calc_moneyness, the same function
    every option deal uses. These are its documented conventions (pricing.py:40-66)."""
    from riskflow import pricing

    class Surface(object):
        # the ONLY thing calc_moneyness reads is the SubType, so this stands in for surfaces that
        # are expensive to build (a Malz surface has to be solved from a delta surface first)
        def __init__(self, surface_type):
            self.surface_type = surface_type

        def get_subtype(self):
            return self.surface_type, 'Sticky_Moneyness'

    boot = bootstrappers.HestonNandiModelParameters({}, DEV, DT)
    k, s, f = 1100.0, 1000.0, 1020.0
    for surface_type, expected in [
            ('Explicit', s / k), ('Relative_Forward', (k - f) / f), ('Malz', np.log(s / k))]:
        surface = Surface(surface_type)
        assert boot.moneyness(k, s, f, surface, False, False) == pytest.approx(expected, rel=1e-12)
        deal_data = utils.DealDataType(
            Instrument=None, Time_dep=None, Calc_res=None,
            Factor_dep={'Volatility': [(None, None, surface.get_subtype())]})
        assert boot.moneyness(k, s, f, surface, False, False) == pytest.approx(float(
            pricing.calc_moneyness(*[torch.tensor(x, dtype=DT) for x in (k, s, f)], deal_data)))
    # and the flags flip the plain-2D coordinate exactly as they do in the pricing path
    assert boot.moneyness(k, s, f, Surface('Explicit'), True, False) == pytest.approx(f / k)
    assert boot.moneyness(k, s, f, Surface('Explicit'), False, True) == pytest.approx(k / s)


@pytest.mark.parametrize('surface_type', ['SVI', 'Skew'])
def test_parametric_surface_is_refused_loudly(surface_type, caplog):
    """SVI/Skew vols are not a table lookup - refuse rather than fit to a mis-looked-up vol."""
    factors = skewed_factors(surface_type, (0.8, 0.9, 1.0, 1.1, 1.2), 1.0)
    prices = skewed_prices()
    with caplog.at_level(logging.ERROR):
        bootstrappers.HestonNandiModelParameters({}, DEV, DT).bootstrap(
            {'Base_Date': BASE_DATE}, {}, factors, ModelParams(), prices, {})
    assert 'HestonNandiModelParameters.SKEW' not in factors
    assert 'Surface_Type {}'.format(surface_type) in caplog.text and 'SKEW' in caplog.text


def test_parametric_surface_is_fine_when_premiums_are_quoted(caplog):
    """...but only the SYNTHESIS path needs the surface - quoted premiums never touch it."""
    factors = skewed_factors('SVI', (0.8, 0.9, 1.0, 1.1, 1.2), 1.0)
    prices = skewed_prices()
    instrument = prices['HestonNandiModelPrices.SKEW']['instrument']
    instrument['Quote_Type'] = 'Premium'
    for option, premium in zip(instrument['European_Options'], (110.0, 30.0, 5.0)):
        option['Quoted_Market_Value'] = premium
    with caplog.at_level(logging.ERROR):
        bootstrappers.HestonNandiModelParameters({}, DEV, DT).bootstrap(
            {'Base_Date': BASE_DATE}, {}, factors, ModelParams(), prices, {})
    assert 'HestonNandiModelParameters.SKEW' in factors
    assert 'Surface_Type' not in caplog.text


# ======================================================================================
# stationarity - a box constraint on the fitted persistence, so it holds at the optimum
# ======================================================================================

def test_stationarity_at_the_optimum(round_trip):
    param = round_trip[0]
    assert param['Alpha'] * param['Gamma_Star'] ** 2 + param['Beta'] < 1.0
    assert param['Omega'] > 0.0 and param['Alpha'] >= 0.0 and 0.0 <= param['Beta'] < 1.0
    assert param['H0'] > 0.0


def test_reparam_is_stationary_everywhere_in_the_box():
    """Not just at the optimum - EVERY point the optimizer can visit is stationary."""
    boot = bootstrappers.HestonNandiModelParameters({}, DEV, DT)
    lo, hi = np.array(boot.bounds).T
    rng = np.random.RandomState(0)
    for x in np.vstack([lo, hi, lo + rng.rand(64, len(lo)) * (hi - lo)]):
        omega, alpha, beta, gamma, h0 = [float(v) for v in boot.reparam(torch.tensor(x, dtype=DT))]
        assert alpha * gamma ** 2 + beta < 1.0
        assert omega > 0.0 and h0 > 0.0


def test_reparam_round_trips_through_unreparam():
    boot = bootstrappers.HestonNandiModelParameters({}, DEV, DT)
    x = np.array([np.log(4.7e-6), 0.96, 0.5, 0.25, np.log(4.04e-4)])
    assert boot.unreparam(*[float(v) for v in boot.reparam(torch.tensor(x, dtype=DT))]) == \
           pytest.approx(x, rel=1e-12)


# ======================================================================================
# plumbing - dispatch, key parity, the written factor
# ======================================================================================

def test_construct_bootstrapper_dispatch():
    boot = bootstrappers.construct_bootstrapper('HestonNandiModelParameters', {})
    assert isinstance(boot, bootstrappers.HestonNandiModelParameters)


def test_bootstrapper_and_riskfactor_class_names_match():
    """The write is price_factors[Factor(self.__class__.__name__, ...)] - nothing checks that the
    risk factor of that name exists, so a mismatch is a late KeyError."""
    assert hasattr(riskfactors, bootstrappers.HestonNandiModelParameters.__name__)


def test_written_factor_constructs(round_trip):
    _, factors, _, _, asset = round_trip
    factor = utils.Factor('HestonNandiModelParameters', (asset[0],))
    obj = riskfactors.construct_factor(factor, factors, ModelParams())
    assert isinstance(obj, riskfactors.HestonNandiModelParameters)
    assert obj.current_value()['Gamma_Star'][0] == factors[utils.check_tuple_name(factor)]['Gamma_Star']


def test_tenor_index_and_current_value_key_parity(round_trip):
    _, factors, _, _, asset = round_trip
    obj = riskfactors.construct_factor(
        utils.Factor('HestonNandiModelParameters', (asset[0],)), factors, ModelParams())
    assert obj.get_tenor_indices().keys() == obj.current_value().keys()
    assert all(v.shape == (1, 1) for v in obj.get_tenor_indices().values())
    assert all(v.shape == (1,) for v in obj.current_value().values())


def test_get_tenors_yields_five_scope_names(round_trip):
    _, factors, _, _, asset = round_trip
    factor = utils.Factor('HestonNandiModelParameters', (asset[0],))
    tenors = utils.get_tenors({factor: riskfactors.construct_factor(factor, factors, ModelParams())})
    assert set(tenors) == {'HestonNandiModelParameters.{}.{}'.format(asset[0], x) for x in
                           riskfactors.HestonNandiModelParameters.parameters}


def test_fields_mapping_matches_the_factor():
    from riskflow.fields import mapping
    assert mapping['Factor']['types']['HestonNandiModelParameters'] == list(
        riskfactors.HestonNandiModelParameters.parameters)
    assert all(x in mapping['Factor']['fields'] for x in
               mapping['Factor']['types']['HestonNandiModelParameters'])
    assert all(x in mapping['MarketPrices']['fields'] for x in
               mapping['MarketPrices']['types']['HestonNandiModelPrices'])


def test_not_registered_as_an_implied_model():
    """HN is consumed as a STATIC dependent factor - registering it in implied_models would mint a
    duplicate AAD leaf (implied_var and static_var under identical scope names)."""
    assert 'HestonNandiModelParameters' not in ModelParams().implied_models.values()


# ======================================================================================
# idempotency - the bootstrapper warm starts off its own output
# ======================================================================================

def test_rerunning_on_its_own_output_is_stable(round_trip):
    param, factors, prices, boot, asset = round_trip
    again = run(boot, factors, prices, asset[0])
    for k in riskfactors.HestonNandiModelParameters.parameters:
        assert again[k] == pytest.approx(param[k], rel=1e-6)


# ======================================================================================
# end to end - Config.bootstrap() off a market data JSON (equity + dividend yield), quotes
# synthesised from the vol surface
# ======================================================================================

def test_config_bootstrap_end_to_end(caplog):
    cx = Config()
    cx.parse_json(FIXTURE)
    assert 'HestonNandiModelParameters.TEST_HN' not in cx.params['Price Factors']

    with caplog.at_level(logging.INFO):
        cx.bootstrap()

    param = cx.params['Price Factors']['HestonNandiModelParameters.TEST_HN']
    assert set(param) == {'Property_Aliases'} | set(riskfactors.HestonNandiModelParameters.parameters)
    assert param['Alpha'] * param['Gamma_Star'] ** 2 + param['Beta'] < 1.0
    # the diagnostics the bootstrapper is required to log
    text = caplog.text
    assert 'persistence' in text and 'long run vol' in text and 'c_premium' in text
    # ...and the guard that a no-op bootstrap can't pass unnoticed didn't fire
    assert 'wrote no' not in text
    # the factor constructs straight out of the bootstrapped config
    obj = riskfactors.construct_factor(
        utils.Factor('HestonNandiModelParameters', ('TEST_HN',)),
        cx.params['Price Factors'], cx.params['Price Factor Interpolation'])
    assert obj.current_value()['H0'][0] == param['H0']


def test_no_op_bootstrap_is_logged_loudly(caplog):
    """The known silent-failure mode: Config.bootstrap swallows construction errors and
    find_models degrades quietly, so a bootstrapper that writes nothing must say so."""
    cx = Config()
    cx.parse_json(FIXTURE)
    cx.params['Market Prices'] = {}
    with caplog.at_level(logging.ERROR):
        cx.bootstrap()
    assert 'wrote no HestonNandiModelParameters.* price factor' in caplog.text
