########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of RiskFlow.
#
# RiskFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# RiskFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RiskFlow.  If not, see <http://www.gnu.org/licenses/>.
########################################################################


# import standard libraries
import time
import logging
from collections import namedtuple

# third party stuff
import numpy as np
import pandas as pd
import torch

# Internal modules
from . import utils, pricing, instruments, riskfactors, stochasticprocess, hn_garch

import scipy.optimize

market_swap_class = namedtuple('market_swap', 'deal_data price weight')
date_desc = {'years': 'Y', 'months': 'M', 'days': 'D'}
# date formatter
date_fmt = lambda x: ''.join(['{0}{1}'.format(v, date_desc[k]) for k, v in x.kwds.items()])


class RiskNeutralInterestRate_State(utils.Calculation_State):
    def __init__(self, scenario_keys, batch_size, device, dtype, nomodel='Constant'):
        super(RiskNeutralInterestRate_State, self).__init__(
            None, torch.ones([1, 1], dtype=dtype, device=device), 2048, None, nomodel, batch_size)
        # these are tensors
        self.t_PreCalc = {}
        self.scenario_keys = scenario_keys
        self.t_random_batch = None
        self.batch_index = 0
        self.t_Scenario_Buffer = {}

    @property
    def t_random_numbers(self):
        return self.t_random_batch[self.batch_index]

    def reset(self, num_batches, numfactors, time_grid):
        # clear the buffers
        self.t_Buffer.clear()
        self.t_PreCalc.clear()

        if self.t_random_batch is None:
            # the sobol engine in torch > 1.8 goes up to dimension 21201 - so this should be fine
            self.sobol = torch.quasirandom.SobolEngine(
                dimension=time_grid.time_grid_years.size * numfactors, scramble=True, seed=1234)
            # skip this many samples
            self.sobol.fast_forward(2048)
            # make sure we don't include 1 or 0
            sample_sobol = self.sobol.draw(self.simulation_batch * num_batches).reshape(
                num_batches, self.simulation_batch, -1)
            sample = torch.erfinv(2 * (0.5 + (1 - torch.finfo(sample_sobol.dtype).eps) * (
                    sample_sobol - 0.5)) - 1).reshape(
                num_batches, self.simulation_batch, -1) * 1.4142135623730951
            self.t_random_batch = sample.transpose(1, 2).reshape(
                num_batches, numfactors, -1, self.simulation_batch).to(self.one.device)


def create_float_cashflows(base_date, cashflow_obj, frequency):
    cashflows = []
    for cashflow, reset in zip(cashflow_obj.schedule, cashflow_obj.Resets.schedule):
        cashflows.append({
            'Payment_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Pay_Day]),
            'Notional': 1.0,
            'Accrual_Start_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Start_Day]),
            'Accrual_End_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_End_Day]),
            'Accrual_Year_Fraction': cashflow[utils.CASHFLOW_INDEX_Year_Frac],
            'Fixed_Amount': cashflow[utils.CASHFLOW_INDEX_FixedAmt],
            'Resets': [[base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Reset_Day]),
                        base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Start_Day]),
                        base_date + pd.offsets.Day(reset[utils.RESET_INDEX_End_Day]),
                        reset[utils.RESET_INDEX_Accrual],
                        frequency, 'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]],
            'Margin': utils.Basis(0.0)
        })
    return cashflows


def normalize(sample):
    '''Simple function to ensure that the sample used for the monte carlo sim has mean 0 and var 1'''
    return (sample - sample.mean(axis=0)) / sample.std(axis=0)


def create_market_swaps(base_date, time_grid, curve_index, vol_surface, curve_factor,
                        instrument_definitions, rate=None):
    # store these benchmark swap definitions if necessary
    benchmarks = []
    # store the benchmark instruments
    all_deals = {}
    # cater for shifted lognormal vols
    shift_parameter = vol_surface.BlackScholesDisplacedShiftValue / 100.0
    for instrument in instrument_definitions:
        # set up the instrument
        effective = base_date + instrument['Start']
        maturity = effective + instrument['Tenor']
        exp_days = (effective - base_date).days
        tenor = (maturity - effective).days / utils.DAYS_IN_YEAR
        expiry = exp_days / utils.DAYS_IN_YEAR
        time_index = np.searchsorted(time_grid.mtm_time_grid, [exp_days], side='right') - 1
        swaption_name = 'Swaption_{}_{}'.format(
            date_fmt(instrument['Start']), date_fmt(instrument['Tenor']))

        float_pay_dates = instruments.generate_dates_backward(
            maturity, effective, instrument['Floating_Frequency'])

        float_cash = utils.generate_float_cashflows(
            base_date, time_grid, float_pay_dates, 1.0, None, None,
            instrument['Floating_Frequency'], pd.DateOffset(month=0),
            utils.get_day_count(instrument['Floating_Day_Count']), 0.0)

        K, pvbp = float_cash.get_par_swap_rate(base_date, curve_factor)

        if instrument['Fixed_Frequency'] != instrument['Floating_Frequency']:
            fixed_pay_dates = instruments.generate_dates_backward(
                maturity, effective, instrument['Fixed_Frequency'])
            fixed_cash = utils.generate_fixed_cashflows(
                base_date, fixed_pay_dates, 1.0, None, utils.get_day_count(instrument['Fixed_Day_Count']), 0.0)
            pv_float = K * pvbp
            pvbp = fixed_cash.get_par_swap_rate(base_date, curve_factor)
            K = pv_float / pvbp
            fixed_cash.set_fixed_amount(K)
            fixed_indices = float_cash[:, utils.CASHFLOW_INDEX_Pay_Day].searchsorted(
                fixed_cash[:, utils.CASHFLOW_INDEX_Pay_Day])

            if not (float_cash[fixed_indices, utils.CASHFLOW_INDEX_Pay_Day] ==
                    fixed_cash[:, utils.CASHFLOW_INDEX_Pay_Day]).all():
                logging.error('Float leg and Fixed legs do not coincide')
                raise Exception('Float leg and Fixed legs do not coincide')

            # set the float leg fixed amount
            float_cash.schedule[fixed_indices, utils.CASHFLOW_INDEX_FixedAmt] = \
                -fixed_cash[:, utils.CASHFLOW_INDEX_FixedAmt]
        else:
            float_cash.set_fixed_amount(-K)

        # get the atm vol
        if instrument['Market_Volatility'].amount:
            vol = instrument['Market_Volatility'].amount
        else:
            vol = vol_surface.ATM(tenor, expiry)[0][0]

        deal_data = utils.DealDataType(
            Instrument=None, Factor_dep={'Cashflows': float_cash, 'Forward': curve_index,
                                         'Discount': curve_index, 'CompoundingMethod': 'None'},
            Time_dep=utils.DealTimeDependencies(time_grid.mtm_time_grid, time_index), Calc_res=None)

        shifted_strike = K + shift_parameter
        # first check if we have the actual premium (not implied)
        if vol_surface.premiums is not None:
            swaption_price = vol_surface.get_premium(date_fmt(instrument['Start']), date_fmt(instrument['Tenor']))
            if vol_surface.delta:
                try:
                    implied_vol = scipy.optimize.brentq(lambda v: pvbp * utils.black_european_option_price(
                        shifted_strike, shifted_strike, 0.0, v, expiry, 1.0, 1.0) - swaption_price, 0.01, vol + .5)
                except:
                    modified_k = vol_surface.get_strike_from_premiums(date_fmt(instrument['Start']),
                                                                      date_fmt(instrument['Tenor']))
                    logging.warning(
                        'Implied vol calc during delta bump failed - calculated strike is {} - using strike from premium file {}'.format(
                            K, modified_k))
                    shifted_strike = modified_k + shift_parameter
                    implied_vol = scipy.optimize.brentq(lambda v: pvbp * utils.black_european_option_price(
                        shifted_strike, shifted_strike, 0.0, v, expiry, 1.0, 1.0) - swaption_price, 0.01, vol + .5)

                swaption_price = pvbp * utils.black_european_option_price(
                    shifted_strike, shifted_strike, 0.0, implied_vol + vol_surface.delta, expiry, 1.0, 1.0)
        else:
            swaption_price = pvbp * utils.black_european_option_price(
                shifted_strike, shifted_strike, 0.0, vol + vol_surface.delta, expiry, 1.0, 1.0)

        # store this
        all_deals[swaption_name] = market_swap_class(
            deal_data=deal_data, price=swaption_price, weight=instrument['Weight'])

        # store the benchmark
        if rate is not None:
            benchmarks.append(
                instruments.construct_instrument(
                    {'Object': 'CFFloatingInterestListDeal',
                     'Reference': swaption_name,
                     'Currency': curve_factor.param['Currency'],
                     'Discount_Rate': '.'.join(rate),
                     'Forecast_Rate': '.'.join(rate),
                     'Buy_Sell': 'Buy',
                     'Cashflows': {'Items': create_float_cashflows(
                         base_date, float_cash, instrument['Floating_Frequency'])}},
                    {})
            )

    return all_deals, benchmarks


class CSForwardPriceModelParameters(object):
    documentation = (
        'Energy',
        ['For Risk Neutral simulation, the Clewlow Strickland Model is calibrated to a set of European Energy',
         'futures options $J$. We use scipy',
         'an integrated curve $\\bar{\\sigma}(t)$ needs to be specified and is',
         'interpreted as the average volatility at time $t$. This is typically obtained from the corresponding',
         'ATM volatility. This is then used to construct a new variance curve $V(t)$ which is defined as',
         '$V(0)=0, V(t_i)=\\bar{\\sigma}(t_i)^2 t_i$ and $V(t)=\\bar{\\sigma}(t_n)^2 t$ for $t>t_n$ where',
         '$t_1,...,t_n$ are discrete points on the ATM volatility curve.',
         '',
         'Points on the curve that imply a decrease in variance (i.e. $V(t_i)<V(t_{i-1})$) are adjusted to',
         '$V(t_i)=\\bar\\sigma(t_i)^2t_i=V(t_{i-1})$. This curve is then used to construct *instantaneous* curves',
         'that are then input to the corresponding stochastic process.',
         '',
         'The relationship between integrated $F(t)=\\int_0^t f_1(s)f_2(s)ds$ and instantaneous curves $f_1, f_2$',
         'where the instantaneous curves are defined on discrete points $P={t_0,t_1,..,t_n}$ with $t_0=0$ is defined',
         'on $P$ by Simpson\'s rule:',
         '',
         '$$F(t_i)=F(t_{i-1})+\\frac{t_i-t_{i-1}}{6}\\Big(f(t_i)+4f(\\frac{t_i+t_{i-1}}{2})+f(t_i)\\Big)$$',
         '',
         'and $f(t)=f_1(t)f_2(t)$. Integrated curves are flat extrapolated and linearly interpolated.'
         ]
    )

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Checks for Declining variance in the ATM vols of the relevant price factor and corrects accordingly.
        '''

        def B(a, t):
            return (1.0 - np.exp(-a * t)) / a if a != 0 else t

        def V(sigma, alpha, T, S):
            return sigma * sigma * np.exp(-2.0 * alpha * S) * B(-2.0 * alpha, T)

        def calc_error(x, options):
            sigma, alpha = x
            error = 0.0

            for option in options:
                discount = np.exp(-option['r'] * option['T'])
                error += option['Weight'] * (option['Premium'] - utils.black_european_option_price(
                    option['Forward'], option['Strike'], 0.0, np.sqrt(V(sigma, alpha, option['T'], option['S'])),
                    1.0, option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0) * discount) ** 2
            return error

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == 'CSForwardPriceModelPrices':
                # get the vol surface
                if 'ForwardPriceVol.' + implied_params['instrument']['Forward_Volatility'] in price_factors:
                    vol_factor = utils.Factor('ForwardPriceVol', utils.check_rate_name(
                        implied_params['instrument']['Forward_Volatility']))
                if 'ForwardPrice.' + implied_params['instrument']['Energy'] in price_factors:
                    energy_factor = utils.Factor('ForwardPrice', utils.check_rate_name(
                        implied_params['instrument']['Energy']))
                if 'InterestRate.' + implied_params['instrument']['Discount_Rate'] in price_factors:
                    discount_factor = utils.Factor('InterestRate', utils.check_rate_name(
                        implied_params['instrument']['Discount_Rate']))

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                    vol_surface.delta = sys_params.get('Volatility_Delta', 0.0)
                    forward = riskfactors.construct_factor(energy_factor, price_factors, factor_interp)
                    discount = riskfactors.construct_factor(discount_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                # need to loop over this and create some market prices.
                quote_type = implied_params['instrument']['Quote_Type']
                for option in implied_params['instrument']['Energy_Futures_Options']:
                    t = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
                    d = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Settlement_Date'] - sys_params['Base_Date']).days)
                    expiry_excel = (option['Expiry_Date'] - utils.excel_offset).days
                    settlement_excel = (option['Settlement_Date'] - utils.excel_offset).days
                    forward_at_exp = forward.current_value(expiry_excel)
                    forward_at_settle = forward.current_value(settlement_excel)
                    r = discount.current_value(t)
                    if quote_type == 'Implied_Volatility':
                        sigma = vol_surface.current_value([[t, d, 1.0]])[0] if not option['Quoted_Market_Value'] else \
                            option['Quoted_Market_Value']
                        sigma += vol_surface.delta
                    else:
                        logging.error('quote_type {} not supported yet'.format(quote_type))
                        continue

                    option['Strike'] = forward_at_exp if not option['Strike'] else option['Strike']
                    option['Forward'] = forward_at_settle
                    option['r'] = r
                    option['S'] = d
                    option['T'] = t
                    option['sigma'] = sigma
                    option['Premium'] = utils.black_european_option_price(
                        option['Forward'], option['Strike'], r, sigma, t,
                        option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0)

                result = scipy.optimize.minimize(
                    calc_error, (0.5, 0.1),
                    args=(implied_params['instrument']['Energy_Futures_Options'],),
                    bounds=[(0.001, 2.5), (-1, 2.0)])

                # log the results
                for option in implied_params['instrument']['Energy_Futures_Options']:
                    vol = np.sqrt(V(result.x[0], result.x[1], option['T'], option['S']) / option['T'])
                    discount = np.exp(-option['r'] * option['T'])
                    fitted_premium = utils.black_european_option_price(
                        option['Forward'], option['Strike'], 0.0,
                        np.sqrt(V(result.x[0], result.x[1], option['T'], option['S'])),
                        1.0, option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0) * discount
                    err = (fitted_premium - option['Premium']) ** 2
                    logging.info(
                        'Commodity {} strike {}, expiry {}, vol {}, c_vol {}, premium {}, c_premium {}, err {}'.format(
                            implied_params['instrument']['Energy'], option['Strike'], option['Expiry_Date'],
                            option['sigma'], vol, option['Premium'], fitted_premium, err))

                price_param = utils.Factor(self.__class__.__name__, market_factor.name)

                price_factors[utils.check_tuple_name(price_param)] = {
                    'Property_Aliases': None,
                    'Sigma': result.x[0],
                    'Alpha': result.x[1]}


class HestonNandiModelParameters(object):
    documentation = (
        'Fx And Equity',
        ['For Risk Neutral simulation, the Heston-Nandi GARCH(1,1) model is calibrated to a set of European',
         'options $J$ on a spot underlying. The model is ASSET CLASS AGNOSTIC - the *Underlying* may be any',
         'spot (0D) price factor (**FxRate**, **EquityPrice**, **CommodityPrice**, **FuturesPrice**) and the',
         '*Volatility* any (moneyness, expiry) vol surface (**FXVol**, **EquityPriceVol**,',
         '**CommodityPriceVol**); the type of each is looked up from the price factors, or named explicitly',
         'with *Underlying_Type* / *Volatility_Type*. Under the locally risk neutral valuation relationship',
         '(LRNVR) $\\lambda^*=-\\frac{1}{2}$, so the model is parameterised directly in $\\gamma^*$:',
         '',
         '$$\\log\\frac{S_{t+1}}{S_t}=(r-q)-\\frac{h_{t+1}}{2}+\\sqrt{h_{t+1}}z_{t+1}$$',
         '',
         '$$h_{t+1}=\\omega+\\beta h_t+\\alpha\\Big(z_t-\\gamma^*\\sqrt{h_t}\\Big)^2$$',
         '',
         'with $z\\sim N(0,1)$ i.i.d. and $h_{t+1}$ predictable (known at $t$), hence the fitted initial',
         'variance is $h_1$ - the variance of the *first* step - and is stored as **H0**. Option values come',
         'from the recursive characteristic function of Heston and Nandi (2000) inverted by Gauss-Legendre',
         'quadrature (see `riskflow.hn_garch`). The optional *Yield* (a dividend, repo, convenience or carry',
         'curve) enters as $q$ - the drift is $r-q$ and the value carries the extra $e^{-qt}$ factor - so',
         'equity, FX and commodity underlyings are all handled by the same objective.',
         '',
         'Writing the persistence as $\\psi=\\beta+\\alpha\\gamma^{*2}$ and the stationary per-step variance as',
         '$m=\\frac{\\omega+\\alpha}{1-\\psi}$, the objective',
         '',
         '$$\\sum_{j\\in J}w_j\\Big(V_j-V_j(\\omega,\\alpha,\\beta,\\gamma^*,h_1)\\Big)^2$$',
         '',
         'is minimized with L-BFGS-B over $\\Big(\\log\\omega,\\psi,l,\\frac{\\gamma^*}{1000},\\log h_1\\Big)$',
         'where $\\alpha=\\frac{l\\psi}{\\gamma^{*2}}$ and $\\beta=\\psi(1-l)$ for a leverage share',
         '$l\\in[0,1]$. Stationarity is therefore a *box constraint on a fitted parameter*',
         '($\\psi\\le1-10^{-6}$) and holds at every point the optimizer visits - there is no penalty term and',
         'no infeasible iterate. Gradients are exact (torch autograd through the inversion).',
         '',
         'Target premia are the Black prices at the corresponding vol surface point (as per the Clewlow',
         'Strickland bootstrapper) unless *Quote_Type* is **Premium**, in which case the quoted values are',
         'used directly. A previously bootstrapped price factor (if present) is used to warm start the fit.',
         '',
         'MONEYNESS CONVENTION. Unlike the other bootstrappers this one queries the surface AWAY FROM',
         'THE MONEY, where the five moneyness conventions in this framework no longer coincide, so the',
         'lookup point is produced by `pricing.calc_moneyness` - the same dispatch every option deal',
         'uses - off the surface *SubType*, with *Use_Forward* and *Invert_Moneyness* (Yes/No, both',
         'defaulting to **No**, i.e. $\\frac{S}{K}$, as they do in the pricing path). Supported',
         '*Surface_Types* are **Explicit**, **Relative_Forward** and **Malz** - the ones whose vol at a',
         'strike is a table lookup. **SVI** and **Skew** surfaces are parametric (the vol needs the',
         'ATM_Ref/wing machinery of the pricing path) and are REFUSED with an error rather than',
         'mis-looked-up: quote those premiums directly with *Quote_Type* **Premium**.'
         ]
    )

    # The Fourier inversion needs double precision - the framework default (float32) destroys the
    # cancellation in P1/P2 - so the dtype this is constructed with is deliberately ignored.
    prec = torch.float64
    # x = (log Omega, psi, leverage share, Gamma_Star/1000, log H0) - see reparam
    bounds = [(np.log(1e-12), np.log(1e-3)), (0.0, 1.0 - 1e-6), (0.0, 1.0),
              (1e-3, 5.0), (np.log(1e-10), np.log(1e-2))]
    # candidate price factor types for each instrument input - the underlying is any spot (0D)
    # factor and the volatility any (moneyness, expiry) surface, so one instrument definition
    # serves FX, equity and commodity underlyings
    factor_types = {'Underlying': ['FxRate', 'EquityPrice', 'CommodityPrice', 'FuturesPrice'],
                    'Volatility': ['FXVol', 'EquityPriceVol', 'CommodityPriceVol'],
                    'Discount_Rate': ['InterestRate'],
                    'Yield': ['DividendRate', 'InterestRate']}
    # Surface_Types whose vol at a strike is a TABLE LOOKUP, hence usable here. SVI/Skew are
    # parametric - their vol needs the ATM_Ref/wing machinery of the pricing path (Factor2D
    # returns the parameters, not a vol), so a synthesised premium would be silently wrong.
    tabular_surfaces = ('Explicit', 'Relative_Forward', 'Malz')

    def __init__(self, param, device, dtype):
        self.device = device
        self.param = param

    @classmethod
    def resolve(cls, instrument, field, price_factors):
        """The factor named by instrument[field], typed by the first candidate that exists in the
        price factors (the FXVol/EquityPriceVol probe of GBMAssetPriceTSModelParameters,
        generalised) or by an explicit instrument[field + '_Type']. None if the field is unset."""
        if not instrument.get(field):
            return None
        rate = utils.check_rate_name(instrument[field])
        types = [instrument[field + '_Type']] if instrument.get(field + '_Type') else cls.factor_types[field]
        return utils.Factor(next(x for x in types if utils.check_tuple_name(
            utils.Factor(x, rate)) in price_factors), rate)

    @staticmethod
    def reparam(x):
        """Maps the fitted vector x to (Omega, Alpha, Beta, Gamma_Star, H0).

        STATIONARITY IS ENFORCED BY CONSTRUCTION, not by a penalty: the optimizer fits the
        persistence psi = Beta + Alpha*Gamma_Star^2 itself (a plain box bound psi <= 1-1e-6) and
        splits it between the two channels with a leverage share l in [0, 1]. Omega and H0 are
        fitted in logs so they stay positive and so their scale (~1e-6) doesn't wreck the line
        search against Gamma_Star (~1e3, hence the /1000).
        """
        psi, lev, gamma = x[1], x[2], x[3] * 1000.0
        return torch.exp(x[0]), lev * psi / gamma ** 2, psi * (1.0 - lev), gamma, torch.exp(x[4])

    @staticmethod
    def unreparam(omega, alpha, beta, gamma, h0):
        """Inverse of reparam (used to warm start off an existing price factor)."""
        psi = beta + alpha * gamma ** 2
        return np.array([np.log(omega), psi, alpha * gamma ** 2 / psi, gamma / 1000.0, np.log(h0)])

    @classmethod
    def moneyness(cls, strike, spot, forward, vol_surface, use_forward, invert_moneyness):
        """The moneyness coordinate to look the vol surface up at.

        There are FIVE conventions in this framework and they are dispatched off the surface's
        SubType, so this DELEGATES to pricing.calc_moneyness - the same function every option deal
        uses - rather than reimplementing the dispatch. calc_moneyness only reads the SubType out
        of deal_data, so a minimal Deal_data carrying this surface's SubType is all it needs.
        """
        deal_data = utils.DealDataType(
            Instrument=None, Time_dep=None, Calc_res=None,
            Factor_dep={'Volatility': [(None, None, vol_surface.get_subtype())]})
        return float(pricing.calc_moneyness(
            *[torch.tensor(float(x), dtype=cls.prec) for x in (strike, spot, forward)],
            deal_data, use_forward, invert_moneyness))

    @staticmethod
    def price(spot, strike, is_call, units, p, n, h0, panels, yield_discount=1.0):
        """Heston-Nandi European option value - puts by put-call parity off the call.

        ``p.r`` is the per step COST OF CARRY r-q (so the simulated spot has the right forward) and
        ``yield_discount`` = exp(-q*t) converts the resulting value back to a discounting at r:
        the internal price is exp(-(r-q)t)[F P1 - K P2], the value is exp(-rt)[F P1 - K P2]. Parity
        survives the rescale, so puts are still call - S + K exp(-(r-q)n) times the same factor."""
        call = hn_garch.hn_call(spot, strike, p, n, h0, panels=panels)
        return units * yield_discount * (call - (1.0 - is_call) * (spot - strike * torch.exp(-p.r * n)))

    def calc_error(self, x, groups, spot, panels, scale):
        """Weighted squared premium error and its exact gradient (autograd).

        ``scale`` is the mean squared quoted premium: L-BFGS-B's gradient tolerance is ABSOLUTE, so
        without it the fit would stop early on a low priced underlying (an fx rate) and late on a
        high priced one. Dividing by a constant leaves the relative Weights untouched."""
        x_t = torch.tensor(x, device=self.device, dtype=self.prec, requires_grad=True)
        omega, alpha, beta, gamma, h0 = self.reparam(x_t)
        error = 0.0
        for n, b, q, strike, is_call, units, weight, premium in groups:
            fitted = self.price(spot, strike, is_call, units,
                                hn_garch.HNParams(omega, alpha, beta, gamma, b), n, h0, panels, q)
            error = error + (weight * (premium - fitted) ** 2).sum() / scale
        error.backward()
        return float(error.detach()), x_t.grad.cpu().numpy()

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Calibrates the risk neutral Heston-Nandi GARCH(1,1) parameters to a set of European options on any
        spot underlying and writes them out as a HestonNandiModelParameters price factor.
        '''

        def tensor(x):
            return torch.tensor(x, device=self.device, dtype=self.prec)

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == 'HestonNandiModelPrices':
                instrument = implied_params['instrument']

                # resolve the underlying spot, its vol surface, the discount curve and any yield
                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(
                        self.resolve(instrument, 'Volatility', price_factors), price_factors, factor_interp)
                    vol_surface.delta = sys_params.get('Volatility_Delta', 0.0)
                    underlying = riskfactors.construct_factor(
                        self.resolve(instrument, 'Underlying', price_factors), price_factors, factor_interp)
                    discount = riskfactors.construct_factor(
                        self.resolve(instrument, 'Discount_Rate', price_factors), price_factors, factor_interp)
                    yield_factor = self.resolve(instrument, 'Yield', price_factors)
                    carry = riskfactors.construct_factor(
                        yield_factor, price_factors, factor_interp) if yield_factor else None
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                spot = float(underlying.current_value()[0])
                quote_type = instrument['Quote_Type']
                steps_per_year = instrument.get('Steps_Per_Year', 252.0)
                panels = instrument.get('Quadrature_Panels', 64)
                use_forward = instrument.get('Use_Forward') == 'Yes'
                invert_moneyness = instrument.get('Invert_Moneyness') == 'Yes'

                # a mis-looked-up vol would produce a wrong-but-converged calibration - the worst
                # outcome - so refuse the surface rather than guess at its convention
                subtype = vol_surface.get_subtype()
                if quote_type == 'Implied_Volatility' and subtype[0] not in self.tabular_surfaces:
                    logging.error(
                        'Cannot bootstrap {0} - volatility {1} has Surface_Type {2} (Moneyness_Rule {3}); '
                        'only {4} surfaces can be queried at a strike. Quote premiums directly '
                        '(Quote_Type Premium) instead'.format(
                            market_price, instrument['Volatility'], subtype[0], subtype[1],
                            '/'.join(self.tabular_surfaces)))
                    continue

                # need to loop over this and create some market prices - group by expiry so that all
                # the strikes of one expiry share a single characteristic function recursion
                expiries = {}
                for option in instrument['European_Options']:
                    t = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
                    r = float(discount.current_value(t))
                    q = float(carry.current_value(t)) if carry is not None else 0.0
                    forward = spot * np.exp((r - q) * t)
                    sign = 1.0 if option['Option_Type'] == 'Call' else -1.0
                    option['Strike'] = forward if not option['Strike'] else option['Strike']
                    option['r'] = r
                    option['q'] = q
                    option['T'] = t
                    # the number of GARCH steps to expiry - the carry is spread over them so that
                    # exp(-b_step*n) is exactly exp(-(r-q)*t)
                    option['n'] = max(int(round(t * steps_per_year)), 1)
                    if quote_type == 'Implied_Volatility':
                        moneyness = self.moneyness(
                            option['Strike'], spot, forward, vol_surface, use_forward, invert_moneyness)
                        sigma = vol_surface.current_value([[moneyness, t]])[0] if not option[
                            'Quoted_Market_Value'] else option['Quoted_Market_Value']
                        sigma += vol_surface.delta
                        option['Moneyness'] = moneyness
                        option['Premium'] = utils.black_european_option_price(
                            forward, option['Strike'], r, sigma, t, option['Units'], sign)
                    elif quote_type == 'Premium':
                        option['Premium'] = option['Units'] * option['Quoted_Market_Value']
                        # back out the Black vol of the quote (seeds the fit and the diagnostics)
                        call = option['Quoted_Market_Value'] + (0.0 if sign > 0 else
                                                                forward - option['Strike']) * np.exp(-r * t)
                        sigma = np.sqrt(hn_garch.bs_implied_total_var(
                            call, spot * np.exp(-q * t), option['Strike'], r * t, 1) / t)
                    else:
                        logging.error('quote_type {} not supported yet'.format(quote_type))
                        continue
                    option['sigma'] = sigma
                    expiries.setdefault(option['n'], []).append(option)

                groups = [(n, tensor((opts[0]['r'] - opts[0]['q']) * opts[0]['T'] / n),
                            tensor(np.exp(-opts[0]['q'] * opts[0]['T'])),
                            tensor([x['Strike'] for x in opts]),
                            tensor([1.0 if x['Option_Type'] == 'Call' else 0.0 for x in opts]),
                            tensor([x['Units'] for x in opts]),
                            tensor([x['Weight'] for x in opts]),
                            tensor([x['Premium'] for x in opts])) for n, opts in expiries.items()]

                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                param_name = utils.check_tuple_name(price_param)
                if param_name in price_factors:
                    # warm start off the previous fit
                    old = price_factors[param_name]
                    x0 = np.clip(self.unreparam(old['Omega'], old['Alpha'], old['Beta'],
                                                old['Gamma_Star'], old['H0']), *np.array(self.bounds).T)
                else:
                    var = np.mean([x['sigma'] for opts in expiries.values()
                                   for x in opts]) ** 2 / steps_per_year
                    x0 = np.array([np.log(0.1 * var), 0.9, 0.5, 0.1, np.log(var)])

                scale = np.mean([x['Premium'] ** 2 for opts in expiries.values() for x in opts])
                result = scipy.optimize.minimize(
                    self.calc_error, x0, args=(groups, spot, panels, scale), jac=True,
                    method='L-BFGS-B', bounds=self.bounds,
                    # the default ftol/gtol are calibrated for an O(1e2) objective - the normalised
                    # one starts at O(1) and a good fit is O(1e-12), so let it run to that
                    options={'ftol': 1e-15, 'gtol': 1e-12})

                omega, alpha, beta, gamma, h0 = [
                    float(x) for x in self.reparam(tensor(result.x))]
                fitted_params = hn_garch.HNParams(omega, alpha, beta, gamma)

                # log the results
                with torch.no_grad():
                    for n, b, q, strike, is_call, units, weight, premium in groups:
                        p = hn_garch.HNParams(*[tensor(x) for x in (omega, alpha, beta, gamma)], b)
                        fitted = self.price(spot, strike, is_call, units, p, n, tensor(h0), panels, q)
                        for option, fitted_premium in zip(expiries[n], fitted.cpu().numpy()):
                            vol = hn_garch.hn_implied_vol(
                                spot, option['Strike'], p, n, tensor(h0), steps_per_year, panels=panels)
                            logging.info(
                                'Underlying {} strike {}, expiry {}, steps {}, vol {}, c_vol {}, premium {}, '
                                'c_premium {}, err {}'.format(
                                    instrument['Underlying'], option['Strike'], option['Expiry_Date'], n,
                                    option['sigma'], vol, option['Premium'], fitted_premium,
                                    (fitted_premium - option['Premium']) ** 2))

                logging.info(
                    'Underlying {} Heston-Nandi Omega {}, Alpha {}, Beta {}, Gamma_Star {}, H0 {}, '
                    'persistence {}, long run vol {}, sse {} ({})'.format(
                        instrument['Underlying'], omega, alpha, beta, gamma, h0,
                        fitted_params.persistence, fitted_params.ann_vol(steps_per_year),
                        result.fun, result.message))

                price_factors[param_name] = {
                    'Property_Aliases': None,
                    'Omega': omega,
                    'Alpha': alpha,
                    'Beta': beta,
                    'Gamma_Star': gamma,
                    'H0': h0}


class GBMAssetPriceTSModelParameters(object):
    documentation = (
        'Fx And Equity',
        ['For Risk Neutral simulation, an integrated curve $\\bar{\\sigma}(t)$ needs to be specified and is',
         'interpreted as the average volatility at time $t$. This is typically obtained from the corresponding',
         'ATM volatility. This is then used to construct a new variance curve $V(t)$ which is defined as',
         '$V(0)=0, V(t_i)=\\bar{\\sigma}(t_i)^2 t_i$ and $V(t)=\\bar{\\sigma}(t_n)^2 t$ for $t>t_n$ where',
         '$t_1,...,t_n$ are discrete points on the ATM volatility curve.',
         '',
         'Points on the curve that imply a decrease in variance (i.e. $V(t_i)<V(t_{i-1})$) are adjusted to',
         '$V(t_i)=\\bar\\sigma(t_i)^2t_i=V(t_{i-1})$. This curve is then used to construct *instantaneous* curves',
         'that are then input to the corresponding stochastic process.',
         '',
         'The relationship between integrated $F(t)=\\int_0^t f_1(s)f_2(s)ds$ and instantaneous curves $f_1, f_2$',
         'where the instantaneous curves are defined on discrete points $P={t_0,t_1,..,t_n}$ with $t_0=0$ is defined',
         'on $P$ by Simpson\'s rule:',
         '',
         '$$F(t_i)=F(t_{i-1})+\\frac{t_i-t_{i-1}}{6}\\Big(f(t_i)+4f(\\frac{t_i+t_{i-1}}{2})+f(t_i)\\Big)$$',
         '',
         'and $f(t)=f_1(t)f_2(t)$. Integrated curves are flat extrapolated and linearly interpolated.'
         ]
    )

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Checks for Declining variance in the ATM vols of the relevant price factor and corrects accordingly.
        '''
        eq_vols = {}
        fx_vols = {}
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == 'GBMAssetPriceTSModelPrices':
                # get the vol surface
                implied_param = utils.check_rate_name(implied_params['instrument']['Asset_Price_Volatility'])
                if 'FXVol.' + implied_params['instrument']['Asset_Price_Volatility'] in price_factors:
                    vol_factor = utils.Factor('FXVol', utils.check_rate_name(
                        implied_params['instrument']['Asset_Price_Volatility']))
                    is_fx = True
                else:
                    vol_factor = utils.Factor('EquityPriceVol', utils.check_rate_name(
                        implied_params['instrument']['Asset_Price_Volatility']))
                    is_fx = False

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                mn_ix = np.searchsorted(vol_surface.moneyness, 1.0)
                atm_vol = [np.interp(1, vol_surface.moneyness[mn_ix - 1:mn_ix + 1], y) for y in
                           vol_surface.get_vols()[:, mn_ix - 1:mn_ix + 1]]

                # store the output
                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                model_param = utils.Factor('GBMAssetPriceTSModelImplied', market_factor.name)

                if vol_surface.expiry.size > 1:
                    dt = np.diff(np.append(0, vol_surface.expiry))
                    var = vol_surface.expiry * np.array(atm_vol) ** 2
                    sig = atm_vol[:1]
                    vol = atm_vol[:1]
                    var_tm1 = var[0]
                    fixed_variance = False

                    for var_t, delta_t, t_i in zip(var[1:], dt[1:] / 3.0, vol_surface.expiry[1:]):
                        M = var_tm1 + delta_t * (sig[-1] ** 2)
                        if var_t < M:
                            fixed_variance = True
                            var_t = M

                        a = delta_t
                        b = sig[-1] * delta_t
                        c = M - var_t

                        sig.append((-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a))
                        vol.append(np.sqrt(var_t / t_i))
                        var_tm1 = var_t

                    if fixed_variance:
                        logging.warning('Fixed declining variance for {0}'.format(market_price))
                else:
                    vol = atm_vol

                if is_fx:
                    fx_vols[rate[-1]] = [utils.Curve(['Integrated'], list(zip(vol_surface.expiry, vol))), implied_param]
                    price_factors[utils.check_tuple_name(price_param)] = {
                        'Property_Aliases': None,
                        'Vol': fx_vols[rate[-1]][0],
                        'Quanto_FX_Volatility': None,
                        'Quanto_FX_Correlation': 0.0}
                    price_models[utils.check_tuple_name(model_param)] = {'Risk_Premium': None}
                else:
                    quanto_fx_corr = price_factors.get(
                        'Correlation.EquityPrice.{}.{}/FxRate.{}.{}'.format(
                            rate[-1], implied_param[-1], *sorted([sys_params['Base_Currency'], implied_param[-1]])),
                        {'Value': 0.0})['Value']
                    price_factors[utils.check_tuple_name(price_param)] = {
                        'Property_Aliases': None,
                        'Vol': utils.Curve(['Integrated'], list(zip(vol_surface.expiry, vol))),
                        'Quanto_FX_Volatility': None,
                        'Quanto_FX_Correlation': quanto_fx_corr}
                    price_models[utils.check_tuple_name(model_param)] = {'Risk_Premium': None}
                    # store this for later quanto correction
                    eq_vols[rate[-1]] = [utils.check_tuple_name(price_param), implied_param[-1]]


class RiskNeutralInterestRateModel(object):
    def __init__(self, param, device, dtype):
        self.param = param
        self.num_batches = 1
        self.batch_size = 8192
        self.device = device
        self.prec = dtype

    def calc_loss_on_ir_curve(self, implied_params, base_date, time_grid, process,
                              implied_obj, ir_factor, vol_surface, resid=lambda x: x * x, jac=False):

        def loss(implied_var):
            # first, reset the shared_mem
            shared_mem.reset(self.num_batches, numfactors, time_grid)
            # now set up the calc
            process.precalculate(base_date, time_grid, stoch_var, shared_mem, 0, implied_tensor=implied_var)
            tensor_swaptions = {}
            # needed to interpolate the zero curve
            delta_scen_t = np.diff(time_grid.scen_time_grid).reshape(-1, 1)

            for batch_index in range(self.num_batches):
                # load up the batch
                shared_mem.batch_index = batch_index
                # simulate the price factor - only need the full curve at the mtm time points
                shared_mem.t_Scenario_Buffer = process.generate(shared_mem)
                # get the discount factors
                Dfs = utils.calc_time_grid_curve_rate(
                    curve_index_reduced, time_grid.calc_time_grid(time_grid.scen_time_grid[:-1]),
                    shared_mem)
                # get the index in the deflation factor just prior to the given grid
                deflation = Dfs.reduce_deflate(delta_scen_t, time_grid.mtm_time_grid, shared_mem)
                # go over the instrument definitions and build the calibration
                for swaption_name, market_data in market_swaps.items():
                    expiry = market_data.deal_data.Time_dep.mtm_time_grid[
                        market_data.deal_data.Time_dep.deal_time_grid[0]]
                    DtT = deflation[expiry]
                    par_swap = pricing.pv_float_cashflow_list(
                        shared_mem, time_grid, market_data.deal_data,
                        pricing.pricer_float_cashflows, settle_cash=False)
                    sum_swaption = torch.sum(torch.relu(DtT * par_swap))
                    if swaption_name in tensor_swaptions:
                        tensor_swaptions[swaption_name] += sum_swaption
                    else:
                        tensor_swaptions[swaption_name] = sum_swaption

            calibrated_swaptions = {k: v / (self.batch_size * self.num_batches) for k, v in tensor_swaptions.items()}
            errors = {k: swap.weight * resid(100.0 * (swap.price / calibrated_swaptions[k] - 1.0))
                      for k, swap in market_swaps.items()}
            return calibrated_swaptions, errors

        # set up the stochastic factors
        stochastic_factors = {ir_factor: process}
        # calculate a reverse lookup for the tenors and store the daycount code
        all_tenors = utils.update_tenors(base_date, stochastic_factors)
        # calculate the curve indices
        index_keys = {'full': utils.Factor(ir_factor.type, ir_factor.name + ('full',)),
                      'reduced': utils.Factor(ir_factor.type, ir_factor.name + ('reduced',))}
        # calculate the tenor curve index
        c_index = instruments.calc_factor_index(ir_factor, {}, stochastic_factors, all_tenors)
        # now edit the curve indices with the correct names - one reduced, one full
        curve_index = [(c_index[utils.FACTOR_INDEX_Stoch], index_keys['full']) + c_index[2:]]
        curve_index_reduced = [(c_index[utils.FACTOR_INDEX_Stoch], index_keys['reduced']) + c_index[2:]]
        # calc the market swap rates and instrument_definitions    
        market_swaps, benchmarks = create_market_swaps(
            base_date, time_grid, curve_index, vol_surface, process.factor,
            implied_params['instrument']['Instrument_Definitions'], ir_factor.name)
        # number of random factors to use
        numfactors = process.num_factors()
        # set up a common context - we leave out the random numbers and pass it in explicitly below
        shared_mem = RiskNeutralInterestRate_State(index_keys, self.batch_size, self.device, self.prec)
        # set up the variables
        implied_var = {}
        stoch_var = torch.tensor(
            process.factor.current_value(), device=self.device, dtype=self.prec, requires_grad=jac)

        for param_name, param_value in implied_obj.current_value(include_quanto=jac).items():
            implied_var[param_name] = torch.tensor(
                param_value, dtype=self.prec, device=self.device, requires_grad=True)

        if jac:
            return stoch_var, implied_var, loss
        else:
            return implied_var, loss, market_swaps, benchmarks

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        base_date = sys_params['Base_Date']
        base_currency = sys_params['Base_Currency']
        master_curve_list = sys_params.get('Master_Curves')

        if sys_params.get('Swaption_Premiums') is not None:
            swaption_premiums = pd.read_csv(sys_params['Swaption_Premiums'], index_col=0)
            ATM_Premiums = swaption_premiums[swaption_premiums['Strike'] == 'ATM']
        else:
            ATM_Premiums = None

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == self.market_factor_type:
                # fetch the factors
                ir_factor = utils.Factor('InterestRate', rate[1:])
                vol_factor = utils.Factor('InterestYieldVol', utils.check_rate_name(
                    implied_params['instrument']['Swaption_Volatility']))

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    swaptionvol = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                    swaptionvol.delta = sys_params.get('Volatility_Delta', 0.0)
                    ir_curve = riskfactors.construct_factor(ir_factor, price_factors, factor_interp)
                    swaptionvol.set_premiums(ATM_Premiums, ir_curve.get_currency())
                except KeyError as k:
                    logging.warning('Missing price factor {} - Unable to bootstrap {}'.format(k.args, market_price))
                    continue
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                if master_curve_list and master_curve_list.get(ir_curve.get_currency()[0]) != rate[1]:
                    logging.warning('curve is not Risk Free {} - skipping and will reassign later'.format(market_price))
                    continue

                # set of dates for the calibration
                mtm_dates = set(
                    [base_date + x['Start'] for x in implied_params['instrument']['Instrument_Definitions']])

                # grab the implied process
                implied_obj, process, vol_tenors = self.implied_process(
                    base_currency, price_factors, price_models, ir_curve, rate)

                # set up the time grid
                time_grid = utils.TimeGrid(mtm_dates, mtm_dates, mtm_dates)
                # add a delta of 10 days to the time_grid_years (without changing the scenario grid
                # this is needed for stochastically deflating the exposure later on
                time_grid.set_base_date(base_date, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))

                # calculate the error
                loss_fn, optimizers, implied_var, market_swaptions, benchmarks = self.calc_loss(
                    implied_params, base_date, time_grid, process, implied_obj, ir_factor, swaptionvol)

                if debug is not None:
                    debug.deals['Deals']['Children'] = [{'instrument': x} for x in benchmarks]
                    try:
                        debug.write_trade_file(market_factor.name[0] + '.aap')
                    except Exception:
                        logging.error('Could not write output file {}'.format(market_factor.name[0] + '.aap'))

                # check the time
                time_now = time.monotonic()
                calibrated_swaptions, errors = loss_fn(implied_var)
                batch_loss = torch.stack(list(errors.values())).sum().cpu().detach().numpy()
                vars = {k: v.cpu().detach().numpy() for k, v in implied_var.items()}
                # initialize the soln with the current values
                soln = (batch_loss, vars)
                logging.info('{} - Batch loss {}'.format(market_factor.name[0], batch_loss))
                for k, v in sorted(vars.items()):
                    logging.info('{} - {}'.format(k, v))

                for k, v in sorted(calibrated_swaptions.items()):
                    value = v.cpu().detach().numpy()
                    price = market_swaptions[k].price
                    logging.debug('{},market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                        k, price, value, 100.0 * (price - value) / price))

                # minimize
                result = None
                num_optimizers = len(optimizers)
                for op_loop in range(num_optimizers):
                    optim = optimizers[op_loop % num_optimizers]
                    x0 = result['x'] if result is not None else optim[1]
                    if optim[0] == 'basin':
                        result = scipy.optimize.basinhopping(
                            optim[2], x0=x0, take_step=optim[3], accept_test=optim[4], T=5.0, niter=50,
                            minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "bounds": optim[5]})
                        batch_loss = float(optim[2](result['x'])[0])
                    elif optim[0] == 'leastsq':
                        result = scipy.optimize.least_squares(
                            optim[2], x0=x0, jac=optim[3], bounds=optim[4])
                        batch_loss = optim[2](result['x']).sum()

                    if batch_loss < soln[0] and process.params_ok:
                        sim_swaptions, errors = loss_fn(implied_var)
                        vars = {k: v.cpu().detach().numpy() for k, v in implied_var.items()}
                        soln = (batch_loss, vars)
                        logging.info('{} - run {} - Batch loss {}'.format(
                            market_factor.name[0], op_loop, batch_loss))
                        for k, v in sorted(vars.items()):
                            logging.info('{} - {}'.format(k, v))
                        for k, v in sim_swaptions.items():
                            value = v.cpu().detach().numpy()
                            price = market_swaptions[k].price
                            logging.info('{},market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                                k, price, value, 100.0 * (price - value) / price))

                # save this
                final_implied_obj = self.save_params(soln[1], price_factors, implied_obj, rate)

                # record the time
                logging.info('This took {} seconds.'.format(time.monotonic() - time_now))


class PCAMixedFactorModelParameters(RiskNeutralInterestRateModel):
    def __init__(self, param, device, dtype):
        super(PCAMixedFactorModelParameters, self).__init__(param, device, dtype)
        self.market_factor_type = 'HullWhite2FactorInterestRateModelPrices'

    def calc_sample(self, time_grid, numfactors=0):
        if numfactors != 3 or self.sample is None:
            self.sobol = torch.quasirandom.SobolEngine(
                dimension=time_grid.time_grid_years.size * numfactors, scramble=True, seed=1234)
            sample = torch.distributions.Normal(0, 1).icdf(
                self.sobol.draw(self.batch_size * self.num_batches)).reshape(
                self.num_batches, self.batch_size, -1)
            self.sample = sample.transpose(1, 2).reshape(
                self.num_batches, numfactors, -1, self.batch_size).to(self.one.device)

        return self.sample

    def calc_loss(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface):
        # get the swaption error and market values
        implied_var, error, calibrated_swaptions, market_swaptions, benchmarks = self.calc_loss_on_ir_curve(
            implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface)

        losses = list(error.values())
        loss = torch.sum(losses)

        loss_with_penatalties = loss + tf.nn.moments(
            implied_var['Yield_Volatility'][1:] - implied_var['Yield_Volatility'][:-1], axes=[0])[1]

        var_to_bounds = {implied_var['Yield_Volatility']: (1e-3, 0.6),
                         implied_var['Reversion_Speed']: (1e-2, 1.8)}

        # optimizer = ScipyBasinOptimizerInterface(loss, var_to_bounds=var_to_bounds)
        optimizer = None

        return loss, optimizer, implied_var, calibrated_swaptions, market_swaptions, benchmarks

    @staticmethod
    def implied_process(base_currency, price_factors, price_models, ir_curve, rate):
        # need to create a process and params as variables to pass to tf
        price_model = utils.Factor('PCAInterestRateModel', rate[1:])
        param_name = utils.check_tuple_name(price_model)
        vol_tenors = np.array([0, 1.5, 3, 6, 12, 24, 48, 84, 120]) / 12.0

        if param_name in price_factors:
            param = price_models[param_name]

            # construct an initial guess - need to read from params
            implied_obj = riskfactors.PCAMixedFactorModelParameters(
                {'Quanto_FX_Volatility': None,
                 'Reversion_Speed': param['Reversion_Speed'],
                 'Yield_Volatility': param['Yield_Volatility']})
        else:
            # need to do a historical calibration
            implied_obj = riskfactors.PCAMixedFactorModelParameters(
                {'Quanto_FX_Volatility': None,
                 'Reversion_Speed': 1.0,
                 'Yield_Volatility': utils.Curve([], list(zip(vol_tenors, [0.1] * vol_tenors.size)))})

        process = stochasticprocess.construct_process(
            price_model.type, ir_curve, price_models[utils.check_tuple_name(price_model)], implied_obj)

        return implied_obj, process, vol_tenors

    @staticmethod
    def save_params(vars, price_factors, implied_obj, rate):
        # construct an initial guess - need to read from params
        param_name = utils.check_tuple_name(
            utils.Factor(type='PCAInterestRateModel', name=rate[1:]))
        param = price_factors[param_name]
        vol_tenors = implied_obj.get_vol_tenors()

        param['Reversion_Speed'] = float(vars['Reversion_Speed'][0])
        param['Yield_Volatility'].array = np.dstack((vol_tenors, vars['Yield_Volatility']))[0]

        # assign this back in case this is a dict-proxy object
        price_factors[param_name] = param


class HullWhite2FactorModelParameters(RiskNeutralInterestRateModel):
    documentation = (
        'Interest Rates',
        ['A set of parameters $\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho$ are estimated from ATM',
         'swaption volatilities. Swaption volatilities are preferred to caplets to better estimate $\\rho$.'
         'Although assuming that $\\sigma_1, \\sigma_2$ are constant makes the calibration of this model',
         'considerably easier, in general, $\\sigma_1, \\sigma_2$ should be allowed a piecewise linear term',
         'structure dependent on the underlying swaptions.',
         '',
         'For a set of $J$ ATM swaptions, we need to minimize:',
         '',
         '$$E=\\sum_{j\\in J} \\omega_j (V_j(\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho)-V_j)^2$$',
         '',
         'Where $V_j(\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho)$ is the price of the $j^{th}$ swaption',
         'under the model, $V_j$ is the market value of the $j^{th}$ swaption and $ \\omega_j$ is the corresponding',
         'weight. The market value is calculated using the standard pricing functions',
         '',
         'To find a good minimum of the model value, basin hopping as implemented [here](https://docs.scipy.org/doc\
/scipy/reference/generated/scipy.optimize.basinhopping.html) as well as',
         'least squares [optimization](https://docs.scipy.org/doc/scipy/reference/generated/\
scipy.optimize.leastsq.html) are used.',
         '',
         'The error $E$ is algorithmically differentiated and then solved via brute-force monte carlo',
         'using tensorflow and scipy.',
         '',
         'If the currency of the interest rate is not the same as the base currency, then a quanto correction needs',
         'to be made. Assume $C$ is the value of the interest rate/FX correlation price factor (can be estimated from',
         'historical data), then the FX rate follows:',
         '',
         '$$d(log X)(t)=(r_0(t)-r(t)-\\frac{1}{2}v(t)^2)dt+v(t)dW(t)$$',
         '',
         'with $r(t)$ the short rate and $r_0(t)$ the short rate in base currency. The short rate with a quanto',
         'correction is:',
         '',
         '$$dr(t)=r_T(0,t)dt+\\sum_{i=1}^2 (\\theta_i(t)-\\alpha_i x_i(t)- \\bar\\rho_i\\sigma_i v(t))dt+\\sigma_i dW_i(t)$$',
         '',
         'where $W_1(t),W_2(t)$ and $W(t)$ are standard Wiener processes under the rate currency\'s risk neutral measure',
         'and $r_T(t,T)$ is the partial derivative of the instantaneous forward rate r(t,T) with respect to the maturity ',
         'date $T$.'
         '',
         'Define:',
         '',
         '$$F(u,v)=\\frac{\\sigma_1u+\\sigma_2v}{\\sqrt{\\sigma_1^2+\\sigma_2^2+2\\rho\\sigma_1\\sigma_2}}$$',
         '',
         'Then $\\bar\\rho_1, \\bar\\rho_2$ are assigned:',
         '',
         '$$\\bar\\rho_1=F(1,\\rho)C$$',
         '',
         '$$\\bar\\rho_2=F(\\rho,1)C$$',
         ]
    )

    def __init__(self, param, device, dtype):
        super(HullWhite2FactorModelParameters, self).__init__(param, device, dtype)
        # HullWhite2FactorModelPrices
        # HullWhite2FactorInterestRateModelPrices
        self.market_factor_type = 'HullWhite2FactorModelPrices'
        self.sigma_bounds = (1e-5, 0.09)
        self.alpha_bounds = (-0.5, 2.4)
        self.corr_bounds = (-.95, 0.95)

    def calc_loss(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface):

        def split_param(x):
            corr = x[2:3]
            alpha = x[:2]
            sigmas = x[3:]
            return sigmas, alpha, corr

        def make_basin_callbacks(step, sigma_min_max, alpha_min_max, corr_min_max):
            def bounds_check(**kwargs):
                x = kwargs["x_new"]
                sigmas, alpha, corr = split_param(x)
                sigma_ok = (sigmas > sigma_min_max[0]).all() and (sigmas < sigma_min_max[1]).all()
                alpha_ok = (alpha > alpha_min_max[0]).all() and (alpha < alpha_min_max[1]).all()
                corre_ok = (corr > corr_min_max[0]).all() and (corr < corr_min_max[1]).all()
                return sigma_ok and alpha_ok and corre_ok and process.params_ok

            def basin_step(x):
                sigmas, alpha, corr = split_param(x)
                # update vars
                sigmas = (sigmas * np.exp(np.random.uniform(-step, step, sigmas.size))).clip(*sigma_min_max)
                alpha = (alpha * np.exp(np.random.uniform(-step, step, alpha.size))).clip(*alpha_min_max)
                corr = (corr + np.random.uniform(-step, step, corr.size)).clip(*corr_min_max)

                return np.concatenate((alpha, corr, sigmas))

            return bounds_check, basin_step

        def make_basin_hopping_loss(loss_fn, implied_vars, device, with_grad=False):
            # makes it possible to call the scipy basinhopper
            def basin_hopper(x):
                for tn_var, np_var in zip(implied_vars.values(), np.split(x, split_param)):
                    tn_var.grad = None
                    tn_var.data = torch.from_numpy(np_var).to(device)

                try:
                    _, error = loss_fn(implied_vars)
                except Exception as e:
                    print("Warning x ({}) - {}".format(x, e.args))
                    return 100.0 * sum(len_vars), [100.0 * sum(len_vars)] * sum(len_vars)
                else:
                    total_loss = torch.sum(torch.stack(list(error.values())))
                    if with_grad:
                        total_loss.backward()
                        grad = torch.cat([x.grad for x in implied_vars.values()]).cpu().detach().numpy()
                        return total_loss.cpu().detach().numpy(), grad
                    else:
                        return total_loss.cpu().detach().numpy()

            len_vars = [len(x) for x in implied_vars.values()]
            split_param = np.cumsum(len_vars[:-1])
            return basin_hopper

        def make_least_squares_loss(loss_fn, implied_vars, device):
            # makes it possible to call the scipy least squares algo
            def calc_loss(x):
                for tn_var, np_var in zip(implied_vars.values(), np.split(x, split_param)):
                    tn_var.grad = None
                    tn_var.data = torch.from_numpy(np_var).to(device)
                _, error = loss_fn(implied_vars)
                return torch.stack(list(error.values()))

            def jacobian(x):
                loss = calc_loss(x)
                # full jacobian - takes a second or so
                jac = torch.stack([torch.cat(torch.autograd.grad(
                    loss, list(implied_vars.values()), x, retain_graph=True))
                    for x in torch.eye(len(loss), device=device)])
                return jac.cpu().numpy()

            def least_squares(x):
                return calc_loss(x).cpu().detach().numpy()

            len_vars = [len(x) for x in implied_vars.values()]
            split_param = np.cumsum(len_vars[:-1])
            return least_squares, jacobian

        # get the swaption error and market values
        implied_var_dict, loss_fn, market_swaptions, benchmarks = self.calc_loss_on_ir_curve(
            implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface)

        bounds = []
        for k, v in implied_var_dict.items():
            if k.startswith('Alpha'):
                bounds.append([self.alpha_bounds])
            elif k == 'Correlation':
                bounds.append([self.corr_bounds])
            else:
                bounds.append([self.sigma_bounds] * len(v))

        var_to_bounds = np.vstack(bounds)
        bounds_ok, make_step = make_basin_callbacks(0.125, self.sigma_bounds, self.alpha_bounds, self.corr_bounds)

        basin_hopper_fn_grad = make_basin_hopping_loss(loss_fn, implied_var_dict, self.device, True)
        x0 = torch.cat(list(implied_var_dict.values())).cpu().detach().numpy()
        lsq_fn, jacobian = make_least_squares_loss(loss_fn, implied_var_dict, self.device)

        optimizers = [('basin', x0, basin_hopper_fn_grad, make_step, bounds_ok, var_to_bounds),
                      ('leastsq', x0, lsq_fn, jacobian, list(zip(*var_to_bounds)))]

        return loss_fn, optimizers, implied_var_dict, market_swaptions, benchmarks

    def implied_process(self, base_currency, price_factors, price_models, ir_curve, rate):
        vol_tenors = np.array([0, 1, 3, 6, 12, 24, 48, 72, 96, 120]) / 12.0
        # construct an initial guess - need to read from params
        param_name = utils.check_tuple_name(
            utils.Factor(type=self.__class__.__name__, name=rate[1:]))

        # check if we need a quanto fx vol
        fx_factor = utils.Factor('GBMAssetPriceTSModelParameters', ir_curve.get_currency())
        ir_factor = utils.Factor('InterestRate', ir_curve.get_currency())
        fx_factor_name = utils.check_tuple_name(fx_factor)
        ir_factor_name = utils.check_tuple_name(ir_factor)

        if fx_factor_name in price_factors:
            quanto_fx = price_factors[fx_factor_name]['Vol']
            curr_pair = sorted((base_currency,) + ir_curve.get_currency())
            correlation_name = 'Correlation.FxRate.{}/{}'.format('.'.join(curr_pair), ir_factor_name)
            # check if the quote is against the base currency
            sign = 1.0
            if curr_pair[0] == base_currency:
                sign = -1.0
                logging.info('Reversing Correlation as {} is quoted against the base currency'.format(correlation_name))
            # the correlation between fx and ir - needed to establish Quanto Correlation 1 and 2
            C = sign * price_factors.get(correlation_name, {'Value': 0.0})['Value']
        else:
            C = None
            quanto_fx = None

        if param_name in price_factors:
            param = price_factors[param_name]
            implied_obj = riskfactors.HullWhite2FactorModelParameters(
                {'Quanto_FX_Volatility': quanto_fx,
                 'short_rate_fx_correlation': C,
                 'Alpha_1': np.clip(param['Alpha_1'], *self.alpha_bounds),
                 'Alpha_2': np.clip(param['Alpha_2'], *self.alpha_bounds),
                 'Correlation': np.clip(param['Correlation'], *self.corr_bounds),
                 'Sigma_1': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_1'].array.T).clip(*self.sigma_bounds)))),
                 'Sigma_2': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_2'].array.T).clip(*self.sigma_bounds))))})
        else:
            implied_obj = riskfactors.HullWhite2FactorModelParameters(
                {'Quanto_FX_Volatility': quanto_fx,
                 'short_rate_fx_correlation': C,
                 'Alpha_1': 0.1, 'Alpha_2': 0.1, 'Correlation': 0.01,
                 'Sigma_1': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size))),
                 'Sigma_2': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size)))})

        # need to create a process and params as variables to pass to tf
        process = stochasticprocess.HullWhite2FactorImpliedInterestRateModel(
            ir_curve, {'Lambda_1': 0.0, 'Lambda_2': 0.0}, implied_obj)

        return implied_obj, process, vol_tenors

    def save_params(self, vars, price_factors, implied_obj, rate):
        param_name = utils.check_tuple_name(
            utils.Factor(type=self.__class__.__name__, name=rate[1:]))
        # grab the sigma tenors
        sig1_tenor, sig2_tenor = implied_obj.get_vol_tenors()
        # store the basic paramters
        param = {'Property_Aliases': None,
                 'Quanto_FX_Volatility': None,
                 'Alpha_1': float(vars['Alpha_1'][0]),
                 'Sigma_1': utils.Curve([], list(zip(sig1_tenor, vars['Sigma_1']))),
                 'Alpha_2': float(vars['Alpha_2'][0]),
                 'Sigma_2': utils.Curve([], list(zip(sig2_tenor, vars['Sigma_2']))),
                 'Correlation': float(vars['Correlation'][0])}

        # grab the quanto fx correlations
        quanto_fx1, quanto_fx2 = implied_obj.get_quanto_correlation(
            vars['Correlation'], [vars['Sigma_1'], vars['Sigma_2']])

        if quanto_fx1 is not None and quanto_fx2 is not None:
            param.update({
                'Quanto_FX_Volatility': implied_obj.param['Quanto_FX_Volatility'],
                'Quanto_FX_Correlation_1': quanto_fx1,
                'Quanto_FX_Correlation_2': quanto_fx2})

        price_factors[param_name] = param
        # return the final implied object
        return riskfactors.HullWhite2FactorModelParameters(param)


def construct_bootstrapper(btype, param, dtype=torch.float32):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return globals().get(btype)(param, device, dtype)
