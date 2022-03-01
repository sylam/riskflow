########################################################################
# Copyright (C)  Shuaib Osman (sosman@investec.co.za)
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
from collections import OrderedDict, namedtuple

# third party stuff
import numpy as np
import pandas as pd
import torch

# Internal modules
from . import utils, pricing, instruments, riskfactors, stochasticprocess

from scipy.stats import norm
import scipy.optimize

curve_jacobian_class = namedtuple('shared_mem', 't_Buffer t_Static_Buffer \
                                precision riskneutral simulation_batch')

market_swap_class = namedtuple('market_swap', 'deal_data price weight')
date_desc = {'years': 'Y', 'months': 'M', 'days': 'D'}
# date formatter
date_fmt = lambda x: ''.join(['{0}{1}'.format(v, date_desc[k]) for k, v in x.kwds.items()])


class RiskNeutralInterestRate_State(utils.Calculation_State):
    def __init__(self, batch_size, device, dtype, nomodel='Constant'):
        super(RiskNeutralInterestRate_State, self).__init__(
            None, torch.ones([1, 1], dtype=dtype, device=device), None, nomodel)
        # these are tensors
        self.t_PreCalc = {}
        self.t_random_batch = None
        self.batch_index = 0
        self.t_Scenario_Buffer = None
        # these are shared parameter states
        self.simulation_batch = batch_size

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
        cashflows.append(OrderedDict([
            ('Payment_Date', base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Pay_Day])),
            ('Notional', 1.0),
            ('Accrual_Start_Date', base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Start_Day])),
            ('Accrual_End_Date', base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_End_Day])),
            ('Accrual_Year_Fraction', cashflow[utils.CASHFLOW_INDEX_Year_Frac]),
            ('Fixed_Amount', cashflow[utils.CASHFLOW_INDEX_FixedAmt]),
            ('Resets', [[base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Reset_Day]),
                         base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Start_Day]),
                         base_date + pd.offsets.Day(reset[utils.RESET_INDEX_End_Day]),
                         reset[utils.RESET_INDEX_Accrual],
                         frequency, 'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]]),
            ('Margin', utils.Basis(0.0))
        ]))
    return cashflows


def normalize(sample):
    '''Simple function to ensure that the sample used for the monte carlo sim has mean 0 and var 1'''
    return (sample - sample.mean(axis=0)) / sample.std(axis=0)


def atm_swap(base_date, curve_factor, time_grid, effective, maturity, float_freq, daycount):
    float_pay_dates = instruments.generate_dates_backward(
        maturity, effective, float_freq)

    float_cash = utils.generate_float_cashflows(
        base_date, time_grid, float_pay_dates, 1.0, None, None,
        float_freq, pd.DateOffset(months=0), utils.get_day_count(daycount), 0.0)

    K, pvbp = float_cash.get_par_swap_rate(base_date, curve_factor)
    float_cash.set_fixed_amount(-K)
    return float_cash, K, pvbp


def atm_depo(base_date, curve_factor, time_grid, maturity, daycount):
    time_in_years = utils.get_day_count_accrual(
        base_date, (maturity - base_date).days, utils.get_day_count(daycount))

    fixed_pay_dates = instruments.generate_dates_backward(
        maturity, base_date, maturity - base_date)

    fixed_cash = utils.generate_fixed_cashflows(
        base_date, fixed_pay_dates, 1.0, None,
        utils.get_day_count(daycount), 0.0)

    fixed_cash.overwrite_rate(utils.CASHFLOW_INDEX_FixedAmt, 1.0)

    return fixed_cash


def create_market_swaps(base_date, time_grid, curve_index, vol_surface, curve_factor,
                        instrument_definitions, rate=None):
    # store these benchmark swap definitions if necessary
    benchmarks = []
    # store the benchmark instruments
    all_deals = {}
    # cater for shifted lognormal vols
    shift_parameter = vol_surface.BlackScholesDisplacedShiftValue / 100.0
    for instrument in instrument_definitions:
        # setup the instrument
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
                implied_vol = scipy.optimize.brentq(lambda v: pvbp * utils.black_european_option_price(
                    shifted_strike, shifted_strike, 0.0, v, expiry, 1.0, 1.0) - swaption_price, 0.01, vol+.5)
                swaption_price = pvbp * utils.black_european_option_price(
                    shifted_strike, shifted_strike, 0.0, implied_vol+vol_surface.delta, expiry, 1.0, 1.0)
        else:
            swaption_price = pvbp * utils.black_european_option_price(
                shifted_strike, shifted_strike, 0.0, vol+vol_surface.delta, expiry, 1.0, 1.0)

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


class InterestRateJacobian(object):
    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param
        self.batch_size = 1

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):

        def fwd_gradients(ys, xs, d_xs):
            """ Forward-mode pushforward analogous to the pullback defined by tf.gradients.
                With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
                the vector being pushed forward."""
            v = tf.placeholder_with_default(
                np.ones([x.value for x in ys.get_shape()], dtype=self.prec), shape=ys.get_shape())
            g = tf.concat(tf.gradients(ys, xs, grad_ys=v), axis=0)
            return tf.gradients(g, v, grad_ys=d_xs)

        base_date = sys_params['Base_Date']
        # do all of this at time 0        
        time_grid = utils.TimeGrid({base_date}, {base_date}, {base_date})
        time_grid.set_base_date(base_date)
        # now prepare for inverse bootstrap
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == 'InterestRatePrices':
                # get the currency 
                curr = implied_params['instrument']['Currency']
                graph = tf.Graph()
                with graph.as_default():
                    # this shouldn't fail - if it does, need to log it and move on
                    try:
                        ir_factor = utils.Factor('InterestRate', rate[1:])
                        ir_curve = riskfactors.construct_factor(ir_factor, price_factors, factor_interp)
                    except Exception:
                        logging.warning('Unable to calculate the Jacobian for {0} - skipping'.format(market_price))
                        continue

                    # calculate a reverse lookup for the tenors and store the daycount code
                    all_tenors = utils.update_tenors(base_date, {ir_factor: ir_curve})
                    # calculate the curve index - need to clean this up - TODO!!!
                    curve_index = [instruments.calc_factor_index(ir_factor, {ir_factor: 0}, {}, all_tenors)]
                    benchmarks = OrderedDict()
                    # state for all calcs
                    shared_mem = curve_jacobian_class(
                        t_Buffer={},
                        t_Static_Buffer=[tf.constant(ir_curve.current_value(), dtype=self.prec)],
                        precision=self.prec,
                        riskneutral=False,
                        simulation_batch=self.batch_size)

                    for child in implied_params['Children']:
                        if child['quote']['DealType'] == 'DepositDeal':
                            maturity = base_date + child['instrument']['Payment_Frequency']
                            fixed_cash = atm_depo(base_date, ir_curve, time_grid, maturity,
                                                  child['instrument']['Accrual_Day_Count'])
                            deal_data = utils.DealDataType(
                                Instrument=None, Factor_dep={'Cashflows': fixed_cash, 'Discount': curve_index},
                                Time_dep=utils.DealTimeDependencies(time_grid.mtm_time_grid, [0]), Calc_res=None)
                            fmt_fn = str if isinstance(child['quote']['Descriptor'], str) else date_fmt
                            benchmarks['Deposit_' + fmt_fn(child['quote']['Descriptor'])] = pricing.pv_fixed_cashflows(
                                shared_mem, time_grid, deal_data, settle_cash=False)
                        elif child['quote']['DealType'] == 'FRADeal':
                            float_cash, K, pvbp = atm_swap(
                                base_date, ir_curve, time_grid,
                                base_date + pd.DateOffset(months=child['quote']['Descriptor'].data[0]),
                                base_date + pd.DateOffset(months=child['quote']['Descriptor'].data[1]),
                                pd.DateOffset(months=np.diff(child['quote']['Descriptor'].data)[0]),
                                child['instrument']['Day_Count'])
                            deal_data = utils.DealDataType(
                                Instrument=None, Factor_dep={'Cashflows': float_cash, 'Forward': curve_index,
                                                             'Discount': curve_index, 'CompoundingMethod': 'None'},
                                Time_dep=utils.DealTimeDependencies(time_grid.mtm_time_grid, [0]), Calc_res=None)
                            benchmarks['FRA_' + str(child['quote']['Descriptor'])] = pricing.pv_float_cashflow_list(
                                shared_mem, time_grid, deal_data, pricing.pricer_float_cashflows, settle_cash=False)
                        elif child['quote']['DealType'] == 'SwapInterestDeal':
                            float_cash, K, pvbp = atm_swap(
                                base_date, ir_curve, time_grid,
                                base_date,
                                base_date + child['quote']['Descriptor'],
                                child['instrument']['Pay_Frequency'],
                                child['instrument']['Pay_Day_Count'])
                            deal_data = utils.DealDataType(
                                Instrument=None, Factor_dep={'Cashflows': float_cash, 'Forward': curve_index,
                                                             'Discount': curve_index, 'CompoundingMethod': 'None'},
                                Time_dep=utils.DealTimeDependencies(time_grid.mtm_time_grid, [0]), Calc_res=None)
                            benchmarks[
                                'Swap_' + date_fmt(child['quote']['Descriptor'])] = pricing.pv_float_cashflow_list(
                                shared_mem, time_grid, deal_data, pricing.pricer_float_cashflows, settle_cash=False)
                        else:
                            raise Exception('quote type not supported')

                    n = ir_curve.tenors.shape[0]
                    dummy_jac_vec = tf.placeholder(self.prec, n)
                    bench = tf.squeeze(tf.concat(list(benchmarks.values()), axis=0), axis=1)
                    jvp = fwd_gradients(bench, shared_mem.t_Static_Buffer, dummy_jac_vec)

                config = tf.ConfigProto(allow_soft_placement=True)

                with tf.Session(graph=graph, config=config) as sess:
                    I = np.eye(n)
                    # store the output
                    price_param = utils.Factor('InterestRateJacobian', market_factor.name)
                    jac = OrderedDict()
                    j = np.array([sess.run(jvp, {dummy_jac_vec: I[i]})[0] for i in range(n)]).T
                    for index, benchmark_name in enumerate(benchmarks.keys()):
                        non_zero = np.where(j[index] != 0.0)
                        jac[benchmark_name] = utils.Curve([], list(zip(ir_curve.tenors[non_zero], j[index][non_zero])))
                    # jac = pd.DataFrame(j, index=list(benchmarks.keys()), columns=ir_curve.tenors)

                price_factors[utils.check_tuple_name(price_param)] = jac


class GBMAssetPriceTSModelParameters(object):
    documentation = (
        'FX and Equity',
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
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == 'GBMAssetPriceTSModelPrices':
                # get the vol surface
                vol_factor = utils.Factor('FXVol', utils.check_rate_name(
                    implied_params['instrument']['Asset_Price_Volatility']))
                # this shouldn't fail - if it does, need to log it and move on
                try:
                    fxvol = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                mn_ix = np.searchsorted(fxvol.moneyness, 1.0)
                atm_vol = [np.interp(1, fxvol.moneyness[mn_ix - 1:mn_ix + 1], y) for y in
                           fxvol.get_vols()[:, mn_ix - 1:mn_ix + 1]]

                # store the output
                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                model_param = utils.Factor('GBMAssetPriceTSModelImplied', market_factor.name)

                if fxvol.expiry.size > 1:
                    dt = np.diff(np.append(0, fxvol.expiry))
                    var = fxvol.expiry * np.array(atm_vol) ** 2
                    sig = atm_vol[:1]
                    vol = atm_vol[:1]
                    var_tm1 = var[0]
                    fixed_variance = False

                    for var_t, delta_t, t_i in zip(var[1:], dt[1:] / 3.0, fxvol.expiry[1:]):
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

                price_factors[utils.check_tuple_name(price_param)] = OrderedDict(
                    [('Property_Aliases', None),
                     ('Vol', utils.Curve(['Integrated'], list(zip(fxvol.expiry, vol)))),
                     ('Quanto_FX_Volatility', None),
                     ('Quanto_FX_Correlation', 0.0)])
                price_models[utils.check_tuple_name(model_param)] = OrderedDict([('Risk_Premium', None)])


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
            # now setup the calc
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

        # calculate a reverse lookup for the tenors and store the daycount code
        all_tenors = utils.update_tenors(base_date, {ir_factor: process})
        # calculate the curve index
        curve_index = [instruments.calc_factor_index(ir_factor, {}, {ir_factor: 0}, all_tenors)]
        # calculate the reduced tenor curve index - note it's the same as above but in the next scenario index (1)
        curve_index_reduced = [instruments.calc_factor_index(ir_factor, {}, {ir_factor: 1}, all_tenors)]
        # calc the market swap rates and instrument_definitions    
        market_swaps, benchmarks = create_market_swaps(
            base_date, time_grid, curve_index, vol_surface, process.factor,
            implied_params['instrument']['Instrument_Definitions'], ir_factor.name)
        # number of random factors to use
        numfactors = process.num_factors()
        # setup a common context - we leave out the random numbers and pass it in explicitly below
        shared_mem = RiskNeutralInterestRate_State(self.batch_size, self.device, self.prec)
        # setup the variables
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
                    swaptionvol.delta = sys_params.get('Swaption_Volatility_Delta', 0.0)
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

                # setup the time grid
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
                for op_loop in range(2 * num_optimizers):
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
                # calculate the jacobians and final premiums
                jacobians, premiums = self.calc_jacobians(
                    implied_params, base_date, time_grid, process, final_implied_obj, ir_factor, swaptionvol)
                price_param = utils.Factor(implied_obj.__class__.__name__ + 'Jacobian', market_factor.name)
                price_factors[utils.check_tuple_name(price_param)] = jacobians
                prem_param = utils.Factor(implied_obj.__class__.__name__ + 'Premiums', market_factor.name)
                price_factors[utils.check_tuple_name(prem_param)] = premiums

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
         'and $r_T(t,T)$ is the partial derivative of the instantaneous forward rate r(t,T) with respect to the maturity.',
         'date $T$.'
         '',
         'Define:',
         '$$F(u,v)=\\frac{\\sigma_1u+\\sigma_2v}{\\sqrt{\\sigma_1^2+\\sigma_2^2+2\\rho\\sigma_1\\sigma_2}}$$',
         '',
         'Then $\\bar\\rho_1, \\bar\\rho_2$ are assigned:',
         '',
         '$$\\bar\\rho_1=F(1,\\rho)C$$',
         '$$\\bar\\rho_2=F(\\rho,1)C$$',
         '',
         'This is simply assumed to work'
         ]
    )

    def __init__(self, param, device, dtype):
        super(HullWhite2FactorModelParameters, self).__init__(param, device, dtype)
        # HullWhite2FactorModelPrices
        # HullWhite2FactorInterestRateModelPrices
        self.market_factor_type = 'HullWhite2FactorModelPrices'
        self.sigma_bounds = (1e-5, 0.09)
        self.alpha_bounds = (1e-5, 2.4)
        self.corr_bounds = (-.95, 0.95)

    def calc_jacobians(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface):
        # reset the stochastic process with the new implied factor
        process.reset_implied_factor(implied_obj)
        # get the swaption error and market values
        stoch_var, implied_var_dict, loss_fn = self.calc_loss_on_ir_curve(
            implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface, jac=True)

        # run the loss function
        jacobians = {}
        premiums, errors = loss_fn(implied_var_dict)
        # add the curve
        implied_var_dict['Curve'] = stoch_var
        for swaption_name, premium in premiums.items():
            grad_swaption = torch.autograd.grad(
                premium, list(implied_var_dict.values()), retain_graph=True)
            var_names = list(implied_var_dict.keys())
            for name, val in zip(var_names, grad_swaption):
                value = val.cpu().numpy()
                non_zero = np.where(value != 0.0)
                if name == 'Curve':
                    curve = utils.Curve([], list(zip(process.factor.get_tenor()[non_zero], value[non_zero])))
                    jacobians.setdefault(swaption_name, {}).setdefault('Curve', curve)
                elif name.startswith('Sigma') or name == 'Quanto_FX_Volatility':
                    curve = utils.Curve([], list(zip(implied_obj.param[name].array[non_zero[0], 0], value[non_zero])))
                    jacobians.setdefault(swaption_name, {}).setdefault(name, curve)
                else:
                    jacobians.setdefault(swaption_name, {}).setdefault(name, float(value[0]))

        all_premiums = {k: float(v.cpu().detach().numpy()) for k, v in premiums.items()}

        return jacobians, all_premiums

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
