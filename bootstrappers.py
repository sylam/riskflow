#########################################################################
# Copyright (C) 2016-2017  Shuaib Osman (sosman@investec.co.za)
# This file is part of Crstal.
#
# Crstal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Crstal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Crstal.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################

"""
All Bootstrappers are defined here.
"""

# import standard libraries
import time
import logging
from collections import OrderedDict, namedtuple

# third party stuff
import hdsobol
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.interpolate

# Internal modules
import utils
import pricing
import instruments
import riskfactors
import stochasticprocess

# misc functions/classes
from calculation import TimeGrid
from tensorflow.contrib.opt import ExternalOptimizerInterface

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

from scipy.stats import norm
from tensorflow.python.client import device_lib

curve_calibration_class = namedtuple('shared_mem', 't_Buffer \
                        t_Scenario_Buffer precision simulation_batch')

curve_jacobian_class = namedtuple('shared_mem', 't_Buffer t_Static_Buffer \
                                precision riskneutral simulation_batch')

market_swap_class = namedtuple('market_swap', 'deal_data price weight pvbp')
date_desc = {'years': 'Y', 'months': 'M', 'days': 'D'}
# date formatter
date_fmt = lambda x: ''.join(['{0}{1}'.format(v, date_desc[k]) for k, v in x.kwds.items()])

constrained = True


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


def logit(p):
    return np.log(p / (1 - p))


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


def create_market_swaps(base_date, time_grid, curve_index, atm_surface, curve_factor,
                        instrument_definitions, rate=None):
    # store these benchmark swap definitions if necessary
    benchmarks = []
    # store the benchmark instruments
    all_deals = {}
    for instrument in instrument_definitions:
        # setup the instrument
        effective = base_date + instrument['Start']
        maturity = effective + instrument['Tenor']
        exp_days = (effective - base_date).days
        tenor = (maturity - effective).days / utils.DAYS_IN_YEAR
        expiry = exp_days / utils.DAYS_IN_YEAR
        time_index = np.searchsorted(time_grid.mtm_time_grid, [exp_days], side='right') - 1
        swaption_name = 'Swap_{0:05.2f}_{1:02.0f}_{2}_{3}'.format(
            expiry, tenor, date_fmt(instrument['Start']), date_fmt(instrument['Tenor']))

        float_pay_dates = instruments.generate_dates_backward(
            maturity, effective, instrument['Floating_Frequency'])

        if instrument['Fixed_Frequency'] != instrument['Floating_Frequency']:
            raise Exception('Fixed_Frequency not equal to Floating_Frequency - Not supported')

        float_cash = utils.generate_float_cashflows(
            base_date, time_grid, float_pay_dates, 1.0, None, None,
            instrument['Floating_Frequency'], instrument['Floating_Frequency'],
            utils.get_day_count(instrument['Day_Count']), 0.0)

        K, pvbp = float_cash.get_par_swap_rate(base_date, curve_factor)
        float_cash.set_fixed_amount(-K)

        # get the atm vol
        if instrument['Market_Volatility'].amount:
            vol = instrument['Market_Volatility'].amount
        else:
            vol = atm_surface(tenor, expiry)[0][0]

        deal_data = utils.DealDataType(
            Instrument=None, Factor_dep={'Cashflows': float_cash, 'Forward': curve_index,
                                         'Discount': curve_index, 'CompoundingMethod': 'None'},
            Time_dep=utils.DealTimeDependencies(time_grid.mtm_time_grid, time_index), Calc_res=None)

        # store this
        all_deals[swaption_name] = market_swap_class(
            deal_data=deal_data,
            price=pvbp * utils.black_european_option_price(K, K, 0.0, vol, expiry, 1.0, 1.0),
            weight=instrument['Weight'],
            pvbp=np.exp(-expiry * curve_factor.current_value(expiry)))

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


def _get_shape_tuple(tensor):
    return tuple(dim.value for dim in tensor.get_shape())


def _prod(array):
    prod = 1
    for value in array:
        prod *= value
    return prod


def _accumulate(list_):
    total = 0
    yield total
    for x in list_:
        total += x
        yield total


class ScipyLeastsqOptimizerInterface(object):
    """Base class for interfaces with external optimization algorithms.

    Subclass this and implement `_minimize` in order to wrap a new optimization
    algorithm.

    `ExternalOptimizerInterface` should not be instantiated directly; instead use
    e.g. `ScipyOptimizerInterface`.

    @@__init__

    @@minimize
    """

    def __init__(self, loss, var_to_bounds, **optimizer_kwargs):

        def fwd_gradients(ys, xs, d_xs):
            """ Forward-mode pushforward analogous to the pullback defined by tf.gradients.
                With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
                the vector being pushed forward."""

            v = self._dummy_ys_vec  # dummy variable
            g = tf.concat([x for x in tf.gradients(ys, xs, grad_ys=v) if x is not None], axis=0)
            return tf.gradients(g, v, grad_ys=d_xs)

        self._loss = tf.stack(loss)
        self._vars = variables.trainable_variables()
        packed_bounds = None
        if var_to_bounds is not None:
            left_packed_bounds = []
            right_packed_bounds = []
            for var in self._vars:
                shape = var.get_shape().as_list()
                bounds = (-np.infty, np.infty)
                if var in var_to_bounds:
                    bounds = var_to_bounds[var]
                    left_packed_bounds.extend(list(np.broadcast_to(bounds[0], shape).flat))
                    right_packed_bounds.extend(list(np.broadcast_to(bounds[1], shape).flat))
            packed_bounds = list([left_packed_bounds, right_packed_bounds])
        self._packed_bounds = packed_bounds

        self._dummy_ys_vec = tf.placeholder_with_default(
            np.ones(self._loss.get_shape()[0].value, dtype=np.float32), shape=self._loss.get_shape())
        self._dummy_jac_vec = array_ops.placeholder(
            self._vars[0].dtype, sum([_get_shape_tuple(x)[0] for x in self._vars]))

        self._update_placeholders = [
            array_ops.placeholder(var.dtype) for var in self._vars
        ]
        self._var_updates = [
            var.assign(array_ops.reshape(placeholder, _get_shape_tuple(var)))
            for var, placeholder in zip(self._vars, self._update_placeholders)
        ]

        time_now = time.clock()
        print('Calculating jacobian')
        # loss_grads = [tf.gradients(resid, self._vars) for resid in loss]
        self.jvp = fwd_gradients(self._loss, self._vars, self._dummy_jac_vec)

        print('Done Calculating jacobian')

        # record the time
        print('This took', time.clock() - time_now, 'seconds.')
        equalities_grads = []
        inequalities_grads = []

        self.optimizer_kwargs = optimizer_kwargs

        self._packed_var = self._pack(self._vars)
        # self._packed_loss_grad = tf.stack([self._pack(grad) for grad in loss_grads])
        self._packed_equality_grads = []
        self._packed_inequality_grads = []
        dims = [_prod(_get_shape_tuple(var)) for var in self._vars]
        accumulated_dims = list(_accumulate(dims))
        self._packing_slices = [
            slice(start, end)
            for start, end in zip(accumulated_dims[:-1], accumulated_dims[1:])
        ]

    def minimize(self, session=None, feed_dict=None, fetches=None,
                 step_callback=None, loss_callback=None, **run_kwargs):
        """Minimize a scalar `Tensor`.

        Variables subject to optimization are updated in-place at the end of
        optimization.

        Note that this method does *not* just return a minimization `Op`, unlike
        `Optimizer.minimize()`; instead it actually performs minimization by
        executing commands to control a `Session`.

        Args:
          session: A `Session` instance.
          feed_dict: A feed dict to be passed to calls to `session.run`.
          fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
            as positional arguments.
          step_callback: A function to be called at each optimization step;
            arguments are the current values of all optimization variables
            flattened into a single vector.
          loss_callback: A function to be called every time the loss and gradients
            are computed, with evaluated fetches supplied as positional arguments.
          **run_kwargs: kwargs to pass to `session.run`.
        """

        session = session or ops.get_default_session()
        feed_dict = feed_dict or {}
        fetches = fetches or []

        loss_callback = loss_callback or (lambda *fetches: None)
        step_callback = step_callback or (lambda xk: None)

        # Construct loss function and associated gradient.
        loss_func = self._make_eval_func(self._loss, session,
                                         feed_dict, fetches, loss_callback)
        n = self._dummy_jac_vec.shape[0].value
        grad_func = [self._make_eval_func(self.jvp, session,
                                          {self._dummy_jac_vec: np.eye(n)[i]},
                                          fetches, loss_callback)
                     for i in range(n)]

        # Get initial value from TF session.
        initial_packed_var_val = session.run(self._packed_var)

        # Perform minimization.
        packed_var_val = self._minimize(
            initial_val=initial_packed_var_val,
            loss_func=loss_func, grad_func=grad_func,
            packed_bounds=self._packed_bounds,
            step_callback=step_callback,
            optimizer_kwargs=self.optimizer_kwargs)
        var_vals = [
            packed_var_val[packing_slice] for packing_slice in self._packing_slices
        ]

        # Set optimization variables to their new values.
        session.run(
            self._var_updates,
            feed_dict=dict(zip(self._update_placeholders, var_vals)),
            **run_kwargs)

    def _minimize(self, initial_val, loss_func, grad_func,
                  packed_bounds, step_callback, optimizer_kwargs):

        def loss_func_wrapper(x):
            return loss_func(x)[0]

        def grad_func_wrapper(x):
            return np.array([grad(x)[0].astype('float64') for grad in grad_func]).T

        def print_fun(x, f, accepted):
            print("at minimum %.4f accepted %d" % (f, int(accepted)))

        minimize_kwargs = {
            'jac': True,
            'callback': step_callback,
            'constraints': [],
            'bounds': packed_bounds,
        }

        import scipy.optimize
        if packed_bounds is not None:
            result = scipy.optimize.least_squares(loss_func_wrapper, initial_val,
                                                  jac=grad_func_wrapper, bounds=packed_bounds)
        else:
            result = scipy.optimize.least_squares(loss_func_wrapper, initial_val,
                                                  jac=grad_func_wrapper)

        message_lines = [
            'Optimization terminated with:',
            '  Message: %s',
            '  Objective function value: %f',
        ]

        message_args = [result.message, result.fun]

        if hasattr(result, 'nit'):
            # Some optimization methods might not provide information such as nit and
            # nfev in the return. Logs only available information.
            message_lines.append('  Number of iterations: %d')
            message_args.append(result.nit)
        if hasattr(result, 'nfev'):
            message_lines.append('  Number of functions evaluations: %d')
            message_args.append(result.nfev)
        logging.info('\n'.join(message_lines), *message_args)

        return result['x']

    @classmethod
    def _pack(cls, tensors):
        """Pack a list of `Tensor`s into a single, flattened, rank-1 `Tensor`."""
        if not tensors:
            return None
        elif len(tensors) == 1:
            return array_ops.reshape(tensors[0], [-1])
        else:
            flattened = [array_ops.reshape(tensor, [-1]) for tensor in tensors]
            return array_ops.concat(flattened, 0)

    def _make_eval_func(self, tensors, session, feed_dict, fetches,
                        callback=None):
        """Construct a function that evaluates a `Tensor` or list of `Tensor`s."""
        if not isinstance(tensors, list):
            tensors = [tensors]
        num_tensors = len(tensors)

        def eval_func(x):
            """Function to evaluate a `Tensor`."""
            augmented_feed_dict = {
                var: x[packing_slice].reshape(_get_shape_tuple(var))
                for var, packing_slice in zip(self._vars, self._packing_slices)
            }

            augmented_feed_dict.update(feed_dict)
            augmented_fetches = tensors + fetches

            augmented_fetch_vals = session.run(augmented_fetches,
                                               feed_dict=augmented_feed_dict)

            if callable(callback):
                callback(*augmented_fetch_vals[num_tensors:])

            return augmented_fetch_vals[:num_tensors]

        return eval_func

    def _make_eval_funcs(self, tensors, session, feed_dict, fetches, callback=None):
        return [self._make_eval_func(tensor, session, feed_dict, fetches, callback)
                for tensor in tensors]


class ScipyBasinOptimizerInterface(ExternalOptimizerInterface):
    """Wrapper allowing `scipy.optimize.basinhopping` to operate a `tf.Session`.
    Implemented exactly the same as ScipyOptimizerInterface
    """

    _DEFAULT_METHOD = 'L-BFGS-B'

    def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                  equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                  packed_bounds, step_callback, optimizer_kwargs):

        def loss_grad_func_wrapper(x):
            # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
            loss, gradient = loss_grad_func(x)
            return loss, gradient.astype('float64')
        
        def print_fun(x, f, accepted):
            print("at minimum %.4f accepted %d" % (f, int(accepted)))
            if f<0.1:
                return True

        optimizer_kwargs = dict(optimizer_kwargs.items())
        method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)
        take_step = optimizer_kwargs.pop('take_step', None)
        accept_test = optimizer_kwargs.pop('accept_test', None)

        minimize_args = [loss_grad_func_wrapper, initial_val]
        minimize_kwargs = {
            'jac': True,
            'callback': step_callback,
            'method': method,
            'constraints': [],
            'bounds': packed_bounds,
        }

        import scipy.optimize
        result = scipy.optimize.basinhopping(
            *minimize_args, minimizer_kwargs=minimize_kwargs, niter=200, T=0.25,
            accept_test=accept_test, callback=print_fun, take_step=take_step)

        message_lines = [
            'Optimization terminated with:',
            '  Message: %s',
            '  Objective function value: %f',
        ]

        message_args = [result.message, result.fun]

        if hasattr(result, 'nit'):
            # Some optimization methods might not provide information such as nit and
            # nfev in the return. Logs only available information.
            message_lines.append('  Number of iterations: %d')
            message_args.append(result.nit)
        if hasattr(result, 'nfev'):
            message_lines.append('  Number of functions evaluations: %d')
            message_args.append(result.nfev)
        logging.info('\n'.join(message_lines), *message_args)

        return result['x']


class InterestRateJacobian(object):
    def __init__(self, param, prec=np.float32):
        self.prec = prec
        self.param = param
        self.batch_size = 1

    def bootstrap(self, sys_params, price_models, price_factors, market_prices, grid, calendars, debug=None):

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
        time_grid = TimeGrid({base_date}, {base_date}, {base_date})
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
                        ir_curve = riskfactors.construct_factor(ir_factor, price_factors)
                    except:
                        logging.warning('Unable to calculate the Jacobian for {0} - skipping'.format(market_price))
                        continue
                    # not really necessary - need to clean this up - TODO!
                    distinct_tenors = OrderedDict()
                    utils.distinct_tenors(distinct_tenors, ir_factor, ir_curve)
                    # calculate a reverse lookup for the tenors and store the daycount code
                    all_tenors = utils.update_tenors(base_date, {}, {ir_factor: ir_curve}, distinct_tenors)
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
                            benchmarks['Deposit_' + fmt_fn(child['quote']['Descriptor'])] = pricing.pvfixedcashflows(
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
                            benchmarks['FRA_' + str(child['quote']['Descriptor'])] = pricing.pvfloatcashflowlist(
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
                            benchmarks['Swap_' + date_fmt(child['quote']['Descriptor'])] = pricing.pvfloatcashflowlist(
                                shared_mem, time_grid, deal_data, pricing.pricer_float_cashflows, settle_cash=False)
                        else:
                            raise Exception('quote type not supported')

                    n = ir_curve.tenors.shape[0]
                    dummy_jac_vec = tf.placeholder(self.prec, n)
                    bench = tf.squeeze(tf.concat(list(benchmarks.values()), axis=0), axis=1)
                    jvp = fwd_gradients(bench, shared_mem.t_Static_Buffer, dummy_jac_vec)

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True

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


class GBMTSImpliedParameters(object):
    def __init__(self, param, prec=np.float32):
        self.prec = prec
        self.param = param

    def bootstrap(self, sys_params, price_models, price_factors, market_prices, grid, calendars, debug=None):
        '''
        Checks for Declining variance in the ATM vols of the relevant price factor and corrects accordingly.
        '''
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == 'GBMTSModelPrices':
                # get the vol surface
                vol_factor = utils.Factor('FXVol', utils.check_rate_name(
                    implied_params['instrument']['Asset_Price_Volatility']))
                
                # this shouldn't fail - if it does, need to log it and move on
                try:
                    fxvol = riskfactors.construct_factor(vol_factor, price_factors)
                except:
                    logging.warning('Unable to bootstrap {0} - skipping'.format(market_price))
                    continue

                mn_ix = np.searchsorted(fxvol.moneyness, 1.0)
                atm_vol = [np.interp(1, fxvol.moneyness[mn_ix - 1:mn_ix + 1], y) for y in
                           fxvol.get_vols()[:, mn_ix - 1:mn_ix + 1]]

                # store the output
                price_param = utils.Factor('GBMTSImpliedParameters', market_factor.name)
                model_param = utils.Factor('GBMAssetPriceTSModelImplied', market_factor.name)
                
                if fxvol.expiry.size>1:
                    dt = np.diff(np.append(0,fxvol.expiry))
                    var = fxvol.expiry*np.array(atm_vol)**2
                    sig = atm_vol[:1]
                    vol = atm_vol[:1]
                    var_tm1 = var[0]
                    fixed_variance = False
                    
                    for var_t, delta_t, t_i in zip(var[1:], dt[1:]/3.0, fxvol.expiry[1:]):
                        M = var_tm1+delta_t*(sig[-1]**2)
                        if var_t<M:
                            fixed_variance = True                            
                            var_t = M
                            
                        a = delta_t
                        b = sig[-1]*delta_t
                        c = M-var_t
                        
                        sig.append( (-b+np.sqrt(b*b-4.0*a*c))/(2.0*a) )
                        vol.append( np.sqrt( var_t/t_i ) )
                        var_tm1 = var_t
                        
                    if fixed_variance:
                        logging.warning('Fixed declining variance for {0}'.format(market_price))
                else:
                    vol = atm_vol

                price_factors[utils.check_tuple_name(price_param)] = OrderedDict(
                    [('Quanto_FX_Volatility', None),
                     ('Vol', utils.Curve([], list(zip(fxvol.expiry, vol)))),
                     ('Quanto_FX_Correlation', 0)])
                price_models[utils.check_tuple_name(model_param)] = OrderedDict([('Risk_Premium', None)])


class RiskNeutralInterestRateModel(object):
    def __init__(self, param, prec=np.float32):
        self.param = param
        self.num_batches = 1
        self.batch_size = 512 * 10
        # needs to be generic
        self.gpu = sorted([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])[0]
        self.prec = prec
        self.sample = None
        # set the global precision - not ideal
        utils.Default_Precision = prec

    def calc_loss_on_ir_curve(self, implied_params, base_date, time_grid, process,
                              implied_obj, ir_factor, atm_surface, resid=tf.square, debug=None):
        # calculate a reverse lookup for the tenors and store the daycount code
        all_tenors = utils.update_tenors(base_date, {ir_factor: process})
        # calculate the curve index - need to clean this up - TODO!!!
        curve_index = [instruments.calc_factor_index(ir_factor, {}, {ir_factor: 0}, all_tenors)]
        # calc the market swap rates and instrument_definitions    
        market_swaps, benchmarks = create_market_swaps(
            base_date, time_grid, curve_index, atm_surface, process.factor,
            implied_params['instrument']['Instrument_Definitions'], ir_factor.name)
        # number of random factors to use
        numfactors = process.num_factors()
        # random sample - use a sobol sequence skipping the first 4000 numbers
        # need to check convergence and error - TODO
        # also, this is in python - might want to use cffi to speed this up
        sample = self.calc_sample(time_grid, numfactors)
        # output of gpu's
        gpu_swaptions = {}
        # setup a common context - we leave out the random numbers and pass it in explicitly below
        shared_mem = curve_calibration_class(
            t_Buffer={},
            t_Scenario_Buffer=[None],
            precision=self.prec,
            simulation_batch=self.batch_size)
        # setup the variables
        implied_var = {}
        with tf.device('/cpu:0'):
            # the curve is treated as constant here - no placeholders
            stoch_var = tf.constant(process.factor.current_value(), dtype=self.prec)
            # store variables on the cpu
            with tf.name_scope("Implied_Input"):
                for param_name, param_value in implied_obj.current_value().items():
                    factor_name = utils.Factor(
                        implied_obj.__class__.__name__, ir_factor.name + (param_name,))
                    tf_variable = tf.get_variable(
                        name=utils.check_scope_name(factor_name),
                        initializer=self.unconstrain(param_name, param_value.astype(self.prec)),
                        dtype=self.prec)
                    implied_var[param_name] = self.constrain(param_name, tf_variable)

            # now setup the calc - on the cpu
            process.precalculate(base_date, time_grid, stoch_var, shared_mem, 0, implied_tensor=implied_var)
            # we need stochastic deflation

            # def calc_stoch_deflator(self, time_grid, curve_index, shared):

            # cached_tensor, interpolation_params = utils.cache_interpolation(shared, curve_index[0], self.stoch_drift[:-1])
            # tensor = utils.TensorBlock(curve_index, [cached_tensor], [interpolation_params], time_grid)
            # self.deflation_drift = tensor.gather_weighted_curve(shared, self.stoch_dt * utils.DAYS_IN_YEAR,
            #                                                        multiply_by_time=False)

            # process.calc_stoch_deflator(time_grid, curve_index, shared_mem)
                
            # needed to interpolate the zero curve 
            delta_scen_t = np.diff(time_grid.scen_time_grid).astype(self.prec)

            with tf.device(self.gpu):
                # run the batch
                for batch_index in range(self.num_batches):
                    # load up the batch
                    batch_sample = sample[batch_index].T.reshape(
                        numfactors, time_grid.time_grid_years.size,-1).astype(self.prec)
                    # simulate the price factor
                    shared_mem.t_Scenario_Buffer[0] = process.generate(shared_mem, batch_sample)
                    # get the discount factors
                    Dfs = utils.calc_time_grid_curve_rate(curve_index, time_grid, shared_mem)
                    # go over the instrument definitions and build the calibration
                    for swaption_name, market_data in market_swaps.items():
                        expiry = market_data.deal_data.Time_dep.mtm_time_grid[
                            market_data.deal_data.Time_dep.deal_time_grid[0]]
                        scenario_index = time_grid.scen_time_grid.searchsorted(expiry, side='right')-1
                        DtT = process.stoch_deflation[scenario_index]
                        par_swap = pricing.pvfloatcashflowlist(
                            shared_mem, time_grid, market_data.deal_data,
                            pricing.pricer_float_cashflows, settle_cash=False)
                        sum_swaption = tf.reduce_sum(tf.nn.relu(DtT*par_swap))
                        if swaption_name in gpu_swaptions:
                            gpu_swaptions[swaption_name] += sum_swaption
                        else:
                            gpu_swaptions[swaption_name] = sum_swaption

            calibrated_swaptions = {k: v/(self.batch_size*self.num_batches) for k, v in gpu_swaptions.items()}
            error = {k: swap.weight * resid(100.0 * (
                    calibrated_swaptions[k] - swap.price)) for k, swap in market_swaps.items()}

        return implied_var, error, calibrated_swaptions, market_swaps, benchmarks

    def bootstrap(self, sys_params, price_models, price_factors, market_prices, grid, calendars, debug=None):
        base_date = sys_params['Base_Date']
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
                    swaptionvol = riskfactors.construct_factor(vol_factor, price_factors)
                    ir_curve = riskfactors.construct_factor(ir_factor, price_factors)
                except:
                    logging.warning('Unable to bootstrap {0} - skipping'.format(market_price))
                    continue

                # if market_factor.name[0]!='ZAR-SWAP':
                #if market_factor.name[0] not in ['ZAR-SWAP','ZAR-JIBAR-3M']:#, 'USD-LIBOR-3M','USD-MASTER']:
                if market_factor.name[0] not in ['ZAR-JIBAR-3M']:#,'USD-LIBOR-3M','USD-MASTER']:
                    continue

                # calc the atm vol
                mn_ix = np.searchsorted(swaptionvol.moneyness, 0.0)
                atm_vol = np.array([np.interp(1, swaptionvol.moneyness[mn_ix - 1:mn_ix + 1], y)
                                    for y in swaptionvol.get_vols()[:, mn_ix - 1:mn_ix + 1]])
                atm_surface = scipy.interpolate.RectBivariateSpline(
                    swaptionvol.tenor, swaptionvol.expiry,
                    atm_vol.reshape(swaptionvol.tenor.size, swaptionvol.expiry.size))

                # set of dates for the calibration
                mtm_dates = set(
                    [base_date + x['Start'] for x in implied_params['instrument']['Instrument_Definitions']])
                # scenario_dates = grid(base_date, max(mtm_dates), '0d 1w(1w) 3m(1m) 5y(2m)')

                time_grid = TimeGrid(mtm_dates, mtm_dates, mtm_dates)
                # add a delta of 10 days to the time_grid_years (without changeing the scenario grid
                time_grid.set_base_date(base_date, delta=10)

                # tenor guess - add all the extra dates till the first expiry (should help with accuracy)
                swaption_expiries = [(x - base_date).days / utils.DAYS_IN_YEAR for x in sorted(mtm_dates)]
                implied_obj, process = self.implied_process(price_factors, price_models, ir_curve, rate)

                # setup the tensorflow calc
                graph = tf.Graph()
                with graph.as_default():
                    # calculate the error
                    loss, optimizers, implied_var, calibrated_swaptions, market_swaptions, benchmarks = self.calc_loss(
                        implied_params, base_date, time_grid, process, implied_obj, ir_factor, atm_surface)
                    # fetch the expiries for reporting
                    expiry_names = {k: v.deal_data.Time_dep.mtm_time_grid[v.deal_data.Time_dep.deal_time_grid[-1]]
                                    for k,v in market_swaptions.items()}
                                    
                if debug is not None:
                    debug.deals['Deals']['Children']=[{'instrument':x} for x in benchmarks]
                    try:
                        debug.write_trade_file (market_factor.name[0]+'.aap')
                    except:
                        logging.warning('Could not write output file {}'.format(market_factor.name[0]+'.aap'))

                # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
                config = tf.ConfigProto(allow_soft_placement=True)

                with tf.Session(graph=graph, config=config) as sess:
                    # init the variables
                    sess.run(tf.global_variables_initializer())
                    # check the time
                    time_now = time.clock()
                    batch_loss, vars = sess.run([loss, implied_var])
                    print('Before run', batch_loss, market_factor.name[0])
                    for k, v in sorted(vars.items()):
                        print(k, v)
                        
                    sim_swaptions = sess.run(calibrated_swaptions)
                    for k, v in sorted(sim_swaptions.items()):
                        price = market_swaptions[k].price
                        print(k, 'market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                            price, v, 100.0 * (price - sim_swaptions[k]) / price))
                        
                    # mimimize
                    soln = None
                    num_optimizers = len(optimizers)
                    for op_loop in range(6):
                        optimizers[op_loop%num_optimizers].minimize(sess)
                        batch_loss, vars = sess.run([loss, implied_var])
                        
                        if soln is None or batch_loss<soln[0]:
                            soln = (batch_loss,vars)
                            print('After run {}'.format(op_loop), batch_loss)
                            for k, v in sorted(vars.items()):
                                print(k, v)
                            sim_swaptions = sess.run(calibrated_swaptions)
                            for k, v in sorted(sim_swaptions.items()):
                                price = market_swaptions[k].price
                                print(k, 'market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                                    price, v, 100.0 * (price - sim_swaptions[k]) / price))
                        
                    # save this
                    self.save_params(soln[1], price_factors, implied_obj, rate)
                    # record the time
                    print('This took', time.clock() - time_now, 'seconds.')


class PCAMixedFactorModelParameters(RiskNeutralInterestRateModel):
    def __init__(self, param, prec=np.float32):
        super(PCAMixedFactorModelParameters, self).__init__(param)
        self.market_factor_type = 'HullWhite2FactorInterestRateModelPrices'

    def calc_sample(self, time_grid, numfactors=0):
        if numfactors != 3 or self.sample is None:
            self.sample = normalize(norm.ppf(hdsobol.gen_sobol_vectors(
                self.batch_size * self.num_batches + 4000, time_grid.scen_time_grid.size * numfactors))[3999:]).reshape(
                self.num_batches, self.batch_size, -1)
        return self.sample

    def calc_loss(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor, atm_surface):
        # get the swaption error and market values
        implied_var, error, calibrated_swaptions, market_swaptions, benchmarks = self.calc_loss_on_ir_curve(
            implied_params, base_date, time_grid, process, implied_obj, ir_factor, atm_surface)

        losses = list(error.values())
        loss = tf.reduce_sum(losses)
        
        loss_with_penatalties = loss + tf.nn.moments(
            implied_var['Yield_Volatility'][1:] - implied_var['Yield_Volatility'][:-1], axes=[0])[1]

        with tf.device(self.gpu):
            # standard tensorflow gradient descent based optimizer
            # optimizer   = tf.train.AdamOptimizer(0.1).minimize(loss_with_penatalties)
            if constrained:
                var_to_bounds = {implied_var['Yield_Volatility']: (1e-3, 0.6),
                                 implied_var['Reversion_Speed']: (1e-2, 1.8)}
            else:
                var_to_bounds = None

            # optimizer = ScipyLeastsqOptimizerInterface(losses, var_to_bounds={} if var_to_bounds is None else var_to_bounds )
            optimizer = ScipyBasinOptimizerInterface(loss, var_to_bounds=var_to_bounds)

        return loss, optimizer, implied_var, calibrated_swaptions, market_swaptions, benchmarks

    def constrain(self, param_name, variable):
        if param_name == 'Yield_Volatility':
            return 0.6 * tf.nn.sigmoid(variable)
        elif param_name == 'Reversion_Speed':
            return 1.8 * tf.nn.sigmoid(variable)
        else:
            raise Exception('Variable not constrained')

    def unconstrain(self, param_name, value):
        if param_name == 'Yield_Volatility':
            return logit(value / 0.6)
        elif param_name == 'Reversion_Speed':
            return logit(value / 1.8)
        else:
            raise Exception('Variable not constrained')

    def implied_process(self, price_factors, price_models, ir_curve, rate):
        # need to create a process and params as variables to pass to tf
        price_model = utils.Factor('PCAInterestRateModel', rate[1:])
        param_name = utils.check_tuple_name(price_model)
        vol_tenors = np.array([0,1.5,3,6,12,24,48,84,120])/12.0
        
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

        return implied_obj, process

    def save_params(self, vars, price_factors, implied_obj, rate):
        # construct an initial guess - need to read from params
        param_name = utils.check_tuple_name(
            utils.Factor(type='PCAInterestRateModel', name=rate[1:]))
        param = price_factors[param_name]
        
        param['Reversion_Speed'] = float(vars['Reversion_Speed'][0])
        param['Yield_Volatility'].array=np.dstack((vol_tenors, vars['Yield_Volatility']))[0]


class HullWhite2FactorModelParameters(RiskNeutralInterestRateModel):
    def __init__(self, param, prec=np.float32):
        super(HullWhite2FactorModelParameters, self).__init__(param)
        self.market_factor_type = 'HullWhite2FactorInterestRateModelPrices'

    def calc_sample(self, time_grid, numfactors=0):
        if numfactors != 2 or self.sample is None:
            self.sample = normalize(norm.ppf(hdsobol.gen_sobol_vectors(
                self.batch_size * self.num_batches + 4000, time_grid.time_grid_years.size * numfactors))[3999:]).reshape(
                self.num_batches, self.batch_size, -1)
        return self.sample

    def calc_loss(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor, atm_surface):

        def make_bounds(implied_var, sigma_bounds, corr_bounds, alpha_bounds):
            return {
                implied_var['Sigma_1']: sigma_bounds,
                implied_var['Sigma_2']: sigma_bounds,
                implied_var['Correlation']: corr_bounds,
                implied_var['Alpha_1']: alpha_bounds,
                implied_var['Alpha_2']: alpha_bounds}
        
        def split_param(x):
            corr = x[-1:]
            alpha = x[-3:-1]
            sigmas = x[:-3]
            return sigmas, alpha, corr
        
        def make_basin_callbacks(step, sigma_min_max, alpha_min_max, corr_min_max):
            
            def bounds_check(**kwargs):
                x = kwargs["x_new"]
                sigmas, alpha, corr = split_param(x)
                sigma_ok = (sigmas > sigma_min_max[0]).all() and (sigmas < sigma_min_max[1]).all()
                alpha_ok = (alpha > alpha_min_max[0]).all() and (alpha < alpha_min_max[1]).all()
                corre_ok = (corr > corr_min_max[0]).all() and (corr < corr_min_max[1]).all()
                return sigma_ok and alpha_ok and corre_ok
            
            def basin_step(x):
                sigmas, alpha, corr = split_param(x)
                #update vars
                sigmas = (sigmas+np.random.uniform(-0.08*step, 0.08*step, sigmas.size)).clip(*sigma_min_max)
                alpha = (alpha+np.random.uniform(-1.5*step, 1.5*step, alpha.size)).clip(*alpha_min_max)
                corr = (corr+np.random.uniform(-step, step, corr.size)).clip(*corr_min_max)
                
                return np.concatenate((sigmas, alpha, corr))
            
            return bounds_check, basin_step
        
        
        # get the swaption error and market values
        implied_var_dict, error_dict, calibrated_swaptions, market_swaptions, benchmarks = self.calc_loss_on_ir_curve(
            implied_params, base_date, time_grid, process, implied_obj, ir_factor, atm_surface)

        error = OrderedDict(error_dict)
        var_list = [implied_var_dict['Sigma_1'],implied_var_dict['Sigma_2'],
                    implied_var_dict['Alpha_1'], implied_var_dict['Alpha_2'],
                    implied_var_dict['Correlation']]
        
        implied_var = OrderedDict(implied_var_dict)
        losses = list(error.values())
        loss = tf.reduce_sum(losses)# +100.0*(
            # tf.nn.moments(implied_var['Sigma_1'][1:]-implied_var['Sigma_1'][:-1],axes=[0])[1]+
            # tf.nn.moments(implied_var['Sigma_2'][1:]-implied_var['Sigma_2'][:-1],axes=[0])[1])

        sigma_bounds = (1e-4, 0.08)
        alpha_bounds = (1e-4, 2.25)
        corr_bounds  = (-.93, 0.93)
        
        # slightly different bounds for the least squares calc
        lq_sigma_bounds = (1e-5, 0.085)
        lq_alpha_bounds = (1e-5, 2.3)
        lq_corr_bounds  = (-.94, 0.94)
        
        with tf.device(self.gpu):
            # standard tensorflow gradient descent based optimizer
            if constrained:
                var_to_bounds = make_bounds(implied_var, sigma_bounds, corr_bounds, alpha_bounds)
                var_to_bounds_lq = make_bounds(implied_var, lq_sigma_bounds, lq_corr_bounds, lq_alpha_bounds)
            else:
                var_to_bounds = None
                var_to_bounds_lq = None
                
            bounds_ok, make_step = make_basin_callbacks(0.25, sigma_bounds, alpha_bounds, corr_bounds)

            optimizers = [
                ScipyBasinOptimizerInterface(
                    loss, var_list = var_list,
                    take_step = make_step,
                    accept_test = bounds_ok,
                    var_to_bounds=var_to_bounds)
                ,ScipyLeastsqOptimizerInterface(
                    losses, var_to_bounds=var_to_bounds_lq)
                ]

        return loss, optimizers, implied_var, calibrated_swaptions, market_swaptions, benchmarks

    def constrain(self, param_name, variable, ident=constrained):
        if ident:
            return variable
        elif param_name in ['Sigma_1', 'Sigma_2']:
            return 0.0001 + 0.07 * tf.nn.sigmoid(variable)
        elif param_name in ['Alpha_1', 'Alpha_2']:
            return 0.0001 + 2.45 * tf.nn.sigmoid(variable)
        elif param_name in ['Correlation']:
            return 0.92*tf.nn.tanh(variable)
        else:
            raise Exception('Variable not constrained')

    def unconstrain(self, param_name, value, ident=constrained):
        if ident:
            return value
        elif param_name in ['Sigma_1', 'Sigma_2']:
            return logit((value-0.0001) /0.07 )
        elif param_name in ['Alpha_1', 'Alpha_2']:
            return logit((value-0.0001) / 2.45 )
        elif param_name in ['Correlation']:
            return np.arctanh(value/0.92)
        else:
            raise Exception('Variable not constrained')

    def implied_process(self, price_factors, price_models, ir_curve, rate):
        vol_tenors = np.array([0,1.5,3,6,12,24,48,84,120])/12.0
        # construct an initial guess - need to read from params
        param_name = utils.check_tuple_name(
            utils.Factor(type='HullWhite2FactorModelParameters', name=rate[1:]))
        
        # check if we need a quanto fx vol
        fx_factor = utils.Factor('GBMTSImpliedParameters', ir_curve.get_currency())
        ir_factor = utils.Factor('InterestRate', ir_curve.get_currency())
        fx_factor_name = utils.check_tuple_name(fx_factor)
        ir_factor_name = utils.check_tuple_name(ir_factor)
        
        if fx_factor_name in price_factors:
            quanto_fx = price_factors[fx_factor_name]['Vol']
            correlation_name = 'Correlation.FxRate.{}/{}'.format('.'.join(
                sorted(('USD',)+ir_curve.get_currency())), ir_factor_name)
            # the correlation between fx and ir - needed to establish Quanto Correlation 1 and 2
            C = price_factors.get(correlation_name,{'Value':0.0})['Value']
        else:
            C = None
            quanto_fx = None
            
        if param_name in price_factors:
            param = price_factors[param_name]
            implied_obj = riskfactors.HullWhite2FactorModelParameters(
                {'Quanto_FX_Volatility': quanto_fx,
                 'short_rate_fx_correlation': C,
                 'Alpha_1': np.clip(param['Alpha_1'],1e-4, 2.25),
                 'Alpha_2': np.clip(param['Alpha_2'],1e-4, 2.25),
                 'Correlation': np.clip(param['Correlation'], -0.95, 0.95),
                 'Sigma_1': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_1'].array.T).clip(1e-4+5e-5,0.08)))),
                 'Sigma_2': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_2'].array.T).clip(1e-4+5e-5,0.08))))})
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

        return implied_obj, process
    
    def save_params(self, vars, price_factors, implied_obj, rate):
        param_name = utils.check_tuple_name(
            utils.Factor(type='HullWhite2FactorModelParameters', name=rate[1:]))
        param = price_factors[param_name]
        
        param['Alpha_1'] = float(vars['Alpha_1'][0])
        param['Alpha_2'] = float(vars['Alpha_2'][0])
        param['Correlation'] = float(vars['Correlation'][0])
        
        # grab the sigma tenors
        sig1_tenor, sig2_tenor = implied_obj.get_vol_tenors()
        param['Sigma_1'].array=np.dstack((sig1_tenor,vars['Sigma_1']))[0]
        param['Sigma_2'].array=np.dstack((sig2_tenor,vars['Sigma_2']))[0]
        
        # grab the quanto fx correlations
        quanto_fx1, quanto_fx2 = implied_obj.get_quanto_correlation(
            vars['Correlation'], [vars['Sigma_1'], vars['Sigma_2']])
        param['Quanto_FX_Correlation_1'] = quanto_fx1
        param['Quanto_FX_Correlation_2'] = quanto_fx2
        param['Quanto_FX_Volatility'] = implied_obj.param['Quanto_FX_Volatility']


def construct_bootstrapper(btype, param):
    return globals().get(btype)(param)
