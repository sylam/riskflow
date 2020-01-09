##########################################################################
# Copyright (C) 2016-2017  Shuaib Osman (sosman@investec.co.za)
# This file is part of RiskFlow.
#
# RiskFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# RiskFlow is distributed in the hope that it will be usefugather_weighted_curvel,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RiskFlow.  If not, see <http://www.gnu.org/licenses/>.
##########################################################################

# utility functions and constants
import utils
# specific modules
import numpy as np
import tensorflow as tf
from collections import OrderedDict

# useful constants
BARRIER_UP = -1.0
BARRIER_DOWN = 1.0
BARRIER_IN = -1.0
BARRIER_OUT = 1.0
OPTION_PUT = -1.0
OPTION_CALL = 1.0


def cash_settle(shared, currency, time_index, value):
    if shared.t_Cashflows is not None:
        shared.t_Cashflows[currency][time_index] += value


class SensitivitiesEstimator(object):
    """ Implements the AAD sensitivities (both first and second derivatives)
       
    Attributes:
        
        value: function output (tensor)
        params: List of model parameters (list of tensor(s))

    """
    def __init__(self, value, params):
        """
        Args:
            cost: Cost function output (tensor)
            params: List of model parameters (list of tensor(s))
        """
        self.cost = value
        self.grad = OrderedDict([(
        var, tf.convert_to_tensor(grad)) for grad, var in zip(
            tf.gradients(value, params, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
            params) if grad is not None])
        self.params = list(self.grad.keys())
        self.P = self.flatten(self.params).get_shape().as_list()[0]

    def flatten(self, params):
        """
        Flattens the list of tensor(s) into a 1D tensor
        
        Args:
            params: List of model parameters (List of tensor(s))
        
        Returns:
            A flattened 1D tensor
        """
        return tf.concat([tf.reshape(_params, [-1]) for _params in params], axis=0)

    def get_Hv_op(self, v):
        """ 
        Implements a Hessian vector product estimator Hv op defined as the 
        matrix multiplication of the Hessian matrix H with the vector v.
    
        Args:      
            v: Vector to multiply with Hessian (tensor)
        
        Returns:
            Hv_op: Hessian vector product op (tensor)
        """
        cost_gradient = self.flatten(self.grad.values())
        vprod = tf.multiply(cost_gradient, tf.stop_gradient(v))
        Hv_op = self.flatten(tf.gradients(vprod, self.params))
        return Hv_op

    def get_H_op(self, prec):
        """ 
        Implements a full Hessian estimator op by forming p Hessian vector 
        products using HessianEstimator.get_Hv_op(v) for all v's in R^P
        
        Args:
            None
        
        Returns:
            H_op: Hessian matrix op (tensor)
        """
        H_op = tf.map_fn(self.get_Hv_op, tf.eye(self.P, self.P, dtype=prec), dtype=prec)
        return H_op


def greeks(shared, deal_data, mtm):    
    greeks_calc = SensitivitiesEstimator(mtm, shared.calc_greeks)
    deal_data.Calc_res['Greeks_First'] = greeks_calc.grad
    # use this only when all the vols and curves are sparsely represented (check greeks_calc.P)
    deal_data.Calc_res['Greeks_Second'] = greeks_calc.get_H_op(shared.precision)


def interpolate(mtm, shared, time_grid, deal_data):
    if deal_data.Time_dep.interp.size > deal_data.Time_dep.deal_time_grid.size:
        # interpolate it
        mtm = utils.gather_interp_matrix(mtm, deal_data.Time_dep, shared)

    # check if we want to store the mtm value for this instrument
    if deal_data.Calc_res is not None:
        deal_data.Calc_res['Value'] = mtm
        if shared.calc_greeks is not None:
            greeks(shared, deal_data, mtm)

    if time_grid.mtm_time_grid.size > 1:
        # pad it with zeros and return
        return tf.pad(mtm, [[0, time_grid.mtm_time_grid.size - deal_data.Time_dep.interp.size], [0, 0]])
    else:
        return mtm


def getbarrierpayoff(direction, eta, phi, strike, barrier):
    '''
    Function to generate the barrier payoff function using these formulae:
    (import sympy with the necessary symbols to see how to derive the code below)
    
    A = phi * ( spot * sympy.exp ( (b-r)*expiry ) * normcdf ( phi * x1 ) -
            strike * sympy.exp( -r*expiry ) * normcdf ( phi * ( x1 - vol ) ) )
    B = phi * ( spot * sympy.exp ( (b-r)*expiry ) * normcdf ( phi * x2 ) -
            strike * sympy.exp( -r*expiry ) * normcdf ( phi * ( x2 - vol ) ) )
    C = phi * ( spot * sympy.exp ( (b-r)*expiry + log_bar*2*(mu+1) ) * normcdf (eta*y1) -
            strike * sympy.exp ( -r*expiry + log_bar*2*mu ) * normcdf ( eta * ( y1 - vol ) ) )
    D = phi * ( spot * sympy.exp ( (b-r)*expiry + log_bar*2*(mu+1) ) * normcdf (eta*y2) -
            strike * sympy.exp ( -r*expiry + log_bar*2*mu ) * normcdf ( eta * ( y2 - vol ) ) )
            
    E = cash_rebate * sympy.exp ( -r*expiry ) * ( normcdf ( eta * ( x2 - vol ) ) -
        sympy.exp(log_bar*2*mu) * normcdf ( eta * ( y2 - vol ) ) )
    F = cash_rebate * ( sympy.exp ( log_bar*(mu+lam) ) * normcdf (eta*z) +
        sympy.exp (log_bar*(mu-lam)) * normcdf ( eta * ( z - 2*lam*vol) ) )

    This is for single Barrier options and based on Merton, Reiner and Rubinstein.
    '''

    def barrier_option(sigma, expiry, cash_rebate, b, r, spot):

        sigma2 = sigma * sigma
        vol = sigma * np.sqrt(expiry)
        mu = (b - 0.5 * sigma2) / sigma2
        log_bar = tf.log(barrier / spot)
        x1 = tf.log(spot / strike) / vol + (1.0 + mu) * vol
        x2 = tf.log(spot / barrier) / vol + (1.0 + mu) * vol

        y1 = tf.log((barrier * barrier) / (spot * strike)) / vol + (1.0 + mu) * vol
        y2 = log_bar / vol + (1.0 + mu) * vol
        lam = tf.sqrt(mu * mu + 2.0 * r / sigma2)
        z = log_bar / vol + lam * vol
        eta_scale = 0.7071067811865476 * eta
        phi_scale = 0.7071067811865476 * phi
        expiry_r = expiry * r

        if direction == BARRIER_IN:
            if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > barrier) or
                    (phi == OPTION_PUT and eta == BARRIER_DOWN and strike <= barrier)):
                # A+E
                return (cash_rebate * (
                        (0.5 * tf.erfc(eta_scale * (-vol + y2)) - 1.0) * tf.exp(2 * log_bar * mu) +
                        0.5 * tf.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * tf.erfc(phi_scale * x1) - 1.0) * tf.exp(b * expiry) +
                                0.5 * strike * tf.erfc(phi_scale * (vol - x1)))) * tf.exp(-expiry_r)
            elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= barrier) or
                  (phi == OPTION_PUT and eta == BARRIER_DOWN and strike > barrier)):
                # B-C+D+E
                return (cash_rebate * (
                        (0.5 * tf.erfc(eta_scale * (-vol + y2)) - 1.0) * tf.exp(2 * log_bar * mu) +
                        0.5 * tf.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * tf.erfc(phi_scale * x2) - 1.0) * tf.exp(b * expiry) +
                                0.5 * strike * tf.erfc(phi_scale * (vol - x2))) + phi * (
                                spot * (0.5 * tf.erfc(eta_scale * y1) - 1.0) * tf.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1)) - spot * (
                                        0.5 * tf.erfc(eta_scale * y2) - 1.0) *
                                tf.exp(expiry * (b - r) + 2 * log_bar * (mu + 1)) + 0.5 * strike * tf.exp(
                            -expiry_r + 2 * log_bar * mu) * tf.erfc(eta_scale * (vol - y1)) -
                                0.5 * strike * tf.exp(-expiry_r + 2 * log_bar * mu) * tf.erfc(
                            eta_scale * (vol - y2))) * tf.exp(expiry_r)) * tf.exp(-expiry_r)
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > barrier) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike <= barrier)):
                # A-B+D+E
                return (cash_rebate * ((0.5 * tf.erfc(eta_scale * (-vol + y2)) - 1.0) * tf.exp(
                    2 * log_bar * mu) + 0.5 * tf.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * tf.erfc(eta_scale * y2) - 1.0) * tf.exp(expiry * (
                                b - r) + 2 * log_bar * (mu + 1)) + 0.5 * strike * tf.exp(
                            -expiry_r + 2 * log_bar * mu) * tf.erfc(
                            eta_scale * (vol - y2))) * tf.exp(expiry_r) - phi * (spot * (0.5 * tf.erfc(
                    phi_scale * x1) - 1.0) * tf.exp(b * expiry) + 0.5 * strike * tf.erfc(
                    phi_scale * (vol - x1))) + phi * (spot * (0.5 * tf.erfc(
                    phi_scale * x2) - 1.0) * tf.exp(b * expiry) + 0.5 * strike * tf.erfc(
                    phi_scale * (vol - x2)))) * tf.exp(-expiry_r)
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= barrier) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike > barrier)):
                # C+ E
                return (cash_rebate * ((0.5 * tf.erfc(eta_scale * (-vol + y2)) - 1.0) * tf.exp(
                    2 * log_bar * mu) + 0.5 * tf.erfc(eta_scale * (vol - x2))) - phi * (spot * (
                        0.5 * tf.erfc(eta_scale * y1) - 1.0) * tf.exp(expiry * (b - r) + 2 * log_bar * (
                        mu + 1)) + 0.5 * strike * tf.exp(-expiry_r + 2 * log_bar * mu) * tf.erfc(eta_scale * (
                        vol - y1))) * tf.exp(expiry_r)) * tf.exp(-expiry_r)
        else:
            if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > barrier) or
                    (phi == OPTION_PUT and eta == BARRIER_DOWN and strike <= barrier)):
                # F
                return -cash_rebate * ((0.5 * tf.erfc(eta_scale * z) - 1.0) * tf.exp(2 * lam * log_bar) -
                                       0.5 * tf.erfc(eta_scale * (2 * lam * vol - z))) * tf.exp(-log_bar * (lam - mu))
            elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= barrier) or
                  (phi == OPTION_PUT and eta == BARRIER_DOWN and strike > barrier)):
                # A - B + C - D + F
                return (-cash_rebate * ((0.5 * tf.erfc(eta_scale * z) - 1.0) * tf.exp(2 * lam * log_bar)
                                        - 0.5 * tf.erfc(eta_scale * (2 * lam * vol - z))) * tf.exp(expiry_r)
                        + phi * (-spot * (0.5 * tf.erfc(eta_scale * y1) - 1.0) * tf.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1)) + spot * (0.5 * tf.erfc(eta_scale * y2) - 1.0) *
                                 tf.exp(expiry * (b - r) + 2 * log_bar * (mu + 1)) - 0.5 * strike * tf.exp(
                                    -expiry_r + 2 * log_bar * mu) *
                                 tf.erfc(eta_scale * (vol - y1)) + 0.5 * strike * tf.exp(-expiry_r + 2 * log_bar * mu) *
                                 tf.erfc(eta_scale * (vol - y2))) * tf.exp(expiry_r + log_bar * (lam - mu)) +
                        phi * (-spot * (0.5 * tf.erfc(phi_scale * x1) - 1.0) * tf.exp(b * expiry) + spot * (
                                0.5 * tf.erfc(phi_scale * x2) - 1.0) * tf.exp(b * expiry) - 0.5 * strike *
                               tf.erfc(phi_scale * (vol - x1)) + 0.5 * strike * tf.erfc(phi_scale * (vol - x2)))
                        * tf.exp(log_bar * (lam - mu))) * tf.exp(-expiry_r - log_bar * (lam - mu))
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > barrier) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike <= barrier)):
                # B - D + F
                return (-cash_rebate * ((0.5 * tf.erfc(eta_scale * z) - 1.0) * tf.exp(2 * lam * log_bar)
                                        - 0.5 * tf.erfc(eta_scale * (2 * lam * vol - z))) * tf.exp(expiry_r) +
                        phi * (spot * (0.5 * tf.erfc(eta_scale * y2) - 1.0) * tf.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1))
                               + 0.5 * strike * tf.exp(-expiry_r + 2 * log_bar * mu) * tf.erfc(
                                    eta_scale * (vol - y2))) * tf.exp(
                            expiry_r + log_bar * (lam - mu)) - phi * (
                                spot * (0.5 * tf.erfc(phi_scale * x2) - 1.0) * tf.exp(
                            b * expiry) + 0.5 * strike * tf.erfc(phi_scale * (vol - x2))) * tf.exp(
                            log_bar * (lam - mu))) * tf.exp(
                    -expiry_r - log_bar * (lam - mu))
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= barrier) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike > barrier)):
                # A-C+F
                return (-cash_rebate * ((0.5 * tf.erfc(eta_scale * z) - 1.0) * tf.exp(2 * lam * log_bar)
                                        - 0.5 * tf.erfc(eta_scale * (2 * lam * vol - z))) * tf.exp(expiry_r) +
                        phi * (spot * (0.5 * tf.erfc(eta_scale * y1) - 1.0) * tf.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1))
                               + 0.5 * strike * tf.exp(-expiry_r + 2 * log_bar * mu) * tf.erfc(
                                    eta_scale * (vol - y1))) * tf.exp(
                            expiry_r + log_bar * (lam - mu)) - phi * (
                                spot * (0.5 * tf.erfc(phi_scale * x1) - 1.0) * tf.exp(
                            b * expiry) + 0.5 * strike * tf.erfc(phi_scale * (vol - x1))) * tf.exp(
                            log_bar * (lam - mu))) * tf.exp(
                    -expiry_r - log_bar * (lam - mu))

    return barrier_option


def pvbarrieroption(shared, time_grid, deal_data, nominal,
                    spot, b, tau, payoff_currency, invert_moneyness=False):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    factor_dep = deal_data.Factor_dep
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing
    phi = OPTION_CALL if deal_data.Instrument.field['Option_Type'] == 'Call' else OPTION_PUT
    eta = BARRIER_DOWN if 'Down' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_UP
    direction = BARRIER_OUT if 'Out' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_IN
    buy_or_sell = 1.0 if deal_data.Instrument.field['Buy_Sell'] == 'Buy' else -1.0
    barrier = deal_data.Instrument.field['Barrier_Price']
    strike = deal_data.Instrument.field['Strike_Price']
    cash_rebate = deal_data.Instrument.field['Cash_Rebate']

    # get the zero curve
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time[:-1], shared)

    expiry = daycount_fn(tau).astype(shared.precision)
    need_spot_at_expiry = deal_time.shape[0] - expiry.size
    spot_prior, spot_at = tf.split(spot, [expiry.size, need_spot_at_expiry])
    moneyness = strike / spot_prior if invert_moneyness else spot_prior / strike
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep['Barrier_Monitoring']:
        adj_barrier = barrier * tf.exp((2.0 * tf.cast(
            barrier > spot, shared.precision) - 1.0) * sigma * factor_dep['Barrier_Monitoring'])
    else:
        adj_barrier = barrier

    # get the payoff function - should be adj_barrier not barrier - TODO
    barrierOption = getbarrierpayoff(direction, eta, phi, strike, barrier)

    r = tf.squeeze(discounts.gather_weighted_curve(
        shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

    expiry_years = expiry.reshape(-1, 1)
    barrier_payoff = buy_or_sell * nominal * barrierOption(sigma, expiry_years, cash_rebate / nominal,
                                                           b, r, spot_prior)

    if need_spot_at_expiry:
        # work out barrier
        if eta == BARRIER_UP:
            touched = tf.cast(tf.logical_and(
                spot[:-1] < barrier, spot[1:] > barrier), shared.precision)
        else:
            touched = tf.cast(tf.logical_and(
                spot[:-1] > barrier, spot[1:] < barrier), shared.precision)

        # barrier payoff
        barrier_touched = tf.pad(tf.cast(tf.cumsum(touched, axis=0) > 0, shared.precision), [[1, 0], [0, 0]])
        first_touch = barrier_touched[1:] - barrier_touched[:-1]
        # final payoff
        payoff_at = buy_or_sell * tf.nn.relu(phi * (spot_at - strike))

        if direction == BARRIER_IN:
            forward = spot_prior * tf.exp(b * expiry_years)
            payoff_prior = utils.black_european_option(
                forward, strike, sigma, expiry, buy_or_sell, phi, shared) * tf.exp(-r * expiry_years)
            european_part = barrier_touched * (nominal * tf.concat([payoff_prior, payoff_at], axis=0))
            barrier_part = (1.0 - barrier_touched) * tf.pad(
                barrier_payoff, [[0, 1], [0, 0]], constant_values=buy_or_sell * cash_rebate)
            combined = european_part + barrier_part
            # settle cashflows (can only happen at the end)
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], combined[-1])
        else:
            # barrier out
            barrier_part = (1.0 - barrier_touched) * tf.concat([barrier_payoff, nominal * payoff_at], axis=0)
            rebate_part = buy_or_sell * cash_rebate * first_touch
            combined = tf.pad(buy_or_sell * cash_rebate * first_touch, [[1, 0], [0, 0]]) + barrier_part
            # settle cashflows (The one at expiry)
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], barrier_part[-1])
            # settle cashflows (The potential rebate knockout)
            if cash_rebate:
                for cash_index, cash in zip(deal_data.Time_dep.deal_time_grid[1:], tf.unstack(rebate_part)):
                    cash_settle(shared, payoff_currency, cash_index, cash)
    else:
        combined = barrier_payoff

    return combined


def pvonetouchoption(shared, time_grid, deal_data, nominal,
                     spot, b, tau, payoff_currency, invert_moneyness=False):
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing

    eta = BARRIER_DOWN if 'Down' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_UP
    buy_or_sell = 1.0 if deal_data.Instrument.field['Buy_Sell'] == 'Buy' else -1.0
    barrier = deal_data.Instrument.field['Barrier_Price']

    # get the zero curve
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time[:-1], shared)

    expiry = daycount_fn(tau).astype(shared.precision)
    need_spot_at_expiry = deal_time.shape[0] - expiry.size
    spot_prior, spot_at = tf.split(spot, [expiry.size, need_spot_at_expiry])
    moneyness = barrier / spot_prior if invert_moneyness else spot_prior / barrier
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep['Barrier_Monitoring']:
        adj_barrier = barrier * tf.exp((2.0 * tf.cast(
            barrier > spot_prior, shared.precision) - 1.0) * sigma * factor_dep['Barrier_Monitoring'])
    else:
        adj_barrier = barrier

    r = tf.squeeze(discounts.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

    # one touch payoff
    expiry_years = expiry.reshape(-1, 1)
    root_tau = np.sqrt(expiry_years)
    mu = b / sigma - 0.5 * sigma
    log_vol = tf.log(barrier / spot_prior) / sigma
    barrovert = log_vol / root_tau
    eta_scale = 0.7071067811865476 * eta

    if deal_data.Instrument.field['Payment_Timing'] == 'Expiry':
        muroot = mu * root_tau
        d1 = muroot - barrovert
        d2 = -muroot - barrovert
        payoff = tf.exp(-r * expiry_years) * 0.5 * (
                tf.erfc(eta_scale * d1) + tf.exp(2.0 * mu * log_vol) * tf.erfc(eta_scale * d2))
    elif deal_data.Instrument.field['Payment_Timing'] == 'Touch':
        lamb = tf.sqrt(tf.nn.relu(mu * mu + 2.0 * r))
        lambroot = lamb * root_tau
        d1 = lambroot - barrovert
        d2 = -lambroot - barrovert
        payoff = 0.5 * (tf.exp((mu - lamb) * log_vol) * tf.erfc(eta_scale * d1) +
                        tf.exp((mu + lamb) * log_vol) * tf.erfc(eta_scale * d2))

    if need_spot_at_expiry:
        # barrier check
        if eta == BARRIER_UP:
            touched = tf.cast(tf.logical_and(
                spot[:-1] < barrier, spot[1:] > barrier), shared.precision)
        else:
            touched = tf.cast(tf.logical_and(
                spot[:-1] > barrier, spot[1:] < barrier), shared.precision)

        barrier_touched = tf.pad(tf.cast(tf.cumsum(touched, axis=0) > 0, shared.precision), [[1, 0], [0, 0]])
        first_touch = barrier_touched[1:] - barrier_touched[:-1]
        barrier_part = (1.0 - barrier_touched) * tf.pad(payoff, [[0, 1], [0, 0]])

        if deal_data.Instrument.field['Payment_Timing'] == 'Touch':
            touch_part = tf.pad(first_touch, [[1, 0], [0, 0]])
            combined = buy_or_sell * nominal * (touch_part + barrier_part)
            for cash_index, cash in zip(deal_data.Time_dep.deal_time_grid[1:], tf.unstack(first_touch)):
                cash_settle(shared, payoff_currency, cash_index, buy_or_sell * nominal * cash)
        else:
            # Expiry
            rebate_part = barrier_touched * tf.pad(tf.exp(-r * expiry_years), [[0, 1], [0, 0]], constant_values=1.0)
            combined = buy_or_sell * nominal * (rebate_part + barrier_part)
            # settle cashflows (The one at expiry)
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], combined[-1])
    else:
        combined = buy_or_sell * nominal * payoff

    return combined


def pricer_float_cashflows(all_resets, cashflows, factor_dep, time_slice, shared):
    margin = cashflows[:, utils.CASHFLOW_INDEX_FloatMargin] * cashflows[:, utils.CASHFLOW_INDEX_Year_Frac]
    all_int = cashflows[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) * all_resets + margin.reshape(1, -1, 1)

    return all_int, margin


def pricer_cap(all_resets, cashflows, factor_dep, time_slice, shared):
    mn_option = all_resets - cashflows[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1)
    expiry = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](
        cashflows[:, utils.CASHFLOW_INDEX_Start_Day] - time_slice)

    # note that the Year Frac is averaged - all the cashflows are supposed to have the same year frac
    # (but practically not - should be ok to do this)

    vols = utils.calc_tenor_cap_time_grid_vol_rate(
        factor_dep['VolSurface'], mn_option, expiry,
        cashflows[:, utils.CASHFLOW_INDEX_Year_Frac].mean(), shared)

    payoff = utils.black_european_option(
        all_resets, cashflows[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1),
        vols, expiry, 1.0, 1.0, shared)

    all_int = cashflows[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) * payoff
    margin = np.zeros(cashflows.shape[0], dtype=shared.precision)

    return all_int, margin


def pricer_floor(all_resets, cashflows, factor_dep, time_slice, shared):
    mn_option = all_resets - cashflows[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1)
    expiry = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](
        cashflows[:, utils.CASHFLOW_INDEX_Start_Day] - time_slice)

    # note that the Year Frac is averaged - all the cashflows are supposed to have the same year frac
    # (but practically not - should be ok to do this)

    vols = utils.calc_tenor_cap_time_grid_vol_rate(
        factor_dep['VolSurface'], mn_option, expiry,
        cashflows[:, utils.CASHFLOW_INDEX_Year_Frac].mean(), shared)

    payoff = utils.black_european_option(
        all_resets, cashflows[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1),
        vols, expiry, 1.0, -1.0, shared)

    all_int = cashflows[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) * payoff
    margin = np.zeros(cashflows.shape[0], dtype=shared.precision)

    return all_int, margin


def pvfloatcashflowlist(shared, time_grid, deal_data, cashflow_pricer, mtm_currency=None, settle_cash=True):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    # first precalc all past resets
    resets = factor_dep['Cashflows'].Resets
    known_resets = resets.known_resets(shared.simulation_batch)
    sim_resets = resets.schedule[(resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                 (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                  deal_time[:, utils.TIME_GRID_MTM].max())]
    if mtm_currency:
        # precalc the FX forwards
        sim_fx_forward = utils.calc_fx_forward(
            mtm_currency, factor_dep['Currency'],
            sim_resets[:, utils.RESET_INDEX_Start_Day], sim_resets[:, :utils.RESET_INDEX_Scenario + 1],
            shared, only_diag=True)

        known_fx = factor_dep['Cashflows'].known_resets(
            shared.simulation_batch, index=utils.CASHFLOW_INDEX_FXResetValue,
            filter_index=utils.CASHFLOW_INDEX_Start_Day)

        # fetch fx rates - note that there is a slight difference between this and the spot fx rate
        old_fx_rates = tf.squeeze(
            tf.concat([tf.stack(known_fx), sim_fx_forward]
                      if known_fx
                      else sim_fx_forward, axis=0), axis=1)

    forwards = utils.calc_time_grid_curve_rate(factor_dep['Forward'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    old_resets = utils.calc_time_grid_curve_rate(factor_dep['Forward'],
                                                 sim_resets[:, :utils.RESET_INDEX_Scenario + 1],
                                                 shared)

    delta_start = (sim_resets[:, utils.RESET_INDEX_Start_Day] -
                   sim_resets[:, utils.RESET_INDEX_Reset_Day]).reshape(-1, 1)
    delta_end = (sim_resets[:, utils.RESET_INDEX_End_Day] -
                 sim_resets[:, utils.RESET_INDEX_Reset_Day]).reshape(-1, 1)

    reset_weights = (sim_resets[:, utils.RESET_INDEX_Weight] /
                     sim_resets[:, utils.RESET_INDEX_Accrual]).reshape(-1, 1, 1)

    reset_values = tf.expm1(old_resets.gather_weighted_curve(
        shared, delta_end, delta_start)) * reset_weights \
        if sim_resets.any() else tf.zeros([0, 1, shared.simulation_batch],
                                          dtype=shared.precision)

    # fetch all fixed resets 
    old_resets = tf.squeeze(
        tf.concat([tf.stack(known_resets), reset_values]
                  if known_resets
                  else reset_values, axis=0), axis=1)

    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        cashflows = factor_dep['Cashflows'].merged()[start_index[index]:]

        cash_pmts, cash_index, cash_counts = np.unique(cashflows[:, utils.CASHFLOW_INDEX_Pay_Day],
                                                       return_index=True, return_counts=True)

        reset_offset = factor_dep['Cashflows'].offsets[start_index[index]][1]
        pmts_offset = cash_index + (cash_counts - 1)

        time_ofs = 0
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        reset_block = resets.schedule[reset_offset:]
        reset_ofs, reset_count = np.unique(resets.split_block_resets(
            reset_offset, time_block), return_counts=True)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block,
                                                  future_pmts, shared)

        # do we need to split the forward block further? 
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        # empty list for payments
        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            time_slice = time_block[time_ofs:time_ofs + size].reshape(-1, 1)
            future_starts = reset_block[offset:, utils.RESET_INDEX_Start_Day] - time_slice
            future_ends = reset_block[offset:, utils.RESET_INDEX_End_Day] - time_slice
            future_weights = (reset_block[offset:, utils.RESET_INDEX_Weight]
                              / reset_block[offset:, utils.RESET_INDEX_Accrual]).reshape(1, -1, 1)
            future_resets = tf.expm1(forward_rates.gather_weighted_curve(
                shared, future_ends, future_starts)) * future_weights

            # now deal with past resets
            past_resets = tf.tile(
                tf.expand_dims(old_resets[reset_offset:reset_offset + offset], axis=0)
                , [size, 1, 1])
            all_resets = tf.concat([past_resets, future_resets], axis=1)

            # handle cashflows that don't pay interest (e.g. bullets)
            if cashflows[:, utils.CASHFLOW_INDEX_NumResets].all():
                reset_cashflows = cashflows
            else:
                reset_cash_index = np.where(cashflows[:, utils.CASHFLOW_INDEX_NumResets])[0]
                reset_cashflows = cashflows[reset_cash_index]
                cash_counts *= (cashflows[:, utils.CASHFLOW_INDEX_NumResets] >= 1).astype(np.int32)
                cash_index = reset_cash_index.searchsorted(cash_index)

            if mtm_currency:
                # now deal with fx rates - note that there should only be 1 reset per cashflow
                future_fx_resets = utils.calc_fx_forward(
                    mtm_currency, factor_dep['Currency'],
                    cashflows[offset:, utils.CASHFLOW_INDEX_FXResetDate],
                    discount_block.time_grid[time_ofs:time_ofs + size], shared)

                past_fx_resets = tf.tile(
                    tf.expand_dims(old_fx_rates[reset_offset:reset_offset + offset], axis=0)
                    , [size, 1, 1])
                all_fx_resets = tf.concat([past_fx_resets, future_fx_resets], axis=1)

                # calculate the Nominal in the correct currency
                Pi = all_fx_resets * cashflows[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)
                Pi_1 = tf.pad(Pi[:, 1:, :], [[0, 0], [0, 1], [0, 0]])

            time_ofs += size

            # now we can price the cashflows
            all_int, all_margin = cashflow_pricer(all_resets, reset_cashflows, factor_dep, time_slice, shared)

            # check if there are a different number of resets per cashflow 
            if (cash_counts.min() != cash_counts.max()):
                interest = tf.pad(all_int, [[0, 0], [0, 1], [0, 0]])
                nominal = np.append(reset_cashflows[:, utils.CASHFLOW_INDEX_Nominal], 0.0)
                margin = np.append(all_margin, 0.0)
            else:
                interest = all_int
                margin = all_margin
                nominal = reset_cashflows[:, utils.CASHFLOW_INDEX_Nominal]

            default_offst = np.ones(cash_index.size, dtype=np.int32) * (interest.shape[1].value - 1)
            total = 0.0

            for i in range(cash_counts.max()):
                offst = default_offst.copy()
                offst[cash_counts > i] = i + cash_index[cash_counts > i]
                int_i = tf.gather(interest, offst, axis=1)

                if mtm_currency:
                    total += int_i * Pi + (Pi - Pi_1)
                elif factor_dep['CompoundingMethod'] == 'None':
                    total += nominal[offst].reshape(1, -1, 1) * int_i
                elif factor_dep['CompoundingMethod'] == 'Include_Margin':
                    total += (total + nominal[offst].reshape(1, -1, 1)) * int_i
                elif factor_dep['CompoundingMethod'] == 'Flat':
                    total += (int_i * nominal[offst].reshape(1, -1, 1)) + total * (
                            int_i - margin[offst].reshape(1, -1, 1))
                else:
                    raise Exception('Floating cashflow list method not implemented')

            payments.append(total + cashflows[pmts_offset, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1))

        # now finish the payments
        all_payments = tf.concat(payments, axis=0) if len(payments) > 1 else payments[0]

        # settle any cashflows
        if settle_cash:
            cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]),
                        all_payments[-1][0])
        # add it to the list
        mtm_list.append(tf.reduce_sum(all_payments * discount_rates, axis=1))

    return tf.concat(mtm_list, axis=0, name='mtm')


def pvfixedcashflows(shared, time_grid, deal_data, ignore_fixed_rate=False, settle_cash=True):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
    settlement_amt = factor_dep.get('Settlement_Amount', 0.0)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    start_index, counts = np.unique(cash_start_idx, return_counts=True)

    for index, [discount_block] in enumerate(utils.split_counts([discounts], counts, shared)):
        cashflows = factor_dep['Cashflows'].schedule[start_index[index]:]

        cash_pmts, cash_index, cash_counts = np.unique(cashflows[:, utils.CASHFLOW_INDEX_Pay_Day],
                                                       return_index=True, return_counts=True)
        # payment times            
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block,
                                                  future_pmts, shared)
        # is this a forward?
        if settlement_amt:
            settlement = settlement_amt * tf.squeeze(utils.calc_discount_rate(
                discount_block,
                (factor_dep['Settlement_Date'] - time_block).reshape(-1, 1), shared), axis=1)
        else:
            settlement = 0.0

        # empty list for payments
        all_int = (1.0 if ignore_fixed_rate else
                   cashflows[:, utils.CASHFLOW_INDEX_FixedRate]) * cashflows[:, utils.CASHFLOW_INDEX_Year_Frac]
        payments = np.zeros_like(cash_index, dtype=shared.precision)

        if cash_counts.min() != cash_counts.max():
            interest = np.append(all_int, 0.0)
            nominal = np.append(cashflows[:, utils.CASHFLOW_INDEX_Nominal], 0.0)
            fixed_amt = np.append(cashflows[:, utils.CASHFLOW_INDEX_FixedAmt], 0.0)
        else:
            interest = all_int
            nominal = cashflows[:, utils.CASHFLOW_INDEX_Nominal]
            fixed_amt = cashflows[:, utils.CASHFLOW_INDEX_FixedAmt]

        default_offst = np.ones(cash_index.size, dtype=np.int32) * (interest.size - 1)

        for i in range(cash_counts.max()):
            offst = default_offst.copy()
            offst[cash_counts > i] = i + cash_index[cash_counts > i]
            int_i = interest[offst]

            if factor_dep.get('Compounding', False):
                payments += (payments + nominal[offst]) * int_i + fixed_amt[offst]
            else:
                payments += int_i * nominal[offst] + fixed_amt[offst]

        # add to the mtm               
        mtm_list.append(tf.reduce_sum(payments.reshape(1, -1, 1) * discount_rates, axis=1) - settlement)

        # settle any cashflows
        if settle_cash:
            if factor_dep.get('Settlement_Date') is not None:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, factor_dep['Settlement_Date']),
                            mtm_list[-1][-1])
            else:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]),
                            payments[0])

    return tf.concat(mtm_list, axis=0, name='mtm')


def pvindexcashflows(shared, time_grid, deal_data, settle_cash=True):
    def calc_index(schedule, sim_schedule):
        weight = schedule[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
        dates = schedule[np.newaxis, :, utils.RESET_INDEX_Reset_Day] - \
                last_pub_block[:, np.newaxis, utils.RESET_INDEX_Reset_Day]
        index_t = tf.expand_dims(last_index_block, axis=1) / utils.calc_discount_rate(
            forecast_block, dates, shared)

        # split if necessary
        if dates[dates < 0].any():
            future_indices = (dates >= 0).all(axis=1).argmin()
            future_index_t, past_index_t = tf.split(
                index_t, [future_indices, dates.shape[0] - future_indices])

            mixed_indices_t = []
            for mixed_dates, mixed_indices in zip(dates[future_indices:], tf.unstack(past_index_t)):
                future_resets = mixed_dates.size - (mixed_dates[::-1] > 0).argmin()
                past_resets_t, future_resets_t = tf.split(
                    mixed_indices, [future_resets, mixed_dates.size - future_resets])
                mixed_indices_t.append(
                    tf.concat([sim_schedule[:future_resets], future_resets_t], axis=0))

            values = weight * tf.concat([future_index_t, tf.stack(mixed_indices_t)], axis=0)
        else:
            values = weight * index_t

        if resets_per_cf > 1:
            return tf.reduce_sum(
                tf.reshape(values,
                           (last_pub_block.shape[0], -1, resets_per_cf, shared.simulation_batch)), axis=2)
        else:
            return values

    def get_index_val(cash_index_vals, schedule, sim_schedule, resets_per_cf, offset):
        if cash_index_vals[cash_index_vals < 0].any():
            num_known = cash_index_vals[cash_index_vals > 0].size
            known_indices = tf.tile(
                cash_index_vals[cash_index_vals > 0].reshape(1, -1, 1).astype(shared.precision),
                (last_pub_block.shape[0], 1, shared.simulation_batch))
            reset_offset = resets_per_cf * (offset + num_known)
            if num_known:
                return tf.concat(
                    [known_indices,
                     calc_index(schedule[reset_offset:], sim_schedule[reset_offset:])], axis=1)
            else:
                return calc_index(schedule[reset_offset:], sim_schedule[reset_offset:])
        else:
            return cash_index_vals.reshape(1, -1, 1)

    def filter_resets(resets, index):
        known_resets = resets.known_resets(shared.simulation_batch)
        sim_resets = resets.schedule[(resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                     (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                      deal_time[:, utils.TIME_GRID_MTM].max())]
        old_resets = utils.calc_time_grid_spot_rate(index, sim_resets[:, :utils.RESET_INDEX_Scenario + 1], shared)
        return tf.concat([tf.concat(known_resets, axis=0), old_resets], axis=0) if known_resets else old_resets

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    resets_per_cf = 2 if (
            factor_dep['IndexMethod'] in [utils.CASHFLOW_METHOD_IndexReferenceInterpolated3M,
                                          utils.CASHFLOW_METHOD_IndexReferenceInterpolated4M]) else 1

    last_published = factor_dep['Cashflows'].Resets.schedule[deal_data.Time_dep.deal_time_grid]
    last_published_index = utils.calc_time_grid_spot_rate(factor_dep['PriceIndex'], deal_time, shared)
    index_forecast = utils.calc_time_grid_curve_rate(factor_dep['ForecastIndex'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    all_base_resets = filter_resets(factor_dep['Base_Resets'], factor_dep['PriceIndex'])
    all_final_resets = filter_resets(factor_dep['Final_Resets'], factor_dep['PriceIndex'])

    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)
    last_pub_blocks = np.split(last_published, start_counts.cumsum())

    for index, (forecast_block, discount_block, last_index_block) in enumerate(
            utils.split_counts([index_forecast, discounts, last_published_index], start_counts, shared)):

        last_pub_block = last_pub_blocks[index]
        cashflows = factor_dep['Cashflows'].schedule[start_index[index]:]

        cash_pmts, cash_index, cash_counts = np.unique(cashflows[:, utils.CASHFLOW_INDEX_Pay_Day],
                                                       return_index=True, return_counts=True)

        # payment times            
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block,
                                                  future_pmts, shared)

        cash_base_index_vals = factor_dep['Cashflows'].offsets[start_index[index]:, utils.CASHFLOW_OFFSET_BaseReference]
        all_base_index_vals = get_index_val(cash_base_index_vals, factor_dep['Base_Resets'].schedule, all_base_resets,
                                            resets_per_cf, start_index[index])
        cash_final_index_vals = factor_dep['Cashflows'].offsets[start_index[index]:,
                                utils.CASHFLOW_OFFSET_FinalReference]
        all_final_index_vals = get_index_val(cash_final_index_vals, factor_dep['Final_Resets'].schedule,
                                             all_final_resets,
                                             resets_per_cf, start_index[index])

        # empty list for payments
        interest = (
                cashflows[:, utils.CASHFLOW_INDEX_FixedRate] * cashflows[:, utils.CASHFLOW_INDEX_Year_Frac]). \
            reshape(1, -1, 1)
        growth = cashflows[:, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1) * (
                all_final_index_vals / all_base_index_vals)
        payment = cashflows[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1) * growth * interest
        payments = None

        for i in range(cash_counts.min()):
            offst = cash_index + i
            int_i = tf.cast(tf.gather(payment, offst, axis=1), shared.precision)
            if payments is None:
                payments = int_i
            else:
                payments += int_i

        # add it to the list
        mtm_list.append(tf.reduce_sum(payments * discount_rates, axis=1))

        # settle any cashflows
        if settle_cash:
            if factor_dep['Settlement_Date'] is not None:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, factor_dep['Settlement_Date']),
                            mtm_list[-1][-1])
            else:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]),
                            payments[-1][0])

    return tf.concat(mtm_list, axis=0, name='mtm')


def pvenergycashflows(shared, time_grid, deal_data):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    # first precalc all past resets
    resets = factor_dep['Cashflows'].Resets
    known_resets = resets.known_resets(shared.simulation_batch)
    sim_resets = resets.schedule[(resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                 (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                  deal_time[:, utils.TIME_GRID_MTM].max())]
    all_resets = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], sim_resets, shared)
    all_fx_spot = utils.calc_fx_cross(factor_dep['ForwardFX'][0], factor_dep['CashFX'][0],
                                      sim_resets, shared)

    reset_values = tf.expand_dims(
        tf.squeeze(all_resets.gather_weighted_curve(
            shared, sim_resets[:, utils.RESET_INDEX_End_Day].reshape(-1, 1), multiply_by_time=False),
            axis=1) * all_fx_spot, axis=1) \
        if sim_resets.any() else tf.zeros([0, 1, shared.simulation_batch], dtype=shared.precision)

    old_resets = tf.squeeze(tf.concat([tf.stack(known_resets), reset_values]
                                      if known_resets
                                      else reset_values, axis=0), axis=1)

    forwards = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        cashflows = factor_dep['Cashflows'].schedule[start_index[index]:]
        cash_offset = factor_dep['Cashflows'].offsets[start_index[index]:]

        cash_pmts, cash_index, cash_counts = np.unique(cashflows[:, utils.CASHFLOW_INDEX_Pay_Day],
                                                       return_index=True, return_counts=True)

        reset_offset = factor_dep['Cashflows'].offsets[start_index[index]][1]

        time_ofs = 0
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        reset_block = resets.schedule[reset_offset:]
        reset_ofs, reset_count = np.unique(resets.split_block_resets(
            reset_offset, time_block), return_counts=True)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block,
                                                  future_pmts, shared)

        # we need to split the forward block further
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        # empty list for payments
        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            # past resets
            past_resets = tf.tile(
                tf.expand_dims(old_resets[reset_offset:reset_offset + offset], axis=0), [size, 1, 1])

            # future resets
            future_ends = np.tile(reset_block[offset:, utils.RESET_INDEX_End_Day], [size, 1])

            if future_ends.any():
                future_resets = forward_rates.gather_weighted_curve(
                    shared, future_ends, multiply_by_time=False)

                forwardfx = utils.calc_fx_forward(
                    factor_dep['ForwardFX'], factor_dep['CashFX'],
                    reset_block[offset:, utils.RESET_INDEX_Start_Day],
                    discounts.time_grid[time_ofs:time_ofs + size], shared)

                all_resets = tf.concat([past_resets, future_resets * forwardfx], axis=1)
            else:
                all_resets = past_resets

            time_ofs += size

            all_payoffs = all_resets * reset_block[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
            payoff = tf.stack(
                [tf.reduce_sum(x, axis=1) for x in tf.split(all_payoffs, cash_offset[:, 0], axis=1)], axis=1)

            # now we can price the cashflows
            payment = cashflows[:, utils.CASHFLOW_INDEX_Nominal] * (
                    cashflows[:, utils.CASHFLOW_INDEX_Start_Mult] * payoff +
                    cashflows[:, utils.CASHFLOW_INDEX_FloatMargin])

            payments.append(payment)

        # now finish the payments
        all_payments = tf.concat(payments, axis=0)

        # settle any cashflows - use this tf.gather(all_payments, time_grid.mtm_time_grid.searchsorted(cash_pmts))!!
        cash_settle(shared, factor_dep['SettleCurrency'],
                    np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), all_payments[-1][0])
        # add it to the list
        mtm_list.append(tf.reduce_sum(all_payments * discount_rates, axis=1))

    return tf.concat(mtm_list, axis=0, name='mtm')


def pvcreditcashflows(shared, time_grid, deal_data):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    surv = utils.calc_time_grid_curve_rate(factor_dep['Name'], deal_time, shared)

    start_index, counts = np.unique(cash_start_idx, return_counts=True)
    for index, (discount_block, surv_block) in enumerate(
            utils.split_counts([discounts, surv], counts, shared)):
        cashflows = factor_dep['Cashflows'].schedule[start_index[index]:]

        cash_pmts, cash_index = np.unique(cashflows[:, utils.CASHFLOW_INDEX_Pay_Day],
                                          return_index=True)
        cash_sts = np.unique(cashflows[:, utils.CASHFLOW_INDEX_Start_Day])

        # payment times            
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        start_pmts = (cash_sts.reshape(1, -1) - time_block.reshape(-1, 1)).clip(0, np.inf)

        Dt_T = utils.calc_discount_rate(discount_block,
                                        future_pmts, shared)
        Dt_Tm1 = utils.calc_discount_rate(discount_block,
                                          start_pmts, shared)

        survival_T = utils.calc_discount_rate(surv_block, future_pmts, shared,
                                              multiply_by_time=False)
        survival_t = utils.calc_discount_rate(surv_block, start_pmts, shared,
                                              multiply_by_time=False)

        interest = cashflows[cash_index, utils.CASHFLOW_INDEX_FixedRate] * cashflows[
            cash_index, utils.CASHFLOW_INDEX_Year_Frac]
        premium = (interest[cash_index] *
                   cashflows[cash_index, utils.CASHFLOW_INDEX_Nominal]).reshape(1, -1, 1) * survival_T * Dt_T
        credit = (1.0 - factor_dep['Recovery_Rate']) * cashflows[cash_index, utils.CASHFLOW_INDEX_Nominal].reshape(
            1, -1, 1) * 0.5 * (Dt_T + Dt_Tm1) * (survival_t - survival_T)

        # settle any cashflows - TODO
        # cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]),
        #                payments[0])

        mtm_list.append(tf.reduce_sum(credit - premium, axis=1))

    return tf.concat(mtm_list, axis=0, name='mtm')


def pvequitycashflows(shared, time_grid, deal_data):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    eq_spot = utils.calc_time_grid_spot_rate(factor_dep['Equity'], deal_time, shared)
    cash, divis = factor_dep['Flows']

    # needed for grouping 
    cash_start_idx = np.searchsorted(cash.schedule[:, utils.CASHFLOW_INDEX_Start_Day],
                                     deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_end_idx = np.searchsorted(cash.schedule[:, utils.CASHFLOW_INDEX_End_Day],
                                   deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_pay_idx = cash.get_cashflow_start_index(deal_time)

    # would prefer to use np.unique here but it does not seem to work with tuples  
    all_idx = OrderedDict()
    for idx in zip(cash_start_idx, cash_end_idx, cash_pay_idx):
        all_idx[idx] = all_idx.setdefault(idx, 0) + 1

    # first precalc all past resets
    samples = cash.Resets
    known_sample = samples.known_resets(shared.simulation_batch, include_today=True)
    known_divs = samples.known_resets(shared.simulation_batch, utils.RESET_INDEX_Weight,
                                      include_today=True)
    sim_samples = samples.schedule[(samples.schedule[:, utils.RESET_INDEX_Value] == 0.0) &
                                   (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                    deal_time[:, utils.TIME_GRID_MTM].max())]

    # now calculate the dividends
    div_samples = divis.Resets

    h_t0_t1 = utils.calc_realized_dividends(
        factor_dep['Equity'], factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
        div_samples, shared, offsets=divis.offsets[:, 0])

    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    repo_discounts = utils.calc_time_grid_curve_rate(factor_dep['Equity_Zero'], deal_time, shared)
    past_samples = utils.calc_time_grid_spot_rate(factor_dep['Equity'],
                                                  sim_samples[:, :utils.RESET_INDEX_Scenario + 1],
                                                  shared)
    # fetch all fixed resets
    if past_samples.shape[1].value!=shared.simulation_batch:
        past_samples = tf.tile(past_samples, [1, shared.simulation_batch])
    
    all_samples = tf.concat(
        [tf.concat(known_sample, axis=0), past_samples], axis=0) if known_sample else past_samples

    cashflows = cash.schedule
    all_index, all_counts = zip(*all_idx.items())

    for index, (discount_block, repo_block, eq_block) in enumerate(
            utils.split_counts([discounts, repo_discounts, eq_spot], np.array(all_counts), shared)):

        start_idx, end_idx, pay_idx = all_index[index]

        cashflow_start = cashflows[start_idx:, utils.CASHFLOW_INDEX_Start_Day].reshape(1, -1)
        cashflow_pay = cashflows[pay_idx:, utils.CASHFLOW_INDEX_Pay_Day].reshape(1, -1)

        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cashflow_pay - time_block.reshape(-1, 1)
        payoffs = []
        discount_rates = utils.calc_discount_rate(discount_block,
                                                  future_pmts, shared)

        # need equity forwards for start and end cashflows
        if pay_idx < end_idx:
            future_end = cashflows[pay_idx, utils.CASHFLOW_INDEX_End_Day].reshape(1, -1)

            St0 = all_samples[2 * pay_idx]
            St1 = all_samples[2 * pay_idx + 1]
            Ht0_t1 = 0 * h_t0_t1[pay_idx]

            payoff = (cashflows[pay_idx, utils.CASHFLOW_INDEX_End_Mult].reshape(1, -1, 1) * St1 -
                      cashflows[pay_idx, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1) * St0 +
                      cashflows[pay_idx, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1) * Ht0_t1)
            payment = payoff * cashflows[pay_idx, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1)

            if factor_dep['PrincipleNotShares']:
                payment /= St0

            payoffs.append(payment)
            # settle cashflow if necessary
            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cashflow_pay[0][0]),
                        payment[0][0])

        if end_idx < start_idx:
            cashflow_end = cashflows[end_idx, utils.CASHFLOW_INDEX_End_Day].reshape(1, -1)
            future_end = cashflow_end - time_block.reshape(-1, 1)
            forward_end = utils.calc_eq_forward(factor_dep['Equity'], factor_dep['Equity_Zero'],
                                                factor_dep['Dividend_Yield'],
                                                np.squeeze(cashflow_end, axis=0),
                                                discount_block.time_grid, shared)
            discount_end = utils.calc_discount_rate(
                repo_block, future_end, shared)

            St0 = all_samples[end_idx * 2]
            Ht0_t = utils.calc_realized_dividends(
                factor_dep['Equity'], factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                utils.calc_dividend_samples(time_block[0], time_block[-1], time_grid), shared)
            payoff = ((cashflows[end_idx, utils.CASHFLOW_INDEX_End_Mult] -
                       cashflows[end_idx, utils.CASHFLOW_INDEX_Dividend_Mult]) * forward_end +
                      cashflows[end_idx, utils.CASHFLOW_INDEX_Dividend_Mult] * (
                          tf.expand_dims(eq_block + Ht0_t, axis=1)) / discount_end -
                      cashflows[end_idx, utils.CASHFLOW_INDEX_Start_Mult] * St0)
            if factor_dep['PrincipleNotShares']:
                payoff /= tf.reshape(St0, [1, 1, -1])

            payoffs.append(payoff * cashflows[end_idx, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1))

        if cashflow_start.any():
            cashflow_end = cashflows[start_idx:, utils.CASHFLOW_INDEX_End_Day].reshape(1, -1)
            future_start = cashflow_start - time_block.reshape(-1, 1)
            future_end = cashflow_end - time_block.reshape(-1, 1)
            forward_start = utils.calc_eq_forward(factor_dep['Equity'], factor_dep['Equity_Zero'],
                                                  factor_dep['Dividend_Yield'], np.squeeze(cashflow_start, axis=0),
                                                  discount_block.time_grid, shared)
            forward_end = utils.calc_eq_forward(factor_dep['Equity'], factor_dep['Equity_Zero'],
                                                factor_dep['Dividend_Yield'], np.squeeze(cashflow_end, axis=0),
                                                discount_block.time_grid, shared)
            discount_start = utils.calc_discount_rate(
                repo_block, future_start, shared)
            discount_end = utils.calc_discount_rate(
                repo_block, future_end, shared)

            if factor_dep['PrincipleNotShares']:
                factor1 = forward_end / forward_start
                factor2 = 1.0
            else:
                factor1 = forward_end
                factor2 = forward_start

            payoff = (cashflows[start_idx:, utils.CASHFLOW_INDEX_End_Mult] -
                      cashflows[start_idx:, utils.CASHFLOW_INDEX_Dividend_Mult]).reshape(1, -1, 1) * factor1 + (
                             cashflows[start_idx:, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1) *
                             (discount_start / discount_end) -
                             cashflows[start_idx:, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1)) * factor2

            payoffs.append(payoff * cashflows[start_idx:, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1))

        # now finish the payments
        payments = tf.concat(payoffs, axis=1) if len(payoffs) > 1 else payoffs[0]
        mtm_list.append(tf.reduce_sum(payments * discount_rates, axis=1))

    return tf.concat(mtm_list, axis=0, name='mtm')


def pvfixedleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0], shared.Report_Currency,
                                 deal_time, shared)

    mtm = FX_rep * pvfixedcashflows(shared, time_grid, deal_data)

    return mtm


def pvenergyleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency,
                                 deal_time, shared)

    mtm = FX_rep * pvenergycashflows(shared, time_grid, deal_data)

    return mtm


def pvfloatleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0], shared.Report_Currency,
                                 deal_time, shared)

    mtm = FX_rep * pvfloatcashflowlist(shared, time_grid, deal_data, pricer_float_cashflows)

    return mtm


def pvcapleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0], shared.Report_Currency,
                                 deal_time, shared)
    return FX_rep * pvfloatcashflowlist(shared, time_grid, deal_data, pricer_cap)


def pvfloorleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0], shared.Report_Currency,
                                 deal_time, shared)
    return FX_rep * pvfloatcashflowlist(shared, time_grid, deal_data, pricer_floor)


def pvindexleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                 deal_time, shared)

    mtm = FX_rep * pvindexcashflows(shared, time_grid, deal_data)

    return mtm


def pvcdsleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                 deal_time, shared)
    mtm = FX_rep * pvcreditcashflows(shared, time_grid, deal_data)

    return mtm


def pvequityleg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                 deal_time, shared)

    mtm = FX_rep * pvequitycashflows(shared, time_grid, deal_data)

    return mtm


def pvfxbarrieroption(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    nominal = deal_data.Instrument.field['Underlying_Amount']
    payoff_currency = deal_data.Instrument.field[deal_data.Instrument.field['Payoff_Currency']]

    curr_curve = utils.calc_time_grid_curve_rate(
        deal_data.Factor_dep['Currency'][1], deal_time[:-1], shared)
    und_curr_curve = utils.calc_time_grid_curve_rate(
        deal_data.Factor_dep['Underlying_Currency'][1], deal_time[:-1], shared)

    spot = utils.calc_fx_cross(deal_data.Factor_dep['Underlying_Currency'][0],
                               deal_data.Factor_dep['Currency'][0], deal_time, shared)

    # need to adjust if there's just 1 timepoint - i.e. base reval
    if time_grid.mtm_time_grid.size > 1:
        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
    else:
        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

    b = tf.squeeze(curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False) -
                   und_curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

    fx_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0], shared.Report_Currency,
                                 deal_time, shared)

    pv = pvbarrieroption(
        shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency,
        invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'])

    mtm = fx_rep * pv

    return mtm


def pveqbarrieroption(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    nominal = deal_data.Instrument.field['Units']
    payoff_currency = deal_data.Instrument.field[deal_data.Instrument.field['Currency']]

    eq_zer_curve = utils.calc_time_grid_curve_rate(
        deal_data.Factor_dep['Equity_Zero'], deal_time[:-1], shared)
    eq_div_curve = utils.calc_time_grid_curve_rate(
        deal_data.Factor_dep['Dividend_Yield'], deal_time[:-1], shared)

    spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)

    # need to adjust if there's just 1 timepoint - i.e. base reval
    if time_grid.mtm_time_grid.size > 1:
        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
    else:
        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

    b = tf.squeeze(eq_zer_curve.gather_weighted_curve(shared, tau.reshape(-1, 1),
                                               multiply_by_time=False) -
                   eq_div_curve.gather_weighted_curve(shared, tau.reshape(-1, 1),
                                               multiply_by_time=False), axis=1)

    fx_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                 deal_time, shared)

    pv = pvbarrieroption(
        shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency,
        invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'])

    mtm = fx_rep * pv

    return mtm


def pvfxonetouch(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    nominal = deal_data.Instrument.field['Cash_Payoff']
    payoff_currency = deal_data.Instrument.field[deal_data.Instrument.field['Payoff_Currency']]
    fx_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0],
                                 shared.Report_Currency, deal_time, shared)

    curr_curve = utils.calc_time_grid_curve_rate(
        deal_data.Factor_dep['Currency'][1], deal_time[:-1], shared)
    und_curr_curve = utils.calc_time_grid_curve_rate(
        deal_data.Factor_dep['Underlying_Currency'][1], deal_time[:-1], shared)

    spot = utils.calc_fx_cross(deal_data.Factor_dep['Underlying_Currency'][0],
                               deal_data.Factor_dep['Currency'][0], deal_time, shared)

    # need to adjust if there's just 1 timepoint - i.e. base reval
    if time_grid.mtm_time_grid.size > 1:
        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
    else:
        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

    b = tf.squeeze(curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False) -
                   und_curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

    pv = pvonetouchoption(
        shared, time_grid, deal_data, nominal,
        spot, b, tau, payoff_currency, invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'])

    mtm = fx_rep * pv

    return mtm
