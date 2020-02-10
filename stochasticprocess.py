##########################################################################
# Copyright (C) 2016-2017  Shuaib Osman (sosman@investec.co.za)
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
##########################################################################

# import standard libraries
import itertools
from collections import OrderedDict

# 3rd party libraries
import numpy as np
import scipy.interpolate

# Internal modules
import utils
import tensorflow as tf

from instruments import get_fx_zero_rate_factor


class GBMAssetPriceModel(object):
    """The Geometric Brownian Motion Stochastic Process"""

    documentation = (
        'Asset Pricing', ['The spot price of an equity or FX rate can be modelled as Geometric Brownian Motion (GBM).',
                          'The model is specified as follows:',
                          '',
                          '$$ dS = \\mu S dt + \\sigma S dZ$$',
                          '',
                          'Its final form is:',
                          '',
                          '$$ S = exp \\Big( (\\mu-\\frac{1}{2}\\sigma^2)t + \\sigma dW(t)  \\Big ) $$',
                          '',
                          'Where:',
                          '',
                          '- $S$ is the spot price of the asset',
                          '- $dZ$ is the standard Brownian motion',
                          '- $\\mu$ is the constant drift of the asset',
                          '- $\\sigma$ is the constant volatility of the asset',
                          '- $dW(t)$ is a standard Wiener Process'])

    def __init__(self, factor, param, implied_factor=None):
        self.factor = factor
        self.param = param

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        dt = np.diff(np.hstack(([0], time_grid.time_grid_years)))
        var = self.param['Vol'] * self.param['Vol'] * dt
        drift = shared.precision(self.param['Drift'] * dt - 0.5 * var).reshape(-1, 1)
        vol = shared.precision(np.sqrt(var)).reshape(-1, 1)

        # store a reference to the current tensor
        self.spot = tensor

        with tf.name_scope(self.__class__.__name__):
            self.f1 = tf.cumsum(drift + vol * shared.t_random_numbers[
                                              process_ofs, :time_grid.scen_time_grid.size], axis=0, name='F1')

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    def theoretical_mean_std(self, t):
        mu = self.factor.current_value() * np.exp(self.param['Drift'] * t)
        var = mu * mu * (np.exp(t * self.param['Vol'] ** 2) - 1.0)
        return mu, np.sqrt(var)

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        with tf.name_scope(self.__class__.__name__):
            return self.spot * tf.exp(self.f1)


class GBMAssetPriceCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, num_business_days=252.0, vol_cuttoff=0.5, drift_cuttoff=0.1):
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Log', num_business_days=num_business_days)
        mu = (stats['Drift'] + 0.5 * (stats['Volatility'] ** 2)).values[0]
        sigma = stats['Volatility'].values[0]

        return utils.CalibrationInfo(
            {'Vol': np.clip(sigma, 0.01, vol_cuttoff), 'Drift': np.clip(mu, -drift_cuttoff, drift_cuttoff)},
            [[1.0]], delta)


class GBMAssetPriceTSModelImplied(object):
    """The Geometric Brownian Motion Stochastic Process with implied drift and vol"""

    documentation = ('Asset Pricing', [
        'GBM with constant drift and vol may not be suited to model risk-neutral asset prices. A generalization that',
        'allows this would be to modify the volatility $\\sigma(t)$ and $\\mu(t)$ to be functions of time $t$.',
        'This can be specified as follows:',
        '',
        '$$ \\frac{dS(t)}{S(t)} = (r(t)-q(t)-v(t)\\sigma(t)\\rho) dt + \\sigma(t) dW(t)$$',
        '',
        'Note that no risk premium curve is captured. Its final form is:',
        '',
        '$$ S(t+\\delta) = F(t,t+\\delta)exp \\Big(\\rho(C(t+\\delta)-C(t)) -\\frac{1}{2}(V(t+\\delta)) - V(t))\
         + \\sqrt{V(t+\\delta) - V(t)}Z  \\Big) $$',
        '',
        'Where:',
        '',
        '- $\\sigma(t)$ is the volatility of the asset at time $t$',
        '- $v(t)$ is the *Quanto FX Volatility* of the asset at time $t$. $\\rho$ is then the *Quanto FX Correlation*',
        '- $V(t) = \\int_{0}^{t} \\sigma(s)^2 ds$',
        '- $C(t) = \\int_{0}^{t} v(s)\\sigma(s) ds$',
        '- $r$ is the interest rate in the asset currency',
        '- $q$ is the yield on the asset (If S is a foreign exchange rate, q is the foreign interest rate)',
        '- $F(t,t+\\delta)$ is the forward asset price at time t',
        '- $S$ is the spot price of the asset',
        '- $Z$ is a sample from the standard normal distribution',
        '- $\\delta$ is the increment in timestep between samples',
        '',
        'In the case that the $S(t)$ represents an FX rate, this can be further simplified to:',
        '',
        '$$S(t)=S(0)\\beta(t)exp\\Big(\\frac{1}{2}\\bar\\sigma(t)^2t+\\int_0^t\\sigma(s)dW(s)\\Big)$$',
        '',
        'Here $C(t)=\\bar\\sigma(t)^2t, \\beta(t)=exp\\Big(\\int_0^t(r(s)-q(s))ds\\Big), \\rho=-1 and v(t)=\\sigma(t)$'
    ])

    def __init__(self, factor, param, implied_factor=None):
        self.factor = factor
        self.param = param
        self.implied = implied_factor

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        vols = self.implied.param['Vol'].array
        self.V = utils.cumulative_integration(lambda s: np.interp(s, *vols.T) ** 2, time_grid.time_grid_years)
        # calc the incremental vol
        delta_V = np.insert(np.diff(self.V), 0, 0).reshape(-1, 1).astype(shared.precision)
        self.delta_scen_t = np.insert(
            np.diff(time_grid.scen_time_grid), 0, 0).reshape(-1, 1).astype(shared.precision)
        # store a reference to the current tensor
        self.spot = tensor
        # store the scenario grid
        self.scen_grid = time_grid.scenario_grid

        with tf.name_scope(self.__class__.__name__):
            self.f1 = tf.cumsum(tf.sqrt(delta_V) * shared.t_random_numbers[
                                                   process_ofs, :time_grid.scen_time_grid.size], axis=0, name='F1')

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        # this is valid for FX factors only - can change this to equities etc. by changing the get_XXX_factor
        # function below
        self.r_t = get_fx_zero_rate_factor(self.factor.get_domestic_currency(None), static_ofs, stoch_ofs,
                                           all_tenors, all_factors)
        self.q_t = get_fx_zero_rate_factor(factor.name, static_ofs, stoch_ofs, all_tenors, all_factors)

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        with tf.name_scope(self.__class__.__name__):
            rt = utils.calc_time_grid_curve_rate(self.r_t, self.scen_grid, shared_mem)
            qt = utils.calc_time_grid_curve_rate(self.q_t, self.scen_grid, shared_mem)

            rt_rates = rt.gather_weighted_curve(shared_mem, self.delta_scen_t)
            qt_rates = qt.gather_weighted_curve(shared_mem, self.delta_scen_t)

            drift = tf.cumsum(tf.squeeze(rt_rates - qt_rates, axis=1), axis=0)

            return self.spot * tf.exp(drift - 0.5 * self.V.reshape(-1, 1) + self.f1)


class GBMPriceIndexModel(object):
    """The Geometric Brownian Motion Stochastic Process for Price Indices - can contain adjustments for seasonality"""

    documentation = ('Inflation',
                     ['The model is specified as follows:',
                      '',
                      '$$ F(t) = exp \\Big( (\\mu-\\frac{\\sigma^2}{2})t + \\sigma W(t) \\Big)$$',
                      '',
                      'Where:',
                      '',
                      '- $\\mu$ is the drift of the price index',
                      '- $\\sigma$ is the volatility of the price index',
                      '- $W(t)$ is a standard Wiener Process under the real-world measure',
                      '',
                      'Note that the simulation of this model is identical to plain Geometric Brownian Motion with the',
                      'exception of modifying the scenario grid to coincide with allowable publication dates obtained',
                      'by the corresponding Price Index'])

    def __init__(self, factor, param, implied_factor=None):
        self.factor = factor
        self.param = param

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        scenario_time_grid = np.array([(x - self.factor.param['Last_Period_Start']).days for x in
                                       self.factor.get_last_publication_dates(ref_date,
                                                                              time_grid.scen_time_grid.tolist())],
                                      dtype=np.float64)
        dt = np.diff(np.hstack(([0], scenario_time_grid / utils.DAYS_IN_YEAR)))
        var = self.param['Vol'] * self.param['Vol'] * dt
        drift = shared.precision(self.param['Drift'] * dt - 0.5 * var).reshape(-1, 1)
        vol = shared.precision(np.sqrt(var)).reshape(-1, 1)

        # store a reference to the current tensor
        self.spot = tensor

        with tf.name_scope(self.__class__.__name__):
            self.f1 = tf.cumsum(drift + vol * shared.t_random_numbers[process_ofs, :dt.size], axis=0, name='F1')

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        with tf.name_scope(self.__class__.__name__):
            return self.spot * tf.exp(self.f1)


class GBMPriceIndexCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, num_business_days=252.0):
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Log', num_business_days=num_business_days)
        mu = (stats['Drift'] + 0.5 * (stats['Volatility'] ** 2)).values[0]
        sigma = stats['Volatility'].values[0]

        return utils.CalibrationInfo({'Vol': sigma, 'Drift': mu, 'Seasonal_Adjustment': None}, [[1.0]], delta)


class HullWhite2FactorImpliedInterestRateModel(object):
    """Hull white 2 factor implied interest rate model for risk neutral simulation of yield curves"""

    documentation = (
        'Interest Rates',
        ['This is a generalization of the 1 factor Hull White model. There are 2 correlated risk ',
         'factors where the $i^{th}$ factor has a volatility curve $\\sigma_i(t)$, constant reversion',
         'speed $\\alpha_i$ and market price of risk $\\lambda_i$.',
         '',
         'Final form of the model is:',
         '$$ D(t,T) = \\frac{D(0,T)}{D(0,t)}exp\\Big(-\\frac{1}{2}\\sum_{i,j=1}^2\\rho_{ij}A_{ij}(t,'
         'T)-\\sum_{i=1}^2B_i(T-t)e^{-\\alpha_it}(Y_i(t) -\\tilde\\rho_i K_i(t) + \\lambda_i H_i('
         't))\\Big) $$',
         '',
         'Where:',
         '',
         '- $B_i(t) = \\frac{(1-e^{-\\alpha_i t})}{\\alpha_i}$, $Y_i(t)=\\int\\limits_0^t e^{\\alpha_i '
         's}\\sigma_i (s) dW_i(s)$, $W_1$ and $W_2$ are correlated Weiner Processes with correlation '
         '$\\rho$ ($\\rho_{ij}=\\rho$ if $i \\neq j$ else 1)',
         '- $A_{ij}(t,T)=B_i(T-t)B_j(T-t)e^{-(\\alpha_i+\\alpha_j)}J_{ij}(t)+\\frac{B_i(T-t)}{\\alpha_j}('
         'e^{-\\alpha_it}I_{ij}(t)-e^{-(\\alpha_i+\\alpha_j)t}J_{ij}(t))+\\frac{B_j(T-t)}{\\alpha_i}(e^{'
         '-\\alpha_jt}I_{ji}(t)-e^{-(\\alpha_i+\\alpha_j)t}J_{ji}(t))$',
         '- $H_i(t)=\\int\\limits_0^t e^{\\alpha_is}\\sigma_i(s)ds$',
         '- $I_{ij}(t)=\\int\\limits_0^t e^{\\alpha_is}\\sigma_i(s)\\sigma_j(s)ds$',
         '- $J_{ij}(t)=\\int\\limits_0^t e^{(\\alpha_i+\\alpha_j)s}\\sigma_i(s)\\sigma_j(s)ds$',
         '- $K_i(t)=\\int\\limits_0^t e^{\\alpha_is}\\sigma_i(s)v(s)ds$',
         '',
         'If the rate and base currencies match, $v(t)=0$ and $\\tilde\\rho_i=0$. Otherwise, $v(t)$ is',
         'the volatility of the rate currency (in base currency) and $\\tilde\\rho_i$ is the correlation',
         'between the FX rate and the $i^{th}$ factor. The increment $Y(t_{k+1})-Y(t_k)$ (where',
         '$0=t_0,t_1,t_2...$ corresponds to the simulation grid) is gaussian with zero mean and covariance',
         'Matrix $C_{ij}=\\rho_{ij}(J_{ij}(t_{k+1})-J_{ij}(t_k))$.',
         '',
         'The cholesky decomposition of $C$ is $$L=\\begin{pmatrix} \\sqrt C_{11} & 0 \\\\ \\frac{C_{12}}{'
         '\\sqrt C_{11}} & \\sqrt {C_{22}-\\frac{C_{12}^2}{C_{11}} } \\\\ \\end{pmatrix}$$',
         'The increment is simulated using $LZ$ where $Z$ is a 2D vector of independent normals at',
         'time step $k$.'])

    def __init__(self, factor, param, implied_factor, clip=(1e-5, 3.0)):
        self.factor = factor
        self.param = param
        self.implied = implied_factor
        self.clip = clip
        self.grid_index = None
        self.BtT = None
        self.C = None
        self.f1 = None
        self.f2 = None

    @staticmethod
    def num_factors():
        return 2

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):

        def interpolate_vol(t, vols, vols_tenor):

            if isinstance(vols, np.ndarray):
                # no tensorflow needed - don't even bother caching
                dt = np.diff(t)
                interp = np.interp(t, vols_tenor, vols)
                return dt, interp[:-1], np.diff(interp) / dt
            else:
                key_code = ('Hull_White', vols.name, tuple(t))

            if key_code not in shared.t_Buffer:
                # linear interpolation of the vols at vol_tenor to the grid t
                dt = np.diff(t)
                interp = utils.interpolate_tensor(t, vols_tenor, vols)
                grad = (interp[1:] - interp[:-1]) / dt
                shared.t_Buffer[key_code] = (dt, interp[:-1], grad)

            # interpolated vol
            return shared.t_Buffer[key_code]

        def calc_H(time_grid, a, vols, vols_tenor):
            # obtained via sympy.simplify(sympy.integrate(sympy.exp(a*s)*(v+m*(s-t)), (s,t,t+dt)))
            # and leaving out the division by a**2 until later

            H = lambda a, t, v, dt, m: ((-a * v + m) * tf.exp(a * t) +
                                        (-a * m * t + a * m * (dt + t) + a * v - m) * tf.exp(a * (dt + t)))

            t = np.union1d(0.0, time_grid)

            # interpolated vol
            dt, interp_vol, m = interpolate_vol(t, vols, vols_tenor)
            integral = tf.cumsum(H(a, t[:-1], interp_vol, dt, m)) / a ** 2

            if 0.0 in time_grid:
                return tf.pad(integral, [[1, 0]])
            else:
                return integral

        def calc_IJK(time_grid, a, volsi, volsj, volsi_tenor, volsj_tenor):

            def IJK(a, t, vi, vj, dt, mi, mj):
                # sympy.simplify(sympy.integrate(sympy.exp(a*s)*(vi+mi*(s-t))*(vj+mj*(s-t)), (s,t,t+dt)))
                # and leaving the division by a**3 until later (and rearranging terms)

                a2, dt2, mi_mj, mj_vi, mi_vj, vi_vj = a * a, dt * dt, mi * mj, mj * vi, mi * vj, vi * vj

                return ((a2 * (
                        dt2 * mi_mj + dt * mi_vj + dt * mj_vi + vi_vj) - 2 * a * dt * mi_mj + 2 * mi_mj - a * mi_vj
                         - a * mj_vi) * tf.exp(a * dt) - a2 * vi_vj + a * mi_vj + a * mj_vi - 2 * mi_mj) * tf.exp(a * t)

            t = np.union1d(0.0, time_grid)

            dt, interp_volsi, mi = interpolate_vol(t, volsi, volsi_tenor)
            dt, interp_volsj, mj = interpolate_vol(t, volsj, volsj_tenor)

            integral = tf.cumsum(IJK(a, t[:-1], interp_volsi, interp_volsj, dt, mi, mj)) / a ** 3

            if 0.0 in time_grid:
                return tf.pad(integral, [[1, 0]])
            else:
                return integral

        # get the factor's tenor points
        factor_tenor = self.factor.get_tenor()

        # store the forward curve    
        self.fwd_curve = utils.calc_curve_forwards(self.factor, tensor, time_grid, shared, ref_date)

        # calculate known functions 
        alpha = [tf.cast(tf.clip_by_value(implied_tensor['Alpha_1'][0], self.clip[0], self.clip[1]), tf.float64),
                 tf.cast(tf.clip_by_value(implied_tensor['Alpha_2'][0], self.clip[0], self.clip[1]), tf.float64)]
        lam = [self.param['Lambda_1'], self.param['Lambda_2']]
        corr = tf.cast(implied_tensor['Correlation'], tf.float64)
        vols = [
            tf.cast(implied_tensor['Sigma_1'], tf.float64, name='Cast.' + implied_tensor['Sigma_1'].name.split(':')[0]),
            tf.cast(implied_tensor['Sigma_2'], tf.float64, name='Cast.' + implied_tensor['Sigma_2'].name.split(':')[0])]
        vols_tenor = [self.implied.param['Sigma_1'].array[:, 0],
                      self.implied.param['Sigma_2'].array[:, 0]]

        # Check if the curve is not the same as the base currency
        if self.implied.param['Quanto_FX_Volatility']:
            quantofx = self.implied.param['Quanto_FX_Volatility'].array
            quantofxcorr = self.implied.get_quanto_correlation(corr, vols)
        else:
            quantofx = np.zeros((1, 2))
            quantofxcorr = [0.0, 0.0]

        H = [calc_H(time_grid.time_grid_years, alpha[i], vols[i], vols_tenor[i]) for i in range(2)]
        I = [[calc_IJK(time_grid.time_grid_years, alpha[i], vols[i], vols[j], vols_tenor[i], vols_tenor[j])
              for j in range(2)] for i in range(2)]
        J = [[calc_IJK(time_grid.time_grid_years, alpha[i] + alpha[j], vols[i], vols[j], vols_tenor[i], vols_tenor[j])
              for j in range(2)] for i in range(2)]
        K = [calc_IJK(time_grid.time_grid_years, alpha[i], vols[i], quantofx[:, 1], vols_tenor[i], quantofx[:, 0])
             for i in range(2)]

        # now calculate the At
        AtT = tf.zeros([time_grid.time_grid_years.size, factor_tenor.size], tf.float64)
        BtT = [(1.0 - tf.exp(-alpha[i] * factor_tenor)) / alpha[i] for i in range(2)]
        CtT = []
        rho = [[1.0, corr], [corr, 1.0]]
        t = time_grid.time_grid_years.reshape(-1, 1)

        for i, j in itertools.product(range(2), range(2)):
            first_part = tf.exp(-(alpha[i] + alpha[j]) * t) * tf.reshape(J[i][j], (-1, 1))
            second_part = tf.exp(-alpha[i] * t) * tf.reshape(I[i][j], (-1, 1)) - first_part
            third_part = tf.exp(-alpha[j] * t) * tf.reshape(I[j][i], (-1, 1)) - first_part

            # get the covariance
            CtT.append(rho[i][j] * J[i][j])

            # all together now
            AtT += rho[i][j] * tf.matmul(
                tf.concat([first_part, second_part, third_part], axis=1),
                tf.stack([BtT[j] * BtT[i], BtT[i] / alpha[j], BtT[j] / alpha[i]]))

        # calculate the Covariance matrix
        c11 = tf.pad(CtT[0][1:] - CtT[0][:-1], [[1, 0]], constant_values=CtT[0][0])
        c12 = tf.pad(CtT[1][1:] - CtT[1][:-1], [[1, 0]], constant_values=CtT[1][0])
        c22 = tf.pad(CtT[3][1:] - CtT[3][:-1], [[1, 0]], constant_values=CtT[3][0])

        # make sure it's ok
        c11_ok = tf.cast(c11 > 0.0, tf.float64)
        c22_ok = tf.cast(c22 > 0.0, tf.float64)
        c12_ok = tf.cast((c12 * c12) < (c11 * c22), tf.float64)

        # fudge factors to prevent underflow
        epsilon = tf.ones_like(c11, dtype=tf.float64) * 1e-12
        zero = tf.zeros_like(c11, dtype=tf.float64)

        # calc the covariance matrix
        C11 = c11_ok * c11 + (1.0 - c11_ok) * epsilon
        C22 = c22_ok * c22 + (1.0 - c22_ok) * epsilon
        C12 = c12_ok * c12

        L = tf.stack([c11_ok * tf.sqrt(C11), zero,
                      c11_ok * (C12 / tf.sqrt(C11)),
                      c12_ok * tf.sqrt(C22 - (C12 * C12) / C11)])

        # get the correlation through time - this will break if the cholesky is not Positive definite
        self.C = tf.cast(tf.reshape(L, [2, 2, -1]), shared.precision)
        # intermediate results
        self.BtT = [tf.cast(tf.reshape(Bi, (-1, 1)), shared.precision) for Bi in BtT]
        self.YtT = [tf.cast(tf.exp(-alpha[i] * t), shared.precision) for i in range(2)]
        self.KtT = [tf.cast(quantofxcorr[i] * tf.reshape(K[i], (-1, 1)), shared.precision) for i in range(2)]
        self.HtT = [tf.cast(lam[i] * tf.reshape(H[i], (-1, 1)), shared.precision) for i in range(2)]
        self.alpha = [tf.cast(alpha[i], shared.precision) for i in range(2)]

        # needed for factor calcs later
        self.F1 = tf.reshape(self.C[0, 0], [-1, 1])
        self.F2 = tf.expand_dims(self.C[1], axis=2)

        # store the grid points used if necessary
        if len(time_grid.scenario_grid) != time_grid.time_grid_years.size:
            self.grid_index = time_grid.scen_time_grid.searchsorted(time_grid.scenario_grid[:, utils.TIME_GRID_MTM])

        self.drift = tf.expand_dims(self.fwd_curve + 0.5 * tf.cast(AtT, shared.precision), 2)

        if hasattr(shared, 't_random_numbers'):
            # only run this if we have random numbers
            with tf.name_scope(self.__class__.__name__):
                self.calc_factors(shared.t_random_numbers[process_ofs, :time_grid.scen_time_grid.size],
                                  shared.t_random_numbers[process_ofs:process_ofs + 2, :time_grid.scen_time_grid.size])

    def calc_stoch_deflator(self, time_grid, curve_index, shared):
        cached_tensor, interpolation_params = utils.cache_interpolation(shared, curve_index[0], self.stoch_drift[:-1])
        tensor = utils.TensorBlock(curve_index, [cached_tensor], [interpolation_params], time_grid)
        self.deflation_drift = tensor.gather_weighted_curve(shared, self.stoch_dt * utils.DAYS_IN_YEAR,
                                                            multiply_by_time=False)

    def calc_points(self, rate, points, time_grid, shared, multiply_by_time=True):
        """ Much slower way to evaluate points at time t - but more accurate and memory efficient"""
        t = rate[utils.FACTOR_INDEX_Daycount](points)
        BtT = [tf.expand_dims((1.0 - tf.exp(-self.alpha[i] * t)), 2) / self.alpha[i] for i in range(2)]
        drift_at_grid = utils.gather_scenario_interp(self.drift, time_grid, shared)
        drift = utils.interpolate_curve(drift_at_grid, rate, None, points, 0)
        f1 = utils.gather_scenario_interp(self.f1, time_grid, shared)
        f2 = utils.gather_scenario_interp(self.f2, time_grid, shared)
        stoch_component = BtT[0] * f1 + BtT[1] * f2
        fwd_curve = drift + stoch_component

        if multiply_by_time:
            return fwd_curve
        else:
            return fwd_curve / t

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    def calc_factors(self, factor1, factor1and2):
        self.f1 = tf.expand_dims((tf.cumsum(factor1 * self.F1, axis=0, name='F1')
                                  - self.KtT[0] + self.HtT[0]) * self.YtT[0], axis=1)
        self.f2 = tf.expand_dims((tf.cumsum(tf.reduce_sum(factor1and2 * self.F2, axis=0), axis=0, name='F2')
                                  - self.KtT[1] + self.HtT[1]) * self.YtT[1], axis=1)

    @property
    def correlation_name(self):
        return 'HWImpliedInterestRate', [('F1',), ('F2',)]

    def generate(self, shared_mem, random_sample=None):
        if random_sample is not None:
            self.calc_factors(random_sample[0], random_sample[:2])

        with tf.name_scope(self.__class__.__name__):
            # check if we need deflators
            if self.grid_index is not None:
                # only generate curves for the given grid
                f1 = tf.gather(self.f1, self.grid_index)
                f2 = tf.gather(self.f2, self.grid_index)
                drift = tf.gather(self.drift, self.grid_index)
            else:
                f1 = self.f1
                f2 = self.f2
                drift = self.drift

            # generate curves based on 
            stoch_component = self.BtT[0] * f1 + self.BtT[1] * f2
            return (drift + stoch_component) / (
                self.factor.get_tenor().astype(shared_mem.precision).reshape(-1, 1))


class HullWhite1FactorInterestRateModel(object):
    """Hull White 1 factor model

    """

    documentation = ('Interest Rates', [
        'The instantaneous spot rate (or short rate) which governs the evolution of the yield curve is modeled as:',
        '',
        '$$ dr(t) = (\\theta (t)-\\alpha r(t) - v(t)\\sigma(t)\\rho)dt + \\sigma(t) dW^*(t)$$',
        '',
        'Where:',
        '',
        '- $\\sigma (t)$ is a deterministic volatility curve',
        '- $\\alpha$ is a constant mean reversion speed',
        '- $\\theta (t)$ is a deterministic curve derived from the vol, mean reversion and initial discount factors',
        '- $v(t)$ is the quanto FX volatility and $\\rho$ is the quanto FX correlation',
        '- $dW^*(t)$ is the risk neutral Wiener process related to the real-world Wiener Process $dW(t)$',
        'by $dW^*(t)=dW(t)+\\lambda dt$ where $\\lambda$ is the market price of risk (assumed to be constant)',
        '',
        'Final form of the model is:',
        '$$ D(t,T) = \\frac{D(0,T)}{D(0,t)}exp\\Big(-\\frac{1}{2}A(t,T)-B(T-t)e^{-\\alpha t}(Y(t) -\\rho K(t) + '
        '\\lambda H(t))\\Big)$$',
        '',
        'Where:',
        '',
        '- $B(t) = \\frac{(1-e^{-\\alpha t})}{\\alpha}, Y(t)=\\int\\limits_0^t e^{\\alpha s}\\sigma (s) dW$',
        '- $A(t,T)=\\frac{B(T-t)e^{-\\alpha T}}{\\alpha}(2I(t)-(e^{-\\alpha t}+e^{-\\alpha T})J(t))$',
        '- $H(t) = \\int\\limits_0^t e^{\\alpha s}\\sigma (s)ds $',
        '- $I(t) = \\int\\limits_0^t e^{\\alpha s}{\\sigma (s)}^2ds $',
        '- $J(t) = \\int\\limits_0^t e^{2\\alpha s}{\\sigma (s)}^2ds $',
        '- $K(t) = \\int\\limits_0^t e^{\\alpha s}v(s){\\sigma (s)}ds $',
        '',
        'The simulation of the random increment $Y(t_{k+1})-Y(t_k)$ (where $0=t_0,t_1,t_2,...$',
        'represents the simulation grid) is normal with zero mean and variance $J(t_{k+1})-J(t_k)$'])

    def __init__(self, factor, param, implied_factor=None):
        self.factor = factor
        self.param = param
        self.H = None
        self.I = None
        self.J = None
        self.K = None

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # ensures that tenors used are the same as the price factor
        factor_tenor = self.factor.get_tenor()
        alpha = self.param['Alpha']

        # store the forward curve    
        self.fwd_curve = utils.calc_curve_forwards(self.factor, tensor, time_grid, shared, ref_date)

        # Really should implement this . . .
        quantofx = self.param['Quanto_FX_Volatility'].array.T if self.param['Quanto_FX_Volatility'] else np.zeros(
            (1, 2))
        quantofxcorr = self.param.get('Quanto_FX_Correlation', 0.0)

        # grab the vols
        vols = self.param['Sigma'].array

        # calculate known functions
        if len(vols) == 1:
            const_vol = vols[0][1]
            tmp = (1.0 / alpha) * (np.exp(alpha * time_grid.time_grid_years) - 1.0)
            self.H = const_vol * tmp
            self.I = const_vol * const_vol * tmp
            self.J = const_vol * const_vol * (1.0 / (2.0 * alpha)) * (
                    np.exp(2.0 * alpha * time_grid.time_grid_years) - 1.0)
            self.K = const_vol * utils.cumulative_integration(lambda s: np.exp(alpha * s) * np.interp(s, *quantofx.T),
                                                              time_grid.time_grid_years)
        else:
            self.H = utils.cumulative_integration(lambda s: np.exp(alpha * s) * np.interp(s, *vols.T),
                                                  time_grid.time_grid_years)
            self.I = utils.cumulative_integration(lambda s: np.exp(alpha * s) * (np.interp(s, *vols.T) ** 2),
                                                  time_grid.time_grid_years)
            self.J = utils.cumulative_integration(lambda s: np.exp(2.0 * alpha * s) * (np.interp(s, *vols.T) ** 2),
                                                  time_grid.time_grid_years)
            self.K = utils.cumulative_integration(
                lambda s: np.exp(alpha * s) * np.interp(s, *vols.T) * np.interp(s, *quantofx.T),
                time_grid.time_grid_years)

        # Now precalculate the A and B matrices
        BtT = (1.0 - np.exp(-alpha * factor_tenor)) / alpha
        AtT = np.array([(BtT / alpha) * np.exp(-alpha * t) * (
                2.0 * It - (np.exp(-alpha * t) + np.exp(-alpha * (t + factor_tenor))) * Jt) for (It, Jt, t) in
                        zip(self.I, self.J, time_grid.time_grid_years)])

        # get the deltas
        delta_KtT = np.hstack((0.0, quantofxcorr * np.diff(np.exp(-alpha * time_grid.time_grid_years) * self.K)))
        delta_HtT = np.hstack(
            (0.0, self.param['Lambda'] * np.diff(np.exp(-alpha * time_grid.time_grid_years) * self.H)))
        delta_var = np.diff(np.exp(-2.0 * alpha * time_grid.time_grid_years) * self.J)

        # now force any negative/zero vars to be the smallest +ve one - this is a hack - should just fix the integrals
        delta_var[delta_var <= 0.0] = delta_var[delta_var > 0].min()
        delta_vol = np.sqrt(np.hstack((0.0, delta_var)))

        self.AtT = AtT.astype(shared.precision)
        self.BtT = BtT.astype(shared.precision)

        with tf.name_scope(self.__class__.__name__):
            self.f1 = tf.cumsum(shared.t_random_numbers[process_ofs, :time_grid.scen_time_grid.size] *
                                delta_vol.reshape(-1, 1) - delta_KtT[0] + delta_HtT[0], axis=0, name='F1')

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    @property
    def correlation_name(self):
        return 'HWInterestRate', [('F1',)]

    def generate(self, shared_mem):
        with tf.name_scope(self.__class__.__name__):
            stoch_component = self.BtT.reshape(-1, 1) * tf.expand_dims(self.f1, 1)
            fwd_component = tf.expand_dims(self.fwd_curve + 0.5 * self.AtT, 2)
            return (fwd_component + stoch_component) / (
                self.factor.get_tenor().astype(shared_mem.precision).reshape(-1, 1))


class HWInterestRateCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Diff',
                                                          num_business_days=num_business_days, max_alpha=4.0)
        # alpha                       = np.percentile(stats['Mean Reversion Speed'], 50)#.mean()
        alpha = stats['Mean Reversion Speed'].mean()
        sigma = stats['Reversion Volatility'].mean()
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo(
            {'Lambda': 0.0, 'Alpha': alpha, 'Sigma': utils.Curve([], [(0.0, sigma)]), 'Quanto_FX_Correlation': 0.0,
             'Quanto_FX_Volatility': 0.0}, correlation_coef, delta)


class HWHazardRateModel(object):
    """Hull White 1 factor hazard Rate model"""

    documentation = ('Survival Rates',
                     ['The Hull-White instantaneous hazard rate process is modeled as:',
                      '',
                      '$$ dh(t) = (\\theta (t)-\\alpha h(t))dt + \\sigma dW^*(t)$$',
                      '',
                      'All symbols defined as per Hull White 1 factor for interest rates.'
                      '',
                      'The final form of the model is',
                      '',
                      '$$ S(t,T) = \\frac{S(0,T)}{S(0,t)}exp\\Big(-\\frac{1}{2}A(t,T)-\\sigma B(T-t)(Y(t) + B(t)'
                      '\\lambda)\\Big)$$',
                      '',
                      'Where:',
                      '',
                      '- $B(t) = \\frac{1-e^{-\\alpha t}}{\\alpha}$, $Y(t) \\sim N(0, \\frac{1-e^{-2 \\alpha t}}{2'
                      '\\alpha})$',
                      '- $A(t,T)=\\sigma^2 B(T-t)\\Big(B(T-t)\\frac{B(2t)}{2}+B(t)^2\\Big)$'])

    def __init__(self, factor, param, implied_factor=None):
        self.factor = factor
        self.param = param

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        alpha = self.param['Alpha']
        factor_tenor = self.factor.get_tenor()

        # store the forward curve    
        self.fwd_curve = utils.calc_curve_forwards(self.factor, tensor, time_grid, shared, ref_date, mul_time=False)
        Bt = ((1.0 - np.exp(-alpha * time_grid.time_grid_years)) / alpha).reshape(-1, 1)
        B2t = ((1.0 - np.exp(-2.0 * alpha * time_grid.time_grid_years)) / alpha).reshape(-1, 1)
        self.BtT = ((1.0 - np.exp(-alpha * factor_tenor)) / alpha).reshape(1, -1)
        self.AtT = np.square(self.param['Sigma']) * self.BtT * (self.BtT * B2t + (Bt * Bt))
        var = (np.square(self.param['Sigma']) / (2.0 * alpha)) * (
                1.0 - np.exp(-2.0 * alpha * time_grid.time_grid_years))
        vol = np.sqrt(np.diff(np.insert(var, 0, 0, axis=0), axis=0))

        with tf.name_scope(self.__class__.__name__):
            delta_Bt = np.diff(np.insert(Bt, 0, 0)).reshape(-1, 1)
            self.f1 = tf.cumsum(shared.t_random_numbers[process_ofs, :time_grid.scen_time_grid.size] *
                                vol.reshape(-1, 1) + self.param['Sigma'] * self.param['Lambda'] * delta_Bt, axis=0,
                                name='F1')

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    @property
    def correlation_name(self):
        return 'HullWhiteProcess', [()]

    def generate(self, shared_mem):
        with tf.name_scope(self.__class__.__name__):
            stoch_component = self.BtT.reshape(1, -1, 1) * tf.expand_dims(self.f1, 1)
            fwd_component = tf.expand_dims(self.fwd_curve + 0.5 * self.AtT, 2)
            return fwd_component + stoch_component


class HWHazardRateCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Diff',
                                                          num_business_days=num_business_days, max_alpha=4.0)
        alpha = stats['Mean Reversion Speed'].mean()
        sigma = stats['Reversion Volatility'].values[0] / tenor[0]
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo({'Alpha': alpha, 'Sigma': sigma, 'Lambda': 0}, correlation_coef, delta)


class CSForwardPriceModel(object):
    """Clewlow-Strickland Model"""

    documentation = ('Energy Pricing',
                     ['For commodity/Energy deals, the Forward price is modeled directly. For each settlement date T,',
                      'the SDE for the forward price is:',
                      '',
                      '$$ dF(t,T) = \\mu F(t,T)dt + \\sigma e^{-\\alpha(T-t)}F(t,T)dW(t)$$',
                      '',
                      'Where:',
                      '',
                      '- $\\mu$ is the drift rate',
                      '- $\\sigma$ is the volatility',
                      '- $\\alpha$ is the mean reversion speed',
                      '- $W(t)$ is the standard Weiner Process',
                      '',
                      'Final form of the model is',
                      '',
                      '$$ F(t,T) = F(0,T)exp\\Big(\\mu t-\\frac{1}{2}\\sigma^2e^{-2\\alpha(T-t)}v(t)+\\sigma '
                      'e^{-\\alpha(T-t)}Y(t)\\Big)$$',
                      '',
                      'Where $Y$ is a standard Ornstein-Uhlenbeck Process with variance:',
                      '$$v(t) = \\frac{1-e^{-2\\alpha t}}{2\\alpha}$$',
                      '',
                      'The spot rate is given by $$S(t)=F(t,t)=F(0,t)exp\\Big(\\mu t-\\frac{1}{2}\\sigma^2v(t)+'
                      '\\sigma Y(t)\\Big)$$'])

    def __init__(self, factor, param, implied_factor=None):
        self.factor = factor
        self.param = param
        self.base_date_excel = None

    @staticmethod
    def num_factors():
        return 1

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    def theoretical_mean_std(self, t):
        Tmt = np.clip((self.factor.get_tenor() - self.base_date_excel) / utils.DAYS_IN_YEAR - t, 0, np.inf)
        ln_var = np.square(self.param['Sigma']) * np.exp(-2.0 * self.param['Alpha'] * Tmt) * (
                1.0 - np.exp(-2.0 * self.param['Alpha'] * t)) / (2.0 * self.param['Alpha'])
        mu = self.factor.current_value() * np.exp(self.param['Drift'] * t + 0.5 * ln_var)
        var = mu * mu * (np.exp(ln_var) - 1.0)
        return mu, np.sqrt(var)

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.initial_curve = tf.reshape(tensor, [1, -1, 1])
        excel_offset = (ref_date - utils.excel_offset).days
        self.base_date_excel = excel_offset
        excel_date_time_grid = time_grid.scen_time_grid + excel_offset
        tenors = (self.factor.get_tenor().reshape(1, -1) -
                  excel_date_time_grid.reshape(-1, 1)).clip(0.0, np.inf) / utils.DAYS_IN_YEAR
        tenor_rel = self.factor.get_tenor() - excel_offset
        delta = tenor_rel.reshape(1, -1).clip(
            time_grid.scen_time_grid[:-1].reshape(-1, 1),
            time_grid.scen_time_grid[1:].reshape(-1, 1)
        ) - time_grid.scen_time_grid[:-1].reshape(-1, 1)
        dt = np.insert(delta, 0, 0, axis=0) / utils.DAYS_IN_YEAR
        # need to scale the vol (as the variance is modelled using an OU Process)
        var_adj = (1.0 - np.exp(-2.0 * self.param['Alpha'] * dt.cumsum(axis=0))) / (2.0 * self.param['Alpha'])
        var = np.square(self.param['Sigma']) * np.exp(-2.0 * self.param['Alpha'] * tenors) * var_adj
        # get the vol
        vol = np.sqrt(np.diff(np.insert(var, 0, 0, axis=0), axis=0))

        with tf.name_scope(self.__class__.__name__):
            drift = np.expand_dims(self.param['Drift'] * dt.cumsum(axis=0) - 0.5 * var, axis=2)
            z_portion = tf.expand_dims(
                shared.t_random_numbers[process_ofs, :time_grid.scen_time_grid.size],
                axis=1) * np.expand_dims(vol, axis=2)
            self.stoch = drift + tf.cumsum(z_portion, axis=0, name='CS')

    @property
    def correlation_name(self):
        return 'ClewlowStricklandProcess', [()]

    def generate(self, shared_mem):
        with tf.name_scope(self.__class__.__name__):
            return self.initial_curve * tf.exp(self.stoch)


class CSForwardPriceCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Log', num_business_days=num_business_days,
                                                          max_alpha=5.0)
        alpha = stats['Mean Reversion Speed'].values[0]
        sigma = stats['Reversion Volatility'].values[0]
        mu = stats['Drift'].values[0] + 0.5 * (stats['Volatility'].values[0]) ** 2
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo({'Sigma': sigma, 'Alpha': alpha, 'Drift': mu}, correlation_coef, delta)


class PCAInterestRateModel(object):
    """The Principle Component Analysis model for interest rate curves Stochastic Process - defines the python
    interface and the low level cuda code"""

    documentation = (
        'Interest Rates',
        ['The parameters of the model are:',
         '- a volatility curve $\\sigma_\\tau$ for each tenor $\\tau$ of the zero curve $r_\\tau$',
         '- a mean reversion parameter $\\alpha$',
         '- eigenvalues $\\lambda_1,\\lambda_2,..,\\lambda_m$ and corresponding eigenvectors $Q_1(\\tau),Q_2(\\tau),...,Q_m(\\tau)$',
         '- optionally a historical yield curve $\\Theta(\\tau)$ for the long run mean of $r_\\tau$',
         '',
         'The stochastic process for the rate at each tenor on the interest rate curve is specified as:',
         '',
         '$$ dr_\\tau = r_\\tau ( u_\\tau  dt + \\sigma_\\tau dY )$$',
         '$$ dY_t = -\\alpha Ydt + dZ$$',
         '',
         'with $dY$  a standard Ornstein-Uhlenbeck process and $dZ$ a Brownian motion. It can be shown that:',
         '$$ Y(t) \\sim N(0, \\frac{1-e^{-2 \\alpha t}}{2 \\alpha})$$ ',
         '',
         'Currently, only the covarience matrix is used to define the eigenvectors with corresponding weight curves',
         '$w_k(\\tau)=Q_k(\\tau)\\frac{\\sqrt\\lambda_k}{\\sigma_\\tau}$ and normalized weight curve'
         '$$B_k(\\tau)=\\frac{w_k(\\tau)}{\\sqrt{\\sum_{l=1}^m w_l(\\tau)^2}}$$'
         '',
         'Final form of the model is',
         '',
         '$$ r_\\tau(t) = R_\\tau(t) exp \\Big( -\\frac{1}{2} \\sigma_\\tau^2 (\\frac{1-e^{-2 \\alpha t}}{2 \\alpha}) + \\sigma_\\tau \\sum_{k=1}^{3} B_k(\\tau) Y_k(t) \\Big)$$',
         '',
         'Where:',
         '',
         '- $r_\\tau(t)$ is the zero rate with a tenor $\\tau$  at time $t$  ($t = 0$ denotes the current rate at tenor $\\tau$)',
         '- $\\alpha$ is the mean reversion level of zero rates',
         '- $Y_k(t)$ is the OU process associated with Principle component $k$',
         '',
         'To simulate the mean rate $R_\\tau(t)$ (note that $R_\\tau(0)=r_\\tau(0)$ ), there are 2 choices:',
         '',
         '**Drift To Forward** where the mean rate is the inital forward rate from $t$ to $t+\\tau$ so that',
         '$$\\frac{D(0,t+\\tau)}{D(0,t)}=e^{R_\\tau(t)\\tau} $$',
         '**Drift To Blend** is a weighted average function of the current rate and a mean reversion level',
         '$\\Theta_\\tau$ $$R_\\tau(t)=[e^{-\\alpha t}r_\\tau (0) + (1-e^{-\\alpha t})\\Theta_\\tau]$$',
         ])

    def __init__(self, factor, param, implied_factor=None):
        self.factor = factor
        self.param = param

        # need to precalculate these for a specific set of tenors
        self.evecs = None
        self.vols = None

    def num_factors(self):
        return len(self.param['Eigenvectors'])

    def theoretical_mean_std(self, ref_date, time_in_days):
        t = self.factor.get_day_count_accrual(ref_date, time_in_days)
        # only works for drift to forward - todo - extend to other cases
        fwd_curve = (self.factor.current_value(t + self.factor.tenors) * (
                t + self.factor.tenors) - self.factor.current_value(
            t) * t) / self.factor.tenors
        sigma = np.interp(self.factor.get_tenor(), *self.param['Yield_Volatility'].array.T)
        sigma2 = (1.0 - np.exp(-2.0 * self.param['Reversion_Speed'] * t)) / (
                2.0 * self.param['Reversion_Speed']) * sigma * sigma
        std_dev = np.sqrt(np.exp(sigma2) - 1.0) * fwd_curve
        return fwd_curve, std_dev

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # ensures that tenors used are the same as the price factor
        factor_tenor = self.factor.get_tenor()

        # rescale and precalculate the eigenvectors
        evecs, evals = np.zeros((factor_tenor.size, self.num_factors())), []
        for index, eigen_data in enumerate(self.param['Eigenvectors']):
            evecs[:, index] = np.interp(factor_tenor, *eigen_data['Eigenvector'].array.T)
            evals.append(eigen_data['Eigenvalue'])

        evecs = np.array(evecs)
        # note that I don't need to divide by the volatility because I normalize
        # across tenors below . . .
        B = evecs.dot(np.diag(np.sqrt(evals)))
        B /= np.linalg.norm(B, axis=1).reshape(-1, 1)

        if implied_tensor is not None:
            vols = implied_tensor['Yield_Volatility']
            self.vols = utils.interpolate_tensor(factor_tenor, self.param['Yield_Volatility'].array[:, 0], vols)
            alpha = implied_tensor['Reversion_Speed']
            var = tf.exp(-2.0 * alpha * time_grid.time_grid_years) / (2.0 * alpha)
            self.vol_adj = -tf.concat([[0.0], var[1:] - var[:-1]], axis=0)
            self.delta_vol = tf.reshape(self.vols, (1, -1, 1)) * tf.reshape(tf.sqrt(self.vol_adj), (-1, 1, 1))
            self.drift = tf.expand_dims(
                -0.5 * tf.expand_dims(self.vols * self.vols, axis=0) *
                tf.expand_dims(self.vol_adj, axis=1), axis=2)
        else:
            self.vols = np.interp(factor_tenor, *self.param['Yield_Volatility'].array.T)
            alpha = self.param['Reversion_Speed']
            self.vol_adj = -np.append([0], np.diff(
                np.exp(-2.0 * alpha * time_grid.time_grid_years) / (2.0 * alpha)))
            self.delta_vol = self.vols.reshape(1, -1, 1) * np.sqrt(self.vol_adj).reshape(-1, 1, 1)
            self.drift = np.expand_dims(
                -0.5 * ((self.vols * self.vols).reshape(1, -1)) * self.vol_adj.reshape(-1, 1), axis=2)

        # normalize the eigenvectors, precalculate the vols and the historical_Yield
        self.evecs = B.T

        # also need to pre-calculate the forward curves at time_grid given and pass that to the cuda kernel
        if self.param['Rate_Drift_Model'] == 'Drift_To_Blend':
            hist_mean = scipy.interpolate.interp1d(*np.hstack(
                ([[0.0], [self.param['Historical_Yield'].array.T[-1][0]]], self.param['Historical_Yield'].array.T)),
                                                   kind='linear', bounds_error=False,
                                                   fill_value=self.param['Historical_Yield'].array.T[-1][-1])
            curve_t0 = self.factor.current_value(self.factor.tenors)
            omega = hist_mean(self.factor.tenors)
            # TODO - convert the code below to tensorflow

        ##            self.fwd_curve = np.array(
        ##                [curve_t0 * np.exp(-alpha * self.factor.get_day_count_accrual(ref_date, t)) + omega *
        ##                 (1.0 - np.exp(-alpha * self.factor.get_day_count_accrual(ref_date, t)))
        ##                 for t in time_grid.scen_time_grid])
        else:
            # calculate the forward curve across time
            self.fwd_curve = utils.calc_curve_forwards(
                self.factor, tensor, time_grid, shared, ref_date) / self.factor.tenors.reshape(1, -1)

        if hasattr(shared, 't_random_numbers'):
            # only run this if we have random numbers
            with tf.name_scope(self.__class__.__name__):
                self.calc_factors(shared.t_random_numbers[
                                  process_ofs:process_ofs + self.num_factors(), :time_grid.scen_time_grid.size])

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    def calc_factors(self, factors):
        pc_portion = tf.reduce_sum(tf.expand_dims(factors, axis=2) *
                                   self.evecs[:, np.newaxis, :, np.newaxis], axis=0) * self.delta_vol
        self.stoch = tf.cumsum(self.drift + pc_portion, axis=0, name='PCA')

    @property
    def correlation_name(self):
        return 'InterestRateOUProcess', [('PC{}'.format(x),) for x in range(1, self.num_factors() + 1)]

    def generate(self, shared_mem, random_sample=None):
        if random_sample is not None:
            self.calc_factors(random_sample)

        with tf.name_scope(self.__class__.__name__):
            fwd_component = tf.expand_dims(self.fwd_curve, 2)
            return fwd_component * tf.exp(self.stoch)


class PCAInterestRateCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 3

    def calibrate(self, data_frame, num_business_days=252.0):
        min_rate = data_frame.min().min()
        force_positive = 0.0 if min_rate > 0.0 else -5.0 * min_rate
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame + force_positive, method='Log',
                                                          num_business_days=num_business_days, max_alpha=4.0)

        standard_deviation = stats['Reversion Volatility'].interpolate()
        covariance = np.dot(standard_deviation.values.reshape(-1, 1),
                            standard_deviation.values.reshape(1, -1)) * correlation
        aki, evecs, evals = utils.PCA(covariance, self.num_factors)
        meanReversionSpeed = stats['Mean Reversion Speed'].mean()
        volCurve = standard_deviation
        reversionLevel = stats['Long Run Mean'].interpolate().bfill().ffill()
        correlation_coef = aki.T

        return utils.CalibrationInfo(
            OrderedDict({
                'Reversion_Speed': meanReversionSpeed,
                'Historical_Yield': utils.Curve([], list(zip(tenor, reversionLevel))),
                'Yield_Volatility': utils.Curve([], list(zip(tenor, volCurve))),
                'Eigenvectors': [
                    OrderedDict({'Eigenvector': utils.Curve([], list(zip(tenor, evec))), 'Eigenvalue': eval})
                    for evec, eval in zip(evecs.real.T, evals.real)],
                'Rate_Drift_Model': self.param['Rate_Drift_Model'],
                'Princ_Comp_Source': self.param['Matrix_Type'],
                'Distribution_Type': self.param['Distribution_Type']
            }),
            correlation_coef,
            delta
        )


def construct_process(sp_type, factor, param, implied_factor=None):
    return globals().get(sp_type)(factor, param, implied_factor)


def construct_calibration_config(calibration_config, param):
    return globals().get(calibration_config['Method'])(calibration_config['PriceModel'], param)
