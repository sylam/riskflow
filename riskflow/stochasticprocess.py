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
import itertools
from collections import OrderedDict

# 3rd party libraries
import numpy as np
import scipy.interpolate
from scipy.linalg import expm as matrix_expm
import torch
import torch.nn.functional as F

# Internal modules
from . import utils
from .instruments import get_fx_zero_rate_factor, get_equity_zero_rate_factor, get_dividend_rate_factor


def piecewise_linear(t, tenor, values, shared):
    if isinstance(values, np.ndarray):
        # no tensors needed - don't even bother caching
        dt = np.diff(t)
        interp = np.interp(t, tenor, values)
        return dt, interp[:-1], np.diff(interp) / dt
    else:
        key_code = ('piecewise_linear', values.data_ptr(), id(tenor), t.tobytes())

    if key_code not in shared.t_PreCalc:
        # linear interpolation of the vols at vol_tenor to the grid t
        dt = values.new(np.diff(t))
        interp = utils.interpolate_tensor(t, tenor, values)
        grad = (interp[1:] - interp[:-1]) / dt
        shared.t_PreCalc[key_code] = (dt, interp[:-1], grad)

    # interpolated vol
    return shared.t_PreCalc[key_code]


def integrate_piecewise_linear(fn_norm, shared, time_grid, tenor1, val1, tenor2=None, val2=None):
    def final_integration_points(only_np, int_points, interp_value):
        # return all but last point and make a tensor if necessary
        return int_points[:-1] if only_np else shared.t_PreCalc.setdefault(
            ('integration', tuple(int_points[:-1])), interp_value.new(int_points[:-1]))

    t = np.union1d(0.0, time_grid)
    np_only = isinstance(val1, np.ndarray)
    max_time = time_grid.max()
    fn, norm = fn_norm
    integration_points = np.union1d(t, tenor1[:tenor1.searchsorted(max_time)])

    if val2 is not None:
        integration_points = np.union1d(integration_points, tenor2[:tenor2.searchsorted(max_time)])
        _, interp_val1, m1 = piecewise_linear(integration_points, tenor1, val1, shared)
        dt, interp_val2, m2 = piecewise_linear(integration_points, tenor2, val2, shared)
        np_only = np_only and isinstance(val2, np.ndarray)
        t_integration_points = final_integration_points(np_only, integration_points, interp_val1)
        int_fn = fn(t_integration_points, interp_val1, interp_val2, dt, m1, m2)
    else:
        dt, interp_val, m = piecewise_linear(integration_points, tenor1, val1, shared)
        t_integration_points = final_integration_points(np_only, integration_points, interp_val)
        int_fn = fn(t_integration_points, interp_val, dt, m)

    if np_only:
        integral = np.pad(np.cumsum(int_fn) / norm, [1, 0], 'constant')
        return integral[integration_points.searchsorted(time_grid)]
    else:
        integral = F.pad(torch.cumsum(int_fn, dim=0) / norm, (1, 0))
        if integration_points.size == time_grid.size:
            return integral
        else:
            return integral[integration_points.searchsorted(time_grid)]


# Hull white analytic integrals for 1 and 2 factor models (assuming piecewise linear vols)

def hw_calc_H(a, exp):
    # sympy.simplify(sympy.integrate(sympy.exp(a * s) * (v + m * (s - t)), (s, t, t + dt)))
    # leave the division till later and simplify
    def H(t, v, dt, m):
        return (-a * v + m) * exp(a * t) + (a * m * dt + a * v - m) * exp(a * (dt + t))

    return H, a * a


def hw_calc_IJK(a, exp):
    # sympy.simplify(sympy.integrate(sympy.exp(a*s)*(vi+mi*(s-t))*(vj+mj*(s-t)), (s,t,t+dt)))
    # leave the division till later and simplify
    def IJK(t, vi, vj, dt, mi, mj):
        a2, dt2, mi_mj, mj_vi_p_mi_vj, vi_vj = a * a, dt * dt, mi * mj, mj * vi + mi * vj, vi * vj

        return ((a2 * (dt2 * mi_mj + dt * mj_vi_p_mi_vj + vi_vj) + 2 * mi_mj * (1 - a * dt) - a * mj_vi_p_mi_vj)
                * exp(a * dt) - a2 * vi_vj + a * mj_vi_p_mi_vj - 2 * mi_mj) * exp(a * t)

    return IJK, a ** 3


class StochasticProcess(object):
    """Base class for all stochastic processes"""

    def __init__(self, factor, param):
        self.factor = factor
        self.param = param
        self.params_ok = True

    def link_references(self, implied_tensor, implied_var, implied_factors):
        """link market variables across different risk factors"""
        pass

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass


class GBMAssetPriceModel(StochasticProcess):
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
        super(GBMAssetPriceModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        dt = np.diff(np.hstack(([0], time_grid.time_grid_years)))
        var = self.param['Vol'] * self.param['Vol'] * dt
        # store params in tensors
        self.drift = tensor.new((self.param['Drift'] * dt - 0.5 * var).reshape(-1, 1))
        self.vol = tensor.new((np.sqrt(var)).reshape(-1, 1))

        # store a reference to the current tensor
        self.spot = tensor

    def theoretical_mean_std(self, t):
        mu = self.factor.current_value() * np.exp(self.param['Drift'] * t)
        var = mu * mu * (np.exp(t * self.param['Vol'] ** 2) - 1.0)
        return mu, np.sqrt(var)

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        f1 = (self.drift + self.vol *
              shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]).cumsum(axis=0)
        return self.spot * torch.exp(f1)


class GBMAssetPriceCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0, vol_cuttoff=0.5, drift_cuttoff=0.1):
        stats, correlation, delta = utils.calc_statistics(
            data_frame, method='Log', num_business_days=num_business_days)
        mu = (stats['Drift'] + 0.5 * (stats['Volatility'] ** 2)).values[0]
        sigma = stats['Volatility'].values[0]

        return utils.CalibrationInfo(
            {'Vol': np.clip(sigma, 0.01, vol_cuttoff), 'Drift': np.clip(mu, -drift_cuttoff, drift_cuttoff)},
            [[1.0]], delta)


class GBMAssetPriceTSModelImplied(StochasticProcess):
    """The Geometric Brownian Motion Stochastic Process with implied drift and vol"""

    documentation = ('Asset Pricing', [
        'GBM with constant drift and vol may not be suited to model risk-neutral asset prices. A generalization that',
        'allows this would be to modify the volatility $\\sigma(t)$ and $\\mu(t)$ to be functions of time $t$.',
        'This can be specified as follows:',
        '',
        '$$ \\frac{dS(t)}{S(t)} = (r(t)-q(t)-v(t)\\sigma(t)\\rho) dt + \\sigma(t) dW(t)$$',
        '',
        'Note that no risk premium curve is captured. For Equity factors, its final form is:',
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
        'Here $C(t)=\\bar\\sigma(t)^2t, \\beta(t)=exp\\Big(\\int_0^t(r(s)-q(s))ds\\Big), \\rho=-1$ and $v(t)=\\sigma('
        't)$'
    ])

    def __init__(self, factor, param, implied_factor=None):
        super(GBMAssetPriceTSModelImplied, self).__init__(factor, param)
        self.implied = implied_factor
        # get the name of the underlying factor
        self.factor_type = self.factor.__class__.__name__
        # potentially handle quanto fx volatility
        self.quanto_fx_tenor = None

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        def calc_vol(t, v, dt, m):
            ''' sympy.simplify(sympy.integrate((v + m * (s - t)) ** 2 , (s, t, t + dt))) '''
            return dt * (dt ** 2 * m ** 2 / 3 + dt * m * v + v ** 2)

        def cal_quanto_fx_vol(t, vi, vj, dt, mi, mj):
            ''' sympy.simplify(sympy.integrate((vi + mi * (s - t)) * (vj + mj * (s - t)), (s, t, t + dt))) '''
            return dt * (2 * dt ** 2 * mi * mj + 3 * dt * mi * vj + 3 * dt * mj * vi + 6 * vi * vj) / 6

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        # calc vols
        vol_tenor = self.implied.param['Vol'].array[:, 0]
        self.V = torch.unsqueeze(integrate_piecewise_linear(
            (calc_vol, 1.0), shared, time_grid.time_grid_years, vol_tenor, implied_tensor['Vol']), dim=1)
        # calc the incremental vol
        self.delta_vol = F.pad(torch.sqrt(self.V[1:] - self.V[:-1]), (0, 0, 1, 0))
        self.delta_scen_t = np.insert(np.diff(time_grid.scen_time_grid), 0, 0).reshape(-1, 1)
        # store a reference to the current tensor
        self.spot = tensor
        # store the scenario grid
        self.scen_grid = time_grid.scenario_grid
        # check if we need to calculate the quanto fx vol
        if self.factor_type == 'EquityPrice' and self.quanto_fx_tenor is not None:
            # need to get the quantofx rate if necessary
            self.C = torch.unsqueeze(integrate_piecewise_linear(
                (cal_quanto_fx_vol, 1.0), shared, time_grid.time_grid_years,
                vol_tenor, implied_tensor['Vol'], self.quanto_fx_tenor, implied_tensor['Quanto_FX_Volatility']),
                dim=1)
            self.rho = implied_tensor['Quanto_FX_Correlation']
        else:
            self.C = 0.0
            self.rho = 0.0

    def link_references(self, implied_tensor, implied_var, implied_factors):
        """link market variables across different risk factors"""
        if self.factor_type == 'EquityPrice':
            fx_implied_index = utils.Factor('FxRate', self.factor.get_currency())
            if fx_implied_index in implied_var:
                FXImplied_vol_factor = utils.Factor(
                    'GBMAssetPriceTSModelParameters', self.factor.get_currency() + ('Vol',))
                # now set the Quanto_FX_Volatility to the same vol as the fx rate
                implied_tensor['Quanto_FX_Volatility'] = implied_var[fx_implied_index][FXImplied_vol_factor]
                if self.implied.param['Quanto_FX_Volatility'] is not None:
                    self.quanto_fx_tenor = self.implied.param['Quanto_FX_Volatility'].array[:, 0]
                else:
                    self.quanto_fx_tenor = implied_factors[
                        utils.Factor('GBMAssetPriceTSModelParameters', self.factor.get_currency())].get_tenor()

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        # this is valid for FX and Equity factors only
        if self.factor_type == 'EquityPrice':
            self.r_t = get_equity_zero_rate_factor(
                factor.name, static_ofs, stoch_ofs, all_tenors, all_factors)
            self.q_t = get_dividend_rate_factor(
                factor.name, static_ofs, stoch_ofs, all_tenors)
        elif self.factor_type == 'FxRate':
            self.r_t = get_fx_zero_rate_factor(
                self.factor.get_domestic_currency(None), static_ofs, stoch_ofs, all_tenors, all_factors)
            self.q_t = get_fx_zero_rate_factor(factor.name, static_ofs, stoch_ofs, all_tenors, all_factors)
        else:
            raise Exception('Unknown factor type {}'.format(self.factor_type))

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        f1 = torch.cumsum(
            self.delta_vol * shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon], dim=0)

        rt = utils.calc_time_grid_curve_rate(self.r_t, self.scen_grid, shared_mem)
        qt = utils.calc_time_grid_curve_rate(self.q_t, self.scen_grid, shared_mem)

        rt_rates = rt.gather_weighted_curve(shared_mem, self.delta_scen_t)
        qt_rates = qt.gather_weighted_curve(shared_mem, self.delta_scen_t)

        drift = torch.cumsum(torch.squeeze(rt_rates - qt_rates, dim=1), dim=0)

        return self.spot * torch.exp(drift - self.rho * self.C - 0.5 * self.V + f1)


class GBMPriceIndexModel(StochasticProcess):
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
        super(GBMPriceIndexModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # store randomnumber id's
        self.z_offset = process_ofs
        # calculate the correct scenario grid
        scenario_time_grid = np.array(
            [(x - self.factor.param['Last_Period_Start']).days
             for x in self.factor.get_last_publication_dates(ref_date, time_grid.scen_time_grid.tolist())],
            dtype=np.float64)
        # record the horizon
        self.scenario_horizon = scenario_time_grid.size

        dt = np.diff(np.hstack(([0], scenario_time_grid / utils.DAYS_IN_YEAR)))
        var = self.param['Vol'] * self.param['Vol'] * dt
        self.drift = tensor.new(self.param['Drift'] * dt - 0.5 * var).reshape(-1, 1)
        self.vol = tensor.new(np.sqrt(var)).reshape(-1, 1)

        # store a reference to the current tensor
        self.spot = tensor

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        f1 = torch.cumsum(self.drift + self.vol *
                          shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon], dim=0)
        return self.spot * torch.exp(f1)


class GBMPriceIndexCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Log', num_business_days=num_business_days)
        mu = (stats['Drift'] + 0.5 * (stats['Volatility'] ** 2)).values[0]
        sigma = stats['Volatility'].values[0]

        return utils.CalibrationInfo({'Vol': sigma, 'Drift': mu, 'Seasonal_Adjustment': None}, [[1.0]], delta)


class HullWhite2FactorImpliedInterestRateModel(StochasticProcess):
    """Hull white 2 factor implied interest rate model for risk neutral simulation of yield curves"""

    documentation = (
        'Interest Rates',
        ['This is a generalization of the 1 factor Hull White model. There are 2 correlated risk ',
         'factors where the $i^{th}$ factor has a volatility curve $\\sigma_i(t)$, constant reversion',
         'speed $\\alpha_i$ and market price of risk $\\lambda_i$.',
         '',
         'Final form of the model is:',
         '',
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
         'The cholesky decomposition of $C$ is',
         '',
         '$$L=\\begin{pmatrix} \\sqrt C_{11} & 0 \\\\ \\frac{C_{12}}{\\sqrt C_{11}} & \\sqrt {C_{22}',
         '-\\frac{C_{12}^2}{C_{11}} } \\\\ \\end{pmatrix}$$',
         '',
         'The increment is simulated using $LZ$ where $Z$ is a 2D vector of independent normals at',
         'time step $k$.'])

    def __init__(self, factor, param, implied_factor, clip=(1e-5, 3.0)):
        super(HullWhite2FactorImpliedInterestRateModel, self).__init__(factor, param)
        self.implied = implied_factor
        self.cache = {}
        self.factor_tenor = None
        self.clip = clip
        self.grid_index = None
        self.BtT = None
        self.C = None

    @staticmethod
    def num_factors():
        return 2

    def reset_implied_factor(self, implied_factor):
        self.implied = implied_factor
        self.cache = {}

    def link_references(self, implied_tensor, implied_var, implied_factors):
        """link market variables across different risk factors"""
        fx_implied_index = utils.Factor('FxRate', self.factor.get_currency())
        if fx_implied_index in implied_var:
            FXImplied_vol_factor = utils.Factor(
                'GBMAssetPriceTSModelParameters', self.factor.get_currency() + ('Vol',))
            # now set the Quanto_FX_Volatility to the same vol as the fx rate
            implied_tensor['Quanto_FX_Volatility'] = implied_var[fx_implied_index][FXImplied_vol_factor]
        else:
            # handle the unlikely case were we need to simulate a curve but not the fx rate.
            quantofx = self.implied.get_quanto_fx()
            if quantofx is not None:
                implied_tensor['Quanto_FX_Volatility'] = implied_tensor['Sigma_1'].new_tensor(quantofx)

    def read_cache(self, ref_date, time_grid, tensor, shared, process_ofs):
        if not self.cache:
            self.z_offset = process_ofs
            self.scenario_horizon = time_grid.scen_time_grid.size
            time_grid_years = np.array([self.factor.get_day_count_accrual(
                ref_date, t) for t in time_grid.scen_time_grid])
            # get the factor's tenor points
            self.factor_tenor = tensor.new(self.factor.get_tenor().reshape(-1, 1))
            # store the forward curve
            fwd_curve = utils.calc_curve_forwards(self.factor, tensor, time_grid_years, shared)
            # flatten the tenor
            self.factor_tenor_full = tensor.new_tensor(self.factor.get_tenor(), dtype=torch.float64)
            # get the quanto vol
            self.quantofx = tensor.new_tensor(
                self.implied.param['Quanto_FX_Volatility'].array[:, 1], dtype=torch.float64)

            # cache tensors/variables
            self.cache['time_grid_years'] = time_grid_years
            self.cache['fwd_curve'] = fwd_curve
            self.cache['t'] = tensor.new(time_grid_years.reshape(-1, 1))

        return self.cache['time_grid_years'], self.cache['fwd_curve'], self.cache['t']

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # get the factor's tenor points
        time_grid_years, fwd_curve, t = self.read_cache(ref_date, time_grid, tensor, shared, process_ofs)

        # calculate known functions
        alpha = [implied_tensor['Alpha_1'][0].type(torch.float64),
                 implied_tensor['Alpha_2'][0].type(torch.float64)]
        lam = [self.param['Lambda_1'], self.param['Lambda_2']]
        corr = implied_tensor['Correlation'].type(torch.float64)
        vols = [implied_tensor['Sigma_1'].type(torch.float64),
                implied_tensor['Sigma_2'].type(torch.float64)]
        vols_tenor = [self.implied.param['Sigma_1'].array[:, 0],
                      self.implied.param['Sigma_2'].array[:, 0]]

        H = [integrate_piecewise_linear(
            hw_calc_H(alpha[i], torch.exp), shared, time_grid_years, vols_tenor[i], vols[i]) for i in range(2)]
        I = [[integrate_piecewise_linear(
            hw_calc_IJK(alpha[i], torch.exp), shared, time_grid_years, vols_tenor[i], vols[i], vols_tenor[j], vols[j])
            for j in range(2)] for i in range(2)]
        J = [[integrate_piecewise_linear(
            hw_calc_IJK(alpha[i] + alpha[j], torch.exp), shared,
            time_grid_years, vols_tenor[i], vols[i], vols_tenor[j], vols[j]) for j in range(2)] for i in range(2)]

        # Check if the curve is not the same as the base currency
        if self.implied.param['Quanto_FX_Volatility'] and self.implied.param['Quanto_FX_Volatility'].array.any():
            quantofx = self.implied.param['Quanto_FX_Volatility'].array
            if 'Quanto_FX_Correlation_1' in implied_tensor and 'Quanto_FX_Correlation_2' in implied_tensor:
                quantofxcorr = [implied_tensor['Quanto_FX_Correlation_1'][0].type(torch.float64),
                                implied_tensor['Quanto_FX_Correlation_2'][0].type(torch.float64)]
                t_quanto_vol = implied_tensor['Quanto_FX_Volatility'].type(torch.float64)
            else:
                quantofxcorr = self.implied.get_quanto_correlation(corr, vols)
                t_quanto_vol = self.quantofx

            K = [integrate_piecewise_linear(
                hw_calc_IJK(alpha[i], torch.exp), shared, time_grid_years,
                vols_tenor[i], vols[i], quantofx[:, 0], t_quanto_vol) for i in range(2)]
        else:
            quantofxcorr = [0.0, 0.0]
            K = [corr.new_zeros(time_grid_years.size) for i in range(2)]

        # now calculate the AtT
        AtT = 0.0
        BtT = [(1.0 - torch.exp(-alpha[i] * self.factor_tenor_full)) / alpha[i] for i in range(2)]
        CtT = []
        rho = [[1.0, corr], [corr, 1.0]]

        for i, j in itertools.product(range(2), range(2)):
            first_part = torch.exp(-(alpha[i] + alpha[j]) * t) * J[i][j].reshape(-1, 1)
            second_part = torch.exp(-alpha[i] * t) * I[i][j].reshape(-1, 1) - first_part
            third_part = torch.exp(-alpha[j] * t) * I[j][i].reshape(-1, 1) - first_part

            # get the covariance
            CtT.append(rho[i][j] * J[i][j])

            # all together now
            AtT += rho[i][j] * torch.matmul(
                torch.cat([first_part, second_part, third_part], dim=1),
                torch.stack([BtT[j] * BtT[i], BtT[i] / alpha[j], BtT[j] / alpha[i]]))

        t_CtT = torch.stack(CtT).T
        # get the change in variance
        delta_CtT = t_CtT[1:] - t_CtT[:-1]
        # check if the entire cholesky is +ve definite
        if (delta_CtT[:, 0] * delta_CtT[:, 3] > delta_CtT[:, 1] * delta_CtT[:, 2]).all():
            # get the correlation through time
            C = F.pad(torch.linalg.cholesky(delta_CtT.reshape(-1, 2, 2)), (0, 0, 0, 0, 1, 0))
            # all good
            self.params_ok = True
        else:
            # need to fix the cholesky
            cholesky = [tensor.new_zeros((2, 2), dtype=torch.float64)]
            for i, C in enumerate(t_CtT[1:] - t_CtT[:-1]):
                if (C[0] > 0.0) & (C[3] > 0.0) & (C[1] * C[2] < C[0] * C[3]):
                    cholesky.append(torch.linalg.cholesky(C.reshape(2, 2)))
                else:
                    cholesky.append(cholesky[-1])

            # the cholesky was broken
            self.params_ok = False
            C = torch.stack(cholesky)

        # intermediate results
        self.BtT = [Bi.type(shared.one.dtype).reshape(-1, 1) for Bi in BtT]
        self.YtT = [torch.exp(-alpha[i] * t).type(shared.one.dtype) for i in range(2)]
        self.KtT = [(quantofxcorr[i] * K[i].reshape(-1, 1)).type(shared.one.dtype) for i in range(2)]
        self.HtT = [(lam[i] * H[i].reshape(-1, 1)).type(shared.one.dtype) for i in range(2)]
        self.alpha = [alpha[i].type(shared.one.dtype) for i in range(2)]

        # needed for factor calcs later
        self.F1 = C[:, 0, 0].reshape(-1, 1).type(shared.one.dtype)
        self.F2 = C[:, 1].t().unsqueeze(axis=2).type(shared.one.dtype)

        # store the grid points used if necessary
        if len(time_grid.scenario_grid) != time_grid_years.size:
            self.grid_index = time_grid.scen_time_grid.searchsorted(time_grid.scenario_grid[:, utils.TIME_GRID_MTM])

        self.drift = torch.unsqueeze(fwd_curve + 0.5 * AtT.type(shared.one.dtype), 2)

    def calc_factors(self, factor1, factor1and2):
        f1 = torch.unsqueeze(
            (torch.cumsum(factor1 * self.F1, dim=0) - self.KtT[0] + self.HtT[0]) * self.YtT[0], dim=1)
        f2 = torch.unsqueeze(
            (torch.cumsum(torch.sum(factor1and2 * self.F2, dim=0), dim=0) - self.KtT[1] + self.HtT[1]) * self.YtT[1],
            dim=1)
        return f1, f2

    @property
    def correlation_name(self):
        return 'HWImpliedInterestRate', [('F1',), ('F2',)]

    def generate(self, shared_mem):

        def sim_curve(drift, Bt0, Bt1, f1, f2, factor_tenor):
            stoch_component = Bt0 * f1 + Bt1 * f2
            return (drift + stoch_component) / factor_tenor

        factor1, factor2 = self.calc_factors(
            shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon],
            shared_mem.t_random_numbers[self.z_offset:self.z_offset + 2, :self.scenario_horizon])

        # check if we need deflators
        if self.grid_index is not None:
            # if we have a grid index, then we want to simulate a reduced curve (just the first 6 months) over a
            # finer time grid - this is useful for stochastic deflation - note that at least 4 tenor points need
            # to be included to correctly handle interpolation
            reduced_tenor_index = max(4, self.factor.tenors.searchsorted(0.5) + 1)

            full_grid_curve = sim_curve(
                self.drift[self.grid_index], self.BtT[0], self.BtT[1],
                factor1[self.grid_index], factor2[self.grid_index], self.factor_tenor)

            partial_grid_curve = sim_curve(
                self.drift[:, :reduced_tenor_index], self.BtT[0][:reduced_tenor_index],
                self.BtT[1][:reduced_tenor_index], factor1, factor2, self.factor_tenor[:reduced_tenor_index])

            return {shared_mem.scenario_keys['full']: full_grid_curve,
                    shared_mem.scenario_keys['reduced']: partial_grid_curve}
        else:
            return sim_curve(self.drift, self.BtT[0], self.BtT[1], factor1, factor2, self.factor_tenor)


class HullWhite1FactorInterestRateModel(StochasticProcess):
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
        '- $H(t) = \\int\\limits_0^t e^{\\alpha s}\\sigma (s)ds$',
        '- $I(t) = \\int\\limits_0^t e^{\\alpha s}{\\sigma (s)}^2ds$',
        '- $J(t) = \\int\\limits_0^t e^{2\\alpha s}{\\sigma (s)}^2ds$',
        '- $K(t) = \\int\\limits_0^t e^{\\alpha s}v(s){\\sigma (s)}ds$',
        '',
        'The simulation of the random increment $Y(t_{k+1})-Y(t_k)$ (where $0=t_0,t_1,t_2,...$',
        'represents the simulation grid) is normal with zero mean and variance $J(t_{k+1})-J(t_k)$'])

    def __init__(self, factor, param, implied_factor=None):
        super(HullWhite1FactorInterestRateModel, self).__init__(factor, param)
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

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        # store the forward curve
        time_grid_years = np.array([self.factor.get_day_count_accrual(
            ref_date, t) for t in time_grid.scen_time_grid])
        self.fwd_curve = utils.calc_curve_forwards(self.factor, tensor, time_grid_years, shared)

        # Really should implement this . . .
        quantofx = self.param['Quanto_FX_Volatility'].array.T if self.param['Quanto_FX_Volatility'] else np.zeros(
            (1, 2))
        quantofxcorr = self.param.get('Quanto_FX_Correlation', 0.0)

        # grab the vols
        vols = self.param['Sigma'].array

        # calculate known functions
        self.H = integrate_piecewise_linear(
            hw_calc_H(alpha, np.exp), shared, time_grid_years, vols[:, 0], vols[:, 1])
        self.I = integrate_piecewise_linear(
            hw_calc_IJK(alpha, np.exp), shared, time_grid_years,
            vols[:, 0], vols[:, 1], vols[:, 0], vols[:, 1])
        self.J = integrate_piecewise_linear(
            hw_calc_IJK(2.0 * alpha, np.exp), shared, time_grid_years,
            vols[:, 0], vols[:, 1], vols[:, 0], vols[:, 1])
        self.K = integrate_piecewise_linear(
            hw_calc_IJK(alpha, np.exp), shared, time_grid_years,
            vols[:, 0], vols[:, 1], quantofx[:, 0], quantofx[:, 1])

        # Now precalculate the A and B matrices
        BtT = (1.0 - np.exp(-alpha * factor_tenor)) / alpha
        AtT = np.array([(BtT / alpha) * np.exp(-alpha * t) * (
                2.0 * It - (np.exp(-alpha * t) + np.exp(-alpha * (t + factor_tenor))) * Jt) for (It, Jt, t) in
                        zip(self.I, self.J, time_grid_years)])

        # get the deltas
        self.delta_KtT = shared.one.new_tensor(
            np.hstack((0.0, quantofxcorr * np.diff(np.exp(-alpha * time_grid_years) * self.K)))
        ).reshape(-1, 1)

        self.delta_HtT = shared.one.new_tensor(np.hstack(
            (0.0, self.param['Lambda'] * np.diff(np.exp(-alpha * time_grid_years) * self.H)))
        ).reshape(-1, 1)

        delta_var = np.diff(np.exp(-2.0 * alpha * time_grid_years) * self.J)
        if delta_var.size:
            # needed for numerical stability
            delta_var[delta_var <= 0.0] = delta_var[delta_var > 0.0].min()
        self.delta_vol = shared.one.new_tensor(np.sqrt(np.hstack((0.0, delta_var))).reshape(-1, 1))

        self.AtT = shared.one.new_tensor(AtT)
        self.BtT = shared.one.new_tensor(BtT.reshape(-1, 1))
        self.fwd_component = torch.unsqueeze(self.fwd_curve + 0.5 * self.AtT, 2)
        self.factor_tenor = shared.one.new_tensor(factor_tenor.reshape(-1, 1))

    @property
    def correlation_name(self):
        return 'HWInterestRate', [('F1',)]

    def generate(self, shared_mem):
        f1 = (shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon] *
              self.delta_vol - self.delta_KtT + self.delta_HtT).cumsum(axis=0)

        stoch_component = self.BtT * torch.unsqueeze(f1, dim=1)
        return (self.fwd_component + stoch_component) / self.factor_tenor


class HWInterestRateCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
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


class HWHazardRateModel(StochasticProcess):
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
        super(HWHazardRateModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        alpha = self.param['Alpha']
        factor_tenor = self.factor.get_tenor()

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        time_grid_years = np.array([self.factor.get_day_count_accrual(
            ref_date, t) for t in time_grid.scen_time_grid])

        # store the forward curve    
        self.fwd_curve = utils.calc_curve_forwards(self.factor, tensor, time_grid_years, shared, mul_time=False)
        Bt = ((1.0 - np.exp(-alpha * time_grid_years)) / alpha).reshape(-1, 1)
        B2t = ((1.0 - np.exp(-2.0 * alpha * time_grid_years)) / alpha).reshape(-1, 1)
        sigma2 = self.param['Sigma'] ** 2
        BtT = ((1.0 - np.exp(-alpha * factor_tenor)) / alpha).reshape(1, -1)
        AtT = sigma2 * BtT * (0.5 * BtT * B2t + Bt ** 2)

        # OU variance
        var = (1.0 / (2.0 * alpha)) * (1.0 - np.exp(-2.0 * alpha * time_grid_years))
        delta_var = np.diff(np.insert(var, 0, 0, axis=0), axis=0)
        self.delta_vol = shared.one.new_tensor(np.sqrt(delta_var).reshape(-1, 1))

        # convert to tensors
        self.AtT = shared.one.new_tensor(AtT)
        self.BtT = shared.one.new_tensor(BtT)
        self.fwd_component = torch.unsqueeze(self.fwd_curve + 0.5 * self.AtT, dim=2)
        self.Bt = shared.one.new_tensor(Bt) if self.param['Lambda'] else 0.0

    @property
    def correlation_name(self):
        return 'HullWhiteProcess', [()]

    def generate(self, shared_mem):
        f1 = (shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon] * self.delta_vol).cumsum(dim=0)
        # add market price of risk (if non-zero):
        f1 = f1 + self.param['Lambda']*self.Bt
        stoch_component = self.param['Sigma'] * torch.unsqueeze(self.BtT, dim=2) * torch.unsqueeze(f1, dim=1)
        return self.fwd_component + stoch_component


class HWHazardRateCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Diff',
                                                          num_business_days=num_business_days, max_alpha=4.0)
        alpha = stats['Mean Reversion Speed'].mean()
        sigma = stats['Reversion Volatility'].values[0] / tenor[0]
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo({'Alpha': alpha, 'Sigma': sigma, 'Lambda': 0}, correlation_coef, delta)


class CSForwardPriceModel(StochasticProcess):
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
                      '',
                      '$$v(t) = \\frac{1-e^{-2\\alpha t}}{2\\alpha}$$',
                      '',
                      'The spot rate is given by',
                      '',
                      '$$S(t)=F(t,t)=F(0,t)exp\\Big(\\mu t-\\frac{1}{2}\\sigma^2v(t)+\\sigma Y(t)\\Big)$$',
                      ''])

    def __init__(self, factor, param, implied_factor=None):
        super(CSForwardPriceModel, self).__init__(factor, param)
        self.base_date_excel = None

    @staticmethod
    def num_factors():
        return 1

    def theoretical_mean_std(self, t):
        Tmt = np.clip((self.factor.get_tenor() - self.base_date_excel) / utils.DAYS_IN_YEAR - t, 0, np.inf)
        ln_var = np.square(self.param['Sigma']) * np.exp(-2.0 * self.param['Alpha'] * Tmt) * (
                1.0 - np.exp(-2.0 * self.param['Alpha'] * t)) / (2.0 * self.param['Alpha'])
        mu = self.factor.current_value() * np.exp(self.param['Drift'] * t + 0.5 * ln_var)
        var = mu * mu * (np.exp(ln_var) - 1.0)
        return mu, np.sqrt(var)

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.initial_curve = tensor.reshape(1, -1, 1)
        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        #  rebase the dates
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

        if implied_tensor is None:
            # need to scale the vol (as the variance is modelled using an OU Process)
            var_adj = (1.0 - np.exp(-2.0 * self.param['Alpha'] * dt.cumsum(axis=0))) / (2.0 * self.param['Alpha'])
            var = np.square(self.param['Sigma']) * np.exp(-2.0 * self.param['Alpha'] * tenors) * var_adj
            # get the vol
            vol = np.sqrt(np.diff(np.insert(var, 0, 0, axis=0), axis=0))
            self.vol = tensor.new(np.expand_dims(vol, axis=2))
            self.drift = tensor.new(np.expand_dims(self.param['Drift'] * dt.cumsum(axis=0) - 0.5 * var, axis=2))
        else:
            # need to scale the vol (as the variance is modelled using an OU Process)
            var_adj = (1.0 - torch.exp(-2.0 * implied_tensor['Alpha'] * tensor.new(dt.cumsum(axis=0)))) / (
                    2.0 * implied_tensor['Alpha'])
            var = torch.square(implied_tensor['Sigma']) * torch.exp(
                -2.0 * implied_tensor['Alpha'] * tensor.new(tenors)) * var_adj
            delta_var = torch.diff(F.pad(var, [0, 0, 1, 0]), dim=0)
            safe_delta = torch.where(delta_var > 0.0, delta_var, torch.ones_like(delta_var))
            vol = torch.where(delta_var > 0.0, torch.sqrt(safe_delta), torch.zeros_like(delta_var))
            self.vol = torch.unsqueeze(vol, dim=2)
            self.drift = torch.unsqueeze(-0.5 * var, dim=2)

    @property
    def correlation_name(self):
        return 'ClewlowStricklandProcess', [()]

    def generate(self, shared_mem):
        z_portion = torch.unsqueeze(
            shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon],
            dim=1) * self.vol

        return self.initial_curve * torch.exp(self.drift + torch.cumsum(z_portion, dim=0))


class CSImpliedForwardPriceModel(CSForwardPriceModel):
    """The Clewlow Strickland Stochastic Process with implied vol and mean reversion"""

    def __init__(self, factor, param, implied_factor=None):
        super(CSImpliedForwardPriceModel, self).__init__(factor, param)
        self.implied = implied_factor
        # set the drift explicitly to zero - copy the other 2 params
        self.param = {'Drift': 0.0, 'Sigma': implied_factor.param['Sigma'], 'Alpha': implied_factor.param['Alpha']}


class CSForwardPriceCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(
            data_frame, method='Log', num_business_days=num_business_days, max_alpha=5.0)
        alpha = stats['Mean Reversion Speed'].values[0]
        sigma = stats['Reversion Volatility'].values[0]
        mu = stats['Drift'].values[0] + 0.5 * (stats['Volatility'].values[0]) ** 2
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo({'Sigma': sigma, 'Alpha': alpha, 'Drift': mu}, correlation_coef, delta)


class PCAInterestRateModel(StochasticProcess):
    """The Principal Component Analysis model for interest rate curves Stochastic Process - defines the python
    interface and the low level cuda code"""

    documentation = (
        'Interest Rates',
        ['The parameters of the model are:',
         '- a volatility curve $\\sigma_\\tau$ for each tenor $\\tau$ of the zero curve $r_\\tau$',
         '- a mean reversion parameter $\\alpha$',
         '- eigenvalues $\\lambda_1,\\lambda_2,..,\\lambda_m$ and corresponding eigenvectors $Q_1(\\tau),Q_2(\\tau),'
         '...,Q_m(\\tau)$',
         '- optionally a historical yield curve $\\Theta(\\tau)$ for the long run mean of $r_\\tau$',
         '',
         'The stochastic process for the rate at each tenor on the interest rate curve is specified as:',
         '',
         '$$ dr_\\tau = r_\\tau ( u_\\tau  dt + \\sigma_\\tau dY )$$',
         '',
         '$$ dY_t = -\\alpha Ydt + dZ$$',
         '',
         'with $dY$  a standard Ornstein-Uhlenbeck process and $dZ$ a Brownian motion. It can be shown that:',
         '',
         '$$ Y(t) \\sim N(0, \\frac{1-e^{-2 \\alpha t}}{2 \\alpha})$$ ',
         '',
         'Currently, only the covarience matrix is used to define the eigenvectors with corresponding weight curves',
         '$w_k(\\tau)=Q_k(\\tau)\\frac{\\sqrt\\lambda_k}{\\sigma_\\tau}$ and normalized weight curve'
         '$$B_k(\\tau)=\\frac{w_k(\\tau)}{\\sqrt{\\sum_{l=1}^m w_l(\\tau)^2}}$$'
         '',
         'Final form of the model is',
         '',
         '$$ r_\\tau(t) = R_\\tau(t) exp \\Big( -\\frac{1}{2} \\sigma_\\tau^2 (\\frac{1-e^{-2 \\alpha t}}{2 \\alpha}) '
         '+ \\sigma_\\tau \\sum_{k=1}^{3} B_k(\\tau) Y_k(t) \\Big)$$',
         '',
         'Where:',
         '',
         '- $r_\\tau(t)$ is the zero rate with a tenor $\\tau$  at time $t$  ($t = 0$ denotes the current rate at '
         'tenor $\\tau$)',
         '- $\\alpha$ is the mean reversion level of zero rates',
         '- $Y_k(t)$ is the OU process associated with Principle component $k$',
         '',
         'To simulate the mean rate $R_\\tau(t)$ (note that $R_\\tau(0)=r_\\tau(0)$ ), there are 2 choices:',
         '',
         '**Drift To Forward** where the mean rate is the inital forward rate from $t$ to $t+\\tau$ so that',
         '',
         '$$\\frac{D(0,t+\\tau)}{D(0,t)}=e^{R_\\tau(t)\\tau}$$',
         '',
         '**Drift To Blend** is a weighted average function of the current rate and a mean reversion level',
         '$\\Theta_\\tau$',
         '',
         '$$R_\\tau(t)=[e^{-\\alpha t}r_\\tau (0) + (1-e^{-\\alpha t})\\Theta_\\tau]$$',
         ''
         ])

    def __init__(self, factor, param, implied_factor=None):
        super(PCAInterestRateModel, self).__init__(factor, param)
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

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        time_grid_years = np.array([self.factor.get_day_count_accrual(
            ref_date, t) for t in time_grid.scen_time_grid])

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

        self.vols = np.interp(factor_tenor, *self.param['Yield_Volatility'].array.T)
        alpha = self.param['Reversion_Speed']

        # Exact OU discretisation: Y_{k+1} = exp(-α Δt_k) Y_k + sqrt((1-exp(-2α Δt_k))/(2α)) Z_{k+1}
        dt_steps   = np.diff(np.append([0.0], time_grid_years))              # [T]
        ou_decay   = np.exp(-alpha * dt_steps)                               # [T]
        ou_std     = np.sqrt((1.0 - np.exp(-2.0 * alpha * dt_steps)) / (2.0 * alpha))  # [T]
        self.ou_decay    = shared.one.new_tensor(ou_decay.reshape(-1, 1))    # [T, 1]
        self.ou_noise    = shared.one.new_tensor(ou_std.reshape(-1, 1))      # [T, 1]
        self.vols_tensor = shared.one.new_tensor(self.vols.reshape(-1, 1))   # [n_tenors, 1]

        # Ito drift: -½ σ_τ² Var(Y(t_k)) = -½ σ_τ² (1-exp(-2α t_k))/(2α)  (full value at each t_k)
        ou_var_cumul = (1.0 - np.exp(-2.0 * alpha * time_grid_years)) / (2.0 * alpha)  # [T]
        self.drift = shared.one.new_tensor(np.expand_dims(
            -0.5 * (self.vols * self.vols).reshape(1, -1) * ou_var_cumul.reshape(-1, 1), axis=2))

        # normalize the eigenvectors
        self.evecs = shared.one.new_tensor(B.T[:, np.newaxis, :, np.newaxis])

        # also need to pre-calculate the forward curves at time_grid given and pass that to the cuda kernel
        if self.param['Rate_Drift_Model'] == 'Drift_To_Blend':
            hist_mean = scipy.interpolate.interp1d(*np.hstack(
                ([[0.0], [self.param['Historical_Yield'].array.T[-1][0]]], self.param['Historical_Yield'].array.T)),
                                                   kind='linear', bounds_error=False,
                                                   fill_value=self.param['Historical_Yield'].array.T[-1][-1])
            curve_t0 = self.factor.current_value(self.factor.tenors)
            omega = hist_mean(self.factor.tenors)
            # R_τ(t) = exp(-α t) r_τ(0) + (1 - exp(-α t)) Θ_τ  vectorised over [T, n_tenors]
            decay    = shared.one.new_tensor(np.exp(-alpha * time_grid_years).reshape(-1, 1))  # [T, 1]
            curve_t0 = shared.one.new_tensor(curve_t0.reshape(1, -1))                          # [1, n_tenors]
            omega    = shared.one.new_tensor(omega.reshape(1, -1))                              # [1, n_tenors]
            fwd_curve = decay * curve_t0 + (1.0 - decay) * omega                               # [T, n_tenors]

        else:
            # calculate the forward curve across time
            fwd_curve = utils.calc_curve_forwards(self.factor, tensor, time_grid_years, shared) / shared.one.new_tensor(
                factor_tenor.reshape(1, -1))

        self.fwd_component = torch.unsqueeze(fwd_curve, dim=2)

    def calc_factors(self, factors):
        # factors: [n_factors, T, n_scenarios]
        #
        # Closed-form vectorisation of Y_{k+1} = d_k * Y_k + n_k * Z_k  (Y_0 = 0):
        #
        #   Y_out[k] = D[k] * cumsum( n[i]/D[i] * Z[i] )[k]
        #   D[k] = cumprod(d)[k] = prod_{j=0}^{k} d[j]
        #
        # Replaces the T-step Python loop with two O(T) CUDA-native ops (cumprod, cumsum).
        # Numerically stable for typical PCA α values (0.01–0.3); avoid α >> 1 on long grids.
        evecs_mat = self.evecs[:, 0, :, 0]                              # [n_factors, n_tenors]
        D = torch.cumprod(self.ou_decay, dim=0)                         # [T, 1]
        Y = D.unsqueeze(0) * torch.cumsum(
            (self.ou_noise / D).unsqueeze(0) * factors, dim=1)          # [n_factors, T, n_scenarios]
        projected = torch.einsum('jt,jks->kts', evecs_mat, Y)          # [T, n_tenors, n_scenarios]
        return self.drift + self.vols_tensor.unsqueeze(0) * projected   # [T, n_tenors, n_scenarios]

    @property
    def correlation_name(self):
        return 'InterestRateOUProcess', [('PC{}'.format(x),) for x in range(1, self.num_factors() + 1)]

    def generate(self, shared_mem):
        stoch = self.calc_factors(
            shared_mem.t_random_numbers[
            self.z_offset:self.z_offset + self.num_factors(), :self.scenario_horizon])

        return self.fwd_component * torch.exp(stoch)


class PCAInterestRateCalibration(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 3

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
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


class RegimeSwitchingOU2FactorModel(StochasticProcess):
    """Two-factor OU carry model with regime switching via a 2-state continuous-time Markov chain.

    Regime-conditional dynamics (z in {0=Low, 1=High}):

        dx_t = kappa_x[z] * (theta_x[z] + beta[z] * y_t - x_t) dt + sigma_x[z] dW_x
        dy_t = kappa_y[z] * (theta_y[z] - y_t)                  dt + sigma_y[z] dW_y
        corr(dW_x, dW_y) = rho[z]

    Exact linear-Gaussian discretisation is used per time step via the Van Loan
    block-matrix method to compute the conditional covariance Omega(dt).

    **Timing convention** (regime frozen over the interval): the regime z_k observed
    at the *start* of [t_k, t_{k+1}] is held constant while the state evolves; only
    after the state update is the next regime z_{k+1} drawn from P(.|z_k, dt).
    That is:

        s_{k+1} = A[z_k] @ s_k + b[z_k] + eps_k,   eps_k ~ N(0, Omega[z_k])
        z_{k+1} ~ P( . | z_k, dt )                  (CTMC transition)
    """

    documentation = (
        'Asset Pricing',
        ['A two-factor mean-reverting carry model with two latent regimes.',
         '',
         'The **fast** carry factor $x_t$ (annualised) and **slow** pressure factor $y_t$',
         'satisfy, conditional on latent regime $z_t \\in \\{0,1\\}$:',
         '',
         '$$ dx_t = \\kappa_x^{[z]}(\\theta_x^{[z]} + \\beta^{[z]} y_t - x_t)\\,dt'
         ' + \\sigma_x^{[z]}\\,dW_x $$',
         '',
         '$$ dy_t = \\kappa_y^{[z]}(\\theta_y^{[z]} - y_t)\\,dt + \\sigma_y^{[z]}\\,dW_y $$',
         '',
         'with $\\operatorname{Corr}(dW_x, dW_y) = \\rho^{[z]}$.',
         '',
         'The regime $z_t$ follows a 2-state continuous-time Markov chain with intensities',
         '$\\lambda_{01}$ (Low$\\to$High) and $\\lambda_{10}$ (High$\\to$Low).',
         '',
         'Simulation uses the **exact linear-Gaussian discretisation** with the'
         ' **regime frozen over each interval**: the regime $z_{t_k}$ at the'
         ' *start* of $[t_k,\\,t_{k+1}]$ governs the state update, and the'
         ' next regime $z_{t_{k+1}}$ is drawn afterwards:',
         '',
         '$$ \\mathbf{s}_{t_{k+1}} = A^{[z_{t_k}]}(\\delta)\\,\\mathbf{s}_{t_k}'
         ' + b^{[z_{t_k}]}(\\delta) + \\varepsilon_k,'
         '\\quad \\varepsilon_k \\sim \\mathcal{N}(0,\\,\\Omega^{[z_{t_k}]}(\\delta)) $$',
         '',
         '$$ z_{t_{k+1}} \\sim P(\\,\\cdot\\mid z_{t_k},\\,\\delta) $$',
         '',
         'where $A(\\delta) = e^{F\\delta}$ ($F$ is the mean-reversion generator),'
         ' $b = (I-A)\\,\\boldsymbol{\\theta}^*$, and $\\Omega$ is obtained via the'
         ' Van Loan block-matrix method.',
         '',
         'Two correlated Gaussian drivers are used per step for the state update;'
         ' regime transitions draw independent uniforms via the quasi-random'
         ' number stream (``shared_mem.quasi_rng``).',
         '',
         'The forward carry price is reconstructed as',
         '',
         '$$ F(t,T) = S(t)\\,\\exp\\!\\bigl((r_t + x_t)\\,\\tau\\bigr) $$',
         ])

    def __init__(self, factor, param, implied_factor=None):
        super(RegimeSwitchingOU2FactorModel, self).__init__(factor, param)
        self._validate_params()

    def _validate_params(self):
        p = self.param
        for key in ('Kappa_X_Low', 'Kappa_X_High', 'Kappa_Y_Low', 'Kappa_Y_High'):
            if p.get(key, 0.0) <= 0.0:
                self.params_ok = False
                return
        for key in ('Sigma_X_Low', 'Sigma_X_High', 'Sigma_Y_Low', 'Sigma_Y_High'):
            if p.get(key, 0.0) < 0.0:
                self.params_ok = False
                return
        for key in ('Rho_Low', 'Rho_High'):
            if abs(p.get(key, 0.0)) >= 1.0:
                self.params_ok = False
                return
        for key in ('Lambda_01', 'Lambda_10'):
            if p.get(key, 0.0) <= 0.0:
                self.params_ok = False
                return

    @staticmethod
    def num_factors():
        return 2

    @staticmethod
    def _unpack_regime(param, label):
        """Return (kx, ky, tx, ty, sx, sy, beta, rho) for label 'Low' or 'High'."""
        return (
            param['Kappa_X_{}'.format(label)],
            param['Kappa_Y_{}'.format(label)],
            param['Theta_X_{}'.format(label)],
            param['Theta_Y_{}'.format(label)],
            param['Sigma_X_{}'.format(label)],
            param['Sigma_Y_{}'.format(label)],
            param['Beta_{}'.format(label)],
            param['Rho_{}'.format(label)],
        )

    @staticmethod
    def _compute_discretization(kx, ky, tx, ty, sx, sy, beta, rho, dt):
        """Exact linear-Gaussian step matrices for one regime over interval dt.

        Uses the centred-variable representation and the Van Loan block-matrix
        method to compute the conditional step covariance Omega(dt).

        Parameters
        ----------
        kx, ky       : mean-reversion speeds for x and y
        tx, ty       : long-run means (theta)
        sx, sy       : volatilities
        beta         : y-to-x coupling
        rho          : instantaneous correlation between dW_x and dW_y
        dt           : time step in years

        Returns
        -------
        A  : (2,2) ndarray  transition matrix  expm(F * dt)
        b  : (2,)  ndarray  mean-reversion drift  (I - A) @ [tx + beta*ty, ty]
        L  : (2,2) ndarray  lower-Cholesky factor of Omega(dt)
        """
        # Mean-reversion generator
        F = np.array([[-kx,  kx * beta],
                      [0.0,  -ky]])
        # Long-run mean of the joint state
        theta_vec = np.array([tx + beta * ty, ty])
        # Transition matrix A = expm(F * dt)
        A = matrix_expm(F * dt)
        # Mean-reversion drift in original coordinates:  b = (I - A) * theta*
        b = (np.eye(2) - A) @ theta_vec
        # Diffusion covariance  Q = Sigma * Corr * Sigma^T
        Q = np.array([[sx * sx,       rho * sx * sy],
                      [rho * sx * sy, sy * sy]])
        # Van Loan block matrix:  M = [[-F, Q], [0, F^T]] * dt
        # expm(M) = [[expm(-F*dt),  expm(-F*dt)*Omega], [0, expm(F^T*dt)]]
        M = np.zeros((4, 4))
        M[:2, :2] = -F * dt
        M[:2, 2:] =  Q * dt
        M[2:, 2:] =  F.T * dt
        eM = matrix_expm(M)
        # Omega = A @ (upper-right 2x2 block of expm(M))
        Omega = A @ eM[:2, 2:]
        # Symmetrise to remove floating-point residuals
        Omega = 0.5 * (Omega + Omega.T)
        # Cholesky — regularise if barely non-PSD
        try:
            L = np.linalg.cholesky(Omega)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(Omega)
            vals = np.clip(vals, 0.0, None)
            L = np.linalg.cholesky(vecs @ np.diag(vals) @ vecs.T + 1e-14 * np.eye(2))
        return A, b, L

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # store random-number slice offset and horizon
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        dt_arr = np.diff(np.hstack(([0.0], time_grid.time_grid_years)))
        n = self.scenario_horizon

        prms_low  = self._unpack_regime(self.param, 'Low')
        prms_high = self._unpack_regime(self.param, 'High')

        # Allocate per-step discretisation arrays
        A_low  = np.zeros((n, 2, 2));  b_low  = np.zeros((n, 2));  L_low  = np.zeros((n, 2, 2))
        A_high = np.zeros((n, 2, 2));  b_high = np.zeros((n, 2));  L_high = np.zeros((n, 2, 2))

        for k, dt in enumerate(dt_arr):
            if dt > 1e-10:
                A_low[k],  b_low[k],  L_low[k]  = self._compute_discretization(*prms_low,  dt)
                A_high[k], b_high[k], L_high[k] = self._compute_discretization(*prms_high, dt)
            else:
                # dt == 0: identity step, no noise
                A_low[k]  = np.eye(2)
                A_high[k] = np.eye(2)

        # Regime transition probabilities from 2-state CTMC over each dt
        lam01    = self.param['Lambda_01']
        lam10    = self.param['Lambda_10']
        lam      = lam01 + lam10
        p_sw     = (1.0 - np.exp(-lam * dt_arr)) if lam > 0.0 else np.zeros(n)
        p01      = (lam01 / lam) * p_sw   # P(Low  -> High | currently Low)
        p10      = (lam10 / lam) * p_sw   # P(High -> Low  | currently High)

        # Initial conditions (stored as Python scalars for use inside generate())
        self.x0        = tensor                                          # tensor [1] or [batch]
        self.init_y    = float(self.param.get('Initial_Y',                0.0))
        self.init_p_hi = float(self.param.get('Initial_Regime_Prob_High', 0.5))

        # Store discretisation matrices as tensors
        def _t(arr):
            return shared.one.new_tensor(arr)

        self.A_low  = _t(A_low);   self.b_low  = _t(b_low);   self.L_low  = _t(L_low)
        self.A_high = _t(A_high);  self.b_high = _t(b_high);  self.L_high = _t(L_high)
        # per-step transition probabilities — shape [T] for scalar indexing in the loop
        self.p01 = _t(p01)
        self.p10 = _t(p10)

    @property
    def correlation_name(self):
        return 'RegimeSwitchingOUProcess', [('F_x',), ('F_y',)]

    def generate(self, shared_mem):
        # Correlated Gaussian drivers for the OU state update: [T, batch]
        Z1 = shared_mem.t_random_numbers[self.z_offset,     :self.scenario_horizon]
        Z2 = shared_mem.t_random_numbers[self.z_offset + 1, :self.scenario_horizon]

        batch = Z1.shape[1]

        # Independent uniforms for regime transitions drawn from the quasi-rng stream
        # quasi_rng(dimension, sample_size) returns (normals, uniforms), shape [sample_size, dimension]
        u_regime = shared_mem.quasi_rng(shared_mem.simulation_batch, self.scenario_horizon)[1].T
        # u_regime: [T, batch]

        # Initial state — broadcast x0 (which may be [1] or [batch]) to [batch]
        x = self.x0.reshape(-1).expand(batch).clone()
        y = Z1.new_full((batch,), self.init_y)
        # Sample initial regime from Bernoulli(p_high); result is float -> cast to long
        z = torch.bernoulli(Z1.new_full((batch,), self.init_p_hi)).long()  # [batch]

        x_path = []

        for k in range(self.scenario_horizon):
            # --- Step 1: state update using regime z_k (frozen over [t_k, t_{k+1}]) ---
            z1k, z2k = Z1[k], Z2[k]   # [batch]

            # Correlated noise for each regime via lower-Cholesky
            # L is lower triangular: eps = [L[0,0]*z1 + L[0,1]*z2,  L[1,0]*z1 + L[1,1]*z2]
            ex0 = self.L_low[k, 0, 0]  * z1k + self.L_low[k, 0, 1]  * z2k
            ey0 = self.L_low[k, 1, 0]  * z1k + self.L_low[k, 1, 1]  * z2k
            ex1 = self.L_high[k, 0, 0] * z1k + self.L_high[k, 0, 1] * z2k
            ey1 = self.L_high[k, 1, 0] * z1k + self.L_high[k, 1, 1] * z2k

            # s_{k+1} = A[z_k] @ s_k + b[z_k] + eps_k
            nx0 = self.A_low[k, 0, 0] * x + self.A_low[k, 0, 1] * y + self.b_low[k, 0]  + ex0
            ny0 = self.A_low[k, 1, 0] * x + self.A_low[k, 1, 1] * y + self.b_low[k, 1]  + ey0
            nx1 = self.A_high[k, 0, 0] * x + self.A_high[k, 0, 1] * y + self.b_high[k, 0] + ex1
            ny1 = self.A_high[k, 1, 0] * x + self.A_high[k, 1, 1] * y + self.b_high[k, 1] + ey1

            is_low = (z == 0)
            x = torch.where(is_low, nx0, nx1)
            y = torch.where(is_low, ny0, ny1)

            # Record s_{k+1}  (state is now at t_{k+1}, regime still z_k)
            x_path.append(x.unsqueeze(0))  # [1, batch]

            # --- Step 2: draw z_{k+1} ~ P(.|z_k, dt) for the *next* interval ---
            u      = u_regime[k]                                         # [batch] in (0,1)
            p_sw_k = torch.where(is_low, self.p01[k], self.p10[k])      # [batch]
            z      = torch.where(u < p_sw_k, 1 - z, z)

        return torch.cat(x_path, dim=0)   # [T, batch]


class RegimeSwitchingOU2FactorCalibration(object):
    """Calibrate RegimeSwitchingOU2FactorModel from a carry-rate time series.

    The observable is the fast carry factor x_t (a scalar, possibly negative).
    The slow pressure factor y_t is latent; its parameters default to a slower
    version of the x-process and can be overridden via ``self.param``.

    **Regime assignment**: a rolling volatility window (``vol_window`` steps)
    is computed; observations above the ``high_vol_percentile``-th percentile
    are labelled High, the rest Low.

    **Per-regime OU x parameters** (Kappa_X, Theta_X, Sigma_X) are estimated
    with ``utils.calc_statistics(method='Diff')`` on each regime subset.

    **y-process defaults** (can be overridden in ``self.param``):
      - Kappa_Y  = Kappa_X / 5   (slower reversion than x)
      - Theta_Y  = 0.0
      - Sigma_Y  = 0.0           (deterministic y; set > 0 to add y-noise)
      - Beta     = 0.0           (no y-to-x coupling)
      - Rho      = 0.0           (zero x-y correlation given regime)

    **Transition intensities** are estimated from empirical sojourn times:
      - Lambda_01 = 1 / mean_steps_in_Low  * num_business_days
      - Lambda_10 = 1 / mean_steps_in_High * num_business_days

    The ``correlation_coef`` returned is the 2×2 identity: Z1, Z2 are drawn
    as independent standard normals by the simulation engine; within-regime
    x-y correlation is applied internally via the Van Loan Cholesky L.
    """

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 2

    @staticmethod
    def _regime_labels(series, vol_window, high_vol_percentile):
        """Return a boolean array: True = High-vol regime."""
        roll_vol = series.rolling(vol_window, min_periods=1).std().fillna(0.0)
        threshold = np.percentile(roll_vol.values, high_vol_percentile)
        return roll_vol.values >= threshold

    @staticmethod
    def _transition_intensity(labels, num_business_days):
        """Estimate CTMC rate from empirical sojourn run-lengths.

        Returns (lambda_01, lambda_10) in annualised units.  If a regime is
        never visited the corresponding intensity defaults to 1.0.
        """
        def _mean_run_length(mask):
            runs, current = [], 0
            for v in mask:
                if v:
                    current += 1
                else:
                    if current:
                        runs.append(current)
                    current = 0
            if current:
                runs.append(current)
            return float(np.mean(runs)) if runs else 1.0

        is_high = labels
        is_low  = ~labels
        mean_low  = _mean_run_length(is_low)   # avg steps in Low before switching
        mean_high = _mean_run_length(is_high)  # avg steps in High before switching
        # rate = 1/(mean sojourn in steps) * num_business_days = annualised
        lam01 = num_business_days / mean_low
        lam10 = num_business_days / mean_high
        return lam01, lam10

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0,
                  vol_window=21, high_vol_percentile=60.0,
                  kappa_max=10.0, sigma_max=2.0):
        series = data_frame.iloc[:, 0]

        # 1. Regime labels
        is_high = self._regime_labels(series, vol_window, high_vol_percentile)
        is_low  = ~is_high

        # 2. Per-regime OU calibration via calc_statistics
        def _ou_params(subset_df):
            stats, _, _ = utils.calc_statistics(
                subset_df, method='Diff', num_business_days=num_business_days)
            kappa = np.clip(stats['Mean Reversion Speed'].values[0], 1e-4, kappa_max)
            # Reversion Volatility is sigma of the OU process (sd of increments * sqrt(2k))
            sigma = np.clip(stats['Reversion Volatility'].values[0] + vol_shift, 0.0, sigma_max)
            theta = float(stats['Long Run Mean'].values[0])
            return kappa, theta, sigma

        # Fall back to full series if a regime is absent
        df_low  = data_frame[is_low]  if is_low.sum()  > 10 else data_frame
        df_high = data_frame[is_high] if is_high.sum() > 10 else data_frame

        kx_low,  tx_low,  sx_low  = _ou_params(df_low)
        kx_high, tx_high, sx_high = _ou_params(df_high)

        # 3. Transition intensities
        lam01, lam10 = self._transition_intensity(is_high, num_business_days)

        # 4.  y-process defaults (overridable via self.param)
        p = self.param
        ky_scale  = p.get('KY_Scale',  0.2)       # Kappa_Y = ky_scale * Kappa_X
        ty_default = p.get('Theta_Y_Default', 0.0)
        sy_default = p.get('Sigma_Y_Default', 0.0)
        beta_default = p.get('Beta_Default',  0.0)
        rho_default  = p.get('Rho_Default',   0.0)

        params = {
            'Kappa_X_Low':   kx_low,
            'Theta_X_Low':   tx_low,
            'Sigma_X_Low':   sx_low,
            'Kappa_Y_Low':   max(kx_low  * ky_scale, 1e-4),
            'Theta_Y_Low':   ty_default,
            'Sigma_Y_Low':   sy_default,
            'Beta_Low':      beta_default,
            'Rho_Low':       rho_default,
            'Kappa_X_High':  kx_high,
            'Theta_X_High':  tx_high,
            'Sigma_X_High':  sx_high,
            'Kappa_Y_High':  max(kx_high * ky_scale, 1e-4),
            'Theta_Y_High':  ty_default,
            'Sigma_Y_High':  sy_default,
            'Beta_High':     beta_default,
            'Rho_High':      rho_default,
            'Lambda_01':     lam01,
            'Lambda_10':     lam10,
        }

        # 5. Correlation matrix: Z1/Z2 are independent inputs; within-regime
        #    x-y correlation is handled entirely by the Van Loan Cholesky L.
        correlation_coef = np.eye(2)

        # Return delta from the full series for time-grid purposes
        _, _, delta = utils.calc_statistics(
            data_frame, method='Diff', num_business_days=num_business_days)

        return utils.CalibrationInfo(params, correlation_coef, delta)


class RegimeSwitchingOU2FactorHMMCalibration(object):
    """HMM-based calibration of RegimeSwitchingOU2FactorModel from an observed carry series.

    This class is a practical calibration of a **regime-switching 1-factor OU**
    for the directly observed carry factor x_t.  It is **not** a full latent-state
    MLE of the continuous-time 2-factor system — the second factor y remains
    inactive by default (Sigma_Y = Beta = Rho = 0).

    Approach
    --------
    1.  A 2-state ``GaussianHMM`` is fitted on the observation vector
        ``[x_t, dx_t]`` (level + increment) with multiple random restarts;
        the best non-degenerate fit (by log-likelihood) is retained.

    2.  States are reordered so that regime 0 = low carry, regime 1 = high carry
        (primary key: higher mean x_t; secondary: higher variance of dx_t).

    3.  Regime-conditional OU parameters (Kappa_X, Theta_X, Sigma_X) are
        estimated via **weighted AR(1) regression** using the HMM posterior
        probabilities as observation weights, which avoids hard-subsetting a
        non-contiguous series.

    4.  CTMC transition intensities are derived analytically from the discrete
        HMM transition matrix.

    Diagnostics are stored on ``self.last_result`` after each call to
    ``calibrate()``.

    Parameters (via ``self.param``)
    --------------------------------
    HMM_Num_States          int    2
    HMM_Covariance_Type     str    'full'
    HMM_N_Iter              int    500
    HMM_Tol                 float  1e-4
    HMM_N_Init              int    10
    HMM_Random_State        int    1234
    HMM_Use_Diff_Feature    bool   True    — use [x_t, dx_t]; False → [x_t] only
    KY_Scale                float  0.2     — Kappa_Y = KY_Scale * Kappa_X
    Theta_Y_Default         float  0.0
    Sigma_Y_Default         float  0.0
    Beta_Default            float  0.0
    Rho_Default             float  0.0
    Kappa_Max               float  10.0
    Sigma_Max               float  2.0
    Min_Regime_Weight       float  0.05    — min total posterior mass for a
                                            regime to be considered non-degenerate
    """

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 2
        self.last_result = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def calibrate(self, data_frame, vol_shift=0.0, num_business_days=252.0, **kwargs):
        """Calibrate from a DataFrame whose first column is the carry series x_t.

        Parameters
        ----------
        data_frame          : pd.DataFrame  (index = dates, col 0 = x_t)
        vol_shift           : float         optional: added to Sigma_X_Low/High after fit
        num_business_days   : float         trading days per year (default 252)

        Returns
        -------
        utils.CalibrationInfo(params, np.eye(2), delta)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                'hmmlearn is required for RegimeSwitchingOU2FactorHMMCalibration. '
                'Install it with: pip install hmmlearn')

        p = self.param
        n_states          = int(p.get('HMM_Num_States',       2))
        cov_type          = str(p.get('HMM_Covariance_Type',  'full'))
        n_iter            = int(p.get('HMM_N_Iter',           500))
        tol               = float(p.get('HMM_Tol',            1e-4))
        n_init            = int(p.get('HMM_N_Init',           10))
        rand_state        = int(p.get('HMM_Random_State',     1234))
        use_diff          = bool(p.get('HMM_Use_Diff_Feature', True))
        ky_scale          = float(p.get('KY_Scale',            0.2))
        ty_default        = float(p.get('Theta_Y_Default',     0.0))
        sy_default        = float(p.get('Sigma_Y_Default',     0.0))
        beta_default      = float(p.get('Beta_Default',        0.0))
        rho_default       = float(p.get('Rho_Default',         0.0))
        kappa_max         = float(p.get('Kappa_Max',           10.0))
        sigma_max         = float(p.get('Sigma_Max',           2.0))
        min_regime_weight = float(p.get('Min_Regime_Weight',   0.05))

        dt = 1.0 / num_business_days

        # --- 1. Prepare series ------------------------------------------------
        series = self._prepare_series(data_frame)

        # --- 2. Build observation matrix -------------------------------------
        obs = self._build_observations(series, use_diff_feature=use_diff)

        # --- 3. Fit HMM (multi-start, best non-degenerate) -------------------
        hmm_model = self._fit_best_hmm(
            obs, n_states, cov_type, n_iter, tol, n_init, rand_state, min_regime_weight)

        # --- 4. Posterior probabilities & Viterbi decoding -------------------
        posterior     = hmm_model.predict_proba(obs)      # [n, n_states]
        decoded_raw   = hmm_model.predict(obs)            # [n]
        log_likelihood = hmm_model.score(obs)

        # --- 5. Reorder states: 0=Low, 1=High --------------------------------
        low_idx, high_idx, posterior, decoded = self._reorder_states(
            hmm_model, obs, posterior, decoded_raw)

        startprob = hmm_model.startprob_[[low_idx, high_idx]]
        trans_mat = hmm_model.transmat_[np.ix_([low_idx, high_idx],
                                                [low_idx, high_idx])]

        # --- 6. Weighted AR(1) per regime ------------------------------------
        x_t   = series.values[:-1]
        x_tp1 = series.values[1:]
        w_low  = posterior[:-1, 0]   # weights for Low regime
        w_high = posterior[:-1, 1]   # weights for High regime

        fit_low  = self._weighted_ar1_fit(x_t, x_tp1, w_low,  dt, kappa_max, sigma_max)
        fit_high = self._weighted_ar1_fit(x_t, x_tp1, w_high, dt, kappa_max, sigma_max)

        kx_low,  tx_low,  sx_low  = fit_low['kappa'],  fit_low['theta'],  fit_low['sigma']
        kx_high, tx_high, sx_high = fit_high['kappa'], fit_high['theta'], fit_high['sigma']

        # Optional vol_shift bump
        sx_low  = min(sx_low  + vol_shift, sigma_max)
        sx_high = min(sx_high + vol_shift, sigma_max)

        # --- 7. Transition intensities from discrete matrix ------------------
        lam01, lam10 = self._transition_probs_to_intensities(trans_mat, dt)

        # --- 8. Initial high-regime probability ------------------------------
        init_p_hi = float(startprob[1])

        # --- 9. Build parameter dict -----------------------------------------
        params = self._build_params(
            kx_low, tx_low, sx_low, kx_high, tx_high, sx_high,
            lam01, lam10, init_p_hi,
            ky_scale, ty_default, sy_default, beta_default, rho_default)

        # --- 10. delta for CalibrationInfo compatibility ---------------------
        _, _, delta = utils.calc_statistics(
            data_frame, method='Diff', num_business_days=num_business_days)

        # --- 11. Store diagnostics -------------------------------------------
        self.last_result = {
            'log_likelihood':             log_likelihood,
            'transition_matrix':          trans_mat,
            'startprob':                  startprob,
            'posterior':                  posterior,
            'decoded_states':             decoded,
            'state_means':                hmm_model.means_[[low_idx, high_idx]],
            'state_covars':               hmm_model.covars_[[low_idx, high_idx]],
            'weighted_fit_low':           fit_low,
            'weighted_fit_high':          fit_high,
            'expected_duration_low_years':  1.0 / lam01,
            'expected_duration_high_years': 1.0 / lam10,
            'observations':               obs,
            'series_used':                series,
        }

        return utils.CalibrationInfo(params, np.eye(2), delta)

    # ------------------------------------------------------------------
    # Helper: data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_series(data_frame):
        """Sort, drop NaN, validate length.  Returns a pd.Series."""
        series = data_frame.iloc[:, 0].sort_index().dropna()
        if len(series) < 250:
            raise ValueError(
                'RegimeSwitchingOU2FactorHMMCalibration requires at least 250 '
                'observations; got {}.'.format(len(series)))
        return series

    # ------------------------------------------------------------------
    # Helper: build HMM observation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _build_observations(series, use_diff_feature=True):
        """Return feature matrix [n_obs, d].

        If use_diff_feature is True: d=2, columns = [x_t, dx_t].
        Otherwise: d=1, column = [x_t].
        """
        x = series.values.astype(float)
        if use_diff_feature:
            dx = np.diff(x, prepend=x[0])   # dx[0] = 0 by convention
            return np.column_stack([x, dx])
        return x.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Helper: fit best HMM over multiple random starts
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_best_hmm(observations, n_states, covariance_type, n_iter, tol,
                      n_init, random_state, min_regime_weight):
        from hmmlearn.hmm import GaussianHMM

        best_model  = None
        best_score  = -np.inf
        n_obs       = observations.shape[0]
        rng         = np.random.default_rng(random_state)

        for trial in range(n_init):
            seed = int(rng.integers(0, 2 ** 31))
            try:
                candidate = GaussianHMM(
                    n_components=n_states,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    tol=tol,
                    random_state=seed,
                    verbose=False,
                )
                candidate.fit(observations)
                ll = candidate.score(observations)

                # Reject degenerate fits: any regime with near-zero occupancy
                posterior = candidate.predict_proba(observations)
                regime_mass = posterior.mean(axis=0)
                if np.any(regime_mass < min_regime_weight):
                    continue

                if ll > best_score:
                    best_score = ll
                    best_model = candidate

            except Exception:
                continue

        if best_model is None:
            # Last resort: refit once without the occupancy filter
            fallback = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
                verbose=False,
            )
            fallback.fit(observations)
            best_model = fallback

        return best_model

    # ------------------------------------------------------------------
    # Helper: reorder states so 0=Low, 1=High by mean carry level
    # ------------------------------------------------------------------

    @staticmethod
    def _reorder_states(model, observations, posterior, decoded):
        """Map HMM states to economic states: 0=Low carry, 1=High carry.

        Primary ranking: mean of the first feature (x_t level).
        Secondary tie-break: variance of the second feature (dx_t).
        """
        n_states  = model.n_components
        x_means   = model.means_[:, 0]

        if n_states == 2:
            # Simple 2-state case
            if x_means[0] <= x_means[1]:
                # state 0 is already Low
                low_idx, high_idx = 0, 1
            else:
                low_idx, high_idx = 1, 0
        else:
            order     = np.argsort(x_means)
            low_idx   = int(order[0])
            high_idx  = int(order[-1])

        # Reorder posterior columns: col 0 = Low, col 1 = High
        reordered_post   = np.column_stack([posterior[:, low_idx],
                                             posterior[:, high_idx]])
        # Remap decoded states
        remap            = {low_idx: 0, high_idx: 1}
        reordered_decoded = np.array([remap.get(s, s) for s in decoded])

        return low_idx, high_idx, reordered_post, reordered_decoded

    # ------------------------------------------------------------------
    # Helper: weighted AR(1) → OU parameters
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_ar1_fit(x_t, x_tp1, weights, dt, kappa_max, sigma_max):
        """Fit  x_{t+1} = a + b * x_t + eps  with observation weights.

        Returns a dict with keys: a, b, q, kappa, theta, sigma.
        """
        eps = 1e-12
        w   = np.clip(weights, 0.0, None)
        w_sum = w.sum()
        if w_sum < eps:
            # Degenerate: no weight — fall back to OLS
            w = np.ones_like(weights)
            w_sum = w.sum()

        # Weighted means
        mx  = (w * x_t).sum()   / w_sum
        my  = (w * x_tp1).sum() / w_sum

        # Weighted (co)variances
        cxx = (w * (x_t   - mx) ** 2).sum() / w_sum
        cxy = (w * (x_t   - mx) * (x_tp1 - my)).sum() / w_sum

        if cxx < eps:
            b = 1.0 - 1e-6   # near-unit root fallback
        else:
            b = cxy / cxx

        b = float(np.clip(b, 1e-6, 1.0 - 1e-6))
        a = my - b * mx

        # Weighted residual variance
        resid = x_tp1 - (a + b * x_t)
        q = (w * resid ** 2).sum() / w_sum
        q = max(q, eps)

        # AR(1) → continuous-time OU
        kappa = float(np.clip(-np.log(b) / dt, 1e-4, kappa_max))
        theta = float(a / (1.0 - b))
        sigma_sq = q * 2.0 * kappa / max(1.0 - np.exp(-2.0 * kappa * dt), eps)
        sigma = float(np.clip(np.sqrt(max(sigma_sq, 0.0)), 0.0, sigma_max))

        return {'a': a, 'b': b, 'q': q, 'kappa': kappa, 'theta': theta, 'sigma': sigma}

    # ------------------------------------------------------------------
    # Helper: discrete transition matrix → CTMC intensities
    # ------------------------------------------------------------------

    @staticmethod
    def _transition_probs_to_intensities(P, dt):
        """Convert 2×2 discrete transition matrix P to annualised CTMC rates.

        Uses  Lambda = -log(max(1-p, eps)) / dt.
        """
        eps   = 1e-12
        p01   = float(np.clip(P[0, 1], 0.0, 1.0))
        p10   = float(np.clip(P[1, 0], 0.0, 1.0))
        lam01 = -np.log(max(1.0 - p01, eps)) / dt
        lam10 = -np.log(max(1.0 - p10, eps)) / dt
        return float(lam01), float(lam10)

    # ------------------------------------------------------------------
    # Helper: assemble output parameter dict
    # ------------------------------------------------------------------

    @staticmethod
    def _build_params(kx_low, tx_low, sx_low, kx_high, tx_high, sx_high,
                      lam01, lam10, init_p_hi,
                      ky_scale, ty_default, sy_default, beta_default, rho_default):
        return {
            'Kappa_X_Low':             kx_low,
            'Theta_X_Low':             tx_low,
            'Sigma_X_Low':             sx_low,
            'Kappa_Y_Low':             max(kx_low  * ky_scale, 1e-4),
            'Theta_Y_Low':             ty_default,
            'Sigma_Y_Low':             sy_default,
            'Beta_Low':                beta_default,
            'Rho_Low':                 rho_default,
            'Kappa_X_High':            kx_high,
            'Theta_X_High':            tx_high,
            'Sigma_X_High':            sx_high,
            'Kappa_Y_High':            max(kx_high * ky_scale, 1e-4),
            'Theta_Y_High':            ty_default,
            'Sigma_Y_High':            sy_default,
            'Beta_High':               beta_default,
            'Rho_High':                rho_default,
            'Lambda_01':               lam01,
            'Lambda_10':               lam10,
            'Initial_Regime_Prob_High': init_p_hi,
        }

class RegimeSwitchingOU1FactorKalmanCalibration(object):
    """Kalman-filter calibration of a latent 1-factor switching OU carry model.

    Observation equation
    --------------------
        m_t = tau_t * x_t + eps_t
        eps_t ~ N(0, R)

    where ``m_t`` is an observed raw log basis (for example ``log(F_t / S_t)``)
    and ``tau_t`` is the time-to-expiry in years. The latent carry state ``x_t``
    follows a regime-dependent Ornstein-Uhlenbeck process:

        dx_t = kappa[z_t] * (theta[z_t] - x_t) dt + sigma[z_t] dW_t

    The latent regime ``z_t`` is a 2-state Markov chain. Estimation uses an
    approximate EM loop with a switching Kalman filter and collapsed regime
    mixtures. Parameters are returned in the existing
    ``RegimeSwitchingOU2FactorModel`` format, with the second factor switched off
    by default (Sigma_Y = Beta = Rho = 0).
    """

    def __init__(self, model, param):
        self.model = model
        self.param = param or {}
        self.num_factors = 2
        self.last_result = {}

    @staticmethod
    def _safe_tau(tau, floor=1e-4):
        return np.clip(np.asarray(tau, dtype=float), floor, np.inf)

    def _extract_observations(self, data_frame):
        df = data_frame.copy()
        cols = list(df.columns)
        lower = {c: str(c).lower() for c in cols}

        tau_col = next((c for c in cols if any(k in lower[c] for k in ['tau', 'tenor', 'expiry'])), None)
        obs_col = next((c for c in cols if any(k in lower[c] for k in ['raw_basis', 'log_basis', 'basis_obs', 'observed_basis'])), None)

        if obs_col is None:
            basis_carry_col = next((c for c in cols if 'basis_carry' in lower[c]), None)
            if basis_carry_col is not None and tau_col is not None:
                obs = df[basis_carry_col].astype(float).values * df[tau_col].astype(float).values
                reconstructed = True
            else:
                non_tau_cols = [c for c in cols if c != tau_col]
                if not non_tau_cols:
                    raise ValueError('Need at least one observation column for Kalman calibration.')
                obs_col = non_tau_cols[0]
                obs = df[obs_col].astype(float).values
                reconstructed = False
        else:
            obs = df[obs_col].astype(float).values
            reconstructed = False

        if tau_col is None:
            raise ValueError('Kalman calibration requires a tau / tenor column in the archive data.')

        tau = df[tau_col].astype(float).values
        valid = np.isfinite(obs) & np.isfinite(tau) & (tau > 0.0)
        if valid.sum() < 50:
            raise ValueError('Need at least 50 valid observations with positive tau for Kalman calibration.')

        obs = obs[valid]
        tau = tau[valid]
        out = df.loc[valid].copy()
        out['_obs'] = obs
        out['_tau'] = tau
        return out, reconstructed

    @staticmethod
    def _weighted_ar1_fit(x_t, x_tp1, weights, dt, kappa_max, sigma_max):
        eps = 1e-12
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        w_sum = w.sum()
        if w_sum < eps:
            w = np.ones_like(x_t, dtype=float)
            w_sum = w.sum()

        mx = np.sum(w * x_t) / w_sum
        my = np.sum(w * x_tp1) / w_sum
        cxx = np.sum(w * (x_t - mx) ** 2) / w_sum
        cxy = np.sum(w * (x_t - mx) * (x_tp1 - my)) / w_sum

        if cxx < eps:
            b = 1.0 - 1e-6
        else:
            b = cxy / cxx
        b = float(np.clip(b, 1e-6, 1.0 - 1e-6))
        a = float(my - b * mx)

        resid = x_tp1 - (a + b * x_t)
        q = float(max(np.sum(w * resid ** 2) / w_sum, eps))

        kappa = float(np.clip(-np.log(b) / max(dt, eps), 1e-4, kappa_max))
        theta = float(a / max(1.0 - b, eps))
        sigma_sq = q * 2.0 * kappa / max(1.0 - np.exp(-2.0 * kappa * dt), eps)
        sigma = float(np.clip(np.sqrt(max(sigma_sq, 0.0)), 0.0, sigma_max))
        return {'a': a, 'b': b, 'q': q, 'kappa': kappa, 'theta': theta, 'sigma': sigma}

    @staticmethod
    def _transition_probs_to_intensities(P, dt):
        eps = 1e-12
        p01 = float(np.clip(P[0, 1], 0.0, 1.0 - eps))
        p10 = float(np.clip(P[1, 0], 0.0, 1.0 - eps))
        lam01 = -np.log(max(1.0 - p01, eps)) / max(dt, eps)
        lam10 = -np.log(max(1.0 - p10, eps)) / max(dt, eps)
        return lam01, lam10

    @staticmethod
    def _normal_pdf(v, s):
        s = max(float(s), 1e-12)
        return np.exp(-0.5 * (v * v) / s) / np.sqrt(2.0 * np.pi * s)

    def _switching_kalman_filter(self, obs, tau, params, trans_mat, init_prob, meas_var_t, dt):
        kappa = np.asarray(params['kappa'], dtype=float)
        theta = np.asarray(params['theta'], dtype=float)
        sigma = np.asarray(params['sigma'], dtype=float)

        n = obs.shape[0]
        n_reg = 2
        trans_mat = np.clip(np.asarray(trans_mat, dtype=float), 1e-8, 1.0)
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)
        init_prob = np.clip(np.asarray(init_prob, dtype=float), 1e-8, 1.0)
        init_prob /= init_prob.sum()
        meas_var_t = np.asarray(meas_var_t, dtype=float)
        meas_var_t = np.clip(meas_var_t, 1e-10, np.inf)

        a = np.exp(-kappa * dt)
        c = (1.0 - a) * theta
        q = sigma * sigma * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
        q = np.clip(q, 1e-10, np.inf)

        regime_prob = np.zeros((n, n_reg))
        state_mean = np.zeros((n, n_reg))
        state_var = np.zeros((n, n_reg))
        pair_prob = np.zeros((n, n_reg, n_reg))
        emission_mass = np.zeros((n, n_reg, n_reg))
        loglik = 0.0

        var0 = np.maximum(sigma * sigma / np.maximum(2.0 * kappa, 1e-6), 1e-4)
        pred_mean = theta.copy()
        pred_var = var0.copy()
        prev_prob = init_prob.copy()

        for t in range(n):
            H = float(tau[t])
            y = float(obs[t])

            w = np.zeros((n_reg, n_reg))
            m_upd = np.zeros((n_reg, n_reg))
            p_upd = np.zeros((n_reg, n_reg))

            for i in range(n_reg):
                for j in range(n_reg):
                    m_pred = a[j] * pred_mean[i] + c[j]
                    p_pred = a[j] * a[j] * pred_var[i] + q[j]
                    R = meas_var_t[t]
                    s = H * H * p_pred + R
                    v = y - H * m_pred
                    lik = self._normal_pdf(v, s)
                    w[i, j] = prev_prob[i] * trans_mat[i, j] * lik
                    K = p_pred * H / s
                    m_upd[i, j] = m_pred + K * v
                    p_upd[i, j] = max((1.0 - K * H) * p_pred, 1e-12)
                    emission_mass[t, i, j] = lik

            total = w.sum()
            if not np.isfinite(total) or total <= 0.0:
                total = 1e-300
                w = np.full_like(w, 1.0 / (n_reg * n_reg))

            pair_prob[t] = w / total
            regime_prob[t] = pair_prob[t].sum(axis=0)
            loglik += np.log(total)

            next_mean = np.zeros(n_reg)
            next_var = np.zeros(n_reg)
            for j in range(n_reg):
                pj = regime_prob[t, j]
                if pj <= 1e-15:
                    next_mean[j] = theta[j]
                    next_var[j] = var0[j]
                    continue
                weights_j = pair_prob[t, :, j] / pj
                mu = np.sum(weights_j * m_upd[:, j])
                var = np.sum(weights_j * (p_upd[:, j] + (m_upd[:, j] - mu) ** 2))
                next_mean[j] = mu
                next_var[j] = max(var, 1e-12)

            state_mean[t] = next_mean
            state_var[t] = next_var
            pred_mean = next_mean
            pred_var = next_var
            prev_prob = regime_prob[t]

        return {
            'loglik': loglik,
            'regime_prob': regime_prob,
            'state_mean': state_mean,
            'state_var': state_var,
            'pair_prob': pair_prob,
            'emission_mass': emission_mass,
        }

    @staticmethod
    def _build_measurement_variance(tau,
                                    base_meas_var=1e-4,
                                    tau_floor=1e-4,
                                    tau_power=1.0,
                                    tau_scale=0.0,
                                    roll_mask=None,
                                    roll_mult=1.0,
                                    max_meas_var=1.0):
        """
        Build time-varying measurement variance R_t.

        R_t = base_meas_var + tau_scale / tau**tau_power

        Parameters
        ----------
        tau : array-like
            Time to expiry in years.
        base_meas_var : float
            Base observation noise floor.
        tau_floor : float
            Lower bound for tau.
        tau_power : float
            1.0 or 2.0 are sensible starting points.
        tau_scale : float
            Strength of tenor-dependent inflation.
        roll_mask : bool array-like or None
            If provided, multiply R_t by roll_mult where roll_mask is True.
        roll_mult : float
            Roll-window inflation multiplier.
        max_meas_var : float
            Clip to avoid absurd variances.
        """
        tau = np.clip(np.asarray(tau, dtype=float), tau_floor, np.inf)
        R_t = base_meas_var + tau_scale / np.power(tau, tau_power)

        if roll_mask is not None:
            roll_mask = np.asarray(roll_mask, dtype=bool)
            R_t = np.where(roll_mask, R_t * roll_mult, R_t)

        return np.clip(R_t, base_meas_var, max_meas_var)

    def calibrate(self, data_frame, vol_shift=0.0, num_business_days=252.0, **kwargs):
        p = self.param
        max_iter = int(p.get('KF_Max_Iter', 25))
        tol = float(p.get('KF_Tol', 1e-5))
        kappa_max = float(p.get('Kappa_Max', 10.0))
        sigma_max = float(p.get('Sigma_Max', 2.0))
        tau_floor = float(p.get('Tau_Floor', 1e-4))
        ky_scale = float(p.get('KY_Scale', 0.2))
        ty_default = float(p.get('Theta_Y_Default', 0.0))
        sy_default = float(p.get('Sigma_Y_Default', 0.0))
        beta_default = float(p.get('Beta_Default', 0.0))
        rho_default = float(p.get('Rho_Default', 0.0))
        min_meas_var = float(p.get('Min_Measurement_Var', 1e-6))
        rolling_clip = float(p.get('Proxy_Clip_Std', 4.0))
        base_meas_var = float(p.get('Base_Measurement_Var', 1e-4))
        tau_meas_scale = float(p.get('Tau_Measurement_Scale', 1e-5))
        tau_meas_power = float(p.get('Tau_Measurement_Power', 1.0))
        roll_meas_mult = float(p.get('Roll_Measurement_Mult', 1.0))
        max_meas_var = float(p.get('Max_Measurement_Var', 1.0))

        df, reconstructed = self._extract_observations(data_frame)
        obs = df['_obs'].astype(float).values
        tau = self._safe_tau(df['_tau'].values, floor=tau_floor)
        dt = 1.0 / num_business_days
        n = obs.shape[0]

        # --- Warm-start: recover a proxy carry series and fit a single OU ----
        proxy = obs / tau
        if rolling_clip > 0.0:
            mu_p, sd_p = np.nanmean(proxy), np.nanstd(proxy)
            proxy = np.clip(proxy, mu_p - rolling_clip * sd_p, mu_p + rolling_clip * sd_p)

        fit_init = self._weighted_ar1_fit(
            proxy[:-1], proxy[1:], np.ones(n - 1), dt, kappa_max, sigma_max)

        # Regime params: Low = quieter, High = more volatile
        ou_params = {
            'kappa': np.clip([fit_init['kappa'] * 0.5, fit_init['kappa'] * 1.5], 1e-4, kappa_max),
            'theta': np.array([fit_init['theta'], fit_init['theta']]),
            'sigma': np.clip([fit_init['sigma'] * 0.7, fit_init['sigma'] * 1.3], 1e-8, sigma_max),
        }

        trans_mat  = np.array([[0.95, 0.05], [0.05, 0.95]])
        init_prob  = np.array([0.5, 0.5])
        roll_mask = None
        if 'ignore_roll' in df.columns:
            roll_mask = df['ignore_roll'].astype(bool).values

        meas_var_t = self._build_measurement_variance(
            tau=tau,
            base_meas_var=max(base_meas_var, min_meas_var),
            tau_floor=tau_floor,
            tau_power=tau_meas_power,
            tau_scale=tau_meas_scale,
            roll_mask=roll_mask,
            roll_mult=roll_meas_mult,
            max_meas_var=max_meas_var
        )

        prev_loglik = -np.inf
        kf_result   = None

        # --- EM loop ----------------------------------------------------------
        for em_iter in range(max_iter):

            # E-step: switching Kalman filter
            kf_result = self._switching_kalman_filter(
                obs, tau, ou_params, trans_mat, init_prob, meas_var_t, dt)
            loglik      = kf_result['loglik']
            regime_prob = kf_result['regime_prob']   # [n, 2]
            state_mean  = kf_result['state_mean']    # [n, 2]
            pair_prob   = kf_result['pair_prob']     # [n, 2, 2]

            # M-step 1: per-regime OU params via weighted AR(1) on filtered means
            new_kappa = np.zeros(2)
            new_theta = np.zeros(2)
            new_sigma = np.zeros(2)
            for j in range(2):
                fit_j = self._weighted_ar1_fit(
                    state_mean[:-1, j], state_mean[1:, j],
                    regime_prob[:-1, j], dt, kappa_max, sigma_max)
                new_kappa[j] = fit_j['kappa']
                new_theta[j] = fit_j['theta']
                new_sigma[j] = min(fit_j['sigma'] + vol_shift, sigma_max)

            ou_params = {
                'kappa': np.clip(new_kappa, 1e-4, kappa_max),
                'theta': new_theta,
                'sigma': np.clip(new_sigma, 1e-8, sigma_max),
            }

            # M-step 2: transition matrix from soft pair counts
            trans_counts = pair_prob.sum(axis=0)           # [2, 2]
            row_sums     = trans_counts.sum(axis=1, keepdims=True)
            row_sums     = np.where(row_sums > 1e-12, row_sums, 1.0)
            trans_mat    = trans_counts / row_sums

            # M-step 3: measurement variance from collapsed residuals
            x_hat = np.einsum('tj,tj->t', regime_prob, state_mean)
            resid2 = (obs - tau * x_hat) ** 2

            base_est = max(float(np.median(resid2)), min_meas_var)

            meas_var_t = self._build_measurement_variance(
                tau=tau,
                base_meas_var=base_est,
                tau_floor=tau_floor,
                tau_power=tau_meas_power,
                tau_scale=tau_meas_scale,
                roll_mask=roll_mask,
                roll_mult=roll_meas_mult,
                max_meas_var=max_meas_var
            )

            # M-step 4: initial regime probability
            init_prob = np.clip(regime_prob[0], 1e-8, 1.0)
            init_prob /= init_prob.sum()

            if abs(loglik - prev_loglik) < tol * max(1.0, abs(prev_loglik)):
                break
            prev_loglik = loglik

        # --- Build output parameter dict -------------------------------------
        kx_low,  tx_low,  sx_low  = float(ou_params['kappa'][0]), float(ou_params['theta'][0]), float(ou_params['sigma'][0])
        kx_high, tx_high, sx_high = float(ou_params['kappa'][1]), float(ou_params['theta'][1]), float(ou_params['sigma'][1])

        lam01, lam10 = self._transition_probs_to_intensities(trans_mat, dt)
        init_p_hi    = float(init_prob[1])

        out_params = {
            'Kappa_X_Low':              kx_low,
            'Theta_X_Low':              tx_low,
            'Sigma_X_Low':              sx_low,
            'Kappa_Y_Low':              max(kx_low  * ky_scale, 1e-4),
            'Theta_Y_Low':              ty_default,
            'Sigma_Y_Low':              sy_default,
            'Beta_Low':                 beta_default,
            'Rho_Low':                  rho_default,
            'Kappa_X_High':             kx_high,
            'Theta_X_High':             tx_high,
            'Sigma_X_High':             sx_high,
            'Kappa_Y_High':             max(kx_high * ky_scale, 1e-4),
            'Theta_Y_High':             ty_default,
            'Sigma_Y_High':             sy_default,
            'Beta_High':                beta_default,
            'Rho_High':                 rho_default,
            'Lambda_01':                lam01,
            'Lambda_10':                lam10,
            'Initial_Regime_Prob_High': init_p_hi,
        }

        _, _, delta = utils.calc_statistics(
            df.iloc[:,:2], method='Diff', num_business_days=num_business_days)

        self.last_result = {
            'log_likelihood':               prev_loglik,
            'em_iterations':                em_iter + 1,
            'transition_matrix':            trans_mat,
            'init_prob':                    init_prob,
            'regime_prob':                  kf_result['regime_prob'] if kf_result else None,
            'state_mean':                   kf_result['state_mean']  if kf_result else None,
            'state_var':                    kf_result['state_var']   if kf_result else None,
            'measurement_var_t':            meas_var_t,
            'measurement_var_base':         float(np.median(meas_var_t)),
            'expected_duration_low_years':  1.0 / lam01,
            'expected_duration_high_years': 1.0 / lam10,
            'params_low':  {'kappa': kx_low,  'theta': tx_low,  'sigma': sx_low},
            'params_high': {'kappa': kx_high, 'theta': tx_high, 'sigma': sx_high},
            'observations':                 obs,
            'tau':                          tau,
            'reconstructed_obs':            reconstructed,
        }

        return utils.CalibrationInfo(out_params, np.eye(2), delta)


class SingleRegimeOU1FactorKalmanModel(StochasticProcess):
    """Single-regime latent 1-factor OU carry model.

    Companion simulation process for ``SingleRegimeOU1FactorKalmanCalibration``.

    State dynamics:

        dx_t = Kappa * (Theta - x_t) dt + Sigma dW_t

    Exact discrete-time step over interval dt:

        a = exp(-Kappa * dt)
        c = (1 - a) * Theta
        q = Sigma^2 * (1 - exp(-2*Kappa*dt)) / (2*Kappa)
        x_{t+1} = a * x_t + c + sqrt(q) * Z_t,   Z_t ~ N(0,1)

    One correlated Gaussian driver is consumed per step.
    The generated path is the latent carry state x_t (annualised), shape [T, batch].

    Parameters
    ----------
    Kappa   : float  Mean-reversion speed (> 0).
    Theta   : float  Long-run carry level.
    Sigma   : float  OU volatility (>= 0).
    Initial_X : float  Initial carry state (default 0.0).
    """

    documentation = (
        'Asset Pricing',
        ['A single-regime latent 1-factor Ornstein-Uhlenbeck carry model.',
         '',
         'The latent carry state $x_t$ follows:',
         '',
         '$$ dx_t = \\kappa(\\theta - x_t)\\,dt + \\sigma\\,dW_t $$',
         '',
         'Exact discrete-time step over interval $\\delta$:',
         '',
         '$$ x_{t+\\delta} = \\theta + e^{-\\kappa\\delta}(x_t - \\theta)'
         ' + \\sigma\\sqrt{\\frac{1-e^{-2\\kappa\\delta}}{2\\kappa}}\\,Z,'
         '\\quad Z\\sim\\mathcal{N}(0,1) $$',
         '',
         'One correlated Gaussian driver is consumed per time step.',
         ''])

    def __init__(self, factor, param, implied_factor=None):
        super(SingleRegimeOU1FactorKalmanModel, self).__init__(factor, param)
        self._validate_params()

    def _validate_params(self):
        p = self.param
        if p.get('Kappa', 0.0) <= 0.0:
            self.params_ok = False
            return
        if p.get('Sigma', 0.0) < 0.0:
            self.params_ok = False

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        dt_arr = np.diff(np.hstack(([0.0], time_grid.time_grid_years)))

        kappa = float(self.param['Kappa'])
        theta = float(self.param['Theta'])
        sigma = float(self.param['Sigma'])

        a = np.exp(-kappa * dt_arr)
        c = (1.0 - a) * theta
        q = sigma * sigma * (1.0 - np.exp(-2.0 * kappa * dt_arr)) / max(2.0 * kappa, 1e-12)
        q = np.clip(q, 0.0, None)

        self.a_arr   = shared.one.new_tensor(a.reshape(-1, 1))           # [T, 1]
        self.c_arr   = shared.one.new_tensor(c.reshape(-1, 1))           # [T, 1]
        self.vol_arr = shared.one.new_tensor(np.sqrt(q).reshape(-1, 1))  # [T, 1]
        self.x0      = tensor

    @property
    def correlation_name(self):
        return 'SingleRegimeOUCarryProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]  # [T, batch]
        # Prefix cumprod of mean-reversion decay: A[k] = prod(a[0..k])       [T, 1]
        A = torch.cumprod(self.a_arr, dim=0)
        # Innovation at each step: b[k] = c[k] + vol[k] * Z[k]              [T, batch]
        b = self.c_arr + self.vol_arr * Z
        # Closed-form recurrence: x[k] = A[k] * (x0 + cumsum(b/A)[k])
        # Derivation: unrolling x_k = a_k*x_{k-1}+b_k gives
        #   x_k = A_k*x0 + sum_{j<k} (A_k/A_j)*b_j = A_k*(x0 + cumsum(b/A)[k])
        return A * (self.x0 + torch.cumsum(b / A, dim=0))                    # [T, batch]


class SingleRegimeOU1FactorKalmanCalibration(object):
    """Kalman-filter calibration of a single-regime latent 1-factor OU carry model.

    A simpler alternative to ``RegimeSwitchingOU1FactorKalmanCalibration`` for
    cases where regime switching is not warranted.

    Observation equation
    --------------------
        m_t = tau_t * x_t + eps_t,  eps_t ~ N(0, R_t)

    where ``m_t`` is the raw log basis (e.g. ``log(F_t / S_t)``) and ``tau_t``
    is the remaining tenor in years. The latent carry state ``x_t`` follows a
    single Ornstein-Uhlenbeck process:

        dx_t = kappa * (theta - x_t) dt + sigma dW_t

    Exact discrete-time transition over interval ``dt``:

        a = exp(-kappa * dt)
        x_{t+1} = a * x_t + (1 - a) * theta + sqrt(q) * w_t
        q = sigma^2 * (1 - exp(-2*kappa*dt)) / (2*kappa)

    Time-varying measurement variance:

        R_t = base_meas_var + tau_meas_scale / tau_t**tau_meas_power

    Estimation uses a quasi-EM loop:
        E-step: standard Kalman filter forward pass to obtain filtered state means.
        M-step 1: weighted AR(1) on filtered means --> updated kappa, theta, sigma.
        M-step 2: residual-based update of measurement variance base.

    Output parameters are in a 1-factor OU format compatible with the
    corresponding stochastic process: Kappa, Theta, Sigma,
    Measurement_Var_Base.
    """

    def __init__(self, model, param):
        self.model = model
        self.param = param or {}
        self.num_factors = 1
        self.last_result = {}

    @staticmethod
    def _safe_tau(tau, floor=1e-4):
        return np.clip(np.asarray(tau, dtype=float), floor, np.inf)

    def _extract_observations(self, data_frame):
        """Return a cleaned copy of data_frame with '_obs' and '_tau' columns appended.

        Column discovery (case-insensitive substring match):
          - tau column : any column whose name contains 'tau', 'tenor', or 'expiry'
          - obs column : any column whose name contains 'raw_spread', 'raw_basis',
                         'log_basis', 'basis_obs', or 'observed_basis'.
            Falls back to the first non-tau column if none of those keywords match.

        Rows with non-finite obs or tau, or with tau <= 0, are dropped.
        Raises ValueError if fewer than 30 valid rows remain.
        """
        df = data_frame.copy()
        cols = list(df.columns)
        lower = {c: str(c).lower() for c in cols}

        tau_col = next(
            (c for c in cols if any(k in lower[c] for k in ['tau', 'tenor', 'expiry'])),
            None
        )
        if tau_col is None:
            raise ValueError('Kalman calibration requires a tau / tenor column in the archive data.')

        obs_col = next(
            (c for c in cols if any(k in lower[c]
                                    for k in ['raw_spread', 'raw_basis', 'log_basis',
                                              'basis_obs', 'observed_basis'])),
            None
        )
        if obs_col is None:
            # Fall back to first non-tau column
            non_tau = [c for c in cols if c != tau_col]
            if not non_tau:
                raise ValueError('Need at least one observation column for Kalman calibration.')
            obs_col = non_tau[0]

        obs = df[obs_col].astype(float).values
        tau = df[tau_col].astype(float).values
        valid = np.isfinite(obs) & np.isfinite(tau) & (tau > 0.0)
        if valid.sum() < 30:
            raise ValueError(
                f'Need at least 30 valid observations with positive tau for Kalman calibration '
                f'(got {int(valid.sum())}).')

        out = df.loc[valid].copy()
        out['_obs'] = obs[valid]
        out['_tau'] = tau[valid]
        return out

    @staticmethod
    def _weighted_ar1_fit(x_t, x_tp1, weights, dt, kappa_max, sigma_max):
        """Fit AR(1) parameters from a weighted state sequence and map to OU params.

        Returns a dict with keys: kappa, theta, sigma, q.
        """
        eps = 1e-12
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        w_sum = w.sum()
        if w_sum < eps:
            w = np.ones_like(x_t, dtype=float)
            w_sum = float(len(x_t))

        mx = np.sum(w * x_t) / w_sum
        my = np.sum(w * x_tp1) / w_sum
        cxx = np.sum(w * (x_t - mx) ** 2) / w_sum
        cxy = np.sum(w * (x_t - mx) * (x_tp1 - my)) / w_sum

        b = float(np.clip(cxy / cxx if cxx > eps else (1.0 - 1e-6), 1e-6, 1.0 - 1e-6))
        a = float(my - b * mx)

        resid = x_tp1 - (a + b * x_t)
        q = float(max(np.sum(w * resid ** 2) / w_sum, eps))

        kappa = float(np.clip(-np.log(b) / max(dt, eps), 1e-4, kappa_max))
        theta = float(a / max(1.0 - b, eps))
        sigma_sq = q * 2.0 * kappa / max(1.0 - np.exp(-2.0 * kappa * dt), eps)
        sigma = float(np.clip(np.sqrt(max(sigma_sq, 0.0)), 0.0, sigma_max))
        return {'kappa': kappa, 'theta': theta, 'sigma': sigma, 'q': q}

    @staticmethod
    def _build_measurement_variance(tau, base_meas_var, tau_floor, tau_power, tau_scale,
                                    min_meas_var, max_meas_var):
        """Build time-varying measurement variance R_t = base_meas_var + tau_scale / tau**tau_power."""
        tau = np.clip(np.asarray(tau, dtype=float), tau_floor, np.inf)
        R_t = base_meas_var + tau_scale / np.power(tau, tau_power)
        return np.clip(R_t, min_meas_var, max_meas_var)

    @staticmethod
    def _kalman_filter(obs, tau, kappa, theta, sigma, meas_var_t, dt):
        """Run a standard linear Kalman filter for the single-regime OU carry model.

        State transition:   x_{t+1} = a*x_t + c + sqrt(q)*w_t,  w_t ~ N(0,1)
        Observation:        m_t = H_t * x_t + eps_t,             eps_t ~ N(0, R_t)
        where a = exp(-kappa*dt), c = (1-a)*theta, H_t = tau_t.

        Returns a dict with: loglik, state_mean [n], state_var [n],
        innovations [n], innov_var [n].
        """
        n = obs.shape[0]
        eps = 1e-12

        a = np.exp(-kappa * dt)
        c = (1.0 - a) * theta
        q = sigma * sigma * (1.0 - np.exp(-2.0 * kappa * dt)) / max(2.0 * kappa, eps)
        q = max(q, 1e-10)

        # Initialise at unconditional (stationary) distribution
        x_filt = float(theta)
        p_filt = sigma * sigma / max(2.0 * kappa, eps)

        state_mean  = np.zeros(n)
        state_var   = np.zeros(n)
        innovations = np.zeros(n)
        innov_var   = np.zeros(n)
        loglik = 0.0

        for t in range(n):
            # --- Predict ---
            x_pred = a * x_filt + c
            p_pred = a * a * p_filt + q

            H = float(tau[t])
            R = float(meas_var_t[t])
            s = H * H * p_pred + R          # innovation variance
            v = float(obs[t]) - H * x_pred  # innovation

            if s > eps:
                loglik += -0.5 * (np.log(2.0 * np.pi * s) + v * v / s)

            # --- Update ---
            K = p_pred * H / max(s, eps)
            x_filt = x_pred + K * v
            p_filt = max((1.0 - K * H) * p_pred, 1e-12)

            state_mean[t]  = x_filt
            state_var[t]   = p_filt
            innovations[t] = v
            innov_var[t]   = s

        return {
            'loglik':      loglik,
            'state_mean':  state_mean,
            'state_var':   state_var,
            'innovations': innovations,
            'innov_var':   innov_var,
        }

    def calibrate(self, data_frame, vol_shift=0.0, num_business_days=252.0, **kwargs):
        p = self.param
        max_iter       = int(p.get('KF_Max_Iter', 30))
        tol            = float(p.get('KF_Tol', 1e-6))
        kappa_max      = float(p.get('Kappa_Max', 10.0))
        sigma_max      = float(p.get('Sigma_Max', 2.0))
        tau_floor      = float(p.get('Tau_Floor', 1e-4))
        base_meas_var  = float(p.get('Base_Measurement_Var', 1e-4))
        tau_meas_scale = float(p.get('Tau_Measurement_Scale', 0.0))
        tau_meas_power = float(p.get('Tau_Measurement_Power', 1.0))
        max_meas_var   = float(p.get('Max_Measurement_Var', 1.0))
        min_meas_var   = float(p.get('Min_Measurement_Var', 1e-8))

        df  = self._extract_observations(data_frame)
        obs = df['_obs'].astype(float).values
        tau = self._safe_tau(df['_tau'].values, floor=tau_floor)
        dt  = 1.0 / num_business_days
        n   = obs.shape[0]

        # --- Warm-start: proxy carry x ≈ obs/tau, then AR(1) OU fit --------
        proxy = obs / tau
        fit = self._weighted_ar1_fit(
            proxy[:-1], proxy[1:], np.ones(n - 1), dt, kappa_max, sigma_max)
        kappa = fit['kappa']
        theta = fit['theta']
        sigma = fit['sigma']

        meas_var_t = self._build_measurement_variance(
            tau=tau,
            base_meas_var=max(base_meas_var, min_meas_var),
            tau_floor=tau_floor,
            tau_power=tau_meas_power,
            tau_scale=tau_meas_scale,
            min_meas_var=min_meas_var,
            max_meas_var=max_meas_var,
        )

        prev_loglik = -np.inf
        kf_result   = None

        # --- Quasi-EM loop ---------------------------------------------------
        for em_iter in range(max_iter):

            # E-step: standard Kalman filter
            kf_result  = self._kalman_filter(obs, tau, kappa, theta, sigma, meas_var_t, dt)
            loglik     = kf_result['loglik']
            state_mean = kf_result['state_mean']    # [n]

            # M-step 1: refit OU params from filtered state means via weighted AR(1)
            fit   = self._weighted_ar1_fit(
                state_mean[:-1], state_mean[1:], np.ones(n - 1), dt, kappa_max, sigma_max)
            kappa = fit['kappa']
            theta = fit['theta']
            sigma = float(np.clip(fit['sigma'] + vol_shift, 0.0, sigma_max))

            # M-step 2: update measurement variance base from squared residuals
            resid2   = (obs - tau * state_mean) ** 2
            base_est = max(float(np.median(resid2)), min_meas_var)

            meas_var_t = self._build_measurement_variance(
                tau=tau,
                base_meas_var=base_est,
                tau_floor=tau_floor,
                tau_power=tau_meas_power,
                tau_scale=tau_meas_scale,
                min_meas_var=min_meas_var,
                max_meas_var=max_meas_var,
            )

            if abs(loglik - prev_loglik) < tol * max(1.0, abs(prev_loglik)):
                break
            prev_loglik = loglik

        out_params = {
            'Kappa':                float(np.clip(kappa, 1e-4, kappa_max)),
            'Theta':                float(theta),
            'Sigma':                float(np.clip(sigma, 0.0, sigma_max)),
            'Measurement_Var_Base': float(np.median(meas_var_t)),
        }

        _, _, delta = utils.calc_statistics(
            df.iloc[:, :2], method='Diff', num_business_days=num_business_days)

        self.last_result = {
            'log_likelihood':     prev_loglik,
            'em_iterations':      em_iter + 1,
            'kappa':              kappa,
            'theta':              theta,
            'sigma':              sigma,
            'measurement_var_t':  meas_var_t,
            'state_mean':         kf_result['state_mean']  if kf_result else None,
            'state_var':          kf_result['state_var']   if kf_result else None,
            'innovations':        kf_result['innovations'] if kf_result else None,
            'observations':       obs,
            'tau':                tau,
        }

        return utils.CalibrationInfo(out_params, [[1.0]], delta)


class LogOUSpotModel(StochasticProcess):
    """Log-space Ornstein-Uhlenbeck spot-price model.

    Latent state X_t = log(S_t) follows:

        dX_t = Kappa * (Theta - X_t) dt + Sigma dW_t

    The exact discretisation over an interval dt is:

        X_{t+dt} = Theta + exp(-Kappa*dt) * (X_t - Theta) + eps
        eps ~ N(0, Sigma^2 * (1 - exp(-2*Kappa*dt)) / (2*Kappa))

    Simulated paths are returned as S_t = exp(X_t), which are always positive.
    One correlated Gaussian driver is consumed per step.
    """

    documentation = (
        'Asset Pricing',
        ['The log-spot $X_t = \\log S_t$ follows a mean-reverting Ornstein-Uhlenbeck process:',
         '',
         '$$ dX_t = \\kappa(\\theta - X_t)\\,dt + \\sigma\\,dW_t $$',
         '',
         'The exact discretisation over a step $\\delta$ is:',
         '',
         '$$ X_{t+\\delta} = \\theta + e^{-\\kappa\\delta}(X_t - \\theta)'
         ' + \\sigma\\sqrt{\\frac{1-e^{-2\\kappa\\delta}}{2\\kappa}}\\,Z,'
         '\\quad Z\\sim\\mathcal{N}(0,1) $$',
         '',
         'Simulated paths are returned as $S_t = \\exp(X_t) > 0$.',
         '',
         'Parameters:',
         '',
         '- **Spot**: Initial spot price $S_0 > 0$',
         '- **Kappa**: Mean-reversion speed $\\kappa > 0$',
         '- **Theta**: Long-run mean of $\\log S$',
         '- **Sigma**: Volatility $\\sigma \\geq 0$',
         '',
         'The stationary distribution of $\\log S$ is'
         ' $\\mathcal{N}\\!\\left(\\theta,\\,\\frac{\\sigma^2}{2\\kappa}\\right)$.'])

    def __init__(self, factor, param, implied_factor=None):
        super(LogOUSpotModel, self).__init__(factor, param)
        self._validate_params()

    def _validate_params(self):
        p = self.param
        if p.get('Kappa', 0.0) <= 0.0:
            self.params_ok = False
            return
        if p.get('Sigma', 0.0) < 0.0:
            self.params_ok = False
            return

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        dt = np.diff(np.hstack(([0.0], time_grid.time_grid_years)))
        kappa = self.param['Kappa']
        sigma = self.param['Sigma']

        # exact OU step: mean-reversion factor and conditional std-dev
        e_kdt   = np.exp(-kappa * dt)
        var_step = sigma * sigma * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)

        self.e_kdt   = tensor.new(e_kdt.reshape(-1, 1))
        self.ou_vol  = tensor.new(np.sqrt(var_step).reshape(-1, 1))
        self.theta   = float(self.param['Theta'])

        # initial log-spot
        self.log_spot0 = float(torch.log(tensor).item()) if tensor.numel() == 1 else torch.log(tensor)

    def theoretical_mean_std(self, t):
        """Theoretical mean and std of S_t = exp(X_t) under the exact OU distribution."""
        kappa = self.param['Kappa']
        sigma = self.param['Sigma']
        theta = self.param['Theta']
        e_kt  = np.exp(-kappa * t)
        mu_X  = theta + e_kt * (self.log_spot0 - theta)
        var_X = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * t)) / (2.0 * kappa)
        # lognormal moments of exp(X) where X ~ N(mu_X, var_X)
        mean_S = np.exp(mu_X + 0.5 * var_X)
        std_S  = mean_S * np.sqrt(np.exp(var_X) - 1.0)
        return mean_S, std_S

    @property
    def correlation_name(self):
        return 'OULogSpotProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]  # [T, batch]
        A = torch.cumprod(self.e_kdt, dim=0)                           # [T, 1]
        b = self.theta * (1.0 - self.e_kdt) + self.ou_vol * Z         # [T, batch]
        log_spot = A * (self.log_spot0 + torch.cumsum(b / A, dim=0))  # [T, batch]
        return torch.exp(log_spot)


class LogOUSpotCalibration(object):
    """Calibrate LogOUSpotModel parameters from historical spot price data.

    Uses ``utils.calc_statistics`` in log space to estimate:

    - **Kappa**  — mean-reversion speed (``Mean Reversion Speed``)
    - **Sigma**  — OU volatility in log space (``Reversion Volatility``)
    - **Theta**  — long-run mean of log S, recovered by inverting the
      lognormal expectation:

          Long Run Mean = exp(theta + sigma^2 / (4*kappa))
          =>  Theta = log(Long Run Mean) - sigma^2 / (4*kappa)

    The ``Spot`` parameter is the current market value and is not calibrated here.
    """

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0,
                  kappa_max=10.0, sigma_max=2.0):
        stats, correlation, delta = utils.calc_statistics(
            data_frame, method='Log', num_business_days=num_business_days)

        kappa = stats['Mean Reversion Speed'].values[0]
        sigma = stats['Reversion Volatility'].values[0]

        # calc_statistics 'Log' returns Long Run Mean = exp(theta + sigma^2/(4*kappa))
        # Invert to recover theta = long-run mean of log(S)
        lrm   = np.clip(stats['Long Run Mean'].values[0], 1e-8, np.inf)
        theta = np.log(lrm) - sigma ** 2 / (4.0 * kappa)

        return utils.CalibrationInfo(
            {'Kappa': np.clip(kappa, 1e-4, kappa_max),
             'Theta': theta,
             'Sigma': np.clip(sigma + vol_shift, 0.0, sigma_max)},
            [[1.0]], delta)


def construct_process(sp_type, factor, param, implied_factor=None):
    return globals().get(sp_type)(factor, param, implied_factor)


def construct_calibration_config(calibration_model, param):
    return globals().get(param['Method'])(calibration_model, param)
