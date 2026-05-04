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
from scipy.linalg import expm as matrix_expm, logm as matrix_logm
from scipy.optimize import minimize as scipy_minimize
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

    @classmethod
    def privileged_layout(cls, param):
        """Static {name: dim} schema of privileged factors this process emits, derivable from
        param alone for the policy's privileged-encoder sizing at construction time."""
        return {}

    def privileged_factors(self, simulated):
        """Privileged factors the asymmetric critic sees but the actor does not — dict of
        (T, B, dim) tensors keyed by name. `simulated` is this process's (T, B) path."""
        return {}


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

        delta_var = np.diff(self.J)
        self.exp_minus_alpha_t = shared.one.new(np.exp(-alpha * time_grid_years)).reshape(-1, 1)

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
              self.delta_vol - self.delta_KtT + self.delta_HtT).cumsum(axis=0) * self.exp_minus_alpha_t

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
        # Initial log-spot
        self.log_spot0 = float(torch.log(tensor).item()) if tensor.numel() == 1 else torch.log(tensor)
        # Anchor theta at the current spot rather than the calibrated long-run mean. Production
        # hedging shouldn't bet that today's price will revert to a historical average — the agent
        # should hedge variance around the current regime, not directional drift toward a stale θ.
        # Disable via `Anchor_Theta_At_Spot: false` for backtests that want absolute calibrated θ.
        if bool(self.param.get('Anchor_Theta_At_Spot', True)):
            log_spot_scalar = self.log_spot0 if isinstance(self.log_spot0, float) else float(self.log_spot0.mean().item())
            self.theta = log_spot_scalar
        else:
            self.theta = float(self.param['Theta'])

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

    @classmethod
    def privileged_layout(cls, param):
        return {'log_deviation': 1, 'kappa': 1, 'sigma': 1}

    def privileged_factors(self, simulated):
        spot = simulated.to(dtype=torch.float32)
        # Use self.theta (post-anchor) so the critic sees the actual θ used during simulation.
        log_dev = (spot.clamp_min(1.0e-9).log() - float(self.theta)).unsqueeze(-1)
        kappa_t = torch.full_like(log_dev, float(self.param['Kappa']))
        sigma_t = torch.full_like(log_dev, float(self.param['Sigma']))
        return {'log_deviation': log_dev, 'kappa': kappa_t, 'sigma': sigma_t}


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


class MarkovSwitchingLogOUSpotModel(StochasticProcess):
    """N-state hidden-Markov LogOU spot-price model. Conditional on the latent regime z_t, the
    log-spot follows the same exact OU discretisation as `LogOUSpotModel`; a discrete Markov
    chain over regimes drives parameter switching. One Gaussian driver per step (num_factors=1);
    regime transitions sampled from the independent quasi-rng uniform stream.

    JSON config:
        States: list of {Kappa, Theta, Sigma} dicts (one per regime, must be >= 2)
        Transition_Matrix: NxN row-stochastic matrix at the calibration time step
        Initial_State_Probs: length-N vector summing to 1
        Calibration_DT_Years: step size of the calibrated P (default 1/252).

    The calibrated P is converted to a CTMC generator Q = log(P)/dt_calib once, then re-discretised
    per simulation step P_step = expm(Q * dt) so a daily-calibrated chain stays faithful when the
    sim grid uses non-daily steps."""

    documentation = (
        'Asset Pricing',
        ['An N-state hidden-Markov extension of the LogOU spot model. The latent regime '
         '$z_t \\in \\{0,...,N-1\\}$ follows a Markov chain with transition matrix $P$; '
         'conditional on $z_t$ the log-spot follows:',
         '',
         '$$ dX_t = \\kappa_{z_t} (\\theta_{z_t} - X_t) dt + \\sigma_{z_t} dW_t $$',
         '',
         'Exact OU discretisation per step (regime frozen over the interval):',
         '',
         '$$ X_{t+\\delta} = \\theta_z + e^{-\\kappa_z\\delta}(X_t - \\theta_z) '
         '+ \\sigma_z \\sqrt{\\frac{1-e^{-2\\kappa_z\\delta}}{2\\kappa_z}}\\,Z $$',
         '',
         'Regime transitions: $z_{t+1} \\sim \\text{Categorical}(P[z_t, :])$, drawn from the '
         'quasi-RNG uniform stream — independent of the OU Gaussian. The configured P is at '
         'calibration step $\\delta_c$; for simulation step $\\delta$ the model uses '
         '$P_\\delta = \\exp(\\log(P) \\delta / \\delta_c)$.',
         '',
         'Parameters:',
         '- **Spot**: Initial spot price.',
         '- **States**: List of $\\{\\kappa, \\theta, \\sigma\\}$ per regime.',
         '- **Transition_Matrix**: NxN row-stochastic matrix at the calibration step.',
         '- **Initial_State_Probs**: Initial regime distribution (length N).',
         '- **Calibration_DT_Years**: Step size $\\delta_c$ of the calibrated $P$ (default 1/252).'])

    def __init__(self, factor, param, implied_factor=None):
        super(MarkovSwitchingLogOUSpotModel, self).__init__(factor, param)
        self._validate_params()

    def _validate_params(self):
        states = self.param.get('States') or []
        if len(states) < 2:
            self.params_ok = False
            return
        for s in states:
            if s.get('Kappa', 0.0) <= 0.0 or s.get('Sigma', 0.0) < 0.0:
                self.params_ok = False
                return
        n = len(states)
        P = self.param.get('Transition_Matrix')
        if not isinstance(P, list) or len(P) != n:
            self.params_ok = False
            return
        for row in P:
            if not isinstance(row, list) or len(row) != n:
                self.params_ok = False
                return
            if abs(sum(row) - 1.0) > 1.0e-6:
                self.params_ok = False
                return
        pi0 = self.param.get('Initial_State_Probs', [1.0 / n] * n)
        if len(pi0) != n or abs(sum(pi0) - 1.0) > 1.0e-6:
            self.params_ok = False

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        dt_arr = np.diff(np.hstack(([0.0], time_grid.time_grid_years)))
        states = self.param['States']
        self.n_states = len(states)
        T = len(dt_arr)

        e_kdt = np.zeros((self.n_states, T))
        ou_vol = np.zeros((self.n_states, T))
        thetas = np.zeros(self.n_states)
        for i, s in enumerate(states):
            kappa = float(s['Kappa'])
            sigma = float(s['Sigma'])
            e_kdt[i] = np.exp(-kappa * dt_arr)
            var_step = sigma * sigma * (1.0 - np.exp(-2.0 * kappa * dt_arr)) / (2.0 * kappa)
            ou_vol[i] = np.sqrt(np.clip(var_step, 0.0, None))
            thetas[i] = float(s['Theta'])

        # Anchor theta at current spot rather than the calibrated long-run mean. Production
        # hedging shouldn't bet that today's price will revert to a stale historical average —
        # the agent should hedge variance around the current regime, not directional drift.
        # Preserve the calibrated *relative* spread between regime means (state 1 still hotter
        # than state 0, etc.) but center the stationary log-mean at log(current_spot).
        # Disable via `Anchor_Theta_At_Spot: false` for backtests that want absolute calibrated θ.
        if bool(self.param.get('Anchor_Theta_At_Spot', True)):
            log_spot_scalar = float(torch.log(tensor).mean().item()) if tensor.numel() > 1 else float(torch.log(tensor).item())
            pi0 = np.array(self.param.get('Initial_State_Probs', [1.0 / self.n_states] * self.n_states))
            calibrated_log_mean = float((pi0 * thetas).sum())
            thetas = thetas + (log_spot_scalar - calibrated_log_mean)
            self.param = dict(self.param)  # don't mutate caller's dict
            self.param['States'] = [
                {**s, 'Theta': float(t)} for s, t in zip(states, thetas)
            ]

        # Convert the calibrated transition matrix to its CTMC generator once, then re-discretise
        # per simulation dt — preserves the daily calibration when the sim grid uses non-daily steps.
        P_calib = np.array(self.param['Transition_Matrix'], dtype=np.float64)
        dt_calib = float(self.param.get('Calibration_DT_Years', 1.0 / 252.0))
        Q = np.real(matrix_logm(P_calib)) / dt_calib
        P_per_step = np.zeros((T, self.n_states, self.n_states))
        for t, dt in enumerate(dt_arr):
            P_per_step[t] = np.real(matrix_expm(Q * dt)) if dt > 1.0e-12 else np.eye(self.n_states)
        P_cum = np.cumsum(P_per_step, axis=2)

        pi0 = np.array(self.param.get('Initial_State_Probs', [1.0 / self.n_states] * self.n_states))
        pi0_cum = np.cumsum(pi0)

        def _t(arr):
            return shared.one.new_tensor(arr)

        self.e_kdt_per_state = _t(e_kdt)
        self.ou_vol_per_state = _t(ou_vol)
        self.thetas = _t(thetas)
        self.P_cum = _t(P_cum)
        self.pi0_cum = _t(pi0_cum)

        self.log_spot0 = float(torch.log(tensor).item()) if tensor.numel() == 1 else torch.log(tensor)

    @property
    def correlation_name(self):
        return 'MarkovSwitchingLogOUProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        T, B = Z.shape
        device = Z.device

        # quasi_rng(dim, sample_size) returns sobol.draw(sample_size) of shape (sample_size, dim).
        # We want (T+1, B) — one initial draw plus one per transition — so pass dim=B, sample_size=T+1.
        u_regime = shared_mem.quasi_rng(shared_mem.simulation_batch, T + 1)[1].contiguous()

        state = torch.searchsorted(self.pi0_cum, u_regime[0]).clamp_max_(self.n_states - 1)
        regimes = torch.empty((T, B), dtype=torch.long, device=device)
        regimes[0] = state
        for t in range(1, T):
            cdf_rows = self.P_cum[t - 1].index_select(0, state)
            state = (cdf_rows < u_regime[t].unsqueeze(1)).sum(dim=1).clamp_max_(self.n_states - 1)
            regimes[t] = state

        t_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, B)
        e_kdt_t = self.e_kdt_per_state[regimes, t_idx]
        ou_vol_t = self.ou_vol_per_state[regimes, t_idx]
        theta_t = self.thetas[regimes]

        A = torch.cumprod(e_kdt_t, dim=0)
        b = theta_t * (1.0 - e_kdt_t) + ou_vol_t * Z
        log_spot = A * (self.log_spot0 + torch.cumsum(b / A, dim=0))
        # Stashed for privileged_factors() called immediately after generate() in the sim loop.
        self.last_regime_path = regimes
        return torch.exp(log_spot)

    @classmethod
    def privileged_layout(cls, param):
        n = len(param.get('States') or [])
        out = {'regime_onehot': n}
        for i in range(n):
            out[f'state{i}_kappa'] = 1
            out[f'state{i}_theta'] = 1
            out[f'state{i}_sigma'] = 1
        return out

    def privileged_factors(self, simulated):
        regimes = self.last_regime_path
        T, B = regimes.shape
        device = regimes.device
        out = {'regime_onehot': torch.nn.functional.one_hot(regimes, num_classes=self.n_states).to(dtype=torch.float32)}
        for i, s in enumerate(self.param['States']):
            for attr in ('Kappa', 'Theta', 'Sigma'):
                out[f'state{i}_{attr.lower()}'] = torch.full((T, B, 1), float(s[attr]), dtype=torch.float32, device=device)
        return out


class CompoundHawkesSpotModel(StochasticProcess):
    """Pure compound bivariate marked Hawkes spot-price model. No diffusion, no
    regime switching — log-price moves entirely through up- and down-jump arrivals
    whose rates self- and cross-excite each other. The mark distribution carries
    all of the volatility information; there is no separate σ.

    Per step δ, for each of K independent Hawkes components k=1..K:

        N⁺_k ~ Poisson(λ⁺_k δ),  N⁻_k ~ Poisson(λ⁻_k δ)        # arrivals
        M⁺_{k,i} ~ Exp(η⁺_k),    M⁻_{k,j} ~ Exp(η⁻_k)          # marks (jump sizes)
        ΣM⁺_k = Σ_{i=1..N⁺_k} M⁺_{k,i}                          # compound Poisson sums
        ΣM⁻_k = Σ_{j=1..N⁻_k} M⁻_{k,j}

    Net log-return aggregated across components:

        r_t = Σ_k (ΣM⁺_k - ΣM⁻_k)
        log_S_t = log_S_{t-1} + r_t

    Intensity update (mark-weighted self- and cross-excitation):

        λ⁺_k(t+δ) = μ⁺_k + e^{-β_k δ}(λ⁺_k(t) - μ⁺_k) + α⁺⁺_k ΣM⁺_k + α⁻⁺_k ΣM⁻_k
        λ⁻_k(t+δ) = μ⁻_k + e^{-β_k δ}(λ⁻_k(t) - μ⁻_k) + α⁺⁻_k ΣM⁺_k + α⁻⁻_k ΣM⁻_k

    Multiple components (K>1) capture multi-scale clustering — e.g., one fast-decay
    process for daily-burst dynamics and one slow-decay process for multi-week regimes.
    Per component, the kernel matrix [α⁺⁺ α⁻⁺; α⁺⁻ α⁻⁻] / (β · E[mark]) must have
    spectral radius < 1 for stationarity.

    JSON config:
        Components: list of dicts, one per Hawkes component:
            {Mu_Plus, Mu_Minus, Beta, Alpha_PP, Alpha_NP, Alpha_PN, Alpha_NN, Eta_Plus, Eta_Minus}

        Mu_*  — baseline arrival intensities (events/year). E.g. Mu=10 ≈ 10 events/yr.
        Beta  — kernel decay rate (1/years). β=5 ≈ 50-day half-life at 252 bd/yr.
        Alpha_** — 4 mark-weighted cross-excitation gains.
        Eta_*  — exponential mark rates: E[|mark|] = 1/η in log-return units.
                 e.g. η=100 ⇒ typical jump = 1%; η=30 ⇒ typical jump = 3.3%.
    """

    documentation = (
        'Asset Pricing',
        ['A pure compound bivariate marked Hawkes spot-price process. Up- and down-jumps '
         'arrive as Poisson events with self- and cross-exciting intensities; jump sizes '
         '(marks) are exponentially distributed and supply all of the volatility — there '
         'is no separate diffusion or regime-switching layer. Per step:',
         '',
         '$$ N^+_k \\sim \\text{Pois}(\\lambda^+_k \\delta), \\quad '
         'N^-_k \\sim \\text{Pois}(\\lambda^-_k \\delta) $$',
         '',
         '$$ M^+_{k,i} \\sim \\text{Exp}(\\eta^+_k), \\quad '
         'M^-_{k,j} \\sim \\text{Exp}(\\eta^-_k) $$',
         '',
         '$$ r_t = \\sum_k \\Big(\\sum_i M^+_{k,i} - \\sum_j M^-_{k,j}\\Big), \\quad '
         '\\log S_t = \\log S_{t-1} + r_t $$',
         '',
         'Intensity update with mark-weighted excitation:',
         '',
         '$$ \\lambda^+_k(t{+}\\delta) = \\mu^+_k + e^{-\\beta_k\\delta}(\\lambda^+_k(t) - \\mu^+_k) '
         '+ \\alpha^{++}_k \\Sigma M^+_k + \\alpha^{-+}_k \\Sigma M^-_k $$',
         '',
         '$$ \\lambda^-_k(t{+}\\delta) = \\mu^-_k + e^{-\\beta_k\\delta}(\\lambda^-_k(t) - \\mu^-_k) '
         '+ \\alpha^{+-}_k \\Sigma M^+_k + \\alpha^{--}_k \\Sigma M^-_k $$',
         '',
         'Multiple components (K>1) capture multi-scale clustering. Per component, '
         'spectral radius of the kernel matrix divided by $\\beta_k \\cdot E[M_k]$ must '
         'be < 1 for stationarity.',
         '',
         'Parameters:',
         '- **Components**: list of dicts, one per Hawkes component '
         '$\\{\\mu^+, \\mu^-, \\beta, \\alpha^{++}, \\alpha^{-+}, \\alpha^{+-}, \\alpha^{--}, \\eta^+, \\eta^-\\}$.'])

    REQUIRED_COMPONENT_KEYS = (
        'Mu_Plus', 'Mu_Minus', 'Beta',
        'Alpha_PP', 'Alpha_NP', 'Alpha_PN', 'Alpha_NN',
        'Eta_Plus', 'Eta_Minus',
    )

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)
        self._validate_params()

    def _validate_params(self):
        comps = self.param.get('Components')
        if not isinstance(comps, list) or len(comps) < 1:
            self.params_ok = False
            return
        for c in comps:
            if not isinstance(c, dict):
                self.params_ok = False
                return
            for k in self.REQUIRED_COMPONENT_KEYS:
                v = c.get(k)
                if not isinstance(v, (int, float)) or v < 0.0:
                    self.params_ok = False
                    return
            if c['Beta'] <= 0.0 or c['Eta_Plus'] <= 0.0 or c['Eta_Minus'] <= 0.0:
                self.params_ok = False
                return

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # process_ofs is retained for parity with peers even though we don't draw from
        # the shared Gaussian pool — Hawkes uses Poisson + Gamma sampling instead.
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        dt_arr = np.diff(np.hstack(([0.0], time_grid.time_grid_years)))
        comps = self.param['Components']
        K = len(comps)

        def _t(arr):
            return shared.one.new_tensor(arr)

        # Per-component scalar parameters, packed (K,) for vectorised loop math.
        self.K = K
        self.mu_plus = _t(np.array([c['Mu_Plus'] for c in comps], dtype=np.float64))
        self.mu_minus = _t(np.array([c['Mu_Minus'] for c in comps], dtype=np.float64))
        self.beta = _t(np.array([c['Beta'] for c in comps], dtype=np.float64))
        self.alpha_pp = _t(np.array([c['Alpha_PP'] for c in comps], dtype=np.float64))
        self.alpha_np = _t(np.array([c['Alpha_NP'] for c in comps], dtype=np.float64))
        self.alpha_pn = _t(np.array([c['Alpha_PN'] for c in comps], dtype=np.float64))
        self.alpha_nn = _t(np.array([c['Alpha_NN'] for c in comps], dtype=np.float64))
        self.eta_plus = _t(np.array([c['Eta_Plus'] for c in comps], dtype=np.float64))
        self.eta_minus = _t(np.array([c['Eta_Minus'] for c in comps], dtype=np.float64))

        # exp(-β_k δ_t) per (component, step) — pre-computed.
        beta_np = np.array([c['Beta'] for c in comps], dtype=np.float64)
        self.decay_per_step = _t(np.exp(-np.outer(beta_np, dt_arr)))  # (K, T)
        self.dt_per_step = _t(dt_arr)                                  # (T,)

        self.log_spot0 = float(torch.log(tensor).item()) if tensor.numel() == 1 else torch.log(tensor)

    @property
    def correlation_name(self):
        return 'CompoundHawkesProcess', [()]

    def generate(self, shared_mem):
        # We don't read the Gaussian pool here, but stay shape-consistent with peers
        # by sizing the output from t_random_numbers / scenario_horizon.
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        T, B = Z.shape
        device = Z.device
        K = self.K

        if isinstance(self.log_spot0, torch.Tensor) and self.log_spot0.numel() > 1:
            log_spot = self.log_spot0.to(device).expand(B).clone()
        else:
            scalar = self.log_spot0 if isinstance(self.log_spot0, float) else float(self.log_spot0.item())
            log_spot = torch.full((B,), scalar, device=device, dtype=torch.float32)

        # Per-component intensities, initialised at baseline. Shape (K, B).
        lam_plus = self.mu_plus.to(dtype=torch.float32).view(K, 1).expand(K, B).contiguous()
        lam_minus = self.mu_minus.to(dtype=torch.float32).view(K, 1).expand(K, B).contiguous()

        log_spot_out = torch.empty((T, B), device=device, dtype=torch.float32)
        h_plus_path = torch.empty((T, B), device=device, dtype=torch.float32)
        h_minus_path = torch.empty((T, B), device=device, dtype=torch.float32)

        # Pre-cast scalar param tensors to float32 once.
        mu_plus = self.mu_plus.to(dtype=torch.float32).view(K, 1)
        mu_minus = self.mu_minus.to(dtype=torch.float32).view(K, 1)
        decay = self.decay_per_step.to(dtype=torch.float32)            # (K, T)
        eta_plus = self.eta_plus.to(dtype=torch.float32).view(K, 1)    # rate of Exp marks
        eta_minus = self.eta_minus.to(dtype=torch.float32).view(K, 1)
        a_pp = self.alpha_pp.to(dtype=torch.float32).view(K, 1)
        a_np = self.alpha_np.to(dtype=torch.float32).view(K, 1)
        a_pn = self.alpha_pn.to(dtype=torch.float32).view(K, 1)
        a_nn = self.alpha_nn.to(dtype=torch.float32).view(K, 1)
        dt_per_step = self.dt_per_step.to(dtype=torch.float32)         # (T,)

        for t in range(T):
            dt = dt_per_step[t]
            decay_t = decay[:, t].view(K, 1)                            # (K, 1)

            # Decay intensities toward baselines (Hawkes mean-reverting kernel).
            lam_plus = mu_plus + decay_t * (lam_plus - mu_plus)
            lam_minus = mu_minus + decay_t * (lam_minus - mu_minus)

            # Sample event counts per (component, path). torch.poisson takes a rate tensor.
            n_plus = torch.poisson(lam_plus * dt)                       # (K, B), real-valued >= 0
            n_minus = torch.poisson(lam_minus * dt)

            # Compound-Poisson mark sums. Σ of N iid Exp(η) is Gamma(N, rate=η);
            # for N=0 the sum is 0 (Gamma(0, ·) is degenerate, mask it out).
            mask_p = (n_plus > 0).to(dtype=torch.float32)
            mask_n = (n_minus > 0).to(dtype=torch.float32)
            shape_p = n_plus.clamp(min=1.0)
            shape_n = n_minus.clamp(min=1.0)
            sum_mp = mask_p * torch.distributions.Gamma(shape_p, eta_plus).sample()    # (K, B)
            sum_mn = mask_n * torch.distributions.Gamma(shape_n, eta_minus).sample()

            # Net log-return summed across components.
            r_t = (sum_mp - sum_mn).sum(dim=0)                          # (B,)
            log_spot = log_spot + r_t
            log_spot_out[t] = log_spot

            # Mark-weighted intensity update.
            lam_plus = lam_plus + a_pp * sum_mp + a_np * sum_mn
            lam_minus = lam_minus + a_pn * sum_mp + a_nn * sum_mn

            # Aggregate (sum across K) for the bundle/feature surface.
            h_plus_path[t] = lam_plus.sum(dim=0)
            h_minus_path[t] = lam_minus.sum(dim=0)

        self.last_h_plus_path = h_plus_path
        self.last_h_minus_path = h_minus_path
        return torch.exp(log_spot_out)

    @classmethod
    def privileged_layout(cls, param):
        K = len(param.get('Components') or [])
        return {
            'hawkes_h_plus_total': 1,
            'hawkes_h_minus_total': 1,
            'hawkes_ratio_total': 1,
            # Per-component intensities for asymmetric critic — gives the critic visibility
            # into multi-scale clustering structure that the actor only sees aggregated.
            'hawkes_h_plus_components': K,
            'hawkes_h_minus_components': K,
        }

    def privileged_factors(self, simulated):
        H_plus = self.last_h_plus_path.to(dtype=torch.float32)
        H_minus = self.last_h_minus_path.to(dtype=torch.float32)
        ratio = H_minus / (H_plus + H_minus + 1.0e-8)
        return {
            'hawkes_h_plus_total': H_plus.unsqueeze(-1),
            'hawkes_h_minus_total': H_minus.unsqueeze(-1),
            'hawkes_ratio_total': ratio.unsqueeze(-1),
        }


class MarkovSwitchingLogOUSpotCalibration(object):
    """Baum-Welch (EM) calibration of MarkovSwitchingLogOUSpotModel from a daily price series.

    Each state is an Ornstein-Uhlenbeck process on log-spot; a discrete N-state Markov chain
    drives parameter switching. Estimation:

        E-step: forward-backward to obtain posterior state probabilities gamma_t(z) and joint
                gamma_t(z, z') over consecutive observations.
        M-step: transition matrix from soft-counts; per-state OU params via weighted MLE on
                (X_t, X_{t+1}) pairs (closed form: regress X_{t+1} = a + b X_t with weights w_t,
                recover kappa = -log(b)/dt, theta = a/(1-b), sigma from residual variance).

    Hyperparameters (read from `param` if provided, else defaults):
        N_States   — number of regimes (default 2)
        N_Iter     — EM iterations cap (default 200)
        Seed       — RNG seed for the small init perturbation (default 42)
        Tol        — relative log-likelihood change for early stopping (default 1e-6)
    """

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    @staticmethod
    def _logsumexp(x, axis=None, keepdims=False):
        m = np.max(x, axis=axis, keepdims=True)
        out = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
        return out if keepdims else np.squeeze(out, axis=axis)

    @staticmethod
    def _emission_logprob(X, kappa, theta, sigma, dt):
        """Per-state log p(X_{t+1} | X_t, z=s) for each transition. Returns (T-1, n_states)."""
        T = len(X) - 1
        n_states = len(kappa)
        out = np.empty((T, n_states))
        for s in range(n_states):
            e_kdt = np.exp(-kappa[s] * dt)
            var = max(sigma[s] ** 2 * (1.0 - np.exp(-2.0 * kappa[s] * dt)) / (2.0 * kappa[s]), 1.0e-12)
            mu = theta[s] + e_kdt * (X[:-1] - theta[s])
            out[:, s] = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (X[1:] - mu) ** 2 / var
        return out

    @classmethod
    def _forward_backward(cls, log_pi, log_P, log_emit):
        T, S = log_emit.shape
        log_alpha = np.full((T, S), -np.inf)
        log_alpha[0] = log_pi + log_emit[0]
        for t in range(1, T):
            log_alpha[t] = cls._logsumexp(log_alpha[t - 1, :, None] + log_P, axis=0) + log_emit[t]
        log_lik = cls._logsumexp(log_alpha[-1])
        log_beta = np.zeros((T, S))
        for t in range(T - 2, -1, -1):
            log_beta[t] = cls._logsumexp(log_P + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1)
        log_gamma = log_alpha + log_beta
        log_gamma -= cls._logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        log_xi = (log_alpha[:-1, :, None] + log_P[None, :, :]
                  + log_emit[1:, None, :] + log_beta[1:, None, :])
        log_xi -= cls._logsumexp(log_xi.reshape(T - 1, -1), axis=1)[:, None, None]
        xi = np.exp(log_xi)
        return gamma, xi, log_lik

    @staticmethod
    def _m_step_ou(X, w, dt, eps=1.0e-12):
        """Weighted OU MLE on (X_t → X_{t+1}) pairs with state-occupancy weights w (length T-1).
        Regress X_{t+1} = a + b X_t with sample weights, recover (kappa, theta, sigma)."""
        sw = max(w.sum(), eps)
        Xt, Xt1 = X[:-1], X[1:]
        mx = (w * Xt).sum() / sw
        my = (w * Xt1).sum() / sw
        sxx = (w * (Xt - mx) ** 2).sum() / sw
        sxy = (w * (Xt - mx) * (Xt1 - my)).sum() / sw
        b = sxy / max(sxx, eps)
        # Clip with margin to keep b in (0, 1) — outside this range OU is non-stationary.
        b = float(np.clip(b, 1.0e-3, 1.0 - 1.0e-6))
        a = my - b * mx
        resid = Xt1 - (a + b * Xt)
        var_resid = max((w * resid ** 2).sum() / sw, eps)
        kappa = -np.log(b) / dt
        theta = a / (1.0 - b)
        sigma2 = var_resid * 2.0 * kappa / max(1.0 - np.exp(-2.0 * kappa * dt), eps)
        return float(kappa), float(theta), float(np.sqrt(max(sigma2, eps)))

    @classmethod
    def _fit_em(cls, prices, dt, n_states=2, n_iter=200, seed=42, tol=1.0e-6):
        rng = np.random.default_rng(seed)
        X = np.log(np.asarray(prices, dtype=np.float64))
        if len(X) < 50:
            raise ValueError('Need at least ~50 observations for stable HMM-LogOU fit')
        log_returns = np.diff(X)
        base_sigma = log_returns.std() / np.sqrt(dt)
        # Init: spread sigma across states so EM has distinguishable regimes from the start.
        if n_states == 2:
            kappa = np.array([0.5, 5.0])
            sigma = np.array([base_sigma * 0.7, base_sigma * 1.5])
        else:
            kappa = rng.uniform(0.2, 5.0, size=n_states)
            sigma = rng.uniform(base_sigma * 0.5, base_sigma * 2.0, size=n_states)
        theta = np.full(n_states, X.mean())
        pi = np.full(n_states, 1.0 / n_states)
        P = np.full((n_states, n_states), 0.05) + 0.85 * np.eye(n_states)
        P /= P.sum(axis=1, keepdims=True)

        log_lik_traj = []
        prev_lik = -np.inf
        for it in range(n_iter):
            log_emit = cls._emission_logprob(X, kappa, theta, sigma, dt)
            gamma, xi, log_lik = cls._forward_backward(np.log(pi + 1e-12), np.log(P + 1e-12), log_emit)
            log_lik_traj.append(float(log_lik))
            if it > 0 and abs(log_lik - prev_lik) < tol * abs(prev_lik):
                break
            prev_lik = log_lik
            pi = gamma[0]
            denom = gamma[:-1].sum(axis=0)
            P_new = xi.sum(axis=0) / np.maximum(denom[:, None], 1.0e-12)
            P_new /= P_new.sum(axis=1, keepdims=True)
            P = P_new
            for s in range(n_states):
                kappa[s], theta[s], sigma[s] = cls._m_step_ou(X, gamma[:, s], dt)

        # Stationary distribution of the converged transition matrix.
        eigvals, eigvecs = np.linalg.eig(P.T)
        stationary = np.real(eigvecs[:, np.isclose(eigvals, 1.0)].flatten())
        if stationary.size == 0:
            stationary = pi
        stationary = stationary / stationary.sum()

        # Order states by sigma (state 0 = least volatile) for deterministic JSON output.
        order = np.argsort(sigma)
        kappa, theta, sigma = kappa[order], theta[order], sigma[order]
        P = P[order][:, order]
        pi = pi[order]
        stationary = stationary[order]

        return {
            'states': [{'Kappa': float(kappa[s]), 'Theta': float(theta[s]), 'Sigma': float(sigma[s])}
                       for s in range(n_states)],
            'transition_matrix': P.tolist(),
            'initial_state_probs': pi.tolist(),
            'stationary_probs': stationary.tolist(),
            'log_likelihood': log_lik_traj[-1],
            'iterations': len(log_lik_traj),
        }

    def calibrate(self, data_frame, vol_shift=0.0, num_business_days=252.0):
        """Fit MS-LogOU via EM on the first column of `data_frame` (assumed daily prices).
        Hyperparameters (N_States, N_Iter, Seed, Tol) read from self.param if provided."""
        n_states = int(self.param.get('N_States', 2))
        n_iter = int(self.param.get('N_Iter', 200))
        seed = int(self.param.get('Seed', 42))
        tol = float(self.param.get('Tol', 1.0e-6))
        prices = data_frame.iloc[:, 0].dropna().astype(float).values
        dt = 1.0 / float(num_business_days)
        fit = self._fit_em(prices, dt, n_states=n_states, n_iter=n_iter, seed=seed, tol=tol)
        states = [
            {
                'Kappa': float(np.clip(s['Kappa'], 1.0e-4, 50.0)),
                'Theta': float(s['Theta']),
                'Sigma': float(np.clip(s['Sigma'] + vol_shift, 0.0, 5.0)),
            }
            for s in fit['states']
        ]
        return utils.CalibrationInfo(
            {
                'States': states,
                'Transition_Matrix': [[float(x) for x in row] for row in fit['transition_matrix']],
                'Initial_State_Probs': [float(x) for x in fit['stationary_probs']],
                'Calibration_DT_Years': dt,
            },
            [[1.0]],
            dt,
        )


class CompoundHawkesSpotCalibration(object):
    """MLE calibration of `CompoundHawkesSpotModel` (K=1 component) from a daily price series.

    Extracts events via a magnitude threshold τ on |r_t|: days with r_t > τ are "up events"
    (mark = r_t), days with r_t < -τ are "down events" (mark = -r_t), and |r_t| ≤ τ are
    non-events. Fits the 9 K=1 parameters {μ⁺, μ⁻, β, α⁺⁺, α⁻⁺, α⁺⁻, α⁻⁻, η⁺, η⁻} by
    maximising the discrete-time Poisson log-likelihood:

        log L_t = log P(N⁺_t, N⁻_t | λ⁺(t), λ⁻(t)) + 1[event]·log f_M(|r_t| | η_sign)

        with N⁺_t, N⁻_t ∈ {0, 1}; the intensities recurse:

        λ⁺(t+1) = μ⁺ + e^{-βδ}(λ⁺(t) - μ⁺) + α⁺⁺ M_t·1[up] + α⁻⁺ M_t·1[down]
        λ⁻(t+1) = μ⁻ + e^{-βδ}(λ⁻(t) - μ⁻) + α⁺⁻ M_t·1[up] + α⁻⁻ M_t·1[down]

    Optimisation in log-parameter space (params = exp(θ)) so positivity is enforced; uses
    L-BFGS-B with multi-start. K>1 calibration requires an EM/SMC scheme to assign events
    to components and is deferred — for K>1 calibrate K=1 then split the kernel manually.

    Hyperparameters (read from `param` if provided):
        Threshold_Quantile  — quantile of |r| above which events are extracted (default 0.5)
        Threshold_Abs       — explicit threshold (overrides quantile if set)
        N_Restarts          — multi-start optimiser restarts (default 4)
        N_Iter              — L-BFGS max iterations per start (default 500)
        Seed                — RNG for init perturbation (default 42)
        Tol                 — relative LL tolerance (default 1e-6)
    """

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    @staticmethod
    def _extract_events(log_returns, threshold):
        """Classify each step as up-event (+1), down-event (-1), or non-event (0)."""
        sign = np.zeros_like(log_returns, dtype=np.int8)
        sign[log_returns > threshold] = 1
        sign[log_returns < -threshold] = -1
        marks = np.abs(log_returns)
        return sign, marks

    @staticmethod
    def _neg_log_lik(log_theta, sign, marks, dt):
        """Discrete-time Poisson log-likelihood under the K=1 Hawkes recursion. log_theta packs
        log of (μ⁺, μ⁻, β, α⁺⁺, α⁻⁺, α⁺⁻, α⁻⁻, η⁺, η⁻); positivity enforced via exp."""
        mu_p, mu_m, beta, a_pp, a_np, a_pn, a_nn, eta_p, eta_m = np.exp(log_theta)
        decay = np.exp(-beta * dt)

        T = sign.shape[0]
        lam_p = mu_p
        lam_m = mu_m
        log_lik = 0.0

        for t in range(T):
            # Discrete-time joint Poisson likelihood for at-most-one-event-per-side per step.
            #   N⁺=1, N⁻=0  (up event):    log(λ⁺δ) − λ⁺δ − λ⁻δ
            #   N⁺=0, N⁻=1  (down event):  log(λ⁻δ) − λ⁺δ − λ⁻δ
            #   N⁺=0, N⁻=0  (no event):    − λ⁺δ − λ⁻δ
            base = -(lam_p + lam_m) * dt
            s = sign[t]
            m = marks[t]
            if s == 1:
                ll_t = np.log(max(lam_p * dt, 1.0e-300)) + base + np.log(eta_p) - eta_p * m
            elif s == -1:
                ll_t = np.log(max(lam_m * dt, 1.0e-300)) + base + np.log(eta_m) - eta_m * m
            else:
                ll_t = base
            log_lik += ll_t

            # Mark-weighted excitation only when an event was observed.
            if s == 1:
                lam_p = mu_p + decay * (lam_p - mu_p) + a_pp * m
                lam_m = mu_m + decay * (lam_m - mu_m) + a_pn * m
            elif s == -1:
                lam_p = mu_p + decay * (lam_p - mu_p) + a_np * m
                lam_m = mu_m + decay * (lam_m - mu_m) + a_nn * m
            else:
                lam_p = mu_p + decay * (lam_p - mu_p)
                lam_m = mu_m + decay * (lam_m - mu_m)

        return -log_lik

    @classmethod
    def _stationarity_radius(cls, mu_p, mu_m, beta, a_pp, a_np, a_pn, a_nn, eta_p, eta_m):
        """Spectral radius of the branching matrix G = αᵀ·diag(E[M⁺], E[M⁻])/β. < 1 ⇒ stationary."""
        kernel = np.array([[a_pp, a_pn], [a_np, a_nn]])
        em = np.array([1.0 / eta_p, 1.0 / eta_m])
        G = kernel.T * em[None, :] / beta
        return float(max(abs(np.linalg.eigvals(G))))

    @classmethod
    def _fit(cls, prices, dt, threshold_quantile=0.5, threshold_abs=None,
             n_restarts=4, n_iter=500, seed=42, tol=1.0e-6):
        rng = np.random.default_rng(seed)
        X = np.log(np.asarray(prices, dtype=np.float64))
        if len(X) < 100:
            raise ValueError('Need at least ~100 observations for stable Hawkes fit')
        log_returns = np.diff(X)
        threshold = (float(threshold_abs) if threshold_abs is not None
                     else float(np.quantile(np.abs(log_returns), threshold_quantile)))
        sign, marks = cls._extract_events(log_returns, threshold)
        n_up = int((sign == 1).sum())
        n_dn = int((sign == -1).sum())
        n_no = int((sign == 0).sum())

        # Method-of-moments init: per-side baseline rate from event count, mark scale from
        # mean of in-class marks, β default ≈ 5 (50d half-life), α small to start near additive.
        T = len(log_returns)
        mu_p_init = max(n_up / (T * dt), 1.0e-3)
        mu_m_init = max(n_dn / (T * dt), 1.0e-3)
        em_p_init = float(marks[sign == 1].mean()) if n_up else float(marks.mean())
        em_m_init = float(marks[sign == -1].mean()) if n_dn else float(marks.mean())
        eta_p_init = 1.0 / max(em_p_init, 1.0e-6)
        eta_m_init = 1.0 / max(em_m_init, 1.0e-6)
        # α scaled so initial branching ratio ≈ 0.3 — meaningful but well stationary.
        target_branch = 0.3
        beta_init = 5.0
        a_init = target_branch * beta_init / max(em_p_init, em_m_init)

        log_theta_0 = np.log(np.array([
            mu_p_init, mu_m_init, beta_init,
            a_init, a_init, a_init, a_init,
            eta_p_init, eta_m_init,
        ]))

        best = None
        for r in range(n_restarts):
            jitter = rng.normal(0.0, 0.3, size=log_theta_0.shape)
            x0 = log_theta_0 if r == 0 else log_theta_0 + jitter
            res = scipy_minimize(
                cls._neg_log_lik, x0, args=(sign, marks, dt),
                method='L-BFGS-B', tol=tol,
                options={'maxiter': n_iter, 'disp': False},
            )
            if best is None or res.fun < best.fun:
                best = res

        params = np.exp(best.x)
        mu_p, mu_m, beta, a_pp, a_np, a_pn, a_nn, eta_p, eta_m = params
        rho = cls._stationarity_radius(*params)

        return {
            'components': [{
                'Mu_Plus': float(mu_p),
                'Mu_Minus': float(mu_m),
                'Beta': float(beta),
                'Alpha_PP': float(a_pp),
                'Alpha_NP': float(a_np),
                'Alpha_PN': float(a_pn),
                'Alpha_NN': float(a_nn),
                'Eta_Plus': float(eta_p),
                'Eta_Minus': float(eta_m),
            }],
            'log_likelihood': float(-best.fun),
            'iterations': int(best.nit),
            'threshold': threshold,
            'event_counts': {'up': n_up, 'down': n_dn, 'none': n_no},
            'mean_mark_up': em_p_init,
            'mean_mark_down': em_m_init,
            'spectral_radius': rho,
            'stationary': rho < 1.0,
        }

    def calibrate(self, data_frame, vol_shift=0.0, num_business_days=252.0):
        """Fit CompoundHawkesSpotModel (K=1) to the first column of `data_frame`. Hyperparameters
        (Threshold_Quantile, Threshold_Abs, N_Restarts, N_Iter, Seed, Tol) read from self.param."""
        threshold_q = float(self.param.get('Threshold_Quantile', 0.5))
        threshold_abs = self.param.get('Threshold_Abs', None)
        n_restarts = int(self.param.get('N_Restarts', 4))
        n_iter = int(self.param.get('N_Iter', 500))
        seed = int(self.param.get('Seed', 42))
        tol = float(self.param.get('Tol', 1.0e-6))
        prices = data_frame.iloc[:, 0].dropna().astype(float).values
        dt = 1.0 / float(num_business_days)
        fit = self._fit(prices, dt, threshold_quantile=threshold_q,
                        threshold_abs=threshold_abs, n_restarts=n_restarts,
                        n_iter=n_iter, seed=seed, tol=tol)
        return utils.CalibrationInfo(
            {'Components': fit['components']},
            [[1.0]],
            dt,
        )


def construct_process(sp_type, factor, param, implied_factor=None):
    return globals().get(sp_type)(factor, param, implied_factor)


def construct_calibration_config(calibration_model, param):
    return globals().get(param['Method'])(calibration_model, param)
