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
import copy
import logging
import itertools

# 3rd party libraries
import numpy as np
import pandas as pd
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


def hmm_forward_backward(log_pi, log_P, log_emit):
    """Log-space forward-backward for a discrete-state HMM. `log_emit` is (T, S) of
    per-step per-state emission log-densities; returns smoothed posteriors `gamma`
    (T, S), pairwise posteriors `xi` (T-1, S, S), and log-likelihood. Used by every
    Markov-style calibration class."""
    from scipy.special import logsumexp
    T, S = log_emit.shape
    log_alpha = np.full((T, S), -np.inf)
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_P, axis=0) + log_emit[t]
    log_lik = logsumexp(log_alpha[-1])
    log_beta = np.zeros((T, S))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_P + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    log_xi = (log_alpha[:-1, :, None] + log_P[None, :, :]
              + log_emit[1:, None, :] + log_beta[1:, None, :])
    log_xi -= logsumexp(log_xi.reshape(T - 1, -1), axis=1)[:, None, None]
    xi = np.exp(log_xi)
    return gamma, xi, log_lik


class StochasticProcess(object):
    """Base class for all stochastic processes"""

    def __init__(self, factor, param):
        self.factor = factor
        self.param = param
        self.params_ok = True

    def copy(self):
        """Shallow copy. Use case: forking a process for nested simulation (inner MC)
        so the fork can be precalculated against a different shared state / time grid
        without clobbering the outer instance's precalc-derived attributes (`spot0`,
        `scenario_horizon`, `z_offset`, etc.). Construction-time references (`factor`,
        `param`, `implied`) are shared by reference, which is intentional — these are
        read-only after setup."""
        return copy.copy(self)

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
        # per-step incremental vol, anchored at V(0)=0 so the first step evolves from today (t=0)
        self.delta_vol = torch.sqrt(self.V - F.pad(self.V[:-1], (0, 0, 1, 0)))
        # we always evolve from today: prepend 0 to the sample times so the step sizes are the diffs
        # [t_1, t_2 - t_1, ...] and each step's drift is read from the curve as-of its start node
        # [0, t_1, ..., t_{N-1}] (when t_1 = 0 the first step is a no-op and we match spot today)
        self.delta_scen_t = np.diff(np.insert(time_grid.scen_time_grid, 0, 0)).reshape(-1, 1)
        # store a reference to the current tensor
        self.spot = tensor
        # the curve-read grid is the step starts: today (t=0) followed by all but the final sample
        today = time_grid.scenario_grid[:1].copy()
        today[:, utils.TIME_GRID_MTM] = 0.0
        self.scen_grid = np.vstack([today, time_grid.scenario_grid[:-1]])
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
            delta_var[delta_var < 0.0] = delta_var[delta_var >= 0.0].min()
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
        # tensor: (n_tenors,) outer / (n_tenors, B) inner. Only `initial_curve` depends
        # on tensor — vol/drift are time-grid + param functions and stay (T, n_tenors, 1).
        if tensor.ndim == 1:
            self.initial_curve = tensor.reshape(1, -1, 1)                   # (1, n_tenors, 1)
        else:
            self.initial_curve = tensor.unsqueeze(0).unsqueeze(-1)          # (1, n_tenors, B, 1)
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
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        if Z.ndim == 2:
            # Outer mode: Z is (T, B). Bit-exact preserves legacy behavior.
            z_portion = Z.unsqueeze(1) * self.vol                       # (T, n_tenors, B)
            return self.initial_curve * torch.exp(self.drift + torch.cumsum(z_portion, dim=0))
        # Inner mode: Z is (T, B, B2). One path per outer × inner.
        vol4 = self.vol.unsqueeze(-1)                                   # (T, n_tenors, 1, 1)
        drift4 = self.drift.unsqueeze(-1)                               # (T, n_tenors, 1, 1)
        z_portion = Z.unsqueeze(1) * vol4                               # (T, n_tenors, B, B2)
        return self.initial_curve * torch.exp(drift4 + torch.cumsum(z_portion, dim=0))


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

        # Anchor at time_grid_years[0] so per-step dt and elapsed time are correct in both
        # outer mode (time_grid_years[0] = 0) and inner-MC kept-base mode (> 0). `elapsed`
        # is time since sim start, used wherever the legacy code used absolute time_grid_years.
        # Exact OU discretisation: Y_{k+1} = exp(-α Δt_k) Y_k + sqrt((1-exp(-2α Δt_k))/(2α)) Z_{k+1}
        dt_steps   = np.diff(np.append([time_grid_years[0]], time_grid_years))  # [T]
        elapsed    = dt_steps.cumsum()                                        # [T] — time since sim start
        ou_decay   = np.exp(-alpha * dt_steps)                               # [T]
        ou_std     = np.sqrt((1.0 - np.exp(-2.0 * alpha * dt_steps)) / (2.0 * alpha))  # [T]
        self.ou_decay    = shared.one.new_tensor(ou_decay.reshape(-1, 1))    # [T, 1]
        self.ou_noise    = shared.one.new_tensor(ou_std.reshape(-1, 1))      # [T, 1]
        self.vols_tensor = shared.one.new_tensor(self.vols.reshape(-1, 1))   # [n_tenors, 1]

        # Ito drift: -½ σ_τ² Var(Y(t_k)) = -½ σ_τ² (1-exp(-2α t_k))/(2α)  (full value at each t_k)
        ou_var_cumul = (1.0 - np.exp(-2.0 * alpha * elapsed)) / (2.0 * alpha)  # [T]
        self.drift = shared.one.new_tensor(np.expand_dims(
            -0.5 * (self.vols * self.vols).reshape(1, -1) * ou_var_cumul.reshape(-1, 1), axis=2))

        # normalize the eigenvectors
        self.evecs = shared.one.new_tensor(B.T[:, np.newaxis, :, np.newaxis])

        # Forward curve at each time-grid point. `tensor` is the current zero curve:
        # (n_tenors,) in outer mode, (n_tenors, B) in inner mode (per-outer-path init).
        factor_tenor_t = shared.one.new_tensor(factor_tenor.reshape(1, -1))                    # [1, n_tenors]
        if self.param['Rate_Drift_Model'] == 'Drift_To_Blend':
            hist_mean = scipy.interpolate.interp1d(*np.hstack(
                ([[0.0], [self.param['Historical_Yield'].array.T[-1][0]]],
                 self.param['Historical_Yield'].array.T)), kind='linear', bounds_error=False,
                    fill_value=self.param['Historical_Yield'].array.T[-1][-1])
            omega = shared.one.new_tensor(hist_mean(self.factor.tenors).reshape(1, -1))        # [1, n_tenors]
            decay = shared.one.new_tensor(np.exp(-alpha * elapsed).reshape(-1, 1))             # [T, 1]
            # R_τ(t) = exp(-α t) r_τ(0) + (1 - exp(-α t)) Θ_τ  (t is time since sim start)
            if tensor.ndim == 1:
                curve_t0 = tensor.reshape(1, -1)                                                # [1, n_tenors]
                fwd_curve = decay * curve_t0 + (1.0 - decay) * omega                            # [T, n_tenors]
            else:
                curve_t0 = tensor.unsqueeze(0)                                                  # [1, n_tenors, B]
                decay_b = decay.unsqueeze(-1)                                                   # [T, 1, 1]
                omega_b = omega.unsqueeze(-1)                                                   # [1, n_tenors, 1]
                fwd_curve = decay_b * curve_t0 + (1.0 - decay_b) * omega_b                      # [T, n_tenors, B]
        else:
            if tensor.ndim == 1:
                fwd_curve = utils.calc_curve_forwards(
                    self.factor, tensor, elapsed, shared) / factor_tenor_t                     # [T, n_tenors]
            else:
                # Per-outer-path forwards: loop over B and stack. Inner pass runs under
                # no_grad so no autograd graph cost; vectorise later if profiling shows it.
                fwd_curve = torch.stack([
                    utils.calc_curve_forwards(
                        self.factor, tensor[:, b], elapsed, shared) / factor_tenor_t
                    for b in range(tensor.shape[1])
                ], dim=-1)                                                                       # [T, n_tenors, B]

        self.fwd_component = fwd_curve.unsqueeze(-1)  # [..., 1] — trailing dim is the inner B2 / outer scenario axis

    def calc_factors(self, factors):
        # factors: [n_factors, T, B] outer / [n_factors, T, B, B2] inner. Branch on ndim;
        # closed-form OU vectorisation is identical, only the broadcasting shapes differ.
        #
        # Closed-form vectorisation of Y_{k+1} = d_k * Y_k + n_k * Z_k  (Y_0 = 0):
        #   Y_out[k] = D[k] * cumsum( n[i]/D[i] * Z[i] )[k],  D[k] = cumprod(d)[k]
        # Two O(T) CUDA-native ops (cumprod, cumsum) replace the T-step Python loop.
        # Numerically stable for typical PCA α (0.01–0.3); avoid α >> 1 on long grids.
        evecs_mat = self.evecs[:, 0, :, 0]                              # [n_factors, n_tenors]
        D = torch.cumprod(self.ou_decay, dim=0)                         # [T, 1]

        if factors.ndim == 3:
            Y = D.unsqueeze(0) * torch.cumsum(
                (self.ou_noise / D).unsqueeze(0) * factors, dim=1)      # [n_factors, T, B]
            projected = torch.einsum('jt,jks->kts', evecs_mat, Y)       # [T, n_tenors, B]
            return self.drift + self.vols_tensor.unsqueeze(0) * projected
        else:
            D4 = D.view(1, -1, 1, 1)                                    # [1, T, 1, 1]
            noise4 = (self.ou_noise / D).view(1, -1, 1, 1)              # [1, T, 1, 1]
            Y = D4 * torch.cumsum(noise4 * factors, dim=1)              # [n_factors, T, B, B2]
            projected = torch.einsum('jt,jkbs->ktbs', evecs_mat, Y)     # [T, n_tenors, B, B2]
            return self.drift.unsqueeze(-1) + self.vols_tensor.view(1, -1, 1, 1) * projected

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
        force_positive = 0.0 #if min_rate > 0.0 else -5.0 * min_rate
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
            {
                'Reversion_Speed': meanReversionSpeed,
                'Historical_Yield': utils.Curve([], list(zip(tenor, reversionLevel))),
                'Yield_Volatility': utils.Curve([], list(zip(tenor, volCurve))),
                'Eigenvectors': [
                    {'Eigenvector': utils.Curve([], list(zip(tenor, evec))), 'Eigenvalue': eval}
                    for evec, eval in zip(evecs.real.T, evals.real)],
                'Rate_Drift_Model': self.param['Rate_Drift_Model'],
                'Princ_Comp_Source': self.param['Matrix_Type'],
                'Distribution_Type': self.param['Distribution_Type']
            },
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

    def calibrate(self, data_frame, vol_shift=0.0, num_business_days=252.0):
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
        # Initial log-spot. AAD: keep on graph — `tensor` is the factor's `Spot` (always
        # scalar for a CommodityPrice/EquityPrice); reshape to 0-d preserves grad.
        self.log_spot0 = torch.log(tensor.reshape(()))
        # Anchor theta at the current spot rather than the calibrated long-run mean. Production
        # hedging shouldn't bet that today's price will revert to a historical average — the agent
        # should hedge variance around the current regime, not directional drift toward a stale θ.
        # Disable via `Anchor_Theta_At_Spot: false` for backtests that want absolute calibrated θ.
        if bool(self.param.get('Anchor_Theta_At_Spot', True)):
            self.theta = self.log_spot0                                                # 0-d tensor, on graph
        else:
            self.theta = tensor.new_tensor(float(self.param['Theta']))                 # constant, no grad

    def theoretical_mean_std(self, t):
        """Theoretical mean and std of S_t = exp(X_t) under the exact OU distribution."""
        kappa = self.param['Kappa']
        sigma = self.param['Sigma']
        theta = self.param['Theta']
        e_kt  = np.exp(-kappa * t)
        # Analytical path — pull log_spot0 to numpy at use site (no AAD here).
        mu_X  = theta + e_kt * (float(self.log_spot0) - theta)
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

        # AAD: keep on graph — `tensor` is the factor's `Spot` (always scalar for a
        # CommodityPrice); reshape to 0-d preserves grad.
        self.log_spot0 = torch.log(tensor.reshape(()))

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
        # Stashed for privileged_factors() called immediately after generate() in the sim loop;
        # also published for cross-process consumers under the (factor_key, kind) convention.
        self.last_regime_path = regimes
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'regimes')] = regimes
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


class MarkovHMMSpotModel(StochasticProcess):
    """N-state hidden-Markov spot-price model. Conditional on regime z_t, the per-step
    innovation is Gaussian (or Student-t if `Nu` is set on the state) with annualised
    `(Mu, Sigma)`. Long-memory autocorrelation comes from regime persistence; fat tails
    from regime mixture plus optional t-emissions. One framework Gaussian per step;
    regime transitions sampled from the independent quasi-RNG uniform stream.

    JSON config:
        States: list of N dicts {Mu, Sigma, [Nu]} per regime (annualised).
        Transition_Matrix: NxN row-stochastic at Calibration_DT_Years.
        Initial_State_Probs: length-N vector summing to 1.
        Calibration_DT_Years: step size of P (default 1/252).
        Log_Price: bool (default False) — emit log returns instead of raw price diffs."""

    documentation = (
        'Asset Pricing',
        ['An N-state hidden-Markov spot-price model with additive Gaussian emissions on the '
         'daily diff $\\Delta S_t = S_t - S_{t-1}$. Latent regime $z_t$ follows a Markov chain '
         'with transition matrix $P$. Conditional on $z_t$:',
         '',
         '$$ \\Delta S_t \\sim \\mathcal{N}(\\mu_{z_t}\\delta,\\, \\sigma_{z_t}^2\\delta) $$',
         '',
         'No mean reversion at the spot level; long-memory autocorrelation arises from regime '
         'persistence and fat tails from regime occupancy.',
         '',
         'Parameters:',
         '- **States**: List of $\\{\\mu, \\sigma\\}$ per regime (annualised).',
         '- **Transition_Matrix**: NxN row-stochastic at the calibration step.',
         '- **Initial_State_Probs**: Initial regime distribution.',
         '- **Calibration_DT_Years**: Step size of $P$ (default 1/252).'])

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        # Anchor at time_grid_years[0] so per-step dt is correct under both outer mode
        # (scen_time_grid[0] = 0) and inner-MC kept-base mode (scen_time_grid[0] > 0).
        tg_years = time_grid.time_grid_years
        dt_arr = np.diff(np.hstack(([tg_years[0]], tg_years)))
        states = self.param['States']
        self.n_states = len(states)
        T = len(dt_arr)

        def _t(arr):
            return shared.one.new_tensor(arr)

        # Annualised (μ, σ) per state; per-step values applied in generate via dt scaling.
        self.mu_per_state = _t(np.array([float(s.get('Mu', 0.0)) for s in states], dtype=np.float64))
        self.sigma_per_state = _t(np.array([float(s['Sigma']) for s in states], dtype=np.float64))
        self.dt_per_step = _t(dt_arr)
        # Optional Student-t degrees of freedom per state. If any state has Nu the model
        # emits t-distributed innovations (rescaled to unit marginal variance so σ retains
        # its standard interpretation). Absent or all-None → Gaussian as before.
        nu_arr = [s.get('Nu') for s in states]
        if any(n is not None for n in nu_arr):
            self.nu_per_state = _t(np.array([float(n) if n is not None else 1.0e6 for n in nu_arr],
                                            dtype=np.float64))
        else:
            self.nu_per_state = None

        # Log-price mode: emissions are log returns; final price is exp(log_spot0 + cumsum).
        # When False, emissions are raw price diffs and price is spot0 + cumsum(dS).
        self.log_price = bool(self.param.get('Log_Price', False))

        # CTMC re-discretisation (same pattern as MarkovSwitchingLogOUSpotModel).
        P_calib = np.array(self.param['Transition_Matrix'], dtype=np.float64)
        dt_calib = float(self.param['Calibration_DT_Years'])
        Q = np.real(matrix_logm(P_calib)) / dt_calib
        P_per_step = np.zeros((T, self.n_states, self.n_states))
        for t, dt in enumerate(dt_arr):
            P_per_step[t] = np.real(matrix_expm(Q * dt)) if dt > 1.0e-12 else np.eye(self.n_states)
        self.P_cum = _t(np.cumsum(P_per_step, axis=2))
        # Raw P kept for the forward belief filter (the cumulative form is sampling-only).
        self.P_per_step = _t(P_per_step)
        self.pi0_cum = _t(np.cumsum(self.param['Initial_State_Probs']))
        self.pi0_probs = _t(np.array(self.param['Initial_State_Probs'], dtype=np.float64))

        # Initial spot. AAD: keep spot0 on the autograd graph so payoff sensitivities
        # w.r.t. the initial spot flow through. Stored as-is so inner-MC mode can pass
        # a `(B,)` vector of per-outer-path initial spots; outer mode is the framework's
        # usual `(1,)` scalar (broadcast at generate-time).
        self.spot0 = tensor

    @property
    def correlation_name(self):
        return 'MarkovHMMSpotProcess', [()]

    def generate(self, shared_mem):
        # Z is (T, B) in outer mode, (T, B, B2) in inner mode.
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        device = Z.device
        dtype = torch.float32

        pi0_cum = self.pi0_cum.to(device=device, dtype=dtype)
        P_cum = self.P_cum.to(device=device, dtype=dtype)
        mu = self.mu_per_state.to(device=device, dtype=dtype)
        sigma = self.sigma_per_state.to(device=device, dtype=dtype)
        dt = self.dt_per_step.to(device=device, dtype=dtype)
        nu_per_state = getattr(self, 'nu_per_state', None)
        nu = nu_per_state.to(device=device, dtype=dtype) if nu_per_state is not None else None

        if Z.ndim == 2:
            # Outer mode: (T, B). Bit-exact preserves of legacy behavior.
            T, B = Z.shape
            u_regime = shared_mem.quasi_rng(shared_mem.simulation_batch, T + 1)[1].contiguous()
            # Per-path regime0 override (diff-ML t=0 randomization): if the buffer
            # carries `(self.factor_key, 'regime0_outer')`, use it as the t=0
            # regime sample instead of the calibrated π_0 draw. Mirrors the
            # existing inner-mode `regime0_inner` pattern.
            regime0_override = shared_mem.t_Scenario_Buffer.get(
                (self.factor_key, 'regime0_outer'))
            if regime0_override is not None:
                state = regime0_override.to(device=device, dtype=torch.long)
            else:
                state = torch.searchsorted(pi0_cum, u_regime[0]).clamp_max_(self.n_states - 1)
            regimes = torch.empty((T, B), dtype=torch.long, device=device)
            regimes[0] = state
            for t in range(1, T):
                cdf_rows = P_cum[t - 1].index_select(0, state)
                state = (cdf_rows < u_regime[t].unsqueeze(1)).sum(dim=1).clamp_max_(self.n_states - 1)
                regimes[t] = state

            Z_dt = Z.to(dtype=dtype)
            if nu is not None:
                nu_t = nu[regimes]                                                   # (T, B)
                # Floor the chi-square draw — an underflow to 0 makes sqrt(nu/W) blow
                # up to inf (or 0*inf=NaN), corrupting ds before the log-path clamp.
                W = torch.distributions.Gamma(nu_t / 2.0, 0.5).sample().to(dtype=dtype).clamp_min(1.0e-6)
                t_innov = Z_dt * torch.sqrt(nu_t / W)
                scale_to_unit_var = torch.sqrt((nu_t - 2.0).clamp_min(1.0e-3) / nu_t)
                innov = t_innov * scale_to_unit_var
            else:
                innov = Z_dt

            mu_t = mu[regimes] * dt.view(T, 1)                                       # (T, B)
            std_t = sigma[regimes] * dt.view(T, 1).sqrt()
            ds = mu_t + std_t * innov

            s0 = self.spot0.expand(B)                                                # (1,) -> (B,)
            if self.log_price:
                log_path = s0.log().unsqueeze(0) + ds.cumsum(dim=0)                  # (T, B)
                # Floor the log-path before exp(): a fat-tailed Student-t innovation can
                # drive it below the float underflow threshold, where exp() returns 0.0
                # and breaks the strictly-positive price-level invariant downstream.
                spot_path = log_path.clamp_min(-10.0).exp()
            else:
                spot_path = s0.unsqueeze(0) + ds.cumsum(dim=0)                       # (T, B)
            # Forward HMM belief filter — outer-mode only. The differential-ML build
            # uses `P(regime_t | prices_{0..t})` as the regime coordinate of `market_t`
            # (the privileged true regime is unavailable to a decision rule at runtime).
            # Detach from autograd: belief is consumed as a state coordinate, not a
            # quantity we differentiate through; the simulator's price-path autograd
            # graph is preserved separately for the deal pricer.
            # Phase 3c: published to BOTH `privileged_factors()` (B-axis dim=1 → ok concat)
            # AND `t_Scenario_Buffer` with B-LAST shape (T, n_states, B) so the buffer's
            # dim=-1 concat works, enabling `_extract_outer_state_at(privileged=True)` to
            # route belief into the V̂ deep-state market block.
            with torch.no_grad():
                belief_path = self._forward_belief(spot_path.detach(), device, dtype)
            self.last_regime_belief = belief_path
            shared_mem.t_Scenario_Buffer[(self.factor_key, 'regime_belief')] = \
                belief_path.permute(0, 2, 1).contiguous()                            # (T, n_states, B)
        else:
            # Inner mode: (T, B, B2). One regime path per outer × inner.
            T, B, B2 = Z.shape
            # Sobol dim = T+1 (inner timesteps, ≪ 21201 cap); samples = B*B2 paths (unbounded).
            # Each path is one (T+1)-dim Sobol point; transpose so timesteps lead.
            u_flat = shared_mem.quasi_rng(T + 1, B * B2)[1].transpose(0, 1).contiguous()  # (T+1, B*B2)
            u_regime = u_flat.reshape(T + 1, B, B2)

            regime0_override = shared_mem.t_Scenario_Buffer.get(
                (self.factor_key, 'regime0_inner'), None)
            regimes = torch.empty((T, B, B2), dtype=torch.long, device=device)
            if regime0_override is not None:
                # Per-outer-path initial regime: shape (B,), expanded across the B2 inner fan-out.
                state = regime0_override.to(device=device, dtype=torch.long)\
                    .view(B, 1).expand(B, B2).contiguous()
            else:
                state = torch.searchsorted(pi0_cum, u_regime[0]).clamp_max_(self.n_states - 1)
            regimes[0] = state
            n_states = self.n_states
            for t in range(1, T):
                cdf_rows = P_cum[t - 1].index_select(0, state.flatten())\
                    .reshape(B, B2, n_states)
                state = (cdf_rows < u_regime[t].unsqueeze(-1)).sum(dim=-1)\
                    .clamp_max_(n_states - 1)
                regimes[t] = state

            Z_dt = Z.to(dtype=dtype)
            if nu is not None:
                nu_t = nu[regimes]                                                   # (T, B, B2)
                # Floor the chi-square draw — an underflow to 0 makes sqrt(nu/W) blow
                # up to inf (or 0*inf=NaN), corrupting ds before the log-path clamp.
                W = torch.distributions.Gamma(nu_t / 2.0, 0.5).sample().to(dtype=dtype).clamp_min(1.0e-6)
                t_innov = Z_dt * torch.sqrt(nu_t / W)
                scale_to_unit_var = torch.sqrt((nu_t - 2.0).clamp_min(1.0e-3) / nu_t)
                innov = t_innov * scale_to_unit_var
            else:
                innov = Z_dt

            mu_t = mu[regimes] * dt.view(T, 1, 1)                                    # (T, B, B2)
            std_t = sigma[regimes] * dt.view(T, 1, 1).sqrt()
            ds = mu_t + std_t * innov

            s0 = self.spot0                                                          # (B,)
            if self.log_price:
                log_path = s0.view(B, 1).log() + ds.cumsum(dim=0)                    # (T, B, B2)
                spot_path = log_path.clamp_min(-10.0).exp()                          # floor: see outer branch
            else:
                spot_path = s0.view(B, 1) + ds.cumsum(dim=0)                         # (T, B, B2)

        # Stashed for privileged_factors() called immediately after generate() in the sim loop;
        # also published for cross-process consumers under the (factor_key, kind) convention.
        self.last_regime_path = regimes
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'regimes')] = regimes
        return spot_path

    @classmethod
    def privileged_layout(cls, param):
        n = len(param.get('States') or [])
        return {'regime_onehot': n, 'regime_belief': n}

    def privileged_factors(self, simulated):
        regimes = self.last_regime_path
        belief = getattr(self, 'last_regime_belief', None)
        out = {
            'regime_onehot': torch.nn.functional.one_hot(
                regimes, num_classes=self.n_states).to(dtype=torch.float32),
        }
        if belief is not None:
            # Shape (T, B, n_states); accumulator concatenates along batch dim (last but one).
            out['regime_belief'] = belief.to(dtype=torch.float32)
        return out

    def _forward_belief(self, spot_path, device, dtype):
        """Forward HMM belief filter — outer-mode only. Returns belief (T, B, n_states)
        where `belief[t, b, r] = P(regime_t = r | observed diffs through time t, path b)`,
        computed in log-space (logsumexp predict + logsumexp normalize) for numerical
        robustness under fat-tailed Student-t emissions. Per-step emission parameters and
        transition matrix match the simulator's exactly (same `mu_per_state`,
        `sigma_per_state`, `nu_per_state`, `dt_per_step`, `P_per_step`) — so on held-out
        sim data the filter is calibrated against the model that generated it.
        """
        T, B = spot_path.shape
        n_states = self.n_states

        pi0_probs = self.pi0_probs.to(device=device, dtype=dtype)
        P_step = self.P_per_step.to(device=device, dtype=dtype)            # (T, n, n)
        dt = self.dt_per_step.to(device=device, dtype=dtype)               # (T,)
        mu = self.mu_per_state.to(device=device, dtype=dtype)              # (n,)
        sigma = self.sigma_per_state.to(device=device, dtype=dtype)        # (n,)
        nu_per_state = getattr(self, 'nu_per_state', None)
        nu = nu_per_state.to(device=device, dtype=dtype) if nu_per_state is not None else None

        # Observed per-step diffs: log returns if Log_Price, raw price diffs otherwise.
        # diffs[t-1] is the observation arriving at time t.
        if self.log_price:
            log_path = spot_path.clamp_min(1.0e-30).log()
            diffs = log_path[1:] - log_path[:-1]                           # (T-1, B)
        else:
            diffs = spot_path[1:] - spot_path[:-1]                         # (T-1, B)

        log_belief = torch.empty((T, B, n_states), dtype=dtype, device=device)
        log_belief[0] = pi0_probs.clamp_min(1.0e-30).log().expand(B, n_states)

        log_2pi = float(np.log(2.0 * np.pi))
        log_pi = float(np.log(np.pi))
        for t in range(1, T):
            # Predict: log_b_pred[r'] = logsumexp_r (log_b[t-1, r] + log_P[t-1, r, r'])
            log_P = P_step[t - 1].clamp_min(1.0e-30).log()                 # (n, n)
            log_b_pred = torch.logsumexp(
                log_belief[t - 1].unsqueeze(-1) + log_P.unsqueeze(0), dim=-2)  # (B, n)

            dt_t = dt[t]
            if float(dt_t) < 1.0e-12:
                # Degenerate step (e.g. forked grid where t=0's neighbour is zero); skip
                # the update — predict-only is a valid filter step under no observation.
                log_belief[t] = log_b_pred - torch.logsumexp(log_b_pred, dim=-1, keepdim=True)
                continue

            mean_r = mu * dt_t                                              # (n,)
            std_r = sigma * dt_t.sqrt()                                     # (n,)
            d = diffs[t - 1].unsqueeze(-1)                                  # (B, 1)
            z = (d - mean_r.unsqueeze(0)) / std_r.unsqueeze(0)              # (B, n)
            if nu is not None:
                # Scaled-t log-pdf (model rescales innov to unit variance ⇒ σ is marginal std).
                #   log f(d) = lgamma((ν+1)/2) − lgamma(ν/2)
                #            − 0.5·log((ν−2)π) − log σ_r
                #            − 0.5·(ν+1)·log(1 + z²/(ν−2))
                nu_eff = (nu - 2.0).clamp_min(1.0e-3)
                log_L = (
                    torch.lgamma((nu + 1.0) / 2.0)
                    - torch.lgamma(nu / 2.0)
                    - 0.5 * (nu_eff.log() + log_pi)
                    - std_r.log()
                ).unsqueeze(0) - 0.5 * (nu + 1.0).unsqueeze(0) * torch.log1p(z.pow(2) / nu_eff.unsqueeze(0))
            else:
                # Gaussian log-pdf: -0.5·log(2π σ²·dt) − 0.5·z²
                log_L = (-0.5 * (log_2pi + 2.0 * std_r.log())).unsqueeze(0) - 0.5 * z.pow(2)

            log_b_unnorm = log_b_pred + log_L
            log_belief[t] = log_b_unnorm - torch.logsumexp(log_b_unnorm, dim=-1, keepdim=True)

        return log_belief.exp()


class MarkovHMMSpotCalibration(object):
    """Calibration of MarkovHMMSpotModel via in-house Baum-Welch on price diffs (or log
    returns when `Log_Price=True`). Per-state emission is Gaussian; M-step uses weighted
    mean and weighted variance. Optional Student-t refit picks a shared ν via
    method-of-moments on the unconditional mixture kurtosis — per-state μ, σ are kept
    and ν enters the simulator's t-rescaling so marginal variance per regime stays at σ².
    States are reordered ascending by σ post-fit. `delta` is the regime-standardised
    innovation series — approximately iid under the calibrated model.

    JSON config (calibration_config.json):
        N_States, N_Iter, Seed, Tol: EM knobs (defaults 3, 200, 42, 1e-6).
        Log_Price: fit on log returns rather than raw diffs (default True).
        Use_Student_T: enable t-refit (default True; False → pure Gaussian).
        Nu_Min, Nu_Max: clamps on ν (defaults 3.0, 50.0; above max → drop t)."""

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    @staticmethod
    def _emission_logprob(diffs, means, sigmas):
        """Per-state log Normal(diffs | μ_s, σ_s²); returns (T, n_states)."""
        var = np.maximum(sigmas ** 2, 1.0e-12)
        return (-0.5 * np.log(2.0 * np.pi * var)
                - 0.5 * (diffs[:, None] - means) ** 2 / var)

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        from scipy import stats as scipy_stats

        n_states = int(self.param.get('N_States', 3))
        n_iter = int(self.param.get('N_Iter', 200))
        seed = int(self.param.get('Seed', 42))
        tol = float(self.param.get('Tol', 1.0e-6))
        use_t = bool(self.param.get('Use_Student_T', True))
        nu_min = float(self.param.get('Nu_Min', 3.0))
        nu_max = float(self.param.get('Nu_Max', 50.0))
        # Log_Price: fit on log returns. Scale-invariant, simulator exp()s to keep prices positive.
        log_price = bool(self.param.get('Log_Price', True))
        dt_calib = 1.0 / float(num_business_days)

        prices = data_frame.iloc[:, 0].astype(np.float64).dropna()
        if log_price:
            diffs = np.log(prices).diff().dropna()
        else:
            diffs = prices.diff().dropna()
        x = diffs.values

        # Init: spread σ across regimes for distinguishable EM start.
        rng = np.random.default_rng(seed)
        base_sigma = x.std()
        means = np.full(n_states, x.mean()) + rng.normal(0, base_sigma * 0.05, n_states)
        sigmas = base_sigma * np.linspace(0.5, 2.0, n_states)
        pi = np.full(n_states, 1.0 / n_states)
        P = np.full((n_states, n_states), 0.05) + 0.85 * np.eye(n_states)
        P /= P.sum(axis=1, keepdims=True)

        prev_lik = -np.inf
        for it in range(n_iter):
            log_emit = self._emission_logprob(x, means, sigmas)
            gamma, xi, log_lik = hmm_forward_backward(
                np.log(pi + 1e-12), np.log(P + 1e-12), log_emit)
            if it > 0 and abs(log_lik - prev_lik) < tol * abs(prev_lik):
                break
            prev_lik = log_lik
            pi = gamma[0]
            denom = gamma[:-1].sum(axis=0)
            P = xi.sum(axis=0) / np.maximum(denom[:, None], 1e-12)
            P /= P.sum(axis=1, keepdims=True)
            w_sum = gamma.sum(axis=0)
            means = (gamma * x[:, None]).sum(axis=0) / np.maximum(w_sum, 1e-12)
            sigmas = np.sqrt(np.maximum(
                (gamma * (x[:, None] - means) ** 2).sum(axis=0) / np.maximum(w_sum, 1e-12),
                1e-12))

        # Reorder states ascending by σ; remap P, π, posterior assignments accordingly.
        order = np.argsort(sigmas)
        means = means[order]
        sigmas = sigmas[order]
        P = P[np.ix_(order, order)]
        regimes = np.argmax(gamma, axis=1)
        remap = {old: new for new, old in enumerate(order)}
        regimes = np.array([remap[s] for s in regimes])
        occ = np.bincount(regimes, minlength=n_states) / len(regimes)
        nus = [None] * n_states

        # Method-of-moments shared ν, derived from the unconditional kurt of the
        # regime mixture: K = (3 + 6/(ν-4)) · Σπ_s σ_s⁴ / Var² - 3. Inverting for ν:
        #     ν = 4 + 6 / [(K_emp + 3)·Var² / Σπ_s σ_s⁴ - 3]
        # Use the *model's* stationary variance (not the sample variance) so the
        # simulator round-trips on kurt — EM convergence may underfit empirical Var.
        if use_t:
            emp_kurt = float(scipy_stats.kurtosis(x, fisher=True))
            mu_total = float((occ * means).sum())
            mix_var = float((occ * sigmas**2).sum() + (occ * (means - mu_total)**2).sum())
            sum_pi_sigma4 = float(np.sum(occ * sigmas**4))
            denom = (emp_kurt + 3.0) * mix_var * mix_var / sum_pi_sigma4 - 3.0
            if denom > 1e-3:
                nu_global = 4.0 + 6.0 / denom
                nu_global = float(np.clip(nu_global, nu_min, nu_max))
                if nu_global < nu_max - 1e-6:
                    nus = [nu_global] * n_states

        # Annualised storage convention so model.precalculate's per-step
        # `μ·δ, σ²·δ` formula gives the calibration-step daily moments at δ=dt_calib.
        mu_year = means / dt_calib
        sigma_year = sigmas / np.sqrt(dt_calib)

        param = {
            'Log_Price': log_price,
            'States': [
                {'Mu': float(m), 'Sigma': float(s), **({'Nu': float(n)} if n is not None else {})}
                for m, s, n in zip(mu_year, sigma_year, nus)
            ],
            'Transition_Matrix': P.tolist(),
            'Initial_State_Probs': occ.tolist(),
            'Calibration_DT_Years': dt_calib,
        }

        # delta = regime-standardised innovation: (diff - μ_state) / σ_state under the
        # posterior regime path. Approximately iid N(0,1) so the framework's correlation
        # consolidation isn't contaminated by regime-induced heteroskedasticity.
        innov = (diffs.values - means[regimes]) / np.where(sigmas[regimes] > 0, sigmas[regimes], 1.0)
        delta = pd.DataFrame({data_frame.columns[0]: innov}, index=diffs.index)

        return utils.CalibrationInfo(param, [[1.0]], delta)


class VARMixedFactorInterestRateModel(StochasticProcess):
    """3-factor VAR(1) curve model on a forward-curve-shaped factor (absolute-date
    tenors, in the same shape as ForwardPrice consumed by CSForwardPriceModel). The
    latent state X(t) = (β_0, β_1, r) — level, slope, curvature — evolves jointly:

        X(t+δ) = μ + Φ_step (X(t) − μ) + diag(σ_step) Z(t)

    Internally the model maintains a 3-slot ladder (front/mid/back contract τs that
    age with sim time and roll forward by `Contract_Cycle_Years` together), evaluates
    the curve at the slot τs via the cross-product orthogonal basis:

        c_i_slot(t) = β_0 + β_1·τ_i_slot(t) + r·w_i_slot(t)

    then *publishes* by linearly interpolating the 3 slot values onto the factor's
    contract tenors (T_j(t) = factor_tenor[j] − sim_t in years). The factor's tenor
    is in absolute Excel-date offsets — one per dated contract — so consumer-side
    `gather_weighted_curve(date_index, multiply_by_time=False)` lands exactly on a
    knot. Slot machinery is purely an internal latent-state representation; the
    published shape is a forward curve.

    Φ_step and σ_step are matrix-power-scaled from the calibration step when
    sim δ ≠ δ_calib. Innovation correlations live in the global Correlations block;
    the framework's global Cholesky delivers Z pre-correlated.

    JSON config:
        Mean: 3-vector [μ_β0, μ_β1, μ_r] (long-run mean per factor)
        Phi: 3x3 list-of-lists (calibration-step transition)
        Sigma: 3-vector — *marginal* innovation std per latent component (post-correlation,
            NOT a Cholesky factor). Innovation cross-correlation lives in the global
            Correlations block; the framework's Cholesky pre-correlates Z, the model
            scales each component by σ. At δ = δ_calib the resulting innovation
            covariance is diag(σ)·ρ·diag(σ) = Σ_calib.
        Calibration_Tenors: 3-vector (τ_i(0) — slot tenors at simulation start, years)
        Contract_Cycle_Years: float (front-slot roll cycle, e.g. 0.25 quarterly)
        Calibration_DT_Years: float (default 1/252)

    Caveat: σ-step scales as σ_calib·√(δ/δ_calib), a Brownian approximation valid for
    δ ≈ δ_calib. precalculate raises if any step ratio drifts >50% from 1 — implement
    Lyapunov-equation step covariance before running on weekly/monthly grids."""

    documentation = (
        'Interest Rates',
        ['A 3-factor VAR(1) model with internal slot ladder, published as a '
         'forward-curve-shaped factor (absolute-date tenors). The latent state '
         '$X = (\\beta_0, \\beta_1, r)$ — level, slope, curvature — evolves jointly:',
         '',
         '$$ X(t+\\delta) = \\boldsymbol{\\mu} + \\Phi (X(t) - \\boldsymbol{\\mu}) '
         '+ \\text{diag}(\\sigma) Z, \\quad Z \\sim \\mathcal{N}(0, \\rho_\\text{innov}) $$',
         '',
         'Internally the curve at the slot τs is reconstructed each step as $c_i^{slot}(t) '
         '= \\beta_0 + \\beta_1 \\tau_i^{slot}(t) + r \\cdot w_i^{slot}(t)$. The factor '
         'output at the dated contract tenors $T_j(t)$ is obtained by linearly '
         'interpolating the three slot values; this is algebraically equivalent to '
         'evaluating the parametric form with linearly-interpolated $w$, but only '
         'evaluates $w$ where it is mathematically defined (at the slot τs). Innovation '
         'correlations $\\rho_\\text{innov}$ live in the global Correlations block.'])

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)

    def num_factors(self):
        return 3

    @property
    def correlation_name(self):
        return 'VARMixedFactorInterestRateProcess', [('B0',), ('B1',), ('R',)]

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        dt_calib = float(self.param['Calibration_DT_Years'])
        sim_t = time_grid.time_grid_years
        # Anchor at sim_t[0] so per-step dt is correct under both outer mode (sim_t[0] = 0)
        # and inner-MC kept-base mode (sim_t[0] > 0). Slot-ladder roll count below uses
        # absolute sim_t intentionally — rolls happen at physical calendar dates.
        dt_arr = np.diff(np.hstack(([sim_t[0]], sim_t)))                                # (T,)

        # σ_step uses a Brownian approximation σ_calib·√(δ/δ_calib) which assumes i.i.d.
        # innovations. The exact VAR(1) per-step covariance is V_stat − Φ^(δ/δ_calib)
        # V_stat (Φ^(δ/δ_calib))ᵀ (Lyapunov), reducing to Σ_calib at δ = δ_calib. The
        # Brownian form is first-order in (1−Φ); error is small for daily sim against a
        # daily-calibrated model (including the 0.69 calendar/business-day ratio) and
        # grows on slow eigenmodes at long sim steps. Reject weekly+ steps to prevent
        # silent mis-pricing of innovation variance. dt = 0 (the initial sim point, not
        # a stochastic step) is excluded from the check.
        nonzero_dt = dt_arr[dt_arr > 1.0e-12]
        if nonzero_dt.size and np.any((nonzero_dt / dt_calib > 2.0)
                                       | (nonzero_dt / dt_calib < 0.5)):
            raise ValueError(
                "VARMixedFactorInterestRateModel σ-step uses a Brownian approximation valid for "
                f"δ ≈ δ_calib; sim grid has non-zero step ratios in "
                f"[{(nonzero_dt/dt_calib).min():.2f}, {(nonzero_dt/dt_calib).max():.2f}], "
                "outside the supported [0.5, 2.0] band. Implement Lyapunov-equation step "
                "covariance for weekly/monthly time grids before running this.")

        # Per-step Φ via matrix-power: Φ_step = Φ_calib^(dt/dt_calib). For dt = dt_calib
        # this is just Φ_calib. Eigendecomposition handles arbitrary dt.
        Phi_calib = np.array(self.param['Phi'], dtype=np.float64)
        eigvals, eigvecs = np.linalg.eig(Phi_calib)
        eigvecs_inv = np.linalg.inv(eigvecs)
        Phi_per_step = np.zeros((len(dt_arr), 3, 3))
        # `Sigma` is the calibrated *marginal* std per latent component (post-correlation),
        # NOT a Cholesky factor. The framework's global Cholesky supplies pre-correlated Z
        # to `generate`; multiplying by sigma_per_step recovers Σ_calib = diag(σ)·ρ·diag(σ)
        # at δ = δ_calib (where ρ comes from the Correlations block).
        sigma_calib = np.array(self.param['Sigma'], dtype=np.float64)                  # (3,)
        sigma_per_step = np.zeros((len(dt_arr), 3))
        for t_idx, dt in enumerate(dt_arr):
            power = dt / dt_calib
            Phi_per_step[t_idx] = np.real(eigvecs @ np.diag(eigvals ** power) @ eigvecs_inv)
            sigma_per_step[t_idx] = sigma_calib * np.sqrt(power)

        self.Phi_per_step = shared.one.new_tensor(Phi_per_step)                        # (T, 3, 3)
        self.sigma_per_step = shared.one.new_tensor(sigma_per_step)                    # (T, 3)
        self.mean_vec = shared.one.new_tensor(np.array(self.param['Mean'], dtype=np.float64))  # (3,)

        # Slot ladder: starting at cal_tau_0 = τ_i(0), each slot ages with sim_t. The
        # front slot rolls when its tenor would otherwise hit zero; all three slots
        # shift forward by Δ_cycle together (futures-ladder convention).
        cal_tau_0 = np.array(self.param['Calibration_Tenors'], dtype=np.float64)       # (3,)
        delta_cycle = float(self.param['Contract_Cycle_Years'])
        n_rolls = np.where(sim_t < cal_tau_0[0], 0,
                           np.floor((sim_t - cal_tau_0[0]) / delta_cycle).astype(int) + 1)
        slot_tenors = cal_tau_0[None, :] + n_rolls[:, None] * delta_cycle - sim_t[:, None]
        if (slot_tenors <= 0).any():
            raise ValueError(f'Slot tenor schedule went non-positive: min={slot_tenors.min():.4f}')

        # Per-step W via cross-product on the slot τs (where w is mathematically defined).
        t1, t2, t3 = slot_tenors[:, 0], slot_tenors[:, 1], slot_tenors[:, 2]
        w_raw = np.stack([t2 - t3, t3 - t1, t1 - t2], axis=-1)
        W_per_step = w_raw / np.linalg.norm(w_raw, axis=-1, keepdims=True)
        W_per_step = W_per_step * np.where(W_per_step[:, 1:2] < 0, -1.0, 1.0)

        D_slot_per_step = np.stack([
            np.ones_like(slot_tenors),
            slot_tenors,
            W_per_step,
        ], axis=-1)                                                                    # (T, 3, 3)
        self.D_slot_per_step = shared.one.new_tensor(D_slot_per_step)
        self.tau_slot_per_step = shared.one.new_tensor(slot_tenors)                    # (T, 3)

        # CSForward-style per-step contract tenors: factor_tenor is in absolute Excel
        # date offsets; convert to remaining years per sim step. Contracts past expiry
        # get a zero output (clamped via expired mask).
        factor_tenor = np.asarray(self.factor.get_tenor(), dtype=np.float64)
        excel_offset = (ref_date - utils.excel_offset).days
        excel_date_time_grid = time_grid.scen_time_grid + excel_offset                 # (T,) Excel days
        contract_T = (factor_tenor[None, :] - excel_date_time_grid[:, None]) / utils.DAYS_IN_YEAR
        self.contract_expired = shared.one.new_tensor(
            (contract_T <= 0.0).astype(np.float64)).bool()                             # (T, n_contracts)
        self.contract_T = shared.one.new_tensor(np.maximum(contract_T, 0.0))           # (T, n_contracts)

        # X_0 from t=0 carries: find X such that linearly-interpolating the slot values
        # c_slot = D_slot[0] @ X at slot τs onto contract Ts reproduces today's curve.
        # The linear-interp/extrap operator is captured by row-mixing of D_slot[0] —
        # for contract j in bracket [slot k, slot k+1]:
        #   M[j, :] = (1-α_j) D_slot[0, k, :] + α_j D_slot[0, k+1, :]
        # This matches the runtime interp in `generate` exactly (including extrapolation
        # outside the slot range).
        #
        # AAD: M is data-independent (built from calibration tenors + sim time grid only),
        # so build it as a constant tensor. `tensor` carries the live curve and may have
        # requires_grad — keep it as a torch tensor through the solve so ∂X_0/∂curve_0 = M⁻¹
        # flows via autograd's standard linear-solve adjoint, and downstream `out` retains
        # the gradient through to the input curve.
        if factor_tenor.size != 3:
            # TODO: relax to N≥3 by switching torch.linalg.solve → torch.linalg.lstsq below
            # and warning when the round-trip residual is non-negligible (which is
            # expected for N>3: 3 latent factors can't exactly span an N-dim curve).
            # Runtime generate() already supports arbitrary N via slot interpolation /
            # linear extrapolation. For N>3 with cross-product w, calibration on >3
            # archive anchors needs an anchor-choice convention (or switch to NS basis).
            raise ValueError(
                f'VARMixedFactorInterestRateModel expects a 3-knot factor (front/mid/back '
                f'contract ladder); got factor_tenor of size {factor_tenor.size}.')
        slot_t0 = slot_tenors[0]
        contract_T0 = np.maximum(contract_T[0], 1e-9)
        idx0 = np.clip(np.searchsorted(slot_t0, contract_T0, side='left'), 1, 2)
        alpha0 = (contract_T0 - slot_t0[idx0 - 1]) / (slot_t0[idx0] - slot_t0[idx0 - 1])
        M_np = (1.0 - alpha0)[:, None] * D_slot_per_step[0, idx0 - 1, :] \
               + alpha0[:, None] * D_slot_per_step[0, idx0, :]                         # (3, 3)
        M_t = shared.one.new_tensor(M_np)
        # tensor: (3,) outer / (3, B) inner — fed directly to solve. linalg.solve
        # treats 1D RHS as a vector and N-D RHS as a batch of column-vector solves
        # (PyTorch's natural broadcasting), so X0 inherits tensor's batch dims.
        curve0 = tensor
        # M_t goes singular when the contract ladder degenerates — fewer than 3 live
        # contracts (e.g. an inner-MC fork started late in the horizon, after the front
        # contracts have expired) collapse the bracketing onto coincident rows. In that
        # case recover X_0 by a ridge-regularised least-squares solve and skip the exact
        # round-trip check (the recovery is approximate by construction). When the ladder
        # is well-conditioned, keep the exact solve + round-trip guard (it catches genuine
        # searchsorted/clip bracketing bugs).
        if float(torch.linalg.cond(M_t)) < 1.0e8:
            X0 = torch.linalg.solve(M_t, curve0)                                       # (3,) or (3, B)
            roundtrip_err_t = (M_t @ X0 - curve0).abs().max()
            rt_tol = 1.0e-10 if M_t.dtype == torch.float64 else 1.0e-5
            if roundtrip_err_t.item() > rt_tol:
                raise ValueError(
                    f'X_0 round-trip failed: max |M·X_0 - curve_0| = {roundtrip_err_t.item():.2e}'
                    f' (tol {rt_tol:.0e} for dtype {M_t.dtype}). '
                    f'Likely boundary mishandling in the slot-bracketing (contract_T0={contract_T0}, '
                    f'slot_t0={slot_t0}).')
        else:
            gram = M_t.transpose(-1, -2) @ M_t
            ridge = 1.0e-6 * gram.diagonal().mean() * torch.eye(
                3, dtype=M_t.dtype, device=M_t.device)
            X0 = torch.linalg.solve(gram + ridge, M_t.transpose(-1, -2) @ curve0)
        self.X0 = X0                                                                   # (3,) — non-leaf, on `tensor`'s graph

    def generate(self, shared_mem):
        # Z: (3, T, B) outer / (3, T, B, B2) inner.
        Z = shared_mem.t_random_numbers[
            self.z_offset:self.z_offset + 3, :self.scenario_horizon]
        n_contracts = self.contract_T.shape[1]

        # einsum is used in both modes for the (3,3) × (3, *batch_shape) products.
        # `@` would silently misinterpret the leading 3 of (3, B, B2) as a batch dim
        # rather than the matmul contraction axis; einsum keeps the contraction explicit.
        if Z.ndim == 3:
            # Outer mode: (3, T, B).
            _, T, B = Z.shape
            mean = self.mean_vec.view(3, 1)                                            # (3, 1)
            X = self.X0.view(3, 1).expand(3, B).clone()                                # (3, B)
            out = torch.empty((T, n_contracts, B), dtype=Z.dtype, device=Z.device)
            for t in range(T):
                X = mean + torch.einsum('ij,j...->i...', self.Phi_per_step[t], X - mean) \
                    + self.sigma_per_step[t].view(3, 1) * Z[:, t, :]                   # (3, B)
                c_slot = torch.einsum('ij,j...->i...', self.D_slot_per_step[t], X)     # (3, B)
                ts = self.tau_slot_per_step[t]
                cT_t = self.contract_T[t]
                idx = torch.clamp(torch.searchsorted(ts, cT_t, right=False), 1, 2)
                alpha = ((cT_t - ts[idx - 1]) / (ts[idx] - ts[idx - 1])).unsqueeze(-1) # (n_contracts, 1)
                out_t = (1.0 - alpha) * c_slot[idx - 1] + alpha * c_slot[idx]          # (n_contracts, B)
                out[t] = torch.where(self.contract_expired[t].unsqueeze(-1),
                                     torch.zeros_like(out_t), out_t)
        else:
            # Inner mode: (3, T, B, B2). X carries the fan-out batch dims through.
            _, T, B, B2 = Z.shape
            mean = self.mean_vec.view(3, 1, 1)                                         # (3, 1, 1)
            X = self.X0.unsqueeze(-1).expand(3, B, B2).clone()                         # (3, B, B2)
            out = torch.empty((T, n_contracts, B, B2), dtype=Z.dtype, device=Z.device)
            for t in range(T):
                X = mean + torch.einsum('ij,j...->i...', self.Phi_per_step[t], X - mean) \
                    + self.sigma_per_step[t].view(3, 1, 1) * Z[:, t, :, :]             # (3, B, B2)
                c_slot = torch.einsum('ij,j...->i...', self.D_slot_per_step[t], X)     # (3, B, B2)
                ts = self.tau_slot_per_step[t]
                cT_t = self.contract_T[t]
                idx = torch.clamp(torch.searchsorted(ts, cT_t, right=False), 1, 2)
                alpha = ((cT_t - ts[idx - 1]) / (ts[idx] - ts[idx - 1]))\
                    .view(-1, 1, 1)                                                    # (n_contracts, 1, 1)
                out_t = (1.0 - alpha) * c_slot[idx - 1] + alpha * c_slot[idx]          # (n_contracts, B, B2)
                expired_view = self.contract_expired[t].view(-1, 1, 1)                 # (n_contracts, 1, 1)
                out[t] = torch.where(expired_view, torch.zeros_like(out_t), out_t)
        return out


class VARMixedFactorInterestRateCalibration(object):
    """Calibration of VARMixedFactorInterestRateModel from floating-tenor curve
    observations. Per day t: solve D(t)·X(t) = c(t) where D = [1, τ, w(τ)] is the 3×3
    design matrix built from that day's τ-vector — exact 3-equations-3-unknowns. Then
    fit VAR(1) to the daily latent series via OLS for Φ, sample mean for μ, and column-
    std of the residuals for σ.

    The simulator-side `Calibration_Tenors` is τ at the *last* calibration date — i.e.
    τ_i(0) of the simulation. `Contract_Cycle_Years` is the front-slot roll cycle,
    auto-detected from the median τ-spacing across slots.

    `delta` is ε_t (3 columns); `correlation` is identity 3×3 so the framework's
    consolidation picks up the innovation cross-correlations directly."""

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 3

    @staticmethod
    def _w_vector(tau):
        """Unit vector orthogonal to (1,1,1) and τ, with sign convention w_2 > 0."""
        w = np.array([tau[2] - tau[1], tau[0] - tau[2], tau[1] - tau[0]], dtype=np.float64)
        n = np.linalg.norm(w)
        if n < 1e-12:
            return np.zeros(3)
        w /= n
        if w[1] < 0:
            w = -w
        return w

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        primary = [c for c in data_frame.columns if not c.split(',', 1)[0].startswith('Tenor.')]
        tenor_cols = [c for c in data_frame.columns if c.split(',', 1)[0].startswith('Tenor.')]

        sub_keys = [c.split(',', 1)[1] for c in primary]
        tenor_lookup = {c.split(',', 1)[0].split('.', 1)[1]: c for c in tenor_cols}
        carry_arr = data_frame[primary].astype(np.float64).values                      # (T_obs, 3)
        tau_arr = np.column_stack([
            data_frame[tenor_lookup[sk]].astype(np.float64).values for sk in sub_keys
        ])                                                                              # (T_obs, 3)

        # Drop NaN + near-expiry rows. Tau_Floor (default 0.03 ≈ 1 BD) eliminates days
        # where 1/τ amplification destabilises the carry estimate.
        tau_floor = float(self.param.get('Tau_Floor', 0.03))
        valid = (~(np.isnan(carry_arr).any(axis=1) | np.isnan(tau_arr).any(axis=1))
                 & (tau_arr[:, 0] > tau_floor))
        carry_arr = carry_arr[valid]
        tau_arr = tau_arr[valid]
        valid_index = data_frame.index[valid]

        X_daily = np.zeros((len(carry_arr), 3))
        for t in range(len(carry_arr)):
            tau_t = tau_arr[t]
            w_t = self._w_vector(tau_t)
            D = np.column_stack([np.ones(3), tau_t, w_t])
            X_daily[t] = np.linalg.solve(D, carry_arr[t])

        mu = X_daily.mean(axis=0)
        X_centered = X_daily - mu
        X_lag = X_centered[:-1]
        X_next = X_centered[1:]
        gram = X_lag.T @ X_lag
        rhs = X_lag.T @ X_next
        Phi = np.linalg.solve(gram, rhs).T
        innov = X_next - X_centered[:-1] @ Phi.T
        sigma = innov.std(axis=0)

        # τ_i(0) for the simulator: τ at the last calibration date (sim picks up here).
        # Cycle: median inter-slot spacing across the calibration window.
        tau_ref = tau_arr[-1]
        cycle = float(np.median(np.diff(tau_arr, axis=1)))

        dt_calib = 1.0 / float(num_business_days)
        param = {
            'Mean': [float(m) for m in mu],
            'Phi': [[float(v) for v in row] for row in Phi],
            'Sigma': [float(s) for s in sigma],
            'Calibration_Tenors': [float(t) for t in tau_ref],
            'Contract_Cycle_Years': cycle,
            'Calibration_DT_Years': dt_calib,
        }

        # delta: per-factor innovation series. Framework concatenates these and computes
        # ρ for both within-factor (β0↔β1 etc.) and cross-factor (e.g. spot↔β0) correlations.
        archive_name = primary[0].split(',', 1)[0]                                     # e.g. 'InterestRate.PLATINUM_CARRY'
        delta = pd.DataFrame(
            innov, index=valid_index[1:],
            columns=[f'{archive_name},B0', f'{archive_name},B1', f'{archive_name},R'])
        # correlation: identity 3×3 — each delta column maps 1-1 to a primitive factor.
        correlation_coef = np.eye(3).tolist()

        return utils.CalibrationInfo(param, correlation_coef, delta)


class BasisLinkedSpotModel(StochasticProcess):
    """Lagged-AR(1) basis driven by a sibling commodity-spot path and its HMM regime:

        b(t) = a · ΔS(t) + φ · b(t-1) + η(t)
        η(t) = σ(s_t) · √((ν-2)/ν) · ε_t,    ε_t ~ t_ν

    ΔS is the linked spot's per-step diff, s_t is the linked spot's HMM regime, and σ(s)
    is the regime-keyed innovation std. Innovation is built from a framework-correlated
    Gaussian Z plus an internal Chi²(ν) draw; the √((ν-2)/ν) rescaling makes σ(s) the
    realised std of η regardless of ν. The linked spot's path and regime path are read
    from `shared_mem.t_Scenario_Buffer`; sim ordering is enforced by
    `dependant_fields['CommodityBasis']` and the `Observed_Commodity` Price Factor field.
    Initial b(0) is taken from the factor's `Spot` value.

    JSON config:
        A: concurrent ΔS loading
        Phi: AR(1) coefficient on b(t-1)
        Nu: Student-t degrees of freedom (shared across regimes)
        Sigma_By_State: list of σ_s indexed by linked-spot HMM state
        Mu: long-run mean of η (typically 0)
        Calibration_DT_Years: float (default 1/252)"""

    documentation = (
        'Asset Pricing',
        ['A lagged-AR(1) basis driven by a sibling commodity-spot path and its HMM regime.',
         '',
         '$$ b(t) = a \\Delta S(t) + \\phi b(t-1) + \\eta(t),'
         '\\quad \\eta(t) = \\sigma(s_t)\\sqrt{(\\nu-2)/\\nu}\\,\\varepsilon_t,'
         '\\quad \\varepsilon_t \\sim t_\\nu $$',
         '',
         'Reads the linked spot path and its HMM regime path from the simulator shared '
         'buffer; the linked spot is simulated first (enforced via `dependant_fields`).',
         '',
         '- **A**: concurrent ΔS loading',
         '- **Phi**: AR(1) coefficient',
         '- **Nu**: Student-t degrees of freedom (shared across regimes)',
         '- **Sigma_By_State**: per-regime innovation std'])

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    @property
    def correlation_name(self):
        return 'BasisLinkedSpotProcess', [()]

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        # The linked CommodityPrice factor is named in this Price Factor's Observed_Commodity
        # field — declared via dependant_fields['CommodityBasis']. At sim time the framework
        # has populated the Price Factor entry from the job file's ExplicitMarketData.
        linked_id = self.factor.param['Observed_Commodity']
        self.linked_key = utils.Factor('CommodityPrice', tuple(linked_id.split('.')))

        self.A = float(self.param['A'])
        self.Phi = float(self.param['Phi'])
        self.Nu = float(self.param['Nu'])
        self.Mu = float(self.param.get('Mu', 0.0))
        self.sigma_by_state = shared.one.new_tensor(np.array(self.param['Sigma_By_State'], dtype=np.float64))
        # AAD: keep b0 on the autograd graph so sensitivities of payoffs w.r.t. the
        # observed initial basis flow through. Stored as-is so inner-MC mode can pass
        # a `(B,)` vector of per-outer-path initial bases; outer mode is `(1,)`.
        self.b0 = tensor

    def generate(self, shared_mem):
        # Z is (T, B) outer / (T, B, B2) inner, correlated.
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        device = Z.device
        dtype = Z.dtype

        # Cross-process reads. The linked spot must have been generated first; the
        # `dependant_fields` declaration on CommodityBasis enforces the ordering by
        # making CommodityPrice a dependency, so the simulator topo-orders it before us.
        # The linked spot path is in *price level* (dollars), not log-space — the HMM
        # process exp()s its log-cumsum before publishing. Path/regime shapes match
        # this process's Z, since both processes ran in the same inner/outer mode.
        linked_path = shared_mem.t_Scenario_Buffer[self.linked_key]
        regimes = shared_mem.t_Scenario_Buffer[(self.linked_key, 'regimes')]
        assert (linked_path > 0).all(), 'linked_path expected to be all positive'

        sigma_t = self.sigma_by_state[regimes]
        # Student-t innovation: η_t = sigma_t · ε_t · √((ν-2)/ν), ε_t ~ t_ν.
        # Identity: ε_t = Z · √(ν/W) where W ~ Chi²(ν). Combine the rescaling so the
        # marginal variance of η_t is sigma_t² regardless of ν.
        nu = self.Nu
        phi = float(self.Phi)
        a = float(self.A)

        if Z.ndim == 2:
            # Outer mode: (T, B). Bit-exact preserve of legacy behavior.
            T, B = Z.shape
            # Floor the chi-square draw — same inf/NaN guard as the linked spot.
            W = torch.distributions.Chi2(nu).sample((T, B)).to(device=device, dtype=dtype).clamp_min(1.0e-6)
            eta = sigma_t * Z * torch.sqrt((nu - 2.0) / W)
            b_init = self.b0.expand(B)
            out = torch.empty((T, B), device=device, dtype=dtype)
            out[0] = b_init
            for t in range(1, T):
                out[t] = a * (linked_path[t] - linked_path[t - 1]) + phi * out[t - 1] + eta[t]
        else:
            # Inner mode: (T, B, B2).
            T, B, B2 = Z.shape
            W = torch.distributions.Chi2(nu).sample((T, B, B2)).to(device=device, dtype=dtype).clamp_min(1.0e-6)
            eta = sigma_t * Z * torch.sqrt((nu - 2.0) / W)
            out = torch.empty((T, B, B2), device=device, dtype=dtype)
            out[0] = self.b0.unsqueeze(-1).expand(B, B2)
            for t in range(1, T):
                out[t] = a * (linked_path[t] - linked_path[t - 1]) + phi * out[t - 1] + eta[t]
        return out


class BasisLinkedSpotCalibration(object):
    """Calibration of BasisLinkedSpotModel. Self-contained: data_frame carries the basis
    column (`CommodityBasis.<basis>,<linked>`) plus the linked CommodityPrice column,
    delivered via the comma-subkey archive-pull. OLS on `b(t) = a·ΔS + φ·b(t-1) + η(t)`
    recovers (a, φ); ν from method-of-moments on the η excess kurt; per-regime σ from
    rolling-vol-tercile partitioning of η — terciles indexed in σ-ascending order to
    match the linked spot's HMM regime convention.

    JSON config (calibration_config.json):
        Nu_Min, Nu_Max: clamps for the MoM ν solve (defaults 3.0, 50.0)
        Vol_Window: rolling window for regime-tercile assignment (default 21 BD)"""

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        from scipy import stats as scipy_stats

        nu_min = float(self.param.get('Nu_Min', 3.0))
        nu_max = float(self.param.get('Nu_Max', 50.0))
        vol_window = int(self.param.get('Vol_Window', 21))
        dt_calib = 1.0 / float(num_business_days)

        # Basis col is `CommodityBasis.<basis>,<linked>`; linked col is `CommodityPrice.<linked>`.
        basis_col = next(c for c in data_frame.columns if c.split('.', 1)[0] == 'CommodityBasis')
        linked_id = basis_col.split(',', 1)[1]
        linked_col = f'CommodityPrice.{linked_id}'
        if linked_col not in data_frame.columns:
            raise ValueError(
                f'BasisLinkedSpotCalibration: required linked-spot column {linked_col!r} '
                f'not in data_frame (have {list(data_frame.columns)}). The framework should '
                f'have auto-pulled it via the comma-subkey on {basis_col!r}.')

        joint = data_frame[[basis_col, linked_col]].astype(np.float64).dropna()
        b = joint[basis_col].values
        lme_v = joint[linked_col].values
        dlme = np.diff(lme_v)
        y = b[1:]
        X = np.column_stack([dlme, b[:-1]])

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_hat, phi_hat = float(coef[0]), float(coef[1])
        eta = y - X @ coef

        eta_kurt = float(scipy_stats.kurtosis(eta, fisher=True))
        nu_hat = float(np.clip(4.0 + 6.0 / max(eta_kurt, 1.0e-3), nu_min, nu_max))

        # Rolling-21d vol of ΔLME → tercile bins (low/mid/high). Index ascending in
        # vol matches the production HMM's σ-ascending state ordering, so per-regime σ
        # values are positionally consistent with the HMM regime path read at sim time.
        rolling_vol = pd.Series(dlme).rolling(vol_window, min_periods=vol_window).std()
        # Align rolling_vol to η (same length n-1)
        rolling_vol = rolling_vol.values
        valid = ~np.isnan(rolling_vol)
        if valid.sum() < 100:
            sigma_by_state = [float(eta.std())] * 3
        else:
            quantiles = np.quantile(rolling_vol[valid], [1.0 / 3, 2.0 / 3])
            tercile = np.zeros(len(eta), dtype=int)
            tercile[rolling_vol > quantiles[0]] = 1
            tercile[rolling_vol > quantiles[1]] = 2
            tercile[~valid] = 1                                                       # leading NaN → mid
            sigma_by_state = [float(eta[tercile == s].std()) if (tercile == s).sum() > 1
                              else float(eta.std()) for s in range(3)]

        param = {
            'A': a_hat,
            'Phi': phi_hat,
            'Nu': nu_hat,
            'Mu': 0.0,
            'Sigma_By_State': sigma_by_state,
            'Calibration_DT_Years': dt_calib,
        }
        delta = pd.DataFrame({basis_col: eta}, index=joint.index[1:])
        return utils.CalibrationInfo(param, [[1.0]], delta)


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
            gamma, xi, log_lik = hmm_forward_backward(np.log(pi + 1e-12), np.log(P + 1e-12), log_emit)
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


def construct_process(sp_type, factor, param, implied_factor=None):
    return globals().get(sp_type)(factor, param, implied_factor)


def construct_calibration_config(calibration_model, param):
    return globals().get(param['Method'])(calibration_model, param)
