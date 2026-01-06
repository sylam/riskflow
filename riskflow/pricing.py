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

import numpy as np
import torch
import torch.nn.functional as F

from . import utils

# useful constants
BARRIER_UP = -1.0
BARRIER_DOWN = 1.0
BARRIER_IN = -1.0
BARRIER_OUT = 1.0
OPTION_PUT = -1.0
OPTION_CALL = 1.0


def cash_settle(shared, currency, time_index, value):
    # need to check if the time_index
    if shared.t_Cashflows is not None and time_index in shared.t_Cashflows.get(currency, []):
        shared.t_Cashflows[currency][time_index] += value


def calc_moneyness(strike, spot, forward, deal_data, use_forward=False, invert_moneyness=False):
    '''
    Deals with different ways of calculating moneyness - either spot/strike, forward/strike
    or for svi/skew surfaces, return log(strike/forward) or the strike
    :param use_forward: use the forward to calculate moneyness (otherwise use the spot)
    :param invert_moneyness: if true, calculate moneyness as strike / forward (or spot)
    :param strike: strike value
    :param spot: spot price tensor
    :param forward: forward price tensor
    :param deal_data: Deal_data struct containing deal specific data
    :return: the moneyness ( or information relevant to calculate the moneyness)
    '''
    subtype = deal_data.Factor_dep['Volatility'][0][utils.FACTOR_INDEX_SubType]
    if subtype[0] in ['SVI', 'Skew']:
        # need to divide the strike by the ATM_Ref (can only be done later)
        # so we just return the strike here (we know the moneyness rule and will do the rest later)
        # otherwise, (Sticky_Moneyness) - we return log of strike over forward
        return strike if subtype[1]=='Sticky_Strike' else torch.log(strike/forward)
    else:
        # regular 2d vol surface assumed to be sticky_moneyness - need to handle other moneyness rules (TODO!)
        forward_or_spot = forward if use_forward else spot
        return strike / forward_or_spot if invert_moneyness else forward_or_spot / strike


class SensitivitiesEstimator(object):
    """ Implements the AAD sensitivities (both first and second derivatives)"""

    def __init__(self, value, params, create_graph=False):
        """
        Args:
            value: function output (tensor)
            params: List of model parameters (list of tensor(s))
        """
        # run the backward pass
        value.backward(retain_graph=True, create_graph=create_graph)
        # store associated gradients
        self.params = {key: tensor for key, tensor in params if tensor.grad is not None}
        self.grad = {key: tensor.grad for key, tensor in self.params.items()}
        self.flat_grad = self.flatten(list(self.grad.values()))
        self.device = self.flat_grad.device
        self.dtype = self.flat_grad.dtype
        self.list_param = list(self.params.values())
        self.P = len(self.flat_grad)

    def report_grad(self):
        return {utils.check_scope_name(factor): tensor.cpu().detach().numpy() for factor, tensor in self.grad.items()}

    def report_hessian(self, allow_unused=False):
        h_op = self.get_H_op(allow_unused)
        hessian = np.zeros((self.P, self.P))

        # store it in a matrix
        j = 0
        for row in h_op:
            v, _ = row.shape
            hessian[j:j + v, j:] = row
            j += v

        # zero out the lower indices
        hessian[np.tril_indices(self.P, k=-1)] = 0.0
        return hessian + np.triu(hessian, k=1).T

    def get_Hv_op(self, v):
        """
        Implements a Hessian vector product estimator Hv op defined as the
        matrix multiplication of the Hessian matrix H with the vector v.

        Args:
            v: Vector to multiply with Hessian (tensor)

        Returns:
            Hv_op: Hessian vector product op (tensor)
        """
        hv = self.flatten(torch.autograd.grad(
            self.flat_grad, self.list_param, grad_outputs=v, only_inputs=True, retain_graph=True))

        return hv

    @staticmethod
    def flatten(params):
        """
        Flattens the list of tensor(s) into a 1D tensor
        
        Args:
            params: List of model parameters (List of tensor(s))
        
        Returns:
            A flattened 1D tensor
        """
        return torch.cat([_params.reshape(-1) for _params in params], dim=0)

    def get_H_op(self, allow_unused):
        """ 
        Implements a Hessian estimator op by forming p Hessian vector
        products using HessianEstimator.get_Hv_op(v) for all v's in R^P
        
        Args:
            None
        
        Returns:
            H_op: Hessian matrix op (tensor)
            :param allow_unused:
        """

        glist = self.grad.values()
        plist = self.list_param
        e = {l: torch.eye(l, dtype=self.dtype, device=self.device) for l in set([len(g) for g in glist])}
        if allow_unused:
            z = {l: torch.zeros(l, dtype=self.dtype, device=self.device) for l in set([len(p) for p in plist])}

        hessian = []

        for i, g in enumerate(glist):
            g_size = len(g)
            row = []
            for block in [torch.autograd.grad(
                    g, plist[i:], grad_outputs=x, only_inputs=True,
                    allow_unused=allow_unused, retain_graph=True) for x in e[g_size]]:

                if allow_unused:
                    row.append(torch.cat([b if b is not None else z[len(p)] for b, p in zip(block, plist[i:])]))
                else:
                    row.append(torch.cat(block))

            hessian.append(torch.stack(row).cpu().detach().numpy())

        return hessian


def greeks(shared, deal_data, mtm):
    greeks_calc = SensitivitiesEstimator(mtm, shared.calc_greeks, create_graph=shared.gamma)
    deal_data.Calc_res['Greeks_First'] = greeks_calc.report_grad()
    # use this only when all the vols and curves are sparsely represented (check greeks_calc.P)
    if shared.gamma:
        deal_data.Calc_res['Greeks_Second'] = greeks_calc.report_hessian(allow_unused=True)


def interpolate(mtm, shared, time_grid, deal_data, interpolate_grid=True):
    if interpolate_grid and deal_data.Time_dep.interp.size > deal_data.Time_dep.deal_time_grid.size:
        # interpolate it
        mtm = utils.gather_interp_matrix(mtm, deal_data.Time_dep, shared)

    # check if we want to store the mtm value for this instrument
    if deal_data.Calc_res is not None:
        shared.save_results(deal_data.Calc_res, {'Value': mtm})
        # deal_data.Calc_res.setdefault('MTM', []).append(mtm_np.sum(axis=1))

    if mtm.shape != (1, 1) and mtm.shape[0] < time_grid.mtm_time_grid.size:
        # pad it with zeros and return
        return F.pad(mtm, [0, 0, 0, time_grid.mtm_time_grid.size - deal_data.Time_dep.interp.size])
    else:
        return mtm


def getbarrierpayoff(direction, eta, phi, strike, H):
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

    def barrier_option(sigma, expiry, cash_rebate, b, r, spot, barrier):

        sigma2 = sigma * sigma
        vol = sigma * torch.sqrt(expiry)
        mu = (b - 0.5 * sigma2) / sigma2
        log_bar = torch.log(barrier / spot)
        x1 = torch.log(spot / strike) / vol + (1.0 + mu) * vol
        x2 = torch.log(spot / barrier) / vol + (1.0 + mu) * vol

        y1 = torch.log((barrier * barrier) / (spot * strike)) / vol + (1.0 + mu) * vol
        y2 = log_bar / vol + (1.0 + mu) * vol
        lam = torch.sqrt(mu * mu + 2.0 * r / sigma2)
        z = log_bar / vol + lam * vol
        eta_scale = 0.7071067811865476 * eta
        phi_scale = 0.7071067811865476 * phi
        expiry_r = expiry * r

        if direction == BARRIER_IN:
            if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or
                    (phi == OPTION_PUT and eta == BARRIER_DOWN and strike <= H)):
                # A+E
                return (cash_rebate * (
                        (0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(2 * log_bar * mu) +
                        0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x1) - 1.0) * torch.exp(b * expiry) +
                                0.5 * strike * torch.erfc(phi_scale * (vol - x1)))) * torch.exp(-expiry_r)
            elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_PUT and eta == BARRIER_DOWN and strike > H)):
                # B-C+D+E
                return (cash_rebate * (
                        (0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(2 * log_bar * mu) +
                        0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x2) - 1.0) * torch.exp(b * expiry) +
                                0.5 * strike * torch.erfc(phi_scale * (vol - x2))) + phi * (
                                spot * (0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1)) - spot * (
                                        0.5 * torch.erfc(eta_scale * y2) - 1.0) *
                                torch.exp(expiry * (b - r) + 2 * log_bar * (mu + 1)) + 0.5 * strike * torch.exp(
                            -expiry_r + 2 * log_bar * mu) * torch.erfc(eta_scale * (vol - y1)) -
                                0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(
                            eta_scale * (vol - y2))) * torch.exp(expiry_r)) * torch.exp(-expiry_r)
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike <= H)):
                # A-B+D+E
                return (cash_rebate * ((0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(
                    2 * log_bar * mu) + 0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * torch.erfc(eta_scale * y2) - 1.0) * torch.exp(expiry * (
                                b - r) + 2 * log_bar * (mu + 1)) + 0.5 * strike * torch.exp(
                            -expiry_r + 2 * log_bar * mu) * torch.erfc(
                            eta_scale * (vol - y2))) * torch.exp(expiry_r) - phi * (spot * (0.5 * torch.erfc(
                    phi_scale * x1) - 1.0) * torch.exp(b * expiry) + 0.5 * strike * torch.erfc(
                    phi_scale * (vol - x1))) + phi * (spot * (0.5 * torch.erfc(
                    phi_scale * x2) - 1.0) * torch.exp(b * expiry) + 0.5 * strike * torch.erfc(
                    phi_scale * (vol - x2)))) * torch.exp(-expiry_r)
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike > H)):
                # C+ E
                return (cash_rebate * ((0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(
                    2 * log_bar * mu) + 0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (spot * (
                        0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(expiry * (b - r) + 2 * log_bar * (
                        mu + 1)) + 0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(eta_scale * (
                        vol - y1))) * torch.exp(expiry_r)) * torch.exp(-expiry_r)
        else:
            if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or
                    (phi == OPTION_PUT and eta == BARRIER_DOWN and strike <= H)):
                # F
                return -cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar) -
                                       0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(
                    -log_bar * (lam - mu))
            elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_PUT and eta == BARRIER_DOWN and strike > H)):
                # A - B + C - D + F
                return (-cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar)
                                        - 0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(expiry_r)
                        + phi * (-spot * (0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1)) + spot * (
                                         0.5 * torch.erfc(eta_scale * y2) - 1.0) *
                                 torch.exp(expiry * (b - r) + 2 * log_bar * (mu + 1)) - 0.5 * strike * torch.exp(
                                    -expiry_r + 2 * log_bar * mu) *
                                 torch.erfc(eta_scale * (vol - y1)) + 0.5 * strike * torch.exp(
                                    -expiry_r + 2 * log_bar * mu) *
                                 torch.erfc(eta_scale * (vol - y2))) * torch.exp(expiry_r + log_bar * (lam - mu)) +
                        phi * (-spot * (0.5 * torch.erfc(phi_scale * x1) - 1.0) * torch.exp(b * expiry) + spot * (
                                0.5 * torch.erfc(phi_scale * x2) - 1.0) * torch.exp(b * expiry) - 0.5 * strike *
                               torch.erfc(phi_scale * (vol - x1)) + 0.5 * strike * torch.erfc(phi_scale * (vol - x2)))
                        * torch.exp(log_bar * (lam - mu))) * torch.exp(-expiry_r - log_bar * (lam - mu))
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike <= H)):
                # B - D + F
                return (-cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar)
                                        - 0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(expiry_r) +
                        phi * (spot * (0.5 * torch.erfc(eta_scale * y2) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1))
                               + 0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(
                                    eta_scale * (vol - y2))) * torch.exp(
                            expiry_r + log_bar * (lam - mu)) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x2) - 1.0) * torch.exp(
                            b * expiry) + 0.5 * strike * torch.erfc(phi_scale * (vol - x2))) * torch.exp(
                            log_bar * (lam - mu))) * torch.exp(
                    -expiry_r - log_bar * (lam - mu))
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike > H)):
                # A-C+F
                return (-cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar)
                                        - 0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(expiry_r) +
                        phi * (spot * (0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1))
                               + 0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(
                                    eta_scale * (vol - y1))) * torch.exp(
                            expiry_r + log_bar * (lam - mu)) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x1) - 1.0) * torch.exp(
                            b * expiry) + 0.5 * strike * torch.erfc(phi_scale * (vol - x1))) * torch.exp(
                            log_bar * (lam - mu))) * torch.exp(
                    -expiry_r - log_bar * (lam - mu))

    return barrier_option


def getpartialbarrierpayoff(isKnockIn, eta, phi, spot, strike, barrier, startBarrier, limit, expiry, r, b, sigma):
    '''
    Function to generate the partial barrier payoff function
    '''

    def BarrierPutCallTransformation(spot, strike, barrier, r, b, upDown):
        return strike, spot, spot * strike / barrier, r - b, -b, -upDown

    def PartialBarrierCalc(forward, strike, log1, log2, rho1, rho2, p1, p2, p3, p4, p5, p6, p7, p8):
        return (forward * (utils.ApproxBivN(p1, p2, rho1) - torch.exp(log1) * utils.ApproxBivN(p3, p4, rho2)) -
                strike * (utils.ApproxBivN(p5, p6, rho1) - torch.exp(log2) * utils.ApproxBivN(p7, p8, rho2)))

    def partial_barrier_option(spot, strike, barrier, r, b, eta):
        rho = torch.sqrt(limit / expiry)
        vol = sigma * torch.sqrt(expiry)
        vollimit = sigma * torch.sqrt(limit)
        halfvv = 0.5 * sigma * sigma
        logSpotOverStrike = torch.log(spot / strike)
        logSpotOverBarrier = torch.log(spot / barrier)
        d1 = (logSpotOverStrike + (b + halfvv) * expiry) / vol
        d2 = d1 - vol
        f1 = (logSpotOverStrike - 2.0 * logSpotOverBarrier + (b + halfvv) * expiry) / vol
        f2 = f1 - vol
        e1 = (logSpotOverBarrier + (b + halfvv) * limit) / vollimit
        e2 = e1 - vollimit
        e3 = e1 - 2.0 * logSpotOverBarrier / vollimit
        e4 = e3 - vollimit
        mu2 = b / halfvv - 1.0
        forward = spot * torch.exp(b * expiry)
        log1 = -logSpotOverBarrier * (mu2 + 2.0)
        log2 = -logSpotOverBarrier * mu2

        if startBarrier:
            etaRho = eta * rho
            pv = PartialBarrierCalc(
                forward, strike, log1, log2, etaRho, etaRho, d1, e1 * eta,
                f1, e3 * eta, d2, e2 * eta, f2, e4 * eta)
        else:
            g1 = (logSpotOverBarrier + (b + halfvv) * expiry) / vol
            g2 = g1 - vol
            g3 = g1 - 2.0 * logSpotOverBarrier / vol
            g4 = g3 - vol

            if eta == 0:  # type B1

                pv = PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, d1, e1, f1, -e3, d2, e2, f2, -e4)
                temp = PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, g1, e1, g3, -e3, g2, e2, g4, -e4)
                temp += PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, -g1, -e1, -g3, -e3, -g2, -e2, -g4, -e4)
                temp -= PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, -d1, -e1, -f1, -e3, -d2, -e2, -f2, -e4)

                pv = torch.where(strike < barrier, pv, temp)
            else:
                if eta == 1:

                    pv = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, g1, e1, g3, -e3, g2, e2, g4, -e4)
                    temp = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, d1, e1, f1, -e3, d2, e2, f2, -e4)

                    pv = torch.where(strike < barrier, pv, temp)

                else:  # eta == -1

                    pv = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, -g1, -e1, -g3, e3, -g2, -e2, -g4, e4)
                    pv -= PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, -d1, -e1, e3, -f1, -d2, -e2, e4, -f2)
                    pv *= (strike < barrier)

        return pv * torch.exp(-r * expiry)

    if isKnockIn:
        bs = utils.black_european_option(
            spot * torch.exp(b * expiry), strike, sigma * torch.sqrt(expiry),
            1.0, 1.0, phi, None) * torch.exp(-r * expiry)

    if phi == -1:
        spot, strike, barrier, r, b, eta = BarrierPutCallTransformation(spot, strike, barrier, r, b, eta)

    pv = partial_barrier_option(spot, strike, barrier, r, b, eta)

    if isKnockIn:
        return bs - pv
    else:
        return pv


def calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared):
    # None means get the ATM vol for this expiry (can change depending on the vol surface type)
    fx_vols = utils.calc_time_grid_vol_rate(factor_dep['FXVol'], None, expiry, shared)

    # b_adj adjusts the carry on the forward, s_adj is to scale the forward directly (as a factor)
    if 'QuantoImpliedCorrelation' in factor_dep:
        # quanto fx deal
        atm_vol = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], None, expiry, shared)
        return {'vol': vols, 'b_adj': -atm_vol * fx_vols * factor_dep['QuantoImpliedCorrelation'], 's_adj': 1.0}
    else:
        tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
        forwardfx = utils.calc_fx_forward(
            factor_dep['Local'], factor_dep['Other'],
            tenor_in_days, deal_time, shared)
        compo_vols = torch.sqrt(
            fx_vols * fx_vols + 2.0 * fx_vols * vols * factor_dep['CompoImpliedCorrelation'] + vols * vols)
        return {'vol': compo_vols, 'b_adj': 0.0, 's_adj':forwardfx}


def smooth_max(x, k, eps=0.01):
    return torch.where(x < k - eps, torch.zeros_like(x), torch.where(
        x > k + eps, x - k, (-k * x + 0.5 * x ** 2 + eps * x + 0.5 * (k - eps) ** 2) / (2 * eps)))


def smooth_relu(x, eps=0.005):
    return torch.where(x < - eps, torch.zeros_like(x), torch.where(
        x > eps, x, (0.5 * x ** 2 + eps * x + 0.5 * eps ** 2) / (2 * eps)))


def smooth_heaviside_up(x, k, eps=0.01):
    return torch.where(x < k - eps, torch.zeros_like(x),
                       torch.where(x > k + eps, torch.ones_like(x), 0.5 + (x - k) / (2.0 * eps)))


def smooth_heaviside_down(x, k, eps=0.01):
    return torch.where(x < k - eps, torch.ones_like(x),
                       torch.where(x > k + eps, torch.zeros_like(x), 0.5 + (k - x) / (2.0 * eps)))


def pv_discrete_barrier_option(shared, time_grid, deal_data, spot, b,
                               tau, invert_moneyness=False, use_forwards=False, isdigital=False):
    def sim_discrete_barrier(S, isBarrierDate):
        # MAIN LOOP
        breached = 0.0
        tMax = len(S)
        mtm_list = torch.zeros_like(S)

        for t in range(tMax):
            if isBarrierDate[t] > 0.0:
                breachEvent = smooth_heaviside_up(
                    S[t], barrier) if eta == BARRIER_UP else smooth_heaviside_down(S[t], barrier)
                # rebate = (direction == BARRIER_OUT) * (breached == 0) & breachEvent
                # mtm_list[t] = cash_rebate * rebate
                breached = breached + breachEvent

        # digital payoff
        if isdigital:
            payoff = smooth_heaviside_down(
                S[-1], strike) if phi == OPTION_PUT else smooth_heaviside_up(S[-1], strike)
        else:
            payoff = smooth_relu((strike - S[-1]) if phi == OPTION_PUT else (S[-1] - strike))

        # pay at expiry
        if direction == BARRIER_IN:
            mtm_list[t] = payoff * (breached > 0) + cash_rebate * (breached == 0)
        else:
            mtm_list[t] = payoff * (breached == 0)

        return mtm_list.mean(axis=2)

    def sim_spot(spot_prices, vols, times, carry, offset, terminationDate, sobol=False, num_sims=1024 * 4):
        timesteps, num_samples = times.shape
        dt = times.unsqueeze(axis=2)
        var = (vols * vols).unsqueeze(axis=1) * dt
        # store params in tensors
        drift = carry * dt - 0.5 * var
        # not strictly speaking correct, but we need to do this to prevent gradients from blowing up
        vol = torch.sqrt(var.clamp(min=1e-4))

        isBarrierDate = BarrierDates[offset:]

        mcmc = []
        for s, r, sigma in zip(spot_prices, drift, vol):
            if sobol:
                z = shared.quasi_rng(shared.simulation_batch, num_samples * num_sims)[0].T.reshape(
                    num_samples, shared.simulation_batch, -1)
            else:
                z = torch.randn([num_samples, shared.simulation_batch, num_sims],
                                dtype=shared.one.dtype, device=shared.one.device)
            f1 = (r.unsqueeze(axis=2) + sigma.unsqueeze(axis=2) * z).cumsum(axis=0)
            mcmc_spot = s.reshape(1, -1, 1) * torch.exp(f1)
            val = sim_discrete_barrier(mcmc_spot, isBarrierDate)
            mcmc.append(val)

        return torch.stack(mcmc)

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    # get the zero curve
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # set up the calculation grid
    samples = factor_dep['Observation_Dates'].reinitialize(shared.one)
    start_idx = samples.get_start_index(deal_time)
    dual_samples = samples.dual()
    start_index, counts = np.unique(start_idx, return_counts=True)

    # work out what we're pricing
    BarrierDates = [1 if x != -1 else -1 for x in factor_dep['Barrier_Dates']]
    phi = OPTION_CALL if deal_data.Instrument.field['Option_Type'] == 'Call' else OPTION_PUT
    eta = BARRIER_DOWN if 'Down' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_UP
    direction = BARRIER_OUT if 'Out' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_IN
    barrier = deal_data.Instrument.field['Barrier_Price']
    strike = deal_data.Instrument.field['Strike_Price']
    cash_rebate = deal_data.Instrument.field.get('Cash_Rebate', 0.0)

    nominal = factor_dep['Buy_Sell'] * (
        deal_data.Instrument.field['Cash_Payoff'] if isdigital else deal_data.Instrument.field['Units'])
    terminationDate = -shared.one.new_ones(shared.simulation_batch, 1)

    sobol = False
    # use a quasi random generator if we are simulating a large batch
    if shared.simulation_batch > 16:
        # reset the sobol counter (so that subsequent runs reuse the same quasi random numbers)
        sobol = True
        shared.reset_qrg()

    expiry = daycount_fn(tau)
    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = spot.new(expiry.reshape(-1, 1))

    expiry_years = factor_dep[expiry_years_key]
    forward = spot * torch.exp(b * expiry_years)
    moneyness = calc_moneyness(shared.one * strike, spot, forward, deal_data, use_forwards, invert_moneyness)

    for index, (discount_block, spot_block, moneyness_block) in enumerate(
            utils.split_counts([discount, spot, moneyness], counts, shared)):

        t_block = discount_block.time_grid
        sample_index_t = start_index[index]
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]
        fixings = (dual_samples.np[np.newaxis, sample_index_t:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis])

        drifts = utils.calc_eq_drift(
            deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield'],
            fixings, t_block, shared, multiply_by_time=False)

        fixing_block = daycount_fn(fixings)
        # note that we're using just 1 vol moneyness - the one at expiry (the last date of the Option)
        # this should actually be the moneyness for each settlement date - TODO!
        expiry = daycount_fn(tenor_block)
        vols = utils.calc_time_grid_vol_rate(
            factor_dep['Volatility'], moneyness_block, expiry, shared)

        if factor_dep.get('Check_Payoff_Type', False):
            # need quanto/compo adjustments
            adj = calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared)
            vols = adj['vol']
            drifts = drifts + torch.unsqueeze(adj['b_adj'], 1)

        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))

        # sometimes the deal kicks out early for this batch - skip redundant calcs
        if terminationDate.any():
            theo_cashflow = nominal * sim_spot(
                spot_block, vols, sample_ts, drifts, sample_index_t, terminationDate,
                sobol=sobol, num_sims=shared.MCMC_sims)
        else:
            theo_cashflow = torch.zeros_like(drifts)

        discount_rates = utils.calc_discount_rate(discount_block, fixings, shared)

        theo_price = (theo_cashflow * discount_rates).sum(axis=1)
        # settle potential cashflows
        cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
            time_grid.mtm_time_grid, t_block[-1][utils.TIME_GRID_MTM]), theo_cashflow[-1][0])
        # if the last cashflow was 0 then the autocall hasn't knocked out yet
        terminationDate = -1.0 * (terminationDate == -1) * (theo_cashflow[-1][0] == 0.0).reshape(-1, 1)
        mtm_list.append(theo_price)

    return torch.cat(mtm_list, dim=0)


def pv_barrier_option(shared, time_grid, deal_data, nominal, spot, b,
                      tau, payoff_currency, invert_moneyness=False, use_forwards=False):
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
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    expiry = daycount_fn(tau)
    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = spot.new(expiry.reshape(-1, 1))

    expiry_years = factor_dep[expiry_years_key]
    forward = spot * torch.exp(b * expiry_years)
    moneyness = calc_moneyness(shared.one * strike, spot, forward, deal_data, use_forwards, invert_moneyness)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep.get('Check_Payoff_Type', False):
        # need quanto/compo adjustments
        adj = calc_vol_adjustment(factor_dep, deal_time, expiry, sigma, shared)
        sigma = adj['vol']
        b = b + adj['b_adj']

    barrierOption = getbarrierpayoff(direction, eta, phi, strike, barrier)
    r = torch.squeeze(discounts.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), dim=1)

    mtm_list = []
    prev_touched = 0.0

    for index, (raw_sig, exp, b_t, r_t, s_t, f_t, cash_index) in enumerate(zip(
            sigma, expiry_years, b, r, spot, forward, deal_data.Time_dep.deal_time_grid)):

        # barrier options are very sensitive to vols - so we clamp them to 5%
        sig = raw_sig.clamp(min=0.05)

        if eta == BARRIER_UP:
            touched = (prev_touched + (s_t > barrier)).clip(max=1.0)
        else:
            touched = (prev_touched + (s_t < barrier)).clip(max=1.0)

        if (1 - touched).any() and expiry[index] > 0:
            if factor_dep['Barrier_Monitoring']:
                barrier_t = barrier * torch.exp(
                    (2.0 * (barrier > s_t) - 1.0) * sig * factor_dep['Barrier_Monitoring'])
            else:
                barrier_t = barrier

            payoff = buy_or_sell * nominal * barrierOption(
                sig, exp, cash_rebate / nominal, b_t, r_t, s_t, barrier_t)
        else:
            payoff = 0.0

        barrier_part = (1.0 - touched) * payoff

        if direction == BARRIER_IN:
            # barrier ? and in
            payoff_european = buy_or_sell * (
                utils.black_european_option(f_t, strike, sig, expiry[index], 1.0, phi, shared) * torch.exp(
                    -r_t * exp) if expiry[index] else torch.relu(phi * (f_t - strike))
            )
            european_part = touched * (nominal * payoff_european)
            rebate_part = buy_or_sell * cash_rebate * (1 - touched) if expiry[index] == 0.0 else 0.0
            mtm_list.append(rebate_part + european_part + barrier_part)
        else:
            # barrier ? and out
            rebate_part = buy_or_sell * cash_rebate * (touched - prev_touched)
            mtm_list.append(rebate_part + barrier_part)
            # settle cashflows (The potential rebate)
            if cash_rebate and expiry[index] > 0.0:
                cash_settle(shared, payoff_currency, cash_index, rebate_part)

        prev_touched = touched

    # settle cashflows (The one at expiry)
    cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], mtm_list[-1])
    return torch.stack(mtm_list)


def pv_one_touch_option(shared, time_grid, deal_data, nominal, spot, b,
                        tau, payoff_currency, invert_moneyness=False, use_forwards=False):
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing

    eta = BARRIER_DOWN if 'Down' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_UP
    buy_or_sell = 1.0 if deal_data.Instrument.field['Buy_Sell'] == 'Buy' else -1.0
    barrier = deal_data.Instrument.field['Barrier_Price']

    # get the zero curve
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    expiry = daycount_fn(tau)
    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = spot.new(expiry.reshape(-1, 1))

    expiry_years = factor_dep[expiry_years_key]
    forward = spot * torch.exp(b * expiry_years)
    moneyness = calc_moneyness(shared.one * barrier, spot, forward, deal_data, use_forwards, invert_moneyness)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep.get('Check_Payoff_Type', False):
        # need quanto/compo adjustments
        adj = calc_vol_adjustment(factor_dep, deal_time, expiry, sigma, shared)
        sigma = adj['vol']
        b = b + adj['b_adj']

    r = torch.squeeze(discounts.gather_weighted_curve(
        shared, tau.reshape(-1, 1), multiply_by_time=False), dim=1)

    mtm_list = []
    prev_touched = 0.0
    eta_scale = 0.7071067811865476 * eta

    for index, (raw_sig, exp, b_t, r_t, s_t, cash_index) in enumerate(zip(
            sigma, expiry_years, b, r, spot, deal_data.Time_dep.deal_time_grid)):

        if eta == BARRIER_UP:
            touched = (prev_touched + (s_t > barrier)).clip(max=1.0)
        else:
            touched = (prev_touched + (s_t < barrier)).clip(max=1.0)

        if (1 - touched).any() and expiry[index] > 0:
            if factor_dep['Barrier_Monitoring']:
                barrier_t = barrier * torch.exp(
                    (2.0 * (barrier > s_t) - 1.0) * raw_sig * factor_dep['Barrier_Monitoring'])
            else:
                barrier_t = barrier

            root_tau = torch.sqrt(exp)
            mu = b_t / raw_sig - 0.5 * raw_sig
            log_vol = torch.log(barrier_t / s_t) / raw_sig
            barrovert = log_vol / root_tau

            if deal_data.Instrument.field['Payment_Timing'] == 'Expiry':
                muroot = mu * root_tau
                d1 = muroot - barrovert
                d2 = -muroot - barrovert
                payoff = torch.exp(-r_t * exp) * 0.5 * (
                        torch.erfc(eta_scale * d1) + torch.exp(2.0 * mu * log_vol) * torch.erfc(eta_scale * d2))

            elif deal_data.Instrument.field['Payment_Timing'] == 'Touch':
                lamb = torch.sqrt(smooth_relu(mu * mu + 2.0 * r_t))
                lambroot = lamb * root_tau
                d1 = lambroot - barrovert
                d2 = -lambroot - barrovert
                payoff = 0.5 * (torch.exp((mu - lamb) * log_vol) * torch.erfc(eta_scale * d1) +
                                torch.exp((mu + lamb) * log_vol) * torch.erfc(eta_scale * d2))
        else:
            payoff = 0.0

        one_touch_part = (1.0 - touched) * buy_or_sell * nominal * payoff

        # settle cashflows (The potential rebate)
        if deal_data.Instrument.field['Payment_Timing'] == 'Touch':
            rebate_part = buy_or_sell * nominal * (touched - prev_touched)
            if rebate_part.any():
                cash_settle(shared, payoff_currency, cash_index, rebate_part)
        elif deal_data.Instrument.field['Payment_Timing'] == 'Expiry' and expiry[index] == 0.0:
            rebate_part = buy_or_sell * nominal * touched
            if rebate_part.any():
                cash_settle(shared, payoff_currency, cash_index, rebate_part)
        else:
            rebate_part = 0.0

        mtm_list.append(rebate_part + one_touch_part)
        prev_touched = touched

    return torch.stack(mtm_list)


def pv_partial_barrier_option(shared, time_grid, deal_data, nominal,
                              spot, b, tau, tau1, payoff_currency, invert_moneyness=False):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    factor_dep = deal_data.Factor_dep
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing
    barrierType = deal_data.Instrument.field['Barrier_Type']
    isKnockIn = barrierType in ['Up_And_In', 'Down_And_In', 'In']

    eta = 0.0
    if barrierType in ['Down_And_Out', 'Down_And_In']:
        eta = BARRIER_DOWN
    elif barrierType in ['Up_And_Out', 'Up_And_In']:
        eta = BARRIER_UP

    phi = OPTION_CALL if deal_data.Instrument.field['Option_Type'] == 'Call' else OPTION_PUT
    buy_or_sell = 1.0 if deal_data.Instrument.field['Buy_Sell'] == 'Buy' else -1.0
    barrier = deal_data.Instrument.field['Barrier_Price']
    strike = deal_data.Instrument.field['Strike_Price']
    # get the zero curve
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time[:-1], shared)

    expiry = daycount_fn(tau)
    limit = daycount_fn(tau1)
    need_spot_at_expiry = deal_time.shape[0] - expiry.size
    spot_prior, spot_at = torch.split(spot, (expiry.size, need_spot_at_expiry))
    moneyness = calc_moneyness(shared.one * strike, spot_prior, spot_prior, deal_data, False, invert_moneyness)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep['Barrier_Monitoring']:
        adj_barrier = barrier * torch.exp(
            (2.0 * (barrier > spot[0][0]).type(shared.one.dtype) - 1.0) * sigma * factor_dep['Barrier_Monitoring'])
    else:
        adj_barrier = barrier

    r = torch.squeeze(discounts.gather_weighted_curve(
        shared, tau.reshape(-1, 1), multiply_by_time=False), dim=1)

    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry), tuple(limit))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = (spot.new(expiry.reshape(-1, 1)), spot.new(limit.reshape(-1, 1)))

    expiry_years, limit_years = factor_dep[expiry_years_key]

    barrier_payoff = buy_or_sell * nominal * getpartialbarrierpayoff(
        isKnockIn, eta, phi, spot_prior, strike, adj_barrier,
        deal_data.Instrument.field['Barrier_At_Start'] == 'Yes',
        limit_years, expiry_years, r, b, sigma)

    if need_spot_at_expiry:
        # work out barrier
        if eta == BARRIER_UP:
            touched = (spot[:-1] < barrier) & (spot[1:] > barrier)
        else:
            touched = (spot[:-1] > barrier) & (spot[1:] < barrier)

        # barrier payoff
        barrier_touched = F.pad((torch.cumsum(touched, dim=0) > 0).type(shared.one.dtype), [0, 0, 1, 0])
        first_touch = barrier_touched[1:] - barrier_touched[:-1]
        # final payoff
        payoff_at = buy_or_sell * torch.relu(phi * (spot_at - strike))

        if direction == BARRIER_IN:
            forward = spot_prior * torch.exp(b * expiry_years)
            payoff_prior = utils.black_european_option(
                forward, strike, sigma, expiry, buy_or_sell, phi, shared) * torch.exp(-r * expiry_years)
            european_part = barrier_touched * (nominal * torch.cat([payoff_prior, payoff_at], dim=0))
            barrier_part = (1.0 - barrier_touched) * F.pad(
                barrier_payoff, [0, 0, 0, 1], value=buy_or_sell * cash_rebate)
            combined = european_part + barrier_part
            # settle cashflows (can only happen at the end)
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], combined[-1])
        else:
            # barrier out
            barrier_part = (1.0 - barrier_touched) * torch.cat([barrier_payoff, nominal * payoff_at], dim=0)
            rebate_part = buy_or_sell * cash_rebate * first_touch
            combined = F.pad(buy_or_sell * cash_rebate * first_touch, [0, 0, 1, 0]) + barrier_part
            # settle cashflows (The one at expiry)
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], barrier_part[-1])
            # settle cashflows (The potential rebate knockout)
            if cash_rebate:
                for cash_index, cash in zip(deal_data.Time_dep.deal_time_grid[1:], rebate_part):
                    cash_settle(shared, payoff_currency, cash_index, cash)
    else:
        combined = barrier_payoff

    return combined


def pv_american_option(shared, time_grid, deal_data, nominal, moneyness, spot, forward):
    def phi(gamma, H, I):
        kappa = (2.0 * safe_b) / sigma2 + 2.0 * gamma - 1.0
        d = (torch.log(H / safe_S) - (safe_b + (gamma - 0.5) * sigma2) * tau) / vol
        lamb = -safe_r + gamma * safe_b + 0.5 * gamma * (gamma - 1.0) * sigma2
        log_IS = torch.log(I / safe_S)
        safe_exp = (kappa * log_IS).clamp(max=25.0)
        ret = utils.norm_cdf(d) - torch.exp(safe_exp) * utils.norm_cdf(d - 2.0 * log_IS / vol)
        return torch.exp(lamb * tau) * ret

    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
    expiry = discount.code[0][utils.FACTOR_INDEX_Daycount](tenor_in_days)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    # make sure there are no zeros
    tau = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    # cost of carry
    b = torch.log(forward / spot) / tau
    # interest rates
    r = torch.squeeze(
        discount.gather_weighted_curve(shared, tenor_in_days.reshape(-1, 1), multiply_by_time=False), dim=1)
    # actual volatility
    vol = sigma * torch.sqrt(tau)
    # adjust if this is a put option
    if factor_dep['Option_Type'] > 0:
        S, K = spot, factor_dep['Strike_Price'] * shared.one
    else:
        S, K = shared.one * factor_dep['Strike_Price'], spot
        r, b = r - b, -b

    sigma2 = sigma * sigma
    american = b < r - 1e-6
    safe_b = american * b
    b_over_sigma2 = safe_b / sigma2
    # pad all non american exercise points (0.375 is arbitrary)
    safe_r = american * r + ~american * 0.375 * sigma2
    # make sure we avoid nans
    safe_sqrt = ((b_over_sigma2 - 0.5) ** 2 + 2.0 * safe_r / sigma2).clamp(min=1e-6)
    beta = (0.5 - b_over_sigma2) + torch.sqrt(safe_sqrt)
    r_b = safe_r - safe_b
    # calculate the barrier
    B_0 = K * torch.maximum(safe_r / r_b, torch.ones_like(safe_r))
    B_inf = K * beta / (beta - 1)

    # check if the strike is > 0
    if (K > 0.0).all():
        h_tau = -(b * tau + 2 * vol) * (B_0 / (B_inf - B_0))
        I = B_0 + (B_inf - B_0) * (1 - torch.exp(h_tau))
        safe_S = torch.min(S - 1e-6, I)
        C_BS = (I - K) * torch.exp(torch.log(safe_S / I) * beta) * (1.0 - phi(beta, I, I))
        x = phi(1.0, I, I)
        y = phi(1.0, K, I)
        C_BS += safe_S * (x - y)
        x = phi(0.0, K, I)
        y = phi(0.0, I, I)
        C_BS += K * (x - y)
    else:
        I = B_0
        C_BS = B_0

    Black = utils.black_european_option(
        S * torch.exp(b * tau), K, vol, 1.0, 1.0, 1.0, shared) * torch.exp(-r * tau)

    C_BS = torch.maximum(Black, C_BS)

    theo_price = (b >= r) * Black + (b < r) * ((S < I) * C_BS + (S >= I) * (S - K))
    early_exercise = (b < r) * (S >= I)

    # check the first knockout
    knocked_out = early_exercise.cumsum(axis=0) > 0
    first_knockout = torch.concat([knocked_out[0].reshape(1, -1), knocked_out[1:] ^ knocked_out[:-1]])
    # make sure we are still in force today
    knocked_out[0] = False
    value = factor_dep['Buy_Sell'] * nominal * theo_price * ~knocked_out

    # handle cashflows
    exercise_val = factor_dep['Buy_Sell'] * nominal * (S - K) * first_knockout
    for t, cashflows in zip(deal_data.Time_dep.deal_time_grid, exercise_val):
        if cashflows.any():
            cash_settle(shared, factor_dep['SettleCurrency'], t, cashflows)

    return value


def pv_european_option(shared, time_grid, deal_data, nominal, moneyness, forward, binary=False):
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
    expiry = discount.code[0][utils.FACTOR_INDEX_Daycount](tenor_in_days)
    vols = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    # check if this is a compo or quanto deal
    if factor_dep.get('Check_Payoff_Type', False):
        # need quanto/compo adjustments
        adj = calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared)
        vols = adj['vol']
        forward = adj['s_adj'] * forward * torch.exp(adj['b_adj'] * shared.one.new(expiry.reshape(-1, 1)))

    if binary:
        theo_price = utils.black_european_option(
            forward, factor_dep['Strike_Price'], vols, expiry,
            factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared, cash_payoff=nominal)
        value = theo_price
    else:
        theo_price = utils.black_european_option(
            forward, factor_dep['Strike_Price'], vols, expiry,
            factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared)
        value = nominal * theo_price

    discount_rates = torch.squeeze(
        utils.calc_discount_rate(discount, tenor_in_days.reshape(-1, 1), shared), dim=1)

    # handle cashflows (if necessary)
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], value[-1])

    return value * discount_rates

def pv_MC_Tarf(shared, time_grid, deal_data, spot):
    """
    One-step survival Monte Carlo for TARF (autograd-friendly).
    - Analytic KO-in-step: adds (1 - p_survive) * remaining_target at each fixing
    - Survival branch: GBM step with Z ~ N(0,1) truncated at z_max from the PnL barrier
    - Accrual: only ITM leg contributes to target; OTM affects PV only (and optional knock-in barrier)
    """

    # --- Accumulated target from past observed fixings --------------------------
    def calc_accum_value(targetValue, accumulated, s, k, C, inverted):
        if inverted:
            iv = (1.0 / s - 1.0 / k) * C * (-1.0)
        else:
            iv = (s - k) * C
        return (accumulated + F.relu(iv).reshape(-1,1)).clamp(max=targetValue)

    def bs_call_put_fwd(F, K, sdt, D):
        """
        Put via parity in forward measure.
        """
        d1 = (torch.log(F / K) + 0.5 * sdt * sdt) / sdt
        d2 = d1 - sdt
        call =  D * (F * utils.norm_cdf(d1) - K * utils.norm_cdf(d2))
        return call, call - D * (F - K)

    def sim_spot_tarf(spot_prices, times, carry, prev_accum, discount_rates,
                      t_block, fixings, settlement, sobol=False, num_sims=1024 * 4):

        # Styles & shapes follow your autocall implementation
        eps = torch.finfo(shared.one.dtype).eps
        # easier to type
        K = strike
        # Per-block results
        mcmc = []
        # Loop over block rows (same zipped signature you use)
        for t, D, s, carry_rate, delta_t, full_t, tau in zip(
                t_block, discount_rates, spot_prices, carry, times, fixings, settlement):

            # reduced_samples = number of GBM “substeps” in this coupon/fixing leg
            reduced_samples = len(delta_t)
            # calculate the total forward rate
            fixing_t = delta_t.cumsum(0).unsqueeze(-1)
            forward = s.unsqueeze(0) * torch.exp((carry_rate * fixing_t))
            # calculate the moneyness based on the forwards
            moneyness = strike / forward if factor_dep['Invert_Moneyness']  else forward / strike
            vols = torch.stack([utils.calc_time_grid_vol_rate(
                factor_dep['Volatility'], mon, s_tau, shared) for mon, s_tau in zip(moneyness, full_t.reshape(-1,1))])
            # RNG (uniforms for truncated draws)
            if reduced_samples:
                if sobol:
                    u = shared.quasi_rng(shared.simulation_batch, reduced_samples * num_sims)[1].T.reshape(
                        reduced_samples, shared.simulation_batch, -1)
                else:
                    u = torch.rand([reduced_samples, shared.simulation_batch, num_sims],
                                   dtype=shared.one.dtype, device=shared.one.device)

                # now we should have antithetic uniform samples
                u = torch.concat([u,1.0-u], dim=-1)

            Sj = torch.unsqueeze(s, 1)  # [batch, 1] as you do
            # Running PV and survival weight
            P = shared.one.new_zeros((shared.simulation_batch, 2*num_sims))
            L = shared.one.new_ones((shared.simulation_batch, 2*num_sims))
            # Remaining target (R) per path at this block start
            remaining_target = targetValue-prev_accum
            # which simulations are still alive?
            q = remaining_target == 0.0
            # update the Live matrix
            L = torch.where(q, torch.zeros_like(L), L)
            # Update the remaining targets
            R = (remaining_target * factor_dep['Notional1']).expand_as(P)
            # Per-fixing notional tensors (broadcastable to [batch, sims])
            N_itm = notional1
            N_otm = notional2
            # Iterate over fixings within this block using your “coupon_index”-like stepper
            for j in range(reduced_samples):
                # dt and forward-carry consistent with your autocall code
                dt = delta_t[j]
                Dj = D[j].reshape(-1, 1)
                use_past_fixing = False

                if dt > 0:
                    fwd_carry = ((carry_rate[j] * full_t[j] - carry_rate[j - 1] * full_t[j - 1]) / dt
                                 if j > 0 else carry_rate[j]).reshape(-1, 1)  # [batch,1]
                    fwd_vol = (torch.sqrt((full_t[j] * vols[j]**2 - full_t[j - 1] * vols[j - 1]**2) / dt)
                               if j > 0 else vols[j]).T
                    vol_dt = fwd_vol * torch.sqrt(dt)
                else:
                    #use past fixing
                    use_past_fixing = True

                if not use_past_fixing:
                    # ---- ONE-STEP SURVIVAL: compute p_survive via PnL barrier -----------------
                    # Standard accrual: KO within this step if (S_i - K)*cp >= R/N_itm and ITM
                    # => spot-level cap B_pnl = K + (R/N_itm) for cp=+1; inverted has reciprocal form
                    N_i = N_itm
                    if not invertedTarget:
                        B_pnl = K + (R / N_i) * callOrPut # [batch, sims]
                    else:
                        # (1/S_i - 1/K)*(-cp) * N_i >= R  =>  1/S_i >= 1/K + (R/N_i)*cp  =>  S_i <= 1 / (1/K + (R/N_i)*cp)
                        rhs = (1.0 / K) + (R / N_i) * callOrPut
                        B_pnl = 1.0 / rhs

                    # Lognormal cap -> z_max
                    # NOTE: use current Sj as “S_{i-1}”
                    z_max = (torch.log(B_pnl/Sj) - (fwd_carry - 0.5 * fwd_vol * fwd_vol) * dt) / vol_dt
                    PhiB = utils.norm_cdf(z_max)  # = P(Z <= z_max)
                    # cv_call, cv_put = bs_call_put_fwd(Sj*torch.exp(fwd_carry*dt), K, vol_dt, Dj)
                    if (callOrPut > 0):
                        p = PhiB  # survival = Z <= z_max
                        Z = utils.norm_icdf(torch.clamp(u[j] * p, eps, 1.0 - eps))
                        # itm_cv, otm_cv = cv_call, cv_put
                    else:
                        p = 1.0 - PhiB  # survival = Z >= z_max
                        Z = utils.norm_icdf(torch.clamp(PhiB + u[j] * p, eps, 1.0 - eps))
                        # itm_cv, otm_cv = cv_put, cv_call,
                    # ---- Analytic KO-in-step contribution -------------------------------------
                    # KO pays the *remaining target* this step, discounted at the j-th discount point
                    P = P + (1.0 - p) * L * R * Dj
                    # GBM increment
                    Sj = Sj * (torch.exp((fwd_carry - 0.5 * fwd_vol * fwd_vol) * dt + vol_dt * Z) if dt > 0 else 1.0)
                else:
                    Sj = all_samples[-min(reduced_samples, num_samples)].reshape(-1, 1)
                    p = 1.0
                # ---- Economic PV for this fixing -----------------------------------------
                intr = ((Sj - K) * callOrPut).clamp(max=remaining_target)  # intrinsic
                itm_mask = (intr > 0.0)  # ITM
                # Optional knock-in on OTM leg
                if barrier > 0.0:
                    barrier_intr = (barrier - Sj) * callOrPut
                    barrier_hit = (barrier_intr >= 0.0).to(Sj.dtype)
                else:
                    barrier_hit = torch.ones_like(Sj)
                # economic per-fixing cashflow (signed)
                cf_itm = F.relu(intr) * N_itm  # ≥ 0
                cf_otm = F.relu(-intr) * N_otm * barrier_hit  # ≤ 0
                cf_step = L * p * (cf_itm - cf_otm)  # signed
                # add discounted PV to P
                P = P + Dj * cf_step
                # ---- Update remaining target R on survivors ------------------------------
                if not invertedTarget:
                    accr = F.relu(intr) * N_itm
                else:
                    inv_intr = (1.0 / Sj - 1.0 / K) * (-callOrPut)
                    accr = F.relu(inv_intr) * N_itm

                R = torch.where(itm_mask, R - accr, R)  # survival construction ensures no overshoot
                # ---- Update survival weight ----------------------------------------------
                L = p * L
                # (Optional) settlement at tau==0
                if j == 0 and tau[0]==0:
                    cash_settle(
                        shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, t[utils.TIME_GRID_MTM]),
                        cf_step.mean(axis=1))
            # End-of-block: push mean PV over sims
            mcmc.append(P.mean(axis=1))

        return torch.stack(mcmc)

    # --- Main block loop --------------------------------
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # now precalc all past resets - Fixings includes settlement and fixings dates
    samples = factor_dep['Fixings'].reinitialize(shared.one)
    start_idx = samples.get_start_index(deal_time)
    start_index, counts = np.unique(start_idx, return_counts=True)

    # make sure we can access the numpy and tensor components
    fx_samples = factor_dep['Price_Fixings'].reinitialize(shared.one)
    known_resets = fx_samples.known_resets(shared.simulation_batch)
    sim_samples = fx_samples.schedule[
        (fx_samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (fx_samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    next_samples = utils.calc_fx_cross(
        factor_dep['Underlying_Currency'][0], factor_dep['Currency'][0],
        sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
    all_samples = torch.cat(
        [torch.cat(known_resets, dim=0), next_samples], dim=0) if known_resets else next_samples
    num_samples = len(all_samples)

    settle_idx = np.searchsorted(factor_dep['Settlement'], deal_time[:, utils.TIME_GRID_MTM]).astype(np.int64)
    # need to get the index of the fixing and the equity
    fixing_indices = counts.cumsum() - 1
    settle_index = settle_idx[fixing_indices]

    # params for the tarf
    targetValue = deal_data.Instrument.field['TargetLevel']
    barrier = deal_data.Instrument.field.get('Barrier', 0.0)
    notional1 = factor_dep['Notional1'] * shared.one
    notional2 = factor_dep['Notional2'] * shared.one
    strike = factor_dep['Strike_Price']
    invertedTarget = deal_data.Instrument.field['InvertedTarget']
    callOrPut = factor_dep['Option_Type']

    # calculate the correct accumulation to date
    acc = shared.one * 0.0
    for sample_val in fx_samples.schedule[:, utils.RESET_INDEX_Value]:
        if sample_val:
            acc = calc_accum_value(targetValue, acc, sample_val * shared.one, strike, callOrPut, invertedTarget)

    accumulation = [acc]
    for sample_val in next_samples:
        accumulation.append(calc_accum_value(
            targetValue, accumulation[-1], sample_val, strike, callOrPut, invertedTarget))

    sobol = False
    # use a quasi random generator if we are simulating a large batch
    if shared.simulation_batch > 16:
        # reset the sobol counter (so that subsequent runs reuse the same quasi random numbers)
        sobol = True
        shared.reset_qrg()

    # split vectors into blocks based on start_index/counts
    for b_idx, (discount_block, spot_block) in enumerate(
            utils.split_counts([discount, spot], counts, shared)):
        # get the starting index for fixings/settlement
        settle_index_local = settle_index[b_idx]
        # calculate the fixing dates
        t_block = discount_block.time_grid
        fixings = (fx_samples[np.newaxis, settle_index_local:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis]).clip(min=0)
        drifts = utils.calc_fx_drift(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            fixings, t_block, shared, multiply_by_time=False)
        fixing_block = daycount_fn(fixings)
        # Discounting and settlement
        settlement = (factor_dep['Settlement'][np.newaxis, settle_index_local:] -
                      t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))
        discount_rates = utils.calc_discount_rate(discount_block, settlement, shared)

        theo_price = factor_dep['Buy_Sell'] * sim_spot_tarf(
            spot_block, sample_ts, drifts, accumulation[settle_index_local], discount_rates,
            t_block, fixing_block, settlement, sobol=sobol, num_sims=shared.MCMC_sims)

        mtm_list.append(theo_price)

    # Reporting currency MTM
    mtm = torch.cat(mtm_list, dim=0)
    return mtm


def pv_MC_AutoCallSwap(shared, time_grid, deal_data, spot, moneyness):
    def sim_autocall(S, isBarrierDate, isFixingDate, isFloatDate, floating, threshold, coupon, terminationDate):
        avg = 0.0
        averageCounter = 0.0
        fx = isQuanto * (fixFXRate - 1.0) + 1.0

        # MAIN LOOP
        resetTime = 0
        breachEvent = barrierIsHit - 0.01
        floatingTime = 0
        sumOfSpots = 0.0

        tMax = len(S)
        mtm_list = S.new_zeros((len(isFixingDate),) + S.shape[1:])

        for t in range(tMax):
            inforce = 1.0 * (terminationDate < 0.0)

            if isBarrierDate[t] > 0.0:
                breachEvent = inforce * (S[t] <= putBarrier)

            if isFixingDate[t] > 0.0:
                sumOfSpots = sumOfSpots + S[t]
                averageCounter = averageCounter + 1.0

            if isFloatDate[t] > 0.0:
                lastKnownFloatingRate = -floating[floatingTime]
                mtm_list[resetTime] = inforce * fx * lastKnownFloatingRate
                floatingTime = floatingTime + 1

                if coupon[t] <= 0.0:
                    resetTime = resetTime + 1

            if coupon[t] > 0.0:
                avg = sumOfSpots / averageCounter

                sumOfSpots = 0.0
                averageCounter = 0.0
                termination = inforce * smooth_heaviside_up(avg, threshold[t] * strike)
                mtm_list[resetTime] += termination * fx * (rebate + coupon[t])
                terminationDate = (1 - termination) * terminationDate + termination * resetTime
                breachEvent = (1 - termination) * breachEvent

                resetTime = resetTime + 1

        alive = 1.0 * (terminationDate < 0.0)
        breached = 1.0 * (breachEvent > 0.0)
        mtm_list[resetTime - 1] += alive * fx * (rebate + breached * (rebate - (strike - avg) / strike))

        return mtm_list.mean(axis=2)

    def sim_spot(spot_prices, vols, times, carry, offset, terminationDate, discount_rates,
                 t_block, fixings, t_tenor, last_fixing, sobol=False, num_sims=1024 * 4):

        timesteps, num_samples = times.shape

        isBarrierDate = BarrierDates[offset:]
        isFloatingDate = Floating[offset:]
        threshold = Threshold[offset:]
        coupon = Coupon[offset:]

        # if there is exactly 1 fixing per coupon date, then there is no averaging
        if factor_dep['no_averaging']:
            # needed for numerical stability
            eps = torch.finfo(shared.one.dtype).eps
            fx = isQuanto * (fixFXRate - 1.0) + 1.0
            mcmc = []
            for t, tau, df, s, v, carry_rate, delta_t, full_t, floating in zip(
                    t_block, t_tenor, discount_rates, spot_prices, vols, carry, times, fixings, floating_leg):

                reduced_samples = len(delta_t)
                # reduced samples can be zero if there's just the floating leg left
                if reduced_samples:
                    if sobol:
                        u = shared.quasi_rng(shared.simulation_batch, reduced_samples * num_sims)[1].T.reshape(
                            reduced_samples, shared.simulation_batch, -1)
                    else:
                        u = torch.rand([reduced_samples, shared.simulation_batch, num_sims],
                                       dtype=shared.one.dtype, device=shared.one.device)
                if last_fixing is None:
                    Sj = torch.unsqueeze(s, 1)
                    fixing_aligned = True
                else:
                    Sj = torch.unsqueeze(all_eq_samples[last_fixing], 1)
                    fixing_aligned = False

                P = torch.zeros((shared.simulation_batch, num_sims), dtype=shared.one.dtype, device=shared.one.device)
                L = torch.ones((shared.simulation_batch, num_sims), dtype=shared.one.dtype, device=shared.one.device)
                # which simulations are still alive?
                q = terminationDate != -1
                # update the Live matrix
                L = torch.where(q, torch.zeros_like(L), L)
                # set the correct shapes for discounting and vols
                D = df.unsqueeze(2)
                vol = v.unsqueeze(1)

                coupon_index = 0

                for j, (coup, thresh, FloatingDate, barrier) in enumerate(
                        zip(coupon, threshold, isFloatingDate, isBarrierDate)):

                    if FloatingDate > 0:
                        P = P + L * fx * -FloatingDate * D[j]

                    if coup > 0:
                        K = thresh * strike
                        dt = delta_t[coupon_index] if fixing_aligned else 0.0
                        if dt > 0:
                            forward_carry = ((carry_rate[coupon_index] * full_t[coupon_index] -
                            carry_rate[coupon_index - 1] * full_t[coupon_index - 1]) / delta_t[coupon_index] \
                                if coupon_index > 0 else carry_rate[coupon_index]).reshape(-1, 1)
                            vol_dt = vol * torch.sqrt(dt)
                            p = utils.norm_cdf(
                                (torch.log(K / Sj) - (forward_carry - 0.5 * vol * vol) * dt) / vol_dt)
                        else:
                            p = torch.where(K > Sj, 1.0, 0.0)
                            if tau == 0.0:
                                terminationDate = torch.where((terminationDate == -1) & (p == 0), reduced_samples, terminationDate)

                        P = P + fx * (1 - p) * L * coup * D[j]
                        L = p * L

                        if fixing_aligned:
                            # prevent underflow or overflow
                            safe_pu = torch.clamp(p * u[coupon_index], min=eps, max=1.0-eps)

                            Sj = Sj * (torch.exp(
                                (forward_carry - 0.5 * vol * vol) * dt + vol_dt * utils.norm_icdf(
                                    safe_pu)) if dt > 0 else 1.0)
                            coupon_index += 1
                        else:
                            fixing_aligned = True

                    if barrier > 0:
                        breach = torch.where(Sj <= putBarrier, 1.0, 0.0)
                        P = P + L * D[j] * fx * breach * (rebate - (1.0 - Sj / strike))

                    if tau == 0.0:
                        cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
                            time_grid.mtm_time_grid, t[utils.TIME_GRID_MTM]), P.mean(axis=1))

                mcmc.append(P.mean(axis=1))
        else:
            isFixingDate = FixingDates[offset:]
            dt = times.unsqueeze(axis=2)
            var = (vols * vols).unsqueeze(axis=1) * dt
            # store params in tensors
            drift = carry * dt - 0.5 * var
            # not strictly speaking correct, but we need to do this to prevent gradients from blowing up
            vol = torch.sqrt(var.clamp(min=1e-4))

            mcmc = []
            for t, tau, D, s, r, sigma, floating in zip(
                    t_block, t_tenor, discount_rates, spot_prices, drift, vol, floating_leg):
                if sobol:
                    z = shared.quasi_rng(shared.simulation_batch, num_samples * num_sims)[0].T.reshape(
                        num_samples, shared.simulation_batch, -1)
                else:
                    z = torch.randn([num_samples, shared.simulation_batch, num_sims],
                                    dtype=shared.one.dtype, device=shared.one.device)
                f1 = (r.unsqueeze(axis=2) + sigma.unsqueeze(axis=2) * z).cumsum(axis=0)
                mcmc_spot = s.reshape(1, -1, 1) * torch.exp(f1)
                val = sim_autocall(
                    mcmc_spot, isBarrierDate, isFixingDate, isFloatingDate, floating, threshold, coupon,
                    terminationDate)
                pv = (val * D).sum(axis=0)
                mcmc.append(pv)

                # settle potential cashflows
                if tau == 0.0:
                    cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
                        time_grid.mtm_time_grid, t[utils.TIME_GRID_MTM]), pv)
            # if the last cashflow wasn't 0 then the autocall hasn't knocked out yet
            terminationDate = -1.0 * (terminationDate == -1) * (pv != 0.0).reshape(-1, 1)

        return torch.stack(mcmc)

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    # get the discount curve and daycount
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # set up the calculation grid
    samples = factor_dep['Fixings'].reinitialize(shared.one)
    start_idx = samples.get_start_index(deal_time)
    dual_samples = samples.dual()
    start_index, counts = np.unique(start_idx, return_counts=True)

    if factor_dep['no_averaging']:
        # resets could be prior to the coupon date - need to store this
        equity_samples = factor_dep['Price_Fixing'].reinitialize(shared.one)
        coupon_samples = factor_dep['Coupon_Fixing'].reinitialize(shared.one)
        eq_start_idx = equity_samples.get_start_index(deal_time)
        cp_start_idx = coupon_samples.get_start_index(deal_time)
        # grab the known fixings and join with any potential simulated fixings
        known_resets = equity_samples.known_resets(shared.simulation_batch)
        sim_samples = equity_samples.schedule[
            (equity_samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
            (equity_samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
        past_samples = utils.calc_time_grid_spot_rate(
            deal_data.Factor_dep['Equity'], sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
        all_eq_samples = torch.cat(
            [torch.cat(known_resets, dim=0), past_samples], dim=0) if known_resets else past_samples
        # need to get the index of the fixing and the equity
        fixing_indices = counts.cumsum() - 1
        eq_start_index = eq_start_idx[fixing_indices]
        cp_start_index = cp_start_idx[fixing_indices]
        coupon_equity_index = equity_samples.schedule[:, utils.RESET_INDEX_Reset_Day].searchsorted(
            coupon_samples.schedule[:, utils.RESET_INDEX_Reset_Day], 'right') - 1
        coupon_equity_index = np.append(coupon_equity_index, coupon_equity_index[-1] + 1)

    # make sure we can price the floating leg (if present)
    if 'Forward' in factor_dep:
        resets = factor_dep['Cashflows'].get_resets(shared.one)
        forward = utils.calc_time_grid_curve_rate(factor_dep['Forward'], deal_time, shared)
        old_resets = resets.get_simulated_resets(
            deal_time[:, utils.TIME_GRID_MTM].max(), factor_dep['Forward'], shared)

        cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
        cash_start_index = dict(zip(start_idx, cash_start_idx))
    else:
        # not used - set it to discount
        forward = discount

    # params for the autocallable
    BarrierDates = [1 if x != -1 else -1 for x in factor_dep['Barrier_Dates']]
    if not factor_dep['no_averaging']:
        FixingDates = [1 if x != -1 else -1 for x in factor_dep['Price_Fixing']]
    Threshold = factor_dep['Autocall_Thresholds']
    Floating = factor_dep['Autocall_Floating']
    Coupon = factor_dep['Autocall_Coupons']
    putBarrier = factor_dep['Barrier']
    isQuanto = 1.0 * (deal_data.Instrument.field.get('Payoff_Type') == 'Quanto')
    fixFXRate = deal_data.Instrument.field.get('FixFXRate', 1.0)
    rebate = deal_data.Instrument.field.get('Rebate', 0.0)
    barrierIsHit = 1.0 * (deal_data.Instrument.field.get('BarrierIsHit') is not None)
    strike = factor_dep['Strike_Price']
    nominal = factor_dep['Buy_Sell'] * deal_data.Instrument.field['Units']
    terminationDate = -shared.one.new_ones(shared.simulation_batch, 1)

    sobol = False
    # use a quasi random generator if we are simulating a large batch
    if shared.simulation_batch > 16:
        # reset the sobol counter (so that subsequent runs reuse the same quasi random numbers)
        sobol = True
        shared.reset_qrg()

    for index, (forward_block, discount_block, spot_block, moneyness_block) in enumerate(
            utils.split_counts([forward, discount, spot, moneyness], counts, shared)):

        t_block = discount_block.time_grid
        sample_index_t = start_index[index]

        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]
        all_fixings = (dual_samples.np[np.newaxis, sample_index_t:, utils.RESET_INDEX_End_Day] -
                       t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        fixings = (factor_dep['Price_Fixing'].schedule[np.newaxis, eq_start_index[index]:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis]) if factor_dep['no_averaging'] else all_fixings

        drifts = utils.calc_eq_drift(
            deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield'],
            fixings, t_block, shared, multiply_by_time=False)
        fixing_block = daycount_fn(fixings)

        # note that we're using just 1 vol moneyness - the one at expiry (the last date of the Option)
        # this should actually be the moneyness for each settlement date - TODO!
        expiry = daycount_fn(tenor_block)
        vols = utils.calc_time_grid_vol_rate(
            factor_dep['Volatility'], moneyness_block, expiry, shared)

        if factor_dep.get('Check_Payoff_Type', False):
            # need quanto/compo adjustments
            adj = calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared)
            vols = adj['vol']
            drifts = drifts + torch.unsqueeze(adj['b_adj'], 1)

        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)])
        ) if fixing_block.any() else fixing_block

        if 'Forward' in factor_dep:
            cashflows = factor_dep['Cashflows'].merged(shared.one, cash_start_index[sample_index_t])
            reset_offset = factor_dep['Cashflows'].offsets[cash_start_index[sample_index_t]][1]
            reset_ofs, reset_count = np.unique(
                np.searchsorted(
                    resets.schedule[reset_offset:, utils.RESET_INDEX_Reset_Day],
                    t_block[:, utils.TIME_GRID_MTM]),
                return_counts=True)

            floating_legs = []
            forward_blocks = forward_block.split_counts(
                reset_count, shared) if len(reset_count) > 1 else [forward_block]

            for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
                time_block = t_block[:, utils.TIME_GRID_MTM]
                time_slice = time_block[:size].reshape(-1, 1)
                reset_block = resets.dual(reset_offset)
                future_starts = reset_block.np[offset:, utils.RESET_INDEX_Start_Day] - time_slice
                future_ends = reset_block.np[offset:, utils.RESET_INDEX_End_Day] - time_slice
                future_weights = (reset_block.tn[offset:, utils.RESET_INDEX_Weight] /
                                  reset_block.tn[offset:, utils.RESET_INDEX_Accrual]).reshape(1, -1, 1)
                future_resets = torch.expm1(forward_rates.gather_weighted_curve(
                    shared, future_ends, future_starts)) * future_weights

                # now deal with past resets
                past_resets = torch.tile(
                    torch.unsqueeze(old_resets[reset_offset:reset_offset + offset], dim=0), [size, 1, 1])
                all_resets = torch.concat([past_resets, future_resets], dim=1) if past_resets.any() else future_resets
                # we now have all the expected floating payments for each scenario
                all_int, _ = pricer_float_cashflows(all_resets, cashflows.tn, shared)
                # need to reshape this for the mcmc pricing
                floating_legs.append(torch.unsqueeze(all_int, dim=-1))

            # Sometimes we need to combine the floating legs
            floating_leg = torch.concat(floating_legs) if len(floating_legs) > 1 else floating_legs[0]
        else:
            # not used - set it to the spot_block
            floating_leg = spot_block

        # calc the discount rates
        discount_rates = utils.calc_discount_rate(discount_block, all_fixings, shared)
        # sometimes the autocall kicks out early for this batch - skip redundant calcs
        if terminationDate.any():
            if factor_dep['no_averaging']:
                fixing_index = coupon_equity_index[cp_start_index[index]]
                last_fixing = None if fixing_index == eq_start_index[index] else fixing_index
            else:
                last_fixing = None
            theo_cashflow = sim_spot(
                spot_block, vols, sample_ts, drifts, sample_index_t, terminationDate, discount_rates,
                t_block, fixing_block, all_fixings[:, 0], last_fixing, sobol=sobol, num_sims=shared.MCMC_sims)
        else:
            theo_cashflow = torch.zeros_like(drifts)
        theo_price = nominal * theo_cashflow
        # theo_price = (theo_cashflow * discount_rates).sum(axis=1)
        # # settle potential cashflows
        # cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
        #     time_grid.mtm_time_grid, t_block[-1][utils.TIME_GRID_MTM]), theo_cashflow[-1][0])
        # # if the last cashflow was 0 then the autocall hasn't knocked out yet
        # terminationDate = -1.0 * (terminationDate == -1) * (theo_cashflow[-1][0] == 0.0).reshape(-1, 1)
        mtm_list.append(theo_price)

    # mtm in reporting currency
    mtm = torch.cat(mtm_list, dim=0)

    return mtm


def pv_discrete_asian_option(shared, time_grid, deal_data, nominal, spot, forward,
                             past_factor_list, invert_moneyness=False, use_forwards=False):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    expiry = daycount_fn(factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
    # make sure there are no zeros
    safe_expiry = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    # cost of carry
    b = torch.log(forward / spot) / safe_expiry
    # now precalc all past resets
    eps = torch.finfo(shared.one.dtype).eps
    samples = factor_dep['Samples'].reinitialize(shared.one)
    known_resets = samples.known_resets(shared.simulation_batch)
    start_idx = samples.get_start_index(deal_time)
    sim_samples = samples.schedule[(samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                   (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                    deal_time[:, utils.TIME_GRID_MTM].max())]

    # check if the spot was simulated - if not, hold it flat
    if spot.shape != forward.shape:
        past_samples = spot.expand(sim_samples.shape[0], shared.simulation_batch)
        spot = spot.expand(*forward.shape)
    else:
        past_sample_factor = [utils.calc_time_grid_spot_rate(
            past_factor, sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
            for past_factor in past_factor_list]
        past_samples = past_sample_factor[0] if len(
            past_sample_factor) == 1 else past_sample_factor[0] / past_sample_factor[1]

    all_samples = torch.cat(
        [torch.cat(known_resets, dim=0), past_samples], dim=0) if known_resets else past_samples
    # make sure we can access the numpy and tensor components
    dual_samples = samples.dual()

    start_index, counts = np.unique(start_idx, return_counts=True)

    for index, (discount_block, spot_block, forward_block, carry_block) in enumerate(
            utils.split_counts([discount, spot, forward, b], counts, shared)):
        t_block = discount_block.time_grid
        sample_index_t = start_index[index]
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]

        sample_tau = daycount_fn(
            dual_samples.np[sample_index_t:, utils.RESET_INDEX_End_Day].reshape(1, -1) -
            t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        sample_ts = carry_block.new(sample_tau)

        weight_t = dual_samples.tn[sample_index_t:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
        normalize = dual_samples.tn[sample_index_t:, utils.RESET_INDEX_Weight].sum()
        average = torch.sum(
            all_samples[:sample_index_t] * dual_samples.tn[:sample_index_t, utils.RESET_INDEX_Weight].reshape(-1, 1),
            dim=0)

        # check if we still need to account for future averages.
        if sample_tau.size:
            # strike_bar can be negative if the average is higher than the original strike price - need to clamp this.
            strike_bar = torch.clamp(
                factor_dep['Strike'] - average.reshape(1, -1).expand(counts[index], -1), min=1e-5)
            moneyness = calc_moneyness(
                strike_bar / normalize.clamp(min=eps), spot_block, forward_block, deal_data, use_forwards,
                invert_moneyness)
            # get the vol per sample tau
            vols = torch.stack([utils.calc_time_grid_vol_rate(
                factor_dep['Volatility'], mon, s_tau, shared) for mon, s_tau in zip(moneyness, sample_tau)])
            if factor_dep.get('Check_Payoff_Type', False):
                # need quanto/compo adjustments
                adj = [calc_vol_adjustment(
                    factor_dep, deal_time, s_tau, vol, shared) for s_tau, vol in zip(sample_tau, vols)]
                vols = torch.stack([x['vol'] for x in adj])
                carry_block = torch.stack([cb + x['b_adj'] for cb, x in zip(carry_block, adj)])
            else:
                carry_block = torch.unsqueeze(carry_block, dim=1)

            sample_ft = weight_t * torch.exp(carry_block * torch.unsqueeze(sample_ts, dim=2))

            M1 = torch.sum(sample_ft, dim=1)
            product_t = sample_ft * torch.exp(torch.unsqueeze(sample_ts, dim=2) * (vols * vols))
            sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
            M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)

            # trick to avoid nans in the gradients
            MM = torch.log(M2.clamp(min=eps)) - 2.0 * torch.log(M1.clamp(min=eps))
            var_t = torch.where((M1 > eps) & (M2 > eps), MM, 0.0)
            vol_t = torch.where(var_t > 0.0, torch.sqrt(var_t.clamp(min=eps)), 0.0)
            tau = 1.0
            forward_price = M1 * spot_block
        else:
            # We're past the averaging period - we know the intrinsic value
            strike_bar = factor_dep['Strike']
            # can't set tau exactly to 0 - it will break the AAD
            tau = 1e-5
            vol_t = torch.zeros_like(average)
            forward_price = average

        if factor_dep['Digital']:
            theo_price = utils.black_european_option(
            forward_price, strike_bar, vol_t, tau,
            factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared, cash_payoff=nominal)
            value = theo_price
        else:
            theo_price = utils.black_european_option(
                forward_price, strike_bar, vol_t, tau,
                factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared)
            value = nominal * theo_price

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(discount_block, tenor_block.reshape(-1, 1), shared),
            dim=1)

        mtm_list.append(value * discount_rates)

    # potential cashflows
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], value[-1])
    # mtm in reporting currency
    mtm = torch.cat(mtm_list, dim=0)

    return mtm


def pv_discrete_double_asian_option(shared, time_grid, deal_data, nominal, spot, forward,
                                    past_factor_list, invert_moneyness=False, use_forwards=False):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    expiry = daycount_fn(factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
    # make sure there are no zeros
    safe_expiry = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    # cost of carry
    b = torch.log(forward / spot) / safe_expiry
    # load the alpha multipliers (usually just 1)
    alphas = [factor_dep['Alpha_1'], factor_dep['Alpha_2']]
    # merge the resets for both samples before calculating the start_idx
    sample_reset_days = np.union1d(
        *[factor_dep[i].schedule[:, utils.RESET_INDEX_Reset_Day] for i in ['Samples_1', 'Samples_2']])
    start_idx = np.searchsorted(sample_reset_days, deal_time[:, utils.TIME_GRID_MTM], side='right').astype(np.int64)
    # set the unique index
    start_index, counts = np.unique(start_idx, return_counts=True)
    # now precalc all past resets

    all_samples = []
    dual_samples = []
    start_samples = []

    for i in ['Samples_1', 'Samples_2']:
        samples = factor_dep[i].reinitialize(shared.one)
        sample_idx = samples.get_start_index(deal_time, offset=1)
        known_resets = samples.known_resets(shared.simulation_batch)
        sim_samples = samples.schedule[(samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                       (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                        deal_time[:, utils.TIME_GRID_MTM].max())]

        # check if the spot was simulated - if not, hold it flat
        if spot.shape != forward.shape:
            past_samples = spot.expand(sim_samples.shape[0], shared.simulation_batch)
            spot = spot.expand(*forward.shape)
        else:
            past_sample_factor = [utils.calc_time_grid_spot_rate(
                past_factor, sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
                for past_factor in past_factor_list]
            past_samples = past_sample_factor[0] if len(
                past_sample_factor) == 1 else past_sample_factor[0] / past_sample_factor[1]

        full_sample = torch.cat(
            [torch.cat(known_resets, dim=0), past_samples], dim=0) if known_resets else past_samples
        # make sure we can access the numpy and tensor components
        dual_sample = samples.dual()
        dual_samples.append(dual_sample)
        # store the sample with the weights applied
        all_samples.append(full_sample * dual_sample.tn[:, utils.RESET_INDEX_Weight].reshape(-1, 1))
        # record the index of this sample relative the merged resets calculated earlier
        start_samples.append({x: y for x, y in zip(start_idx, sample_idx)})

    for index, (discount_block, spot_block, forward_block, carry_block) in enumerate(
            utils.split_counts([discount, spot, forward, b], counts, shared)):
        t_block = discount_block.time_grid
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]

        # only do moment matching for tenors prior to expiry
        if tenor_block.any():
            # use at the money vols
            moneyness_block = (forward_block if use_forwards else spot_block) / spot_block
            moneyness = 1.0 / moneyness_block if invert_moneyness else moneyness_block
            vols = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, daycount_fn(tenor_block), shared)

            mu = []
            sigma = []
            lambdas = []
            sample_fts = []
            sample_tss = []

            for alpha, start_idx, dual_sample, all_sample in zip(alphas, start_samples, dual_samples, all_samples):
                # get the sample at time t
                sample_index_t = start_idx[index]
                sample_ts = carry_block.new(
                    daycount_fn(dual_sample.np[sample_index_t:, utils.RESET_INDEX_End_Day].reshape(1, -1) -
                                t_block[:, utils.TIME_GRID_MTM, np.newaxis]))

                weight_t = dual_sample.tn[sample_index_t:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
                sample_ft = weight_t * torch.exp(
                    torch.unsqueeze(carry_block, dim=1) * torch.unsqueeze(sample_ts, dim=2))
                M1 = torch.sum(sample_ft, dim=1)
                # realized average so far
                average = torch.sum(all_sample[:sample_index_t], dim=0)

                product_t = sample_ft * torch.exp(
                    torch.unsqueeze(sample_ts, dim=2) * torch.unsqueeze(vols * vols, dim=1))
                sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
                M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)

                # trick to avoid nans in the gradients
                MM = torch.log(M2) - 2.0 * torch.log(M1)
                MM_ok = MM.clamp(min=1e-6)
                vol_t = torch.sqrt(MM_ok)

                mu.append(M1)
                sigma.append(vol_t)
                sample_fts.append(sample_ft)
                sample_tss.append(sample_ts)
                lambdas.append(alpha * average)

            sum_rho = 0.0
            for i in range(sample_fts[0].shape[1]):
                min_ts = torch.minimum(sample_tss[0][:, i].reshape(-1, 1), sample_tss[1])
                sample_fi = torch.unsqueeze(sample_fts[0][:, i], dim=1)
                sum_rho += sample_fi * sample_fts[1] * torch.exp(
                    torch.unsqueeze(min_ts, dim=2) * torch.unsqueeze(vols * vols, dim=1))

            M_rho = torch.sum(sum_rho, dim=1)
            MM_rho = (torch.log(M_rho) - torch.log(mu[0]) - torch.log(mu[1])) / (sigma[0] * sigma[1])
            K_bar = factor_dep['Alpha_0'] * factor_dep['Strike'] - lambdas[0] + lambdas[1]

            theo_price = factor_dep['Buy_Sell'] * utils.Bjerksund_Stensland(
                factor_dep['Option_Type'], -factor_dep['Option_Type'],
                -factor_dep['Option_Type'] * K_bar, alphas[0] * spot_block * mu[0], alphas[1] * spot_block * mu[1],
                K_bar, sigma[0], sigma[1], MM_rho, factor_dep['Option_Type'])
        else:
            lambdas = [alpha * all_sample.sum(axis=0) for alpha, all_sample in zip(alphas, all_samples)]
            K_bar = factor_dep['Alpha_0'] * factor_dep['Strike'] - lambdas[0] + lambdas[1]
            theo_price = factor_dep['Buy_Sell'] * smooth_relu(-factor_dep['Option_Type'] * K_bar)

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(discount_block, tenor_block.reshape(-1, 1), shared),
            dim=1)

        cash = nominal * theo_price
        mtm_list.append(cash * discount_rates)

    # potential cashflows
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])
    # mtm in reporting currency
    mtm = torch.cat(mtm_list, dim=0)

    return mtm


def pv_energy_option(shared, time_grid, deal_data, nominal):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # first precalc all past resets
    samples = factor_dep['Cashflow'].get_resets(shared.one)
    known_samples = samples.known_resets(shared.simulation_batch)
    start_idx = samples.get_start_index(deal_time)
    sim_samples = samples.schedule[
        (samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    fx_spot = utils.calc_fx_cross(
        factor_dep['ForwardFX'][0], factor_dep['CashFX'][0], sim_samples, shared)
    fx_rep = utils.calc_fx_cross(
        factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)
    all_samples = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], sim_samples, shared)

    sample_values = all_samples.gather_weighted_curve(
        shared, sim_samples[:, utils.RESET_INDEX_End_Day, np.newaxis] if sim_samples.size else np.zeros((1, 1)),
        multiply_by_time=False) * torch.unsqueeze(fx_spot, dim=1)

    past_samples = torch.squeeze(
        torch.cat([torch.stack(known_samples), sample_values], dim=0)
        if known_samples else sample_values, dim=1)

    forwards = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    # need the tensor and numpy data
    dual_samples = samples.dual()
    start_index, start_counts = np.unique(start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        t_block = discount_block.time_grid
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM].reshape(-1, 1)

        sample_t = dual_samples[start_index[index]:]
        average = torch.sum(
            past_samples[:start_index[index]] *
            dual_samples.tn[:start_index[index], utils.RESET_INDEX_Weight].reshape(-1, 1),
            dim=0)

        if sample_t.np.any():
            sample_ts = np.tile(
                sample_t.np[np.newaxis, :, utils.RESET_INDEX_End_Day],
                [t_block.shape[0], 1])
            weight_t = sample_t.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)

            future_resets = forward_block.gather_weighted_curve(
                shared, sample_ts, multiply_by_time=False)

            forwardfx = utils.calc_fx_forward(
                factor_dep['ForwardFX'], factor_dep['CashFX'],
                sample_t.np[:, utils.RESET_INDEX_Start_Day], t_block, shared)

            sample_ft = weight_t * future_resets * forwardfx

            # needed for vol lookup
            sample_block = daycount_fn(
                sample_t.np[:, utils.RESET_INDEX_Start_Day].reshape(1, -1)
                - t_block[:, utils.TIME_GRID_MTM, np.newaxis])

            delivery_block = daycount_fn(
                sample_t.np[:, utils.RESET_INDEX_End_Day].reshape(1, -1) - factor_dep['Basedate']
                - t_block[:, utils.TIME_GRID_MTM, np.newaxis]
            )

            M1 = torch.sum(sample_ft, dim=1)
            strike_bar = factor_dep['Strike'] - average
            moneyness = M1 / strike_bar
            ref_vols = utils.calc_delivery_time_grid_vol_rate(
                factor_dep['ReferenceVol'], moneyness, sample_block,
                delivery_block, t_block, shared)

            vol2 = ref_vols * ref_vols

            # Note - need to allow for compo deals
            if 'FXCompoVol' in factor_dep:
                fx_vols = torch.stack(
                    [utils.calc_time_grid_vol_rate(factor_dep['FXCompoVol'], ones.reshape(-1, 1), sb, shared)
                     for ones, sb in zip(vol2.new_ones(*sample_block.shape), sample_block)])
                vol2 += fx_vols * fx_vols + 2.0 * fx_vols * ref_vols * factor_dep['ImpliedCorrelation']

            product_t = sample_ft * torch.exp(
                sample_ft.new(np.expand_dims(sample_block, axis=2)) * vol2)

            # do an exclusive cumsum on axis=1
            sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
            M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)
            MM = torch.log(M2) - 2.0 * torch.log(M1)
            # trick to allow the gradients to be defined
            MM_ok = MM.clamp(min=1e-5)
            vol_t = torch.sqrt(MM_ok)
            theo_price = utils.black_european_option(
                M1, strike_bar, vol_t, 1.0,
                factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared)
        else:
            forward_p = average.reshape(1, -1)
            theo_price = factor_dep['Buy_Sell'] * smooth_relu(
                factor_dep['Option_Type'] * (forward_p - factor_dep['Strike']))

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(discount_block, tenor_block, shared), dim=1)

        cash = nominal * theo_price
        mtm_list.append(cash * discount_rates)

    # potential cashflows
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])
    # mtm in reporting currency
    mtm = fx_rep * torch.cat(mtm_list, dim=0)

    return mtm


def pricer_float_cashflows(all_resets, t_cash, shared):
    margin = (t_cash[:, utils.CASHFLOW_INDEX_FloatMargin] * t_cash[:, utils.CASHFLOW_INDEX_Year_Frac])
    all_int = all_resets * t_cash[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) + margin.reshape(1, -1, 1)

    return all_int, margin


def _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, call_or_put, shared):
    mn_option = all_resets - t_cash[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1)
    expiry = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](expiries)
    digital_payoff = factor_dep['Digital_Payoff_Rate'] if 'Digital_Payoff_Rate' in factor_dep else 0.0

    vols = utils.calc_tenor_cap_time_grid_vol_rate(
        factor_dep['VolSurface'], mn_option, expiry, tenor, shared)
    dist, shf = factor_dep['VolSurface'][0][utils.FACTOR_INDEX_SubType]
    pricing_fn = utils.black_european_option if dist=='Lognormal' else utils.bachelier_european_option
    payoff = pricing_fn(
        all_resets, t_cash[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1),
        vols, expiry, 1.0, call_or_put, shared, cash_payoff=digital_payoff, shift=shf.amount)

    all_int = t_cash[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) * payoff
    margin = shared.one.new_zeros(len(t_cash))

    return all_int, margin


def pricer_cap(all_resets, t_cash, factor_dep, expiries, tenor, shared):
    return _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, 1.0, shared)


def pricer_floor(all_resets, t_cash, factor_dep, expiries, tenor, shared):
    return _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, -1.0, shared)


def pv_float_cashflow_list(shared: utils.Calculation_State, time_grid: utils.TimeGrid, deal_data: utils.DealDataType,
                           cashflow_pricer, mtm_currency=None, settle_cash=True):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    # first precalc all past resets
    resets = factor_dep['Cashflows'].get_resets(shared.one)

    if mtm_currency:
        # precalc the FX forwards
        sim_fx_forward = utils.calc_fx_forward(
            mtm_currency, factor_dep['Currency'], factor_dep['Cashflows'].FXResets[:, utils.RESET_INDEX_Reset_Day],
            factor_dep['Cashflows'].FXResets[:, :utils.RESET_INDEX_Scenario + 1], shared, only_diag=True)

        known_fx = factor_dep['Cashflows'].known_fx_resets(shared.simulation_batch)

        # fetch fx rates - note that there is a slight difference between this and the spot fx rate
        old_fx_rates = (torch.cat([torch.stack(known_fx), sim_fx_forward], dim=0)
                        if known_fx else sim_fx_forward).squeeze(dim=1)

    forwards = utils.calc_time_grid_curve_rate(factor_dep['Forward'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    old_resets = resets.get_simulated_resets(
        deal_time[:, utils.TIME_GRID_MTM].max(), factor_dep['Forward'], shared)

    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        # cashflows is a dual representation
        cashflows = factor_dep['Cashflows'].merged(shared.one, start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

        reset_offset = factor_dep['Cashflows'].offsets[start_index[index]][1]
        pmts_offset = cash_index + (cash_counts - 1)

        time_ofs = 0
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        reset_block = resets.dual(reset_offset)
        reset_ofs, reset_count = np.unique(resets.split_block_resets(
            reset_offset, time_block), return_counts=True)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        # do we need to split the forward block further?
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        # empty list for payments
        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            time_slice = time_block[time_ofs:time_ofs + size].reshape(-1, 1)
            future_starts = reset_block.np[offset:, utils.RESET_INDEX_Start_Day] - time_slice
            future_ends = reset_block.np[offset:, utils.RESET_INDEX_End_Day] - time_slice
            future_weights = (reset_block.tn[offset:, utils.RESET_INDEX_Weight]
                              / reset_block.tn[offset:, utils.RESET_INDEX_Accrual]).reshape(1, -1, 1)
            future_resets = torch.expm1(forward_rates.gather_weighted_curve(
                shared, future_ends, future_starts)) * future_weights

            # now deal with past resets
            all_resets = torch.cat(
                [old_resets[reset_offset:reset_offset + offset].expand(size, -1, -1), future_resets], dim=1)

            # handle cashflows that don't pay interest (e.g. bullets)
            if cashflows.np[:, utils.CASHFLOW_INDEX_NumResets].all():
                reset_cashflows = cashflows
            else:
                reset_cash_index = np.where(cashflows.np[:, utils.CASHFLOW_INDEX_NumResets])[0]
                reset_cashflows = cashflows[reset_cash_index]
                cash_counts *= (cashflows.np[:, utils.CASHFLOW_INDEX_NumResets] >= 1).astype(np.int32)
                cash_index = reset_cash_index.searchsorted(cash_index)

            if mtm_currency:
                fx_past_index = start_index[index]
                if cashflows.np[:, utils.CASHFLOW_INDEX_FXResetDate].min() < time_slice.max():
                    # get the past fx rates - only works if not forward starting
                    fx_end_index = fx_past_index+1
                else:
                    # forward starting
                    fx_end_index = fx_past_index

                past_fx_resets = old_fx_rates[fx_past_index: fx_end_index].expand(size, -1, -1)

                # now deal with fx rates
                future_fx_resets = utils.calc_fx_forward(
                    mtm_currency, factor_dep['Currency'],
                    cashflows.np[past_fx_resets.shape[1]:, utils.CASHFLOW_INDEX_FXResetDate],
                    discount_block.time_grid[time_ofs:time_ofs + size], shared)

                all_fx_resets = torch.cat([past_fx_resets, future_fx_resets], dim=1)

                # calculate the Nominal in the correct currency
                Pi = all_fx_resets * cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)
                Pi_1 = F.pad(Pi[:, 1:, :], [0, 0, 0, 1, 0, 0])

            time_ofs += size

            if all_resets.shape[1] != reset_cashflows.np.shape[0]:
                # check if the resets need to be averaged (compounded) before being applied (i.e. OIS)
                reset_per_cashflows = factor_dep['Cashflows'].offsets[start_index[index]:, 0]
                reset_split = tuple(reset_per_cashflows[reset_per_cashflows > 0])
                accrual = reset_block.tn[:, utils.RESET_INDEX_Accrual]  # should align with all_resets dim=1
                all_resets = torch.stack(
                    [((torch.expm1(torch.log1p(r * b.reshape(1, -1, 1)).sum(dim=1))) / b.sum())
                     for r, b in zip(all_resets.split(reset_split, 1), accrual.split(reset_split))])

            # check if we need extra information to price caps or floors
            if cashflow_pricer in [pricer_cap, pricer_floor]:
                expiries = cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day] - time_slice
                # note that the tenor (Year Frac) is averaged
                # all the cashflows are supposed to have the same year frac
                # (but practically not - should be ok to do this)
                tenor = cashflows.np[:, utils.CASHFLOW_INDEX_Year_Frac].mean()
                all_int, all_margin = cashflow_pricer(
                    all_resets, reset_cashflows.tn, factor_dep, expiries, tenor, shared)
            else:
                all_int, all_margin = cashflow_pricer(all_resets, reset_cashflows.tn, shared)

            # handle the common case of no compounding or OIS compounding
            if mtm_currency is None and factor_dep['CompoundingMethod'] in ['None', 'OIS']:
                interest = all_int * reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)
                if (cash_counts == 1).all():
                    total = interest
                else:
                    split_interest = torch.split(interest, tuple(cash_counts), dim=1)
                    total = torch.stack([i.sum(dim=1) for i in split_interest], dim=1)
            else:
                # check if there are a different number of resets per cashflow
                if cash_counts.min() != cash_counts.max():
                    interest = F.pad(all_int, [0, 0, 0, 1, 0, 0])
                    # nom = torch.ones_like(reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal])
                    # nominal = F.pad(nom, [0, 1])
                    nominal = F.pad(reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal], [0, 1])
                    margin = F.pad(all_margin, [0, 1])
                else:
                    interest = all_int
                    margin = all_margin
                    nominal = reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal]

                default_offst = np.ones(cash_index.size, dtype=np.int32) * (interest.shape[1] - 1)
                total = 0.0

                for i in range(cash_counts.max()):
                    offst = default_offst.copy()
                    offst[cash_counts > i] = i + cash_index[cash_counts > i]
                    int_i = interest[:, offst]

                    if mtm_currency:
                        total = total + int_i * Pi + (Pi - Pi_1)
                    elif factor_dep['CompoundingMethod'] == 'Include_Margin':
                        total = total + int_i * (total + nominal[offst].reshape(1, -1, 1))
                    elif factor_dep['CompoundingMethod'] == 'Flat':
                        total = total + (int_i * nominal[offst].reshape(1, -1, 1)) + total * (
                                int_i - margin[offst].reshape(1, -1, 1))
                    elif factor_dep['CompoundingMethod'] == 'Exclude_Margin':
                        total = total + (int_i * nominal[offst].reshape(1, -1, 1)) + total * (
                                int_i - margin[offst].reshape(1, -1, 1))
                    else:
                        raise Exception('Floating cashflow list method not implemented')

            payments.append(total + cashflows.tn[
                pmts_offset, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1))

        # now finish the payments
        all_payments = torch.cat(payments, dim=0) if len(payments) > 1 else payments[0]

        # settle any cashflows
        if settle_cash:
            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), all_payments[-1][0])
        # add it to the list
        mtm_list.append(torch.sum(all_payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_fixed_cashflows(shared, time_grid, deal_data, ignore_fixed_rate=False, settle_cash=True):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
    settlement_amt = factor_dep.get('Settlement_Amount', 0.0)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    start_index, counts = np.unique(cash_start_idx, return_counts=True)

    for index, [discount_block] in enumerate(utils.split_counts([discounts], counts, shared)):
        cashflows = factor_dep['Cashflows'].merged(shared.one, start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)
        # payment times            
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)
        # is this a forward?
        if settlement_amt:
            settlement = settlement_amt * torch.squeeze(utils.calc_discount_rate(discount_block, (
                    factor_dep['Settlement_Date'] - time_block).reshape(-1, 1), shared), dim=1)
        else:
            settlement = 0.0

        # empty list for payments
        all_int = (1.0 if ignore_fixed_rate else cashflows.tn[:, utils.CASHFLOW_INDEX_FixedRate]
                   ) * cashflows.tn[:, utils.CASHFLOW_INDEX_Year_Frac]

        # use this to cache the payments for this call - note that we need to include ignore_fixed_rate
        payment_key = ('Payments', index, ignore_fixed_rate)

        if factor_dep.get(payment_key) is None:
            payments = 0.0

            if cash_counts.min() != cash_counts.max():
                interest = F.pad(all_int, [0, 1])
                nominal = F.pad(cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal], [0, 1])
                fixed_amt = F.pad(cashflows.tn[:, utils.CASHFLOW_INDEX_FixedAmt], [0, 1])
            else:
                interest = all_int
                nominal = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal]
                fixed_amt = cashflows.tn[:, utils.CASHFLOW_INDEX_FixedAmt]

            default_offst = np.ones(cash_index.size, dtype=np.int32) * (len(interest) - 1)

            for i in range(cash_counts.max()):
                offst = default_offst.copy()
                offst[cash_counts > i] = i + cash_index[cash_counts > i]
                int_i = interest[offst]

                if factor_dep.get('Compounding', False):
                    payments += (payments + nominal[offst]) * int_i + fixed_amt[offst]
                else:
                    payments += int_i * nominal[offst] + fixed_amt[offst]

            # cache this payment tensor
            factor_dep[payment_key] = payments

        # add to the mtm
        mtm_list.append(torch.sum(
            discount_rates * factor_dep[payment_key].reshape(1, -1, 1), dim=1) - settlement)

        # settle any cashflows
        if settle_cash:
            if factor_dep.get('Settlement_Date') is not None:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, factor_dep['Settlement_Date']),
                            mtm_list[-1][-1])
            else:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]),
                            factor_dep[payment_key][0])

    return torch.cat(mtm_list, dim=0)


def pv_index_cashflows(shared, time_grid, deal_data, settle_cash=True):
    def calc_index(schedule, sim_schedule):
        weight = schedule.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
        dates = schedule.np[np.newaxis, :, utils.RESET_INDEX_Reset_Day] - \
                last_pub_block[:, np.newaxis, utils.RESET_INDEX_Reset_Day]
        index_t = torch.unsqueeze(last_index_block, dim=1) / utils.calc_discount_rate(
            forecast_block, dates, shared)

        # split if necessary
        if dates[dates < 0].any():
            future_indices = (dates >= 0).all(axis=1).argmin()
            future_index_t, past_index_t = torch.split(
                index_t, (future_indices, dates.shape[0] - future_indices))

            mixed_indices_t = []
            for mixed_dates, mixed_indices in zip(dates[future_indices:], past_index_t):
                future_resets = mixed_dates.size - (mixed_dates[::-1] > 0).argmin()
                past_resets_t, future_resets_t = torch.split(
                    mixed_indices, (future_resets, mixed_dates.size - future_resets))
                mixed_indices_t.append(
                    torch.cat([sim_schedule[:future_resets], future_resets_t], dim=0))

            values = weight * torch.cat([future_index_t, torch.stack(mixed_indices_t)], dim=0)
        else:
            values = weight * index_t

        if resets_per_cf > 1:
            return torch.sum(values.reshape(
                last_pub_block.shape[0], -1, resets_per_cf, shared.simulation_batch), dim=2)
        else:
            return values

    def get_index_val(cash_index_vals, schedule, sim_schedule, resets_per_cf, offset):
        if (cash_index_vals.np < 0).any():
            num_known = cash_index_vals.np[cash_index_vals.np > 0].size
            reset_offset = resets_per_cf * (offset + num_known)
            if num_known:
                known_indices = cash_index_vals.tn[cash_index_vals.np < 0].reshape(
                    1, -1, 1).expand(last_pub_block.shape[0], -1, shared.simulation_batch)
                return torch.cat([known_indices, calc_index(
                    schedule[reset_offset:], sim_schedule[reset_offset:])], dim=1)
            else:
                return calc_index(schedule[reset_offset:], sim_schedule[reset_offset:])
        else:
            return cash_index_vals.tn.reshape(1, -1, 1)

    def filter_resets(resets, index):
        known_resets = resets.known_resets(shared.simulation_batch)
        sim_resets = resets.schedule[(resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                     (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                      deal_time[:, utils.TIME_GRID_MTM].max())]
        old_resets = utils.calc_time_grid_spot_rate(index, sim_resets[:, :utils.RESET_INDEX_Scenario + 1], shared)
        return torch.cat([torch.cat(known_resets, dim=0), old_resets], dim=0) if known_resets else old_resets

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    resets_per_cf = 2 if (factor_dep['IndexMethod'] in [
        utils.CASHFLOW_METHOD_IndexReferenceInterpolated3M,
        utils.CASHFLOW_METHOD_IndexReferenceInterpolated4M]) else 1

    last_published = factor_dep['Cashflows'].Resets.schedule[deal_data.Time_dep.deal_time_grid]
    last_published_index = utils.calc_time_grid_spot_rate(factor_dep['PriceIndex'], deal_time, shared)
    index_forecast = utils.calc_time_grid_curve_rate(factor_dep['ForecastIndex'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    base_resets = factor_dep['Base_Resets'].reinitialize(shared.one)
    final_resets = factor_dep['Final_Resets'].reinitialize(shared.one)

    all_base_resets = filter_resets(base_resets, factor_dep['PriceIndex'])
    all_final_resets = filter_resets(final_resets, factor_dep['PriceIndex'])

    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)
    last_pub_blocks = np.split(last_published, start_counts.cumsum())

    for index, (forecast_block, discount_block, last_index_block) in enumerate(
            utils.split_counts([index_forecast, discounts, last_published_index], start_counts, shared)):

        last_pub_block = last_pub_blocks[index]
        # cashflows is a dual representation (numpy and tensor) of the same cashflows
        cashflows = factor_dep['Cashflows'].merged(shared.one, start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

        # payment times            
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        all_base_index_vals = get_index_val(
            cashflows[:, utils.CASHFLOW_INDEX_BaseReference], base_resets.dual(),
            all_base_resets, resets_per_cf, start_index[index])
        all_final_index_vals = get_index_val(
            cashflows[:, utils.CASHFLOW_INDEX_FinalReference], final_resets.dual(),
            all_final_resets, resets_per_cf, start_index[index])

        # empty list for payments
        interest = (cashflows.tn[:, utils.CASHFLOW_INDEX_FixedRate] *
                    cashflows.tn[:, utils.CASHFLOW_INDEX_Year_Frac]).reshape(1, -1, 1)
        growth = cashflows.tn[:, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1) * (
                all_final_index_vals / all_base_index_vals)
        payment_all = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1) * growth * interest

        # reduce if any counts are duplicated
        if (cash_counts == 1).all():
            payments = payment_all
        else:
            payments = torch.stack([payment.sum(dim=1) for payment in torch.split(
                payment_all, tuple(cash_counts), dim=1)], dim=1)

        # add it to the list
        mtm_list.append(torch.sum(payments * discount_rates, dim=1))

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

    return torch.cat(mtm_list, dim=0)


def pv_energy_cashflows(shared, time_grid, deal_data):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    # first precalc all past resets
    resets = factor_dep['Cashflows'].get_resets(shared.one)
    known_resets = resets.known_resets(shared.simulation_batch)
    sim_resets = resets.schedule[(resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                 (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                  deal_time[:, utils.TIME_GRID_MTM].max())]
    all_resets = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], sim_resets, shared)
    all_fx_spot = utils.calc_fx_cross(factor_dep['ForwardFX'][0], factor_dep['CashFX'][0],
                                      sim_resets, shared)

    reset_values = torch.unsqueeze(
        torch.squeeze(all_resets.gather_weighted_curve(
            shared, sim_resets[:, utils.RESET_INDEX_End_Day].reshape(-1, 1), multiply_by_time=False),
            dim=1) * all_fx_spot, dim=1) \
        if sim_resets.any() else shared.fillvalue

    old_resets = torch.squeeze(
        torch.cat([torch.stack(known_resets), reset_values], dim=0) if known_resets else reset_values, dim=1)

    forwards = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        # cashflows = factor_dep['Cashflows'].schedule[start_index[index]:]
        cashflows = factor_dep['Cashflows'].merged(shared.one, start_index[index])
        # cash_offset = factor_dep['Cashflows'].offsets[start_index[index]:]

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

        reset_offset = int(cashflows.np[0, utils.CASHFLOW_INDEX_ResetOffset])

        time_ofs = 0
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        reset_block = resets.dual(reset_offset)
        reset_ofs, reset_count = np.unique(
            resets.split_block_resets(reset_offset, time_block), return_counts=True)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        # we need to split the forward block further
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        # empty list for payments
        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            # past resets
            past_resets = torch.unsqueeze(
                old_resets[reset_offset:reset_offset + offset], dim=0).expand(size, -1, -1)

            # future resets
            future_ends = np.tile(reset_block.np[offset:, utils.RESET_INDEX_End_Day], [size, 1])

            if future_ends.any():
                future_resets = forward_rates.gather_weighted_curve(
                    shared, future_ends, multiply_by_time=False)

                forwardfx = utils.calc_fx_forward(
                    factor_dep['ForwardFX'], factor_dep['CashFX'],
                    reset_block.np[offset:, utils.RESET_INDEX_Start_Day],
                    discounts.time_grid[time_ofs:time_ofs + size], shared)

                all_resets = torch.cat([past_resets, future_resets * forwardfx], dim=1)
            else:
                all_resets = past_resets

            time_ofs += size

            all_payoffs = all_resets * reset_block.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
            split_payoffs = tuple(cashflows.np[:, utils.CASHFLOW_INDEX_NumResets].astype(np.int32))
            payoff = torch.stack([torch.sum(x, dim=1) for x in torch.split(
                all_payoffs, split_payoffs, dim=1)], dim=1)

            # now we can price the cashflows
            payment = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal] * (
                    cashflows.tn[:, utils.CASHFLOW_INDEX_Start_Mult] * payoff +
                    cashflows.tn[:, utils.CASHFLOW_INDEX_FloatMargin])

            payments.append(payment)

        # now finish the payments
        all_payments = torch.cat(payments, dim=0)

        # settle any cashflows
        cash_settle(shared, factor_dep['SettleCurrency'],
                    np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), all_payments[-1][0])
        # add it to the list
        mtm_list.append(torch.sum(all_payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_credit_cashflows(shared, time_grid, deal_data):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    surv = utils.calc_time_grid_curve_rate(factor_dep['Name'], deal_time, shared)
    start_index, counts = np.unique(cash_start_idx, return_counts=True)

    for index, (discount_block, surv_block) in enumerate(
            utils.split_counts([discounts, surv], counts, shared)):
        # get the duel cashflow at the correct index
        cashflows = factor_dep['Cashflows'].merged(shared.one, start_index[index])
        cash_pmts, cash_index = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True)
        cash_sts = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day])

        # payment times            
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        start_pmts = (cash_sts.reshape(1, -1) - time_block.reshape(-1, 1))

        Dt_T = utils.calc_discount_rate(discount_block, future_pmts, shared)
        Dt_Tm1 = utils.calc_discount_rate(discount_block, start_pmts.clip(min=0), shared)

        survival_T = utils.calc_discount_rate(surv_block, future_pmts, shared, multiply_by_time=False)
        survival_t = utils.calc_discount_rate(surv_block, start_pmts.clip(min=0), shared, multiply_by_time=False)

        interest = (cashflows.tn[cash_index, utils.CASHFLOW_INDEX_FixedRate] *
                    cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Year_Frac])

        marginal_PD = survival_t - survival_T
        if factor_dep['Accrue_Fee']:
            past_accrual = cashflows.tn.new(
                daycount_fn(-start_pmts.clip(max=0))) / cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Year_Frac]
            adjustment = (past_accrual + 0.5 * (1-past_accrual)).unsqueeze(dim=2)
        else:
            adjustment = 0.0

        # note the minus sign here
        premium = -(interest[cash_index] * cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Nominal]).reshape(1, -1, 1)
        pv_premium = premium * (survival_T + adjustment * marginal_PD) * Dt_T
        pv_credit = (1.0 - factor_dep['Recovery_Rate']) * cashflows.tn[
            cash_index, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1) * 0.5 * (
                            Dt_T + Dt_Tm1) * marginal_PD

        # settle any cashflows
        cash_settle(shared, factor_dep['SettleCurrency'],
                    np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), premium[0, 0, 0])

        mtm_list.append(torch.sum(pv_credit + pv_premium, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_equity_cashflows(shared, time_grid, deal_data):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    eq_spot = utils.calc_time_grid_spot_rate(factor_dep['Equity'], deal_time, shared)
    cash = factor_dep['Flows']

    # needed for grouping 
    cash_start_idx = np.searchsorted(
        cash.schedule[:, utils.CASHFLOW_INDEX_Start_Day], deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_end_idx = np.searchsorted(
        cash.schedule[:, utils.CASHFLOW_INDEX_End_Day], deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_pay_idx = cash.get_cashflow_start_index(deal_time)

    # first precalc all past resets
    all_samples = []

    for samples in cash.get_resets(shared.one).split_groups(2):
        known_sample = samples.known_resets(shared.simulation_batch, include_today=True)
        sim_samples = samples.schedule[
            (samples.schedule[:, utils.RESET_INDEX_Value] == 0.0) &
            (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]

        past_samples = utils.calc_time_grid_spot_rate(
            factor_dep['Equity'], sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)

        # fetch all fixed resets
        if past_samples.shape[1] != shared.simulation_batch:
            past_samples = past_samples.expand(sim_samples.shape[0], shared.simulation_batch)

        all_samples.append(torch.cat(
            [torch.cat(known_sample, dim=0), past_samples], dim=0) if known_sample else past_samples)

    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    repo_discounts = utils.calc_time_grid_curve_rate(factor_dep['Equity_Zero'], deal_time, shared)

    # cashflows = cash.schedule
    cashflows = cash.merged(shared.one)
    # all_index, all_counts = zip(*all_idx.items())
    all_index, all_counts = np.unique(list(
        zip(cash_start_idx, cash_end_idx, cash_pay_idx)), axis=0, return_counts=True)

    for index, (discount_block, repo_block, eq_block) in enumerate(
            utils.split_counts([discounts, repo_discounts, eq_spot], all_counts, shared)):

        start_idx, end_idx, pay_idx = all_index[index]

        cashflow_start = cashflows.np[start_idx:, utils.CASHFLOW_INDEX_Start_Day].reshape(1, -1)
        cashflow_pay = cashflows.np[pay_idx:, utils.CASHFLOW_INDEX_Pay_Day].reshape(1, -1)

        payoffs = []
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cashflow_pay - time_block.reshape(-1, 1)
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        # need equity forwards for start and end cashflows
        if pay_idx < end_idx:
            St0 = torch.unsqueeze(all_samples[0][pay_idx:end_idx], 0)
            St1 = torch.unsqueeze(all_samples[1][pay_idx:end_idx], 0)

            Ht0_t1 = utils.calc_realized_dividends(
                St0, factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                utils.calc_dividend_samples(
                    cashflows.np[pay_idx:end_idx, utils.CASHFLOW_INDEX_Start_Day],
                    cashflows.np[end_idx - 1:end_idx, utils.CASHFLOW_INDEX_End_Day], time_grid), shared)

            units = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1)
            end_mult = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_End_Mult].reshape(1, -1, 1)
            div_mult = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1)
            start_mult = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1)
            payoff = (end_mult * St1 - start_mult * St0 + div_mult * Ht0_t1)

            payment = payoff * units

            if factor_dep['PrincipleNotShares']:
                payment /= St0

            payoffs.append(payment.expand(time_block.size, -1, -1) if time_block.size > 1 else payment)

            # settle cashflow if necessary
            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cashflow_pay[0][0]),
                        torch.sum(payment, dim=1)[0])

        if end_idx < start_idx:
            cashflow_end = cashflows.np[end_idx, utils.CASHFLOW_INDEX_End_Day].reshape(1, -1)
            future_end = cashflow_end - time_block.reshape(-1, 1)
            forward_end = utils.calc_eq_forward(
                factor_dep['Equity'], factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                np.squeeze(cashflow_end, axis=0), discount_block.time_grid, shared)

            discount_end = utils.calc_discount_rate(repo_block, future_end, shared)

            St0 = torch.unsqueeze(all_samples[0][end_idx:start_idx], dim=0)
            Ht0_t = utils.calc_realized_dividends(
                St0, factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                utils.calc_dividend_samples(
                    cashflows.np[end_idx:start_idx, utils.CASHFLOW_INDEX_Start_Day], time_block, time_grid), shared)

            units = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1)
            end_mult = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_End_Mult].reshape(1, -1, 1)
            div_mult = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1)
            start_mult = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1)

            payoff = (end_mult - div_mult) * forward_end + div_mult * (
                    torch.unsqueeze(eq_block, dim=1) + Ht0_t) / discount_end - start_mult * St0

            if factor_dep['PrincipleNotShares']:
                payoff /= St0

            payoffs.append(payoff * units)

        if cashflow_start.any():
            cashflow_end = cashflows.np[start_idx:, utils.CASHFLOW_INDEX_End_Day].reshape(1, -1)
            future_start = cashflow_start - time_block.reshape(-1, 1)
            future_end = cashflow_end - time_block.reshape(-1, 1)
            forward_start = utils.calc_eq_forward(
                factor_dep['Equity'], factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                np.squeeze(cashflow_start, axis=0), discount_block.time_grid, shared)
            forward_end = utils.calc_eq_forward(
                factor_dep['Equity'], factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                np.squeeze(cashflow_end, axis=0), discount_block.time_grid, shared)

            discount_start = utils.calc_discount_rate(repo_block, future_start, shared)
            discount_end = utils.calc_discount_rate(repo_block, future_end, shared)

            if factor_dep['PrincipleNotShares']:
                factor1 = forward_end / forward_start
                factor2 = 1.0
            else:
                factor1 = forward_end
                factor2 = forward_start

            units = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1)
            end_mult = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_End_Mult].reshape(1, -1, 1)
            div_mult = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1)
            start_mult = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1)

            payoff = (end_mult - div_mult) * factor1 + (
                    div_mult * (discount_start / discount_end) - start_mult) * factor2

            payoffs.append(payoff * units)

        # now finish the payments
        payments = torch.cat(payoffs, dim=1) if len(payoffs) > 1 else payoffs[0]
        mtm_list.append(torch.sum(payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


@utils.log_exception
def pv_fixed_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0], shared.Report_Currency,
                                 deal_time, shared)

    mtm = pv_fixed_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_energy_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency,
                                 deal_time, shared)

    mtm = pv_energy_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_float_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0], shared.Report_Currency,
                                 deal_time, shared)
    model = deal_data.Factor_dep.get('Model', pricer_float_cashflows)
    mtm = pv_float_cashflow_list(shared, time_grid, deal_data, model) * FX_rep

    return mtm


@utils.log_exception
def pv_index_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                 deal_time, shared)

    mtm = pv_index_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_cds_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                 deal_time, shared)
    mtm = pv_credit_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_equity_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                 deal_time, shared)

    mtm = pv_equity_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm
