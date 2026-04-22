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
import itertools
import pandas as pd
import numpy as np
import torch
from functools import reduce
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# load up some useful data types
from collections import namedtuple, defaultdict
# import the risk factors (also known as price factors)
from .riskfactors import construct_factor
# import the stochastic processes
from .stochasticprocess import construct_process
# import the currency/curve lookup factors 
from .instruments import (get_fxrate_factor, get_recovery_rate, get_interest_factor, get_survival_factor)
# import the hessian function
from .pricing import SensitivitiesEstimator
# import the documentation and utils modules
from . import utils, pricing, construct_instrument
from .hedge_runtime import construct_torchrl_runtime
from .torchrl_hedge import run_torchrl_execution


class Aggregation(object):
    '''Container class that represents the base Instrument for aggregation'''
    def __init__(self, name):
        self.reval_dates = None
        self.field = {'Reference': name}
        self.accum_dependencies = True

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets,
        all_factors, all_tenors, time_grid, calendars):
        pass

    def get_report_dates(self):
        return self.reval_dates

    def set_report_dates(self, reval_dates):
        self.reval_dates = reval_dates

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        shared.save_results(deal_data.Calc_res, {'Value': accum})
        return accum

class DealStructure(object):
    def __init__(self, obj, store_results=False):
        self.obj = utils.DealDataType(
            Instrument=obj, Factor_dep={}, Time_dep=None, Calc_res={} if store_results else None)
        # gather a list of deal dependencies
        self.dependencies = []
        # maintain a list of container objects
        self.sub_structures = []
        # Do we want to store each deal level MTM explicitly?
        self.store_results = store_results

    @staticmethod
    def calc_time_dependency(base_date, deal, time_grid):
        # calculate the additional (dynamic) dates that this instrument needs to be evaluated at
        deal_time_dep = None
        try:
            reval_dates = deal.get_reval_dates(clip_expiry=True)
            if len(time_grid.scenario_dates) == 1:
                if len(reval_dates) > 0 and max(reval_dates) < base_date:
                    raise utils.InstrumentExpired(deal.field.get('Reference', 'Unknown Instrument Reference'))
                deal_time_dep = time_grid.calc_deal_grid({base_date})
            else:
                deal_time_dep = time_grid.calc_deal_grid(reval_dates)
        except utils.InstrumentExpired as e:
            logging.warning('skipping expired deal {0}'.format(e.args))

        return deal_time_dep

    def add_deal_to_structure(self, base_date, deal, static_offsets, stochastic_offsets,
                              all_factors, all_tenors, time_grid, calendars, stats):
        """
        The logic is as follows: a structure contains deals - a set of deals are netted off and then the rules that the
        structure itself contains is applied.
        """
        deal_time_dep = self.calc_time_dependency(base_date, deal, time_grid)
        if deal_time_dep is not None:
            # calculate dependencies based on field names
            try:
                self.dependencies.append(
                    utils.DealDataType(Instrument=deal,
                                       Factor_dep=deal.calc_dependencies(
                                           base_date, static_offsets, stochastic_offsets,
                                           all_factors, all_tenors, time_grid, calendars),
                                       Time_dep=deal_time_dep,
                                       Calc_res={} if self.store_results else None))
                stats['Deals loaded'] = stats.setdefault('Deals loaded', 0) + 1
            except Exception as e:
                logging.error('{0} {1} - Skipped'.format(deal.field['Object'], e.args))
                stats['Deals Skipped'] = stats.setdefault('Deals Skipped', 0) + 1

    def finalize_struct(self, base_date, time_grid):
        all_report_dates = [set(
            x.obj.Instrument.get_report_dates(time_grid, base_date)) for x in self.sub_structures]
        self.obj.Instrument.set_report_dates(
            reduce(set.union, all_report_dates) if all_report_dates else time_grid.mtm_dates)
        # copy across the reporting dates to the time_grid
        time_grid.set_report_dates(base_date, self.obj.Instrument.get_report_dates())

    def add_structure_to_structure(self, struct, base_date, static_offsets, stochastic_offsets,
                                   all_factors, all_tenors, time_grid, calendars, stats):
        # get the dependencies
        struct_time_dep = self.calc_time_dependency(base_date, struct.obj.Instrument, time_grid)
        if struct_time_dep is not None:
            try:
                struct.obj = utils.DealDataType(
                    Instrument=struct.obj.Instrument,
                    Factor_dep=struct.obj.Instrument.calc_dependencies(
                        base_date, static_offsets, stochastic_offsets,
                        all_factors, all_tenors, time_grid, calendars),
                    Time_dep=struct_time_dep,
                    Calc_res={} if self.store_results or struct.obj.Instrument.accum_dependencies else None)
                # Structure object representing a netted set of cashflows
                self.sub_structures.append(struct)
                stats['Structs loaded'] = stats.setdefault('Structs loaded', 0) + 1
            except Exception as e:
                logging.error('{0} {1} - Skipped'.format(struct.obj.Instrument.field['Object'], e.args))
                stats['Structs Skipped'] = stats.setdefault('Structs Skipped', 0) + 1

    def resolve_structure(self, shared, time_grid):
        """
        Resolves the Structure
        """

        accum = 0.0 * shared.one

        if self.sub_structures:
            # process sub structures
            for structure in self.sub_structures:
                logging.root.name = structure.obj.Instrument.field.get('Reference', 'root')
                # reset cashflows if this structure accumulates its dependencies (e.g. netting sets)
                if structure.obj.Instrument.accum_dependencies and hasattr(shared, 'reset_cashflows'):
                    shared.reset_cashflows(time_grid)
                struct = structure.resolve_structure(shared, time_grid)
                if (struct != struct).any():
                    logging.critical('Netting set contains NANS! Please Investigate! - skipping for now')
                    continue
                if structure.obj.Instrument.accum_dependencies and hasattr(shared, 'save_cashflows'):
                    shared.save_cashflows(structure.obj.Calc_res, time_grid)
                if structure.obj.Instrument.field.get('Reference', 'root').startswith('FLIP'):
                    logging.warning('Netting set starts with FLIP - inverting MTM')
                    struct = -struct
                accum = accum + struct

        if self.dependencies and self.obj.Instrument.accum_dependencies:
            # accumulate the mtm's
            deal_tensors = 0.0

            for deal_data in self.dependencies:
                logging.root.name = deal_data.Instrument.field.get('Reference', 'root')
                mtm = deal_data.Instrument.calculate(shared, time_grid, deal_data)
                deal_tensors = deal_tensors + mtm

            accum = accum + deal_tensors

        # postprocessing code for working out the mtm of all deals, collateralization etc..
        if hasattr(self.obj.Instrument, 'post_process'):
            # the actual answer for this netting set
            logging.root.name = self.obj.Instrument.field.get('Reference', 'root')
            try:
                accum = self.obj.Instrument.post_process(accum, shared, time_grid, self.obj, self.dependencies)
            except RuntimeError as e:
                logging.error('Runtime error Deal skipped - {}'.format(e.args))
                raise
            except Exception as e:
                logging.critical('Deal skipped - {}'.format(e.args))

        return accum

    def resolve_hedge_structure(self, shared, time_grid):
        """
        Resolves the Structure
        """
        def merge_features(cumulative, new_features):
            cumulative['features'] =  new_features['features'] if cumulative.get('features') is None else torch.concat(
                [cumulative['features'], new_features['features']], 2)
            cumulative['feature_names'] =  cumulative.setdefault('feature_names', () ) + new_features['feature_names']

            for ccy, payoff  in new_features['liability_cashflows'].items():
                cum_ccy = cumulative.setdefault('liability_cashflows', {}).setdefault(ccy, shared.one.new_zeros(1, 1))
                cumulative['liability_cashflows'][ccy] = cum_ccy + payoff
            cumulative['liability_mtm'] = new_features['liability_mtm'] if cumulative.get('liability_mtm') is None else (
                cumulative['liability_mtm'] + new_features['liability_mtm']
            )

        accum = {}

        if self.sub_structures:
            # process sub structures
            for structure in self.sub_structures:
                logging.root.name = structure.obj.Instrument.field.get('Reference', 'root')
                features = structure.resolve_hedge_structure(shared, time_grid)
                merge_features(accum, features)


        if self.dependencies and self.obj.Instrument.accum_dependencies:
            # accumulate the mtm's
            deal_features = {}

            for deal_data in self.dependencies:
                logging.root.name = deal_data.Instrument.field.get('Reference', 'root')
                features = deal_data.Instrument.build_features(shared, time_grid, deal_data)
                merge_features(deal_features, features)

            merge_features(accum, deal_features)

        return accum


class ScenarioTimeGrid(object):
    def __init__(self, cutoff_date, global_time_grid, base_date):
        scen_grid = global_time_grid.scen_time_grid
        offset = scen_grid.searchsorted((cutoff_date - base_date).days) + 1
        self.scen_time_grid = scen_grid[:offset]
        self.time_grid_years = self.scen_time_grid / utils.DAYS_IN_YEAR
        self.scenario_grid = global_time_grid.scenario_grid[:offset]


class Calculation(object):

    def __init__(self, config, prec=torch.float32, device=torch.device('cpu')):
        """
        Construct a new calculation - all calculations must set up their own tensors.
        """

        self.config = config
        self.dtype = prec
        self.time_grid = None
        self.device = device

        # the risk factor data
        self.static_factors = {}
        self.static_var = {}
        self.stoch_factors = {}
        self.stoch_var = {}
        self.all_factors = {}
        self.all_tenors = {}

        self.base_date = None
        self.tenor_size = None
        self.tenor_offset = None

        # the deal structure
        self.netting_sets = None

        # performance and admin feedback
        self.calc_stats = {}
        # the calculation parameters (defined by calling execute)
        self.params = {}
        # Index for the gradients of this calculation (if requested)
        self.gradient_index = None
        # output of calc stored here
        self.output = {}

    def execute(self, params):
        pass

    def make_factor_index(self, tensors):
        # need to match the indices back
        tenors = utils.get_tenors(self.all_factors)
        self.gradient_index = {}
        for name, var in tensors:
            factor_name = utils.check_scope_name(name)
            name_size, pad_size = tenors[factor_name].shape
            padding = 3 - pad_size
            indices = np.pad(tenors[factor_name], [[0, 0], [0, padding]], 'constant')
            self.gradient_index[factor_name] = (indices, pad_size)

    def gradients_as_df(self, grad, header='Gradient', display_val=False):
        if isinstance(grad, dict):
            # get the factor values from all_factors if necessary
            factor_values = {utils.check_scope_name(k): v.factor if hasattr(v, 'factor') else v
                             for k, v in self.all_factors.items()} if display_val else {}
            hessian_index = ([], [])
            factor, rate, tenor, values = [], [], [], []
            for name, v in grad.items():
                # first derivative
                non_zero = np.where(v)[0]
                grad_index, index_len = self.gradient_index[name]
                hessian_index[0].append([name] * v.shape[0])
                hessian_index[1].append(grad_index)
                values.append(v[non_zero])
                rate.append([name] * non_zero.size)
                tenor.append(grad_index[non_zero])
                # store the actual factor value if required
                if display_val:
                    if name in factor_values:
                        rate_non_zero = grad_index[non_zero][:, :index_len].astype(np.float64)
                        factor_val = factor_values[name].current_value(rate_non_zero).flatten()
                    else:
                        sub_rate, param = name.rsplit('.', 1)
                        sub_rate_non_zero = grad_index[non_zero].shape[0]
                        factor_val = factor_values[sub_rate].current_value()[param].flatten()[:sub_rate_non_zero]

                    if factor_val.size == non_zero.size:
                        factor.append(factor_val)

            tenors = np.vstack(tenor)
            self.hessian_index = (np.hstack(hessian_index[0]), np.vstack(hessian_index[1]))

            data = {'Rate': np.hstack(rate), 'Tenor': tenors[:, 0], 'Tenor2': tenors[:, 1],
                    'Tenor3': tenors[:, 2], header: np.hstack(values)}
            index = ['Rate', 'Tenor', 'Tenor2', 'Tenor3']

            if display_val:
                data['Value'] = np.hstack(factor)

            df = pd.DataFrame(data).set_index(index).sort_index(level=[0, 1, 2, 3])
        else:
            # second derivative
            multi_index = []
            non_zero = np.where(grad)
            headers = [None, header]
            for extra_index, full_index in zip(headers, non_zero):
                index = np.unique(full_index)
                rate = self.hessian_index[0][index]
                tenor = self.hessian_index[1][index]
                m_index = [rate, tenor[:, 0], tenor[:, 1], tenor[:, 2]]
                if extra_index is not None:
                    m_index = [[extra_index] * rate.size] + m_index
                multi_index.append(pd.MultiIndex.from_arrays(m_index))

            values = grad[grad.any(axis=1)][:, grad.any(axis=0)]
            # sort both axis
            df = pd.DataFrame(values, index=multi_index[0], columns=multi_index[1]).sort_index(
                level=[0, 1, 2, 3], axis=0).sort_index(level=[0, 1, 2, 3], axis=1)

        return df

    def set_deal_structures(self, deals, output, deal_level_mtm=False):

        for node in deals:
            # get the instrument
            instrument = node['Instrument']
            # should we skip it?
            if node.get('Ignore') == 'True':
                self.calc_stats['Ignored'] = self.calc_stats.setdefault('Ignored', 0) + 1
                continue

            # logging info
            logging.root.name = instrument.field.get('Reference', '<undefined>')
            if node.get('Children'):
                struct = DealStructure(instrument, store_results=deal_level_mtm)
                self.set_deal_structures(node['Children'], struct, deal_level_mtm)
                output.add_structure_to_structure(
                    struct, self.base_date, self.static_factors, self.stoch_factors, self.all_factors,
                    self.all_tenors, self.time_grid, self.config.holidays, self.calc_stats)
                continue

            output.add_deal_to_structure(
                self.base_date, instrument, self.static_factors, self.stoch_factors,
                self.all_factors, self.all_tenors, self.time_grid, self.config.holidays, self.calc_stats)


class CMC_State(utils.Calculation_State):
    def __init__(self, cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                 seed, job_id, num_jobs, scale_survival=False, nomodel='Constant', keep_mtm=False):
        super(CMC_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, batch_size)
        # these are tensors
        self.t_PreCalc = {}
        self.t_cholesky = cholesky
        self.t_random_numbers = None
        self.t_Scenario_Buffer = {}
        # these are shared parameter states
        self.sobol = {}
        # idea is to reuse quasi rng numbers where applicable (but still using enough randomness)
        self.t_quasi_rng = {}
        self.t_quasi_rng_batch = {}
        # set the random seed - seed each job by its offset
        torch.manual_seed(seed + job_id)
        # needed if we are running across multiple gpu's
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.keep_mtm = keep_mtm
        # do we need to scale the mtm by the survival probability in the final answer?
        self.scale_survival = scale_survival

    def quasi_rng(self, dimension, sample_size):
        # may need to parameterize these
        seed = 1234
        fast_forward = 1024

        if dimension not in self.sobol:
            self.sobol[dimension] = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True, seed=seed)
            # skip this many samples
            self.sobol[dimension].fast_forward(fast_forward)

        # hash the tensor
        batch_key = (dimension, sample_size)
        batch_num = self.t_quasi_rng_batch.setdefault(batch_key, 0)
        sample_key = (dimension, sample_size, batch_num)

        if sample_key not in self.t_quasi_rng:
            sample_sobol = self.sobol[dimension].draw(sample_size, dtype=self.one.dtype)
            u = (0.5 + (1 - torch.finfo(sample_sobol.dtype).eps) * (sample_sobol - 0.5)).to(self.one.device)
            self.t_quasi_rng[sample_key] = (utils.norm_icdf(u), u)

        # update the batch key
        self.t_quasi_rng_batch[batch_key] += 1
        # return the cached batch of quasi random numbers
        return self.t_quasi_rng[sample_key]

    def reset_qrg(self):
        self.t_quasi_rng_batch = {}
        
    def reset_cashflows(self, time_grid):
        # reset the cashflows
        self.t_Cashflows = {k: {t_i: self.one.new_zeros(self.simulation_batch)
                                for t_i in np.where(v >= 0)[0]} for k, v in time_grid.CurrencyMap.items()}

    def save_cashflows(self, output, time_grid):
        dates = np.array(sorted(time_grid.mtm_dates))
        for currency, values in self.t_Cashflows.items():
            cash_index = dates[sorted(values.keys())]
            output.setdefault('cashflows', {}).setdefault(currency, []).append(
                pd.DataFrame(
                    [v.cpu().detach().numpy() for _, v in sorted(values.items())], index=cash_index))

    @staticmethod
    def save_results(output, tensors):
        for k, v in tensors.items():
            output.setdefault(k, []).append(v.detach().cpu().numpy().astype(np.float64))

    def reset(self, num_factors, time_grid: utils.TimeGrid, use_antithetic=False):
        # update the random numbers
        sample_size = self.simulation_batch // 2 if use_antithetic else self.simulation_batch
        correlated_sample = torch.matmul(
            self.t_cholesky, torch.randn(
                num_factors, sample_size * time_grid.scen_time_grid.size,
                dtype=self.one.dtype, device=self.one.device)
        ).reshape(num_factors, time_grid.scen_time_grid.size, -1)

        if use_antithetic:
            self.t_random_numbers = torch.concat([correlated_sample, -correlated_sample], dim=-1)
        else:
            self.t_random_numbers = correlated_sample

        self.reset_cashflows(time_grid)

        # clear the buffers
        self.t_Buffer.clear()


class Credit_Monte_Carlo(Calculation):
    documentation = ('Calculations', [
        'A profile is a curve $V(t)$ with values specified at a discrete set of future dates $0=t_0<t_1<...<t_m$ with',
        'values at other dates  obtained via linear interpolation or zero extrapolation i.e. if $t_{i-1}<t<t_i$ then',
        '$V(t)$ is a linear interpolation of $V(t_{i-1})$ and $V(t_i)$; otherwise $V(t)=0$.',
        '',
        'The valuation models described earlier are used to construct the profile. The profile dates $t_1,...,t_m$ are',
        'obtained by taking the following union:',
        '',
        '- The deal\'s maturity date.',
        '- The dates in the **Time Grid** up to the deal\'s maturity date.',
        '- Deal specific dates such as payment and exercise dates.',
        '',
        'Deal specific dates improve the accuracy of the profile by showing the effect of cashflows, exercises etc.',
        '',
        '### Aggregation',
        '',
        'If $U$ and $V$ are profiles, then the set $U+V$ is the union of profile dates $U$ and $V$. If $E$ is the',
        'credit exposure profile in reporting',
        'currency (**Currency**), then:',
        '',
        '$$E = \\sum_{d} V_d$$',
        '',
        'where $V_d$ is the valuation profile of the $d^{th}$ deal. Note that Netting is always assumed to be',
        '**True**.',
        '',
        '#### Peak Exposure',
        '',
        'This is the simulated exposure at **Percentile** $q$ where $0<q<1$ (typically q=.95 or .99).',
        '',
        '#### Expected Exposure',
        '',
        'This is the profile defined by taking the average of the positive simulated exposures i.e. for each profile',
        'date $t$,',
        '',
        '$$\\bar E(t)=\\frac{1}{N}\\sum_{k=1}^N \\max(E(t)(k),0).$$'
        '',
        '#### Exposure Deflation',
        '',
        'Exposure at time $t$ is simulated in units of the time $t$ reporting currency. Exposure deflation converts',
        'this to time $0$ reporting currency i.e.',
        '',
        '$$V^*(t)=\\frac{V(t)}{\\beta(t)}$$',
        '',
        'where',
        '',
        '$$\\beta(t)=\\exp\\Big(\\int_0^t r(s)ds\\Big).$$',
        '',
        'This can be approximated by:',
        '',
        '$$\\beta(t)=\\prod_{i=0}^n\\frac{1}{D(s_{i-1},s_i)}$$',
        '',
        'where $0=s_0<...<s_n=t$. The discrete set of dates $s_1,...,s_{n-1}$ are model-dependent.'
        '',
        '### Credit Valuation Adjustment',
        '',
        'This represents the adjustment to the market value of the portfolio accounting for the risk of default. Only',
        'unilateral CVA (i.e accounting for the counterparty risk of default but ignoring the potential default of',
        'the investor) is calculated. It is given by:',
        '',
        '$$C=\\Bbb E(L(\\tau)),$$',
        '',
        'where the expectation is taken with respect to the risk-neutral measure, and ',
        '',
        '$$L(t)=(1-R)\\max(E^*(t),0),$$',
        '',
        'with:',
        '',
        '- $R$ the counterparty recovery rate',
        '- $\\tau$ the counterparty time to default',
        '- $E^*(t)$ the exposure at time $t$ deflated by the money market account.',
        '',
        'If **Deflate Stochastically** is **No** then the deflated expected exposure is assumed to be deterministic',
        'i.e. $E^*(t)=E(t)D(0,t)$. Note that if $T$ is the end date of the portfolio exposure then $E^*(t)=0$ for',
        '$t>T$.',
        '',
        'Now,',
        '',
        '$$\\Bbb E(L(\\tau))=\\Bbb E\\Big(\\int_0^T L(t)(-dH(t))\\Big)$$',
        '',
        '$$H(t)=\\exp\\Big(-\\int_0^t h(u)du\\Big)$$',
        '',
        'where $h(t)$ is the stochastic hazard rate. There are two ways to calculate the expectation:',
        '',
        'If **Stochastic Hazard** is **No** then $H(t)=\\Bbb P(\\tau > t)=S(0,t)$, the risk neutral survival',
        'probability to time $t$ and',
        '',
        '$$C=\\int_0^T \\Bbb E(L(t))(-dH(t))\\approx\\sum_{i=1}^m C_i,$$',
        '',
        'with',
        '',
        '$$C_i=\\Big(\\frac{\\Bbb E(L(t_{i-1}))+\\Bbb E(L(t_i))}{2}\\Big)(S(0,t_{i-1})-S(0,t_i))$$',
        '',
        'and $0=t_0<...<t_m=T$ are the time points on the exposure profile. Note that the factor models used should',
        'be risk-neutral to give risk neutral simulations of $\\Bbb E^*(t)$.',
        '',
        'If **Stochastic Hazard** is **Yes** then $S(t,u)$ is the simulated survival probability at time $t$ for',
        'maturity $u$ and is related to $H$ by',
        '',
        '$$S(t,u)=\\Bbb E(\\frac{H(u)}{H(t)}\\vert\\mathcal F(t)).$$',
        '',
        'where $\\mathcal F$ is the filtration given by the risk factor processes. For small $u-t$, the approximation',
        '$H(u)\\approx H(t)S(t,u)$ is accurate so that',
        '',
        '$$C\\approx\\sum_{i=1}^m C_i$$',
        '',
        'and',
        '',
        '$$C_i=\\Bbb E\\Big[\\Big(\\frac{L(t_{i-1}))+L(t_i)}{2}\\Big)(H_{i-1}-H_i)\\Big],$$',
        '',
        'again, $0=t_0<...<t_m=T$ are the time points on the exposure profile and',
        '',
        '$$H_i=S(0,t_1)S(t_1,t_2)...S(t_{i-1},t_i).$$',
        '',
        '### Funding Valuation Adjustment',
        '',
        'Not Posting (or receiving) collateral can imply a funding cost (or benefit) when there is a spread between a',
        'party\'s interal cost of funding and the rate that would be recieved should the counterparty place collateral.'
        'The discounted expectation of this cost (or benefit) summed across all time horizons and scenarios constitutes',
        'a funding value adjustment and can be expressed as:',
        '',
        '$$FCA=\\int_{0}^{T} \\Bbb{E}\\Big(max(V(t),0)[f_{fc,C}(t)-f_{rf,C}(t)]SP_C(t)\\Big)dt$$',
        '',
        '$$FBA=\\int_{0}^{T} \\Bbb{E}\\Big(min(V(t),0)[f_{fb,C}(t)-f_{rf,C}(t)]SP_C(t)\\Big)dt$$',
        '',
        'where',
        '',
        '- $T$ is the exposure horizon',
        '- $V(t)$ is the deflated funding profile at time $t$',
        '- $f_{fc,C}(t)$ and $f_{fb,C}(t)$ is the funding cost and benefit spreads respectively',
        '- $f_{rf,C}(t)$ is the risk-free rate',
        '- $SP_C(t)$ is the survival probability of the Counterparty.',
        '',
        'FVA against the counterparty is then calculated as $FVA = FCA + FBA$',
        '',
        'At the bank wide level the $FCA$ and $FBA$ is calculated as:',
        '',
        '$$FCA_{bank}=\\int_{0}^{T} \\Bbb{E}\\Bigg(max\\Big(\\sum_i V(t)SP_{Ci},0\\Big)[f_{fc,C}(t)-f_{rf,C}(t)]\\Bigg)dt$$',
        '',
        '$$FBA_{bank}=\\int_{0}^{T} \\Bbb{E}\\Bigg(min\\Big(\\sum_i V(t)SP_{Ci},0\\Big)[f_{fb,C}(t)-f_{rf,C}(t)]\\Bigg)dt$$',
        '',
        'The idea is the same as defined above except that $i$, the counterparty index, sums over all uncollateralized ',
        'or partially collateralized counterparties (one-way CSA or high threshold CSA).',
        '',
        'Calculation parameters are extended from the Base Valuation with these new fields:',
        '',
        ' - **Deflation Interest Rate** - the interest rate price factor to PV the exposure to today',
        ' - **Batch Size** - The number of simulations per batch. Smaller Batch sizes are more likely to fit in memory.'
        '  This needs to be balanced with speed - larger batch sizes will run quicker.',
        ' - **Simulation Batches** - Number of batches to run. Total number of sumulations is **Simulation Batches** * '
        '**Batch Size**. Note that **Batch Size** is usually a power of 2.',
        ' - **Antithetic** - Use antithethic variables - we run twice the number of simulations using the negative of ',
        '  the random sample for the second run',
        ' - **Calc Scenarios** - return the simulated price factors used in the calculation ',
        ' - **Dynamic Scenario Dates** - Generate scenarios not just on the **Time Grid**, but also on all potential '
        'cashflow settlement dates. Needed to accurately calculate liquidity and settlement dynamics on collateralized ',
        'portfolios.',
        ' - **Generate Cashflows** - returns the simulated cashflows during the simulation period'
    ])

    def __init__(self, config, **kwargs):
        super(Credit_Monte_Carlo, self).__init__(config, **kwargs)
        self.reset_dates = None
        self.settlement_currencies = None
        # used to store any jacobian matrices
        self.jacobians = {}
        # implied factors
        self.implied_factors = {}
        # we represent the calc as a combination of static, stochastic and implied parameters
        self.implied_var = {}

        # potentially store the full list of variables
        self.all_var = None

    def update_factors(self, params, base_date, job_id, num_jobs):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, self.input_time_grid)

        self.update_time_grid(base_date, reset_dates, settlement_currencies,
                              dynamic_scenario_dates=params.get('Dynamic_Scenario_Dates', 'No') == 'Yes')

        return self._build_factor_state(
            dependent_factors, stochastic_factors, implied_factors, params, base_date, job_id, num_jobs)

    def _build_factor_state(self, dependent_factors, stochastic_factors, implied_factors,
                            params, base_date, job_id, num_jobs):
        """Construct factor objects, tensors, shared memory and precalculate processes.

        Called by update_factors after the time grid and dependency sets are known.
        Subclasses that build their own dependency sets (e.g. HedgeMonteCarlo) can
        call this directly instead of going through calculate_dependencies.
        """
        # now construct the stochastic factors and static factors for the simulation
        self.stoch_factors.clear()

        for price_model, price_factor in stochastic_factors.items():
            factor_obj = construct_factor(
                price_factor, self.config.params['Price Factors'],
                self.config.params['Price Factor Interpolation'],
                base_date=base_date)
            implied_factor = implied_factors.get(price_model)
            try:
                if implied_factor:
                    implied_obj = construct_factor(
                        implied_factor, self.config.params['Price Factors'],
                        self.config.params['Price Factor Interpolation'],
                        base_date=base_date)
                    self.implied_factors[implied_factor] = implied_obj
                else:
                    implied_obj = None
            except KeyError as e:
                logging.error('Implied Factor {0} missing in market data file'.format(e.args))

            self.stoch_factors[price_factor] = construct_process(
                price_model.type, factor_obj,
                self.config.params['Price Models'][utils.check_tuple_name(price_model)], implied_obj)

        self.static_factors = {}
        for price_factor in set(dependent_factors).difference(stochastic_factors.values()):
            try:
                self.static_factors.setdefault(
                    price_factor, construct_factor(
                        price_factor, self.config.params['Price Factors'],
                        self.config.params['Price Factor Interpolation'],
                        base_date=base_date)
                )
            except KeyError as e:
                logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.args))

        self.all_factors = self.stoch_factors.copy()
        self.all_factors.update(self.static_factors)
        self.all_factors.update(self.implied_factors)
        self.num_factors = sum([v.num_factors() for v in self.stoch_factors.values()])

        # get the tenor offset (if any)
        tenor_offset = params.get('Tenor_Offset', 0.0)
        # check if we need gradients for any sub calc
        greeks = bool(np.any([params[k].get('Gradient', 'No') == 'Yes' for k, v in params.items() if type(v) == dict]))
        sensitivities = params.get('Gradient_Variables', 'All')

        # now get the stochastic risk factors ready - these will be generated from the price models
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                # check if there are any implied factors linked here
                if hasattr(value, 'implied'):
                    vars = {}
                    calc_grad = greeks and sensitivities in ['All', 'Implied']
                    for param_name, param_value in value.implied.current_value().items():
                        factor_name = utils.Factor(value.implied.__class__.__name__, key.name + (param_name,))
                        vars[factor_name] = torch.tensor(
                            param_value, device=self.device, dtype=self.dtype, requires_grad=calc_grad)
                    self.implied_var[key] = vars

                # check the daycount for the tenor_offset
                if tenor_offset:
                    factor_tenor_offset = utils.get_day_count_accrual(
                        base_date, tenor_offset, value.factor.get_day_count() if hasattr(
                            value.factor, 'get_day_count') else utils.DAYCOUNT_ACT365)
                else:
                    factor_tenor_offset = 0.0

                # record the offset of this risk factor
                current_val = value.factor.current_value(offset=factor_tenor_offset)
                calc_grad = greeks and sensitivities in ['All', 'Factors']
                self.stoch_var[key] = torch.tensor(
                    current_val, device=self.device, dtype=self.dtype, requires_grad=calc_grad)

        # and then get the static risk factors ready - these will just be looked up
        calc_grad = greeks and sensitivities in ['All', 'Factors']
        for key, value in self.static_factors.items():
            if key.type not in utils.DimensionLessFactors:
                # check the daycount for the tenor_offset
                if tenor_offset:
                    factor_tenor_offset = utils.get_day_count_accrual(
                        base_date, tenor_offset, value.get_day_count() if hasattr(
                            value, 'get_day_count') else utils.DAYCOUNT_ACT365)
                else:
                    factor_tenor_offset = 0.0
                # record the offset of this risk factor
                current_val = value.current_value(offset=factor_tenor_offset)
                if isinstance(current_val, dict):
                    for k, v in current_val.items():
                        self.static_var[utils.Factor(key.type, key.name + (k,))] = torch.tensor(
                            v, device=self.device, dtype=self.dtype, requires_grad=calc_grad)
                else:
                    self.static_var[key] = torch.tensor(
                        current_val, device=self.device, dtype=self.dtype, requires_grad=calc_grad)

        # set up the device and allocate memory
        shared_mem = self.__init_shared_mem(
            int(params['Random_Seed']), params.get('NoModel', 'Constant'),
            params['Currency'], params.get('MCMC_Simulations', 2048),
            job_id, num_jobs, calc_greeks=sensitivities if greeks else None)

        # calculate a reverse lookup for the tenors and store the daycount code
        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        # now initialize all stochastic factors
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                if key in self.implied_var:
                    implied_tensor = {k.name[-1]: v for k, v in self.implied_var[key].items()}
                    value.link_references(implied_tensor, self.implied_var, self.implied_factors)
                else:
                    implied_tensor = None
                value.precalculate(
                    base_date, ScenarioTimeGrid(dependent_factors[key], self.time_grid, base_date),
                    self.stoch_var[key], shared_mem, self.process_ofs[key], implied_tensor=implied_tensor)
                if not value.params_ok:
                    logging.warning('Stochastic factor {} has been modified'.format(utils.check_scope_name(key)))

        # now check if any of the stochastic processes depend on other processes
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                value.calc_references(key, self.static_factors, self.stoch_factors, self.all_tenors, self.all_factors)

        return shared_mem

    def update_time_grid(self, base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=False):
        # work out the scenario and dynamic dates
        dynamic_dates = set([x for x in reset_dates if x > base_date])

        # we are repeating a period till the last reset date
        if self.input_time_grid.strip().endswith(')'):
            base_mtm_dates = self.config.parse_grid(base_date, max(dynamic_dates), self.input_time_grid)
            mtm_dates = base_mtm_dates.union(dynamic_dates)
        else:
            # we are only running the calc till the last period specified (and clipping everything else)
            max_date = base_date + self.config.periodparser.parseString(
                self.input_time_grid.strip().split(' ')[-1].upper())[0]
            base_mtm_dates = self.config.parse_grid(base_date, max_date, self.input_time_grid)
            reset_dates = [x for x in dynamic_dates if x <= max_date]
            mtm_dates = base_mtm_dates.union(reset_dates)

        if dynamic_scenario_dates:
            scenario_dates = mtm_dates
        else:
            scenario_dates = self.config.parse_grid(
                base_date, max(dynamic_dates), self.input_time_grid, past_max_date=True)

        # set up the scenario and time grids
        self.time_grid = utils.TimeGrid(scenario_dates, mtm_dates, base_mtm_dates)
        self.base_date = base_date
        self.reset_dates = reset_dates
        self.time_grid.set_base_date(base_date)

        # Set the settlement dates
        self.time_grid.set_currency_settlement(settlement_currencies)
        self.settlement_currencies = settlement_currencies

    def get_cholesky_decomp(self):
        # create the correlation matrix
        correlation_matrix = np.eye(self.num_factors, dtype=np.float64)
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        # prepare the correlation matrix (and the offsets of each stochastic process)
        correlation_factors = []
        self.process_ofs = {}
        for key, value in self.stoch_factors.items():
            proc_corr_type, proc_corr_factors = value.correlation_name
            for sub_factors in proc_corr_factors:
                # record the offset of this factor model
                self.process_ofs.setdefault(key, len(correlation_factors))
                # record the name of needed correlation lookup
                correlation_factors.append(utils.Factor(proc_corr_type, key.name + sub_factors))

        for index1 in range(self.num_factors):
            for index2 in range(index1 + 1, self.num_factors):
                factor1, factor2 = utils.check_tuple_name(correlation_factors[index1]), utils.check_tuple_name(
                    correlation_factors[index2])
                key = (factor1, factor2) if (factor1, factor2) in self.config.params['Correlations'] else (
                    factor2, factor1)
                rho = self.config.params['Correlations'].get(key, 0.0) if factor1 != factor2 else 1.0
                correlation_matrix[index1, index2] = rho
                correlation_matrix[index2, index1] = rho

        raw_eigval, raw_eigvec = np.linalg.eig(correlation_matrix)
        eigval, eigvec = np.real(raw_eigval), np.real(raw_eigvec)
        # need to do cholesky
        while (eigval < 1e-8).any():
            # matrix not positive definite - find a close positive definite matrix
            if self.config.params['System Parameters']['Correlations_Healing_Method'] == 'Eigenvalue_Raising':
                logging.warning('Correlation matrix (size {0}) not positive definite - raising eigenvalues'.format(
                    correlation_matrix.shape))
                P_plus_B = eigvec.dot(np.diag(np.maximum(eigval, 1e-4))).dot(eigvec.T)
                diagonal_norm = np.diag(1.0 / np.sqrt(P_plus_B.diagonal()))
                new_correlation_matrix = diagonal_norm.dot(P_plus_B).dot(diagonal_norm)
            else:
                logging.warning('Correlation matrix (size {0}) not positive definite - alternating Projections'.format(
                    correlation_matrix.shape))

                C = correlation_matrix.astype(np.float64).copy()
                B = correlation_matrix.astype(np.float64).copy()

                # don't do more than 100 iterations - if we need to do this much, behaviour is undefined
                for k in range(100):
                    eigval, eigvec = np.linalg.eig(B)
                    P_plus_B = eigvec.dot(np.diag(np.maximum(eigval, 1e-4))).dot(eigvec.T)
                    nC = P_plus_B + np.diag(1.0 - P_plus_B.diagonal())
                    D = nC - P_plus_B
                    B += D

                    # exit early
                    if np.abs(C - nC).max() < 1e-08 * np.abs(nC).max():
                        break

                    C = nC

                new_correlation_matrix = nC

            correlation_matrix = new_correlation_matrix
            # check again
            raw_eigval, raw_eigvec = np.linalg.eig(correlation_matrix)
            eigval, eigvec = np.real(raw_eigval), np.real(raw_eigvec)

        correlation_matrix = torch.tensor(
            correlation_matrix, device=self.device, dtype=self.dtype, requires_grad=False)
        # return the cholesky decomp
        return torch.linalg.cholesky(correlation_matrix)

    def __init_shared_mem(self, seed, nomodel, reporting_currency, mcmc_sim, job_id, num_jobs, calc_greeks=None):
        # check if we need to report gradients
        if calc_greeks is not None:
            implied_vars = list(itertools.chain(*[x.items() for x in self.implied_var.values()]))
            if calc_greeks == 'Implied':
                self.all_var = implied_vars
            elif calc_greeks == 'Factors':
                self.all_var = self.stoch_var + self.static_var
            else:
                self.all_var = implied_vars + list(self.stoch_var.items()) + list(self.static_var.items())
            # build our index
            self.make_factor_index(self.all_var)

        # Now create a shared state with the cholesky decomp
        # check the calculation parameters to see if we need to tweak the calculation a bit:
        scale_by_survival = False
        if self.params.get('Funding_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
            scale_by_survival = True

        shared_mem = CMC_State(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, job_id, num_jobs, scale_by_survival, keep_mtm=self.params.get('Keep_MtM', 'No')=='Yes')

        return shared_mem

    def report(self, output):
        for result, data in output.items():
            if result == 'scenarios':
                scen = {}
                scenario_date_index = pd.DatetimeIndex(sorted(self.time_grid.scenario_dates))
                if self.params['Calc_Scenarios'] == 'At_Percentile':
                    # calc pfe
                    dates = np.array(sorted(self.time_grid.mtm_dates))[self.time_grid.report_index]
                    mtms = pd.DataFrame(np.concatenate(output['mtm'], axis=-1).astype(np.float64), index=dates)
                    percentiles = self.params.get('Percentile', '95').replace(' ', '').split(',')
                    profiles = {x: np.percentile(mtms.values, float(x), axis=1) for x in percentiles}
                    index = {x: np.argmin(np.abs(mtms.values - profiles[x][:, np.newaxis]), axis=1) for x in percentiles}

                    # now only extract the scenarios at percentile points
                    for factor_key, factor_values in data.items():
                        factor_name = utils.check_tuple_name(factor_key)
                        values = np.concatenate(factor_values, axis=-1)  # Shape: (num_rows, num_scenarios)
                        value_len = values.shape[0]
                        if len(values.shape) == 2:
                            columns = pd.MultiIndex.from_product(
                                [[0.0], percentiles], names=['tenor', 'scenario'])
                            vals = np.dstack([values[np.arange(value_len), i[:value_len]] for i in index.values()])
                            scen[factor_name] = pd.DataFrame(
                                vals.reshape(value_len, -1), index=scenario_date_index[:value_len], columns=columns).T
                        else:
                            tenors = self.all_tenors[factor_key][0].tenor
                            columns = pd.MultiIndex.from_product(
                                [tenors, percentiles], names=['tenor', 'scenario'])
                            vals = np.dstack([values[np.arange(value_len), :, i[:value_len]] for i in index.values()])
                            scen[factor_name] = pd.DataFrame(
                                vals.reshape(value_len, -1),
                                index=scenario_date_index[:value_len], columns=columns).T
                else:
                    for k, v in data.items():
                        factor_name = utils.check_tuple_name(k)
                        values = np.concatenate(v, axis=-1)
                        if len(values.shape) == 2:
                            columns = pd.MultiIndex.from_product(
                                [[0.0], np.arange(values.shape[-1])], names=['tenor', 'scenario'])
                            scen[factor_name] = pd.DataFrame(
                                values, index=scenario_date_index[:values.shape[0]], columns=columns).T
                        else:
                            tenors = self.all_tenors[k][0].tenor
                            columns = pd.MultiIndex.from_product(
                                [tenors, np.arange(values.shape[-1])], names=['tenor', 'scenario'])
                            scen[factor_name] = pd.DataFrame(
                                values.reshape(values.shape[0], -1),
                                index=scenario_date_index[:values.shape[0]], columns=columns).T
                self.output.setdefault('scenarios', scen)
            elif result == 'cashflows':
                self.output.setdefault('cashflows', {k: pd.concat(v, axis=1) for k, v in data.items()})
            elif result in ['cva', 'collva', 'fva', 'legacy_fva']:
                self.output.setdefault(result, np.array(data, dtype=np.float64).mean())
            elif result == 'collva_t':
                self.output.setdefault(result, np.array(data, dtype=np.float64).mean(axis=0))
            elif result in ['grad_cva', 'grad_collva', 'grad_fva', 'grad_legacy_fva']:
                grad = {}
                for k, v in data.items():
                    grad[k] = v.astype(np.float64) / self.params['Simulation_Batches']
                self.output.setdefault(result, self.gradients_as_df(grad, display_val=True))
            elif result == 'CS01':
                columns = pd.MultiIndex.from_arrays(
                    [data['Par_CDS'].keys(), data['Par_CDS'].values()], names=['Tenor', 'Par CDS'])
                self.output.setdefault(result, pd.DataFrame(columns=columns, index=data['Tenor'], data=np.transpose(
                    [x - data['Shifted_Log_Prob'][0] for x in data['Shifted_Log_Prob'][1:]])))
            elif result in ['grad_cva_hessian']:
                self.output.setdefault(result, self.gradients_as_df(
                    data.astype(np.float64) / self.params['Simulation_Batches']))
            elif result in ['mtm', 'collateral']:
                dates = np.array(sorted(self.time_grid.mtm_dates))[self.time_grid.report_index]
                self.output.setdefault(result, pd.DataFrame(
                    np.concatenate(data, axis=-1).astype(np.float64), index=dates))
            elif result == 'gross_mtm':
                dates = np.array(sorted(self.time_grid.mtm_dates))
                self.output.setdefault(result, pd.DataFrame(
                    np.concatenate(data, axis=-1).astype(np.float64), index=dates))
            else:
                self.output.setdefault(result, np.concatenate(data, axis=-1).astype(np.float64))

        return self.output

    def execute(self, params, job_id=0, num_jobs=1):
        # get the rundate
        base_date = pd.Timestamp(params['Run_Date'])

        # Define the base and scenario grids
        self.input_time_grid = params['Time_grid']
        # needed if we are using multiprocessing across gpu's
        params['Simulation_Batches'] = params['Simulation_Batches'] // num_jobs
        self.batch_size = params['Batch_Size']
        self.numscenarios = self.batch_size * params['Simulation_Batches']

        # store the params
        self.params = params
        # set the name of the root logger to this netting set (makes tracking errors easier)
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)

        # store the stats for the batches
        self.calc_stats['Batch_Size'] = self.batch_size
        self.calc_stats['Simulation_Batches'] = self.params['Simulation_Batches']
        self.calc_stats['Random_Seed'] = params['Random_Seed']

        # update the factors and obtain shared state
        shared_mem = self.update_factors(params, base_date, job_id, num_jobs)

        # set up the all instruments
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(
            self.config.deals['Deals']['Children'], self.netting_sets, deal_level_mtm=params.get('DealLevel', False))
        self.netting_sets.finalize_struct(base_date, self.time_grid)

        # clear the output
        output = defaultdict(list)
        # reset the tensors - used for storing simulation data
        tensors = {}
        # record how long it took to run the calc (python + pytorch)
        execution_label = 'Tensor_Execution_Time ({})'.format(self.device.type)
        self.calc_stats[execution_label] = time.monotonic()

        for run in range(self.params['Simulation_Batches']):

            # need to refresh random numbers and zero out buffers
            shared_mem.reset(
                self.num_factors, self.time_grid, use_antithetic=params.get('Antithetic', 'No') == 'Yes')

            # simulate the price factors
            for key, value in self.stoch_factors.items():
                shared_mem.t_Scenario_Buffer[key] = value.generate(shared_mem)

            # construct the valuations

            # use these lines below to track down any issues that prevent gradients from flowing (debugging only)
            # with torch.autograd.detect_anomaly():
            tensors['mtm'] = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
            #    m = tensors['mtm'].mean()
            #    m.backward()

            # is this the final run?
            final_run = run == self.params['Simulation_Batches'] - 1

            # now calculate all the valuation adjustments (if necessary)
            if params.get('Collateral_Valuation_Adjustment', {}).get(
                    'Calculate', 'No') == 'Yes' and shared_mem.simulation_batch > 1:

                if params['Collateral_Valuation_Adjustment'].get('Gradient', 'No') == 'Yes':
                    tensors['collva_t'] = torch.mean(shared_mem.t_Credit['Funding'], dim=1)
                    tensors['collva'] = torch.sum(tensors['collva_t'])

                    # calculate all the derivatives of fva
                    sensitivity = SensitivitiesEstimator(tensors['collva'], self.all_var)

                    if final_run:
                        output['grad_collva'] = sensitivity.report_grad()
                        # store the size of the Gradient
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P

            if params.get('Initial_Margin', {}).get('Calculate', 'No') == 'Yes':
                def calc_buckets(liq_w, tenor):
                    liquidity = {}
                    for col in liq_w.T.iterrows():
                        left_limit = 0.0
                        right_limit = 0.0
                        series = col[1].dropna()
                        if '<=' in series.index[0]:
                            left_limit = 1.0
                            y = series.index.map(lambda x: float(x.replace('<=', '').replace('y', '')))
                        elif '>=' in series.index[-1]:
                            right_limit = 1.0
                            y = series.index.map(lambda x: float(x.replace('>=', '').replace('y', '')))
                        else:
                            y = series.index.map(lambda x: float(x.replace('y', '')))
                        liquidity[col[0]] = np.interp(tenor, y, series.values, left=left_limit, right=right_limit)
                    return liquidity

                def calc_max(liquidity_charge, tenor1, tenor2):
                    return torch.where(
                        liquidity_charge[tenor1]*liquidity_charge[tenor2] < 0,
                        torch.max(torch.abs(liquidity_charge[tenor1]), torch.abs(liquidity_charge[tenor2])),
                        torch.abs(liquidity_charge[tenor1])+torch.abs(liquidity_charge[tenor2]))

                # time_grid = self.netting_sets.sub_structures[0].obj.Time_dep.deal_time_grid
                time_grid = self.time_grid.report_index
                liq_w = pd.read_csv(params['Initial_Margin']['Liquidity_Weights'], index_col=0)
                irs = pd.read_csv(params['Initial_Margin']['IRS_Weights'], index_col=0)
                local_curves = [k for k, v in self.all_factors.items() if len(
                    k.name) == 1 and k.type == 'InterestRate' and (
                                    v.factor.param if hasattr(v, 'factor') else v.param).get(
                    'Currency') == params['Initial_Margin']['Local_Currency']]
                base_ccy = get_fxrate_factor(
                    utils.check_rate_name(self.config.params['System Parameters']['Base_Currency']),
                    self.static_factors, self.stoch_factors)
                fx_report = utils.calc_fx_cross(
                    shared_mem.Report_Currency, base_ccy, self.time_grid.time_grid[time_grid], shared_mem)
                IM_currency = get_fxrate_factor(
                    utils.check_rate_name(params['Initial_Margin']['IM_Currency']),
                    self.static_factors, self.stoch_factors)
                fx_IM_report = utils.calc_fx_cross(
                    base_ccy, IM_currency, self.time_grid.time_grid[time_grid], shared_mem)
                if local_curves[0] in shared_mem.t_Scenario_Buffer:
                    scen_buf = shared_mem.t_Scenario_Buffer
                else:
                    scen_buf = shared_mem.t_Static_Buffer

                local_tenor = {k: self.all_tenors[k][0].tenor for k in local_curves}
                # round the tenor to 1 dp to ensure accurate bucket lookup
                local_shifts = {k: calc_buckets(liq_w, t.round(1)) for k, t in local_tenor.items()}

                all_shifts = {}
                liquidity_deltas = {}

                for d in local_shifts.values():  # you can list as many input dicts as you want here
                    for key, value in d.items():
                        all_shifts.setdefault(key, []).append(value)

                # switch off cashflows
                shared_mem.t_Cashflows = None

                for tenor, shifts in all_shifts.items():
                    # bump the scenarios
                    deltas = {}
                    for curvename, shift in zip(local_shifts.keys(), shifts):
                        deltas[curvename] = shared_mem.one.new_tensor(shift.reshape(1, -1, 1) * 0.01 * 0.01)
                        scen_buf[curvename] += deltas[curvename]

                    # reset the cache
                    shared_mem.t_Buffer.clear()
                    # calc the liquidity change in base_currency - simple delta
                    liquidity_deltas[tenor] = (self.netting_sets.resolve_structure(
                        shared_mem, self.time_grid) - tensors['mtm']) * fx_report

                    # unbump the scenarios
                    for curvename, shift in zip(local_shifts.keys(), shifts):
                        scen_buf[curvename] -= deltas[curvename]

                liquidity_charge = {}
                for tenor, values in liquidity_deltas.items():
                    curve_tenor = utils.tenor_diff(irs[tenor].dropna().index.astype(np.float64).values)
                    curve_weights = shared_mem.one.new_tensor(irs[tenor].dropna().values)
                    index, index_next, alpha = curve_tenor.get_index(values)
                    liquidity_charge[tenor] = values * (
                            curve_weights[index] * (1 - alpha) + curve_weights[index_next] * alpha)

                IM_liquidity_charge = (calc_max(
                    liquidity_charge, '2y', '5y') + calc_max(liquidity_charge, '10y', '30y')) * fx_IM_report

                shared_mem.t_Buffer.clear()
                for int_rate in [k for k in shared_mem.t_Scenario_Buffer.keys() if
                                 k.type == 'InterestRate' and len(k.name) == 1]:
                    #calc pv01
                    shared_mem.t_Scenario_Buffer[int_rate] += 0.01 * 0.01
                for int_rate in [k for k in shared_mem.t_Static_Buffer if
                                 k.type == 'InterestRate' and len(k.name) == 1]:
                    # calc pv01
                    shared_mem.t_Static_Buffer[int_rate] += 0.01 * 0.01

                IM_delta_charge = (self.netting_sets.resolve_structure(
                    shared_mem, self.time_grid) - tensors['mtm']) * fx_report * fx_IM_report

                tensors['LCH_Margin'] = params['Initial_Margin']['Delta_Factor']*IM_delta_charge+IM_liquidity_charge

            if params.get('Funding_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
                time_grid = self.time_grid.report_index
                mtm_grid = self.time_grid.mtm_time_grid[time_grid]

                funding_cost = get_interest_factor(
                    utils.check_rate_name(params['Funding_Valuation_Adjustment']['Funding_Cost_Interest_Curve']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                funding_benefit = get_interest_factor(
                    utils.check_rate_name(params['Funding_Valuation_Adjustment']['Funding_Benefit_Interest_Curve']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                riskfree = get_interest_factor(
                    utils.check_rate_name(params['Funding_Valuation_Adjustment']['Risk_Free_Curve']),
                    self.static_factors, self.stoch_factors, self.all_tenors)

                deflation = utils.calc_time_grid_curve_rate(riskfree, np.zeros((1, 3)), shared_mem)
                DF_rf = torch.squeeze(torch.exp(-deflation.gather_weighted_curve(
                    shared_mem, mtm_grid.reshape(1, -1))), dim=0)

                Vk_plus_ti = torch.relu(tensors['mtm'] * DF_rf)
                Vk_minus_ti = torch.relu(-tensors['mtm'] * DF_rf)
                Vk_star_ti_p = (Vk_plus_ti[1:] + Vk_plus_ti[:-1]) / 2
                Vk_star_ti_m = (Vk_minus_ti[1:] + Vk_minus_ti[:-1]) / 2

                # note that for FVA, we already scale the exposure matrix by the survival probability
                if params['Funding_Valuation_Adjustment'].get('Deflate_Stochastically', 'No') == 'Yes':
                    delta_scen_t = np.diff(mtm_grid).reshape(-1, 1)

                    discount_fund_cost = utils.calc_time_grid_curve_rate(
                        funding_cost, self.time_grid.time_grid[time_grid[:-1]], shared_mem)
                    discount_fund_benefit = utils.calc_time_grid_curve_rate(
                        funding_benefit, self.time_grid.time_grid[time_grid[:-1]], shared_mem)
                    discount_rf = utils.calc_time_grid_curve_rate(
                        riskfree, self.time_grid.time_grid[time_grid[:-1]], shared_mem)

                    delta_fund_cost_rf = torch.squeeze(
                        torch.exp(discount_fund_cost.gather_weighted_curve(shared_mem, delta_scen_t)) -
                        torch.exp(discount_rf.gather_weighted_curve(shared_mem, delta_scen_t)), dim=1)

                    delta_fund_benefit_rf = torch.squeeze(
                        torch.exp(discount_fund_benefit.gather_weighted_curve(shared_mem, delta_scen_t)) -
                        torch.exp(discount_rf.gather_weighted_curve(shared_mem, delta_scen_t)), dim=1)
                else:
                    zero_fund_cost = utils.calc_time_grid_curve_rate(funding_cost, np.zeros((1, 3)), shared_mem)
                    zero_fund_benefit = utils.calc_time_grid_curve_rate(funding_benefit, np.zeros((1, 3)), shared_mem)
                    zero_rf = utils.calc_time_grid_curve_rate(riskfree, np.zeros((1, 3)), shared_mem)

                    Dt_T_fund_cost = torch.squeeze(torch.exp(
                        -zero_fund_cost.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)
                    Dt_T_fund_benefit = torch.squeeze(torch.exp(
                        -zero_fund_benefit.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)
                    Dt_T_rf = torch.squeeze(torch.exp(
                        -zero_rf.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)

                    delta_fund_cost_rf = (
                        (Dt_T_fund_cost[:-1] / Dt_T_fund_cost[1:]) - (Dt_T_rf[:-1] / Dt_T_rf[1:]))
                    delta_fund_benefit_rf = (
                        (Dt_T_fund_benefit[:-1] / Dt_T_fund_benefit[1:]) - (Dt_T_rf[:-1] / Dt_T_rf[1:]))

                FCA_t = torch.sum(delta_fund_cost_rf * Vk_star_ti_p, dim=0)
                FCA = torch.mean(FCA_t)
                FBA_t = torch.sum(delta_fund_benefit_rf * Vk_star_ti_m, dim=0)
                FBA = torch.mean(FBA_t)

                tensors['fva'] = FCA - FBA

                if params['Funding_Valuation_Adjustment'].get('Gradient', 'No') == 'Yes':
                    # calculate all the derivatives of fva
                    sensitivity = SensitivitiesEstimator(tensors['fva'], self.all_var)

                    if final_run:
                        output['grad_fva'] = sensitivity.report_grad()
                        # store the size of the Gradient
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P

            if params.get('Credit_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
                discount = get_interest_factor(
                    utils.check_rate_name(params['Deflation_Interest_Rate']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                survival = get_survival_factor(
                    utils.check_rate_name(params['Credit_Valuation_Adjustment']['Counterparty']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                recovery = get_recovery_rate(
                    utils.check_rate_name(params['Credit_Valuation_Adjustment']['Counterparty']), self.all_factors)

                time_grid = self.time_grid.report_index
                # Calculates unilateral CVA with or without stochastic deflation.
                mtm_grid = self.time_grid.mtm_time_grid[time_grid]
                delta_scen_t = np.hstack((0.0, np.diff(mtm_grid)))

                if params['Credit_Valuation_Adjustment']['Deflate_Stochastically'] == 'Yes':
                    zero = utils.calc_time_grid_curve_rate(
                        discount, self.time_grid.time_grid[time_grid], shared_mem)
                    Dt_T = torch.exp(-torch.squeeze(zero.gather_weighted_curve(
                        shared_mem, delta_scen_t.reshape(-1, 1))).cumsum(dim=0))
                else:
                    zero = utils.calc_time_grid_curve_rate(discount, np.zeros((1, 3)), shared_mem)
                    Dt_T = torch.squeeze(torch.exp(
                        -zero.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)

                pv_exposure = torch.relu(tensors['mtm'] * Dt_T)

                if params['Credit_Valuation_Adjustment']['Stochastic_Hazard_Rates'] == 'Yes':
                    surv = utils.calc_time_grid_curve_rate(
                        survival, self.time_grid.time_grid[time_grid], shared_mem)
                    St_T = torch.exp(-torch.cumsum(torch.squeeze(surv.gather_weighted_curve(
                        shared_mem, delta_scen_t.reshape(-1, 1), multiply_by_time=False), dim=1),
                        dim=0))
                else:
                    surv = utils.calc_time_grid_curve_rate(survival, np.zeros((1, 3)), shared_mem)
                    St_T = torch.squeeze(torch.exp(-surv.gather_weighted_curve(
                        shared_mem, mtm_grid.reshape(1, -1), multiply_by_time=False)), dim=0)

                prob = St_T[:-1] - St_T[1:]
                tensors['cva'] = (1.0 - recovery) * (
                        0.5 * (pv_exposure[1:] + pv_exposure[:-1]) * prob).mean(axis=1).sum()

                if params['Credit_Valuation_Adjustment'].get('Gradient', 'No') == 'Yes':
                    # potentially fetch ir jacobian matrices for base curves
                    base_ir_curves = [x for x in self.stoch_var.keys() if
                                      x.type == 'InterestRate' and len(x.name) == 1]
                    self.jacobians = {}
                    for ir_factor in base_ir_curves:
                        jacobian_factor = utils.Factor('InterestRateJacobian', ir_factor.name)
                        ir_curve = self.stoch_factors[utils.Factor('InterestRate', ir_factor.name)].factor
                        var_name = 'Stochastic_Input/{0}:0'.format(utils.check_tuple_name(ir_factor))
                        try:
                            jac = construct_factor(
                                jacobian_factor, self.config.params['Price Factors'],
                                self.config.params['Price Factor Interpolation'])
                            jac.update(ir_curve)
                            self.jacobians[var_name] = jac.current_value()
                            logging.info('jacobian present for {0} - will attempt inverse bootstrap'.format(
                                utils.check_tuple_name(ir_factor)))
                        except KeyError as e:
                            pass

                    # calculate all the derivatives of cva
                    hessian = params['Credit_Valuation_Adjustment'].get('Hessian', 'No') == 'Yes'
                    sensitivity = SensitivitiesEstimator(
                        tensors['cva'], self.all_var, create_graph=hessian)

                    if final_run:
                        output['grad_cva'] = sensitivity.report_grad()
                        # store the size of the Gradient
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P

                        # now fetch the CDS tenors and calculate the CDS spreads
                        CDS_tenors = params.get('Credit_Valuation_Adjustment', {}).get('CDS_Tenors')
                        if CDS_tenors and recovery < 1.0:
                            # calculate cds sensitivities
                            CDS_rates, shifted_tenor, shifted_curves = utils.calc_cds_rates(
                                recovery, survival[0], discount[0], self.params['Base_Date'],
                                CDS_tenors, self.all_factors)

                            output['CS01'] = {
                                'Par_CDS': CDS_rates,
                                'Tenor': shifted_tenor,
                                'Shifted_Log_Prob': shifted_curves
                            }

                        if hessian:
                            # calculate the hessian matrix - warning - make sure you have enough memory
                            output['grad_cva_hessian'] = sensitivity.report_hessian()

            # store all output tensors
            for k, v in tensors.items():
                output[k].append(v.cpu().detach().numpy())

            # fetch cashflows if necessary
            if self.params.get('Generate_Cashflows', 'No') == 'Yes':
                dates = np.array(sorted(self.time_grid.mtm_dates))
                for currency, values in shared_mem.t_Cashflows.items():
                    cash_index = dates[sorted(values.keys())]
                    output.setdefault('cashflows', {}).setdefault(currency, []).append(
                        pd.DataFrame(
                            [v.cpu().detach().numpy() for _, v in sorted(values.items())], index=cash_index))

            # add any scenarios if necessary
            if self.params.get('Calc_Scenarios', 'No') != 'No':
                for key, value in self.stoch_factors.items():
                    output.setdefault('scenarios', {}).setdefault(key, []).append(
                        shared_mem.t_Scenario_Buffer[key].cpu().detach().numpy())

        self.calc_stats[execution_label] = time.monotonic() - self.calc_stats[execution_label]

        # store the results
        results = {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Jacobians': self.jacobians}
        results['Results'] = self.report(output)

        return results


class Base_Reval_State(utils.Calculation_State):
    def __init__(self, static_buffer, one, mcmc_sims, report_currency, calc_greeks, gamma, nomodel='Constant'):
        super(Base_Reval_State, self).__init__(static_buffer, one, mcmc_sims, report_currency, nomodel, 1)
        self.calc_greeks = calc_greeks
        self.gamma = gamma

    @staticmethod
    def save_results(output, tensors):
        for k, v in tensors.items():
            output[k] = np.float64(v) if isinstance(v, float) else v.detach().cpu().numpy().astype(np.float64)


class Base_Revaluation(Calculation):
    """Simple deal revaluation - Use this to reconcile with the source system"""
    documentation = ('Calculations',
                     ['This applies the valuation models mentioned earlier to the portfolio per deal.',
                      '',
                      'The inputs are:',
                      '',
                      '- **Currency** of the output.',
                      '- **Run_Date** at which the marketdata should be applied (i.e. $t_0$)',
                      '- **MCMC Simulations** the number of Monte Carlo simulations to use for deals that require ',
                      '  Monte Carlo pricing (e.g. Autocalls, TARF\'s etc.)',
                      '- **Random Seed** the seed for the Monte Carlo Pricer',
                      '- **Greeks** calculate all First order sensitivities (partial derivatives) of the portfolio ',
                      '  with respect to the relevant Price Factors (Default is not to calculate this)',
                      '',
                      'The output is a dictionary containing the DealStructure and the calculation computation ',
                      'statistics.'
                      ])

    def __init__(self, config, **kwargs):
        super(Base_Revaluation, self).__init__(config, **kwargs)
        self.base_date = None

        # Cuda related variables to store the state of the device between calculations
        self.shared_memClass = namedtuple('shared_mem',
                                          't_Buffer t_Static_Buffer t_Feed_dict t_Cashflows calc_greeks \
                                          gpus riskneutral precision simulation_batch Report_Currency')

        # prepare the risk factor output matrix . .
        self.static_var = {}

    def update_factors(self, params, base_date):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, '0d', False)

        # update the time grid
        self.update_time_grid(base_date)

        self.static_factors = {}
        for price_factor in dependent_factors:
            try:
                self.static_factors.setdefault(
                    price_factor, construct_factor(
                        price_factor, self.config.params['Price Factors'],
                        self.config.params['Price Factor Interpolation'],
                        base_date=base_date))
            except KeyError as e:
                logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.args))

        self.all_factors = self.static_factors

        calc_grad = params.get('Greeks', 'No') != 'No'
        # and then get the static risk factors ready - these will just be looked up
        for key, value in self.static_factors.items():
            if key.type not in utils.DimensionLessFactors:
                current_val = value.current_value()
                if isinstance(current_val, dict):
                    for k, v in current_val.items():
                        self.static_var[utils.Factor(key.type, key.name + (k,))] = torch.tensor(
                            v, device=self.device, dtype=self.dtype, requires_grad=calc_grad)
                else:
                    self.static_var[key] = torch.tensor(
                        current_val, device=self.device, dtype=self.dtype, requires_grad=calc_grad)

        # set up the device and allocate memory
        shared_mem = self.__init_shared_mem(
            params['Currency'], params.get('MCMC_Simulations', 8 * 4096), calc_grad, params.get('Random_Seed', 1))

        # calculate a reverse lookup for the tenors and store the daycount code
        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        return shared_mem

    def update_time_grid(self, base_date):
        # set up the scenario and time grids
        self.time_grid = utils.TimeGrid({base_date}, {base_date}, {base_date})
        self.base_date = base_date
        self.time_grid.set_base_date(base_date)

    def __init_shared_mem(self, reporting_currency, mcmc_sim, calc_greeks, random_seed):
        # fix the seed if we need to price mc instruments
        torch.manual_seed(random_seed)

        # name of the base currency
        base_currency = utils.Factor(
            'FxRate', (self.config.params['System Parameters']['Base_Currency'],))

        # now decide what we want to calculate greeks with respect to
        all_vars_concat = None
        if calc_greeks:
            all_vars_concat = [x for x in self.static_var.items() if x[0] != base_currency]
            self.make_factor_index(list(self.static_var.items()))

        # allocate memory on the device
        return Base_Reval_State(
            self.static_var, torch.ones([1, 1], dtype=self.dtype, device=self.device),
            mcmc_sim, get_fxrate_factor(utils.check_rate_name(reporting_currency), self.static_factors, {}),
            all_vars_concat, self.params['Greeks'] == 'All')

    def report(self):

        def check_prices(n, parent):

            def format_row(deal, data, val, greeks):
                data['Deal Currency'] = deal.Factor_dep.get(
                    'Local_Currency', deal.Instrument.field.get('Currency', self.params['Currency']))
                try:
                    data['Ref_MTM'] = float(deal.Instrument.field.get('MtM', 0.0))
                except ValueError:
                    data['Ref_MTM'] = 0.0
                for k, v in val.items():
                    if k.startswith('Greeks'):
                        greeks.setdefault(k, []).append(
                            self.gradients_as_df(v, header=deal.Instrument.field.get('Reference'), display_val=True))
                    elif k == 'Value':
                        data[k] = float(v)
                # update any tags
                if deal.Instrument.field.get('Tags'):
                    data.update(dict(zip(tag_titles, deal.Instrument.field['Tags'][0].split(','))))

            block = []
            greeks = {}

            if 'Greeks_First' in n.obj.Calc_res:
                format_row(n.obj, dict(parent), n.obj.Calc_res, greeks)

            for sub_struct in n.sub_structures:
                data = dict(parent + [(field, sub_struct.obj.Instrument.field.get(field, 'Root'))
                                      for field in ['Reference', 'Object']])
                format_row(sub_struct.obj, data, sub_struct.obj.Calc_res, greeks)
                block.append(data)
                sub_block, sub_greeks = check_prices(
                    sub_struct, [('Parent', data['Reference'])])
                block.extend(sub_block)
                # aggregate the sub structure greeks
                for k, v in sub_greeks.items():
                    greeks.setdefault(k, []).extend(v)

            valuations = [deal.Calc_res for deal in n.dependencies]
            for deal, val in zip(n.dependencies, valuations):
                data = dict(parent + [(field, deal.Instrument.field.get(field, '?'))
                                      for field in ['Reference', 'Object']])
                format_row(deal, data, val, greeks)
                block.append(data)

            return block, greeks

        # clear the output
        self.output = {}
        # load any tag titles
        tag_titles = self.config.deals['Attributes'].get('Tag_Titles', '').split(',')
        mtm, greeks = check_prices(
            self.netting_sets, [('Parent', self.netting_sets.obj.Instrument.field.get('Reference'))])

        # calculate the grand total
        data = dict(
            [(field, self.netting_sets.obj.Instrument.field.get(field, 'Root')) for field in ['Reference', 'Object']])
        data['Value'] = sum([float(x.obj.Calc_res['Value']) for x in self.netting_sets.sub_structures])
        mtm.insert(0, data)

        # write out the dataframe
        self.output['mtm'] = pd.DataFrame(mtm)
        for greek_name, greek_val in greeks.items():
            # this guarantees that the multiindex is uniquely defined when we write it out
            if greek_name == 'Greeks_Second':
                summary = pd.concat(greek_val, axis=1).groupby(level=[0, 1, 2, 3, 4], axis=1).sum()
            elif greek_name == 'Greeks_First':
                summary = pd.concat(greek_val, axis=1).groupby(level=0, axis=1).sum()
            else:
                raise Exception('Unknown Greek requested', greek_name)
            self.output.setdefault(greek_name, summary)

        return self.output

    def execute(self, params):
        # get the rundate
        base_date = pd.Timestamp(params['Run_Date'])
        # store the params
        self.params = params
        # update the factors
        shared_mem = self.update_factors(params, base_date)
        # set the logging name
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        self.calc_stats['Deal_Setup_Time'] = time.monotonic()
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(
            self.config.deals['Deals']['Children'], self.netting_sets, deal_level_mtm=True)

        # record the (pure python) dependency setup time
        self.calc_stats['Deal_Setup_Time'] = time.monotonic() - self.calc_stats['Deal_Setup_Time']
        self.calc_stats['Graph_Setup_Time'] = time.monotonic()

        # now ask the netting set to construct each deal - no looping required (just 1 timepoint)
        mtm = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
        # record the graph loading time
        self.calc_stats['Graph_Setup_Time'] = time.monotonic() - self.calc_stats['Graph_Setup_Time']
        # populate the mtm at the netting set
        ns_obj = self.netting_sets.obj
        # make sure the netting set object has a reference and a mtm
        if ns_obj.Instrument.field.get('Reference') is None:
            ns_obj.Instrument.field['Reference'] = self.config.deals['Attributes'].get(
                'Reference', self.config.file_ref)

        if shared_mem.calc_greeks is not None:
            # record the cuda execution stats
            self.calc_stats['Greek_Execution_Time'] = time.monotonic()
            pricing.greeks(shared_mem, ns_obj, mtm)
            self.calc_stats['Greek_Execution_Time'] = time.monotonic() - self.calc_stats['Greek_Execution_Time']

        # return a dictionary of output
        return {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Results': self.report()}


@dataclass
class HedgeScenarioResult:
    """Output of ``HedgeMonteCarlo.execute()``.

    All arrays use a consistent path axis N = ``batch_size * simulation_batches``.

    Attributes
    ----------
    base_date:
        Simulation start date (t = 0).
    time_grid_days:
        Integer day offsets from ``base_date`` for each scenario step, shape [T].
    factor_paths:
        Raw stochastic factor paths keyed by ``utils.Factor``.  Shape is [T, N]
        for scalar factors (e.g. spot, carry) or [T, n_tenors, N] for curve factors.
    n_paths:
        Total number of simulated paths (``batch_size * simulation_batches``).
    n_time_steps:
        Number of scenario time steps T.
    sim_instruments:
        RiskFlow-priced MTM paths for every tradable instrument priced in the
        hedging problem,
        keyed by the instrument's ``Reference`` label.  Shape is [T, N].
        These are the primary outputs used by downstream adapters and policies.
    metadata:
        Miscellaneous run parameters (dates, batch config, calc stats).
    """
    base_date: pd.Timestamp
    business_day: pd.offsets.CustomBusinessDay
    time_grid_days: np.ndarray                       # [T]
    factor_paths: Dict['utils.Factor', np.ndarray]  # factor_key -> [T, ..., N]
    n_paths: int
    n_time_steps: int
    metadata: Dict[str, Any]
    sim_instruments: Dict[str, np.ndarray] = field(default_factory=dict)  # label -> [T, N]

    @property
    def scenario_dates(self) -> pd.DatetimeIndex:
        """DatetimeIndex corresponding to ``time_grid_days``."""
        return pd.DatetimeIndex(
            [self.base_date + pd.Timedelta(days=int(d)) for d in self.time_grid_days]
        )


@dataclass
class HedgeRuntimeExecutionResult:
    """High-level result for HedgeMonteCarlo's TorchRL bundle handoff.

    ``scenario_result`` optionally retains the full materialized scenario data.
    ``torchrl_bundle`` is the generic tensor package for downstream TorchRL
    consumption. Evaluation output is a plain dict for the active TorchRL path.
    """
    scenario_result: Optional[HedgeScenarioResult]
    torchrl_bundle: Optional[Dict[str, Any]]
    evaluation_summary: Optional[Any]
    optimizer_diagnostics: Optional[Dict[str, Any]]
    policy_artifact: Optional[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class HedgeMonteCarlo(Credit_Monte_Carlo):

    @staticmethod
    def _factor_bundle_key(factor_key):
        if hasattr(factor_key, 'type') and hasattr(factor_key, 'name'):
            factor_name = factor_key.name
            if isinstance(factor_name, tuple):
                factor_name = '.'.join(part for part in factor_name)
            return '{}:{}'.format(factor_key.type, factor_name)
        return factor_key

    def _build_scenario_result(self, base_date, business_day, time_grid_days, factor_blocks, params):
        trade_mtm = {
            x.Instrument.field['Reference']: np.concatenate(x.Calc_res['Value'], 1)
            for x in self.netting_sets.dependencies
            if x.Calc_res.get('Value')
        }
        factor_paths = {key: np.concatenate(blocks, axis=-1) for key, blocks in factor_blocks.items()}
        return HedgeScenarioResult(
            base_date=base_date,
            business_day=business_day,
            time_grid_days=time_grid_days.copy(),
            factor_paths=factor_paths,
            n_paths=self.numscenarios,
            n_time_steps=len(time_grid_days),
            sim_instruments=trade_mtm,
            metadata={
                'base_date': str(base_date),
                'time_grid': params['Time_Grid'],
                'batch_size': self.batch_size,
                'num_batches': params['Simulation_Batches'],
                'random_seed': params['Random_Seed'],
                'n_stoch_factors': self.num_factors,
                'calc_stats': dict(self.calc_stats),
            },
        )

    def _build_torchrl_bundle(self, base_date, business_day, time_grid_days, tradable_blocks, factor_tensor_blocks, hedge_profile_blocks, num_batches):
        def pad_time_axis(tensor, target_steps):
            current_steps = int(tensor.shape[0])
            if current_steps >= target_steps:
                return tensor[:target_steps]
            if current_steps == 0:
                raise ValueError('Cannot pad an empty tensor along the time axis')
            pad_shape = (target_steps - current_steps,) + tuple(tensor.shape[1:])
            pad_block = tensor[-1:].expand(*pad_shape)
            return torch.cat([tensor, pad_block], dim=0)

        tradables = {
            instrument_name: torch.cat(blocks, dim=1)
            for instrument_name, blocks in tradable_blocks.items()
        }
        factors = {
            factor_name: torch.cat(blocks, dim=-1)
            for factor_name, blocks in factor_tensor_blocks.items()
        }
        hedge_profile = None
        if hedge_profile_blocks and hedge_profile_blocks.get('features'):
            hedge_profile = {
                'features': torch.cat(hedge_profile_blocks['features'], dim=1),
                'feature_names': tuple(hedge_profile_blocks.get('feature_names', ())),
                'liability_cashflows': {
                    currency: torch.cat(blocks, dim=1)
                    for currency, blocks in hedge_profile_blocks.get('liability_cashflows', {}).items()
                },
                'liability_mtm': torch.cat(hedge_profile_blocks['liability_mtm'], dim=1),
            }
        if hedge_profile is not None:
            aligned_time_steps = int(hedge_profile['features'].shape[0])
        else:
            candidate_lengths = [int(time_grid_days.shape[0])]
            candidate_lengths.extend(int(tensor.shape[0]) for tensor in tradables.values())
            candidate_lengths.extend(int(tensor.shape[0]) for tensor in factors.values())
            aligned_time_steps = min(candidate_lengths) if candidate_lengths else 0
        bundle = {
            'time_grid_days': pad_time_axis(time_grid_days, aligned_time_steps),
            'tradables': {
                instrument_name: pad_time_axis(tensor, aligned_time_steps)
                for instrument_name, tensor in tradables.items()
            },
            'meta': {
                'base_date': base_date,
                'business_day': business_day,
                'num_batches': int(num_batches),
            },
        }
        if factors:
            bundle['factors'] = {
                factor_name: pad_time_axis(tensor, aligned_time_steps)
                for factor_name, tensor in factors.items()
            }
        if hedge_profile is not None:
            bundle['hedge_profile'] = {
                'features': hedge_profile['features'][:aligned_time_steps],
                'feature_names': hedge_profile['feature_names'],
                'liability_cashflows': {
                    currency: tensor[:aligned_time_steps]
                    for currency, tensor in hedge_profile['liability_cashflows'].items()
                },
                'liability_mtm': hedge_profile['liability_mtm'][:aligned_time_steps],
            }
        return bundle

    def update_factors(self, params, base_date, job_id, num_jobs):
        """Override: build dependent_factors from the generic Scenario_Factors JSON dict."""
        dependent_factors, stochastic_factors, _, reset_dates, _ = self.config.calculate_dependencies(
            params, base_date, self.input_time_grid)

        # Size the time grid from Futures_Expiries (or a 2-year fallback)
        max_expiry = max(reset_dates)
        reset_dates = self.config.parse_grid(base_date, max_expiry, self.input_time_grid, past_max_date=True)
        reset_dates.update({base_date, max_expiry})
        # generate scerarios at each grid date
        self.update_time_grid(base_date, reset_dates, {}, dynamic_scenario_dates=True)

        # Use the last scenario grid date so ScenarioTimeGrid covers the extra step from past_max_date=True
        last_scen_date = base_date + pd.DateOffset(days=int(self.time_grid.scen_time_grid[-1]))
        dependent_factors = {k: last_scen_date for k in dependent_factors}
        stochastic_factors, additional_factors = self.config.find_models(dependent_factors)

        return self._build_factor_state(
            dependent_factors, stochastic_factors, additional_factors, params, base_date, job_id, num_jobs)

    def execute(self, params, job_id=0, num_jobs=1):
        """Run hedging simulation batches and package the result for TorchRL.

        The calculation lifecycle is intentionally split into three phases:

        - setup before the simulation loop
        - tensor accumulation inside the simulation loop
        - bundle finalization after the loop

        RiskFlow remains responsible for:

        - simulation
        - tradable pricing
        - time grid construction

        The output handoff is a minimal dictionary-driven tensor bundle for the
        TorchRL path. Learning and policy execution are intentionally not run
        here.
        """
        def read_instruments(instruments_dict):
            instruments = []
            for obj_type, obj_data in instruments_dict.items():
                for ref, obj_field in obj_data.items():
                    ins_obj = {'Object': obj_type, 'Reference': ref}
                    ins_obj.update(obj_field)
                    instruments.append({'Instrument':construct_instrument(ins_obj, self.config.params['Valuation Configuration'])})
            return instruments

        base_date = pd.Timestamp(params['Run_Date'])
        self.input_time_grid = params['Time_Grid']
        params['Simulation_Batches'] = params['Simulation_Batches'] // num_jobs
        self.batch_size = params['Batch_Size']
        self.numscenarios = self.batch_size * params['Simulation_Batches']
        self.params = params
        # keep the mtm
        self.params['Keep_MtM'] = 'Yes'

        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        self.calc_stats['Batch_Size'] = self.batch_size
        self.calc_stats['Simulation_Batches'] = params['Simulation_Batches']
        self.calc_stats['Random_Seed'] = params['Random_Seed']

        execution_mode = params.get('Execution_Mode', 'simulate_only')
        retain_scenario_result = params.get('Retain_Scenario_Result','Yes')=='Yes'
        hedging_problem = params.get('Hedging_Problem', {})

        instruments = read_instruments(hedging_problem.get('Tradable_Instruments', {}))
        liabilities = read_instruments(hedging_problem.get('Liabilities', {}))
        # store it away for deal resolution
        self.config.deals['Deals']['Children'] = instruments + liabilities
        shared_mem = self.update_factors(params, base_date, job_id, num_jobs)
        # Build the valuation structure first; the hedging runtime will consume
        # the live factor and instrument tensors produced by this same loop.
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(instruments, self.netting_sets, deal_level_mtm=True)
        self.netting_sets.finalize_struct(base_date, self.time_grid)

        self.liabilities = DealStructure(Aggregation('contracts'), store_results=False)
        self.set_deal_structures(liabilities, self.liabilities, deal_level_mtm=False)
        self.liabilities.finalize_struct(base_date, self.time_grid)

        t_days_arr = self.time_grid.scenario_grid[:, utils.TIME_GRID_MTM]  # [T]
        execution_label = 'Tensor_Execution_Time ({})'.format(self.device.type)
        self.calc_stats[execution_label] = time.monotonic()

        normalized_runtime = construct_torchrl_runtime(params)

        # Accumulate one numpy array per stochastic factor across batches
        factor_blocks = {k: [] for k in self.stoch_factors} if retain_scenario_result else {}
        factor_tensor_blocks = {
            self._factor_bundle_key(key): [] for key in self.stoch_factors
        }
        tradable_blocks = defaultdict(list)
        hedge_profile_blocks = {
            'features': [],
            'feature_names': None,
            'liability_cashflows': defaultdict(list),
            'liability_mtm': [],
        }
        # get the calendar for business day
        bus_day = self.config.holidays.get(
            self.params['Calendar'], {'businessday': pd.offsets.BDay(1)})['businessday']

        for run in range(params['Simulation_Batches']):
            shared_mem.reset(
                self.num_factors, self.time_grid,
                use_antithetic=params.get('Antithetic', 'No') == 'Yes')

            for key, proc in self.stoch_factors.items():
                shared_mem.t_Scenario_Buffer[key] = proc.generate(shared_mem)

            _ = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
            # grab the liability features
            features = self.liabilities.resolve_hedge_structure(shared_mem, self.time_grid)
            hedge_profile_blocks['features'].append(features['features'].detach().clone())
            current_feature_names = tuple(features.get('feature_names', ()))
            if hedge_profile_blocks['feature_names'] is None:
                hedge_profile_blocks['feature_names'] = current_feature_names
            elif hedge_profile_blocks['feature_names'] != current_feature_names:
                raise ValueError('resolve_hedge_structure returned inconsistent feature_names across batches')
            for currency, payoff in features.get('liability_cashflows', {}).items():
                hedge_profile_blocks['liability_cashflows'][currency].append(payoff.detach().clone())
            hedge_profile_blocks['liability_mtm'].append(features['liability_mtm'].detach().clone())

            if retain_scenario_result:
                for key in self.stoch_factors:
                    factor_blocks[key].append(shared_mem.t_Scenario_Buffer[key].cpu().detach().numpy())

            if factor_tensor_blocks is not None:
                for key in self.stoch_factors:
                    factor_tensor_blocks[self._factor_bundle_key(key)].append(
                        shared_mem.t_Scenario_Buffer[key].detach().clone()
                    )

            # grab the simulated instruments and collect them into a generic bundle
            trade_tensors = {
                x.Instrument.field['Reference']: x.Calc_res['tensor']
                for x in self.netting_sets.dependencies
                if x.Calc_res.get('tensor') is not None
            }

            if tradable_blocks is not None:
                for instrument_name, instrument_tensor in trade_tensors.items():
                    tradable_blocks[instrument_name].append(instrument_tensor.detach().clone())

            shared_mem.t_Buffer.clear()

        self.calc_stats[execution_label] = time.monotonic() - self.calc_stats[execution_label]

        scenario_result = self._build_scenario_result(
            base_date, bus_day, t_days_arr, factor_blocks, params) if retain_scenario_result else None
        torchrl_bundle = None if normalized_runtime is None else self._build_torchrl_bundle(
            base_date,
            bus_day,
            shared_mem.one.new_tensor(t_days_arr),
            tradable_blocks,
            factor_tensor_blocks,
            hedge_profile_blocks,
            params['Simulation_Batches'],
        )
        evaluation_summary = None
        optimizer_diagnostics = None
        policy_artifact = None
        runtime_present = False
        runtime_diagnostics = {}
        optimization_result = None
        if torchrl_bundle is not None and normalized_runtime is not None:
            optimization_result = run_torchrl_execution(torchrl_bundle, normalized_runtime)
        if optimization_result is not None:
            evaluation_summary = optimization_result['evaluation_output']
            optimizer_diagnostics = optimization_result['optimizer_diagnostics']
            policy_artifact = optimization_result['policy_artifact']
            runtime_present = True
            runtime_diagnostics = {
                'num_episodes': int(evaluation_summary.get('diagnostics', {}).get('num_episodes', 0)),
                'trainer_type': evaluation_summary.get('diagnostics', {}).get('trainer_type'),
                'riskflow_simulation_pricing_time_seconds': float(self.calc_stats.get(execution_label, 0.0)),
                'torchrl_rollout_time_seconds': float((optimizer_diagnostics or {}).get('torchrl_rollout_time_seconds', 0.0)),
                'replay_storage_time_seconds': float(evaluation_summary.get('timing', {}).get('replay_storage_time_seconds', 0.0)),
                'gradient_update_time_seconds': float(evaluation_summary.get('timing', {}).get('gradient_update_time_seconds', 0.0)),
                'final_evaluation_time_seconds': float(evaluation_summary.get('timing', {}).get('evaluation_time_seconds', 0.0)),
                'total_fit_time_seconds': float(evaluation_summary.get('timing', {}).get('total_fit_time_seconds', 0.0)),
                'accounting_mode': normalized_runtime.get('accounting_mode'),
                'tradable_names': tuple(normalized_runtime.get('names', {}).get('tradables', ())),
                'cash_account_names': tuple(normalized_runtime.get('names', {}).get('cash_accounts', ())),
            }

        return HedgeRuntimeExecutionResult(
            scenario_result=scenario_result,
            torchrl_bundle=torchrl_bundle,
            evaluation_summary=evaluation_summary,
            optimizer_diagnostics=optimizer_diagnostics,
            policy_artifact=policy_artifact,
            metadata={
                'execution_mode': execution_mode,
                'torchrl_bundle_present': torchrl_bundle is not None,
                'retain_scenario_result': retain_scenario_result,
                'num_batches': params['Simulation_Batches'],
                'num_paths': self.numscenarios,
                'optimizer_diagnostics_present': optimizer_diagnostics is not None,
                'runtime_present': runtime_present,
                'runtime_diagnostics': runtime_diagnostics,
            },
        )


def construct_calculation(calc_type, config, **kwargs):
    return globals().get(calc_type)(config, **kwargs)


if __name__ == '__main__':
    pass
