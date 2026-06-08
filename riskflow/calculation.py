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

import os
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
from .torchrl_hedge import (
    run_torchrl_execution, HedgeRuntimeExecutionResult, build_torchrl_bundle,
)

# Inner-MC memory cap (M2g): generation and pricing both scale with B_outer*B_inner —
# the curve-interpolation / path slabs OOM past ~32k flat on a 10GB card. `_run_inner_mc_at_t`
# runs the whole inner MC in outer-path sub-chunks of at most this many flat samples, so
# peak memory tracks the chunk and B_outer scales freely. Overridable via the env var
# RF_INNER_MC_FLAT_LIMIT — raise it on a large GPU (fewer chunks = faster).
_INNER_MC_FLAT_LIMIT = int(os.environ.get('RF_INNER_MC_FLAT_LIMIT', 32768))


def _concat_inner_chunks(chunks, want_raw_samples):
    """Concatenate per-outer-chunk `_run_inner_mc_chunk` results back to full B_outer.
    The outer-path axis is dim 0 of every per-chunk tensor; chunk order is outer-path
    order. Scalars (`t`, `cutoff_idx`) are taken from the first chunk."""
    out = {'features': torch.cat([c['features'] for c in chunks], dim=0)}
    if want_raw_samples:
        first = chunks[0]
        out['t'], out['cutoff_idx'] = first['t'], first['cutoff_idx']
        for key in ('L_T', 'market_t', 'market_t1'):
            out[key] = torch.cat([c[key] for c in chunks], dim=0)
        for key in ('F_t1', 'dF_T', 'dF_min'):
            out[key] = {ref: torch.cat([c[key][ref] for c in chunks], dim=0)
                        for ref in first[key]}
    return out


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
            new_legs = new_features.get('legs')
            if new_legs is not None:
                if 'legs' in cumulative:
                    cumulative['legs']['features'] = torch.cat(
                        [cumulative['legs']['features'], new_legs['features']], dim=2)
                    cumulative['legs']['ids'] = cumulative['legs']['ids'] + new_legs['ids']
                else:
                    cumulative['legs'] = {
                        'features': new_legs['features'],
                        'ids': list(new_legs['ids']),
                        'feature_names': new_legs['feature_names'],
                        'id_names': new_legs['id_names'],
                    }
            new_mtm = new_features.get('mtm')
            if new_mtm is not None:
                cumulative['mtm'] = new_mtm if cumulative.get('mtm') is None else cumulative['mtm'] + new_mtm

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
                 seed, job_id, num_jobs, scale_survival=False, nomodel='Constant', keep_tensor=False):
        super(CMC_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, batch_size, keep_tensor=keep_tensor)
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
            margin = 1.0e-6
            u = sample_sobol.clamp(min=margin, max=1.0 - margin).to(self.one.device)
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


class CMC_State_Inner(CMC_State):
    """Inner-MC variant of CMC_State for nested simulation. Each of `simulation_batch`
    outer-path states fans out into `simulation_sub_batch` (B2) independent forward
    sample paths. Base `reset()` is inherited unchanged so outer-mode usage of this
    state object is transparent. `reset_inner()` swaps in the inner-shape random
    numbers: `(num_factors, T, simulation_batch, simulation_sub_batch)` instead of
    the base `(num_factors, T, simulation_batch)`. Stochastic processes dispatch on
    `Z.ndim` to pick between outer and inner code paths. Antithetic sampling is not
    supported in inner mode. quasi_rng is inherited — callers handle any reshape."""

    def __init__(self, cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                 seed, job_id, num_jobs, simulation_sub_batch=0,
                 scale_survival=False, nomodel='Constant', keep_tensor=False):
        super().__init__(cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                         seed, job_id, num_jobs, scale_survival=scale_survival,
                         nomodel=nomodel, keep_tensor=keep_tensor)
        # 0 (default) = inner mode unused; base `reset()` works, `reset_inner()` raises.
        self.simulation_sub_batch = simulation_sub_batch

    def reset_inner(self, num_factors, time_grid: utils.TimeGrid, use_antithetic=False):
        if self.simulation_sub_batch <= 1:
            raise ValueError(
                f'reset_inner requires simulation_sub_batch > 1; got {self.simulation_sub_batch}. '
                f'Pass a positive Inner_Sub_Batch in params to enable nested simulation.')
        if use_antithetic:
            raise ValueError('CMC_State_Inner.reset_inner does not support antithetic sampling.')
        T = time_grid.scen_time_grid.size
        B = self.simulation_batch
        B2 = self.simulation_sub_batch
        # Sobol-based correlated Gaussian: draw T*B*B2 quasi-normal vectors of dim num_factors,
        # transpose to (num_factors, T*B*B2), correlate via cholesky, reshape.
        Z_normal, _ = self.quasi_rng(num_factors, T * B * B2)                        # (T*B*B2, num_factors)
        correlated_sample = torch.matmul(
            self.t_cholesky, Z_normal.transpose(0, 1)
        ).reshape(num_factors, T, B, B2)
        self.t_random_numbers = correlated_sample

        self.reset_cashflows(time_grid)
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
        shared_mem = self._init_shared_mem(
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
                # Hand the process its own factor key so it can publish auxiliaries to
                # t_Scenario_Buffer under the documented (factor_key, kind) convention.
                value.factor_key = key
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
            max_date = min(max(dynamic_dates), base_date + self.config.periodparser.parseString(
                self.input_time_grid.strip().split(' ')[-1].upper())[0])
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

    def _init_shared_mem(self, seed, nomodel, reporting_currency, mcmc_sim, job_id, num_jobs, calc_greeks=None):
        # Single-underscore (overridable): HedgeMonteCarlo overrides to construct
        # CMC_State_Inner with simulation_sub_batch from params.
        if calc_greeks is not None:
            implied_vars = list(itertools.chain(*[x.items() for x in self.implied_var.values()]))
            if calc_greeks == 'Implied':
                self.all_var = implied_vars
            elif calc_greeks == 'Factors':
                self.all_var = self.stoch_var + self.static_var
            else:
                self.all_var = implied_vars + list(self.stoch_var.items()) + list(self.static_var.items())
            self.make_factor_index(self.all_var)

        scale_by_survival = (
            self.params.get('Funding_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes')

        return CMC_State(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, job_id, num_jobs, scale_by_survival, nomodel=self.params.get('NoModel', 'Constant'),
            keep_tensor=self.params.get('Keep_Tensor', 'No') == 'Yes')

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
        super(Base_Reval_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, 1, False)
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


class HedgeMonteCarlo(Credit_Monte_Carlo):
    documentation = ('Calculations', [
        'A specialisation of `Credit_Monte_Carlo` that wires the same simulated scenario',
        'engine into a TorchRL training loop. Instead of producing exposure profiles, the',
        'calculation rolls a structured policy through every simulated path, learning to',
        'hedge a portfolio of liabilities by trading a configured set of futures (or other',
        'instruments) over time. The Monte Carlo engine is unchanged — the additions are:',
        '',
        '- A **bundle** built per simulation batch containing the trajectories the policy',
        '  needs at decision time (tradable prices, liability MtM, factor history, leg',
        '  metadata, AAD-derived hedge ratios, privileged regime indicators for the critic).',
        '- A **runtime** dict normalised from the JSON `Hedging_Problem` block (action',
        '  space, position limits, cash accounts, objective, policy and optimiser configs).',
        '- A **policy** (default: structured entity transformer with optional 1D-conv',
        '  temporal encoder) that consumes the entity-tokenised bundle slice at each',
        '  decision step and emits a discrete trade per instrument.',
        '- A **PPO optimiser** that updates the policy from the rollouts. Supports',
        '  asymmetric utility rewards, dense tracking shaping, per-step penalties,',
        '  CVaR-α path-level advantage weighting, asymmetric Huber on the value head, and',
        '  an entropy floor — all gated behind config flags.',
        '',
        'The configuration contract is documented in the',
        '[Hedging_Problem](../json/index.md#calculation) section of the JSON reference.',
        '',
        '### Execution modes',
        '',
        '- `Execution_Mode = "optimize_policy"` — train the policy in-process. Returns the',
        '  trained policy artifact alongside diagnostic metrics (per-epoch losses,',
        '  reward / |trade| / entropy, advantage stats).',
        '- `Execution_Mode = "simulate_only"` — build the bundle and run a deterministic',
        '  argmax eval against a saved policy artifact (or an untrained policy if no',
        '  artifact is set). Useful for offline analysis and post-training reporting.',
        '',
        '### Output',
        '',
        '`out[\'Results\']` contains the trained policy artifact (when `optimize_policy`),',
        'the simulated bundle, the normalized runtime, and an evaluation summary with',
        'terminal P&L statistics and a position-limit audit. Wallclock and device',
        'statistics are in `out[\'Stats\']`.',
        '',
        '### Reusing the inherited Credit Monte Carlo simulator',
        '',
        'Because this class inherits from `Credit_Monte_Carlo`, the underlying scenario',
        'engine — random factor generation, instrument valuation across paths, calendar',
        'handling, AAD graph construction — is identical. Anything that can be priced for',
        'a credit-exposure run can be priced as a hedging-problem leg or tradable. The only',
        'difference is what we do with the simulated MtMs: aggregate into exposures, or',
        'feed into a learning loop.'
    ])

    @staticmethod
    def _factor_bundle_key(factor_key):
        return utils.check_tuple_name(factor_key) if hasattr(factor_key, 'type') and hasattr(factor_key, 'name') else factor_key

    def _init_shared_mem(self, seed, nomodel, reporting_currency, mcmc_sim, job_id, num_jobs, calc_greeks=None):
        """Override: HedgeMonteCarlo doesn't compute greeks or FVA, so skip the parent's
        make_factor_index / scale_survival setup. Build CMC_State_Inner directly so the
        same shared_mem hosts outer (inherited `reset()`) and inner (`reset_inner()`) modes."""
        return CMC_State_Inner(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, job_id, num_jobs,
            simulation_sub_batch=int(self.params.get('Inner_Sub_Batch', 0)),
            keep_tensor=self.params.get('Keep_Tensor', 'No') == 'Yes')

    def update_factors(self, params, base_date, job_id, num_jobs):
        """Override: build dependent_factors from the generic Scenario_Factors JSON dict."""
        dependent_factors, stochastic_factors, _, reset_dates, settlement_currencies = self.config.calculate_dependencies(
            params, base_date, self.input_time_grid)

        # Size the time grid from Futures_Expiries (or a 2-year fallback). If a
        # liability-end cap was set upstream, clip max_expiry there — hedge maturities
        # past liability end are dropped from the simulation horizon; the hedges
        # themselves will be priced through liability end and any residual position
        # closes out at fair value there.
        max_expiry = min(max(reset_dates), self.liability_end_date)
        reset_dates = self.config.parse_grid(base_date, max_expiry, self.input_time_grid, past_max_date=True)
        reset_dates.update({base_date, max_expiry})
        # generate scerarios at each grid date
        self.update_time_grid(base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=True)

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
        self.params['Keep_Tensor'] = 'Yes'

        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        self.calc_stats['Batch_Size'] = self.batch_size
        self.calc_stats['Simulation_Batches'] = params['Simulation_Batches']
        self.calc_stats['Random_Seed'] = params['Random_Seed']

        execution_mode = params.get('Execution_Mode', 'simulate_only')
        hedging_problem = params.get('Hedging_Problem', {})

        instruments = read_instruments(hedging_problem.get('Tradable_Instruments', {}))
        liabilities = read_instruments(hedging_problem.get('Liabilities', {}))
        # store it away for deal resolution
        self.config.deals['Deals']['Children'] = instruments + liabilities
        # Liability-driven time-grid cap. Design choice (not a bug): historically the
        # simulator priced every hedge instrument to its own maturity, which extended
        # the time grid to the latest hedge expiry. For hedge-MC that's wasteful —
        # past the liability terminal there is nothing to hedge, and any residual
        # hedge position is closed out at fair value at that point. Cap the global
        # time grid at the liability's last cashflow / reval date so outer and inner
        # sim both stop there. Picked up by update_factors below.
        # `reset(holidays)` populates each liability's reval_dates / settlement_currencies
        # from its `field` — this is the same call walk_groups makes a moment later, so
        # the double-reset is idempotent. Without it the attrs don't exist yet.
        liability_dates = set()
        for liab in liabilities:
            liab['Instrument'].reset(self.config.holidays)
            liability_dates |= liab['Instrument'].get_reval_dates(clip_expiry=True)
        self.liability_end_date = max(liability_dates)
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

        normalized_runtime = construct_torchrl_runtime(
            params, stoch_factors=self.stoch_factors,
        )

        # Inner-MC setup. Copies are forked after outer setup precalc has populated
        # `factor_key`/`spot0`/etc., so inner-mode precalc on the copies doesn't clobber
        # outer-instance attrs read by outer generate each batch. `shared_mem` is a
        # CMC_State_Inner — outer batches use inherited `reset()`, inner uses `reset_inner()`.
        conditional_features_blocks = [] if params.get('Inner_MC_Enabled', 'No')=='Yes' else None
        tradable_refs = sorted(normalized_runtime['names']['hedges']) if conditional_features_blocks is not None else ()

        # solve_hedge: inner MC runs in the backward DP/MPC sweep, not the outer loop.
        # Cache the per-batch outer scenario buffer so inner MC can fork on demand later.
        # gate0_fork_smoke: one-step differentiable fork from the outer-realised state at
        # a configured t (the differential-ML Gate 0 — see _gate0_fork_smoke). Needs the
        # same inner process copies and outer snapshot as solve_hedge.
        solve_hedge_mode = str(execution_mode).lower() == 'solve_hedge'
        gate0_mode = str(execution_mode).lower() == 'gate0_fork_smoke'
        gate1_mode = str(execution_mode).lower() == 'gate1_belief_check'
        if conditional_features_blocks is not None or gate0_mode or gate1_mode:
            self.stoch_factors_inner = {k: proc.copy() for k, proc in self.stoch_factors.items()}
        outer_state_blocks = defaultdict(list) if (solve_hedge_mode or gate0_mode or gate1_mode) else None

        factor_tensor_blocks = {
            self._factor_bundle_key(key): [] for key in self.stoch_factors
        }
        tradable_blocks = defaultdict(list)
        hedge_profile_blocks = {
            'legs_features': [],
            'legs_ids': None,
            'legs_feature_names': None,
            'legs_id_names': None,
            'mtm': [],
            # Per-(tradable, factor) hedge ratios (in *contract* units), computed per-batch in
            # the sim loop directly from liability and tradable AAD deltas — no separate
            # spot_deltas accumulator needed since the ratio is the only thing the policy reads.
            'hedge_ratios': defaultdict(list),
            'realized_cashflows': defaultdict(list),
        }
        # Per-batch privileged-factor accumulator. Keyed by (factor_name, factor_attr) where
        # factor_attr is whatever the process exposes via `privileged_factors()`. Concatenated
        # along the batch dim after all simulation batches finish.
        privileged_factor_blocks = defaultdict(list)
        # get the calendar for business day
        bus_day = self.config.holidays.get(
            self.params['Calendar'], {'businessday': pd.offsets.BDay(1)})['businessday']
        # Identify which factors correspond to user-declared underlyings.
        # Spot_Price_History is keyed by commodity name; the matching factor key is
        declared_underlyings = self.params['Hedging_Problem']['Portfolio_State'].get('Spot_Price_History',{}).keys()

        for run in range(params['Simulation_Batches']):
            shared_mem.reset(
                self.num_factors, self.time_grid,
                use_antithetic=params.get('Antithetic', 'No') == 'Yes')

            for key, proc in self.stoch_factors.items():
                simulated = proc.generate(shared_mem)
                if utils.check_tuple_name(key) in declared_underlyings:
                    simulated = simulated.detach().requires_grad_(True)
                shared_mem.t_Scenario_Buffer[key] = simulated
                # Each process owns its privileged-factor surface; ask it what to expose. Default
                # implementation returns {} so processes opt in by overriding privileged_factors.
                priv = proc.privileged_factors(simulated)
                if priv:
                    factor_name = key.name[0] if key.name else str(key)
                    for attr_name, tensor in priv.items():
                        privileged_factor_blocks[(factor_name, attr_name)].append(tensor.detach().clone())

            # solve_hedge: snapshot this batch's outer scenario buffer (factor paths +
            # the (spot_key,'regimes') aux key) for on-demand inner-MC forking later.
            if outer_state_blocks is not None:
                for key, tensor in shared_mem.t_Scenario_Buffer.items():
                    outer_state_blocks[key].append(tensor.detach().clone())

            _ = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
            # clear hedge cashflows so t_Cashflows after the next call holds only liability cashflows
            shared_mem.reset_cashflows(self.time_grid)
            # grab the liability features
            features = self.liabilities.resolve_hedge_structure(shared_mem, self.time_grid)
            legs = features.get('legs')
            if legs is not None:
                hedge_profile_blocks['legs_features'].append(legs['features'].detach().clone())
                if hedge_profile_blocks['legs_ids'] is None:
                    hedge_profile_blocks['legs_ids'] = list(legs['ids'])
                    hedge_profile_blocks['legs_feature_names'] = legs['feature_names']
                    hedge_profile_blocks['legs_id_names'] = legs['id_names']
            mtm = features.get('mtm')
            liability_spot_deltas = None
            if mtm is not None:
                liability_spot_deltas = pricing.extract_spot_deltas(mtm, shared_mem.t_Scenario_Buffer)
                hedge_profile_blocks['mtm'].append(mtm.detach().clone())

            mtm_grid_size = self.time_grid.mtm_time_grid.size
            for currency, by_time in (shared_mem.t_Cashflows or {}).items():
                dense = shared_mem.one.new_zeros(mtm_grid_size, shared_mem.simulation_batch)
                for t_idx, payoff in by_time.items():
                    dense[int(t_idx)] = payoff
                hedge_profile_blocks['realized_cashflows'][str(currency)].append(dense.detach().clone())

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
                    # Hedge ratio per (tradable, factor) in *contract* units:
                    #     ratio_contracts = (dL/dS) / (dF_i/dS) / contract_size_i
                    # i.e. "how many contracts of this tradable to short to delta-flatten the
                    # liability against this factor at this (t, b)". AAD-extracted before the
                    # .detach() below. Cash accounts and other constant-tensor instruments have
                    # no backward graph — skipped (their hedge_ratio is undefined anyway).
                    if instrument_tensor.requires_grad and liability_spot_deltas:
                        td_deltas = pricing.extract_spot_deltas(instrument_tensor, shared_mem.t_Scenario_Buffer)
                        contract_size = float(
                            normalized_runtime['tradables'][instrument_name].get('contract_size', 1.0)
                        ) or 1.0
                        ratios = {}
                        T_td = instrument_tensor.shape[0]
                        for factor_name, td_delta in td_deltas.items():
                            liab_delta = liability_spot_deltas.get(factor_name)
                            if liab_delta is None:
                                continue
                            # Liability MtM lives over the full deal period; tradable lives until
                            # its expiry (T_td <= T_liab). Crop liability to the tradable's grid.
                            liab_aligned = liab_delta[:T_td]
                            nonzero = td_delta.abs() > 1e-6
                            safe_td = torch.where(nonzero, td_delta, torch.ones_like(td_delta))
                            ratios[factor_name] = torch.where(
                                nonzero,
                                liab_aligned / (safe_td * contract_size),
                                torch.zeros_like(liab_aligned),
                            )
                        hedge_profile_blocks['hedge_ratios'][instrument_name].append(ratios)
                    tradable_blocks[instrument_name].append(instrument_tensor.detach().clone())

            # RL path only: precompute conditional features during the outer loop.
            # solve_hedge defers inner MC to the backward sweep (see inner_mc_fn below).
            if conditional_features_blocks is not None and not solve_hedge_mode:
                conditional_features_blocks.append(self._run_inner_mc_pass(
                    run, shared_mem, base_date, params, tradable_refs).cpu())

            shared_mem.t_Buffer.clear()

        self.calc_stats[execution_label] = time.monotonic() - self.calc_stats[execution_label]

        if outer_state_blocks is not None:
            # Concatenate per-batch outer scenario snapshots along the batch (last) dim:
            # spot (T, B), curve (T, n_tenors, B), regimes (T, B) all carry B last.
            self._outer_scenario_buffer = {
                key: torch.cat(blocks, dim=-1) for key, blocks in outer_state_blocks.items()
            }

        if gate0_mode:
            # Differential-ML Gate 0: one-step differentiable fork smoke. No bundle, no
            # solver, no runtime — the simulator's per-path init grad-pathway is the only
            # thing under test. Return early with the diagnostic in evaluation_summary.
            gate0_result = self._gate0_fork_smoke(params, shared_mem, base_date)
            return HedgeRuntimeExecutionResult(
                torchrl_bundle=None,
                runtime=None,
                evaluation_summary={'gate0': gate0_result},
                optimizer_diagnostics=None,
                policy_artifact=None,
                metadata={
                    'execution_mode': execution_mode,
                    'torchrl_bundle_present': False,
                    'num_batches': params['Simulation_Batches'],
                    'num_paths': self.numscenarios,
                    'optimizer_diagnostics_present': False,
                    'runtime_present': False,
                    'runtime_diagnostics': {},
                },
            )

        if gate1_mode:
            # Differential-ML Gate 1: forward HMM belief filter — (a) calibration check
            # (bucket by predicted belief, empirical regime frequency matches) AND (b)
            # discriminative-belief check (V_r per pure regime at an interior t — if the
            # spread of V_r is meaningful, belief is informative for hedging value).
            gate1_result = self._gate1_belief_check(
                params, shared_mem, base_date, privileged_factor_blocks)
            return HedgeRuntimeExecutionResult(
                torchrl_bundle=None,
                runtime=None,
                evaluation_summary={'gate1': gate1_result},
                optimizer_diagnostics=None,
                policy_artifact=None,
                metadata={
                    'execution_mode': execution_mode,
                    'torchrl_bundle_present': False,
                    'num_batches': params['Simulation_Batches'],
                    'num_paths': self.numscenarios,
                    'optimizer_diagnostics_present': False,
                    'runtime_present': False,
                    'runtime_diagnostics': {},
                },
            )

        torchrl_bundle = None if normalized_runtime is None else build_torchrl_bundle(
            base_date,
            bus_day,
            shared_mem.one.new_tensor(t_days_arr),
            tradable_blocks,
            factor_tensor_blocks,
            hedge_profile_blocks,
            params['Simulation_Batches'],
            self.stoch_factors,
            runtime=normalized_runtime,
            privileged_factor_blocks=privileged_factor_blocks,
        )
        if torchrl_bundle is not None and conditional_features_blocks:
            torchrl_bundle['conditional_features'] = torch.cat(
                conditional_features_blocks, dim=1)
            torchrl_bundle['conditional_feature_names'] = [
                'prob_loss',
                'expected_loss_given_loss',
                'variance_loss_given_loss',
            ] + [
                f'{stat}_{ref}'
                for ref in tradable_refs
                for stat in ('expected_terminal_move_given_loss', 'expected_min_move_given_loss')
            ]
        if torchrl_bundle is not None and solve_hedge_mode:
            # Closure lets the solver fork inner MC on demand without a calc handle —
            # captures `self` (the inner-MC machinery), the cached outer state, shared_mem.
            torchrl_bundle['inner_mc_fn'] = lambda t: self._run_inner_mc_at_t(
                t, self._outer_scenario_buffer, shared_mem, base_date, tradable_refs)
            # Grad-enabled twin: per-process state-at-t leaves attached so the solver can
            # `.backward()` from any function of inner outputs back to state-at-t and read
            # gradient labels (Phase 3b deep, differential ML twin-loss).
            torchrl_bundle['inner_mc_grad_fn'] = lambda t: self._run_inner_mc_at_t(
                t, self._outer_scenario_buffer, shared_mem, base_date, tradable_refs,
                with_grad=True)
            # One-step variant for the diff-ML twin-loss bootstrap label gen — fork
            # at `t`, advance ONE step to `t+1` under autograd, skip the pricing chain
            # (the deal pricer needs the full horizon and would produce meaningless
            # `L_T` against a 2-point grid). Returns `market_t`, `market_t1`, and the
            # per-process `state_t_leaves` autograd leaves. Restricting the AAD tape to
            # a single step is what keeps the spec's twin-loss architecture on a 3090.
            torchrl_bundle['inner_mc_grad_fn_one_step'] = lambda t: self._run_inner_mc_at_t(
                t, self._outer_scenario_buffer, shared_mem, base_date, tradable_refs,
                with_grad=True, max_inner_steps=1, return_market_only=True)
            # Calc handle for DifferentialSolver's `sample_exogenous` seam (gate4
            # diff-ML build): the solver reads `_outer_scenario_buffer` + uses
            # `_extract_outer_state_at` directly; no monkey-patching of framework
            # primitives. **TODO refactor (A):** promote `sample_exogenous` to a
            # `HedgeMonteCarlo` bundle source (i.e., emit per-t exogenous slices into
            # `torchrl_bundle` so the solver doesn't reach back into the calc) — until
            # then this attribute lets the solver read what's already there without
            # owning the buffer.
            torchrl_bundle['_calc_handle'] = self
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
            torchrl_bundle=torchrl_bundle,
            runtime=normalized_runtime,
            evaluation_summary=evaluation_summary,
            optimizer_diagnostics=optimizer_diagnostics,
            policy_artifact=policy_artifact,
            metadata={
                'execution_mode': execution_mode,
                'torchrl_bundle_present': torchrl_bundle is not None,
                'num_batches': params['Simulation_Batches'],
                'num_paths': self.numscenarios,
                'optimizer_diagnostics_present': optimizer_diagnostics is not None,
                'runtime_present': runtime_present,
                'runtime_diagnostics': runtime_diagnostics,
            },
        )

    # ------------------------------------------------------------------
    # Inner-MC subsystem
    #
    # Parallel to the outer-loop body inlined in `execute()`. Forks the simulator
    # from each outer-path state at each outer timestep, runs inner MC to terminal
    # under `no_grad`, reduces to conditional features per outer path. Outer process
    # instances are not touched — inner uses shallow copies (`StochasticProcess.copy`)
    # so per-instance precalc state (spot0, scenario_horizon, z_offset, ...) doesn't
    # bleed across the outer/inner boundary.
    # ------------------------------------------------------------------

    def _find_spot_key(self):
        """Return the unique CommodityPrice factor key from self.stoch_factors. Raises
        if zero or multiple — v1 inner-MC is specified for a single-spot regime."""
        spot_keys = [k for k in self.stoch_factors if k.type == 'CommodityPrice']
        if len(spot_keys) != 1:
            raise ValueError(
                f'Inner MC expects exactly one CommodityPrice factor; found {len(spot_keys)}: {spot_keys}'
            )
        return spot_keys[0]

    def _get_implied_for(self, key):
        """Mirror of the outer `_build_factor_state` implied-tensor derivation. Returns
        None if the factor has no implied variables."""
        if key in self.implied_var:
            return {k.name[-1]: v for k, v in self.implied_var[key].items()}
        return None

    def _extract_outer_state_at(self, t, key, scenario_buffer, privileged=False):
        """Per-factor state for `key` at time index `t`, sliced from `scenario_buffer`
        with the batch axis trailing.

        `privileged=False` (default): the raw factor state — the shape each process's
        `precalculate` expects as its per-path initial state.
        `privileged=True`: the informative market-state representation the value function
        consumes — the regime posterior (one-hot of the regime path the regime-switching
        process publishes under `(key, 'regimes')`) for a regime-switching spot, since
        the raw price is redundant with the tradable futures; the carry curve as-is for
        a rate factor. Extend the per-type dispatch here to support a new factor type —
        the DP / MPC solvers stay factor-agnostic."""
        outer = scenario_buffer[key]
        if key.type in ('CommodityPrice', 'CommodityBasis'):
            regimes = scenario_buffer.get((key, 'regimes'))
            proc = self.stoch_factors_inner.get(key)
            if privileged and regimes is not None and getattr(proc, 'n_states', None):
                # Phase 3c: prefer the filtered belief `P(regime_t | prices_{0..t})` when
                # available AND the buffer's shape matches the current mode. Outer mode:
                # regimes (T, B), belief (T, n_states, B) → belief.dim() == regimes.dim() + 1.
                # Inner mode: regimes (T, B, B2), belief still outer-shape (no inner filter)
                # → dim mismatch ⇒ fall back to the inner-regime one-hot (degenerate point-
                # mass belief at the sampled regime; the V̂ was trained on belief vectors
                # but a one-hot is at the boundary of belief space, so the query is well-
                # defined though not statistically pure).
                belief = scenario_buffer.get((key, 'regime_belief'))
                if belief is not None and belief.dim() == regimes.dim() + 1:
                    return belief[t]                       # (n_states, ...batch) — B last
                onehot = torch.nn.functional.one_hot(
                    regimes[t].long(), num_classes=proc.n_states).to(dtype=outer.dtype)
                return onehot.movedim(-1, 0)               # (n_states, ...batch)
            return outer[t, :]
        if key.type == 'ForwardRate':
            return outer[t, :, :]                          # carry-factor curve (compact)
        if key.type in ('ForwardPrice', 'InterestRate'):
            # In privileged mode these curves are excluded from the V̂ market state —
            # their hedging-relevant content is already carried by the tradable futures
            # prices in the deep state, and dumping every tenor bloats the V̂ basis past
            # the training-row count. Raw mode (the precalculate fork state) is unchanged.
            return outer[t, :0, :] if privileged else outer[t, :, :]
        raise NotImplementedError(
            f'Inner MC has no per-path initial-state extractor for factor type {key.type!r}'
        )

    def _gate0_fork_smoke(self, params, shared_mem, base_date):
        """Gate 0 of the differential-ML build (see differential_ml_redesign_v14.md §8 / 6.3).

        Smoke-tests that the simulator can fork one step from an arbitrary per-path
        observable state under autograd. PASS = (a) finite pathwise differentials of the
        one-step factor outputs w.r.t. each process's initial state; (b) VAR `X_0`
        round-trip guard holds (enforced in `precalculate` itself — exception propagates);
        (c) the spot/regime/basis block forks jointly (basis reads the linked spot &
        regimes from the buffer via `dependant_fields` topo order).

        Designer-sampled states come from the outer scenario buffer (the simulator's
        forward marginal at `t` from today's market — a correct joint draw across all
        factors). For each, the per-process initial state is detached + re-attached as
        a leaf with `requires_grad=True`, then the inner-mode `generate()` runs with
        autograd ON. No pricing, no bundle, no solver.
        """
        gate0_cfg = params.get('Hedging_Problem', {}).get('Gate0', {})
        scen_t = self.time_grid.scen_time_grid
        t = int(gate0_cfg.get('Fork_T_Index', scen_t.size // 2))
        if not (0 <= t < scen_t.size - 1):
            raise ValueError(
                f'Gate0.Fork_T_Index={t} out of range [0, {scen_t.size - 1}) — needs at '
                f'least one outer step ahead to fork into.')
        grad_finite_tol = float(gate0_cfg.get('Grad_Finite_Tolerance', 1.0e6))

        spot_key = self._find_spot_key()
        outer_buf = self._outer_scenario_buffer
        B_outer = outer_buf[spot_key].shape[-1]
        B_inner = shared_mem.simulation_sub_batch
        if B_inner < 2:
            raise ValueError(
                f'Gate 0 requires Inner_Sub_Batch >= 2 (current: {B_inner}). The inner-MC '
                f'random-number setup requires B_inner >= 2; set Inner_Sub_Batch in the JSON.')

        t_days = int(scen_t[t])
        inner_time_grid = self.time_grid.truncate_to(base_date, t_days)
        inner_base_date = base_date + pd.Timedelta(days=t_days)
        if inner_time_grid.scen_time_grid.size < 2:
            raise ValueError(
                f'Gate0.Fork_T_Index={t} leaves no inner horizon (truncated grid size '
                f'{inner_time_grid.scen_time_grid.size}); pick an earlier t.')

        shared_mem.simulation_batch = B_outer
        shared_mem.reset_inner(self.num_factors, inner_time_grid)
        shared_mem.t_Scenario_Buffer[(spot_key, 'regime0_inner')] = outer_buf[(spot_key, 'regimes')][t]

        leaves = {}
        outputs = {}

        # Generate inner step per process, autograd ON. Topo order matches outer (basis
        # depends on linked spot + regimes — `dependant_fields` enforces).
        for key, proc_inner in self.stoch_factors_inner.items():
            if key.type in utils.DimensionLessFactors:
                continue
            init_state = self._extract_outer_state_at(t, key, outer_buf).detach().clone()
            init_state.requires_grad_(True)
            leaves[key] = init_state

            # VAR X_0 round-trip is enforced inside `precalculate` itself (stochastic
            # process raises if max |M·X_0 - curve_0| exceeds tolerance, ~1e-5 fp32 /
            # 1e-10 fp64) — reaching the next line means that gate passed.
            proc_inner.precalculate(
                inner_base_date, inner_time_grid, init_state, shared_mem, self.process_ofs[key],
                implied_tensor=self._get_implied_for(key),
            )
            simulated = proc_inner.generate(shared_mem)
            shared_mem.t_Scenario_Buffer[key] = simulated
            # Inner-time index 0 = initial state, index 1 = one step ahead. Take t+1 slice.
            outputs[key] = simulated[1] if simulated.dim() >= 1 else simulated

        # Joint-fork sanity: by the time we reach here, basis (if present) has run, which
        # required the linked spot + regimes from the buffer. So consistency is asserted by
        # construction; just record what was visible at the time of basis.generate.
        basis_keys = [k for k in outputs if k.type == 'CommodityBasis']
        joint_fork_consistency = {
            'spot_in_buffer_at_basis_call': spot_key in shared_mem.t_Scenario_Buffer,
            'regime_aux_in_buffer_at_basis_call': (spot_key, 'regimes') in shared_mem.t_Scenario_Buffer,
            'basis_factors_present': [str(k.name) for k in basis_keys],
        }

        # Scalar output to backprop from. Any non-finite output is a PASS-blocker on its
        # own — record before backward so we still get partial diagnostics on the leaves.
        nonfinite_outputs = []
        output_sum = None
        for key, out in outputs.items():
            if not torch.isfinite(out).all():
                nonfinite_outputs.append(str(key.name))
            out_sum = out.sum()
            output_sum = out_sum if output_sum is None else output_sum + out_sum

        notes = []
        pass_overall = True

        if output_sum is None:
            return {
                'pass_overall': False, 'fork_t_index': t,
                'B_outer': int(B_outer), 'B_inner': int(B_inner),
                'per_process': {}, 'joint_fork_consistency': joint_fork_consistency,
                'notes': ['no factor outputs produced — empty stoch_factors_inner?'],
            }

        if nonfinite_outputs:
            pass_overall = False
            notes.append(f'non-finite factor outputs at t+1: {nonfinite_outputs}')
            # Still attempt backward; non-finite outputs will produce non-finite grads,
            # which the per-process loop below will record.

        output_sum.backward()

        per_process = {}
        for key, leaf in leaves.items():
            g = leaf.grad
            factor_name = '.'.join(key.name) if key.name else str(key)
            rec = {
                'factor_type': key.type,
                'factor_name': factor_name,
                'init_state_shape': tuple(leaf.shape),
            }
            if g is None:
                rec.update({'grad_finite': False, 'grad_norm_mean': None,
                            'grad_norm_max': None, 'grad_zero_frac': None})
                pass_overall = False
                notes.append(f'{factor_name} ({key.type}): grad is None')
            else:
                g_flat = g.flatten()
                finite_mask = torch.isfinite(g_flat)
                all_finite = bool(finite_mask.all().item())
                if not all_finite:
                    pass_overall = False
                    notes.append(
                        f'{factor_name} ({key.type}): {int((~finite_mask).sum().item())}/'
                        f'{g_flat.numel()} non-finite grad entries')
                g_abs = g_flat.abs().nan_to_num(0.0)
                max_norm = float(g_abs.max().item())
                if max_norm > grad_finite_tol:
                    pass_overall = False
                    notes.append(
                        f'{factor_name} ({key.type}): max |grad| = {max_norm:.3e} > tol '
                        f'{grad_finite_tol:.3e}')
                # All-zero grad on a factor that physically influences the one-step output
                # is also a smoke failure (means the autograd path was severed somewhere).
                zero_frac = float((g_abs == 0).to(dtype=torch.float32).mean().item())
                if zero_frac > 0.999:
                    pass_overall = False
                    notes.append(
                        f'{factor_name} ({key.type}): grad is ~all-zero ({zero_frac:.3%}) — '
                        f'autograd path severed?')
                rec.update({
                    'grad_finite': all_finite,
                    'grad_norm_mean': float(g_abs.mean().item()),
                    'grad_norm_max': max_norm,
                    'grad_zero_frac': zero_frac,
                })
            per_process[factor_name] = rec

        # Clean buffer state so this call doesn't leak into subsequent execute() runs.
        shared_mem.t_Buffer.clear()
        shared_mem.t_Scenario_Buffer.clear()
        shared_mem.t_quasi_rng.clear()

        return {
            'pass_overall': pass_overall,
            'fork_t_index': t,
            't_days': t_days,
            'B_outer': int(B_outer),
            'B_inner': int(B_inner),
            'per_process': per_process,
            'joint_fork_consistency': joint_fork_consistency,
            'notes': notes,
        }

    def _gate1_belief_check(self, params, shared_mem, base_date, privileged_factor_blocks):
        """Gate 1 of the differential-ML build (see differential_ml_redesign_v14.md §7-R1b / §8).

        Two checks on the forward HMM belief filter `P(regime_t | prices_{0..t})` that
        replaces the privileged true regime as `market_t`'s regime coordinate in the new
        architecture:

        (a) **Calibration** — bucket all (t, path) pairs by `belief[r]` (each regime
            independently); within each bucket the empirical frequency of (true regime
            == r) should approximate the bucket midpoint. A miscalibrated filter would
            confidently predict the wrong regime — diagnoses filter-bugs that produce
            output but not truth.

        (b) **Discriminative belief / non-trivial gradient** — the value function `C_t`
            is approximately linear in belief locally (decomposition
            `C_t(belief) ≈ Σ_r belief[r] · V_r` over a fixed observable state, where
            V_r = E[U(W_T) | regime_t = r, other state held fixed]). So `∂C/∂belief[r] = V_r`
            and the gradient is non-trivial iff V_r spread across r is non-trivial. We
            compute V_r by inner-MC fork with `regime0_inner` forced constant per r at
            an interior `t`, taking the symlog utility of the realized terminal liability
            under the null hedge policy. Belief entropy distribution across paths is a
            cheaper precondition (uniform-everywhere belief → V_r decomposition is moot).
        """
        gate1_cfg = params.get('Hedging_Problem', {}).get('Gate1', {})
        n_buckets = int(gate1_cfg.get('Calibration_Buckets', 10))
        min_samples = int(gate1_cfg.get('Calibration_Min_Samples', 50))
        max_bucket_err = float(gate1_cfg.get('Calibration_Max_Abs_Err', 0.05))
        min_v_r_spread_rel = float(gate1_cfg.get('Discriminative_Min_V_Spread_Rel', 0.05))

        spot_key = self._find_spot_key()
        spot_name = spot_key.name[0] if spot_key.name else str(spot_key)

        belief_blocks = privileged_factor_blocks.get((spot_name, 'regime_belief'), [])
        onehot_blocks = privileged_factor_blocks.get((spot_name, 'regime_onehot'), [])
        if not belief_blocks or not onehot_blocks:
            return {
                'pass_overall': False,
                'notes': [f'missing privileged blocks for spot {spot_name!r}: '
                          f'belief_blocks={len(belief_blocks)}, onehot_blocks={len(onehot_blocks)}'],
            }

        belief = torch.cat(belief_blocks, dim=1)                       # (T, B, n_states)
        onehot = torch.cat(onehot_blocks, dim=1)                       # (T, B, n_states)
        T, B, n_states = belief.shape

        # (a) Calibration check — marginal (gating) + per-bucket (diagnostic).
        # Marginal calibration is the necessary property: E[belief[r]] should equal
        # empirical P(true regime = r) by the tower property of conditional expectation.
        # Per-bucket calibration is a finer (sufficient) check; for rare regimes it can
        # legitimately miss at mid-confidence buckets (the optimal Bayesian filter trades
        # off prior + likelihood, and the rare-regime prior dominates differently across
        # buckets — see ML R1b discussion). We report both; gating is on marginal +
        # extreme-bucket sanity (close-to-the-decision-boundary buckets at the tails).
        max_marginal_err = float(gate1_cfg.get('Calibration_Max_Marginal_Err', 0.02))
        marginal_calibration = {}
        pass_calibration = True
        notes = []
        for r in range(n_states):
            mean_belief = float(belief[..., r].mean().item())
            emp_freq = float(onehot[..., r].mean().item())
            err = abs(mean_belief - emp_freq)
            marginal_calibration[r] = {
                'mean_belief_r': mean_belief, 'empirical_freq_r': emp_freq, 'abs_err': err,
            }
            if err > max_marginal_err:
                pass_calibration = False
                notes.append(
                    f'regime {r}: marginal calibration error {err:.4f} > tol {max_marginal_err:.4f} '
                    f'(E[belief]={mean_belief:.4f} vs empirical={emp_freq:.4f})')

        # Per-bucket detail (diagnostic only — does not gate PASS).
        calibration_per_regime = {}
        edges = torch.linspace(0.0, 1.0, n_buckets + 1)
        for r in range(n_states):
            b_r = belief[..., r].flatten()
            true_r = onehot[..., r].flatten()
            per_bucket = {}
            for k in range(n_buckets):
                lo, hi = float(edges[k]), float(edges[k + 1])
                mask = (b_r >= lo) & ((b_r <= hi) if k == n_buckets - 1 else (b_r < hi))
                n_in = int(mask.sum().item())
                mid = (lo + hi) / 2.0
                if n_in < min_samples:
                    per_bucket[round(mid, 3)] = {'n': n_in, 'emp': None, 'err': None}
                    continue
                emp = float(true_r[mask].mean().item())
                per_bucket[round(mid, 3)] = {'n': n_in, 'emp': emp, 'err': abs(emp - mid)}
            calibration_per_regime[r] = per_bucket

        # Belief entropy: how localized is the filter?
        p = belief.clamp_min(1.0e-30)
        H = -(p * p.log()).sum(dim=-1)                                # (T, B)
        H_flat = H.flatten()
        H_max = float(np.log(n_states))
        belief_entropy_stats = {
            'H_max': H_max,
            'mean': float(H_flat.mean().item()),
            'p10': float(H_flat.quantile(0.10).item()),
            'p50': float(H_flat.quantile(0.50).item()),
            'p90': float(H_flat.quantile(0.90).item()),
            'frac_localized_below_95pct_H_max': float(
                (H_flat < 0.95 * H_max).to(dtype=torch.float32).mean().item()),
        }

        # (b) Discriminative check: V_r at t_eval per pure regime, via inner-MC fork.
        # t_eval defaults to T-2 (one inner step ahead exists; equivalently T_dec-1 for
        # a deal whose last fixing matches the horizon).
        scen_t = self.time_grid.scen_time_grid
        t_eval = int(gate1_cfg.get('Discriminative_T_Index', scen_t.size - 2))
        if not (0 <= t_eval < scen_t.size - 1):
            raise ValueError(
                f'Gate1.Discriminative_T_Index={t_eval} out of range [0, {scen_t.size - 1}).')

        outer_buf = self._outer_scenario_buffer
        outer_regimes = outer_buf[(spot_key, 'regimes')]               # (T, B) long
        # Build per-regime versions of the buffer that override outer_regimes[t_eval] to a
        # constant value of r. `_run_inner_mc_at_t` reads `outer_regimes[t_eval]` and feeds
        # it into `(spot_key, 'regime0_inner')` for the inner HMM, so the override propagates.
        L_T_per_regime = []
        for r in range(n_states):
            outer_buf_r = dict(outer_buf)
            regimes_overridden = outer_regimes.clone()
            regimes_overridden[t_eval] = r
            outer_buf_r[(spot_key, 'regimes')] = regimes_overridden
            res = self._run_inner_mc_at_t(
                t_eval, outer_buf_r, shared_mem, base_date,
                tradable_refs=(), want_raw_samples=True)
            L_T_per_regime.append(res['L_T'])                          # (B_outer, B_inner)

        # Shared utility scale across regimes: median(|L_T|) over all paths × regimes,
        # floored to $1k (matches torchrl_hedge.resolve_utility_scale's degenerate-floor
        # behaviour). c shared ⇒ V_r comparisons are on a common monotone transform.
        all_abs = torch.cat([l.abs().flatten() for l in L_T_per_regime])
        c = max(float(all_abs.median().item()), 1.0e3)

        v_r_per_regime = []
        for L_T in L_T_per_regime:
            u = torch.sign(L_T) * torch.log1p(L_T.abs() / c)
            v_r_per_regime.append(float(u.mean().item()))

        v_max, v_min = max(v_r_per_regime), min(v_r_per_regime)
        denom = max(abs(v_max), abs(v_min), 1.0e-30)
        v_r_spread_rel = (v_max - v_min) / denom
        pass_discriminative = v_r_spread_rel > min_v_r_spread_rel
        if not pass_discriminative:
            notes.append(
                f'V_r spread {v_r_spread_rel:.3f} <= tol {min_v_r_spread_rel:.3f} — '
                f'belief is degenerate w.r.t. value (uninformative for hedging)')

        return {
            'pass_overall': pass_calibration and pass_discriminative,
            'pass_calibration': pass_calibration,
            'pass_discriminative': pass_discriminative,
            'T': int(T), 'B': int(B), 'n_states': int(n_states),
            'marginal_calibration': marginal_calibration,
            'calibration_per_regime': calibration_per_regime,
            'belief_entropy_stats': belief_entropy_stats,
            'discriminative_t_index': t_eval,
            'utility_scale_c': c,
            'v_r_per_regime': v_r_per_regime,
            'v_r_spread_abs': v_max - v_min,
            'v_r_spread_rel': v_r_spread_rel,
            'notes': notes,
        }

    def _calc_inner_features(self, inner_mtm, inner_trade_tensors, tradable_refs, cutoff_idx):
        """Per-outer-path features from inner-MC outputs: P(L_T<0), E[L_T|loss],
        Var[L_T|loss], and per-tradable E[F_T-F_t|loss] / E[min_s F_s-F_t|loss].
        Output `(B_outer, 3 + 2*len(tradable_refs))`. `clamp_min(1)` sentinel: zero-loss
        paths yield all-zero features; `prob_loss == 0` flags them. Pure function — also
        reusable as a standalone diagnostic.

        Sign convention: `liability` MTM follows the deal's natural Payer/Receiver sign
        (see `_evaluate_objective`: net_pnl = pnl_excess + liability). For a Receiver,
        L > 0 = money flows in = profit; L < 0 = loss. The hedger conditions on the
        loss tail, so the mask is `L_T < 0`.

        `inner_mtm[-2]` (not [-1]): the time-grid construction appends one extra date
        past the liability terminal to show the netting set settling to zero ("clean
        exit"). [-1] is that zero; [-2] is the pre-settlement terminal MTM.

        Tradable tensors are sliced at `cutoff_idx` — the post-gather per-deal tensor
        covers `[0, expiry]` over the full outer interp, and positions `[0, cutoff_idx)`
        hold garbage from the post-gather interpolation against a sliced deal_time_grid
        (legitimately unused — we never read them). The valid future is `[cutoff_idx:]`."""
        L_T = inner_mtm[-2]                                            # (B_outer, B_inner)
        loss_mask = (L_T < 0).to(dtype=L_T.dtype)                      # float, same dtype as L_T
        loss_count = loss_mask.sum(dim=-1).clamp_min(1)

        prob_loss = loss_mask.mean(dim=-1)                             # (B_outer,)
        expected_loss_given_loss = (L_T * loss_mask).sum(dim=-1) / loss_count
        centered = (L_T - expected_loss_given_loss.unsqueeze(-1)) * loss_mask
        variance_loss_given_loss = (centered * centered).sum(dim=-1) / loss_count

        per_contract = []
        for ref in tradable_refs:
            td = inner_trade_tensors.get(ref)
            if td is None:
                # Tradable expired before this inner-MC fork — no deal, no moves.
                zero = loss_count.new_zeros(loss_count.shape)           # (B_outer,)
                per_contract.append(zero)
                per_contract.append(zero)
                continue
            td = td[cutoff_idx:]                                       # (T_inner, B_outer, B_inner)
            df_terminal = td[-1] - td[0]
            df_min = td.min(dim=0).values - td[0]
            per_contract.append((df_terminal * loss_mask).sum(dim=-1) / loss_count)
            per_contract.append((df_min * loss_mask).sum(dim=-1) / loss_count)

        return torch.stack(
            [prob_loss, expected_loss_given_loss, variance_loss_given_loss] + per_contract,
            dim=-1)

    def _restricted_struct(self, outer_struct, cutoff_mtm_idx):
        """Build a fresh DealStructure mirroring outer_struct but with each deal's
        Time_dep restricted to events at mtm positions >= cutoff_mtm_idx via
        `DealTimeDependencies.copy_restricted`. Factor_dep is shared by reference
        (static factor lookups, time-grid-independent for the deal types used in
        inner-MC); Calc_res is fresh so inner pricing doesn't clobber outer storage.
        Returns a DealStructure with possibly fewer dependencies (deals fully in
        the past are dropped). Does not recurse into sub_structures — inner-MC use
        case has a flat dependency list."""
        inner = DealStructure(outer_struct.obj.Instrument, store_results=outer_struct.store_results)
        for dd in outer_struct.dependencies:
            new_td = dd.Time_dep.copy_restricted(cutoff_mtm_idx)
            if new_td is None:
                continue
            inner.dependencies.append(utils.DealDataType(
                Instrument=dd.Instrument,
                Factor_dep=dd.Factor_dep,
                Time_dep=new_td,
                Calc_res={} if outer_struct.store_results else None,
            ))
        return inner

    def _run_inner_mc_at_t(self, t, outer_scenario_buffer, shared_mem, base_date,
                           tradable_refs, want_raw_samples=True, with_grad=False,
                           max_inner_steps=None, return_market_only=False):
        """Run inner MC at a single outer timestep `t`, forking from `outer_scenario_buffer`
        — a snapshot of the outer `t_Scenario_Buffer` (factor keys plus the
        `(spot_key,'regimes')` aux key, batch dim B_outer).

        This is the per-`t` body of the former `_run_inner_mc_pass`, lifted so the DP/MPC
        backward sweep can call it on demand outside the outer loop (via the closure
        `bundle['inner_mc_fn']`). The RL `optimize_policy` path still sweeps every `t`
        through the `_run_inner_mc_pass` wrapper below.

        `want_raw_samples=False` returns just `{'features': (B_outer, 3+2*n_tradables)}`.
        `want_raw_samples=True` additionally returns the raw inner samples the solvers need:
            L_T           (B_outer, B_inner)             liability terminal MTM
            F_t1          {ref: (B_outer, B_inner)}      futures price at outer t+1
            dF_T          {ref: (B_outer, B_inner)}      F_T - F_t
            dF_min        {ref: (B_outer, B_inner)}      min_s F_s - F_t over (t, T]
            market_t1     (B_outer, B_inner, market_dim)  inner market state at outer t+1
            market_t      (B_outer, market_dim)           outer-realised market state at t
            t, cutoff_idx
        `market_t`/`market_t1` are every simulated factor's state concatenated — the
        column block is generic (factor-type dispatch lives in `_extract_outer_state_at`),
        so the DP/MPC solvers consume it without knowing what the factors are.

        M2g: this is a dispatcher — the whole inner MC (generation + pricing) runs in
        outer-path sub-chunks (`_run_inner_mc_chunk`) of at most `_INNER_MC_FLAT_LIMIT`
        flat samples, so peak memory tracks the chunk and B_outer scales freely. Each
        chunk draws its own Sobol stream (valid quasi-MC per outer path; the inner MC has
        no cross-outer-path CRN to preserve), so a chunked run is statistically — not
        bitwise — equivalent to a single pass."""
        spot_key = self._find_spot_key()
        B_outer = outer_scenario_buffer[spot_key].shape[-1]
        B_inner = shared_mem.simulation_sub_batch
        n_features = 3 + 2 * len(tradable_refs)

        t_days = int(self.time_grid.scen_time_grid[t])
        inner_time_grid = self.time_grid.truncate_to(base_date, t_days)

        # Terminal / past-end — no inner horizon. Emit zero features; `prob_loss == 0`
        # flags this downstream. The DP sweep does not call here at terminal (it uses the
        # closed-form V_T); the RL wrapper does, hence the guard.
        if inner_time_grid.scen_time_grid.size < 2:
            result = {'features': shared_mem.one.new_zeros(B_outer, n_features)}
            if want_raw_samples:
                result.update(t=t, cutoff_idx=t, L_T=None, market_t=None, market_t1=None,
                              F_t1={}, dF_T={}, dF_min={})
            return result

        # In HedgeMonteCarlo scen_time_grid == mtm_time_grid (dynamic_scenario_dates),
        # so the same `t` indexes both the scenario buffer and the mtm grid.
        cutoff_idx = t
        inner_base_date = base_date + pd.Timedelta(days=t_days)

        # `max_inner_steps=1` truncates the inner grid to {t, t+1} for the diff-ML
        # one-step bootstrap-label generation (`inner_mc_grad_fn_one_step` closure on
        # the bundle). Restricting the AAD tape to a single forward step is the whole
        # reason the spec's twin-loss design fits on a 3090 — a full t→T_dec inner
        # horizon would multiply the tape by ~30× for no use (the bootstrap label only
        # reads `market_{t+1}`). `return_market_only=True` (paired) then skips the
        # deal-pricing chain, which would produce meaningless `L_T` against a 2-point
        # grid anyway (`_restricted_struct` drops deals whose horizon extends past t+1).
        if max_inner_steps is not None and \
                inner_time_grid.scen_time_grid.size > max_inner_steps + 1:
            dates_sorted = sorted(inner_time_grid.scenario_dates)[:max_inner_steps + 1]
            kept = set(dates_sorted)
            inner_time_grid = utils.TimeGrid(
                kept,
                kept & set(inner_time_grid.mtm_dates),
                kept & set(inner_time_grid.base_MTM_dates))
            inner_time_grid.set_base_date(inner_base_date)

        chunk = max(1, _INNER_MC_FLAT_LIMIT // B_inner)
        if chunk >= B_outer:
            return self._run_inner_mc_chunk(
                t, cutoff_idx, outer_scenario_buffer, shared_mem, inner_base_date,
                inner_time_grid, tradable_refs, want_raw_samples, with_grad=with_grad,
                return_market_only=return_market_only)
        if with_grad:
            # Chunked grad mode would require gradient accumulation across chunks via
            # per-chunk .backward()'s; punt on that until a real B_outer needs it.
            raise NotImplementedError(
                f'with_grad=True requires single-chunk inner-MC; B_outer={B_outer} > '
                f'chunk={chunk}. Reduce B_outer or raise _INNER_MC_FLAT_LIMIT.')
        results = []
        for lo in range(0, B_outer, chunk):
            hi = min(lo + chunk, B_outer)
            outer_buf = {k: v[..., lo:hi] for k, v in outer_scenario_buffer.items()}
            results.append(self._run_inner_mc_chunk(
                t, cutoff_idx, outer_buf, shared_mem, inner_base_date,
                inner_time_grid, tradable_refs, want_raw_samples,
                return_market_only=return_market_only))
        return _concat_inner_chunks(results, want_raw_samples)

    def _run_inner_mc_chunk(self, t, cutoff_idx, outer_buf, shared_mem, inner_base_date,
                            inner_time_grid, tradable_refs, want_raw_samples,
                            with_grad=False, return_market_only=False):
        """Inner MC for one outer-path sub-chunk — generation, buffer stuffing, a single
        pricing pass, extraction — all at `B_outer_chunk × B_inner` flat. Peak memory is
        the chunk, not the full B_outer; `_run_inner_mc_at_t` loops this and concatenates.

        Two coordinate systems (unchanged): processes generate against the shifted-base
        `inner_time_grid`; pricers run against the full outer `self.time_grid` with each
        deal's Time_dep restricted via `copy_restricted`. Buffer stuffing prepends the
        outer-realized past (broadcast across B_inner) so path-dependent payoffs see the
        realized fixings.

        `with_grad=True` lifts the no_grad wrapper and reattaches `requires_grad_(True)`
        on each process's per-path initial state. Used by the differential-label
        computation in LsmDpSolver — autograd flows from state-at-t through inner sim
        + deal pricing to terminal utility, giving `∂y_target/∂state_t` for the
        twin-loss (value + gradient) OLS fit. Default False preserves the value-only path."""
        spot_key = self._find_spot_key()
        B_outer = outer_buf[spot_key].shape[-1]
        B_inner = shared_mem.simulation_sub_batch
        B_flat = B_outer * B_inner
        outer_regimes = outer_buf[(spot_key, 'regimes')]               # (T_outer, B_outer) long

        grad_ctx = (torch.enable_grad() if with_grad else torch.no_grad())
        # Track per-process initial state leaves when with_grad — exposed via the result
        # dict so the caller can `.backward()` from any function of the inner outputs and
        # read `.grad` per process/per outer path.
        state_t_leaves = {} if with_grad else None
        with grad_ctx:
            shared_mem.simulation_batch = B_outer
            shared_mem.reset_inner(self.num_factors, inner_time_grid)
            # Per-outer-path initial regime — read by the inner HMM generate.
            shared_mem.t_Scenario_Buffer[(spot_key, 'regime0_inner')] = outer_regimes[t]

            market_t1_parts = []
            for key, proc_inner in self.stoch_factors_inner.items():
                if key.type in utils.DimensionLessFactors:
                    continue
                init_state = self._extract_outer_state_at(t, key, outer_buf)
                if with_grad:
                    # Leaf with grad: differentiates inner-sim + pricing back to state_t.
                    init_state = init_state.detach().clone().requires_grad_(True)
                    state_t_leaves[key] = init_state
                proc_inner.precalculate(
                    inner_base_date, inner_time_grid,
                    init_state,
                    shared_mem, self.process_ofs[key],
                    implied_tensor=self._get_implied_for(key),
                )
                simulated = proc_inner.generate(shared_mem)
                shared_mem.t_Scenario_Buffer[key] = simulated
                if want_raw_samples:
                    # Market state at outer t+1 (inner-time index 1) — generic privileged
                    # extractor; reads the live buffer (factor path + any regime aux its
                    # generate() just published). (factor_flat, B, SB).
                    s = self._extract_outer_state_at(
                        1, key, shared_mem.t_Scenario_Buffer, privileged=True)
                    market_t1_parts.append(s.reshape(-1, B_outer, B_inner))

            # Diff-ML one-step short-circuit — skip the pricing chain (`_restricted_struct`
            # would drop deals whose horizon extends past t+1, producing meaningless
            # `L_T`). The AAD leaves are already attached to the inner generate, so
            # backward through `market_t1` reaches `state_t_leaves` — exactly what the
            # twin-loss differential label gen needs. `L_T`/`F_t1`/`dF_T` return as
            # `None`/empty dicts; the diff-ML solver never reads them on this path.
            if return_market_only and want_raw_samples:
                market_t1 = torch.cat(market_t1_parts, dim=0).permute(1, 2, 0).contiguous()
                market_t_parts, market_t_widths = [], []
                for key in self.stoch_factors_inner:
                    if key.type in utils.DimensionLessFactors:
                        continue
                    block = self._extract_outer_state_at(
                        t, key, outer_buf, privileged=True).reshape(-1, B_outer)
                    market_t_parts.append(block)
                    market_t_widths.append((key, block.shape[0]))
                market_t = torch.cat(market_t_parts, dim=0).permute(1, 0).contiguous()
                # `features` placeholder — the RL feature builder is pricing-dependent
                # and explicitly skipped on the one-step diff-ML path. Consumers on
                # this branch never read it.
                _n_features = 3 + 2 * len(tradable_refs)
                result = {'features': shared_mem.one.new_zeros(B_outer, _n_features)}
                result.update(
                    t=t, cutoff_idx=cutoff_idx, L_T=None,
                    F_t1={}, dF_T={}, dF_min={},
                    market_t=market_t, market_t1=market_t1)
                if with_grad:
                    result['state_t_leaves'] = state_t_leaves
                    result['state_t_leaf_widths'] = market_t_widths
                shared_mem.simulation_batch = B_outer
                shared_mem.t_Buffer.clear()
                shared_mem.t_Scenario_Buffer.clear()
                shared_mem.t_quasi_rng.clear()
                return result

            # Stuff the outer-realized past into each factor buffer + flatten (B,SB)→B*SB.
            for key in self.stoch_factors_inner:
                if key.type in utils.DimensionLessFactors:
                    continue
                inner_path = shared_mem.t_Scenario_Buffer[key]                  # (T_inner, ..., B, SB)
                outer_past = outer_buf[key][:cutoff_idx]                         # (cutoff, ..., B)
                outer_past_b2 = outer_past.unsqueeze(-1).expand(*outer_past.shape, B_inner)
                stuffed = torch.cat([outer_past_b2, inner_path], dim=0)          # (T_outer, ..., B, SB)
                shared_mem.t_Scenario_Buffer[key] = stuffed.reshape(
                    *stuffed.shape[:-2], B_flat)

            # Single-pass pricing — the chunk is sized so B_flat fits the memory budget.
            shared_mem.t_Buffer.clear()
            shared_mem.simulation_batch = B_flat
            # `fillvalue` is a batch-sized empty tensor frozen at State construction (the
            # energy-leg reset code uses it as the empty-cat fallback) — it must track the
            # current simulation_batch or cash_settle size-mismatches.
            shared_mem.fillvalue = shared_mem.one.new_zeros((0, 1, B_flat))
            # Per-chunk restricted DealStructures: same instruments + Factor_dep,
            # fresh Time_dep slicing off past events, fresh Calc_res.
            inner_netting_sets = self._restricted_struct(self.netting_sets, cutoff_idx)
            inner_liabilities = self._restricted_struct(self.liabilities, cutoff_idx)
            inner_netting_sets.resolve_structure(shared_mem, self.time_grid)
            shared_mem.reset_cashflows(self.time_grid)
            mtm_flat = inner_liabilities.resolve_hedge_structure(
                shared_mem, self.time_grid)['mtm']
            inner_mtm = mtm_flat.reshape(*mtm_flat.shape[:-1], B_outer, B_inner)
            inner_trade_tensors = {
                x.Instrument.field['Reference']: x.Calc_res['tensor'].reshape(
                    *x.Calc_res['tensor'].shape[:-1], B_outer, B_inner)
                for x in inner_netting_sets.dependencies
                if x.Calc_res.get('tensor') is not None
            }
            result = {'features': self._calc_inner_features(
                inner_mtm, inner_trade_tensors, tradable_refs, cutoff_idx)}

            if want_raw_samples:
                F_t1, dF_T, dF_min = {}, {}, {}
                zero_bs = inner_mtm[-2].new_zeros(inner_mtm[-2].shape)   # (B_outer, B_inner)
                for ref in tradable_refs:
                    td = inner_trade_tensors.get(ref)
                    if td is None:
                        # Tradable expired before this fork — zero moves, no position.
                        F_t1[ref] = dF_T[ref] = dF_min[ref] = zero_bs
                        continue
                    td = td[cutoff_idx:]                                # (T_inner, B_outer, B_inner)
                    # td has < 2 time points when the tradable's last deal event is at t
                    # (it expires this step) — no t+1 slice; freeze it (dF == 0).
                    F_t1[ref] = (td[1] if td.shape[0] >= 2 else td[-1]).clone()
                    dF_T[ref] = (td[-1] - td[0]).clone()
                    dF_min[ref] = (td.min(dim=0).values - td[0]).clone()
                # Market state — every simulated factor's informative state (regime
                # posterior, carry curve, …) concatenated; factor order is the
                # `stoch_factors_inner` iteration order, identical for market_t/market_t1.
                market_t1 = torch.cat(market_t1_parts, dim=0).permute(1, 2, 0).contiguous()
                market_t_parts = []
                market_t_widths = []
                for key in self.stoch_factors_inner:
                    if key.type in utils.DimensionLessFactors:
                        continue
                    block = self._extract_outer_state_at(
                        t, key, outer_buf, privileged=True).reshape(-1, B_outer)
                    market_t_parts.append(block)
                    market_t_widths.append((key, block.shape[0]))
                market_t = torch.cat(market_t_parts, dim=0).permute(1, 0).contiguous()
                result.update(
                    t=t, cutoff_idx=cutoff_idx, L_T=inner_mtm[-2].clone(),
                    F_t1=F_t1, dF_T=dF_T, dF_min=dF_min,
                    market_t=market_t, market_t1=market_t1)
                if with_grad:
                    # Pair each leaf with the privileged-column width it occupies in
                    # market_t — the differential-label projection in LsmDpSolver needs
                    # this to write per-leaf gradients into the right deep-state columns
                    # without re-deriving factor-type widths (which would silently drift
                    # if the privileged extractor's per-type packing changes).
                    result['state_t_leaves'] = state_t_leaves
                    result['state_t_leaf_widths'] = market_t_widths

            shared_mem.simulation_batch = B_outer
            shared_mem.t_Buffer.clear()
            shared_mem.t_Scenario_Buffer.clear()
            # Drop the Sobol sample cache — it is keyed by sample_size and would otherwise
            # grow unbounded across chunks / t-steps. Each chunk re-draws a fresh,
            # independent quasi-MC stream (the engine advances); the pricer's per-pass
            # `reset_qrg` caching is intact within a chunk — only cleared between them.
            shared_mem.t_quasi_rng.clear()

        return result

    def _run_inner_mc_pass(self, run, shared_mem, base_date, params, tradable_refs):
        """Thin wrapper over `_run_inner_mc_at_t` for the RL `optimize_policy` path:
        sweep every outer `t`, collect the per-`t` feature row, stack to
        `(T_outer, B_outer, 3 + 2*len(tradable_refs))`. The solver path instead calls
        `_run_inner_mc_at_t` on demand via `bundle['inner_mc_fn']`."""
        outer_scenario_buffer = dict(shared_mem.t_Scenario_Buffer)
        per_t = [
            self._run_inner_mc_at_t(
                t, outer_scenario_buffer, shared_mem, base_date,
                tradable_refs, want_raw_samples=False)['features']
            for t in range(self.time_grid.scen_time_grid.size)
        ]
        return torch.stack(per_t, dim=0)


def construct_calculation(calc_type, config, **kwargs):
    return globals().get(calc_type)(config, **kwargs)


if __name__ == '__main__':
    pass
