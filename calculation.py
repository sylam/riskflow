#########################################################################
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
#########################################################################

# import standard libraries

import os
import time
import logging
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf

# needed to list available devices
from tensorflow.python.client import device_lib
# load up some useful data types
from collections import OrderedDict, namedtuple, defaultdict
# import the risk factors (also known as price factors)
from riskfactors import construct_factor
# import the stochastic processes
from stochasticprocess import construct_process
# import the currency/curve lookup factors 
from instruments import get_fxrate_factor, get_recovery_rate, get_interest_factor, get_survival_factor
# import the hessian function
from pricing import SensitivitiesEstimator
# import the documentation module
import documentation
# import the utils
import utils


class InstrumentExpired(Exception):
    def __init__(self, message):
        self.message = message


class Aggregation(object):
    def __init__(self, name):
        self.field = {'Reference': name}


class DealStructure(object):
    def __init__(self, obj, deal_level_mtm=False):
        # parent object - note that all parent objects MUST have an aggregate method that can aggregate partitions
        self.obj = utils.DealDataType(Instrument=obj, Factor_dep=None, Time_dep=None, Calc_res=None)
        # gather a list of deal dependencies
        self.dependencies = []
        # maintain a list of container objects
        self.sub_structures = []
        # do we need to create sub_partitions?
        self.sub_partitions = OrderedDict()
        # Do we want to store each deal level MTM explicitly?
        self.deal_level_mtm = deal_level_mtm

    def calc_time_dependency(self, base_date, deal, time_grid):
        # calculate the additional (dynamic) dates that this instrument needs to be evaluated at
        deal_time_dep = None
        try:
            reval_dates = deal.get_reval_dates(clip_expiry=True)
            if len(time_grid.scenario_dates) == 1:
                if len(reval_dates) > 0 and max(reval_dates) < base_date:
                    raise InstrumentExpired(deal.field.get('Reference', 'Unknown Instrument Reference'))
                deal_time_dep = time_grid.calc_deal_grid({base_date})
            else:
                deal_time_dep = time_grid.calc_deal_grid(reval_dates)
        except InstrumentExpired as e:
            logging.warning('skipping expired deal {0}'.format(e.args))

        return deal_time_dep

    def add_deal_to_structure(self, base_date, deal, static_offsets, stochastic_offsets, all_factors,
                              all_tenors, time_grid, calendars):
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
                                       Factor_dep=deal.calc_dependencies(base_date, static_offsets,
                                                                         stochastic_offsets,
                                                                         all_factors, all_tenors,
                                                                         time_grid, calendars),
                                       Time_dep=deal_time_dep,
                                       Calc_res=OrderedDict() if self.deal_level_mtm else None))
            except Exception as e:
                logging.error('{0}.{1} {2} - Skipped'.format(deal.field['Object'],
                                                             deal.field.get('Reference', '?'), e.args))

    def add_structure_to_structure(self, struct, base_date, static_offsets, stochastic_offsets,
                                   all_factors, all_tenors, time_grid, calendars):
        # get the dependencies
        struct_time_dep = self.calc_time_dependency(base_date, struct.obj.Instrument, time_grid)
        try:
            if struct_time_dep is not None:
                struct.obj = utils.DealDataType(
                    Instrument=struct.obj.Instrument,
                    Factor_dep=struct.obj.Instrument.calc_dependencies(base_date,
                                                                       static_offsets,
                                                                       stochastic_offsets,
                                                                       all_factors, all_tenors,
                                                                       time_grid, calendars),
                    Time_dep=struct_time_dep,
                    Calc_res=OrderedDict() if self.deal_level_mtm else None)
            # Structure object representing a netted set of cashflows
            self.sub_structures.append(struct)
        except Exception as e:
            logging.error('{0}.{1} {2} - Skipped'.format(struct.obj.Instrument.field['Object'],
                                                         struct.obj.Instrument.field.get('Reference', '?'), e.args))

    def build_partitions(self):
        # sort the data by the tags
        self.dependencies.sort(key=lambda x: x.Instrument.field['Tags'])
        # get the distinct tags
        unique_tags = set([x.Instrument.field['Tags'] for x in self.dependencies])
        if len(unique_tags) > 1:
            for tag in sorted(unique_tags):
                # TODO
                pass

    def resolve_structure(self, shared, time_grid):
        """
        Resolves the Structure
        """

        accum = None

        if self.sub_structures:
            # process sub structures
            for structure in self.sub_structures:
                struct = structure.resolve_structure(shared, time_grid)
                accum = struct if accum is None else accum + struct

        if self.dependencies and self.obj.Instrument.accum_dependencies:
            # accumulate the mtm's
            deal_tensors = 0.0

            for deal_data in self.dependencies:
                deal_ref = utils.Factor('Deal', (deal_data.Instrument.field.get('Reference', 'Unknown'),))

                with tf.name_scope(utils.check_scope_name(deal_ref)):
                    mtm = deal_data.Instrument.calculate(shared, time_grid, deal_data)
                    deal_tensors += mtm

            netting = deal_tensors
            accum = netting if accum is None else accum + netting

        # postprocessing code for working out the mtm of all deals, collateralization etc..
        if hasattr(self.obj.Instrument, 'post_process'):
            # the actual answer for this netting set
            accum = self.obj.Instrument.post_process(accum, shared, time_grid, self.obj, self.dependencies)

        return accum


class ScenarioTimeGrid(object):
    def __init__(self, cutoff_date, global_time_grid, base_date):
        scen_grid = global_time_grid.scen_time_grid
        offset = scen_grid.searchsorted((cutoff_date - base_date).days) + 1
        self.scen_time_grid = scen_grid[:offset]
        self.time_grid_years = self.scen_time_grid / utils.DAYS_IN_YEAR
        self.scenario_grid = global_time_grid.scenario_grid[:offset]


class TimeGrid(object):
    def __init__(self, scenario_dates, MTM_dates, base_MTM_dates):
        self.scenario_dates = scenario_dates
        self.base_MTM_dates = base_MTM_dates
        self.CurrencyMap = {}
        self.set_mtm_dates(MTM_dates)

    def set_mtm_dates(self, MTM_dates):
        self.mtm_dates = MTM_dates
        self.date_lookup = dict([(x, i) for i, x in enumerate(sorted(MTM_dates))])

    def calc_time_grid(self, time_in_days):
        dvt = np.concatenate(([1], np.diff(self.scen_time_grid), [1]))
        scen_index = self.scen_time_grid.searchsorted(time_in_days, side='right')
        index = (scen_index - 1).clip(0, self.scen_time_grid.size - 1)
        alpha = ((time_in_days - self.scen_time_grid[index]) / dvt[scen_index]).clip(0, 1)
        return np.dstack([alpha, time_in_days, index])[0]

    def set_base_date(self, base_date, delta=None):
        # leave the grids in terms of the number of days - note that it's possible to have the scenario_dates
        # the same as the mtm_dates (for more accurate margin period of risk on collateralized netting sets)
        self.mtm_time_grid = np.array([(x - base_date).days for x in sorted(self.mtm_dates)])
        self.scen_time_grid = np.array([(x - base_date).days for x in sorted(self.scenario_dates)])

        self.base_time_grid = set([self.date_lookup[x] for x in self.base_MTM_dates])
        self.time_grid = self.calc_time_grid(self.mtm_time_grid)

        # store the scenario time_grid
        self.scenario_grid = np.zeros((self.scen_time_grid.size, 3))
        self.scenario_grid[:, utils.TIME_GRID_MTM] = self.scen_time_grid
        self.scenario_grid[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(self.scen_time_grid.size)

        # deal with the case that we need a very fine time_grid - note we do this after calculating the
        # scenario_grid as setting a non-null delta is a way to generate scenarios without calculating the
        # whole risk factor
        if delta is not None:
            delta_grid = np.arange(0, self.scen_time_grid.max(), delta)
            self.scen_time_grid = np.union1d(self.scen_time_grid, delta_grid)

        self.time_grid_years = self.scen_time_grid / utils.DAYS_IN_YEAR

    def get_scenario_offset(self, days_from_base):
        prev_scen_index = self.scen_time_grid[self.scen_time_grid <= days_from_base].size - 1
        scenario_grid_delta = np.float64(
            (self.scen_time_grid[prev_scen_index + 1] - self.scen_time_grid[prev_scen_index]) if (
                    self.scen_time_grid.size > 1 and self.scen_time_grid.size > prev_scen_index + 1) else 1.0)
        return (days_from_base - self.scen_time_grid[prev_scen_index]) / scenario_grid_delta, prev_scen_index

    def set_currency_settlement(self, currencies):
        self.CurrencyMap = {}
        for currency, dates in currencies.items():
            settlement_dates = sorted([self.date_lookup[x] for x in dates if x in self.date_lookup])
            if settlement_dates:
                currency_lookup = np.zeros(self.mtm_time_grid.size, dtype=np.int32) - 1
                currency_lookup[settlement_dates] = np.arange(len(settlement_dates))
                self.CurrencyMap.setdefault(currency, currency_lookup)

    def calc_deal_grid(self, dates):
        try:
            dynamic_dates = self.base_time_grid.union([self.date_lookup[x] for x in dates])
        except KeyError as e:
            # if there is at least one reset date in the set of dates, then return it, else the deal has expired
            r = [self.date_lookup[x] for x in dates if x in self.date_lookup]
            if r:
                dynamic_dates = self.base_time_grid.union(r)
            else:
                if max(dates) < min(self.date_lookup.keys()):
                    raise InstrumentExpired(e)

                # include this instrument but don't bother pricing it through time
                return utils.DealTimeDependencies(self.mtm_time_grid, np.array([0]))

        # now construct the full deal grid 
        deal_time_grid = np.array(sorted(dynamic_dates))
        # find the last dynamic date - should be the expiry date
        expiry = self.date_lookup[max(dates)]
        # calculate the interpolation points etc.
        return utils.DealTimeDependencies(self.mtm_time_grid, deal_time_grid[deal_time_grid <= expiry])


class Calculation(object):

    def __init__(self, config, prec=np.float32):
        """
        Construct a new calculation - all calculations must setup the GPU ( by setting the precision )
        and by specifying a configuration (market data + porfolio)
        """

        self.config = config
        self.prec = prec
        self.time_grid = None
        self.shared_mem = None

        # the risk factor data
        self.static_factors = OrderedDict()
        self.static_ofs = OrderedDict()
        self.stoch_factors = OrderedDict()
        self.stoch_ofs = OrderedDict()
        self.all_factors = {}
        self.all_tenors = {}

        # these are used for implied calibration        
        self.implied_ofs = OrderedDict()

        self.dependent_factors = None
        self.base_date = None

        self.tenor_size = None
        self.tenor_offset = None

        # the deal structure
        self.netting_sets = None

        # generate slides from calculation?
        self.documentation = documentation.NoElaboration()
        # what's the name of this calculation?
        self.name = 'Unnamed'
        # performance and admin feedback
        self.calc_stats = {}
        # the calculation parameters (defined by calling execute)
        self.params = {}
        # Index for the gradients of this calculation (if requested)
        self.gradient_index = None

    def feed(self, **args):
        pass

    def execute(self, params):
        pass

    def make_factor_index(self, tensors):
        # need to match the indices back
        tenors = utils.get_tenors(self.all_factors)
        self.gradient_index = {}
        for var in tensors:
            name = var.name[:-2]
            factor_name = name.split('/')[-1]
            name_size, pad_size = tenors[factor_name].shape
            padding = 3 - pad_size
            indices = np.pad(tenors[factor_name], [[0, 0], [0, padding]], 'constant')
            self.gradient_index[name] = (indices, pad_size)

    def gradients_as_df(self, grad, header='Gradient', display_val=False):
        if isinstance(grad, dict):
            # get the factor values from all_factors if necessary
            factor_values = {utils.check_scope_name(k): v.factor if hasattr(v, 'factor') else v
                             for k, v in self.all_factors.items()} if display_val else {}
            hessian_index = ([], [])
            factor, rate, tenor, values = [], [], [], []
            for k, v in grad.items():
                # first derivative
                name = k.name[:-2]
                non_zero = np.where(v)[0]
                grad_index, index_len = self.gradient_index[name]
                hessian_index[0].append([name] * v.shape[0])
                hessian_index[1].append(grad_index)
                values.append(v[non_zero])
                rate.append([name] * non_zero.size)
                tenor.append(grad_index[non_zero])
                # store the actual factor value if required
                if display_val:
                    rate_name = name.split('/')[-1]
                    if rate_name in factor_values:
                        rate_non_zero = grad_index[non_zero][:, :index_len].astype(np.float64)
                        factor.append(factor_values[rate_name].current_value(rate_non_zero).flatten())
                    else:
                        sub_rate, param = rate_name.rsplit('.', 1)
                        factor.append(factor_values[sub_rate].current_value()[param].flatten())

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
            df = pd.DataFrame(values, index=multi_index[0],
                              columns=multi_index[1]).sort_index(level=[0, 1, 2, 3], axis=0)
        return df

    def set_deal_structures(self, deals, output, num_scenarios, batch_size, tagging=None, deal_level_mtm=False):
        partitioning = False
        for node in deals:
            # get the instrument
            instrument = node['instrument']
            # should we skip it?
            if node.get('Ignore') == 'True':
                continue
            # apply a tag if provided (used for partitioning)
            if tagging:
                instrument.field['Tags'] = tagging(instrument.field)
                partitioning = True
            if node.get('Children'):
                struct = DealStructure(instrument, deal_level_mtm=deal_level_mtm)
                logging.info('Analysing Group {0}'.format(instrument.field.get('Reference', '<undefined>')))
                self.set_deal_structures(node['Children'], struct, num_scenarios, batch_size, tagging, deal_level_mtm)
                output.add_structure_to_structure(struct, self.base_date, self.static_ofs, self.stoch_ofs,
                                                  self.all_factors, self.all_tenors, self.time_grid,
                                                  self.config.holidays)
                continue
            output.add_deal_to_structure(self.base_date, instrument, self.static_ofs, self.stoch_ofs, self.all_factors,
                                         self.all_tenors, self.time_grid, self.config.holidays)

        # check if we need to partition this structure
        if partitioning:
            output.build_partitions()


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
        '- The dates in the **Base Time Grid** up the the deal\'s maturity date.',
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
        'This is the simulated exposure at percentile $q$ where $0<q<1$ (typically q=.95 or .99).',
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
        '$$\\Bbb E(L(\\tau))=\\Bbb E\\Big(\\int_0^T L(t)(-dH(t))\\Big),$$',
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
        'Posting (or recieving) collateral can imply a funding cost (or benefit) when there is a spread between a',
        'party\'s interal cost of funding and the contractual interest rate paid on the collateral balance. The',
        'discounted expectation of this cost (or benefit) summed across all time horizons and scenarios constitutes',
        'a funding value adjustment and can be expressed as:',
        '',
        '$$\\frac{1}{m}\\sum_{j=1}^m \\sum_{k=0}^{T-1} B_j(t_k)S_j(t_k)\\Big(\\frac{D_j^c(t_k,t_{k+1})}'
        '{D_j^f(t_k,t_{k+1})}-1\\Big)D_j^c(0,t_k),$$',
        '',
        'where',
        '',
        '- $B_j(t)$ is the number of units of the collateral portfolio for scenario $j$ at time $t$',
        '- $S_j(t)$ is the base currecy value of one unit of the collateral asset for scenario $j$ at time $t$',
        '- $D_j^c(t)$ is the discount rate for the collateral rate at time t for scenario $j$',
        '- $D_j^f(t)$ is the discount rate for the funding rate at time t for scenario $j$',
        '',
        'Note that only cash collateral is supported presently although this can be extended.'
    ])

    def __init__(self, config, **kwargs):
        super(Credit_Monte_Carlo, self).__init__(config, **kwargs)
        self.cholesky = None
        self.stochastic_factors = None
        self.reset_dates = None
        self.settlement_currencies = None

        # global variables to store the state of the graph between calculations
        self.shared_mem_class = namedtuple('shared_mem', 't_Buffer t_random_numbers \
                                                        t_Scenario_Buffer t_Static_Buffer t_Cashflows \
                                                        t_Feed_dict t_Credit gpus riskneutral precision \
                                                        simulation_batch Report_Currency')

        # used to store the Scenarios (if requested)
        self.Scenarios = {}
        # used to store any jacobian matrices
        self.jacobians = {}
        # used to store the resulting tensors of the calculation
        self.tensors = OrderedDict()
        # implied factors
        self.implied_factors = OrderedDict()
        # we represent the calc as a combination of static, stochastic and implied parameters
        self.stoch_values = []
        self.stoch_var = []
        self.static_values = []
        self.static_var = []
        self.implied_var = []
        self.implied_values = []
        # potentially store the full list of variables
        self.all_var = None

        self.stoch_ofs = OrderedDict()
        self.static_ofs = OrderedDict()
        self.implied_ofs = OrderedDict()

    def update_factors(self, params, base_date):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, self.input_time_grid)

        # update the time grid
        logging.info('Updating timegrid')
        self.update_time_grid(base_date, reset_dates, settlement_currencies,
                              dynamic_scenario_dates=params.get('Dynamic_Scenario_Dates', 'No') == 'Yes')

        # now construct the stochastic factors and static factors for the simulation
        self.stoch_factors.clear()

        for price_model, price_factor in stochastic_factors.items():
            factor_obj = construct_factor(price_factor, self.config.params['Price Factors'])
            implied_factor = implied_factors.get(price_model)
            try:
                if implied_factor:
                    implied_obj = construct_factor(implied_factor, self.config.params['Price Factors'])
                    self.implied_factors[implied_factor] = implied_obj
                else:
                    implied_obj = None
            except KeyError as e:
                logging.warning(
                    'Implied Factor {0} missing in market data file - Attempting to Proxy with EUR-MASTER'.format(
                        e.args))
                # Hardcoding - Fix this - TODO
                proxy_factor = utils.Factor(implied_factor.type, ('EUR-MASTER',))
                self.config.params['Price Factors'][utils.check_tuple_name(implied_factor)] = \
                    self.config.params['Price Factors'][utils.check_tuple_name(proxy_factor)]
                implied_obj = construct_factor(implied_factor, self.config.params['Price Factors'])

            self.stoch_factors[price_factor] = construct_process(price_model.type,
                                                                 factor_obj,
                                                                 self.config.params['Price Models']
                                                                 [utils.check_tuple_name(price_model)],
                                                                 implied_obj)

        self.static_factors = OrderedDict()
        for price_factor in set(dependent_factors).difference(stochastic_factors.values()):
            try:
                self.static_factors.setdefault(
                    price_factor, construct_factor(price_factor, self.config.params['Price Factors']))
            except KeyError as e:
                logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.args))

        self.all_factors = self.stoch_factors.copy()
        self.all_factors.update(self.static_factors)
        self.all_factors.update(self.implied_factors)
        self.num_factors = sum([v.num_factors() for v in self.stoch_factors.values()])

        # work out distinct tenor sets
        # distinct_tenors = OrderedDict()

        # now get the stochastic risk factors ready - these will be generated from the price models
        with tf.name_scope("Stochastic_Input"):
            for key, value in self.stoch_factors.items():
                if key.type not in utils.DimensionLessFactors:
                    # check if there are any implied factors linked here
                    with tf.name_scope("Implied/"):
                        if hasattr(value, 'implied'):
                            vals, vars = [], OrderedDict()
                            self.implied_ofs.setdefault(key, len(self.implied_var))
                            for param_name, param_value in value.implied.current_value().items():
                                factor_name = utils.Factor(value.implied.__class__.__name__,
                                                           key.name + (param_name,))
                                vals.append(param_value)
                                vars[param_name] = tf.placeholder(self.prec, shape=len(param_value),
                                                                  name=utils.check_scope_name(factor_name))
                            self.implied_values.append(vals)
                            self.implied_var.append(vars)

                    # record the offset of this risk factor
                    self.stoch_ofs.setdefault(key, len(self.stoch_var))
                    # utils.distinct_tenors(distinct_tenors, key, value.factor)
                    current_val = value.factor.current_value()
                    self.stoch_values.append(current_val)
                    self.stoch_var.append(
                        tf.placeholder(self.prec,
                                       shape=len(value.factor.current_value()),
                                       name=utils.check_scope_name(key)))

        # and then get the the static risk factors ready - these will just be looked up
        with tf.name_scope("Static_Input"):
            for key, value in self.static_factors.items():
                if key.type not in utils.DimensionLessFactors:
                    # record the offset of this risk factor
                    self.static_ofs.setdefault(key, len(self.static_var))
                    # utils.distinct_tenors(distinct_tenors, key, value)
                    current_val = value.current_value()
                    self.static_values.append(current_val)
                    self.static_var.append(
                        tf.placeholder(self.prec,
                                       shape=len(current_val),
                                       name=utils.check_scope_name(key)))

        # update the correlations
        self.update_correlation()
        # check if we need gradients for any xVA
        greeks = np.any([params[x].get('Gradient', 'No') == 'Yes' for x in params.keys() if x.endswith('VA')])

        # setup the device and allocate memory
        self.__refresh_shared_mem(params['Random_Seed'],
                                  params.get('NoModel', 'Constant'),
                                  params['Currency'],
                                  calc_greeks=greeks)

        # calculate a reverse lookup for the tenors and store the daycount code
        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        # now initialize all stochastic factors
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                with tf.name_scope(utils.check_scope_name(key)):
                    implied_index = self.implied_ofs.get(key, -1)

                    value.precalculate(
                        base_date, ScenarioTimeGrid(
                            dependent_factors[key], self.time_grid, base_date),
                        self.stoch_var[self.stoch_ofs[key]],
                        self.shared_mem, self.process_ofs[key],
                        implied_tensor=self.implied_var[implied_index] if implied_index > -1 else None)

        # now check if any of the stochastic processes depend on other processes
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                # precalculate any values for the stochastic process
                value.calc_references(key, self.static_ofs, self.stoch_ofs, self.all_tenors, self.all_factors)

        # store the state info
        self.dependent_factors, self.stochastic_factors = dependent_factors, stochastic_factors

    def update_time_grid(self, base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=False):
        # work out the scenario and dynamic dates
        dynamic_dates = set([x for x in reset_dates if x > base_date])
        base_mtm_dates = self.config.parse_grid(base_date, max(dynamic_dates), self.input_time_grid)
        mtm_dates = base_mtm_dates.union(dynamic_dates)
        if dynamic_scenario_dates:
            scenario_dates = mtm_dates
        else:
            scenario_dates = self.config.parse_grid(base_date, max(dynamic_dates), self.input_time_grid,
                                                    past_max_date=True)

        # setup the scenario and base time grids
        self.time_grid = TimeGrid(scenario_dates, mtm_dates, base_mtm_dates)
        self.base_date = base_date
        self.reset_dates = reset_dates
        self.time_grid.set_base_date(base_date)

        # Set the settlement dates
        self.time_grid.set_currency_settlement(settlement_currencies)
        self.settlement_currencies = settlement_currencies

    def update_correlation(self):
        # create the correlation matrix
        correlation_matrix = np.eye(self.num_factors, dtype=self.prec)

        # prepare the correlation matrix (and the offsets of each stochastic process)
        correlation_factors = []
        self.process_ofs = OrderedDict()
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

        # need to do cholesky
        eigval, eigvec = np.linalg.eig(correlation_matrix.astype(np.float64))
        if not (eigval > 1e-8).all():
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

        self.correlation_matrix = correlation_matrix
        with tf.name_scope("Correlation_Input"):
            self.input_correlation = tf.placeholder(self.prec, shape=correlation_matrix.shape,
                                                    name='correlation_matrix')
            # set the cholesky decomp
            self.cholesky = tf.cholesky(self.input_correlation)

    def __refresh_shared_mem(self, seed, nomodel, reporting_currency, calc_greeks=False, debug_batch_size=128):
        # setup the Feed dictionary - the inputs needed to evaluate the graph
        feed = {self.input_correlation: self.correlation_matrix}
        feed.update(dict(zip(self.stoch_var, self.stoch_values)))
        feed.update(dict(zip(self.static_var, self.static_values)))
        feed.update(
            dict(zip(itertools.chain(*[x.values() for x in self.implied_var]),
                     itertools.chain(*self.implied_values))))
        if not isinstance(self.batch_size, int):
            feed.update({self.batch_size: debug_batch_size})
            # this makes a difference in tensorflow's internals for some reason
            # might be possible to remove this once fixed
            cashflow_size = [self.batch_size]
        else:
            cashflow_size = self.batch_size

        if calc_greeks:
            implied_vars = list(itertools.chain(*[x.values() for x in self.implied_var]))
            self.all_var = self.stoch_var + self.static_var + implied_vars
            # build our index                 
            self.make_factor_index(self.all_var)

        # Now apply that cholesky to our random numbers.
        # There may be a better way to do this as the cholesky is lower triangular

        with tf.name_scope('Setup'):
            self.shared_mem = self.shared_mem_class(
                t_Buffer={},
                t_random_numbers=tf.reshape(
                    tf.matmul(
                        self.cholesky,
                        tf.random_normal(
                            [self.num_factors, self.batch_size * self.time_grid.scen_time_grid.size],
                            seed=seed, dtype=self.prec),
                    ),
                    (self.num_factors, self.time_grid.scen_time_grid.size, -1), name='d_random_numbers'),
                t_Scenario_Buffer=[None] * len(self.stoch_factors),
                t_Static_Buffer=self.static_var,
                t_Cashflows={k: {t_i: tf.zeros(cashflow_size, dtype=self.prec) for t_i in np.where(v >= 0)[0]}
                             for k, v in self.time_grid.CurrencyMap.items()},
                t_Feed_dict=feed,
                t_Credit={},
                gpus=[x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'],
                riskneutral=nomodel == 'RiskNeutral',
                precision=self.prec,
                simulation_batch=self.batch_size,
                Report_Currency=get_fxrate_factor(utils.check_rate_name(reporting_currency),
                                                  self.static_ofs, self.stoch_ofs)
            )

    def feed(self, batch_size, correlation=None, stoch_vars={}, static_vars={},
             options=None, run_meta=None, writer=None):

        # record the performance stats
        self.calc_stats['Tensor_Execution_Time'] = time.clock()

        # config = tf.ConfigProto(device_count = {'GPU': 0})
        config = tf.ConfigProto(allow_soft_placement=True)

        # clear the output
        self.output = {}
        # update the feed
        feed = self.shared_mem.t_Feed_dict.copy()
        # copy the static params
        feed.update({self.graph.get_tensor_by_name('Static_Input/{0}:0'.format(k)): v for
                     k, v in static_vars.items()})
        # copy the stoch params
        feed.update({self.graph.get_tensor_by_name('Stochastic_Input/{0}:0'.format(k)): v for
                     k, v in stoch_vars.items()})
        # update the correlation if necessary
        if correlation is not None:
            # note - I don't check if the new matrix is positive definite - this will break if it is
            feed.update({self.input_correlation: correlation})

        # change the batch size
        if not isinstance(self.batch_size, int):
            feed[self.batch_size] = batch_size

        with tf.Session(graph=self.graph, config=config) as sess:
            output = defaultdict(list)
            tensors = list(self.tensors.values())
            for run in range(self.params['Simulation_Batches']):
                # get the output tensors
                for k, v in zip(self.tensors.keys(), sess.run(tensors, feed_dict=feed,
                                                              options=options, run_metadata=run_meta)):
                    output[k].append(v)

                # fetch cashflows if necessary
                if self.params.get('Generate_Cashflows', 'No') == 'Yes':
                    dates = np.array(sorted(self.time_grid.mtm_dates))
                    for currency, values in self.shared_mem.t_Cashflows.items():
                        curr = sess.run(values, feed_dict=feed, options=options, run_metadata=run_meta)
                        cash_index = dates[sorted(curr.keys())]
                        output.setdefault('cashflows', {}).setdefault(currency, []).append(
                            pd.DataFrame([v for _, v in sorted(curr.items())], index=cash_index))

                # add any scenarios if necessary
                if self.params.get('Calc_Scenarios', 'No') == 'Yes':
                    for key, value in self.stoch_factors.items():
                        output.setdefault('scenarios', {}).setdefault(key, []).append(
                            sess.run(self.shared_mem.t_Scenario_Buffer[self.stoch_ofs[key]],
                                     feed_dict=feed, options=options, run_metadata=run_meta))
                if writer:
                    writer.add_run_metadata(run_meta, 'run {}'.format(run), run)

        for result, data in output.items():
            if result == 'scenarios':
                self.output.setdefault('scenarios', {k: np.concatenate(v, axis=-1) for k, v in data.items()})
            elif result == 'cashflows':
                self.output.setdefault('cashflows', {k: pd.concat(v, axis=1) for k, v in data.items()})
            elif result in ['cva', 'collva', 'fva']:
                self.output.setdefault(result, np.array(data, dtype=np.float64).mean())
            elif result == 'collva_t':
                self.output.setdefault(result, np.array(data, dtype=np.float64).mean(axis=0))
            elif result in ['grad_cva', 'grad_collva', 'grad_fva']:
                grad = dict.fromkeys(data[0].keys())
                for k in grad.keys():
                    values = np.array([v[k].astype(np.float64) for v in data])
                    grad[k] = np.mean(values, axis=0)
                self.output.setdefault(result, grad)
            elif result in ['grad_cva_hessian']:
                self.output.setdefault(result, np.stack(data).astype(np.float64).mean(axis=0))
            else:
                self.output.setdefault(result, np.concatenate(data, axis=-1).astype(np.float64))

        self.calc_stats['Tensor_Execution_Time'] = time.clock() - self.calc_stats['Tensor_Execution_Time']
        return self.output

    def execute(self, params, feedgraph=True):
        # get the rundate
        base_date = pd.Timestamp(params['Run_Date'])
        # check if we need to produce a slideshow (i.e. documentation)
        if params['Generate_Slideshow'] == 'Yes':
            # we need to generate scenarios to draw pictures . . .
            params['Calc_Scenarios'] = 'Yes'
            # check if we need to compare the results to an externally produced result
            compare_output = pd.read_csv(params['PFE_Recon_File'], index_col=0, parse_dates=True) \
                if params['PFE_Recon_File'] else None
            # set the name of this calculation
            self.name = params['calc_name'][0]
            # create some slides
            self.documentation = documentation.Elaboration(self.name)
        else:
            compare_output = ''
            self.documentation = documentation.NoElaboration()

        # Define the base and scenario grids
        self.input_time_grid = params['Time_grid']
        self.numscenarios = params['Batch_Size'] * params['Simulation_Batches']

        # reset the default graph
        tf.reset_default_graph()
        # create a TF graph to represent this calculation
        self.graph = tf.Graph()
        # set the precision in the utils module - Nasty hack - TODO
        utils.Default_Precision = self.prec
        # store the params                
        self.params = params
        # Dynamic Batch?
        dynamic = params.get('Dynamic', 'No') == 'Yes'
        # build the TF graph
        with self.graph.as_default() as g:
            self.batch_size = tf.placeholder(tf.int32, shape=[], name='Batch_Size') if dynamic else params['Batch_Size']
            self.calc_stats['Factor_Setup_Time'] = time.clock()
            self.update_factors(params, base_date)

            # record the setup time
            self.calc_stats['Factor_Setup_Time'] = time.clock() - self.calc_stats['Factor_Setup_Time']

            # now that the memory is setup, all python objects have had a chance to sync with the device
            # (in terms of indices, offsets etc.)
            self.calc_stats['Deal_Setup_Time'] = time.clock()
            self.netting_sets = DealStructure(Aggregation('root'))
            self.set_deal_structures(self.config.deals['Deals']['Children'], self.netting_sets, self.numscenarios,
                                     params['Batch_Size'], tagging=None,
                                     deal_level_mtm=params.get('DealLevel', False))

            # record the (pure python) dependency setup time
            self.calc_stats['Deal_Setup_Time'] = time.clock() - self.calc_stats['Deal_Setup_Time']

            # record how long it took to build the graph
            self.calc_stats['Graph_Setup_Time'] = time.clock()

            # simulate the price factors
            for key, value in self.stoch_factors.items():
                with tf.name_scope(utils.check_scope_name(key)):
                    self.shared_mem.t_Scenario_Buffer[self.stoch_ofs[key]] = value.generate(self.shared_mem)

            # construct the valuations
            self.tensors['mtm'] = self.netting_sets.resolve_structure(self.shared_mem, self.time_grid)

            # now calculate all the valuation adjustments (if necessary)
            if 'CollVA' in params and 'Funding' in self.shared_mem.t_Credit:
                with tf.name_scope('CollVA'):
                    self.tensors['collva_t'] = tf.reduce_mean(self.shared_mem.t_Credit['Funding'], axis=1)
                    self.tensors['collva'] = tf.reduce_sum(self.tensors['collva_t'])
                    self.tensors['collateral'] = self.shared_mem.t_Credit['Collateral']

                    if params['CollVA'].get('Gradient', 'No') == 'Yes':
                        # calculate all the derivatives of collva
                        self.tensors['grad_collva'] = SensitivitiesEstimator(
                            self.tensors['collva'], self.all_var).grad

            if 'FVA' in params:
                time_grid = self.netting_sets.sub_structures[0].obj.Time_dep.deal_time_grid
                mtm_grid = self.time_grid.mtm_time_grid[time_grid]

                funding = get_interest_factor(utils.check_rate_name(params['FVA']['Funding_Interest_Curve']),
                                              self.static_ofs, self.stoch_ofs, self.all_tenors)
                riskfree = get_interest_factor(utils.check_rate_name(params['FVA']['Risk_Free_Curve']),
                                               self.static_ofs, self.stoch_ofs, self.all_tenors)

                if 'Counterparty' in params['FVA']:
                    survival = get_survival_factor(utils.check_rate_name(params['FVA']['Counterparty']),
                                                   self.static_ofs,
                                                   self.stoch_ofs, self.all_tenors)
                    surv = utils.calc_time_grid_curve_rate(survival, np.zeros((1, 3)), self.shared_mem)
                    St_T = tf.squeeze(tf.exp(-surv.gather_weighted_curve(
                        self.shared_mem, mtm_grid.reshape(1, -1), multiply_by_time=False)), axis=0)
                else:
                    St_T = tf.constant([[1.0]], self.prec)

                with tf.name_scope('FVA'):
                    deflation = utils.calc_time_grid_curve_rate(riskfree, np.zeros((1, 3)), self.shared_mem)
                    DF_rf = tf.squeeze(tf.exp(-deflation.gather_weighted_curve(
                        self.shared_mem, mtm_grid.reshape(1, -1))), axis=0)

                    if params['FVA'].get('Stochastic_Funding', 'No') == 'Yes':
                        Vk_plus_ti = tf.nn.relu(self.tensors['mtm'] * DF_rf)
                        Vk_minus_ti = tf.nn.relu(-self.tensors['mtm'] * DF_rf)
                        Vk_star_ti_p = (Vk_plus_ti[1:] + Vk_plus_ti[:1]) / 2
                        Vk_star_ti_m = (Vk_minus_ti[1:] + Vk_minus_ti[:1]) / 2

                        delta_scen_t = np.hstack((0.0, np.diff(mtm_grid))).astype(self.prec).reshape(-1, 1)

                        discount_fund = utils.calc_time_grid_curve_rate(funding,
                                                                        self.time_grid.time_grid[time_grid],
                                                                        self.shared_mem)
                        discount_rf = utils.calc_time_grid_curve_rate(riskfree, self.time_grid.time_grid[time_grid],
                                                                      self.shared_mem)

                        delta_fund_rf = tf.squeeze(
                            tf.exp(discount_fund.gather_weighted_curve(self.shared_mem, delta_scen_t)) -
                            tf.exp(discount_rf.gather_weighted_curve(self.shared_mem, delta_scen_t)), axis=1) * St_T

                        FCA_t = tf.reduce_mean(delta_fund_rf[1:] * Vk_star_ti_p, axis=1)
                        FCA = tf.reduce_sum(FCA_t)
                        FBA_t = tf.reduce_mean(delta_fund_rf[1:] * Vk_star_ti_m, axis=1)
                        FBA = tf.reduce_sum(FBA_t)
                    else:
                        pass

                    self.tensors['fva'] = FCA - FBA

                    if params['FVA'].get('Gradient', 'No') == 'Yes':
                        # calculate all the derivatives of fva
                        self.tensors['grad_fva'] = SensitivitiesEstimator(
                            self.tensors['fva'], self.all_var).grad

            if 'CVA' in params:
                discount = get_interest_factor(utils.check_rate_name(params['Deflation_Interest_Rate']),
                                               self.static_ofs, self.stoch_ofs, self.all_tenors)
                survival = get_survival_factor(utils.check_rate_name(params['CVA']['Counterparty']),
                                               self.static_ofs,
                                               self.stoch_ofs, self.all_tenors)
                recovery = get_recovery_rate(utils.check_rate_name(params['CVA']['Counterparty']), self.all_factors)
                # only looks at the first netting set - should be fine . . .
                time_grid = self.netting_sets.sub_structures[0].obj.Time_dep.deal_time_grid

                # Calculates unilateral CVA with or without stochastic deflation.
                with tf.name_scope('CVA'):
                    mtm_grid = self.time_grid.mtm_time_grid[time_grid]
                    delta_scen_t = np.hstack((0.0, np.diff(mtm_grid))).astype(self.prec)

                    if params['CVA']['Deflate_Stochastically'] == 'Yes':
                        zero = utils.calc_time_grid_curve_rate(discount, self.time_grid.time_grid[time_grid],
                                                               self.shared_mem)
                        Dt_T = tf.exp(-tf.cumsum(tf.squeeze(zero.gather_weighted_curve(
                            self.shared_mem, delta_scen_t.reshape(-1, 1))), axis=0))
                    else:
                        zero = utils.calc_time_grid_curve_rate(discount, np.zeros((1, 3)), self.shared_mem)
                        Dt_T = tf.squeeze(tf.exp(-zero.gather_weighted_curve(
                            self.shared_mem, mtm_grid.reshape(1, -1))), axis=0)

                    pv_exposure = tf.nn.relu(self.tensors['mtm'] * Dt_T)

                    if params['CVA']['Stochastic_Hazard'] == 'Yes':
                        surv = utils.calc_time_grid_curve_rate(survival, self.time_grid.time_grid[time_grid],
                                                               self.shared_mem)
                        St_T = tf.exp(-tf.cumsum(tf.squeeze(surv.gather_weighted_curve(
                            self.shared_mem, delta_scen_t.reshape(-1, 1), multiply_by_time=False), axis=1),
                            axis=0))
                    else:
                        surv = utils.calc_time_grid_curve_rate(survival, np.zeros((1, 3)), self.shared_mem)
                        St_T = tf.squeeze(tf.exp(-surv.gather_weighted_curve(
                            self.shared_mem, mtm_grid.reshape(1, -1), multiply_by_time=False)), axis=0)

                    prob = St_T[:-1] - St_T[1:]

                    self.tensors['cva'] = (1.0 - recovery) * tf.reduce_sum(
                        tf.reduce_mean(0.5 * (pv_exposure[1:] + pv_exposure[:-1]) * prob, axis=1), axis=0)

                    if params['CVA'].get('Gradient', 'No') == 'Yes':
                        # potentially fetch ir jacobian matrices for base curves
                        base_ir_curves = [x for x in self.stoch_ofs.keys() if
                                          x.type == 'InterestRate' and len(x.name) == 1]
                        self.jacobians = {}
                        for ir_factor in base_ir_curves:
                            jacobian_factor = utils.Factor('InterestRateJacobian', ir_factor.name)
                            ir_curve = self.stoch_factors[utils.Factor('InterestRate', ir_factor.name)].factor
                            var_name = 'Stochastic_Input/{0}:0'.format(utils.check_tuple_name(ir_factor))
                            try:
                                jac = construct_factor(jacobian_factor, self.config.params['Price Factors'])
                                jac.update(ir_curve)
                                self.jacobians[var_name] = jac.current_value()
                                logging.info('jacobian present for {0} - will attempt inverse bootstrap'.format(
                                    utils.check_tuple_name(ir_factor)))
                            except KeyError as e:
                                pass

                        # calculate all the derivatives of cva
                        sensitivity = SensitivitiesEstimator(self.tensors['cva'], self.all_var)
                        self.tensors['grad_cva'] = sensitivity.grad
                        # store the size of the Gradient
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P
                        if params['CVA'].get('Hessian', 'No') == 'Yes':
                            # calculate the hessian matrix - warning - this will probably kill your box
                            self.tensors['grad_cva_hessian'] = sensitivity.get_H_op(self.prec)

            self.calc_stats['Graph_Setup_Time'] = time.clock() - self.calc_stats['Graph_Setup_Time']

        results = {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Jacobians': self.jacobians}

        if params.get('Debug', 'No') != 'No' and os.path.isdir(params['Debug']):
            # record the number of tensors used (lower is better) - this is an expensive op
            self.calc_stats['Graph_Nodes'] = len(self.graph.as_graph_def().node)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            writer = tf.summary.FileWriter(params['Debug'], graph=self.graph)
            results['Results'] = self.feed(params['Batch_Size'], options=run_options, run_meta=run_metadata,
                                           writer=writer)
        else:
            results['Results'] = self.feed(params['Batch_Size']) if feedgraph else None

        return results


class Base_Revaluation(Calculation):
    """Simple deal revaluation - Use this to reconcile with the source system"""
    documentation = ('Calculations',
                     ['This applies the valuation models mentioned earlier to the portfolio per deal.',
                      '',
                      'The only inputs are:',
                      '',
                      '- **Currency** of the output.',
                      '- **Run_Date** at which the marketdata should be applied (i.e. $t_0$)',
                      '',
                      'The output is a dictionary containing the DealStructure and the calculation computation',
                      'statistics.'
                      ])

    def __init__(self, config, **kwargs):
        super(Base_Revaluation, self).__init__(config, **kwargs)
        self.dependent_factors = None
        self.base_date = None

        # Cuda related variables to store the state of the device between calculations
        self.shared_memClass = namedtuple('shared_mem',
                                          't_Buffer t_Static_Buffer t_Feed_dict t_Cashflows calc_greeks \
                                          gpus riskneutral precision simulation_batch Report_Currency')

        # prepare the risk factor output matrix . .
        self.static_var = []
        self.static_values = []
        self.static_ofs = OrderedDict()

    def update_factors(self, params, base_date):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, '0d', False)

        # update the time grid
        self.update_time_grid(base_date)

        self.static_factors = OrderedDict()
        for price_factor in dependent_factors:
            try:
                self.static_factors.setdefault(price_factor,
                                               construct_factor(
                                                   price_factor,
                                                   self.config.params['Price Factors']))
            except KeyError as e:
                logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.args))

        self.all_factors = self.static_factors

        # work out distinct tenor sets
        # distinct_tenors = OrderedDict()
        with tf.name_scope("Static_Input"):
            for key, value in self.static_factors.items():
                if key.type not in utils.DimensionLessFactors:
                    # record the offset of this risk factor
                    self.static_ofs.setdefault(key, len(self.static_var))
                    # utils.distinct_tenors(distinct_tenors, key, value)
                    current_val = value.current_value()
                    self.static_values.append(current_val)
                    self.static_var.append(
                        tf.placeholder(self.prec,
                                       shape=len(current_val),
                                       name=utils.check_scope_name(key)))

        # setup the device and allocate memory
        self.__refresh_shared_mem(params['Currency'], params.get('Greeks', 'No') == 'Yes')

        # calculate a reverse lookup for the tenors and store the daycount code
        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        # store the state info
        self.dependent_factors = dependent_factors

    def update_time_grid(self, base_date):
        # setup the scenario and base time grids
        self.time_grid = TimeGrid({base_date}, {base_date}, {base_date})
        self.base_date = base_date
        self.time_grid.set_base_date(base_date)

    def __refresh_shared_mem(self, reporting_currency, calc_greeks):

        # setup the Feed dictionary - the inputs needed to evaluate the graph
        feed = dict(zip(self.static_var, self.static_values))
        static_buffer = self.static_var
        # name of the base currency
        base_currency = 'Static_Input/FxRate.{}'.format(
            self.config.params['System Parameters']['Base_Currency']),

        # now decide what we want to calculate greeks with respect to
        all_vars_concat = None
        if calc_greeks:
            all_vars_concat = [x for x in self.static_var if not x.name.startswith(
                base_currency)]
            self.make_factor_index(self.static_var)

        # allocate memory on the device
        with tf.name_scope('Setup'):
            self.shared_mem = self.shared_memClass(
                t_Buffer={},
                t_Static_Buffer=static_buffer,
                t_Feed_dict=feed,
                t_Cashflows=None,
                calc_greeks=all_vars_concat,
                gpus=[x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'],
                riskneutral=False,
                precision=self.prec,
                simulation_batch=1,
                Report_Currency=get_fxrate_factor(utils.check_rate_name(reporting_currency),
                                                  self.static_ofs, {})
            )

    def feed(self, static_vars={}):

        def check_prices(sess, feed, n, parent=[]):

            def format_row(deal, data, val, greeks):
                data['Deal Currency'] = deal.Factor_dep.get(
                    'Local_Currency', deal.Instrument.field.get('Currency'))
                for k, v in val.items():
                    if k.startswith('Greeks'):
                        greeks.setdefault(k, []).append(
                            self.gradients_as_df(v, header=deal.Instrument.field.get('Reference')))
                    else:
                        data[k] = float(v)
                # update any tags
                if deal.Instrument.field.get('Tags') is not None:
                    data.update(dict(zip(tag_titles, deal.Instrument.field['Tags'][0].split(','))))

            block = []
            greeks = {}
            for sub_struct in n.sub_structures:
                data = OrderedDict(parent + [(field, sub_struct.obj.Instrument.field.get(field, 'Root'))
                                             for field in ['Reference', 'Object']])
                format_row(sub_struct.obj, data, sess.run(sub_struct.obj.Calc_res, feed), greeks)
                block.append(data)
                sub_block, sub_greeks = check_prices(
                    sess, feed, sub_struct, parent + [('Parent' + str(len(parent)), data['Reference'])])
                block.extend(sub_block)
                # aggregate the sub structure greeks
                for k, v in sub_greeks.items():
                    greeks.setdefault(k, []).extend(v)

            valuations = sess.run([deal.Calc_res for deal in n.dependencies], feed)
            for deal, val in zip(n.dependencies, valuations):
                data = OrderedDict(parent + [(field, deal.Instrument.field.get(field, '?'))
                                             for field in ['Reference', 'Object']])
                format_row(deal, data, val, greeks)
                block.append(data)

            return block, greeks

        # update the feed
        feed = self.shared_mem.t_Feed_dict.copy()
        # copy the new params
        feed.update({self.graph.get_tensor_by_name('Static_Input/{0}:0'.format(k)): v for
                     k, v in static_vars.items()})

        # record the cuda execution stats
        self.calc_stats['Tensor_Execution_Time'] = time.clock()

        # clear the output
        self.output = {}
        # load any tag titles
        tag_titles = self.config.deals['Attributes'].get('Tag_Titles', '').split(',')
        # setup the config
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(graph=self.graph, config=config) as sess:
            mtm, greeks = check_prices(sess, feed, self.netting_sets)

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

        self.calc_stats['Tensor_Execution_Time'] = time.clock() - self.calc_stats['Tensor_Execution_Time']

        return self.output

    def execute(self, params, feedgraph=True):
        # get the rundate
        base_date = pd.Timestamp(params['Run_Date'])
        # reset the default graph
        tf.reset_default_graph()
        # create a TF graph to represent this calculation
        self.graph = tf.Graph()
        # store the params
        self.params = params
        # set the precision in the utils module - Nasty hack - Need to fix this - TODO
        utils.Default_Precision = self.prec
        # build the TF graph
        with self.graph.as_default() as g:
            self.update_factors(params, base_date)

            self.calc_stats['Deal_Setup_Time'] = time.clock()
            self.netting_sets = DealStructure(Aggregation('root'), deal_level_mtm=True)
            self.set_deal_structures(self.config.deals['Deals']['Children'], self.netting_sets, 1, 1,
                                     deal_level_mtm=True)

            # record the (pure python) dependency setup time
            self.calc_stats['Deal_Setup_Time'] = time.clock() - self.calc_stats['Deal_Setup_Time']

            self.calc_stats['Graph_Setup_Time'] = time.clock()

            # now ask the netting set to construct each deal - no looping required (just 1 timepoint)
            mtm = self.netting_sets.resolve_structure(self.shared_mem, self.time_grid)

            # record the graph loading time
            self.calc_stats['Graph_Setup_Time'] = time.clock() - self.calc_stats['Graph_Setup_Time']

        # return a dictionary of output
        return {'Netting': self.netting_sets, 'Stats': self.calc_stats,
                'Results': self.feed() if feedgraph else None}


def construct_calculation(calc_type, config, **kwargs):
    return globals().get(calc_type)(config, **kwargs)


if __name__ == '__main__':
    pass
