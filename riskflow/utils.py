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

import calendar
import traceback
from functools import reduce, wraps
from collections import namedtuple, OrderedDict
from typing import Tuple, List

import logging
import scipy.stats
import pandas as pd
import numpy as np

import torch

# For dealing with excel dates and dataframes
excel_offset = pd.Timestamp('1899-12-30 00:00:00')


def array_type(x): return np.array(x)


# Days in year - could set this to 365.0 or 365.25 if you want that bit extra time
DAYS_IN_YEAR = 365.25

# daycount codes
DAYCOUNT_None = -1
DAYCOUNT_ACT365 = 0
DAYCOUNT_ACT360 = 1
DAYCOUNT_ACT365IDSA = 2
DAYCOUNT_ACT30_360 = 3
DAYCOUNT_ACT30_E360 = 4
DAYCOUNT_ACTACTICMA = 5

# factor codes
FACTOR_INDEX_Stoch = 0  # either True for stochastic or False for static
FACTOR_INDEX_Offset = 1  # index to get the factor name
FACTOR_INDEX_SubType = 2  # index to get the factor subtype (if any)
# these are indices to get the tenors relevant to interpolate the risk factor in question
FACTOR_INDEX_Tenor_Index = 3
FACTOR_INDEX_Daycount = 4  # daycount code
FACTOR_INDEX_ExcelCalcDate = 4
FACTOR_INDEX_Moneyness_Index = 3
FACTOR_INDEX_Expiry_Index = 4
FACTOR_INDEX_VolTenor_Index = 5
FACTOR_INDEX_Flat_Index = 5
FACTOR_INDEX_Surface_Flat_Index = 6

# cashflow codes 
CASHFLOW_INDEX_Start_Day = 0
CASHFLOW_INDEX_End_Day = 1
CASHFLOW_INDEX_Pay_Day = 2

CASHFLOW_INDEX_Year_Frac = 3
# can also use this index for equity swaplet multipliers
CASHFLOW_INDEX_Start_Mult = 3

CASHFLOW_INDEX_Nominal = 4
# can also use this index for equity swaplet multipliers
CASHFLOW_INDEX_End_Mult = 4

CASHFLOW_INDEX_FixedAmt = 5

# Cashflow code for Float payments
CASHFLOW_INDEX_FloatMargin = 6
# Cashflow code for Fixed payments
CASHFLOW_INDEX_FixedRate = 6
# Cashflow code for caps/floor payments
CASHFLOW_INDEX_Strike = 6
# Cashflow code for equity swaplet multipliers
CASHFLOW_INDEX_Dividend_Mult = 6
# Cashflow code for possible FX resets
CASHFLOW_INDEX_FXResetDate = 7
CASHFLOW_INDEX_FXResetValue = 8
# used by inflation cashflows
CASHFLOW_INDEX_BaseReference = 9
CASHFLOW_INDEX_FinalReference = 10
CASHFLOW_OFFSET_Settle = 2

# Number of resets/fixings for this cashflow (0 for fixed cashflows)
CASHFLOW_INDEX_NumResets = 9
# offset in the reset/fixings array for this cashflow
CASHFLOW_INDEX_ResetOffset = 10
# Boolean (0 or 1) value that determines if this cashflow is settled (1) or accumulated (0)
CASHFLOW_INDEX_Settle = 11

# Cashflow calculation methods 
CASHFLOW_METHOD_IndexReference2M = 1
CASHFLOW_METHOD_IndexReference3M = 2
CASHFLOW_METHOD_IndexReferenceInterpolated3M = 3
CASHFLOW_METHOD_IndexReferenceInterpolated4M = 4

CASHFLOW_METHOD_Equity_Shares = 0
CASHFLOW_METHOD_Equity_Principal = 1
CASHFLOW_METHOD_Average_Interest = 0

CASHFLOW_METHOD_Compounding_Include_Margin = 2
CASHFLOW_METHOD_Compounding_Flat = 3
CASHFLOW_METHOD_Compounding_Exclude_Margin = 4
CASHFLOW_METHOD_Compounding_None = 5

CASHFLOW_METHOD_Fixed_Compounding_No = 0
CASHFLOW_METHOD_Fixed_Compounding_Yes = 1

CASHFLOW_IndexMethodLookup = {'IndexReference2M': CASHFLOW_METHOD_IndexReference2M,
                              'IndexReference3M': CASHFLOW_METHOD_IndexReference3M,
                              'IndexReferenceInterpolated3M': CASHFLOW_METHOD_IndexReferenceInterpolated3M,
                              'IndexReferenceInterpolated4M': CASHFLOW_METHOD_IndexReferenceInterpolated4M}

# reset codes - note that the first 3 fields correspond with the TIME_GRID
# (so that a reset can be treated as a timepoint)
RESET_INDEX_Time_Grid = 0
RESET_INDEX_Reset_Day = 1
RESET_INDEX_Scenario = 2
RESET_INDEX_Start_Day = 3
RESET_INDEX_End_Day = 4
RESET_INDEX_Weight = 5
RESET_INDEX_Value = 6
# used to store the reset accrual period
RESET_INDEX_Accrual = 7
# used to store any fx averaging (can't be used with accrual periods)
RESET_INDEX_FXValue = 7

# modifiers for dealing with a sequence of cashflows
SCENARIO_CASHFLOWS_FloatLeg = 0
SCENARIO_CASHFLOWS_Cap = 1
SCENARIO_CASHFLOWS_Floor = 2
SCENARIO_CASHFLOWS_Energy = 3
SCENARIO_CASHFLOWS_Index = 4
SCENARIO_CASHFLOWS_Equity = 5

# Constants for the time grid
TIME_GRID_PriorScenarioDelta = 0
TIME_GRID_MTM = 1
TIME_GRID_ScenarioPriorIndex = 2

# Collateral Cash Valuation mode
CASH_SETTLEMENT_Received_Only = 0
CASH_SETTLEMENT_Paid_Only = 1
CASH_SETTLEMENT_All = 2

# Factor sizes
FACTOR_SIZE_CURVE = 4
FACTOR_SIZE_RATE = 2

# Named tuples to make life easier
Factor = namedtuple('Factor', 'type name')
RateInfo = namedtuple('RateInfo', 'model_name archive_name calibration')
CalibrationInfo = namedtuple('CalibrationInfo', 'param correlation delta')
DealDataType = namedtuple('DealDataType', 'Instrument Factor_dep Time_dep Calc_res')
Partition = namedtuple('Partition', 'DealMTMs Collateral_Cash Funding_Cost Cashflows')
Collateral = namedtuple('Collateral', 'Haircut Amount Currency Funding_Rate Collateral_Rate Collateral')

# define 1, 2 and 3d risk factors - add more as development proceeds
DimensionLessFactors = ['DiscountRate', 'ReferenceVol', 'Correlation']
OneDimensionalFactors = ['InterestRate', 'InflationRate', 'DividendRate', 'SurvivalProb', 'ForwardPrice']
TwoDimensionalFactors = ['FXVol', 'EquityPriceVol', 'CommodityPriceVol']
ThreeDimensionalFactors = ['InterestRateVol', 'InterestYieldVol', 'ForwardPriceVol']
ImpliedFactors = ['HullWhite2FactorModelParameters', 'GBMAssetPriceTSModelParameters', 'PCAMixedFactorModelParameters']

# weekends and weekdays
WeekendMap = {'Friday and Saturday': 'Sun Mon Tue Wed Thu',
              'Saturday and Sunday': 'Mon Tue Wed Thu Fri',
              'Sunday': 'Mon Tue Wed Thu Fri Sat',
              'Saturday': 'Sun Mon Tue Wed Thu Fri',
              'Friday': 'Sat Sun Mon Tue Wed Thu'}


# Custom Exceptions
class InstrumentExpired(Exception):
    def __init__(self, message):
        self.message = message


def log_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as k:
            # we are okay to pass keyerrors back to the calling code
            raise
        except Exception as e:
            # Log the exception with traceback and context
            logging.error(f"An error occurred in function '{func.__name__}': {e}")
            logging.error(traceback.format_exc())
            # Re-raise the exception
            raise

    return wrapper


# Defined types - things like percentages, basis points etc.

class Descriptor:
    """Useful for arbitrary storage values"""

    def __init__(self, value):
        self.data = value
        self.descriptor_type = 'X'

    def __str__(self):
        return self.descriptor_type.join([str(x) for x in self.data])


class Percent:
    def __init__(self, amount):
        self.amount = amount / 100.0

    def __str__(self):
        return '%g%%' % (self.amount * 100.0)

    def __float__(self):
        return self.amount

    def __lt__(self, other):
        return self.amount < other.amount

    def __eq__(self, other):
        if isinstance(other, Percent):
            return self.amount == other.amount
        return NotImplemented

    def __hash__(self):
        return hash(self.amount)

    def __mul__(self, other):
        return self.amount * other

    def __repr__(self):
        return str(self)

    # define right multiply
    __rmul__ = __mul__


class Basis:
    def __init__(self, amount):
        self.amount = amount / 10000.0
        self.points = amount

    def __str__(self):
        return '%d bp' % self.points

    def __float__(self):
        return self.amount

    def __lt__(self, other):
        return self.amount < other.amount

    def __eq__(self, other):
        if isinstance(other, Basis):
            return self.amount == other.amount
        return NotImplemented

    def __hash__(self):
        return hash(self.points)

    def __mul__(self, other):
        return self.amount * other

    def __repr__(self):
        return str(self)

    # define right multiply
    __rmul__ = __mul__


class Curve:
    def __init__(self, meta, data):
        self.meta = meta
        self.array = array_type(sorted(data)) if isinstance(data, list) else data

    def __str__(self):
        def format1darray(data):
            return '(%s)' % ','.join(['%.12g' % y for y in data])

        array_rep = format1darray(self.array) if len(self.array.shape) == 1 else ','.join(
            [format1darray(x) for x in self.array])
        meta_rep = ','.join([str(x) for x in self.meta])
        return '[%s,%s]' % (meta_rep, array_rep) if meta_rep else '[%s]' % array_rep


class Offsets:
    lookup = {'months': 'm', 'days': 'd', 'years': 'y', 'weeks': 'w'}

    def __init__(self, data):
        self.grid = isinstance(data[0], list)
        self.data = data

    def __str__(self):
        ofs_fmt = lambda ofs: ''.join(['%d%s' % (v, Offsets.lookup[k]) for k, v in ofs.kwds.items()])
        if self.grid:
            periods = [ofs_fmt(value[0]) if len(value) == 1 else '{0}({1})'.format(*map(ofs_fmt, value)) for value in
                       self.data]
            return '{0}'.format(' '.join(periods))
        else:
            periods = [ofs_fmt(value) for value in self.data]
            return '[{0}]'.format(','.join(periods))


class DateList:
    def __init__(self, data):
        self.data = OrderedDict(data)
        self.dates = set()

    def __str__(self):
        return '\\'.join(
            ['%s=%.12g' % ('%02d%s%04d' % (x[0].day, calendar.month_abbr[x[0].month], x[0].year), x[1]) for x in
             self.data.items()]) + '\\'

    def sum_range(self, run_date, cuttoff_date):
        return sum([val for date, val in self.data.items() if run_date > date > cuttoff_date], 0.0)

    def prepare_dates(self):
        self.dates = set(self.data.keys())

    def consume(self, cuttoff, date):
        datelist = set([x for x in self.dates if x >= cuttoff]) if cuttoff else self.dates
        if datelist:
            closest_date = min(datelist, key=lambda x: np.abs((x - date).days))
            if closest_date <= date:
                self.dates.remove(closest_date)
            return closest_date, self.data[closest_date]
        else:
            return None, 0.0


class CreditSupportList:
    def __init__(self, data):
        self.data = OrderedDict(data)

    def value(self):
        return next(iter(self.data.values()))

    def __str__(self):
        return '\\'.join(['%d=%.12g' % (rating, amount) for rating, amount in self.data.items()]) + '\\'


class DateEqualList:
    def __init__(self, data):
        self.data = OrderedDict([(x[0], x[1:]) for x in data])

    def value(self):
        return self.data.values()

    def get(self, field):
        return self.data.get(field)

    def sum_range(self, run_date, cuttoff_date, index):
        return sum([val[index] for date, val in self.data.items() if run_date > date > cuttoff_date], 0.0)

    def __str__(self):
        return '[' + ','.join(['%s=%s' % (
            '%02d%s%04d' % (date.day, calendar.month_abbr[date.month], date.year), '='.join([str(y) for y in value]))
                               for date, value in self.data.items()]) + ']'


class Interpolation(object):
    def __init__(self, tensor, interp_params):
        self.tensor = tensor
        self.indexed_tensor = tensor.reshape(-1, tensor.shape[-1])
        self.interp_params = []
        for param in interp_params:
            self.interp_params.append(param.reshape(-1, param.shape[-1]))


class CurveTenor(object):
    def __init__(self, tenor_points, interp):
        # linear interpolation by default
        points = np.array(tenor_points)
        min_tenor = points.min()
        max_tenor = points.max()
        # check that dividends are defined >0
        if interp == 'Dividend':
            tenor_delta = (1.0 / np.array(tenor_points[:-1]).clip(1e-5, np.inf)) - \
                          (1.0 / np.array(tenor_points[1:]).clip(1e-5, np.inf))
            min_tenor = max(1e-5, min_tenor)
            max_tenor = max(1e-5, max_tenor)
        else:
            tenor_delta = np.diff(points)

        self.tenor = points
        self.delta = np.append(tenor_delta, 1.0)
        self.type = interp
        self.min = min_tenor
        self.max = max_tenor
        self.max_index = max(points.shape[0] - 1, 0)
        self.tensor_cache = {}

    def get_index(self, tenor_points_in_years):
        if isinstance(tenor_points_in_years, torch.Tensor):
            clipped_points = tenor_points_in_years.clip(self.min, self.max)
            tenor = self.tensor_cache.setdefault('tenor', tenor_points_in_years.new(self.tenor))
            delta = self.tensor_cache.setdefault('delta', tenor_points_in_years.new(self.delta))
            index = torch.searchsorted(tenor, clipped_points, right=True) - 1
        else:
            clipped_points = np.clip(tenor_points_in_years, self.min, self.max)
            tenor = self.tenor
            delta = self.delta
            index = tenor.searchsorted(clipped_points, side='right') - 1

        index_next = (index + 1).clip(0, self.max_index)

        if self.type == 'Dividend':
            alpha = (1.0 / tenor[index].clip(min=1e-5) -
                     1.0 / clipped_points) / delta[index]
        else:
            alpha = (clipped_points - tenor[index]) / delta[index]

        return index, index_next, alpha


@torch.jit.script
class Calculation_State(object):
    """
    Note that all pricing functions depend on this class being correctly setup. All calculations
    should inherit from this calculation state and extend accordingly
    """

    def __init__(self, static_buffer, unit, mcmc_sims, report_currency: List[Tuple[bool, int]],
                 nomodel: str, simulation_batch: int):
        # these are tensors
        self.t_Buffer = {}
        self.t_Static_Buffer = static_buffer
        # storing a unit tensor allows the dtype and device to be encoded in the calculation state
        self.one = unit
        self.fillvalue = unit.new_zeros((0, 1, simulation_batch))
        self.simulation_batch = simulation_batch
        self.Report_Currency = report_currency
        self.t_Cashflows = None
        # these are shared parameter states
        self.riskneutral = nomodel == 'RiskNeutral'
        self.MCMC_sims = mcmc_sims


# often we need a numpy array and its tensor equivalent at the same time
class DualArray:
    def __init__(self, tensor, ndarray):
        self.np = ndarray
        self.tn = tensor

    def __getitem__(self, x):
        return DualArray(self.tn[x], self.np[x])


# Tensor specific classes that's used internally
class TensorSchedule(object):
    def __init__(self, schedule, offsets):
        self.schedule = np.array(schedule)
        self.offsets = np.array(offsets)
        self.cache = {}
        self.unit = None

    def __getitem__(self, x):
        return self.schedule[x]

    def __len__(self):
        return len(self.schedule)

    def count(self):
        return self.schedule.shape[0]

    def reinitialize(self, unit):
        if self.unit is None:
            # set the unit tensor (this implicitly defines the dtype and the device)
            self.unit = unit
            self.cache = {}
        return self

    def dual(self, index=0):
        '''Returns just the schedule as a dual'''
        if 'dual' not in self.cache:
            self.cache['dual'] = DualArray(self.unit.new_tensor(self.schedule), self.schedule)
        return self.cache['dual'][index:]

    def merged(self, unit, index=0):
        '''Returns the schedule and offsets as a dual'''
        if self.unit is None:
            self.reinitialize(unit)

        if 'dual' not in self.cache:
            if len(self.schedule) < len(self.offsets):
                raise Exception('Schedule and offset mismatch')
                merged = np.concatenate((self.schedule, self.offsets[:len(self.schedule)]), axis=1)
            else:
                merged = np.concatenate((self.schedule, self.offsets), axis=1)
            self.cache['dual'] = DualArray(self.unit.new_tensor(merged), merged)
        return self.cache['dual'][index:]


class DealTimeDependencies(object):
    def __init__(self, mtm_time_grid, deal_time_grid):
        self.mtm_time_grid = mtm_time_grid
        self.delta = np.hstack(((mtm_time_grid[deal_time_grid[1:]] -
                                 mtm_time_grid[deal_time_grid[:-1]]), [1]))
        self.interp = mtm_time_grid[mtm_time_grid <= mtm_time_grid[deal_time_grid[-1]]]
        self.deal_time_grid = deal_time_grid
        # store the indices for linear interpolation
        self.update_indices()

    def assign(self, time_dependencies):
        # only assign up to the max of this set of dependencies
        expiry = self.deal_time_grid[-1]
        query = time_dependencies.deal_time_grid <= expiry
        self.delta = time_dependencies.delta[query]
        self.deal_time_grid = time_dependencies.deal_time_grid[query]
        self.interp = self.mtm_time_grid[self.mtm_time_grid <= self.mtm_time_grid[expiry]]
        # store the indices for linear interpolation
        self.update_indices()

    def update_indices(self):
        self.index = np.searchsorted(self.deal_time_grid, np.arange(self.interp.size), side='right') - 1
        self.index_next = (self.index + 1).clip(0, self.deal_time_grid.size - 1)
        self.alpha = (np.array((self.interp - self.interp[self.deal_time_grid[self.index]]) /
                               self.delta[self.index]).reshape(-1, 1))
        self.t_alpha = None

    def fetch_index_by_day(self, days):
        return self.interp.searchsorted(days)


# calculation time grid
class TimeGrid(object):
    def __init__(self, scenario_dates, MTM_dates, base_MTM_dates):
        self.scenario_dates = scenario_dates
        self.base_MTM_dates = base_MTM_dates
        self.CurrencyMap = {}
        self.report_index = None
        self.mtm_dates = MTM_dates
        self.date_lookup = dict([(x, i) for i, x in enumerate(sorted(MTM_dates))])

    def set_report_dates(self, base_date, report_dates):
        report_days = [(x - base_date).days for x in sorted(report_dates)]
        self.report_index = (self.mtm_time_grid.searchsorted(
            report_days, side='right') - 1).clip(0, self.mtm_time_grid.size - 1)

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
        self.scenario_grid[:, TIME_GRID_MTM] = self.scen_time_grid
        self.scenario_grid[:, TIME_GRID_ScenarioPriorIndex] = np.arange(self.scen_time_grid.size)

        # deal with the case that we need a very fine time_grid - note we do this after calculating the
        # scenario_grid as setting a non-null delta is a way to generate scenarios without calculating the
        # whole risk factor
        if delta is not None:
            delta_days, delta_tenors = delta
            delta_grid = np.union1d(np.arange(0, self.scen_time_grid.max(), delta_days), delta_tenors.round())
            self.scen_time_grid = np.union1d(self.scen_time_grid, delta_grid)

        self.time_grid_years = self.scen_time_grid / DAYS_IN_YEAR

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
            r = [self.date_lookup.get(x, max(self.date_lookup.values())) for x in dates]
            if r:
                dynamic_dates = self.base_time_grid.union(r)
            else:
                if max(dates) < min(self.date_lookup.keys()):
                    raise InstrumentExpired(e)

                # include this instrument but don't bother pricing it through time
                return DealTimeDependencies(self.mtm_time_grid, np.array([0]))

        # now construct the full deal grid
        deal_time_grid = np.array(sorted(dynamic_dates))
        # find the last dynamic date - should be the expiry date or the end of the grid
        expiry = self.date_lookup.get(max(dates), max(self.date_lookup.values()))
        # calculate the interpolation points etc.
        return DealTimeDependencies(self.mtm_time_grid, deal_time_grid[deal_time_grid <= expiry])


class TensorResets(TensorSchedule):
    def __init__(self, schedule, offsets):
        super(TensorResets, self).__init__(schedule, offsets)

        # Assign the offsets directly to the resets
        self.schedule[:, RESET_INDEX_Scenario] = self.offsets

    def known_resets(self, num_scenarios, index=RESET_INDEX_Value,
                     filter_index=RESET_INDEX_Reset_Day, include_today=False):
        key = ('known_resets', num_scenarios, include_today)
        if self.cache.get(key) is None:
            if include_today:
                # we only include today if we are dealing with equity resets
                self.cache[key] = [self.unit.new_full((1, num_scenarios), x[index])
                                   for x in self.schedule if x[filter_index] <= 0.0 and x[index] > 0]
            else:
                self.cache[key] = [self.unit.new_full((1, num_scenarios), x[index])
                                   for x in self.schedule if x[filter_index] < 0.0]
        return self.cache[key]

    def get_simulated_resets(self, max_time, forward, shared):
        within_horizon = (self.offsets > -1) & (self.schedule[:, RESET_INDEX_Reset_Day] <= max_time)
        sim_resets = self.dual()[within_horizon]
        known_resets = self.known_resets(shared.simulation_batch)
        old_resets = calc_time_grid_curve_rate(
            forward, sim_resets.np[:, :RESET_INDEX_Scenario + 1], shared)
        delta_start = (sim_resets.np[:, RESET_INDEX_Start_Day] -
                       sim_resets.np[:, RESET_INDEX_Reset_Day]).reshape(-1, 1)
        delta_end = (sim_resets.np[:, RESET_INDEX_End_Day] -
                     sim_resets.np[:, RESET_INDEX_Reset_Day]).reshape(-1, 1)
        reset_weights = (sim_resets.tn[:, RESET_INDEX_Weight] /
                         sim_resets.tn[:, RESET_INDEX_Accrual]).reshape(-1, 1, 1)

        reset_values = torch.expm1(
            old_resets.gather_weighted_curve(shared, delta_end, delta_start)) * reset_weights \
            if sim_resets.np.any() else shared.fillvalue

        # fetch all fixed resets
        return torch.squeeze(
            torch.concat(
                [shared.fillvalue if not known_resets else torch.stack(known_resets), reset_values], dim=0)
            , dim=1)

    def split_block_resets(self, reset_offset, t, date_offset=0):
        all_resets = self.schedule[reset_offset:]
        future_resets = np.searchsorted(all_resets[:, RESET_INDEX_Reset_Day] - date_offset, t)
        return future_resets

    def get_start_index(self, time_grid, offset=0):
        """Read the start index (relative to the time_grid) of each reset"""
        return np.searchsorted(self.schedule[:, RESET_INDEX_Reset_Day] - offset,
                               time_grid[:, TIME_GRID_MTM]).astype(np.int64)

    def split_groups(self, group_size):
        if self.cache.get(('groups', group_size)) is None:
            groups = []
            for i in range(group_size):
                group = TensorResets(self.schedule[i::group_size], self.offsets[i::group_size])
                groups.append(group.reinitialize(self.unit))
            self.cache[('groups', group_size)] = groups
        return self.cache.get(('groups', group_size))


class TensorCashFlows(TensorSchedule):
    def __init__(self, schedule, offsets):
        # check which cashflows are settlements (as opposed to accumulations)
        for cashflow, next_cashflow, cash_ofs in zip(schedule[:-1], schedule[1:], offsets[:-1]):
            if (next_cashflow[CASHFLOW_INDEX_Pay_Day] != cashflow[CASHFLOW_INDEX_Pay_Day]) or (
                    cashflow[CASHFLOW_INDEX_FixedAmt] != 0):
                cash_ofs[CASHFLOW_OFFSET_Settle] = 1

        # last cashflow always settles (if it's not marked as such) otherwise, it's a forward
        if offsets[-1][CASHFLOW_OFFSET_Settle] == 0:
            offsets[-1][CASHFLOW_OFFSET_Settle] = 1

        # Add Resets field
        self.Resets = None
        # call superclass
        super(TensorCashFlows, self).__init__(schedule, offsets)

    def get_resets(self, unit):
        return self.Resets.reinitialize(unit)

    def known_fx_resets(self, num_scenarios, index=CASHFLOW_INDEX_FXResetValue,
                        filter_index=CASHFLOW_INDEX_FXResetDate):

        if self.cache.get(('known_fx_resets', num_scenarios)) is None:
            self.cache[('known_fx_resets', num_scenarios)] = [
                self.Resets.unit.new_full((1, num_scenarios), x[index])
                for x in self.schedule if x[filter_index] < 0.0]
        return self.cache.get(('known_fx_resets', num_scenarios))

    def get_par_swap_rate(self, base_date, ir_curve):
        """Used to calculate the par swap rate for these cashflows given an interest rate curve"""
        Dt = ir_curve.get_day_count_accrual(base_date, self.schedule[:, CASHFLOW_INDEX_Pay_Day])
        D = np.exp(-ir_curve.current_value(Dt) * Dt) * self.schedule[:, CASHFLOW_INDEX_Year_Frac]
        if self.Resets is not None:
            T = ir_curve.get_day_count_accrual(base_date, self.Resets.schedule[:, RESET_INDEX_End_Day])
            t = ir_curve.get_day_count_accrual(base_date, self.Resets.schedule[:, RESET_INDEX_Start_Day])
            a = self.Resets.schedule[:, RESET_INDEX_Accrual]
            r = (np.exp(ir_curve.current_value(T) * T - ir_curve.current_value(t) * t) - 1.0) / a
            return (D * r).sum() / D.sum(), D.sum()
        else:
            return D.sum()

    def insert_cashflow(self, cashflow):
        """Inserts a cashflow at the beginning of the cashflow schedule - useful to model a fixed payment at the
        beginning of a schedule of cashflows"""
        self.schedule = np.vstack((cashflow, self.schedule))
        self.offsets = np.vstack(([0, 0, 1], self.offsets))

    def set_fixed_amount(self, rate):
        """sets the fixed amount to the rate provided"""
        self.schedule[:, CASHFLOW_INDEX_FixedAmt] = rate * self.schedule[:, CASHFLOW_INDEX_Nominal] * \
                                                    self.schedule[:, CASHFLOW_INDEX_Year_Frac]

    def add_maturity_accrual(self, reference_date, daycount_code):
        """Adjusts the last cashflow's daycount accrual fraction to include the maturity date"""
        last_cashflow = self.schedule[-1]
        last_cashflow[CASHFLOW_INDEX_Year_Frac] = get_day_count_accrual(
            reference_date + pd.offsets.Day(last_cashflow[CASHFLOW_INDEX_End_Day]),
            last_cashflow[CASHFLOW_INDEX_End_Day] - last_cashflow[CASHFLOW_INDEX_Start_Day] + 1, daycount_code)

    def set_resets(self, schedule, offsets):
        self.Resets = TensorResets(schedule, offsets)

    def overwrite_rate(self, attribute_index, value):
        """
        Overwrites the strike/fixed_amount/float_rate defined in the cashflow schedule
        """
        for cashflow in self.schedule:
            cashflow[attribute_index] = value
        self.cache = None

    def set_future_fx_resets(self, max_time, time_grid):
        FXResets = []
        valid = (self.schedule[:, CASHFLOW_INDEX_FXResetDate] <= max_time) & (
                self.schedule[:, CASHFLOW_INDEX_FXResetDate] >= 0)
        for cashflow in self.schedule:
            Reset_Day = cashflow[CASHFLOW_INDEX_FXResetDate]
            Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
            FXResets.append([Time_Grid, Reset_Day, Scenario])
        self.FXResets = np.array(FXResets)[valid]

    def add_mtm_payments(self, base_date, principal_exchange, effective_date, day_count):
        ''' MTM CCIRS's only need a zero marker for the nominal should the effective date be in the future '''
        if (principal_exchange in ['Start_Maturity', 'Start']) and base_date <= effective_date:
            dummy_cashflow = make_cashflow(
                base_date, base_date - pd.offsets.Day(1), effective_date,
                effective_date, 0.0, get_day_count(day_count), 0.0, 0.0)
            self.insert_cashflow(dummy_cashflow)

    def add_fixed_payments(self, base_date, principal_exchange, effective_date, day_count, principal):
        ''' Regular CCIRS's might need to exchange principle at the start and end '''
        if (principal_exchange in ['Start_Maturity', 'Start']) and base_date <= effective_date:
            self.insert_cashflow(
                make_cashflow(base_date, effective_date, effective_date, effective_date, 0.0, get_day_count(day_count),
                              -principal, 0.0))

        if principal_exchange in ['Start_Maturity', 'Maturity']:
            self.schedule[-1][CASHFLOW_INDEX_FixedAmt] = principal

    def get_cashflow_start_index(self, time_grid, field_index=CASHFLOW_INDEX_Pay_Day, last_payment=None):
        """Read the start index (relative to the time_grid) of each cashflow"""
        t_grid = time_grid[:, TIME_GRID_MTM]
        if last_payment:
            t_grid = time_grid[:, TIME_GRID_MTM].copy()
            t_grid[t_grid > last_payment] = self.schedule[:, CASHFLOW_INDEX_Pay_Day].max() + 1
        return np.searchsorted(self.schedule[:, field_index], t_grid).astype(np.int64)


def split_tensor(tensor, counts):
    return torch.split(tensor, tuple(counts)) if tensor.shape[0] == counts.sum() else [tensor] * counts.size


# @torch.jit.script
def calc_hermite_curve(t_a, g, c, curve_t0, curve_t1):
    one_minus_ta = (1.0 - t_a)
    return curve_t0 * one_minus_ta + t_a * (curve_t1 + one_minus_ta * (g + t_a * c))


class CurveTensor(object):
    '''
    This is a container for a curve tensor - a curve typically has tenor points per timepoint per scenario.
    The original simulation grid that gets computed at the start of each MC run is large enough as it is so
    we need a way to index into this original grid while keeping track of indices.
    Also contains information about any interpolation method other than linear.
    Note that the curve tensor is used directly by the tensorblock object
    '''

    def __init__(self, interp_obj: Interpolation, index, alpha):
        self.interp_obj = interp_obj
        self.index = index
        self.time_index = self.index.reshape(-1, 1) * interp_obj.tensor.shape[1]
        if alpha is not None:
            self.alpha = interp_obj.tensor.new(alpha) if isinstance(alpha, np.ndarray) else alpha
            self.index_next = (index + 1).clip(0, interp_obj.tensor.shape[0] - 1)
            self.time_index_next = self.index_next.reshape(-1, 1) * interp_obj.tensor.shape[1]
        else:
            self.alpha = None
            self.index_next = None
            self.time_index_next = None

    def interp_value(self):
        if self.alpha is not None:
            return self.interp_obj.tensor[self.index] * (1 - self.alpha) + \
                self.interp_obj.tensor[self.index_next] * self.alpha
        else:
            return self.interp_obj.tensor[self.index]

    def split(self, counts):
        sub_alpha = split_tensor(self.alpha, counts) if self.alpha is not None else [None] * counts.size

        return [CurveTensor(self.interp_obj, sub_index, sub_alpha)
                for sub_index, sub_alpha in zip(np.split(
                self.index, counts.cumsum()[:-1]), sub_alpha)]

    def interpolate_risk_neutral(self, curve_component, points, time_grid, time_multiplier):
        t = time_grid[:, 1].reshape(-1, 1)
        T = points + t
        return self.interpolate_curve(
            curve_component, t, time_multiplier) - self.interpolate_curve(
            curve_component, T, time_multiplier)

    def interpolate_curve(self, curve_component, points, time_factor):
        # our tensor object
        tensor = self.interp_obj.indexed_tensor
        # check the points being queried
        time_size, point_size = points.shape

        if point_size > 0:
            # get the points in years
            tenor_points_in_years = curve_component[FACTOR_INDEX_Daycount](points)

            curve_tenor = curve_component[FACTOR_INDEX_Tenor_Index]
            i1, i2, a = curve_tenor.get_index(tenor_points_in_years)

            # check if time_index is non-zero (valid if this was a stochastic factor)
            offset = self.time_index if self.time_index.any() else 0
            i00 = offset + i1
            i01 = offset + i2

            if self.time_index_next is not None:
                i10 = self.time_index_next + i1
                i11 = self.time_index_next + i2

            t_w2 = tensor.new(a).unsqueeze(dim=2)
            t_w1 = 1.0 - t_w2

            tenors = tensor.new(tenor_points_in_years).unsqueeze(dim=2)
            mult = tenors if time_factor else 1.0

            if curve_tenor.type.startswith('Hermite'):
                g, c = self.interp_obj.interp_params
                if curve_tenor.type == 'HermiteRT':
                    mult = mult / tenors.clamp(curve_tenor.min, curve_tenor.max)

                val = calc_hermite_curve(t_w2, g[i00,], c[i00,], tensor[i00,], tensor[i01,])

                if self.alpha is not None:
                    # need to linearly interpolate between 2 time points
                    val_t1 = calc_hermite_curve(t_w2, g[i10,], c[i10,], tensor[i10,], tensor[i11,])

                    val = (1 - self.alpha) * val + self.alpha * val_t1
            else:
                if curve_tenor.type == 'LinearRT':
                    mult = mult / tenors.clamp(curve_tenor.min, curve_tenor.max)
                # default to linear
                val = tensor[i00,] * t_w1 + tensor[i01,] * t_w2

                if self.alpha is not None:
                    val_t1 = tensor[i10,] * t_w1 + tensor[i11,] * t_w2

                    val = (1 - self.alpha) * val + self.alpha * val_t1

            return val * mult
        else:
            # return a null tensor
            return tensor.new_zeros([time_size, 0, tensor.shape[-1]])


class TensorBlock(object):
    def __init__(self, code, tensors: List[CurveTensor], time_grid: np.ndarray):
        self.code = code
        self.time_grid = time_grid
        self.curve_tensors = tensors
        self.local_cache = {}

    def split_counts(self, counts, shared):

        key_code = ('tensorblock', tuple([x[:2] for x in self.code]),
                    tuple(self.time_grid[:, TIME_GRID_MTM]),
                    tuple(counts))

        if key_code not in shared.t_Buffer:
            rate_tensor = zip(*[sub_tensor.split(counts) for sub_tensor in self.curve_tensors])
            time_block = np.split(self.time_grid, counts.cumsum())
            shared.t_Buffer[key_code] = [TensorBlock(self.code, tensor, time_t)
                                         for tensor, time_t in zip(rate_tensor, time_block)]

        return shared.t_Buffer[key_code]

    def gather_weighted_curve(self, shared, end_points,
                              start_points=None, multiply_by_time=True):

        # @torch.jit.script
        def calc_curve(time_multiplier, points):
            temp_curve = None
            for curve_tensor, curve_component in zip(self.curve_tensors, self.code):
                # handle static curves
                if not curve_component[FACTOR_INDEX_Stoch] and shared.riskneutral:
                    scaled_val = curve_tensor.interpolate_risk_neutral(
                        curve_component, end_points, self.time_grid, time_multiplier)
                else:
                    scaled_val = curve_tensor.interpolate_curve(curve_component, points, time_multiplier)

                if temp_curve is None:
                    temp_curve = scaled_val
                else:
                    temp_curve += scaled_val

            return temp_curve

        local_cache_key = (tuple(tuple(x) for x in end_points),
                           tuple(tuple(x) for x in start_points) if start_points is not None else None,
                           multiply_by_time)

        if local_cache_key not in self.local_cache:

            curve_points = calc_curve(1 if multiply_by_time else 0, end_points)

            if start_points is not None:
                curve_points -= calc_curve(1 if multiply_by_time else 0, start_points)
            self.local_cache[local_cache_key] = curve_points

        return self.local_cache[local_cache_key]

    def reduce_deflate(self, delta_scen_t, time_points, shared):
        DtT = torch.exp(-torch.squeeze(self.gather_weighted_curve(shared, delta_scen_t)).cumsum(dim=0))
        # we need the index just prior - note this needs to be checked in the calling code
        indices = self.time_grid[:, TIME_GRID_MTM].searchsorted(time_points) - 1
        return {t: DtT[index] for t, index in zip(time_points, indices)}


# date generation utils

def cds_dates(base, num_months):
    base_month = base.month
    initial = pd.DateOffset(months=(3 - base_month % 3) % 3, day=20)
    months = pd.DateOffset(months=3)
    last_date = (base + initial) if base.day < 20 else (base + initial + months)
    res = [last_date]

    while last_date < base + pd.DateOffset(months=num_months):
        last_date = last_date + months
        res.append(last_date)

    return res


def calc_cds_rates(R, survival, discount, base_date, CDS_tenors, all_factors, bump=0.01 * 0.01):
    def calc_par_cds(S_j, cds_tenor, delta=0.0, start_time=None, end_time=None):
        if delta:
            S_vals = S_j.copy()
            S_vals[start_time: end_time] += delta * S_ti[start_time: end_time]
        else:
            S_vals = S_j

        h = (S_vals[1:] - S_vals[:-1]) / (S_ti[1:] - S_ti[:-1])
        S = np.exp(-S_vals)
        F = D * S
        V_prot = ((F[:-1] - F[1:]) * h) / (h + f)

        cds_pay_dates = cds_dates(base_date, int(cds_tenor * 12))
        # insert the previous standard date (3 months prior)
        cds_pay_dates.insert(0, cds_pay_dates[0] - pd.DateOffset(months=3))
        tau = np.array([survival[FACTOR_INDEX_Daycount]((x - base_date).days) for x in cds_pay_dates])
        alpha = tau[1:] - tau[:-1]
        n = S_ti.searchsorted(tau[1:])
        v_fee = -tau[0]
        prev_n = 0

        for alpha_j, prev_tau, n_j in zip(alpha, tau[:-1], n):
            sub_i = slice(prev_n, n_j)
            sub_i_p1 = slice(prev_n + 1, n_j + 1)
            h_plus_f = h[sub_i] + f[sub_i]
            A_j = ((1 + h_plus_f * (S_ti[sub_i] - prev_tau)) * F[sub_i] - (
                    1 + h_plus_f * (S_ti[sub_i_p1] - prev_tau)) * F[sub_i_p1]) * h[sub_i] / h_plus_f ** 2
            v_fee += alpha_j * D[n_j] * S[n_j] + A_j.sum()
            prev_n = n_j

        v_prot = (1.0 - R) * V_prot[:n_j].sum()
        return v_prot / v_fee, n[-1]

    max_cds_dates = cds_dates(base_date, int(max(CDS_tenors) * 12))
    time_to_add = [survival[FACTOR_INDEX_Daycount]((x - base_date).days) for x in max_cds_dates]

    S_proc = all_factors[survival[FACTOR_INDEX_Offset]]
    D_proc = all_factors[discount[FACTOR_INDEX_Offset]]
    S_factor = S_proc.factor if hasattr(S_proc, 'factor') else S_proc
    D_factor = D_proc.factor if hasattr(D_proc, 'factor') else D_proc

    # calculate the piecewise hazard rate, forward rate and survival and discount curves
    S_ti = np.union1d(S_factor.get_tenor(), time_to_add)
    D_vals = D_factor.current_value(S_ti) * S_ti
    f = (D_vals[1:] - D_vals[:-1]) / (S_ti[1:] - S_ti[:-1])
    D = np.exp(-D_vals)

    S_vals_0 = S_factor.current_value(S_ti)
    CDS_rates = {}
    for tenor in CDS_tenors:
        CDS_rates[tenor] = calc_par_cds(S_vals_0, tenor)

    if bump:
        T = 0
        S_j = [S_vals_0]

        for k, v in CDS_rates.items():
            delta_j = scipy.optimize.brentq(
                lambda x: calc_par_cds(S_j[-1], k, delta=x, start_time=T, end_time=v[1] + 1)[0] - (v[0] + bump), -0.1,
                0.1)

            S_j.append(S_j[-1].copy())
            S_j[-1][T: v[1] + 1] += delta_j * S_ti[T: v[1] + 1]
            T = v[1] + 1

        return {k: v[0] for k, v in CDS_rates.items()}, S_ti, S_j
    else:
        return {k: v[0] for k, v in CDS_rates.items()}


# dataframe manipulation

def filter_data_frame(df, from_date, to_date, rate=None):
    index1 = (pd.Timestamp(from_date) - excel_offset).days
    index2 = (pd.Timestamp(to_date) - excel_offset).days
    return df.loc[index1:index2] if rate is None else df.loc[index1:index2][
        [col for col in df.columns if col.startswith(rate)]]


# Math Type stuff

def hermite_interpolation(tenors, rates):
    def calc_ri(t, r):
        r_i = ((np.diff(r[:-1]) * np.diff(t[1:])) / np.diff(t[:-1]) +
               (np.diff(r[1:]) * np.diff(t[:-1])) / np.diff(t[1:])) / (t[2:] - t[:-2])
        r_1 = (((r[1] - r[0]) * (t[2] + t[1] - 2.0 * t[0])) / (t[1] - t[0]) -
               (r[2] - r[1]) * (t[1] - t[0]) / (t[2] - t[1])) / (t[2] - t[0])
        r_n = -1.0 / (t[-1] - t[-3]) * ((r[-2] - r[-3]) * (t[-1] - t[-2]) / (t[-2] - t[-3]) -
                                        (r[-1] - r[-2]) * (2.0 * t[-1] - t[-2] - t[-3]) / (t[-1] - t[-2]))
        return np.append(np.append(r_1, r_i), r_n)

    def calc_gi(t, r, ri):
        return np.append(np.diff(t), 0.0) * ri - np.append(np.diff(r), 0.0)

    def calc_ci(t, r, ri):
        return np.append(2.0 * np.diff(r) - np.diff(t) * (ri[:-1] + ri[1:]), 0.0)

    ri = calc_ri(tenors, rates)
    gi = calc_gi(tenors, rates, ri)
    ci = calc_ci(tenors, rates, ri)
    return gi, ci


# @torch.jit.script
def norm_cdf(x):
    return 0.5 * (torch.erfc(x * -0.7071067811865475))


def norm_icdf(x):
    return 1.4142135623730951 * torch.erfinv(2.0 * x - 1.0)


def BivN(P, Q, rho):
    from scipy.stats import multivariate_normal
    mvn = np.vectorize(lambda x: multivariate_normal(cov=[[1.0, x], [x, 1.0]]))
    z2 = mvn(rho)
    cdf = np.vectorize(lambda z, x, y: z.cdf([x, y]))
    return cdf(z2, P, Q)


def ApproxBivN(P, Q, rho):
    # this is an approximation of the bivariate normal integral accurate to around 4 decimal
    # places - based on the paper from A Simple Approximation for Bivariate Normal Integral
    # Based on Error Function and its Application on Probit Model
    # with Binary Endogenous Regressor (Wen-Jen Tsay and Peng-Hsuan Ke)
    # might want to improve the accuracy of this but this is fast and vectorized

    # work out the cases
    denom = torch.sqrt(1.0 - rho * rho)
    a = -rho / denom
    b = P / denom
    numer = a * Q + b

    case1 = (a > 0.0) & (numer >= 0.0)
    case2 = (a > 0.0) & (numer < 0.0)
    case3 = (a < 0.0) & (numer >= 0.0)
    case4 = (a < 0.0) & (numer < 0.0)

    c1 = -1.0950081470333
    c2 = -0.75651138383854
    r2 = 1.4142135623730951
    ma2c2 = 1.0 - a * a * c2
    two_sq_ma2c2 = 2.0 * torch.sqrt(ma2c2)
    a2c1_2 = a * a * c1 * c1
    q_part = r2 * (Q - a * c2 * (a * Q + b))
    root4_p = torch.exp((a2c1_2 + 2 * b * (r2 * c1 + b * c2)) / (4.0 * ma2c2)) / (2.0 * two_sq_ma2c2)
    root4_m = torch.exp((a2c1_2 - 2 * b * (r2 * c1 - b * c2)) / (4.0 * ma2c2)) / (2.0 * two_sq_ma2c2)
    erf2_p = torch.erf((q_part + a * c1) / two_sq_ma2c2)
    erf2_m = torch.erf((q_part - a * c1) / two_sq_ma2c2)
    erf_p1 = (r2 * b) / (a * two_sq_ma2c2)
    erf_p2 = (a * a * c1) / (a * two_sq_ma2c2)
    erf1 = torch.erf(erf_p1 + erf_p2)
    erf3 = torch.erf(erf_p1 - erf_p2)
    final = norm_cdf(P) * norm_cdf(Q)

    for c, f in enumerate([case1, case2, case3, case4]):
        if f.any():
            if c == 0:
                case = .5 * (
                        torch.erf(Q / r2) + torch.erf(b / (r2 * a))) + root4_m * (
                               1.0 - erf3) - root4_p * (erf2_m + erf1)
            elif c == 1:
                case = root4_m * (1 + erf2_p)
            elif c == 2:
                case = .5 * (1 + torch.erf(Q / r2)) - root4_p * (1.0 + erf2_m)
            else:
                case = .5 * (1 - torch.erf(b / (r2 * a))) - root4_p * (1.0 - erf1) + root4_m * (erf2_p + erf3)

            final[f] = case[f]

    return final


def black_european_option_price(F, X, r, vol, tenor, buyOrSell, callOrPut):
    stddev = vol * np.sqrt(tenor)
    sign = 1.0 if (F > 0.0 and X > 0.0) else -1.0
    d1 = (np.log(F / X) + 0.5 * stddev * stddev) / stddev
    d2 = d1 - stddev
    return buyOrSell * callOrPut * (F * scipy.stats.norm.cdf(callOrPut * sign * d1) -
                                    X * scipy.stats.norm.cdf(callOrPut * sign * d2)) * np.exp(-r * tenor)


def Bjerksund_Stensland(A1, A2, B, x1, x2, K, sigma1, sigma2, rho, callOrPut):
    a = x2 + K
    b = x2 / a
    sigma1_2 = sigma1 * sigma1
    sigma2_2 = sigma2 * sigma2
    # make sure the variance is at least 1e-6
    v2 = torch.clamp(sigma1_2 - 2 * rho * sigma1 * b * sigma2 + b * b * sigma2_2, min=1e-6)
    v = torch.sqrt(v2)
    d = torch.log(x1 / a) / v
    d1 = d + v / 2
    d2 = d - (sigma1_2 - 2 * rho * sigma1 * sigma2 - b * b * sigma2_2 + 2 * b * sigma2_2) / (2 * v)
    d3 = d - (sigma1_2 - b * b * sigma2_2) / (2 * v)

    return A1 * x1 * norm_cdf(callOrPut * d1) + A2 * x2 * norm_cdf(callOrPut * d2) + B * norm_cdf(callOrPut * d3)


def black_european_option(F, X, vol, tenor, buyorsell, callorput, shared, cash_payoff=0.0):
    # calculates the black function WITHOUT discounting

    if isinstance(tenor, float):
        guard = (vol > 0.0) & (X > 0.0)
        stddev = vol.clamp(min=1e-5) * np.sqrt(tenor)
        strike = max(X, 1e-5) if isinstance(X, float) else X.clamp(min=1e-5)
    else:
        tenor_np = tenor.clip(min=0.0)
        tau_key = ('tenor', tenor_np.shape + tuple(tenor_np.flatten()))
        if tau_key not in shared.t_Buffer:
            shared.t_Buffer[tau_key] = vol.new(np.sqrt(tenor_np))

        tau = shared.t_Buffer[tau_key]
        guard = tau > 0.0

        if len(guard.shape) > 1:
            guard = torch.unsqueeze(guard, dim=2)
            sigma = vol * torch.unsqueeze(tau, dim=2)
        else:
            guard = torch.unsqueeze(guard, dim=1)
            sigma = vol * tau.reshape(-1, 1)

        stddev = sigma.clamp(min=1e-5)
        strike = X

    # make sure the forward is always >1e-5
    forward = torch.clamp(F, min=1e-5)

    if isinstance(strike, float) and strike == 0:
        # need to check if this is a put option (value is 0)
        # or a call option (value is just the forward)
        adjustment = 1.0 if callorput == 1.0 else 0.0
        prem = forward * adjustment
        value = forward * adjustment
    else:
        d1 = torch.log(forward / strike) / stddev + 0.5 * stddev
        d2 = d1 - stddev
        if cash_payoff:
            prem = cash_payoff * norm_cdf(callorput * d2)
            value = cash_payoff * (callorput * (forward - X) > 0) * shared.one
        else:
            prem = callorput * (forward * norm_cdf(callorput * d1) - X * norm_cdf(callorput * d2))
            value = torch.relu(callorput * (forward - X))
    return buyorsell * torch.where(guard, prem, value)


# tenor manipulation
def get_tenors(factor_dict):
    all_tenor = {}
    for factor_name, data in factor_dict.items():
        factor = data.factor if hasattr(data, 'factor') else data
        if hasattr(factor, 'get_tenor_indices'):
            indices = factor.get_tenor_indices()
            if isinstance(indices, dict):
                for k, v in indices.items():
                    new_factor_name = Factor(factor_name.type, factor_name.name + (k,))
                    all_tenor.setdefault(check_scope_name(new_factor_name), v)
            else:
                all_tenor.setdefault(check_scope_name(factor_name), indices)
    return all_tenor


def tenor_diff(tenor_points, interp='Linear'):
    return CurveTenor(tenor_points, interp)


def update_tenors(base_date, all_factors):
    def daycount_fn(base_date, daycount):
        def calc_daycount(time_in_days):
            return get_day_count_accrual(base_date, time_in_days, daycount)

        return calc_daycount

    all_tenors = {}
    for factor, factor_obj in all_factors.items():
        risk_factor = factor_obj.factor if hasattr(factor_obj, 'factor') else factor_obj

        if factor.type in OneDimensionalFactors or (
                factor.type in TwoDimensionalFactors and risk_factor.get_subtype()[0] in ['SVI', 'Skew']):
            tenor_points = risk_factor.get_tenor()

            if factor.type == 'DividendRate':
                tenor_data = tenor_diff(tenor_points, 'Dividend')
            elif factor.type in ['InterestRate', 'InflationRate']:
                tenor_data = tenor_diff(tenor_points, risk_factor.interpolation[0])
            else:
                tenor_data = tenor_diff(tenor_points)

            daycount = risk_factor.get_day_count()
            all_tenors[factor] = [tenor_data, daycount_fn(base_date, daycount)]

        # this is a surface of some kind
        elif factor.type in TwoDimensionalFactors:
            # we're going to dynamically interpolate when needed
            expiry_map = []
            for moneyness_points in risk_factor.index_map.values():
                expiry_map.append(tenor_diff(moneyness_points))
            # store the moneyness and expiry first
            all_tenors[factor] = [tenor_diff(risk_factor.get_moneyness()),
                                  tenor_diff(risk_factor.get_expiry()), expiry_map]

        elif factor.type in ThreeDimensionalFactors:
            if factor.type == 'ForwardPriceVol':
                # can interpolate dynamically when needed
                expiry_map = []
                for expiry_points in risk_factor.index_map[risk_factor.EXPIRY_INDEX]:
                    expiry_map.append(tenor_diff(expiry_points[0]))
                moneyness_map = []
                for moneyness_points in risk_factor.index_map[risk_factor.MONEYNESS_INDEX]:
                    moneyness_map.append(tenor_diff(moneyness_points[0]))
                # store the moneyness, expiry and tenor points
                all_tenors[factor] = [moneyness_map, expiry_map,
                                      tenor_diff(risk_factor.get_tenor()), risk_factor.index_map]
            else:
                # full surface defined - do not interpolate dynamically
                for dim_index, data in enumerate(
                        [risk_factor.get_moneyness(), risk_factor.get_expiry(), risk_factor.get_tenor()]):
                    all_tenors.setdefault(factor, [0, 0, 0])[dim_index] = tenor_diff(data)

    return all_tenors


# indexing ops manipulating large tensors
def interpolate_tensor(t, tenor, rate_tensor):
    dvt = np.concatenate(([1], np.diff(tenor), [1]))
    tenor_index = tenor.searchsorted(t, side='right')
    index = (tenor_index - 1).clip(0, tenor.size - 1)
    index_next = tenor_index.clip(0, tenor.size - 1)
    alpha = rate_tensor.new(((t - tenor[index]) / dvt[tenor_index]).clip(0, 1))
    return rate_tensor[index] * (1 - alpha) + rate_tensor[index_next] * alpha


# indexing ops manipulating large tensors
def gather_interp_matrix(mtm, deal_time_dep, shared):
    if deal_time_dep.alpha.any():
        if deal_time_dep.t_alpha is None:
            deal_time_dep.t_alpha = mtm.new(deal_time_dep.alpha)
        return mtm[deal_time_dep.index] * (1 - deal_time_dep.t_alpha) + \
            mtm[deal_time_dep.index_next] * deal_time_dep.t_alpha
    else:
        return mtm[deal_time_dep.index]


def gather_scenario_interp(interp_obj, time_grid, shared, as_curve_tensor=True):
    # calc the time interpolation weights
    index = time_grid[:, TIME_GRID_ScenarioPriorIndex].astype(np.int64)
    alpha_shape = tuple([-1] + [1] * (len(interp_obj.tensor.shape) - 1))
    alpha = time_grid[:, TIME_GRID_PriorScenarioDelta].reshape(alpha_shape)
    curve_tensor = CurveTensor(interp_obj, index, alpha if alpha.any() else None)
    return curve_tensor if as_curve_tensor else curve_tensor.interp_value()


def split_counts(rates, counts, shared):
    splits = []
    for rate in rates:
        if isinstance(rate, torch.Tensor):
            splits.append(split_tensor(rate, counts))
        else:
            splits.append(rate.split_counts(counts, shared))

    return zip(*splits)


def calc_fx_cross(rate1, rate2, time_grid, shared):
    key_code = ('fxcross', rate1[0], rate2[0], tuple(time_grid[:, TIME_GRID_MTM]))
    if rate1 != rate2:
        if key_code not in shared.t_Buffer:
            shared.t_Buffer[key_code] = calc_time_grid_spot_rate(
                rate1, time_grid, shared) / calc_time_grid_spot_rate(
                rate2, time_grid, shared)
    else:
        shared.t_Buffer[key_code] = shared.one
    return shared.t_Buffer[key_code]


def calc_discount_rate(block, tenors_in_days, shared, multiply_by_time=True):
    key_code = ('discount', tuple([x[:2] for x in block.code]),
                tuple(block.time_grid[:, TIME_GRID_MTM]),
                tenors_in_days.shape, tuple(tenors_in_days.ravel()))

    if key_code not in shared.t_Buffer:
        discount_rates = torch.exp(-block.gather_weighted_curve(
            shared, tenors_in_days, multiply_by_time=multiply_by_time))
        shared.t_Buffer[key_code] = discount_rates

    return shared.t_Buffer[key_code]


def calc_spot_forward(curve, T, time_grid, shared, only_diag):
    """
    Function for calculating the forward price of FX or EQ rates taking
    into account risk neutrality for static curves
    """
    curve_grid = calc_time_grid_curve_rate(curve, time_grid, shared)
    T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
    weights = np.diag(T_t).reshape(-1, 1) if only_diag else T_t
    return curve_grid.gather_weighted_curve(shared, weights)


def calc_dividend_samples(start_day, samples, time_grid):
    reset_start_day = start_day.clip(min=0)
    time_grid_scenario = [time_grid.get_scenario_offset(x) for x in reset_start_day]
    scenario = [x[1] for x in time_grid_scenario]
    time_interp = [x[0] for x in time_grid_scenario]
    resets = [TensorResets([[Time_Grid, reset_start, -1, reset_start, reset_end, 0.0, 0.0, 0.0]
                            for reset_end in samples], [scenario_offset] * len(samples))
              for Time_Grid, reset_start, scenario_offset in zip(time_interp, reset_start_day, scenario)]
    return resets


def calc_realized_dividends(s_t0, repo, div_yield, div_reset_stack, shared):
    # Calculate exp(sr) * (1 - exp(-sq))
    sr_minus_sq = torch.stack([
        torch.exp(torch.squeeze(calc_spot_forward(
            repo, div_resets[:, RESET_INDEX_End_Day], div_resets, shared, True), dim=1)
        ) * (1.0 - torch.exp(
            -torch.squeeze(calc_spot_forward(
                div_yield, div_resets[:, RESET_INDEX_End_Day], div_resets, shared, True), dim=1))
             )
        for div_resets in div_reset_stack], dim=1)

    return s_t0 * sr_minus_sq


def calc_eq_drift(repo, div_yield, weights, time_grid, shared, multiply_by_time=True):
    repo_curve_grid = calc_time_grid_curve_rate(repo, time_grid, shared)
    div_curve_grid = calc_time_grid_curve_rate(div_yield, time_grid, shared)
    return repo_curve_grid.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time) - div_curve_grid.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time)


def calc_eq_forward(equity, repo, div_yield, T, time_grid, shared, only_diag=False):
    T_scalar = isinstance(T, int)
    key_code = ('eqforward', equity[0], div_yield[0][:2], only_diag,
                T if T_scalar else tuple(T),
                tuple(time_grid[:, TIME_GRID_MTM]))

    if key_code not in shared.t_Buffer:
        T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
        spot = calc_time_grid_spot_rate(equity, time_grid, shared)

        if T_t.any():
            drift = torch.exp(
                calc_spot_forward(repo, T, time_grid, shared, only_diag) -
                calc_spot_forward(div_yield, T, time_grid, shared, only_diag))
        else:
            drift = shared.one.new_ones(
                [time_grid.shape[0], 1 if only_diag else T_t.size, 1])

        shared.t_Buffer[key_code] = spot * torch.squeeze(drift, dim=1) \
            if T_scalar else torch.unsqueeze(spot, dim=1) * drift

    return shared.t_Buffer[key_code]


def calc_fx_drift(local, other, weights, time_grid, shared, multiply_by_time=True):
    repo_local = calc_time_grid_curve_rate(local[1], time_grid, shared)
    repo_other = calc_time_grid_curve_rate(other[1], time_grid, shared)
    return repo_other.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time) - repo_local.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time)


def calc_fx_forward(local, other, T, time_grid, shared, only_diag=False):
    T_scalar = isinstance(T, int)
    key_code = ('fxforward', local[0][0], other[0][0], only_diag,
                T if T_scalar else tuple(T),
                tuple(time_grid[:, TIME_GRID_MTM]))
    if key_code not in shared.t_Buffer:
        if local[0] != other[0]:
            T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
            fx_spot = calc_fx_cross(local[0], other[0], time_grid, shared)

            if T_t.any():
                weights = np.diag(T_t).reshape(-1, 1) if only_diag else T_t
                drift = torch.exp(calc_fx_drift(local, other, weights, time_grid, shared))
            else:
                drift = fx_spot.new_ones([time_grid.shape[0], 1 if only_diag else T_t.size, 1])

            shared.t_Buffer[key_code] = fx_spot * torch.squeeze(drift, dim=1) \
                if T_scalar else torch.unsqueeze(fx_spot, dim=1) * drift
        else:
            shared.t_Buffer[key_code] = shared.one

    return shared.t_Buffer[key_code]


def gather_flat_surface(flat_surface, code, expiry, shared, calc_std):
    # cache the time surface interpolation matrix
    time_code = ('surface_flat', code[:2], tuple(expiry), calc_std)

    if time_code not in shared.t_Buffer:
        expiry_tenor = code[FACTOR_INDEX_Expiry_Index]
        moneyness_max_index = np.array([x.tenor.shape[0] for x in code[FACTOR_INDEX_Flat_Index]])
        exp_index = np.cumsum(np.append(0, moneyness_max_index[:-1]))
        time_modifier = np.sqrt(expiry).reshape(-1, 1) if calc_std else 1.0
        index, index_next, alpha = expiry_tenor.get_index(expiry)
        alpha = flat_surface.new(alpha.reshape(-1, 1, 1))
        subset = np.union1d(index, index_next)

        block_indices, block_alphas = [], []
        new_moneyness_tenor = reduce(np.union1d, [code[FACTOR_INDEX_Flat_Index][x].tenor for x in subset])

        for tenor_index in subset:
            moneyness_tenor = code[FACTOR_INDEX_Flat_Index][tenor_index]
            moneyness_index, moneyness_index_next, moneyness_alpha = moneyness_tenor.get_index(
                new_moneyness_tenor)

            block_indices.append(exp_index[tenor_index] + np.stack([moneyness_index, moneyness_index_next]))
            block_alphas.append(np.stack([1.0 - moneyness_alpha, moneyness_alpha]))

        # need to interpolate back to the tenor level
        money_indices, money_alpha = np.array(block_indices), np.array(block_alphas)
        subset_index = subset.searchsorted(index)
        tenor_money_indices = flat_surface.new_tensor(money_indices[subset_index], dtype=torch.int64)
        tenor_money_alpha = flat_surface.new(money_alpha[subset_index])
        subset_index_next = subset.searchsorted(index_next)
        tenor_money_alpha_next = flat_surface.new(money_alpha[subset_index_next])
        tenor_money_indices_next = flat_surface.new_tensor(money_indices[subset_index_next], dtype=torch.int64)

        surface = time_modifier * torch.sum(
            flat_surface.take(tenor_money_indices) * tenor_money_alpha * (1.0 - alpha) +
            flat_surface.take(tenor_money_indices_next) * tenor_money_alpha_next * alpha, dim=1)

        shared.t_Buffer[time_code] = (surface.reshape(-1), tenor_diff(new_moneyness_tenor))

    return shared.t_Buffer[time_code]


def gather_surface_interp(surface, code, expiry, shared, calc_std):
    # cache the time surface interpolation matrix
    time_code = ('surface_interp', code[:2], tuple(expiry), calc_std)

    if time_code not in shared.t_Buffer:
        expiry_tenor = code[FACTOR_INDEX_Expiry_Index]
        index, index_next, alpha = expiry_tenor.get_index(expiry)
        time_modifier = np.sqrt(expiry) if calc_std else 1.0
        alpha = surface.new(alpha).reshape(-1, 1)

        shared.t_Buffer[time_code] = (surface[index] * (1 - alpha) + surface[index_next] * alpha) * time_modifier

    return shared.t_Buffer[time_code]


def calc_moneyness_vol_rate(moneyness, expiry, key_code, shared):
    def calc_skew(x, t, atm_vol, s, L, R, C, D, lam, rho):
        skew_key = ('skew_params', t) + key_code[FACTOR_INDEX_Offset][0]

        if skew_key not in shared.t_Buffer:
            s2LC = s + 2.0 * L * C
            gamma = s2LC / (-2.0 * C * lam)
            beta = s2LC * (1.0 + 1.0 / lam)
            alpha = atm_vol + C * ((s - beta) + C * (L - gamma))

            # Right wing
            s2RD = s + 2.0 * R * D
            gamma_r = s2RD / (-2.0 * D * rho)
            beta_r = s2RD * (1.0 + 1.0 / rho)
            alpha_r = atm_vol + D * ((s - beta_r) + D * (R - gamma_r))

            shared.t_Buffer[skew_key] = (gamma, beta, alpha, gamma_r, beta_r, alpha_r)

        gamma, beta, alpha, gamma_r, beta_r, alpha_r = shared.t_Buffer[skew_key]
        lam_ok = lam.all()
        rho_ok = rho.all()

        # the 6 regions of the skew - check for 0 lam and rho - hold flat
        r1 = torch.ones_like(x) * (
            (alpha + C * (beta * (1.0 + lam) + gamma * (1.0 + lam) ** 2 * C)) if lam_ok else (atm_vol + C * (s + L * C)))
        r2 = alpha + x * (beta + gamma * x) if lam_ok else atm_vol + C * (s + L * C)
        r3 = atm_vol + x * (s + L * x)
        r4 = atm_vol + x * (s + R * x)
        r5 = alpha_r + x * (beta_r + gamma_r * x) if rho_ok else atm_vol + D * (s + R * D)
        r6 = torch.ones_like(x) * (
            (alpha_r + D * (beta_r * (1.0 + rho) + gamma_r * (1.0 + rho) ** 2 * D)) if rho_ok else (atm_vol + D * (s + R * D)))

        return torch.where(
            x <= (1 + lam) * C, r1,
                torch.where(x <= C, r2,
                            torch.where(x<=0, r3,
                                        torch.where(x<=D, r4,
                                                    torch.where(x<(1+rho)*D, r5, r6)
                                                    )
                                        )
                            )
                )

    if key_code[0] == 'vol_time_grid' and key_code[FACTOR_INDEX_Offset][0][0] in ['SVI', 'Skew']:
        surface, rate_code, calc_std = shared.t_Buffer[key_code]
        expiry_tenor = rate_code[FACTOR_INDEX_Tenor_Index]
        time_modifier = np.sqrt(expiry).reshape(-1, 1) if calc_std else 1.0
        index, index_next, alpha = expiry_tenor.get_index(expiry)
        alpha = shared.one.new(alpha.reshape(-1, 1))

        # need to calculate the correct way to query the vol surface
        if moneyness is None:
            moneyness = 0.0 * shared.one
        else:
            if rate_code[FACTOR_INDEX_SubType][1] == 'Sticky_Strike':
                atm_ref = surface['ATM_Ref'][index] * (1 - alpha) + surface['ATM_Ref'][index_next] * alpha
                moneyness = torch.log(moneyness / atm_ref)

        if rate_code[FACTOR_INDEX_SubType][0] == 'Skew':
            vol_prior = calc_skew(moneyness, tuple(index), surface['ATM_Vol'][index], surface['s'][index],
                                  surface['L'][index], surface['R'][index], surface['C'][index],
                                  surface['D'][index], surface['lam'][index], surface['rho'][index])
            vol_post = calc_skew(moneyness, tuple(index_next), surface['ATM_Vol'][index_next], surface['s'][index_next],
                                  surface['L'][index_next], surface['R'][index_next], surface['C'][index_next],
                                  surface['D'][index_next], surface['lam'][index_next], surface['rho'][index_next])
            vol = vol_prior * (1 - alpha) + vol_post * alpha
            return vol * time_modifier

        elif rate_code[FACTOR_INDEX_SubType][0] == 'SVI':
            k_m_prior = moneyness - surface['m'][index]
            var_prior = surface['a'][index] + surface['b'][index] * (
                    surface['rho'][index] * k_m_prior + torch.sqrt(k_m_prior ** 2 + surface['sigma'][index] ** 2))
            k_m_post = moneyness - surface['m'][index_next]
            var_post = surface['a'][index_next] + surface['b'][index_next] * (
                    surface['rho'][index_next] * k_m_post + torch.sqrt(
                k_m_post ** 2 + surface['sigma'][index_next] ** 2))
            variance = var_prior * (1 - alpha) + var_post * alpha
            return torch.sqrt(variance) * time_modifier
    else:
        surface, moneyness_tenor = shared.t_Buffer[key_code]
        max_index = np.prod(surface.shape) - 1
        if moneyness is None:
            moneyness = shared.one
        index, _, alpha = moneyness_tenor.get_index(moneyness)
        expiry_indices = np.arange(expiry.size).astype(np.int32)
        expiry_index_key = ('expiry_tenor', tuple(expiry_indices), moneyness_tenor.tenor.size)

        if expiry_index_key not in shared.t_Buffer:
            shared.t_Buffer[expiry_index_key] = shared.one.new_tensor(
                np.array([expiry_indices * moneyness_tenor.tenor.size]),
                dtype=torch.int32).T

        expiry_offsets = shared.t_Buffer[expiry_index_key]
        vol_index = index + expiry_offsets

        vol_index_next = torch.clamp(vol_index + 1, 0, max_index)
        vols = surface[vol_index] * (1.0 - alpha) + surface[vol_index_next] * alpha
        return vols


def calc_time_grid_vol_rate(code, moneyness, expiry, shared, calc_std=False):
    keys = []
    for rate in code:
        if rate[FACTOR_INDEX_SubType][0] in ['SVI', 'Skew']:
            keys.append((rate[FACTOR_INDEX_SubType][0], tuple(rate[:1] + tuple(rate[1]))))
        else:
            keys.append(('vol2d', rate[:2]))

    key_code = ('vol_time_grid', tuple(keys), tuple(expiry), calc_std)

    if key_code not in shared.t_Buffer:
        spread = None
        # We only support one vol stack at the moment - but can extend this to 2 or more
        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                if rate[FACTOR_INDEX_SubType][0] in ['SVI', 'Skew']:
                    spread = {x.name[-1]: shared.t_Static_Buffer[x].reshape(-1, 1) for x in rate[FACTOR_INDEX_Offset]}
                else:
                    spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

        # either interpolate a flat vol surface or a svi/skew vol param
        if code[0][FACTOR_INDEX_SubType][0] in ['SVI', 'Skew']:
            shared.t_Buffer[key_code] = (spread, code[0], calc_std)
        else:
            shared.t_Buffer[key_code] = gather_flat_surface(
                spread, code[0], expiry, shared, calc_std)

    return calc_moneyness_vol_rate(moneyness, expiry, key_code, shared)


def calc_tenor_time_grid_vol_rate(code, moneyness, expiry, tenor, shared, calc_std=False):
    key_code = ('vol3d', tuple([x[:2] for x in code]),
                tuple(expiry.flatten()), tenor, calc_std)

    if key_code not in shared.t_Buffer:
        vol_spread = None

        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                vol_spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

        tenor_index = code[0][FACTOR_INDEX_VolTenor_Index]
        space = vol_spread.reshape(tenor_index.tenor.size, -1)
        index, index_next, alpha = tenor_index.get_index(tenor)

        spread = (1.0 - alpha) * space[index] + alpha * space[index_next]

        surface = spread.reshape(-1, code[0][FACTOR_INDEX_Moneyness_Index].tenor.size)
        flat_vol_time = gather_surface_interp(surface, code[0], expiry, shared, calc_std).reshape(-1, )

        shared.t_Buffer[key_code] = (flat_vol_time, code[0][FACTOR_INDEX_Moneyness_Index])

    return calc_moneyness_vol_rate(moneyness, expiry, key_code, shared)


def calc_tenor_cap_time_grid_vol_rate(code, moneyness, expiry, tenor, shared, calc_std=False):
    key_code = ('vol3d_cap', tuple([x[:2] for x in code]), tenor, calc_std, tuple(expiry.flatten()))

    if key_code not in shared.t_Buffer:
        vol_spread = None

        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                vol_spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

        tenor_index = code[0][FACTOR_INDEX_VolTenor_Index]
        space = vol_spread.reshape(tenor_index.tenor.size, -1)
        index, index_next, alpha = tenor_index.get_index(tenor)

        spread = space[index] * (1.0 - alpha) + space[index_next] * alpha

        surface = spread.reshape(-1, code[0][FACTOR_INDEX_Moneyness_Index].tenor.size)
        result = []
        for exp, mon in zip(expiry, moneyness):
            time_exp = key_code[:-1] + tuple(exp)
            if time_exp not in shared.t_Buffer:
                flat_vol_time = gather_surface_interp(
                    surface, code[0], exp, shared, calc_std).reshape(-1)
                shared.t_Buffer[time_exp] = (flat_vol_time, code[0][FACTOR_INDEX_Moneyness_Index])
            result.append(calc_moneyness_vol_rate(mon, exp, time_exp, shared))

        shared.t_Buffer[key_code] = torch.stack(result)

    return shared.t_Buffer[key_code]


def calc_delivery_time_grid_vol_rate(code, moneyness, expiry, delivery, time_grid, shared):
    # can't cache this function as moneyness is generally stochastic
    vol_spread = None

    for rate in code:
        # Only static moneyness/expiry vol surfaces are supported for now
        if rate[FACTOR_INDEX_Stoch]:
            raise Exception("Stochastic vol surfaces not yet implemented")
        else:
            vol_spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
            break

    index_map = code[0][FACTOR_INDEX_Surface_Flat_Index]
    tenor_index = code[0][FACTOR_INDEX_VolTenor_Index]
    expiry_index = code[0][FACTOR_INDEX_Expiry_Index]
    money_index = code[0][FACTOR_INDEX_Moneyness_Index]

    # need to know the moneyness offset for a particular expiry offset
    expiry_offset = np.cumsum([0] + [x.tenor.size for x in expiry_index])
    t_index, t_index_next, alpha = tenor_index.get_index(delivery)
    alpha_tensor = vol_spread.new(alpha).unsqueeze(2)

    space = []
    tenor_cache = {}
    for current_tenor_index in [t_index, t_index_next]:
        result = []
        for tenor_sub_index, exp, mon in zip(current_tenor_index, expiry, moneyness):
            expiry_tenor_map = [expiry_index[to].get_index(e) for to, e in zip(tenor_sub_index, exp)]
            time_slice = []
            for tenor_offset, (e_index, e_index_next, e_alpha) in zip(tenor_sub_index, expiry_tenor_map):
                tenor_exp_key = (tenor_offset, e_index, e_index_next)
                if tenor_exp_key not in tenor_cache:
                    if expiry_index[tenor_offset].tenor.size > 1:
                        # need to interpolate the expiry
                        moneyness_00 = expiry_offset[tenor_offset] + e_index
                        moneyness_01 = expiry_offset[tenor_offset] + e_index_next

                        m_prior = vol_spread[slice(*index_map[2][moneyness_00][1:])]
                        m_next = vol_spread[slice(*index_map[2][moneyness_01][1:])]

                        # grab 2 moneyness layers
                        m_index_1, m_index_next_1, m_alpha_1 = money_index[moneyness_00].get_index(mon)
                        m_index_2, m_index_next_2, m_alpha_2 = money_index[moneyness_01].get_index(mon)

                        exp_prior = m_prior[m_index_1] * (1 - m_alpha_1) + m_prior[m_index_next_1] * m_alpha_1
                        exp_next = m_next[m_index_2] * (1 - m_alpha_2) + m_next[m_index_next_2] * m_alpha_2
                        tenor_cache[tenor_exp_key] = exp_prior * (1 - e_alpha) + exp_next * e_alpha

                    else:
                        # go straight to moneyness
                        moneyness_0 = expiry_offset[tenor_offset]
                        m_slice = vol_spread[slice(*index_map[2][moneyness_0][1:])]
                        m_index, m_index_next, m_alpha = money_index[moneyness_0].get_index(mon)
                        tenor_cache[tenor_exp_key] = m_slice[m_index] * (1 - m_alpha) + m_slice[m_index_next] * m_alpha

                time_slice.append(tenor_cache[tenor_exp_key])
            result.append(torch.stack(time_slice))
        space.append(result)

    interpolated_vols = [prior * (1 - a) + next * a for prior, next, a in zip(space[0], space[1], alpha_tensor)]

    return torch.stack(interpolated_vols)


def hermite_interpolation_tensor(t, rate_tensor):
    rate_diff = (rate_tensor[:, 1:, :] - rate_tensor[:, :-1, :])
    time_diff = t[:, 1:, :] - t[:, :-1, :]

    # calc r_i
    r_i = ((rate_diff[:, :-1, :] * time_diff[:, 1:, :]) / time_diff[:, :-1, :] +
           (rate_diff[:, 1:, :] * time_diff[:, :-1, :]) / time_diff[:, 1:, :]) / (
                  t[:, 2:, :] - t[:, :-2, :])
    r_1 = ((rate_diff[:, 0] * (t[:, 2, :] + t[:, 1, :] - 2.0 * t[:, 0, :])) / time_diff[:, 0, :] -
           (rate_diff[:, 1] * time_diff[:, 0, :]) / time_diff[:, 1, :]) / (t[:, 2, :] - t[:, 0, :])

    r_n = (-1.0 / (t[:, -1, :] - t[:, -3, :])) * (
            (rate_diff[:, -2] * time_diff[:, -1, :]) / time_diff[:, -2, :] -
            (rate_diff[:, -1] * (2.0 * t[:, -1, :] - t[:, -2, :] - t[:, -3, :])) / time_diff[:, -1, :])

    ri = torch.cat([torch.unsqueeze(r_1, dim=1), r_i, torch.unsqueeze(r_n, dim=1)], dim=1)

    # zero
    zero = torch.unsqueeze(torch.zeros_like(r_1), dim=1)
    # calc g_i
    gi = torch.cat([time_diff * ri[:, :-1, :] - rate_diff, zero], dim=1)
    # calc c_i
    ci = torch.cat([2.0 * rate_diff - time_diff * (ri[:, :-1, :] + ri[:, 1:, :]), zero], dim=1)

    return gi, ci


def make_curve_tensor(tensor, curve_component, time_grid, shared):
    key_code = (curve_component[FACTOR_INDEX_Tenor_Index].type, curve_component[:2],
                tuple(tensor.shape))

    if key_code not in shared.t_Buffer:
        # check if we are using all the tenor points
        if key_code[0] != 'Linear':
            curve_tenor = curve_component[FACTOR_INDEX_Tenor_Index].tenor
            if curve_tenor.size != tensor.shape[1]:
                t = tensor.new(curve_tenor[:tensor.shape[1]]).reshape(1, -1, 1)
            else:
                t = tensor.new(curve_tenor).reshape(1, -1, 1)

        if key_code[0] in ['HermiteRT', 'Hermite']:
            mod_tenor = tensor
            if curve_component[FACTOR_INDEX_Tenor_Index].type == 'HermiteRT':
                mod_tenor = tensor * t

            g, c = hermite_interpolation_tensor(t, mod_tenor)
            shared.t_Buffer[key_code] = Interpolation(mod_tenor, [g, c])

        elif key_code[0] == 'LinearRT':
            shared.t_Buffer[key_code] = Interpolation(tensor * t, [])
        else:
            shared.t_Buffer[key_code] = Interpolation(tensor, [])

    if time_grid is not None:
        return gather_scenario_interp(shared.t_Buffer[key_code], time_grid, shared)
    else:
        return CurveTensor(shared.t_Buffer[key_code], np.zeros(1, dtype=np.int64), None)


def calc_time_grid_curve_rate(code, time_grid, shared):
    time_hash = tuple(time_grid[:, TIME_GRID_MTM])
    code_hash = tuple([x[:2] for x in code])

    key_code = ('curve', code_hash, time_hash)

    if key_code not in shared.t_Buffer:
        value = []

        for rate in code:
            rate_code = ('curve_factor', rate[:2], time_hash)

            # check if the curve factors are already available
            if rate_code not in shared.t_Buffer:
                if rate[FACTOR_INDEX_Stoch]:
                    tensor = shared.t_Scenario_Buffer[rate[FACTOR_INDEX_Offset]]
                    spread = make_curve_tensor(tensor, rate, time_grid, shared)
                else:
                    tensor = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                    spread = make_curve_tensor(tensor.reshape(1, -1, 1), rate, None, shared)

                # store it
                shared.t_Buffer[rate_code] = spread

            # append the curve and its (possible) interpolation parameters
            value.append(shared.t_Buffer[rate_code])

            shared.t_Buffer[key_code] = TensorBlock(code=code, tensors=value, time_grid=time_grid)

    return shared.t_Buffer[key_code]


def calc_time_grid_spot_rate(rate, time_grid, shared):
    key_code = ('spot', tuple(rate[0][:2]), tuple(time_grid[:, TIME_GRID_MTM]))

    if key_code not in shared.t_Buffer:
        if rate[0][FACTOR_INDEX_Stoch]:
            tensor = shared.t_Scenario_Buffer[rate[0][FACTOR_INDEX_Offset]]
            value = gather_scenario_interp(Interpolation(tensor, []), time_grid, shared, as_curve_tensor=False)
        else:
            tensor = shared.t_Static_Buffer[rate[0][FACTOR_INDEX_Offset]]
            value = tensor.reshape(1, -1)

        shared.t_Buffer[key_code] = value

    return shared.t_Buffer[key_code]


def calc_curve_forwards(factor, tensor, time_grid_years, shared, mul_time=True):
    def prepare_tenors(factor_tenor, time_grid, extrapolate):
        """Prepare tenor grid with optional extrapolation."""
        tnr = factor_tenor.copy()
        amended_tensor = tensor
        if extrapolate:
            max_tenor = time_grid.max() + factor_tenor.max()
            tnr = np.append(tnr, max_tenor)
            # flat extrapolated gradient
            point_at_inf = tensor[-1:] + max_tenor * (tensor[-1:] - tensor[-2:-1]) / (
                    factor_tenor[-1] - factor_tenor[-2])
            amended_tensor = torch.cat([tensor, point_at_inf])

        tnr_d = np.diff(tnr, append=tnr.max() + 1)
        return tensor.new(tnr), tensor.new(tnr_d), amended_tensor

    def scale_for_rt(tnr, tensor, is_rt):
        """Scale tensor for rate*time interpolation."""
        if is_rt:
            return tensor * tnr
        return tensor

    def calculate_interp_params(tnr, tnr_d, time_grid):
        """Vectorized calculation of interpolation indices and weights."""
        #get the max index
        max_tnr_index = tnr.size()[0] - 1
        # Batch calculate all time + tenor combinations
        time_tenor = time_grid.view(-1, 1) + tnr.view(1, -1)

        # Find interpolation indices
        left_idx = (torch.searchsorted(tnr, time_tenor, right=True) - 1).clamp(min=0)
        right_idx = (left_idx + 1).clamp(max=max_tnr_index)

        left_time_idx = (torch.searchsorted(tnr, time_grid, right=True) - 1).clamp(min=0)
        right_time_idx = (left_time_idx + 1).clamp(max=max_tnr_index)

        alpha_1 = (time_tenor.clamp(max=tnr.max()) - tnr[left_idx]) / tnr_d[left_idx]
        alpha_2 = (time_grid - tnr[left_time_idx]).clamp(min=0.0) / tnr_d[left_time_idx]

        return (alpha_1, left_idx, right_idx), (alpha_2, left_time_idx, right_time_idx)

    def hermite_interpolation(tensor, tnr, indices_t, indices_time, time_grid, is_rt, mul_time):
        """Handle Hermite interpolation variants."""
        # Calculate Hermite coefficients
        t = tnr.view(1, -1, 1)
        time_grid_tenor = time_grid.view(-1, 1) + tnr.view(1, -1)
        norm = (time_grid_tenor if mul_time else 1.0, time_grid if mul_time else 1.0)
        if is_rt:
            norm = (norm[0] / time_grid_tenor.clamp(tnr.min(), tnr.max()),
                    norm[1] / time_grid.clamp(tnr.min(), tnr.max()))
        g, c = [torch.squeeze(x) for x in hermite_interpolation_tensor(t, tensor.reshape(1, -1, 1))]

        # Interpolate values
        alpha_1, ten_t, ten_t_next = indices_t
        alpha_2, ten_t1, ten_t1_next = indices_time
        val = calc_hermite_curve(alpha_1, g[ten_t], c[ten_t], tensor[ten_t], tensor[ten_t_next])
        val_t = calc_hermite_curve(alpha_2, g[ten_t1], c[ten_t1], tensor[ten_t1], tensor[ten_t1_next])
        return val * norm[0] - (val_t * norm[1]).reshape(-1, 1)

    def linear_interpolation(tensor, tnr, indices_t, indices_time, time_grid, is_rt, mul_time, extrapolate):
        """Handle linear interpolation variants."""
        alpha_1, ten_t, ten_t_next = indices_t
        alpha_2, ten_t1, ten_t1_next = indices_time

        # Linear interpolation
        val = alpha_1 * tensor[ten_t_next] + (1 - alpha_1) * tensor[ten_t]
        val_t = alpha_2 * tensor[ten_t1_next] + (1 - alpha_2) * tensor[ten_t1]

        if extrapolate:
            tnr = tnr[:-1]
            val = val[:, :-1]

        time_grid_tenor = time_grid.view(-1, 1) + tnr.view(1, -1)
        norm = (time_grid_tenor if mul_time else 1.0, time_grid if mul_time else 1.0)

        if is_rt:
            norm = (norm[0] / time_grid_tenor.clamp(tnr.min(), tnr.max()),
                    norm[1] / time_grid.clamp(tnr.min(), tnr.max()))

        return val * norm[0] - (val_t * norm[1]).reshape(-1, 1)

    """Compute interpolated forward curves with support for multiple interpolation methods."""
    # Configuration
    interp_method = factor.interpolation[0]
    is_rt = interp_method.endswith('RT')
    is_hermite = interp_method.startswith('Hermite')
    is_linear = 'Linear' in interp_method
    extrapolate = 'Extrapolate' in interp_method

    # Preprocess tensors
    factor_tenor = factor.get_tenor()
    time_grid = tensor.new(time_grid_years)
    tnr, tnr_d, tensor = prepare_tenors(factor_tenor, time_grid_years, extrapolate)

    # Handle RT scaling
    tensor = scale_for_rt(tnr, tensor, is_rt)

    # Calculate interpolation indices and weights
    indices_t, indices_time = calculate_interp_params(tnr, tnr_d, time_grid)

    # Perform interpolation
    if is_hermite:
        return hermite_interpolation(tensor, tnr, indices_t, indices_time, time_grid, is_rt, mul_time)
    else:
        return linear_interpolation(tensor, tnr, indices_t, indices_time, time_grid, is_rt, mul_time, extrapolate)


def PCA(matrix, num_redim=0):
    # Compute eigenvalues and sort into descending order
    evals, evecs = np.linalg.eig(matrix)
    indices = np.argsort(evals)[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]

    if num_redim > 0:
        evecs = evecs[:, :num_redim]
        evals = evals[:num_redim]

    var = np.diag(matrix)
    aki = evecs * np.sqrt(var.reshape(-1, 1).dot(1.0 / evals.reshape(1, -1)))
    # correlation = (np.identity(var.size)/np.sqrt(var)).dot(evecs).dot(np.identity(evals.size)*np.sqrt(evals))

    return aki, evecs, evals


def calc_statistics(data_frame, method='Log', num_business_days=252.0, frequency=1, max_alpha=4.0):
    """Currently only frequency==1 is supported"""

    def calc_alpha(x, y):
        return (-num_business_days * np.log(
            1.0 + ((x - x.mean(axis=0)) * (y - y.mean(axis=0))).mean(axis=0) / ((y - y.mean(axis=0)) ** 2.0).mean(
                axis=0))).clip(0.001, max_alpha)

    def calc_sigma2(x, y, alpha):
        return (x.var(axis=0) - ((1 - np.exp(-alpha / num_business_days)) ** 2) * y.var(axis=0)) * (
                (2.0 * alpha) / (1 - np.exp(-2.0 * alpha / num_business_days)))

    def calc_theta(x, y, alpha):
        return y.mean(axis=0) + x.mean(axis=0) / (1.0 - np.exp(-alpha / num_business_days))

    def calc_log_theta(theta, sigma2, alpha):
        return np.exp(theta + sigma2 / (4.0 * alpha))

    # TODO - implement weighting
    # delta = frequency / num_business_days

    transform = {'Diff': lambda x: x, 'Log': lambda x: np.log(x.clip(0.0001, np.inf))}[method]
    transformed_df = transform(data_frame)

    # can implement decay weights here if needed

    data = transformed_df.diff(frequency).shift(-frequency)
    y = transformed_df  #
    alpha = calc_alpha(data, y)
    theta = calc_theta(data, y, alpha)
    sigma2 = calc_sigma2(data, y, alpha)

    if method == 'Log':
        theta = calc_log_theta(theta, sigma2, alpha)
        # get rid of any infs
        theta.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ignore any outlier greater than 2 std deviations from the median
        median = theta.median()
        theta[np.abs(theta - median) > (2 * theta.std())] = np.nan

    stats = pd.DataFrame({
        'Volatility': data.std(axis=0) * np.sqrt(num_business_days),
        'Drift': data.mean(axis=0) * num_business_days,
        'Mean Reversion Speed': alpha,
        'Long Run Mean': theta,
        'Reversion Volatility': np.sqrt(sigma2)
    })

    correlation = data.corr()
    return stats, correlation, data


# Graph operations - needed for dependency solving

def traverse_dependents(x, adj):
    queue = adj[x][:]
    for i in queue:
        yield i
        if adj[i]:
            queue.extend([t for t in adj[i] if t not in queue])


def topological_sort(graph_unsorted):
    """
    Repeatedly go through all the nodes in the graph, moving each of
    the nodes that has all its edges resolved, onto a sequence that
    forms our sorted graph. A node has all of its edges resolved and
    can be moved once all the nodes its edges point to, have been moved
    from the unsorted graph onto the sorted one.

    NB - this destroys the graph_unsorted dictionary that was passed in
    and just returns the keys of the sorted graph
    """

    graph_sorted = []

    # Run until the unsorted graph is empty.
    while graph_unsorted:

        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append(node)

        if not acyclic:
            raise RuntimeError("A cyclic dependency occurred")

    return graph_sorted


# Data transformation utilities for constructing cashflows, calculating accruals etc.

def get_day_count(code):
    if code == 'ACT_365':
        return DAYCOUNT_ACT365
    elif code == 'ACT_360':
        return DAYCOUNT_ACT360
    elif code == '_30_360':
        return DAYCOUNT_ACT30_360
    elif code == '_30E_360':
        return DAYCOUNT_ACT30_E360
    elif code == 'ACT_365_ISDA':
        return DAYCOUNT_ACT365IDSA
    elif code == 'ACT_ACT_ICMA':
        return DAYCOUNT_ACTACTICMA
    else:
        raise Exception('Daycount {} Not implemented'.format(code))


def get_day_count_accrual(reference_date, time_in_days, code):
    """Need to complete this implementation. time_in_days is incremental"""

    if code == DAYCOUNT_ACT360:
        return time_in_days / 360.0
    elif code == DAYCOUNT_ACT365:
        return time_in_days / 365.0
    elif code in (DAYCOUNT_ACT365IDSA, DAYCOUNT_ACTACTICMA):
        # TODO
        return time_in_days / 365.0
    elif code == DAYCOUNT_ACT30_360:
        e1 = min(reference_date.day, 30)
        new_date = end_date = reference_date
        if isinstance(time_in_days, np.ndarray):
            ret = []
            for ed in time_in_days.tolist():
                end_date += pd.DateOffset(days=ed)
                e2 = 30 if end_date.day >= 30 and new_date.day >= 30 else end_date.day
                ret.append(((e2 - e1) + 30 * (end_date.month - new_date.month) +
                            360 * (end_date.year - new_date.year)) / 360.0)
                new_date = end_date
            return ret
        else:
            end_date = reference_date + pd.DateOffset(days=time_in_days)
            e2 = 30 if end_date.day >= 30 and reference_date.day >= 30 else end_date.day
            return ((e2 - e1) + 30 * (end_date.month - reference_date.month) +
                    360 * (end_date.year - reference_date.year)) / 360.0
    elif code == DAYCOUNT_ACT30_E360:
        e1 = min(reference_date.day, 30)
        new_date = end_date = reference_date
        if isinstance(time_in_days, np.ndarray):
            ret = []
            for ed in time_in_days.tolist():
                end_date += pd.DateOffset(days=ed)
                e2 = min(end_date.day, 30)
                ret.append(((e2 - e1) + 30 * (end_date.month - new_date.month) +
                            360 * (end_date.year - new_date.year)) / 360.0)
                new_date = end_date
            return ret
        else:
            end_date = reference_date + pd.DateOffset(days=time_in_days)
            e2 = min(end_date.day, 30)
            return ((e2 - e1) + 30 * (end_date.month - reference_date.month) +
                    360 * (end_date.year - reference_date.year)) / 360.0
    elif code == DAYCOUNT_None:
        return time_in_days


def get_fieldname(field, obj):
    """Needed to evaluate nested fields - e.g. collateral fields"""
    if isinstance(field, tuple):
        if len(field) == 1:
            try:
                return [element.get(field[0]) for element in obj if element.get(field[0])]
            except:
                return [obj[field[0]]] if obj.get(field[0]) else []
        else:
            return get_fieldname(field[1:], obj[field[0]] if obj.get(field[0]) else ({} if len(field) > 2 else [{}]))
    else:
        return [obj[field]] if obj.get(field) else []


def check_rate_name(name):
    """Needed to ensure that name is a tuple - Rate names need to be in upper case"""
    return tuple([x.upper() for x in name]) if type(name) == tuple else tuple(name.split('.'))


def check_tuple_name(factor):
    """Opposite of check_rate_name - used to make sure the name is a flat name"""
    return '.'.join((factor.type,) + factor.name) if type(factor.name) == tuple else factor


def check_scope_name(factor):
    """Uses check_tuple_name but makes sure TF can use the result as a scope name"""
    return check_tuple_name(factor).translate(
        str.maketrans({'#': '_', ':': '_', ' ': '_', '(': '_', '/': '_', '+': '_', '%': '_', '*': '_', ')': '_'}))


def check_fx_name(fx_correlation):
    """FX rates must be sorted alphabetically - however, often we need correlations with non-alphabetical fx rates.
    In this case, we need to know we're actually using the reverse pair (i.e. -1*rho) as opposed to the sorted name"""
    ccy1, ccy2 = fx_correlation
    return (1.0, (ccy1, ccy2)) if ccy1 < ccy2 else (-1.0, (ccy2, ccy1))


def check_tensor_name(name, scope):
    return '/'.join(name.split('/')[:2] + [scope]).translate(
        str.maketrans({'#': '_', ':': '_', ' ': '_', '(': '_', '+': '_', ')': '_'}))


def make_cashflow(reference_date, start_date, end_date, pay_date, nominal, daycount_code, fixed_amount, spread_or_rate):
    """
    Constructs a single cashflow vector with the provided parameters - can be used to manually construct nominal
    or fixed payments.
    """
    cashflow_days = [(x - reference_date).days for x in [start_date, end_date, pay_date]]
    return np.array(
        cashflow_days + [get_day_count_accrual(reference_date, cashflow_days[1] - cashflow_days[0], daycount_code),
                         nominal, fixed_amount, spread_or_rate, 0, 0])


def get_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, spread_or_rate):
    """
    Generates a vector of Start_day, End_day, Pay_day, Year_Frac, Nominal, FixedAmount (=0)
    and rate/spread from the parameters provided. Note that the length of the nominal array must
    be 1 less than the reset_dates (Since there is no nominal on the first reset date i.e.
    Effective date).
    The nominal could also be just a single number representing a vanilla (constant) profile

    Returns a vector of days (and nominals) relative to the reference date
    """

    amort_offsets = np.array([((k - reference_date).days, v) for k, v in amort.data.items()] if amort else [])
    day_offsets = np.array([(x - reference_date).days for x in reset_dates])

    nominal_amount, nominal_sign = [np.abs(nominal)], 1 if nominal > 0 else -1
    amort_index = 0
    for offset in day_offsets[1:]:
        amort_to_add = 0.0
        while amort_index < amort_offsets.shape[0] and amort_offsets[amort_index][0] <= offset:
            amort_to_add += amort_offsets[amort_index][1]
            amort_index += 1
        nominal_amount.append(nominal_amount[-1] - amort_to_add)
    nominal_amount = nominal_sign * np.array(nominal_amount)

    # we want the earliest negative number
    last_payment = np.where(day_offsets >= 0)[0]

    # calculate the index of the earliest cashflow
    previous_index = max(last_payment[0] - 1 if last_payment.size else day_offsets.size, 0)
    cashflows_left = day_offsets[previous_index:]
    rates = spread_or_rate if isinstance(nominal, np.ndarray) else [spread_or_rate] * (reset_dates.size - 1)
    ref_date = (reference_date + pd.offsets.Day(cashflows_left[0])) \
        if cashflows_left.any() else reference_date

    # order is start_day, end_day, pay_day, daycount_accrual, nominal, fixed amount, FxResetDate, FXResetValue

    return zip(cashflows_left[:-1], cashflows_left[1:], cashflows_left[1:],
               get_day_count_accrual(ref_date, np.diff(cashflows_left), daycount_code),
               nominal_amount[previous_index:], np.zeros(cashflows_left.size - 1), rates[previous_index:],
               np.zeros(cashflows_left.size - 1), np.zeros(cashflows_left.size - 1))


def generate_float_cashflows(reference_date, time_grid, reset_dates, nominal, amort, known_rate_list, reset_tenor,
                             reset_frequency, daycount_code, spread):
    """
    Generates a vector of Start_day, End_day, Pay_day, Year_Frac, Nominal, FixedAmount (=0)
    and spread from the parameters provided. Note that the length of the nominal array must
    be 1 less than the reset_dates (Since there is no nominal on the first reset date i.e.
    Effective date).
    The nominal could also be just a single number representing a vanilla (constant) profile

    Returns a vector of days (and nominals) relative to the reference date, as well as as
    the structure of resets
    """

    cashflow_schedule = list(get_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, spread))
    cashflow_reset_offsets = []
    all_resets = []
    reset_scenario_offsets = []

    # prepare to consume reset dates
    known_rates = known_rate_list if known_rate_list is not None else DateList({})
    known_rates.prepare_dates()

    min_date = None
    for cashflow in cashflow_schedule:
        r = []
        if next(iter(reset_frequency.kwds.values())) == 0.0:
            reset_days = np.array([reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_Start_Day]))])
            reset_tenor = pd.offsets.Day(cashflow[CASHFLOW_INDEX_End_Day] - cashflow[CASHFLOW_INDEX_Start_Day])
        else:
            reset_days = pd.date_range(reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_Start_Day])),
                                       reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_End_Day])),
                                       freq=reset_frequency, inclusive='left')
            reset_tenor = reset_frequency if next(iter(reset_tenor.kwds.values())) == 0.0 else reset_tenor

        for reset_day in reset_days:
            Reset_Day = (reset_day - reference_date).days
            Start_Day = (reset_day - reference_date).days
            End_Day = (reset_day + reset_tenor - reference_date).days
            Accrual = get_day_count_accrual(reference_date, End_Day - Start_Day, daycount_code)
            Weight = 1.0 / reset_days.size
            Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)

            # match the closest reset
            closest_date, Value = known_rates.consume(min_date, reset_day)
            if closest_date is not None:
                min_date = closest_date if min_date is None else max(min_date, closest_date)

            # only add a reset if it's in the past
            r.append([Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
                      Value / 100.0 if reset_day < reference_date else 0.0, Accrual])
            reset_scenario_offsets.append(Scenario)

            if Start_Day == End_Day:
                raise Exception("Reset Start and End Days coincide")

        # attach the reset_offsets to the cashflow - assume each cashflow is a settled one (not accumulated)
        cashflow_reset_offsets.append([len(r), len(all_resets), 1])
        # store resets
        all_resets.extend(r)

    cashflows = TensorCashFlows(cashflow_schedule, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def generate_fixed_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, fixed_rate):
    """
    Generates a vector of Start_day, End_day, Pay_day, Year_Frac, Nominal, FixedAmount (=0)
    and the fixed rate from the parameters provided. Note that the length of the nominal array must
    be 1 less than the reset_dates (Since there is no nominal on the first reset date i.e.
    Effective date).
    The nominal could also be just a single number representing a vanilla (constant) profile

    Returns a vector of days (and nominals) relative to the reference date
    """
    cashflow_schedule = list(get_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, fixed_rate))
    # Add the null resets to the end
    dummy_resets = np.zeros((len(cashflow_schedule), 3))

    return TensorCashFlows(cashflow_schedule, dummy_resets)


def make_fixed_cashflows(reference_date, position, cashflows, settlement_date):
    """
    Generates a vector of fixed cashflows from a data source taking nominal amounts into account.
    """
    cash = []
    reset_offsets = []

    for cashflow in sorted(
            cashflows['Items'], key=lambda x: (x['Payment_Date'], x.get('Accrual_Start_Date', x['Payment_Date']))):
        rate = cashflow['Rate'] if isinstance(cashflow['Rate'], float) else cashflow['Rate'].amount
        if cashflow['Payment_Date'] >= reference_date and (
                (cashflow['Payment_Date'] >= settlement_date) if settlement_date else True):
            # check the accrual dates - if none set it to the payment date
            Accrual_Start_Date = cashflow['Accrual_Start_Date'] if cashflow[
                'Accrual_Start_Date'] else cashflow['Payment_Date']
            Accrual_End_Date = cashflow['Accrual_End_Date'] if cashflow[
                'Accrual_End_Date'] else cashflow['Payment_Date']

            cash.append([(Accrual_Start_Date - reference_date).days, (Accrual_End_Date - reference_date).days,
                         (cashflow['Payment_Date'] - reference_date).days,
                         cashflow['Accrual_Year_Fraction'], position * cashflow['Notional'],
                         position * cashflow.get('Fixed_Amount', 0.0), rate, 0.0, 0.0])

            # needed to deal with forward settlement
            reset_offsets.append([0, 0, 0 if settlement_date is None else -(settlement_date - reference_date).days])

    return TensorCashFlows(cash, reset_offsets)


def make_sampling_data(reference_date, time_grid, samples):
    all_resets = []
    reset_scenario_offsets = []
    D = float(sum([x[-1] for x in samples]))

    for sample in sorted(samples):
        Reset_Day = (sample[0] - reference_date).days
        Start_Day = Reset_Day
        End_Day = Reset_Day
        Weight = sample[-1] / D
        Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
        # only add a reset if its in the past
        all_resets.append(
            [Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
             sample[-2] if sample[0] < reference_date else 0.0, 0.0])
        reset_scenario_offsets.append(Scenario)

    return TensorResets(all_resets, reset_scenario_offsets)


def make_fixing_data(reference_date, time_grid, fixings):
    all_resets = []
    reset_scenario_offsets = []

    for fixing in sorted(fixings):
        Reset_Day = (fixing[0] - reference_date).days
        Start_Day = Reset_Day
        End_Day = Reset_Day
        Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
        # only add a reset if it's in the past
        all_resets.append(
            [Time_Grid, Reset_Day, -1, Start_Day, End_Day, 1.0,
             fixing[-1] if fixing[0] < reference_date else 0.0, 0.0])
        reset_scenario_offsets.append(Scenario)

    return TensorResets(all_resets, reset_scenario_offsets)


def make_simple_fixed_cashflows(reference_date, position, cashflows):
    """
    Generates a vector of fixed cashflows from a data source only looking at the actual fixed value.
    """
    cash = OrderedDict()
    for cashflow in sorted(cashflows['Items'], key=lambda x: x['Payment_Date']):
        if cashflow['Payment_Date'] >= reference_date:
            tenor = (cashflow['Payment_Date'] - reference_date).days
            if tenor in cash:
                cash[tenor][5] += position * cashflow['Fixed_Amount']
            else:
                cash.setdefault(tenor, [tenor, tenor, tenor, 1.0, 0.0,
                                        position * cashflow['Fixed_Amount'], 0.0, 0.0, 0.0])

    # Add the null resets to the end
    dummy_resets = np.zeros((len(cash), 3))

    return TensorCashFlows(list(cash.values()), dummy_resets)


def make_energy_fixed_cashflows(reference_date, position, cashflows):
    """
    Generates a vector of fixed cashflows from a data source only looking at the actual fixed value.
    """
    cash = []
    for cashflow in sorted(cashflows['Items'], key=lambda x: x['Payment_Date']):
        if cashflow['Payment_Date'] >= reference_date:
            cash.append(
                [(cashflow['Payment_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days,
                 (cashflow['Payment_Date'] - reference_date).days,
                 1.0, 0.0, position * cashflow['Volume'] * cashflow['Fixed_Price'], 0.0, 0.0, 0.0])

    # Add the null resets to the end
    dummy_resets = np.zeros((len(cash), 3))

    return TensorCashFlows(cash, dummy_resets)


def make_equity_swaplet_cashflows(base_date, time_grid, position, cashflows, current_spot):
    """
    Generates a vector of equity cashflows from a data source.
    """
    cash = []
    all_resets = []
    cashflow_reset_offsets = []
    reset_scenario_offsets = []

    for cashflow in sorted(cashflows['Items'], key=lambda x: (x['Payment_Date'], x['End_Date'], x['Start_Date'])):
        if cashflow['Payment_Date'] >= base_date:
            cash.append([(cashflow['Start_Date'] - base_date).days, (cashflow['End_Date'] - base_date).days,
                         (cashflow['Payment_Date'] - base_date).days, cashflow.get('Start_Multiplier', 1.0),
                         cashflow.get('End_Multiplier', 1.0), position * cashflow['Amount'],
                         cashflow.get('Dividend_Multiplier', 1.0), 0.0, 0.0])

            r = []
            for reset in ['Start', 'End']:
                Reset_Day = (cashflow[reset + '_Date'] - base_date).days
                Start_Day = Reset_Day
                # we map the weight of the reset with the prior dividends
                Weight = cashflow.get('Known_Dividend_Sum', 0.0)

                # Need to use this reset to estimate future dividends
                Time_Grid, Scenario = time_grid.get_scenario_offset(max(Reset_Day, 0))

                # only add a reset if it's in the past - if its 0, then replace it with the current spot
                if Start_Day <= 0:
                    known_price = cashflow.get('Known_' + reset + '_Price', 0.0)
                    if Start_Day == 0 and not known_price:
                        logging.warning(
                            'Known_{}_Price not set at base_date - setting to current spot'.format(reset))
                        reset_price = current_spot
                    else:
                        reset_price = known_price
                else:
                    reset_price = 0.0

                r.append([Time_Grid, Reset_Day, -1, Start_Day, Start_Day, Weight,
                          reset_price,
                          cashflow.get('Known_' + reset + '_FX_Rate', 0.0) if Start_Day <= 0 else 0.0])
                reset_scenario_offsets.append(Scenario)

            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def make_index_cashflows(base_date, time_grid, position, cashflows, price_index, index_rate,
                         settlement_date, reference_name, isBond=True):
    """
    Generates a vector of index-linked cashflows from a data source given the price_index and index_rate price factors.
    """

    def IndexReference2M(pricing_date, lagged_date, resets, offsets):
        Fixing_Day = (pricing_date - pd.DateOffset(months=2)).to_period('M').to_timestamp('D')
        Rel_Day = (Fixing_Day - lagged_date).days
        Value = index_rate.get_reference_value(Fixing_Day) if Fixing_Day <= lagged_date else 0.0

        Time_Grid, Scenario = time_grid.get_scenario_offset(Rel_Day) if Rel_Day >= 0.0 else (0, -1)
        resets.append([Time_Grid, Rel_Day, -1, Rel_Day, Rel_Day, 1.0, Value, 0.0])
        offsets.append(Scenario)

    def IndexReference3M(pricing_date, lagged_date, resets, offsets):
        Fixing_Day = (pricing_date - pd.DateOffset(months=3)).to_period('M').to_timestamp('D')
        Rel_Day = (Fixing_Day - lagged_date).days
        Value = index_rate.get_reference_value(Fixing_Day) if Fixing_Day <= lagged_date else 0.0

        Time_Grid, Scenario = time_grid.get_scenario_offset(Rel_Day) if Rel_Day >= 0.0 else (0, -1)
        resets.append([Time_Grid, Rel_Day, -1, Rel_Day, Rel_Day, 1.0, Value, 0.0])
        offsets.append(Scenario)

    def IndexReferenceInterpolated3M(pricing_date, lagged_date, resets, offsets):
        T1 = pricing_date.to_period('M').to_timestamp('D')
        Sample_Day_1 = (pricing_date - pd.DateOffset(months=3)).to_period('M').to_timestamp('D')
        Sample_Day_2 = (pricing_date - pd.DateOffset(months=2)).to_period('M').to_timestamp('D')
        w = (pricing_date - T1).days / float(((T1 + pd.DateOffset(months=1)) - T1).days)
        Weights = [(Sample_Day_1, (1.0 - w)), (Sample_Day_2, w)]

        for Day, Weight in Weights:
            Rel_Day = (Day - lagged_date).days
            Value = index_rate.get_reference_value(Day) if Day <= lagged_date else 0.0
            Time_Grid, Scenario = time_grid.get_scenario_offset(Rel_Day) if Rel_Day >= 0.0 else (0, -1)

            resets.append([Time_Grid, Rel_Day, -1, Rel_Day, Rel_Day, Weight, Value, 0.0])
            offsets.append(Scenario)

    def IndexReferenceInterpolated4M(pricing_date, lagged_date, resets, offsets):
        T1 = pricing_date.to_period('M').to_timestamp('D')
        Sample_Day_1 = (pricing_date - pd.DateOffset(months=4)).to_period('M').to_timestamp('D')
        Sample_Day_2 = (pricing_date - pd.DateOffset(months=3)).to_period('M').to_timestamp('D')
        w = (pricing_date - T1).days / float(((T1 + pd.DateOffset(months=1)) - T1).days)
        Weights = [(Sample_Day_1, (1.0 - w)), (Sample_Day_2, w)]

        for Day, Weight in Weights:
            Rel_Day = (Day - lagged_date).days
            Value = index_rate.get_reference_value(Day) if Day <= lagged_date else 0.0
            Time_Grid, Scenario = time_grid.get_scenario_offset(Rel_Day) if Rel_Day >= 0.0 else (0, -1)

            resets.append([Time_Grid, Rel_Day, -1, Rel_Day, Rel_Day, Weight, Value, 0.0])
            offsets.append(Scenario)

    cash = []
    cashflow_reset_offsets = []
    # resets at different points in time
    time_resets = []
    time_scenario_offsets = []
    # resets per cashflow
    base_resets = []
    base_scenario_offsets = []
    final_resets = []
    final_scenario_offsets = []

    for cashflow in sorted(cashflows['Items'], key=lambda x: x['Payment_Date']):
        if cashflow['Payment_Date'] >= base_date and (
                (cashflow['Payment_Date'] >= settlement_date) if settlement_date else True):
            Pay_Date = (cashflow['Payment_Date'] - base_date).days
            Accrual_Start_Date = (cashflow['Accrual_Start_Date'] - base_date).days \
                if cashflow.get('Accrual_Start_Date') else Pay_Date
            Accrual_End_Date = (cashflow['Accrual_End_Date'] - base_date).days \
                if cashflow.get('Accrual_End_Date') else Pay_Date
            base_reference_date = cashflow.get('Base_Reference_Date') \
                if cashflow.get('Base_Reference_Date') else base_date
            final_reference_date = cashflow.get('Final_Reference_Date') \
                if cashflow.get('Final_Reference_Date') else base_date

            cash.append([Accrual_Start_Date, Accrual_End_Date, Pay_Date, cashflow['Accrual_Year_Fraction'],
                         position * cashflow['Notional'], cashflow['Rate_Multiplier'], cashflow['Yield'].amount, 0.0,
                         0.0])

            # attach the base and final reference dates to the cashflow
            cashflow_reset_offsets.append(
                [cashflow['Base_Reference_Value'] if cashflow['Base_Reference_Value'] else -(
                        base_reference_date - base_date).days,
                 cashflow['Final_Reference_Value'] if cashflow['Final_Reference_Value'] else -(
                         final_reference_date - base_date).days,
                 Pay_Date if settlement_date is None else -(settlement_date - base_date).days])

            if isBond:
                locals()[reference_name](
                    base_reference_date, base_date, base_resets, base_scenario_offsets)
                locals()[reference_name](
                    final_reference_date, base_date, final_resets, final_scenario_offsets)

    # set the cashflows
    cashflows = TensorCashFlows(sorted(cash), cashflow_reset_offsets)
    # check if the paydays are still sorted
    if (cashflows.schedule[:, CASHFLOW_INDEX_Pay_Day] != sorted(cashflows.schedule[:, CASHFLOW_INDEX_Pay_Day])).any():
        logging.error("Cashflow Pay Day not in sorted order - check accrual dates")

    if isBond:
        mtm_grid = time_grid.time_grid[:, TIME_GRID_MTM]

        for last_published_date in index_rate.get_last_publication_dates(base_date, mtm_grid):
            # calc the number of days since last published date to the base date
            Rel_Day = (last_published_date - base_date).days
            Value = index_rate.get_reference_value(last_published_date) if last_published_date <= index_rate.param[
                'Last_Period_Start'] else 0.0

            time_resets.append([0.0, Rel_Day, Rel_Day, Rel_Day, -1, 1.0, Value, 0.0])
            time_scenario_offsets.append(0)

        cashflows.set_resets(time_resets, time_scenario_offsets)

        return cashflows, TensorResets(base_resets, base_scenario_offsets), TensorResets(
            final_resets, final_scenario_offsets)

    else:
        for eval_time in time_grid.time_grid[:, TIME_GRID_MTM]:
            actual_time = base_date + pd.DateOffset(days=eval_time)

            locals()[reference_name](
                actual_time, index_rate.param['Last_Period_Start'], time_resets, time_scenario_offsets)

        cashflows.set_resets(time_resets, time_scenario_offsets)

        return cashflows


def make_float_cashflows(reference_date, time_grid, position, cashflows):
    """
    Generates a vector of floating cashflows from a data source.
    """
    cash = []
    all_resets = []
    cashflow_reset_offsets = []
    reset_scenario_offsets = []

    for cashflow in sorted(
            cashflows['Items'], key=lambda x: (x['Payment_Date'], x['Accrual_End_Date'], x['Accrual_Start_Date'])):

        if cashflow['Payment_Date'] >= reference_date:
            # potential FX resets
            fx_reset_date = (cashflow.get('FX_Reset_Date') - reference_date).days \
                if cashflow.get('FX_Reset_Date') else 0.0
            fx_reset_val = cashflow.get('Known_FX_Rate', 0.0)

            cash.append([(cashflow['Accrual_Start_Date'] - reference_date).days,
                         (cashflow['Accrual_End_Date'] - reference_date).days,
                         (cashflow['Payment_Date'] - reference_date).days,
                         cashflow['Accrual_Year_Fraction'], position * cashflow['Notional'],
                         position * cashflow.get('Fixed_Amount', 0.0), cashflow['Margin'].amount,
                         fx_reset_date, fx_reset_val])

            r = []
            for reset in cashflow['Resets']:
                # check if the reset end day is valid
                Actual_End_Day = reset[1] + cashflow['Rate_Tenor'] if reset[2] == reset[1] else reset[2]

                # create the reset vector
                Reset_Day = (reset[0] - reference_date).days
                Start_Day = (reset[1] - reference_date).days
                End_Day = (Actual_End_Day - reference_date).days
                Accrual = reset[3]
                Weight = 1.0 / len(cashflow['Resets'])
                Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
                # only add a reset if it's in the past
                r.append([Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
                          reset[-1].amount if reset[0] < reference_date else 0.0, Accrual])
                reset_scenario_offsets.append(Scenario)

            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def make_energy_cashflows(reference_date, time_grid, position, cashflows, reference, forwardsample, fxsample,
                          calendars):
    """
    Generates a vector of floating/fixed cashflows from a data source
    using the energy model. NOTE - Need to allow for fxSample different from the forwardsample - TODO!
    """
    cash = []
    all_resets = []
    cashflow_reset_offsets = []
    reset_scenario_offsets = []
    forward_calendar_bday = calendars.get(forwardsample.get_holiday_calendar(), {'businessday': 'B'})['businessday']

    for cashflow in sorted(cashflows['Items'], key=lambda x: (x['Payment_Date'], x['Period_End'], x['Period_Start'])):
        if cashflow['Payment_Date'] >= reference_date:
            cash.append(
                [(cashflow['Period_Start'] - reference_date).days, (cashflow['Period_End'] - reference_date).days,
                 (cashflow['Payment_Date'] - reference_date).days, cashflow.get('Price_Multiplier', 1.0),
                 position * cashflow['Volume'], 0.0, cashflow.get('Fixed_Basis', 0.0), 0.0, 0.0])

            r = []
            bunsiness_dates = pd.date_range(
                cashflow['Period_Start'], cashflow['Period_End'], freq=forward_calendar_bday)

            if forwardsample.get_sampling_convention() == 'ForwardPriceSampleDaily':
                # create daily samples
                reset_dates = bunsiness_dates

            elif forwardsample.get_sampling_convention() == 'ForwardPriceSampleBullet':
                # create one sample
                reset_dates = [bunsiness_dates[-1]]

            resets_in_excel_format = [(x - reference.start_date).days for x in reset_dates]
            reference_date_excel = (reference_date - reference.start_date).days

            # retrieve the fixing dates from the reference curve and adding an offset
            fixing_dates = reference.get_fixings().array[
                               np.searchsorted(reference.get_tenor(), resets_in_excel_format) + int(
                                   forwardsample.param.get('Offset'))][:, 1]

            for reset_day, fixing_day in zip(resets_in_excel_format, fixing_dates):
                Reset_Day = reset_day - reference_date_excel
                Start_Day = reset_day - reference_date_excel
                End_Day = fixing_day
                Weight = 1.0 / len(reset_dates)
                Time_Grid, Scenario = time_grid.get_scenario_offset(Start_Day)
                # only add a reset if its in the past
                r.append([Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
                          cashflow['Realized_Average'] or 0.0, cashflow['FX_Realized_Average'] or 0.0])
                reset_scenario_offsets.append(Scenario)

            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def compress_deal_data(deals):
    def filter_deals(deals, values):
        filtered = []
        unfiltered = []
        for deal in deals:
            (filtered if deal['Instrument'].field['Reference'] in values else unfiltered).append(deal)
        return filtered, unfiltered

    def compress_CFFixedInterestListDeal(unders, ref, use_ref_as_tag=False):
        compressed = []
        all_rate = {}
        all_notional = {}
        for deal in unders:
            buy_sell = 1.0 if deal['Instrument'].field['Buy_Sell'] == 'Buy' else -1.0
            prop_key = tuple(sorted(
                [(k, v) for k, v in deal['Instrument'].field['Cashflows'].items() if k != 'Items']))
            rate_list = all_rate.setdefault(prop_key, {})
            notional_list = all_notional.setdefault(prop_key, {})
            for cf in deal['Instrument'].field['Cashflows']['Items']:
                key = tuple(sorted(
                    [(k, v) for k, v in cf.items() if k not in ['Notional', 'Rate']]))
                notional = buy_sell * cf['Notional']
                rate_list[key] = rate_list.setdefault(key, 0.0) + cf['Rate'] * notional
                notional_list[key] = notional_list.setdefault(key, 0.0) + notional

        # finish this off
        for prop_index, (cf_prop, rate_list) in enumerate(all_rate.items()):
            leg = []
            notional_list = all_notional[cf_prop]
            for key, val in rate_list.items():
                notional = notional_list[key]
                cashflow = dict(key)
                if notional:
                    cashflow['Notional'] = notional
                    cashflow['Rate'] = Percent(100.0 * val / notional)
                else:
                    cashflow['Notional'] = val
                    cashflow['Rate'] = Percent(100.0)
                leg.append(cashflow)

            # sort it
            final = sorted(leg, key=lambda x: (x['Payment_Date'], x['Accrual_Start_Date'], x['Accrual_End_Date']))
            # use an exisiting deal to edit the cashflows
            deal = unders[prop_index]
            deal['Instrument'].field['Buy_Sell'] = 'Buy'
            deal['Instrument'].field['Cashflows'] = dict(cf_prop)
            deal['Instrument'].field['Cashflows']['Items'] = final
            if use_ref_as_tag:
                deal['Instrument'].field['Reference'] = 'Compressed_CFFixed_{}_{}'.format(
                    'Buy', deal['Instrument'].field['Currency'])
                deal['Instrument'].field['Tags'] = list(ref)
            else:
                deal['Instrument'].field['Reference'] = 'Compressed_CFFixed_{}_{}'.format('Buy', ref)
            compressed.append(deal)

        return compressed

    def compress_CFFloatingInterestListDeal(unders, ref, use_ref_as_tag=False):
        compressed = []
        all_margin = {}
        all_notional = {}
        for deal in unders:
            buy_sell = 1.0 if deal['Instrument'].field['Buy_Sell'] == 'Buy' else -1.0
            prop_key = tuple(sorted(
                [(k, v) for k, v in deal['Instrument'].field['Cashflows'].items() if k != 'Items']))
            margin_list = all_margin.setdefault(prop_key, {})
            notional_list = all_notional.setdefault(prop_key, {})
            for cf in deal['Instrument'].field['Cashflows']['Items']:
                cf_key = tuple(sorted(
                    [(k, v) for k, v in cf.items() if k not in ['Notional', 'Resets', 'Margin']]))
                reset_key = tuple(sorted([tuple(x) for x in cf['Resets']]))
                key = (cf_key, reset_key)
                notional = buy_sell * cf['Notional']
                margin_list[key] = margin_list.setdefault(key, 0.0) + cf['Margin'] * notional
                notional_list[key] = notional_list.setdefault(key, 0.0) + notional

        # finish this off
        prop_index = 0
        for cf_prop, margin_list in all_margin.items():
            leg = []
            existing_deals = unders[prop_index:]
            notional_list = all_notional[cf_prop]
            for key, val in margin_list.items():
                notional = notional_list[key]
                cashflow = dict(key[0])
                cashflow['Resets'] = [list(x) for x in list(key[1])]
                if notional:
                    cashflow['Notional'] = notional
                    cashflow['Margin'] = Basis(10000.0 * val / notional)
                    leg.append(cashflow)
                elif val:
                    cashflow['Notional'] = val
                    cashflow['Margin'] = Basis(10000.0)
                    leg.append(cashflow)
                    logging.warning('Float Cashflow Nominal compressed to 0.0 and margin is not 0 - TEST')
                else:
                    logging.info('Float Cashflow Nominal compressed to 0.0 and margin is 0 - will be skipped')

            # check that there are no overlapping resets (if so - create a new leg)
            final = sorted(leg, key=lambda x: (x['Payment_Date'], x['Accrual_Start_Date'], x['Accrual_End_Date']))
            # can just check the first reset because we sorted them earlier
            splits = [i + 1 for i, (x, y) in enumerate(
                zip(final[:-1], final[1:])) if x['Resets'][0][0] > y['Resets'][0][0]]

            if len(splits) >= len(existing_deals):
                # can happen with e.g. prime linked swaps (many resets per day)
                # check to see if we must edit the tag
                for deal in existing_deals:
                    if use_ref_as_tag:
                        deal['Instrument'].field['Tags'] = list(ref)
                    # add the deal uncompressed
                    compressed.append(deal)
            else:
                for i, (deal, m, n) in enumerate(zip(existing_deals, [0] + splits, splits + [None])):
                    legnum = '_Leg{}'.format(i) if splits else ''
                    deal['Instrument'].field['Buy_Sell'] = 'Buy'
                    deal['Instrument'].field['Cashflows'] = dict(cf_prop)
                    deal['Instrument'].field['Cashflows']['Items'] = final[m:n]
                    if use_ref_as_tag:
                        deal['Instrument'].field['Reference'] = 'Compressed_CFFloat_{}_{}{}'.format(
                            'Buy', deal['Instrument'].field['Currency'], legnum)
                        deal['Instrument'].field['Tags'] = list(ref)
                    else:
                        deal['Instrument'].field['Reference'] = 'Compressed_CFFloat_{}_{}{}'.format('Buy', ref, legnum)

                    compressed.append(deal)

                # move the existing deal index forward
                prop_index += i + 1

        return compressed

    # return this as our compressed portfolio
    reduced_deals = deals
    # first try and compress equity_swaps
    equity_swaps = [x for x in reduced_deals if x['Instrument'].field['Object'] == 'EquitySwapletListDeal']
    # don't bother if there are less than 400 swaps
    if equity_swaps and len(equity_swaps) > 400:
        logging.info('Compressing {} EquitySwaplets'.format(len(equity_swaps)))
        eq_unders = {}
        ir_unders = {}
        eq_swap_ref = {x['Instrument'].field['Reference']: x['Instrument'].field['Equity'] for x in equity_swaps}
        all_eq_swap, all_other = filter_deals(reduced_deals, eq_swap_ref.keys())

        # first load all compressible deals
        for k in all_eq_swap:
            key = tuple(
                sorted([(field, tuple(value) if isinstance(value, list) else value)
                        for field, value in k['Instrument'].field.items()
                        if field not in ['Reference', 'Buy_Sell', 'Cashflows']]))

            if k['Instrument'].field['Object'] == 'EquitySwapletListDeal':
                # need to split buys and sells because there could be at different prices for the same day
                buy_sell = (('Buy_Sell', k['Instrument'].field['Buy_Sell']),)
                eq_unders.setdefault(key + buy_sell, []).append(k)
            else:
                # pair up with the equity leg so that it's easy to track funding per stock
                under_eq = eq_swap_ref[k['Instrument'].field['Reference']]
                ir_unders.setdefault(key + (under_eq,), []).append(k)

        # now compress
        eq_compressed = {}
        for k, unders in eq_unders.items():
            cf_list = {}
            for deal in unders:
                for cf in deal['Instrument'].field['Cashflows']['Items']:
                    key = tuple([(k, v) for k, v in cf.items() if k != 'Amount'])
                    cf_list[key] = cf_list.setdefault(key, 0.0) + cf['Amount']

            # edit the last deal
            deal['Instrument'].field['Cashflows']['Items'] = [dict(k + (('Amount', v),)) for k, v in cf_list.items()]
            deal['Instrument'].field['Reference'] = 'Compressed_EQSwaplet_{}_{}'.format(
                deal['Instrument'].field['Buy_Sell'], deal['Instrument'].field['Equity'])
            eq_compressed.setdefault(deal['Instrument'].field['Equity'], []).append(deal)

        ir_compressed = {}
        for k, unders in ir_unders.items():
            ir_compressed.setdefault(k[-1], []).extend(compress_CFFloatingInterestListDeal(unders, k[-1]))

        for k, v in eq_compressed.items():
            all_other.extend(v)
            all_other.extend(ir_compressed[k])

        reduced_deals = all_other

    # now try and compress ir_swaps - not ideal looking for ',Swap,' in tags - TODO!
    ir_swaps = [x for x in reduced_deals if x['Instrument'].field['Object'] == 'StructuredDeal'
                and ',Swap,' in x['Instrument'].field['Tags'][0]]

    # switched off - need to test and improve this
    if False and ir_swaps and len(ir_swaps) > 200:
        logging.info('Compressing {} IR Swaps'.format(len(ir_swaps)))
        float_unders = {}
        fixed_unders = {}
        swap_refs = [x['Instrument'].field['Reference'] for x in ir_swaps]
        all_ir_swap, all_other = filter_deals(reduced_deals, swap_refs)

        # first load all compressible deals
        for structure in all_ir_swap:
            tags = tuple(structure['Instrument'].field['Tags'])
            for k in structure['Children']:
                key = tuple(
                    sorted([(field, value) for field, value in k['Instrument'].field.items()
                            if field not in ['Reference', 'Tags', 'Buy_Sell', 'Cashflows']])) + (tags,)

                if k['Instrument'].field['Object'] == 'CFFloatingInterestListDeal':
                    float_unders.setdefault(key, []).append(k)
                else:
                    fixed_unders.setdefault(key, []).append(k)

        fixed_compressed = []
        for k, unders in fixed_unders.items():
            fixed_compressed.extend(compress_CFFixedInterestListDeal(unders, k[-1], use_ref_as_tag=True))

        float_compressed = []
        for k, unders in float_unders.items():
            float_compressed.extend(compress_CFFloatingInterestListDeal(unders, k[-1], use_ref_as_tag=True))

        # add it and continue
        all_other.extend(fixed_compressed)
        all_other.extend(float_compressed)

        reduced_deals = all_other

    return reduced_deals


def compress_no_compounding(cashflows, groupsize, check_resets=True):
    '''

    :param cashflows: cashflows to compress
    :param groupsize: -1 to keep all resets (and just regroup them), otherwise, sample this many groups per cashflow
    :param check_resets: make sure all resets are in the future
    :return: the compressed cashflows if we can approximate many resets by fewer groups otherwise, return the
            original cashflows

    Needs more Testing - !TODO!
    '''
    cash_pmts, cash_index, cash_counts = np.unique(
        cashflows.schedule[:, CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

    if (cashflows.offsets[:, 0] == 1).all():
        if (cash_counts > abs(groupsize)).any():
            # can compress
            cash, cashflow_reset_offsets = [], []
            all_resets, reset_scenario_offsets = [], []
            for pay_day, index, num_cf in zip(*[cash_pmts, cash_index, cash_counts]):
                cashflow_schedule = cashflows.schedule[index:index + num_cf]
                cashflow_offsets = cashflows.offsets[index:index + num_cf]
                reset_offset = cashflows.offsets[index:index + num_cf, 1]
                nominals = np.unique(cashflow_schedule[:, CASHFLOW_INDEX_Nominal])
                margins = np.unique(cashflow_schedule[:, CASHFLOW_INDEX_FloatMargin])

                if groupsize == -1 and nominals.size == 1 and margins.size == 1:
                    # we can compress this
                    cash.append(
                        [cashflow_schedule[0, CASHFLOW_INDEX_Start_Day],
                         cashflow_schedule[-1, CASHFLOW_INDEX_End_Day],
                         pay_day,
                         cashflow_schedule[:, CASHFLOW_INDEX_Year_Frac].sum(),
                         cashflow_schedule[:, CASHFLOW_INDEX_Nominal].mean(),
                         cashflow_schedule[:, CASHFLOW_INDEX_FixedAmt].sum(),
                         cashflow_schedule[:, CASHFLOW_INDEX_FloatMargin].mean(),
                         cashflow_schedule[0, CASHFLOW_INDEX_FXResetDate],
                         cashflow_schedule[0, CASHFLOW_INDEX_FXResetValue]])

                    cashflow_reset_offsets.append([num_cf, index, 1])
                    all_resets.extend(cashflows.Resets[reset_offset].tolist())
                    reset_scenario_offsets.extend(cashflows.Resets.offsets[reset_offset].tolist())

                elif nominals.size <= groupsize and margins.size <= groupsize and (check_resets and not (
                        cashflows.Resets[reset_offset, RESET_INDEX_Reset_Day] < 0).any() or not check_resets):
                    # we can compress this
                    for cash_group, ofs_group in zip(*map(
                            lambda x: np.array_split(x, groupsize), [cashflow_schedule, cashflow_offsets])):
                        cash.append(
                            [cash_group[0, CASHFLOW_INDEX_Start_Day],
                             cash_group[-1, CASHFLOW_INDEX_End_Day],
                             pay_day,
                             cash_group[:, CASHFLOW_INDEX_Year_Frac].sum(),
                             # not strictly correct - need to break this up - TODO
                             cash_group[:, CASHFLOW_INDEX_Nominal].mean(),
                             cash_group[:, CASHFLOW_INDEX_FixedAmt].sum(),
                             # not strictly correct - need to break this up - TODO
                             cash_group[:, CASHFLOW_INDEX_FloatMargin].mean(),
                             cash_group[0, CASHFLOW_INDEX_FXResetDate],
                             cash_group[0, CASHFLOW_INDEX_FXResetValue]])

                        reset_index = ofs_group[ofs_group[:, 1].size // 2, 1]
                        cashflow_reset_offsets.append([1, len(all_resets), 0])
                        reset_scenario_offsets.append(cashflows.Resets.offsets[reset_index])
                        all_resets.append(cashflows.Resets[reset_index].tolist())

                else:
                    # copy as is
                    cash.extend(cashflow_schedule.tolist())
                    all_resets.extend(cashflows.Resets[reset_offset].tolist())
                    reset_scenario_offsets.extend(cashflows.Resets.offsets[reset_offset].tolist())
                    cashflow_reset_offsets.extend(cashflows.offsets[index:index + num_cf].tolist())

            approx_cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
            approx_cashflows.set_resets(all_resets, reset_scenario_offsets)
            if cashflows.Resets.count() == approx_cashflows.Resets.count():
                logging.warning('Cashflows rebased from {} resets'.format(cashflows.Resets.count()))
            else:
                logging.warning('Cashflows reduced from {} resets to {} resets'.format(
                    cashflows.Resets.count(), approx_cashflows.Resets.count()))
            return approx_cashflows

    return cashflows


if __name__ == '__main__':
    pass
