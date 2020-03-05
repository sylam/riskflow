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

import calendar
from functools import reduce
from collections import namedtuple, OrderedDict

import logging
import scipy.stats
import pandas as pd
import numpy as np
import tensorflow as tf

# For dealing with excel dates and dataframes
excel_offset = pd.Timestamp('1899-12-30 00:00:00')

# set the default precision for tf
Default_Precision = np.float32


def array_type(x): return np.array(x)


# Days in year - could set this to 365.25 if you want that bit extra time
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
FACTOR_INDEX_Offset = 1  # index into the corresponding tensor list
FACTOR_INDEX_Tenor_Index = 2  # Actual tenor tensor and its delta
FACTOR_INDEX_Daycount = 3  # daycount code
FACTOR_INDEX_Process = 4  # Stochastic process code
FACTOR_INDEX_ExcelCalcDate = 3
FACTOR_INDEX_Moneyness_Index = 2
FACTOR_INDEX_Expiry_Index = 3
FACTOR_INDEX_VolTenor_Index = 4
FACTOR_INDEX_Flat_Index = 4
FACTOR_INDEX_Surface_Flat_Moneyness_Index = 5
FACTOR_INDEX_Surface_Flat_Expiry_Index = 6

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
CASHFLOW_OFFSET_BaseReference = 0
CASHFLOW_OFFSET_FinalReference = 1
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
# CASHFLOW_METHOD_Average_Rate	 				= 1

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

CASHFLOW_CompoundingMethodLookup = {'None': CASHFLOW_METHOD_Compounding_None,
                                    'Flat': CASHFLOW_METHOD_Compounding_Flat,
                                    'Include_Margin': CASHFLOW_METHOD_Compounding_Include_Margin}

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
TwoDimensionalFactors = ['FXVol', 'EquityPriceVol']
ThreeDimensionalFactors = ['InterestRateVol', 'InterestYieldVol', 'ForwardPriceVol']
ImpliedFactors = ['HullWhite2FactorModelParameters', 'GBMTSImpliedParameters', 'PCAMixedFactorModelParameters']

# weekends and weekdays
WeekendMap = {'Friday and Saturday': 'Sun Mon Tue Wed Thu',
              'Saturday and Sunday': 'Mon Tue Wed Thu Fri',
              'Sunday': 'Mon Tue Wed Thu Fri Sat',
              'Saturday': 'Sun Mon Tue Wed Thu Fri',
              'Friday': 'Sat Sun Mon Tue Wed Thu'}


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


class Basis:
    def __init__(self, amount):
        self.amount = amount / 10000.0
        self.points = amount

    def __str__(self):
        return '%d bp' % self.points

    def __float__(self):
        return self.amount


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

    def last(self):
        return self.data.values()[-1] if self.data.values() else 0.0

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


# Tensor specific classes that's used internally
class TensorSchedule(object):
    def __init__(self, schedule, offsets):
        self.schedule = np.array(schedule)
        self.offsets = np.array(offsets)
        self.cache = None
        self.dtype = None

    def known_resets(self, num_scenarios, index=RESET_INDEX_Value,
                     filter_index=RESET_INDEX_Reset_Day, include_today=False):
        filter_fn = (lambda x: x <= 0.0) if include_today else (lambda x: x < 0.0)
        try:
            return [tf.fill([1, num_scenarios], x[index].astype(Default_Precision))
                    for x in self.schedule if filter_fn(x[filter_index])]
        except:
            print('asd')

    def __getitem__(self, x):
        return self.schedule[x]

    def count(self):
        return self.schedule.shape[0]

    def merged(self):
        if self.cache is None:
            self.cache = np.concatenate((self.schedule, self.offsets), axis=1)
        return self.cache


class DealTimeDependencies(object):
    def __init__(self, mtm_time_grid, deal_time_grid):
        self.mtm_time_grid = mtm_time_grid
        self.delta = np.hstack(((mtm_time_grid[deal_time_grid[1:]] -
                                 mtm_time_grid[deal_time_grid[:-1]]), [1]))
        self.interp = mtm_time_grid[mtm_time_grid <= mtm_time_grid[deal_time_grid[-1]]]
        self.deal_time_grid = deal_time_grid

    def assign(self, time_dependencies):
        # only assign up to the max of this set of dependencies
        expiry = self.deal_time_grid[-1]
        query = time_dependencies.deal_time_grid <= expiry
        self.delta = time_dependencies.delta[query]
        self.deal_time_grid = time_dependencies.deal_time_grid[query]
        self.interp = self.mtm_time_grid[self.mtm_time_grid <= self.mtm_time_grid[expiry]]

    def fetch_index_by_day(self, days):
        return self.interp.searchsorted(days)


class TensorResets(TensorSchedule):
    def __init__(self, schedule, offsets):
        super(TensorResets, self).__init__(schedule, offsets)

        # Assign the offsets directly to the resets
        self.schedule[:, RESET_INDEX_Scenario] = self.offsets

    def split_resets(self, reset_offset, t):
        all_resets = self.schedule[reset_offset:]
        future_resets = all_resets[all_resets[:, RESET_INDEX_Reset_Day] >= t]
        past_resets = all_resets[all_resets[:, RESET_INDEX_Reset_Day] < t] if reset_offset else np.array([])
        return past_resets, future_resets

    def split_block_resets(self, reset_offset, t, date_offset=0):
        all_resets = self.schedule[reset_offset:]
        future_resets = np.searchsorted(all_resets[:, RESET_INDEX_Reset_Day] - date_offset, t)
        return future_resets

    def get_start_index(self, time_grid, offset=0):
        """Read the start index (relative to the time_grid) of each reset"""
        return np.searchsorted(self.schedule[:, RESET_INDEX_Reset_Day] - offset,
                               time_grid[:, TIME_GRID_MTM]).astype(np.int64)


class TensorCashFlows(TensorSchedule):
    def __init__(self, schedule, offsets):
        # check which cashflows are settlements (as opposed to accumulations)
        for cashflow, next_cashflow, cash_ofs in zip(schedule[:-1], schedule[1:], offsets[:-1]):
            if (next_cashflow[CASHFLOW_INDEX_Pay_Day] != cashflow[CASHFLOW_INDEX_Pay_Day]) or (
                    cashflow[CASHFLOW_INDEX_FixedAmt] != 0):
                cash_ofs[2] = 1

        # last cashflow always settles (if it's not marked as such) otherwise, it's a forward
        if offsets[-1][2] == 0:
            offsets[-1][2] = 1

        # call superclass
        super(TensorCashFlows, self).__init__(schedule, offsets)
        self.Resets = None

        # store the days needed to access curves
        self.payment_days = np.unique(self.schedule[:, CASHFLOW_INDEX_Pay_Day])

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

    def add_mtm_payments(self, base_date, principal_exchange, effective_date, day_count):
        ''' MTM CCIRS's only need a zero marker for the nominal should the effective date be in the future '''
        if (principal_exchange in ['Start_Maturity', 'Start']) and base_date <= effective_date:
            self.insert_cashflow(
                make_cashflow(base_date, effective_date, effective_date, effective_date, 0.0, get_day_count(day_count),
                              0.0, 0.0))

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
    return tf.split(tensor, counts) if tensor.shape[0] == counts.sum() else [tensor] * counts.size


def non_zero(tensor):
    if tensor.get_shape()[0] > 1:
        flat = tf.squeeze(tensor)
        non_zero = tf.where(tf.cast(flat, tf.bool))
        return tf.squeeze(tf.gather(flat, non_zero), axis=1)
    else:
        return tensor


def interpolate_curve_indices(all_tenor_points, curve_component, time_factor=1.0):
    # will only work if all_tenor_points is 2 dimensional
    tenor, delta, curvetype = curve_component[FACTOR_INDEX_Tenor_Index][:3]
    daycount = curve_component[FACTOR_INDEX_Daycount]
    max_tenor_index = max(tenor.size - 1, 0)

    a, i1, w1, i2, w2 = [], [], [], [], []
    for time_index, tenor_points in enumerate(all_tenor_points):
        tenor_points_in_years = daycount(tenor_points)
        index = np.clip(
            np.searchsorted(tenor, tenor_points_in_years, side='right') - 1,
            0, max_tenor_index)
        index_next = np.clip(index + 1, 0, max_tenor_index)
        if curvetype == 'Dividend':
            min_tenor = max(1e-5, tenor.min())
            max_tenor = max(1e-5, tenor.max())
            alpha = (1.0 / tenor[index].clip(1e-5, np.inf) -
                     1.0 / tenor_points_in_years.clip(min_tenor, max_tenor)) / delta[index]
        else:
            alpha = (tenor_points_in_years.clip(tenor.min(), tenor.max()) -
                     tenor[index]) / delta[index]

        time_modifier = time_factor * tenor_points_in_years if time_factor else 1.0

        t = np.ones_like(index) * time_index
        i1.append(np.dstack((t, index))[-1])
        i2.append(np.dstack((t, index_next))[-1])
        a.append(alpha)
        w1.append((1.0 - alpha) * time_modifier)
        w2.append(alpha * time_modifier)

    weight1, weight2, alpha = np.array(w1), np.array(w2), np.expand_dims(np.array(a), axis=2)

    return [alpha, np.array(i1), np.expand_dims(weight1, axis=2), np.array(i2), np.expand_dims(weight2, axis=2)]


def interpolate_curve(curve_tensor, curve_component, curve_interp, points, time_factor):
    a, i1, w1, i2, w2 = interpolate_curve_indices(
        points, curve_component, time_factor)

    if not curve_component[FACTOR_INDEX_Stoch]:
        # flatten the indices
        i1[:, :, 0] *= 0
        i2[:, :, 0] *= 0

    if curve_component[FACTOR_INDEX_Tenor_Index][2].startswith('Hermite'):
        g, c = curve_interp
        tenors = np.expand_dims(
            curve_component[FACTOR_INDEX_Daycount](points), axis=2)
        if curve_component[FACTOR_INDEX_Tenor_Index][2] == 'HermiteRT':
            mult = None if time_factor else 1.0 / tenors
        else:
            mult = tenors if time_factor else None

        val = tf.gather_nd(curve_tensor, i1) * (1.0 - a) + (
            (tf.gather_nd(curve_tensor, i2) * a + a * (1.0 - a) * tf.gather_nd(g, i1) +
             a * a * (1.0 - a) * tf.gather_nd(c, i1)) if a.any() else 0.0)

        interp_val = val * mult if mult is not None else val
    else:
        # default to linear
        interp_val = tf.gather_nd(curve_tensor, i1) * w1 + (
            tf.gather_nd(curve_tensor, i2) * w2 if w2.any() else 0.0)

    return interp_val


def interpolate_risk_neutral(points, curve_tensor, curve_component, curve_interp, time_grid, time_multiplier):
    t = time_grid[:, 1].reshape(-1, 1)
    T = points + t
    return interpolate_curve(
        curve_tensor, curve_component, curve_interp, t, time_multiplier) - interpolate_curve(
        curve_tensor, curve_component, curve_interp, T, time_multiplier)


class TensorBlock(object):
    def __init__(self, code, tensor, interp, time_grid):
        self.code = code
        self.time_grid = time_grid
        self.interp = interp
        self.tensor = tensor
        self.local_cache = {}

    def split_counts(self, counts, shared):

        key_code = ('tensorblock', tuple([x[:2] for x in self.code]),
                    tuple(self.time_grid[:, TIME_GRID_MTM]),
                    tuple(counts))

        if key_code not in shared.t_Buffer:
            if isinstance(self.tensor, tf.Tensor):
                rate_tensor = split_tensor(self.tensor, counts)
                interp_tensor = None
            else:
                rate_tensor = zip(*[split_tensor(sub_tensor, counts) for sub_tensor in self.tensor])
                interp_tensor = []
                for sub_tensor in self.interp:
                    if sub_tensor is not None:
                        elements = zip(*[split_tensor(sub_interp, counts) for sub_interp in sub_tensor])
                        interp_tensor.append(elements)
                    else:
                        interp_tensor.append([None] * counts.size)

            time_block = np.split(self.time_grid, counts.cumsum())
            shared.t_Buffer[key_code] = [TensorBlock(self.code, tensor, interp, time_t)
                                         for tensor, interp, time_t in zip(
                    rate_tensor, zip(*interp_tensor), time_block)]

        return shared.t_Buffer[key_code]

    def gather_weighted_curve(self, shared, end_points,
                              start_points=None, multiply_by_time=True):

        def calc_curve(time_multiplier, points):
            temp_curve = None
            for curve_tensor, curve_interp, curve_component in zip(self.tensor, self.interp, self.code):
                # handle static curves
                if not curve_component[FACTOR_INDEX_Stoch] and shared.riskneutral:
                    scaled_val = interpolate_risk_neutral(end_points, curve_tensor, curve_component,
                                                          curve_interp, self.time_grid, time_multiplier)
                else:
                    scaled_val = interpolate_curve(curve_tensor, curve_component, curve_interp, points, time_multiplier)

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

    def reduce_deflate(self, time_points, shared):
        temp_curve = 0
        for curve_tensor in self.tensor:
            temp_curve += curve_tensor
        DtT = tf.exp(-tf.cumsum(temp_curve, axis=0))
        # we need the index just prior - note this needs to be checked in the calling code
        indices = self.time_grid[:, TIME_GRID_MTM].searchsorted(time_points) - 1
        return {t: tf.gather(DtT, index) for t, index in zip(time_points, indices)}


# dataframe manipulation

def filter_data_frame(df, from_date, to_date, rate=None):
    index1 = (pd.Timestamp(from_date) - excel_offset).days
    index2 = (pd.Timestamp(to_date) - excel_offset).days
    return df.ix[index1:index2] if rate is None else df.ix[index1:index2][
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


def norm_cdf(x):
    return 0.5 * (tf.erfc(x * -0.7071067811865475))


def black_european_option_price(F, X, r, vol, tenor, buyOrSell, callOrPut):
    stddev = vol * np.sqrt(tenor)
    sign = 1.0 if (F > 0.0 and X > 0.0) else -1.0
    d1 = (np.log(F / X) + 0.5 * stddev * stddev) / stddev
    d2 = d1 - stddev
    return buyOrSell * callOrPut * (F * scipy.stats.norm.cdf(callOrPut * sign * d1) -
                                    X * scipy.stats.norm.cdf(callOrPut * sign * d2)) * np.exp(-r * tenor)


def black_european_option(F, X, vol, tenor, buyorsell, callorput, shared):
    # calculates the black function WITHOUT discounting
    stack = False
    if isinstance(tenor, float):
        guard = tf.logical_and(vol > 0.0, X > 0)
        stddev = tf.where(vol > 0, vol * np.sqrt(tenor), tf.ones_like(vol))
        strike = tf.where(X > 0, X, tf.ones_like(X))
    else:
        guard = tenor > 0.0
        tau = np.sqrt(tenor.clip(0.0, np.inf))
        ones = tf.ones_like(vol)

        if len(guard.shape) > 1:
            stack = True
            stddev = tf.stack(
                [tf.where(g, s * t.reshape(-1, 1), o) for g, t, s, o in
                 zip(guard, tau, tf.unstack(vol), tf.unstack(ones))])
        else:
            sigma = vol * tau.reshape(-1, 1)
            stddev = tf.where(guard, sigma, tf.ones_like(sigma))

        strike = X

    # make sure the forward is always >1e-5
    forward = tf.maximum(F, 1e-5)

    if isinstance(strike, float) and strike == 0:
        prem = forward
        value = forward
    else:
        d1 = tf.log(forward / strike) / stddev + 0.5 * stddev
        d2 = d1 - stddev
        prem = callorput * (forward * norm_cdf(callorput * d1) - X * norm_cdf(callorput * d2))
        value = tf.nn.relu(callorput * (forward - X))

    if stack:
        return tf.stack(
            [tf.where(g, p, v) for g, p, v in zip(guard, tf.unstack(prem), tf.unstack(value))])
    else:
        return buyorsell * tf.where(guard, prem, value)


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
    points = np.array(tenor_points)
    # linear interpolation by default
    return (points, np.append(np.diff(points), 1.0), interp)


def update_tenors(base_date, all_factors):
    def daycount_fn(base_date, daycount):
        def calc_daycount(time_in_days):
            return get_day_count_accrual(base_date, time_in_days, daycount)

        return calc_daycount

    all_tenors = {}
    for factor, factor_obj in all_factors.items():
        risk_factor = factor_obj.factor if hasattr(factor_obj, 'factor') else factor_obj

        if factor.type in OneDimensionalFactors:
            tenor_points = risk_factor.get_tenor()
            # linear interpolation by default
            tenor_data = tenor_diff(tenor_points)

            if factor.type == 'DividendRate':
                # change the tenor delta to use dividend interpolation
                tenor_delta = (1.0 / np.array(tenor_points[:-1]).clip(1e-5, np.inf)) - \
                              (1.0 / np.array(tenor_points[1:]).clip(1e-5, np.inf))
                tenor_data = (np.array(tenor_points), np.hstack((tenor_delta, [1.0])), 'Dividend')
            elif factor.type == 'InterestRate' and risk_factor.interpolation[0] != 'Linear':
                tenor_data = tenor_data[:2] + risk_factor.interpolation

            daycount = risk_factor.get_day_count()
            all_tenors[factor] = [tenor_data, daycount_fn(base_date, daycount), factor_obj]

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
                moneyness_map, expiry_index_map = [], []
                for delivery_expiry_points in risk_factor.index_map.values():
                    expiry_map, exp_index = [], []
                    for exp_day, expiry_points in delivery_expiry_points.items():
                        expiry_map.append(tenor_diff(expiry_points))
                        exp_index.append(exp_day)
                    expiry_index_map.append(tenor_diff(exp_index))
                    moneyness_map.append(expiry_map)
                # store the moneyness and expiry first
                all_tenors[factor] = [tenor_diff(risk_factor.get_moneyness()),
                                      tenor_diff(risk_factor.get_expiry()),
                                      tenor_diff(risk_factor.get_tenor()), moneyness_map, expiry_index_map]
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
    alpha = ((t - tenor[index]) / dvt[tenor_index]).clip(0, 1)
    return (1 - alpha) * tf.gather(rate_tensor, index) + alpha * tf.gather(rate_tensor, index_next)


# indexing ops manipulating large tensors
def gather_interp_matrix(mtm, deal_time_dep, shared):
    index = np.searchsorted(deal_time_dep.deal_time_grid,
                            np.arange(deal_time_dep.interp.size), side='right') - 1
    alpha = np.array((deal_time_dep.interp -
                      deal_time_dep.interp[deal_time_dep.deal_time_grid[index]])
                     / deal_time_dep.delta[index]).reshape(-1, 1)
    index_next = (index + 1).clip(0, deal_time_dep.deal_time_grid.size - 1)
    return (tf.gather(mtm, index) * (1 - alpha) + alpha * tf.gather(mtm, index_next)) \
        if alpha.any() else tf.gather(mtm, index)


def gather_scenario_interp(tensor, time_grid, shared):
    # cache the time interpolation matrix
    index = time_grid[:, TIME_GRID_ScenarioPriorIndex].astype(np.int64)
    alpha_shape = tuple([-1] + [1] * (len(tensor.shape) - 1))
    alpha = time_grid[:, TIME_GRID_PriorScenarioDelta].reshape(alpha_shape)
    index_next = (index + 1).clip(0, tensor.shape[0].value - 1)

    return (tf.gather(tensor, index) * (1 - alpha) + tf.gather(tensor, index_next) * alpha) \
        if alpha.any() else tf.gather(tensor, index)


def split_counts(rates, counts, shared):
    splits = []
    for rate in rates:
        if isinstance(rate, tf.Tensor):
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

        return shared.t_Buffer[key_code]
    else:
        return tf.constant([[1.0]], dtype=shared.precision)


def calc_discount_rate(block, tenors_in_days, shared, multiply_by_time=True):
    key_code = ('discount', tuple([x[:2] for x in block.code]),
                tuple(block.time_grid[:, TIME_GRID_MTM]),
                tenors_in_days.shape, tuple(tenors_in_days.ravel()))

    if key_code not in shared.t_Buffer:
        discount_rates = tf.exp(-block.gather_weighted_curve(shared, tenors_in_days,
                                                             multiply_by_time=multiply_by_time))
        shared.t_Buffer[key_code] = discount_rates

    return shared.t_Buffer[key_code]


def calc_spot_forward(curve, T, time_grid, shared, only_diag):
    '''
    Function for calculating the forward price of FX or EQ rates taking
    into account risk neutrality for static curves
    '''
    curve_grid = calc_time_grid_curve_rate(curve, time_grid, shared)
    T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
    weights = np.diag(T_t).reshape(-1, 1) if only_diag else T_t
    return curve_grid.gather_weighted_curve(shared, weights)


def calc_dividend_samples(t, T, time_grid):
    divi_scenario_offsets = []
    samples = np.linspace(t, T, max(10, (T - t) / 30.0))

    d = []
    for reset_start, reset_end in zip(samples[:-1], samples[1:]):
        Time_Grid, Scenario = time_grid.get_scenario_offset(reset_start)
        d.append([Time_Grid, reset_start, -1, reset_start, reset_end, 0.0, 0.0, 0.0])
        divi_scenario_offsets.append(Scenario)

    return TensorResets(d, divi_scenario_offsets)


def calc_realized_dividends(equity, repo, div_yield, div_resets, shared, offsets=None):
    S = calc_time_grid_spot_rate(
        equity, div_resets[:, :RESET_INDEX_Scenario + 1], shared)

    sr = tf.squeeze(
        calc_spot_forward(
            repo, div_resets[:, RESET_INDEX_End_Day], div_resets, shared, True),
        axis=1)
    sq = tf.exp(-tf.squeeze(
        calc_spot_forward(
            div_yield, div_resets[:, RESET_INDEX_End_Day], div_resets, shared, True),
        axis=1))

    if offsets is not None:
        a = []
        for s, r, q in split_counts([S, sr, sq], offsets, shared):
            a.append(tf.reduce_sum(
                s * tf.exp(tf.cumsum(r, reverse=True)) * (1 - tf.reshape(q, [-1, 1])), axis=0))
        return tf.stack(a)
    else:
        return tf.reduce_sum(
            S * tf.exp(tf.cumsum(sr, reverse=True)) * (1 - tf.reshape(sq, [-1, 1])), axis=0)


def calc_eq_forward(equity, repo, div_yield, T, time_grid, shared, only_diag=False):
    T_scalar = isinstance(T, int)
    key_code = ('eqforward', equity[0], div_yield[0][:2], only_diag,
                T if T_scalar else tuple(T),
                tuple(time_grid[:, TIME_GRID_MTM]))

    if key_code not in shared.t_Buffer:
        T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
        spot = calc_time_grid_spot_rate(equity, time_grid, shared)

        if T_t.any():
            drift = tf.exp(
                calc_spot_forward(repo, T, time_grid, shared, only_diag) -
                calc_spot_forward(div_yield, T, time_grid, shared, only_diag))
        else:
            drift = tf.ones([time_grid.shape[0], 1 if only_diag else T_t.size, 1],
                            dtype=shared.precision)

        shared.t_Buffer[key_code] = spot * tf.squeeze(drift, axis=1) \
            if T_scalar else tf.expand_dims(spot, axis=1) * drift

    return shared.t_Buffer[key_code]


def calc_fx_forward(local, other, T, time_grid, shared, only_diag=False):
    T_scalar = isinstance(T, int)
    key_code = ('fxforward', local[0][0], other[0][0], only_diag,
                T if T_scalar else tuple(T),
                tuple(time_grid[:, TIME_GRID_MTM]))

    if local[0] != other[0]:
        if key_code not in shared.t_Buffer:
            T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
            fx_spot = calc_fx_cross(local[0], other[0], time_grid, shared)

            if T_t.any():
                weights = np.diag(T_t).reshape(-1, 1) if only_diag else T_t
                repo_local = calc_time_grid_curve_rate(local[1], time_grid, shared)
                repo_other = calc_time_grid_curve_rate(other[1], time_grid, shared)
                drift = tf.exp(repo_other.gather_weighted_curve(shared, weights) -
                               repo_local.gather_weighted_curve(shared, weights))
            else:
                drift = tf.ones([time_grid.shape[0], 1 if only_diag else T_t.size, 1],
                                dtype=shared.precision)

            shared.t_Buffer[key_code] = fx_spot * tf.squeeze(drift, axis=1) \
                if T_scalar else tf.expand_dims(fx_spot, axis=1) * drift

        return shared.t_Buffer[key_code]
    else:
        return tf.constant([[1.0]], dtype=shared.precision)


def gather_flat_surface(flat_surface, code, expiry, shared, calc_std):
    # cache the time surface interpolation matrix
    time_code = ('surface_flat', code[:2], tuple(expiry), calc_std)

    if time_code not in shared.t_Buffer:
        expiry_tenor = code[FACTOR_INDEX_Expiry_Index]

        moneyness_max_index = np.array([x[0].size for x in code[FACTOR_INDEX_Flat_Index]])
        exp_index = np.cumsum(np.append(0, moneyness_max_index[:-1]))
        index = np.clip(np.searchsorted(expiry_tenor[0], expiry, side='right') - 1, 0, expiry_tenor[0].size - 1)
        time_modifier = np.sqrt(expiry).reshape(-1, 1) if calc_std else 1.0
        index_next = np.clip(index + 1, 0, expiry_tenor[0].size - 1)
        subset = np.union1d(index, index_next)
        alpha = np.clip((expiry - expiry_tenor[0][index]) /
                        expiry_tenor[1][index], 0, 1.0).reshape(-1, 1, 1)

        block_indices, block_alphas = [], []
        new_moneyness_tenor = reduce(np.union1d, [code[FACTOR_INDEX_Flat_Index][x][0] for x in subset])

        for tenor_index in subset:
            max_index = moneyness_max_index[tenor_index] - 1
            moneyness_tenor = code[FACTOR_INDEX_Flat_Index][tenor_index]
            moneyness_index = np.clip(moneyness_tenor[0].searchsorted(
                new_moneyness_tenor, side='right') - 1, 0, max_index)
            moneyness_index_next = np.clip(moneyness_index + 1, 0, max_index)
            moneyness_alpha = np.clip((new_moneyness_tenor - moneyness_tenor[0][moneyness_index]) /
                                      moneyness_tenor[1][moneyness_index], 0, 1.0)
            block_indices.append(exp_index[tenor_index] + np.stack([moneyness_index, moneyness_index_next]))
            block_alphas.append(np.stack([1 - moneyness_alpha, moneyness_alpha]))

        # need to interpolate back to the tenor level
        money_indices, money_alpha = np.array(block_indices), np.array(block_alphas)
        subset_index = subset.searchsorted(index)
        tenor_money_indices = money_indices[subset_index]
        tenor_money_alpha = money_alpha[subset_index]
        subset_index_next = subset.searchsorted(index_next)
        tenor_money_alpha_next = money_alpha[subset_index_next]
        tenor_money_indices_next = money_indices[subset_index_next]

        surface = time_modifier * tf.reduce_sum(tf.gather(
            flat_surface, tenor_money_indices) * tenor_money_alpha * (1.0 - alpha) + tf.gather(
            flat_surface, tenor_money_indices_next) * tenor_money_alpha_next * alpha, axis=1)

        shared.t_Buffer[time_code] = (tf.reshape(surface, [-1]), tenor_diff(new_moneyness_tenor))

    return shared.t_Buffer[time_code]


def gather_surface_interp(surface, code, expiry, shared, calc_std):
    # cache the time surface interpolation matrix
    time_code = ('surface_interp', code[:2], tuple(expiry), calc_std)

    if time_code not in shared.t_Buffer:
        expiry_tenor = code[FACTOR_INDEX_Expiry_Index]
        index = np.clip(np.searchsorted(expiry_tenor[0], expiry, side='right') - 1, 0, expiry_tenor[0].size - 1)
        time_modifier = np.sqrt(expiry) if calc_std else 1.0
        index_next = np.clip(index + 1, 0, expiry_tenor[0].size - 1)
        alpha = np.clip((expiry - expiry_tenor[0][index]) /
                        expiry_tenor[1][index], 0, 1.0).reshape(-1, 1)

        shared.t_Buffer[time_code] = tf.gather(surface, index) * (1 - alpha) * time_modifier + (
            tf.gather(surface, index_next) * alpha * time_modifier if alpha.any() else 0.0)

    return shared.t_Buffer[time_code]


def calc_moneyness_vol_rate(moneyness, expiry, key_code, shared):
    # work out the moneyness - this is a way to fake np.searchsorted - clean this up
    surface, moneyness_tenor = shared.t_Buffer[key_code]
    max_index = np.prod(surface.shape.as_list()) - 1
    clipped_moneyness = tf.clip_by_value(moneyness,
                                         moneyness_tenor[0].min(),
                                         moneyness_tenor[0].max())
    flat_moneyness = tf.reshape(clipped_moneyness, (-1, 1))
    cmp = tf.cast(flat_moneyness >= np.append(moneyness_tenor[0], [np.inf]), dtype=tf.int32)
    index = tf.argmin(cmp[:, 1:] - cmp[:, :-1], axis=1, output_type=tf.int32)
    alpha = (tf.squeeze(flat_moneyness) -
             tf.gather(moneyness_tenor[0].astype(shared.precision), index)
             ) / tf.gather(moneyness_tenor[1].astype(shared.precision), index)

    expiry_indices = np.arange(expiry.size).astype(np.int32)
    expiry_offsets = tf.concat([tf.fill([shared.simulation_batch], x)
                                for x in expiry_indices * moneyness_tenor[0].size], axis=0)
    reshape = True
    if expiry_offsets.shape != index.shape:
        reshape = False
        vol_index = index + expiry_offsets[:index.shape[0].value]
    else:
        vol_index = index + expiry_offsets

    vol_index_next = tf.clip_by_value(vol_index + 1, 0, max_index)

    vols = tf.gather(surface, vol_index) * (1.0 - alpha) + \
           tf.gather(surface, vol_index_next) * alpha

    return tf.reshape(vols, (-1, shared.simulation_batch)) if reshape else tf.reshape(vols, [-1, 1])


def calc_time_grid_vol_rate(code, moneyness, expiry, shared, calc_std=False):
    key_code = ('vol2d', tuple([x[:2] for x in code]), tuple(expiry), calc_std)

    if key_code not in shared.t_Buffer:
        spread = None

        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

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
        space = tf.reshape(vol_spread, (tenor_index[0].size, -1))
        index = np.clip(np.searchsorted(tenor_index[0], tenor, side='right') - 1, 0, tenor_index[0].size - 2)
        alpha = np.clip((tenor - tenor_index[0][index]) / tenor_index[1][index], 0, 1.0)
        spread = (1.0 - alpha) * space[index] + alpha * space[min(index + 1, tenor_index[0].size - 1)]

        surface = tf.reshape(spread, (-1, code[0][FACTOR_INDEX_Moneyness_Index][0].size))
        flat_vol_time = tf.reshape(
            gather_surface_interp(surface, code[0], expiry, shared, calc_std), (-1,))

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
        space = tf.reshape(vol_spread, (tenor_index[0].size, -1))
        index = np.clip(np.searchsorted(tenor_index[0], tenor, side='right') - 1, 0, tenor_index[0].size - 2)
        alpha = np.clip((tenor - tenor_index[0][index]) / tenor_index[1][index], 0, 1.0)
        spread = (1.0 - alpha) * space[index] + alpha * space[min(index + 1, tenor_index[0].size - 1)]

        surface = tf.reshape(spread, (-1, code[0][FACTOR_INDEX_Moneyness_Index][0].size))
        result = []
        for exp, mon in zip(expiry, tf.unstack(moneyness)):
            time_exp = key_code[:-1] + tuple(exp)
            if time_exp not in shared.t_Buffer:
                flat_vol_time = tf.reshape(
                    gather_surface_interp(surface, code[0], exp, shared, calc_std), (-1,))
                shared.t_Buffer[time_exp] = (flat_vol_time, code[0][FACTOR_INDEX_Moneyness_Index])
            result.append(calc_moneyness_vol_rate(mon, exp, time_exp, shared))

        shared.t_Buffer[key_code] = tf.stack(result)

    return shared.t_Buffer[key_code]


def calc_delivery_time_grid_vol_rate(code, moneyness, expiry, delivery, time_grid, shared, calc_std=False):
    key_code = ('vol3d_energy', tuple([x[:2] for x in code]),
                tuple(expiry.ravel()), tuple(time_grid[:, TIME_GRID_MTM]),
                tuple(delivery.ravel()), calc_std)

    money_index = code[0][FACTOR_INDEX_Moneyness_Index]

    if key_code not in shared.t_Buffer:
        vol_spread = None

        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                vol_spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

        # get the moneyness from the flat surface
        moneyness_slice_size = np.array([len(x) for x in code[0][FACTOR_INDEX_Surface_Flat_Moneyness_Index]])
        moneyness_slice_index = np.append(0, moneyness_slice_size.cumsum())[:-1]
        moneyness_slice = []

        for delivery_index, expiry_index, vol_index in zip(
                code[0][FACTOR_INDEX_Surface_Flat_Moneyness_Index],
                code[0][FACTOR_INDEX_Surface_Flat_Expiry_Index], moneyness_slice_index):
            # get the expiry index
            index_expiry = np.clip(
                np.searchsorted(expiry_index[0], expiry, side='right') - 1,
                0, expiry_index[0].size - 1)
            index_expiry_p1 = np.clip(index_expiry + 1, 0, expiry_index[0].size - 1)

            alpha_exp = np.clip((expiry - expiry_index[0][index_expiry]) /
                                expiry_index[1][index_expiry], 0, 1.0)

            # get the delivery index per expiry
            index_del_00 = [[np.clip(np.searchsorted(
                delivery_index[int(y[0])][0], y[1], side='right') - 1, 0, delivery_index[int(y[0])][0].size - 1)
                             for y in x] for x in np.dstack((index_expiry, delivery))]
            index_del_01 = [[np.clip(
                y[0] + 1, 0, delivery_index[int(y[1])][0].size - 1) for y in x]
                for x in np.dstack((index_del_00, index_expiry))]
            index_del_10 = [[np.clip(np.searchsorted(
                delivery_index[int(y[0])][0], y[1], side='right') - 1, 0, delivery_index[int(y[0])][0].size - 1)
                             for y in x] for x in np.dstack((index_expiry_p1, delivery))]
            index_del_11 = [[np.clip(
                y[0] + 1, 0, delivery_index[int(y[1])][0].size - 1) for y in x]
                for x in np.dstack((index_del_10, index_expiry_p1))]

            # get the interpolation factors
            alpha_del_0 = np.vstack(
                [[np.clip((y[1] - delivery_index[int(y[2])][0][int(y[0])])
                          / delivery_index[int(y[2])][1][int(y[0])], 0, 1.0)
                  for y in x] for x in np.dstack((index_del_00, delivery, index_expiry))])

            alpha_del_1 = np.vstack(
                [[np.clip((y[1] - delivery_index[int(y[2])][0][int(y[0])])
                          / delivery_index[int(y[2])][1][int(y[0])], 0, 1.0)
                  for y in x] for x in np.dstack((index_del_10, delivery, index_expiry_p1))])

            delivery_slice_size = np.array([x[0].size for x in delivery_index])
            delivery_slice_index = vol_index + np.append(0, delivery_slice_size.cumsum())

            index_00 = delivery_slice_index[index_expiry] + np.vstack(index_del_00)
            index_01 = delivery_slice_index[index_expiry] + np.vstack(index_del_01)
            index_10 = delivery_slice_index[index_expiry_p1] + np.vstack(index_del_10)
            index_11 = delivery_slice_index[index_expiry_p1] + np.vstack(index_del_11)

            moneyness_slice.append(
                (1.0 - alpha_exp) * (1.0 - alpha_del_0) * tf.gather(vol_spread, index_00) +
                (1.0 - alpha_exp) * alpha_del_0 * tf.gather(vol_spread, index_01) +
                alpha_exp * (1.0 - alpha_del_1) * tf.gather(vol_spread, index_10) +
                alpha_exp * alpha_del_1 * tf.gather(vol_spread, index_11))

        shared.t_Buffer[key_code] = tf.stack(moneyness_slice)

    surface = shared.t_Buffer[key_code]

    clipped_moneyness = tf.clip_by_value(moneyness,
                                         money_index[0].min(),
                                         money_index[0].max())

    flat_moneyness = tf.reshape(clipped_moneyness, (-1, 1))
    cmp = tf.cast(flat_moneyness >= np.append(money_index[0], [np.inf]), dtype=tf.int32)
    index = tf.argmin(cmp[:, 1:] - cmp[:, :-1], axis=1, output_type=tf.int32)
    alpha = (tf.squeeze(flat_moneyness) -
             tf.gather(money_index[0].astype(shared.precision), index)
             ) / tf.gather(money_index[1].astype(shared.precision), index)

    tenor_surface = tf.reshape(tf.transpose(surface, [1, 0, 2]), [-1, surface.shape[2]])
    vol_index = tf.reshape(index, (-1, shared.simulation_batch))
    vol_alpha = tf.reshape(alpha, (-1, shared.simulation_batch, 1))

    surface_offset = (np.arange(surface.shape[1].value) *
                      surface.shape[0].value).reshape(-1, 1)

    vols = tf.gather(tenor_surface, vol_index + surface_offset) * (1.0 - vol_alpha) + \
           tf.gather(tenor_surface, tf.clip_by_value(
               vol_index + 1, 0, money_index[0].size - 1) + surface_offset) * vol_alpha

    return tf.transpose(vols, [0, 2, 1])


def hermite_interpolation_tensor(t, rate_tensor):
    rate_diff = (rate_tensor[:, 1:, :] - rate_tensor[:, :-1, :])
    time_diff = np.diff(t, axis=1)

    # calc r_i
    r_i = ((rate_diff[:, :-1, :] * time_diff[:, 1:, :]) / time_diff[:, :-1, :] +
           (rate_diff[:, 1:, :] * time_diff[:, :-1, :]) / time_diff[:, 1:, :]) / (
                  t[:, 2:, :] - t[:, :-2, :])
    r_1 = ((tf.gather(rate_diff, 0, axis=1) * (t[:, 2, :] + t[:, 1, :] - 2.0 * t[:, 0, :])) / time_diff[:, 0, :] -
           (tf.gather(rate_diff, 1, axis=1) * time_diff[:, 0, :]) / time_diff[:, 1, :]) / (t[:, 2, :] - t[:, 0, :])

    r_n = (-1.0 / (t[:, -1, :] - t[:, -3, :])) * (
            (tf.gather(rate_diff, -2, axis=1) * time_diff[:, -1, :]) / time_diff[:, -2, :] -
            (tf.gather(rate_diff, -1, axis=1) * (2.0 * t[:, -1, :] - t[:, -2, :] - t[:, -3, :])) / time_diff[:, -1, :])

    ri = tf.concat([tf.expand_dims(r_1, axis=1), r_i, tf.expand_dims(r_n, axis=1)], axis=1)

    # zero
    zero = tf.expand_dims(tf.zeros_like(r_1), axis=1)
    # calc g_i
    gi = tf.concat([time_diff * ri[:, :-1, :] - rate_diff, zero], axis=1)
    # calc c_i
    ci = tf.concat([2.0 * rate_diff - time_diff * (ri[:, :-1, :] + ri[:, 1:, :]), zero], axis=1)

    return gi, ci


def cache_interpolation(shared, curve_component, tensor):
    key_code = (curve_component[FACTOR_INDEX_Tenor_Index][2], curve_component[:2],
                tuple(tensor.shape.as_list()))

    if key_code not in shared.t_Buffer:

        if key_code[0] in ['HermiteRT', 'Hermite']:
            t = curve_component[FACTOR_INDEX_Tenor_Index][0].reshape(1, -1, 1)
            mod_tenor = tensor
            if curve_component[FACTOR_INDEX_Tenor_Index][2] == 'HermiteRT':
                mod_tenor = tensor * t

            g, c = hermite_interpolation_tensor(t, mod_tenor)
            shared.t_Buffer[key_code] = mod_tenor, (g, c)
        else:
            shared.t_Buffer[key_code] = tensor, None

    return shared.t_Buffer[key_code]


def calc_time_grid_curve_rate(code, time_grid, shared, points=None, multiply_by_time=False):
    time_hash = tuple(time_grid[:, TIME_GRID_MTM])
    code_hash = tuple([x[:2] for x in code])
    points_hash = tuple(tuple(x) for x in points) if points is not None else None

    key_code = ('curve', code_hash, time_hash, points_hash, multiply_by_time)

    if key_code not in shared.t_Buffer:
        with tf.name_scope(None):
            value, interp = [], []

            for rate in code:
                rate_code = ('curve_factor', rate[:2], time_hash, points_hash, multiply_by_time)
                interp_scenario = rate[FACTOR_INDEX_Tenor_Index][2] != 'Linear' or points is None

                # if points are defined, calculate directly - otherwise interpolate from
                # the scenario and static buffers
                if interp_scenario:
                    # check if the curve factors are already available
                    if rate_code not in shared.t_Buffer:
                        if rate[FACTOR_INDEX_Stoch]:
                            tensor = shared.t_Scenario_Buffer[rate[FACTOR_INDEX_Offset]]
                            with tf.name_scope(check_tensor_name(tensor.name, 'curve')):
                                cached_tensor, interpolation_params = cache_interpolation(
                                    shared, rate, tensor)
                                spread = gather_scenario_interp(cached_tensor, time_grid, shared)
                        else:
                            tensor = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                            with tf.name_scope(check_tensor_name(tensor.name, 'curve')):
                                spread, interpolation_params = cache_interpolation(
                                    shared, rate, tf.reshape(tensor, (1, -1, 1)))

                        # cache interpolation if necessary
                        interp_interp = [gather_scenario_interp(params, time_grid, shared)
                                         for params in interpolation_params] \
                            if interpolation_params is not None else None

                        # store it
                        shared.t_Buffer[rate_code] = (spread, interp_interp)
                else:
                    if rate_code not in shared.t_Buffer:
                        # if points are defined, calculate directly - otherwise interpolate from
                        # the scenario and static buffers
                        if rate[FACTOR_INDEX_Stoch]:
                            spread = rate[FACTOR_INDEX_Process].calc_points(rate, points, time_grid, shared,
                                                                            multiply_by_time)
                        else:
                            tensor = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                            if shared.riskneutral:
                                spread = interpolate_risk_neutral(points, tensor, rate, None, time_grid,
                                                                  multiply_by_time)
                            else:
                                spread = interpolate_curve(tensor, rate, None, points, multiply_by_time)

                        # store it
                        shared.t_Buffer[rate_code] = (spread, None)

                # append the curve and its (possible) interpolation parameters
                value.append(shared.t_Buffer[rate_code][0])
                interp.append(shared.t_Buffer[rate_code][1])

            shared.t_Buffer[key_code] = TensorBlock(code=code, tensor=value,
                                                    interp=interp, time_grid=time_grid)

    return shared.t_Buffer[key_code]


def calc_time_grid_spot_rate(rate, time_grid, shared):
    key_code = ('spot', tuple(rate[0][:2]), tuple(time_grid[:, TIME_GRID_MTM]))

    if key_code not in shared.t_Buffer:
        with tf.name_scope(None):
            if rate[0][FACTOR_INDEX_Stoch]:
                tensor = shared.t_Scenario_Buffer[rate[0][FACTOR_INDEX_Offset]]
                with tf.name_scope(check_tensor_name(tensor.name, 'spot')):
                    value = gather_scenario_interp(tensor, time_grid, shared)
            else:
                tensor = shared.t_Static_Buffer[rate[0][FACTOR_INDEX_Offset]]
                with tf.name_scope(check_tensor_name(tensor.name, 'spot')):
                    value = tf.reshape(tensor, (1, -1))

            shared.t_Buffer[key_code] = value

    return shared.t_Buffer[key_code]


def calc_curve_forwards(factor, tensor, time_grid, shared, ref_date, mul_time=True):
    factor_tenor = factor.get_tenor()

    tnr, tnr_d = factor_tenor, np.hstack((np.diff(factor_tenor), [1]))
    max_dim = factor_tenor.size - 1

    # calculate the tenors and indices used for lookups
    ten_t1 = np.array([np.clip(
        np.searchsorted(factor_tenor,
                        factor.get_day_count_accrual(ref_date, t),
                        side='right') - 1,
        0, max_dim) for t in time_grid.scen_time_grid])

    ten_t = np.array([np.clip(
        np.searchsorted(factor_tenor,
                        factor_tenor + factor.get_day_count_accrual(ref_date, t),
                        side='right') - 1,
        0, max_dim) for t in time_grid.scen_time_grid])

    ten_t1_next = np.clip(ten_t1 + 1, 0, max_dim)
    ten_t_next = np.clip(ten_t + 1, 0, max_dim)

    ten_tv = np.clip(np.array(
        [factor_tenor + factor.get_day_count_accrual(ref_date, t)
         for t in time_grid.scen_time_grid])
        , 0, factor_tenor.max())

    alpha_1 = (ten_tv - tnr[ten_t]) / tnr_d[ten_t]
    alpha_2 = (time_grid.time_grid_years - tnr[ten_t1]).clip(0, np.inf) / tnr_d[ten_t1]

    if factor.interpolation[0].startswith('Hermite'):
        t = tnr.reshape(1, -1, 1)
        if factor.interpolation[0] == 'HermiteRT':
            mod = tf.reshape(tensor, [1, -1, 1]) * t
            norm = None
        else:
            mod = tf.reshape(tensor, [1, -1, 1])
            norm = (ten_tv, time_grid.time_grid_years) if mul_time else (1.0, 1.0)

        g, c = [tf.squeeze(x) for x in hermite_interpolation_tensor(t, mod)]
        sq = tf.squeeze(mod)

        val = tf.gather(sq, ten_t) * (1.0 - alpha_1) + tf.gather(sq, ten_t_next) * alpha_1 + \
              alpha_1 * (1.0 - alpha_1) * tf.gather(g, ten_t) + \
              alpha_1 * alpha_1 * (1.0 - alpha_1) * tf.gather(c, ten_t)

        val_t = tf.gather(sq, ten_t1) * (1.0 - alpha_2) + tf.gather(sq, ten_t1_next) * alpha_2 + \
                alpha_2 * (1.0 - alpha_2) * tf.gather(g, ten_t1) + \
                alpha_2 * alpha_2 * (1.0 - alpha_2) * tf.gather(c, ten_t1)

        if norm is None:
            return val - tf.reshape(val_t, [-1, 1])
        else:
            return val * norm[0] - tf.reshape(val_t * norm[1], [-1, 1])
    else:
        fwd1 = alpha_1 * tf.gather(tensor, ten_t_next) + (1 - alpha_1) * tf.gather(tensor, ten_t)
        fwd2 = alpha_2 * tf.gather(tensor, ten_t1_next) + (1 - alpha_2) * tf.gather(tensor, ten_t1)

        norm = (time_grid.time_grid_years.reshape(-1, 1) + tnr,
                time_grid.time_grid_years) if mul_time else (1.0, 1.0)

        return fwd1 * norm[0] - tf.reshape(fwd2 * norm[1], (-1, 1))


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

        # ignore any outlier greater then 2 std deviations from the median
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
    Repeatedly go through all of the nodes in the graph, moving each of
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
            return [element.get(field[0]) for element in obj if element.get(field[0])]
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
                                       freq=reset_frequency, closed='left')
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

            # only add a reset if its in the past
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
    # Pay_Date if settlement_date is None else -(settlement_date - base_date).days ])
    cash = []
    reset_offsets = []

    for cashflow in cashflows['Items']:
        rate = cashflow['Rate'] if isinstance(cashflow['Rate'], float) else cashflow['Rate'].amount
        if cashflow['Payment_Date'] >= reference_date and (
                (cashflow['Payment_Date'] >= settlement_date) if settlement_date else True):
            Accrual_Start_Date = cashflow['Accrual_Start_Date'] if cashflow['Accrual_Start_Date'] else cashflow[
                'Payment_Date']
            Accrual_End_Date = cashflow['Accrual_End_Date'] if cashflow['Accrual_End_Date'] else cashflow[
                'Payment_Date']
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
    D = float(sum([x[2] for x in samples]))

    for sample in sorted(samples):
        Reset_Day = (sample[0] - reference_date).days
        Start_Day = Reset_Day
        End_Day = Reset_Day
        Weight = sample[2] / D
        Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
        # only add a reset if its in the past
        all_resets.append(
            [Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
             sample[1] if sample[0] < reference_date else 0.0, 0.0])
        reset_scenario_offsets.append(Scenario)

    return TensorResets(all_resets, reset_scenario_offsets)


def make_simple_fixed_cashflows(reference_date, position, cashflows):
    """
    Generates a vector of fixed cashflows from a data source only looking at the actual fixed value.
    """
    cash = OrderedDict()
    for cashflow in cashflows['Items']:
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
    for cashflow in cashflows['Items']:
        if cashflow['Payment_Date'] >= reference_date:
            cash.append(
                [(cashflow['Payment_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days,
                 (cashflow['Payment_Date'] - reference_date).days,
                 1.0, 0.0, position * cashflow['Volume'] * cashflow['Fixed_Price'], 0.0, 0.0, 0.0])

    # Add the null resets to the end
    dummy_resets = np.zeros((len(cash), 3))

    return TensorCashFlows(cash, dummy_resets)


def make_equity_swaplet_cashflows(base_date, time_grid, position, cashflows):
    """
    Generates a vector of equity cashflows from a data source.
    """
    cash = []
    all_resets = []
    all_divs = []
    cashflow_reset_offsets = []
    cashflow_divi_offsets = []
    reset_scenario_offsets = []
    divi_scenario_offsets = []
    dividend_sample_frq = pd.DateOffset(months=1)

    for cashflow in cashflows['Items']:
        if cashflow['Payment_Date'] >= base_date:
            cash.append([(cashflow['Start_Date'] - base_date).days, (cashflow['End_Date'] - base_date).days,
                         (cashflow['Payment_Date'] - base_date).days,
                         cashflow['Start_Multiplier'], cashflow['End_Multiplier'], position * cashflow['Amount'],
                         cashflow['Dividend_Multiplier'], 0.0, 0.0])

            r = []
            for reset in ['Start', 'End']:
                Reset_Day = (cashflow[reset + '_Date'] - base_date).days
                Start_Day = Reset_Day
                # we map the weight of the reset with the prior dividends
                Weight = cashflow.get('Known_Dividend_Sum', 0.0)

                # Need to use this reset to estimate future dividends
                Time_Grid, Scenario = time_grid.get_scenario_offset(max(Reset_Day, 0))

                # only add a reset if its in the past
                r.append([Time_Grid, Reset_Day, -1, Start_Day, 0, Weight,
                          cashflow['Known_' + reset + '_Price'] if Start_Day <= 0 else 0.0,
                          cashflow['Known_' + reset + 'FX_Rate'] if Start_Day <= 0 else 0.0])
                reset_scenario_offsets.append(Scenario)

            d = []
            for reset in pd.date_range(cashflow['Start_Date'], cashflow['End_Date'],
                                       freq=dividend_sample_frq)[:10]:
                Reset_Day = (reset - base_date).days
                Start_Day = Reset_Day
                End_Day = (reset + dividend_sample_frq - base_date).days
                # Need to use this reset to estimate future dividends
                Time_Grid, Scenario = time_grid.get_scenario_offset(max(Reset_Day, 0))

                # only add a reset if its in the past
                d.append([Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight, 0.0, 0.0])
                divi_scenario_offsets.append(Scenario)

            cashflow_divi_offsets.append([len(d), len(all_divs), 0])
            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)
            # store divs
            all_divs.extend(d)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)
    dividends = TensorCashFlows(cash, cashflow_divi_offsets)
    dividends.set_resets(all_divs, divi_scenario_offsets)

    return cashflows, dividends


def make_index_cashflows(base_date, time_grid, position, cashflows, price_index, index_rate, settlement_date,
                         isBond=True):
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

    for cashflow in cashflows['Items']:
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
                locals()[price_index.param['Reference_Name']](base_reference_date, base_date,
                                                              base_resets, base_scenario_offsets)
                locals()[price_index.param['Reference_Name']](final_reference_date, base_date,
                                                              final_resets, final_scenario_offsets)

    # set the cashflows
    cashflows = TensorCashFlows(sorted(cash), cashflow_reset_offsets)

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

        return cashflows, TensorResets(base_resets, base_scenario_offsets), TensorResets(final_resets,
                                                                                         final_scenario_offsets)

    else:
        for eval_time in time_grid.time_grid[:, TIME_GRID_MTM]:
            actual_time = base_date + pd.DateOffset(days=eval_time)

            locals()[price_index.param['Reference_Name']](actual_time, index_rate.param['Last_Period_Start'],
                                                          time_resets, time_scenario_offsets)

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

    for cashflow in cashflows['Items']:
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
                # only add a reset if its in the past
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

    for cashflow in cashflows['Items']:
        if cashflow['Payment_Date'] >= reference_date:
            cash.append(
                [(cashflow['Period_Start'] - reference_date).days, (cashflow['Period_End'] - reference_date).days,
                 (cashflow['Payment_Date'] - reference_date).days, cashflow.get('Price_Multiplier', 1.0),
                 position * cashflow['Volume'], 0.0, cashflow.get('Fixed_Basis', 0.0), 0.0, 0.0])

            r = []
            bunsiness_dates = pd.date_range(cashflow['Period_Start'], cashflow['Period_End'],
                                            freq=forward_calendar_bday)

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
                          cashflow['Realized_Average'], cashflow['FX_Realized_Average']])
                reset_scenario_offsets.append(Scenario)

            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def compress_no_compounding(cashflows, groupsize):
    cash_pmts, cash_index, cash_counts = np.unique(
        cashflows.schedule[:, CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

    if (cashflows.offsets[:, 0] == 1).all() and (cash_counts > groupsize).any():
        # can compress
        cash, cashflow_reset_offsets = [], []
        all_resets, reset_scenario_offsets = [], []
        for pay_day, index, num_cf in zip(*[cash_pmts, cash_index, cash_counts]):
            cashflow_schedule = cashflows.schedule[index:index + num_cf]
            cashflow_offsets = cashflows.offsets[index:index + num_cf]
            reset_offset = cashflows.offsets[index:index + num_cf, 1]
            nominals = np.unique(cashflow_schedule[:, CASHFLOW_INDEX_Nominal])
            margins = np.unique(cashflow_schedule[:, CASHFLOW_INDEX_FloatMargin])
            if nominals.size <= groupsize and margins.size <= groupsize and not (
                    cashflows.Resets[reset_offset, RESET_INDEX_Reset_Day] < 0).any():

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
        logging.warning('Cashflows reduced from {} resets to {} resets'.format(cashflows.Resets.count(),
                                                                               approx_cashflows.Resets.count()))
        return approx_cashflows
    else:
        return cashflows


if __name__ == '__main__':
    import pickle

    with open(r'C:\temp\cf.obj', 'rb') as f:
        cf = pickle.load(f)
    with open(r'C:\temp\tg.obj', 'rb') as f:
        tg = pickle.load(f)


    def make_df(cf):
        resets = pd.DataFrame(cf.Resets.schedule,
                              columns=['time', 'reset', 'scen', 'start', 'end', 'weight', 'value', 'Accrual'])
        cashflows = pd.DataFrame(cf.schedule,
                                 columns=['start', 'end', 'pay', 'year_frac', 'nominal', 'fixed', 'float_margin',
                                          'fxreset', 'fxval'])
        return cashflows, resets


    cashflows, resets = make_df(cf)
    com_cf = compress_no_compounding(cf, 3)

    com_cashflows, com_resets = make_df(com_cf)
