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
from functools import reduce, wraps
from collections import namedtuple, deque
from typing import Tuple, List
from dataclasses import dataclass

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
# for equity swaps, we need to adjust days based on settlement
CASHFLOW_INDEX_Start_Adj = 7
CASHFLOW_INDEX_FXResetValue = 8
CASHFLOW_INDEX_End_Adj = 8

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
OneDimensionalFactors = ['InterestRate', 'InflationRate', 'DividendRate', 'SurvivalProb', 'ForwardPrice', 'ForwardRate']
TwoDimensionalFactors = ['FXVol', 'EquityPriceVol', 'CommodityPriceVol']
ThreeDimensionalFactors = ['InterestRateVol', 'InterestYieldVol', 'ForwardPriceVol']

# weekends and weekdays
WeekendMap = {'Friday and Saturday': 'Sun Mon Tue Wed Thu',
              'Saturday and Sunday': 'Mon Tue Wed Thu Fri',
              'Sunday': 'Mon Tue Wed Thu Fri Sat',
              'Saturday': 'Sun Mon Tue Wed Thu Fri',
              'Friday': 'Sat Sun Mon Tue Wed Thu'}


@dataclass
class DeferredDeal:
    payload: dict


# Custom Exceptions
class InstrumentExpired(Exception):
    def __init__(self, message):
        self.message = message


def is_fatal_pricing_error(e):
    """Exceptions a deal-level pricing guard must NOT swallow into a scalar-0 mark: the machine
    running out of memory. That says the FRAMEWORK is wrong rather than the deal, and it produces
    a silently missing mark if caught — inside an inner-MC fork a missing tradable mark reads as
    an expired contract and retires the instrument from the hedge set. Everything else keeps the
    canonical skip."""
    return isinstance(e, (MemoryError, torch.cuda.OutOfMemoryError)) or (
        isinstance(e, RuntimeError) and 'out of memory' in str(e).lower())


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
            logging.exception("An error occurred in function '%s'", func.__name__)
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

    def __add__(self, other):
        return self.amount + other

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
        self.data = dict(data)
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
        self.data = dict(data)

    def value(self):
        return next(iter(self.data.values()))

    def __str__(self):
        return '\\'.join(['%d=%.12g' % (rating, amount) for rating, amount in self.data.items()]) + '\\'


class DateEqualList:
    def __init__(self, data):
        self.data = {x[0]: x[1:] for x in data}

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


def select_rows(operand, pos):
    """Row subset of a per-time-row operand (indices, interp weights, tenors, alpha) for a routed
    group. A leading dim of 1 is broadcasting against the time axis and must be left alone."""
    return operand if pos is None or not torch.is_tensor(operand) or operand.shape[0] == 1 \
        else operand[pos]


class ScenarioBlock(object):
    """One physical scenario tensor and where it sits in the LOGICAL grid.

    `first_row` is the block's first logical scenario row. `batch_index` maps each logical batch
    column to the column of THIS block that supplies it — `None` when the block is already at the
    logical width. An inner-MC fork's realized past holds one outer column per `Inner_Sub_Batch`
    flat columns, so its map is the flattening the fork itself performed: passing it as data is
    what stops the two ends having to agree by arithmetic.
    """

    def __init__(self, tensor, first_row=0, batch_index=None):
        self.tensor = tensor
        self.first_row = first_row
        self.batch_index = batch_index
        self.n_rows = tensor.shape[0]

    def project(self, val):
        """A read at this block's width, taken up to the logical grid's.

        Applied to the RESULT of a read, never to the stored tensor: projecting the tensor would
        materialize the block at the logical width and hand back exactly the memory the block
        split exists to save. It is a batch-axis gather, so it commutes with the time blend and
        with `combine` — which is what lets the caller project first and blend after."""
        return val if self.batch_index is None else val.index_select(-1, self.batch_index)

    def __mul__(self, other):
        return ScenarioBlock(self.tensor * other, self.first_row, self.batch_index)


class ScenarioSource(object):
    """A factor's scenario grid as the pricer sees it: a SEQUENCE of `ScenarioBlock`s under one
    logical shape, each at its own batch width.

    Ordinary generation publishes a bare tensor and never builds one of these, so base valuation,
    credit Monte Carlo and the outer hedge loop never meet this class. An inner-MC fork publishes
    TWO blocks: the outer-realized past at `Batch_Size`, then the forked rows at
    `Batch_Size x Inner_Sub_Batch`. Every past row is identical across the inner draws, so joining
    them into one tensor writes the realized past out `Inner_Sub_Batch` times — 98% of the stuffed
    buffer at the production operating point, dragging a same-shaped slab of Hermite coefficients
    with it.

    Write-once and read-only: built after every process's `generate` has published, and carrying
    only the operations `make_curve_tensor` performs on a raw buffer value, so anything else fails
    loud rather than silently materializing.
    """

    def __init__(self, *blocks):
        self.blocks = blocks
        self.cuts = np.cumsum([b.n_rows for b in blocks[:-1]], dtype=np.int64)
        self.shape = (sum(b.n_rows for b in blocks),) + tuple(blocks[-1].tensor.shape[1:])

    def new(self, *args, **kwargs):
        return self.blocks[-1].tensor.new(*args, **kwargs)

    def __mul__(self, other):
        # the LinearRT/HermiteRT tenor rescale — elementwise over the tenor axis, so per block
        return ScenarioSource(*[b * other for b in self.blocks])


class Interpolation(object):
    """Tenor and time interpolation over ONE physical scenario tensor.

    A leaf, and the only class base valuation / credit Monte Carlo / the outer hedge loop ever
    build. `build` prepares what a given interpolation kind stores — an RT kind folds the tenor
    into the values, Hermite derives its coefficient pair — and `eval` looks at the kind and
    returns just what it needs. Dividend curves are plain here; what makes them different lives
    in `CurveTenor`.

    It knows nothing about inner MC, block boundaries, logical rows or batch fan-out: the rows
    reaching it are already in its own frame, and it flattens them against its OWN tenor stride,
    which is what lets a tenor segment be the same kind of object.
    """

    def __init__(self, tensor, interp_params):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)
        self.indexed_tensor = tensor.reshape(-1, tensor.shape[-1])
        self.interp_params = [p.reshape(-1, p.shape[-1]) for p in interp_params]

    @classmethod
    def build(cls, tensor, kind, tenor):
        """What an interpolation of `kind` stores: the values, and whatever it derives from them.
        Rate*time folds the tenor into the values; Hermite derives its coefficient pair."""
        if kind == 'Linear':
            return cls(tensor, [])
        t = tensor.new(tenor[:tensor.shape[1]]).reshape(1, -1, 1)
        if kind in ('Hermite', 'HermiteRT'):
            values = tensor * t if kind == 'HermiteRT' else tensor
            return cls(values, hermite_interpolation_tensor(t, values))
        if kind == 'LinearRT':
            return cls(tensor * t, [])
        return cls(tensor, [])

    def route(self, index, has_alpha):
        """A leaf IS the whole grid — there is nothing to route."""
        return None

    def read_at(self, tenor_data, rows, i1, i2, w2):
        """The RAW value at one time point — before the rate*time scaling, which `combine` applies
        after any time blend.

        Scenario rows are flattened into this tensor's (row, tenor) frame HERE, against its OWN
        stride: a tenor segment's stride is its own, which is the whole reason `CurveTensor` hands
        out rows rather than a flat offset. `rows is None` means every row is row 0 — a static
        curve, or a stochastic one gathered only at the base date — and skips the add entirely."""
        base = None if rows is None else rows.reshape(-1, 1) * self.shape[1]
        i0, i1x = (i1, i2) if base is None else (base + i1, base + i2)
        if tenor_data[0].startswith('Hermite'):
            g, c = self.interp_params
            return calc_hermite_curve(
                w2, g[i0,], c[i0,], self.indexed_tensor[i0,], self.indexed_tensor[i1x,])
        # default to linear
        return self.indexed_tensor[i0,] * (1.0 - w2) + self.indexed_tensor[i1x,] * w2

    def blend(self, raw, nxt, alpha):
        """Linear time interpolation between two raw reads."""
        return (1 - alpha) * raw + alpha * nxt

    def project(self, block, raw):
        """A raw read taken up to the logical batch width by the block that produced it."""
        return block.project(raw)

    def combine(self, raw, tenor_data, i2, tnr, time_factor):
        """Raw read -> curve value: the rate*time scaling this kind asks for. Elementwise in
        `raw`, so it commutes with the time blend and with a block projection — which is why it
        runs ONCE, after both, rather than inside either."""
        kind, tnr_min, tnr_max = tenor_data
        tenors = tnr.unsqueeze(-1)
        mult = tenors if time_factor else 1.0
        if kind.endswith('RT'):
            mult = mult / tenors.clamp(tnr_min, tnr_max)
        return raw * mult

    def eval(self, tenor_data, index, index_next, alpha, i1, i2, w2, tnr, time_factor, route=None):
        raw = self.read_at(tenor_data, index, i1, i2, w2)
        if alpha is not None:
            # the t+1 read is taken BEFORE either weighting, so no full-width term is held across it
            raw = self.blend(raw, self.read_at(tenor_data, index_next, i1, i2, w2), alpha)
        return self.combine(raw, tenor_data, i2, tnr, time_factor)

    def gather_rows(self, index, index_next, alpha, route=None):
        """Whole rows at `index` — the 0D spot path."""
        if alpha is None:
            return self.tensor[index]
        return self.tensor[index] * (1 - alpha) + self.tensor[index_next] * alpha


class SegmentedInterpolation(object):
    """A curve whose tenor axis is split at a near index, each side interpolated its own way
    (`Near_Interpolation`). A SIBLING of `Interpolation`, not a subclass: it composes leaves over
    TENOR, exactly as `RoutedInterpolation` composes strategies over SCENARIO ROWS, and the two
    compositions are orthogonal.

    Segments are middle-dim slices with their own tenor divisors, so each owns its own flat
    stride — which is the reason `CurveTensor` hands out scenario ROWS and lets the strategy
    flatten them.
    """

    def __init__(self, tensor, spec, tenor):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)
        self.indexed_tensor = tensor.reshape(-1, tensor.shape[-1])
        self.spec = spec
        # this only works for 2 segments - checked when the factor is built
        self.cutoff = spec[0][1]
        self.segments = [Interpolation.build(tensor[:, s:e + 1, :], kind, tenor[s:e + 1])
                         for s, e, kind in spec]


    def route(self, index, has_alpha):
        return None

    def seg_tenors(self, seg_i, i1, i2):
        """`i1, i2` in segment `seg_i`'s own tenor frame."""
        s, e, _kind = self.spec[seg_i]
        if seg_i == 0:
            return i1.clamp(max=e), i2.clamp(max=e)
        return (i1 - s).clamp(min=0), (i2 - s).clamp(min=0)

    def read_at(self, tenor_data, rows, i1, i2, w2):
        """One raw read PER SEGMENT — evaluated on the full tenor set and selected in `combine`.
        More work than we need, but it keeps every segment a plain leaf."""
        return [seg.read_at((seg_spec[2], tnr_min, tnr_max), rows,
                            *self.seg_tenors(k, i1, i2), w2)
                for k, (seg, (seg_spec, tnr_min, tnr_max))
                in enumerate(zip(self.segments, zip(*tenor_data)))]

    def blend(self, raw, nxt, alpha):
        return [seg.blend(a, b, alpha) for seg, a, b in zip(self.segments, raw, nxt)]

    def project(self, block, raw):
        return [seg.project(block, v) for seg, v in zip(self.segments, raw)]

    def combine(self, raw, tenor_data, i2, tnr, time_factor):
        """Each segment's own scaling, then the tenor select between them. `tenor_data` is the
        `(spec, (min, split), (split, max))` triple, so `zip(*tenor_data)` is one segment's
        `((start, end, kind), tnr_min, tnr_max)`."""
        vals = [seg.combine(v, (seg_spec[2], tnr_min, tnr_max),
                            self.seg_tenors(k, i2, i2)[1], tnr, time_factor)
                for k, (seg, v, (seg_spec, tnr_min, tnr_max))
                in enumerate(zip(self.segments, raw, zip(*tenor_data)))]
        return torch.where((i2 <= self.cutoff).unsqueeze(-1), vals[0], vals[1])

    def eval(self, tenor_data, index, index_next, alpha, i1, i2, w2, tnr, time_factor, route=None):
        raw = self.read_at(tenor_data, index, i1, i2, w2)
        if alpha is not None:
            raw = self.blend(raw, self.read_at(tenor_data, index_next, i1, i2, w2), alpha)
        return self.combine(raw, tenor_data, i2, tnr, time_factor)

    def gather_rows(self, index, index_next, alpha, route=None):
        if alpha is None:
            return self.tensor[index]
        return self.tensor[index] * (1 - alpha) + self.tensor[index_next] * alpha


class RoutedInterpolation(object):
    """One logical scenario grid over several physical blocks — an inner-MC fork's realized past
    and its forked rows — each carrying its OWN interpolation, built recursively from the same
    curve tenor. A segmented curve inside a fork is therefore a `RoutedInterpolation` of
    `SegmentedInterpolation`s and needs no special case here.

    It owns exactly the composite concerns: which block holds a row, rebasing a logical row into
    that block's frame, projecting a narrow block's read up to the logical batch width, and
    reassembling the groups in the caller's row order. The interpolations stay unaware of all of it.
    """

    def __init__(self, source, curve_tenor):
        self.blocks = source.blocks
        self.shape, self.cuts = source.shape, source.cuts
        self.strategies = tuple(build_interpolation(b.tensor, curve_tenor) for b in source.blocks)
        # the last block is the one already at the logical batch width, so it answers the
        # tensor-shaped questions a leaf answers for itself
        self.tensor = source.blocks[-1].tensor
        self.indexed_tensor = self.strategies[-1].indexed_tensor

    def route(self, index, has_alpha):
        """Group a gather's rows by the block that owns each of its two reads: `(row positions,
        block for the t read, block for the t+1 read)`, positions `None` when ONE group covers
        every row. A time-interpolated read reaches `index + 1`, so a row just below a cut reads
        ACROSS it and names two blocks — classify on where a read ENDS, not where it starts.
        Decided from the numpy indices a `CurveTensor` already holds, so it costs no device sync,
        and once per `CurveTensor` rather than once per gather."""
        hi = np.minimum(index + 1, self.shape[0] - 1) if has_alpha else index
        at_t = np.searchsorted(self.cuts, index, side='right')
        at_t1 = np.searchsorted(self.cuts, hi, side='right')
        # an empty gather (a step with no resets in range) names no rows: one group, empty
        pairs = np.unique(np.stack([at_t, at_t1]), axis=1) if index.size else np.zeros((2, 1), int)
        if pairs.shape[1] == 1:
            return ((None, int(pairs[0, 0]), int(pairs[1, 0])),)
        return tuple((torch.tensor(np.flatnonzero((at_t == t0) & (at_t1 == t1)),
                                   dtype=torch.int64, device=self.tensor.device), int(t0), int(t1))
                     for t0, t1 in pairs.T)

    def routed(self, route, n_rows, read):
        """Run `read` per routed group and put the groups back in the caller's row order. A group
        covering every row answers directly."""
        out = None
        for pos, at_t, at_t1 in route:
            val = read(pos, at_t, at_t1)
            if pos is None:
                return val
            out = val.new_empty((n_rows,) + tuple(val.shape[1:])) if out is None else out
            out.index_copy_(0, pos, val)
        return out

    def local(self, rows, block):
        """Logical scenario rows in `block`'s own frame."""
        return rows if rows is None or not block.first_row else rows - block.first_row

    def eval(self, tenor_data, index, index_next, alpha, i1, i2, w2, tnr, time_factor, route):
        def group(pos, at_t, at_t1):
            b0, b1 = self.blocks[at_t], self.blocks[at_t1]
            s0, s1 = self.strategies[at_t], self.strategies[at_t1]
            rows = self.local(select_rows(index, pos), b0)
            weight, t1, t2 = (select_rows(x, pos) for x in (w2, i1, i2))
            nxt = None if index_next is None else self.local(select_rows(index_next, pos), b1)
            # the read is per block, but the SPEC is the curve's
            raw = s0.project(b0, s0.read_at(tenor_data, rows, t1, t2, weight))
            if alpha is not None:
                # projection and the time blend are both linear, and `combine` runs after both,
                # so the routed path is the same arithmetic in the same order as an unrouted one
                raw = s0.blend(raw, s1.project(b1, s1.read_at(tenor_data, nxt, t1, t2, weight)),
                               select_rows(alpha, pos))
            return s0.combine(raw, tenor_data, t2, select_rows(tnr, pos), time_factor)

        return self.routed(route, (i1 if index is None else index).shape[0], group)

    def gather_rows(self, index, index_next, alpha, route):
        """Whole-row gather — the 0D spot path. The same block routing as `eval`, on the
        scenario-row axis rather than the flattened (row, tenor) one."""
        def group(pos, at_t, at_t1):
            b0, b1 = self.blocks[at_t], self.blocks[at_t1]
            if alpha is None:
                return b0.project(
                    self.strategies[at_t].tensor[self.local(select_rows(index, pos), b0)])
            a = select_rows(alpha, pos)
            return b0.project(
                self.strategies[at_t].tensor[self.local(select_rows(index, pos), b0)]) * (1 - a) + \
                b1.project(
                    self.strategies[at_t1].tensor[
                        self.local(select_rows(index_next, pos), b1)]) * a

        return self.routed(route, index.shape[0], group)


def build_interpolation(value, curve_tenor):
    """The one constructor for a curve's interpolation, recursive in the scenario axis.

        bare tensor    + a kind string  -> Interpolation
        bare tensor    + a segment list -> SegmentedInterpolation
        ScenarioSource + either         -> RoutedInterpolation, whose per-block children are built
                                           by this function again

    So a segmented curve inside an inner-MC fork composes rather than special-cases, and the
    ordinary path builds exactly the leaf it always did."""
    if isinstance(value, ScenarioSource):
        return RoutedInterpolation(value, curve_tenor)
    if isinstance(curve_tenor.type, str):
        return Interpolation.build(value, curve_tenor.type, curve_tenor.tenor)
    return SegmentedInterpolation(value, curve_tenor.type, curve_tenor.tenor)


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
            if not self.tensor_cache:
                self.tensor_cache['tenor'] = tenor_points_in_years.new(self.tenor)
                self.tensor_cache['delta'] = tenor_points_in_years.new(self.delta)
            tenor = self.tensor_cache['tenor']
            delta = self.tensor_cache['delta']
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
                 nomodel: str, simulation_batch: int, keep_tensor: bool):
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
        # keep individual calculation results per dependency?
        self.keep_tensor = keep_tensor


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

    def copy_restricted(self, cutoff_mtm_index):
        """Fresh DealTimeDependencies covering only deal events at mtm positions
        >= cutoff_mtm_index. delta/interp/indices/alpha are recomputed via __init__
        for the sliced view, so the post-pricing interpolate path produces output
        aligned with mtm_time_grid (terminal lands at the deal's expiry position).
        Returns None if all events are past the cutoff."""
        keep = self.deal_time_grid >= cutoff_mtm_index
        if not keep.any():
            return None
        return type(self)(self.mtm_time_grid, self.deal_time_grid[keep])

    def copy_window(self, from_mtm_index, to_mtm_index):
        """Fresh DealTimeDependencies covering only deal events at mtm positions in
        [from_mtm_index, to_mtm_index] — the one-step inner-MC fork prices at exactly
        {t, t+1}, so the AAD tape and the scenario buffer stop at t+1 (interp is rebuilt
        up to the last kept event; nothing downstream indexes past it). Assumes
        hedge-mode deals reval on every mtm date (dense event grids). Returns None if
        no event falls inside the window (deal expired before the fork)."""
        keep = (self.deal_time_grid >= from_mtm_index) & (self.deal_time_grid <= to_mtm_index)
        if not keep.any():
            return None
        return type(self)(self.mtm_time_grid, self.deal_time_grid[keep])

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

    def truncate_to(self, original_base_date, t_days):
        """Return a new TimeGrid covering [t_days, T] of the original, with base shifted
        forward by t_days. Used by nested-simulation drivers (inner MC) to construct a
        truncated horizon starting at an outer timestep. `original_base_date` is the
        base date this grid was originally set against (caller's responsibility — TimeGrid
        does not store its base date)."""
        new_base_date = original_base_date + pd.Timedelta(days=int(t_days))
        new_scenario_dates = [d for d in sorted(self.scenario_dates) if d >= new_base_date]
        new_mtm_dates = [d for d in sorted(self.mtm_dates) if d >= new_base_date]
        new_base_mtm = [d for d in self.base_MTM_dates if d in new_mtm_dates]
        new_grid = TimeGrid(new_scenario_dates, new_mtm_dates, new_base_mtm)
        if new_scenario_dates:
            new_grid.set_base_date(new_base_date)
        else:
            # Past-end caller's grid: keep `scen_time_grid` queryable (empty) so size
            # checks like `grid.scen_time_grid.size < 2` work without AttributeError.
            new_grid.scen_time_grid = np.array([], dtype=np.int64)
            new_grid.time_grid_years = np.array([], dtype=np.float64)
        return new_grid

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

    def total_abs_nominal(self):
        """Summed |notional| across the schedule."""
        return float(np.abs(self.schedule[:, CASHFLOW_INDEX_Nominal]).sum())

    def last_pay_day(self):
        """Latest payment day (offset in days from base_date)."""
        return float(self.schedule[:, CASHFLOW_INDEX_Pay_Day].max())

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


def split_array(array, counts):
    """`split_tensor` on the numpy side — keeps a CurveTensor's CPU scenario indices in step with
    its device ones, so a per-deal slice re-derives its row routing without a device sync."""
    return np.split(array, counts.cumsum()[:-1]) if array.shape[0] == counts.sum() \
        else [array] * counts.size


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

    def __init__(self, interp_obj, index, alpha, np_index=None):
        self.interp_obj = interp_obj
        self.np_index = index if isinstance(index, np.ndarray) else np_index
        self.index = torch.tensor(
            index, dtype=torch.int64, device=interp_obj.tensor.device) if isinstance(index, np.ndarray) else index
        # SCENARIO ROWS, not a flattened (row, tenor) offset: the strategy owns that flattening
        # because a tenor SEGMENT has its own stride, and a segment inside an inner-MC fork would
        # otherwise need a different flat index from the one its sibling block uses.
        if alpha is not None:
            self.alpha = self.interp_obj.tensor.new(alpha) if isinstance(alpha, np.ndarray) else alpha
            self.index_next = (self.index + 1).clamp(0, self.interp_obj.shape[0] - 1)
        else:
            self.alpha = self.index_next = None
        # A curve every one of whose rows is row 0 (a static factor, or a stochastic one gathered
        # only at the base date) skips the flattening add entirely. Decided off the NUMPY indices,
        # so asking costs no device sync.
        self.rows = None if not self.np_index.any() and self.alpha is None else self.index
        # Which of the source's row blocks owns each row this gather reads — also decided off the
        # CPU-side indices, once per CurveTensor rather than per gather. A leaf answers with its
        # whole-grid group.
        self.route = self.interp_obj.route(self.np_index, self.alpha is not None)

    def interp_value(self):
        return self.interp_obj.gather_rows(self.index, self.index_next, self.alpha, self.route)

    def split(self, counts):
        sub_alpha = split_tensor(self.alpha, counts) if self.alpha is not None else [None] * counts.size
        sub_index = split_tensor(self.index, counts)
        return [CurveTensor(self.interp_obj, sub_index, sub_alpha, np_index=sub_np)
                for sub_index, sub_alpha, sub_np in
                zip(sub_index, sub_alpha, split_array(self.np_index, counts))]

    def interpolate_risk_neutral(self, curve_component, points, time_grid, time_multiplier):
        t = time_grid[:, 1].reshape(-1, 1)
        T = points + t
        return self.interpolate_curve(
            curve_component, T, time_multiplier) - self.interpolate_curve(
            curve_component, t, time_multiplier)

    def interpolate_curve(self, curve_component, points, time_factor):
        # our tensor object
        tensor = self.interp_obj.indexed_tensor
        # check the points being queried
        time_size, point_size = points.shape

        if point_size > 0:
            # get the points in years
            tenor_points_in_years = tensor.new(curve_component[FACTOR_INDEX_Daycount](points))
            curve_tenor = curve_component[FACTOR_INDEX_Tenor_Index]
            i1, i2, a = curve_tenor.get_index(tenor_points_in_years)

            if isinstance(curve_tenor.type, str):
                tenor_data = (curve_tenor.type, curve_tenor.min, curve_tenor.max)
            else:
                split_tenor = curve_tenor.tenor[curve_tenor.type[0][1]]
                tenor_data = (curve_tenor.type, (curve_tenor.min, split_tenor),
                              (split_tenor, curve_tenor.max))

            return self.interp_obj.eval(
                tenor_data, self.rows, self.index_next, self.alpha, i1, i2, a.unsqueeze(dim=-1),
                tenor_points_in_years, time_factor, route=self.route)
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
        
        local_cache_key = (end_points.shape, end_points.tobytes(),
                           (start_points.shape, start_points.tobytes()) if start_points is not None else None,
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


class DerivedForwardCurve(object):
    '''
    A forward curve reconstructed from simulated components - F(t,T) = S(t) exp(c(T)(T-t) + r(t,T)(T-t))
    where S is a spot price tensor (time, batch), c is a carry TensorBlock quoted at absolute (excel date)
    tenors and r is a repo/funding TensorBlock quoted at relative year tenors. t_excel maps each time row
    to its excel date offset so gathers take the same absolute-date end_points as a ForwardPrice factor.
    Duck-types the TensorBlock surface used by curve pricing (gather_weighted_curve, split_counts, time_grid).
    Note that F(t,t) = S(t) exactly.
    '''

    def __init__(self, spot, carry, repo, t_excel, time_grid):
        self.spot = spot
        self.carry = carry
        self.repo = repo
        self.t_excel = t_excel
        self.time_grid = time_grid

    def split_counts(self, counts, shared):
        cum_counts = counts.cumsum()
        return [DerivedForwardCurve(*sub_block) for sub_block in zip(
            torch.split(self.spot, tuple(counts)),
            self.carry.split_counts(counts, shared), self.repo.split_counts(counts, shared),
            np.split(self.t_excel, cum_counts), np.split(self.time_grid, cum_counts))]

    def gather_weighted_curve(self, shared, end_points, start_points=None, multiply_by_time=False):
        tenor_in_days = end_points - self.t_excel.reshape(-1, 1)
        cost_of_carry = self.carry.gather_weighted_curve(
            shared, end_points, multiply_by_time=False) * self.spot.new_tensor(
            tenor_in_days / DAYS_IN_YEAR).unsqueeze(-1) + self.repo.gather_weighted_curve(shared, tenor_in_days)
        return self.spot.unsqueeze(1) * torch.exp(cost_of_carry)


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
            S_vals[start_time: end_time] += delta * (S_ti[start_time: end_time] - S_ti[start_time])
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
        S_j = [S_vals_0]
        start = 0

        for k, v in CDS_rates.items():
            end = v[1] + 1
            delta_j = scipy.optimize.brentq(
                lambda x: calc_par_cds(S_vals_0, k, delta=x, start_time=start, end_time=end)[0] - (v[0] + bump), -0.1,
                0.1)

            S_j.append(S_vals_0.copy())
            S_j[-1][start: end] += delta_j * (S_ti[start: end] - S_ti[start])
            start = v[1]

        return {k: v[0] for k, v in CDS_rates.items()}, S_ti, S_j
    else:
        return {k: v[0] for k, v in CDS_rates.items()}


def calc_par_cds(R, D, f, S_ti, S_j, tau, delta=0.0, start_time=None, end_time=None):
    if delta:
        S_vals = S_j.copy()
        S_vals[start_time: end_time] += delta * S_ti[start_time: end_time]
    else:
        S_vals = S_j

    h = (S_vals[1:] - S_vals[:-1]) / (S_ti[1:] - S_ti[:-1])
    S = np.exp(-S_vals)
    F = D * S
    V_prot = ((F[:-1] - F[1:]) * h) / (h + f)

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
    return v_prot / v_fee


def index_cds_par_spread(
    H0_names, tau, D, R, f, S_ti, hazard_scale, eps=1e-14
):
    H = hazard_scale * H0_names                 # (N,M)
    N, M = H.shape

    dt = S_ti[1:] - S_ti[:-1]                   # (M-1,)
    h = (H[:, 1:] - H[:, :-1]) / dt             # (N,M-1)

    S = np.exp(-H)                               # (N,M)
    F = S * D[None, :]                           # (N,M)

    hp = h + f[None, :]
    hp = np.where(np.abs(hp) < eps, np.sign(hp) * eps + eps, hp)

    V_prot = ((F[:, :-1] - F[:, 1:]) * h) / hp   # (N,M-1)

    alpha = tau[1:] - tau[:-1]
    n = S_ti.searchsorted(tau[1:])               # match calc_par_cds
    # Optional: assert tau points are on-grid (safer)
    # (need to check indices bounds first)
    if np.any(n >= len(S_ti)):
        raise ValueError("tau contains points beyond S_ti range")
    if not np.all(S_ti[n] == tau[1:]):
        raise ValueError("tau[1:] must be exact grid points in S_ti")

    v_fee = -tau[0] * N
    prev_n = 0

    for alpha_j, prev_tau, n_j in zip(alpha, tau[:-1], n):
        sub_i = slice(prev_n, n_j)
        sub_i_p1 = slice(prev_n + 1, n_j + 1)

        hp_seg = h[:, sub_i] + f[sub_i][None, :]
        hp_seg = np.where(np.abs(hp_seg) < eps, np.sign(hp_seg) * eps + eps, hp_seg)

        term0 = (1.0 + hp_seg * (S_ti[sub_i][None, :] - prev_tau)) * F[:, sub_i]
        term1 = (1.0 + hp_seg * (S_ti[sub_i_p1][None, :] - prev_tau)) * F[:, sub_i_p1]
        A_j = (term0 - term1) * h[:, sub_i] / (hp_seg ** 2)

        v_fee += alpha_j * np.sum(D[n_j] * S[:, n_j]) + np.sum(A_j)
        prev_n = n_j

    n_last = n[-1]
    v_prot_total = (1.0 - R) * np.sum(V_prot[:, :n_last])

    return v_prot_total / v_fee


def calibrate_index_hazard_scale(
        base_date,
        index,
        curves,
        discount,
        maturity,
        b_lo = 0.05,
        b_hi = 5.0,
        tol = 1e-10,
        max_iter = 100
    ):

    def func(b: float) -> float:
        s = index_cds_par_spread(
            H0, tau, D, R, f, S_ti, b)
        return s - target

    index_factor = index.factor if hasattr(index, 'factor') else index
    discount_factor = discount.factor if hasattr(discount, 'factor') else discount
    pay_times = cds_dates(base_date, int((maturity-base_date).days/365+.5) * 12)
    tau = np.array([index_factor.get_day_count_accrual(base_date, (x - base_date).days) for x in pay_times])

    # calculate the piecewise hazard rate, forward rate and survival and discount curves
    S_ti = np.union1d(index_factor.get_tenor().clip(max=max(tau)), tau)
    D_vals = discount_factor.current_value(S_ti) * S_ti
    f = (D_vals[1:] - D_vals[:-1]) / (S_ti[1:] - S_ti[:-1])
    D = np.exp(-D_vals)
    R = index_factor.recovery_rate()
    S_vals_0 = index_factor.current_value(S_ti)

    # the target is the index
    target = calc_par_cds(R, D, f, S_ti, S_vals_0, tau)
    # all the cumulative hazard rates at time 0
    H0 = np.array([cv.current_value(S_ti) for cv in curves])

    flo = func(b_lo)
    fhi = func(b_hi)
    if flo == 0.0:
        return float(b_lo)
    if fhi == 0.0:
        return float(b_hi)
    if flo * fhi > 0:
        raise ValueError(
            f"Root not bracketed for b in [{b_lo},{b_hi}]. "
            f"f(lo)={flo:.6g}, f(hi)={fhi:.6g}. "
            f"Try widening bracket or check index_spread_running."
        )

    return scipy.optimize.brentq(func, b_lo, b_hi)

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


def norm_pdf(x):
    return 0.3989422804014327 * torch.exp(-0.5 * x * x)


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


# ======================================================================================
# Characteristic-function (Fourier) inversion primitive for affine option pricers
# ======================================================================================
#
# A MODEL-AGNOSTIC European vanilla / digital pricer for any model whose aggregate
# log-return R = log(S_T/S_t) has a known characteristic function.  The quadrature
# machinery lives here once; a model supplies ONLY its log-CF and reuses everything below
# with zero inversion code.  Heston-Nandi is the first client (the HN section immediately
# below); Heston, Bates and Variance-Gamma would each just pass their own ``logcf`` closure
# (see the ``cf_european_probabilities`` documentation for the plug-in contract).
#
# Float64 is mandatory: the S*P1 - K*e^{-rn}*P2 assembly is a cancellation of two O(1)
# probabilities and float32 destroys it.

def gauss_legendre(a, b, panels, order=8, dtype=torch.float64, device='cpu'):
    """Composite Gauss-Legendre nodes/weights on [a, b], ASCENDING, endpoints excluded.

    ``panels`` sub-intervals each carry an ``order``-point rule; because the panel edges
    (hence the interval endpoints ``a`` and ``b``) are never sampled, an integrand with a
    removable singularity at an endpoint - e.g. the ``1/(i*phi)`` of a Fourier inversion at
    ``phi = 0`` - can be integrated on ``[0, phi_max]`` directly without a hole.
    """
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    nodes = (mid + half * x[None, :]).ravel()
    wts = (half * w[None, :]).ravel()
    o = np.argsort(nodes)
    return (torch.tensor(nodes[o], dtype=dtype, device=device),
            torch.tensor(wts[o], dtype=dtype, device=device))


def complex_log_unwrap(w, dim=-1):
    """Complex log with the branch fixed by continuity ALONG ``dim`` (the phi grid).

    ``dim`` must be an axis along which phi varies smoothly and monotonically, anchored at
    its first entry (smallest phi, where ``w`` is near ``1+0j``).  This is the general guard
    against the discrete "Heston trap": taking the principal branch of ``log(1 - 2*alpha*B)``
    independently at each backward step of an affine A/B recursion is wrong whenever that
    argument winds around the origin.  ``torch.round`` carries zero gradient, which is
    correct because the winding correction is a locally-constant integer.  Reduces to the
    principal branch when the size along ``dim`` is 1.
    """
    two_pi = 2.0 * np.pi
    ang = torch.angle(w)
    if w.shape[dim] > 1:
        d = torch.diff(ang, dim=dim)
        d = d - two_pi * torch.round(d / two_pi)
        first = ang.narrow(dim, 0, 1)
        ang = torch.cat([first, first + torch.cumsum(d, dim=dim)], dim=dim)
    return torch.complex(torch.log(torch.abs(w)), ang)


def cf_adaptive_phi_max(logcf, carry, dtype=torch.float64, device='cpu',
                        log_tol=-40.0, start=8.0, cap=2.0 ** 24):
    """Smallest power-of-two ``phi_max`` at which the inversion integrand has decayed.

    Doubles ``phi`` until ``Re(logcf) - ln(phi)`` drops below ``log_tol`` on BOTH inversion
    contours (``i*phi`` and ``i*phi + 1``), the +1 (share-measure) contour being normalised
    by the log forward-growth ``carry`` = log E[S_T/S_t].  ``logcf`` must already be reduced
    to the SLOWEST-DECAYING state in the batch (for a stochastic-variance model that is the
    smallest instantaneous variance, whose integrand envelope decays slowest by Jensen), so
    the scan stays cheap - one recursion per doubling on a 2-element phi.  A closed-form
    cutoff is wrong here because the envelope decays slower than the pure-Gaussian
    ``exp(-phi^2 V/2)``.  Runs under ``no_grad`` (the result is a scalar quadrature bound).
    """
    with torch.no_grad():
        phi = float(start)
        while phi < cap:
            z = torch.tensor([phi], dtype=dtype, device=device) * 1j
            m0 = logcf(z).real
            m1 = logcf(z + 1.0).real - carry
            if float(torch.maximum(m0, m1).max()) - np.log(phi) < log_tol:
                return phi
            phi *= 2.0
        return phi


def cf_european_probabilities(logcf, log_moneyness, carry, phi_max, panels=256, order=8,
                              dtype=torch.float64, device='cpu', want=3):
    """The two exercise probabilities P1, P2 of a European claim, by Fourier inversion.

    MODEL-AGNOSTIC.  Given a model whose aggregate log-return ``R = log(S_T/S_t)`` has the
    generalised (complex-phi) characteristic function ``E_t[(S_T/S_t)^phi] = exp(logcf(phi))``,

        P2 = 1/2 + (1/pi) Int_0^inf Re[ e^{-i phi m} exp(logcf(i phi))            / (i phi) ] d phi
        P1 = 1/2 + (1/pi) Int_0^inf Re[ e^{-i phi m} exp(logcf(i phi + 1) - carry)/ (i phi) ] d phi

    with ``m = log_moneyness = ln(K/S)`` and ``carry = ln E_t[S_T/S_t]`` (= r*n under the
    risk-neutral measure, r the per-step rate).  Then a vanilla is priced by the caller as
    ``S*P1 - K*e^{-carry}*P2`` (call), and a digital / CDF by ``Q(R <= b) = 1 - P2`` evaluated
    at ``m = b`` (spot-free by construction).  ``want`` is a bit mask: 1 = P1, 2 = P2, 3 =
    both; a CDF (``want = 2``) is exactly half the cost of a price.

    THE PLUG-IN CONTRACT.  ``logcf(phi)`` receives a complex tensor whose trailing axis is
    the quadrature grid and must return ``log E_t[(S_T/S_t)^phi]`` broadcasting to
    ``(batch, node)`` - the state (e.g. the instantaneous variance) is captured by the
    closure, so a single strike-vector priced against one variance is one ``batch`` row.
    For an AFFINE model that log-CF is ``A(phi) + B(phi)*V_t`` with ``(A, B)`` from the
    model's own backward recursion (use :func:`complex_log_unwrap` inside that recursion for
    the branch of any ``log(1 - ...)`` term); a Levy model returns ``A(phi)`` alone.  A
    future Heston / Bates / VG model therefore adds ZERO inversion code: it supplies its
    ``logcf`` closure and ``carry``, resolves ``phi_max`` via :func:`cf_adaptive_phi_max`
    (feeding it the same closure reduced to the worst-case state), and calls this function.

    Differentiable w.r.t. every leaf reachable through ``logcf`` and ``carry`` (spot, strike
    and the model parameters); float64 is required for the P1-P2 cancellation.
    """
    lm = log_moneyness.unsqueeze(-1)
    nodes, wts = gauss_legendre(0.0, phi_max, panels, order, dtype, device)
    iphi = nodes * 1j
    shift = torch.exp(-1j * nodes * lm) / iphi                # K^{-i phi} S^{i phi} / (i phi)
    out = []
    for bit, off, disc in ((1, 1.0, carry), (2, 0.0, 0.0)):
        if not (want & bit):
            out.append(None)
            continue
        d = (shift * torch.exp(logcf(iphi + off) - disc)).real
        out.append(0.5 + (d * wts).sum(-1) / np.pi)
    return out[0], out[1]


# ======================================================================================
# Heston-Nandi GARCH(1,1): params + A/B recursion + daily-step recursion + semi-analytic pricing.
# House style (mirrors black_european_option_price / Bjerksund_Stensland / hn_variance_step): the
# math is FREE FUNCTIONS taking the GARCH params as explicit trailing args (omega, alpha, beta,
# gamma_star, r); each consumer unpacks its name->tensor mapping (the HestonNandiModelParameters
# factor block / t_Static_Buffer) into those args by the canonical names below - no params class.
# Plugs into the model-agnostic CF machinery above. Theory/conventions: the harvested
# HestonNandiImpliedSpotModel.documentation (stochasticprocess) and tests/test_hn_garch.py.

# The HestonNandiModelParameters price factor's parameters, in canonical order - the SINGLE source
# of that name set, shared with riskfactors.HestonNandiModelParameters.parameters (that class
# derives its list from here; the dependency edge only goes DOWN - utils never imports riskfactors).
# Every HN consumption site unpacks a name->tensor mapping into the explicit function args by these
# names. NOTE: ``r`` (the per-step cost of carry) and the per-call H0 (the variance STATE seeding
# h_1) are NOT factor parameters - only these five scalars are.
HN_PARAM_NAMES = ('Omega', 'Alpha', 'Beta', 'Gamma_Star', 'H0')


def hn_persistence(alpha, beta, gamma_star):
    """psi = beta + alpha * gamma*^2 (the GARCH persistence; must be < 1 for stationarity)."""
    return beta + alpha * gamma_star ** 2


def hn_stationary_var(omega, alpha, beta, gamma_star):
    """E[h] = (omega + alpha) / (1 - psi), the per-step stationary variance."""
    return (omega + alpha) / (1.0 - hn_persistence(alpha, beta, gamma_star))


def hn_ann_vol(omega, alpha, beta, gamma_star, steps_per_year=252.0):
    """Long-run annualised vol sqrt(E[h] * steps_per_year); float or tensor per the inputs."""
    v = hn_stationary_var(omega, alpha, beta, gamma_star) * steps_per_year
    return float(v) ** 0.5 if not torch.is_tensor(v) else v.sqrt()


def hn_ab(phi, n_steps, omega, alpha, beta, gamma_star, r, unwrap=True, phi_dim=-1, theta=None):
    """Backward A/B recursion for ``n_steps`` steps.  Returns ``(A, B)``.

    ``phi`` : real OR complex tensor.  If complex it is assumed to vary smoothly and ascending
              along ``phi_dim`` (needed for the branch unwrap).
    Result satisfies E_t[S_{t+n}^phi] = S_t^phi * exp(A + B * h_{t+1}); i.e. the HN affine log-CF
    of the aggregate log-return is ``A + B * h1`` (the closure handed to the model-agnostic
    inversion primitive :func:`cf_european_probabilities`).

    ``theta`` seeds the recursion at B = theta instead of 0, which makes the result the JOINT
    transform E_t[exp(phi*R_n + theta*h_{t+n+1})] - same recursion, one different initial
    condition.  Differentiating in theta is how the terminal variance's moments (and its
    covariance with the aggregate return) come out exactly; see :func:`hn_aggregate_moments`.
    """
    A = torch.zeros_like(phi)
    B = torch.zeros_like(phi) if theta is None else theta * torch.ones_like(phi)
    lin = phi * (gamma_star - 0.5) - 0.5 * gamma_star ** 2   # <-- the -phi/2 is the LRNVR drift
    half_sq = 0.5 * (phi - gamma_star) ** 2
    phir = phi * r
    for _ in range(int(n_steps)):
        w = 1.0 - 2.0 * alpha * B
        logw = complex_log_unwrap(w, dim=phi_dim) if (unwrap and w.is_complex()) else torch.log(w)
        A = A + phir + B * omega - 0.5 * logw
        B = lin + beta * B + half_sq / w
    return A, B


def hn_logmgf(phi, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw):
    """log E_t[exp(phi * R_n)] where R_n = log(S_{t+n}/S_t).  = A + B*h1."""
    A, B = hn_ab(phi, n_steps, omega, alpha, beta, gamma_star, r, **kw)
    return A + B * h1


def auto_phi_max(n_steps, h1, omega, alpha, beta, gamma_star, r,
                 log_tol=-40.0, start=8.0, cap=2.0 ** 24):
    """Smallest power-of-two phi_max with Re(A + B*h1) - ln(phi) < log_tol.

    The HN glue for the model-agnostic scan :func:`cf_adaptive_phi_max`: it reduces the batch to
    the extreme h1 (the smallest, whose integrand decays slowest) so the scan runs on a 2-element
    phi and checks BOTH inversion contours (i*phi and i*phi+1, the latter normalised by the log
    forward-growth r*n).
    """
    h1t = torch.as_tensor(h1).detach()
    hs = torch.stack([h1t.min(), h1t.max()]).to(omega.dtype).reshape(-1, 1)
    carry = torch.as_tensor(r).detach() * int(n_steps)
    return cf_adaptive_phi_max(
        lambda z: hn_logmgf(z, n_steps, hs, omega, alpha, beta, gamma_star, r), carry,
        omega.dtype, omega.device, log_tol, start, cap)


def _p1_p2(logm, n_steps, h1, omega, alpha, beta, gamma_star, r,
           phi_max, panels, order, unwrap, want=3):
    """P1, P2 for log-moneyness ``logm`` = ln(K/S).  ``logm``/``h1`` broadcast together.

    Thin HN glue over the model-agnostic Fourier-inversion primitive
    :func:`cf_european_probabilities`: it hands over the HN affine log-CF ``A + B*h1`` (from
    :func:`hn_ab`) as the ``logcf`` closure and the log forward-growth ``r*n`` as the P1-contour
    normalisation.  ``want`` is a bit mask: 1 = P1, 2 = P2, 3 = both.
    """
    logm = torch.as_tensor(logm, dtype=omega.dtype, device=omega.device)
    h1 = torch.as_tensor(h1, dtype=omega.dtype, device=omega.device)
    logm, h1 = torch.broadcast_tensors(logm, h1)
    if phi_max is None:
        phi_max = auto_phi_max(n_steps, h1, omega, alpha, beta, gamma_star, r)
    if panels is None:
        panels = 256
    hh = h1.unsqueeze(-1)

    def logcf(phi):
        A, B = hn_ab(phi, n_steps, omega, alpha, beta, gamma_star, r, unwrap=unwrap)
        return A + B * hh

    return cf_european_probabilities(
        logcf, logm, r * n_steps, phi_max, panels, order, omega.dtype, omega.device, want)


def hn_call(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r,
            phi_max=None, panels=None, order=8, unwrap=True):
    """European CALL, ``n_steps`` steps to expiry, spot ``S``, strike ``K``.

    ``h1`` is the (predictable) variance of the FIRST step; ``r`` the PER-STEP cost of carry.
    Differentiable w.r.t. (omega, alpha, beta, gamma_star, r, h1, S, K).
    """
    S = torch.as_tensor(S, dtype=omega.dtype, device=omega.device)
    K = torch.as_tensor(K, dtype=omega.dtype, device=omega.device)
    P1, P2 = _p1_p2(torch.log(K / S), n_steps, h1, omega, alpha, beta, gamma_star, r,
                    phi_max, panels, order, unwrap)
    return S * P1 - K * torch.exp(-r * n_steps) * P2


def hn_put(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw):
    """European PUT.  By put-call parity off :func:`hn_call` (the parity residual of the inversion
    itself is tested separately via the phi=1 martingale identity)."""
    S = torch.as_tensor(S, dtype=omega.dtype, device=omega.device)
    K = torch.as_tensor(K, dtype=omega.dtype, device=omega.device)
    return (hn_call(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw)
            - S + K * torch.exp(-r * n_steps))


def hn_cdf_logret(x, n_steps, h1, omega, alpha, beta, gamma_star, r,
                  phi_max=None, panels=None, order=8, unwrap=True):
    """EXACT  Q( R_n <= x )  where R_n = log(S_{t+n}/S_t), by Fourier inversion.

    This is the quantity the one-step-survival loop needs for an UP barrier at S*exp(x)
    (survival = stay below).  Spot-free by construction.  ``x`` and ``h1`` broadcast together.
    """
    _, P2 = _p1_p2(x, n_steps, h1, omega, alpha, beta, gamma_star, r,
                   phi_max, panels, order, unwrap, want=2)
    return 1.0 - P2


# --------------------------------------------------------------------------------------
# The daily-step recursion -- ONE SOURCE OF TRUTH for the HN step
# --------------------------------------------------------------------------------------
#
# The predictable-variance recursion h_{t+1} = omega + beta*h_t + alpha*(z_t - gamma*sqrt(h_t))^2
# lives ONLY in ``hn_variance_step``.  Every consumer routes through it: the one-step-survival
# (OSS) Monte Carlo pricers in ``riskflow/pricing.py`` (which call ``hn_daily_advance`` /
# ``hn_unmonitored_substeps``), the ``HestonNandiImpliedSpotModel`` diffusion in
# ``riskflow/stochasticprocess.py``, AND the standalone daily simulator in
# ``tests/hn_reference.py``.  So a single mutation-kill matrix on the step covers every pricer.

def hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star):
    """The HN predictable-variance recursion h_{t+1} = omega + beta*h + alpha*(z - gamma*sqrt(h))^2.

    ``sh`` = sqrt(h) is passed in (the caller already needs it for the log-spot step), so the
    square root is computed exactly once.  All args broadcast on the simulation axis.
    """
    return omega + beta * h + alpha * (z - gamma_star * sh) ** 2


def hn_daily_advance(Sj, h, b_step, z, omega, alpha, beta, gamma_star):
    """One daily Heston-Nandi step under the risk-neutral (LRNVR) measure. Returns (Sj, h).

    Advances the log-spot by ``(b_step - 0.5*h) + sqrt(h)*z`` and recurses the predictable
    variance (via :func:`hn_variance_step`). ``z`` is either a fresh unconditional normal (an
    unmonitored sub-step) or the survival-truncated final draw of a monitored interval; in BOTH
    cases the recursion is fed the REALISED z (the survival-conditioned law - leverage-asymmetric
    under truncation, DO NOT 'fix' back to an unconditional draw, see the pv_MC_Tarf note).
    ``b_step`` is the per-step cost-of-carry (r-q). All args broadcast on the trailing sim axis.
    """
    sh = torch.sqrt(h)
    Sj = Sj * torch.exp((b_step - 0.5 * h) + sh * z)
    h = hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star)
    return Sj, h


def hn_log_substep(log_S, h, z, b_step, omega, alpha, beta, gamma_star):
    """One unmonitored HN day, accumulating the LOG increment: the same step as
    :func:`hn_daily_advance` with the exponential left to the caller.

    Kept separate because it is the chain the OSS pricers repeat n_sub times per fixing, and at
    their batch shapes it is bandwidth-bound - ~13 elementwise kernels over a multi-million
    element tensor, of which only the last two carry any state forward. Fused it is one kernel
    and 5.9x faster, bit-identical forward and gradient (the compile is lazy, ~0.4s once, and
    dynamic so a new batch shape does not retrace).
    """
    sh = torch.sqrt(h)
    return (log_S + (b_step - 0.5 * h) + sh * z,
            hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star))


hn_log_substep = torch.compile(hn_log_substep, dynamic=True)


def hn_aggregate_moments(n_steps, omega, alpha, beta, gamma_star):
    """Exact moments of the pair (aggregate log-return X, terminal variance h_end) over
    ``n_steps`` unmonitored HN days, as AFFINE coefficients in the entry variance h1.

    Returns ``(a, b)``, each a 7-entry 1-D tensor, with every moment recovered as
    ``a[i] + b[i] * h1``:

        [0 .. 3]  cumulants kappa_1 .. kappa_4 of X
        [4]       E[h_end]
        [5]       Var(h_end)
        [6]       Cov(X, h_end)

    HN is affine, so the joint transform is exp(A + B*h1) and every derivative at the origin
    splits the same way - which is what makes the sampler O(1) per path: these are SCALARS
    computed once per interval, then evaluated elementwise.  Taken by autodiff of the A/B
    recursion rather than hand algebra.

    The carry ``r`` is left out (set to 0): it enters ``hn_ab`` only as ``+phi*r`` per step, so
    it shifts kappa_1 by ``n_steps*r`` and touches nothing else.  The caller adds it, which lets
    a per-path carry ride a scalar recursion.

    Derivatives are taken by central finite-difference stencils on ONE vectorised ``hn_ab`` call
    rather than by nested autograd: ``create_graph`` to fourth order re-walks the n-step graph on
    every level and measured 816 ms at n=63, 17x the exact walk it exists to replace.  The
    stencil stays differentiable in the parameters (it is a linear combination of evaluations),
    so HN greeks still flow.

    TWO step sizes, because the errors pull opposite ways: truncation is O(delta^2) and dominates
    the low derivatives, while round-off enters kappa_4 as eps/delta^4 and explodes below
    delta~0.05 (measured: at delta=0.005 kappa_1 is exact to 1.3e-6 but kappa_4 is off by 330x).
    So kappa_1/kappa_2 and the covariance - which set the drift and the scale, and must be right
    - use the fine step, and kappa_3/kappa_4 - which only weight Cornish-Fisher correction terms
    - use the coarse one.
    """
    lo, hi = 0.005, 0.2
    h_scale = hn_stationary_var(omega, alpha, beta, gamma_star)
    e = 0.05 / h_scale                                          # theta scale: e*h_end ~ 0.05
    grid = ((0.0, 0.0), (-lo, 0.0), (lo, 0.0),
            (-2 * hi, 0.0), (-hi, 0.0), (hi, 0.0), (2 * hi, 0.0),
            (0.0, -e), (0.0, e), (lo, e), (lo, -e), (-lo, e), (-lo, -e))
    one = torch.ones_like(h_scale)
    A, B = hn_ab(torch.stack([x * one for x, _ in grid]), n_steps, omega, alpha, beta, gamma_star,
                 torch.zeros_like(h_scale), unwrap=False,
                 theta=torch.stack([y * one for _, y in grid]))

    def moments(f):
        z, ml, pl, m2, m1, p1, p2, tm, tp, pp, pm, mp, mm = f.unbind(0)
        return torch.stack([
            (pl - ml) / (2 * lo),                                       # kappa_1
            (pl - 2 * z + ml) / lo ** 2,                                # kappa_2
            (p2 - 2 * p1 + 2 * m1 - m2) / (2 * hi ** 3),                # kappa_3
            (p2 - 4 * p1 + 6 * z - 4 * m1 + m2) / hi ** 4,              # kappa_4
            (tp - tm) / (2 * e),                                        # E[h_end]
            (tp - 2 * z + tm) / e ** 2,                                 # Var(h_end)
            (pp - pm - mp + mm) / (4 * lo * e)])                        # Cov(X, h_end)

    return moments(A), moments(B)


def declared_spot(code, name):
    """Pass a resolved spot code through, saying ONCE whether it is simulated.

    A static spot is held flat across the whole time grid at pricing - legitimate, but it makes
    the exposure profile a deterministic forward, which is worth knowing before reading the
    numbers. Said here, in calc_dependencies, because it is a compile-time fact: the code tuple
    already carries the flag, and the alternative is a warning that repeats every batch.
    """
    if not code[0][FACTOR_INDEX_Stoch]:
        logging.warning('%s is not simulated - spot is held flat across the time grid',
                        check_tuple_name(code[0][FACTOR_INDEX_Offset]))
    return code


def spot_on_deal_grid(spot, deal_time, shared):
    """Give ``spot`` the shape every pricer assumes: (len(deal_time), n_scenarios).

    A SIMULATED spot already has it. A static one arrives as a single row and is tiled up.
    The test is therefore on ROWS - the axis actually being corrected. Testing columns instead
    (`spot.shape[1] != b.shape[1]`) reads a legitimate broadcast pair - a simulated spot with B
    columns against a static curve with 1 - as a defect, and then tiles the ROWS by
    len(deal_time), squaring the grid: 37 dates became 1369 and every barrier deal was silently
    skipped under credit Monte Carlo.
    """
    return spot if spot.shape[0] == len(deal_time) else spot.tile(
        len(deal_time), shared.simulation_batch)


def hn_cached_moments(shared, n_steps, omega, alpha, beta, gamma_star):
    """:func:`hn_aggregate_moments` memoised in ``shared.t_PreCalc`` on (n_steps, parameters).

    Skipped when the parameters carry gradients, because the key is parameter VALUES and under
    AAD that is not enough to identify the entry: two underlyings calibrated to the same numbers
    collide, and the second would be handed moments whose graph runs back to the FIRST one's
    leaves - a silent misattribution of its greeks. (Keying on tensor identity is not the fix
    either; id() is recycled after GC.) Reuse across batches is otherwise safe - the leaves are
    minted once per calculation and SensitivitiesEstimator retains the graph.
    """
    if omega.requires_grad:
        return hn_aggregate_moments(n_steps, omega, alpha, beta, gamma_star)
    key = ('hn_moments', n_steps, omega.item(), alpha.item(), beta.item(), gamma_star.item())
    if key not in shared.t_PreCalc:
        shared.t_PreCalc[key] = hn_aggregate_moments(n_steps, omega, alpha, beta, gamma_star)
    return shared.t_PreCalc[key]


def hn_quantile_table(n_steps, omega, alpha, beta, gamma_star, h_grid, n_u=192, n_sd=7.0):
    """EXACT inverse of the n-step aggregate return law, tabulated over (u, h1).

    HN is affine, so the law is a one-parameter family in the entry variance h1: tabulating it
    on a small h1 grid captures the whole family, and a path's draw is then a 2-D lookup.
    :func:`hn_cdf_logret` gives Q(R_n <= x) EXACTLY by Fourier inversion, so this carries no
    distributional approximation at all - unlike a moment expansion, it has no validity ceiling
    in the tail, which is precisely where an OSS barrier reads.

    Built by evaluating F on an x-grid and inverting by interpolation rather than root-finding:
    one batched CDF call per interval (45-125 ms, near-flat in grid size since the cost is the
    Fourier integration, not the grid). Returns ``(u_grid, x_table)`` with ``x_table[i, j]`` the
    return at probability ``u_grid[j]`` for ``h_grid[i]``.

    Float64 Fourier inversion leaves round-off wobble of order 1e-14 where F has saturated to
    1.0; a cumulative max makes the sequence usable for inversion without touching the body,
    which is strictly monotone as it stands.
    """
    a, b = hn_aggregate_moments(n_steps, omega, alpha, beta, gamma_star)
    mean = (a[0] + b[0] * h_grid).reshape(-1, 1)
    sd = (a[1] + b[1] * h_grid).clamp_min(0.0).sqrt().reshape(-1, 1)
    x = mean + sd * torch.linspace(-n_sd, n_sd, 4 * n_u, dtype=h_grid.dtype,
                                   device=h_grid.device).reshape(1, -1)
    F = hn_cdf_logret(x, n_steps, h_grid.reshape(-1, 1), omega, alpha, beta, gamma_star,
                      torch.zeros_like(omega)).cummax(dim=1).values
    # Tabulated against z = Phi^-1(u), NOT u: a uniform grid leaves the outer ~1% of draws off
    # the end of the table, where clamping truncates both tails and costs over a percent of
    # standard deviation. In z the far tail is near-linear, so the caller's draw is a normal and
    # anything past the grid extrapolates along the edge segment instead of flattening.
    z_grid = torch.linspace(-n_sd, n_sd, n_u, dtype=h_grid.dtype, device=h_grid.device)
    u = norm_cdf(z_grid).expand(F.shape[0], -1).contiguous()
    j = torch.searchsorted(F.contiguous(), u).clamp(1, F.shape[1] - 1)
    F0, F1 = F.gather(1, j - 1), F.gather(1, j)
    x0, x1 = x.gather(1, j - 1), x.gather(1, j)
    w = ((u - F0) / (F1 - F0).clamp_min(1e-300)).clamp(0.0, 1.0)
    return z_grid, x0 + w * (x1 - x0)


def hn_table_substeps(Sj, h, b_step, n_steps, hn_params, shared, num_sims, antithetic):
    """O(1) stand-in for :func:`hn_unmonitored_substeps` - same signature, same (Sj, h) contract.

    Nothing observes the interval, so only the joint law of (aggregate return, terminal variance)
    is needed, not the path. The aggregate is drawn from the EXACT tabulated inverse CDF, and the
    terminal variance from its exact regression on the realised aggregate plus a moment-matched
    residual - so the leverage correlation that carries vol clustering across the interval
    survives. The interval's h1 bracket comes from the previous interval's exact moments rather
    than a hardcoded span.

    Engaged where the calculation amortises a precalculation (see the dispatch in the pricers);
    a single valuation walks the interval exactly instead, because one table build costs more
    than the walk it would replace.
    """
    # An empty block carries no paths to draw for, and the interval's h range - which the table
    # is bracketed to - is undefined. The exact walk returns empty from empty by construction;
    # this has to say so, because min() over nothing raises rather than degenerating.
    if not n_steps or not h.numel():
        return Sj, h
    omega, alpha, beta, gamma_star = hn_params
    key = ('hn_qtable', n_steps, omega.item(), alpha.item(), beta.item(), gamma_star.item())
    if key not in shared.t_PreCalc:
        # The h grid is derived from the MODEL, not from the realised range of this call: the
        # entry variance differs at every block and batch, so bracketing it there gives every
        # call its own key, the cache never hits, and each pays a fresh Fourier inversion -
        # measured 7x SLOWER than the walk it replaces. Anchored on the stationary variance and
        # spanning four decades around it, the key is (n, parameters) alone and the table is
        # built once. omega is the model's own floor on h; the top is far past any realised path.
        lr = hn_stationary_var(omega, alpha, beta, gamma_star)
        h_grid = torch.logspace(float(torch.log10(lr)) - 2.0, float(torch.log10(lr)) + 2.0, 64,
                                dtype=h.dtype, device=h.device)
        shared.t_PreCalc[key] = (h_grid,) + hn_quantile_table(
            n_steps, omega, alpha, beta, gamma_star, h_grid)
    h_grid, z_grid, x_table = shared.t_PreCalc[key]

    zc = torch.randn([shared.simulation_batch, num_sims], dtype=shared.one.dtype,
                     device=shared.one.device)
    z = torch.cat([zc, -zc], dim=-1) if antithetic else zc
    x = interp2d_lookup(z, h, z_grid, h_grid, x_table) + n_steps * b_step

    a, b = hn_cached_moments(shared, n_steps, omega, alpha, beta, gamma_star)
    k1, k2 = a[0] + b[0] * h, a[1] + b[1] * h
    slope = (a[6] + b[6] * h) / k2
    resid = ((a[5] + b[5] * h) - slope * slope * k2).clamp_min(0.0)
    h_end = ((a[4] + b[4] * h) + slope * (x - n_steps * b_step - k1)
             + resid.sqrt() * torch.randn_like(z)).clamp_min(omega)
    return Sj * x.exp(), h_end


def interp2d_lookup(u, y, u_grid, y_grid, table):
    """Bilinear read of ``table[y, u]`` at scattered ``(u, y)``, both grids ascending.

    ``y`` is looked up in LOG space because the grids that need this - conditional variances -
    are built log-spaced, and is CLAMPED to the grid, which is safe only because the caller
    brackets it to the realised range. The ``u`` axis EXTRAPOLATES along the edge segment
    instead: it carries a normal quantile whose far tail is near-linear, and flattening it
    there would truncate the distribution - which is how a discretised scheme quietly biases
    its own tail.
    """
    ly, lg = y.log(), y_grid.log()
    i = torch.searchsorted(lg, ly.reshape(-1).contiguous()).clamp(1, lg.numel() - 1).reshape(y.shape)
    wy = ((ly - lg[i - 1]) / (lg[i] - lg[i - 1])).clamp(0.0, 1.0)
    j = torch.searchsorted(u_grid, u.reshape(-1).contiguous()).clamp(1, u_grid.numel() - 1).reshape(u.shape)
    wu = (u - u_grid[j - 1]) / (u_grid[j] - u_grid[j - 1])            # unclamped: extrapolates
    g = lambda ii, jj: table[ii.reshape(-1), jj.reshape(-1)].reshape(u.shape)
    lo = g(i - 1, j - 1) + wu * (g(i - 1, j) - g(i - 1, j - 1))
    hi = g(i, j - 1) + wu * (g(i, j) - g(i, j - 1))
    return lo + wy * (hi - lo)


def hn_unmonitored_substeps(Sj, h, b_step, n_steps, hn_params, shared, num_sims, antithetic):
    """Advance (Sj, h) through ``n_steps`` UNCONDITIONAL (unmonitored) daily HN steps. These carry
    no barrier - the OSS truncation applies only on the monitored final step (done by the caller).
    A monitored interval of n_sub days passes ``n_steps = n_sub - 1`` here; a non-monitored interval
    (e.g. the run from the last barrier date to expiry) passes the full ``n_steps = n_sub``. Fresh
    regular-stream normals per step (Sobol/antithetic variance reduction is reserved for the
    truncated final draw); with ``antithetic`` the normal is negated on the paired half (z, -z) to
    align with the u<->1-u halves of the final draw (TARF/barrier), otherwise a plain num_sims-wide
    normal (autocall, whose final draw is not antithetic). ``hn_params`` = (omega, alpha, beta,
    gamma_star).

    Nothing observes the spot between these steps - only the fixing does - so the walk runs in log
    space and exponentiates ONCE. The per-step exp/multiply round-trip was pure cost and the less
    accurate spelling of the same number.
    """
    if not n_steps:                                              # a daily fixing walks nothing
        return Sj, h
    log_S = torch.zeros_like(b_step)
    for _ in range(n_steps):
        zc = torch.randn([shared.simulation_batch, num_sims],
                         dtype=shared.one.dtype, device=shared.one.device)
        z = torch.cat([zc, -zc], dim=-1) if antithetic else zc
        log_S, h = hn_log_substep(log_S, h, z, b_step, *hn_params)
    return Sj * log_S.exp(), h


# --------------------------------------------------------------------------------------
# Correlated sub-stepping -- exact within-interval dynamics between coarse scenario nodes
# --------------------------------------------------------------------------------------
# A coarse exposure grid (PFE/CVA nodes weeks apart) still owes each factor the dynamics it
# would have had on the calibration clock: forwarding the variance deterministically and
# drawing one aggregate Gaussian was measured 29%->2000% wrong on tail probabilities at
# |z|=2-3 (gates/hn_aggregate_bias.csv) -- precisely the quantiles PFE reads.  Instead the
# interval walks its own sub-steps, and the framework's correlated draw enters as the
# sqrt(variance)-weighted combination of the sub-step normals, so cross-factor correlation
# rides the interval's dominant direction while the orthogonal complement supplies the
# within-interval vol-of-vol the mean bridge lost.  Freeze h and the aggregate return
# collapses back to sqrt(sum var)*z_fw -- the old bridge -- so this is its strict refinement.

def substep_schedule(f):
    """Trading-time lengths spanning each interval of `f` calibration steps: whole steps, then
    the fractional remainder.  A scenario grid is a CALENDAR object and the recursion is
    calibrated per trading day, so f is essentially never an integer (f = 252k/365.25 on a
    k-calendar-day step) -- rounding it to whole days makes node variance a step function of
    grid spacing, -13% on the framework's own default CVA grid.  len == 1 is the exact
    fractional step every fine grid already took; longer is a coarse-grid walk.
    """
    schedule = []
    for x in f:
        whole = int(x)
        rem = float(x) - whole
        steps = (1.0,) * whole + ((rem,) if rem > 1e-9 else ())
        schedule.append(steps or (float(x),))       # dt == 0 (the t=0 anchor) stays one null step
    return schedule


def substep_normals(sqrt_var, z_fw):
    """n iid N(0,1) sub-step draws Z whose weighted combination REPRODUCES the framework draw:
    w'Z = z_fw exactly, w = ``sqrt_var`` normalized along the leading (sub-step) axis.

    Z = e + w*(z_fw - w'e) with e fresh iid normals: Cov(Z) = (I - ww') + ww' = I given
    z_fw ~ N(0,1) independent of e (framework draws are marginally standard; e is drawn here),
    and w is F_t-measurable so this holds conditionally, per interval.  ``sqrt_var`` is
    (n, ...batch), ``z_fw`` (...batch).

    The weights decide only WHICH linear functional of the walk carries the cross-factor
    correlation -- every marginal is invariant to them.  sqrt of the mean-forwarded variance
    contribution E[h_j]*dt_j is the interval's own return loading, matching a correlated
    sibling that shares its variance profile (the second GARCH-family factor); a sibling with
    flat per-day variance would want uniform weights instead.  Neither dominates, so this is
    a modelling choice, not an exactness claim -- see test_weights_match_the_return_loading.
    """
    w = sqrt_var / (sqrt_var ** 2).sum(0, keepdim=True).sqrt()
    e = torch.randn_like(w)
    return e + w * (z_fw - (w * e).sum(0))


def hn_correlated_substeps(h, z_fw, sub_dt, omega, alpha, beta, gamma_star):
    """Walk one coarse scenario interval as the `sub_dt` fractional Heston-Nandi steps that
    span it.  Returns (h_end, var_sum, r_sum): terminal variance, realized integrated variance
    sum(h_j*dt_j) (the caller's -1/2 convexity drift), and the innovation sum(sqrt(h_j*dt_j)*z_j)
    -- so the interval return carry - var_sum/2 + r_sum is a price-martingale by iterated
    expectations, exact at every sub-step.  Each step is the same fractional recursion the fine
    grid takes (exactly `hn_variance_step` at dt=1), so the two branches agree in the limit.
    """
    psi = hn_persistence(alpha, beta, gamma_star)
    var_bar, mean = [], h
    for dt in sub_dt:                                        # E[h_{j+1}] = h + dt*(omega+alpha+psi*h - h)
        var_bar.append(mean * dt)
        mean = mean + dt * (omega + alpha + psi * mean - mean)
    z = substep_normals(torch.stack(var_bar).sqrt(), z_fw)
    var_sum, r_sum = torch.zeros_like(h), torch.zeros_like(h)
    for j, dt in enumerate(sub_dt):
        sh = h.sqrt()
        var_j = h * dt
        var_sum = var_sum + var_j
        r_sum = r_sum + var_j.sqrt() * z[j]
        h = h + dt * (hn_variance_step(h, sh, z[j], omega, alpha, beta, gamma_star) - h)
    return h, var_sum, r_sum


def garch_correlated_substeps(h, z_fw, sub_dt, omega, alpha, beta, nu):
    """Walk one coarse scenario interval as the `sub_dt` fractional GARCH(1,1)-t steps that span
    it.  Returns (h_end, var_sum, r_sum) with r_j = sqrt(h_j*dt_j)*eps_j: each eps_j is EXACTLY
    standardized Student-t, built by t-scaling the conditioned sub-step normals with fresh
    Gammas -- the same scale mixture GARCHSpotModel.generate uses per step, so the correlated
    draw rides the Gaussian kernel of the interval.  Same fractional recursion as the fine grid.
    """
    var_bar, mean = [], h
    for dt in sub_dt:                                        # E[h_{j+1}] adds alpha*E[r^2] = alpha*h*dt
        var_bar.append(mean * dt)
        mean = mean + dt * (omega - (1.0 - beta) * mean) + alpha * mean * dt
    z = substep_normals(torch.stack(var_bar).sqrt(), z_fw)
    W = torch.distributions.Gamma(nu / 2.0, 0.5).sample(z.shape).clamp_min(1.0e-6)
    eps = z * torch.sqrt(nu / W) * torch.sqrt((nu - 2.0).clamp_min(1.0e-3) / nu)
    var_sum, r_sum = torch.zeros_like(h), torch.zeros_like(h)
    for j, dt in enumerate(sub_dt):
        var_j = h * dt
        var_sum = var_sum + var_j
        r = var_j.sqrt() * eps[j]
        r_sum = r_sum + r
        h = h + dt * (omega - (1.0 - beta) * h) + alpha * r * r
    return h, var_sum, r_sum


# --------------------------------------------------------------------------------------
# Black-Scholes reference + HN implied vol (the HN smile/skew diagnostic and the bootstrapper seed)
# --------------------------------------------------------------------------------------
# bs_call_np is a thin ADAPTER over the canonical ``black_european_option_price`` (total-variance
# parameterisation); bs_implied_total_var bisects on it for the smile/skew diagnostics.

def bs_call_np(S, K, r, n, total_var):
    """BS call from TOTAL variance (r, n in per-step units) -- a thin adapter over the canonical
    ``black_european_option_price`` (F = S*e^{r*n}; vol=sqrt(tv), tenor=1 so stddev^2 = tv)."""
    return float(black_european_option_price(
        S * np.exp(r * n), K, r * n, np.sqrt(total_var), 1.0, 1.0, 1.0))



def bs_implied_total_var(price, S, K, r, n, lo=1e-12, hi=25.0, tol=1e-14, iters=200):
    """Bisection on TOTAL variance (no time units, so this is convention-free)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bs_call_np(S, K, r, n, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def hn_implied_vol(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r,
                   steps_per_year=252.0, **kw):
    """Annualised BS implied vol of the HN price (for the smile/skew diagnostics)."""
    c = float(hn_call(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw))
    tv = bs_implied_total_var(c, float(S), float(K), float(r), int(n_steps))
    return np.sqrt(tv / (int(n_steps) / steps_per_year))


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


def bachelier_european_option(F, X, vol, tenor, buyorsell, callorput, shared, cash_payoff=0.0, shift=0.0):
    # calculates the bachelier function WITHOUT discounting
    # shift is not used but needed to have the same sig as black_european_option

    if isinstance(tenor, float):
        guard = (vol > 0.0) & (tenor > 0.0)
        stddev = vol.clamp(min=1e-5) * np.sqrt(max(tenor, 0.0))
    else:
        tenor_np = tenor.clip(min=0.0)
        tau_key = ('tenor', tenor_np.shape, tenor_np.tobytes())
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

    # Degenerate case: zero stddev (should be guarded by clamp, but keep intrinsic as fallback)
    intrinsic = torch.relu(callorput * (F - X))

    if cash_payoff:
        # Digital option under Bachelier:
        # price = cash * N(callorput * d), d = (F - K) / stddev
        d = (F - X) / stddev
        prem = cash_payoff * norm_cdf(callorput * d)
        value = cash_payoff * (callorput * (F - X) > 0) * shared.one
    else:
        mu = callorput * (F - X)  # positive when option is in-the-money
        mu_per_sig = mu / stddev
        prem = mu * norm_cdf(mu_per_sig) + stddev * norm_pdf(mu_per_sig)
        value = intrinsic

    return buyorsell * torch.where(guard, prem, value)

def black_european_option(F, X, vol, tenor, buyorsell, callorput, shared, cash_payoff=0.0, shift=0.0):
    # calculates the black function WITHOUT discounting

    if isinstance(tenor, float):
        guard = (vol > 0.0) & (X > 0.0)
        stddev = vol.clamp(min=1e-5) * np.sqrt(tenor)
        strike = max(X, 1e-5) if isinstance(X, float) else X.clamp(min=1e-5)
    else:
        tenor_np = tenor.clip(min=0.0)
        tau_key = ('tenor', tenor_np.shape, tenor_np.tobytes())
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

    if isinstance(strike, float) and strike == 0 and not shift:
        # need to check if this is a put option (value is 0)
        # or a call option (value is just the forward)
        adjustment = 1.0 if callorput == 1.0 else 0.0
        prem = forward * adjustment
        value = forward * adjustment
    else:
        # handle shifted vol surfaces
        if shift:
            forward = torch.clamp(F, min=1e-5-shift) + shift
            strike = strike + shift
        d1 = torch.log(forward / strike) / stddev + 0.5 * stddev
        d2 = d1 - stddev
        if cash_payoff:
            prem = cash_payoff * norm_cdf(callorput * d2)
            value = cash_payoff * (callorput * (forward - strike) > 0) * shared.one
        else:
            prem = callorput * (forward * norm_cdf(callorput * d1) - strike * norm_cdf(callorput * d2))
            value = torch.relu(callorput * (forward - strike))
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
                if len(risk_factor.interpolation)>1:
                    interpolation_type = tuple([(x[0], x[1], x[2][0]) for x in risk_factor.interpolation])
                else:
                    interpolation_type = risk_factor.interpolation[0][0]
                tenor_data = tenor_diff(tenor_points, interpolation_type)
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
    alpha_shape = tuple([-1] + [1] * (len(interp_obj.shape) - 1))
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
    key_code = ('fxcross', rate1[0], rate2[0], time_grid[:, TIME_GRID_MTM].tobytes())
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
                time_grid[:, TIME_GRID_MTM].tobytes())

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
                time_grid[:, TIME_GRID_MTM].tobytes())
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

        if code[FACTOR_INDEX_SubType][0] == 'Malz':
            # interpolate along variance for term
            term_prior = flat_surface.new(expiry_tenor.tenor[index].reshape(-1, 1, 1))
            term_post = flat_surface.new(expiry_tenor.tenor[index_next].reshape(-1, 1, 1))
            t_expiry = flat_surface.new(expiry.clip(min=expiry_tenor.min).reshape(-1, 1))
            var_prior = term_prior * flat_surface.take(tenor_money_indices)**2
            var_post = term_post * flat_surface.take(tenor_money_indices_next)**2
            var_surface = time_modifier * torch.sum(
                var_prior * tenor_money_alpha * (1.0 - alpha) +
                var_post * tenor_money_alpha_next * alpha, dim=1)
            surface = torch.sqrt(var_surface/t_expiry)
        else:
            # interpolate along volatility
            surface = time_modifier * torch.sum(
                flat_surface.take(tenor_money_indices) * tenor_money_alpha * (1.0 - alpha) +
                flat_surface.take(tenor_money_indices_next) * tenor_money_alpha_next * alpha, dim=1)

        shared.t_Buffer[time_code] = (surface.reshape(-1), code, tenor_diff(new_moneyness_tenor))

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
        surface, rate_code, moneyness_tenor = shared.t_Buffer[key_code]
        max_index = np.prod(surface.shape) - 1
        if moneyness is None:
            moneyness = shared.one * (0.0 if rate_code[FACTOR_INDEX_SubType][0]=='Malz' else 0.0)
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

        shared.t_Buffer[key_code] = (flat_vol_time, code[0], code[0][FACTOR_INDEX_Moneyness_Index])

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
        shared.t_Buffer[key_code] = spread.reshape(-1, code[0][FACTOR_INDEX_Moneyness_Index].tenor.size)

    surface = shared.t_Buffer[key_code]
    result = []
    for exp, mon in zip(expiry, moneyness):
        time_exp = key_code[:-1] + tuple(exp)
        if time_exp not in shared.t_Buffer:
            flat_vol_time = gather_surface_interp(
                surface, code[0], exp, shared, calc_std).reshape(-1)
            shared.t_Buffer[time_exp] = (flat_vol_time, code[0], code[0][FACTOR_INDEX_Moneyness_Index])
        result.append(calc_moneyness_vol_rate(mon, exp, time_exp, shared))

    return torch.stack(result)


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


def make_curve_tensor(tensor, curve_component, time_grid, shared, n_batch_dims=1):
    # n_batch_dims > 1: the curve carries multiple trailing batch axes (e.g. a nested
    # inner-MC curve shaped (scen, n_tenors, B, B2)). Collapse them into ONE batch axis
    # up front so the rest of the curve stack — hermite params, the Interpolation
    # (scen*n_tenors, batch) indexing, gather_scenario_interp's rank-adaptive alpha —
    # stays rank-agnostic and unchanged. Default 1 preserves the legacy
    # (scen, n_tenors, B) single-batch path exactly. The caller reshapes the gathered
    # result's trailing batch axis back to (B, B2). The (B,B2) curve gathers all happen inside a
    # process's `generate`, i.e. BEFORE a fork publishes its block sequence, so a multi-block
    # source never reaches here — it carries no `reshape` and would say so.
    if n_batch_dims > 1:
        tensor = tensor.reshape(*tensor.shape[:-n_batch_dims], -1)
    curve_tenor = curve_component[FACTOR_INDEX_Tenor_Index]
    key_code = (curve_tenor.type, curve_component[:2], tuple(tensor.shape))

    if key_code not in shared.t_Buffer:
        # One recursive factory: a bare tensor becomes a leaf (or a tenor-segmented composite of
        # leaves), a fork's `ScenarioSource` becomes a scenario-routed composite whose per-block
        # children are built by the same call. Hermite coefficients are DEFERRED inside the leaf —
        # built by the first gather, for the rows that gather names.
        shared.t_Buffer[key_code] = build_interpolation(tensor, curve_tenor)

    if time_grid is not None:
        return gather_scenario_interp(shared.t_Buffer[key_code], time_grid, shared)
    else:
        return CurveTensor(shared.t_Buffer[key_code], np.zeros(1, dtype=np.int64), None)


def calc_time_grid_curve_rate(code, time_grid, shared, n_batch_dims=1):
    # n_batch_dims > 1: gather a curve whose simulated state carries extra trailing batch
    # axes (nested inner-MC: (scen, n_tenors, B, B2)). Threaded to make_curve_tensor, which
    # collapses them to one batch axis; the gathered result's trailing axis is then B*B2,
    # which the caller reshapes back. Default 1 = legacy single-batch behaviour, untouched.
    time_hash = time_grid[:, TIME_GRID_MTM].tobytes()
    code_hash = tuple(x[:2] for x in code)

    key_code = ('curve', code_hash, time_hash, n_batch_dims)

    if key_code not in shared.t_Buffer:
        value = []

        for rate in code:
            rate_code = ('curve_factor', rate[:2], time_hash, n_batch_dims)

            # check if the curve factors are already available
            if rate_code not in shared.t_Buffer:
                if rate[FACTOR_INDEX_Stoch]:
                    tensor = shared.t_Scenario_Buffer[rate[FACTOR_INDEX_Offset]]
                    spread = make_curve_tensor(tensor, rate, time_grid, shared, n_batch_dims=n_batch_dims)
                else:
                    # static curve: no scenario batch axes, n_batch_dims is irrelevant.
                    tensor = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                    spread = make_curve_tensor(tensor.reshape(1, -1, 1), rate, None, shared)

                # store it
                shared.t_Buffer[rate_code] = spread

            # append the curve and its (possible) interpolation parameters
            value.append(shared.t_Buffer[rate_code])

        shared.t_Buffer[key_code] = TensorBlock(code=code, tensors=value, time_grid=time_grid)

    return shared.t_Buffer[key_code]


def calc_time_grid_spot_rate(rate, time_grid, shared):
    # `rate` is a CODE (list of resolved factor indices), mirroring calc_time_grid_curve_rate:
    # element 0 is the primary spot; any tail elements are ObservedBasis components. The spot is
    # the SUM of the gathered components (composed spot = primary + basis), the get_* layer having
    # already turned the explicit deal fields into indices. A single-element code is the plain
    # spot — same ops in the same order as before, so bit-identical.
    key_code = ('spot', tuple(tuple(r[:2]) for r in rate), time_grid[:, TIME_GRID_MTM].tobytes())

    if key_code not in shared.t_Buffer:
        value = None
        for r in rate:
            if r[FACTOR_INDEX_Stoch]:
                tensor = shared.t_Scenario_Buffer[r[FACTOR_INDEX_Offset]]
                component = gather_scenario_interp(
                    build_interpolation(tensor, tenor_diff(np.zeros(1))),
                    time_grid, shared, as_curve_tensor=False)
            else:
                tensor = shared.t_Static_Buffer[r[FACTOR_INDEX_Offset]]
                component = tensor.reshape(1, -1)
            value = component if value is None else value + component

        shared.t_Buffer[key_code] = value

    return shared.t_Buffer[key_code]


def calc_curve_forwards(factor, tensor, time_grid_years, shared, mul_time=True):
    # `tensor` is the curve: (n_tenors,) calibrated, or (n_tenors, B) for a BATCH of per-path
    # curves. Every op below is elementwise or a tenor-axis gather, so the batch axis just rides
    # along as a trailing broadcast dim — no reduction reassociates and the batched result is
    # bitwise equal to looping the columns. `nb == 0` makes every `_bcast` a no-op reshape, so
    # the 1-D path executes exactly the arithmetic it always did.
    nb = tensor.dim() - 1

    def _bcast(x):
        """Right-pad tenor/time-shaped `x` with the curve's trailing batch axes."""
        return x.reshape(*x.shape, *([1] * nb))

    def prepare_tenors(factor_tenor, time_grid, extrapolate):
        """Prepare tenor grid with optional extrapolation."""
        tnr = factor_tenor.copy()
        amended_tensor = tensor
        if extrapolate:
            max_tenor = time_grid.max() + factor_tenor.max()
            tnr = np.append(tnr, max_tenor)
            # flat extrapolated gradient
            point_at_inf = tensor[-1:] + time_grid.max() * (tensor[-1:] - tensor[-2:-1]) / (
                    factor_tenor[-1] - factor_tenor[-2])
            amended_tensor = torch.cat([tensor, point_at_inf])

        tnr_d = np.diff(tnr, append=tnr.max() + 1)
        return tensor.new(tnr), tensor.new(tnr_d), amended_tensor

    def scale_for_rt(tnr, tensor, is_rt):
        """Scale tensor for rate*time interpolation."""
        if is_rt:
            return tensor * _bcast(tnr)
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

    def hermite_interpolation_new(tensor, tnr, is_rt, mul_time, full_tnr=None):

        def interp(values, indices_t):
            norm = _bcast(values) if mul_time else 1.0
            alpha, ten_t, ten_t_next = indices_t
            if is_rt:
                norm = norm / _bcast(values.clamp(full_tnr.min(), full_tnr.max()))
            return calc_hermite_curve(
                _bcast(alpha), g[ten_t], c[ten_t], tensor[ten_t], tensor[ten_t_next]) * norm

        """Handle Hermite interpolation variants."""
        t = tnr.view(1, -1, 1)
        if full_tnr is None:
            full_tnr = tnr
        # (1, n_tenors, 1) calibrated / (1, n_tenors, B) batched. Squeeze the leading axis only
        # when batched — a plain squeeze() would also eat the batch axis at B == 1.
        gc = hermite_interpolation_tensor(t, tensor.reshape(1, tensor.shape[0], -1))
        g, c = [x.squeeze(0) for x in gc] if nb else [torch.squeeze(x) for x in gc]

        return interp

    def linear_interpolation_new(tensor, tnr, is_rt, mul_time, extrapolate, full_tnr=None):

        def interp(values, indices_t):
            norm = _bcast(values) if mul_time else 1.0
            alpha, ten_t, ten_t_next = indices_t

            if is_rt:
                norm = norm / _bcast(values.clamp(full_tnr.min(), full_tnr.max()))
            alpha = _bcast(alpha)
            val = alpha * tensor[ten_t_next] + (1 - alpha) * tensor[ten_t]

            # `> 1 + nb` is the tenor axis test: it selects the (time x tenor) call and skips the
            # time-only one. Reduces to the original `len(val.shape) > 1` when nb == 0.
            if extrapolate and val.dim() > 1 + nb:
                val = val[:, :-1]
            return val * norm

        if extrapolate:
            tnr = tnr[:-1]

        if full_tnr is None:
            full_tnr = tnr

        return interp

    def calc_fwd_interpolated_new(method, l_tnr,  l_tensor, full_tnr=None):
        is_rt = method.endswith('RT')
        is_hermite = method.startswith('Hermite')
        is_linear = 'Linear' in method

        # Handle RT scaling
        tensor = scale_for_rt(l_tnr, l_tensor, is_rt)

        # Perform interpolation
        # basic idea - (tenor_pts+t)*f(tenor_pts+t) - t*f(t) for t in the time_grid
        if is_hermite:
            return hermite_interpolation_new(tensor, l_tnr, is_rt, mul_time, full_tnr=full_tnr)
        else:
            return linear_interpolation_new(tensor, l_tnr, is_rt, mul_time, extrapolate, full_tnr=full_tnr)

    # Preprocess tensors
    if len(factor.interpolation)>1:
        interp_method = [x[-1] for x in factor.interpolation]
        extrapolate = 'Extrapolate' in interp_method[-1][0]
    else:
        interp_method = factor.interpolation[0][0]
        extrapolate = 'Extrapolate' in interp_method

    factor_tenor = factor.get_tenor()
    time_grid = tensor.new(time_grid_years)
    tnr, tnr_d, tensor = prepare_tenors(factor_tenor, time_grid_years, extrapolate)
    # Calculate interpolation indices and weights
    indices_t, indices_time = calculate_interp_params(tnr, tnr_d, time_grid)

    """Compute interpolated forward curves with support for multiple interpolation methods."""
    # see if we have more than 1 interpolation object defined
    M = time_grid.view(-1, 1) + tnr.view(1, -1)
    if len(factor.interpolation)==1:
        f = calc_fwd_interpolated_new(interp_method, tnr, tensor)
        # insert the tenor axis: (T,) -> (T, 1) calibrated, (T, B) -> (T, 1, B) batched
        t_leg = f(time_grid, indices_time)
        return f(M, indices_t) - t_leg.reshape(t_leg.shape[0], 1, *t_leg.shape[1:])
    elif len(factor.interpolation)==2:
        cuttoff_index = factor.interpolation[0][1]
        cuttoff_tenor = tnr[cuttoff_index]
        if False:
            # near leg
            n = calc_fwd_interpolated_new(interp_method[0][0], tnr, tensor)
            near_tT = n(M, indices_t)
            near_t = n(time_grid, indices_time)
            # far leg
            f = calc_fwd_interpolated_new(interp_method[1][0], tnr, tensor)
            far_tT = f(M, indices_t)
            far_t = f(time_grid, indices_time)
            mask_near = M < cuttoff_tenor
            time_t = torch.where(time_grid < cuttoff_tenor, near_t, far_t)
        else:
            # near leg
            n = calc_fwd_interpolated_new(interp_method[0][0], tnr[:cuttoff_index+1], tensor[:cuttoff_index+1])
            near_tT = n(
                M,
                (indices_t[0],indices_t[1].clamp(max=cuttoff_index), indices_t[2].clamp(max=cuttoff_index)))
            near_t = n(
                time_grid,
                (indices_time[0], indices_time[1].clamp(max=cuttoff_index), indices_time[2].clamp(max=cuttoff_index)))
            # far leg
            f = calc_fwd_interpolated_new(interp_method[1][0], tnr[cuttoff_index:], tensor[cuttoff_index:])
            far_tT = f(
                M,
                (indices_t[0],(indices_t[1]-cuttoff_index).clamp(min=0), (indices_t[2]-cuttoff_index).clamp(min=0)))
            far_t = f(
                time_grid,
                (indices_time[0], (indices_time[1]-cuttoff_index).clamp(min=0), (indices_time[2]-cuttoff_index).clamp(min=0)))
            mask_near = _bcast(M <= cuttoff_tenor)
            time_t = torch.where(_bcast(time_grid <= cuttoff_tenor), near_t, far_t)
        return (torch.where(mask_near, near_tT, far_tT)
                - time_t.reshape(time_t.shape[0], 1, *time_t.shape[1:]))
    else:
        raise ValueError("More than 2 Interpolation Segments not supported")


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
    seen = set(adj[x])
    queue = deque(adj[x])
    while queue:
        i = queue.popleft()
        yield i
        for t in adj[i]:
            if t not in seen:
                seen.add(t)
                queue.append(t)


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


# 0D spot factor types whose NAME may carry a composed reference: a primary spot plus one or
# more ObservedBasis periods, positional like the InterestRate curve+basis parent chain
# (InterestRate.USD_SOFR.FUNDING; here CommodityPrice.PLATINUM_CME.LME_CME).
BASIS_COMPOSABLE_TYPES = ('FxRate', 'EquityPrice', 'CommodityPrice')


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
    cash = {}
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


def make_equity_swaplet_cashflows(base_date, time_grid, position, cashflows, current_spot, busday):
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
                         cashflow.get('Dividend_Multiplier', 1.0),
                         (cashflow['Start_Date'] + busday - base_date).days,
                         (cashflow['End_Date'] + busday - base_date).days])

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
    # calculate the business day ajustment on the mtm time grid
    bus_offset = np.array([((x + busday) - x).days for x in sorted(time_grid.mtm_dates)])
    return cashflows, bus_offset


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

            resets_in_excel_format = np.array([(x - reference.start_date).days for x in reset_dates])
            reference_date_excel = (reference_date - reference.start_date).days

            # retrieve the fixing dates from the reference curve and adding an offset
            fixing_dates = reference.get_fixings(resets_in_excel_format + forwardsample.param.get('Offset', 0))

            for reset_day, fixing_day in zip(resets_in_excel_format, fixing_dates):
                Reset_Day = reset_day - reference_date_excel
                # Start_Day = reset_day - reference_date_excel
                Start_Day = reset_day
                End_Day = fixing_day
                Weight = 1.0 / len(reset_dates)
                Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
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
