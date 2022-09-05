########################################################################
# Copyright (C)  Shuaib Osman (sosman@investec.co.za)
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
import logging
from collections import OrderedDict
from functools import reduce

# utility functions and constants
from . import utils, pricing

# specific modules
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def adjust_date(bus_day, modified, date):
    adj_date = bus_day.rollforward(date) if bus_day else date
    return bus_day.rollback(date) if (modified and adj_date.month != date.month) else adj_date


def generate_dates_backward(end_date, start_date, date_offset, bus_day=None, clip=True, modified=False):
    i, new_date = 1, end_date
    dates = [adjust_date(bus_day, modified, new_date)]
    date_kwds = date_offset.kwds.items()
    while new_date > start_date:
        period = pd.DateOffset(**{k: i * v for k, v in date_kwds})
        new_date = max(start_date, end_date - period) if clip else end_date - period
        dates.append(adjust_date(bus_day, modified, new_date))
        i += 1
    dates.reverse()
    return pd.DatetimeIndex(dates)


def generate_dates_forward(end_date, start_date, date_offset, bus_day=None, clip=True, modified=False):
    i, new_date = 1, start_date
    dates = [adjust_date(bus_day, modified, new_date)]
    date_kwds = date_offset.kwds.items()
    while new_date < end_date:
        period = pd.DateOffset(**{k: i * v for k, v in date_kwds})
        new_date = min(end_date, start_date + period) if clip else start_date + period
        dates.append(adjust_date(bus_day, modified, new_date))
        i += 1
    return pd.DatetimeIndex(dates)


def calc_factor_index(field, static_offsets, stochastic_offsets, all_tenors={}):
    """Utility function to determine if a factor is static or stochastic and returns its offset in the scenario block"""
    if static_offsets.get(field) is not None:
        return tuple([False, static_offsets[field]] + all_tenors.get(field, []))
    elif stochastic_offsets.get(field) is not None:
        return tuple([True, stochastic_offsets[field]] + all_tenors.get(field, []))
    else:
        raise Exception('Cannot find {}'.format(utils.check_tuple_name(field)))


def calc_factor_value(field, static_offsets, stochastic_offsets, all_factors):
    """Utility function to determine if a factor is static or stochastic and returns its offset in the scenario block"""
    if static_offsets.get(field) is not None:
        return all_factors[field].current_value()
    elif stochastic_offsets.get(field) is not None:
        return all_factors[field].factor.current_value()
    else:
        raise Exception('Cannot find value for {}'.format(utils.check_tuple_name(field)))


def get_recovery_rate(name, all_factors):
    """Read the Recovery Rate on a Survival Probability Price Factor"""
    survival_prob = all_factors.get(utils.Factor('SurvivalProb', name))
    return survival_prob.factor.recovery_rate() if hasattr(survival_prob, 'factor') else survival_prob.recovery_rate()


def get_interest_rate_currency(name, all_factors):
    """Read the Recovery Rate on a Survival Probability Price Factor"""
    ir_curve = all_factors.get(utils.Factor('InterestRate', name))
    return ir_curve.factor.get_currency() if hasattr(ir_curve, 'factor') else ir_curve.get_currency()


def get_inflation_index_name(fieldname, all_factors):
    """Read the Name of the Price Index price factor linked to this inflation index"""
    inflation = all_factors.get(utils.Factor('InflationRate', fieldname))
    return inflation.factor.param.get('Price_Index') if hasattr(inflation, 'factor') \
        else inflation.param.get('Price_Index')


def get_forwardprice_vol(fieldname, all_factors):
    """Read the Forward Price volatility factor linked to this Reference Vol"""
    pricevol = all_factors.get(utils.Factor('ReferenceVol', fieldname))
    return pricevol.get_forwardprice_vol()


def get_inflation_index_objects(inflation_name, index_name, all_factors):
    """Read the Name of the Price Index price factor linked to this inflation index"""
    inflation = all_factors.get(utils.Factor('InflationRate', inflation_name))
    inflation_factor = inflation.factor if hasattr(inflation, 'factor') else inflation
    index = all_factors.get(utils.Factor('PriceIndex', index_name))
    index_factor = index.factor if hasattr(index, 'factor') else index
    return inflation_factor, index_factor


def get_fxrate_factor(fieldname, static_offsets, stochastic_offsets):
    """Read the index of the FX rate price factor"""
    return [calc_factor_index(utils.Factor('FxRate', fieldname),
                              static_offsets, stochastic_offsets)]


def get_fxrate_spot(fieldname, static_offsets, stochastic_offsets, all_factors):
    """Read the spot of the FX rate price factor"""
    return calc_factor_value(
        utils.Factor('FxRate', fieldname), static_offsets, stochastic_offsets, all_factors)[0]


def get_forwardprice_sampling(fieldname, all_factors):
    """Read the sampling offset for the Sampling_type price factor"""
    return all_factors.get(utils.Factor('ForwardPriceSample', fieldname))


def get_forwardprice_factor(cashflow_currency, static_offsets, stochastic_offsets,
                            all_tenors, all_factors, reference_factor, forward_factor, base_date):
    """Read the Forward price factor of a reference Factor - adjusts the all_tenors lookup to
    include the excel_date version of the base_date"""

    forward_price = reference_factor.get_forwardprice()
    # note that in future we could stack forwardprices together
    forward_offset = [calc_factor_index(
        utils.Factor('ForwardPrice', forward_price), static_offsets, stochastic_offsets, all_tenors)]
    forward_fx_ofs = get_fx_and_zero_rate_factor(
        forward_factor.get_currency(), static_offsets, stochastic_offsets, all_tenors, all_factors)

    if cashflow_currency != forward_factor.get_currency():
        cashflow_fx_ofs = get_fx_and_zero_rate_factor(
            cashflow_currency, static_offsets, stochastic_offsets, all_tenors, all_factors)
        return [forward_offset, forward_fx_ofs, cashflow_fx_ofs]
    else:
        return [forward_offset, forward_fx_ofs, forward_fx_ofs]


def get_reference_factor_objects(fieldname, all_factors):
    """Read the Reference and Forward price factors"""
    reference = all_factors.get(utils.Factor('ReferencePrice', fieldname))
    reference_factor = (reference.factor if hasattr(reference, 'factor') else reference)
    forward_price_name = reference_factor.get_forwardprice()
    forward = all_factors.get(utils.Factor('ForwardPrice', forward_price_name))
    forward_factor = (forward.factor if hasattr(forward, 'factor') else forward)
    return reference_factor, forward_factor


def get_implied_correlation(rate1, rate2, all_factors):
    correlation_name = rate1[:-1] + ('{0}/{1}'.format(rate1[-1], rate2[0]),) + rate2[1:] + (rate1[1],)
    implied_correlation = all_factors.get(utils.Factor('Correlation', correlation_name))
    return implied_correlation.current_value() if implied_correlation else 0.0


def get_equity_rate_factor(fieldname, static_offsets, stochastic_offsets):
    """Read the index of the Equity rate price factor"""
    return [calc_factor_index(utils.Factor('EquityPrice', fieldname),
                              static_offsets, stochastic_offsets)]


def get_equity_spot(fieldname, static_offsets, stochastic_offsets, all_factors):
    """Read the spot of the FX rate price factor"""
    return calc_factor_value(
        utils.Factor('EquityPrice', fieldname), static_offsets, stochastic_offsets, all_factors)[0]


def get_equity_currency_factor(fieldname, static_offsets, stochastic_offsets, all_factors):
    """Read the index of the Equity's Currency price factor"""
    equity_factor = all_factors.get(utils.Factor('EquityPrice', fieldname))
    fxrate = (equity_factor.factor if hasattr(equity_factor, 'factor') else equity_factor).get_currency()
    return [calc_factor_index(utils.Factor('FxRate', fxrate),
                              static_offsets, stochastic_offsets)]


def get_dividend_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the dividend rate price factor"""
    return [calc_factor_index(utils.Factor('DividendRate', fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


def get_interest_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the interest rate price factor"""
    return [calc_factor_index(utils.Factor('InterestRate', fieldname[:x]),
                              static_offsets, stochastic_offsets, all_tenors)
            for x in range(1, len(fieldname) + 1)]


def get_equity_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Read the equity's interest rate price factor"""
    equity_factor = all_factors.get(utils.Factor('EquityPrice', fieldname))
    ir_curve = (equity_factor.factor if hasattr(equity_factor, 'factor') else equity_factor).get_repo_curve_name()
    return get_interest_factor(ir_curve, static_offsets, stochastic_offsets, all_tenors)


def get_discount_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Get the interest rate curve linked to this discount rate price factor"""
    discount_curve = all_factors.get(utils.Factor('DiscountRate', fieldname)).get_interest_rate()
    return get_interest_factor(discount_curve, static_offsets, stochastic_offsets, all_tenors)


def get_fx_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Read the Currency's interest rate price factor"""
    fx_factor = all_factors.get(utils.Factor('FxRate', fieldname))
    ir_curve = (fx_factor.factor if hasattr(fx_factor, 'factor') else fx_factor).get_repo_curve_name(fieldname)
    return get_interest_factor(ir_curve, static_offsets, stochastic_offsets, all_tenors)


def get_fx_and_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Get the FX rate and it's repo curve"""
    return [get_fxrate_factor(fieldname, static_offsets, stochastic_offsets),
            get_fx_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors)]


def get_price_index_factor(fieldname, static_offsets, stochastic_offsets):
    """Read the index of the Price Index price factor"""
    return [calc_factor_index(utils.Factor('PriceIndex', fieldname), static_offsets, stochastic_offsets)]


def get_survival_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the inflation rate price factor"""
    return [calc_factor_index(utils.Factor('SurvivalProb', fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


def get_inflation_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the inflation rate price factor"""
    return [calc_factor_index(utils.Factor('InflationRate', fieldname[:x]),
                              static_offsets, stochastic_offsets, all_tenors)
            for x in range(1, len(fieldname) + 1)]


def get_fx_vol_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the fx vol price factor"""
    return [calc_factor_index(utils.Factor('FXVol', fieldname), static_offsets, stochastic_offsets,
                              all_tenors)]


def get_equity_price_vol_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the Equity Price vol price factor"""
    return [calc_factor_index(utils.Factor('EquityPriceVol', fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


def get_interest_vol_factor(fieldname, tenor, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the interest vol price factor"""
    pricefactor = 'InterestRateVol' if pd.Timestamp('1900-01-01') + tenor <= pd.Timestamp('1900-01-01') + pd.DateOffset(
        years=1) else 'InterestYieldVol'
    return [calc_factor_index(utils.Factor(pricefactor, fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


def get_forward_price_vol_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the forward vol price factor"""
    return [calc_factor_index(utils.Factor('ForwardPriceVol', fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


class Deal(object):
    """
    Base class for representing a trade/deal. Needs to be able to aggregate sub-deals
    (e.g. a netting set can be a "deal") and calculate dynamic dates for resets.
    """
    documentation = ''

    def __init__(self, params, valuation_options):
        # valuation options
        self.options = valuation_options
        # instrument parameters
        self.field = params
        # is this instrument path dependent
        self.path_dependent = False
        # should this deal use the accumulator (evaluate child dependencies?)
        self.accum_dependencies = False

    def reset(self):
        self.reval_dates = set()
        self.settlement_currencies = {}

    def add_grid_dates(self, parser, base_date, grid):
        pass

    def add_reval_date_offset(self, offset, relative_to_settlement=True):
        if relative_to_settlement:
            for fixings in self.settlement_currencies.values():
                self.reval_dates.update([x + pd.DateOffset(days=offset) for x in fixings])
        else:
            fixings = reduce(
                set.union, [{x + pd.DateOffset(days=ofs) for ofs in offset}
                            for x in self.reval_dates])
            self.reval_dates.update(fixings)

    def add_reval_dates(self, dates, currency=None):
        self.reval_dates.update(dates)
        if currency:
            self.settlement_currencies.setdefault(currency, set()).update(dates)

    def get_reval_dates(self, clip_expiry=False):
        if clip_expiry and bool(self.settlement_currencies):
            last_cashflow = max([max(fixing) for fixing in self.settlement_currencies.values()])
            return {x for x in self.reval_dates if x <= last_cashflow}
        else:
            return self.reval_dates

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        if self.path_dependent:
            self.add_grid_dates(parser, base_date, grid if node_children is None else node_resets)

        node_resets.update(self.get_reval_dates())
        for settlement_currency, settlement_dates in self.get_settlement_currencies().items():
            node_settlements.setdefault(settlement_currency, set()).update(settlement_dates)

    def get_settlement_currencies(self):
        return self.settlement_currencies

    def calculate(self, shared, time_grid, deal_data):
        # generate the theo price
        mtm = self.generate(shared, time_grid, deal_data)
        # interpolate it
        return pricing.interpolate(mtm, shared, time_grid, deal_data)

    def generate(self, shared, time_grid, deal_data):
        raise Exception('generate in class {} not implemented yet for deal {}'.format(
            self.__class__.__name__, deal_data.Instrument.field.get('Reference')))


class NettingCollateralSet(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Agreement_Currency': ['FxRate'],
                     'Funding_Rate': ['DiscountRate'],
                     'Balance_Currency': ['FxRate'],
                     ('Collateral_Assets', 'Cash_Collateral', 'Currency'): ['FxRate'],
                     ('Collateral_Assets', 'Cash_Collateral', 'Funding_Rate'): ['InterestRate'],
                     ('Collateral_Assets', 'Cash_Collateral', 'Collateral_Rate'): ['InterestRate'],
                     ('Collateral_Assets', 'Equity_Collateral', 'Equity'): ['EquityPrice'],
                     ('Collateral_Assets', 'Bond_Collateral', 'Currency'): ['FxRate'],
                     ('Collateral_Assets', 'Bond_Collateral', 'Discount_Rate'): ['InterestRate']
                     }

    documentation = ('Collateral',
                     ['The general approach to simulating collateral is as follows:',
                      '',
                      'Define:',
                      '',
                      '- $t$ the simulation time',
                      '- $V(t)$ a realization of the uncollateralized portfolio for a scenario',
                      '- $\\hat V(t)$ a realization of collateralized portfolio for a scenario',
                      '- $B(t)$ the number of units of the collateral portfolio that should be held if the collateral '
                      'agreement was honoured by both parties. This includes minimum transfer amounts and is piecewise '
                      'constant between collateral call dates',
                      '- $S(t)$ is the value of one unit of the collateral portfolio in base currency for a scenario',
                      '- $\\delta_s$ the length of the settlement period',
                      '- $\\delta_l$ the length of the liquidation period',
                      '- $t_s$ the start of the settlement period',
                      '- $t_l$ the start of the liquidation period',
                      '- $t_e$ the end of the liquidation period',
                      '- $C(t_1,t_2)$ the value in base currency of cash in all currencies accumulated by the portfolio'
                      ' over the interval $[t_1,t_2]$',
                      '',
                      'The closeout period (when the counterparty becomes insolvent) starts at $t_s$ and ends at $t_e$',
                      '(when the position and collateral have been liquidated). This period is further divided into a',
                      'settlement period $\\delta_s$ followed by a liquidation period $\\delta_l$ such that the',
                      'liquidation begins at $t_l$ when the settlement period ends. After the settlement period,',
                      'neither party pays cashflows or transfers collateral but the market risk on the portfolio and',
                      'collateral continue until the end of liquidation period.',
                      '',
                      'The collateralized portfolio at time $t$ is the difference between the sum of the liquidated',
                      'portfolio value and the cash accumulated during the closeout period $C(t_s,t_e)$ and the',
                      'liquidated value of the $B$ units of the collateral portfolio $S(t_e)$ held:'
                      '',
                      '$$\\hat V(t)=V(t_e)+C(t_s,t_e)-\\min\\{B(u):t_s\\le u\\le t_l\\}S(t_e)$$',
                      '',
                      'The calculation of $C(t_s,t_e), B(t)$ and $S(t)$ is described below.',
                      ''
                      '### Standard and Forward looking closeout',
                      '',
                      'Usually exposure is reported at the end of the closeout period i.e. :',
                      '',
                      '- $t_s=t-\\delta_l-\\delta_s$',
                      '- $t_l=t-\\delta_l$',
                      '- $t_e=t$',
                      '',
                      'However, it is also possible to model the settlement and liquidation periods to come after $t$.',
                      'This forward looking closeout implies:',
                      '',
                      '- $t_s=t$',
                      '- $t_l=t+\\delta_s$',
                      '- $t_e=t+\\delta_s+\\delta_l$',
                      '',
                      'Note that Standard closeout causes exposure to be reported one closeout period after portfolio',
                      'maturity (as the mechanics of default are still present).',
                      '',
                      '### Cashflow Accounting',
                      '',
                      'Define (for time $t$):',
                      '',
                      '- $C_r(t)$ the unsigned cumulative base currency amount of cash received per scenario',
                      '- $C_p(t)$ the unsigned cumulative base currency amount of cash paid per scenario',
                      '- $C_i(t)$ the signed net cash amount paid per scenario per currency $i$ (positive if received,'
                      ' negative if paid)',
                      '- $X_i(t)$ the exchange rate from currency $i$ to base currency',
                      '',
                      'Interest over the closeout period is assumed negligible and hence not calculated. If **Exclude'
                      ' Paid Today** is **No**,',
                      'then the portfolio value at $t$ includes cash paid on $t$ and hence:',
                      '',
                      '$$C_r(t)=\\sum_{t_j<t}\\sum_i X_i(t_j)\\max(0,C^i(t_j))$$',
                      '$$C_p(t)=\\sum_{t_j<t}\\sum_i X_i(t_j)\\max(0,-C^i(t_j))$$',
                      '',
                      'Otherwise (when **Exclude Paid Today** is **Yes**):',
                      '',
                      '$$C_r(t)=\\sum_{t_j\\le t}\\sum_i X_i(t_j)\\max(0,C^i(t_j))$$',
                      '$$C_p(t)=\\sum_{t_j\\le t}\\sum_i X_i(t_j)\\max(0,-C^i(t_j))$$',
                      '',
                      '#### Settlement risk mechanics',
                      '',
                      'The **Cash Settlement Risk** mechanics are determined depending on the order of cash payments'
                      ' verses collateral.',
                      '',
                      '- **Received Only** assumes that collateral is transfered before cash. All cash is retained'
                      ' during the settlement period.',
                      '',
                      '$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_s)+C_p(t_s)$$',
                      '',
                      '- **Paid Only** assumes that cash is transferred before collateral. Cash is paid by both sides',
                      'and no cash is retained during the settlement period.',
                      '',
                      '$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_l)+C_p(t_l)$$',
                      '',
                      '- **All** assumes that the ordering of cash and collateral is not fixed (i.e. paid and received'
                      ' cash are at risk). Only received cash is retained during the settlement period.',
                      '',
                      '$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_s)+C_p(t_l)$$',
                      '',
                      '### Collateral Accounting',
                      '',
                      'Define:',
                      '',
                      '- $A(t_i)$ the agreed value in base currency of collateral that should be held per scenario if',
                      ' the CSA was honoured by both parties *ignoring minimum transfer amounts*. Collateral is not',
                      'held simultaneously by both parties. Received collateral is positive and posted is negative.',
                      '- $A(0)$ the initial value of collateral in base currency specified by the **Opening Balance**.',
                      '- $X(t)$ the exchange from the CSA currency to the base currency. All amounts on the CSA',
                      '(Thresholds, minimum transfers etc.) are expressed in the **Agreement Currency**.',
                      '- $I$ the independent amount of collateral in agreement currency that is either posted (negative)'
                      ' or received (positive).',
                      '- $H(t)$ the received threshold of collateral: if the portfolio value is above this, the agreed '
                      'collateral value must be increased by the difference.',
                      '- $G(t)$ the posted threshold of collateral: if the portfolio value is below this, the agreed '
                      'collateral value must be decreased by the difference.',
                      '- $M_r(t)$ the minimum received transfer amount. The collateral held will not increase unless '
                      'the increase is at least this amount.',
                      '- $M_p(t)$ the minimum posted transfer amount. The collateral posted will not increase unless '
                      'the increase is at least this amount.',
                      '- $S_h(t)$ is the value in base currency of one unit of the collateral portfolio after haircuts.',
                      '- $t_i, i>0$ are collateral call dates. Note that $t_0=0$ and that in general, $t_0$ need not be'
                      ' a collateral call date.',
                      '',
                      'We then have the following relationships:',
                      '',
                      '$$A(t_i)=X(t_i)I+\\begin{cases} V(t_i)-X(t_i)H(t_i), '
                      '&\\text{if } V(t_i)>X(t_i)H(t_i)\\\\ V(t_i)-X(t_i)G(t_i), '
                      '&\\text{if } V(t_i)<X(t_i)G(t_i) \\end{cases}$$',
                      '',
                      'In general, the presence of minimum transfer amounts introduce a path dependency on $B(t)$ and,',
                      'as such, is not a simple function of $A(t_i)$. Instead, it can be expressed via the following',
                      'recurrence:',
                      '',
                      '$$B(t_i)=\\begin{cases} \\frac{A(0)}{S_h(0)}, &\\text{if } i=0\\\\',
                      '\\frac{A(t_i)}{S_h(t_i)}, &\\text{if } A(t_i)-S(t_i)B(t_{i-1})\\ge M_r(t_i)X(t_i)'
                      '\\text{ and } i>0,\\\\',
                      '&\\text{or if } S(t_i)B(t_{i-1})-A(t_i)\\ge M_p(t_i)X(t_i)\\text{ and } i>0\\\\',
                      'B(t_{i-1}), &\\text{otherwise}\\end{cases}$$',
                      '',
                      'Since $B(t_i)$ is constant between call dates, $B(t)=B(t_{i^*})$, where $t_{i^*}$ is the closest',
                      'call date on or before $t$. Since $B(t)$ is path dependent, it requires calculation on all',
                      'collateral call dates. Due to this being fairly prohibitive, the recurrence is only evaluated',
                      'at collateral call dates associated with a particular simulation time grid. This approximation',
                      'can be improved by using a finer simulation grid (also note that the path dependency dissipates',
                      'as the minimum transfer amounts reduce to zero).',
                      '',
                      '### Collateral Portfolio',
                      '',
                      'Define:',
                      '',
                      '- $a_i$ the number of units of the $i^{th}$ asset in the collateral portfolio.',
                      '- $S_i(t)$ the base currency value per unit of the $i^{th}$ asset.',
                      '',
                      'Currently, the collateral portfolio may only consist of equities and cash and therefore will be',
                      'sensitive to interest rates, FX and equity prices over the closeout period. The relative',
                      'amounts of each collateral asset in the collateral portfolio is held constant. The percent of',
                      'the collateral portfolio represented by a given asset may change as its price relative to other',
                      'assets change. The value of the portfolio is therefore:',
                      '',
                      '$$S(t)=\\sum_i a_i S_i(t)$$',
                      '',
                      'Haircuts may be defined for each asset class, including cash. The value of each asset allowing',
                      'for haircuts is as follows:',
                      '',
                      '$$S_{h,i}=(1-h)S_i(t)$$',
                      '',
                      'Haircuts must be strictly less than one.',
                      ])

    def __init__(self, params, valuation_options):
        super(NettingCollateralSet, self).__init__(params, valuation_options)
        self.path_dependent = True
        self.calendar = None
        self.options = {'Cash_Settlement_Risk': utils.CASH_SETTLEMENT_Received_Only,
                        'Forward_Looking_Closeout': False,
                        'Use_Optimal_Collateral': False,
                        'Exclude_Paid_Today': False}
        self.options.update(valuation_options)

        # make sure collateral default parameters are defined
        self.field.setdefault('Settlement_Period', 0)
        self.field.setdefault('Liquidation_Period', 0)
        self.field.setdefault('Opening_Balance', 0.0)

    def reset(self, calendars):
        super(NettingCollateralSet, self).reset()
        # allow the accumulator
        self.accum_dependencies = True
        # if we allow collateral, this instrument is now path dependent
        if self.field.get('Collateralized', 'False') == 'True':
            self.path_dependent = True
            calendar = calendars.get(self.field.get('Calendars'))
            if calendar:
                self.calendar = calendar['businessday']

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):

        def calc_liquidation_settlement_dates(date):
            if self.options['Forward_Looking_Closeout']:
                tl = date + pd.DateOffset(days=self.field['Settlement_Period'])
                ts = date
            else:
                tl = max(date - pd.DateOffset(days=self.field['Liquidation_Period']), base_date)
                ts = max(date - pd.DateOffset(days=self.field['Settlement_Period'] + self.field['Liquidation_Period']),
                         base_date)
            return ts, tl

        # have to reset the original instrument and let the children nodes determine the outcome
        if self.field.get('Collateralized', 'False') == 'True':
            # update each child element with extra reval dates
            for child in node_children:
                # child.add_reval_dates({max(child.get_reval_dates()) + pd.offsets.Day(1)})
                child.add_reval_date_offset(1)
                settle_liquid = {self.field['Settlement_Period'],
                                 self.field['Settlement_Period'] + self.field['Liquidation_Period']}
                child.add_reval_date_offset(settle_liquid, relative_to_settlement=False)
                node_resets.update(child.get_reval_dates())
                # add an offset for this instrument
                # child.add_reval_date_offset(1)

            # Load the time grid
            grid_dates = parser(base_date, max(node_resets), grid)

            # load up all dynamic dates
            for dates in node_settlements.values():
                valid_fixings = np.clip(list(dates), base_date, max(dates))
                grid_dates.update(valid_fixings)

            # determine the new dates to add
            node_additions = set()
            # add a date just after the rundate
            node_additions.add(base_date + pd.DateOffset(days=1))
            # check if we need to add dates for collateral
            base_call_date = self.field['Base_Collateral_Call_Date'] if self.field.get(
                'Base_Collateral_Call_Date') else base_date
            call_freq = self.field['Collateral_Call_Frequency'] if self.field.get(
                'Collateral_Call_Frequency') else pd.DateOffset(days=1)

            if call_freq.kwds != {'days': 1}:
                # self.calendar
                coll_dates = generate_dates_forward(
                    max(grid_dates), base_call_date, call_freq,
                    bus_day=self.calendar, modified=True)
                grid_dates.update(coll_dates)

            # the current complete grid
            full_grid = grid_dates.union({x for x in node_resets if x >= base_date})

            for t in sorted(full_grid):
                ts, tl = calc_liquidation_settlement_dates(t)
                node_additions.update({t, ts, tl})

            # now set the reval dates
            fresh_grid = full_grid.union(node_additions)
            self.reval_dates = set()

            for date in sorted(fresh_grid):
                ts, tl = calc_liquidation_settlement_dates(date)
                if ts in fresh_grid and tl in fresh_grid:
                    self.reval_dates.add(date)

            # add more valuation nodes
            node_resets.update(node_additions)

            # finally let the children know about the new grid
            final_grid = np.array(sorted(node_resets))
            for child in node_children:
                if child.path_dependent:
                    child.finalize_dates(parser, base_date, grid, node_children, node_resets, node_settlements)
                else:
                    child_expiry = max(child.get_reval_dates())
                    child.add_reval_dates(set(final_grid[:final_grid.searchsorted(child_expiry)]))
        else:
            for child in node_children:
                child.add_reval_date_offset(1)
                node_resets.update(child.get_reval_dates())

            self.reval_dates = node_resets

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {}
        field_index = {}

        # only set this up if this is a collateralized deal
        if self.field.get('Collateralized', 'False') == 'True':
            field['Agreement_Currency'] = utils.check_rate_name(self.field['Agreement_Currency'])
            field['Funding_Rate'] = utils.check_rate_name(self.field['Funding_Rate']) if self.field.get(
                'Funding_Rate') else None

            # apparently this should default to the base currency not the agreement currency,
            # but I think this is a better default..
            field['Balance_Currency'] = utils.check_rate_name(self.field['Balance_Currency']) if self.field.get(
                'Balance_Currency') else field['Agreement_Currency']

            field_index['Agreement_Currency'] = get_fxrate_factor(
                field['Agreement_Currency'], static_offsets, stochastic_offsets)
            field_index['Balance_Currency'] = get_fxrate_factor(
                field['Balance_Currency'], static_offsets, stochastic_offsets)

            # get the settlement currencies loaded
            field_index['Settlement_Currencies'] = OrderedDict()
            for currency in time_grid.CurrencyMap.keys():
                field_index['Settlement_Currencies'].setdefault(
                    currency, get_fxrate_factor(utils.check_rate_name(currency), static_offsets, stochastic_offsets))

            # handle equity collateral
            collateral_defined = False
            field_index['Equity_Collateral'] = []
            # handle bond collateral
            field_index['Bond_Collateral'] = []
            # get the collateral currency loaded
            field_index['Cash_Collateral'] = []

            collateral_assets = self.field.get('Collateral_Assets')

            if collateral_assets:
                if collateral_assets.get('Equity_Collateral'):
                    collateral_equity = self.field['Collateral_Assets']['Equity_Collateral']

                    for col_equity in collateral_equity:
                        equity_rate = utils.check_rate_name(col_equity['Equity'])
                        field_index['Equity_Collateral'].append(
                            utils.Collateral(Haircut=float(col_equity['Haircut_Posted']),
                                             Amount=col_equity['Units'],
                                             Currency=get_equity_currency_factor(
                                                 equity_rate, static_offsets, stochastic_offsets, all_factors),
                                             Funding_Rate=None, Collateral_Rate=None,
                                             Collateral=get_equity_rate_factor(
                                                 equity_rate, static_offsets, stochastic_offsets))
                        )
                    collateral_defined = True

                if collateral_assets.get('Bond_Collateral'):
                    collateral_bond = self.field['Collateral_Assets']['Bond_Collateral']

                    for col_bond in collateral_bond:
                        if np.array(list(col_bond['Coupon_Interval'].kwds.values())).any():
                            reset_dates = generate_dates_backward(
                                base_date + col_bond['Maturity'], base_date, col_bond['Coupon_Interval'])
                        else:
                            reset_dates = np.array([base_date, base_date + col_bond['Maturity']])
                        fixed_cash = utils.generate_fixed_cashflows(
                            base_date, reset_dates, col_bond['Principal'], None, 0, float(col_bond['Coupon_Rate']))
                        # make sure there's a nominal repayment at maturity
                        fixed_cash.add_fixed_payments(base_date, 'Maturity', base_date, 0, col_bond['Principal'])
                        discount = get_interest_factor(utils.check_rate_name(col_bond['Discount_Rate']),
                                                       static_offsets, stochastic_offsets,
                                                       all_tenors)
                        time_index = np.searchsorted(time_grid.mtm_time_grid, (max(reset_dates) - base_date).days)

                        field_index['Bond_Collateral'].append(
                            utils.Collateral(Haircut=float(col_bond['Haircut_Posted']),
                                             Amount=1,
                                             Currency=get_fxrate_factor(
                                                 utils.check_rate_name(col_bond['Currency']),
                                                 static_offsets, stochastic_offsets),
                                             Funding_Rate=None, Collateral_Rate=None,
                                             Collateral=utils.DealDataType(
                                                 Instrument=None,
                                                 Factor_dep={'Cashflows': fixed_cash, 'Discount': discount},
                                                 Time_dep=utils.DealTimeDependencies(
                                                     time_grid.mtm_time_grid, np.arange(time_index)),
                                                 Calc_res=None))
                        )
                    collateral_defined = True

                if collateral_assets.get('Cash_Collateral'):
                    collateral_cash = self.field['Collateral_Assets']['Cash_Collateral']
                    for col_cash in collateral_cash:
                        field_index['Cash_Collateral'].append(
                            utils.Collateral(Haircut=float(col_cash['Haircut_Posted']),
                                             Amount=col_cash['Amount'],
                                             Currency=get_fxrate_factor(
                                                 utils.check_rate_name(col_cash['Currency']),
                                                 static_offsets, stochastic_offsets),
                                             Funding_Rate=get_interest_factor(
                                                 utils.check_rate_name(
                                                     col_cash['Funding_Rate']), static_offsets,
                                                 stochastic_offsets, all_tenors)
                                             if col_cash.get('Funding_Rate') else None,
                                             Collateral_Rate=get_interest_factor(
                                                 utils.check_rate_name(
                                                     col_cash['Collateral_Rate']),
                                                 static_offsets, stochastic_offsets, all_tenors)
                                             if col_cash.get('Collateral_Rate') else None,
                                             Collateral=1.0)
                        )
                    collateral_defined = True

            if not collateral_defined:
                # default the collateral to the be balance currency
                field_index['Cash_Collateral'] = [
                    utils.Collateral(Haircut=0.0,
                                     Amount=1.0,
                                     Currency=get_fxrate_factor(
                                         field['Balance_Currency'], static_offsets, stochastic_offsets),
                                     Funding_Rate=None, Collateral_Rate=None,
                                     Collateral=1.0)]

            # check if the independent amount has been mapped
            if self.field['Credit_Support_Amounts'].get('Independent_Amount'):
                field_index['Independent_Amount'] = self.field['Credit_Support_Amounts']['Independent_Amount'].value()
            else:
                field_index['Independent_Amount'] = 0.0

            # now get the closeout mechanics right
            t = time_grid.time_grid[:, utils.TIME_GRID_MTM]
            # collateral call dates (interpolated - assuming daily)
            base_call_date = self.field['Base_Collateral_Call_Date'] if self.field.get(
                'Base_Collateral_Call_Date') else base_date
            call_freq = self.field['Collateral_Call_Frequency'] if self.field.get(
                'Collateral_Call_Frequency') else pd.DateOffset(days=1)
            call_dates = np.array([(x - base_date).days for x in sorted(self.get_reval_dates())])
            call_mask = np.ones(time_grid.mtm_time_grid.size, dtype=np.int32)

            if call_freq.kwds != {'days': 1}:
                all_call_days = generate_dates_forward(
                    max(self.get_reval_dates()), base_call_date, call_freq,
                    bus_day=self.calendar, modified=True)
                approx_calls = pd.DatetimeIndex(sorted(time_grid.mtm_dates)).intersection(all_call_days)
                mod_call_dates = np.array([(x - base_date).days for x in approx_calls])

                call_mask[np.searchsorted(time_grid.mtm_time_grid, mod_call_dates)] = 2
                call_mask -= 1

            # store the mask    
            field_index['call_mask'] = call_mask

            if self.options['Forward_Looking_Closeout']:
                field_index['Ts'] = (
                        np.searchsorted(t, call_dates, side='right').astype(np.int32) - 1
                ).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Tl'] = (
                        np.searchsorted(t, call_dates + self.field['Settlement_Period'], side='right').astype(
                            np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Te'] = (
                        np.searchsorted(t,
                                        call_dates + self.field['Settlement_Period'] + self.field['Liquidation_Period'],
                                        side='right').astype(np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
            else:
                field_index['Ts'] = (
                        np.searchsorted(t,
                                        call_dates - self.field['Settlement_Period'] - self.field['Liquidation_Period'],
                                        side='right').astype(np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Tl'] = (
                        np.searchsorted(t, call_dates - self.field['Liquidation_Period'], side='right').astype(
                            np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Te'] = (
                        np.searchsorted(t, call_dates, side='right').astype(np.int32) - 1
                ).clip(0, time_grid.mtm_time_grid.size - 1)

        return field_index

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        # calc v^t = v(te) + C(ts,te) - min(B(u); ts<=u<=tl}S(te)
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        if self.field.get('Collateralized', 'False') == 'True' and hasattr(shared, 't_Credit'):
            C_base = 0.0

            for curr, fx_factor in factor_dep['Settlement_Currencies'].items():
                Ci = []
                T = sorted(shared.t_Cashflows[curr].keys())
                for index, pad in enumerate(np.diff([-1] + T + [time_grid.time_grid.shape[0]]) - 1):
                    if index < len(T):
                        cashflow_at_index = torch.unsqueeze(shared.t_Cashflows[curr][T[index]], axis=0)
                        Ci.append(F.pad(cashflow_at_index, [0, 0, pad, 0]))

                base_Ci = torch.cat(Ci, axis=0)
                # pad the last bit
                C_base += F.pad(utils.calc_time_grid_spot_rate(
                    fx_factor, time_grid.time_grid[:-pad], shared) * base_Ci, [0, 0, 0, pad])

            if not self.options['Exclude_Paid_Today']:
                C_block = C_base[:-1]
                Cf_Rec = F.pad(torch.cumsum(torch.relu(C_block), axis=0), [0, 0, 1, 0])
                Cf_Pay = F.pad(torch.cumsum(torch.relu(-C_block), axis=0), [0, 0, 1, 0])
            else:
                Cf_Rec = torch.cumsum(torch.relu(C_base), axis=0)
                Cf_Pay = torch.cumsum(torch.relu(-C_base), axis=0)

            # calc collateral values
            St = torch.zeros_like(accum)

            for cash_col in factor_dep['Cash_Collateral']:
                St += (1.0 - cash_col.Haircut) * cash_col.Amount * utils.calc_time_grid_spot_rate(
                    cash_col.Currency, time_grid.time_grid, shared)

            for bond_col in factor_dep['Bond_Collateral']:
                bond = pricing.pv_fixed_cashflows(shared, time_grid, bond_col.Collateral, settle_cash=False)
                padding = time_grid.time_grid.shape[0] - bond.shape[0]
                St += (1.0 - bond_col.Haircut) * utils.calc_time_grid_spot_rate(
                    bond_col.Currency, time_grid.time_grid, shared) * F.pad(bond, [0, 0, 0, padding])

            for equity_col in factor_dep['Equity_Collateral']:
                St += (1.0 - equity_col.Haircut) * equity_col.Amount * utils.calc_time_grid_spot_rate(
                    equity_col.Currency, time_grid.time_grid, shared) * utils.calc_time_grid_spot_rate(
                    equity_col.Collateral, time_grid.time_grid, shared)

            # calc the collateral amount    
            fx_base = utils.calc_time_grid_spot_rate(shared.Report_Currency, time_grid.time_grid, shared)
            fx_agreement = utils.calc_time_grid_spot_rate(factor_dep['Agreement_Currency'], time_grid.time_grid, shared)
            H = self.field['Credit_Support_Amounts']['Received_Threshold'].value() * fx_agreement
            G = self.field['Credit_Support_Amounts']['Posted_Threshold'].value() * fx_agreement
            min_received = self.field['Credit_Support_Amounts']['Minimum_Received'].value()
            min_posted = self.field['Credit_Support_Amounts']['Minimum_Posted'].value()

            Vt = accum * fx_base

            if self.options['Exclude_Paid_Today']:
                mtm_today_adj = torch.cat([Cf_Rec[0].reshape(1, -1), Cf_Rec[1:] - Cf_Rec[:-1]], axis=0) - \
                                torch.cat([Cf_Pay[0].reshape(1, -1), Cf_Pay[1:] - Cf_Rec[:-1]], axis=0)
                Vt -= mtm_today_adj

            At = factor_dep['Independent_Amount'] * fx_agreement + (Vt - H) * (Vt > H) + (Vt - G) * (Vt < G)

            Bt = self.field['Opening_Balance'] * utils.calc_time_grid_spot_rate(
                factor_dep['Balance_Currency'], np.array([[0, 0, 0]]), shared) / St[0]

            # scan for the correct amount of the collateral portfolio
            fx_St = fx_agreement / St
            Bt_new = At / St
            Mr = Bt_new - min_received * fx_St
            Mp = min_posted * fx_St + Bt_new

            if factor_dep['call_mask'].all():
                # daily collateral
                Sim_Bt = [Bt[0]]
                for mr, mp, bt in zip(Mr[1:], Mp[1:], Bt_new[1:]):
                    mask = ((Sim_Bt[-1] < mr) | (Sim_Bt[-1] > mp))
                    Sim_Bt.append(bt * mask + Sim_Bt[-1] * (~mask))
            else:
                # collateral according to call_mask
                Sim_Bt = [Bt[0]]
                Call_mask = factor_dep['call_mask'][1:].astype(np.bool)
                for mr, mp, bt, cm in zip(Mr[1:], Mp[1:], Bt_new[1:], Call_mask[1:]):
                    if cm:
                        mask = ((Sim_Bt[-1] < mr) | (Sim_Bt[-1] > mp))
                        Sim_Bt.append(bt * mask + Sim_Bt[-1] * (~mask))
                    else:
                        Sim_Bt.append(Sim_Bt[-1])

            Bt = torch.stack(Sim_Bt)

            # now calculate the collateral account and Net exposure
            fx_base = utils.calc_time_grid_spot_rate(shared.Report_Currency, deal_time, shared)

            Vte = accum[factor_dep['Te']] * fx_base

            if self.options['Exclude_Paid_Today']:
                mtm_today_adj = (
                    Cf_Rec[factor_dep['Te']] - F.pad(Cf_Rec[factor_dep['Te'][1:] - 1], [0, 0, 1, 0])) - (
                    Cf_Pay[factor_dep['Te']] - F.pad(Cf_Pay[factor_dep['Te'][1:] - 1], [0, 0, 1, 0]))

                Vte -= mtm_today_adj

            if self.options['Cash_Settlement_Risk'] == utils.CASH_SETTLEMENT_Received_Only:
                C_ts_te = (Cf_Rec[factor_dep['Te']] - Cf_Rec[factor_dep['Ts']]) - \
                          (Cf_Pay[factor_dep['Te']] - Cf_Pay[factor_dep['Ts']])
            elif self.options['Cash_Settlement_Risk'] == utils.CASH_SETTLEMENT_Paid_Only:
                C_ts_te = (Cf_Rec[factor_dep['Te']] - Cf_Rec[factor_dep['Tl']]) - \
                          (Cf_Pay[factor_dep['Te']] - Cf_Pay[factor_dep['Tl']])
            else:
                C_ts_te = (Cf_Rec[factor_dep['Te']] - Cf_Rec[factor_dep['Ts']]) - \
                          (Cf_Pay[factor_dep['Te']] - Cf_Pay[factor_dep['Tl']])

            # note that this should be the minimum Bt from factor_dep['Ts'] to factor_dep['Tl']
            Ste = St[factor_dep['Te']]
            Bte = Bt[factor_dep['Te']]
            # Go from time Ts one step at a time and keep track of the minimum
            base_i = factor_dep['Ts'][:-1]
            # work out how many step from time Ts to time Tl
            delta_T = factor_dep['Tl'][:-1] - base_i
            min_Bt = Bt[base_i]
            b_index = np.zeros_like(delta_T)

            for i in range(delta_T.max() if delta_T.size else 0):
                b_index[delta_T > i] = i + 1
                min_Bt = torch.min(min_Bt, Bt[base_i + b_index])

            # zero out the last row
            min_Bt = F.pad(min_Bt, [0, 0, 0, 1])

            # Store results
            shared.t_Credit['Gross MTM'] = Vte
            shared.t_Credit['Collateral'] = Bte * Ste

            # The net MTM of the netting set
            net_accum = (Vte + C_ts_te - min_Bt * Ste) / fx_base

            if len(factor_dep['Cash_Collateral']) == 1 and factor_dep['Cash_Collateral'][0].Collateral_Rate is not None:
                cash_col = factor_dep['Cash_Collateral'][0]
                cash_base = utils.calc_time_grid_spot_rate(cash_col.Currency, deal_time, shared)
                mtm_grid = deal_time[:, utils.TIME_GRID_MTM]

                # calculate collateral valuation adjustment
                delta_scen_t = np.diff(mtm_grid).reshape(-1, 1)
                discount_funding = utils.calc_time_grid_curve_rate(cash_col.Funding_Rate, deal_time[:-1], shared)
                discount_collateral = utils.calc_time_grid_curve_rate(cash_col.Collateral_Rate, deal_time[:-1], shared)
                discount_collateral_t0 = utils.calc_time_grid_curve_rate(
                    cash_col.Collateral_Rate, np.zeros((1, 3)), shared)

                Dc_over_f_tT_m1 = torch.expm1(
                    torch.squeeze(discount_funding.gather_weighted_curve(shared, delta_scen_t) -
                                  discount_collateral.gather_weighted_curve(shared, delta_scen_t), axis=1)
                )

                Dc0_T = torch.exp(
                    -torch.squeeze(discount_collateral_t0.gather_weighted_curve(
                        shared, mtm_grid[:-1].reshape(1, -1)), axis=0))

                shared.t_Credit['Funding'] = (shared.t_Credit['Collateral'][:-1] * cash_base *
                                              Dc_over_f_tT_m1 * Dc0_T) / fx_base[0]

            return net_accum
        else:
            # copy the mtm
            return accum


class MtMCrossCurrencySwapDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Pay_Interest_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol'],
                     'Pay_Discount_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol'],
                     'Receive_Interest_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol'],
                     'Receive_Discount_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol'],
                     'Pay_Currency': ['FxRate'],
                     'Pay_Discount_Rate': ['DiscountRate'],
                     'Pay_Interest_Rate': ['InterestRate'],
                     'Receive_Currency': ['FxRate'],
                     'Receive_Discount_Rate': ['DiscountRate'],
                     'Receive_Interest_Rate': ['InterestRate']}

    documentation = ('Interest Rates',
                     ['This currency swap adjusts the notional of one leg to capture any changes in the FX Spot rate',
                      'since the last reset. At each reset, the principal of the adjusted leg is set to the principal',
                      'of the unadjusted leg multiplied by the spot FX rate. MtM cross currency swaps are path',
                      'dependent.',
                      '',
                      'The unadjusted leg is either a fixed or floating interest rate list and is valued as such,',
                      'however, the floating adjusted leg is valued as',
                      '',
                      '$$\\sum_{i=1}^n(P_i(t)(L_i(t)+m)\\alpha_i+A_i(t))D(t,t_i),$$',
                      '',
                      'where',
                      '',
                      '- $A_i(t)=P_i(t)-P_{i+1}(t)$ for $i\\lt n$ and $A_n(t)=P_n(t)$',
                      '- $P_i(t)$ is the expected principal $P_i(t)=F(t,t_{i-1})\\tilde P_i$,',
                      '- $\\tilde P_i$ is the unadjusted leg principal for the $i^{th}$ period.',
                      '- $F(t,T)$ is the forward FX rate for settlement at time $T$.'
                      ])

    def __init__(self, params, valuation_options):
        super(MtMCrossCurrencySwapDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(MtMCrossCurrencySwapDeal, self).reset()
        self.paydates = generate_dates_backward(
            self.field['Maturity_Date'], self.field['Effective_Date'],
            self.field.get('Pay_Frequency', pd.DateOffset(months=6)))
        self.recdates = generate_dates_backward(
            self.field['Maturity_Date'], self.field['Effective_Date'],
            self.field.get('Receive_Frequency', pd.DateOffset(months=6)))
        self.add_reval_dates(self.paydates, self.field['Pay_Currency'])
        self.add_reval_dates(self.recdates, self.field['Receive_Currency'])
        self.child_map = {}

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        # have to reset the original instrument and let the child node decide
        super(MtMCrossCurrencySwapDeal, self).reset()
        for currency, dates in node_settlements.items():
            if 'Start' in self.field['Principal_Exchange'] and base_date <= self.field['Effective_Date']:
                dates.add(self.field['Effective_Date'])
            self.add_reval_dates(dates, currency)
        return super(MtMCrossCurrencySwapDeal, self).finalize_dates(
            parser, base_date, grid, node_children, node_resets, node_settlements)

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        if not self.child_map:
            for index, child in enumerate(child_dependencies):
                # make the child price to the same grid as the parent
                child.Time_dep.assign(deal_data.Time_dep)
                # work out which leg is which
                if child.Factor_dep['Currency'] == factor_dep[factor_dep['MTM']]['Currency']:
                    daycount = self.field.get(factor_dep['MTM'] + '_Day_Count', 'ACT_365')
                    # add a zero nominal payment at the beginning if forward starting
                    child.Factor_dep['Cashflows'].add_mtm_payments(
                        factor_dep['base_date'], self.field['Principal_Exchange'],
                        self.field['Effective_Date'], daycount)
                    self.child_map.setdefault('MTM', child)
                else:
                    daycount = self.field.get(factor_dep['Other'] + '_Day_Count', 'ACT_365')
                    # get the last nominal amount
                    capital = child.Factor_dep['Cashflows'].schedule[-1][utils.CASHFLOW_INDEX_Nominal]
                    child.Factor_dep['Cashflows'].add_fixed_payments(
                        factor_dep['base_date'], self.field['Principal_Exchange'],
                        self.field['Effective_Date'], daycount, capital)
                    self.child_map.setdefault('Static', child)

        FX_rep = utils.calc_time_grid_spot_rate(shared.Report_Currency, deal_time, shared)
        FX_static = utils.calc_time_grid_spot_rate(
            self.child_map['Static'].Factor_dep['Currency'][0], deal_time, shared)
        FX_mtm = utils.calc_time_grid_spot_rate(
            self.child_map['MTM'].Factor_dep['Currency'][0], deal_time, shared)

        if self.child_map['Static'].Factor_dep.get('Forward'):
            static_leg = pricing.pv_float_cashflow_list(shared, time_grid, self.child_map['Static'],
                                                        pricing.pricer_float_cashflows)
        else:
            static_leg = pricing.pv_fixed_cashflows(shared, time_grid, self.child_map['Static'])

        mtm_leg = pricing.pv_float_cashflow_list(
            shared, time_grid, self.child_map['MTM'], pricing.pricer_float_cashflows,
            mtm_currency=self.child_map['Static'].Factor_dep['Currency'])

        mtm = (static_leg * FX_static + mtm_leg * FX_mtm) / FX_rep

        # interpolate the Theo price
        return pricing.interpolate(mtm, shared, time_grid, deal_data)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Pay_Currency': utils.check_rate_name(self.field['Pay_Currency'])}
        field['Pay_Discount_Rate'] = utils.check_rate_name(self.field['Pay_Discount_Rate']) if self.field[
            'Pay_Discount_Rate'] else field['Pay_Currency']
        field['Pay_Interest_Rate'] = utils.check_rate_name(self.field['Pay_Interest_Rate']) if self.field[
            'Pay_Interest_Rate'] else field['Pay_Discount_Rate']

        field['Receive_Currency'] = utils.check_rate_name(self.field['Receive_Currency'])
        field['Receive_Discount_Rate'] = utils.check_rate_name(self.field['Receive_Discount_Rate']) if self.field[
            'Receive_Discount_Rate'] else field['Receive_Currency']
        field['Receive_Interest_Rate'] = utils.check_rate_name(self.field['Receive_Interest_Rate']) if self.field[
            'Receive_Interest_Rate'] else field['Receive_Discount_Rate']

        field_index = {'Pay': {}, 'Receive': {}}
        self.isQuanto = get_interest_rate_currency(
            field['Receive_Interest_Rate'], all_factors) != field['Receive_Currency']
        if self.isQuanto:
            # TODO - Deal with Quanto Interest Rate swaps
            pass
        else:
            field_index['Pay']['Currency'] = get_fx_and_zero_rate_factor(
                field['Pay_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Pay']['Forward'] = get_interest_factor(
                field['Pay_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Pay']['Discount'] = get_discount_factor(
                field['Pay_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Receive']['Currency'] = get_fx_and_zero_rate_factor(
                field['Receive_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Receive']['Forward'] = get_interest_factor(
                field['Receive_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Receive']['Discount'] = get_discount_factor(
                field['Receive_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)

        # TODO - complete cashflow definitions..

        # Which side is the mtm leg?
        field_index['MTM'] = self.field['MtM_Side']
        field_index['Other'] = {'Pay': 'Receive', 'Receive': 'Pay'}[self.field['MtM_Side']]
        field_index['base_date'] = base_date

        return field_index


class FXNonDeliverableForward(Deal):
    factor_fields = {'Buy_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Sell_Currency': ['FxRate'],
                     'Settlement_Currency': ['FxRate']}

    documentation = ('FX and Equity',
                     ['An FX non-deliverable forward effectively an FX Forward deal that is cash settled in a',
                      '(potentially) third currency. The deal pays',
                      '',
                      '$$A \\tilde X(t)-BX(t)$$',
                      '',
                      'in **settlement currency** at the **settlement date** $T$ where:',
                      '',
                      '- $A$ is the buy currency',
                      '- $B$ is the sell currency',
                      '- $\\tilde X(t)$ is the price of the buy currency in settlement currency',
                      '- $X$ is the price of the sell currency in settlement currency',
                      '',
                      'The value of the deal in settlement currency at time $t$, ($t \\le T$), is',
                      ''
                      '$$ \\Big(A \\tilde F(t,T)-BF(t,T)\\Big)D(t,T),$$',
                      '',
                      'where:',
                      '',
                      '- $\\tilde F(t,T)$ is the forward price of the buy currency in settlement currency',
                      '- $F(t,T)$ is the forward price of the sell currency in settlement currency'
                      ])

    def __init__(self, params, valuation_options):
        super(FXNonDeliverableForward, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FXNonDeliverableForward, self).reset()
        self.add_reval_dates({self.field['Settlement_Date']}, self.field['Settlement_Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Buy_Currency': utils.check_rate_name(self.field['Buy_Currency']),
                 'Sell_Currency': utils.check_rate_name(self.field['Sell_Currency']),
                 'Settlement_Currency': utils.check_rate_name(self.field['Settlement_Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Settlement_Currency']

        field_index = {
            'BuyFX': get_fx_and_zero_rate_factor(
                field['Buy_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SellFX': get_fx_and_zero_rate_factor(
                field['Sell_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleFX': get_fx_and_zero_rate_factor(
                field['Settlement_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Maturity': (self.field['Settlement_Date'] - base_date).days,
            # needed for reporting
            'Local_Currency': '{0}.{1}'.format(self.field['Buy_Currency'], self.field['Sell_Currency'])
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

        buy_forward = utils.calc_fx_forward(
            factor_dep['BuyFX'], factor_dep['SettleFX'], factor_dep['Maturity'], deal_time, shared)
        sell_forward = utils.calc_fx_forward(
            factor_dep['SellFX'], factor_dep['SettleFX'], factor_dep['Maturity'], deal_time, shared)
        FX_rep = utils.calc_fx_cross(
            factor_dep['SettleFX'][0], shared.Report_Currency, deal_time, shared)

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(
                discount,
                (factor_dep['Maturity'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1), shared),
            axis=1)

        cash = (buy_forward * self.field['Buy_Amount'] - sell_forward * self.field['Sell_Amount'])

        # settle the cash
        pricing.cash_settle(
            shared, self.field['Settlement_Currency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])

        return cash * discount_rates * FX_rep


class FXSwapDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Near_Buy_Far_Sell_Ccy': ['FxRate'],
                     'Near_Buy_Far_Sell_Discount_Rate': ['DiscountRate'],
                     'Near_Sell_Far_Buy_Ccy': ['FxRate'],
                     'Near_Sell_Far_Buy_Discount_Rate': ['DiscountRate']}

    documentation = ('FX and Equity', [
        'An FX swap is a combination of an FX forward deal with near settlement date $t_1$ and',
        'an FX forward deal in the opposite direction with far settlement date $t_2$, where $t_1 < t_2$.',
        'The base-currency value of the FX swap at time $t$, $t \\le t_1$, is',
        '',
        '$$A_1 \\tilde D(t,t_1) \\tilde X(t)-B_1 D(t,t_1)X(t) + B_2 \\tilde D(t,t_2) \\tilde X(t)-A_2 D(t,t_2) \\tilde X(t)$$',
        '',
        'where $A_1$ ($A_2$) is amount of the buy currency bought (sold) at $t_1$, ($t_2$), and $B_1$ ($B_2$)',
        'is amount of the sell currency sold (bought) at $t_1$ ($t_2$). Typically, $t_1$ is the spot',
        'settlement date and $A_2 = A_1$'])

    def __init__(self, params, valuation_options):
        super(FXSwapDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FXSwapDeal, self).reset()
        self.add_reval_dates({self.field['Near_Settlement_Date'], self.field['Far_Settlement_Date']},
                             self.field['Near_Buy_Far_Sell_Ccy'])
        self.add_reval_dates({self.field['Near_Settlement_Date'], self.field['Far_Settlement_Date']},
                             self.field['Near_Sell_Far_Buy_Ccy'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'NearBuyFarSell_Currency': utils.check_rate_name(self.field['Near_Buy_Far_Sell_Ccy']),
                 'NearBuyFarSell_DiscountRate': utils.check_rate_name(self.field['Near_Buy_Far_Sell_Discount_Rate']),
                 'NearSellFarBuy_Currency': utils.check_rate_name(self.field['Near_Sell_Far_Buy_Ccy']),
                 'NearSellFarBuy_DiscountRate': utils.check_rate_name(self.field['Near_Sell_Far_Buy_Discount_Rate'])}

        field_index = {'NearBuyFX': get_fxrate_factor(field['NearBuyFarSell_Currency'], static_offsets,
                                                      stochastic_offsets),
                       'NearBuyDiscount': get_discount_factor(field['NearBuyFarSell_DiscountRate'], static_offsets,
                                                              stochastic_offsets, all_tenors, all_factors),
                       'NearSellFX': get_fxrate_factor(field['NearSellFarBuy_Currency'], static_offsets,
                                                       stochastic_offsets),
                       'NearSellDiscount': get_discount_factor(field['NearSellFarBuy_DiscountRate'], static_offsets,
                                                               stochastic_offsets, all_tenors, all_factors),
                       'Maturity': (self.field['Far_Settlement_Date'] - base_date).days,
                       'Near_Maturity': (self.field['Near_Settlement_Date'] - base_date).days}

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        mtm = 0
        remaining_tenor = (factor_dep['Maturity'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1)

        near_buy_discount = utils.calc_time_grid_curve_rate(factor_dep['NearBuyDiscount'], deal_time, shared)
        near_sell_discount = utils.calc_time_grid_curve_rate(factor_dep['NearSellDiscount'], deal_time, shared)

        if factor_dep['Near_Maturity'] >= 0:
            near_deal_time = deal_time[:np.searchsorted(
                deal_time[:, utils.TIME_GRID_MTM], factor_dep['Near_Maturity'], side='right')]
            near = (factor_dep['Near_Maturity'] - near_deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1)

            NearBuy_rep = utils.calc_fx_cross(
                factor_dep['NearBuyFX'], shared.Report_Currency, near_deal_time, shared)
            NearSell_rep = utils.calc_fx_cross(
                factor_dep['NearSellFX'], shared.Report_Currency, near_deal_time, shared)

            near_sell_discount_rate = torch.squeeze(utils.calc_discount_rate(near_sell_discount, near, shared), axis=1)
            near_buy_discount_rate = torch.squeeze(utils.calc_discount_rate(near_buy_discount, near, shared), axis=1)

            mtm_near = self.field['Near_Buy_Amount'] * near_buy_discount_rate * NearBuy_rep - \
                       self.field['Near_Sell_Amount'] * near_sell_discount_rate * NearSell_rep

            mtm = F.pad(mtm_near, [0, 0, 0, remaining_tenor.size - near.size])

            pricing.cash_settle(
                shared, self.field['Near_Buy_Far_Sell_Ccy'],
                deal_data.Time_dep.fetch_index_by_day(factor_dep['Near_Maturity']), self.field['Near_Buy_Amount'])
            pricing.cash_settle(
                shared, self.field['Near_Sell_Far_Buy_Ccy'],
                deal_data.Time_dep.fetch_index_by_day(factor_dep['Near_Maturity']), -self.field['Near_Sell_Amount'])

        FX_NearBuy_rep = utils.calc_fx_cross(
            factor_dep['NearBuyFX'], shared.Report_Currency, deal_time, shared)
        FX_NearSell_rep = utils.calc_fx_cross(
            factor_dep['NearSellFX'], shared.Report_Currency, deal_time, shared)

        pricing.cash_settle(
            shared, self.field['Near_Sell_Far_Buy_Ccy'],
            deal_data.Time_dep.fetch_index_by_day(factor_dep['Maturity']), self.field['Far_Buy_Amount'])
        pricing.cash_settle(
            shared, self.field['Near_Buy_Far_Sell_Ccy'],
            deal_data.Time_dep.fetch_index_by_day(factor_dep['Maturity']), -self.field['Far_Sell_Amount'])

        far_buy_discount_rate = torch.squeeze(
            utils.calc_discount_rate(near_sell_discount, remaining_tenor, shared), axis=1)
        far_sell_discount_rate = torch.squeeze(
            utils.calc_discount_rate(near_buy_discount, remaining_tenor, shared), axis=1)

        mtm += self.field['Far_Buy_Amount'] * far_buy_discount_rate * FX_NearSell_rep - \
               self.field['Far_Sell_Amount'] * far_sell_discount_rate * FX_NearBuy_rep

        return mtm


class FXForwardDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Buy_Currency': ['FxRate'],
                     'Buy_Discount_Rate': ['DiscountRate'],
                     'Sell_Currency': ['FxRate'],
                     'Sell_Discount_Rate': ['DiscountRate']}

    documentation = ('FX and Equity', [
        'An FX forward is an agreement to buy an amount $A$ of one currency in exchange for an amount $B$ of another currency at settlement date $T$.',
        'The value of the deal in base currency at time $t$, ($t \\le T$), is',
        '',
        '$$A \\tilde D(t,T) \\tilde X(t)-BD(t,T)X(t)$$',
        '',
        'where',
        '',
        '- $D$ is the sell currency discount factor',
        '- $\\tilde D$ is the buy currency discount factor',
        '- $X$ is the price of the sell currency in base currency',
        '- $\\tilde X$ is the price of the buy currency in base currency'])

    def __init__(self, params, valuation_options):
        super(FXForwardDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FXForwardDeal, self).reset()
        self.add_reval_dates({self.field['Settlement_Date']}, self.field['Buy_Currency'])
        self.add_reval_dates({self.field['Settlement_Date']}, self.field['Sell_Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Buy_Currency': utils.check_rate_name(self.field['Buy_Currency'])}
        field['Buy_Discount_Rate'] = utils.check_rate_name(self.field['Buy_Discount_Rate']) if self.field[
            'Buy_Discount_Rate'] else field['Buy_Currency']
        field['Sell_Currency'] = utils.check_rate_name(self.field['Sell_Currency'])
        field['Sell_Discount_Rate'] = utils.check_rate_name(self.field['Sell_Discount_Rate']) if self.field[
            'Sell_Discount_Rate'] else field['Buy_Currency']

        field_index = {
            'BuyFX': get_fxrate_factor(field['Buy_Currency'], static_offsets, stochastic_offsets),
            'BuyDiscount': get_discount_factor(
                field['Buy_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SellFX': get_fxrate_factor(field['Sell_Currency'], static_offsets, stochastic_offsets),
            'SellDiscount': get_discount_factor(
                field['Sell_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Maturity': (self.field['Settlement_Date'] - base_date).days
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        # the tenors and the scenario grid should already be loaded up in the Buffer space in constant memory
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        FX_Buy_rep = utils.calc_fx_cross(
            factor_dep['BuyFX'], shared.Report_Currency, deal_time, shared)
        FX_Sell_rep = utils.calc_fx_cross(
            factor_dep['SellFX'], shared.Report_Currency, deal_time, shared)

        buy_discount = utils.calc_time_grid_curve_rate(factor_dep['BuyDiscount'], deal_time, shared)
        sell_discount = utils.calc_time_grid_curve_rate(factor_dep['SellDiscount'], deal_time, shared)
        remaining_tenor = (factor_dep['Maturity'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1)

        buy_discount_rate = torch.squeeze(utils.calc_discount_rate(buy_discount, remaining_tenor, shared), axis=1)
        sell_discount_rate = torch.squeeze(utils.calc_discount_rate(sell_discount, remaining_tenor, shared), axis=1)

        # settle the cash
        pricing.cash_settle(
            shared, self.field['Buy_Currency'], deal_data.Time_dep.deal_time_grid[-1], self.field['Buy_Amount'])
        pricing.cash_settle(
            shared, self.field['Sell_Currency'], deal_data.Time_dep.deal_time_grid[-1], self.field['Sell_Amount'])

        return self.field['Buy_Amount'] * FX_Buy_rep * buy_discount_rate - \
               self.field['Sell_Amount'] * FX_Sell_rep * sell_discount_rate


class StructuredDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate']}

    required_fields = {
        'Currency': 'ID of the FX rate price factor used to define the settlement currency. For example, USD.'
    }

    def __init__(self, params, valuation_options):
        super(StructuredDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(StructuredDeal, self).reset()

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        if node_children is not None:
            # have to reset the original instrument and let the child node decide
            for child in node_children:
                node_resets.update(child.get_reval_dates())

        self.reval_dates = node_resets
        for currency, dates in node_settlements.items():
            self.add_reval_dates(dates, currency)

        return super(StructuredDeal, self).finalize_dates(
            parser, base_date, grid, node_children, node_resets, node_settlements)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {}
        field['Currency'] = utils.check_rate_name(self.field['Currency'])

        field_index = {}
        field_index['Currency'] = get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets)

        return field_index

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        net_mtm = 0.0

        for child in child_dependencies:
            mtm = child.Instrument.calculate(shared, time_grid, child)
            net_mtm += mtm

        # return the interpolated value (without interpolating the time_grid)
        return pricing.interpolate(net_mtm, shared, time_grid, deal_data, interpolate_grid=False)

    def generate(self, shared, time_grid, deal_data):
        return 0.0


class SwapInterestDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Interest_Rate': ['InterestRate'],
                     'Interest_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol'],
                     'Discount_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol']}

    def __init__(self, params, valuation_options):
        super(SwapInterestDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(SwapInterestDeal, self).reset()
        self.paydates = generate_dates_backward(self.field['Maturity_Date'], self.field['Effective_Date'],
                                                self.field['Pay_Frequency'])
        self.recdates = generate_dates_backward(self.field['Maturity_Date'], self.field['Effective_Date'],
                                                self.field['Receive_Frequency'])
        self.add_reval_dates(self.paydates, self.field['Currency'])
        self.add_reval_dates(self.recdates, self.field['Currency'])
        # this swap could be quantoed - TODO
        self.isQuanto = None

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        logging.warning('SwapInterestDeal {0} - TODO'.format(self.field['Reference']))

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['Interest_Rate'] = utils.check_rate_name(
            self.field['Interest_Rate']) if self.field['Interest_Rate'] else field['Discount_Rate']

        field_index = {'SettleCurrency': self.field['Currency']}
        self.isQuanto = get_interest_rate_currency(field['Interest_Rate'], all_factors) != field['Currency']
        if self.isQuanto:
            # TODO - Deal with Quanto Interest Rate swaps
            pass
        else:
            field_index['Forward'] = get_interest_factor(
                field['Interest_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Discount'] = get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Currency'] = get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)

        if self.field['Pay_Rate_Type'] == 'Fixed':
            field_index['FixedCashflows'] = utils.generate_fixed_cashflows(
                base_date, self.paydates, -self.field['Principal'], self.field['Amortisation'],
                utils.get_day_count(self.field['Pay_Day_Count']), self.field['Swap_Rate'] / 100.0)
            field_index['FloatCashflows'] = utils.generate_float_cashflows(
                base_date, time_grid, self.recdates, self.field['Principal'], self.field['Amortisation'],
                self.field['Known_Rates'], self.field['Receive_Interest_Frequency'], self.field['Index_Tenor'],
                utils.get_day_count(self.field['Receive_Day_Count']), self.field['Floating_Margin'] / 10000.0)
        else:
            field_index['FixedCashflows'] = utils.generate_fixed_cashflows(
                base_date, self.recdates, self.field['Principal'], self.field['Amortisation'],
                utils.get_day_count(self.field['Receive_Day_Count']), self.field['Swap_Rate'] / 100.0)
            field_index['FloatCashflows'] = utils.generate_float_cashflows(
                base_date, time_grid, self.paydates, -self.field['Principal'], self.field['Amortisation'],
                self.field['Known_Rates'], self.field['Pay_Interest_Frequency'], self.field['Index_Tenor'],
                utils.get_day_count(self.field['Pay_Day_Count']), self.field['Floating_Margin'] / 10000.0)

        field_index['CompoundingMethod'] = self.field.get('Compounding_Method', 'None')
        field_index['InterestYieldVol'] = np.zeros(1, dtype=np.int32)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        fixed = deal_data.Factor_dep.copy()
        fixed['Cashflows'] = fixed['FixedCashflows']
        fixed['Compounding'] = self.field['Fixed_Compounding'] == 'Yes'
        fixed_leg = pricing.pv_fixed_leg(shared, time_grid, utils.DealDataType(
            Instrument=deal_data.Instrument, Factor_dep=fixed,
            Time_dep=deal_data.Time_dep, Calc_res=deal_data.Calc_res))

        float = deal_data.Factor_dep.copy()
        float['Cashflows'] = float['FloatCashflows']
        float['Model'] = pricing.pricer_float_cashflows
        float_leg = pricing.pv_float_leg(shared, time_grid, utils.DealDataType(
            Instrument=deal_data.Instrument, Factor_dep=float,
            Time_dep=deal_data.Time_dep, Calc_res=deal_data.Calc_res))

        return fixed_leg + float_leg


class CFFixedInterestListDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate']}

    documentation = (
        'Interest Rates', ['A series of fixed interest cashflows as described [here](#fixed-interest-cashflows)'])

    def __init__(self, params, valuation_options):
        super(CFFixedInterestListDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(CFFixedInterestListDeal, self).reset()

        paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        if self.field.get('Settlement_Date'):
            resetdates = {x for x in paydates if x < self.field['Settlement_Date']}
            resetdates.add(self.field['Settlement_Date'])
        else:
            resetdates = paydates

        self.add_reval_dates(resetdates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']

        field_index = {}
        buy_sell = 1 if self.field['Buy_Sell'] == 'Buy' else -1
        field_index['SettleCurrency'] = self.field['Currency']
        field_index['Currency'] = get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        field_index['Discount'] = get_discount_factor(
            field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        field_index['Cashflows'] = utils.make_fixed_cashflows(
            base_date, buy_sell, self.field['Cashflows'], self.field.get('Settlement_Date'))

        field_index['Compounding'] = self.field['Cashflows'].get('Compounding', 'No') == 'Yes'
        field_index['Settlement_Date'] = (self.field.get('Settlement_Date') -
                                          base_date).days if self.field.get('Settlement_Date') else None
        field_index['Settlement_Amount'] = self.field.get('Settlement_Amount', 0.0) * buy_sell

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_fixed_leg(shared, time_grid, deal_data)


class CFFixedListDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate']}

    def __init__(self, params, valuation_options):
        super(CFFixedListDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(CFFixedListDeal, self).reset()
        self.paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        self.add_reval_dates(self.paydates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']

        field_index = {
            'SettleCurrency': self.field['Currency'], 'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Cashflows': utils.make_simple_fixed_cashflows(
                base_date, 1 if self.field['Buy_Sell'] == 'Buy' else -1, self.field['Cashflows'])
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_fixed_leg(shared, time_grid, deal_data)


class FixedCashflowDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate']}

    documentation = (
        'Interest Rates', ['The time $t$ value of a fixed cashflow amount $C$ paid at time $T$ is $D(t,T)C$.'])

    def __init__(self, params, valuation_options):
        super(FixedCashflowDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FixedCashflowDeal, self).reset()
        self.add_reval_dates({self.field['Payment_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Amount': (1 if self.field.get('Buy_Sell', 'Buy') == 'Buy' else -1) * self.field['Amount'],
            'Payment_Date': (self.field['Payment_Date'] - base_date).days,
            'Local_Currency': self.field['Currency']
        }

        # needed for reporting
        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
        fx_rep = utils.calc_fx_cross(
            factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        discount_rates = torch.squeeze(
            utils.calc_discount_rate(
                discount, (factor_dep['Payment_Date'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1),
                shared),
            axis=1)

        mtm = factor_dep['Amount'] * discount_rates * fx_rep

        # settle the cashflow
        pricing.cash_settle(
            shared, self.field['Currency'], deal_data.Time_dep.deal_time_grid[-1], factor_dep['Amount'])

        return mtm


class CFFloatingInterestListDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Discount_Rate_Cap_Volatility': ['InterestRateVol'],
                     'Discount_Rate_Swaption_Volatility': ['InterestYieldVol'],
                     'Forecast_Rate': ['InterestRate'],
                     'Forecast_Rate_Cap_Volatility': ['InterestRateVol'],
                     'Forecast_Rate_Swaption_Volatility': ['InterestYieldVol']}

    documentation = (
        'Interest Rates', ['A series of floating interest cashflows as described [here](#floating-interest-cashflows)'])

    def __init__(self, params, valuation_options):
        super(CFFloatingInterestListDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(CFFloatingInterestListDeal, self).reset()
        reset_dates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        self.add_reval_dates(reset_dates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['Forecast_Rate'] = utils.check_rate_name(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else \
            field['Discount_Rate']

        field_index = {
            'SettleCurrency': self.field['Currency'],
            'Forward': get_interest_factor(
                field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'VolSurface': np.zeros(1, dtype=np.int32),
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        }

        float_cashflows = utils.make_float_cashflows(
            base_date, time_grid, 1 if self.field['Buy_Sell'] == 'Buy' else -1, self.field['Cashflows'])

        field_index['CompoundingMethod'] = self.field['Cashflows'].get('Compounding_Method', 'None')

        # check if the CompoundingMethod is null (None)
        if field_index['CompoundingMethod'] is None:
            field_index['CompoundingMethod'] = 'None'

        # potentially compress the cashflow list for faster computation
        if field_index['CompoundingMethod'] == 'None' and self.options.get('OIS_Cashflow_Group_Size', 0) > 0:
            field_index['Cashflows'] = utils.compress_no_compounding(
                float_cashflows, self.options['OIS_Cashflow_Group_Size'])
        else:
            field_index['Cashflows'] = float_cashflows

        field_index['Model'] = pricing.pricer_float_cashflows
        if self.field['Cashflows'].get('Properties'):
            first_prop = self.field['Cashflows']['Properties'][0]
            if first_prop.get('Cap_Multiplier', 0.0) or first_prop.get('Floor_Multiplier', 0.0):
                field_index['Model'] = pricing.pricer_cap if first_prop.get(
                    'Cap_Multiplier') is not None else pricing.pricer_floor
                field_index['VolSurface'] = get_interest_vol_factor(
                    utils.check_rate_name(self.field['Forecast_Rate_Cap_Volatility']), pd.DateOffset(months=3),
                    static_offsets, stochastic_offsets, all_tenors)

                field_index['Cashflows'].overwrite_rate(
                    utils.CASHFLOW_INDEX_Strike,
                    float(first_prop['Cap_Strike']) if
                    first_prop.get('Cap_Multiplier') is not None else float(first_prop['Floor_Strike']))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_float_leg(shared, time_grid, deal_data)


class YieldInflationCashflowListDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Index': ['InflationRate']}

    documentation = ('Inflation', ['This pays a fixed coupon on an inflation indexed principal. Define the following: ',
                                   '',
                                   '- $P$ the principal amount',
                                   '- $T_b$ the base reference date',
                                   '- $T_f$ the final reference date',
                                   '- $t_1$ the accrual start date',
                                   '- $t_2$ the accrual end date',
                                   '- $\\alpha$ the accrual daycount from $t_1$ to $t_2$',
                                   '- $r$ the fixed yield',
                                   '- $A\\ne 0$ is the rate multiplier',
                                   '',
                                   'The cashflow payoff is',
                                   '',
                                   '$$P\\Big(A\\frac{I_R(t,T_f)}{I_R(t,T_b)}\\Big)r\\alpha$$',
                                   '',
                                   ])

    def __init__(self, params, valuation_options):
        super(YieldInflationCashflowListDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(YieldInflationCashflowListDeal, self).reset()

        paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        if self.field.get('Is_Forward_Deal', 'No') == 'Yes':
            resetdates = {x for x in paydates if x < self.field['Settlement_Date']}
            resetdates.add(self.field['Settlement_Date'])
        else:
            resetdates = paydates

        self.add_reval_dates(resetdates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['Index'] = utils.check_rate_name(self.field['Index'])
        field['PriceIndex'] = utils.check_rate_name(get_inflation_index_name(field['Index'], all_factors))

        field_index = {
            'ForecastIndex': get_inflation_factor(
                field['Index'], static_offsets, stochastic_offsets, all_tenors),
            'PriceIndex': get_price_index_factor(
                field['PriceIndex'], static_offsets, stochastic_offsets),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets)
        }

        inflation_factor, index_factor = get_inflation_index_objects(field['Index'], field['PriceIndex'], all_factors)
        field_index['IndexMethod'] = utils.CASHFLOW_IndexMethodLookup[inflation_factor.get_reference_name()]

        field_index['Cashflows'], field_index['Base_Resets'], field_index['Final_Resets'] = utils.make_index_cashflows(
            base_date, time_grid, 1 if self.field['Buy_Sell'] == 'Buy' else -1, self.field['Cashflows'],
            inflation_factor, index_factor, self.field.get('Settlement_Date'))

        field_index['SettleCurrency'] = self.field['Currency']
        field_index['Settlement_Date'] = (self.field.get('Settlement_Date') -
                                          base_date).days if self.field.get('Settlement_Date') else None

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_index_leg(shared, time_grid, deal_data)


class CapDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Forecast_Rate': ['InterestRate'],
                     'Forecast_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol'],
                     'Discount_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol']}

    def __init__(self, params, valuation_options):
        super(CapDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(CapDeal, self).reset()
        self.resetdates = generate_dates_backward(
            self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Payment_Interval'])
        self.add_reval_dates(self.resetdates, self.field['Currency'])
        # this swap could be quantoed
        self.isQuanto = None

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        # have to reset the original instrument and let the child node decide
        super(CapDeal, self).reset()
        for currency, dates in node_settlements.items():
            self.add_reval_dates(dates, currency)
        return super(CapDeal, self).finalize_dates(parser, base_date, grid, node_children, node_resets,
                                                   node_settlements)

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        mtm_list = []

        for child in child_dependencies:
            # make the child price to the same grid as the parent
            child.Time_dep.assign(deal_data.Time_dep)
            # price the child
            mtm_list.append(child.Instrument.calculate(shared, time_grid, child))

        mtm = torch.sum(torch.stack(mtm_list), axis=0)

        # return the interpolated value (without interpolating the time_grid)
        return pricing.interpolate(mtm, shared, time_grid, deal_data, interpolate_grid=False)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['Forecast_Rate'] = utils.check_rate_name(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else \
            field['Discount_Rate']
        field['Forecast_Rate_Volatility'] = utils.check_rate_name(self.field['Forecast_Rate_Volatility'])

        field_index = {}
        Principal = self.field.get('Principal', 1000000.0)
        Amortisation = self.field.get('Amortisation')
        Known_Rates = self.field.get('Known_Rates')

        self.isQuanto = get_interest_rate_currency(field['Forecast_Rate'], all_factors) != field['Currency']
        if self.isQuanto:
            # TODO - Deal with Quanto Interest Rate swaps
            pass
        else:
            field_index['Forward'] = get_interest_factor(
                field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Discount'] = get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Currency'] = get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['VolSurface'] = get_interest_vol_factor(
                field['Forecast_Rate_Volatility'], self.field['Payment_Interval'],
                static_offsets, stochastic_offsets, all_tenors)
            field_index['Cashflows'] = utils.generate_float_cashflows(
                base_date, time_grid, self.resetdates,
                (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * Principal,
                Amortisation, Known_Rates, self.field['Index_Tenor'], self.field['Reset_Frequency'],
                utils.get_day_count(self.field['Accrual_Day_Count']), self.field['Cap_Rate'] / 100.0)

        return field_index


class FloorDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Forecast_Rate': ['InterestRate'],
                     'Forecast_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol'],
                     'Discount_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol']}

    def __init__(self, params, valuation_options):
        super(FloorDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FloorDeal, self).reset()
        self.resetdates = generate_dates_backward(self.field['Maturity_Date'], self.field['Effective_Date'],
                                                  self.field['Payment_Interval'])
        self.add_reval_dates(self.resetdates, self.field['Currency'])
        # this swap could be quantoed
        self.isQuanto = None

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        # have to reset the original instrument and let the child node decide
        super(FloorDeal, self).reset()
        for currency, dates in node_settlements.items():
            self.add_reval_dates(dates, currency)
        return super(FloorDeal, self).finalize_dates(
            parser, base_date, grid, node_children, node_resets, node_settlements)

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        mtm_list = []

        for child in child_dependencies:
            # make the child price to the same grid as the parent
            child.Time_dep.assign(deal_data.Time_dep)
            # price the child
            mtm_list.append(child.Instrument.calculate(shared, time_grid, child))

        mtm = torch.sum(torch.stack(mtm_list), axis=0)

        # return the interpolated value (without interpolating the time_grid)
        return pricing.interpolate(mtm, shared, time_grid, deal_data, interpolate_grid=False)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['Forecast_Rate'] = utils.check_rate_name(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else \
            field['Discount_Rate']
        field['Forecast_Rate_Volatility'] = utils.check_rate_name(self.field['Forecast_Rate_Volatility'])

        field_index = {}
        Principal = self.field.get('Principal', 1000000.0)
        Amortisation = self.field.get('Amortisation')
        Known_Rates = self.field.get('Known_Rates')

        self.isQuanto = get_interest_rate_currency(field['Forecast_Rate'], all_factors) != field['Currency']
        if self.isQuanto:
            # TODO - Deal with Quanto Interest Rate swaps
            pass
        else:
            field_index['Forward'] = get_interest_factor(
                field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Discount'] = get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Currency'] = get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['VolSurface'] = get_interest_vol_factor(
                field['Forecast_Rate_Volatility'], self.field['Payment_Interval'],
                static_offsets, stochastic_offsets, all_tenors)
            field_index['Cashflows'] = utils.generate_float_cashflows(
                base_date, time_grid, self.resetdates,
                (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * Principal,
                Amortisation, Known_Rates, self.field['Index_Tenor'], self.field['Reset_Frequency'],
                utils.get_day_count(self.field['Accrual_Day_Count']), self.field['Floor_Rate'] / 100.0)

        return field_index


class SwaptionDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Forecast_Rate': ['InterestRate'],
                     'Forecast_Rate_Volatility': ['InterestRateVol', 'InterestYieldVol']}

    documentation = ('Interest Rates',
                     ['Let $t_0, T_1$ and $T_2$ be the **Option Expiry Date**, **Swap Effective Date** and **Swap '
                      'Maturity Date** respectively of the swaption deal ($t_0 \\le T_1 \\lt T2$). If the deal is',
                      'cash settled, then let $T$ be the **Settlement Date**.',
                      '',
                      'The value of the underlying swap is',
                      '',
                      '$$U(t)=\\delta(V_{float}(t)-V_{fixed}(t))$$',
                      '',
                      'where $V_{float}(t)$ is the value of floating interest rate cashflows, $V_{fixed}(t)$ the',
                      'value of fixed interest cashflows and $\\delta$ is either $+1$ for payer swaptions and $-1$',
                      'for receiver swaptions.',
                      '',
                      'If the fixed leg has payments at times $t_2,...,t_n$, then the Present value of a Basis Point is',
                      '',
                      '$$F(t)=\\sum_{i=2}^n P_i \\alpha_i D(t,t_i)$$',
                      '',
                      'where $P_i$ is the principal amount and $\\alpha_i$ is the accrual year fraction for the',
                      '$i^{th}$ fixed interest cashflow. The forward swap rate is',
                      '',
                      '$$s(t)=\\frac{V_{float}(t)}{F(t)}.$$',
                      '',
                      'Define the *effective* strike rate as',
                      '',
                      '$$K(t)=\\frac{V_{fixed}(t)}{F(t)}$$',
                      '',
                      'Note that presently only zero-margin floating cashflow lists are supported (but this can be',
                      'extended). The value of the underlying swap is given by $U(t)=\\delta(s(t)-K(t))F(t)$. If',
                      'both fixed and floating cashflows have the same payment and accrual dates, then $K(t)=r$',
                      'where $r$ is the constant fixed rate on the fixed interest cashflow list.',
                      '',
                      '#### Physically Settled Swaptions',
                      '',
                      'If the **Settlement Style** is **Physical** and $U(t_0)\\ge 0$, then the option holder',
                      'receives the underlying swap and the value of the deal for $t\\ge t_0$ is $U(t)$. Note that',
                      'physical settlement has significant path dependency.',
                      '',
                      '#### Cash Settled Swaptions',
                      '',
                      'If the **Settlement Style** is **Cash**, then the option holder receives $\\max(U(t_0),0)$',
                      'on settlement date $T$. The value of the deal at $t\\lt t_0$ is',
                      '$F(t)\\mathcal B_\\delta(s(r),K(0),\\sigma\\sqrt{(t_0-t)})D(t_0,T)$. Note that this assumes a',
                      'lognormal distribution of the forecast rate and uses the Black Model as usual.',
                      '',
                      '#### Swap Rate Volatility',
                      '',
                      'Forward starting (where the effective date of the underlying swap is several months or years',
                      'after the option expiry) and  amortizing swaptions are not currently supported. This can be',
                      'extended as needed. Otherwise, $\\sigma$ is the volatility of the underlying rate at time $t$',
                      'for expiry $t_0$, tenor $\\tau=T_2-T_1$ and strike $K(0)$'
                      ])

    def __init__(self, params, valuation_options):
        super(SwaptionDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(SwaptionDeal, self).reset()
        self.add_reval_dates({self.field['Option_Expiry_Date']}, self.field['Currency'])
        # calc the underlying swap dates
        self.paydates = generate_dates_backward(self.field['Swap_Maturity_Date'], self.field['Swap_Effective_Date'],
                                                self.field['Pay_Frequency'])
        self.recdates = generate_dates_backward(self.field['Swap_Maturity_Date'], self.field['Swap_Effective_Date'],
                                                self.field['Receive_Frequency'])

        if self.field['Settlement_Style'] == 'Physical':
            self.add_reval_dates(self.paydates, self.field['Currency'])
            self.add_reval_dates(self.recdates, self.field['Currency'])
        else:
            self.add_reval_dates({self.field['Settlement_Date']}, self.field['Currency'])

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        # have to reset the original instrument and let the child node decide
        super(SwaptionDeal, self).reset()
        self.add_reval_dates({self.field['Option_Expiry_Date']}, self.field['Currency'])

        if self.field['Settlement_Style'] == 'Cash':
            # cash settle - do not include node cashflows
            node_resets.clear()
            node_settlements.clear()
            self.add_reval_dates({self.field['Settlement_Date']}, self.field['Currency'])
        else:
            self.add_reval_dates(node_resets, self.field['Currency'])

            for child in node_children:
                child.add_reval_dates({self.field['Option_Expiry_Date']}, self.field['Currency'])
                child.add_reval_date_offset(1)

        return super(SwaptionDeal, self).finalize_dates(parser, base_date, grid, node_children, node_resets,
                                                        node_settlements)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['Forecast_Rate'] = utils.check_rate_name(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else \
            field['Discount_Rate']
        field['Forecast_Rate_Volatility'] = utils.check_rate_name(self.field['Forecast_Rate_Volatility'])

        field_index = {'SettleCurrency': self.field['Currency'],
                       'Forward': get_interest_factor(
                           field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Discount': get_discount_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'VolSurface': get_interest_vol_factor(
                           field['Forecast_Rate_Volatility'], pd.DateOffset(years=2),
                           static_offsets, stochastic_offsets, all_tenors),
                       'Expiry': (self.field['Option_Expiry_Date'] - base_date).days}

        # need to check defaults
        Principal = self.field.get('Principal', 1000000.0)
        Pay_Amortisation = self.field.get('Pay_Amortisation')
        Receive_Amortisation = self.field.get('Receive_Amortisation')
        Receive_Day_Count = self.field.get('Receive_Day_Count', 'ACT_365')
        Pay_Day_Count = self.field.get('Pay_Day_Count', 'ACT_365')
        Floating_Margin = self.field.get('Floating_Margin', 0.0)
        Index_Day_Count = self.field.get('Index_Day_Count', 'ACT_365')

        if self.field['Payer_Receiver'] == 'Payer':
            field_index['FixedCashflows'] = utils.generate_fixed_cashflows(
                base_date, self.paydates, -Principal, Pay_Amortisation,
                utils.get_day_count(Pay_Day_Count), self.field['Swap_Rate'] / 100.0)
            field_index['FloatCashflows'] = utils.generate_float_cashflows(
                base_date, time_grid, self.recdates, Principal, Receive_Amortisation,
                None, self.field['Receive_Frequency'], self.field['Index_Tenor'],
                utils.get_day_count(Receive_Day_Count), Floating_Margin / 10000.0)
        else:
            field_index['FixedCashflows'] = utils.generate_fixed_cashflows(
                base_date, self.recdates, Principal, Receive_Amortisation,
                utils.get_day_count(Receive_Day_Count), self.field['Swap_Rate'] / 100.0)
            field_index['FloatCashflows'] = utils.generate_float_cashflows(
                base_date, time_grid, self.paydates, -Principal, Pay_Amortisation, None,
                self.field['Pay_Frequency'], self.field['Index_Tenor'],
                utils.get_day_count(Pay_Day_Count), Floating_Margin / 10000.0)

        # Cash settled?
        field_index['Cash_Settled'] = self.field['Settlement_Style'] != 'Physical'

        if self.field['Settlement_Style'] == 'Physical':
            # remember to potentially deliver the underlying swap if it's in the money
            field_index['FixedStartIndex'] = field_index['FixedCashflows'].get_cashflow_start_index(time_grid.time_grid)
            field_index['FloatStartIndex'] = field_index['FloatCashflows'].get_cashflow_start_index(time_grid.time_grid)
        else:
            field_index['FixedStartIndex'] = np.zeros(1, dtype=np.int32)
            field_index['FloatStartIndex'] = np.zeros(1, dtype=np.int32)

        # might want to change this
        field_index['Underlying_Swap_maturity'] = utils.get_day_count_accrual(
            base_date, (self.field['Swap_Maturity_Date'] - self.field['Swap_Effective_Date']).days,
            utils.get_day_count(Index_Day_Count))

        return field_index

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        mtm_list = []
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]
        child_map = {}

        for child in child_dependencies:
            # make the child price to the same grid as the parent
            child.Time_dep.assign(deal_data.Time_dep)
            child_map[child.Instrument.field['Object']] = child

        FX_rep = utils.calc_fx_cross(factor_dep['Currency'], shared.Report_Currency,
                                     deal_time, shared)

        delta = 1.0 if self.field['Payer_Receiver'] == 'Payer' else -1.0
        pvbp = -delta * pricing.pv_fixed_cashflows(
            shared, time_grid, child_map['CFFixedInterestListDeal'], ignore_fixed_rate=True, settle_cash=False)
        vfixed = -delta * pricing.pv_fixed_cashflows(
            shared, time_grid, child_map['CFFixedInterestListDeal'], ignore_fixed_rate=False, settle_cash=False)
        vfloat = delta * pricing.pv_float_cashflow_list(shared, time_grid, child_map['CFFloatingInterestListDeal'],
                                                        pricing.pricer_float_cashflows, settle_cash=False)
        if child_map['CFFloatingInterestListDeal'].Factor_dep[
               'Cashflows'].schedule[:, utils.CASHFLOW_INDEX_FloatMargin].any():
            # note that the margin index is the same as the fixed rate index - so this should pv the margin amounts
            vmargin = delta * pricing.pv_fixed_cashflows(
                shared, time_grid, child_map['CFFloatingInterestListDeal'], ignore_fixed_rate=False, settle_cash=False)
        else:
            vmargin = 0.0

        st = (vfloat - vmargin) / pvbp
        Kt = (vfixed - vmargin) / pvbp
        mn = st - Kt

        tenor = daycount_fn(factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

        if factor_dep['Cash_Settled']:
            vols = utils.calc_tenor_time_grid_vol_rate(
                factor_dep['VolSurface'], mn, tenor,
                factor_dep['Underlying_Swap_maturity'], shared)

            theo_price = utils.black_european_option(
                st, self.field['Swap_Rate'] / 100.0, vols, tenor,
                1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0, delta, shared)

            mtm = FX_rep * pvbp * theo_price

        else:
            expiry = tenor[tenor >= 0.0]
            counts = (expiry.size, tenor.size - expiry.size)

            F_option, F_swap = torch.split(pvbp, counts)
            spot_option, spot_swap = torch.split(st, counts)
            mn_option, mn_swap = torch.split(mn, counts)

            vols = utils.calc_tenor_time_grid_vol_rate(
                factor_dep['VolSurface'], mn_option, expiry,
                factor_dep['Underlying_Swap_maturity'], shared)

            theo_price = utils.black_european_option(
                spot_option, self.field['Swap_Rate'] / 100.0, vols, expiry,
                1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0, delta, shared)

            value = F_option * theo_price
            Ut_swap = delta * mn_swap * F_swap

            if Ut_swap.shape[0]:
                Ut_mask = Ut_swap * (Ut_swap[0] >= 0)
                mtm = FX_rep * torch.cat([value, Ut_mask], axis=0)
            else:
                mtm = FX_rep * value

        # if there's an amortization schedule, then we have to worry about adjusting the underlying tenor
        # for now, assume there's no such schedule (i.e. vanilla swaption)

        # interpolate the Theo price
        return pricing.interpolate(mtm, shared, time_grid, deal_data)

    def generate(self, shared, time_grid, deal_data):
        # Should just call the pricing function here - copy the pricing code for post_process,
        # put it in the pricing module, and call it here - also replace the post_process with
        # the moved pricing function - TODO
        raise Exception('generate in {0} - Not implemented yet'.format(self.__class__.__name__))


class FXDiscreteExplicitAsianOption(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Underlying_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'FX_Volatility': ['FXVol']}

    documentation = ('FX and Equity', ['A path independent option described [here](#discrete-asian-options)'])

    def __init__(self, params, valuation_options):
        super(FXDiscreteExplicitAsianOption, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FXDiscreteExplicitAsianOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['FX_Volatility'] = utils.check_rate_name(self.field['FX_Volatility'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_fx_vol_factor(
                field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'Samples': utils.make_sampling_data(base_date, time_grid, self.field['Sampling_Data']),
            'Strike': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        FX_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
        # get pricing data
        spot = utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0],
            deal_data.Factor_dep['Currency'][0], deal_time, shared)
        forward = utils.calc_fx_forward(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            deal_data.Factor_dep['Expiry'], deal_time, shared)

        mtm = pricing.pv_discrete_asian_option(
            shared, time_grid, deal_data, self.field['Underlying_Amount'], spot,
            forward, [deal_data.Factor_dep['Underlying_Currency'][0], deal_data.Factor_dep['Currency'][0]],
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'], use_forwards=True) * FX_rep

        return mtm


class FXDiscreteExplicitDoubleAsianOption(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Underlying_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'FX_Volatility': ['FXVol']}

    documentation = ('FX and Equity', ['A path independent option described [here](#discrete-double-asian-options)'])

    def __init__(self, params, valuation_options):
        super(FXDiscreteExplicitDoubleAsianOption, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FXDiscreteExplicitDoubleAsianOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['FX_Volatility'] = utils.check_rate_name(self.field['FX_Volatility'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_fx_vol_factor(
                field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'Alpha_0': self.field.get('Strike_Multiplier', 1.0),
            'Alpha_1': self.field.get('Sampling_Multiplier_1', 1.0),
            'Alpha_2': self.field.get('Sampling_Multiplier_2', 1.0),
            'Samples_1': utils.make_sampling_data(base_date, time_grid, self.field['Sampling_Data_1']),
            'Samples_2': utils.make_sampling_data(base_date, time_grid, self.field['Sampling_Data_2']),
            'Strike': self.field.get('Strike_Price', 0.0),
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        FX_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
        # get pricing data
        spot = utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0],
            deal_data.Factor_dep['Currency'][0], deal_time, shared)
        forward = utils.calc_fx_forward(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            deal_data.Factor_dep['Expiry'], deal_time, shared)

        mtm = pricing.pv_discrete_double_asian_option(
            shared, time_grid, deal_data, self.field['Underlying_Amount'], spot,
            forward, [deal_data.Factor_dep['Underlying_Currency'][0], deal_data.Factor_dep['Currency'][0]],
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'], use_forwards=True) * FX_rep

        return mtm


class EquityDiscreteExplicitAsianOption(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Equity': ['EquityPrice', 'DividendRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Equity_Volatility': ['EquityPriceVol']}

    documentation = ('FX and Equity', ['A path independent option described [here](#discrete-asian-options)'])

    def __init__(self, params, valuation_options):
        super(EquityDiscreteExplicitAsianOption, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(EquityDiscreteExplicitAsianOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['Equity_Volatility'] = utils.check_rate_name(self.field['Equity_Volatility'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_equity_price_vol_factor(
                field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Samples': utils.make_sampling_data(base_date, time_grid, self.field['Sampling_Data']),
            'Strike': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0
        }

        # map the past fixings
        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        FX_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency,
                                     deal_time, shared)
        # get pricing data
        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        forward = utils.calc_eq_forward(
            deal_data.Factor_dep['Equity'], deal_data.Factor_dep['Equity_Zero'],
            deal_data.Factor_dep['Dividend_Yield'], deal_data.Factor_dep['Expiry'], deal_time, shared)

        mtm = pricing.pv_discrete_asian_option(
            shared, time_grid, deal_data, self.field['Units'],
            spot, forward, [deal_data.Factor_dep['Equity']]) * FX_rep

        return mtm


class EquityOptionDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Equity': ['EquityPrice', 'DividendRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Equity_Volatility': ['EquityPriceVol']}

    documentation = ('FX and Equity', ['A vanilla option described [here](Definitions#european-options)'])

    def __init__(self, params, valuation_options):
        super(EquityOptionDeal, self).__init__(params, valuation_options)
        self.path_dependent = self.field['Option_Style'] == 'American'

    def reset(self, calendars):
        super(EquityOptionDeal, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.field['Currency'])

    def add_grid_dates(self, parser, base_date, grid):
        # we need to monitor the option for potential early exercise
        if isinstance(grid, str):
            grid_dates = parser(base_date, self.field['Expiry_Date'], grid)
            self.reval_dates.update(grid_dates)
            self.settlement_currencies.setdefault(self.field['Currency'], set()).update(grid_dates)
        else:
            for curr, cash_flow in self.settlement_currencies.items():
                last_pmt = max(cash_flow)
                delta = set([x for x in grid if x < last_pmt])
                self.settlement_currencies[curr].update(delta)
                self.reval_dates.update(delta)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['Equity_Volatility'] = utils.check_rate_name(self.field['Equity_Volatility'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_equity_price_vol_factor(
                field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Strike_Price': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Option_Style': self.field['Option_Style'],
            'Expiry': (self.field['Expiry_Date'] - base_date).days
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)

        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        forward = utils.calc_eq_forward(
            deal_data.Factor_dep['Equity'], deal_data.Factor_dep['Equity_Zero'],
            deal_data.Factor_dep['Dividend_Yield'], deal_data.Factor_dep['Expiry'], deal_time, shared)
        moneyness = spot / deal_data.Factor_dep['Strike_Price']

        if deal_data.Factor_dep['Option_Style'] == 'European':
            mtm = pricing.pv_european_option(
                shared, time_grid, deal_data, self.field['Units'], moneyness, forward) * fx_rep
        else:
            mtm = pricing.pv_american_option(
                shared, time_grid, deal_data, self.field['Units'], moneyness, spot, forward) * fx_rep

        return mtm


class QEDI_CustomAutoCallSwap(Deal):
    class EquityOptionDeal(Deal):
        factor_fields = {'Currency': ['FxRate'],
                         'Equity': ['EquityPrice', 'DividendRate'],
                         'Discount_Rate': ['DiscountRate'],
                         'Equity_Volatility': ['EquityPriceVol']}

        documentation = ('FX and Equity', ['An exotic option described [here](Definitions#QEDI-options)'])

        def __init__(self, params, valuation_options):
            super(QEDI_CustomAutoCallSwap, self).__init__(params, valuation_options)
            self.path_dependent = self.field['Option_Style'] == 'American'

        def reset(self, calendars):
            super(QEDI_CustomAutoCallSwap, self).reset()
            self.add_reval_dates({self.field['Expiry_Date']}, self.field['Currency'])

        def add_grid_dates(self, parser, base_date, grid):
            # we need to monitor the option for potential early exercise
            if isinstance(grid, str):
                grid_dates = parser(base_date, self.field['Expiry_Date'], grid)
                self.reval_dates.update(grid_dates)
                self.settlement_currencies.setdefault(self.field['Currency'], set()).update(grid_dates)
            else:
                for curr, cash_flow in self.settlement_currencies.items():
                    last_pmt = max(cash_flow)
                    delta = set([x for x in grid if x < last_pmt])
                    self.settlement_currencies[curr].update(delta)
                    self.reval_dates.update(delta)

        def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                              calendars):
            field = {'Currency': utils.check_rate_name(self.field['Currency']),
                     'Equity': utils.check_rate_name(self.field['Equity'])}
            field['Discount_Rate'] = utils.check_rate_name(
                self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
            field['Equity_Volatility'] = utils.check_rate_name(self.field['Equity_Volatility'])

            field_index = {
                'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                'SettleCurrency': self.field['Currency'],
                'Discount': get_discount_factor(
                    field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
                'Equity_Zero': get_equity_zero_rate_factor(
                    field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                'Dividend_Yield': get_dividend_rate_factor(
                    field['Equity'], static_offsets, stochastic_offsets, all_tenors),
                'Volatility': get_equity_price_vol_factor(
                    field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
                'Strike_Price': self.field['Strike_Price'],
                'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
                'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
                'Option_Style': self.field['Option_Style'],
                'Expiry': (self.field['Expiry_Date'] - base_date).days
            }

            return field_index

        def generate(self, shared, time_grid, deal_data):
            deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
            fx_rep = utils.calc_fx_cross(
                deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)

            spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
            forward = utils.calc_eq_forward(
                deal_data.Factor_dep['Equity'], deal_data.Factor_dep['Equity_Zero'],
                deal_data.Factor_dep['Dividend_Yield'], deal_data.Factor_dep['Expiry'], deal_time, shared)
            moneyness = spot / deal_data.Factor_dep['Strike_Price']

            if deal_data.Factor_dep['Option_Style'] == 'European':
                mtm = pricing.pv_european_option(
                    shared, time_grid, deal_data, self.field['Units'], moneyness, forward) * fx_rep
            else:
                mtm = pricing.pv_american_option(
                    shared, time_grid, deal_data, self.field['Units'], moneyness, spot, forward) * fx_rep

            return mtm


class EquityBarrierOption(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Equity': ['EquityPrice', 'DividendRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Equity_Volatility': ['EquityPriceVol']}

    documentation = ('FX and Equity', ['A path dependent option described [here](#single-barrier-options)'])

    def __init__(self, params, valuation_options):
        super(EquityBarrierOption, self).__init__(params, valuation_options)
        self.path_dependent = True

    def reset(self, calendars):
        super(EquityBarrierOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.field['Payoff_Currency'])

    def add_grid_dates(self, parser, base_date, grid):
        # a cash rebate is paid on touch if the option knocks out
        if self.field['Cash_Rebate']:
            if isinstance(grid, str):
                grid_dates = parser(base_date, self.field['Expiry_Date'], grid)
                self.reval_dates.update(grid_dates)
                self.settlement_currencies.setdefault(self.field['Payoff_Currency'], set()).update(grid_dates)
            else:
                for curr, cash_flow in self.settlement_currencies.items():
                    last_pmt = max(cash_flow)
                    delta = set([x for x in grid if x < last_pmt])
                    self.settlement_currencies[curr].update(delta)
                    self.reval_dates.update(delta)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['Equity_Volatility'] = utils.check_rate_name(self.field['Equity_Volatility'])

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Discount': get_discount_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
                       'Equity_Zero': get_equity_zero_rate_factor(
                           field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Dividend_Yield': get_dividend_rate_factor(
                           field['Equity'], static_offsets, stochastic_offsets, all_tenors),
                       'Volatility': get_equity_price_vol_factor(
                           field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
                       'Barrier_Monitoring': 0.5826 * np.sqrt(
                           (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days / 365.0),
                       'Expiry': (self.field['Expiry_Date'] - base_date).days}

        # discrete barrier monitoring requires adjusting the barrier by 0.58
        # ( -(scipy.special.zetac(0.5)+1)/np.sqrt(2.0*np.pi) ) * sqrt (monitoring freq)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = deal_data.Instrument.field['Units']
        payoff_currency = deal_data.Instrument.field['Payoff_Currency']

        eq_zer_curve = utils.calc_time_grid_curve_rate(deal_data.Factor_dep['Equity_Zero'], deal_time[:-1], shared)
        eq_div_curve = utils.calc_time_grid_curve_rate(deal_data.Factor_dep['Dividend_Yield'], deal_time[:-1], shared)

        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)

        # need to adjust if there's just 1 timepoint - i.e. base reval
        if time_grid.mtm_time_grid.size > 1:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
        else:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

        b = torch.squeeze(
            eq_zer_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False) -
            eq_div_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

        fx_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)

        pv = pricing.pv_barrier_option(shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency)
        mtm = pv * fx_rep

        return mtm


class EquityForwardDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Equity': ['EquityPrice', 'DividendRate'],
                     'Discount_Rate': ['DiscountRate']}

    documentation = ('FX and Equity', ['Described [here](Definitions#forwards)'])

    def __init__(self, params, valuation_options):
        super(EquityForwardDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(EquityForwardDeal, self).reset()
        self.add_reval_dates({self.field['Maturity_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        # field['Equity_Volatility']		= utils.check_rate_name(self.field['Equity_Volatility'])

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Discount': get_discount_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
                       'Equity_Zero': get_equity_zero_rate_factor(
                           field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Dividend_Yield': get_dividend_rate_factor(
                           field['Equity'], static_offsets, stochastic_offsets, all_tenors),
                       'Expiry': (self.field['Maturity_Date'] - base_date).days}

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
        forward = utils.calc_eq_forward(
            factor_dep['Equity'], factor_dep['Equity_Zero'],
            factor_dep['Dividend_Yield'], factor_dep['Expiry'], deal_time, shared)
        fx_rep = utils.calc_fx_cross(
            factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        nominal = (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * self.field['Units']

        discount_rates = torch.squeeze(utils.calc_discount_rate(
            discount, (factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1), shared),
            axis=1)

        cash = nominal * (forward - self.field['Forward_Price'])

        # store the settled cashflow
        pricing.cash_settle(shared, self.field['Currency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])

        return cash * discount_rates * fx_rep


class EquityDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Equity': ['EquityPrice']}

    documentation = ('FX and Equity', ['Described [here](Definitions#forwards)'])

    def __init__(self, params, valuation_options):
        super(EquityDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(EquityDeal, self).reset()
        self.add_reval_dates({self.field['Investment_Horizon']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity'])}

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
                       'Expiry': (self.field['Investment_Horizon'] - base_date).days}
        # TODO - Add more detail for dividend payments etc.
        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        spot = utils.calc_time_grid_spot_rate(factor_dep['Equity'], deal_time, shared)
        fx_rep = utils.calc_fx_cross(factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        nominal = (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * self.field['Units']
        cash = nominal * spot
        return cash * fx_rep


class EquitySwapletListDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Equity_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Equity': ['EquityPrice', 'DividendRate'],
                     'Equity_Volatility': ['EquityPriceVol']}

    documentation = ('FX and Equity', ['Described [here](#equity-swaps)'])

    def __init__(self, params, valuation_options):
        super(EquitySwapletListDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(EquitySwapletListDeal, self).reset()
        self.paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        self.add_reval_dates(self.paydates, self.field['Currency'])
        # this swap could be quantoed
        self.isQuanto = None

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['Equity_Currency'] = utils.check_rate_name(self.field['Equity_Currency'])

        effective_date = min([x['Start_Date'] for x in self.field['Cashflows']['Items']])
        start_dividend_sum = self.field['Known_Dividends'].sum_range(
            base_date, effective_date) if self.field.get('Known_Dividends') else 0.0

        field_index = {}
        self.isQuanto = field['Equity_Currency'] != field['Currency']

        field_index['PrincipleNotShares'] = 1 if self.field['Amount_Type'] == 'Principal' else 0
        field_index['SettleCurrency'] = self.field['Currency']

        if self.isQuanto:
            # TODO - Deal with Quanto swaps
            pass
        else:
            field_index['Currency'] = get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets)
            field_index['Discount'] = get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Equity'] = get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets)
            field_index['Dividend_Yield'] = get_dividend_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Equity_Zero'] = get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Flows'] = utils.make_equity_swaplet_cashflows(
                base_date, time_grid, 1 if self.field['Buy_Sell'] == 'Buy' else -1, self.field['Cashflows'])

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_equity_leg(shared, time_grid, deal_data)


class EquitySwapLeg(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Payoff_Currency': ['FxRate'],
                     'Equity_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Equity': ['EquityPrice', 'DividendRate']}

    documentation = ('FX and Equity', ['Described [here](#equity-swaps)'])

    def __init__(self, params, valuation_options):
        super(EquitySwapLeg, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(EquitySwapLeg, self).reset()
        self.bus_pay_day = calendars.get(self.field.get('Payment_Calendars', self.field['Accrual_Calendars']),
                                         {'businessday': pd.offsets.BDay(1)})['businessday']
        paydates = {self.field['Maturity_Date'] + self.bus_pay_day * int(self.field['Payment_Offset'])}
        self.add_reval_dates(paydates, self.field['Currency'])
        # this swap could be quantoed
        self.isQuanto = None

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {
            'Currency': utils.check_rate_name(self.field['Currency']),
            'Equity': utils.check_rate_name(self.field['Equity'])
        }
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] \
            else field['Currency']
        field['Payoff_Currency'] = utils.check_rate_name(self.field['Payoff_Currency'])

        # Implicitly we assume that units is the number of shares (Principle is 0) and that Dividend Timing is
        # "Terminal" - need to add support for dividends..
        start_prices = self.field['Equity_Known_Prices'].data.get(self.field['Effective_Date'], (None, None)) if \
            self.field['Equity_Known_Prices'] else (None, None)
        end_prices = self.field['Equity_Known_Prices'].data.get(self.field['Maturity_Date'], (None, None)) if \
            self.field['Equity_Known_Prices'] else (None, None)
        start_dividend_sum = self.field['Known_Dividends'].sum_range(base_date, self.field['Effective_Date']) if \
            self.field['Known_Dividends'] else 0.0

        # sometimes the equity price is listed but in the past - need to check for this
        if self.field['Equity_Known_Prices']:
            if start_prices == (None, None):
                earlier_dates = [x for x in self.field['Equity_Known_Prices'].data.keys() if
                                 x < self.field['Effective_Date']]
                start_prices = self.field['Equity_Known_Prices'].data[
                    max(earlier_dates)] if earlier_dates else (None, None)
            # if the end price is not provided, use the current spot price as a proxy
            if end_prices == (None, None):
                current_price = get_equity_spot(field['Equity'], static_offsets, stochastic_offsets, all_factors)
                fx_reset = 1.0 if field['Currency'] == field['Payoff_Currency'] else get_fxrate_spot(
                    field['Currency'], static_offsets, stochastic_offsets, all_factors) / get_fxrate_spot(
                    field['Payoff_Currency'], static_offsets, stochastic_offsets, all_factors)

                end_prices = [current_price, fx_reset]

        field['cashflow'] = {'Items':
            [{
                'Payment_Date': self.field['Maturity_Date'] + self.bus_pay_day * int(self.field['Payment_Offset']),
                'Start_Date': self.field['Effective_Date'],
                'End_Date': self.field['Maturity_Date'],
                'Start_Multiplier': 1.0,
                'End_Multiplier': 1.0,
                'Known_Start_Price': start_prices[0],
                'Known_Start_FX_Rate': start_prices[1],
                'Known_End_Price': end_prices[0],
                'Known_End_FX_Rate': end_prices[1],
                'Known_Dividend_Sum': start_dividend_sum,
                'Dividend_Multiplier': 1.0 if self.field['Include_Dividends'] == 'Yes' else 0.0,
                'Amount': self.field['Units']
                if self.field['Principal_Fixed_Variable'] == 'Variable' else self.field['Principal']
            }]
        }

        field_index = {'PrincipleNotShares': 1 if self.field['Principal_Fixed_Variable'] == 'Principal' else 0,
                       'SettleCurrency': self.field['Currency']}

        self.isQuanto = field['Payoff_Currency'] != field['Currency']

        if self.isQuanto:
            # TODO - Deal with Quanto Equity Swaps
            raise Exception("EquitySwapLeg Compo deal - TODO")
        else:
            field_index['Currency'] = get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets)
            field_index['Discount'] = get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Equity'] = get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets)
            field_index['Dividend_Yield'] = get_dividend_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Equity_Zero'] = get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Flows'] = utils.make_equity_swaplet_cashflows(
                base_date, time_grid, 1 if self.field['Buy_Sell'] == 'Buy' else -1, field['cashflow'])

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_equity_leg(shared, time_grid, deal_data)


class FXOneTouchOption(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Underlying_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'FX_Volatility': ['FXVol']}

    documentation = ('FX and Equity', [
        'A path dependent FX Option described [here](#one-touch-and-no-touch-binary-options-and-rebates)'])

    def __init__(self, params, valuation_options):
        super(FXOneTouchOption, self).__init__(params, valuation_options)
        self.path_dependent = True

    def reset(self, calendars):
        super(FXOneTouchOption, self).reset()
        self.payoff_ccy = self.field[self.field['Payoff_Currency']] if self.field['Payoff_Currency'] in self.field \
            else self.field['Payoff_Currency']
        self.add_reval_dates({self.field['Expiry_Date']}, self.payoff_ccy)

    def add_grid_dates(self, parser, base_date, grid):
        # only if the payoff is american (Touch) should we add potential payoff dates
        if self.field['Payment_Timing'] == 'Touch':
            if isinstance(grid, str):
                grid_dates = parser(base_date, self.field['Expiry_Date'], grid)
                self.reval_dates.update(grid_dates)
                self.settlement_currencies.setdefault(self.payoff_ccy, set()).update(grid_dates)
            else:
                # this is called if the grid is fully defined i.e. a set of dates
                for curr, cash_flow in self.settlement_currencies.items():
                    last_pmt = max(cash_flow)
                    delta = set([x for x in grid if x < last_pmt])
                    self.settlement_currencies[curr].update(delta)
                    self.reval_dates.update(delta)

    def add_reval_date_offset(self, offset, relative_to_settlement=True):
        # don't add any extra reval dates if this is a touch option
        if relative_to_settlement:
            if self.field['Payment_Timing'] != 'Touch':
                for curr, fixings in self.settlement_currencies.items():
                    new_dates = [x + pd.DateOffset(days=offset) for x in fixings]
                    self.reval_dates.update(new_dates)
        else:
            fixings = reduce(
                set.union, [{x + pd.DateOffset(days=ofs) for ofs in offset}
                            for x in self.reval_dates])
            self.reval_dates.update(fixings)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['FX_Volatility'] = utils.check_rate_name(self.field['FX_Volatility'])

        field_index = {'Currency': get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_fx_vol_factor(
                field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'Barrier_Monitoring': 0.5826 * np.sqrt(
                (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days / 365.0),
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])}
        # adjustment for discrete barrier monitoring
        # needed for reporting

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = deal_data.Instrument.field['Cash_Payoff']
        payoff_currency = deal_data.Instrument.payoff_ccy
        fx_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0],
                                     shared.Report_Currency, deal_time, shared)

        curr_curve = utils.calc_time_grid_curve_rate(
            deal_data.Factor_dep['Currency'][1], deal_time[:-1], shared)
        und_curr_curve = utils.calc_time_grid_curve_rate(
            deal_data.Factor_dep['Underlying_Currency'][1], deal_time[:-1], shared)

        spot = utils.calc_fx_cross(deal_data.Factor_dep['Underlying_Currency'][0],
                                   deal_data.Factor_dep['Currency'][0], deal_time, shared)

        # need to adjust if there's just 1 timepoint - i.e. base reval
        if time_grid.mtm_time_grid.size > 1:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
        else:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

        b = torch.squeeze(
            curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False) -
            und_curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

        pv = pricing.pv_one_touch_option(
            shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency,
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'], use_forwards=True)

        mtm = pv * fx_rep

        return mtm


class FXBarrierOption(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Underlying_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'FX_Volatility': ['FXVol']}

    documentation = ('FX and Equity', ['A path dependent FX Option described [here](#single-barrier-options)'])

    def __init__(self, params, valuation_options):
        super(FXBarrierOption, self).__init__(params, valuation_options)
        self.path_dependent = True

    def reset(self, calendars):
        super(FXBarrierOption, self).reset()
        self.payoff_ccy = self.field[self.field['Payoff_Currency']] if self.field['Payoff_Currency'] in self.field \
            else self.field['Payoff_Currency']
        self.add_reval_dates({self.field['Expiry_Date']}, self.payoff_ccy)

    def add_grid_dates(self, parser, base_date, grid):
        # a cash rebate is paid on touch if the option knocks out
        if self.field['Cash_Rebate']:
            if isinstance(grid, str):
                grid_dates = parser(base_date, self.field['Expiry_Date'], grid)
                self.reval_dates.update(grid_dates)
                self.settlement_currencies.setdefault(self.payoff_ccy, set()).update(grid_dates)
            else:
                # this is called if the grid is fully defined i.e. a set of dates
                for curr, cash_flow in self.settlement_currencies.items():
                    last_pmt = max(cash_flow)
                    delta = set([x for x in grid if x < last_pmt])
                    self.settlement_currencies[curr].update(delta)
                    self.reval_dates.update(delta)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['FX_Volatility'] = utils.check_rate_name(self.field['FX_Volatility'])

        field_index = {'Currency': get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_fx_vol_factor(
                field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Barrier_Monitoring': 0.5826 * np.sqrt(
                (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days / 365.0),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'settlement_currency': 1.0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])}

        # discrete barrier monitoring requires adjusting the barrier by 0.58
        # ( -(scipy.special.zetac(0.5)+1)/np.sqrt(2.0*np.pi) ) * sqrt (monitoring freq)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = deal_data.Instrument.field['Underlying_Amount']
        payoff_currency = deal_data.Instrument.payoff_ccy

        curr_curve = utils.calc_time_grid_curve_rate(
            deal_data.Factor_dep['Currency'][1], deal_time[:-1], shared)
        und_curr_curve = utils.calc_time_grid_curve_rate(
            deal_data.Factor_dep['Underlying_Currency'][1], deal_time[:-1], shared)

        spot = utils.calc_fx_cross(deal_data.Factor_dep['Underlying_Currency'][0],
                                   deal_data.Factor_dep['Currency'][0], deal_time, shared)

        # need to adjust if there's just 1 timepoint - i.e. base reval
        if time_grid.mtm_time_grid.size > 1:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
        else:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

        b = torch.squeeze(
            curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False) -
            und_curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)

        pv = pricing.pv_barrier_option(
            shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency,
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'], use_forwards=True)

        mtm = pv * fx_rep

        return mtm


class FXPartialTimeBarrierOption(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Underlying_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'FX_Volatility': ['FXVol']}

    documentation = ('FX and Equity', ['A partial path dependent FX Option described [here](#partial-barrier-options)'])

    def __init__(self, params, valuation_options):
        super(FXPartialTimeBarrierOption, self).__init__(params, valuation_options)
        self.path_dependent = True

    def reset(self, calendars):
        super(FXPartialTimeBarrierOption, self).reset()
        self.payoff_ccy = self.field.get('Payoff_Currency', self.field['Currency'])
        self.add_reval_dates({self.field['Expiry_Date'], self.field['Barrier_Limit_Date']}, self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['FX_Volatility'] = utils.check_rate_name(self.field['FX_Volatility'])

        field_index = {'Currency': get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_fx_vol_factor(
                field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Barrier_Monitoring': 0.5826 * np.sqrt(
                (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days / 365.0),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Limit_Date': (self.field['Barrier_Limit_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'settlement_currency': 1.0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])}

        # discrete barrier monitoring requires adjusting the barrier by 0.58
        # ( -(scipy.special.zetac(0.5)+1)/np.sqrt(2.0*np.pi) ) * sqrt (monitoring freq)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = deal_data.Instrument.field['Underlying_Amount']
        payoff_currency = deal_data.Instrument.payoff_ccy

        curr_curve = utils.calc_time_grid_curve_rate(
            deal_data.Factor_dep['Currency'][1], deal_time[:-1], shared)
        und_curr_curve = utils.calc_time_grid_curve_rate(
            deal_data.Factor_dep['Underlying_Currency'][1], deal_time[:-1], shared)

        spot = utils.calc_fx_cross(deal_data.Factor_dep['Underlying_Currency'][0],
                                   deal_data.Factor_dep['Currency'][0], deal_time, shared)

        # need to adjust if there's just 1 timepoint - i.e. base reval
        if time_grid.mtm_time_grid.size > 1:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
            tau1 = (deal_data.Factor_dep['Limit_Date'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
        else:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
            tau1 = (deal_data.Factor_dep['Limit_Date'] - deal_time[:, utils.TIME_GRID_MTM])

        b = torch.squeeze(
            curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False) -
            und_curr_curve.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), axis=1)

        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)

        pv = pricing.pv_partial_barrier_option(
            shared, time_grid, deal_data, nominal, spot, b, tau, tau1, payoff_currency,
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'])

        mtm = pv * fx_rep

        return mtm


class FXOptionDeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Underlying_Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'FX_Volatility': ['FXVol']}

    documentation = (
        'FX and Equity', ['A path independent vanilla FX Option described [here](Definitions#european-options)'])

    def __init__(self, params, valuation_options):
        super(FXOptionDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FXOptionDeal, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['FX_Volatility'] = utils.check_rate_name(self.field['FX_Volatility'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_fx_vol_factor(
                field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': field['Currency'][0] == field['FX_Volatility'][0],
            'Strike_Price': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])
        }

        # needed for reporting

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
        forward = utils.calc_fx_forward(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            deal_data.Factor_dep['Expiry'], deal_time, shared)

        moneyness = deal_data.Factor_dep['Strike_Price'] / forward if deal_data.Factor_dep['Invert_Moneyness'] \
            else forward / deal_data.Factor_dep['Strike_Price']

        mtm = pricing.pv_european_option(
            shared, time_grid, deal_data, self.field['Underlying_Amount'], moneyness, forward) * fx_rep

        return mtm


class DealDefaultSwap(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Name': ['SurvivalProb']}

    documentation = ('Credit',
                     ['This is a bilateral agreement where the buyer purchases protection from the seller against',
                      'default of a reference entity with period fixed payments. Should default of the reference',
                      'entity occur, the seller pays the buyer the default amount and payment ceases. The default',
                      'amount is $P(1-R)$, where $P$ is the principal amount. Note that there could also be an',
                      'accrued fee (but is currently ignored).',
                      '',
                      'Assuming the default payment does not occur prior to the effective date of the swap, the',
                      'value of this deal at $t$ is',
                      '',
                      '$$\\sum_{i=1}^n P_i(1-R)V_i(t)-\\sum_{i=1}^n P_i c\\alpha_i D(t,T_i)S(t,t_i)$$',
                      '',
                      'where $c$ is the fixed payment rate and',
                      '',
                      '$$V_i(t)=\\frac{\\bar h_i}{f_i+\\bar h_i}\\Big((D(t,\\tilde t_{i-1})S(t,\\tilde t_{i-1})-D(t,\\tilde t_i)S(t,\\tilde t_i)\\Big)$$',
                      '$$\\bar h_i=\\frac{1}{\\tilde t_i-\\tilde t_{i-1}}\\log\\Big(\\frac{S(t,\\tilde t_{i-1})}{S(t,\\tilde t_i)}\\Big)$$',
                      '$$f_i=\\frac{1}{\\tilde t_i-\\tilde t_{i-1}}\\log\\Big(\\frac{D(t,\\tilde t_{i-1})}{D(t,\\tilde t_i)}\\Big)$$',
                      '',
                      'Note that this is further approximated by',
                      '',
                      '$$V_i(t)=\\frac{D(t,\\tilde t_{i-1})+D(t,\\tilde t_i)}{2}\\Big(S(t,\\tilde t_{i-1})-S(t,\\tilde t_i)\\Big)$$',
                      '',
                      'which is accurate for low to moderate rates of default.'
                      ])

    def __init__(self, params, valuation_options):
        super(DealDefaultSwap, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(DealDefaultSwap, self).reset()
        bus_day = calendars.get(self.field['Calendars'], {'businessday': pd.offsets.Day(1)})['businessday']
        if list(self.field['Pay_Frequency'].kwds.values()) == [0]:
            self.resetdates = pd.DatetimeIndex([self.field['Effective_Date'], self.field['Maturity_Date']])
        else:
            self.resetdates = generate_dates_backward(
                self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Pay_Frequency'], bus_day=bus_day)
        self.add_reval_dates(self.resetdates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['Name'] = utils.check_rate_name(self.field['Name'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Name': get_survival_factor(field['Name'], static_offsets, stochastic_offsets, all_tenors),
            'Recovery_Rate': get_recovery_rate(field['Name'], all_factors)
        }

        pay_rate = self.field['Pay_Rate'] / 100.0 if isinstance(
            self.field['Pay_Rate'], float) else self.field['Pay_Rate'].amount

        field_index['Cashflows'] = utils.generate_fixed_cashflows(
            base_date, self.resetdates, (1 if self.field['Buy_Sell'] == 'Buy' else -1) * self.field['Principal'],
            self.field['Amortisation'], utils.get_day_count(self.field['Accrual_Day_Count']), pay_rate)

        # include the maturity date in the daycount
        field_index['Cashflows'].add_maturity_accrual(base_date, utils.get_day_count(self.field['Accrual_Day_Count']))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_cds_leg(shared, time_grid, deal_data)


class FRADeal(Deal):
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Interest_Rate': ['InterestRate']}

    def __init__(self, params, valuation_options):
        super(FRADeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FRADeal, self).reset()
        self.pay_date = self.field['Maturity_Date'] if self.field.get(
            'Payment_Timing', 'End') == 'End' else self.field['Effective_Date']
        self.add_reval_dates({self.pay_date}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['Interest_Rate'] = utils.check_rate_name(
            self.field['Interest_Rate']) if self.field['Interest_Rate'] else field['Discount_Rate']
        field['Reset_Date'] = self.field['Reset_Date'] if self.field.get('Reset_Date') else self.field['Effective_Date']
        field['Use_Known_Rate'] = self.field.get('Use_Known_Rate', 'No')
        field['Known_Rate'] = self.field.get('Known_Rate', 0.0)

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_discount_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Forward': get_interest_factor(
                field['Interest_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Daycount': utils.get_day_count(self.field['Day_Count']),
            'CompoundingMethod': 'None', 'SettleCurrency': self.field['Currency']
        }

        Accrual_fraction = utils.get_day_count_accrual(
            base_date, (self.field['Maturity_Date'] - self.field['Effective_Date']).days, field_index['Daycount'])

        cashflows = {'Items':
            [{
                'Payment_Date': self.pay_date,
                'Accrual_Start_Date': self.field['Effective_Date'],
                'Accrual_End_Date': self.field['Maturity_Date'],
                'Accrual_Year_Fraction': Accrual_fraction,
                'Notional': self.field['Principal'],
                'Margin': utils.Basis(-100.0 * self.field['FRA_Rate']),
                'Resets': [
                    [field['Reset_Date'], field['Reset_Date'],
                     self.field['Maturity_Date'], Accrual_fraction,
                     field['Use_Known_Rate'], field['Known_Rate']]
                ]
            }]
        }

        field_index['VolSurface'] = np.zeros(1, dtype=np.int32)
        field_index['Cashflows'] = utils.make_float_cashflows(
            base_date, time_grid, 1 if self.field['Borrower_Lender'] == 'Borrower' else -1, cashflows)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_float_leg(shared, time_grid, deal_data)


class FloatingEnergyDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Sampling_Type': ['ForwardPriceSample'],
                     'FX_Sampling_Type': ['ForwardPriceSample'],
                     'Reference_Type': ['ReferencePrice'],
                     'Payoff_Currency': ['FxRate']}

    documentation = ('Energy', [
        'The time $t$ value of an energy cashflow paid at $T$ indexed to volume $V$ of energy at a price determined by',
        'the reference price $S^R$ is',
        '',
        '$$V (A S^R(t,t_s^s,t_e^s,\\mathcal S)+b)D(t,T)$$',
        '',
        'where $A$ is the **Price Multiplier** and $b$ is the **Fixed Basis**.'
    ])

    def __init__(self, params, valuation_options):
        super(FloatingEnergyDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FloatingEnergyDeal, self).reset()
        paydates = set([x['Payment_Date'] for x in self.field['Payments']['Items']])
        self.add_reval_dates(paydates,
                             self.field['Payoff_Currency'] if self.field['Payoff_Currency'] else self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(
            self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
        field['Sampling_Type'] = utils.check_rate_name(self.field['Sampling_Type'])
        field['FX_Sampling_Type'] = utils.check_rate_name(self.field['FX_Sampling_Type']) if self.field[
            'FX_Sampling_Type'] else None
        field['Reference_Type'] = utils.check_rate_name(self.field['Reference_Type'])
        field['Payoff_Currency'] = utils.check_rate_name(self.field['Payoff_Currency']) if self.field[
            'Payoff_Currency'] else field['Currency']

        field_index = {}
        reference_factor, forward_factor = get_reference_factor_objects(field['Reference_Type'], all_factors)
        forward_sample = get_forwardprice_sampling(field['Sampling_Type'], all_factors)
        fx_sample = get_forwardprice_sampling(field['FX_Sampling_Type'], all_factors) if field[
            'FX_Sampling_Type'] else None

        field_index['ForwardPrice'], field_index['ForwardFX'], field_index['CashFX'] = get_forwardprice_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors,
            all_factors, reference_factor, forward_factor, base_date)

        field_index['Payoff_Currency'] = get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets)
        field_index['Discount'] = get_discount_factor(
            field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        field_index['Cashflows'] = utils.make_energy_cashflows(
            base_date, time_grid, -1.0 if self.field['Payer_Receiver'] == 'Payer' else 1.0,
            self.field['Payments'], reference_factor, forward_sample, fx_sample, calendars)

        field_index['SettleCurrency'] = self.field['Payoff_Currency']

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_energy_leg(shared, time_grid, deal_data)


class FixedEnergyDeal(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate']}

    documentation = ('Energy', [
        'The time $t$ value of a fixed energy cashflow paid at $T$ indexed to volume $V$ of energy at a fixed price $K$ is',
        '',
        '$$V K D(t,T)$$',
    ])

    def __init__(self, params, valuation_options):
        super(FixedEnergyDeal, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(FixedEnergyDeal, self).reset()
        self.paydates = set([x['Payment_Date'] for x in self.field['Payments']['Items']])
        self.add_reval_dates(self.paydates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']

        field_index = {'SettleCurrency': self.field['Currency'],
                       'Currency': get_fx_and_zero_rate_factor(
                           field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Discount': get_discount_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Cashflows': utils.make_energy_fixed_cashflows(
                           base_date, -1.0 if self.field['Payer_Receiver'] == 'Payer' else 1.0, self.field['Payments'])}

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_fixed_leg(shared, time_grid, deal_data)


class EnergySingleOption(Deal):
    # dependent price factors for this instrument
    factor_fields = {'Currency': ['FxRate'],
                     'Discount_Rate': ['DiscountRate'],
                     'Sampling_Type': ['ForwardPriceSample'],
                     'FX_Sampling_Type': ['ForwardPriceSample'],
                     'Reference_Type': ['ReferencePrice'],
                     'Reference_Volatility': ['ReferenceVol'],
                     'Payoff_Currency': ['FxRate']}

    documentation = ('Energy', [
        'A European option on an energy forward contract can be priced using the Black model with a volatility derived',
        'using the moment matching approach described earlier. Consider a European option with a reference price',
        '$S^R(t,t_s^s,t_e^s,\\mathcal S)$, where $t_s^s$ is the usual start of the sampling period and $t_e^s$ is the',
        'end of the period and also the expiry date of the option. The value of the option with strike $K$ and',
        'settlement date $T$ is',
        '',
        '$$\\mathcal B_\\delta (S^R(t,t_s^s,t_e^s,\\mathcal S),K,w(t,t_e^s,t_s^s,t_e^s,\\mathcal S))D(t,T)$$',
        '',
        'where $w(t,t_e^s,t_s^s,t_e^s,\\mathcal S)$ is the standard deviation of $S^R(t,t_s^s,t_e^s,\\mathcal S)$',
        'and $\\mathcal B_\\delta$ is the Black formula.'
    ])

    def __init__(self, params, valuation_options):
        super(EnergySingleOption, self).__init__(params, valuation_options)

    def reset(self, calendars):
        super(EnergySingleOption, self).reset()
        self.paydates = {self.field['Settlement_Date']}
        self.add_reval_dates(self.paydates,
                             self.field['Payoff_Currency'] if self.field['Payoff_Currency'] else self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = utils.check_rate_name(self.field['Discount_Rate']) if self.field['Discount_Rate'] else \
            field['Currency']
        field['Sampling_Type'] = utils.check_rate_name(self.field['Sampling_Type'])
        field['FX_Sampling_Type'] = utils.check_rate_name(self.field['FX_Sampling_Type']) if self.field[
            'FX_Sampling_Type'] else None
        field['Reference_Type'] = utils.check_rate_name(self.field['Reference_Type'])
        field['Reference_Volatility'] = utils.check_rate_name(self.field['Reference_Volatility'])
        field['Payoff_Currency'] = utils.check_rate_name(self.field['Payoff_Currency']) if self.field[
            'Payoff_Currency'] else field['Currency']

        field['cashflow'] = {'Payment_Date': self.field['Settlement_Date'],
                             'Period_Start': self.field['Period_Start'],
                             'Period_End': self.field['Period_End'],
                             'Volume': 1.0,
                             'Fixed_Price': self.field['Strike'],
                             'Realized_Average': self.field['Realized_Average'],
                             'FX_Period_Start': self.field['FX_Period_Start'],
                             'FX_Period_End': self.field['FX_Period_End'],
                             'FX_Realized_Average': self.field['FX_Realized_Average']}

        field_index = {}
        reference_factor, forward_factor = get_reference_factor_objects(field['Reference_Type'], all_factors)
        forward_sample = get_forwardprice_sampling(field['Sampling_Type'], all_factors)
        fx_sample = get_forwardprice_sampling(field['FX_Sampling_Type'], all_factors) if field[
            'FX_Sampling_Type'] else None
        forward_price_vol = get_forwardprice_vol(field['Reference_Volatility'], all_factors)

        if field['Currency'] != forward_factor.get_currency():
            fx_lookup = tuple(sorted([field['Currency'][0], forward_factor.get_currency()[0]]))
            field_index['FXCompoVol'] = get_fx_vol_factor(fx_lookup, static_offsets, stochastic_offsets, all_tenors)
            field_index['ImpliedCorrelation'] = get_implied_correlation(
                ('FxRate',) + fx_lookup, ('ReferencePrice',) + forward_price_vol, all_factors)

        # make a pricing cashflow
        cashflow = utils.make_energy_cashflows(base_date, time_grid, 1, {'Items': [field['cashflow']]},
                                               reference_factor,
                                               forward_sample, fx_sample, calendars)
        # turn it into a sampling object
        field_index['Cashflow'] = cashflow
        # store the base date in excel format
        field_index['Basedate'] = (base_date - reference_factor.start_date).days

        field_index['ForwardPrice'], field_index['ForwardFX'], field_index['CashFX'] = get_forwardprice_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors,
            all_factors, reference_factor, forward_factor, base_date)

        field_index['Payoff_Currency'] = get_fxrate_factor(
            field['Payoff_Currency'], static_offsets, stochastic_offsets)
        field_index['Discount'] = get_discount_factor(
            field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        field_index['ReferenceVol'] = get_forward_price_vol_factor(
            forward_price_vol, static_offsets, stochastic_offsets, all_tenors)
        field_index['SettleCurrency'] = self.field['Payoff_Currency']
        field_index['Buy_Sell'] = 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0
        field_index['Option_Type'] = 1.0 if self.field['Option_Type'] == 'Call' else -1.0
        field_index['Expiry'] = (self.field['Settlement_Date'] - base_date).days
        field_index['Strike'] = self.field['Strike']

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_energy_option(shared, time_grid, deal_data, self.field['Volume'])


def construct_instrument(param, all_valuation_options):
    if param.get('Object') not in globals():
        logging.error('Instrument {0} not defined'.format(param.get('Object')))
        return {}
    else:
        deal_options = all_valuation_options.get(param.get('Object'), {})
        return globals().get(param.get('Object'))(param, deal_options)


if __name__ == '__main__':
    pass
