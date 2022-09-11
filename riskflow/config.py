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

import os
import calendar
# import standard libraries
import json
import logging
import operator
# useful types
from collections import OrderedDict
from functools import reduce
# import parsing libraries
from typing import Dict, Any, Union
from xml.etree.ElementTree import ElementTree, Element

import numpy as np
import pandas as pd
from pyparsing import Literal, Word, nums, OneOrMore, delimitedList, oneOf, Optional, Group

# import libraries
from . import utils
from .bootstrappers import construct_bootstrapper
from .instruments import construct_instrument, Deal
from .stochasticprocess import construct_calibration_config, construct_process

# define datetime routines
Timestamp = pd.Timestamp
DateOffset = pd.DateOffset


def get_grid_grammar():
    """
    Contains the grammar definition rules for parsing the grid data
    """

    def push_int(strg, loc, toks):
        return int(toks[0])

    def push_single_period(strg, loc, toks):
        return Context.offset_lookup[toks[1]], toks[0]

    def push_period(strg, loc, toks):
        ofs = dict(toks.asList())
        return DateOffset(**ofs)

    def push_date_grid(strg, loc, toks):
        return toks[0][0] if len(toks) == 1 else utils.Offsets(toks.asList())

    lpar = Literal("(").suppress()
    rpar = Literal(")").suppress()
    decimal = Literal(".")

    integer = (Word("+-" + nums, nums) + ~decimal).setName('int').setParseAction(push_int)
    single_period = (integer + oneOf(['D', 'M', 'Y', 'W'], caseless=True)).setName('single_period').setParseAction(
        push_single_period)
    period = OneOrMore(single_period).setName('period').setParseAction(push_period)
    grid = delimitedList(Group(period + Optional(lpar + period + rpar)),
                         delim=' ').leaveWhitespace().setParseAction(push_date_grid)

    return grid, period


class ModelParams(object):
    def __init__(self, state=None):
        # valid risk factor subtypes (only Interest rates at the moment)
        self.valid_subtype = {'BasisSpread': 'BasisSpread'}
        # these models need these additional price factors
        self.implied_models = {'GBMAssetPriceTSModelImplied': 'GBMAssetPriceTSModelParameters',
                               'HullWhite2FactorImpliedInterestRateModel': 'HullWhite2FactorModelParameters'}

        self.modeldefaults = OrderedDict()
        self.modelfilters = OrderedDict()
        if state:
            defaults, filters = state
            for factor, model in defaults.items():
                self.append(factor, (), model)
            for factor, mappings in filters.items():
                for condition, model in mappings:
                    self.append(factor, tuple(condition), model)

    def append(self, price_factor, price_filter, stoch_proc):
        if price_filter == ():
            self.modeldefaults.setdefault(price_factor, stoch_proc)
        else:
            self.modelfilters.setdefault(price_factor, []).append((price_filter, stoch_proc))

    def write_to_file(self, file_handle):
        # write out the defaults first, then any filters that apply
        filters_written = set()
        for factor, model_name in self.modeldefaults.items():
            rules = self.modelfilters.get(factor)
            if rules:
                for (attrib, value), model in rules:
                    file_handle.write('{0}={1} where {2} = "{3}"\n'.format(factor, model, attrib, value))
                filters_written.add(factor)
            file_handle.write('{0}={1}\n'.format(factor, model_name))

        # make sure we write all the filters out
        for factor, rules in self.modelfilters.items():
            if factor not in filters_written:
                for (attrib, value), model in rules:
                    file_handle.write('{0}={1} where {2} = "{3}"\n'.format(factor, model, attrib, value))

    def additional_factors(self, model, factor):
        add_factor = self.implied_models.get(model)
        return utils.Factor(add_factor, factor.name) if add_factor else None

    def search(self, factor, actual_factor, ignore_subtype=False):
        """
        :param ignore_subtype:
        :param factor: Riskfactor of type utils.Factor
        :param actual_factor: corresponding dictionary of parameters for the factor loaded from a marketdata context
        :return: the model associated with the factor (taking any overrides into account)
        """
        price_factor_type = factor.type + ('' if ignore_subtype else self.valid_subtype.get(
            actual_factor.get('Sub_Type'), ''))

        # look for a filter rule
        rule = self.modelfilters.get(price_factor_type)
        if rule:
            factor_attribs = dict({k.lower(): v for k, v in actual_factor.items()}, **{'id': '.'.join(factor.name)})
            for (attrib, value), model in rule:
                if factor_attribs.get(attrib.strip().lower()) == value.strip():
                    return model
        return self.modeldefaults.get(price_factor_type)


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        return_value = {'.Unknown': str(type(obj))}
        if isinstance(obj, utils.Curve):
            return_value = {'.Curve': {'meta': obj.meta, 'data': obj.array.tolist()}}
        elif isinstance(obj, Deal):
            return_value = {'.Deal': obj.field}
        elif isinstance(obj, ModelParams):
            return_value = {'.ModelParams': {'modeldefaults': obj.modeldefaults, 'modelfilters': obj.modelfilters}}
        elif isinstance(obj, utils.Descriptor):
            return_value = {'.Descriptor': obj.data}
        elif isinstance(obj, utils.Percent):
            return_value = {'.Percent': 100.0 * obj.amount}
        elif isinstance(obj, utils.Basis):
            return_value = {'.Basis': 10000.0 * obj.amount}
        elif isinstance(obj, utils.Offsets):
            return_value = {'.Offsets': obj.data}
        elif isinstance(obj, utils.CreditSupportList):
            return_value = {'.CreditSupportList': [[rating, value] for rating, value in obj.data.items()]}
        elif isinstance(obj, utils.DateEqualList):
            return_value = {
                '.DateEqualList': [[date.strftime('%Y-%m-%d')] + list(value) for date, value in obj.data.items()]}
        elif isinstance(obj, utils.DateList):
            return_value = {'.DateList': [[date.strftime('%Y-%m-%d'), value] for date, value in obj.data.items()]}
        elif isinstance(obj, DateOffset):
            return_value = {'.DateOffset': obj.kwds}
        elif isinstance(obj, Timestamp):
            return_value = {'.Timestamp': obj.strftime("%Y-%m-%d")}
        else:
            logging.error('Error Saving file - Encoding object ' + str(obj) + ' failed')
        return return_value


class Context(object):
    """
    Reads (parses) a JSON market data and deals file.
    Also writes out these files once the data has been modified.
    Provides support for parsing grids of dates and working out dynamic dates for the
    given portfolio of deals.
    """

    # class level lookups
    month_lookup = dict((m, i) for i, m in enumerate(calendar.month_abbr) if m)
    offset_lookup = {'M': 'months', 'D': 'days', 'Y': 'years', 'W': 'weeks'}
    reverse_offset = {'months': 'M', 'days': 'D', 'years': 'Y', 'weeks': 'W'}

    def __init__(self, base_currency='USD'):
        self.deals = {'Deals': {'Children': []}}
        self.calibrations = {'CalibrationConfig': {'MarketDataArchiveFile': {}, 'Calibrations': []}}
        self.calendars = ElementTree(Element('Calendars'))
        self.holidays = {}
        self.archive = None
        self.archive_columns = {}

        # the default state of the system
        self.version = None
        self.params = {
            'System Parameters':
                {'Base_Currency': base_currency,
                 'Base_Date': '',
                 'Correlations_Healing_Method': 'Eigenvalue_Raising'
                 },
            'Model Configuration': ModelParams(),
            'Price Factors': {},
            'Price Factor Interpolation': ModelParams(),
            'Price Models': {},
            'Correlations': {},
            'Market Prices': {},
            'Valuation Configuration': {},
            'Bootstrapper Configuration': {}
        }  # type: Dict[str, Union[ModelParams, Dict[Any, Any]]]

        # make sure that there are no default calibration mappings
        self.calibration_process_map = {}
        self.gridparser, self.periodparser = get_grid_grammar()

    def parse_grid(self, run_date, max_date, grid, past_max_date=False):
        """
        Construct a set of dates (NOT adjusted to the next business day) as specified in the grid.
        Dates are capped at max_date (but may include the next date after if past_max_date is True)
        """

        parsed_grid = self.gridparser.parseString(grid)[0]
        offsets = parsed_grid.data if isinstance(parsed_grid, utils.Offsets) else [[parsed_grid]]
        fixed_dates = [(run_date + code[0], code[1] if len(code) > 1 else None) for code in offsets] + \
                      [(Timestamp.max, None)]
        dates = set()
        finish = False

        for date_rule, next_date_rule in zip(fixed_dates[:-1], fixed_dates[1:]):
            next_date = date_rule[0]
            if next_date > max_date:
                break
            else:
                dates.add(next_date)
            if date_rule[1]:
                while True:
                    next_date = next_date + date_rule[1]
                    if next_date > max_date:
                        finish = True
                        break
                    if next_date > next_date_rule[0]:
                        break
                    dates.add(next_date)

            if finish:
                break

        if past_max_date:
            dates.add(next_date)

        return dates

    def parse_calendar_file(self, filename):
        """
        Parses the xml calendar file in filename
        """
        self.holidays = {}
        self.calendars = ElementTree(file=filename)
        for elem in self.calendars.iter():
            if elem.attrib.get('Location'):
                holidays = dict(tuple(x.split("|")) for x in elem.attrib['Holidays'].split(', '))
                self.holidays[elem.attrib['Location']] = {
                    'businessday': pd.tseries.offsets.CustomBusinessDay(
                        holidays=holidays.keys(), weekmask=utils.WeekendMap[elem.attrib['Weekends']]),
                    'holidays': holidays}

    def fetch_all_calibration_factors(self, override={}):
        """
        Assumes valid marketdata and calibration config files have been loaded (via parse_json)
        and returns the list of price factors that have mapped price models.
        The return value of this method is suitable to pass to the calibrate_factors method.
        """
        model_factor = {}
        for factor in self.params.get('Price Factors', {}):
            price_factor = utils.check_rate_name(factor)
            model_config = self.params['Model Configuration'].search(utils.Factor(price_factor[0], price_factor[1:]),
                                                                     self.params['Price Factors'][factor])
            model = override.get(model_config, model_config)
            if model:
                subtype = self.params['Price Factors'][factor].get('Sub_Type')
                model_name = utils.Factor(model, price_factor[1:])
                archive_name = utils.Factor(price_factor[0] + (subtype if subtype and subtype != "None" else ''),
                                            price_factor[1:])
                model_factor[factor] = utils.RateInfo(utils.check_tuple_name(model_name),
                                                      utils.check_tuple_name(archive_name),
                                                      self.calibration_process_map.get(model))

        remaining_factor = {}
        remaining_rates = set([col.split(',')[0] for col in self.archive.columns]).difference(model_factor.keys())
        for factor in remaining_rates:
            price_factor = utils.check_rate_name(factor)
            model = self.params['Model Configuration'].search(utils.Factor(price_factor[0], price_factor[1:]), {})
            if model:
                model_name = utils.Factor(model, price_factor[1:])
                archive_name = utils.Factor(price_factor[0], price_factor[1:])
                remaining_factor[factor] = utils.RateInfo(utils.check_tuple_name(model_name),
                                                          utils.check_tuple_name(archive_name),
                                                          self.calibration_process_map.get(model))

        return {'present': model_factor, 'absent': remaining_factor}

    def bootstrap(self):
        """
        Runs all the bootstrappers - this happens in one process with debugging on by default. For multiprocessing
        bootstrapping, call the construct_bootstrapper method directly
        """
        # need to implement ordered dicts in the params obj - TODO
        for bootstrapper_name, params in sorted(self.params['Bootstrapper Configuration'].items()):
            # need parsers here - but for now, can just use the name to know what to do
            try:
                bootstrapper = construct_bootstrapper(bootstrapper_name, params)
            except:
                logging.warning('Cannot execute Bootstrapper for {0} - skipping'.format(bootstrapper_name))
                continue

            # run the bootstrapper on the market prices and store them in the price factors/price models
            bootstrapper.bootstrap(self.params['System Parameters'],
                                   self.params['Price Models'],
                                   self.params['Price Factors'],
                                   self.params['Price Factor Interpolation'],
                                   self.params['Market Prices'],
                                   self.holidays,
                                   debug=self)

    def calibrate_factors(self, from_date, to_date, factors, smooth=0.0, correlation_cuttoff=0.2,
                          overwrite_correlations=True):
        """
        Assumes a valid calibration JSON configuration file is loaded first, then proceeds to strip out only data
        between from_date and to_date. The calibration rules as specified by the calibration configuration file is then
        applied to the factors given. Note that factors needs to be a list of utils.RateInfo (obtained via a call to
        fetch_all_calibration_factors). Also note that this method overwrites the Price Model section of the config
        file in memory. To save the changes, an explicit call to write_marketdata_json must be made.
        """
        correlation_names = []
        consolidated_df = None
        ak = []
        num_indexes = 0
        num_factors = 0
        total_rates = reduce(operator.concat,
                             [self.archive_columns[rate.archive_name] for rate in factors.values()], [])
        factor_data = utils.filter_data_frame(self.archive, from_date, to_date)[total_rates]

        for rate_name, rate_value in sorted(factors.items()):
            df = factor_data[[col for col in factor_data.columns if col.split(',')[0] == rate_value.archive_name]]
            # now remove spikes
            data_frame = df[np.abs(df - df.median()) <= (smooth * df.std())].interpolate(method='index') \
                if smooth else df
            # log it
            logging.info(
                'Calibrating {0} (archive name {1}) with raw shape {2} and cleaned non-null shape {3}'.format(
                    rate_name, rate_value.archive_name, str(df.shape), str(data_frame.dropna().shape)))

            # calibrate
            try:
                result = rate_value.calibration.calibrate(data_frame, num_business_days=252.0)
            except:
                logging.error('Data errors in factor {0} resulting in flawed calibration - skipping'.format(
                    rate_value.archive_name))
                continue

            # check that it makes sense . . .
            if (np.array(result.correlation).max() > 1) or (np.array(result.correlation).min() < -1) or (
                    result.delta.std() == 0).any():
                logging.error('Data errors in factor {0} resulting in incorrect correlations - skipping'.format(
                    rate_value.archive_name))
                continue

            model_tuple = utils.check_rate_name(rate_value.model_name)
            model_factor = utils.Factor(model_tuple[0], model_tuple[1:])

            # get the correlation name
            process_name, addons = construct_process(model_factor.type, None, result.param).correlation_name

            for sub_factors in addons:
                correlation_names.append(
                    utils.check_tuple_name(utils.Factor(process_name, model_factor.name + sub_factors)))

            consolidated_df = result.delta if consolidated_df is None else pd.concat(
                [consolidated_df, result.delta], axis=1)
            # store the correlation coefficients
            ak.append(result.correlation)

            # store result back in market data file
            self.params['Price Models'][rate_value.model_name] = result.param

            num_indexes += result.delta.shape[1]
            num_factors += rate_value.calibration.num_factors

        a = np.zeros((num_factors, num_indexes))
        rho = consolidated_df.corr()
        offset = 0
        row_num = 0
        for coeff in ak:
            for factor_index, factor in enumerate(coeff):
                a[row_num + factor_index, offset:offset + len(factor)] = factor
            row_num += len(coeff)
            offset += len(factor)

        # cheating here
        factor_correlations = a.dot(rho).dot(a.T).clip(-1.0, 1.0)

        # see if we need to delete the old correlations
        if overwrite_correlations:
            self.params['Correlations'] = {}

        for index1 in range(len(correlation_names) - 1):
            for index2 in range(index1 + 1, len(correlation_names)):
                if np.fabs(factor_correlations[index1, index2]) > correlation_cuttoff:
                    self.params['Correlations'][(correlation_names[index1], correlation_names[index2])] = \
                        factor_correlations[index1, index2]

    def calculate_dependencies(self, options, base_date, base_MTM_dates, calc_dates=True):
        """
        Works out the risk factors (and risk models) in the given set of deals.
        These factors are cross-referenced in the marketdata file and matched by
        name. This can be extended as needed.

        Returns the dependant factors, the stochastic models, the full list of
        reset dates and optionally the potential currency settlements
        """

        def get_rates(factor, instrument):
            rates_to_add = {factor: []}
            factor_name = utils.check_tuple_name(factor)

            if factor.type in dependant_fields:
                linked_factors = [utils.Factor(dep_field[1], utils.check_rate_name(
                    self.params['Price Factors'][factor_name][dep_field[0]])) for dep_field in
                                  dependant_fields[factor.type] if
                                  self.params['Price Factors'][factor_name][dep_field[0]]]
                for linked_factor in linked_factors:
                    # add it assuming no dependencies
                    rates_to_add.setdefault(linked_factor, [])
                    # update and dependencies
                    if linked_factor.type in dependant_fields:
                        rates_to_add.update(get_rates(linked_factor, instrument))
                    # link it to the original factor
                    rates_to_add[factor].append(linked_factor)

                    # check that we include any nested factors
                    if linked_factor.type in nested_fields:
                        for i in range(1, len(linked_factor.name) + 1):
                            rates_to_add.update({utils.Factor(linked_factor.type, linked_factor.name[:i]): [
                                utils.Factor(linked_factor.type, linked_factor.name[:i - 1])] if i > 1 else []})

            if factor.type in nested_fields:
                for i in range(1, len(factor.name) + 1):
                    rates_to_add.update({utils.Factor(factor.type, factor.name[:i]): [
                        utils.Factor(factor.type, factor.name[:i - 1])] if i > 1 else []})

            if factor.type in conditional_fields:
                for conditional_factor in conditional_fields[factor.type](
                        instrument, self.params['Price Factors'][factor_name], self.params['Price Factors']):
                    rates_to_add[factor].append(conditional_factor)
                    rates_to_add.update({conditional_factor: []})

            return rates_to_add

        def walk_groups(deals, price_factors, factor_tenors):

            def get_price_factors(rates_to_add, rate_tenors, instrument):
                reval_dates = instrument.get_reval_dates()
                for field_name, factor_types in instrument.factor_fields.items():
                    for field_value in utils.get_fieldname(field_name, instrument.field):
                        for factor in [utils.Factor(factor_type, utils.check_rate_name(field_value))
                                       for factor_type in factor_types]:
                            # store the max tenor for just the factor alone
                            if reval_dates:
                                rate_tenors.setdefault(factor, set()).update({max(reval_dates)})
                            # now look at dependent factors
                            if factor not in rates_to_add:
                                try:
                                    rates_to_add.update(get_rates(factor, instrument))
                                except KeyError as e:
                                    logging.warning('Price Factor {0} not found in market data'.format(e))
                                    if factor.type == 'DiscountRate':
                                        logging.info('Creating default Discount {0}'.format(factor))
                                        self.params['Price Factors'][utils.check_tuple_name(factor)] = OrderedDict(
                                            [('Interest_Rate', '.'.join(factor.name))])
                                        # try again
                                        rates_to_add.update(get_rates(factor, instrument))
                                    else:
                                        logging.error('Skipping Price Factor')

            resets = {base_date}
            children = []
            settlement_currencies = {}

            for node in deals:
                # get the instrument
                instrument = node['Instrument']

                if node.get('Ignore') == 'True':
                    continue

                # get a list of children ready to pass back to the parent
                children.append(instrument)

                if node.get('Children'):
                    node_children, node_resets, node_settlements = walk_groups(
                        node['Children'], price_factors, factor_tenors)

                    # sort out dates and calendars
                    instrument.reset(self.holidays)

                    if calc_dates:
                        instrument.finalize_dates(self.parse_grid, base_date, base_MTM_dates, node_children,
                                                  node_resets, node_settlements)
                    # get it's price factors
                    get_price_factors(price_factors, factor_tenors, instrument)

                    # merge dates
                    resets.update(node_resets)
                    for key, value in node_settlements.items():
                        settlement_currencies.setdefault(key, set()).update(value)
                else:
                    # sort out dates and calendars
                    instrument.reset(self.holidays)

                    if calc_dates:
                        instrument.finalize_dates(self.parse_grid, base_date, base_MTM_dates, None, resets,
                                                  settlement_currencies)
                    # get its price factors
                    get_price_factors(price_factors, factor_tenors, instrument)

            return children, resets, settlement_currencies

        def add_interest_rate(curve_name):
            interest_rate_factor = utils.Factor(
                'InterestRate', utils.check_rate_name(curve_name))

            # make sure to add the reset dates to this interest rate (and dependents)
            dependent_factor_tenors[interest_rate_factor] = reset_dates
            dependent_factors.update(get_rates(interest_rate_factor, {}))
            for sub_factor in utils.traverse_dependents(interest_rate_factor, dependent_factors):
                dependent_factor_tenors[sub_factor] = reset_dates

        # derived fields are fields that embed other risk factors
        dependant_fields = {'FxRate': [('Interest_Rate', 'InterestRate')],
                            'DiscountRate': [('Interest_Rate', 'InterestRate')],
                            'ForwardPrice': [('Currency', 'FxRate')],
                            'ReferencePrice': [('ForwardPrice', 'ForwardPrice')],
                            'ReferenceVol': [('ForwardPriceVol', 'ForwardPriceVol'),
                                             ('ReferencePrice', 'ReferencePrice')],
                            'InflationRate': [('Price_Index', 'PriceIndex')],
                            'EquityPrice': [('Interest_Rate', 'InterestRate'), ('Currency', 'FxRate')]}

        # nested fields need to include all their children
        nested_fields = {'InterestRate'}

        # conditional fields need to potentially include correlation and fx vol surfaces (e.g. reference prices)
        conditional_fields = {
            'ReferenceVol': lambda instrument, factor_fields, params:
            [utils.Factor('Correlation', tuple('FxRate.{0}.{1}/ReferencePrice.{2}.{0}'.format(
                params[utils.check_tuple_name(utils.Factor('ForwardPrice', (instrument.field['Reference_Type'],)))]
                ['Currency'], instrument.field['Currency'], instrument.field['Reference_Type']).split('.')))] if
            instrument.field['Currency'] != params[utils.check_tuple_name(
                utils.Factor('ForwardPrice', (instrument.field['Reference_Type'],)))]['Currency'] else [],
            'ForwardPrice': lambda instrument, factor_fields, params: [utils.Factor('FXVol', tuple(
                sorted([instrument.field['Currency'], factor_fields['Currency']])))] if
            instrument.field['Currency'] != factor_fields['Currency'] else []}

        # the list of returned factors
        dependent_factors = set()
        stochastic_factors = OrderedDict()
        additional_factors = OrderedDict()

        # complete list of reset dates referenced
        reset_dates = set()
        # complete list of currency settlement dates
        currency_settlement_dates = {}

        # only if we have a portfolio of trades can we calculate its dependencies
        if self.deals:
            # get the reporting currency
            report_currency = options['Currency']
            # get the base currency factor
            base_factor = utils.Factor(
                'FxRate', utils.check_rate_name(self.params['System Parameters']['Base_Currency']))

            # add the base Fx rate
            dependent_factors = get_rates(base_factor, {})

            # store the max date the factor is needed
            dependent_factor_tenors = {}

            # grab all the factor fields in the portfolio
            dependant_deals, reset_dates, currency_settlement_dates = walk_groups(
                self.deals['Deals']['Children'], dependent_factors, dependent_factor_tenors)

            # additional factors from the options passed in
            report_factor = utils.Factor('FxRate', utils.check_rate_name(report_currency))
            report_currency_dependencies = get_rates(report_factor, {})

            # add the base dependencies to the reporting currency
            if report_factor != base_factor:
                report_currency_dependencies[report_factor].append(base_factor)

            dependent_factors.update(report_currency_dependencies)
            # make sure the reporting currency is around till the end
            dependent_factor_tenors[report_factor] = reset_dates
            for reporting_factor in utils.traverse_dependents(report_factor, dependent_factors):
                dependent_factor_tenors[reporting_factor] = reset_dates

            # check if we need to fetch survival data for CVA
            if options.get('CVA'):
                dependent_factors.update(get_rates(
                    utils.Factor('SurvivalProb', utils.check_rate_name(options['CVA']['Counterparty'])),
                    {})
                )
            # check if we need to fetch curve data for FVA
            if options.get('FVA'):
                # add curves
                add_interest_rate(options['FVA']['Funding_Interest_Curve'])
                add_interest_rate(options['FVA']['Risk_Free_Curve'])

                # need to weight the FVA by the survival prob of the counterparty (if defined)
                if 'Counterparty' in options['FVA']:
                    dependent_factors.update(get_rates(
                        utils.Factor('SurvivalProb', utils.check_rate_name(options['FVA']['Counterparty'])), {})
                    )
            # Check deflation
            if options.get('Deflation_Interest_Rate'):
                add_interest_rate(options['Deflation_Interest_Rate'])

            # update the linked factor max tenors
            missing_tenors = {}
            for k, v in dependent_factor_tenors.items():
                for linked_factor in utils.traverse_dependents(k, dependent_factors):
                    missing_tenors.setdefault(linked_factor, set()).update(v)

            # make sure the base currency is always first
            for k, v in dependent_factors.items():
                if k.type == 'FxRate' and k != base_factor and base_factor not in v:
                    v.append(base_factor)

            # now sort the factors taking any factor dependencies into account
            sorted_factors = utils.topological_sort(dependent_factors)
            # merge missing tenors
            for k, v in missing_tenors.items():
                dependent_factor_tenors.setdefault(k, set()).update(v)
            # now get the last tenor for each factor
            dependent_factors = {k: max(dependent_factor_tenors.get(k, reset_dates)) for k in sorted_factors}

            # now lookup the processes
            for factor in sorted_factors:
                stoch_proc = self.params['Model Configuration'].search(factor, self.params['Price Factors'].get(
                    utils.check_tuple_name(factor), {}))
                # might need implied parameters
                additional_factor = self.params['Model Configuration'].additional_factors(stoch_proc, factor)
                if stoch_proc and factor.name[0] != self.params['System Parameters']['Base_Currency']:
                    factor_model = utils.Factor(stoch_proc, factor.name)

                    if utils.check_tuple_name(factor_model) in self.params['Price Models']:
                        stochastic_factors.setdefault(factor_model, factor)
                        if additional_factor:
                            additional_factors.setdefault(factor_model, additional_factor)
                    else:
                        logging.error(
                            'Risk Factor {0} using stochastic process {1} missing in Price Models section'.format(
                                utils.check_tuple_name(factor), stoch_proc))

        return dependent_factors, stochastic_factors, additional_factors, reset_dates, currency_settlement_dates

    def parse_json(self, filename):

        def as_internal(dct):
            if '.Curve' in dct:
                return utils.Curve(dct['.Curve']['meta'], dct['.Curve']['data'])
            elif '.Percent' in dct:
                return utils.Percent(dct['.Percent'])
            elif '.Deal' in dct:
                return construct_instrument(dct['.Deal'], self.params['Valuation Configuration'])
            elif '.Basis' in dct:
                return utils.Basis(dct['.Basis'])
            elif '.Descriptor' in dct:
                return utils.Descriptor(dct['.Descriptor'])
            elif '.DateList' in dct:
                return utils.DateList(OrderedDict([(Timestamp(date), val) for date, val in dct['.DateList']]))
            elif '.DateEqualList' in dct:
                return utils.DateEqualList([[Timestamp(values[0])] + values[1:] for values in dct['.DateEqualList']])
            elif '.CreditSupportList' in dct:
                return utils.CreditSupportList(dct['.CreditSupportList'])
            elif '.DateOffset' in dct:
                return DateOffset(**dct['.DateOffset'])
            elif '.Offsets' in dct:
                return utils.Offsets(dct['.Offsets'])
            elif '.Timestamp' in dct:
                return Timestamp(dct['.Timestamp'])
            elif '.ModelParams' in dct:
                return ModelParams((dct['.ModelParams']['modeldefaults'], dct['.ModelParams']['modelfilters']))
            return dct

        with open(filename, 'rt') as f:
            self.last_file_loaded = filename
            self.file_ref = os.path.splitext(os.path.split(self.last_file_loaded)[-1])[0]
            data = json.load(f, object_hook=as_internal)

        if 'MarketData' in data:
            market_data = data['MarketData']
            correlations = {}
            for rate1, rate_list in market_data['Correlations'].items():
                for rate2, correlation in rate_list.items():
                    correlations.setdefault((rate1, rate2), correlation)

            # update the correlations
            market_data['Correlations'] = correlations
            self.params = market_data
            self.version = data['Version']

        elif 'Deals' in data:
            self.deals = data

        elif 'CalibrationConfig' in data:
            # store the calibration config
            self.calibrations = data

            # load the archive file - must be a tab separated file
            mkt_data_details = self.calibrations['CalibrationConfig']['MarketDataArchiveFile']
            self.archive = pd.read_csv(mkt_data_details['name'], skiprows=mkt_data_details['skiprows'],
                                       sep='\t', index_col=mkt_data_details['index_column'])
            # load the calibration
            self.calibration_process_map = dict((calibration['PriceModel'], construct_calibration_config(calibration))
                                                for calibration in
                                                self.calibrations['CalibrationConfig']['Calibrations'])
            # store a lookup to all columns
            self.archive_columns = {}

            for col in self.archive.columns:
                self.archive_columns.setdefault(col.split(',')[0], []).append(col)

    def read_json(self, filename):

        def as_internal(dct):
            if '.Curve' in dct:
                return utils.Curve(dct['.Curve']['meta'], dct['.Curve']['data'])
            elif '.Percent' in dct:
                return utils.Percent(dct['.Percent'])
            elif '.Deal' in dct:
                return construct_instrument(dct['.Deal'], self.params['Valuation Configuration'])
            elif '.Basis' in dct:
                return utils.Basis(dct['.Basis'])
            elif '.Descriptor' in dct:
                return utils.Descriptor(dct['.Descriptor'])
            elif '.DateList' in dct:
                return utils.DateList(OrderedDict([(Timestamp(date), val) for date, val in dct['.DateList']]))
            elif '.DateEqualList' in dct:
                return utils.DateEqualList([[Timestamp(values[0])] + values[1:] for values in dct['.DateEqualList']])
            elif '.CreditSupportList' in dct:
                return utils.CreditSupportList(dct['.CreditSupportList'])
            elif '.DateOffset' in dct and '.Offset' in dct:
                return [self.periodparser.parseString(dct['.DateOffset'])[0],
                        self.periodparser.parseString(dct['.Offset'])[0]]
            elif '.DateOffset' in dct:
                return self.periodparser.parseString(dct['.DateOffset'])[0]
            elif '.Grid' in dct:
                return utils.Offsets([(x if isinstance(x, list) else [x]) for x in dct['.Grid']])
            elif '.Timestamp' in dct:
                return Timestamp(dct['.Timestamp'])
            elif '.ModelParams' in dct:
                return ModelParams((dct['.ModelParams']['modeldefaults'], dct['.ModelParams']['modelfilters']))
            return dct

        with open(filename, 'rt') as f:
            self.last_file_loaded = filename
            self.file_ref = os.path.splitext(os.path.split(self.last_file_loaded)[-1])[0]
            data = json.load(f, object_hook=as_internal)

        return data

    def write_marketdata_json(self, json_filename):
        # backup old data
        old_correlations = self.params['Correlations']

        # need to serialize out new data
        correlations = {}
        for correlation, value in old_correlations.items():
            correlations.setdefault(correlation[0], {}).setdefault(correlation[1], value)

        # create new keys for json serialization
        self.params['Correlations'] = correlations

        with open(json_filename, 'wt') as f:
            f.write(json.dumps({'MarketData': self.params,
                                'Version': self.version}, separators=(',', ':'), cls=CustomJsonEncoder))

        # restore state
        self.params['Correlations'] = old_correlations

    def write_tradedata_json(self, json_filename):
        with open(json_filename, 'wt') as f:
            f.write(json.dumps(self.deals, separators=(',', ':'), cls=CustomJsonEncoder))

    def write_calibration_json(self, json_filename):
        with open(json_filename, 'wt') as f:
            f.write(json.dumps(self.calibrations, separators=(',', ':'), cls=CustomJsonEncoder))


if __name__ == '__main__':
    pass
