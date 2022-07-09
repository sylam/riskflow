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
import calendar

# import parsing libraries
from pyparsing import *
from xml.etree.ElementTree import ElementTree, Element, iterparse

# useful types
from collections import OrderedDict

# needed types
from . import utils
from .config import ModelParams, Context
from .stochasticprocess import construct_calibration_config
from .instruments import construct_instrument

try:
    # load up extra libraries
    # Note that PyPy does not have access to numpy and pandas (by default) - but it's stupidly fast at parsing
    import pandas as pd
    import numpy as np

    # define datetime routines
    Timestamp = pd.Timestamp
    DateOffset = pd.DateOffset
    array_transform = lambda x: x.tolist()

except ImportError:
    import csv


    # define dummy objects for parsing with PyPy
    class DateOffset(object):
        def __init__(self, **kwds):
            self.kwds = kwds


    class Timestamp(object):
        def __init__(self, date):
            self.date = date

        def strftime(self, fmt):
            return self.date


    array_transform = lambda x: x
    Default_Precision = None


def copy_dict(source_dict, diffs):
    """Returns a copy of source_dict, updated with the new key-value pairs in diffs."""
    result = OrderedDict(source_dict)
    result.update(diffs)
    return result


def parse_market_prices(prices):
    market_prices = {}
    for rate, data in OrderedDict(prices).items():
        market_prices[rate] = {'instrument': OrderedDict((k, v) for k, v in data.items() if k != 'Points')}
        children = market_prices[rate].setdefault('Children', [])
        for point in data.get('Points', []):
            children.append({'quote': OrderedDict((k, v) for k, v in point.items() if k != 'Deal'),
                             'instrument': point['Deal']})
    return market_prices


def format_market_prices(data):
    points = []
    for point in data['Children']:
        # make sure the order is correct
        quote = sorted(point['quote'].items(),
                       key=lambda x: {'Descriptor': -1, 'DealType': -2}.get(x[0], 0))
        points.append(copy_dict(quote, {'Deal': point['instrument']}))
    return points


def drawobj(obj):
    """Tries to emulate a human readable format"""
    buffer = []
    if isinstance(obj, list):
        for value in obj:
            if isinstance(value, tuple):
                buffer += ['.'.join(value)]
            elif isinstance(value, DateOffset):
                buffer += [''.join(['%d%s' % (v, Context.reverse_offset[k]) for k, v in value.kwds.items()])]
            elif isinstance(value, float):
                buffer += ['%.12g' % value]
            elif isinstance(value, Timestamp):
                buffer += ['%02d%s%04d' % (value.day, calendar.month_abbr[value.month], value.year)]
            else:
                buffer += [str(value) if value else '']
    else:
        for key, value in obj.items():
            if isinstance(value, dict):
                buffer += ['='.join([key, '[%s]' % drawobj(value)])]
            elif isinstance(value, list):
                temp = []
                for sub_value in value:
                    if isinstance(sub_value, str):
                        temp += [sub_value]
                    elif isinstance(sub_value, utils.Curve):
                        temp += [str(sub_value)]
                    else:
                        temp += ['[' + drawobj(sub_value) + ']']
                buffer += ['='.join([key, '[%s]' % (
                    ','.join(temp) if key in ['InstantaneousDrift', 'InstantaneousVols', 'Eigenvectors', 'Resets',
                                              'Points', 'Instrument_Definitions'] else ''.join(temp))])]
            elif isinstance(value, tuple):
                buffer += ['='.join([key, '.'.join(value)])]
            elif isinstance(value, DateOffset):
                buffer += [
                    '='.join([key, ''.join(['%d%s' % (v, Context.reverse_offset[k]) for k, v in value.kwds.items()])])]
            elif isinstance(value, float):
                buffer += ['='.join([key, '%.12g' % value])]
            elif isinstance(value, Timestamp):
                buffer += ['='.join([key, '%02d%s%04d' % (value.day, calendar.month_abbr[value.month], value.year)])]
            else:
                buffer += ['='.join([key, str(value) if value else ''])]

    return ','.join(buffer)


class AdaptivContext(Context):
    """
    Reads (parses) an Adaptiv Analytics compatible marketdata and deals file.
    Also writes out these files once the data has been modified.
    """

    def __init__(self, *args, **kwargs):
        super(AdaptivContext, self).__init__(*args, **kwargs)
        self.calibrations = ElementTree(Element('CalibrationConfig'))
        self.sourcemtm = None

        # setup an empty calibration config
        for elem_tag in ['MarketDataArchiveFile', 'DealsFile', 'Calibrations', 'BootstrappingPriceFactorSelections']:
            self.calibrations.getroot().append(Element(elem_tag))

        # the default state of the system
        self.version = None
        self.params.update({
            'System Parameters':
                {'Base_Currency': 'USD',
                 'Description': '',
                 'Base_Date': '',
                 'Exclude_Deals_With_Missing_Market_Data': 'Yes',
                 'Proxying_Rules_File': '',
                 'Script_Base_Scenario_Multiplier': 1,
                 'Correlations_Healing_Method': 'Eigenvalue_Raising',
                 'Grouping_File': ''
                 }
        })

        # make sure that there are no default calibration mappings
        self.calibration_process_map = {}
        self.parser, self.lineparser, self.gridparser, self.periodparser, self.assignparser = self.grammar()

    def parse_source_mtm_file(self, filename, columns=('ThVal', 'AA Base Valuation')):
        try:
            recon = pd.read_csv(filename, index_col=0)
            recon.index = recon.index.astype(np.str)
            self.sourcemtm = recon[columns].to_dict()
        except:
            self.sourcemtm = {}
            with open(filename) as f:
                recon = csv.reader(f)
                header = recon.next()
                offsets = [header.index(column) for column in columns]

                for line in recon:
                    for offset in offsets:
                        self.sourcemtm.setdefault(header[offset], {}).setdefault(line[0], line[offset])

    def parse_calibration_file(self, filename):
        """
        Parses the xml calibration file in filename (and loads up the .ada file)
        """

        self.calibrations = ElementTree(file=filename)
        self.calibration_process_map = {}

        for elem in self.calibrations.getroot():
            if elem.tag == 'MarketDataArchiveFile':
                self.archive = pd.read_csv(elem.text, skiprows=3, sep='\t', index_col=0)
            elif elem.tag == 'Calibrations':
                for calibration in elem:
                    param = OrderedDict(self.lineparser.parseString(calibration.text).asList())
                    calibration_config = construct_calibration_config(calibration.attrib, param)
                    self.calibration_process_map.setdefault(calibration_config.model, calibration_config)

        # store a lookup to all columns
        self.archive_columns = {}

        for col in self.archive.columns:
            self.archive_columns.setdefault(col.split(',')[0], []).append(col)

    def parse_market_file(self, filename):
        '''Parses the AA marketdata .dat file in filename'''

        self.params = OrderedDict()
        self.parser.parseFile(filename)

    def parse_trade_file(self, filename, calc_filename=None):
        '''
        Parses the xml .aap file in filename
        '''
        self.deals = {'Deals': {'Children': []}}
        deals = [self.deals['Deals']['Children']]
        path = []
        attrib = []
        elements = []

        for event, elem in iterparse(filename, events=('start', 'end')):

            if event == 'start':
                elements.append(elem)
                path.append(elem.tag)
                attrib.append(elem.attrib)

            elif event == 'end':
                # process the tag
                if elem.text and elem.text.strip() != '':
                    # get the field data
                    fields = OrderedDict(self.lineparser.parseString(elem.text).asList())
                    # check if we can amend it with sourceMTM's
                    if self.sourcemtm and 'Reference' in fields:
                        fields['MtM'] = self.sourcemtm['AA Base Valuation'].get(fields['Reference'])
                    if path[-2:] == ['Deal', 'Properties']:
                        # make a new node
                        node = {'instrument': construct_instrument(
                            fields, self.params['Valuation Configuration']), 'Children': []}
                        # add it to the collection
                        deals[-1].append(node)
                        # add any modifiers
                        node.update(attrib[-2])
                        # go deeper
                        if len(elements[-2].getchildren()) > 1:
                            deals.append(node['Children'])

                    elif path[-2:] == ['Deals', 'Deal']:
                        # make a new node
                        node = {'instrument': construct_instrument(
                            fields, self.params['Valuation Configuration'])}
                        # add it
                        deals[-1].append(node)
                        # add any modifiers
                        node.update(elem.attrib)

                if path[-2:] == ['Deal', 'Deals']:
                    # go back
                    deals.pop()

                _ = path.pop()
                _ = elements.pop()
                last_attrib = attrib.pop()

        # store the last attribute
        self.deals['Attributes'] = last_attrib

        # check if we need to load a calculation
        if calc_filename is not None:
            with open(calc_filename, 'rt') as f:
                calc_data = f.read()
                self.deals['Calculation'] = OrderedDict(self.lineparser.parseString(calc_data).asList())

    def write_trade_file(self, filename):
        def amend(xml_parent, internal_deals):
            for node in internal_deals:
                instrument = node['Instrument']
                deal = Element('Deal')
                fields = OrderedDict(
                    [(k, v) for k, v in sorted(instrument.field.items(), key=lambda x: -1 if x[0] == 'Object' else 0)])

                if node.get('Ignore'):
                    deal.attrib['Ignore'] = node['Ignore']

                if node.get('Children'):
                    properties = Element('Properties')
                    properties.text = drawobj(fields)
                    deal.append(properties)
                    deals = Element('Deals')
                    deal.append(deals)
                    amend(deals, node['Children'])
                else:
                    deal.text = drawobj(fields)

                xml_parent.append(deal)

        # Writes the state of these deals to filename (loosely compatible with adaptiv analytics)
        xml_deals = ElementTree(
            Element('Deals', attrib=self.deals.get('Attributes', {self.version[0]: self.version[1]})))
        amend(xml_deals.getroot(), self.deals['Deals']['Children'])
        xml_deals.write(filename)

    def write_market_file(self, filename):
        """Writes out the internal state of this config object out to filename"""
        # need to explicitly store the config file in this order because some people don't like standards
        sections = ['System Parameters', 'Model Configuration', 'Price Factors', 'Price Factor Interpolation',
                    'Price Models', 'Correlations', 'Valuation Configuration', 'Market Prices',
                    'Bootstrapper Configuration']
        with open(filename, 'wt') as f:
            f.write('='.join(self.version) + '\n\n')
            for k in sections:
                v = self.params.get(k, {})
                f.write('<{0}>'.format(k) + '\n')
                if k == 'Correlations':
                    for (f1, f2), corr in sorted(v.items()):
                        f.write(','.join([f1, f2, str(corr)]) + '\n')
                elif k == 'Market Prices':
                    for rate, data in sorted(v.items()):
                        points = format_market_prices(data)
                        # rewrite the madness
                        f.write(rate + ',' + drawobj(
                            copy_dict(data['instrument'], {'Points': points} if points else {})) + '\n')
                elif k in ['Price Factors', 'Price Models']:
                    for key, value in sorted(v.items()):
                        f.write(key + ',' + drawobj(value) + '\n')
                elif k == 'Valuation Configuration':
                    for dealtype, param in sorted(v.items()):
                        deal_func = param['Valuation']
                        params = {k: v for k, v in param.items() if k != 'Valuation'}
                        f.write(dealtype + '=' + deal_func + ((',' + drawobj(params)) if params else '') + '\n')
                elif isinstance(v, dict):
                    for key, value in sorted(v.items()):
                        f.write(key + '=' + ('' if value is None else drawobj([value])) + '\n')
                elif v.__class__.__name__ == 'ModelParams':
                    v.write_to_file(f)
                else:
                    raise Exception("Unknown model section {0} in writing market data {1}".format(k, filename))
                f.write('\n')

    def convert_market_data_json(self, dat_filename, json_filename):
        # read in the market_data_file
        self.parse_market_file(dat_filename)
        # write out a json file
        self.write_marketdata_json(json_filename)

    def convert_trade_data_json(self, aap_filename, json_filename, calc_filename=None):
        # read in the trade_data_file
        self.parse_trade_file(aap_filename, calc_filename)
        # write out a json file
        self.write_tradedata_json(json_filename)

    def grammar(self):
        '''
        Contains the grammer definition rules for parsing the market data file.
        Mostly complete but may need to be extended as needed. This is EXTREMELY slow in CPython.
        Please call this only in PyPy.
        '''

        def pushDate(strg, loc, toks):
            return Timestamp('{0}-{1:02d}-{2:02d}'.format(toks[2], Context.month_lookup[toks[1]], int(toks[0])))

        def pushInt(strg, loc, toks):
            return int(toks[0])

        def pushFloat(strg, loc, toks):
            return float(toks[0])

        def pushPercent(strg, loc, toks):
            return utils.Percent(toks[0])

        def pushBasis(strg, loc, toks):
            return utils.Basis(toks[0])

        def pushIdent(strg, loc, toks):
            return (toks[0], None if len(toks) == 1 else toks[1])

        def pushChain(strg, loc, toks):
            return OrderedDict({toks[0]: toks[1]})

        def pushSinglePeriod(strg, loc, toks):
            return (Context.offset_lookup[toks[1]], toks[0])

        def pushPeriod(strg, loc, toks):
            ofs = dict(toks.asList())
            return DateOffset(**ofs)

        def pushID(strg, loc, toks):
            entry = OrderedDict()
            for k, v in toks[1:]:
                entry[k] = v
            return toks[0], entry

        def pushParam(strg, loc, toks):
            entry = OrderedDict([('Valuation', toks[1])])
            for k, v in toks[2:]:
                entry[k] = v
            return toks[0], entry

        def pushCurve(strg, loc, toks):
            '''need to allow differentiation between adding a spread and scaling by a factor'''
            return utils.Curve([], toks[0].asList()) if len(toks) == 1 else utils.Curve(toks[:-1], toks[-1].asList())

        def pushList(strg, loc, toks):
            return toks.asList()

        def pushOffset(strg, loc, toks):
            return utils.Offsets(toks[0].asList())

        def pushDateGrid(strg, loc, toks):
            return toks[0][0] if len(toks) == 1 else utils.Offsets(toks.asList())

        def pushDescriptor(strg, loc, toks):
            return utils.Descriptor(toks[0].asList())

        def pushDateList(strg, loc, toks):
            return utils.DateList(toks.asList())

        def pushDateEqualList(strg, loc, toks):
            return utils.DateEqualList(toks.asList())

        def pushCreditSupportList(strg, loc, toks):
            return utils.CreditSupportList(toks.asList())

        def pushObj(strg, loc, toks):
            obj = OrderedDict()
            for token in toks.asList():
                if isinstance(token, OrderedDict) or isinstance(token, list) or isinstance(token, str) or isinstance(
                        token, utils.Curve):
                    obj.setdefault('_obj', []).append(token)
                else:
                    key, val = token
                    obj.setdefault(key, val)
            return [obj.get('_obj', obj)]

        def pushTuple(strg, loc, toks):
            return tuple(toks[0])

        def pushName(strg, loc, toks):
            return '.'.join(toks[0]) if len(toks[0]) > 1 else toks[0][0]

        def pushKeyVal(strg, loc, toks):
            return (toks[0], toks[1].rstrip())

        def pushRule(strg, loc, toks):
            return (toks[0], toks[1].strip()[1:-1])

        def pushMdlCfg(strg, loc, toks):
            return (toks[0], toks[1].rstrip(), toks[2] if len(toks) > 2 else ())

        def pushSection(strg, loc, toks):
            if toks[0] == '<Correlations>':
                self.params.setdefault(toks[0][1:-1], {(p1, p2): c for p1, p2, c in toks[1:]})
            elif toks[0] in ['<Model Configuration>', '<Price Factor Interpolation>']:
                # need to filter out paramters
                model_params = ModelParams()
                for elem in toks[1:]:
                    model_params.append(elem[0], elem[2] if len(elem[-1]) > 0 else (), elem[1])
                self.params.setdefault(toks[0][1:-1], model_params)
            elif toks[0] == '<Market Prices>':
                # change the way market prices are expressed due AA being fucking insane
                market_prices = parse_market_prices(toks[1:])
                self.params.setdefault(toks[0][1:-1], market_prices)
            else:
                self.params.setdefault(toks[0][1:-1], OrderedDict(toks[1:]))
            return toks[0][1:-1]

        def pushCorrel(strg, loc, toks):
            return tuple(toks)

        def pushConfig(strg, loc, toks):
            self.version = [x.rstrip() for x in toks[:2]]

        e = CaselessLiteral("E")
        undef = Keyword("<undefined>")
        reference = Keyword("Reference")
        counterparty = Keyword("Counterparty")

        # headings
        aa_format = Keyword("AnalyticsVersion")
        correl_sec = Keyword("<Correlations>")
        market_p = Keyword("<Market Prices>")
        bootstrap = Keyword("<Bootstrapper Configuration>")
        valuation = Keyword("<Valuation Configuration>")
        price_mod = Keyword("<Price Models>")
        factor_int = Keyword("<Price Factor Interpolation>")
        price_fac = Keyword("<Price Factors>")
        model_cfg = Keyword("<Model Configuration>")
        sys_param = Keyword("<System Parameters>")

        # reserved words
        where = Keyword("where").suppress()

        equals = Literal("=").suppress()
        zero = Empty().setParseAction(lambda strg, loc, toks: 0.0)
        null = Empty().suppress()
        eol = LineEnd().suppress()

        lapar = Literal("<").suppress()
        rapar = Literal(">").suppress()
        lsqpar = Literal("[").suppress()
        rsqpar = Literal("]").suppress()
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        percent = Literal("%").suppress()
        backslash = Literal('\\').suppress()
        decimal = Literal(".")
        comma = Literal(",").suppress()

        ident = Word(alphas, alphas + nums + "_-")
        desc = Word(alphas+',', alphas+nums+", ()/%_-.").setName('Description')
        arbstring = Word(alphas+nums+'_', alphas+nums+"/*_ :+-()%&#$").setName('ArbString')
        namedId = Group(delimitedList(arbstring, delim='.', combine=False)).setName('namedId').setParseAction(pushName)
        integer = (Word("+-" + nums, nums) + ~decimal).setName('int').setParseAction(pushInt)
        fnumber = Combine(Word("+-" + nums, nums) + Optional(decimal + Optional(Word(nums))) + Optional(
            e + Word("+-" + nums, nums))).setName('float').setParseAction(pushFloat)
        date = (integer + oneOf(list(calendar.month_abbr)[1:]) + integer).setName('date').setParseAction(pushDate)
        dateitem = Group(date + equals + fnumber).setParseAction(pushTuple)
        dateitemfx = Group(date + OneOrMore(equals + fnumber)).setParseAction(pushTuple)
        credit_item = Group(integer + equals + fnumber + backslash).setParseAction(pushTuple)

        datelist = (backslash | delimitedList(dateitem, delim=backslash) + Optional(backslash)).setParseAction(
            pushDateList)
        datelistdel = (lsqpar + Optional(delimitedList(dateitemfx)) + rsqpar).setParseAction(pushDateEqualList)
        creditlist = (backslash | OneOrMore(credit_item)).setParseAction(pushCreditSupportList)

        # WHY DO I need to put a ' ' here?
        percentage = (fnumber + ZeroOrMore(' ') + percent).setName('percent').setParseAction(pushPercent)
        basis_pts = (fnumber + ' bp').setName('bips').setParseAction(pushBasis)
        point = (lpar + Group(delimitedList(fnumber)) + rpar).setParseAction(pushTuple)
        single_period = (integer + oneOf(['D', 'M', 'Y', 'W'], caseless=True)).setName('single_period').setParseAction(
            pushSinglePeriod)
        period = OneOrMore(single_period).setName('period').setParseAction(pushPeriod)
        descriptor = Group(integer + Literal('X').suppress() + integer).setParseAction(pushDescriptor)

        obj = Forward()
        chain = (ident + equals + obj).setParseAction(pushChain)
        listofstf = delimitedList(
            lsqpar + Group(delimitedList(date | percentage | period | fnumber | namedId | zero)) + rsqpar).setParseAction(
            pushList)

        curve = (lsqpar + Optional(delimitedList(integer | fnumber | ident) + comma) + Group(
            delimitedList(point)) + rsqpar).setParseAction(pushCurve)
        tenors = (lsqpar + Group(delimitedList(period)) + rsqpar).setParseAction(pushOffset)
        grid = delimitedList(Group(period + Optional(lpar + period + rpar)),
                             delim=' ').leaveWhitespace().setParseAction(pushDateGrid)
        assign = (((reference | counterparty) + equals + namedId) | (ident + equals + (
                chain | creditlist | datelistdel | datelist | date | grid | percentage | basis_pts | descriptor | fnumber | namedId | undef | curve | tenors | obj | null))).leaveWhitespace().setParseAction(
            pushIdent)
        obj << (lsqpar + (delimitedList(curve) | listofstf | delimitedList(
            assign | OneOrMore(obj)) | desc) + rsqpar).setParseAction(pushObj)
        line = (namedId + comma + delimitedList(assign) + Optional(eol)).setParseAction(pushID)
        correl = (namedId + comma + namedId + comma + fnumber).setParseAction(pushCorrel)

        header = aa_format + equals + restOfLine
        system = (~lapar + ident + equals + (date | ident | integer | null).leaveWhitespace()).setParseAction(pushIdent)
        todo = (~lapar + ident + equals + restOfLine).setParseAction(pushKeyVal)
        param = (~lapar + ident + equals + ident + Optional(comma + delimitedList(assign)) + Optional(
            eol)).setParseAction(pushParam)

        rule = (~lapar + ident + equals + restOfLine).setParseAction(pushRule)
        modelcfg = (~lapar + ident + equals + ident + Optional(where + rule)).setParseAction(pushMdlCfg)

        section = ((correl_sec + ZeroOrMore(correl)) |
                   (sys_param + OneOrMore(system)) |
                   (bootstrap + ZeroOrMore(todo)) |
                   (valuation + OneOrMore(param)) |
                   ((model_cfg | factor_int) + ZeroOrMore(modelcfg)) |
                   ((price_fac | price_mod | market_p) + ZeroOrMore(line))).setParseAction(pushSection)

        marketfile = (header + OneOrMore(section)).setParseAction(pushConfig)

        marketfile.ignore(cppStyleComment)

        return marketfile, delimitedList(assign), grid, period, line


if __name__ == '__main__':
    pass
