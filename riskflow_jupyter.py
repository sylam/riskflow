import os
import json
import copy
import operator
import logging
import itertools
import numpy as np
import pandas as pd
import ipywidgets as widgets

from functools import reduce
from abc import ABCMeta, abstractmethod
from IPython.display import display  # Used to display widgets in the notebook

# riskflow specific stuff
import riskflow as rf
# load the widgets - might be able to use more native objects instead of unicode text - TODO!
from riskflow_widgets import FileChooser, Tree, Table, Flot, Three, to_json


def load_table_from_vol(vol):
    '''
    :param vol: a riskfactor representing an array of 2d or 3d surfaces in the
                'param' dict (usually stored as the 'Surface' key)
    :return: a list of lists representing the columns and rows (expiry and moneyness) where
            the vols are in the right cell.
    '''

    def make_table(array, index1=0, index2=1, index3=2):
        sparse_matrix = {}
        for k in array:
            sparse_matrix.setdefault(k[index1], {}).setdefault(k[index2], k[index3])

        df = pd.DataFrame(sparse_matrix).sort_index(level=[0, 1])

        return np.vstack([
            [0.0] + df.columns.values.round(5).tolist(),
            np.hstack(
                [df.index.values.round(5).reshape(-1, 1).tolist(),
                 df.round(5).replace({np.nan: None})]
            )]
        ).tolist()

    if vol.__class__.__name__ in rf.utils.TwoDimensionalFactors:
        return make_table(vol.param['Surface'].array)
    elif vol.__class__.__name__ in rf.utils.ThreeDimensionalFactors:
        vol_space = {}
        for t in vol.get_tenor():
            surface = vol.param['Surface'].array
            vol_space.setdefault(
                t, make_table(
                    surface[surface[:, vol.TENOR_INDEX] == t],
                    index1=vol.MONEYNESS_INDEX, index2=vol.EXPIRY_INDEX, index3=3)
            )
        return vol_space


# portfolio parent data
class DealCache:
    Json = 0
    Deal = 1
    Parent = 2
    Count = 3


# basic logging class
class MyStream(object):
    def __init__(self, log_widget):
        self.widget = log_widget
        self.widget.value = u''

    def flush(self):
        pass

    def write(self, record):
        self.widget.value += record


# Code for custom pages go here
class TreePanel(metaclass=ABCMeta):
    '''
    Base class for tree-based screens with a dynamic panel on the right to show data for the screen.
    '''

    def __init__(self, context):
        # load the config object - to load and store state
        self.context = context
        # load up relevant information from the config and define the tree type
        self.parse_config()

        # set up the view containters
        self.right_container = widgets.VBox()
        # update the style of the right container
        self.right_container._dom_classes = ['rightframe']

        self.tree.selected = u""
        self.tree._dom_classes = ['generictree']

        # event handlers
        self.tree.observe(self._on_selected_changed, 'selected')
        self.tree.observe(self._on_created_changed, 'created')
        self.tree.observe(self._on_deleted_changed, 'deleted')
        # self.tree.on_displayed(self._on_displayed)
        # default layout for the widgets
        self.default_layout = widgets.Layout(min_height='30px')

        # interface for the tree and the container
        self.main_container = widgets.HBox(
            children=[self.tree, self.right_container],
            layout=widgets.Layout(border='5px outset black'),
            _dom_classes=['mainframe']
        )

    @property
    def config(self):
        # this changes depending on the context
        return self.context.current_cfg

    @staticmethod
    def load_fields(field_names, field_data):
        storage = {}
        for k, v in field_names.items():
            storage[k] = {}
            if isinstance(v, dict):
                storage[k] = TreePanel.load_fields(v, field_data)
            else:
                for property_name in v:
                    field_meta = field_data[property_name].copy()
                    if 'sub_fields' in field_meta:
                        field_meta.update(TreePanel.load_fields(
                            {'sub_fields': field_meta['sub_fields']}, field_data))
                    field_meta['default'] = field_meta['value']
                    storage[k].setdefault(property_name, field_meta)
        return storage

    @staticmethod
    def get_value_for_widget(config, section_name, field_meta):
        '''Code for mapping between the library (model) and the UI (view) goes here.
           Handles both deal and factor data.

           For deals, config is the instrument object from the model and the section_name
           is usually the 'field' attribute.

           For factors, config is the config section from the model and the section_name
           is the name of the factor/process.

            The field_meta contains the actual field names.
        '''

        def get_repr(obj, field_name, default_val):

            if isinstance(obj, rf.utils.Curve):
                if obj.meta and obj.meta[0] != 'Integrated':
                    # need to use a threeview
                    t = getattr(rf.riskfactors, rate_type)({field_name: obj})
                    vol_space = load_table_from_vol(t)
                    return_value = to_json(vol_space)
                else:
                    return_value = to_json([{'label': 'Rate', 'data': [[x, y] for x, y in obj.array]}])
            elif isinstance(obj, rf.utils.Percent):
                return_value = obj.amount
            elif isinstance(obj, rf.utils.Basis):
                return_value = obj.amount
            elif isinstance(obj, rf.utils.Descriptor):
                return_value = str(obj)
            elif isinstance(obj, rf.utils.DateEqualList):
                data = [
                    [get_repr(date, 'Date', default_val)] +
                    [get_repr(sub_val, 'Value', default_val) for sub_val in value]
                    for date, value in obj.data.items()]
                return_value = to_json(data)
            elif isinstance(obj, rf.utils.DateList):
                data = [[get_repr(date, 'Date', default_val),
                         get_repr(value, 'Value', default_val)] for date, value in obj.data.items()]
                return_value = to_json(data)
            elif isinstance(obj, rf.utils.CreditSupportList):
                data = [[get_repr(value, 'Value', default_val),
                         get_repr(rating, 'Value', default_val)] for rating, value in obj.data.items()]
                return_value = to_json(data)
            elif isinstance(obj, list):
                if field_name == 'Eigenvectors':
                    madness = []
                    for i, element in enumerate(obj):
                        madness.append(
                            {'label': str(element['Eigenvalue']),
                             'data': [[x, y] for x, y in element['Eigenvector'].array]})
                    return_value = to_json(madness)
                elif field_name in ['Properties', 'Items', 'Cash_Collateral', 'Equity_Collateral',
                                    'Bond_Collateral', 'Commodity_Collateral']:
                    data = []
                    for flow in obj:
                        data.append(
                            [get_repr(flow.get(field_name), field_name, rf.fields.default.get(widget_type, default_val))
                             for field_name, widget_type in zip(field_meta['col_names'], field_meta['obj'])])
                    return_value = to_json(data)
                elif field_name == 'Resets':
                    headings = ['Reset_Date', 'Start_Date', 'End_Date', 'Year_Fraction', 'Rate_Tenor', 'Day_Count',
                                'Rate_Frequency', 'Rate_Fixing', 'Use Known Rate', 'Known_Rate']
                    widgets = ['DatePicker', 'DatePicker', 'DatePicker', 'Float', 'Period',
                               'Text', 'Period', 'Float', 'Text', 'Percent']
                    data = []
                    for flow in obj:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = to_json(data)
                elif field_name in ['Description', 'Tags']:
                    return_value = to_json(obj)
                elif field_name == 'TARF_ExpiryDates':
                    headings = ['Date', 'Settle', 'Value']
                    widgets = ['DatePicker', 'DatePicker', 'Float']
                    data = []
                    for flow in obj:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = to_json(data)
                elif field_name == 'Sampling_Data':
                    headings = ['Date', 'Price', 'Weight']
                    widgets = ['DatePicker', 'Float', 'Float']
                    data = []
                    for flow in obj:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = to_json(data)
                elif field_name in ['Price_Fixing', 'Autocall_Thresholds', 'Autocall_Coupons']:
                    headings = ['Date', 'Value']
                    widgets = ['DatePicker', 'Float']
                    data = []
                    for flow in obj:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = to_json(data)
                elif field_name == 'Barrier_Dates':
                    headings = ['Date', 'Value']
                    widgets = ['DatePicker', 'Float']
                    data = []
                    for flow in [obj]:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = to_json(data)
                
                else:
                    raise Exception('Unknown Array Field type {0}'.format(field_name))
            elif isinstance(obj, pd.DateOffset):
                return_value = ''.join(['%d%s' % (v, rf.config.Context.reverse_offset[k]) for k, v in obj.kwds.items()])
            elif isinstance(obj, pd.Timestamp):
                # leave datepickers unaltered but convert everything else to a string
                return_value = obj if field_meta['widget'] == 'DatePicker' else obj.strftime('%Y-%m-%d')
            elif obj is None:
                return_value = default_val
            else:
                return_value = obj

            # return the value
            return return_value

        # update an existing factor
        rate_type = rf.utils.check_rate_name(section_name)[0]
        field_name = field_meta['description'].replace(' ', '_')
        if section_name in config and field_name in config[section_name]:
            # return the value in the config obj
            obj = config[section_name][field_name]
            return get_repr(obj, field_name, field_meta['value'])
        else:
            return field_meta['value']

    @abstractmethod
    def parse_config(self):
        pass

    @abstractmethod
    def generate_handler(self, field_name, widget_elements, label):
        pass

    def define_input(self, label, widget_elements):
        # label this container
        wig = [widgets.HTML(value=self.get_label(label))]

        for field_name, element in widget_elements.items():
            # skip this element if it's not visible
            if element.get('isvisible') == 'False':
                continue
            if element['widget'] == 'Dropdown':
                w = widgets.Dropdown(
                    options=element['values'], description=element['description'],
                    layout=dict(width='420px'), value=element['value'])
            elif element['widget'] == 'Text':
                w = widgets.Text(description=element['description'], value=str(element['value']))
            elif element['widget'] == 'Container':
                new_label = label + [
                    element['description']] if isinstance(label, list) else [element['description']]
                w = self.define_input([x.replace(' ', '_') for x in new_label], element['sub_fields'])
            elif element['widget'] == 'Flot':
                w = Flot(description=element['description'],
                         hot_settings=to_json(element.get('hot_settings', {})),
                         flot_settings=to_json(element.get('flot_settings', {})),
                         value=element['value'])
            elif element['widget'] == 'Three':
                w = Three(description=element['description'], value=element['value'])
            elif element['widget'] == 'Integer':
                w = widgets.IntText(description=element['description'], value=element['value'])
            elif element['widget'] == 'HTML':
                w = widgets.HTML(value=element['value'])
            elif element['widget'] == 'Table':
                w = Table(description=element['description'],
                          settings=to_json({
                              'columns': element['sub_types'],
                              'colHeaders': element['col_names'],
                              'manualColumnMove': True,
                              'minSpareRows': 1,
                              'startRows': 1,
                              'startCols': len(element['col_names']),
                              'width': element.get('width', 400),
                              'height': element.get('height', 200)
                          }),
                          value=element['value']
                          )
            elif element['widget'] == 'Float':
                w = widgets.FloatText(description=element['description'], value=element['value'])
            elif element['widget'] == 'DatePicker':
                w = widgets.DatePicker(description=element['description'],
                                       value=pd.Timestamp(element['value']), layout=dict(width='420px'))
            elif element['widget'] == 'BoundedFloat':
                w = widgets.BoundedFloatText(min=element['min'], max=element['max'],
                                             description=element['description'], value=element['value'])
            else:
                raise Exception('Unknown widget field')

            if element['widget'] != 'Container':
                # add an observer to any non-containter widget
                w.observe(self.generate_handler(field_name, widget_elements, label), 'value')

            wig.append(w)

        container = widgets.VBox(children=wig, layout=widgets.Layout(padding='5px', border='outset'))
        return container

    @abstractmethod
    def get_label(self, label):
        pass

    @abstractmethod
    def calc_frames(self, selection):
        pass

    def _on_selected_changed(self, change):
        self.right_container.children = self.calc_frames(change['new'])

    @abstractmethod
    def create(self, newval):
        pass

    @abstractmethod
    def delete(self, newval):
        pass

    def _on_created_changed(self, change):
        if self.tree.created:
            # print 'create',val
            self.create(change['new'])
            # reset the created flag
            self.tree.unobserve(self._on_created_changed, 'created')
            self.tree.created = ''
            self.tree.observe(self._on_created_changed, 'created')

    def _on_deleted_changed(self, change):
        if self.tree.deleted:
            # print 'delete',val
            self.delete(change['new'])
            # reset the deleted flag
            self.tree.unobserve(self._on_deleted_changed, 'deleted')
            self.tree.deleted = ''
            self.tree.observe(self._on_deleted_changed, 'deleted')

    def show(self):
        display(self.main_container)


class PortfolioPage(TreePanel):

    def __init__(self, config):
        super(PortfolioPage, self).__init__(config)

    def get_label(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format(label[0], label[1]) if type(
            label) is tuple else '<h4>{0}</h4>'.format('.'.join(label))

    def set_value_from_widget(self, instrument, field_meta, new_val):

        def checkArray(new_obj):
            return [x for x in new_obj if x[0] is not None] if new_obj is not None else None

        def set_repr(obj, obj_type):
            if obj_type == 'Percent':
                return rf.utils.Percent(100.0 * obj)
            elif obj_type == 'Basis':
                return rf.utils.Basis(10000.0 * obj)
            elif obj_type == 'Period':
                return self.config.periodparser.parseString(obj)[0]
            elif obj_type == 'DateList':
                new_obj = checkArray(json.loads(obj))
                return rf.utils.DateList(
                    {pd.Timestamp(date): val for date, val in new_obj}) if new_obj is not None else None
            elif obj_type == 'DateEqualList':
                new_obj = checkArray(json.loads(obj))
                return rf.utils.DateEqualList(
                    [[pd.Timestamp(item[0])] + item[1:] for item in new_obj]) if new_obj is not None else None
            elif obj_type == 'DateValueList':
                new_obj = checkArray(json.loads(obj))
                return [[pd.Timestamp(item[0])] + item[1:] for item in new_obj] if new_obj is not None else None
            elif obj_type == 'CreditSupportList':
                new_obj = checkArray(json.loads(obj))
                return rf.utils.CreditSupportList(
                    {rating: val for val, rating in new_obj}) if new_obj is not None else None
            elif obj_type == 'ResetArray':
                resets = json.loads(obj)
                field_types = ['DatePicker', 'DatePicker', 'DatePicker', 'Float', 'Period',
                               'Text', 'Period', 'Float', 'Text', 'Percent']
                return [[set_repr(data_obj, data_type) for data_obj, data_type in zip(reset, field_types)]
                        for reset in resets]
            elif isinstance(obj_type, list):
                if field_name == 'Description':
                    return json.loads(obj)
                elif field_name == 'Sampling_Data':
                    new_obj = checkArray(json.loads(obj))
                    return [[set_repr(data_obj, data_type) for data_obj, data_type, field in
                             zip(data_row, obj_type, field_meta['col_names'])] for data_row in
                            new_obj] if new_obj else []
                elif field_name in ['Items', 'Properties', 'Cash_Collateral',
                                    'Equity_Collateral', 'Bond_Collateral', 'Commodity_Collateral']:
                    new_obj = checkArray(json.loads(obj))
                    return [{field: set_repr(data_obj, data_type) for data_obj, data_type, field in
                             zip(data_row, obj_type, field_meta['col_names'])} for data_row in
                            new_obj] if new_obj else []
                else:
                    raise Exception('Unknown list field type {0}'.format(field_name))
            elif obj_type == 'DatePicker':
                # check for NaT's (the obj==obj check will fail for NaT)
                return pd.Timestamp(obj) if obj and obj == obj else None
            elif obj_type == 'Float':
                try:
                    new_obj = obj if obj == '<undefined>' else float(obj)
                except ValueError:
                    print('FLOAT', obj, field_meta)
                    new_obj = 0.0
                return new_obj
            else:
                # default is to just return the object
                return obj

        field_name = field_meta['description'].replace(' ', '_')
        obj_type = field_meta.get('obj', field_meta['widget'])
        instrument[field_name] = set_repr(new_val, obj_type)

    def generate_handler(self, field_name, widget_elements, label):

        def handleEvent(change):
            # update the json representation
            widget_elements[field_name]['value'] = change['new']
            # find the instrument obj
            instrument_obj = self.current_deal['Instrument']
            # get the fields
            instrument = instrument_obj.field
            # update the value in the model
            if isinstance(label, list):
                for key in label:
                    if key not in instrument:
                        # should only happen if a container attribute was not set (should be a dict by default)
                        instrument[key] = {}
                    # if this is a nested container, check if it was set to an empty string
                    # (if it was, set it to a dict)
                    if instrument[key] == '':
                        instrument[key] = {}
                    # go down the list
                    instrument = instrument[key]

            self.set_value_from_widget(instrument, widget_elements[field_name], change['new'])

        return handleEvent

    def parse_config(self):

        def load_items(config, field_name, field_data, storage):
            # make sure we get the right set of properties
            properties = field_data[field_name] if field_name in field_data else field_data[
                config[field_name]['Object']]
            for property_name, property_data in sorted(properties.items()):
                value = copy.deepcopy(property_data)
                if 'sub_fields' in value:
                    new_field_name = value['description'].replace(' ', '_')
                    load_items(config[field_name], new_field_name, {new_field_name: value['sub_fields']},
                               value['sub_fields'])
                value['value'] = self.get_value_for_widget(config, field_name, value)

                storage[property_name] = value

        def walkPortfolio(deals, path, instrument_fields, parent, parent_cache):

            for node in deals:
                # get the instrument
                instrument = node['Instrument']
                # get its name
                reference = instrument.field.get('Reference')
                # update the parent cache
                count = parent_cache[path][DealCache.Count].setdefault(reference, 0)
                # get the deal_id (unique key)
                deal_id = '{0}{1}'.format(
                    reference, ':{0}'.format(count) if count else '') if reference else '{0}'.format(count)
                # establish the name
                name = "{0}.{1}".format(instrument.field['Object'], deal_id)
                # increment the counter
                parent_cache[path][DealCache.Count][reference] += 1
                # full path name
                path_name = path + (name,)

                # if node.attrib.get('Ignore')=='True':
                #    continue

                node_data = {}
                load_items(instrument.__dict__, 'field', instrument_fields, node_data)

                json_data = {"text": name,
                             "type": "group" if node.get('Children') else "default",
                             "data": {},
                             "children": []}

                parent.append(json_data)
                parent_cache[path_name] = [node_data, node, deals, {} if 'Children' in node else None]

                if node.get('Children'):
                    walkPortfolio(node['Children'], path_name, instrument_fields, json_data['children'], parent_cache)

        deals_to_append = []
        self.data = {(): [{}, self.config.deals['Deals'], None, {}]}

        # map all the fields to one flat hierarchy
        instrument_types = {
            key: reduce(operator.concat, [rf.fields.mapping['Instrument']['sections'][group] for group in groups]) for
            key, groups in rf.fields.mapping['Instrument']['types'].items()}

        # get the fields from the master list
        self.instrument_fields = self.load_fields(instrument_types, rf.fields.mapping['Instrument']['fields'])
        # fill it with data
        walkPortfolio(self.config.deals['Deals']['Children'], (), self.instrument_fields, deals_to_append, self.data)

        self.tree_data = [{"text": "Postions",
                           "type": "root",
                           "id": "ROOT",
                           "state": {"opened": True, "selected": True},
                           "children": deals_to_append}]

        type_data = {"root": {"icon": "fa fa-folder text-primary", "valid_children": ["group"]},
                     "group": {"icon": "fa fa-folder", "valid_children": ["group", "default"]},
                     "default": {"icon": "fa fa-file", "valid_children": []}}

        context_menu = {}
        for k, v in sorted(rf.fields.mapping['Instrument']['groups'].items()):
            group_type = v[0]
            for instype in sorted(v[1]):
                context_menu.setdefault(k, {}).setdefault(instype, group_type)

        # tree widget data
        self.tree = Tree(
            plugins=["contextmenu", "sort", "types", "search", "unique"],
            settings=to_json({'contextmenu': context_menu, 'events': ['create', 'delete']}),
            type_data=to_json(type_data),
            value=to_json(self.tree_data)
        )

        # have a placeholder for the selected model (deal)
        self.current_deal = None
        self.current_deal_parent = None

    def calc_frames(self, selection):
        key = tuple(selection)
        frames = []
        frame, self.current_deal, self.current_deal_parent, count = self.data.get(key, [{}, None, None, 0])

        # factor_fields
        if frame:
            # get the object type - this should always be defined
            obj_type = frame['Object']['value']

            for frame_name in rf.fields.mapping['Instrument']['types'][obj_type]:
                # load the values:
                instrument_fields = rf.fields.mapping['Instrument']['sections'][frame_name]
                frame_fields = {k: v for k, v in frame.items() if k in instrument_fields}
                frames.append(self.define_input((frame_name, key[-1]), frame_fields))

        return frames

    def create(self, val):
        key = tuple(val)
        instrument_type = key[-1][:key[-1].find('.')]
        reference = key[-1][key[-1].find('.') + 1:]

        # load defaults for the new object
        fields = self.instrument_fields.get(instrument_type)
        ins = {}

        for value in fields.values():
            self.set_value_from_widget(ins, value, value['value'])

        # set it up
        ins['Object'] = instrument_type
        ins['Reference'] = reference

        # Now check if this is a group or a regular deal
        if instrument_type in rf.fields.mapping['Instrument']['groups']['New Structure'][1]:
            deal = {'Instrument': rf.instruments.construct_instrument(
                ins, self.config.params['Valuation Configuration']), 'Children': []}
        else:
            deal = {'Instrument': rf.instruments.construct_instrument(
                ins, self.config.params['Valuation Configuration'])}

        # add it to the xml
        parent = self.data[key[:-1]][DealCache.Deal]
        # add this to parent
        parent['Children'].append(deal)

        # store it away
        view_data = copy.deepcopy(self.instrument_fields.get(instrument_type))

        # make sure we record the instrument type
        for field in ['Object', 'Reference']:
            view_data[field]['value'] = ins[field]

        # update the cache
        count = self.data[key[:-1]][DealCache.Count].setdefault(reference, 0)
        # increment it
        self.data[key[:-1]][DealCache.Count][reference] += 1
        # store it away
        self.data[key] = [view_data, deal, parent, {} if 'Children' in deal else None]

    def delete(self, val):
        key = tuple(val)
        reference = key[-1][key[-1].find('.') + 1:]
        parent = self.data[key][DealCache.Parent]

        print(key, parent)
        # delete the deal
        parent['Children'].remove(self.data[key][DealCache.Deal])
        # decrement the count
        self.data[key[:-1]][DealCache.Count][reference] -= 1
        # delte the view data
        del self.data[key]


class RiskFactorsPage(TreePanel):
    def __init__(self, config):
        super(RiskFactorsPage, self).__init__(config)

    def parse_config(self):

        def load_items(config, factor_data, field_data, storage):
            for property_name, property_data in sorted(field_data[factor_data.type].items()):
                value = copy.deepcopy(property_data)
                value['value'] = self.get_value_for_widget(config, rf.utils.check_tuple_name(factor_data), value)
                storage[property_name] = value

        risk_factor_fields = self.load_fields(
            rf.fields.mapping['Factor']['types'], rf.fields.mapping['Factor']['fields'])
        risk_process_fields = self.load_fields(
            rf.fields.mapping['Process']['types'], rf.fields.mapping['Process']['fields'])

        # only 1 level of config parameters here - unlike the other 2
        self.sys_config = next(iter(self.load_fields(
            rf.fields.mapping['System']['types'], rf.fields.mapping['System']['fields']).values()))
        for value in self.sys_config.values():
            value['value'] = self.get_value_for_widget(self.config.params, 'System Parameters', value)

        possible_risk_process = {}
        for k, v in rf.fields.mapping['Process_factor_map'].items():
            fields_to_add = {}
            for process in v:
                fields_to_add[process] = risk_process_fields[process]
            possible_risk_process[k] = fields_to_add

        loaded_data = []
        self.data = {}
        factor_process_map = {}

        for price_factor, price_data in self.config.params['Price Factors'].items():
            raw_factor = rf.utils.check_rate_name(price_factor)
            factor = rf.utils.Factor(raw_factor[0], raw_factor[1:])
            factor_to_append = {}
            process_to_append = {}
            process_data = possible_risk_process[factor.type].copy()

            factor_process_map[price_factor] = ''

            load_items(self.config.params['Price Factors'], factor, risk_factor_fields, factor_to_append)
            stoch_proc = self.config.params['Model Configuration'].search(factor, price_data)
            if stoch_proc:
                factor_model = rf.utils.Factor(stoch_proc, factor.name)
                load_items(self.config.params['Price Models'], factor_model, risk_process_fields, process_to_append)
                process_data[stoch_proc] = process_to_append
                factor_process_map[price_factor] = stoch_proc

            factor_node = {'text': price_factor,
                           'type': 'default',
                           'data': {},
                           'children': []}

            loaded_data.append(factor_node)
            self.data[price_factor] = {'Factor': factor_to_append, 'Process': process_data}

        self.tree_data = [{"text": "Price Factors",
                           "type": "root",
                           "state": {"opened": True, "selected": False},
                           "children": loaded_data}]

        type_data = {
            "root": {
                "icon": "fa fa-folder", "valid_children": ["default"]
            },
            "default": {
                "icon": "fa fa-file", "valid_children": []
            }
        }

        # simple lookup to match the params in the config file to the json in the UI
        self.config_map = {'Factor': 'Price Factors', 'Process': 'Price Models', 'Config': 'System Parameters'}
        # fields to store new objects
        self.new_factor = {'Factor': risk_factor_fields, 'Process': possible_risk_process}
        # set the context menu to all risk factors defined
        context_menu = {"New Risk Factor": {x: "default" for x in sorted(risk_factor_fields.keys())}}

        # tree widget data
        self.tree = Tree(
            plugins=["contextmenu", "sort", "unique", "types", "search", "checkbox"],
            settings=to_json({'contextmenu': context_menu, 'events': ['create', 'delete']}),
            type_data=to_json(type_data),
            value=to_json(self.tree_data)
        )

    def get_label(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format(label[0], label[1] + (
            '' if label[1] in self.config.params[self.config_map[label[0]]] else ' - Unsaved'))

    def set_value_from_widget(self, frame_name, section_name, field_meta, new_val):

        def check_array(new_obj):
            return np.array(new_obj[:-1] if (new_obj[-1][0] is None) else new_obj, dtype=np.float64)

        def set_tablefrom_vol(vols, tenor=None):
            # skip the header and the nulls at the end
            curve_array = []
            null_filter = slice(1, -1) if (vols[0][-1] is None) else slice(1, None)
            moneyness = vols[0][null_filter]
            for exp_vol in vols[null_filter]:
                exp = exp_vol[0]
                curve_array.extend(
                    [([money, exp, tenor, vol] if (tenor is not None) else [money, exp, vol]) for money, vol in
                     zip(moneyness, exp_vol[null_filter])])
            return curve_array

        def set_repr(obj, obj_type):
            if obj_type == 'Flot':
                new_obj = json.loads(obj)
                if field_name == 'Eigenvectors':
                    obj = []
                    for new_eigen_data in new_obj:
                        eigen_data = {}
                        eigen_data['Eigenvector'] = rf.utils.Curve([], check_array(new_eigen_data['data']))
                        eigen_data['Eigenvalue'] = float(new_eigen_data['label'])
                        obj.append(eigen_data)
                    return obj
                else:
                    return rf.utils.Curve([], check_array(new_obj[0]['data']))
            elif obj_type == 'Three':
                new_obj = json.loads(obj)
                if rate_type in rf.utils.TwoDimensionalFactors:
                    # 2d Surfaces
                    return rf.utils.Curve([2, 'Linear'], np.array(set_tablefrom_vol(new_obj), dtype=np.float64))
                else:
                    # 3d Spaces
                    vol_cube = []
                    for tenor, vol_surface in new_obj.items():
                        vol_cube.extend(set_tablefrom_vol(vol_surface, float(tenor)))
                    return rf.utils.Curve([3, 'Linear'], np.array(vol_cube, dtype=np.float64))
            elif obj_type == 'Percent':
                return rf.utils.Percent(100.0 * obj)
            elif obj_type == 'DatePicker':
                # might want to make this none by default
                return pd.Timestamp(obj) if obj else None
            else:
                return obj

        rate_type = rf.utils.check_rate_name(section_name)[0]
        field_name = field_meta['description'].replace(' ', '_')
        obj_type = field_meta.get('obj', field_meta['widget'])
        config = self.config.params[self.config_map[frame_name]]

        if section_name == "":
            config[field_name] = set_repr(new_val, obj_type)
        elif section_name in config and field_name in config[section_name]:
            config[section_name][field_name] = set_repr(new_val, obj_type)
        elif new_val != field_meta['default']:
            config.setdefault(section_name, {}).setdefault(field_name, set_repr(new_val, obj_type))
            # store the new object with all the usual defaults
            if frame_name == 'Factor':
                frame_defaults = self.data[self.tree.selected[-1]][frame_name]
            elif frame_name == 'Config':
                frame_defaults = self.sys_config
            else:
                frame_defaults = self.data[self.tree.selected[-1]][frame_name][rate_type]

            for new_field_meta in frame_defaults.values():
                new_field_name = new_field_meta['description'].replace(' ', '_')
                new_obj_type = new_field_meta.get('obj', new_field_meta['widget'])
                if new_field_name not in config[section_name]:
                    config[section_name][new_field_name] = set_repr(new_field_meta['value'], new_obj_type)

    def generate_handler(self, field_name, widget_elements, label):
        def handleEvent(change):
            # update the json representation
            widget_elements[field_name]['value'] = change['new']
            # update the value in the config object
            self.set_value_from_widget(label[0], label[1], widget_elements[field_name], change['new'])

        return handleEvent

    def create(self, val):
        factor_type = val[:val.find('.')]
        # load defaults for the new riskfactor
        self.data[val] = {'Factor': copy.deepcopy(self.new_factor['Factor'].get(factor_type)),
                          'Process': copy.deepcopy(self.new_factor['Process'].get(factor_type))}

    def delete(self, val):
        factor = rf.utils.check_rate_name(val)
        # delete the factor
        if val in self.config.params['Price Factors']:
            del self.config.params['Price Factors'][val]
        for model in self.config.params['Price Models']:
            if rf.utils.check_rate_name(model)[1:] == factor[1:]:
                del self.config.params['Price Models'][model]
                # also delete any correlations this model had
                # TODO!

        # delte the view data
        del self.data[val]

    def correlation_frame(self, stoch_proc_list):

        def generate_handler(correlation_matrix, process_names, process_lookup):
            def handleEvent(change):
                # update the value in the config object
                correlations = json.loads(change['new'])
                for proc1, proc2 in itertools.combinations(process_names, 2):
                    i, j = process_lookup[proc1], process_lookup[proc2]
                    key = (proc1, proc2) if (proc1, proc2) in correlation_matrix else (proc2, proc1)
                    correlation_matrix[key] = correlations[j][i] if correlations[j][i] != 0.0 else correlations[i][j]

            return handleEvent

        # generate dummy processes
        stoch_factors = {stoch_proc: rf.stochasticprocess.construct_process(
            stoch_proc.type, None, self.config.params['Price Models'][rf.utils.check_tuple_name(stoch_proc)])
            for stoch_proc in stoch_proc_list
            if rf.utils.check_tuple_name(stoch_proc) in self.config.params['Price Models']}

        num_factors = sum([x.num_factors() for x in stoch_factors.values()])

        # prepare the correlation matrix (and the offsets of each stochastic process)
        correlation_factors = []
        correlation_matrix = np.eye(num_factors, dtype=np.float32)

        for key, value in stoch_factors.items():
            proc_corr_type, proc_corr_factors = value.correlation_name
            for sub_factors in proc_corr_factors:
                correlation_factors.append(
                    rf.utils.check_tuple_name(rf.utils.Factor(proc_corr_type, key.name + sub_factors)))

        for index1 in range(num_factors):
            for index2 in range(index1 + 1, num_factors):
                factor1, factor2 = correlation_factors[index1], correlation_factors[index2]
                key = (factor1, factor2) if (factor1, factor2) in self.config.params['Correlations'] else (
                    factor2, factor1)
                rho = self.config.params['Correlations'].get(key, 0.0) if factor1 != factor2 else 1.0
                correlation_matrix[index1, index2] = rho
                correlation_matrix[index2, index1] = rho

        # label this container
        wig = [widgets.HTML(value='<h4>Correlation:</h4>')]

        w = Table(description="Matrix",
                  settings=to_json(
                      {
                          'columns': [{"type": "numeric",
                                       "numericFormat": {"pattern": "0.0000"}}] * num_factors,
                          'startCols': num_factors,
                          'startRows': num_factors,
                          'rowHeaders': correlation_factors,
                          'colHeaders': correlation_factors,
                          'width': 700,
                          'height': 300
                      }),
                  value=to_json(correlation_matrix.tolist())
                  )

        w.observe(generate_handler(
            self.config.params['Correlations'], correlation_factors,
            {x: i for i, x in enumerate(correlation_factors)}), 'value')
        wig.append(w)

        return widgets.VBox(children=wig)

    def model_config_frame(self):

        def generate_handler(model_config):
            def handleEvent(change):
                # update the value in the config object
                model_config.modeldefaults = {}
                model_config.modelfilters = {}
                for config in json.loads(change['new']):
                    if config and config[0] is not None:
                        factor, process = config[0].split('.')
                        if (config[1] is not None) and (config[2] is not None):
                            filter_on, value = config[1], config[2]
                            model_config.modelfilters.setdefault(factor, []).append(((filter_on, value), process))
                            continue
                        model_config.modeldefaults.setdefault(factor, process)

            return handleEvent

        # label this container
        wig = [widgets.HTML(value='<h4>Stochastic Process Mapping:</h4>')]

        optionmap = {k: v for k, v in rf.fields.mapping['Process_factor_map'].items() if v}
        col_types = [{"type": "dropdown",
                      "source": sorted(
                          reduce(operator.concat,
                                 [['{0}.{1}'.format(key, val) for val in values]
                                  for key, values in optionmap.items()], [])
                      ),
                      "strict": True}, {}, {}]
        model_config = [['{0}.{1}'.format(k, v), None, None] for k, v in
                        sorted(self.config.params['Model Configuration'].modeldefaults.items())]

        model_config.extend(
            reduce(operator.concat, [
                [['{0}.{1}'.format(k, rule[1]), rule[0][0], rule[0][1]] for rule in v]
                for k, v in sorted(self.config.params['Model Configuration'].modelfilters.items())
            ], [])
        )

        w = Table(description="Model Configuration",
                  settings=to_json({
                      'columns': col_types,
                      'startCols': 3,
                      'colHeaders': ["Risk_Factor.Stochastic_Process", "Where", "Equals"],
                      'contextMenu': True,
                      'minSpareRows': 1,
                      'width': 700,
                      'height': 300
                  }),
                  value=to_json(sorted(model_config) if model_config else [[None, None, None]])
                  )

        w.observe(generate_handler(self.config.params['Model Configuration']), 'value')
        wig.append(w)

        return widgets.VBox(children=wig)

    def calc_frames(self, selection):
        # riskfactors are only 1 level deep (might want to extend this) - TODO
        frame = None
        frames = []
        sps = set()
        if selection:
            key = selection[0]
            frame = self.data.get(key, {})
            if frame:
                # get the name
                factor_name = rf.utils.check_rate_name(key)
                factor = rf.utils.Factor(factor_name[0], factor_name[1:])
                # load the values:
                for frame_name, frame_value in sorted(frame.items()):
                    if frame_name == 'Process':
                        stoch_proc = self.config.params['Model Configuration'].search(
                            factor, self.config.params['Price Factors'].get(key, {}))
                        if stoch_proc:
                            full_name = rf.utils.Factor(stoch_proc, factor.name)
                            sps.add(full_name)
                            frames.append(
                                self.define_input(
                                    (frame_name, rf.utils.check_tuple_name(full_name)), frame_value[stoch_proc])
                            )
                        else:
                            frames.append(self.define_input((frame_name, ''), {}))
                    elif frame_name == 'Factor':
                        frames.append(self.define_input((frame_name, rf.utils.check_tuple_name(factor)), frame_value))

                # need to add a frame for correlations
                if self.tree.checked:
                    for selected in self.tree.checked:
                        factor_name = rf.utils.check_rate_name(selected)
                        factor = rf.utils.Factor(factor_name[0], factor_name[1:])
                        stoch_proc = self.config.params['Model Configuration'].search(factor, self.config.params[
                            'Price Factors'].get(selected, {}))
                        if stoch_proc:
                            full_name = rf.utils.Factor(stoch_proc, factor.name)
                            sps.add(full_name)
                    if len(sps) > 1:
                        frames.append(self.correlation_frame(sorted(sps)))

        # show the system config screen if there are no factors selected
        if not frame:
            frames.append(self.define_input(('Config', ''), self.sys_config))
            frames.append(self.model_config_frame())

        return frames


class CalculationPage(TreePanel):
    def __init__(self, config):
        super(CalculationPage, self).__init__(config)
        self.output = {}

    def get_label(self, label):
        """
        :param label: a list of text describing this calculation.
        :return: just prints the last 2 labels as the section and subsection header
        """
        return '<h4>{0}:</h4><i>{1}</i>'.format(*label[-2:])

    def parse_config(self):

        def load_items(config, field_name, field_data, storage):
            properties = field_data[field_name] if field_name in field_data else field_data[
                config[field_name]['Object']]
            for property_name, property_data in properties.items():
                value = copy.deepcopy(property_data)
                if 'sub_fields' in value:
                    load_items(config[field_name], property_name, {property_name: value['sub_fields']},
                               value['sub_fields'])
                value['value'] = self.get_value_for_widget(config, field_name, value)
                storage[property_name] = value

        self.data = {}

        # get the fields from the master list
        self.calculation_fields = self.load_fields(
            rf.fields.mapping['Calculation']['types'],
            rf.fields.mapping['Calculation']['fields']
        )

        calculation_to_add = []
        if self.config.deals.get('Calculation'):
            calc_type = self.config.deals['Calculation']['Object']
            ref_name = self.config.deals['Attributes'].get('Reference', 'Unknown')
            key = '{}.{}'.format(calc_type, ref_name)
            calculation_to_add.append({
                "text": key,
                "type": "default",
                "data": {},
                "children": []
            })
            node_data = {}
            load_items(self.config.deals, 'Calculation', self.calculation_fields, node_data)
            self.data[key] = {
                'frames': node_data, 'calc': calc_type, 'output': {}
            }

        self.tree_data = [{"text": "Calculations",
                           "type": "root",
                           "state": {"opened": True, "selected": False},
                           "children": calculation_to_add}]

        type_data = {"root": {"valid_children": ["default"]},
                     "default": {"valid_children": []}}

        # tree widget data
        # set the context menu to all risk factors defined
        context_menu = {
            "Create New Calculation": dict.fromkeys(
                rf.fields.mapping['Calculation']['types'].keys(), 'default')
        }

        # tree widget data
        self.tree = Tree(
            plugins=["contextmenu", "sort", "unique", "types", "search"],
            settings=to_json({'contextmenu': context_menu, 'events': ['create', 'delete']}),
            type_data=to_json(type_data),
            value=to_json(self.tree_data)
        )

    def get_results(self, selection, frame, key):

        def get_values(x):
            return {k: v['value'] if v['widget'] != 'Container' else get_values(
                v['sub_fields']) for k, v in x.items()}

        def make_float_widgets(x):
            return {k: {'widget': 'Float', 'description': k.replace('_', ' '), 'value': v} for k, v in x.items()}

        def make_container(label, x):
            return {label: {'widget': 'Container', 'description': label.replace('_', ' '),
                            'value': {k: v['value'] for k, v in x.items()}, 'sub_fields': x}}

        def make_results(results):
            out = {}
            for k, v in sorted(results.items(), key=lambda x: x[0][::-1]):
                filename_field = './tmp/{}.{}.csv'.format(key, k)
                link = {'widget': 'HTML', 'value': '<a href="{}" title="Click to Download">Download {}</a>'.format(filename_field, k)}
                if isinstance(v, pd.DataFrame):
                    # date index
                    if v.index.dtype.type == np.datetime64:
                        # clip the columns if there are too many to display
                        clipped_cols = v.columns[:8]
                        label_name = k.replace('_', ' ') if len(v.columns)<=8 else 'First {} Columns'.format(len(clipped_cols))  
                        Widget = {'widget': 'Flot', 'description': label_name,
                                  'hot_settings': {
                                      'columns': [{}] +
                                                 [{"type": "numeric",
                                                   "numericFormat": {"pattern": "0,0.00"}}] * len(clipped_cols),
                                      'manualColumnResize': True
                                  },
                                  'flot_settings': {'xaxis': {'mode': "time", 'timeBase': "milliseconds"}},
                                  'value': to_json(
                                      [{'label': c, 'data': [[x, y] for x, y in zip(v.index.view(np.int64) // 1000000,
                                                                                    v[c].values)]} for c in clipped_cols])
                                  }
                    else:
                        # multiindex not yet supported
                        r = v.reset_index()
                        dtypes = r.dtypes
                        r = r.replace({np.nan: None})
                        Widget = {'widget': 'Table', 'description': k.replace('_', ' '),
                                  'width': 650,
                                  'height': 300,
                                  'sub_types': [{"type": "numeric",
                                                 "numericFormat": {"pattern": "0,0.0000"}}
                                                if x.type in [np.float32, np.float64] else {} for x in dtypes],
                                  'col_names': r.columns.tolist(),
                                  'obj': ['Float' if x.type in [np.float32, np.float64] else 'Text' for x in dtypes],
                                  'value': to_json([x[-1].tolist() for x in r.iterrows()])
                                  }
                                  
                    with open(filename_field, 'wt') as fp:
                        v.to_csv(fp)                        
                    subfields = {k: Widget, 'link': link}
                    widget = {'widget': 'Container', 'description': k.replace('_', ' '), 'value': subfields, 'sub_fields': subfields}
                            
                elif isinstance(v, float):
                    widget = {'widget': 'Float', 'description': k.replace('_', ' '), 'value': v}
                    
                elif isinstance(v, dict):
                    if k=='scenarios':
                        multi_index = pd.MultiIndex.from_tuples(
                            reduce(operator.concat, [[(k2,)+ v1 for v1 in v2.index] for k2,v2 in v.items()]), names = ['factor', 'tenor', 'scenario'])
                    else:
                        multi_index = pd.MultiIndex.from_tuples(reduce(operator.concat, [[(k2, v1) for v1 in v2.index] for k2,v2 in v.items()]))
                    df = pd.concat(v.values()).set_index(multi_index)
                    with open(filename_field, 'wt') as fp:
                        df.to_csv(fp)
                    widget = link
                else:
                    print('output widget for ', k, ' - TODO')
                    continue

                out[k] = widget

            return out

        def click(widget):
            calc = frame['calc']
            input = frame['frames']
            param = get_values(input)
            # send the name of the calc to the calculation engine
            param.update({'calc_name': key, 'Object': calc})
            # update the parameters for the current calculation
            rf.update_dict(self.config.deals['Calculation'], param)
            # get the output
            calc_obj, calc_output = self.context.run_job()
            # all calculations provide stats
            output = make_container('Stats', make_float_widgets(calc_output['Stats']))
            # now look at the results and find suitable widgets
            output.update(make_container('Output', make_results(calc_output['Results'])))
            # flag the gui that we have output
            frame['output'] = output
            # trigger redraw
            self._on_selected_changed({'new': selection})
            # store the output
            self.output[key] = calc_output

        return click

    def calc_frames(self, selection):
        frames = []
        if selection:
            key = selection[0]
            frame = self.data.get(key, {})

            if frame:
                input = self.define_input(key.split('.'), frame['frames'])
                # add the calc button
                execute_button = widgets.Button(description='Execute', tooltip='Run the Calculation')
                execute_button.on_click(self.get_results(selection, frame, key))

                input.children = input.children + (execute_button,)

                frames.append(input)
                # now the output
                if frame['output']:
                    output = self.define_input([key, 'Results'], frame['output'])
                    frames.append(output)

        return frames

    def generate_handler(self, field_name, widget_elements, label):
        def handleEvent(change):
            # update the json representation
            widget_elements[field_name]['value'] = change['new']
            # we could update the value in the config object if we wanted to save the calculation back,
            # but we won't
            # self.SetValueFromWidget ( label[0], label[1], widget_elements[field_name], new_value )

        return handleEvent

    def create(self, val):
        key = val[0]
        calc_type = key[:key.find('.')]
        reference = key[key.find('.') + 1:]
        # print(key, calc_type, reference)
        # load defaults for the new riskfactor
        self.data[key] = {
            'frames': copy.deepcopy(self.calculation_fields.get(calc_type)),
            'calc': calc_type, 'output': {}
        }

    def delete(self, val):
        if val[0] in self.data:
            del self.data[val[0]]


class Workbench(object):
    '''
    Main class for the UI
    Needs to be able to load and save config files/trade files (in JSON)
    '''

    def __init__(self, path='', path_transform={}, file_transform={}):

        def select_json(chooser):
            self.context.file_map = {k: v for k, v in self.file_transform.items() if k}
            self.context.path_map = {k: v for k, v in self.path_transform.items() if k}
            logging.info('loading {}'.format(chooser.selected))
            self.context.load_json(chooser.selected, compress=self.compress_json.value)
            self.reload()
            # select the portfolio index
            self.tabs.selected_index = 1
            logging.info('{} loaded'.format(chooser.selected))

        def make_new_map(map_type, map_data):

            def make_frame(kv_map):

                def lock_key(kv_map, key):
                    def handle(change):
                        kv_map[key.value] = change.new

                    return handle

                w = []
                for k, v in kv_map.items():
                    old = widgets.Text(description='{} name (Old)'.format(map_type), value=k)
                    new = widgets.Text(description='{} name (New)'.format(map_type), value=v)
                    new.observe(lock_key(kv_map, old), 'value')
                    w.append(widgets.VBox(children=[old, new],
                                          layout=widgets.Layout(padding='5px', border='outset', max_width='700px')))
                return w

            def add_map_button(e):
                map_data[''] = ''
                frame.children = make_frame(map_data)

            def del_map_button(e):
                if map_data:
                    last = list(map_data.keys())[-1]
                    del map_data[last]
                    frame.children = make_frame(map_data)

            add_map = widgets.Button(description='Add new {} map'.format(map_type),
                                     tooltip='Remaps occurances of the old name with the new one when loading JSON')
            add_map.on_click(add_map_button)

            del_map = widgets.Button(description='Delete {} map'.format(map_type),
                                     tooltip='Removes the last mapping occurance')
            del_map.on_click(del_map_button)
            frame = widgets.VBox(children=make_frame(map_data))
            buttons = widgets.HBox(children=[add_map, del_map])
            return widgets.VBox(children=[frame, buttons])

        self.context = rf.Context()
        self.path = path
        self.path_transform = path_transform
        self.file_transform = file_transform

        # load the file selectors

        self.path_map = make_new_map('Path', self.path_transform)
        self.file_map = make_new_map('File', self.file_transform)
        self.json_file = FileChooser(path, title='Load JSON File', filter_pattern='*.json')        
        self.json_file.register_callback(select_json)
        self.compress_json = widgets.Checkbox(
            value=False, description='Compress deals', disabled=False, indent=False)
        self.filename = widgets.VBox(children=[self.path_map, self.file_map, widgets.HBox(
            children=[self.json_file, self.compress_json])], layout=widgets.Layout(border='5px outset black'),
            _dom_classes=['mainframe'])
            
        # the main containers/widgets
        self.main = None
        self.log_area = widgets.Textarea()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logstream = MyStream(self.log_area)
        self.loghandler = logging.StreamHandler(self.logstream)
        self.loghandler.setFormatter(formatter)

        self.log = logging.getLogger()
        self.log.handlers = []
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(self.loghandler)
        # build the UI
        self.portfolio = PortfolioPage(self.context)
        self.factors = RiskFactorsPage(self.context)
        self.calculations = CalculationPage(self.context)

        self.tabs = widgets.Tab(children=[self.filename,
                                          self.portfolio.main_container,
                                          self.factors.main_container,
                                          self.calculations.main_container])

        self.tabs.set_title(0, 'JSON File')
        self.tabs.set_title(1, 'Portfolio')
        self.tabs.set_title(2, 'Price Factors')
        self.tabs.set_title(3, 'Calculations')

        # draw a new one
        self.main = widgets.VBox()
        self.main.children = [self.tabs, self.log_area]

        display(self.main)

    def reload(self):
        self.portfolio = PortfolioPage(self.context)
        self.factors = RiskFactorsPage(self.context)
        self.calculations = CalculationPage(self.context)

        self.tabs.children = [self.filename, self.portfolio.main_container,
                              self.factors.main_container, self.calculations.main_container]

    def __del__(self):
        self.loghandler.close()


if __name__ == '__main__':
    rundate = '2022-07-07'
    if os.name == 'nt':
        path = os.path.join('U:\\CVA_JSON', rundate)
        path_transform = {
            '\\\\ICMJHBMVDROPPRD\\AdaptiveAnalytics\\Inbound\\MarketData':
                '\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData'}
    else:
        path = os.path.join('/media/vretiel/3EFA4BCDFA4B7FDF/Media/Data/crstal/CVA_JSON', rundate)
        path_transform = {
            '//ICMJHBMVDROPPRD/AdaptiveAnalytics/Inbound/MarketData':
                '/media/vretiel/3EFA4BCDFA4B7FDF/Media/Data/crstal/CVA_JSON'}

    cx = rf.Context(
        path_transform=path_transform,
        file_transform={
            'CVAMarketData_Calibrated.dat': 'CVAMarketData_Calibrated_New.json',
            'MarketData.dat': 'MarketData.json'
        })
    cx.load_json(os.path.join(path, 'InputAAJ_CrB_JPMorgan_Chase_NYK_ISDA.json'))
    # cx.load_json(os.path.join(path, 'InputAAJ_CrB_BNP_Paribas__Paris__ISDA.json'))
    # cp = CalculationPage(cx)
    rp = RiskFactorsPage(cx)
