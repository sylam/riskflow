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
from IPython.display import display  # Used to display widgets in the notebook

# riskflow specific stuff
import riskflow as rf
# load the widgets
from riskflow_widgets import Tree, Table, Flot, FlotTree, Three


def to_json(string):
    """converts a string to json - skips the whitespace for a smaller output"""
    return json.dumps(string, separators=(',', ':'))


# portfolio parent data
class DealCache:
    Json = 0
    Deal = 1
    Parent = 2
    Count = 3


# Code for custom pages go here
class TreePanel(object):
    '''
    Base class for tree-based screens with a dynamic panel on the right to show data for the screen.
    '''

    def __init__(self, config):
        # load the config object - to load and store state
        self.config = config

        # load up relevant information from the config and define the tree type
        self.parse_config()

        # setup the view containters
        self.right_container = widgets.VBox()
        # update the style of the right container
        self.right_container._dom_classes = ['rightframe']

        self.tree.selected = u""
        self.tree._dom_classes = ['generictree']

        # event handlers
        self.tree.observe(self._on_selected_changed, 'selected')
        self.tree.observe(self._on_created_changed, 'created')
        self.tree.observe(self._on_deleted_changed, 'deleted')
        self.tree.on_displayed(self._on_displayed)
        # default layout for the widgets
        self.default_layout = widgets.Layout(min_height='30px')

        # interface for the tree and the container
        self.main_container = widgets.HBox(
            children=[self.tree, self.right_container],
            layout=widgets.Layout(border='5px outset black'),
            _dom_classes=['mainframe']
        )

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
           Handles both deal as well as factor data.

           For deals, config is the instrument object from the model and the section_name
           is usually the 'field' attribute.

           For factors, config is the config section from the model and the section_name
           is the name of the factor/process.

            The field_meta contains the actual field names.
        '''

        def load_table_from_vol(vol_factor, vols):
            table = [['Term to maturity/Moneyness'] + list(vol_factor.get_moneyness())]
            for term, vol in zip(vol_factor.get_expiry(), vols):
                table.append([term] + (list(vol) if isinstance(vol, np.ndarray) else [vol]))
            return table

        def get_repr(obj, field_name, default_val):

            if isinstance(obj, rf.utils.Curve):
                if obj.meta and obj.meta[0]!='Integrated':
                    # need to use a threeview
                    if obj.meta[0] == 2:
                        t = rf.riskfactors.Factor2D({'Surface': obj})
                        vols = t.get_vols()
                        vol_space = load_table_from_vol(t, vols)
                    else:
                        t = rf.riskfactors.Factor3D({'Surface': obj})
                        vol_cube = t.get_vols()
                        if len(vol_cube.shape) == 2:
                            vol_space = {t.get_tenor()[0]: load_table_from_vol(t, vol_cube)}
                        elif len(vol_cube.shape) == 3:
                            vol_space = {}
                            for index, tenor in enumerate(t.get_tenor()):
                                vol_space.setdefault(tenor, load_table_from_vol(t, vol_cube[index]))

                    return_value = to_json(vol_space)
                else:
                    return_value = to_json([{'label': 'None', 'data': [[x, y] for x, y in obj.array]}])
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
                elif field_name == 'Sampling_Data':
                    headings = ['Date', 'Price', 'Weight']
                    widgets = ['DatePicker', 'Float', 'Float']
                    data = []
                    for flow in obj:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = to_json(data)
                else:
                    raise Exception('Unknown Array Field type {0}'.format(field_name))
            elif isinstance(obj, pd.DateOffset):
                return_value = ''.join(['%d%s' % (v, rf.config.Context.reverse_offset[k]) for k, v in obj.kwds.items()])
            elif isinstance(obj, pd.Timestamp):
                return_value = obj.strftime('%Y-%m-%d')
            elif obj is None:
                return_value = default_val
            else:
                return_value = obj

            # return the value
            return return_value

        # update an existing factor
        field_name = field_meta['description'].replace(' ', '_')
        if section_name in config and field_name in config[section_name]:
            # return the value in the config obj
            obj = config[section_name][field_name]
            return get_repr(obj, field_name, field_meta['value'])
        else:
            return field_meta['value']

    def parse_config(self):
        pass

    def generate_handler(self, field_name, widget_elements, label):
        pass

    def define_input(self, label, widget_elements):
        # label this container
        wig = [widgets.HTML()]
        vals = [self.get_label(label)]

        for field_name, element in sorted(widget_elements.items()):
            # skip this element if it's not visible
            if element.get('isvisible') == 'False':
                continue
            if element['widget'] == 'Dropdown':
                w = widgets.Dropdown(options=element['values'], description=element['description'],
                                     layout=dict(width='420px'))
                vals.append(element['value'])
            elif element['widget'] == 'Text':
                w = widgets.Text(description=element['description'])
                vals.append(str(element['value']))
            elif element['widget'] == 'Container':
                new_label = label + [
                    element['description']] if isinstance(label, list) else [element['description']]
                w, v = self.define_input([x.replace(' ', '_') for x in new_label], element['sub_fields'])
                vals.append(v)
            elif element['widget'] == 'Flot':
                w = Flot(description=element['description'])
                vals.append(element['value'])
            elif element['widget'] == 'Three':
                w = Three(description=element['description'])
                vals.append(element['value'])
            elif element['widget'] == 'Integer':
                w = widgets.IntText(description=element['description'])
                vals.append(element['value'])
            elif element['widget'] == 'TreeFlot':
                w = FlotTree(description=element['description'])
                w.type_data = element['type_data']
                w.profiles = element['profiles']
                vals.append(element['value'])
            elif element['widget'] == 'HTML':
                w = widgets.HTML()
                vals.append(element['value'])
            elif element['widget'] == 'Table':
                w = Table(description=element['description'], colTypes=to_json(element['sub_types']),
                          colHeaders=element['col_names'])
                vals.append(element['value'])
            elif element['widget'] == 'Float':
                w = widgets.FloatText(description=element['description'])
                vals.append(element['value'])
            elif element['widget'] == 'DatePicker':
                w = widgets.DatePicker(description=element['description'], layout=dict(width='420px'))
                vals.append(pd.Timestamp(element['value']))
            elif element['widget'] == 'BoundedFloat':
                w = widgets.BoundedFloatText(min=element['min'], max=element['max'], description=element['description'])
                vals.append(element['value'])
            else:
                raise Exception('Unknown widget field')

            if element['widget'] != 'Container':
                w.observe(self.generate_handler(field_name, widget_elements, label), 'value')

            wig.append(w)

        container = widgets.VBox(children=wig)
        return container, vals

    def get_label(self, label):
        return ''

    def calc_frames(self, selection):
        pass

    def _on_selected_changed(self, change):

        def update_frame(frame, values):
            # set the style
            frame._dom_classes = ['genericframe']
            # update the values in the frame
            for index, child in enumerate(frame.children):
                # recursively update the frames if needed
                if isinstance(values[index], list):
                    update_frame(child, values[index])
                else:
                    child.value = values[index]

        frames = self.calc_frames(change['new'])

        # set the value of all widgets in the frame . . .
        # need to do this last in case all widgets haven't fully rendered in the DOM
        for container, value in frames:
            update_frame(container, value)

    def create(self, newval):
        pass

    def delete(self, newval):
        pass

    def _on_created_changed(self, change):
        if self.tree.created:
            # print 'create',val
            self.create(change['new'])
            # reset the created flag
            self.tree.created = ''

    def _on_deleted_changed(self, change):
        if self.tree.deleted:
            # print 'delete',val
            self.delete(change['new'])
            # reset the deleted flag
            self.tree.deleted = ''

    def _on_displayed(self, e):
        self.tree.value = to_json(self.tree_data)

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
                return pd.Timestamp(obj) if obj else None
            elif obj_type == 'Float':
                try:
                    new_obj = obj if obj == '<undefined>' else float(obj)
                except ValueError:
                    print('FLOAT', obj, field_meta)
                    new_obj = 0.0
                return new_obj
            elif obj_type == 'Container':
                return json.loads(obj)
            else:
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
            settings=to_json({'contextmenu': context_menu, 'events':['create','delete']}),
            type_data=to_json(type_data),
            #, value=to_json(self.tree_data)
        )

        # have a placeholder for the selected model (deal)
        self.current_deal = None
        self.current_deal_parent = None

    def calc_frames(self, selection):
        key = tuple(selection)
        frame, self.current_deal, self.current_deal_parent, count = self.data.get(key, [{}, None, None, 0])

        # factor_fields
        if frame:
            frames = []
            # get the object type - this should always be defined
            obj_type = frame['Object']['value']
            # get the instrument obj
            # instrument_obj = self.current_deal['Instrument']

            for frame_name in rf.fields.mapping['Instrument']['types'][obj_type]:
                # load the values:
                instrument_fields = rf.fields.mapping['Instrument']['sections'][frame_name]
                frame_fields = {k: v for k, v in frame.items() if k in instrument_fields}
                frames.append(self.define_input((frame_name, key[-1]), frame_fields))

            # only store the container (first component)
            self.right_container.children = [x[0] for x in frames]
            return frames
        else:
            # load up a set of defaults
            self.right_container = widgets.VBox()
            return []

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

        type_data = {"root": {"icon": "fa fa-folder", "valid_children": ["default"]},
                     "default": {"icon": "fa fa-file", "valid_children": []}}

        # simple lookup to match the params in the config file to the json in the UI
        self.config_map = {'Factor': 'Price Factors', 'Process': 'Price Models', 'Config': 'System Parameters'}
        # fields to store new objects
        self.new_factor = {'Factor': risk_factor_fields, 'Process': possible_risk_process}
        # set the context menu to all risk factors defined
        context_menu = {"New Risk Factor": {x: "default" for x in sorted(risk_factor_fields.keys())}}

        # tree widget data
        self.tree = Tree(
            plugins = ["contextmenu", "sort", "unique", "types", "search", "checkbox"],
            settings = to_json({'contextmenu': context_menu, 'events': ['create', 'delete']}),
            type_data = to_json(type_data)
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
                frame_defaults = self.data[self.tree.selected][frame_name]
            elif frame_name == 'Config':
                frame_defaults = self.sys_config
            else:
                frame_defaults = self.data[self.tree.selected][frame_name][rate_type]

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

        container = widgets.VBox()

        # label this container
        wig = [widgets.HTML()]
        vals = ['<h4>Correlation:</h4>']

        col_types = '[' + ','.join(['{ "type": "numeric", "format": "0.0000" }'] * len(correlation_factors)) + ']'

        w = Table(description="Matrix", colTypes=col_types, colHeaders=correlation_factors)
        vals.append(to_json(correlation_matrix.tolist()))
        w.observe(generate_handler(
            self.config.params['Correlations'], correlation_factors,
            {x: i for i, x in enumerate(correlation_factors)}), 'value')
        wig.append(w)

        # print correlation_factors, correlation_matrix, col_headers

        container.children = wig
        return container, vals

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

        container = widgets.VBox()

        # label this container
        wig = [widgets.HTML()]
        vals = ['<h4>Stochastic Process Mapping:</h4>']

        optionmap = {k: v for k, v in rf.fields.mapping['Process_factor_map'].items() if v}
        col_types = to_json([{"type": "dropdown", "source": sorted(reduce(operator.concat, [
            ['{0}.{1}'.format(key, val) for val in values] for key, values in optionmap.items()], [])), "strict": True},
                                {}, {}])
        model_config = [['{0}.{1}'.format(k, v), None, None] for k, v in
                        sorted(self.config.params['Model Configuration'].modeldefaults.items())]
        model_config.extend(reduce(operator.concat,
                                   [[['{0}.{1}'.format(k, rule[1]), rule[0][0], rule[0][1]] for rule in v] for k, v in
                                    sorted(self.config.params['Model Configuration'].modelfilters.items())], []))
        w = Table(description="Model Configuration", colTypes=col_types,
                  colHeaders=["Risk_Factor.Stochastic_Process", "Where", "Equals"])
        vals.append(to_json(sorted(model_config) if model_config else [[None, None, None]]))
        w.observe(generate_handler(self.config.params['Model Configuration']), 'value')
        wig.append(w)

        container.children = wig
        return container, vals

    def calc_frames(self, selection):
        # riskfactors are only 1 level deep (might want to extend this) - TODO
        key = selection[0]
        frame = self.data.get(key, {})
        frames = []
        sps = set()

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

        self.right_container.children = [x[0] for x in frames]

        return frames


class CalculationPage(TreePanel):
    def __init__(self, config):
        super(CalculationPage, self).__init__(config)

    def GetLabel(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format(label[0], label[1])

    def ParseConfig(self):
        self.data = {}
        self.calculation_fields = self.load_fields(rf.fields.mapping['Calculation']['types'],
                                                   rf.fields.mapping['Calculation']['fields'])

        self.tree_data = [{"text": "Calculations",
                           "type": "root",
                           "state": {"opened": True, "selected": False},
                           "children": []}]

        type_data = {"root": {"valid_children": ["default"]},
                     "default": {"valid_children": []}}

        # tree widget data
        self.tree = CalculationTree()
        self.tree.type_data = to_json(type_data)
        self.tree.calculation_types = to_json(dict.fromkeys(rf.fields.mapping['Calculation']['types'].keys(), 'default'))

    def GenerateSlides(self, calc, filename, output):
        nb = nbf.new_notebook()
        cells = []

        for cell in calc.slides:
            nb_cell = nbf.new_text_cell('markdown', cell['text'])
            nb_cell['metadata'] = cell['metadata']
            cells.append(nb_cell)

        nb['metadata'] = {'celltoolbar': 'Slideshow'}
        nb['worksheets'].append(nbf.new_worksheet(cells=cells))

        with open(filename + '.ipynb', 'w') as f:
            nbf.write(nb, f, 'ipynb')

        # now generate the slideshow
        call([r'C:\python27\Scripts\ipython', 'nbconvert', filename + '.ipynb', '--to', 'slides', '--reveal-prefix',
              "http://cdn.jsdelivr.net/reveal.js/2.6.2"])

        # let the gui know about the slides - need to create a custom widget for the label - this is hacky
        output[
            'value'] = '<div class="widget-hbox-single"><div class="widget-hlabel" style="display: block;">Slides</div><a href=\'{0}\' target=''_blank''>{0}</a></div>'.format(
            filename + '.slides.html')

    def GetResults(self, selection, frame, key):

        def Define_Click(widget):
            calc = frame['calc']
            input = frame['frames']['input']
            output = frame['frames']['output']

            param = {k: v['value'] for k, v in input.items()}
            # send the name of the calc to the calculation engine
            param['calc_name'] = key
            result = calc.Execute(param)

            # Disable unneeded fields
            for k, v in input.items():
                if v.get('Output'):
                    output[v['Output']]['isvisible'] = 'True' if v['value'] == 'Yes' else 'False'

            # Format the results
            calc.FormatOutput(result, output)

            # generate slides
            if calc.slides:
                self.GenerateSlides(calc, key[0], output['Slideshow'])

            # flag the gui that we have output
            frame['output'] = True
            # trigger redraw
            self._on_selected_changed({'new': selection})

        return Define_Click

    def CalcFrames(self, selection):
        key = tuple(json.loads(selection))
        frame = self.data.get(key, {})

        if frame:
            frames = []
            input = self.DefineInput(key[-1].split('.'), frame['frames']['input'])

            # add the calc button
            execute_button = widgets.Button(description='Execute')
            execute_button.on_click(self.GetResults(selection, frame, key))

            input[0].children = input[0].children + (execute_button,)
            input[1].append('')

            frames.append(input)

            # now the output
            if frame['output']:
                output = self.DefineInput([key[-1], 'Results'], frame['frames']['output'])
                frames.append(output)

            self.right_container.children = [x[0] for x in frames]
            return frames
        else:
            # load up a set of defaults
            self.right_container = widgets.VBox()
            return []

    def GenerateHandler(self, field_name, widget_elements, label):
        def handleEvent(change):
            # update the json representation
            widget_elements[field_name]['value'] = change['new']
            # update the value in the config object
            # self.SetValueFromWidget ( label[0], label[1], widget_elements[field_name], new_value )

        return handleEvent

    def Create(self, val):
        key = tuple(json.loads(val))
        calc_type = key[-1][:key[-1].find('.')]
        reference = key[-1][key[-1].find('.') + 1:]
        # print key, calc_type, reference
        # load defaults for the new riskfactor
        self.data[key] = {'frames': copy.deepcopy(self.calculation_fields.get(calc_type)),
                          'calc': ConstructCalculation(calc_type, self.config), 'output': False}

    def Delete(self, val):
        del self.data[tuple(json.loads(val))]
