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
from riskflow_widgets import Tree as PortfolioTree, Table


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
            table = [['Term to maturity/Moneyness'] + list(vol_factor.GetMoneyness())]
            for term, vol in zip(vol_factor.GetExpiry(), vols):
                table.append([term] + (list(vol) if isinstance(vol, np.ndarray) else [vol]))
            return table

        def get_repr(obj, field_name, default_val):

            if isinstance(obj, rf.utils.Curve):
                if obj.meta:
                    # need to use a threeview
                    if obj.meta[0] == 2:
                        t = rf.riskfactors.Factor2D({'Surface': obj})
                        vols = t.GetVols()
                        vol_space = load_table_from_vol(t, vols)
                    else:
                        t = rf.riskfactors.Factor3D({'Surface': obj})
                        vol_cube = t.GetVols()
                        if len(vol_cube.shape) == 2:
                            vol_space = {t.GetTenor()[0]: load_table_from_vol(t, vol_cube)}
                        elif len(vol_cube.shape) == 3:
                            vol_space = {}
                            for index, tenor in enumerate(t.GetTenor()):
                                vol_space.setdefault(tenor, load_table_from_vol(t, vol_cube[index]))

                    return_value = json.dumps(vol_space)
                else:
                    return_value = json.dumps([{'label': 'None', 'data': [[x, y] for x, y in obj.array]}])
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
                return_value = json.dumps(data)
            elif isinstance(obj, rf.utils.DateList):
                data = [[get_repr(date, 'Date', default_val),
                         get_repr(value, 'Value', default_val)] for date, value in obj.data.items()]
                return_value = json.dumps(data)
            elif isinstance(obj, rf.utils.CreditSupportList):
                data = [[get_repr(value, 'Value', default_val),
                         get_repr(rating, 'Value', default_val)] for rating, value in obj.data.items()]
                return_value = json.dumps(data)
            elif isinstance(obj, list):
                if field_name == 'Eigenvectors':
                    madness = []
                    for i, element in enumerate(obj):
                        madness.append(
                            {'label': str(element['Eigenvalue']),
                             'data': [[x, y] for x, y in element['Eigenvector'].array]})
                    return_value = json.dumps(madness)
                elif field_name in ['Properties', 'Items', 'Cash_Collateral', 'Equity_Collateral',
                                    'Bond_Collateral', 'Commodity_Collateral']:
                    data = []
                    for flow in obj:
                        data.append(
                            [get_repr(flow.get(field_name), field_name, rf.fields.default.get(widget_type, default_val))
                             for field_name, widget_type in zip(field_meta['col_names'], field_meta['obj'])])
                    return_value = json.dumps(data)
                elif field_name == 'Resets':
                    headings = ['Reset_Date', 'Start_Date', 'End_Date', 'Year_Fraction', 'Use Known Rate', 'Known_Rate']
                    widgets = ['DatePicker', 'DatePicker', 'DatePicker', 'Float', 'Text', 'Float']
                    data = []
                    for flow in obj:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = json.dumps(data)
                elif field_name in ['Description', 'Tags']:
                    return_value = json.dumps(obj)
                elif field_name == 'Sampling_Data':
                    headings = ['Date', 'Price', 'Weight']
                    widgets = ['DatePicker', 'Float', 'Float']
                    data = []
                    for flow in obj:
                        data.append([get_repr(item, field, rf.fields.default.get(widget_type, default_val)) for
                                     field, item, widget_type in zip(headings, flow, widgets)])
                    return_value = json.dumps(data)
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
                w = Table(description=element['description'], colTypes=json.dumps(element['sub_types']),
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
        self.tree.value = json.dumps(self.tree_data)

    def show(self):
        display(self.main_container)


class PortfolioPage(TreePanel):

    def __init__(self, config):
        super(PortfolioPage, self).__init__(config)

    def get_label(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format(label[0], label[1]) if type(
            label) is tuple else '<h4>{0}</h4>'.format('.'.join(label))

    def SetValueFromWidget(self, instrument, field_meta, new_val):

        def checkArray(new_obj):
            return [x for x in new_obj if x[0] is not None] if new_obj is not None else None

        def setRepr(obj, obj_type):
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
                return [[setRepr(data_obj, data_type) for data_obj, data_type in zip(reset, field_types)]
                        for reset in resets]
            elif isinstance(obj_type, list):
                if field_name == 'Description':
                    return json.loads(obj)
                elif field_name == 'Sampling_Data':
                    new_obj = checkArray(json.loads(obj))
                    return [[setRepr(data_obj, data_type) for data_obj, data_type, field in
                             zip(data_row, obj_type, field_meta['col_names'])] for data_row in
                            new_obj] if new_obj else []
                elif field_name in ['Items', 'Properties', 'Cash_Collateral',
                                    'Equity_Collateral', 'Bond_Collateral', 'Commodity_Collateral']:
                    new_obj = checkArray(json.loads(obj))
                    return [{field: setRepr(data_obj, data_type) for data_obj, data_type, field in
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

        instrument[field_name] = setRepr(new_val, obj_type)

        # try:
        #     instrument[field_name] = setRepr(new_val, obj_type)
        # except:
        #    logging.debug('Field {0} could not be set to {1} - obj type {2}'.format(field_name, new_val, obj_type))

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

            self.SetValueFromWidget(instrument, widget_elements[field_name], change['new'])

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
        self.tree = PortfolioTree(
            plugins=["sort", "types", "search", "unique"],
            context_menu=json.dumps(context_menu),
            type_data=json.dumps(type_data),
            #, value=json.dumps(self.tree_data)
        )

        # have a placeholder for the selected model (deal)
        self.current_deal = None
        self.current_deal_parent = None

    def calc_frames(self, selection):
        key = tuple(json.loads(selection))
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
        key = tuple(json.loads(val))
        instrument_type = key[-1][:key[-1].find('.')]
        reference = key[-1][key[-1].find('.') + 1:]

        # load defaults for the new object
        fields = self.instrument_fields.get(instrument_type)
        ins = {}

        for value in fields.values():
            self.SetValueFromWidget(ins, value, value['value'])

        # set it up
        ins['Object'] = instrument_type
        ins['Reference'] = reference

        # Now check if this is a group or a regular deal
        if instrument_type in rf.fields.mapping['Instrument']['groups']['STR'][1]:
            deal = {'instrument': rf.instrument.construct_instrument(ins), 'Children': []}
        else:
            deal = {'instrument': rf.instrument.construct_instrument(ins)}

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
        key = tuple(json.loads(val))
        reference = key[-1][key[-1].find('.') + 1:]
        parent = self.data[key][DealCache.Parent]

        print(key, parent)
        # delete the deal
        parent['Children'].remove(self.data[key][DealCache.Deal])
        # decrement the count
        self.data[key[:-1]][DealCache.Count][reference] -= 1
        # delte the view data
        del self.data[key]
