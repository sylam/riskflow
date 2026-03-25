import os
import copy
import operator
import logging
import numpy as np
import pandas as pd
import ipywidgets as widgets

from base64 import b64encode
from functools import reduce
from IPython.display import display, Javascript

# load the widgets - might be able to use more native objects instead of unicode text - TODO!
from riskflow_widgets import FileDragUpload, Tree, Table, Flot, Three, to_json
from riskflow_jupyter import TreePanel

class MainPage(TreePanel):
    def __init__(self, config):
        super(MainPage, self).__init__(config)
        self.output = {}

    def get_label(self, label):
        """
        :param label: a list of text describing this calculation.
        :return: just prints the last label as the subsection header
        """

        return '<h4>{0}:</h4>'.format(label[-1])

    def parse_config(self):

        self.data = {}
        calculation_to_add = []
        if self.config.deals.get('Calculation'):
            calc_type = self.config.deals['Calculation']['Object']
            ref_name = self.config.deals['Attributes'].get('Reference', 'Unknown')
            key = self.config.deals['Calculation'].get('calc_name', '{}.{}'.format(calc_type, ref_name))
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
                           "state": {"opened": True, "selected": True},
                           "children": calculation_to_add}]

        type_data = {"root": {"valid_children": ["default"]},
                     "default": {"valid_children": []}}

        # tree widget data
        self.tree = Tree(
            plugins=["sort", "unique", "types", "search"],
            settings=to_json({'events': []}),
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
                link = {'widget': 'HTML',
                        'value': '<a href="{}" title="Right Click (Save link as) to Download">Download {}</a>'.format(filename_field, k)}
                if isinstance(v, pd.DataFrame):
                    # date index
                    if v.index.dtype.type == np.datetime64:
                        # clip the columns if there are too many to display
                        clipped_cols = v.columns[:8]
                        Widget = {'widget': 'Flot', 'description': '',
                                  'hot_settings': {
                                      'columns': [{}] +
                                                 [{"type": "numeric",
                                                   "numericFormat": {"pattern": "0,0.00"}}] * len(clipped_cols),
                                      'manualColumnResize': True
                                  },
                                  'flot_settings': {'xaxis': {'mode': "time", 'timeBase': "milliseconds"}},
                                  'value': to_json(
                                      [{'label': c, 'data': [[x, y] for x, y in zip(v.index.view(np.int64) // 1000000,
                                                                                    v[c].values)]} for c in
                                       clipped_cols])
                                  }
                    else:
                        # multiindex not yet supported
                        r = v.reset_index()
                        dtypes = r.dtypes
                        r = r.replace({np.nan: None})
                        Widget = {'widget': 'Table', 'description': '',
                                  'width': 700,
                                  'height': 300,
                                  'sub_types': [{"type": "numeric",
                                                 "numericFormat": {"pattern": "0,0.0000"}}
                                                if x.type in [np.float32, np.float64] else {} for x in dtypes],
                                  'col_names': r.columns.tolist(),
                                  'obj': ['Float' if x.type in [np.float32, np.float64] else 'Text' for x in dtypes],
                                  'value': to_json([x[-1].tolist() for x in r.iterrows()])
                                  }

                    os.makedirs(os.path.dirname(filename_field), exist_ok=True)
                    with open(filename_field, 'wt') as fp:
                        v.to_csv(fp)
                    subfields = {k: Widget, 'link': link}
                    widget = {'widget': 'Container', 'description': k.replace('_', ' '), 'value': subfields,
                              'sub_fields': subfields}

                elif isinstance(v, float):
                    widget = {'widget': 'Float', 'description': k.replace('_', ' '), 'value': v}

                elif isinstance(v, dict):
                    if k == 'scenarios':
                        multi_index = pd.MultiIndex.from_tuples(
                            reduce(operator.concat, [[(k2,) + v1 for v1 in v2.index] for k2, v2 in v.items()]),
                            names=['factor', 'tenor', 'scenario'])
                    else:
                        multi_index = pd.MultiIndex.from_tuples(
                            reduce(operator.concat, [[(k2, v1) for v1 in v2.index] for k2, v2 in v.items()]))
                    df = pd.concat(v.values()).set_index(multi_index)
                    with open(filename_field, 'wt') as fp:
                        df.to_csv(fp)
                    widget = link
                else:
                    logging.warning('output widget for {}  - TODO'.format(k))
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
            try:
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
            except Exception as e:
                logging.root.name = key
                logging.error('Cannot execute calculation - {}'.format(e.args))

        return click

    def download_config(self, key, output):
    
        def trigger_download(text, filename, kind='text/json'):
            try:
                content_b64 = b64encode(text.encode('utf-8')).decode('utf-8')
                data_url = f'data:{kind};charset=utf-8;base64,{content_b64}'
                js_code = f"""                
                    var a = document.createElement('a');
                    a.setAttribute('download', '{filename}');
                    a.setAttribute('href', '{data_url}');
                    a.style.display = 'none';
                    a.click()
                """
                with output:
                    display(Javascript(js_code))
            except Exception as e:
                logging.error('Error generating download JS: {e}')
                
        def click(widget):
            trigger_download(self.context.save_json(None), '{}.json'.format(key))

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
                    # allow saving the calculation
                    # need an output widget
                    download_output = widgets.Output()
                    # hide it
                    download_output.layout.display = 'none'
                    save_button = widgets.Button(description='Export', tooltip='Save the Calculation (as JSON)')
                    save_button.on_click(self.download_config(key, download_output))
                    # add it to the output
                    output.children = output.children + (save_button, download_output)

        return frames

    def generate_handler(self, field_name, widget_elements, label):
        def handleEvent(change):
            # update the json representation
            widget_elements[field_name]['value'] = change['new']

        return handleEvent


class Dashboard(object):
    '''
    Main class for the UI
    Needs to be able to load and save config files/trade files (in JSON)
    '''

    def __init__(self, config):
        # the main containers/widgets
        self.main = None

        # build the UI
        self.mainboard = MainPage(self.context, self.default_rundate)
        self.tabs = widgets.Tab(children=[self.mainboard.main_container])
        self.tabs.set_title(0, 'XVA')

        # draw a new one
        self.main = widgets.VBox()
        self.main.children = [self.tabs]

        display(self.main)

if __name__ == '__main__':
    config = ""
    d = Dashboard(config)
