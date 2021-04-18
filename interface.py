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
import logging
import numpy as np
import pandas as pd


def diag_ir(out, calc, factor, tenor_point, aa=None):
    ir = out['Results']['scenarios'][factor]
    size = ir.shape[0]
    perc = np.array([50, 0.50, 2.50, 5.00, 95.00, 97.50, 99.50])
    ten_ir = calc.all_factors[factor].factor.tenors
    all = [pd.DataFrame(np.percentile(i, 100.0 - perc, axis=1), index=perc, columns=ten_ir).T for i in ir]
    comp = pd.DataFrame([x.iloc[ten_ir.searchsorted(tenor_point)] for x in all],
                        index=calc.time_grid.time_grid_years[:size])
    if aa is not None:
        return comp.reindex(comp.index.union(aa['Time (Years)'])).interpolate('index').reindex(aa['Time (Years)'])
    else:
        return comp


def bootstrap(path, rundate, reuse_cal=True):
    from riskflow.adaptiv import AdaptivContext

    context = AdaptivContext()

    if reuse_cal and os.path.isfile(os.path.join(path, rundate, 'CVAMarketData.json')):
        context.parse_json(os.path.join(path, rundate, 'CVAMarketData.json'))
    else:
        context.parse_json(os.path.join(path, rundate, 'MarketData.json'))

    context.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)
    context.parse_calendar_file(os.path.join(path, 'calendars.cal'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    context.bootstrap()
    context.write_marketdata_json(os.path.join(path, rundate, 'MarketDataCal.json'))
    context.write_market_file(os.path.join(path, rundate, 'MarketDataCal.dat'))


if __name__ == '__main__':
    import matplotlib

    # matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt

    plt.interactive(True)
    # make pandas pretty print
    pd.options.display.float_format = '{:,.5f}'.format

    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # set the log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import riskflow as rf

    folder = 'CVA'
    path = rf.getpath(['E:\\Data\\crstal\\{}'.format(folder),
                       '/media/vretiel/Media/Data/crstal/{}'.format(folder),
                       'G:\\Credit Quants\\CRSTAL\\{}'.format(folder),
                       'G:\\{}'.format(folder)])

    rundate = '2021-02-05'

    bootstrap(path, rundate, reuse_cal=True)

    if 0:
        cx = rf.load_market_data(rundate, path, json_name='MarketData.json')
        cx_new = rf.load_market_data(rundate, path, json_name='CVAMarketData_Calibrated_New.json')

        for factor in [x for x in cx_new.params['Price Factors'].keys()
                       if x.startswith('HullWhite2FactorModelParameters')]:
            # override it
            cx.params['Price Factors'][factor] = cx_new.params['Price Factors'][factor]

        cx.parse_json(os.path.join(path, rundate, 'CrB_Kathu_Solar_Park_ISDA.json'))

        if 1:
            calc, out, res = rf.run_cmc(cx, overrides={'Calc_Scenarios': 'No',
                                                       'Dynamic_Scenario_Dates': 'No',
                                                       # 'Run_Date': '2020-03-09',
                                                       # 'Tenor_Offset': -3/365.0,
                                                       # 'Time_grid':'1m 5m 1362d',
                                                       'Batch_Size': 2048,
                                                       'Simulation_Batches': 10,
                                                       'CVA': {'Gradient': 'No', 'Hessian': 'No'}}, prec=np.float32)

        else:
            calc, out = rf.run_baseval(cx)
