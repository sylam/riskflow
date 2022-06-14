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


def check_correlation(scenarios, ir_factor, fx_factor):
    timepoints, _, sims = scenarios[ir_factor].shape
    f = [np.array([np.corrcoef(scenarios[ir_factor][:, 10 * j, i],
                               np.log(scenarios[fx_factor][:timepoints, i]))[0][1] for i in range(sims)])
         for j in range(20)]
    return np.array(f)


def bootstrap(path, rundate, reuse_cal=True):
    from riskflow.adaptiv import AdaptivContext

    context = AdaptivContext()
    cal_file = 'CVAMarketData_Calibrated_New.json'
    if reuse_cal and os.path.isfile(os.path.join(path, rundate, cal_file)):
        # context.parse_json(os.path.join(path, rundate, cal_file))
        context.parse_json(os.path.join(path, rundate, 'MarketData.json'))
        context_tmp = AdaptivContext()
        context_tmp.parse_json(os.path.join(path, rundate, cal_file))
        for factor in [x for x in context_tmp.params['Price Factors'].keys()
                       if x.startswith('HullWhite2FactorModelParameters') or
                          x.startswith('GBMAssetPriceTSModelParameters')]:
            # override it
            context.params['Price Factors'][factor] = context_tmp.params['Price Factors'][factor]
    else:
        context.parse_json(os.path.join(path, rundate, 'MarketData.json'))

    context.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)
    context.params['System Parameters'][
        'Swaption_Premiums'] = 'T:\\OTHER\\ICE\\Archive\\IR_Volatility_Swaption_2022-02-22_0000_LON.csv'
    # for mp in list(context.params['Market Prices'].keys()):
    #     if mp.startswith('HullWhite2FactorInterestRateModelPrices') and not (
    #             mp.endswith('AUD-AONIA') or mp.endswith('ZAR-JIBAR-3M')):
    #         del context.params['Market Prices'][mp]

    context.parse_calendar_file(os.path.join(path, 'calendars.cal'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    context.bootstrap()
    # context.write_marketdata_json(os.path.join(path, rundate, cal_file))
    # context.write_market_file(os.path.join(path, rundate, 'MarketDataCal.dat'))


def plot_matrix(df, title='Full Gamma'):
    f = plt.figure(figsize=(15, 15))
    plt.matshow(df, fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns.get_level_values(1), fontsize=3, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns.get_level_values(1), fontsize=3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=5)
    plt.title(title, fontsize=10)


def checkdate(outputs, inputs, date):
    import glob
    loaded = [os.path.split(x)[1].replace('.json', '') for x in glob.glob(os.path.join(inputs, date, 'CrB*.json'))]
    calced = [os.path.split(x)[1].replace('.csv', '')[15:] for x in glob.glob(
        os.path.join(outputs, 'CVA_{}*.csv'.format(date)))]
    return set(loaded).difference(calced)


def get_missing_rates(aa, model_factor, from_date, to_date, MDR=500):
    '''
    Simple function to determine what can be calibrated and what's missing in the .ada file.
    Needs a crstal context and a list of all potential price factors to calibrate between the
    provided from_date and to_date.

    MDR is an acronym for Minumum Data Required (default is 500)
    '''
    from riskflow.utils import filter_data_frame

    to_calibrate = {}
    missing = {}
    data_window = filter_data_frame(aa.archive, from_date, to_date)
    for factor_name, factor_model in sorted(model_factor.items()):
        if factor_model.archive_name not in aa.archive_columns:
            missing[factor_name] = -1
        else:
            data = data_window[aa.archive_columns[factor_model.archive_name]]
            valid = (data[data.columns[1:]].count() > MDR).all() if data.shape[1] > 1 else (data.count() > MDR).all()
            if valid:
                to_calibrate[factor_name] = factor_model
            else:
                missing[factor_name] = data.count().values[0]
    return missing, to_calibrate


def calibrate_PFE(arena_path, rundate, scratch="Z:\\"):
    from riskflow.utils import calc_statistics, excel_offset, filter_data_frame
    from riskflow.adaptiv import AdaptivContext

    aa = AdaptivContext()
    aa.parse_json(os.path.join(arena_path, rundate, 'MarketData.json'))
    # aa.ParseCalibrationfile(os.path.join(scratch, 'calibration_prd_linux.config'))
    # aa.ParseCalibrationfile(os.path.join(scratch, 'calibration_uat_linux.config'))
    aa.parse_calibration_file(os.path.join(scratch, 'calibration.config'))

    end_date = excel_offset + pd.offsets.Day(aa.archive.index.max())
    # now work out 3 years prior for the start date
    start_date = end_date - pd.DateOffset(years=3)
    # the zar-swap curve needs to include a downturn - The earliest data we have is from 2007-01-01
    swap_start_date = pd.Timestamp('2007-01-01')

    # print out the dates used
    print('Calibrating from', start_date, 'to', end_date)

    # now check if all rates are present
    all_factors = aa.fetch_all_calibration_factors(override={})
    model_factor = all_factors['present']
    model_factor.update(all_factors['absent'])

    # report which rates are missing or have less than MDR data points and which have enough to calibrate
    missing, to_calibrate = get_missing_rates(aa, model_factor, start_date, end_date, 600)
    # EquityPrice.ZAR_SHAR
    # EquityPrice.ZAR_TFGP

    for i in ['EquityPrice.ZAR_SHAR', 'EquityPrice.ZAR_TFGP']:
        del to_calibrate[i]

    smoothing_std = 2.15
    # perform an actual calibration - first, clear out the old parameters
    aa.params['Price Models'] = {}
    # if there are any issues with the data, it will be logged here
    aa.calibrate_factors(
        start_date, end_date, to_calibrate, smooth=smoothing_std, correlation_cuttoff=0.1, overwrite_correlations=True)


if __name__ == '__main__':
    import glob
    import matplotlib.pyplot as plt

    plt.interactive(True)
    # make pandas pretty print
    pd.options.display.float_format = '{:,.5f}'.format
    pd.set_option("display.max_rows", 500, "display.max_columns", 20)

    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # set the log level
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    import riskflow as rf

    env = ''
    paths = {}
    for folder in ['Input_JSON', 'CVA_JSON', 'CVA_UAT', 'CVA', 'PFE', 'PFE_UAT', 'Upgrade']:
        paths[folder] = rf.getpath(
            [os.path.join('E:\\Data\\crstal\\CVA', folder),
             # os.path.join('/media/vretiel/Media/Data/crstal', folder),
             os.path.join('U:\\', folder),
             os.path.join('S:\\CCR_PFE_EE_NetCollateral', folder),
             os.path.join('S:\\CCR_Tagged', folder),
             os.path.join('N:\\Archive', folder),
             os.path.join('G:\\', folder)])

    path_json = paths['CVA_JSON']
    # path_json = paths['Input_JSON']
    # path = paths['CVA_UAT']
    # path = paths['CVA']
    path = paths['PFE']

    # rundate = '2021-11-12'
    rundate = '2022-06-13'
    # rundate = '2021-09-14'
    # calibrate_PFE(path, rundate)
    # bootstrap(path, rundate, reuse_cal=True)

    # empty context
    cx = rf.Context(
        path_transform={
            '\\\\ICMJHBMVDROPPRD\\AdaptiveAnalytics\\Inbound\\MarketData':
                '\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData'},
        file_transform={
            'CVAMarketData_Calibrated.dat': 'CVAMarketData_Calibrated_New.json',
            'MarketData.dat': 'MarketData.json'
        })

    spreads = {
        'USD': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'EUR': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'ZAR': {'FVA@Equity': {'collateral': -10, 'funding': 15}, 'FVA@Income': {'collateral': -10, 'funding': 15}}}

    curves = {'USD': {'collateral': 'USD-OIS', 'funding': 'USD-LIBOR-3M'},
              'EUR': {'collateral': 'EUR-EONIA', 'funding': 'EUR-EURIBOR-3M'},
              'ZAR': {'collateral': 'ZAR-SWAP', 'funding': 'ZAR-SWAP'}}

    curves_rfr = {
        'USD': {'collateral': 'USD-SOFR', 'funding': 'USD-LIBOR-3M'},
        'EUR': {'collateral': 'EUR-ESTR', 'funding': 'EUR-EURIBOR-3M'}}

    for json in glob.glob(os.path.join(path_json, rundate, 'Input*.json')):
        try:
            cx.load_json(json)
        except:
            print('error in json', json)
            continue

        if not cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            print('no children for crb {} - skipping'.format(json))
            continue

        # grab the netting set
        ns = cx.current_cfg.deals['Deals']['Children'][0]['Instrument']
        agreement_currency = ns.field.get('Agreement_Currency', 'ZAR')
        factor = rf.utils.Factor('InterestRate', ('ZAR-SWAP',))

        # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Funding_Rate'] = 'USD-LIBOR-3M.FUNDING'
        # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Collateral_Rate'] = 'USD-OIS'
        # ns.field['Collateral_Call_Frequency']=pd.DateOffset(weeks=1)
        # ns.field['Collateralized'] = 'False'

        overrides = {
            'Calc_Scenarios': 'No',
            # 'Run_Date': '2021-10-11',
            # 'Tenor_Offset': 2.0,
            # 'Time_grid': '0d 2d 1w(1w) 3m(1m)',
            'Random_Seed': 1254,
            'Generate_Cashflows': 'No',
            # 'Currency': 'USD',
            # 'Deflation_Interest_Rate': 'ZAR-SWAP',
            'Batch_Size': 2500,
            'Simulation_Batches': 2,
            'CollVA': {'Gradient': 'Yes'},
            'CVA': {'Gradient': 'No', 'Hessian': 'No'}
        }

        if ns.field['Collateralized'] == 'True':
            overrides['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            overrides['Dynamic_Scenario_Dates'] = 'No'

        # calc, out, res = cx.Credit_Monte_Carlo(overrides=overrides, CVA=False, CollVA=False, FVA=False)
        # calc, out = cx.Base_Valuation(overrides={'Currency': 'ZAR'})
        output = json.replace('json', 'csv')
        filename = os.path.split(output)[1]
        outfile = os.path.join('C:\\temp', filename)

        calc, out = cx.Base_Valuation()
        out['Results']['mtm'].to_csv(outfile)

        mike_filename = 'N:\\Archive\\PFE\\{}\\{}'.format(rundate, filename.replace('InputAAJ_', ''))

        if os.path.exists(mike_filename):
            mike = pd.read_csv(mike_filename)
            me = pd.read_csv(outfile)
            adaptiv = float(mike.head(1)['PFE'])
            myval = float(me.head(1)['Value'])

            if adaptiv != 0 and np.abs((adaptiv-myval)/adaptiv).max() > 0.01:
                print('Mismatch - Collateralized is {}, '.format(ns.field['Collateralized']), outfile, adaptiv, myval)
                if ns.field['Collateralized'] != 'True':
                    print('investigate', json)
        else:
            print('skipping', mike_filename)