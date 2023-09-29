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
    # cal_file = 'CVAMarketData_Calibrated_New.json'
    # cal_file = 'CVAMarketData_test.json'
    cal_file = 'CVAMarketData_TST.json'
    if reuse_cal and os.path.isfile(os.path.join(path, rundate, cal_file)):
        # context.parse_json(os.path.join(path, rundate, cal_file))
        context.parse_json(os.path.join(path, rundate, 'CVAMarketData_TST.json'))
        context_tmp = AdaptivContext()
        context_tmp.parse_json(os.path.join(path, rundate, cal_file))
        for factor in [x for x in context_tmp.params['Price Factors'].keys()
                       if x.startswith('HullWhite2FactorModelParameters') or
                          x.startswith('GBMAssetPriceTSModelParameters')]:
            # override it
            context.params['Price Factors'][factor] = context_tmp.params['Price Factors'][factor]
    else:
        # context.parse_json(os.path.join(path, rundate, cal_file))
        context.parse_json(os.path.join(path, '', cal_file))

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


def compress(data):
    import joblib
    # t=data['Calc']['Deals']['Deals']['Children'][0]['Children'][1121]['Instrument'].field
    # print (data['Children'][0]['Children'][1121]['Instrument'].field)
    # deals = data['Calc']['Deals']['Deals']['Children'][0]['Children']
    deals = joblib.load('C:\\temp\\gs.obj')
    equity_swaps = [x for x in deals if x['Instrument'].field['Object'] == 'EquitySwapletListDeal']
    if equity_swaps:
        eq_swap_ref = {x['Instrument'].field['Reference']: x['Instrument'].field['Equity'] for x in equity_swaps}
        all_other = [y for y in deals if y['Instrument'].field['Reference'] not in eq_swap_ref.keys()]
        all_eq_swap = [y for y in deals if y['Instrument'].field['Reference'] in eq_swap_ref.keys()]
        eq_unders = {}
        ir_unders = {}
        # first load all compressable deals
        for k in all_eq_swap:
            key = tuple([k['Instrument'].field[x] for x in k['Instrument'].field.keys()
                         if x not in ['Tags', 'Reference', 'Cashflows']])
            key_tag = key + tuple(k['Instrument'].field['Tags'])
            if k['Instrument'].field['Object'] == 'EquitySwapletListDeal':
                eq_unders.setdefault(key_tag, []).append(k)
            else:
                under_eq = eq_swap_ref[k['Instrument'].field['Reference']]
                ir_unders.setdefault(key_tag + (under_eq,), []).append(k)

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
            deal['Instrument'].field['Reference'] = 'COMPRESSED_{}_{}'.format(
                deal['Instrument'].field['Buy_Sell'], deal['Instrument'].field['Equity'])
            eq_compressed.setdefault(deal['Instrument'].field['Equity'], []).append(deal)

        ir_compressed = {}
        for k, unders in ir_unders.items():
            margin_list = {}
            notional_list = {}
            for deal in unders:
                for cf in deal['Instrument'].field['Cashflows']['Items']:
                    cf_key = tuple([(k, v) for k, v in cf.items() if k not in ['Notional', 'Resets', 'Margin']])
                    reset_key = tuple([tuple(x) for x in cf['Resets']])
                    key = (cf_key, reset_key)
                    margin_list[key] = margin_list.setdefault(key, 0.0) + cf['Margin'].amount * cf['Notional']
                    notional_list[key] = notional_list.setdefault(key, 0.0) + cf['Notional']

            # finish this off
            final = []
            for key, val in margin_list.items():
                notional = notional_list[key]
                cashflow = dict(key[0])
                cashflow['Notional'] = notional
                cashflow['Resets'] = [list(x) for x in list(key[1])]
                cashflow['Margin'] = rf.utils.Basis(10000.0 * val / notional)
                final.append(cashflow)

            # edit the last deal
            deal['Instrument'].field['Cashflows']['Items'] = final
            deal['Instrument'].field['Reference'] = 'COMPRESSED_{}_{}'.format(
                deal['Instrument'].field['Buy_Sell'], k[-1])
            ir_compressed.setdefault(k[-1], []).append(deal)

        for k, v in eq_compressed.items():
            all_other.extend(v)
            all_other.extend(ir_compressed[k])

        return all_other
    else:
        return deals


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
    for folder in ['JSON', 'COLLVA', 'CVA_SARB', 'Input_JSON', 'CVA_JSON',
                   'Autocall_PreTrade', 'FVA_JSON', 'CVA', 'PFE', 'PFE_UAT', 'Upgrade']:
        paths[folder] = rf.getpath(
            [os.path.join('Y:\\CollVA', folder),
             os.path.join('/media/vretiel/Media/Data/crstal', folder),
             # os.path.join('Z:\\CVA_JSON\\TEST', folder),
             os.path.join('R:\\Riskflow', folder),
             # os.path.join('S:\\CCR_PFE_EE_NetCollateral', folder),
             os.path.join('S:\\Riskflow\\TEST', folder),
             os.path.join('N:\\Archive', folder)])

    # path_json = paths['FVA_JSON']
    path_json = paths['CVA']
    # path = paths['CVA_UAT']
    # path = paths['CVA']
    path = paths['PFE']

    rundate = '2023-09-15'
    # rundate = '2023-05-31'
    # calibrate_PFE(path, rundate)
    # bootstrap(path_json, '', reuse_cal=True)
    # bootstrap('Z:\\', rundate, reuse_cal=False)

    # empty context
    cx = rf.StressedContext(
        path_transform={
            '\\\\ICMJHBMVDROPPRD\\AdaptiveAnalytics\\Inbound\\MarketData':
                '\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData',
            #'\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData\\CVAMarketDataBackup':
            #    '\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData',
            '\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData':
                '\\\\icmjhbmvdropuat\\AdaptiveAnalytics\\Inbound\\MarketData'
            # '\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData':
            #    os.path.join('\\\\ICMJHBMVDROPUAT\\AdaptiveAnalytics\\Inbound\\MarketData\\CVA_JSON', rundate),
        },
        file_transform={
            'CVAMarketData_Calibrated.dat': 'CVAMarketData_TST_New.json',
            'CVAMarketData_Calibrated_New.json': 'CVAMarketData_TST_New.json',
            # 'CVAMarketData_Calibrated_New.json': 'MarketData.json',
            'MarketData.dat': 'MarketData.json'
        })

    # md = rf.load_market_data(rundate, path, json_name=os.path.join(env, 'MarketData.json'))

    spreads = {
        'USD': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'EUR': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'GBP': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'ZAR': {'FVA@Equity': {'collateral': -10, 'funding': 15}, 'FVA@Income': {'collateral': -10, 'funding': 15}}}

    curves = {'USD': {'collateral': 'USD-OIS', 'funding': 'USD-LIBOR-3M'},
              'EUR': {'collateral': 'EUR-EONIA', 'funding': 'EUR-EURIBOR-3M'},
              'GBP': {'collateral': 'GBP-SONIA', 'funding': 'GBP-SONIA'},
              'ZAR': {'collateral': 'ZAR-SWAP', 'funding': 'ZAR-SWAP'}}

    curves_rfr = {
        'USD': {'collateral': 'USD-SOFR', 'funding': 'USD-LIBOR-3M'},
        'EUR': {'collateral': 'EUR-ESTR', 'funding': 'EUR-EURIBOR-3M'}}

    # for json in glob.glob(os.path.join(path_json, rundate, 'Combination*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Soc_Gen_Paris_*.json')):
    for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_ACWA_Power_SolarReserve_Redstone_So_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_CS_Int_London_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Goldman_Sachs_Int_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_JPMorgan_Chase_NYK_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_M_Stanley___Co_Int_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Investec_Life_Limited_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_M_Lynch_Int_Ldn_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_ACWA_Power_SolarReserve_Redstone_So_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Deutsche_Bank_AG_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_AutoX__Pty__Ltd_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputJSON_ZAR_CrB_Tiger_Consumer_Brands_Limited_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Redefine_Properties_Limited*.json')):

        # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Standard_Bank_*.json')):
        # for json in glob.glob(os.path.join(path_json, rundate, '*otus*.json')):
        cx.load_json(json, compress=True)
        try:
            if 'GBMAssetPriceModel.EUR_SX5E' not in cx.current_cfg.params['Price Models']:
                cx.current_cfg.params['Price Models']['GBMAssetPriceModel.EUR_SX5E'] = cx.current_cfg.params[
                    'Price Models']['GBMAssetPriceModel.EUR_SX5EEX']
            if 'EquityPriceVol.USD_SPX.ZAR' not in cx.current_cfg.params['Price Factors']:
                cx.current_cfg.params['Price Factors']['EquityPriceVol.USD_SPX.ZAR'] = cx.current_cfg.params[
                    'Price Factors']['EquityPriceVol.USD_SPX.USD']
        except:
            pass

        # if 'HullWhite2FactorModelParameters.USD-OIS' not in cx.current_cfg.params['Price Factors']:
        #     cx.current_cfg.params['Price Factors']['HullWhite2FactorModelParameters.USD-OIS'] = cx.current_cfg.params[
        #         'Price Factors']['HullWhite2FactorModelParameters.USD-SOFR']

        if not cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            print('no children for crb {} - skipping'.format(json))
            continue

        # Update any missing models
        try:
            cx.current_cfg.params['Price Factors'].update(md.params['Price Factors'])
        except:
            pass

        # grab the netting set
        ns = cx.current_cfg.deals['Deals']['Children'][0]['Instrument']
        agreement_currency = ns.field.get('Agreement_Currency', 'ZAR')
        factor = rf.utils.Factor('InterestRate', ('ZAR-SWAP',))

        # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Funding_Rate'] = 'USD-LIBOR-3M.FUNDING'
        # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Collateral_Rate'] = 'USD-OIS'
        # ns.field['Collateral_Assets']['Cash_Collateral'] = []
        # ns.field['Collateral_Call_Frequency']=pd.DateOffset(weeks=1)
        # ns.field['Collateralized'] = 'True'

        overrides = {
            'Calc_Scenarios': 'No',
            # 'Run_Date': '2021-10-11',
            # 'Tenor_Offset': 2.0,
            # 'Time_grid': '0d 2d 1w(1w) 3m(1m)',
            'Random_Seed': '1',
            'Generate_Cashflows': 'Yes',
            'Currency': 'ZAR',
            'Antithetic': 'Yes',
            # 'Deflation_Interest_Rate': 'ZAR-SWAP',
            'Batch_Size': 1024,
            'Simulation_Batches': 2,
            # 'COLLVA': {'Gradient': 'Yes'},
            'CVA': {'Gradient': 'Yes', 'Hessian': 'No'}
        }

        if ns.field['Collateralized'] == 'True':
            overrides['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            overrides['Dynamic_Scenario_Dates'] = 'No'

        for i in cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            # if i['Instrument'].field['Reference'] != 'USDIBLIX12DIGIPUT18189230%20280629':
            # if i['Instrument'].field['Reference'] != '137335521':
            if False and not i['Instrument'].field['Object'].startswith('QEDI'):
            # if not i['Instrument'].field['Reference'].startswith('ZARSPI'):
                i['Ignore'] = 'True'
            else:
                i['Ignore'] = 'False'

        # calculation_currency = 'USD'

        calc, out = cx.Base_Valuation(overrides={'Currency': 'ZAR'})
        for i in range(3):
            # cx.stress_config(['InterestRate', 'InflationRate'])
            calc, out = cx.Credit_Monte_Carlo(overrides=overrides)
            print(i, 'stress', out['Results']['cva'])
            cx.restore_config()
            calc, out = cx.Credit_Monte_Carlo(overrides=overrides)
            print(i, 'restore', out['Results']['cva'])


        # cx.stress_config(['ForwardPrice'])

        calc, out1 = cx.Credit_Monte_Carlo(overrides=overrides)
        cx.restore_config()
        cx.stress_config(['InterestRate', 'InflationRate'])
        calc, out2 = cx.Credit_Monte_Carlo(overrides=overrides)
        cx.restore_config()
        calc, out = cx.Credit_Monte_Carlo(overrides=overrides)
        calc, params = rf.run_cmc(
            cx.current_cfg, overrides=overrides, CVA=False, FVA=False, CollVA=False, LegacyFVA=True)
        del params['CVA']
        partial_FVA = calc.calc_individual_FVA(
            params, spreads=spreads[calculation_currency], discount_curves=curves[calculation_currency])

        output = json.replace('json', 'csv')
        filename = os.path.split(output)[1]
        outfile = os.path.join('C:\\temp', filename)

        try:
            assets = list(ns.field['Collateral_Assets'].keys()).pop()
            if assets is None and ns['Collateralized'] == 'True':
                assets = 'Cash_Collateral'
        except:
            assets = 'Cash_Collateral' if ns.field['Collateralized'] == 'True' else 'None'

        collva_sect = cx.current_cfg.deals['Calculation'].get(
            'Collateral_Valuation_Adjustment', {'Calculate': 'Yes' if assets == 'Cash_Collateral' else 'No'})

        # sensible defaults
        collva_sect['Collateral_Curve'] = collva_sect.get(
            'Collateral_Curve', curves[agreement_currency]['collateral'])
        collva_sect['Funding_Curve'] = collva_sect.get(
            'Funding_Curve', curves[agreement_currency]['funding'])
        collva_sect['Collateral_Spread'] = collva_sect.get(
            'Collateral_Spread', spreads[agreement_currency]['FVA@Income']['collateral'])
        collva_sect['Funding_Spread'] = collva_sect.get(
            'Funding_Spread', spreads[agreement_currency]['FVA@Income']['funding'])
        cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment'] = collva_sect

        if 'Funding_Valuation_Adjustment' in cx.current_cfg.deals['Calculation']:
            del cx.current_cfg.deals['Calculation']['Funding_Valuation_Adjustment']

        calc, out = cx.Base_Valuation()
        # calc, out = cx.run_job(overrides)
        # out['Results']['collateral_profile'].to_csv(outfile)
        # calc, out = cx.Base_Valuation()
        # out['Results']['mtm'].to_csv(outfile)

        mike_filename = 'N:\\Archive\\PFE\\{}\\{}'.format(rundate, filename.replace('InputAAJ_', ''))

        if False and os.path.exists(mike_filename):
            mike = pd.read_csv(mike_filename, index_col=0)
            me = pd.read_csv(outfile, index_col=0)
            adaptiv = float(mike.head(1)['PFE'])
            if 0:
                myval = float(me.head(1)['Value'])

                if adaptiv != 0 and np.abs((adaptiv - myval) / adaptiv).max() > 0.01:
                    print('Mismatch - Collateralized is {}, '.format(ns.field['Collateralized']), outfile, adaptiv,
                          myval)
                    if ns.field['Collateralized'] != 'True':
                        print('investigate', json)
            else:
                common_index = mike.index.intersection(me.index)
                adaptiv_pfe = mike.reindex(common_index)
                my_pfe = me.reindex(common_index)
                abs_error = (my_pfe - adaptiv_pfe).abs()
                error = ((my_pfe - adaptiv_pfe) / adaptiv_pfe)['PFE'].dropna().abs().max()
                if error > 0.02:
                    print('{0:06.2f} Error Mismatch - Crb {1} Collateralized is {2}'.format(
                        error, json, ns.field['Collateralized']))
                else:
                    print('Crb {} is within 2%'.format(json))
        else:
            print('skipping', mike_filename)
