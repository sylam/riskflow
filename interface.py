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
import torch
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


def bootstrap(path, rundate, device, reuse_cal=True):
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
        'Swaption_Premiums'] = '~/Downloads/IR_Volatility_Swaption_2021-03-17_0000_LON.csv'
    # for mp in list(context.params['Market Prices'].keys()):
    #     if mp.startswith('HullWhite2FactorInterestRateModelPrices') and not (
    #             mp.endswith('AUD-AONIA') or mp.endswith('ZAR-JIBAR-3M')):
    #         del context.params['Market Prices'][mp]

    context.parse_calendar_file(os.path.join(path, 'calendars.cal'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    context.bootstrap(device)
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


if __name__ == '__main__':
    import matplotlib

    # matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt

    # import seaborn as sns

    plt.interactive(True)
    # make pandas pretty print
    pd.options.display.float_format = '{:,.5f}'.format

    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # set the log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import riskflow as rf

    env = ''
    paths = {}
    for folder in ['CVA', 'Arena', 'Debug', 'Upgrade']:
        paths[folder] = rf.getpath(
            [os.path.join('E:\\Data\\crstal\\CVA', folder),
             os.path.join('G:\\Credit Quants\\CRSTAL', folder),
             os.path.join('/media/vretiel/Media/Data/crstal', folder),
             os.path.join('G:\\', folder)])

    path = paths['CVA']

    rundate = '2021-03-17'
    # rundate = '2021-03-24'

    gpudevice = torch.device("cuda:0")
    cpudevice = torch.device("cpu")

    # bootstrap(path, rundate, device=gpudevice, reuse_cal=True)

    if 1:
        # cx_prod = rf.load_market_data(rundate, path, json_name=os.path.join('', 'MarketData.json'))
        cx = rf.load_market_data(rundate, path, json_name=os.path.join(env, 'MarketData.json'))
        # cx.params['Price Factor Interpolation'] = rf.config.ModelParams()

        # cx = rf.load_market_data(rundate, path, json_name=os.path.join(env, 'MarketDataCVA.json'))
        # cx_arena = rf.load_market_data(rundate, path, json_name=os.path.join(env, 'MarketData.json'))
        # cx.params['Price Factors'].update(cx_arena.params['Price Factors'])
        # cx = rf.load_market_data(rundate, path, json_name=os.path.join(env, 'MarketData.json'))
        try:
            cx_new = rf.load_market_data(rundate, path, json_name='CVAMarketData_Calibrated_New.json')

            for factor in [x for x in cx_new.params['Price Factors'].keys() if x.startswith(
                    'HullWhite2FactorModelParameters') or x.startswith('GBMAssetPriceTSModelParameters')]:
                # override it
                cx.params['Price Factors'][factor] = cx_new.params['Price Factors'][factor]

            logging.info('Using Implied calibration for Interest Rates and FX')
        except:
            logging.info('Not using Implied calibration for Interest Rates and FX')

        # cx.parse_trade_file(os.path.join(path, rundate, 'ccirs.aap'))
        # cx.parse_json(os.path.join(path, rundate, env, 'CrB_NatWest_Markets_Plc_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_FirstRand_Bank_Ltd_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Shanta_Mining_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Ukhamba_Holdings_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_SAPPI_SA_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_AVI_Financial_Services__Pty__Limited_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_AutoX_ISDA.json'))

        # cx.parse_json(os.path.join(path, rundate, 'CrB_Eskom_Hld_SOC_Ltd_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_RCL_Foods_Treasury_NonISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Redefine_Properties_Limited_ISDA.json'))
        cx.parse_json(os.path.join(path, rundate, 'CrB_Kathu_Solar_Park_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Sanlam_Developing_Markets_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_The_Core_Computer_Business_Limited_ISDA.json'))

        # cx.parse_json(os.path.join(path, rundate, 'CrB_Land___Agricul_Bnk_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_ABSA_Bank_Jhb_ISDA.json'))

        # cx.parse_json(os.path.join(path, rundate, 'CrB_Growthpoint_Properties_Ltd_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_BNP_Paribas__Paris__ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Nedbank_Ltd_ISDA.json'))

        # cx.parse_json(os.path.join(path, rundate, 'CrB_NatWest_Markets_Plc_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_CS_Sec__Europe__Ltd_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Nomura_International_Plc_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_RCL_Foods_Treasury_NonISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Goldman_Sachs_Int_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Momentum_Metro_Life_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_Omnia_Group_ISDA.json'))
        # cx.parse_json(os.path.join(path, rundate, 'CrB_CS_Int_London_ISDA.json'))

        # cx.parse_json(os.path.join(path, rundate, 'CrB_Prescient_Toyota_SA_Prov_Fd_ISDA.json'))
        # cx.parse_trade_file(os.path.join(path, rundate, 'equityswap.aap'))

        cx.params['Price Factors']['InterestRate.USD-LIBOR-3M.FUNDING'] = rf.makeflatcurve(
            'USD', 65, daycount='ACT_365', tenor=30)
        # cx.params['Price Factors']['InterestRate.ZAR-SWAP.FUNDING'] = rf.makeflatcurve(
        #     'ZAR', 25, daycount='ACT_365', tenor=30)
        # cx.params['Price Factors']['InterestRate.ZAR-SWAP.OIS'] = rf.makeflatcurve(
        #     'ZAR', -40, daycount='ACT_365', tenor=30)
        #
        # for x in cx.deals['Deals']['Children'][0]['Children']:
        #     x['Ignore'] = 'False'
        #     # if 'fxnon' in x['instrument'].field['Object'].lower():
        #     if 'ZAR-USDCCSJI3M-LI3M201008-211008054' in x['instrument'].field['Reference']:
        #         x['Ignore'] = 'False'
        #

        # cx.deals['Deals']['Children'][0]['instrument'].field['Collateral_Assets'] = {
        #     'Cash_Collateral': [{
        #         'Currency': 'USD',
        #         'Collateral_Rate': 'USD-OIS',
        #         'Funding_Rate': 'USD-LIBOR-3M.FUNDING',
        #         'Haircut_Posted': 0.0,
        #         'Amount': 1.0}]}

        if 1:
            # grab the netting set
            ns = cx.deals['Deals']['Children'][0]['instrument']
            factor = rf.utils.Factor('InterestRate', ('ZAR-SWAP',))

            # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Funding_Rate'] = 'USD-LIBOR-3M.FUNDING'
            # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Collateral_Rate'] = 'USD-OIS'
            # ns.field['Collateral_Call_Frequency']=pd.DateOffset(weeks=1)
            # ns.field['Collateralized'] = 'False'

            overrides = {'Calc_Scenarios': 'Yes',
                         # 'Run_Date': '2020-08-21',
                         # 'Tenor_Offset': 2.0,
                         'Time_grid': '0d 2d 1w(1w) 3m(1m)',
                         'Random_Seed': 1254,
                         'Generate_Cashflows': 'No',
                         'Currency': 'ZAR',
                         'Deflation_Interest_Rate': 'ZAR-SWAP',
                         'Batch_Size': 1024,
                         'Simulation_Batches': 2,
                         'CollVA': {'Gradient': 'No'},
                         'CVA': {'Gradient': 'Yes', 'Hessian': 'No'}}

            if ns.field['Collateralized'] == 'True':
                overrides['Dynamic_Scenario_Dates'] = 'Yes'
            else:
                overrides['Dynamic_Scenario_Dates'] = 'No'

            calc, out, res = rf.run_cmc(cx, prec=torch.float32, device=gpudevice,
                                        overrides=overrides, CVA=True, CollVA=False, FVA=False)

            # factors_to_add = {}
            # for factor in calc.stoch_factors.keys():
            #     if factor.type == 'InterestRate':
            #         factor_name = rf.utils.check_tuple_name(factor)
            #         print(factor, cx.params['Price Factors'][factor_name]['Interpolation'])
            #         prod = rf.riskfactors.construct_factor(factor, cx_prod.params['Price Factors'])
            #         uat = rf.riskfactors.construct_factor(factor, cx.params['Price Factors'])
            #         cx.params['Price Factors'][factor_name]['Curve'] = rf.utils.Curve(
            #             [], list(zip(prod.tenors, uat.current_value(prod.tenors))))
            #
            # cx.params['Price Factor Interpolation'].modelfilters = {}
            # calc2, out2, res2 = rf.run_cmc(
            #     cx, prec=torch.float32, device=cpudevice, overrides=overrides, CVA=True, CollVA=False, FVA=False)
            # j = check_correlation(out['Results']['scenarios'],
            #                       ('InterestRate', ('ZAR-JIBAR-3M',)), ('FxRate', ('ZAR',)))
            # s = check_correlation(out['Results']['scenarios'],
            #                       ('InterestRate', ('ZAR-SWAP',)), ('FxRate', ('ZAR',)))

        else:
            calc, out = rf.run_baseval(cx, prec=torch.float64, device=gpudevice,
                                       overrides={'Currency': 'ZAR', 'Greeks': 'No'})
