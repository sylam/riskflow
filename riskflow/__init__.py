########################################################################
# Copyright (C) Shuaib Osman (vretiel@gmail.com)
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

__author__ = "Shuaib Osman"
__license__ = "Free for non-commercial use"
__all__ = ['version_info', '__version__', '__author__', '__license__', 'Context', 'makeflatcurve', 'getpath',
           'set_collateral', 'load_market_data', 'run_baseval', 'run_cmc', 'update_dict']

import os
import json
import torch
import pathlib
import logging
import numpy as np
import pandas as pd
import collections.abc
import torch.multiprocessing as mp

from ._version import version_info, __version__
from . import fields
from . import utils
from .adaptiv import AdaptivContext
from .config import CustomJsonEncoder


def update_dict(d, u):
    for k, v in u.items():
        d[k] = update_dict(d.get(k, {}), v) if isinstance(v, collections.abc.Mapping) else v
    return d


def makeflatcurve(curr, bps, daycount='ACT_365', tenor=30):
    """
    generates a constant (flat) curve in basis points with the given daycount and tenor
    :return: a dictionary containing the curve definition
    """
    return {'Currency': curr, 'Curve': utils.Curve([], [[0, bps * 0.01 * 0.01], [tenor, bps * 0.01 * 0.01]]),
            'Day_Count': daycount, 'Property_Aliases': None, 'Sub_Type': 'None'}


def getpath(pathlist):
    """
    returns the first valid path in pathlist
    """
    for path in pathlist:
        if os.path.isdir(path):
            return path


def summarize_data(data, percentiles):
    pos_exposure = data.clip(0.0, np.inf)
    neg_exposure = data.clip(-np.inf, 0.0)

    exposure = {
        'EE': np.mean(pos_exposure, axis=1),
        'ENE': np.mean(neg_exposure, axis=1)}

    if percentiles:
        extra = {'PFE_{}'.format(x): np.percentile(data, float(x), axis=1) for x in percentiles.split(',')}
        exposure.update(extra)

    return pd.DataFrame(exposure)

def set_collateral(cfg, Agreement_Currency, Balance_Currency, Opening_Balance, Received_Threshold=0.0,
                   Posted_Threshold=0.0, Minimum_Received=100000.0, Minimum_Posted=100000.0, Liquidation_Period=10.0):
    """
    Loads CSA details on the root netting set in the given context
    """
    cfg.deals['Deals']['Children'][0]['Instrument'].field.update(
        {'Agreement_Currency': Agreement_Currency, 'Opening_Balance': Opening_Balance,
         'Apply_Closeout_When_Uncollateralized': 'No', 'Collateralized': 'True', 'Settlement_Period': 0.0,
         'Balance_Currency': Balance_Currency, 'Liquidation_Period': Liquidation_Period,
         'Credit_Support_Amounts':
             {'Received_Threshold': utils.CreditSupportList({1: Received_Threshold}),
              'Minimum_Received': utils.CreditSupportList({1: Minimum_Received}),
              'Posted_Threshold': utils.CreditSupportList({1: Posted_Threshold}),
              'Minimum_Posted': utils.CreditSupportList({1: Minimum_Posted})
              }
         }
    )


def load_market_data(rundate, path, json_name='MarketData.json', calendar_name='calendars.cal'):
    """
    Loads a json marketdata file and corresponding calendar (assumed to be named 'calendars.cal')
    :param rundate: folder inside path where the marketdata file resides
    :param path: root folder for the marketdata, calendar and trades
    :param json_name: name of the marketdata json file (default MarketData.json)
    :return: a context object with the data and calendars loaded
    """

    config = AdaptivContext()
    config.parse_json(os.path.join(path, rundate, json_name))
    if os.path.isfile(os.path.join(path, 'calendars.cal')):
        config.parse_calendar_file(os.path.join(path, calendar_name))
    else:
        logging.warning('Calendar file {} not loaded'.format(calendar_name))

    config.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)

    return config


def run_baseval(context, prec=torch.float64, overrides=None):
    """
    Runs a base valuation calculation on the provided context
    :param prec:
    :param context: a Context object
    :param overrides: a dictionary of overrides to replace the context's  calculation parameters
    :return: a tuple containing the calculation object and the output dictionary
    """
    from .calculation import construct_calculation
    calc_params = context.deals.get('Calculation',
                                    {'Base_Date': context.params['System Parameters']['Base_Date'],
                                     'Currency': 'ZAR'})

    # check if the gpu is available
    if torch.cuda.is_available():
        # make sure we try to be deterministic as possible
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    rundate = calc_params['Base_Date'].strftime('%Y-%m-%d')
    params_bv = {'Run_Date': rundate, 'MCMC_Simulations': 4096 * 8, 'Greeks': 'No'}
    params_bv.update(calc_params)

    if overrides is not None:
        update_dict(params_bv, overrides)

    calc = construct_calculation('Base_Revaluation', context, device=device, prec=prec)
    out = calc.execute(params_bv)
    return calc, out


def run_cmc(context, prec=torch.float32, overrides=None, LegacyFVA=False, job_id=0, num_jobs=1, res_queue=None):
    """
    Runs a credit monte carlo calculation on the provided context
    :param res_queue: If not None, returns the results in this queue
    :param num_jobs: number of jobs spawned - usually just 1 (i.e. the parent)
    :param job_id: used if multiprocessing is set (index of this process in a group of workers)
    :param LegacyFVA: True if all we want is a way to calculate an FVA per deal
    :param context: a Context object
    :param overrides: a dictionary of overrides to replace the context's  calculation parameters
    :param prec: the numerical precision to use (default float32)
    :return: a tuple containing the calculation object, output dictionary and exposure profile
    """
    from .calculation import construct_calculation
    calc_params = context.deals.get('Calculation',
                                    {'Base_Date': context.params['System Parameters']['Base_Date'],
                                     'Base_Time_Grid': '0d 2d 1w(1w) 1m(1m) 3m(3m)'})

    # check if the gpu is available
    if torch.cuda.is_available():
        # make sure we try to be deterministic as possible
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        device = torch.device("cuda", job_id)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    rundate = calc_params['Base_Date'].strftime('%Y-%m-%d')
    time_grid = str(calc_params.get('Base_Time_Grid', '0d 2d 1w(1w) 1m(1m) 3m(3m)'))

    params_mc = {'Time_grid': time_grid, 'Run_Date': rundate,
                 'Tenor_Offset': 0.0, 'Batch_Size': 1024, 'Simulation_Batches': 1}

    params_mc.update(calc_params)

    if overrides is not None:
        update_dict(params_mc, overrides)

    if params_mc.get('Credit_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
        cva_sect = params_mc['Credit_Valuation_Adjustment']
        if cva_sect.get('CDS_Tenors'):
            # add extra tenors to the survival probability curve and interpolate it to calculate CDS rates
            survivalprob = context.params['Price Factors']['SurvivalProb.{}'.format(cva_sect['Counterparty'])]
            daycount = lambda time_in_days: utils.get_day_count_accrual(
                params_mc['Base_Date'], time_in_days, utils.DAYCOUNT_ACT365)
            to_add = [daycount((x - params_mc['Base_Date']).days) for x in
                      utils.cds_dates(params_mc['Base_Date'], max(cva_sect.get('CDS_Tenors')) * 12)]
            new_terms = np.union1d(to_add, survivalprob['Curve'].array[:, 0])
            survivalprob['Curve'].array = np.array(
                list(zip(new_terms, np.interp(new_terms, *survivalprob['Curve'].array.T))))

    if 'LegacyFVA' in params_mc:
        legacy_fva_sect = params_mc['LegacyFVA']
        # get the agreement currency
        calc_currency = calc_params['Currency']
        collateral_curve = legacy_fva_sect['Collateral_Curve']
        funding_curve = legacy_fva_sect['Funding_Curve']

        collateral_curve = '{}.COLLATERAL'.format(collateral_curve)
        context.params['Price Factors']['InterestRate.{}'.format(collateral_curve)] = makeflatcurve(
            calc_currency, legacy_fva_sect['Collateral_Spread'])

        funding_curve = '{}.FUNDING'.format(funding_curve)
        context.params['Price Factors']['InterestRate.{}'.format(funding_curve)] = makeflatcurve(
            calc_currency, legacy_fva_sect['Funding_Spread'])

    if params_mc.get('Collateral_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
        # setup collva calc
        ns = context.deals['Deals']['Children'][0]['Instrument'].field
        collva_sect = context.deals['Calculation']['Collateral_Valuation_Adjustment']
        # get the agreement currency        
        calc_currency = ns.get('Balance_Currency', 'ZAR')

        if 'Cash_Collateral' not in ns.get('Collateral_Assets', {}):
            # get the funding and collateral rates
            collateral_curve = collva_sect['Collateral_Curve']
            funding_curve = collva_sect['Funding_Curve']

            if collva_sect['Collateral_Spread']:
                collateral_curve = '{}.COLLATERAL'.format(collateral_curve)
                context.params['Price Factors']['InterestRate.{}'.format(collateral_curve)] = makeflatcurve(
                    calc_currency, collva_sect['Collateral_Spread'])

            if collva_sect['Funding_Spread']:
                funding_curve = '{}.FUNDING'.format(funding_curve)
                context.params['Price Factors']['InterestRate.{}'.format(funding_curve)] = makeflatcurve(
                    calc_currency, collva_sect['Funding_Spread'])

            ns['Collateral_Assets'] = {
                'Cash_Collateral': [{
                    'Currency': calc_currency,
                    'Collateral_Rate': collateral_curve,
                    'Funding_Rate': funding_curve,
                    'Haircut_Posted': 0.0,
                    'Amount': 1.0}]}

        elif len(ns['Collateral_Assets']['Cash_Collateral']) > 1:
            # make sure we just take the first definition
            ns['Collateral_Assets']['Cash_Collateral'] = [ns['Collateral_Assets']['Cash_Collateral'][0]]

    calc = construct_calculation('Credit_Monte_Carlo', context, device=device, prec=prec)
    if LegacyFVA:
        return calc, params_mc
    else:
        out = calc.execute(params_mc, job_id, num_jobs)

        if res_queue is not None:
            # parent process must summarize data
            res_queue.put({'Results': out['Results'], 'Stats': out['Stats']})
        else:
            # summarize the results for easy review
            out['Results']['exposure_profile'] = summarize_data(
                out['Results']['mtm'], params_mc.get('Percentile', '95').replace(' ', ''))

            return calc, out


class Context:
    def __init__(self, path_transform={}, file_transform={}):
        # needed if the json file contains paths to window's files but linux is needed
        self.path_map = path_transform
        # needed if the name of the file referenced needs to be changed (e.g. from .dat to .json)
        self.file_map = file_transform
        self.config_cache = {}
        self.holiday_cfg_cache = {}
        self.current_cfg = AdaptivContext()
        # there may be a stressed config file attached to the context
        self.stressed_config_file = None

    def load_config(self, path_name):
        new_cfg = AdaptivContext()
        new_cfg.parse_json(path_name)

        # check we need to set the base_date
        if new_cfg.params['System Parameters'].get('Base_Date') is None:
            # set it to now
            new_cfg.params['System Parameters']['Base_Date'] = pd.Timestamp.now()

        # set its version
        new_cfg.version = ['JSONVersion', '22.05.30']
        return new_cfg

    def parse_path(self, file_path):
        # file_path is assumed to be a window's path - so we need to check if we're in a posix world
        if os.name == 'posix':
            file_path = pathlib.PureWindowsPath(file_path).as_posix()

        path, filename = os.path.split(file_path)
        return os.path.join(self.path_map.get(path, path), self.file_map.get(filename, filename))

    def load_json(self, jobfilename, compress=True):
        cfg = self.current_cfg
        # read the raw json data
        data = self.current_cfg.read_json(jobfilename)

        if 'MergeMarketData' in data['Calc']:
            market_data = data['Calc']['MergeMarketData']
            # check if there's a stressed marketdata defined (record but don't load it)
            self.stressed_config_file = market_data.get('StressedMarketDataFile')

            if market_data['MarketDataFile'] not in self.config_cache:
                new_cfg = self.load_config(self.parse_path(market_data['MarketDataFile']))

                # check if a calendar is loaded
                if 'CalendDataFile' in data['Calc']:
                    # parse calendar file
                    new_cfg.parse_calendar_file(self.parse_path(data['Calc']['CalendDataFile']))
                    # store a link
                    self.holiday_cfg_cache[market_data['MarketDataFile']] = data['Calc']['CalendDataFile']

                self.config_cache[market_data['MarketDataFile']] = new_cfg

            cfg = self.config_cache[market_data['MarketDataFile']]
            for section, section_data in market_data['ExplicitMarketData'].items():
                cfg.params[section].update(section_data)

            if 'CalendDataFile' in data['Calc'] and data['Calc'][
                'CalendDataFile'] != self.holiday_cfg_cache[market_data['MarketDataFile']]:
                # parse calendar file again
                cfg.parse_calendar_file(self.parse_path(data['Calc']['CalendDataFile']))
                # store a link
                self.holiday_cfg_cache[market_data['MarketDataFile']] = data['Calc']['CalendDataFile']

        if 'Deals' in data['Calc']:
            cfg.deals = {'Attributes': {
                'Tag_Titles': data['Calc']['Deals'].get('Tag_Titles', ''),
                'Reference': data['Calc']['Deals'].get('Reference')}}
            # try to compress the deal data if possible
            deals = data['Calc']['Deals']['Deals']
            if compress:
                deals['Children'][0]['Children'] = utils.compress_deal_data(deals['Children'][0]['Children'])

            cfg.deals.update({'Deals': deals})
            cfg.deals.update({'Calculation': data['Calc']['Calculation']})

        #  set the current context to newly loaded one
        self.current_cfg = cfg
        # return this object
        return self

    def save_json(self, json_filename):
        '''
        Quick and dirty method to save a JSON back to file - experimental - not fully implemented
        :param json_filename:
        :return: None (saves the job to JSON file)
        '''

        cfg = self.current_cfg
        try:
            md, cal = list(self.holiday_cfg_cache.items())[0]
        except:
            md, cal = '', ''

        final_json = {
            "Calc":
                {
                    "Calculation": cfg.deals['Calculation'],
                    "Deals": {
                        "Tag_Titles": cfg.deals['Attributes'].get('Tag_Titles', ''),
                        "Reference": cfg.deals['Attributes'].get('Reference', ''),
                        "Deals": cfg.deals['Deals']
                        },
                    "MergeMarketData": {
                        "MarketDataFile": md,
                        "ExplicitMarketData": {
                            "System Parameters": {},
                            "Price Factors": cfg.params['Price Factors']
                        }
                    },
                    "CalendDataFile": cal
                }
            }

        data = json.dumps(final_json, separators=(',', ':'), cls=CustomJsonEncoder)
        if json_filename is not None:
            with open(json_filename, 'wt') as f:
                f.write(data)
        else:
            return data

    def run_job(self, overrides=None, runparallel=False):
        # check what calc we should run
        if self.current_cfg.deals['Calculation']['Object'] == 'CreditMonteCarlo':
            return self.Credit_Monte_Carlo(overrides, runparallel)
        elif self.current_cfg.deals['Calculation']['Object'] == 'BaseValuation':
            return self.Base_Valuation(overrides)
        else:
            raise Exception('Unknown Calculation {}'.format(self.current_cfg.deals['Calculation']['Object']))

    def Credit_Monte_Carlo(self, overrides=None, runparallel=False):
        if runparallel:
            num_workers = torch.cuda.device_count()
            results = mp.Queue()
            workers = [mp.Process(target=run_cmc, args=(
                self.current_cfg, torch.float32, overrides, False, i, num_workers, results))
                            for i in range(num_workers)]

            # start all children
            for w in workers:
                w.start()

            # now collate the results
            post_processing = []
            for i in range(num_workers):
                post_processing.append(results.get())

            # close the queue
            results.close()

            # terminate all worker processes
            for w in workers:
                w.join()
                if w.is_alive():
                    w.close()

            post_results = {}
            for output in post_processing:
                data = dict(output)
                for k, v in data.items():
                    post_results.setdefault(k, []).append(v)
            # return the results
            return post_results
        else:
            return run_cmc(self.current_cfg, overrides=overrides)

    def Base_Valuation(self, overrides=None):
        return run_baseval(self.current_cfg, overrides=overrides)


class StressedContext(Context):

    def __init__(self, path_transform={}, file_transform={}):
        super(StressedContext, self).__init__(path_transform, file_transform)
        self.stressed_cfg = None
        self.current_models = None
        self.current_factors = None

    def restore_config(self):
        if self.current_models is not None:
            self.current_cfg.params['Price Models'].update(self.current_models)
        if self.current_factors is not None:
            self.current_cfg.params['Price Factors'].update(self.current_factors)

    def stress_config(self, rate_group):
        # calculate the current factors to stress
        factors_to_override, models_to_override = self.calc_stress_config(rate_group)
        # back up the old factors and models
        self.current_factors = {k: self.current_cfg.params['Price Factors'][k] for k in factors_to_override.keys()}
        self.current_models = {k: self.current_cfg.params['Price Models'][k] for k in models_to_override.keys()}
        # override the models
        self.current_cfg.params['Price Factors'].update(factors_to_override)
        self.current_cfg.params['Price Models'].update(models_to_override)

    def calc_stress_config(self, rate_group):
        '''
        :param rate_group: Rate group (InterestRate, FxRate etc.)
        :return:
        '''
        # check if the stressed file is loaded
        if self.stressed_config_file not in self.config_cache:
            self.stressed_cfg = self.load_config(self.parse_path(self.stressed_config_file))
            self.config_cache[self.stressed_config_file] = self.stressed_cfg

        self.stressed_cfg = self.config_cache[self.stressed_config_file]

        factors_to_override = {}
        models_to_override = {}

        for factor_type in rate_group:
            factors = [utils.Factor(factor_type, utils.check_rate_name(i)[1:])
                       for i in self.current_cfg.params['Price Factors'].keys() if i.startswith(factor_type)]
            factor_models, additional_factors = self.current_cfg.find_models(factors)
            for factor in [utils.check_tuple_name(x) for x in additional_factors.values()]:
                try:
                    factors_to_override[factor] = self.stressed_cfg.params['Price Factors'][factor]
                except KeyError as k:
                    logging.warning(
                        "Skipping Stressed Price Factor {} - not present in stressed file".format(k))
            for factor_model in [utils.check_tuple_name(x) for x in factor_models.keys()]:
                try:
                    models_to_override[factor_model] = self.stressed_cfg.params['Price Models'][factor_model]
                except KeyError as k:
                    logging.warning(
                        "Skipping Stressed Price Model {} - not present in stressed file".format(k))

        return factors_to_override, models_to_override
