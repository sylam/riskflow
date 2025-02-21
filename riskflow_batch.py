########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
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
import sys
import time
import json
import glob
import traceback
import pandas as pd
import numpy as np

from collections import defaultdict
from multiprocessing import Process, Queue, Lock


def rename_factor(cx, old_name, new_name):
    for old_local_df in [x for x in cx.current_cfg.params['Price Models'] if x.endswith(old_name)]:
        new_local_df = old_local_df.replace(old_name, new_name)
        cx.current_cfg.params['Price Models'][new_local_df] = cx.current_cfg.params['Price Models'][old_local_df]

    for old_local_param in [x for x in cx.current_cfg.params['Price Factors'] if x.endswith(
            'Parameters.{}'.format(old_name))]:
        new_local_param = old_local_param.replace(old_name, new_name)
        cx.current_cfg.params['Price Factors'][new_local_param] = cx.current_cfg.params['Price Factors'][
            old_local_param]
            

class JOB(object):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        self.cx = cx
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.netting_set = netting_set
        self.stats = stats
        self.logger = log
        self.splits = None
        self.params = {'Calc_Scenarios': 'No',
                       'Generate_Cashflows': 'No'}

        # load trades (and marketdata)
        self.cx.load_json(os.path.join(self.input_path, self.rundate, self.netting_set))

        # get the netting set
        self.ns = self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument']
        # get the agreement currency
        self.agreement_currency = self.ns.field.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.field.get('Balance_Currency', self.agreement_currency)

    def perform_calc(self):
        # create the calculation    
        if self.valid():
            try:
                if self.splits:
                    orig_netting_set = self.netting_set
                    for group, indices in enumerate(self.splits):
                        self.netting_set = orig_netting_set.replace('.json', '_{}.json'.format(group))
                        netting = os.path.splitext(self.netting_set)[0]
                        self.cx.deals['Attributes']['Reference'] = netting
                        self.cx.deals['Deals']['Children'][0]['Children'] = list(indices)
                        self.run_calc()
                else:
                    self.run_calc()
            except Exception as e:
                self.logger(self.netting_set, "!! CRITICAL ERROR In Calc !! - {} - Skipping".format(e.args))

    def valid(self):
        return False

    def run_calc(self):
        pass


class PFE(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(PFE, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = self.cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params[
            'Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self):
        self.params['Run_Date'] = self.rundate
        filename = 'PFE_{}_{}.csv'.format(
            self.params['Run_Date'], self.cx.current_cfg.deals['Attributes']['Reference'])
        legacy_filename = 'OutputAAJ_{}.aaj'.format(self.cx.current_cfg.deals['Attributes']['Reference'])

        # load up CVA calc params
        if self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            self.logger(self.netting_set, 'is uncollateralized')
            self.params['Dynamic_Scenario_Dates'] = 'No'

        # do 20000 sims in batches of 1024
        self.params['Simulation_Batches'] = 20
        self.params['Batch_Size'] = 1024
        # set the percentiles to 95 and 99
        self.params['Percentile'] = '95, 99'
        # do more MCMC simulations
        self.params['MCMC_Simulations'] = 8192

        calc, out = self.cx.run_job(overrides=self.params)
        profile = out['Results']['exposure_profile']
        legacy_exposure_profile = self.cx.current_cfg.parse_output_results(profile, calc.params['Currency'])
        PFE_key = [x for x in profile.keys() if x.startswith('PFE')][0]
        out['Stats'].update({PFE_key: profile[PFE_key].max(), 'Currency': calc.params['Currency']})
        # set the currency in the profile
        profile.index.name = calc.params['Currency']
        # write the results
        profile.to_csv(os.path.join(self.outputdir, filename))
        if not os.path.isdir(os.path.join(self.outputdir, self.rundate)):
            os.mkdir(os.path.join(self.outputdir, self.rundate))
        with open(os.path.join(self.outputdir, self.rundate, legacy_filename), 'w') as f:
            f.write(legacy_exposure_profile)
        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class CollateralBaseVal(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(CollateralBaseVal, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)

        # get the collateral assets
        try:
            assets = self.ns.field['Collateral_Assets'].keys()[0]
            if assets is None and self.ns.field['Collateralized'] == 'True':
                assets = 'Cash'
        except:
            assets = 'Cash' if self.ns.field['Collateralized'] == 'True' else 'None'

        self.collateral = assets
        self.params = {'Run_Date': rundate}

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] or self.collateral != 'Cash':
            return False
        else:
            return True

    def run_calc(self):
        import riskflow as rf

        if 'Reference' not in self.cx.current_cfg.deals['Attributes']:
            self.cx.current_cfg.deals['Attributes']['Reference'] = os.path.splitext(self.netting_set)[0]
        ending = self.params['Run_Date'] + '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        filename = 'BaseVal_' + ending
        filename_greeks = 'BaseVal_Delta_' + ending
        filename_greeks_second = 'BaseVal_Gamma_' + ending

        self.params.update({'Currency': self.agreement_currency, 'Greeks': 'All'})

        try:
            _, out = self.cx.Base_Valuation(overrides=self.params)
            out['Results']['mtm'].to_csv(os.path.join(self.outputdir, 'Greeks', filename))
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
            self.logger(self.netting_set, 'Failed to baseval')
        else:
            self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
            out['Stats'].update({'Currency': self.params['Currency']})
            out['Stats'].update({'MTM': out['Results']['mtm']['Value'].head(1).values[0]})
            out['Results']['Greeks_First'].to_csv(os.path.join(self.outputdir, 'Greeks', filename_greeks))
            if 'Greeks_Second' in out['Results']:
                out['Results']['Greeks_Second'].to_csv(os.path.join(self.outputdir, 'Greeks', filename_greeks_second))


class CVA_GRAD(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(CVA_GRAD, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        # change the currency
        self.params['Currency'] = self.cx.current_cfg.deals['Calculation']['Currency']
        # does this netting set have any combination deals?
        self.cominations = 0
        # rename some curves to better handle CSA discounting
        rename_factor(self.cx, 'ZAR-SWAP', 'ZAR-OIS')
        rename_factor(self.cx, 'ZAR-JIBAR-3M', 'ZAR-JIBAR-3M-OIS')

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:  
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals)
            
            cpy = self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment'].get('Counterparty')        
            if self.cx.current_cfg.params['Price Factors'].get('SurvivalProb.{}'.format(cpy) ,{}).get('Recovery_Rate')==1.0:
                #100% recovery - skip
                self.logger(self.netting_set, 'Netting set has 100% Recovery - skipping')
                return False
            else:
                return True
                

    def run_calc(self):
        import riskflow as rf

        filename = 'CVA_' + self.rundate + '_' + self.params['Currency']+ '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
         # make sure other xva is switched off
        if 'Funding_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Funding_Valuation_Adjustment']
        if 'Collateral_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment']
            
        # how many simulations do we want (in thousands)
        num_1ksims = 16
        if self.combinations:
            # use smaller batches if there are comination deals present
            self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
            self.params['Simulation_Batches'] = 4 * num_1ksims
            self.params['Batch_Size'] = 256
        else:
            self.params['Simulation_Batches'] = num_1ksims
            self.params['Batch_Size'] = 1024
            
        # load up CVA calc params
        if self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            self.logger(self.netting_set, 'is uncollateralized')
            self.params['Dynamic_Scenario_Dates'] = 'No'

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping gradient CVA calc as file already exists')
            self.params['Credit_Valuation_Adjustment'] = {'Gradient': 'No'}
        else:
            self.params['Credit_Valuation_Adjustment'] = {'Gradient': 'Yes'}

        calc_complete = False
        num_tries = 0

        while not calc_complete:
            try:
                calc, out = self.cx.run_job(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries>2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))                        
                    out = {'Stats': {}}
                    out['Stats'].update({'CVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': {}}
                out['Stats'].update({'CVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_cva' in out['Results']:
                    grad_cva = out['Results']['grad_cva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    # store the CVA as part of the stats
                    out['Stats'].update({
                        'CVA': out['Results']['cva'],
                        'Currency': self.params['Currency']})                        
                    # log the netting set
                    self.logger(self.netting_set, 'CVA calc complete')
                else:
                    out['Stats'].update({
                        'CVA': out['Results']['cva'] if 'cva' in out['Results'] else np.nan,
                        'Currency': self.params['Currency']})
                    self.logger(self.netting_set, 'CVA calc complete - (gradients already present)')
                                    
                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class COLLVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(COLLVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from riskflow.utils import Curve
        # change the currency
        self.params['Currency'] = self.balance_currency
        # does this netting set have any combination deals?
        self.cominations = 0

        if self.ns.field.get('Collateral_Assets') is None:
            self.logger(self.netting_set, 'Check balance currency {}'.format(self.balance_currency))

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] or self.ns.field.get(
            'Collateralized', 'False') == 'False':
            return False
        else:
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals)
            return True

    def run_calc(self):
        filename = 'COLLVA_' + self.rundate + '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        num_deals = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        self.logger(self.netting_set, 'Netting set has {} deals'.format(num_deals))

        spreads = {
            'USD': {'collateral': 0, 'funding': 65},
            'EUR': {'collateral': 0, 'funding': 65},
            'GBP': {'collateral': 0, 'funding': 65},
            'ZAR': {'collateral': -10, 'funding': 15}
            }
    
        curves = {'USD': {'collateral': 'USD-OIS', 'funding': 'USD-SOFR.USD-SOFR3M_CAS'},
                  'EUR': {'collateral': 'EUR-EONIA', 'funding': 'EUR-EURIBOR-3M'},
                  'GBP': {'collateral': 'GBP-SONIA', 'funding': 'GBP-SONIA'},
                  'ZAR': {'collateral': 'ZAR-SWAP', 'funding': 'ZAR-SWAP'}}

        collva_sect = self.cx.current_cfg.deals['Calculation'].get(
            'Collateral_Valuation_Adjustment', {'Calculate': 'Yes'})
        # sensible defaults
        collva_sect['Collateral_Curve'] = collva_sect.get(
            'Collateral_Curve', curves[self.balance_currency]['collateral'])
        collva_sect['Funding_Curve'] = collva_sect.get(
            'Funding_Curve', curves[self.balance_currency]['funding'])
        collva_sect['Collateral_Spread'] = collva_sect.get(
            'Collateral_Spread', spreads[self.balance_currency]['collateral'])
        collva_sect['Funding_Spread'] = collva_sect.get(
            'Funding_Spread', spreads[self.balance_currency]['funding'])
        self.cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment'] = collva_sect
        # make sure other xva is switched off
        if 'Credit_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment']
        if 'Funding_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Funding_Valuation_Adjustment']
        # how many simulations do we want (in thousands)
        num_1ksims = 16
        if self.combinations:
            # use smaller batches if there are comination deals present
            self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
            self.params['Simulation_Batches'] = 4 * num_1ksims
            self.params['Batch_Size'] = 256
        else:
            self.params['Simulation_Batches'] = 2 * num_1ksims
            self.params['Batch_Size'] = 512

        self.params['Dynamic_Scenario_Dates'] = 'Yes'
        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping gradient COLLVA calc as file already exists')
            self.params['COLLVA'] = {'Gradient': 'No'}
            self.params['Simulation_Batches'] = num_1ksims
            self.params['Batch_Size'] = 1024
        else:
            self.params['COLLVA'] = {'Gradient': 'Yes'}

        calc_complete = False
        num_tries = 0
        CSA = {k: v.value() if hasattr(v, 'value') else v for k, v in self.ns.field['Credit_Support_Amounts'].items()}
        while not calc_complete:
            try:
                calc, out = self.cx.run_job(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries > 2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))                        
                    out = {'Stats': CSA}
                    out['Stats'].update({'CollVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': CSA}
                out['Stats'].update({'CollVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_collva' in out['Results']:
                    grad_collva = out['Results']['grad_collva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    grad_collva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    out['Stats'].update(CSA)
                    # store the CollVA as part of the stats
                    out['Stats'].update({
                        'CollVA': out['Results']['collva'],
                        'Opening_CSA_Balance': self.ns.field['Opening_Balance'],
                        'MtM_t0': out['Results']['mtm'][0].mean(),
                        'Calc_Collateral_t2': out['Results']['collateral'][1].mean(),
                        'Agreement_Currency': self.agreement_currency,
                        'Currency': self.params['Currency']})
                    # log the netting set
                    self.logger(self.netting_set, 'CollVA calc complete')
                else:
                    out['Stats'].update({
                        'CollVA': out['Results']['collva'] if 'collva' in out['Results'] else np.nan,
                        'Opening_CSA_Balance': self.ns.field['Opening_Balance'],
                        'MtM_t0': out['Results']['mtm'][0].mean() if 'mtm' in out['Results'] else np.nan,
                        'Calc_Collateral_t2': out['Results']['collateral'][1].mean() if 'collateral' in out['Results'] else np.nan,
                        'Agreement_Currency': self.agreement_currency,
                        'Currency': self.params['Currency']})
                    self.logger(self.netting_set, 'CollVA calc complete - (gradients already present)')

                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class Legacy_FVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(Legacy_FVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)

        # hardcoded list to excluded trades
        exclusions = pd.read_csv('/mnt/MarketData/FVAStaticData/excluded trades.csv').groupby('Netting')
        if os.path.splitext(netting_set)[0] in exclusions.groups:
            to_exclude = exclusions.get_group(os.path.splitext(netting_set)[0])['Reference']
            for i in cx.current_cfg.deals['Deals']['Children'][0]['Children']:
                if i['Instrument'].field['Reference'] in to_exclude.values:
                    self.logger(self.netting_set, 'Excluding deal {}'.format(i['Instrument'].field['Reference']))
                    i['Ignore'] = 'True'
                else:
                    i['Ignore'] = 'False'

        # get the calculation currency        
        self.calculation_currency = self.ns.field.get('Balance_Currency', 'ZAR')
        # get the collateral assets
        try:
            self.assets = list(self.ns.field['Collateral_Assets'].keys()).pop()
            if self.assets is None and self.ns.field['Collateralized'] == 'True':
                self.assets = 'Cash_Collateral'
        except:
            self.assets = 'Cash_Collateral' if self.ns.field['Collateralized'] == 'True' else 'None'

        self.logger(self.netting_set, 'asset is {}'.format(self.assets))

        # load up a calendar (for theta)
        self.business_day = cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        self.params['Currency'] = self.balance_currency
        # does this netting set have any combination deals?
        self.cominations = 0

        if self.ns.field.get('Collateral_Assets') is None:
            self.logger(self.netting_set, 'Check balance currency {}'.format(self.balance_currency))

        num_children = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        if num_children > 1000:
            # need to split this into smaller groups
            self.splits = np.array_split(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'].copy(), num_children//500)
     
    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] or self.ns.field.get('Collateralized', 'False') == 'False' or self.assets!='Cash_Collateral':
            return False
        else:
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals)
            return True

    def run_calc(self):
        import riskflow as rf
        netting = os.path.splitext(self.netting_set)[0]
        filename = 'Legacy_FVA_' + self.rundate + '_' + self.cx.current_cfg.deals['Attributes'].get('Reference', netting) + '.csv'
        num_deals = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        self.logger(self.netting_set, 'Netting set has {} deals'.format(num_deals))
        
        spreads = {
            'USD': {'collateral': 0, 'funding': 65},
            'EUR': {'collateral': 0, 'funding': 65},
            'GBP': {'collateral': 0, 'funding': 65},
            'ZAR': {'collateral': -10, 'funding': 15}
            }
        
        curves = {'USD': {'collateral': 'USD-OIS-STATIC-OLD', 'funding': 'USD-SOFR.USD-SOFR3M_CAS'},
                  'EUR': {'collateral': 'EUR-EONIA', 'funding': 'EUR-EURIBOR-3M'},
                  'GBP': {'collateral': 'GBP-SONIA', 'funding': 'GBP-SONIA'},
                  'ZAR': {'collateral': 'ZAR-SWAP', 'funding': 'ZAR-SWAP'}}
        
        #calculation parameters
        overrides = {
            'Run_Date':self.rundate,
            'Batch_Size': 1024,
            'Simulation_Batches': 5,
            'Random_Seed':4126, 
            'MCMC_Simulations': 8192,
            'Calc_Scenarios':'No',
            'Currency': self.balance_currency,
            'Dynamic_Scenario_Dates': 'No',
            'LegacyFVA': {
                    'Funding_Curve': curves[self.balance_currency]['funding'],
                    'Funding_Spread': spreads[self.balance_currency]['funding'],
                    'Collateral_Curve': curves[self.balance_currency]['collateral'],
                    'Collateral_Spread': spreads[self.balance_currency]['collateral'],
                    'Gradient': 'No'
                },
            }

        # make sure other xva is switched off
        if 'Credit_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment']
        if 'Funding_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Funding_Valuation_Adjustment']

        calc_complete = False
        num_tries = 0
        try:
            CSA = {k:v.value() if hasattr(v,'value') else v for k,v in self.ns.field['Credit_Support_Amounts'].items()}
        except:
            CSA = {}

        # switch off collateral
        self.ns.field['Collateralized']='False'
                    
        while not calc_complete:
            try:
                calc, out = self.cx.run_job(self.params)
                # calc, out = rf.run_cmc(self.cx, overrides=overrides)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
                overrides['Simulation_Batches'] *= 2
                overrides['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(overrides['Batch_Size']))
                num_tries += 1
                if num_tries>2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(overrides['Batch_Size']))
                    out = {'Stats': CSA}
                    out['Stats'].update({'legacy_fva': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': CSA}
                out['Stats'].update({'legacy_fva': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_legacy_fva' in out['Results']:
                    grad_legacy_fva = out['Results']['grad_legacy_fva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes'].get('Reference', netting)})
                    grad_legacy_fva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    out['Stats'].update(CSA)
                    # store the CollVA as part of the stats
                    out['Stats'].update({
                        'legacy_fva': out['Results']['legacy_fva'],
                        'Opening_CSA_Balance': self.ns.field['Opening_Balance'],
                        'MtM_t0': out['Results']['mtm'].values[0].mean(),
                        'CSA': self.assets,
                        'Agreement_Currency': self.agreement_currency,
                        'Currency': self.params['Currency']})
                    # log the netting set
                    self.logger(self.netting_set, 'Legacy FVA calc complete')
                else:
                    out['Stats'].update({
                        'legacy_fva': out['Results']['legacy_fva'] if 'legacy_fva' in out['Results'] else np.nan,
                        'Opening_CSA_Balance': self.ns.field['Opening_Balance'],
                        'MtM_t0': out['Results']['mtm'].values[0].mean() if 'mtm' in out['Results'] else np.nan,
                        'CSA': self.assets,
                        'Agreement_Currency': self.agreement_currency,
                        'Currency': self.params['Currency']})
                    self.logger(self.netting_set, 'Legacy FVA calc complete - (gradients already present)')

                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class FVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(FVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from riskflow.utils import Curve
        # change the currency
        self.params['Currency'] = self.cx.current_cfg.deals['Calculation']['Currency']
        # does this netting set have any combination deals?
        self.cominations = 0
        # rename some curves to better handle CSA discounting
        rename_factor(self.cx, 'ZAR-SWAP', 'ZAR-OIS')
        rename_factor(self.cx, 'ZAR-JIBAR-3M', 'ZAR-JIBAR-3M-OIS')

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals)
            
            if self.cx.current_cfg.deals['Attributes'].get('Reference','').endswith('CPY'):
                #internal deal
                self.logger(self.netting_set, 'Netting set is internal - only filtering specific portfolios')
                for i in self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
                    tags = i['Instrument'].field.get('Tags',[''])[0].split(',')                    
                    if tags and tags[1] in ['IR Prime Strat','IR Cpty Prime NonCSA','IR Cpty Swap Internal','IR Cpty CPI KSP Hedge','IR Cpty Vol Internal','IR Cpty Swaps NonCSA','IR Prime Hedge','IR Prime Fixed']:
                    # if False and not str(i['Instrument'].field['Reference']) in ['CrB_BNP_Paribas__Paris__ISDA', 'CrB_Citibank_NA_NY_ISDA']:
                        i['Ignore'] = 'False'
                    else:
                        i['Ignore'] = 'True'
            
            # if assets.replace('_Collateral','')=='Equity':
            if self.ns.field.get('Credit_Support_Amounts', {}).get('Independent_Amount') is not None:
                self.logger(self.netting_set, 'Netting set has an Independent Amount - assuming Equity collar deals and Skipping')
                return False
            else:
                return True

    def run_calc(self):
        
        filename = 'FVA_' + self.rundate + '_' + self.params['Currency']+ '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        num_deals = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        self.logger(self.netting_set, 'Netting set has {} deals'.format(num_deals))

        # make sure other xva is switched off
        if 'Credit_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment']
        if 'Collateral_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment']
        # how many simulations do we want (in thousands)
        num_1ksims = 16
        if self.combinations:
            # use smaller batches if there are comination deals present
            self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
            self.params['Simulation_Batches'] = 4 * num_1ksims
            self.params['Batch_Size'] = 256
        else:
            self.params['Simulation_Batches'] = num_1ksims
            self.params['Batch_Size'] = 1024

        self.params['Dynamic_Scenario_Dates'] = 'Yes'
        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping gradient FVA calc as file already exists')
            self.params['Funding_Valuation_Adjustment'] = {'Gradient': 'No'}
        else:
            self.params['Funding_Valuation_Adjustment'] = {'Gradient': 'Yes'}

        calc_complete = False
        num_tries = 0

        while not calc_complete:
            try:
                calc, out = self.cx.run_job(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries > 2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))                        
                    out = {'Stats': {}}
                    out['Stats'].update({'FVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': {}}
                out['Stats'].update({'FVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_fva' in out['Results']:
                    grad_fva = out['Results']['grad_fva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    grad_fva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    # store the FVA as part of the stats
                    out['Stats'].update({
                        'FVA': out['Results']['fva'],
                        'Currency': self.params['Currency']})
                    # log the netting set
                    self.logger(self.netting_set, 'FVA calc complete')
                else:
                    out['Stats'].update({
                        'FVA': out['Results']['fva'] if 'fva' in out['Results'] else np.nan,
                        'Currency': self.params['Currency']})
                    self.logger(self.netting_set, 'FVA calc complete - (gradients already present)')

                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class SA_CVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(SA_CVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        # does this netting set have any combination deals?
        self.cominations = 0
        if 'HullWhite2FactorModelParameters.USD-OIS' not in self.cx.current_cfg.params['Price Factors']:
            self.cx.current_cfg.params['Price Factors']['HullWhite2FactorModelParameters.USD-OIS'] = self.cx.current_cfg.params['Price Factors']['HullWhite2FactorModelParameters.USD-SOFR']
        if self.cx.stressed_config_file is None:
            from conf import UAT_MARKETDATA
            short_date = ''.join(rundate[2:].split('-')[::-1])
            self.cx.stressed_config_file = UAT_MARKETDATA+"\\CVAMarketDataBackup\\CVAMarketData_Calibrated_Vega_{}.json".format(short_date)
            self.logger(self.netting_set, 'setting Stressed market file to hardcoded value - {}'.format(self.cx.stressed_config_file))

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals) 
            return True

    def calc_vega(self, num_2ksims, calc, output):
    
        def run_vega(vega_factors):
            if [x for x in calc.all_factors.keys() if x.type in vega_factors]:
                # stress the cfg
                self.cx.stress_config(vega_factors)
                try:
                    _, vega = self.cx.run_job(self.params)
                except:
                    self.logger(self.netting_set, 'vega failed')
                    vega = output
                # restore the config
                self.cx.restore_config()
                vega_result = vega['Results']['cva'] - output['Results']['cva']
                self.logger(self.netting_set, 'vega calc for {} is {}'.format(vega_factors, vega_result))
                return vega_result
            else:
                self.logger(self.netting_set, 'skipping vega calc for {}'.format(vega_factors))
                return 0.0
                
        # turn off gradients
        self.params['Credit_Valuation_Adjustment']['Gradient'] = 'No'
        
        results = {}
        results['CVA_IR_Vega'] = run_vega(['InterestRate', 'InflationRate'])
        results['CVA_CM_Vega'] = run_vega(['ForwardPrice'])  
          
        return results


    def run_calc(self):
        filename = self.rundate + '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        num_deals = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        self.logger(self.netting_set, 'Netting set has {} deals'.format(num_deals))
        # how many simulations do we want (in thousands) - must be divisible by 4
        num_2ksims = 8
        # load up CVA calc params
        if self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
            # change the liquidation period to 10 days (as per regs)
            self.ns.field['Liquidation_Period'] = 10.0
            if self.combinations:
                # use smaller batches if there are comination deals present
                self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
                self.params['Simulation_Batches'] = 4 * num_2ksims
                self.params['Batch_Size'] = 512
            else:
                self.params['Simulation_Batches'] = 2 * num_2ksims
                self.params['Batch_Size'] = 1024
        else:
            self.params['Dynamic_Scenario_Dates'] = 'No'
            self.params['Simulation_Batches'] = num_2ksims
            self.params['Batch_Size'] = 2048
            self.logger(self.netting_set, 'is uncollateralized')

        # get the calculation parameters for CVA
        cva_sect = self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment']

        # update the params
        self.params['Currency'] = 'ZAR'
        self.params['Deflation_Interest_Rate'] = 'ZAR-SWAP'
        self.params['Credit_Valuation_Adjustment'] = cva_sect
        self.params['Credit_Valuation_Adjustment']['CDS_Tenors'] = [0.5, 1, 3, 5, 10]
        self.params['Credit_Valuation_Adjustment']['Gradient'] = 'Yes'

        # this produces (or not) the gamma matrix
        cva_sect['Hessian'] = 'No'
        calc_complete = False
        num_tries = 0
        while not calc_complete:
            try:
                calc, out = self.cx.run_job(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries>2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))
                    out = {'Stats': {}}
                    out['Stats'].update({'CVA_IR_Vega': np.nan, 'CVA_CM_Vega': np.nan, 'CVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': {}}
                out['Stats'].update({'CVA_IR_Vega': np.nan, 'CVA_CM_Vega': np.nan, 'CVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_cva' in out['Results']:
                    grad_cva = out['Results']['grad_cva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
                    grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', 'SACVA_' + filename))
                    # write out the CS01
                    if 'CS01' in out['Results']: 
                        out['Results']['CS01'].to_csv(os.path.join(self.outputdir, 'Greeks', 'CS01_'+filename))
                    # now calculate the vega sensitivities
                    out['Stats'].update(self.calc_vega(num_2ksims, calc, out))
                    
                    # log the netting set
                    self.logger(self.netting_set, 'CVA calc complete')
                else:
                    self.logger(self.netting_set, '!!! Critical !!! - CVA NOT calc complete - Check gradients and Vega calc')

                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


def work(id, lock, queue, results, job, rundate, input_path, outputdir):
    def log(netting_set, msg):
        lock.acquire()
        print('JOB %s:' % id, '{0}: {1}'.format(netting_set, msg))
        lock.release()

    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)

    # now load the library 
    import riskflow as rf
    from conf import PROD_MARKETDATA, UAT_MARKETDATA
    # remap paths from windows to linux
    path_transform = {
        UAT_MARKETDATA: '/mnt/MarketData',
        PROD_MARKETDATA: '/mnt/MarketDataProd',
        UAT_MARKETDATA + '/CVAMarketDataBackup': '/mnt/MarketData/CVAMarketDataBackup',
        PROD_MARKETDATA + '/CVAMarketDataBackup': '/mnt/MarketDataProd/CVAMarketDataBackup'
    }
    # a context object loads (and caches) one or more configs
    # remap any paths as necessary
    if job == 'PFE':
        cx = rf.Context(path_transform=path_transform, file_transform={
            'CVAMarketData_Calibrated_New.json': 'MarketData.json',
            'MarketData.dat': 'MarketData.json'
        })
    elif job == 'SA_CVA':
        # this uses 2 marketdata files - a "normal" file and a "stressed" file which contains the vega recalibrations
        cx = rf.StressedContext(path_transform=path_transform)
    else:
        cx = rf.Context(path_transform=path_transform)

    # log results
    logs = {}

    while True:
        # get the task
        task = queue.get()
        if task is None:
            break

        try:
            obj = globals().get(job)(cx, rundate, input_path, outputdir, task, logs, log)
        except Exception as e:
            log(task, "!! CRITICAL ERROR In JSON !! - {} - Skipped".format(e.args))
        else:
            obj.perform_calc()

    # empty the queue
    queue.put(None)
    # get ready to send the results
    result = []

    # write out the logs
    if 'Stats' in logs:
        stats_file = os.path.join(outputdir, 'Stats', '{0}_Stats_{1}_JOB_{2}.csv'.format(job, rundate, id))
        pd.DataFrame(data=logs['Stats']).T.to_csv(stats_file)
        result.append(('Stats', stats_file))

    results.put(result)


class Parent:
    def __init__(self, num_jobs):
        self.queue = Queue()
        self.results = Queue()
        self.lock = Lock()
        self.NUMBER_OF_PROCESSES = num_jobs

    def start(self, job, rundate, input_path, outputdir, wildcard='CrB*.json'):
        print("starting {0} workers in {1}".format(self.NUMBER_OF_PROCESSES, input_path))

        self.workers = [Process(target=work, args=(
            i, self.lock, self.queue, self.results, job, rundate, input_path, outputdir))
                        for i in range(self.NUMBER_OF_PROCESSES)]

        # start all children
        for w in self.workers:
            w.start()

        crbs = map(lambda x: os.path.split(x)[-1],
                   sorted(glob.glob(os.path.join(input_path, rundate, wildcard)), key=os.path.getsize)[::-1])

        # load the crbs on the queue
        for netting_set in crbs:
            self.queue.put(netting_set)

        self.queue.put(None)

        # now collate the results
        post_processing = []
        for i in range(self.NUMBER_OF_PROCESSES):
            post_processing.append(self.results.get())

        # close the queue
        self.queue.close()
        self.results.close()

        # terminate all worker processes
        for w in self.workers:
            w.join()
            if w.is_alive():
                w.close()

        post_results = {}
        for output in post_processing:
            data = dict(output)
            for k, v in data.items():
                post_results.setdefault(k, []).append(pd.read_csv(v, index_col=0))

        # write out the combined data
        for k, v in post_results.items():
            if v:
                out_path = os.path.join(outputdir, k, '{0}_{1}_{2}_Total.csv'.format(job, k, rundate))
                pd.concat(v).to_csv(out_path)


def main():
    import argparse

    jobs = [cls.__name__ for cls in globals().values() if
            isinstance(cls, type) and hasattr(cls, 'valid') and cls.__name__ != 'JOB']
    # setup the arguments
    parser = argparse.ArgumentParser(description='Run a riskflow batch on a directory of .json netting sets.')
    parser.add_argument('num_jobs', type=int, help='the number of gpu\'s to use (if available) else cpu\'s')
    parser.add_argument('job', type=str, help='the job name', choices=jobs)
    parser.add_argument('rundate', type=str, help='batch rundate')
    parser.add_argument('input_path', type=str, help='directory containing the input files (note that the rundate is '
                                                     'assumed to be a directory within)')
    parser.add_argument('output_path', type=str, help='output directory')
    parser.add_argument('filename', type=str, help='filename(s) in input_path to run - wildcards allowed')

    # get the arguments
    args = parser.parse_args()

    Parent(args.num_jobs).start(
        args.job, args.rundate, args.input_path, args.output_path, args.filename)
    return 0


if __name__ == '__main__':
    sys.exit(main())
