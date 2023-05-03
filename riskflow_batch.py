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
import sys
import time
import json
import glob
import traceback
import pandas as pd
import numpy as np

from collections import defaultdict
from multiprocessing import Process, Queue, Lock


class JOB(object):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        self.cx = cx
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.netting_set = netting_set
        self.stats = stats
        self.logger = log
        self.params = {'Calc_Scenarios': 'No',
                       'Run_Date': rundate,
                       'Time_grid': '0d 2d 1w(1w) 3m(1m) 1y(3m)'}
        # load trades (and marketdata)
        self.cx.load_json(os.path.join(self.input_path, self.rundate, self.netting_set))
        # get the netting set
        self.ns = self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field
        # get the agreement currency
        self.agreement_currency = self.ns.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.get('Balance_Currency', self.agreement_currency)

    def perform_calc(self):
        # create the calculation    
        if self.valid():
            self.run_calc()

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
        filename = 'PFE_{}_{}.csv'.format(
            self.params['Run_Date'], self.cx.current_cfg.deals['Attributes']['Reference'])

        # load up CVA calc params
        if self.ns.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            self.logger(self.netting_set, 'is uncollateralized')
            self.params['Dynamic_Scenario_Dates'] = 'No'

        # do 20000 sims in batches of 1024
        self.params['Simulation_Batches'] = 10 * 2
        self.params['Batch_Size'] = 1024
        try:
            calc, out = self.cx.run_job(overrides=self.params)
        except RuntimeError as e:  # Out of memory
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
            self.logger(self.netting_set, 'Exception: Could not complete')
        else:        
            profile = out['Results']['profile']
            out['Stats'].update({'PFE': profile['PFE'].max(), 'Currency': calc.params['Currency']})            
            # write the grad
            profile.index.name = calc.params['Currency']
            profile.to_csv(os.path.join(self.outputdir, filename))
            self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class CVA_GRAD(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(CVA_GRAD, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = self.cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params[
            'Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        # allow cva overrides
        self.params['CVA'] = {}

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self):
        filename = 'CVA_{}_{}.csv'.format(
            self.params['Run_Date'], self.cx.current_cfg.deals['Attributes']['Reference'])

        # load up CVA calc params
        if self.ns.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            self.logger(self.netting_set, 'is uncollateralized')
            self.params['Dynamic_Scenario_Dates'] = 'No'

        # do 10000 sims in batches of 512
        self.params['Simulation_Batches'] = 1
        self.params['Batch_Size'] = 512

        # work out the next business day
        next_day = self.business_day.rollforward(
            pd.Timestamp(self.params['Run_Date']) + pd.offsets.Day(1)).strftime('%Y-%m-%d')

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping CVA gradient calc as file already exists')
            self.params['CVA']['Gradient'] = 'No'
            calc, out = self.cx.run_job(overrides=self.params)
            stats = out['Stats']
            # store the CVA as part of the stats
            out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': calc.params['Currency']})
            # now calc theta
            self.params.update({'Run_Date': next_day})
            calc, theta = self.cx.run_job(overrides=self.params)
            out['Stats'].update({'CVA_Theta': theta['Results']['cva']})
            # store cva
            self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
        else:
            self.params['CVA']['Gradient'] = 'Yes'
            calc_complete = False
            while not calc_complete:
                try:
                    calc, out = self.cx.run_job(overrides=self.params)                  
                except RuntimeError as e:  # Out of memory
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(
                        exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
                    self.params['Simulation_Batches'] *= 2
                    self.params['Batch_Size'] //= 2
                    self.logger(self.netting_set,
                        'Exception: OOM - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                    if self.params['Batch_Size']<128:
                        self.logger(self.netting_set, 'Batchsize too small - skipping')
                        break
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(
                        exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
                    self.logger(self.netting_set, 'Unhandled Exception: Skipping')
                    break
                else:
                    calc_complete = True            
                    
            if calc_complete:      
                grad_cva = out['Results']['grad_cva'].rename(
                    columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': calc.params['Currency']})
                # write the grad
                grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                # run the next day
                self.params['Run_Date'] = next_day
                # switch off gradients
                self.params['CVA']['Gradient'] = 'No'
                _, theta = self.cx.run_job(overrides=self.params)
                out['Stats'].update({'CVA_Theta': theta['Results']['cva']})
                self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
                # log the netting set
                self.logger(self.netting_set, 'CVA calc complete')
            else:
              self.logger(self.netting_set, 'Error CVA calc NOT complete!')            
        

class COLLVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.netting_set = os.path.splitext(netting_set)[0] + '.json'
        self.stats = stats
        self.logger = log
        self.params = {'calc_name': ('cmc', 'calc1'),
                       'Time_grid': '0d 2d 1w(1w) 1m(1m) 3m(3m) 1y(1y)',
                       'Run_Date': rundate, 'Currency': 'ZAR', 'Random_Seed': 5126,
                       'Calc_Scenarios': 'No', 'Dynamic_Scenario_Dates': 'Yes',
                       'Debug': 'No', 'NoModel': 'Constant', 'Partition': 'None',
                       'Generate_Slideshow': 'No', 'PFE_Recon_File': ''}

        from riskflow.adaptiv import AdaptivContext
        self.cx = AdaptivContext()
        # load up the CVA marketdata file
        self.cx.parse_json(os.path.join(self.input_path, rundate, 'MarketDataCVA.json'))
        # update the cva data with the arena data - should also have the parameters overridden
        self.cx.params['Price Factors'].update(cx.params['Price Factors'])
        # load trade
        self.cx.parse_json(os.path.join(self.input_path, self.rundate, self.netting_set))
        # get the netting set
        self.ns = self.cx.deals['Deals']['Children'][0]['Instrument'].field
        # get the agreement currency
        self.agreement_currency = self.ns.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.get('Balance_Currency', self.agreement_currency)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from riskflow.utils import Curve
        # change the currency
        self.params['Currency'] = self.agreement_currency

        def makeflatcurve(curr, bps, tenor=30):
            return {'Currency': curr, 'Curve': Curve([], [[0, bps * 0.01 * 0.01], [tenor, bps * 0.01 * 0.01]]),
                    'Day_Count': 'ACT_365',
                    'Property_Aliases': None, 'Sub_Type': 'None'}

        if self.agreement_currency == 'ZAR':
            self.cx.params['Price Factors']['InterestRate.ZAR-SWAP.OIS'] = makeflatcurve('ZAR', -15)
            self.cx.params['Price Factors']['InterestRate.ZAR-SWAP.FUNDING'] = makeflatcurve('ZAR', 10)
            self.cx.deals['Deals']['Children'][0]['Instrument'].field['Collateral_Assets'] = {
                'Cash_Collateral': [{
                    'Currency': 'USD',
                    'Collateral_Rate': 'ZAR-SWAP.OIS',
                    'Funding_Rate': 'ZAR-SWAP.FUNDING',
                    'Haircut_Posted': 0.0,
                    'Amount': 1.0}]}
        else:
            # set the OIS curve to use the master curve
            self.cx.params['Price Factors']['HullWhite2FactorModelParameters.USD-OIS'] = cx.params[
                'Price Factors']['HullWhite2FactorModelParameters.USD-MASTER']
            self.cx.params['Price Factors']['InterestRate.USD-LIBOR-3M.FUNDING'] = makeflatcurve('USD', 65)
            self.cx.deals['Deals']['Children'][0]['Instrument'].field['Collateral_Assets'] = {
                'Cash_Collateral': [{
                    'Currency': 'USD',
                    'Collateral_Rate': 'USD-OIS',
                    'Funding_Rate': 'USD-LIBOR-3M.FUNDING',
                    'Haircut_Posted': 0.0,
                    'Amount': 1.0}]}

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children'] or self.cx.deals[
            'Deals']['Children'][0]['Instrument'].field.get(
            'Collateralized', 'False') == 'False' or self.agreement_currency != 'USD':
            return False
        else:
            for x in self.cx.deals['Deals']['Children'][0]['Children']:
                if 'ZAR_COHN' in x['Instrument'].field['Reference']:
                    x['Ignore'] = 'True'
            return True

    def run_calc(self):
        import riskflow as rf

        filename = 'COLLVA_' + self.params['Run_Date'] + '_' + self.cx.deals['Attributes']['Reference'] + '.csv'
        num_deals = len(self.cx.deals['Deals']['Children'][0]['Children'])
        num_sims = 10240
        guess_batch = int(25600 / num_deals + .5)
        batch_size = min(2 ** int(np.log(guess_batch) / np.log(2) + .5), 128)

        self.logger(self.netting_set, 'Netting set has {} deals - initial batch size {}'.format(num_deals, batch_size))

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping COLLVA calc as file already exists')
        else:
            self.params['CollVA'] = {'Gradient': 'Yes'}
            self.params['Simulation_Batches'] = num_sims // batch_size
            self.params['Batch_Size'] = batch_size

            calc_complete = False
            while not calc_complete:
                try:
                    _, out, _ = rf.run_cmc(self.cx, overrides=self.params, CollVA=True)
                except RuntimeError as e:  # Out of memory
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)
                    self.params['Simulation_Batches'] *= 2
                    self.params['Batch_Size'] //= 2
                    self.logger(self.netting_set,
                                'Exception: OOM - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                    time.sleep(10)
                else:
                    calc_complete = True

            grad_collva = out['Results']['grad_collva'].rename(
                columns={'Gradient': self.cx.deals['Attributes']['Reference']})
            # store the CollVA as part of the stats
            out['Stats'].update({'CollVA': out['Results']['collva'], 'Currency': self.params['Currency']})
            grad_collva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
            self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
            # log the netting set
            self.logger(self.netting_set, 'CollVA calc complete')


class SA_CVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(SA_CVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self):
        from riskflow.calculation import construct_calculation

        filename = 'CVA_' + self.params['Run_Date'] + '_' + self.cx.deals['Attributes']['Reference'] + '.csv'

        # load up CVA calc params
        if self.ns.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
            self.params['Simulation_Batches'] = 80
            self.params['Batch_Size'] = 256
        else:
            self.params['Dynamic_Scenario_Dates'] = 'No'
            self.params['Simulation_Batches'] = 40
            self.params['Batch_Size'] = 512
            self.logger(self.netting_set, 'is uncollateralized')

        # get the calculation parameters for CVA
        cva_sect = self.cx.deals['Calculation']['Credit_Valuation_Adjustment']

        # update the params
        self.params['Currency'] = 'ZAR'
        self.params['Deflation_Interest_Rate'] = 'ZAR-SWAP'
        self.params['CVA'] = {'Counterparty': cva_sect['Counterparty'],
                              'Deflate_Stochastically': cva_sect['Deflate_Stochastically'],
                              'Stochastic_Hazard': cva_sect['Stochastic_Hazard_Rates']}

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping CVA gradient calc as file already exists')
            self.params['CVA']['Gradient'] = 'No'
            out = calc.execute(self.params)
            stats = out['Stats']
            # store the CVA as part of the stats
            out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
        else:
            self.params['CVA']['Gradient'] = 'Yes'

            calc_complete = False
            while not calc_complete:
                try:
                    out = calc.execute(self.params)
                except RuntimeError as e:  # Out of memory
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)
                    self.params['Simulation_Batches'] *= 2
                    self.params['Batch_Size'] //= 2
                    self.logger(self.netting_set,
                                'Exception: OOM - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                    time.sleep(1)
                else:
                    calc_complete = True

            grad_cva = out['Results']['grad_cva'].rename(
                columns={'Gradient': self.cx.deals['Attributes']['Reference']})
            # store the CVA as part of the stats
            out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
            # write the grad
            grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
            # store it
            self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


def work(id, lock, queue, results, job, rundate, input_path, outputdir):
    def log(netting_set, msg):
        lock.acquire()
        print('JOB %s: ' % id)
        print('Netting set {0}: {1}'.format(netting_set, msg))
        lock.release()

    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)

    # now load the library 
    import riskflow as rf
    # a context object loads (and caches) one or more configs
    # remap any paths as necessary
    cx = rf.Context(
        path_transform={
            '//ICMJHBMVDROPPRD/AdaptiveAnalytics/Inbound/MarketData': '/mnt/MarketData'
        },
        file_transform={
            'CVAMarketData_Calibrated.dat': 'CVAMarketData_Calibrated_New.json',
            'MarketData.dat': 'MarketData.json'
        })

    # log results
    logs = {}

    while True:
        # get the task
        task = queue.get()
        if task is None:
            break

        obj = globals().get(job)(cx, rundate, input_path, outputdir, task, logs, log)
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
    if 'CSA' in logs:
        csa_file = os.path.join(outputdir, 'CSA', '{0}_{1}_CSA_JOB_{2}.csv'.format(job, rundate, id))
        pd.DataFrame(logs['CSA']).T.to_csv(csa_file)
        result.append(('CSA', csa_file))
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

        post_results = {'Stats': [], 'CSA': []}
        for output in post_processing:
            data = dict(output)
            for k, v in data.items():
                post_results[k].append(pd.read_csv(v, index_col=0))

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
