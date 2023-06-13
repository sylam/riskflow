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
import glob
import time
import shutil
import logging
import tempfile
import traceback
import pandas as pd

from multiprocessing import Process, Queue, Manager


def work(job_id, queue, result, price_factors, price_factor_interp,
         price_models, sys_params, holidays):
    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(job_id)
    # set the log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # log to file
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        force=True,
                        filename='bootstrap_{}.log'.format(job_id),
                        filemode='w')

    from riskflow.bootstrappers import construct_bootstrapper

    bootstrappers = {}

    # perform the bootstrapping
    while True:
        task = queue.get()
        if task is not None:
            bootstrapper_name, params, job_price = task
        else:
            queue.put(None)
            break
        try:
            name = list(job_price.keys())[0]
            if bootstrapper_name not in bootstrappers:
                bootstrappers[bootstrapper_name] = construct_bootstrapper(
                    bootstrapper_name, params)
            bootstrapper = bootstrappers[bootstrapper_name]
            bootstrapper.bootstrap(
                sys_params, price_models, price_factors, price_factor_interp[0], job_price, holidays)

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
            result.put('Cannot execute Bootstrapper for {0} - {1}'.format(name, e.args))
        else:
            result.put('{} - Job {} Ok'.format(name, job_id))


class Parent(object):
    def __init__(self, num_jobs):
        self.queue = Queue()
        self.result = Queue()
        self.manager = Manager()
        self.NUMBER_OF_PROCESSES = num_jobs
        self.path = None
        self.cx = None
        self.ref = None
        self.daily = False

    def start(self, rundate, input_path, calendar, outfile='CVAMarketDataCal', premium_file=None, delta=0):
        # disable gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        # set the logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M')

        from riskflow.adaptiv import AdaptivContext

        # create the context
        self.cx = AdaptivContext()
        # load calendars
        self.cx.parse_calendar_file(calendar)
        # store the path
        self.path = input_path
        # load marketdata
        if rundate is None:
            self.daily = True
            self.path = os.path.split(input_path)[0]
            self.outfile = outfile
            self.cx.parse_json(input_path)
            # load up the old file if present
            old_output_name = os.path.join(self.path, outfile + '.json')
            if os.path.isfile(old_output_name):
                self.ref = AdaptivContext()
                self.ref.parse_json(old_output_name)
                params_to_bootstrap = self.cx.params['Bootstrapper Configuration'].keys()
                for factor in [x for x in self.ref.params['Price Factors'].keys()
                               if x.split('.', 1)[0] in params_to_bootstrap]:
                    # override it
                    self.cx.params['Price Factors'][factor] = self.ref.params['Price Factors'][factor]
            rundate = pd.Timestamp.now().strftime('%Y-%m-%d')
        elif os.path.isfile(os.path.join(self.path, rundate, 'MarketDataCal.json')):
            self.cx.parse_json(os.path.join(self.path, rundate, 'MarketDataCal.json'))
        elif os.path.isfile(os.path.join(self.path, rundate, 'MarketData.json')):
            self.cx.parse_json(os.path.join(self.path, rundate, 'MarketData.json'))
        else:
            logging.error('Cannot find market data for rundate {}'.format(rundate))
            return

        # update the rundate if necessary
        if self.cx.params['System Parameters']['Base_Date'] is None:
            logging.info('Setting  rundate {}'.format(rundate))
            self.cx.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)

        if premium_file is not None:
            waited = 0
            base = ''
            while True:
                logging.info('Watching for premium file in {} for rundate {}'.format(premium_file, rundate))
                prem = glob.glob(os.path.join(premium_file, base, 'IR_Volatility_Swaption_{}*.csv'.format(rundate)))
                if prem:
                    if prem[0].endswith('_RFR.csv'):
                        rfr, libor = prem[0], prem[1]
                    else:
                        rfr, libor = prem[1], prem[0]

                    # load up the files
                    rfr_premiums = pd.read_csv(rfr, index_col=0)
                    libor_premiums = pd.read_csv(libor, index_col=0)

                    with tempfile.NamedTemporaryFile(delete=False) as fp:
                        valid_rfr = rfr_premiums[
                            rfr_premiums['Currency'].isin(['AUD', 'CHF', 'EUR', 'GBP', 'JPY', 'USD'])]
                        valid_libor = libor_premiums[libor_premiums['Currency'].isin(['CAD', 'ZAR'])]
                        pd.concat([valid_rfr, valid_libor]).to_csv(fp)
                        logging.info('Merging {} and {} as {}'.format(rfr, libor, fp.name))
                        prem[0] = fp.name
                    break
                else:
                    if waited > 60:
                        logging.info('Waited an hour for Premium file in {} - exiting'.format(premium_file))
                        sys.exit(1)
                    else:
                        time.sleep(60)
                        waited += 1
                        base = '..' if base == '' else ''

            logging.info('Setting swaption premiums from {}'.format(prem[0]))
            self.cx.params['System Parameters']['Swaption_Premiums'] = prem[0]

        if delta:
            logging.info('Setting implied vol delta to {}'.format(delta))
            self.cx.params['System Parameters']['Volatility_Delta'] = delta / 100.0

        # load up master curves
        self.cx.params['System Parameters']['Master_Curves'] = {
            'AUD': 'AUD-AONIA',
            'CAD': 'CAD-MASTER',
            'CHF': 'CHF-OIS',
            'EUR': 'EUR-ESTR',
            'GBP': 'GBP-SONIA',
            'JPY': 'JPY-TONAR',
            'USD': 'USD-SOFR',
            'ZAR': 'ZAR-SWAP'
        }

        # load the params
        price_factors = self.manager.dict(self.cx.params['Price Factors'])
        price_factor_interp = self.manager.list([self.cx.params['Price Factor Interpolation']])
        price_models = self.manager.dict(self.cx.params['Price Models'])
        sys_params = self.manager.dict(self.cx.params['System Parameters'])
        holidays = self.manager.dict(self.cx.holidays)

        logging.info("starting {0} workers in {1}".format(self.NUMBER_OF_PROCESSES, input_path))
        self.workers = [Process(target=work, args=(
            i, self.queue, self.result, price_factors, price_factor_interp,
            price_models, sys_params, holidays)) for i in range(self.NUMBER_OF_PROCESSES)]

        for w in self.workers:
            w.start()

        # load the bootstrapper on to the queue - note - order is important here - hence python 3.6
        for bootstrapper_name, params in self.cx.params['Bootstrapper Configuration'].items():
            # get the market price id and any options for bootstrapping
            market_price, _, *options = params.split(',', 2)
            # get the market prices for this bootstrapper
            market_prices = {k: v for k, v in self.cx.params['Market Prices'].items() if
                             k.startswith(market_price)}
            # number of return statuses needed
            status_required = 0
            for market_price in market_prices.keys():
                status_required += 1
                self.queue.put((bootstrapper_name, options, {market_price: market_prices[market_price]}))

            for i in range(status_required):
                logging.info(self.result.get())

        # tell the children it's over
        self.queue.put(None)
        # store the results back in the parent context
        self.cx.params['Price Factors'] = price_factors.copy()
        self.cx.params['Price Models'] = price_models.copy()
        # finish up
        # close the queues
        self.queue.close()
        self.result.close()

        # join the children to this process
        for w in self.workers:
            w.join()

        # write out the data
        logging.info('Parent: All done - saving data')

        if self.daily:
            # write out the calibrated data
            self.cx.write_marketdata_json(os.path.join(self.path, self.outfile + '.json'))
            self.cx.write_market_file(os.path.join(self.path, self.outfile + '.dat'))
            logfilename = os.path.join(self.path, self.outfile + '.log')
        else:
            self.cx.write_marketdata_json(os.path.join(self.path, rundate, 'MarketDataCal.json'))
            self.cx.write_market_file(os.path.join(self.path, rundate, 'MarketDataCal.dat'))
            logfilename = os.path.join(self.path, rundate, 'MarketDataCal.log')

        # copy the logs across
        with open(logfilename, 'wb') as wfd:
            for f in glob.glob('bootstrap*.log'):
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Bootstrap xVA risk neutral Calibration.')
    parser.add_argument('num_jobs', type=int, help='number of processes to run in parallel')
    parser.add_argument('task', type=str, help='the task name', choices=['Historical', 'Daily', 'CopyHW'])

    hist = parser.add_argument_group('Historical', 'options for bootstrapping past data')
    hist.add_argument('-i', '--input_path', type=str, help='root directory containing rundates with marketdata')
    hist.add_argument('-s', '--start', type=str, help='start rundate')
    hist.add_argument('-e', '--end', type=str, help='end rundate')

    market = parser.add_argument_group('Daily', 'options for calibration of a single marketdata file')
    market.add_argument('-m', '--market_file', type=str, help='marketdata.json filename and path')
    market.add_argument('-p', '--premium_file', type=str, help='swaption premium csv filename and path', default=None)
    market.add_argument('-d', '--delta', type=int,
                        help='amount to add (in percentage) to implied swaption vol (default 0)', default=0)
    market.add_argument('-o', '--output_file', type=str, help='output adaptiv filename (uses the path of the '
                                                              'market_file) - do not include the extention .dat')
    parser.add_argument_group('CopyHW', 'options for copying the HW2 factor model to non RF curves')

    # get the arguments
    args = parser.parse_args()
    # parse the files
    if args.task == 'Historical':
        for rundate in [x for x in sorted(os.listdir(args.input_path))
                        if args.start < x < args.end and os.path.isdir(os.path.join(args.input_path, x))]:
            Parent(args.num_jobs).start(rundate, args.input_path, os.path.join(args.input_path, 'calendars.cal'))
    elif args.task == 'Daily':
        calendar = os.path.join(
            os.path.split(args.market_file)[0], 'calendars.cal')
        Parent(args.num_jobs).start(
            None, args.market_file, calendar, outfile=args.output_file, premium_file=args.premium_file,
            delta=args.delta)
    elif args.task == 'CopyHW':
        import numpy as np
        import riskflow.utils as utils
        from riskflow.riskfactors import construct_factor
        from riskflow.adaptiv import AdaptivContext
        from riskflow.bootstrappers import master_curve_list
        # load the context
        context = AdaptivContext()
        context.parse_json(args.market_file)
        # get the hw2factor params
        hw2params = {c: context.params['Price Factors']['HullWhite2FactorModelParameters.{}'.format(x)]
                     for c, x in master_curve_list.items()}
        # get all ir_base curve names
        ir_curves = {curve_name: construct_factor(
            utils.Factor('InterestRate', (curve_name,)), context.params['Price Factors'],
            context.params['Price Factor Interpolation']) for curve_name in np.unique(
            [x.split('.')[1] for x in context.params['Price Factors'].keys() if x.startswith('InterestRate.')])}

        params_to_create = {}
        for ir_curve_name, ir_curve in ir_curves.items():
            ccy = ir_curve.get_currency()[0]
            if ccy in master_curve_list:
                params_to_create['HullWhite2FactorModelParameters.{}'.format(ir_curve_name)] = hw2params[ccy]

        # delete old parameters          
        for k in list(context.params['Price Factors'].keys()):
            if k.startswith('HullWhite2FactorModelParameters.'):
                del context.params['Price Factors'][k]
        # write out the new ones        
        context.params['Price Factors'].update(params_to_create)
        # write out the data
        context.write_marketdata_json(args.market_file)
        # remove any jacobians before we write out the .dat
        jacobians = [i for i in context.params['Price Factors'].keys() if
                     i.startswith('HullWhite2FactorModelParametersJacobian')]
        for i in jacobians:
            del context.params['Price Factors'][i]
        # remove the Swaption_Premiums param if it exists
        if 'Swaption_Premiums' in context.params['System Parameters']:
            del context.params['System Parameters']['Swaption_Premiums']
        if 'Volatility_Delta' in context.params['System Parameters']:
            del context.params['System Parameters']['Volatility_Delta']
        if 'Master_Curves' in context.params['System Parameters']:
            del context.params['System Parameters']['Master_Curves']
        context.write_market_file(args.market_file.replace('.json', '.dat'))
    else:
        logging.error('Invalid Job - aborting')


if __name__ == '__main__':
    sys.exit(main())
