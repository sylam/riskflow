import os
import glob
import shutil
import logging
import pandas as pd

from multiprocessing import Process, Queue, Manager


def work(job_id, queue, result, price_factors,
         price_models, sys_params, holidays):
    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(job_id)
    # set the log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # log to file
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='bootstrap_{}.log'.format(job_id),
                        filemode='w')

    from bootstrappers import construct_bootstrapper

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
                bootstrappers[bootstrapper_name] = construct_bootstrapper(bootstrapper_name, params)
            bootstrapper = bootstrappers[bootstrapper_name]
            bootstrapper.bootstrap(sys_params, price_models, price_factors, job_price, holidays)

        except Exception as e:
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

    def start(self, rundate, input_path, calendar, outfile='CVAMarketDataCal'):
        # disable gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        # set the log level for the parent
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # set the logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M')

        from adaptiv import AdaptivContext

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

        # load the params
        price_factors = self.manager.dict(self.cx.params['Price Factors'])
        price_models = self.manager.dict(self.cx.params['Price Models'])
        sys_params = self.manager.dict(self.cx.params['System Parameters'])
        holidays = self.manager.dict(self.cx.holidays)

        logging.info("starting {0} workers in {1}".format(self.NUMBER_OF_PROCESSES, input_path))
        self.workers = [Process(target=work, args=(
            i, self.queue, self.result, price_factors, price_models, sys_params, holidays)) for i in
                        range(self.NUMBER_OF_PROCESSES)]

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
        for i in range(self.NUMBER_OF_PROCESSES):
            self.workers[i].join()

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bootstrap xVA risk neutral Calibration.')
    parser.add_argument('num_jobs', type=int, help='number of processes to run in parallel')
    parser.add_argument('task', type=str, help='the task name', choices=['Historical', 'Daily'])

    hist = parser.add_argument_group('Historical', 'options for bootstrapping past data')
    hist.add_argument('-i', '--input_path', type=str, help='root directory containing rundates with marketdata')
    hist.add_argument('-s', '--start', type=str, help='start rundate')
    hist.add_argument('-e', '--end', type=str, help='end rundate')

    market = parser.add_argument_group('Daily', 'options for calibration of a single marketdata file')
    market.add_argument('-m', '--market_file', type=str, help='marketdata.json filename and path')
    market.add_argument('-o', '--output_file', type=str, help='output adaptiv filename (uses the path of the '
                                                              'market_file) - do not include the extention .dat')

    # get the arguments
    args = parser.parse_args()
    # parse the files
    if args.task == 'Historical':
        for rundate in [x for x in sorted(os.listdir(args.input_path))
                        if args.start < x < args.end and os.path.isdir(os.path.join(args.input_path, x))]:
            Parent(args.num_jobs).start(rundate, args.input_path, os.path.join(args.input_path, 'calendars.cal'))
    elif args.task == 'Daily':
        calendar = os.path.join(
            os.path.split(args.market_file)[0], 'Calendars.cal')
        Parent(args.num_jobs).start(None, args.market_file, os.path.join(
            os.path.split(args.market_file)[0], 'Calendars.cal'), outfile=args.output_file)
    else:
        logging.error('Invalid Job - aborting')
