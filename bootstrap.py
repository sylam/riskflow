import os
import itertools
import pandas as pd

from multiprocessing import Process, Queue, Lock, Manager


def getpath(pathlist, uat=False):
    for path in pathlist:
        if os.path.isdir(path):
            return os.path.join(path, 'UAT') if uat else path


def work(job_id, lock, queue, errors,
         price_factors, price_models, sys_params, holidays):

    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(job_id)
    # set the log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from bootstrappers import construct_bootstrapper

    # perform the bootstrapping
    while True:
        task = queue.get()
        if task is None:
            break
        else:
            bootstrapper_name, params, job_prices = task

        try:
            bootstrapper = construct_bootstrapper(bootstrapper_name, params)
            bootstrapper.bootstrap(sys_params, price_models, price_factors, job_prices, holidays)
        except Exception as e:
            error = 'Cannot execute Bootstrapper for {0} - {1}'.format(bootstrapper_name, e.args)
            errors.put(error)
        else:
            # run the bootstrapper on the market prices and store them in the price factors/price models
            errors.put('{}: {} Ok'.format(bootstrapper_name, job_id))


class Parent(object):
    def __init__(self, num_jobs):
        self.queue = Queue()
        self.errors = Queue()
        self.lock = Lock()
        self.manager = Manager()
        self.NUMBER_OF_PROCESSES = num_jobs
        self.cx = None

    def start(self, rundate, input_path, calendar):
        # disable gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        # set the log level for the parent
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from adaptiv import AdaptivContext

        # create the context
        self.cx = AdaptivContext()
        # load calendars
        self.cx.parse_calendar_file(calendar)
        # load marketdata
        if os.path.isfile(os.path.join(input_path, rundate, 'MarketDataCal.json')):
            self.cx.parse_json(os.path.join(input_path, rundate, 'MarketDataCal.json'))
        else:
            self.cx.parse_json(os.path.join(input_path, rundate, 'MarketData.json'))

        # load the rundate
        self.cx.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)

        # load the params
        price_factors = self.manager.dict(self.cx.params['Price Factors'])
        price_models = self.manager.dict(self.cx.params['Price Models'])
        sys_params = self.manager.dict(self.cx.params['System Parameters'])
        holidays = self.manager.dict(self.cx.holidays)

        print("starting {0} workers in {1}".format(self.NUMBER_OF_PROCESSES, input_path))
        self.workers = [Process(target=work, args=(
            i, self.lock, self.queue, self.errors, price_factors,
            price_models, sys_params, holidays)) for i in range(self.NUMBER_OF_PROCESSES)]

        for w in self.workers:
            w.start()

        # load the bootstrapper on to the queue - note - order is important here - hence python 3.6
        for bootstrapper_name, params in self.cx.params['Bootstrapper Configuration'].items():
            # get the market price id and any options for bootstrapping
            market_price, _, *options = params.split(',', 2)
            # get the market prices for this bootstrapper
            market_prices = {k: v for k, v in self.cx.params['Market Prices'].items() if k.startswith(market_price)}
            # number of return statuses needed
            status_required = 0
            for job_num in range(self.NUMBER_OF_PROCESSES):
                job_prices = dict(itertools.compress(
                    market_prices.items(),
                    [i % self.NUMBER_OF_PROCESSES == job_num for i in range(len(market_prices))]))

                # only put data on the queue if there's work to do
                if job_prices:
                    self.queue.put((bootstrapper_name, options, job_prices))
                    status_required += 1

            # fetch and print the results
            for i in range(status_required):
                child_status = self.errors.get()
                self.lock.acquire()
                print('Parent: {}'.format(child_status))
                self.lock.release()

        for i in range(self.NUMBER_OF_PROCESSES):
            # tell the children it's over
            self.queue.put(None)

        # finish up
        self.stop(rundate, input_path)

    def stop(self, rundate, input_path):
        # join the children to this process
        for i in range(self.NUMBER_OF_PROCESSES):
            self.workers[i].join()

        # write out the data
        self.lock.acquire()
        print('Parent: All done - saving data')
        self.lock.release()

        self.cx.write_marketdata_json(os.path.join(input_path, rundate, 'MarketDataCal.json'))
        self.cx.write_market_file(os.path.join(input_path, rundate, 'MarketDataCal.dat'))


if __name__ == '__main__':
    # import matplotlib

    # matplotlib.use('Qt4Agg')
    # import matplotlib.pyplot as plt
    # from adaptiv import AdaptivContext

    # plt.interactive(True)
    # make pandas pretty print
    # pd.options.display.float_format = '{:,.5f}'.format

    cva_path = getpath(['E:\\Data\\crstal\\CVA',
                        'G:\\Credit Quants\\CRSTAL\\CVA',
                        'G:\\CVA'])

    cva_path_uat = getpath(['E:\\Data\\crstal\\CVA',
                            'G:\\Credit Quants\\CRSTAL\\CVA',
                            'G:\\CVA'], uat=True)

    # cx = AdaptivContext()

    for rundate in [x for x in sorted(os.listdir(cva_path)) \
                    if x > '2020-01-05' and os.path.isdir(os.path.join(cva_path, x)) and x < '2020-01-18']:
        Parent(4).start(rundate, cva_path, os.path.join(cva_path, 'calendars.cal'))

        # if os.path.isfile(cva_path.format(rundate, 'MarketDataCal.json')):
        #     cx.parse_json(cva_path.format(rundate, 'MarketDataCal.json'))
        # else:
        #     cx.parse_json(cva_path.format(rundate, 'MarketData.json'))

        # cx.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)
        # cx.bootstrap()
        # cx.write_marketdata_json(cva_path.format(rundate, 'MarketDataCal.json'))
        # cx.write_market_file(cva_path.format(rundate, 'MarketDataCal.dat'))
