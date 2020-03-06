import os
import sys
import json
import glob
import traceback
import pandas as pd
import numpy as np

from collections import defaultdict
from multiprocessing import Process, Queue, Lock


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return_value = {'.array': obj.tolist()}
        return return_value


def CustomJsonDecoder(dct):
    if '.array' in dct:
        return np.array(dct['.array'])
    return dct


def merge_profiles(id, merge_data, rundate, outputdir):
    pivots = []

    # remove any existing totals
    for prev_totals in glob.glob(outputdir + os.sep + 'Tagged_{}_Total_{}.csv'.format(rundate, id)):
        os.remove(prev_totals)

    pivot_labels = ['Reference', 'Acquirer', 'Portfolio', 'Instrument', 'SalesTeam']
    if 'Factor' in merge_data.values()[0]:
        pivot_labels.insert(1, merge_data.values()[0]['Factor'])

    for name, data in merge_data.items():
        # details for indexing
        profile = pd.read_csv(
            outputdir + os.sep + 'Tagged_{0}_{1}_{2}_{3}.csv'.format(rundate, data['currency'], data['csa'], name))
        position = pd.pivot_table(profile, values=profile.columns[4 + len(data):].tolist(), index=pivot_labels,
                                  aggfunc=np.sum)
        position['Netting'] = name
        position['CSA'] = data['csa']
        position['Currency'] = data['currency']
        pivots.append(position)

    # write it out
    filename = outputdir + os.sep + 'Tagged_{}_Total_{}.csv'.format(rundate, id)
    pd.concat(pivots, axis=0, join='outer').to_csv(filename)
    return ('Merge', filename)


class JOB(object):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log,
                 time_grid='0d 2d 1w(1w) 3m(1m) 2y(3m)'):
        self.cx = cx
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.netting_set = netting_set
        self.stats = stats
        self.logger = log
        # load trade
        self.cx.parse_json(os.path.join(self.input_path, self.rundate, self.netting_set))
        # default cmc parameters - note - fix this to load up all the parameters from cx.deals['Calculation']
        self.params = {'calc_name': ('cmc', 'calc1'),
                       'Time_grid': str(self.cx.deals['Calculation']['Base_Time_Grid']),
                       'Run_Date': self.cx.deals['Calculation']['Base_Date'].strftime('%Y-%m-%d'),
                       'Currency': self.cx.deals['Calculation']['Currency'],
                       'Deflation_Interest_Rate': self.cx.deals['Calculation']['Deflation_Interest_Rate'],
                       'Simulation_Batches': 10,
                       'Batch_Size': 512,
                       'Random_Seed': 5126,
                       'Calc_Scenarios': 'No',
                       'Debug': 'No',
                       'NoModel': 'Constant',
                       'Partition': 'None',
                       'Generate_Slideshow': 'No',
                       'PFE_Recon_File': ''}
        # get the netting set
        self.ns = self.cx.deals['Deals']['Children'][0]['instrument'].field
        # get the agreement currency
        self.agreement_currency = self.ns.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.get('Balance_Currency', self.agreement_currency)

    def perform_calc(self):
        # should be loaded correctly
        from calculation import construct_calculation
        # create the calculation    
        calc = construct_calculation('Credit_Monte_Carlo', self.cx)
        # call the custom calc
        if self.valid():
            self.run_calc(calc)

    def valid(self):
        return False

    def get_filename(self, csa_stat):
        # filename to write the results to
        return 'Tagged_{0}_{1}_{2}_{3}.csv'.format(self.rundate, self.balance_currency, csa_stat, self.netting_set)

    def run_calc(self, calc):
        pass


class PFE(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(PFE, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self, calc):
        deals = defaultdict(int)
        for deal in self.cx.deals['Deals']['Children'][0]['Children']:
            deals[deal['instrument'].field['Object']] += 1

        instruments_in_file = ', '.join(deals)
        # turn on dynamic scenarios (more accurate)
        self.params['Dynamic_Scenario_Dates'] = 'Yes'
        recon_file = os.path.join(self.input_path, self.rundate, self.netting_set[:-4] + 'csv')

        if os.path.isfile(recon_file):
            # load up the file to compare to - csv
            aa_output = pd.read_csv(recon_file, index_col=0)
            # change the currency to match the output from adaptiv
            self.params['Currency'] = aa_output.index.name
            try:
                # run the PFE
                n = calc.execute(self.params)
                exposure = n['Results']['mtm'].clip(0.0, np.inf).astype(np.float64)
                dates = np.array(sorted(calc.time_grid.mtm_dates))[
                    calc.netting_sets.sub_structures[0].obj.Time_dep.deal_time_grid]
                pfe = pd.DataFrame(index=sorted([x.strftime('%Y-%m-%d') for x in dates]),
                                   data={'My_PFE': np.percentile(exposure, 95, axis=1),
                                         'My_EE': exposure.mean(axis=1)})

                result = pd.concat([aa_output, pfe], axis=1)
                self.stats.setdefault('Stats', {})[self.netting_set] = calc.calc_stats
                result['PFE_ABS_ERROR'] = (result['PFE'] - result['My_PFE'])
                result['PFE_REL_ERROR'] = result['PFE_ABS_ERROR'] / result['PFE']
                if ((np.abs(result['PFE_ABS_ERROR']) > 10000) & (np.abs(result['PFE_REL_ERROR']) > 0.1)).any():
                    self.stats['Stats'][self.netting_set]['PFE_ABS_ERROR'] = np.abs(result['PFE_ABS_ERROR']).max()
                    self.stats['Stats'][self.netting_set]['PFE_REL_ERROR'] = np.abs(result['PFE_REL_ERROR']).max()
                if np.abs(result['PFE_REL_ERROR'][0]) > 0.01:
                    self.stats['Stats'][self.netting_set]['Mismatch on Day 0'] = np.abs(result['PFE_REL_ERROR'][0])
                self.stats['Stats'][self.netting_set].update({'instruments': instruments_in_file})
                result.to_csv(self.outputdir + os.sep + 'RECON_' + self.netting_set[:-4] + 'csv')
            except:
                self.logger(self.netting_set, 'Error: PFE did not complete')
        else:
            self.logger(self.netting_set, 'Error: could not find Adaptiv Output')


class CVA_GRAD(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(CVA_GRAD, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self, calc):
        from calculation import construct_calculation

        filename = 'CVA_' + self.params['Run_Date'] + '_' + self.cx.deals['Attributes']['Reference'] + '.csv'

        # load up CVA calc params
        if self.cx.deals['Deals']['Children'][0]['instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
            self.params['Simulation_Batches'] = 20
            self.params['Batch_Size'] = 256
        else:
            self.params['Dynamic_Scenario_Dates'] = 'No'
            self.params['Simulation_Batches'] = 10
            self.params['Batch_Size'] = 512
            self.logger(self.netting_set, 'is uncollateralized')

        # get the calculation parameters for CVA
        cva_sect = self.cx.deals['Calculation']['Credit_Valuation_Adjustment']

        # update the params
        self.params['CVA'] = {'Counterparty': cva_sect['Counterparty'],
                              'Deflate_Stochastically': cva_sect['Deflate_Stochastically'],
                              'Stochastic_Hazard': cva_sect['Stochastic_Hazard_Rates']}

        # work out the next business day
        next_day = self.business_day.rollforward(
            pd.Timestamp(self.params['Run_Date']) + pd.offsets.Day(1)).strftime('%Y-%m-%d')
        
        if os.path.isfile( os.path.join(self.outputdir, 'Greeks', filename) ):
            self.logger(self.netting_set, 'Warning: skipping CVA gradient calc as file already exists')
            self.params['CVA']['Gradient'] = 'No'
            out = calc.execute(self.params)
            stats = out['Stats']
            # store the CVA as part of the stats
            out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
            # now calc theta
            self.params.update({'Run_Date': next_day})
            calc = construct_calculation('Credit_Monte_Carlo', self.cx)
            theta = calc.execute(self.params)
            out['Stats'].update({'CVA_Theta': theta['Results']['cva']})
            # store cva
            self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
        else:
            self.params['CVA']['Gradient'] = 'Yes'
            att = 0
            while att <= 3:
                try:
                    out = calc.execute(self.params)
                    att = 10
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)
                    att += 1
                    self.params['Simulation_Batches'] *= 2
                    self.params['Batch_Size'] //= 2
                    self.logger(self.netting_set, 'Exception: ' + str(e.args))
                    self.logger(self.netting_set, 'Halving memory to {0} and Doubling Batches to {1}'.format(
                        self.params['Batch_Size'], self.params['Simulation_Batches']))

                    # create a new calculation
                    calc = construct_calculation('Credit_Monte_Carlo', self.cx)
                else:
                    stats = out['Stats']
                    grad_cva = calc.gradients_as_df(out['Results']['grad_cva']).rename(
                        columns={'Gradient': self.cx.deals['Attributes']['Reference']})
                    # store the CVA as part of the stats
                    out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
                    # write the grad                    
                    grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    # now calc theta
                    self.params.update({'Run_Date': next_day, 'Gradient': 'No'})
                    calc = construct_calculation('Credit_Monte_Carlo', self.cx)
                    theta = calc.execute(self.params)
                    out['Stats'].update({'CVA_Theta': theta['Results']['cva']})
                    # store it
                    self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class CVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(CVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self, calc):
        from calculation import construct_calculation

        # load up CVA calc params
        if self.cx.deals['Deals']['Children'][0]['instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
            self.params['Simulation_Batches'] = 20
            self.params['Batch_Size'] = 256
        else:
            self.params['Dynamic_Scenario_Dates'] = 'No'
            self.params['Simulation_Batches'] = 10
            self.params['Batch_Size'] = 512
            self.logger(self.netting_set, 'is uncollateralized')

        # get the calculation parameters for CVA
        cva_sect = self.cx.deals['Calculation']['Credit_Valuation_Adjustment']

        # update the params
        self.params['CVA'] = {'Counterparty': cva_sect['Counterparty'],
                              'Deflate_Stochastically': cva_sect['Deflate_Stochastically'],
                              'Stochastic_Hazard': cva_sect['Stochastic_Hazard_Rates'],
                              'Gradient': 'No'}
        att = 0
        while att <= 3:
            try:
                out = calc.execute(self.params)
                att = 10
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                att += 1
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set, 'Exception: ' + str(e.args))
                self.logger(self.netting_set, 'Halving memory to {0} and Doubling Batches to {1}'.format(
                    self.params['Batch_Size'], self.params['Simulation_Batches']))

                # create a new calculation
                calc = construct_calculation('Credit_Monte_Carlo', self.cx)
            else:
                stats = out['Stats']
                # store the CVA as part of the stats
                out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
                # store it
                self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


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

        from adaptiv import AdaptivContext
        self.cx = AdaptivContext()
        old_cx = AdaptivContext()
        # load marketdata
        old_cx.parse_json(os.path.join(self.input_path, rundate, 'MarketData.json'))
        # load up the CVA marketdata file
        self.cx.parse_json(os.path.join(self.input_path, rundate, 'MarketDataCVA.json'))
        # update the cva data with the arena data
        self.cx.params['Price Factors'].update(old_cx.params['Price Factors'])
        # load trade
        self.cx.parse_json(os.path.join(self.input_path, self.rundate, self.netting_set))
        # get the netting set
        self.ns = self.cx.deals['Deals']['Children'][0]['instrument'].field
        # get the agreement currency
        self.agreement_currency = self.ns.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.get('Balance_Currency', self.agreement_currency)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from utils import Curve
        # change the currency
        self.params['Currency'] = self.agreement_currency

        def makeflatcurve(curr, bps, tenor=30):
            return {'Currency': curr, 'Curve': Curve([], [[0, bps * 0.01 * 0.01], [tenor, bps * 0.01 * 0.01]]),
                    'Day_Count': 'ACT_365',
                    'Property_Aliases': None, 'Sub_Type': 'None'}

        if self.agreement_currency == 'ZAR':
            self.cx.params['Price Factors']['InterestRate.ZAR-SWAP.OIS'] = makeflatcurve('ZAR', -15)
            self.cx.params['Price Factors']['InterestRate.ZAR-SWAP.FUNDING'] = makeflatcurve('ZAR', 10)
            self.cx.deals['Deals']['Children'][0]['instrument'].field['Collateral_Assets']['Cash_Collateral'][0][
                'Collateral_Rate'] = 'ZAR-SWAP.OIS'
            self.cx.deals['Deals']['Children'][0]['instrument'].field['Collateral_Assets']['Cash_Collateral'][0][
                'Funding_Rate'] = 'ZAR-SWAP.FUNDING'
        else:
            self.cx.params['Price Factors']['InterestRate.USD-LIBOR-3M.FUNDING'] = makeflatcurve('USD', 65)
            self.cx.deals['Deals']['Children'][0]['instrument'].field['Collateral_Assets']['Cash_Collateral'][0][
                'Collateral_Rate'] = 'USD-OIS'
            self.cx.deals['Deals']['Children'][0]['instrument'].field['Collateral_Assets']['Cash_Collateral'][0][
                'Funding_Rate'] = 'USD-LIBOR-3M.FUNDING'

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children'] or self.cx.deals[
            'Deals']['Children'][0]['instrument'].field.get('Collateralized', 'False') == 'False':
            return False
        else:
            return True

    def run_calc(self, calc):
        filename = 'COLLVA_' + self.params['Run_Date'] + '_' + self.crb_default + '.csv'

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping COLLVA calc as file already exists')
        else:
            self.params['CollVA'] = {'Gradient': 'Yes'}

            # make sure the margin period of risk is 10 business days (approx 12 calendar days)
            self.cx.deals['Deals']['Children'][0]['instrument'].field['Liquidation_Period'] = 12.0
            self.params['Simulation_Batches'] = 20
            self.params['Batch_Size'] = 256

            try:
                out = calc.execute(self.params)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)

                self.logger(self.netting_set, 'Exception: ' + str(e.args))
            else:
                stats = out['Stats']
                grad_collva = calc.gradients_as_df(out['Results']['grad_collva']).rename(
                    columns={'Gradient': self.cx.deals['Attributes']['Reference']})
                # store the CollVA as part of the stats
                out['Stats'].update({'CollVA': out['Results']['collva'], 'Currency': self.params['Currency']})
                grad_collva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
                # log the netting set
                self.logger(self.netting_set, 'CollVA calc complete')


class CVADEFAULT(JOB):
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

        from adaptiv import AdaptivContext
        self.cx = AdaptivContext()
        old_cx = AdaptivContext()
        # load marketdata
        old_cx.parse_json(os.path.join(self.input_path, rundate, 'MarketData.json'))
        # load up the CVA marketdata file
        self.cx.parse_json(os.path.join(self.input_path, rundate, 'MarketDataCVA.json'))
        # update the cva data with the arena data
        self.cx.params['Price Factors'].update(old_cx.params['Price Factors'])
        # load trade
        self.cx.parse_json(os.path.join(self.input_path, self.rundate, self.netting_set))
        # get the netting set
        self.ns = self.cx.deals['Deals']['Children'][0]['instrument'].field
        # get the agreement currency
        self.agreement_currency = self.ns.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.get('Balance_Currency', self.agreement_currency)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from utils import Curve
        # set the survival curve to a default value
        self.crb_default = self.cx.deals['Attributes']['Reference']
        self.cx.params['Price Factors'].setdefault('SurvivalProb.' + self.crb_default,
                                                   {'Recovery_Rate': 0.5, 'Curve': Curve(
                                                       [],
                                                       [[0.0, 0.0], [.5, .01], [1, .02], [3, .07], [5, .15], [10, .35],
                                                        [20, .71]]), 'Property_Aliases': None})

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self, calc):
        filename = 'CVA_' + self.params['Run_Date'] + '_' + self.crb_default + '.csv'

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping CVA calc as file already exists')
        else:
            self.params['CVA'] = {'Deflation': self.cx.deals['Calculation'].get('Deflation_Interest_Rate', 'ZAR-SWAP'),
                                  'Gradient': 'Yes'}

            if self.cx.deals['Deals']['Children'][0]['instrument'].field.get('Collateralized') == 'True':
                self.logger(self.netting_set, 'is collateralized')
                # make sure the margin period of risk is 10 business days (approx 12 calendar days)
                self.cx.deals['Deals']['Children'][0]['instrument'].field['Liquidation_Period'] = 12.0
                self.params['Simulation_Batches'] = 20
                self.params['Batch_Size'] = 256
            else:
                self.params['Simulation_Batches'] = 10
                self.params['Batch_Size'] = 512
                self.logger(self.netting_set, 'is uncollateralized')

            # get the calculation parameters for CVA
            default_cva = {'Deflate_Stochastically': 'Yes', 'Stochastic_Hazard_Rates': 'No',
                           'Counterparty': self.crb_default}
            cva_sect = self.cx.deals.get('Calculation', {'Credit_Valuation_Adjustment': default_cva}).get(
                'Credit_Valuation_Adjustment', default_cva)

            # update the params
            self.params['CVA']['Counterparty'] = cva_sect['Counterparty']
            self.params['CVA']['Deflate_Stochastically'] = cva_sect['Deflate_Stochastically']
            self.params['CVA']['Stochastic_Hazard'] = cva_sect['Stochastic_Hazard_Rates']
            # now adjust the survival curve - need intervals of .25 years
            sc = self.cx.params['Price Factors']['SurvivalProb.' + cva_sect['Counterparty']]['Curve'].array.copy()
            tenors = np.arange(0, sc[-1][0], .25)
            self.cx.params['Price Factors']['SurvivalProb.' + cva_sect['Counterparty']]['Curve'].array = np.array(
                list(zip(tenors, np.interp(tenors, sc[:, 0], sc[:, 1]))))

            try:
                out = calc.execute(self.params)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)

                self.logger(self.netting_set, 'Exception: ' + str(e.args))
            else:
                stats = out['Stats']
                grad_cva = calc.gradients_as_df(out['Results']['grad_cva']).rename(
                    columns={'Gradient': self.cx.deals['Attributes']['Reference']})
                # store the CVA as part of the stats
                out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
                grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
                # log the netting set
                self.logger(self.netting_set, 'CVA calc complete')


class FVADEFAULT(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.netting_set = os.path.splitext(netting_set)[0] + '.json'
        self.stats = stats
        self.logger = log
        self.params = {'calc_name': ('cmc', 'calc1'),
                       'Time_grid': '0d 2d 1w(1w) 1m(1m) 3m(3m) 1y(1y)',
                       'Run_Date': rundate, 'Currency': 'USD', 'Random_Seed': 5126,
                       'Calc_Scenarios': 'No', 'Dynamic_Scenario_Dates': 'Yes',
                       'Debug': 'No', 'NoModel': 'Constant', 'Partition': 'None',
                       'Generate_Slideshow': 'No', 'PFE_Recon_File': ''}

        from adaptiv import AdaptivContext
        self.cx = AdaptivContext()
        old_cx = AdaptivContext()
        # load marketdata
        old_cx.parse_json(os.path.join(self.input_path, rundate, 'MarketData.json'))
        # load up the CVA marketdata file
        self.cx.parse_json(os.path.join(self.input_path, rundate, 'MarketDataCVA.json'))
        # update the cva data with the arena data
        self.cx.params['Price Factors'].update(old_cx.params['Price Factors'])
        # load trade
        self.cx.parse_json(os.path.join(self.input_path, self.rundate, self.netting_set))
        # get the netting set
        self.ns = self.cx.deals['Deals']['Children'][0]['instrument'].field
        # get the agreement currency
        self.agreement_currency = self.ns.get('Agreement_Currency', 'USD')
        # get the balance currency
        self.balance_currency = self.ns.get('Balance_Currency', self.agreement_currency)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from utils import Curve

        def makeflatcurve(curr, bps, tenor=30):
            return {'Currency': curr, 'Curve': Curve([], [[0, bps * 0.01 * 0.01], [tenor, bps * 0.01 * 0.01]]),
                    'Day_Count': 'ACT_365',
                    'Property_Aliases': None, 'Sub_Type': 'None'}

            # set the survival curve to a default value

        self.crb_default = self.cx.deals['Attributes']['Reference']
        # set up funding curves - ??? 
        # self.cx.params['Price Factors']['InterestRate.ZAR-SWAP.OIS'] = makeflatcurve('ZAR',-15)
        # self.cx.params['Price Factors']['InterestRate.ZAR-SWAP.FUNDING'] = makeflatcurve('ZAR', 10)

        self.cx.params['Price Factors']['InterestRate.USD-LIBOR-3M.FUNDING'] = makeflatcurve('USD', 65)

        self.cx.params['Price Factors'].setdefault('SurvivalProb.' + self.crb_default,
                                                   {'Recovery_Rate': 0.5, 'Curve': Curve(
                                                       [],
                                                       [[0.0, 0.0], [.5, .01], [1, .02], [3, .07], [5, .15], [10, .35],
                                                        [20, .71], [30, 1.0]]), 'Property_Aliases': None})

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self, calc):
        filename = 'FVA_' + self.params['Run_Date'] + '_' + self.crb_default + '.csv'

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping FVA calc as file already exists')
        else:
            self.params['FVA'] = {'Funding_Interest_Curve': 'USD-LIBOR-3M.FUNDING',
                                  'Risk_Free_Curve': 'USD-OIS',
                                  'Stochastic_Funding': 'Yes', 'Counterparty': self.crb_default, 'Gradient': 'Yes'}

            if self.cx.deals['Deals']['Children'][0]['instrument'].field.get('Collateralized', 'False') != 'True':
                # only calc FVA for uncollateralized counterparties
                self.params['Simulation_Batches'] = 10
                self.params['Batch_Size'] = 512
                self.logger(self.netting_set, 'is uncollateralized')

                try:
                    out = calc.execute(self.params)
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)
                    self.logger(self.netting_set, 'Exception: ' + str(e.args))
                else:
                    stats = out['Stats']
                    if 'grad_fva' in out['Results']:
                        grad_fva = calc.gradients_as_df(out['Results']['grad_fva']).rename(
                            columns={'Gradient': self.cx.deals['Attributes']['Reference']})
                        grad_fva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    # store the FVA as part of the stats
                    out['Stats'].update({'FVA': out['Results']['fva'], 'Currency': self.params['Currency']})
                    self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
                    # log the netting set
                    self.logger(self.netting_set, 'FVA calc complete')


class CVAVega(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.netting_set = os.path.splitext(netting_set)[0] + '.json'
        self.stats = stats
        self.logger = log
        self.params = {'calc_name': ('cmc', 'calc1'),
                       'Time_grid': '0d 2d 1w(1w) 3m(1m) 2y(3m)', 'Run_Date': rundate,
                       'Currency': 'ZAR', 'Random_Seed': 5126,
                       'Calc_Scenarios': 'No',
                       'Dynamic_Scenario_Dates': 'Yes',
                       'Debug': 'No', 'NoModel': 'Constant',
                       'Partition': 'None', 'Generate_Slideshow': 'No',
                       'PFE_Recon_File': ''}

        from adaptiv import AdaptivContext
        self.cx = AdaptivContext()
        self.cx.parse_json(os.path.join(self.input_path, rundate, 'MarketDataCVA.json'))
        self.noshift = AdaptivContext()
        self.noshift.parse_json(os.path.join(self.input_path, rundate, 'MarketDataNoShift.json'))
        self.shift = AdaptivContext()
        self.shift.parse_json(os.path.join(self.input_path, rundate, 'MarketDataShift.json'))

        # update the cva data with the arena data
        self.cx.params['Price Factors'].update(self.noshift.params['Price Factors'])

        self.cx.parse_json(os.path.join(self.input_path, rundate, self.netting_set))
        # get the netting set
        self.ns = self.cx.deals['Deals']['Children'][0]['instrument'].field
        # get the agreement currency
        self.agreement_currency = self.ns.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.get('Balance_Currency', self.agreement_currency)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from utils import Curve
        # set the survival curve to a default value
        self.crb_default = self.cx.deals['Attributes']['Reference']
        self.cx.params['Price Factors'].setdefault('SurvivalProb.' + self.crb_default,
                                                   {'Recovery_Rate': 0.5, 'Curve': Curve(
                                                       [],
                                                       [[0.0, 0.0], [.5, .01], [1, .02], [3, .07], [5, .15], [10, .35],
                                                        [20, .71]]), 'Property_Aliases': None})

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self, calc):
        from calculation import construct_calculation

        self.params['CVA'] = {'Deflation': self.cx.deals['Calculation'].get('Deflation_Interest_Rate', 'ZAR-SWAP'),
                              'Gradient': 'No'}

        if self.cx.deals['Deals']['Children'][0]['instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            self.params['Simulation_Batches'] = 20
            self.params['Batch_Size'] = 256
        else:
            self.params['Simulation_Batches'] = 10
            self.params['Batch_Size'] = 512
            self.logger(self.netting_set, 'is uncollateralized')

        # get the calculation parameters for CVA
        default_cva = {'Deflate_Stochastically': 'Yes', 'Stochastic_Hazard_Rates': 'No',
                       'Counterparty': self.crb_default}
        cva_sect = self.cx.deals.get('Calculation', {'Credit_Valuation_Adjustment': default_cva}).get(
            'Credit_Valuation_Adjustment', default_cva)

        # update the params
        self.params['CVA']['Counterparty'] = cva_sect['Counterparty']
        self.params['CVA']['Deflate_Stochastically'] = cva_sect['Deflate_Stochastically']
        self.params['CVA']['Stochastic_Hazard'] = cva_sect['Stochastic_Hazard_Rates']

        try:
            out = calc.execute(self.params)
            self.cx.params['Price Factors'].update(self.shift.params['Price Factors'])
            calcshift = construct_calculation('Credit_Monte_Carlo', self.cx)
            outshift = calcshift.execute(self.params)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)

            self.logger(self.netting_set, 'Exception: ' + str(e.args))
        else:
            stats = out['Stats']
            stats.update({'CVA': out['Results']['cva'], 'CVA_Shift': outshift['Results']['cva'],
                          'Currency': self.params['Currency']})
            self.stats.setdefault('Stats', {})[self.netting_set] = stats
            # log the netting set
            self.logger(self.netting_set, 'CVA calc complete NoShift:{}, Shift:{}'.format(
                out['Results']['cva'], outshift['Results']['cva']))


class BaseVal(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        self.cx = cx
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.stats = stats
        self.logger = log
        self.business_day = cx.holidays['Johannesburg']['businessday']
        self.netting_set = os.path.splitext(netting_set)[0] + '.json'
        cx.parse_json(os.path.join(self.input_path, rundate, self.netting_set))
        self.params = {'calc_name': ('cmc', 'calc1'), 'Run_Date': rundate}

    def valid(self):
        if not self.cx.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self, calc):
        # needed for theta
        next_day = self.business_day.rollforward(
            pd.Timestamp(self.params['Run_Date']) + pd.offsets.Day(1)).strftime('%Y-%m-%d')
        ending = self.params['Run_Date'] + '_' + self.cx.deals['Attributes']['Reference'] + '.csv'
        filename = 'BaseVal_' + ending
        filename_theta = 'BaseVal_Theta_' + ending
        filename_greeks = 'BaseVal_Delta_' + ending
        filename_greeks_second = 'BaseVal_Gamma_' + ending

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping BaseVal calc as file already exists')
        else:
            # should be loaded correctly
            from calculation import construct_calculation
            calc = construct_calculation('Base_Revaluation', self.cx, prec=np.float64)
            self.params.update({'Currency': 'ZAR', 'Greeks': 'No'})
            base_val = calc.execute(self.params)
            base_val['Results']['mtm'].to_csv(os.path.join(self.outputdir, 'Greeks', filename))
            try:
                self.params.update({'Run_Date': next_day, 'Greeks': 'Yes'})
                calc = construct_calculation('Base_Revaluation', self.cx, prec=np.float64)
                out = calc.execute(self.params)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
                self.logger(self.netting_set, 'Failed to baseval')
            else:
                self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
                out['Results']['mtm'].to_csv(os.path.join(self.outputdir, 'Greeks', filename_theta))
                out['Results']['Greeks_First'].to_csv(os.path.join(self.outputdir, 'Greeks', filename_greeks))
                out['Results']['Greeks_Second'].to_csv(os.path.join(self.outputdir, 'Greeks', filename_greeks_second))


def work(id, lock, queue, results, job, rundate, input_path, calendar, outputdir):
    def log(netting_set, msg):
        lock.acquire()
        print('JOB %s: ' % id)
        print('Netting set {0}: {1}'.format(netting_set, msg))
        lock.release()

    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)
    # set the log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # now load the cuda context
    from adaptiv import AdaptivContext

    # create the crstal context
    cx = AdaptivContext()
    # load calendars
    cx.parse_calendar_file(calendar)
    # load marketdata
    cx.parse_json(os.path.join(input_path, rundate, 'MarketData.json'))

    if os.path.isfile(os.path.join(input_path, rundate, 'CVAMarketData_Calibrated_New.json')):
        cx_new = AdaptivContext()
        cx_new.parse_json(os.path.join(input_path, rundate, 'CVAMarketData_Calibrated_New.json'))
        log("Parent", "Overriding Calibration")
        for factor in [x for x in cx_new.params['Price Factors'].keys()
                       if x.startswith('HullWhite2FactorModelParameters')]:
            # override it
            cx.params['Price Factors'][factor] = cx_new.params['Price Factors'][factor]

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
    if 'Merge' in logs:
        result.append(merge_profiles(id, logs['Merge'], rundate, outputdir))
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

    def start(self, job, rundate, input_path, calendar, outputdir, wildcard='CrB*.json'):
        print("starting {0} workers in {1}".format(self.NUMBER_OF_PROCESSES, input_path))
        self.workers = [Process(target=work, args=(
            i, self.lock, self.queue, self.results, job, rundate, input_path, calendar, outputdir))
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

        # join to this proces
        for i in range(self.NUMBER_OF_PROCESSES):
            self.workers[i].join()

        post_results = {'Stats': [], 'CSA': [], 'Merge': []}
        for output in post_processing:
            data = dict(output)
            for k, v in data.items():
                post_results[k].append(pd.read_csv(v, index_col=0))

        # write out the combined data
        for k, v in post_results.items():
            if v:
                out_path = os.path.join(outputdir, k, '{0}_{1}_{2}_Total.csv'.format(job, k, rundate))
                pd.concat(v).to_csv(out_path)


if __name__ == '__main__':
    import argparse

    jobs = [cls.__name__ for cls in globals().values() if
            isinstance(cls, type) and hasattr(cls, 'valid') and cls.__name__ != 'JOB']
    # setup the arguments
    parser = argparse.ArgumentParser(description='Run a riskflow batch on a directory of .json netting sets.')
    parser.add_argument('num_jobs', type=int, help='the number of gpu\'s to use (if available) else cpu\'s')
    parser.add_argument('job', type=str, help='the job name', choices=jobs)
    parser.add_argument('rundate', type=str, help='batch rundate')
    parser.add_argument('input_path', type=str,
                        help='directory containing the input files (note that the rundate is assumed to be a directory within)')
    parser.add_argument('calendar_file', type=str, help='calendar file to use')
    parser.add_argument('output_path', type=str, help='output directory')
    parser.add_argument('filename', type=str, help='filename(s) in input_path to run - wildcards allowed')

    # get the arguments
    args = parser.parse_args()

    # print args.rundate, args.input_path, args.calendar_file, args.output_path
    Parent(args.num_jobs).start(args.job, args.rundate, args.input_path, args.calendar_file, args.output_path,
                                args.filename)
