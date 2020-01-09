import os
import pandas as pd


def getpath(pathlist, uat=False):
    for path in pathlist:
        if os.path.isdir(path[:-8]):
            return (path[:-4:] + os.sep + 'UAT' + path[-4:]) if uat else path


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt
    from adaptiv import AdaptivContext

    plt.interactive(True)
    # make pandas pretty print
    pd.options.display.float_format = '{:,.5f}'.format

    if 'cx' not in locals():
        rundate = '2019-12-31'
        # rundate = '2019-11-13'

        cva_path = getpath(['E:\\Data\\crstal\\CVA\\{0}\\{1}',
                            'G:\\Credit Quants\\CRSTAL\\CVA\\{0}\\{1}',
                            'G:\\CVA\\{0}\\{1}'])

        cva_path_uat = getpath(['E:\\Data\\crstal\\CVA\\{0}\\{1}',
                                'G:\\Credit Quants\\CRSTAL\\CVA\\{0}\\{1}',
                                'G:\\CVA\\{0}\\{1}'], uat=True)

        pfe_path = getpath(['E:\\Data\\crstal\\Arena\\{0}\\{1}',
                            'G:\\Credit Quants\\CRSTAL\\Arena\\{0}\\{1}',
                            'G:\\Arena\\{0}\\{1}'])

        cx = AdaptivContext()

        if os.path.isfile(cva_path.format(rundate, 'MarketDataCal.json')):
            cx.parse_json(cva_path.format(rundate, 'MarketDataCal.json'))
        else:
            cx.parse_json(cva_path.format(rundate, 'MarketData.json'))

        cx.parse_calendar_file(pfe_path.format('calendars.cal', '')[:-1])
        cx.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)

    if 1:
        # cx.params['Bootstrapper Configuration'] = {'PCAMixedFactorModelParameters': 'Test'}
        cx.bootstrap()
        cx.write_marketdata_json(cva_path.format(rundate, 'MarketDataCal2.json'))
        cx.write_market_file(cva_path.format(rundate, 'MarketDataCal2.dat'))


