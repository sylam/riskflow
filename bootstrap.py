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

    cva_path = getpath(['E:\\Data\\crstal\\CVA\\{0}\\{1}',
                        'G:\\Credit Quants\\CRSTAL\\CVA\\{0}\\{1}',
                        'G:\\CVA\\{0}\\{1}'])

    cva_path_uat = getpath(['E:\\Data\\crstal\\CVA\\{0}\\{1}',
                            'G:\\Credit Quants\\CRSTAL\\CVA\\{0}\\{1}',
                            'G:\\CVA\\{0}\\{1}'], uat=True)

    cx = AdaptivContext()
    for rundate in [x for x in sorted(os.listdir(cva_path[:-8])) if x>'2019-11-20'
                    and os.path.isdir(cva_path[:-8]+os.sep+x) ]:
        
        cx.parse_json(cva_path.format(rundate, 'MarketData.json'))
        cx.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)
        cx.bootstrap()
        cx.write_marketdata_json(cva_path.format(rundate, 'MarketDataCal.json'))
        cx.write_market_file(cva_path.format(rundate, 'MarketDataCal.dat'))


