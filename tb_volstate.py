"""Trade-date vol state as an EXTERNAL observable + LOO'd vol-conditioned band rule.

The GARCH md is only recalibrated quarterly (21 distinct GARCH-CALIB dates across the 48 trades),
so the logged H0 is stale for 2 of every 3 months. Here the state is made per-month and ex-ante:
take (omega, alpha, beta, H0) from the most recent calibration at or BEFORE the trade date and
filter h forward with realized CME log-returns up to the last close STRICTLY before the trade
date. Also computes model-free trailing realized vols. Nothing here uses post-trade information.
"""
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from production_walk_forward import build_corrected_archive, CME_COL

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = f'{ROOT}/artifacts/walk_forward'
PAT = re.compile(
    r'GARCH-CALIB (\d{4}-\d{2}-\d{2}): omega=(\S+) alpha=(\S+) beta=(\S+) nu=(\S+) H0=(\S+) '
    r'LR-vol=(\S+) \| basis sigma=(\S+) phi=(\S+) kappa=(\S+) \|')


def calib_table():
    rows = {}
    for g in (0, 1):
        for line in open(f'{WF}/garch48_gpu{g}.log'):
            m = PAT.search(line)
            if m:
                d = pd.Timestamp(m.group(1))
                rows[d] = dict(date=d, omega=float(m.group(2)), alpha=float(m.group(3)),
                               beta=float(m.group(4)), nu=float(m.group(5)), H0=float(m.group(6)),
                               lr_vol=float(m.group(7)), basis_sigma=float(m.group(8)),
                               phi=float(m.group(9)), kappa=float(m.group(10)))
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r['date'])).set_index('date')


def trade_date_of(m):
    return (pd.Timestamp(f'{m[:4]}-{m[4:]}-01') + pd.offsets.BDay(0)).normalize()


def month_list(start, n):
    return [(pd.Timestamp(start + '-01') + pd.offsets.MonthBegin(m)).strftime('%Y%m') for m in range(n)]


def build():
    raw = pd.read_csv(f'{ROOT}/data/pl_exp.csv', index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    P = arch[CME_COL].dropna()
    lr = np.log(P).diff().dropna()
    cal = calib_table()
    out = []
    for m in month_list('2020-01', 48):
        td = trade_date_of(m)
        prior = cal.index[cal.index <= td]
        assert len(prior), m
        c = cal.loc[prior[-1]]
        # filter h forward from the calibration date to the last close strictly before the trade date
        win = lr[(lr.index > c.name) & (lr.index < td)]
        h = c.H0
        for r in win.to_numpy():
            h = c.omega + c.alpha * r * r + c.beta * h
        lr_var = c.omega / max(1.0 - c.alpha - c.beta, 1e-12)
        hist = lr[lr.index < td]
        out.append(dict(
            tag=m, trade_date=td.date(), calib_date=prior[-1].date(),
            stale_days=int((td - prior[-1]).days),
            h_trade=h, sigma_trade=float(np.sqrt(h * 252)),
            h_ratio=float(h / lr_var), lr_vol=float(c.lr_vol),
            H0_logged=c.H0, h_ratio_logged=float(c.H0 / lr_var),
            rv20=float(hist[-20:].std(ddof=1) * np.sqrt(252)),
            rv60=float(hist[-60:].std(ddof=1) * np.sqrt(252)),
            rv120=float(hist[-120:].std(ddof=1) * np.sqrt(252)),
            basis_sigma=c.basis_sigma, nu=c.nu))
    df = pd.DataFrame(out)
    df.to_csv(f'{ROOT}/tb_volstate.csv', index=False)
    return df


if __name__ == '__main__':
    df = build()
    print(df.to_string(index=False))
    print('\ndistinct calib dates used:', df.calib_date.nunique(), ' stale_days: mean %.0f max %d'
          % (df.stale_days.mean(), df.stale_days.max()))
