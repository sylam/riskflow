"""Walk-forward backtest of the platinum average-price-swap hedge.

Protocol (per user spec): quarterly recalibration, monthly trades from --start.
For each trade date d (first business day of month):
  1. calibrate models on archive data <= quarter start (calibrate_platinum.py, no lookahead),
  2. book a 3m average swap: averaging over the FULL 3rd calendar month after d, paid +5d,
     strike = fair - $8/oz margin (fair = equal-weight mean of F(0,t_i) from the archive
     row at d: S0*exp((c(tau_i)+r(tau_i))*tau_i) over NY business-day fixings),
  3. TRAIN the production policy in the calibrated world (one-step forks, trust region),
  4. REPLAY the frozen policy on the realized archive path (Historical_Replay) with
     B regime-replicas; greedy/textbook/nohedge verdicts all roll the SAME real path,
  5. score net terminal wealth / 2500 oz -> $/oz.

Usage: python backtest_walk_forward.py --start 2020-01 --months 12 [--train-batch 8192]
Writes artifacts/backtest/<stamp>/trades.csv + per-trade configs/diags.
JSON-is-the-contract: load_json + run_job only.
"""
import argparse
import copy
import datetime
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import riskflow as rf

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

ROOT = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')
ARCHIVE = os.path.join(ROOT, 'data', 'plat_archive.csv')
MARKETDATA = os.path.join(ROOT, 'data', 'MarketDataRF_platinum.json')
CAL_CONFIG = os.path.join(ROOT, 'artifacts', 'calibration_config.json')
EXCEL = pd.Timestamp('1899-12-30')
VOLUME = 2500.0
MARGIN = 8.0            # $/oz in the dealer's favour (strike = fair - margin, receiver of float)
CONTRACT_SIZE = 50


def _ts(d):
    return {'.Timestamp': pd.Timestamp(d).strftime('%Y-%m-%d')}


def archive_frame():
    return pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)


def fair_strike(row, trade_date, fixings):
    """Equal-weight mean of F(0,t_i) = S0*exp((c(tau_i)+r(tau_i))*tau_i) over fixing days."""
    s0 = row['CommodityPrice.PLATINUM_LME']
    taus = np.array([row[f'Tenor.PLATINUM_TAU{i}'] for i in (1, 2, 3)])
    carry = np.array([row[f'ForwardRate.PLATINUM_CARRY,PLATINUM_TAU{i}'] for i in (1, 2, 3)])
    sofr = sorted((float(c.split(',')[1]), c) for c in row.index
                  if c.startswith('InterestRate.USD-SOFR'))
    sofr_t = np.array([t for t, _ in sofr])
    sofr_v = np.array([row[c] for _, c in sofr])
    tau_f = np.array([(f - trade_date).days for f in fixings]) / 365.25
    c = np.interp(tau_f, taus, carry)
    r = np.interp(tau_f, sofr_t, sofr_v)
    return float((s0 * np.exp((c + r) * tau_f)).mean())


def build_trade_cfg(template, arch, trade_date, calibrated_md, seed):
    """Per-trade config: fixture reshaped to the trade date, all levels off the archive row."""
    row = arch.loc[:trade_date].iloc[-1]
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    mm = cfg['Calc']['MergeMarketData']
    emd = mm['ExplicitMarketData']
    pf = emd['Price Factors']
    hp = calc['Hedging_Problem']

    calc['Base_Date'] = _ts(trade_date)
    calc['Random_Seed'] = seed
    mm['MarketDataFile'] = calibrated_md

    # --- deal: averaging over the full 3rd calendar month, paid +5d --------------------
    avg_start = (trade_date + pd.offsets.MonthBegin(3)).normalize()
    avg_end = (avg_start + pd.offsets.MonthEnd(0)).normalize()
    pay = avg_end + pd.Timedelta(days=5)
    fixings = pd.bdate_range(avg_start, avg_end)
    k_fair = fair_strike(row, trade_date, fixings)
    deal = hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']
    item = deal['Payments']['Items'][0]
    item.update({'Payment_Date': _ts(pay), 'Period_Start': _ts(avg_start),
                 'Period_End': _ts(avg_end), 'FX_Period_Start': _ts(avg_start),
                 'FX_Period_End': _ts(avg_end), 'Volume': VOLUME,
                 'Realized_Average': 0.0, 'FX_Realized_Average': 0.0,
                 'Fixed_Basis': -(k_fair - MARGIN)})

    # --- tradables: CME strip at the tau ladder + LME twin at tau2 ---------------------
    s0 = row['CommodityPrice.PLATINUM_LME']
    b0 = row['CommodityBasis.LME_CME,PLATINUM_LME']
    taus = [row[f'Tenor.PLATINUM_TAU{i}'] for i in (1, 2, 3)]
    carry = [row[f'ForwardRate.PLATINUM_CARRY,PLATINUM_TAU{i}'] for i in (1, 2, 3)]
    mats = [trade_date + pd.Timedelta(days=round(t * 360)) for t in taus]
    legs = [(f'PL_M{i + 1}', mats[i], 'LME_CME', s0 + b0, carry[i], taus[i]) for i in range(3)]
    legs.append(('PL_LME_M2', mats[1], 'LME_FLAT', s0, carry[1], taus[1]))

    futs, positions, setts, margins, limits = {}, {}, {}, {}, {}
    for name, mat, basis, spot_b, c, tau in legs:
        futs[name] = {'Maturity_Date': _ts(mat), 'Currency': 'USD', 'Carry': 'PLATINUM_CARRY',
                      'Repo_Rate': 'USD-SOFR', 'Implied_Basis': basis,
                      'Contract_Size': CONTRACT_SIZE}
        positions[name] = 0
        setts[name] = round(spot_b * float(np.exp(c * tau)), 4)
        margins[name] = {'Method': 'per_contract', 'Amount': round(0.085 * setts[name] * CONTRACT_SIZE, 0)}
        limits[name] = {'Min_Position': -50, 'Max_Position': 0}
    hp['Tradable_Instruments']['CommodityFutureDeal'] = futs
    ps = hp['Portfolio_State']
    ps['Positions'] = positions
    ps['Settlement_Prices'] = setts
    ps['Initial_Margin'] = margins
    hp['Evaluator']['Position_Limits'] = limits
    cash = hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']
    cash['Investment_Horizon'] = _ts(pay)

    # --- realized history prefix (last 35 business days <= trade date) -----------------
    # STRICTLY before the trade date: including the trade-date spot duplicates sim-day-0
    # in the bundle's time grid, which shifts every [hist:]-indexed read by one day and
    # desynchronizes decisions from accrual (manufactures spurious P&L).
    hist = arch.loc[arch.index < trade_date, 'CommodityPrice.PLATINUM_LME'].iloc[-35:]
    ps['Spot_Price_History'] = {'CommodityPrice.PLATINUM_LME': {
        'Dates': [_ts(d) for d in hist.index], 'Prices': [float(x) for x in hist.values]}}

    # --- price-factor levels off the archive row ----------------------------------------
    pf['CommodityPrice.PLATINUM_LME']['Spot'] = float(s0)
    pf['CommodityBasis.LME_CME']['Spot'] = float(b0)
    pf['CommodityBasis.LME_FLAT'] = {'Spot': 0.0, 'Observed_Commodity': 'PLATINUM_LME'}
    emd.setdefault('Price Models', {})['BasisLinkedSpotModel.LME_FLAT'] = {
        'A': 0.0, 'Phi': 0.0, 'Nu': 5.0, 'Mu': 0.0, 'Sigma_By_State': [0.0, 0.0, 0.0],
        'Calibration_DT_Years': 0.003968253968253968}
    pf['ForwardRate.PLATINUM_CARRY']['Curve']['.Curve']['data'] = [
        [float((m - EXCEL).days), float(c)] for m, c in zip(mats, carry)]
    sofr_cols = sorted(((float(c.split(',')[1]), c) for c in arch.columns
                        if c.startswith('InterestRate.USD-SOFR')))
    pf['InterestRate.USD-SOFR']['Curve']['.Curve']['data'] = [
        [t, float(row[c])] for t, c in sofr_cols]
    # identity fixing curve wide enough for any backtest date
    pf['ReferencePrice.PLATINUM']['Fixing_Curve']['.Curve']['data'] = [
        [40000, 40000], [50000, 50000]]
    return cfg, k_fair


def run_job(cfg, name, run_dir):
    json.dump(cfg, open(os.path.join(run_dir, name + '.json'), 'w'), indent=1, default=str)
    cx = rf.Context()
    cx.load_json((json.dumps(cfg, default=str), name + '.json'))
    _, result = cx.run_job()
    diag = (result.evaluation_summary or {}).get('diagnostics') or {}
    json.dump(diag, open(os.path.join(run_dir, f'diag_{name}.json'), 'w'), indent=1, default=str)
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2020-01')
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--train-batch', type=int, default=4096,
                    help='Outer paths PER training batch (see --batches).')
    ap.add_argument('--batches', type=int, default=2,
                    help='Training stream length: N-1 fit batches then a held-out one.')
    ap.add_argument('--replay-batch', type=int, default=32)
    ap.add_argument('--inner', type=int, default=64)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--levels', type=int, default=9)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--recal-months', type=int, default=3)
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'backtest', f'{stamp}_{args.start}_{args.months}m')
    os.makedirs(run_dir, exist_ok=True)
    template = json.load(open(FIXTURE))
    arch = archive_frame()

    solver_common = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': args.levels,
        'Training_Action_Chunk_Size': 64, 'T_Min': 0, 'DiffV2_Fit_Iters': args.iters,
        'DiffV2_Cost_Aware_Argmax': 'Yes',
        'DiffV2_Per_Column_Grad_Norm': 'Yes',
    }
    rows = []
    calibrated_md = None
    for m in range(args.months):
        trade_date = (pd.Timestamp(args.start + '-01') + pd.offsets.MonthBegin(m)
                      + pd.offsets.BDay(0)).normalize()
        if m % args.recal_months == 0:
            cal_end = trade_date.strftime('%Y-%m-%d')
            calibrated_md = os.path.abspath(os.path.join(run_dir, f'md_{cal_end}.json'))
            logging.info('=== CALIBRATE up to %s ===', cal_end)
            subprocess.run([sys.executable, 'calibrate_platinum.py', '--marketdata', MARKETDATA,
                            '--calibration-config', CAL_CONFIG, '--end', cal_end,
                            '--out', calibrated_md], check=True, cwd=ROOT,
                           stdout=subprocess.DEVNULL)
        tag = trade_date.strftime('%Y%m')
        cfg, k_fair = build_trade_cfg(template, arch, trade_date, calibrated_md, args.seed)
        calc = cfg['Calc']['Calculation']
        hp = calc['Hedging_Problem']

        # --- TRAIN in the calibrated (simulated) world --------------------------------
        ckpt = os.path.abspath(os.path.join(run_dir, f'value_fn_{tag}.pt'))
        # a solve is a stream: --batches - 1 fit batches of --train-batch paths, then a held-out one
        calc.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': args.train_batch,
                     'Simulation_Batches': args.batches,
                     'Inner_Sub_Batch': args.inner, 'Inner_MC_Enabled': 'Yes',
                     'Inner_Antithetic': 'Yes'})
        hp['Randomize_Initial_State'] = 'Yes'
        hp['Solver'] = dict(solver_common, DiffV2_Save_Value_Fn=ckpt)
        logging.info('=== TRAIN %s (fair=%.2f, strike=%.2f) ===', tag, k_fair, k_fair - MARGIN)
        train_diag = run_job(cfg, f'train_{tag}', run_dir)

        # --- ROLL the frozen policy day-by-day on the realized path via the stepper ----
        # Driver prep: dense daily observed factor paths from the base date; the framework's
        # DiffV2_Stepper_Rollout drives _decide off the stepper's own wealth (real accounting).
        obs_base = pd.Timestamp(calc['Base_Date']['.Timestamp'])
        obs_dates = pd.DatetimeIndex([obs_base + pd.Timedelta(days=i) for i in range(220)])
        obs_rows = arch.reindex(arch.index.union(obs_dates)).ffill().loc[obs_dates]
        obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{tag}.npz'))
        np.savez(obs_npz, **{'CommodityPrice.PLATINUM_LME': obs_rows['CommodityPrice.PLATINUM_LME'].to_numpy(),
                             'CommodityBasis.LME_CME': obs_rows['CommodityBasis.LME_CME,PLATINUM_LME'].to_numpy()})
        cfg_r, _ = build_trade_cfg(template, arch, trade_date, calibrated_md, args.seed)
        calc_r = cfg_r['Calc']['Calculation']
        # the roll is a frozen eval: a stream of one, all of it unseen
        calc_r.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': 1,
                       'Simulation_Batches': 1,
                       'Inner_Sub_Batch': args.inner, 'Inner_MC_Enabled': 'Yes',
                       'Inner_Antithetic': 'Yes', 'Observed_Scenario': obs_npz})
        hp_r = calc_r['Hedging_Problem']
        hp_r['Randomize_Initial_State'] = 'No'
        hp_r['Solver'] = dict(solver_common, DiffV2_Load_Value_Fn=[ckpt],
                              DiffV2_Stepper_Rollout='Yes')
        logging.info('=== ROLL %s (stepper, realized path) ===', tag)
        diag = run_job(cfg_r, f'roll_{tag}', run_dir)
        v = diag.get('stepper_verdict') or {}
        rec = {'trade': tag, 'fair': round(k_fair, 2), 'strike': round(k_fair - MARGIN, 2),
               'train_u': ((train_diag.get('verdict') or {}).get('greedy') or {}).get('u_mean')}
        for pol in ('greedy', 'textbook', 'nohedge'):
            s = v.get(pol) or {}
            rec[f'{pol}_usd_oz'] = (round(s['wT_mean'] / VOLUME, 2)
                                    if s.get('wT_mean') is not None else None)
        rec['greedy_usd_oz_p5'] = (round((v.get('greedy') or {}).get('wT_p5', 0) / VOLUME, 2)
                                   if v.get('greedy') else None)
        rows.append(rec)
        logging.info('TRADE %s: greedy=%s tb=%s nh=%s $/oz', tag, rec['greedy_usd_oz'],
                     rec['textbook_usd_oz'], rec['nohedge_usd_oz'])
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, 'trades.csv'), index=False)

    df = pd.DataFrame(rows)
    print('\n===== WALK-FORWARD BACKTEST ($/oz) =====')
    print(df.to_string(index=False))
    print(f"\nmean greedy: {df['greedy_usd_oz'].mean():+.2f} $/oz | "
          f"textbook: {df['textbook_usd_oz'].mean():+.2f} | "
          f"nohedge: {df['nohedge_usd_oz'].mean():+.2f} | margin charged: {MARGIN:.2f}")
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
