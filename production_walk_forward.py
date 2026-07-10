"""Production walk-forward backtest of the platinum average-price-swap hedge.

For each monthly trade date over [--start, --start + --months):
  1. CALIBRATE the stochastic models on archive data up to the trade date (no lookahead) via
     calibrate_platinum.py -> a per-date calibrated MarketDataRF (recalibrated every
     --recal-months).
  2. STRIKE a 3-month average-price swap at the fair forward minus a margin ($/oz in the
     dealer's favour), with all market levels read off the archive row at the trade date.
  3. TRAIN the day-1 hedge policy in that calibrated (simulated) world — production_solver's
     validated best config — and save the frozen value function.
  4. ROLL the frozen policy day-by-day along the REALIZED archive path via the framework's
     stepper (DiffV2_Stepper_Rollout: real futures accounting — variation margin, financing,
     per-instrument expiry, forced-flat; decisions made causally off the stepper's own wealth).
  5. RECORD terminal P&L in $/oz for greedy / textbook / no-hedge and the fair strike.
Aggregate across all trades and print the table.

Point it at any (archive CSV, calibration config, source MarketDataRF, deal template). The deal
specification lives in build_deal_config() — adapt it for a different product/hedge set; the
solver and the rollout are product-agnostic. Repeatable: each run writes a fresh artifacts dir.

Usage:
    python production_walk_forward.py \
        --archive data/plat_archive.csv \
        --calibration-config artifacts/calibration_config.json \
        --marketdata data/MarketDataRF_platinum.json \
        --deal-template tests/fixtures/policy_test_simulate_only.json \
        --start 2021-01 --months 24 --recal-months 3 --margin 8 --batch 8192 --seed 7

CRITICAL correctness notes baked in:
  * Spot_Price_History is STRICTLY BEFORE the trade date — including the trade-date spot
    duplicates sim-day-0 in the bundle time grid and desynchronizes decisions from accrual.
  * The realized path enters via a driver-built Observed_Scenario npz (spot + basis); the
    framework prices the tradables + liability on it. Carry/repo curves are set to the
    trade-date levels (they barely move over a 3-month window).
  * The liability is priced in the Components forward-curve world (ForwardCurve=Components),
    so its carry risk is hedgeable by the same ForwardRate the futures reference.

JSON-is-the-contract: import riskflow, load_json, run_job. No internal imports, no monkey-patching.
"""
import argparse
import copy
import csv
import datetime
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from production_solver import apply_config, run

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCEL = pd.Timestamp('1899-12-30')
CONTRACT_SIZE = 50
CARRY_TENORS = ('PLATINUM_TAU1', 'PLATINUM_TAU2', 'PLATINUM_TAU3')

# --- archive column names (adapt for a different data source) ---
SPOT_COL = 'CommodityPrice.PLATINUM_LME'
BASIS_COL = 'CommodityBasis.LME_CME,PLATINUM_LME'
CARRY_COL = 'ForwardRate.PLATINUM_CARRY'          # + ',<tenor>'
SOFR_PREFIX = 'InterestRate.USD-SOFR'


def _ts(d):
    return {'.Timestamp': pd.Timestamp(d).strftime('%Y-%m-%d')}


def fair_strike(row, trade_date, fixings):
    """Fair forward strike = equal-weight mean over the averaging fixings of
    F(0, t_i) = S0 * exp((carry(tau_i) + repo(tau_i)) * tau_i)."""
    s0 = row[SPOT_COL]
    taus = np.array([row[f'Tenor.{t}'] for t in CARRY_TENORS])
    carry = np.array([row[f'{CARRY_COL},{t}'] for t in CARRY_TENORS])
    sofr = sorted((float(c.split(',')[1]), c) for c in row.index if c.startswith(SOFR_PREFIX))
    sofr_t = np.array([t for t, _ in sofr])
    sofr_v = np.array([row[c] for _, c in sofr])
    tau_f = np.array([(f - trade_date).days for f in fixings]) / 365.25
    c = np.interp(tau_f, taus, carry)
    r = np.interp(tau_f, sofr_t, sofr_v)
    return float((s0 * np.exp((c + r) * tau_f)).mean())


def build_deal_config(template, arch, trade_date, calibrated_md, margin, volume):
    """Reshape the deal template to the trade date, reading every market level off the archive
    row. Returns (cfg, fair_strike). Adapt this function for a different product/hedge set.

    Deal: a 3-month average-price swap (Receiver), averaged over the full 3rd calendar month
    after the trade date, paid +5 days, struck at fair - margin. Hedges: the CME futures strip
    at the 3 carry tenors plus an LME-native future at the mid tenor (basis-free identity leg).
    """
    row = arch.loc[:trade_date].iloc[-1]
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    mm = cfg['Calc']['MergeMarketData']
    emd = mm['ExplicitMarketData']
    pf = emd['Price Factors']
    hp = calc['Hedging_Problem']

    calc['Base_Date'] = _ts(trade_date)
    mm['MarketDataFile'] = calibrated_md

    # --- liability: average-price swap struck at fair - margin -----------------------------
    avg_start = (trade_date + pd.offsets.MonthBegin(3)).normalize()
    avg_end = (avg_start + pd.offsets.MonthEnd(0)).normalize()
    pay = avg_end + pd.Timedelta(days=5)
    fixings = pd.bdate_range(avg_start, avg_end)
    k_fair = fair_strike(row, trade_date, fixings)
    item = next(iter(hp['Liabilities']['FloatingEnergyDeal'].values()))['Payments']['Items'][0]
    item.update({'Payment_Date': _ts(pay), 'Period_Start': _ts(avg_start), 'Period_End': _ts(avg_end),
                 'FX_Period_Start': _ts(avg_start), 'FX_Period_End': _ts(avg_end), 'Volume': volume,
                 'Realized_Average': 0.0, 'FX_Realized_Average': 0.0,
                 'Fixed_Basis': -(k_fair - margin)})

    # --- tradables: CME strip at the tenor ladder + LME twin at the mid tenor --------------
    s0 = float(row[SPOT_COL])
    b0 = float(row[BASIS_COL])
    taus = [float(row[f'Tenor.{t}']) for t in CARRY_TENORS]
    carry = [float(row[f'{CARRY_COL},{t}']) for t in CARRY_TENORS]
    mats = [trade_date + pd.Timedelta(days=round(t * 360)) for t in taus]
    legs = [(f'PL_M{i + 1}', mats[i], 'LME_CME', s0 + b0, carry[i], taus[i]) for i in range(3)]
    legs.append(('PL_LME_M2', mats[1], 'LME_FLAT', s0, carry[1], taus[1]))

    futs, positions, setts, margins, limits = {}, {}, {}, {}, {}
    for name, mat, basis, spot_b, c, tau in legs:
        futs[name] = {'Maturity_Date': _ts(mat), 'Currency': 'USD', 'Carry': 'PLATINUM_CARRY',
                      'Repo_Rate': 'USD-SOFR', 'Implied_Basis': basis, 'Contract_Size': CONTRACT_SIZE}
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
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']['Investment_Horizon'] = _ts(pay)

    # --- realized spot history: STRICTLY before the trade date (see module docstring) ------
    hist = arch.loc[arch.index < trade_date, SPOT_COL].iloc[-35:]
    ps['Spot_Price_History'] = {SPOT_COL: {
        'Dates': [_ts(d) for d in hist.index], 'Prices': [float(x) for x in hist.values]}}

    # --- market-factor levels off the archive row ------------------------------------------
    pf[SPOT_COL]['Spot'] = s0
    pf['CommodityBasis.LME_CME']['Spot'] = b0
    pf['CommodityBasis.LME_FLAT'] = {'Spot': 0.0, 'Observed_Commodity': 'PLATINUM_LME'}
    emd.setdefault('Price Models', {})['BasisLinkedSpotModel.LME_FLAT'] = {
        'A': 0.0, 'Phi': 0.0, 'Nu': 5.0, 'Mu': 0.0, 'Sigma_By_State': [0.0, 0.0, 0.0],
        'Calibration_DT_Years': 0.003968253968253968}
    pf['ForwardRate.PLATINUM_CARRY']['Curve']['.Curve']['data'] = [
        [float((m - EXCEL).days), c] for m, c in zip(mats, carry)]
    sofr_cols = sorted((float(c.split(',')[1]), c) for c in arch.columns if c.startswith(SOFR_PREFIX))
    pf['InterestRate.USD-SOFR']['Curve']['.Curve']['data'] = [[t, float(row[c])] for t, c in sofr_cols]
    pf['ReferencePrice.PLATINUM']['Fixing_Curve']['.Curve']['data'] = [[40000, 40000], [50000, 50000]]
    return cfg, k_fair


def observed_scenario_npz(arch, base_date, path):
    """Dense daily realized (spot, basis) from the base date, forward-filled, written to an npz
    the calc's Observed_Scenario reads. Keys are the full factor names the seam matches on."""
    base = pd.Timestamp(base_date)
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=i) for i in range(220)])
    rows = arch.reindex(arch.index.union(dates)).ffill().loc[dates]
    np.savez(path, **{'CommodityPrice.PLATINUM_LME': rows[SPOT_COL].to_numpy(),
                      'CommodityBasis.LME_CME': rows[BASIS_COL].to_numpy()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True, help='Historical archive CSV (dates x factor columns).')
    ap.add_argument('--calibration-config', required=True, help='Calibration config JSON.')
    ap.add_argument('--marketdata', required=True, help='Source MarketDataRF (Model Configuration).')
    ap.add_argument('--deal-template', required=True, help='Deal/hedge template JSON (Components world).')
    ap.add_argument('--start', default='2021-01', help='First trade month, YYYY-MM.')
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--recal-months', type=int, default=3, help='Recalibrate every N months.')
    ap.add_argument('--margin', type=float, default=8.0, help='$/oz dealer margin (strike = fair - margin).')
    ap.add_argument('--volume', type=float, default=2500.0, help='Swap volume (oz).')
    ap.add_argument('--batch', type=int, default=8192, help='Training outer paths (scale up on a big GPU).')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    arch = pd.read_csv(args.archive, index_col=0, parse_dates=True)
    template = json.load(open(args.deal_template))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'walk_forward', f'{stamp}_{args.start}_{args.months}m')
    os.makedirs(run_dir, exist_ok=True)
    logging.info('run dir: %s', run_dir)

    rows, calibrated_md = [], None
    for m in range(args.months):
        trade_date = (pd.Timestamp(args.start + '-01') + pd.offsets.MonthBegin(m)
                      + pd.offsets.BDay(0)).normalize()
        tag = trade_date.strftime('%Y%m')

        # 1. calibrate up to the trade date (no lookahead), recal every N months
        if m % args.recal_months == 0:
            calibrated_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}.json'))
            logging.info('=== CALIBRATE through %s ===', trade_date.date())
            subprocess.run([sys.executable, 'calibrate_platinum.py', '--marketdata', args.marketdata,
                            '--calibration-config', args.calibration_config,
                            '--end', trade_date.strftime('%Y-%m-%d'), '--out', calibrated_md],
                           check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

        # 2+3. build the trade config, train the policy on the calibrated world
        cfg, k_fair = build_deal_config(template, arch, trade_date, calibrated_md, args.margin, args.volume)
        ckpt = os.path.abspath(os.path.join(run_dir, f'value_fn_{tag}.pt'))
        train = apply_config(copy.deepcopy(cfg), batch=args.batch, seed=args.seed, save=ckpt)
        logging.info('=== TRAIN %s (fair=%.2f, strike=%.2f) ===', tag, k_fair, k_fair - args.margin)
        run(train, f'train_{tag}')

        # 4. roll the frozen policy day-by-day on the realized path (stepper accounting)
        obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{tag}.npz'))
        observed_scenario_npz(arch, trade_date, obs_npz)
        roll_cfg = apply_config(copy.deepcopy(cfg), batch=1, seed=args.seed, load=[ckpt],
                                stepper_rollout=True, randomize_initial_state=False)
        roll_cfg['Calc']['Calculation']['Observed_Scenario'] = obs_npz
        logging.info('=== ROLL %s (stepper, realized path) ===', tag)
        diag = run(roll_cfg, f'roll_{tag}')
        json.dump(diag, open(os.path.join(run_dir, f'diag_{tag}.json'), 'w'), indent=1, default=str)

        # 5. record $/oz
        v = diag.get('stepper_verdict') or {}
        rec = {'trade': tag, 'fair': round(k_fair, 2), 'strike': round(k_fair - args.margin, 2)}
        for pol in ('greedy', 'textbook', 'nohedge'):
            s = v.get(pol) or {}
            rec[f'{pol}_usd_oz'] = round(s['wT_mean'] / args.volume, 2) if s.get('wT_mean') is not None else None
        rows.append(rec)
        logging.info('TRADE %s: greedy=%s nohedge=%s $/oz', tag, rec['greedy_usd_oz'], rec['nohedge_usd_oz'])
        with open(os.path.join(run_dir, 'trades.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    df = pd.DataFrame(rows)
    print('\n===== WALK-FORWARD BACKTEST ($/oz) =====')
    print(df.to_string(index=False))
    print(f"\nmean  greedy: {df['greedy_usd_oz'].mean():+.2f}  |  nohedge: {df['nohedge_usd_oz'].mean():+.2f}  "
          f"|  margin: {args.margin:+.2f}  ($/oz)")
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
