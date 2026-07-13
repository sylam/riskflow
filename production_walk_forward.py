"""Production walk-forward backtest of the platinum average-price-swap hedge — corrected world.

For each monthly trade date over [--start, --start + --months):
  1. CALIBRATE the stochastic models on the corrected archive up to the trade date (no
     lookahead) via calibrate_platinum.py, recalibrated every --recal-months. Logs the standing
     E[dF|b]≈0 martingale guard (dP~b slope ≈ 0; dS~b carries the LBMA catch-up).
  2. STRIKE a 3-month average-price swap on the pure LBMA fixing at fair - margin ($/oz in the
     dealer's favour), all market levels read off the corrected archive row at the trade date.
  3. TRAIN the day-1 hedge policy in that calibrated (simulated) world — production_solver's
     validated best config — and save the frozen value function.
  4. ROLL the frozen policy day-by-day along the REALIZED archive path via the framework's
     stepper (DiffV2_Stepper_Rollout: real futures accounting — variation margin, financing,
     per-instrument expiry, forced-flat; decisions made causally off the stepper's own wealth).
  5. RECORD greedy/no-hedge terminal P&L in $/oz plus the portfolio causal bound and PASS/FAIL.
Aggregate across all trades and print the table (mean/min/max, positives, bound-PASS count).

CORRECTED (composed-spot) architecture, all validated:
  * CommodityPrice.PLATINUM_CME = P is the martingale primary (MarkovHMMSpotModel).
  * CommodityBasis.LME_CME (BasisLinkedSpotModel, Linked_Commodity=PLATINUM_CME) carries the
    published basis b = P - S — the LBMA catch-up.
  * CommodityPrice.PLATINUM_LME = P - b is the observable LBMA fixing (BasisComposedSpotModel;
    routed via the source MarketDataRF modelfilters, never calibrated — it carries no params).
  * The swap references pure PLATINUM_LME; the CME futures reference PLATINUM_LME +
    Implied_Basis=LME_CME (= P, martingale) so E[dF|b] ≈ 0 (unexecutable reversion not harvested).

The raw futures file (--archive, default data/pl_exp.csv) is decomposed internally into the
corrected series by build_corrected_archive() (CME-implied continuous spot P, published basis,
clean calendar-spread carries, SOFR). The deal spec lives in build_deal_config(); the solver and
the rollout are product-agnostic. Repeatable: each run writes a fresh artifacts dir.

Usage:
    python production_walk_forward.py \
        --archive data/pl_exp.csv \
        --calibration-config artifacts/calibration_config.json \
        --marketdata data/MarketDataRF_platinum.json \
        --deal-template tests/fixtures/policy_test_simulate_only.json \
        --start 2021-01 --months 24 --recal-months 3 --margin 8 --batch 8192 --seed 7

CRITICAL correctness notes baked in:
  * Spot_Price_History is STRICTLY BEFORE the trade date — including the trade-date spot
    duplicates sim-day-0 in the bundle time grid and desynchronizes decisions from accrual.
  * The realized path enters via a driver-built Observed_Scenario npz (CME primary + published
    basis); the framework composes the LBMA fixing and prices the tradables + liability on it.
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

# --- corrected archive column names ---
CME_COL = 'CommodityPrice.PLATINUM_CME'          # martingale primary P
LME_COL = 'CommodityPrice.PLATINUM_LME'          # LBMA fixing S = P - b
BASIS_COL = 'CommodityBasis.LME_CME,PLATINUM_CME'  # published basis b = P - S (CME - LBMA)
CARRY_COL = 'ForwardRate.PLATINUM_CARRY'          # + ',<tenor>'
SOFR_PREFIX = 'InterestRate.USD-SOFR'

# validated objective (matches artifacts/platinum_hedge_shipping.json)
OBJECTIVE = {'Object': 'AsymmetricUtility_Huber', 'Huber_Aversion': 2.5, 'Huber_Delta': 1.0,
             'Floor_Penalty': 10.0, 'Surplus_Reward': 1.0, 'Power': 1.0,
             'Expiry_Penalty': 1.0, 'Expiry_Threshold_Days': 4.0,
             'Post_Deal_Trade_Penalty': 1.0, 'Position_Bounds_Penalty': 0.25,
             'Per_Instrument_Bounds_Penalty': 0.5}


def _ts(d):
    return {'.Timestamp': pd.Timestamp(d).strftime('%Y-%m-%d')}


def build_corrected_archive(raw):
    """Decompose the raw futures file (PL1/PL2/PL3 + taus + repo + LBMA spot + SOFR) into the
    corrected series. Returns a DataFrame with the martingale primary P (CME-implied continuous
    spot), the published basis b = P - S, clean calendar-spread carry knots, tenors, the SOFR
    curve, and the LBMA fixing S. (The calibration CSV drops S — it is a composed factor with no
    archive series; it routes to BasisComposedSpotModel and is never calibrated.)"""
    F1, F2, F3 = raw['PL1'].astype(float), raw['PL2'].astype(float), raw['PL3'].astype(float)
    t1, t2, t3 = raw['PL1_tau'].astype(float), raw['PL2_tau'].astype(float), raw['PL3_tau'].astype(float)
    r1, r2, r3 = raw['PL1_rf'].astype(float), raw['PL2_rf'].astype(float), raw['PL3_rf'].astype(float)
    S = raw['CommodityPrice.PLATINUM'].astype(float)
    fwd12 = (r2 * t2 - r1 * t1) / (t2 - t1)                # forward SOFR between contract tenors
    fwd23 = (r3 * t3 - r2 * t2) / (t3 - t2)
    c12 = np.log(F2 / F1) / (t2 - t1) - fwd12              # calendar-spread carry F1->F2
    c23 = np.log(F3 / F2) / (t3 - t2) - fwd23              # F2->F3
    c1 = c12                                               # front: flat-extrapolate c12 to spot
    c2 = (c1 * t1 + c12 * (t2 - t1)) / t2                  # cumulative carry knots c(tau_i)
    c3 = (c2 * t2 + c23 * (t3 - t2)) / t3
    P = F1 * np.exp(-(r1 + c1) * t1)                       # CME-implied continuous spot (primary)
    out = pd.DataFrame(index=raw.index)
    out[CME_COL] = P
    out[BASIS_COL] = P - S                                 # published sign (CME - LBMA)
    for i, (c, t) in enumerate(zip((c1, c2, c3), (t1, t2, t3)), 1):
        out[f'{CARRY_COL},PLATINUM_TAU{i}'] = c
        out[f'Tenor.PLATINUM_TAU{i}'] = t
    for c in raw.columns:
        if c.startswith(SOFR_PREFIX):
            out[c] = raw[c].astype(float)
    out[LME_COL] = S
    out.index.name = 'Date'
    return out


def guard_e_df_b(arch, cal_end):
    """Standing E[dF|b]≈0 guard on data ≤ cal_end. The tradeable's synthetic spot is P (CME,
    martingale), so dP(t+1) regressed on b(t) must have slope ≈ 0; dS(t+1)~b(t) shows the
    catch-up lives in the LBMA leg. Returns {'dP~b': (slope, t), 'dS~b': (slope, t)} with b in
    the natural S-P sign (dS~b ≈ -(1-phi) < 0)."""
    sub = arch.loc[:cal_end]
    b = sub[LME_COL] - sub[CME_COL]                        # natural sign S - P
    out = {}
    for nm, col in (('dP~b', CME_COL), ('dS~b', LME_COL)):
        y = sub[col].diff().shift(-1)
        d = pd.DataFrame({'y': y, 'x': b}).dropna()
        X = np.column_stack([np.ones(len(d)), d['x'].values])
        coef, *_ = np.linalg.lstsq(X, d['y'].values, rcond=None)
        resid = d['y'].values - X @ coef
        se = np.sqrt((resid ** 2).sum() / (len(d) - 2) / ((d['x'] - d['x'].mean()) ** 2).sum())
        out[nm] = (float(coef[1]), float(coef[1] / se))
    return out


def fair_strike(row, trade_date, fixings):
    """Fair forward strike = equal-weight mean over the averaging fixings of
    F(0, t_i) = S0 * exp((carry(tau_i) + repo(tau_i)) * tau_i), S0 the LBMA fixing."""
    s0 = row[LME_COL]
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
    """Reshape the deal template to the trade date in the corrected world, reading every market
    level off the corrected archive row. Returns (cfg, info) where info carries the strike and
    the dates the causal bound / observed path need. Adapt this for a different product.

    Deal: a 3-month average-price swap (Receiver) on the pure LBMA fixing, averaged over the full
    3rd calendar month after the trade date, paid +5 days, struck at fair - margin. Hedges: the
    CME futures strip at the 3 carry tenors, each referencing PLATINUM_LME + Implied_Basis=LME_CME
    (= P, the martingale primary).
    """
    row = arch.loc[:trade_date].iloc[-1]
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    mm = cfg['Calc']['MergeMarketData']
    emd = mm['ExplicitMarketData']
    hp = calc['Hedging_Problem']

    calc['Base_Date'] = _ts(trade_date)
    mm['MarketDataFile'] = calibrated_md
    hp['Objective'] = dict(OBJECTIVE)

    s0 = float(row[LME_COL])
    p0 = float(row[CME_COL])
    b0 = float(row[BASIS_COL])            # published P - S
    taus = [float(row[f'Tenor.{t}']) for t in CARRY_TENORS]
    carry = [float(row[f'{CARRY_COL},{t}']) for t in CARRY_TENORS]
    mats = [trade_date + pd.Timedelta(days=round(t * 360)) for t in taus]

    # --- liability: average-price swap on pure PLATINUM_LME, struck at fair - margin ----------
    avg_start = (trade_date + pd.offsets.MonthBegin(3)).normalize()
    avg_end = (avg_start + pd.offsets.MonthEnd(0)).normalize()
    pay = avg_end + pd.Timedelta(days=5)
    fixings = pd.bdate_range(avg_start, avg_end)
    k_fair = fair_strike(row, trade_date, fixings)
    deal = next(iter(hp['Liabilities']['FloatingEnergyDeal'].values()))
    deal['Commodity'] = 'PLATINUM_LME'
    deal['Reference_Type'] = 'PLATINUM'
    deal['Payments']['Items'][0].update({
        'Payment_Date': _ts(pay), 'Period_Start': _ts(avg_start), 'Period_End': _ts(avg_end),
        'FX_Period_Start': _ts(avg_start), 'FX_Period_End': _ts(avg_end), 'Volume': volume,
        'Realized_Average': 0.0, 'FX_Realized_Average': 0.0, 'Fixed_Basis': -(k_fair - margin)})

    # --- tradables: CME strip at the tenor ladder (LBMA + LME_CME basis = P) -------------------
    futs, positions, setts, margins, limits = {}, {}, {}, {}, {}
    for i, (mat, c, tau) in enumerate(zip(mats, carry, taus), 1):
        name = f'PL_M{i}'
        futs[name] = {'Maturity_Date': _ts(mat), 'Currency': 'USD', 'Carry': 'PLATINUM_CARRY',
                      'Repo_Rate': 'USD-SOFR', 'Implied_Basis': 'LME_CME', 'Contract_Size': CONTRACT_SIZE}
        positions[name] = 0
        setts[name] = round(p0 * float(np.exp(c * tau)), 4)
        margins[name] = {'Method': 'per_contract', 'Amount': round(0.085 * setts[name] * CONTRACT_SIZE, 0)}
        limits[name] = {'Min_Position': -50, 'Max_Position': 0}
    hp['Tradable_Instruments']['CommodityFutureDeal'] = futs
    ps = hp['Portfolio_State']
    ps['Positions'] = positions
    ps['Settlement_Prices'] = setts
    ps['Initial_Margin'] = margins
    hp['Evaluator']['Position_Limits'] = limits
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']['Investment_Horizon'] = _ts(pay)

    # --- realized spot history (both legs), STRICTLY before the trade date --------------------
    hist = arch.loc[arch.index < trade_date].iloc[-35:]
    ps['Spot_Price_History'] = {c: {'Dates': [_ts(d) for d in hist.index],
                                    'Prices': [float(x) for x in hist[c]]}
                                for c in (LME_COL, CME_COL)}

    # --- corrected minimal Price Factors off the archive row ---------------------------------
    emd['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': '', 'Interest_Rate': 'USD-SOFR', 'Spot': 1.0},
        LME_COL: {'Currency': 'USD', 'Interest_Rate': 'USD-SOFR', 'Forward_Rate': 'PLATINUM_CARRY',
                  'Spot': s0, 'Implied_Basis': 'LME_CME', 'Property_Aliases': ''},
        CME_COL: {'Currency': 'USD', 'Interest_Rate': 'USD-SOFR', 'Forward_Rate': 'PLATINUM_CARRY',
                  'Spot': p0, 'Property_Aliases': ''},
        'CommodityBasis.LME_CME': {'Spot': b0, 'Observed_Commodity': 'PLATINUM_LME',
                                   'Linked_Commodity': 'PLATINUM_CME'},
        'ReferencePrice.PLATINUM': {'Fixing_Curve': {'.Curve': {'meta': [], 'data': [[40000, 40000], [50000, 50000]]}},
                                    'ForwardPrice': None, 'Property_Aliases': ''},
        'ForwardRate.PLATINUM_CARRY': {'Currency': 'USD', 'Curve': {'.Curve': {'meta': [], 'data': [
            [float((m - EXCEL).days), c] for m, c in zip(mats, carry)]}}},
        'InterestRate.USD-SOFR': {'Day_Count': 'ACT_365', 'Currency': 'USD', 'Sub_Type': None,
                                  'Curve': {'.Curve': {'meta': [], 'data': [
                                      [t, float(row[c])] for t, c in
                                      sorted((float(x.split(',')[1]), x) for x in arch.columns if x.startswith(SOFR_PREFIX))]}}},
        'ForwardPriceSample.USD': {'Offset': 0, 'Holiday_Calendar': 'New York',
                                   'Sampling_Convention': 'ForwardPriceSampleDaily'},
    }
    emd.setdefault('Price Models', {})['BasisComposedSpotModel.PLATINUM_LME'] = {}
    emd['Valuation Configuration'] = {'FloatingEnergyDeal': {'ForwardCurve': 'Components'}}

    info = {'k_fair': k_fair, 'mats': mats, 'pay': pay}
    return cfg, info


def observed_scenario_npz(arch, base_date, path):
    """Dense daily realized (CME primary, published basis) from the base date, forward-filled,
    written to an npz the calc's Observed_Scenario reads. The framework composes the LBMA fixing
    S = P - b internally. Keys are the full factor names the seam matches on."""
    base = pd.Timestamp(base_date)
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=i) for i in range(220)])
    rows = arch.reindex(arch.index.union(dates)).ffill().loc[dates]
    np.savez(path, **{CME_COL: rows[CME_COL].to_numpy(),
                      'CommodityBasis.LME_CME': rows[BASIS_COL].to_numpy()})


def pf_bound(arch, trade_date, mats, pay):
    """Portfolio causal bound, Σ_t max_leg max(0, -dF_obs), on the reconstructed observed CME
    forward strip ($/oz). A hedge can only lose (vs no-hedge) by the sum over days of the worst
    adverse forward move it could be caught in; greedy P&L must stay ≤ nohedge + this bound."""
    bdays = pd.bdate_range(trade_date, pay)
    sub = arch.reindex(arch.index.union(bdays)).ffill().loc[bdays]
    sofr = sorted((float(c.split(',')[1]), c) for c in arch.columns if c.startswith(SOFR_PREFIX))
    F_legs = []
    for mat in mats:
        tau_t = np.array([(mat - d).days for d in bdays]) / 365.25
        live = tau_t > 0
        ct = np.array([np.interp(tt, [sub[f'Tenor.PLATINUM_TAU{j}'].iloc[k] for j in (1, 2, 3)],
                                 [sub[f'{CARRY_COL},PLATINUM_TAU{j}'].iloc[k] for j in (1, 2, 3)])
                       for k, tt in enumerate(tau_t)])
        rt = np.array([np.interp(tt, [t for t, _ in sofr], [sub[c].iloc[k] for _, c in sofr])
                       for k, tt in enumerate(tau_t)])
        F = sub[CME_COL].to_numpy() * np.exp((ct + rt) * np.clip(tau_t, 0, None))
        F_legs.append(np.where(live, F, np.nan))
    dF = np.diff(np.array(F_legs), axis=1)
    dg = np.nanmax(np.where(np.isnan(dF), -np.inf, np.maximum(0.0, -dF)), axis=0)
    return float(np.where(np.isfinite(dg), dg, 0.0).sum())


def calibrate(marketdata, cal_config, cal_end, out_md):
    """Calibrate the corrected models on the archive up to cal_end (no lookahead)."""
    logging.info('=== CALIBRATE through %s ===', cal_end)
    subprocess.run([sys.executable, 'calibrate_platinum.py', '--marketdata', marketdata,
                    '--calibration-config', cal_config, '--end', cal_end, '--out', out_md],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag):
    """Train the day-1 policy and roll it on the realized path; return the recorded row."""
    cfg, info = build_deal_config(template, arch, trade_date, calibrated_md, args.margin, args.volume)
    ckpt = os.path.abspath(os.path.join(run_dir, f'value_fn_{tag}.pt'))

    train = apply_config(copy.deepcopy(cfg), batch=args.batch, seed=args.seed, save=ckpt)
    if args.fit_iters is not None:
        train['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Fit_Iters'] = args.fit_iters
    logging.info('=== TRAIN %s (fair=%.2f, strike=%.2f) ===', tag, info['k_fair'], info['k_fair'] - args.margin)
    tdiag = run(train, f'train_{tag}')

    obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{tag}.npz'))
    observed_scenario_npz(arch, trade_date, obs_npz)
    roll = apply_config(copy.deepcopy(cfg), batch=1, seed=args.seed, load=[ckpt],
                        stepper_rollout=True, randomize_initial_state=False)
    roll['Calc']['Calculation']['Observed_Scenario'] = obs_npz
    logging.info('=== ROLL %s (stepper, realized path) ===', tag)
    rdiag = run(roll, f'roll_{tag}')
    json.dump(rdiag, open(os.path.join(run_dir, f'diag_{tag}.json'), 'w'), indent=1, default=str)

    sv = rdiag.get('stepper_verdict') or {}
    gr = (sv.get('greedy') or {}).get('wT_mean')
    nh = (sv.get('nohedge') or {}).get('wT_mean')
    bound = pf_bound(arch, trade_date, info['mats'], info['pay'])
    tv = (tdiag.get('verdict') or {}).get('greedy') or {}
    q = np.array(sv.get('greedy_q_traj') or [[0.0]])
    return {
        'trade': tag, 'fair': round(info['k_fair'], 2), 'strike': round(info['k_fair'] - args.margin, 2),
        'train_u': None if tv.get('u_mean') is None else round(tv['u_mean'], 4),
        'V_0': tdiag.get('V_0'),
        'greedy_usd_oz': None if gr is None else round(gr / args.volume, 2),
        'nohedge_usd_oz': None if nh is None else round(nh / args.volume, 2),
        'pf_bound': round(bound, 2),
        'bound_pass': (None if (gr is None or nh is None)
                       else bool(gr / args.volume <= nh / args.volume + bound + 1e-6)),
        'churn': round(float(np.abs(np.diff(q, axis=0)).sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', default='data/pl_exp.csv', help='Raw futures archive CSV (decomposed internally).')
    ap.add_argument('--calibration-config', default='artifacts/calibration_config.json', help='Calibration config JSON.')
    ap.add_argument('--marketdata', default='data/MarketDataRF_platinum.json', help='Source MarketDataRF (Model Configuration).')
    ap.add_argument('--deal-template', default='tests/fixtures/policy_test_simulate_only.json',
                    help='Deal/hedge template JSON (Components world).')
    ap.add_argument('--start', default='2021-01', help='First trade month, YYYY-MM.')
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--recal-months', type=int, default=3, help='Recalibrate every N months.')
    ap.add_argument('--margin', type=float, default=8.0, help='$/oz dealer margin (strike = fair - margin).')
    ap.add_argument('--volume', type=float, default=2500.0, help='Swap volume (oz).')
    ap.add_argument('--batch', type=int, default=8192, help='Training outer paths (scale up on a big GPU).')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--fit-iters', type=int, default=None, help='Override DiffV2_Fit_Iters (smoke tests).')
    args = ap.parse_args()

    raw = pd.read_csv(args.archive, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    template = json.load(open(args.deal_template))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'walk_forward', f'{stamp}_{args.start}_{args.months}m')
    os.makedirs(run_dir, exist_ok=True)
    logging.info('run dir: %s', run_dir)

    # corrected calibration inputs (written once): archive CSV (drop the composed LBMA fixing),
    # source MarketDataRF with the PLATINUM_LME -> BasisComposedSpotModel routing, calibration
    # config pointing at that archive.
    arch_csv = os.path.abspath(os.path.join(run_dir, 'archive_cme.csv'))
    arch.drop(columns=[LME_COL]).to_csv(arch_csv)
    md_src = json.load(open(args.marketdata))
    md_src['MarketData']['Model Configuration']['.ModelParams']['modelfilters'] = {
        'CommodityPrice': [[['ID', 'PLATINUM_LME'], 'BasisComposedSpotModel']]}
    md_cal = os.path.abspath(os.path.join(run_dir, 'marketdata_corrected.json'))
    json.dump(md_src, open(md_cal, 'w'), indent=1)
    cal_src = json.load(open(args.calibration_config))
    cal_src['CalibrationConfig']['MarketDataArchiveFile']['name'] = arch_csv
    cal_cfg = os.path.abspath(os.path.join(run_dir, 'calibration_config.json'))
    json.dump(cal_src, open(cal_cfg, 'w'), indent=1)

    rows, calibrated_md = [], None
    for m in range(args.months):
        trade_date = (pd.Timestamp(args.start + '-01') + pd.offsets.MonthBegin(m) + pd.offsets.BDay(0)).normalize()
        tag = trade_date.strftime('%Y%m')

        if m % args.recal_months == 0:
            calibrated_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}.json'))
            cal_end = trade_date.strftime('%Y-%m-%d')
            calibrate(md_cal, cal_cfg, cal_end, calibrated_md)
            g = guard_e_df_b(arch, cal_end)
            logging.info('GUARD %s: dP~b slope=%+.4f t=%+.2f (martingale) | dS~b slope=%+.4f t=%+.2f (catch-up)',
                         cal_end, g['dP~b'][0], g['dP~b'][1], g['dS~b'][0], g['dS~b'][1])

        try:
            rec = one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag)
        except ValueError as e:
            # Stale quarterly calibration: a VAR carry front slot can expire between cal-date and
            # trade-date (tau -> 0), tripping the float32 X_0 round-trip. Still no lookahead:
            # recalibrate AT the trade date and retry.
            logging.warning('TRADE %s: stale-md failure (%s); FALLBACK fresh calibration', tag, e)
            fresh_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}_fresh.json'))
            calibrate(md_cal, cal_cfg, trade_date.strftime('%Y-%m-%d'), fresh_md)
            rec = one_trade(template, arch, trade_date, fresh_md, args, run_dir, tag)
            rec['fresh_md'] = True

        rows.append(rec)
        logging.info('TRADE %s: greedy=%s nohedge=%s $/oz  bound=%s PASS=%s  churn=%s',
                     tag, rec['greedy_usd_oz'], rec['nohedge_usd_oz'], rec['pf_bound'], rec['bound_pass'], rec['churn'])
        with open(os.path.join(run_dir, 'trades.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    df = pd.DataFrame(rows)
    g = df['greedy_usd_oz']
    print('\n===== WALK-FORWARD BACKTEST ($/oz) =====')
    print(df.to_string(index=False))
    print(f"\ngreedy  mean {g.mean():+.2f}  min {g.min():+.2f}  max {g.max():+.2f}  "
          f"positives {int((g > 0).sum())}/{len(df)}  |  nohedge mean {df['nohedge_usd_oz'].mean():+.2f}  "
          f"|  bound-PASS {int(df['bound_pass'].sum())}/{len(df)}  |  margin {args.margin:+.2f} ($/oz)")
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
