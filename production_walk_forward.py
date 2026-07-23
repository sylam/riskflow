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
  * CommodityBasis.LME_CME (BasisLinkedSpotModel, Observed_Commodity=PLATINUM_CME) carries the
    published basis b = S - P (LME - CME) — the LBMA catch-up.
  * CommodityPrice.PLATINUM_LME = P + b is the observable LBMA fixing (BasisComposedSpotModel;
    routed via the source MarketDataRF modelfilters, never calibrated — it carries no params).
  * The swap references pure PLATINUM_LME; the CME futures reference the primary through the
    identity basis CME_FLAT (Observed_Commodity=PLATINUM_CME, zero dynamics) so synthetic = P + 0
    = P (martingale) and E[dF|b] ≈ 0 (unexecutable reversion not harvested). The dependency graph
    is acyclic: CME -> {CME_FLAT, LME_CME} -> LME (no cycle, so no cycle-breaking needed).

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
        --start 2021-01 --months 24 --recal-months 3 --margin 8 --batch 8192 --seeds 7 42 314

Pass several --seeds for a seed-ensemble deployment (per trade: train one checkpoint per seed,
then ONE roll with DiffV2_Load_Value_Fn = [all checkpoints] = ensemble argmax). --run-dir <dir>
resumes a killed lane in place (completed trades + already-trained seed checkpoints are skipped).

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
import glob
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# Must precede the first torch/riskflow import (pulled in via production_solver below):
# PYTORCH_CUDA_ALLOC_CONF is parsed once at first allocator use. expandable_segments frees
# ~20GB of reserved allocator slack, bit-identical results.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from production_solver import apply_config, run

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCEL = pd.Timestamp('1899-12-30')
CONTRACT_SIZE = 50
CARRY_TENORS = ('PLATINUM_TAU1', 'PLATINUM_TAU2', 'PLATINUM_TAU3')

# --- corrected archive column names ---
CME_COL = 'CommodityPrice.PLATINUM_CME'          # martingale primary P
LME_COL = 'CommodityPrice.PLATINUM_LME'          # LBMA fixing S = P - b
BASIS_COL = 'CommodityBasis.LME_CME,PLATINUM_CME'  # published basis b = S - P (LME - CME)
CARRY_COL = 'ForwardRate.PLATINUM_CARRY'          # + ',<tenor>'
SOFR_PREFIX = 'InterestRate.USD-SOFR'

# validated objective (matches artifacts/platinum_hedge_shipping.json)
OBJECTIVE = {'Object': 'AsymmetricUtility_Huber', 'Huber_Aversion': 6.0, 'Huber_Delta': 1.0}
# (RL-era keys Floor_Penalty/Surplus_Reward/Power/Expiry_*/Post_Deal_Trade_Penalty/
#  *_Bounds_Penalty removed 2026-07-24: read by nothing - _normalize_objective_config
#  consumes only Object/Huber_*/CARA_Gamma/Utility_Scale_*.)


def _ts(d):
    return {'.Timestamp': pd.Timestamp(d).strftime('%Y-%m-%d')}


def build_corrected_archive(raw):
    """Decompose the raw futures file (PL1/PL2/PL3 + taus + repo + LBMA spot + SOFR) into the
    corrected series. Returns a DataFrame with the martingale primary P (CME-implied continuous
    spot), the published basis b = S - P (LME - CME), clean calendar-spread carry knots, tenors,
    the SOFR curve, and the LBMA fixing S. (The calibration CSV drops S — it is a composed factor
    with no archive series; it routes to BasisComposedSpotModel and is never calibrated.)"""
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
    out[BASIS_COL] = S - P                                 # published sign (LBMA - CME) = LME - CME
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


def delta_corridor_schedule(trade_date, fixings, band):
    """Deterministic causal delta-ramp corridor on the SIGNED total position Σq_i, keyed by
    sim-grid step t (= calendar-day offset from the trade date; the deal runs on the daily
    '0d 1d(1d)' grid). The remaining-average exposure is FULL (-50 contracts) before the first
    fixing, declines LINEARLY to 0 across the averaging window as fixings crystallize, and is 0
    after the last fixing — every date known at the trade date (causal). `band` (fraction of the
    50-contract notional) is the half-width of slack around the ramp:
        Min_Total = -(ramp + band·50) clamped to [-50, 0];  Max_Total = min(0, -(ramp − band·50)).
    Returns a list of {Step, Min_Total, Max_Total} knots (one per calendar day over the window;
    piecewise-constant between knots — grid_at(t) reads the rightmost knot with Step <= t)."""
    first = pd.Timestamp(fixings[0]); last = pd.Timestamp(fixings[-1])
    f0 = (first - pd.Timestamp(trade_date)).days
    fN = (last - pd.Timestamp(trade_date)).days
    span = max(fN - f0, 1)
    half = band * 50.0

    def knot(remaining):
        ramp = 50.0 * remaining
        lo = max(-50.0, min(0.0, -(ramp + half)))
        hi = min(0.0, -(ramp - half))
        return round(lo, 4), round(hi, 4)

    knots = {0: knot(1.0)}                                    # full short before the first fixing
    for off in range(f0, fN + 1):                            # linear decline across the fixings
        knots[off] = knot(max(0.0, min(1.0, (fN - off) / span)))
    return [{'Step': int(s), 'Min_Total': lo, 'Max_Total': hi}
            for s, (lo, hi) in sorted(knots.items())]


def build_deal_config(template, arch, trade_date, calibrated_md, margin, volume, delta_corridor=None,
                      spot_model='hmm'):
    """Reshape the deal template to the trade date in the corrected world, reading every market
    level off the corrected archive row. Returns (cfg, info) where info carries the strike and
    the dates the causal bound / observed path need. Adapt this for a different product.

    Deal: a 3-month average-price swap (Receiver) on the pure LBMA fixing, averaged over the full
    3rd calendar month after the trade date, paid +5 days, struck at fair - margin. Hedges: the
    CME futures strip at the 3 carry tenors, each referencing the primary through the identity
    basis Implied_Basis=CME_FLAT (synthetic = P + 0 = P, the martingale primary).
    """
    row = arch.loc[:trade_date].iloc[-1]
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    mm = cfg['Calc']['MergeMarketData']
    emd = mm['ExplicitMarketData']
    hp = calc['Hedging_Problem']

    calc['Base_Date'] = _ts(trade_date)
    calc['Scenario_Factors'] = ['CommodityBasis.LME_CME']  # reached only via the composed LME spot
    mm['MarketDataFile'] = calibrated_md
    hp['Objective'] = dict(OBJECTIVE)

    s0 = float(row[LME_COL])
    p0 = float(row[CME_COL])
    b0 = float(row[BASIS_COL])            # published S - P (LME - CME)
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

    # --- tradables: CME strip at the tenor ladder (primary via CME_FLAT identity basis = P) -----
    futs, positions, setts, margins, limits = {}, {}, {}, {}, {}
    for i, (mat, c, tau) in enumerate(zip(mats, carry, taus), 1):
        name = f'PL_M{i}'
        futs[name] = {'Maturity_Date': _ts(mat), 'Currency': 'USD', 'Carry': 'PLATINUM_CARRY',
                      'Repo_Rate': 'USD-SOFR', 'Implied_Basis': 'CME_FLAT', 'Contract_Size': CONTRACT_SIZE}
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
    # Optional causal delta-ramp corridor on Σq_i (both TRAIN and ROLL configs carry it — the DP
    # should learn within the corridor; for the roll-only validation phase re-rolls activate it on
    # corridor-free checkpoints). The ramp is deterministic from THIS deal's fixing calendar.
    if delta_corridor is not None:
        hp['Evaluator']['Total_Position_Schedule'] = delta_corridor_schedule(
            trade_date, fixings, delta_corridor)
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
        'CommodityBasis.LME_CME': {'Spot': b0, 'Observed_Commodity': 'PLATINUM_CME'},
        'CommodityBasis.CME_FLAT': {'Spot': 0.0, 'Observed_Commodity': 'PLATINUM_CME'},
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
    emd_pm = emd.setdefault('Price Models', {})
    emd_pm['BasisComposedSpotModel.PLATINUM_LME'] = {}
    # CME_FLAT is the zero-dynamics identity basis (not calibrated — carries no archive series);
    # inject its params so the futures' synthetic = P + 0 = P references the martingale primary.
    # Sigma form must match the primary: regime-conditional (HMM publishes 'regimes') vs flat
    # (GARCH publishes no regimes — a Sigma_By_State basis would fail loud on the missing key).
    flat = {'A': 0.0, 'Phi': 0.0, 'Nu': 5.0, 'Mu': 0.0}
    emd_pm['BasisLinkedSpotModel.CME_FLAT'] = (
        {**flat, 'Sigma': 0.0} if spot_model == 'garch' else {**flat, 'Sigma_By_State': [0.0, 0.0, 0.0]})
    emd['Valuation Configuration'] = {'FloatingEnergyDeal': {'ForwardCurve': 'Components'}}

    info = {'k_fair': k_fair, 'mats': mats, 'pay': pay}
    return cfg, info


def observed_scenario_npz(arch, base_date, path):
    """Dense daily realized (CME primary, published basis) from the base date, forward-filled,
    written to an npz the calc's Observed_Scenario reads. The framework composes the LBMA fixing
    S = P + b internally. Keys are the full factor names the seam matches on."""
    base = pd.Timestamp(base_date)
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=i) for i in range(220)])
    if dates.max() > arch.index[-1]:
        raise ValueError(
            f'Observed window {base.date()}+220d ends {dates.max().date()} past archive end '
            f'{arch.index[-1].date()}: the realized roll would run on fabricated flat prices')
    rows = arch.reindex(arch.index.union(dates)).ffill().loc[dates]
    np.savez(path, **{CME_COL: rows[CME_COL].to_numpy(),
                      'CommodityBasis.LME_CME': rows[BASIS_COL].to_numpy()})


def pf_bound(arch, trade_date, mats, pay):
    """Portfolio causal bound, Σ_t max_leg max(0, -dF_obs), on the reconstructed observed CME
    forward strip ($/oz). A hedge can only lose (vs no-hedge) by the sum over days of the worst
    adverse forward move it could be caught in; greedy P&L must stay ≤ nohedge + this bound."""
    bdays = pd.bdate_range(trade_date, pay)
    if bdays.max() > arch.index[-1]:
        raise ValueError(
            f'Bound window {pd.Timestamp(trade_date).date()}..{pd.Timestamp(pay).date()} ends past '
            f'archive end {arch.index[-1].date()}: the causal bound would use fabricated flat prices')
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


def _atomic_write_csv(path, rows):
    """Atomic CSV write: temp file in the same dir + os.replace (POSIX-atomic) so a SIGTERM/OOM
    mid-flush can't truncate trades.csv. Fieldnames = the ordered union across rows (a FAILED
    row carries an extra 'error' column the success rows lack)."""
    fieldnames = list({k: None for r in rows for k in r})
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    """Atomic JSON write (tmp + os.replace) so a killed lane leaves either the old file or the
    complete new one — never a truncated per-trade row sidecar the resume path would misread."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def calibrate(marketdata, cal_config, cal_end, out_md):
    """Calibrate the corrected models on the archive up to cal_end (no lookahead)."""
    logging.info('=== CALIBRATE through %s ===', cal_end)
    subprocess.run([sys.executable, 'calibrate_platinum.py', '--marketdata', marketdata,
                    '--calibration-config', cal_config, '--end', cal_end, '--out', out_md],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def garchify_md(hmm_md, arch, cal_end, out_md):
    """Swap ONLY the model layer of a HMM-calibrated md into the GARCH world (causal, series ≤
    cal_end); everything else (carry VAR, SOFR, composed-LME routing) is byte-for-byte untouched:
      * GARCHSpotModel.PLATINUM_CME from GARCHSpotCalibration on the primary P series (same
        |r|<0.25 guard and H0 = filtered variance at the trade date — reuses the calibrator).
      * BasisLinkedSpotModel.LME_CME → flat-Sigma OU: keep the HMM A/Phi/Nu (the OU structure);
        σ = unconditional innovation std of b up to the trade date (NO regime conditioning, NO
        shipping constant).
      * modeldefaults.CommodityPrice → GARCHSpotModel; Correlations mirror the carry-spot entry
        under GARCHSpotProcess.PLATINUM_CME.
    Logs the calibrated params (ω, α, β, ν, H0, LR vol, basis σ/κ) + a martingale-by-state
    diagnostic at INFO. Returns the GARCH primary param dict."""
    from riskflow.stochasticprocess import GARCHSpotCalibration   # the framework's calibrator
    md = json.load(open(hmm_md))
    m = md['MarketData']
    pm = m['Price Models']

    P = arch[CME_COL].loc[:cal_end].astype(float)
    block = GARCHSpotCalibration('GARCHSpotModel', {}).calibrate(pd.DataFrame({CME_COL: P}), 0.0).param
    block['Convexity_Correction'] = 'Yes'        # price-martingale tradeable (no harvestable Jensen drift)
    pm.pop('MarkovHMMSpotModel.PLATINUM_CME', None)
    pm['GARCHSpotModel.PLATINUM_CME'] = block
    m['Model Configuration']['.ModelParams']['modeldefaults']['CommodityPrice'] = 'GARCHSpotModel'

    corr = m.setdefault('Correlations', {})
    hmm_corr = corr.pop('MarkovHMMSpotProcess.PLATINUM_CME', None)
    if hmm_corr is not None:
        corr['GARCHSpotProcess.PLATINUM_CME'] = dict(hmm_corr)

    lme = pm['BasisLinkedSpotModel.LME_CME']
    A, Phi = float(lme['A']), float(lme['Phi'])
    b = arch[BASIS_COL].loc[:cal_end].astype(float)            # published basis S - P
    eta = (b - A * P.diff() - Phi * b.shift(1)).dropna()       # AR(1)-on-ΔP innovations, no regime split
    basis_sigma = float(eta.std())
    lme.pop('Sigma_By_State', None)
    lme['Sigma'] = basis_sigma

    _atomic_write_json(out_md, md)

    persist = block['Alpha'] + block['Beta']
    lr_vol = float(np.sqrt(block['Omega'] / (1.0 - persist) / block['Calibration_DT_Years']))
    kappa = float(-np.log(Phi) / lme['Calibration_DT_Years']) if 0.0 < Phi < 1.0 else float('nan')
    # martingale-by-state on realized data: filter h, bucket ΔlogP by log-h decile, E[r|bucket]≈μ=0
    r = np.log(P).diff().dropna().values
    r = r[np.abs(r) < 0.25]
    h = np.empty_like(r)
    h[0] = block['H0']
    for i in range(1, len(r)):
        h[i] = block['Omega'] + block['Alpha'] * r[i - 1] ** 2 + block['Beta'] * h[i - 1]
    lh = np.log(h)
    edges = np.quantile(lh, np.linspace(0.0, 1.0, 11))
    worst = 0.0
    for k in range(10):
        msk = (lh >= edges[k]) & (lh <= edges[k + 1])
        if msk.sum() < 20:
            continue
        se = r[msk].std() / np.sqrt(int(msk.sum()))
        worst = max(worst, abs(r[msk].mean()) / se if se > 0 else 0.0)
    logging.info('GARCH-CALIB %s: omega=%.4e alpha=%.4f beta=%.4f nu=%.2f H0=%.4e LR-vol=%.4f | '
                 'basis sigma=%.3f phi=%.3f kappa=%.2f | martingale-by-state worst|E[r|h-decile]|/se=%.2f',
                 cal_end, block['Omega'], block['Alpha'], block['Beta'], block['Nu'], block['H0'],
                 lr_vol, basis_sigma, Phi, kappa, worst)
    return block


def one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag):
    """Train the day-1 policy for EACH seed and roll the frozen seed-ensemble on the realized
    path; return the recorded row. One checkpoint per seed (value_fn_<tag>_s<seed>.pt), then ONE
    stepper roll with DiffV2_Load_Value_Fn = [all checkpoints] — the framework evaluates each
    member in its own standardization frame and averages continuations before the argmax
    (cross-fit winner's-curse reduction; logs 'ENSEMBLE argmax over N value fns'). With a single
    seed the roll loads a 1-element list, bit-identical to the pre-ensemble behaviour.

    Seed-level idempotency: a seed whose checkpoint already exists in the run dir is skipped, so a
    killed lane re-pointed at the same run dir resumes mid-trade without retraining. Row diagnostics
    train_u / V_0 report the primary member (seeds[0]); train_u_seeds carries the per-seed spread."""
    cfg, info = build_deal_config(template, arch, trade_date, calibrated_md, args.margin, args.volume,
                                  delta_corridor=args.delta_corridor, spot_model=args.spot_model)
    # Model tag stamped into every per-trade artifact name so a GARCH lane pointed at an HMM run
    # dir (or vice-versa) never reuses the wrong checkpoint/md/obs; HMM names are unchanged.
    sfx = '' if args.spot_model == 'hmm' else f'_{args.spot_model}'

    ckpts, train_us, v0s, market_dim = [], [], [], None
    for seed in args.seeds:
        ckpt = os.path.abspath(os.path.join(run_dir, f'value_fn_{tag}{sfx}_s{seed}.pt'))
        ckpts.append(ckpt)
        if os.path.exists(ckpt):
            logging.info('=== TRAIN %s seed=%d SKIP (checkpoint exists) ===', tag, seed)
            train_us.append(None)
            v0s.append(None)
            continue
        train = apply_config(copy.deepcopy(cfg), batch=args.batch, seed=seed, save=ckpt)
        if args.fit_iters is not None:
            train['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Fit_Iters'] = args.fit_iters
        logging.info('=== TRAIN %s seed=%d (fair=%.2f, strike=%.2f) ===',
                     tag, seed, info['k_fair'], info['k_fair'] - args.margin)
        tdiag = run(train, f'train_{tag}_s{seed}')
        tv = (tdiag.get('verdict') or {}).get('greedy') or {}
        train_us.append(None if tv.get('u_mean') is None else round(tv['u_mean'], 4))
        v0s.append(tdiag.get('V_0'))
        market_dim = tdiag.get('market_dim')

    obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{tag}{sfx}.npz'))
    observed_scenario_npz(arch, trade_date, obs_npz)
    roll = apply_config(copy.deepcopy(cfg), batch=1, seed=args.seeds[0], load=ckpts,
                        stepper_rollout=True, randomize_initial_state=False)
    # The ROLL's inner draws are independent of training's (validated 64): at Batch_Size=1 a
    # large inner sub-batch is nearly free and shrinks the causal one-step forecast noise
    # (the argmax input is E_inner[C_{t+1}]) that dominates single-realized-path dispersion.
    roll['Calc']['Calculation']['Inner_Sub_Batch'] = args.roll_inner
    roll['Calc']['Calculation']['Observed_Scenario'] = obs_npz
    logging.info('=== ROLL %s (stepper, realized path, %d-seed ensemble, inner=%d) ===',
                 tag, len(ckpts), args.roll_inner)
    rdiag = run(roll, f'roll_{tag}')
    json.dump(rdiag, open(os.path.join(run_dir, f'diag_{tag}{sfx}.json'), 'w'), indent=1, default=str)

    sv = rdiag.get('stepper_verdict') or {}
    gr = (sv.get('greedy') or {}).get('wT_mean')
    nh = (sv.get('nohedge') or {}).get('wT_mean')
    bound = pf_bound(arch, trade_date, info['mats'], info['pay'])
    q = np.array(sv.get('greedy_q_traj') or [[0.0]])
    return {
        'trade': tag, 'spot_model': args.spot_model,
        'fair': round(info['k_fair'], 2), 'strike': round(info['k_fair'] - args.margin, 2),
        'n_seeds': len(args.seeds), 'market_dim': market_dim,
        'train_u': train_us[0],
        'train_u_seeds': train_us,
        'V_0': v0s[0],
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
    ap.add_argument('--spot-model', choices=['hmm', 'garch'], default='hmm',
                    help='CommodityPrice model for the primary. hmm (default): MarkovHMMSpotModel, '
                         'unchanged. garch: per-trade-date GARCHSpotModel (causal recalibration) + '
                         'flat-Sigma basis; swaps ONLY the model layer of the calibrated md.')
    ap.add_argument('--start', default='2021-01', help='First trade month, YYYY-MM.')
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--recal-months', type=int, default=3, help='Recalibrate every N months.')
    ap.add_argument('--margin', type=float, default=8.0, help='$/oz dealer margin (strike = fair - margin).')
    ap.add_argument('--volume', type=float, default=2500.0, help='Swap volume (oz).')
    ap.add_argument('--batch', type=int, default=8192, help='Training outer paths (scale up on a big GPU).')
    ap.add_argument('--seeds', type=int, nargs='+', default=[7],
                    help='Training seed(s) / ensemble members. >1 trains one checkpoint per seed '
                         'and rolls the seed-ensemble argmax (each member in its own frame, '
                         'continuations averaged) on the realized path.')
    ap.add_argument('--seed', type=int, default=None,
                    help='Backward-compat alias for a single --seeds value (overrides --seeds if set).')
    ap.add_argument('--run-dir', default=None,
                    help='Reuse this run directory instead of a fresh timestamped one. Point a '
                         'restarted lane at its prior dir to resume: completed trades (per-trade '
                         'row sidecars) and already-trained seed checkpoints are skipped.')
    ap.add_argument('--roll-inner', type=int, default=256,
                    help='Inner_Sub_Batch for the realized-path ROLL only (training stays at the '
                         'validated 64). The roll is Batch_Size=1, so a large inner sub-batch is '
                         'nearly free and de-noises the causal one-step argmax forecast. '
                         'Default 256; pass 64 to recover the pre-lever roll behavior.')
    ap.add_argument('--fit-iters', type=int, default=None, help='Override DiffV2_Fit_Iters (smoke tests).')
    ap.add_argument('--delta-corridor', type=float, default=None,
                    help='Enforce a causal delta-ramp corridor on the SIGNED total position: '
                         'BAND = half-width as a fraction of the 50-contract notional (e.g. 0.40). '
                         'Off by default. Emits Evaluator.Total_Position_Schedule on train+roll.')
    args = ap.parse_args()
    if args.seed is not None:      # fold the legacy single-seed alias into the seed list
        args.seeds = [args.seed]

    raw = pd.read_csv(args.archive, index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    template = json.load(open(args.deal_template))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.run_dir or os.path.join(
        'artifacts', 'walk_forward', f'{stamp}_{args.start}_{args.months}m_{args.spot_model}')
    os.makedirs(run_dir, exist_ok=True)
    logging.info('run dir: %s  spot_model: %s  seeds: %s', run_dir, args.spot_model, args.seeds)

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

    # Resume: a killed lane re-pointed at its run dir (--run-dir) skips completed trades. Only
    # SUCCESS rows are sidecar-persisted (row_<tag>.json, atomic), so a FAILED trade retries.
    done = {}
    for f in sorted(glob.glob(os.path.join(run_dir, 'row_*.json'))):
        r = json.load(open(f))
        # Fail loud if this run dir was built with a DIFFERENT spot model — a GARCH lane pointed at
        # an HMM run dir (or vice-versa) must never reuse/overwrite the wrong world's rows.
        prior = r.get('spot_model', 'hmm')
        if prior != args.spot_model:
            raise RuntimeError(
                f'run dir {run_dir} holds spot_model={prior!r} rows (e.g. {os.path.basename(f)}), '
                f'but --spot-model={args.spot_model!r}. Use a fresh --run-dir.')
        done[r['trade']] = r
    if done:
        logging.info('RESUME: %d completed trade(s) found in run dir; will skip them', len(done))

    rows, calibrated_md = [], None
    for m in range(args.months):
        trade_date = (pd.Timestamp(args.start + '-01') + pd.offsets.MonthBegin(m) + pd.offsets.BDay(0)).normalize()
        tag = trade_date.strftime('%Y%m')

        # One month's failure (OOM / archive gap / corrupt-checkpoint UnpicklingError / a
        # calibrate() that raises) must degrade to a single FAILED row, not discard every
        # remaining month. The broad guard logs the FULL traceback (logging.exception) so the
        # cause survives; the inner ValueError guard keeps the stale-md fresh-calibration retry.
        try:
            if m % args.recal_months == 0:
                # Recalibrate every boundary even for a resumed/skipped trade: calibrated_md must
                # be set for the later non-skipped months, and re-running (a subprocess) is cheap
                # and can't trust a possibly-truncated md from a killed lane.
                hmm_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}.json'))
                cal_end = trade_date.strftime('%Y-%m-%d')
                calibrate(md_cal, cal_cfg, cal_end, hmm_md)
                # GARCH world: swap only the model layer of the freshly-calibrated md (causal),
                # writing a model-tagged md so the HMM cache is never overwritten.
                calibrated_md = hmm_md if args.spot_model == 'hmm' else \
                    os.path.abspath(os.path.join(run_dir, f'md_{tag}_garch.json'))
                if args.spot_model == 'garch':
                    garchify_md(hmm_md, arch, cal_end, calibrated_md)
                g = guard_e_df_b(arch, cal_end)
                logging.info('GUARD %s: dP~b slope=%+.4f t=%+.2f (martingale) | dS~b slope=%+.4f t=%+.2f (catch-up)',
                             cal_end, g['dP~b'][0], g['dP~b'][1], g['dS~b'][0], g['dS~b'][1])
            if tag in done:
                logging.info('TRADE %s: SKIP (already completed, resuming)', tag)
                rec = done[tag]
            else:
                try:
                    rec = one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag)
                except ValueError as e:
                    # Stale quarterly calibration: a VAR carry front slot can expire between cal-date and
                    # trade-date (tau -> 0), tripping the float32 X_0 round-trip. Still no lookahead:
                    # recalibrate AT the trade date and retry.
                    logging.warning('TRADE %s: stale-md failure (%s); FALLBACK fresh calibration', tag, e)
                    fresh_end = trade_date.strftime('%Y-%m-%d')
                    fresh_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}_fresh.json'))
                    calibrate(md_cal, cal_cfg, fresh_end, fresh_md)
                    if args.spot_model == 'garch':
                        garch_fresh = os.path.abspath(os.path.join(run_dir, f'md_{tag}_fresh_garch.json'))
                        garchify_md(fresh_md, arch, fresh_end, garch_fresh)
                        fresh_md = garch_fresh
                    rec = one_trade(template, arch, trade_date, fresh_md, args, run_dir, tag)
                    rec['fresh_md'] = True
                logging.info('TRADE %s: greedy=%s nohedge=%s $/oz  bound=%s PASS=%s  churn=%s',
                             tag, rec['greedy_usd_oz'], rec['nohedge_usd_oz'], rec['pf_bound'], rec['bound_pass'], rec['churn'])
                _atomic_write_json(os.path.join(run_dir, f'row_{tag}.json'), rec)  # mark done for resume
        except Exception as e:
            logging.exception('TRADE %s FAILED', tag)
            rec = {'trade': tag, 'spot_model': args.spot_model, 'fair': None, 'strike': None,
                   'n_seeds': len(args.seeds), 'market_dim': None,
                   'train_u': None, 'train_u_seeds': None, 'V_0': None,
                   'greedy_usd_oz': None, 'nohedge_usd_oz': None, 'pf_bound': None,
                   'bound_pass': None, 'churn': None, 'error': str(e)}

        rows.append(rec)
        _atomic_write_csv(os.path.join(run_dir, 'trades.csv'), rows)

    df = pd.DataFrame(rows)
    g = df['greedy_usd_oz']
    print('\n===== WALK-FORWARD BACKTEST ($/oz) =====')
    print(df.to_string(index=False))
    print(f"\ngreedy  mean {g.mean():+.2f}  min {g.min():+.2f}  max {g.max():+.2f}  "
          f"positives {int((g > 0).sum())}/{len(df)}  |  nohedge mean {df['nohedge_usd_oz'].mean():+.2f}  "
          f"|  bound-PASS {int(df['bound_pass'].sum())}/{len(df)}  |  margin {args.margin:+.2f} ($/oz)  "
          f"|  seeds {args.seeds}")
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
