"""CLI wrapper around `MarkovSwitchingLogOUSpotCalibration`. Loads a CSV of daily prices,
fits the HMM-LogOU via Baum-Welch, prints diagnostics, writes the calibrated params as JSON.

    python calibrate_hmm_logou.py --csv data/platinum_eod.csv --states 3 --iters 200
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskflow.stochasticprocess import MarkovSwitchingLogOUSpotCalibration


def report(fit, n_states):
    states = fit['states']
    print(f"\nConverged in {fit['iterations']} iterations")
    print(f"Final log-likelihood: {fit['log_likelihood']:,.2f}")
    print(f"\nPer-state OU parameters (sorted by σ):")
    print(f"{'state':>6}  {'kappa':>10}  {'theta':>10}  {'sigma':>10}  {'half_life_d':>12}  {'stat_std_log':>14}")
    for i, s in enumerate(states):
        half_life_days = np.log(2) / s['Kappa'] * 252
        stat_std = s['Sigma'] / np.sqrt(2 * s['Kappa'])
        print(f"{i:>6}  {s['Kappa']:>10.3f}  {s['Theta']:>10.3f}  {s['Sigma']:>10.3f}"
              f"  {half_life_days:>12.1f}  {stat_std:>14.4f}")
    print(f"\nTransition matrix (P[from, to]):")
    for row in fit['transition_matrix']:
        print(f"  {row}")
    print(f"\nStationary probs: {fit['stationary_probs']}")
    avg_dur = [1.0 / (1.0 - fit['transition_matrix'][i][i]) for i in range(n_states)]
    print(f"Average regime duration (business days): {[f'{d:.0f}' for d in avg_dur]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/platinum_eod.csv')
    ap.add_argument('--price-col', default='PL=F')
    ap.add_argument('--states', type=int, default=2)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='artifacts/hmm_logou_fit.json')
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['Date']).set_index('Date').sort_index()
    print(f"Loaded {len(df)} prices from {df.index[0].date()} to {df.index[-1].date()}")

    calib = MarkovSwitchingLogOUSpotCalibration(
        model=None,
        param={'N_States': args.states, 'N_Iter': args.iters, 'Seed': args.seed},
    )
    # Drop into the EM directly (returns the diagnostic dict including LL trajectory).
    prices = df[[args.price_col]].dropna().iloc[:, 0].astype(float).values
    fit = calib._fit_em(prices, dt=1.0 / 252.0, n_states=args.states, n_iter=args.iters, seed=args.seed)
    report(fit, args.states)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fit, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
