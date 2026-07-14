"""Generic calibration runner for a problem-specific MarketDataRF.

Reads a MarketDataRF.json (which carries the Price Factors + Model Configuration that
routes each factor to its stochastic process via modeldefaults / modelfilters) and a
calibration_config.json (which carries the archive + per-model calibration methods),
runs `Config.calibrate_factors`, and writes a calibrated MarketDataRF (Price Models
populated, Correlations populated).

To make a new variant, clone the input MarketDataRF, edit Price Factors / Model
Configuration, and re-run with --marketdata pointing at the clone. The script itself
is generic — it does not know about platinum.

Usage:
    python calibrate_platinum.py --marketdata data/MarketDataRF_platinum.json \\
        --out artifacts/MarketDataRF_platinum_calibrated.json
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from riskflow.config import Config
from riskflow.utils import excel_offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marketdata', default='./data/MarketDataRF_platinum.json',
                        help='Source MarketDataRF.json — Model Configuration drives factor routing.')
    parser.add_argument('--calibration-config', default='./artifacts/calibration_config.json',
                        help='Calibration_config.json (test variants live in artifacts/).')
    parser.add_argument('--out', default='./data/MarketDataRF_platinum_calibrated.json',
                        help='Output path for the calibrated MarketDataRF (the deliverable).')
    parser.add_argument('--start', default=None, help='YYYY-MM-DD; defaults to archive start')
    parser.add_argument('--end', default=None, help='YYYY-MM-DD; defaults to archive end')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='Spike-removal threshold (σ). 0 disables (the default — '
                             'platinum diffs are not spike-prone enough to need it).')
    parser.add_argument('--correlation-cutoff', type=float, default=0.1,
                        help='Drop |ρ| below this from the Correlations block.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s %(name)s: %(message)s',
    )

    aa = Config()
    aa.parse_json(args.marketdata)
    aa.parse_json(args.calibration_config)

    # Routing is fully driven by MarketDataRF's Model Configuration (modeldefaults +
    # modelfilters; the latter supports filters like ["ID", "PLATINUM_CARRY"] that match
    # on factor name). The script doesn't override anything here.
    #
    # The MarketData typically has empty Price Factors — those get filled at job time by
    # ExplicitMarketData in the policy/calc cfg. So the source of factor *discovery* is
    # the archive. We merge 'present' (Price Factors with routed models) and 'absent'
    # (archive-only factors that route via Model Configuration) — same pattern as
    # example.calibrate_PFE.
    all_factors = aa.fetch_all_calibration_factors(override={})
    discovered = dict(all_factors['present'])
    discovered.update(all_factors['absent'])

    # Keep only those whose archive data is actually present (factors that route to a
    # model but have no archive series — e.g. FxRate.USD — would error in calibrate).
    keep = {n: r for n, r in discovered.items() if r.archive_name in aa.archive_columns}
    skipped = [n for n in discovered if n not in keep]
    if skipped:
        for n in skipped:
            logging.info(f'no archive data — skipping calibration of {n}')

    if not keep:
        raise SystemExit('No factors with both a routed model and archive data — check '
                         'Model Configuration and the archive.')

    if args.start is None:
        start_date = excel_offset + pd.offsets.Day(int(aa.archive.index.min()))
    else:
        start_date = pd.Timestamp(args.start)
    if args.end is None:
        end_date = excel_offset + pd.offsets.Day(int(aa.archive.index.max()))
    else:
        end_date = pd.Timestamp(args.end)

    print(f'Calibrating {len(keep)} factors from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}')
    for name, info in sorted(keep.items()):
        cal_name = type(info.calibration).__name__ if info.calibration else '(none)'
        print(f'  {name}  →  {info.model_name}  ({cal_name})')

    # Wipe any prior fits so the output reflects only this run.
    aa.params['Price Models'] = {}

    aa.calibrate_factors(
        start_date, end_date, keep,
        smooth=args.smooth,
        correlation_cuttoff=args.correlation_cutoff,
        overwrite_correlations=True,
    )

    print('\n=== Price Models written ===')
    for k in sorted(aa.params['Price Models']):
        print(f'  {k}')

    print('\n=== Correlations ===')
    if not aa.params.get('Correlations'):
        print('  (none above cutoff)')
    for k, v in sorted(aa.params.get('Correlations', {}).items()):
        a, b = k
        print(f'  {a:55s} ↔  {b:55s}  {v:+.4f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    aa.write_marketdata_json(args.out)
    sz = Path(args.out).stat().st_size / 1024
    print(f'\nWrote {args.out}  ({sz:.1f} KB)')


if __name__ == '__main__':
    main()
