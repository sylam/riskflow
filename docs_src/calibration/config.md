# Calibration Config

`calibration_config.json` declares the historical archive location and the calibration
method to use for each stochastic process class. It is loaded into the same `Config`
object as the MarketData JSON via a second `parse_json` call.

```json
{
  "CalibrationConfig": {
    "MarketDataArchiveFile": {
      "name": "./data/plat_archive.csv",
      "skiprows": 0,
      "index_column": 0,
      "sep": ","
    },
    "Calibrations": {
      "MarkovHMMSpotModel": {
        "ID": "",
        "Method": "MarkovHMMSpotCalibration",
        "N_States": 3,
        "Use_Student_T": true,
        "Log_Price": true
      },
      "VARMixedFactorInterestRateModel": {
        "ID": "",
        "Method": "VARMixedFactorInterestRateCalibration"
      },
      "BasisLinkedSpotModel": {
        "ID": "",
        "Method": "BasisLinkedSpotCalibration",
        "Vol_Window": 21
      }
    }
  }
}
```

## `MarketDataArchiveFile`

Describes how to load the historical archive into a pandas DataFrame.

| Field | Description |
|---|---|
| `name` | Path to the archive CSV |
| `skiprows` | Rows to skip at the top of the file (typically 0) |
| `index_column` | Column index to use as the date index (typically 0) |
| `sep` | Field separator (default `\t` for legacy archives; set `,` for CSV) |

The archive's index is normalised to Excel-offset integers at load time. Date strings
in the index column (e.g. `2009/11/10`) are parsed and converted automatically.

## `Calibrations`

A dict mapping **stochastic process class name** (the model in `Price Models.<Model>.<Factor>`)
to its calibration parameters. Each entry must contain:

- `Method` — the calibration class name (resolved via `globals()` in `stochasticprocess.py`)
- Any model-specific tuning knobs (clamps, EM iteration caps, RNG seeds, etc.)

Calibration is keyed by **model class**, not by factor instance. All factors routed to the
same model class share the same calibration config. Per-factor configuration that depends
on data (e.g. number of states, Student-t flag) lives in this block; per-factor
configuration that depends on the factor (e.g. linked sibling factor) flows through the
[archive subkey convention](cross_factor.md), not through this config.

Calibration classes that don't need any tuning can have a minimal entry containing just
`ID` and `Method`. Classes that need to be discoverable but have no data archive (e.g. for
specialised use cases) can be omitted entirely — the framework will skip factors whose
class isn't registered.
