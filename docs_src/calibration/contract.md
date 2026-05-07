# Calibration Class Contract

A calibration class implements one method:

```python
def calibrate(self, data_frame, vol_shift, num_business_days=252.0, **kwargs):
    ...
    return utils.CalibrationInfo(param, correlation, delta)
```

Once registered (see below), the framework discovers it automatically and runs it as part
of `calibrate_factors`. New model authors should follow this contract verbatim — there
are no other framework hooks.

## Inputs

| Argument | Description |
|---|---|
| `data_frame` | The factor's archive columns, plus any related-factor columns the framework auto-pulled via the [subkey convention](cross_factor.md) |
| `vol_shift` | Optional volatility floor shift (model-specific; most classes pass through unused) |
| `num_business_days` | Days-per-year for annualisation. Default `252.0` |
| `**kwargs` | Reserved for future framework-passed context. Existing classes ignore. |

The class receives `param` at construction time (the
[calibration_config](config.md) entry for its model class). Tuning knobs come from there.

## Output: `CalibrationInfo`

A namedtuple with three fields. All three are load-bearing:

### `param`

A plain dict that becomes the JSON content of `Price Models.<ModelClass>.<FactorName>`.
Its shape must match what the simulator's process class reads at `precalculate` time —
the calibration writes, the simulator reads, no schema validation in between, so
consistency is on the author.

```python
param = {
    'A': float(a_hat),
    'Phi': float(phi_hat),
    'Sigma_By_State': sigma_by_state_list,
    'Calibration_DT_Years': 1.0 / num_business_days,
}
```

### `correlation`

A `(num_factors, num_delta_columns)` matrix that maps the factor's primitive Gaussian
factors (declared via `correlation_name` on the process class) to the columns of `delta`.
For most single-factor processes this is `[[1.0]]`. For multi-factor processes (e.g. PCA,
VAR) it's typically the identity matrix or a per-PC loadings matrix.

The framework uses this matrix during the consolidate-and-correlate step:

```
factor_correlations = a · ρ · aᵀ
```

where `a` is the block-diagonal stack of all factors' `correlation` matrices and `ρ` is
the empirical Pearson correlation of all `delta` columns concatenated.

### `delta`

A `pandas.DataFrame` of standardised innovation residuals, one column per primitive
factor. Indexed by the calibration date range. The framework concatenates all factors'
`delta` columns and computes pairwise Pearson correlations across them — these become
the entries in the global `Correlations` block (subject to the absolute-value cutoff).

For a calibration to participate in cross-factor correlation discovery, its `delta` should
be approximately *iid N(0, 1)* under the calibrated model. Examples:

- **HMM with regime-switching emissions:** delta = `(diff_t - μ_state(t)) / σ_state(t)` —
  standardised under the decoded regime.
- **OU/AR(1):** delta = innovation residual after subtracting the mean-reversion term.
- **VAR(1):** delta = per-factor innovation column (one per primitive Gaussian).

If `delta` columns are not approximately iid, cross-factor correlation estimates will be
contaminated by within-factor structure (autocorrelation, heteroskedasticity, regime
dependence). The framework's correlation cutoff filters out small spurious entries but
won't repair systematic bias.

## Registration

Calibration classes are resolved by string lookup at config load time:

```python
def construct_calibration_config(calibration_model, param):
    return globals().get(param['Method'])(calibration_model, param)
```

To add a new calibration class:

1. Define the class in `riskflow/stochasticprocess.py` (or wherever the matching model
   class lives — `globals()` resolution requires same module).
2. Add an entry under `Calibrations` in `calibration_config.json` with `Method` set to the
   class name. Tuning parameters go in the same entry.
3. Register the model class itself in `riskflow/fields.py`'s `Process_factor_map` for
   the relevant factor type.

No other registration is needed — the framework picks up the new method automatically.

## Independence requirement

Calibration classes operate on the data passed to `calibrate()`. They do **not** reach
into other factors' archives, calibrated parameters, or process instances. If a
calibration needs information from a sibling factor, the
[archive subkey convention](cross_factor.md) brings that data into `data_frame` and the
class stays self-contained.

Avoiding cross-class coupling keeps each calibration:

- **Reproducible** — same `data_frame` in, same params out, regardless of run order.
- **Testable** — fits run on isolated data slices without setting up the full Config.
- **Stable under refactor** — renaming or restructuring sibling factors doesn't ripple
  through unrelated calibration classes.
