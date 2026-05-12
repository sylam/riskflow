# Excel + xlwings integration (MVP)

This folder provides a free Excel frontend path using `xlwings` and your local Python installation of RiskFlow.

## What is implemented

- Excel-callable pricing functions:
  - `RF_PRICE_JSON(job_json, overrides_json="")`
  - `RF_PRICE_PATH(job_path, overrides_json="")`
  - `RF_SOLVE_JSON(job_json, solve_spec_json, overrides_json="")`
  - `RF_SOLVE_PATH(job_path, solve_spec_json, overrides_json="")`
  - `RF_GET_LAST_RESULT(field="status")`
- Queue-driven processing hook (for button/macro):
  - `RF_PROCESS_NEXT_REQUEST()`
- Standalone local worker:
  - `python -m excel_integration.worker --verbose`
- Queue backend abstraction:
  - `file` backend works now (local folders)
  - `solace` backend adapter is intentionally pluggable in `queue_clients.py`

## Install

From project root:

```bash
pip install xlwings
```

For Solace backend:

```bash
pip install solace-pubsubplus
```

(If needed, also install your RiskFlow deps from `requirements.txt`.)

## Configure backend

Environment variables:

- `RF_QUEUE_BACKEND=file` or `solace`
- `RF_FILE_QUEUE_ROOT=.rf_queue` (used by file backend)

For Solace mode (adapter wiring target):

- `RF_SOLACE_HOST`
- `RF_SOLACE_VPN`
- `RF_SOLACE_USERNAME`
- `RF_SOLACE_PASSWORD`
- `RF_SOLACE_REQUEST_QUEUE` (default `riskflow/requests`)
- `RF_SOLACE_RESULT_TOPIC` (default `riskflow/results`)

## Message schema

Request JSON (from Solace or file queue):

```json
{
  "request_id": "abc-123",
  "job_path": "C:/jobs/Trade01.json",
  "overrides": {
    "Simulation_Batches": 10,
    "Batch_Size": 512
  }
}
```

You may send `job_json` instead of `job_path`.

Optional solve mode:

```json
{
  "request_id": "abc-123",
  "job_path": "C:/jobs/Structure01.json",
  "solve_spec": {
    "method": "brentq",
    "variables": [
      {
        "name": "strike",
        "path": "/Calc/Deals/Deals/Children/0/Instrument/field/Strike",
        "initial": 100,
        "lower": 50,
        "upper": 200
      }
    ],
    "targets": [
      {
        "name": "net_mtm",
        "metric": "net_mtm",
        "target": 0.0,
        "weight": 1.0
      }
    ]
  }
}
```

Result JSON:

```json
{
  "request_id": "abc-123",
  "status": "ok",
  "elapsed_ms": 220,
  "summary": {"stats": {}, "exposure_profile": []},
  "result": {"Results": {}, "Stats": {}}
}
```

Solve-mode result adds fields:

```json
{
  "mode": "solve",
  "method": "brentq",
  "solution": {"strike": 102.34},
  "targets": {"net_mtm": 0.0002},
  "residual_norm": 0.0002
}
```

On failure:

```json
{
  "request_id": "abc-123",
  "status": "error",
  "error": "...",
  "traceback": "..."
}
```

## Excel wiring (xlwings)

1. Open your workbook.
2. In xlwings add-in config, set UDF module to:
   - `excel_integration.xlwings_udfs`
3. Use formulas like:
   - `=RF_PRICE_PATH("C:\\jobs\\Trade01.json", "")`
  - `=RF_SOLVE_PATH("C:\\jobs\\Structure01.json", A1, "")` where `A1` contains a JSON `solve_spec`
   - `=RF_GET_LAST_RESULT("status")`
4. Add a button/macro that runs:
   - `RunPython ("import excel_integration.xlwings_udfs as rf; rf.RF_PROCESS_NEXT_REQUEST()")`

This gives you a local “pull one request, price, push result” control loop from Excel.

## Solve spec details

- `variables[*].path` is a JSON Pointer path in the input job JSON.
- `variables[*].lower` / `upper` define bounds used by the solver.
- `targets[*]` supports either:
  - `metric: "net_mtm"` (top portfolio MTM), or
  - `path: "result.path.here"` in the priced response (dot/bracket path syntax).
- Solver methods:
  - `brentq`: 1 variable + 1 target root solve (requires sign change across bounds)
  - `least_squares`: multi-variable or constrained fit to one/many targets

## Local test without Solace

1. Set backend:
   - `RF_QUEUE_BACKEND=file`
2. Enqueue a request from Excel formula:
   - `=RF_ENQUEUE_FILE_REQUEST("C:\\jobs\\Trade01.json")`
3. Process it:
   - run `RF_PROCESS_NEXT_REQUEST()` from button or macro
4. Read result files in:
   - `.rf_queue/results`

## Solace integration point

`SolaceQueueClient` is now implemented in `queue_clients.py` and does the following:

- Connects using `RF_SOLACE_HOST`, `RF_SOLACE_VPN`, `RF_SOLACE_USERNAME`, `RF_SOLACE_PASSWORD`
- Pulls one JSON request from the persistent queue `RF_SOLACE_REQUEST_QUEUE`
- Publishes JSON result to topic `RF_SOLACE_RESULT_TOPIC`

Switch backends by setting:

```bash
RF_QUEUE_BACKEND=solace
```

Then run:

```bash
python -m excel_integration.worker --verbose
```
