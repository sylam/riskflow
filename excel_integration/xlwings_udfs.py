from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import xlwings as xw

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_textbook_hedge as hedge_runner

from .config import load_settings
from .portfolio_service import (
    CALC_COLS,
    PORTFOLIO_COLS,
    RISKFACTOR_COLS,
    all_instrument_types,
    flatten_calc_params,
    flatten_portfolio,
    flatten_risk_factors,
    get_deal_defaults,
    instrument_types_grouped,
    unflatten_calc_params,
    update_risk_factors_from_rows,
)
from .pricing_service import build_portfolio_job_json, price_job
from .queue_clients import build_queue_client, FileQueueClient

_STATE_FILE = Path(".rf_excel_state.json")


def _save_state(data: dict[str, Any]) -> None:
    _STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")


def _load_state() -> dict[str, Any]:
    if not _STATE_FILE.exists():
        return {}
    return json.loads(_STATE_FILE.read_text(encoding="utf-8"))


@xw.func
@xw.arg("job_json", doc="Full RiskFlow calculation JSON as text")
@xw.arg("overrides_json", doc="Optional overrides dict as JSON string")
def RF_PRICE_JSON(job_json: str, overrides_json: str = "") -> str:
    payload: dict[str, Any] = {"job_json": job_json}
    if overrides_json and str(overrides_json).strip():
        payload["overrides"] = json.loads(overrides_json)
    result = price_job(payload)
    _save_state(result)
    return json.dumps(result, ensure_ascii=False, default=str)


@xw.func
@xw.arg("job_path", doc="Path to a RiskFlow job JSON file")
@xw.arg("overrides_json", doc="Optional overrides dict as JSON string")
def RF_PRICE_PATH(job_path: str, overrides_json: str = "") -> str:
    payload: dict[str, Any] = {"job_path": job_path}
    if overrides_json and str(overrides_json).strip():
        payload["overrides"] = json.loads(overrides_json)
    result = price_job(payload)
    _save_state(result)
    return json.dumps(result, ensure_ascii=False, default=str)


@xw.func
@xw.arg("job_json", doc="Full RiskFlow calculation JSON as text")
@xw.arg("solve_spec_json", doc="Solve specification as JSON string")
@xw.arg("overrides_json", doc="Optional overrides dict as JSON string")
def RF_SOLVE_JSON(job_json: str, solve_spec_json: str, overrides_json: str = "") -> str:
    payload: dict[str, Any] = {"job_json": job_json, "solve_spec": json.loads(solve_spec_json)}
    if overrides_json and str(overrides_json).strip():
        payload["overrides"] = json.loads(overrides_json)
    result = price_job(payload)
    _save_state(result)
    return json.dumps(result, ensure_ascii=False, default=str)


@xw.func
@xw.arg("job_path", doc="Path to a RiskFlow job JSON file")
@xw.arg("solve_spec_json", doc="Solve specification as JSON string")
@xw.arg("overrides_json", doc="Optional overrides dict as JSON string")
def RF_SOLVE_PATH(job_path: str, solve_spec_json: str, overrides_json: str = "") -> str:
    payload: dict[str, Any] = {"job_path": job_path, "solve_spec": json.loads(solve_spec_json)}
    if overrides_json and str(overrides_json).strip():
        payload["overrides"] = json.loads(overrides_json)
    result = price_job(payload)
    _save_state(result)
    return json.dumps(result, ensure_ascii=False, default=str)


@xw.sub
def RF_PROCESS_NEXT_REQUEST() -> None:
    settings = load_settings()
    client = build_queue_client(settings)
    request = client.pull_request()
    if request is None:
        _save_state({"status": "idle", "message": "No request available"})
        return

    result = price_job(request)
    client.push_result(result)
    _save_state(result)


@xw.func
def RF_GET_LAST_RESULT(field: str = "status"):
    data = _load_state()
    return data.get(field, "")


@xw.func
def RF_ENQUEUE_FILE_REQUEST(job_path: str, request_id: str = "") -> str:
    settings = load_settings()
    client = build_queue_client(settings)
    if not isinstance(client, FileQueueClient):
        raise ValueError("RF_ENQUEUE_FILE_REQUEST is only available when RF_QUEUE_BACKEND=file")

    rid = client.enqueue_request({"job_path": job_path}, request_id or None)
    return rid


# =============================================================================
# Portfolio / Risk Factor / Calculations sheet UDFs
# =============================================================================

def _get_or_create_sheet(wb: xw.Book, name: str) -> xw.Sheet:
    """Return named sheet, creating it if it doesn't exist."""
    if name not in [s.name for s in wb.sheets]:
        wb.sheets.add(name)
    return wb.sheets[name]


def _write_table(sheet: xw.Sheet, headers: list[str], rows: list[dict]) -> None:
    """Clear sheet and write a header row followed by data rows."""
    sheet.clear()
    sheet["A1"].value = headers
    if rows:
        data = [[r.get(col, "") for col in headers] for r in rows]
        sheet["A2"].value = data


def _read_table(sheet: xw.Sheet) -> list[list[Any]]:
    """
    Read all used rows from a sheet as a list of lists (header row included).
    Returns an empty list when the sheet is blank.
    """
    used = sheet.used_range
    if used is None:
        return []
    values = used.value
    if values is None:
        return []
    # used_range.value returns a single list (not nested) when only one row.
    if values and not isinstance(values[0], list):
        values = [values]
    return values


HEDGE_CONFIG_SHEET = "HedgeConfig"
HEDGE_SCHEDULE_SHEET = "HedgeSchedule"
HEDGE_SUMMARY_SHEET = "HedgeSummary"
HEDGE_PATHS_SHEET = "HedgePaths"
DEFAULT_CONFIG_SHEET = "Config"
DEFAULT_HEDGE_JOB_CELL = "B1"


def _write_param_rows(sheet: xw.Sheet, params: list[tuple[str, Any]]) -> None:
    sheet.clear()
    sheet["A1"].value = [["Param", "Value"]] + [[key, value] for key, value in params]
    sheet.autofit()


def _read_param_sheet(sheet: xw.Sheet) -> dict[str, Any]:
    raw = _read_table(sheet)
    if not raw:
        return {}
    return {
        str(row[0]).strip(): row[1]
        for row in raw[1:]
        if row and len(row) >= 2 and str(row[0]).strip()
    }


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _normalize_path_text(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text


def _resolve_hedge_job_path(wb: xw.Book, job_path: str = "") -> str:
    if job_path and str(job_path).strip():
        return _normalize_path_text(job_path)
    state = _load_state()
    if state.get("hedge_job_path"):
        return _normalize_path_text(state["hedge_job_path"])
    try:
        config_value = wb.sheets[DEFAULT_CONFIG_SHEET][DEFAULT_HEDGE_JOB_CELL].value
        if config_value and str(config_value).strip():
            return _normalize_path_text(config_value)
    except Exception:
        pass
    try:
        return _normalize_path_text(wb.sheets[HEDGE_CONFIG_SHEET]["B2"].value)
    except Exception:
        return ""


def _hedge_config_rows(job_path: str, cfg: dict[str, Any]) -> list[tuple[str, Any]]:
    calc = cfg["Calc"]["Calculation"]
    liabilities = calc["Hedging_Problem"]["Liabilities"]["FloatingEnergyDeal"]
    liability_name = next(iter(liabilities))
    item = liabilities[liability_name]["Payments"]["Items"][0]
    policy_order = ", ".join(
        calc["Hedging_Problem"].get("Policy", {}).get("Action_Space", {}).get("Instrument_Order", [])
    )
    return [
        ("liability_name", liability_name),
        ("strike", -float(item.get("Fixed_Basis", 0.0))),
        ("batch_size", calc.get("Batch_Size", "")),
        ("seed", calc.get("Random_Seed", "")),
        ("output_root", "artifacts/textbook_runs"),
        ("tag", "excel"),
        ("instrument_order", policy_order),
    ]


def _write_default_config_sheet(wb: xw.Book, job_path: str) -> None:
    sheet = _get_or_create_sheet(wb, DEFAULT_CONFIG_SHEET)
    if not sheet["A1"].value:
        sheet["A1"].value = "job_path"
    sheet[DEFAULT_HEDGE_JOB_CELL].value = job_path
    sheet.autofit()


def _read_hedge_config(wb: xw.Book) -> dict[str, Any]:
    if HEDGE_CONFIG_SHEET not in [sheet.name for sheet in wb.sheets]:
        raise ValueError("RF_LOAD_HEDGE_JSON must be run before schedule generation.")
    params = _read_param_sheet(wb.sheets[HEDGE_CONFIG_SHEET])
    job_path = _resolve_hedge_job_path(wb)
    if not job_path:
        raise ValueError("Config!B1 is missing job_path.")
    return {
        "job_path": job_path,
        "liability_name": str(params.get("liability_name", "")).strip() or None,
        "strike": _coerce_optional_float(params.get("strike")),
        "batch_size": _coerce_optional_int(params.get("batch_size")),
        "seed": _coerce_optional_int(params.get("seed")),
        "output_root": str(params.get("output_root", "artifacts/textbook_runs")).strip() or "artifacts/textbook_runs",
        "tag": str(params.get("tag", "excel")).strip(),
    }


def _write_filterable_table(sheet: xw.Sheet, headers: list[str], rows: list[dict[str, Any]]) -> None:
    _write_table(sheet, headers, rows)
    row_count = max(len(rows), 1) + 1
    col_count = max(len(headers), 1)
    table_range = sheet.range((1, 1), (row_count, col_count))
    try:
        table_range.api.AutoFilter()
    except Exception as exc:
        logging.warning("Could not apply AutoFilter on sheet %s: %s", sheet.name, exc)
    sheet.autofit()


def _highlight_schedule_rows(sheet: xw.Sheet, row_count: int, col_count: int) -> None:
    for row_idx in range(2, row_count + 2):
        is_business_day = bool(sheet.range((row_idx, 2)).value)
        is_decision_step = bool(sheet.range((row_idx, 3)).value)
        row_range = sheet.range((row_idx, 1), (row_idx, col_count))
        if not is_business_day:
            row_range.color = (235, 235, 235)
        elif is_decision_step:
            row_range.color = (226, 239, 218)
        else:
            row_range.color = (255, 242, 204)


def _read_schedule_rows(sheet: xw.Sheet) -> list[dict[str, Any]]:
    raw = _read_table(sheet)
    if not raw:
        return []
    headers = [str(cell).strip() for cell in raw[0]]
    rows: list[dict[str, Any]] = []
    for raw_row in raw[1:]:
        row = {headers[idx]: raw_row[idx] for idx in range(min(len(headers), len(raw_row)))}
        if not str(row.get("date", "")).strip():
            continue
        rows.append(row)
    return rows


def _read_csv_rows(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_output_root(wb: xw.Book, output_root: str) -> Path:
    root = Path(output_root)
    if root.is_absolute():
        return root

    workbook_dir = None
    try:
        if wb.fullname:
            workbook_dir = Path(wb.fullname).resolve().parent
    except Exception:
        workbook_dir = None

    if workbook_dir is not None:
        return workbook_dir / root
    return Path.cwd() / root


def _summary_rows(out_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    metric_values: dict[str, Any] = {
        "mean": summary["mean"],
        "median": summary["median"],
        "min": summary["min"],
        "std": summary["std"],
    }

    summary_csv = out_dir / "excel_summary.csv"
    if summary_csv.exists():
        for row in _read_csv_rows(str(summary_csv)):
            metric = str(row.get("metric", "")).strip()
            if metric:
                metric_values[metric] = row.get("net_pnl", "")

    ordered_metrics = ["output_dir", "mean", "median", "std", "min", "p5", "p95", "max"]
    rows = [{"metric": "output_dir", "value": str(out_dir)}]
    for metric in ordered_metrics[1:]:
        if metric in metric_values:
            rows.append({"metric": metric, "value": metric_values[metric]})

    for metric, value in metric_values.items():
        if metric not in ordered_metrics[1:]:
            rows.append({"metric": metric, "value": value})
    return rows


@xw.sub
def RF_LOAD_HEDGE_JSON(job_path: str = "") -> None:
    wb = xw.Book.caller()
    resolved_job_path = _resolve_hedge_job_path(wb, job_path) or hedge_runner.FIXTURE
    if not resolved_job_path:
        raise ValueError("RF_LOAD_HEDGE_JSON: provide job_path or populate HedgeConfig!B2.")

    cfg = hedge_runner.load_job_config(resolved_job_path)
    _write_default_config_sheet(wb, resolved_job_path)
    sheet = _get_or_create_sheet(wb, HEDGE_CONFIG_SHEET)
    _write_param_rows(sheet, _hedge_config_rows(resolved_job_path, cfg))
    _save_state({**_load_state(), "hedge_job_path": resolved_job_path})


@xw.sub
def RF_BUILD_HEDGE_SCHEDULE() -> None:
    wb = xw.Book.caller()
    config = _read_hedge_config(wb)
    cfg = hedge_runner.load_simulation_config(
        config["job_path"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        strike=config["strike"],
        liability_name=config["liability_name"],
    )
    result = hedge_runner.run_simulation_config(cfg)
    rows = hedge_runner.build_schedule_rows(result)
    headers = ["date", "is_business_day", "is_decision_step"] + hedge_runner.runtime_instrument_order(result)
    sheet = _get_or_create_sheet(wb, HEDGE_SCHEDULE_SHEET)
    _write_filterable_table(sheet, headers, rows)
    _highlight_schedule_rows(sheet, len(rows), len(headers))


@xw.sub
def RF_RUN_HEDGE_SCHEDULE() -> None:
    wb = xw.Book.caller()
    config = _read_hedge_config(wb)
    if HEDGE_SCHEDULE_SHEET not in [sheet.name for sheet in wb.sheets]:
        raise ValueError("RF_BUILD_HEDGE_SCHEDULE must be run before RF_RUN_HEDGE_SCHEDULE.")

    cfg = hedge_runner.load_simulation_config(
        config["job_path"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        strike=config["strike"],
        liability_name=config["liability_name"],
    )
    result = hedge_runner.run_simulation_config(cfg)
    instruments = hedge_runner.runtime_instrument_order(result)
    schedule_rows = _read_schedule_rows(wb.sheets[HEDGE_SCHEDULE_SHEET])
    targets = hedge_runner.position_targets_from_rows(schedule_rows, instruments)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{config['tag']}" if config["tag"] else ""
    out_dir = _resolve_output_root(wb, config["output_root"]) / f"{timestamp}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stepper, last = hedge_runner.run_position_target_schedule(result, targets)
    stepper.write_diagnostic_csvs(str(out_dir), label="excel")
    summary = hedge_runner.summarize_terminal_pnl(last)
    summary_rows = _summary_rows(out_dir, summary)

    summary_sheet = _get_or_create_sheet(wb, HEDGE_SUMMARY_SHEET)
    _write_filterable_table(summary_sheet, ["metric", "value"], summary_rows)

    paths_csv = out_dir / "excel_paths.csv"
    if paths_csv.exists():
        path_rows = _read_csv_rows(str(paths_csv))
        path_headers = list(path_rows[0].keys()) if path_rows else ["case"]
    else:
        path_rows = []
        path_headers = ["case"]
    paths_sheet = _get_or_create_sheet(wb, HEDGE_PATHS_SHEET)
    _write_filterable_table(paths_sheet, path_headers, path_rows)


@xw.sub
def RF_LOAD_PORTFOLIO(job_path: str = "") -> None:
    """
    Load a RiskFlow JSON file and populate three sheets:
      • Portfolio    — one row per deal (flat hierarchy with Fields_JSON)
      • RiskFactors  — one row per price factor
      • Calculations — Param/Value table for the calculation section

    Macro usage (ribbon button): leave job_path blank and the macro reads cell
    A1 of a sheet named 'Config' (or prompts with a message box).
    """
    import riskflow as rf

    wb = xw.Book.caller()

    # Resolve job_path: fall back to a Config sheet cell or raise.
    if not job_path or not str(job_path).strip():
        try:
            job_path = str(wb.sheets["Config"]["B1"].value).strip()
        except Exception:
            pass
    if not job_path or not str(job_path).strip():
        raise ValueError(
            "RF_LOAD_PORTFOLIO: provide job_path argument or put the path in Config!B1."
        )

    logging.info("RF_LOAD_PORTFOLIO: loading %s", job_path)
    ctx = rf.Context()
    ctx.load_json(job_path)
    cfg = ctx.current_cfg

    # ── Portfolio sheet ──────────────────────────────────────────────────────
    port_rows = flatten_portfolio(cfg.deals["Deals"])
    port_sheet = _get_or_create_sheet(wb, "Portfolio")
    _write_table(port_sheet, PORTFOLIO_COLS, port_rows)

    # ── RiskFactors sheet ────────────────────────────────────────────────────
    rf_rows = flatten_risk_factors(cfg.params.get("Price Factors", {}))
    rf_sheet = _get_or_create_sheet(wb, "RiskFactors")
    _write_table(rf_sheet, RISKFACTOR_COLS, rf_rows)

    # ── Calculations sheet ───────────────────────────────────────────────────
    calc_rows = flatten_calc_params(cfg.deals.get("Calculation", {}))
    calc_sheet = _get_or_create_sheet(wb, "Calculations")
    _write_table(calc_sheet, CALC_COLS, calc_rows)

    # Store the market data path so price/solve macros can find it.
    try:
        import json as _json
        md_path = list(ctx.config_cache.keys())[0] if ctx.config_cache else job_path
    except Exception:
        md_path = job_path
    _save_state({"loaded_job": job_path, "market_data_path": md_path})

    logging.info(
        "RF_LOAD_PORTFOLIO: wrote %d deals, %d factors",
        len(port_rows),
        len(rf_rows),
    )


@xw.sub
def RF_SAVE_PORTFOLIO(job_path: str = "") -> None:
    """
    Read the Portfolio + RiskFactors + Calculations sheets and save a new job JSON.
    The market data file reference is preserved (only deals + calc section change).
    """
    wb = xw.Book.caller()

    if not job_path or not str(job_path).strip():
        try:
            job_path = str(wb.sheets["Config"]["B2"].value).strip()
        except Exception:
            pass
    if not job_path or not str(job_path).strip():
        raise ValueError(
            "RF_SAVE_PORTFOLIO: provide job_path argument or put the output path in Config!B2."
        )

    state = _load_state()
    market_data_path = state.get("market_data_path", "")
    if not market_data_path:
        raise ValueError(
            "RF_SAVE_PORTFOLIO: no market data path in state — run RF_LOAD_PORTFOLIO first."
        )

    portfolio_rows, calc_params = _read_portfolio_and_calc(wb)

    # Optionally apply risk factor edits.
    import riskflow as rf
    ctx = rf.Context()
    ctx.load_json(market_data_path)
    rf_raw = _read_table(wb.sheets["RiskFactors"])[1:]  # skip header
    update_risk_factors_from_rows(ctx.current_cfg.params["Price Factors"], rf_raw)

    job_json = build_portfolio_job_json(
        market_data_path,
        portfolio_rows,
        calc_params,
        attributes=state.get("attributes"),
    )

    Path(job_path).write_text(job_json, encoding="utf-8")
    logging.info("RF_SAVE_PORTFOLIO: saved to %s", job_path)


@xw.sub
def RF_PRICE_PORTFOLIO(overrides_json: str = "") -> None:
    """
    Price the portfolio defined in the Portfolio + Calculations sheets.
    Results are written to a 'Results' sheet. Market data path is taken from
    the last RF_LOAD_PORTFOLIO call (stored in .rf_excel_state.json).
    """
    wb = xw.Book.caller()
    state = _load_state()
    market_data_path = state.get("market_data_path", "")
    if not market_data_path:
        raise ValueError("RF_PRICE_PORTFOLIO: run RF_LOAD_PORTFOLIO first.")

    portfolio_rows, calc_params = _read_portfolio_and_calc(wb)

    overrides: dict | None = None
    if overrides_json and str(overrides_json).strip():
        overrides = json.loads(overrides_json)

    job_json = build_portfolio_job_json(market_data_path, portfolio_rows, calc_params)
    payload: dict[str, Any] = {"job_json": job_json}
    if overrides:
        payload["overrides"] = overrides

    result = price_job(payload)
    _save_state(result)
    _write_results(wb, result)


@xw.sub
def RF_SOLVE_PORTFOLIO(solve_spec_json: str = "", overrides_json: str = "") -> None:
    """
    Run a goal-seek solve on the portfolio.

    solve_spec_json — JSON string matching the solve_spec schema:
        {
          "method": "brentq",
          "variables": [{"name": "strike", "path": "/Calc/Deals/...", "initial": 100, "lower": 50, "upper": 200}],
          "targets":   [{"name": "net_mtm", "metric": "net_mtm", "target": 0.0}]
        }

    If solve_spec_json is blank the macro reads from the SolveSpec sheet
    (a 2-column Param | Value table with a "solve_spec_json" row).
    """
    wb = xw.Book.caller()
    state = _load_state()
    market_data_path = state.get("market_data_path", "")
    if not market_data_path:
        raise ValueError("RF_SOLVE_PORTFOLIO: run RF_LOAD_PORTFOLIO first.")

    # Read solve spec from sheet if not provided.
    if not solve_spec_json or not str(solve_spec_json).strip():
        try:
            ss_raw = _read_table(wb.sheets["SolveSpec"])
            ss_params = {str(r[0]): r[1] for r in ss_raw[1:] if r and r[0]}
            solve_spec_json = str(ss_params.get("solve_spec_json", "")).strip()
        except Exception:
            pass
    if not solve_spec_json or not str(solve_spec_json).strip():
        raise ValueError(
            "RF_SOLVE_PORTFOLIO: provide solve_spec_json or populate the SolveSpec sheet."
        )

    portfolio_rows, calc_params = _read_portfolio_and_calc(wb)

    overrides: dict | None = None
    if overrides_json and str(overrides_json).strip():
        overrides = json.loads(overrides_json)

    job_json = build_portfolio_job_json(market_data_path, portfolio_rows, calc_params)
    payload: dict[str, Any] = {
        "job_json": job_json,
        "solve_spec": json.loads(solve_spec_json),
    }
    if overrides:
        payload["overrides"] = overrides

    result = price_job(payload)
    _save_state(result)
    _write_results(wb, result)


# ---------------------------------------------------------------------------
# Helper: read Portfolio + Calculations sheets into Python objects
# ---------------------------------------------------------------------------

def _read_portfolio_and_calc(
    wb: xw.Book,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Read the Portfolio and Calculations sheets and return
    (portfolio_row_dicts, calc_params_dict).
    """
    # Portfolio: first row = headers.
    port_raw = _read_table(wb.sheets["Portfolio"])
    headers = [str(h).strip() for h in port_raw[0]] if port_raw else PORTFOLIO_COLS
    portfolio_rows: list[dict[str, Any]] = []
    for raw_row in port_raw[1:]:
        obj = str(raw_row[headers.index("Object")] if "Object" in headers else "").strip()
        if not obj:
            continue
        row = {h: raw_row[i] for i, h in enumerate(headers) if i < len(raw_row)}
        portfolio_rows.append(row)

    # Calculations.
    calc_raw = _read_table(wb.sheets["Calculations"])
    calc_params = unflatten_calc_params(calc_raw[1:] if calc_raw else [])

    return portfolio_rows, calc_params


# ---------------------------------------------------------------------------
# Helper: write results to the Results sheet
# ---------------------------------------------------------------------------

def _write_results(wb: xw.Book, result: dict[str, Any]) -> None:
    """Write a price/solve result dict to the Results sheet."""
    results_sheet = _get_or_create_sheet(wb, "Results")
    results_sheet.clear()

    rows: list[list[Any]] = [
        ["status", result.get("status", "")],
        ["mode", result.get("mode", "")],
        ["elapsed_ms", result.get("elapsed_ms", "")],
    ]

    if result.get("status") == "error":
        rows.append(["error", result.get("error", "")])
    else:
        summary = result.get("summary", {})
        for k, v in summary.items():
            rows.append([k, json.dumps(v, default=str) if isinstance(v, (dict, list)) else v])

        if result.get("mode") == "solve":
            rows.append(["── solution ──", ""])
            for k, v in (result.get("solution") or {}).items():
                rows.append([k, v])
            rows.append(["residual_norm", result.get("residual_norm", "")])
            rows.append(["── targets ──", ""])
            for k, v in (result.get("targets") or {}).items():
                rows.append([k, v])

    results_sheet["A1"].value = rows


# ---------------------------------------------------------------------------
# Read-only info UDFs (worksheet formulas)
# ---------------------------------------------------------------------------

@xw.func
@xw.ret(expand="table")
def RF_GET_DEAL_TYPES() -> list[list[str]]:
    """
    Return all known instrument types as a vertical list (for data validation).
    Call from a single cell; the result will spill downward.
    """
    return [[t] for t in all_instrument_types()]


@xw.func
@xw.ret(expand="table")
def RF_GET_DEAL_TYPES_GROUPED() -> list[list[str]]:
    """
    Return all instrument types as a two-column list: Group | Type.
    Useful for building a categorised dropdown via Excel data validation.
    """
    rows: list[list[str]] = []
    for group, types in instrument_types_grouped().items():
        for t in types:
            rows.append([group, t])
    return rows


@xw.func
@xw.arg("instrument_type", doc="RiskFlow instrument type name, e.g. 'FRADeal'")
def RF_GET_DEAL_DEFAULTS(instrument_type: str) -> str:
    """
    Return a Fields_JSON string with schema-default values for the given
    instrument type.  Paste the result into the Fields_JSON column of the
    Portfolio sheet to seed a new row.
    """
    return get_deal_defaults(str(instrument_type).strip())
