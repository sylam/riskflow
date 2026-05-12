from __future__ import annotations

import json
import os
import re
import tempfile
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import riskflow


def _to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None
        pd = None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return value.reset_index().to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    return str(value)


def _extract_summary(output: dict[str, Any]) -> dict[str, Any]:
    results = output.get("Results") if isinstance(output, dict) else None
    summary: dict[str, Any] = {}
    if isinstance(results, dict):
        for key in ("mtm", "exposure_profile"):
            if key in results:
                summary[key] = _to_jsonable(results[key])
    if "Stats" in output:
        summary["stats"] = _to_jsonable(output.get("Stats"))
    return summary


def _load_job_data(payload: dict[str, Any]) -> dict[str, Any]:
    job_path = payload.get("job_path")
    job_json = payload.get("job_json")

    if not job_path and not job_json:
        raise ValueError("Payload must include either 'job_path' or 'job_json'.")

    if job_path:
        with Path(job_path).open("rt", encoding="utf-8") as f:
            return json.load(f)

    if isinstance(job_json, str):
        return json.loads(job_json)

    if isinstance(job_json, dict):
        return deepcopy(job_json)

    raise ValueError("'job_json' must be a JSON string or dict.")


def _run_job_data(job_data: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("wt", suffix=".json", encoding="utf-8", delete=False) as tmp:
            json.dump(job_data, tmp, ensure_ascii=False, separators=(",", ":"), default=str)
            temp_path = tmp.name

        context = riskflow.Context()
        context.load_json(temp_path)
        _, out = context.run_job(overrides=overrides)
        return out
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _json_pointer_set(doc: Any, pointer: str, value: Any) -> None:
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer}")

    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]]
    if not tokens:
        raise ValueError("JSON pointer cannot target the entire document root")

    current = doc
    for token in tokens[:-1]:
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(f"Path token '{token}' not found in pointer '{pointer}'")
            current = current[token]
        else:
            raise KeyError(f"Cannot traverse token '{token}' in pointer '{pointer}'")

    last = tokens[-1]
    if isinstance(current, list):
        current[int(last)] = value
    elif isinstance(current, dict):
        if last not in current:
            raise KeyError(f"Final token '{last}' not found in pointer '{pointer}'")
        current[last] = value
    else:
        raise KeyError(f"Cannot set token '{last}' in pointer '{pointer}'")


def _get_path_value(data: Any, path: str) -> Any:
    if not path:
        raise ValueError("Target path cannot be empty")

    current = data
    token_pattern = re.compile(r"([^.\[\]]+)|(\[(\d+)\])")
    for segment in path.split("."):
        if segment == "":
            continue
        tokens = token_pattern.findall(segment)
        if not tokens:
            raise KeyError(f"Invalid path segment '{segment}' in path '{path}'")

        for name_token, bracket_token, index_token in tokens:
            if name_token:
                if not isinstance(current, dict) or name_token not in current:
                    raise KeyError(f"Path '{path}' missing key '{name_token}'")
                current = current[name_token]
            elif bracket_token:
                if not isinstance(current, list):
                    raise KeyError(f"Path '{path}' expected list before index {index_token}")
                current = current[int(index_token)]

    return current


def _evaluate_target(raw_out: dict[str, Any], target: dict[str, Any], json_out: dict[str, Any]) -> float:
    metric = str(target.get("metric", "")).strip().lower()
    if metric == "net_mtm":
        mtm = raw_out["Results"]["mtm"]
        if "Value" in mtm.columns and len(mtm.index) > 0:
            return float(mtm.iloc[0]["Value"])
        raise KeyError("Cannot evaluate metric 'net_mtm': Results.mtm['Value'] missing")

    path = target.get("path")
    if not path:
        raise ValueError("Each target must define either 'metric' or 'path'.")

    value = _get_path_value(json_out, str(path))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Target path '{path}' is not numeric (value={value})") from exc


def _prepare_solver_inputs(solve_spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    variables = solve_spec.get("variables") or []
    targets = solve_spec.get("targets") or []
    method = str(solve_spec.get("method", "least_squares")).strip().lower()

    if not variables:
        raise ValueError("solve_spec.variables must contain at least one variable")
    if not targets:
        raise ValueError("solve_spec.targets must contain at least one target")

    for idx, var in enumerate(variables):
        if "path" not in var:
            raise ValueError(f"Variable at index {idx} is missing required field 'path'")

    return variables, targets, method


def solve_job(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    request_id = payload.get("request_id", "")

    try:
        from scipy import optimize

        overrides = payload.get("overrides")
        if isinstance(overrides, str) and overrides.strip():
            overrides = json.loads(overrides)

        solve_spec = payload.get("solve_spec")
        if isinstance(solve_spec, str):
            solve_spec = json.loads(solve_spec)
        if not isinstance(solve_spec, dict):
            raise ValueError("Payload must include solve_spec as object or JSON string")

        base_job_data = _load_job_data(payload)
        variables, targets, method = _prepare_solver_inputs(solve_spec)

        x0 = [float(v.get("initial", 0.0)) for v in variables]
        lower_bounds = [float(v.get("lower", -1.0e12)) for v in variables]
        upper_bounds = [float(v.get("upper", 1.0e12)) for v in variables]
        scale = [float(v.get("scale", 1.0)) for v in variables]

        if len(variables) == 1 and method not in {"least_squares", "brentq"}:
            method = "brentq"
        elif len(variables) > 1 and method == "brentq":
            method = "least_squares"

        last_eval: dict[str, Any] = {}

        def evaluate(x_values: list[float]) -> list[float]:
            nonlocal last_eval
            candidate = deepcopy(base_job_data)
            solved_vars: dict[str, float] = {}
            for i, variable in enumerate(variables):
                solved_value = float(x_values[i])
                solved_vars[str(variable.get("name", f"x{i+1}"))] = solved_value
                _json_pointer_set(candidate, str(variable["path"]), solved_value)

            out = _run_job_data(candidate, overrides=overrides)
            json_out = _to_jsonable(out)

            residuals: list[float] = []
            target_values: dict[str, float] = {}
            for j, target in enumerate(targets):
                target_name = str(target.get("name", f"target{j+1}"))
                current_val = _evaluate_target(out, target, json_out)
                target_goal = float(target.get("target", 0.0))
                weight = float(target.get("weight", 1.0))
                residuals.append((current_val - target_goal) * weight)
                target_values[target_name] = current_val

            last_eval = {
                "variables": solved_vars,
                "target_values": target_values,
                "residuals": residuals,
                "raw_output": out,
            }
            return residuals

        if method == "brentq":
            if len(variables) != 1:
                raise ValueError("brentq requires exactly one variable")
            if len(targets) != 1:
                raise ValueError("brentq requires exactly one target")

            lb, ub = lower_bounds[0], upper_bounds[0]
            f_lb = evaluate([lb])[0]
            f_ub = evaluate([ub])[0]
            if f_lb == 0.0:
                x_sol = lb
            elif f_ub == 0.0:
                x_sol = ub
            elif f_lb * f_ub > 0:
                raise ValueError(
                    "brentq requires a sign change across bounds. Try wider bounds or use method='least_squares'."
                )
            else:
                x_sol = optimize.brentq(lambda x: evaluate([float(x)])[0], lb, ub)
            residuals = evaluate([float(x_sol)])
            solution = [float(x_sol)]
        else:
            res = optimize.least_squares(
                lambda x: evaluate([float(v) for v in x]),
                x0=x0,
                bounds=(lower_bounds, upper_bounds),
                x_scale=scale,
            )
            solution = [float(v) for v in res.x]
            residuals = evaluate(solution)

        solved_payload = deepcopy(base_job_data)
        solved_vars: dict[str, float] = {}
        for i, variable in enumerate(variables):
            var_name = str(variable.get("name", f"x{i+1}"))
            solved_vars[var_name] = solution[i]
            _json_pointer_set(solved_payload, str(variable["path"]), solution[i])

        solved_out = _run_job_data(solved_payload, overrides=overrides)

        return {
            "request_id": request_id,
            "status": "ok",
            "mode": "solve",
            "method": method,
            "elapsed_ms": int((time.time() - started) * 1000),
            "solution": solved_vars,
            "residual_norm": float(sum(r * r for r in residuals) ** 0.5),
            "targets": last_eval.get("target_values", {}),
            "summary": _extract_summary(solved_out),
            "result": _to_jsonable(solved_out),
        }

    except Exception as exc:
        return {
            "request_id": request_id,
            "status": "error",
            "mode": "solve",
            "elapsed_ms": int((time.time() - started) * 1000),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def build_portfolio_job_json(
    market_data_path: str,
    portfolio_rows: list[dict[str, Any]],
    calc_params: dict[str, Any],
    attributes: dict[str, Any] | None = None,
) -> str:
    """
    Build a complete RiskFlow job JSON string from portfolio rows + calc params.

    The produced string can be passed directly as ``job_json`` to ``price_job``
    or ``solve_job``, which will write it to a temp file and load it via the
    normal ``Context.load_json`` path (preserving all market data references).

    :param market_data_path: absolute path to the market data / full job JSON.
    :param portfolio_rows:   list of row dicts as produced by
                             ``portfolio_service.flatten_portfolio``.
    :param calc_params:      dict for the ``Calculation`` section (must include
                             at minimum ``{'Object': 'BaseValuation', ...}``).
    :param attributes:       optional ``{'Reference': ..., 'Tag_Titles': ...}``.
    :returns: compact JSON string suitable for ``price_job({'job_json': ...})``.
    """
    from .portfolio_service import unflatten_portfolio
    from riskflow.config import CustomJsonEncoder

    # Load just enough context to get the Valuation Configuration, which is
    # needed by construct_instrument.  Market data itself is referenced by path
    # and will be re-loaded when the job runs.
    ctx = riskflow.Context()
    ctx.load_json(market_data_path)
    val_config = ctx.current_cfg.params.get("Valuation Configuration", {})

    deals_root = unflatten_portfolio(portfolio_rows, val_config)

    attrs = attributes or {}
    job = {
        "Calc": {
            "Calculation": calc_params,
            "Deals": {
                "Tag_Titles": attrs.get("Tag_Titles", ""),
                "Reference": attrs.get("Reference", ""),
                "Deals": deals_root,
            },
            "MergeMarketData": {
                "MarketDataFile": market_data_path,
                "ExplicitMarketData": {},
            },
        }
    }

    return json.dumps(job, separators=(",", ":"), cls=CustomJsonEncoder)


def price_job(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("solve_spec") is not None:
        return solve_job(payload)

    started = time.time()
    request_id = payload.get("request_id", "")

    try:
        overrides = payload.get("overrides")
        if isinstance(overrides, str) and overrides.strip():
            overrides = json.loads(overrides)

        job_data = _load_job_data(payload)
        out = _run_job_data(job_data, overrides=overrides)

        return {
            "request_id": request_id,
            "status": "ok",
            "mode": "price",
            "elapsed_ms": int((time.time() - started) * 1000),
            "summary": _extract_summary(out),
            "result": _to_jsonable(out),
        }

    except Exception as exc:
        return {
            "request_id": request_id,
            "status": "error",
            "mode": "price",
            "elapsed_ms": int((time.time() - started) * 1000),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
