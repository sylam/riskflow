"""
Portfolio service for the Excel integration.

Handles the bidirectional translation between:
  - RiskFlow's nested deal tree (Deal objects, Curve, Percent, etc.)
  - Flat Excel table rows with a Fields_JSON column for full fidelity.

Sheet layouts expected / produced:
  Portfolio   : Level | Object | Reference | Ignore | Currency | Buy_Sell |
                Effective_Date | Maturity_Date | Discount_Rate | Fields_JSON
  RiskFactors : Factor_Name | Fields_JSON
  Calculations: Param | Value
"""
from __future__ import annotations

import json
import logging
import operator
from datetime import datetime
from functools import reduce
from typing import Any

import pandas as pd
import riskflow as rf
from riskflow.config import Config, CustomJsonEncoder
from riskflow.instruments import construct_instrument

# ---------------------------------------------------------------------------
# Sheet column layouts (order matters — used for Excel range reads/writes)
# ---------------------------------------------------------------------------

PORTFOLIO_COLS: list[str] = [
    "Level", "Object", "Reference", "Ignore",
    "Currency", "Buy_Sell", "Effective_Date", "Maturity_Date",
    "Discount_Rate", "Fields_JSON",
]

RISKFACTOR_COLS: list[str] = ["Factor_Name", "Fields_JSON"]

CALC_COLS: list[str] = ["Param", "Value"]

# Fields whose values are extracted into dedicated Excel columns AND kept
# inside Fields_JSON for completeness.
_STANDARD_FIELD_NAMES: frozenset[str] = frozenset(
    {"Currency", "Buy_Sell", "Effective_Date", "Maturity_Date", "Discount_Rate"}
)

# Widget types that produce complex JSON blobs (not suitable for plain cells).
_COMPLEX_WIDGETS: frozenset[str] = frozenset(
    {"Flot", "Three", "Table", "Container"}
)

# ---------------------------------------------------------------------------
# Lazily-created singleton Config instance (needed for periodparser).
# ---------------------------------------------------------------------------
_CFG: Config | None = None


def _get_cfg() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config()
    return _CFG


# ---------------------------------------------------------------------------
# JSON encode / decode helpers
# ---------------------------------------------------------------------------

def _make_as_internal():
    """Return an object_hook that decodes riskflow JSON special types."""
    cfg = _get_cfg()

    def as_internal(dct: dict) -> Any:
        if ".Curve" in dct:
            return rf.utils.Curve(dct[".Curve"]["meta"], dct[".Curve"]["data"])
        if ".Percent" in dct:
            return rf.utils.Percent(dct[".Percent"])
        if ".Deal" in dct:
            # Re-construct the instrument from its encoded field dict.
            return construct_instrument(dct[".Deal"], {})
        if ".Basis" in dct:
            return rf.utils.Basis(dct[".Basis"])
        if ".Descriptor" in dct:
            return rf.utils.Descriptor(dct[".Descriptor"])
        if ".DateList" in dct:
            return rf.utils.DateList(
                {pd.Timestamp(d): v for d, v in dct[".DateList"]}
            )
        if ".DateEqualList" in dct:
            return rf.utils.DateEqualList(
                [[pd.Timestamp(v[0])] + v[1:] for v in dct[".DateEqualList"]]
            )
        if ".CreditSupportList" in dct:
            return rf.utils.CreditSupportList(dct[".CreditSupportList"])
        if ".DateOffset" in dct and ".Offset" in dct:
            return [
                cfg.periodparser.parseString(dct[".DateOffset"])[0],
                cfg.periodparser.parseString(dct[".Offset"])[0],
            ]
        if ".DateOffset" in dct:
            return cfg.periodparser.parseString(dct[".DateOffset"])[0]
        if ".Grid" in dct:
            return rf.utils.Offsets(
                [(x if isinstance(x, list) else [x]) for x in dct[".Grid"]]
            )
        if ".Timestamp" in dct:
            return pd.Timestamp(dct[".Timestamp"])
        return dct

    return as_internal


def decode_fields_json(fields_json: str) -> dict[str, Any]:
    """Decode a Fields_JSON string to a riskflow-compatible field dict."""
    return json.loads(fields_json, object_hook=_make_as_internal())


def encode_fields_json(field_dict: dict[str, Any]) -> str:
    """Encode a field dict to a compact JSON string via CustomJsonEncoder."""
    return json.dumps(field_dict, separators=(",", ":"), cls=CustomJsonEncoder)


# ---------------------------------------------------------------------------
# Cell-value conversions
# ---------------------------------------------------------------------------

def _to_cell(v: Any) -> Any:
    """Convert a riskflow field value to an Excel-friendly scalar or JSON string."""
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, rf.utils.Percent):
        return v.amount
    if isinstance(v, rf.utils.Basis):
        return v.amount
    if isinstance(v, (str, int, float, bool)):
        return v
    if v is None:
        return ""
    # Complex type (Curve, DateList, list, etc.) → JSON string.
    return json.dumps(v, separators=(",", ":"), cls=CustomJsonEncoder)


# ---------------------------------------------------------------------------
# Flatten: deal tree → Excel rows
# ---------------------------------------------------------------------------

def _flatten_node(node: dict, level: int, rows: list[dict]) -> None:
    """Recursively emit one row dict per deal node."""
    instrument = node.get("Instrument")
    if instrument is None:
        return

    field: dict = instrument.field if hasattr(instrument, "field") else {}

    row: dict[str, Any] = {
        "Level": level,
        "Object": field.get("Object", ""),
        "Reference": field.get("Reference", ""),
        "Ignore": node.get("Ignore", "False"),
        "Currency": _to_cell(field.get("Currency", "")),
        "Buy_Sell": _to_cell(field.get("Buy_Sell", "")),
        "Effective_Date": _to_cell(field.get("Effective_Date")),
        "Maturity_Date": _to_cell(field.get("Maturity_Date")),
        "Discount_Rate": _to_cell(field.get("Discount_Rate", "")),
        "Fields_JSON": encode_fields_json(field),
    }
    rows.append(row)

    for child in node.get("Children", []):
        _flatten_node(child, level + 1, rows)


def flatten_portfolio(deals_root: dict) -> list[dict]:
    """
    Flatten the nested deal tree to a list of row dicts (one per deal).

    :param deals_root: ctx.current_cfg.deals['Deals'] — a dict with 'Children'.
    :returns: list of row dicts keyed by PORTFOLIO_COLS.
    """
    rows: list[dict] = []
    for child in deals_root.get("Children", []):
        _flatten_node(child, 0, rows)
    return rows


# ---------------------------------------------------------------------------
# Unflatten: Excel rows → deal tree
# ---------------------------------------------------------------------------

def _apply_standard_overrides(
    field_dict: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge non-empty standard column values back into the decoded field dict.
    Standard columns win over what was in Fields_JSON for those keys.
    """
    result = dict(field_dict)

    for key in ("Currency", "Buy_Sell", "Discount_Rate"):
        val = row.get(key)
        if val is not None and str(val).strip():
            result[key] = str(val).strip()

    for date_key in ("Effective_Date", "Maturity_Date"):
        val = row.get(date_key)
        if val is not None and val != "":
            try:
                result[date_key] = pd.Timestamp(val)
            except Exception:
                pass

    return result


def unflatten_portfolio(
    rows: list[dict[str, Any]], val_config: dict
) -> dict:
    """
    Reconstruct the nested deal structure from a list of portfolio row dicts.

    :param rows: output of flatten_portfolio (or built from Excel range).
    :param val_config: ctx.current_cfg.params.get('Valuation Configuration', {})
    :returns: dict with 'Children' list — suitable for ctx.current_cfg.deals['Deals'].
    """
    root: dict[str, Any] = {"Children": []}
    # Stack entries: (level, node_dict_with_Children)
    stack: list[tuple[int, dict]] = [(-1, root)]

    for row in rows:
        obj = str(row.get("Object") or "").strip()
        if not obj:
            continue

        level = int(row.get("Level") or 0)
        fields_json = str(row.get("Fields_JSON") or "").strip()

        # Decode Fields_JSON → base field dict.
        if fields_json and fields_json not in ("{}", ""):
            try:
                field_dict = decode_fields_json(fields_json)
            except Exception as exc:
                logging.warning("Cannot decode Fields_JSON for %s: %s", obj, exc)
                field_dict = {}
        else:
            # No Fields_JSON: seed from defaults.
            field_dict = {}

        # Set canonical keys.
        field_dict["Object"] = obj
        field_dict["Reference"] = str(row.get("Reference") or "").strip()

        # Allow standard columns to override Fields_JSON values.
        field_dict = _apply_standard_overrides(field_dict, row)

        # Fill missing fields from schema defaults if Fields_JSON was absent.
        if not fields_json or fields_json in ("{}", ""):
            try:
                defaults = decode_fields_json(get_deal_defaults(obj))
                for k, v in defaults.items():
                    field_dict.setdefault(k, v)
            except Exception:
                pass

        # Construct the Deal instrument object.
        try:
            instrument = construct_instrument(field_dict, val_config)
        except Exception as exc:
            logging.warning("construct_instrument failed for %s: %s", obj, exc)
            instrument = {}

        is_group = obj in ("NettingCollateralSet", "StructuredDeal")
        node: dict[str, Any] = {"Instrument": instrument}

        ignore_val = str(row.get("Ignore") or "False").strip()
        if ignore_val.lower() == "true":
            node["Ignore"] = "True"

        if is_group:
            node["Children"] = []

        # Pop stack until we reach the correct parent level.
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()

        stack[-1][1]["Children"].append(node)

        if is_group:
            stack.append((level, node))

    return root


# ---------------------------------------------------------------------------
# Deal schema helpers
# ---------------------------------------------------------------------------

def get_deal_defaults(instrument_type: str) -> str:
    """
    Return a Fields_JSON string with schema-default values for given instrument type.
    Only scalar/simple fields are included (complex widgets are skipped).
    """
    instr_mapping = rf.fields.mapping["Instrument"]
    section_names: list[str] = instr_mapping["types"].get(instrument_type, [])
    all_fields: dict = instr_mapping["fields"]

    defaults: dict[str, Any] = {"Object": instrument_type}
    for section in section_names:
        for fname in instr_mapping["sections"].get(section, []):
            meta = all_fields.get(fname)
            if meta and meta.get("widget") not in _COMPLEX_WIDGETS:
                defaults[fname] = meta.get("value", "")

    return encode_fields_json(defaults)


def instrument_types_grouped() -> dict[str, list[str]]:
    """Return all instrument types keyed by their group name."""
    return {
        group_name: sorted(type_list)
        for group_name, (_, type_list) in rf.fields.mapping["Instrument"]["groups"].items()
    }


def all_instrument_types() -> list[str]:
    """Return a sorted flat list of every known instrument type."""
    types: set[str] = set()
    for _, type_list in rf.fields.mapping["Instrument"]["groups"].values():
        types.update(type_list)
    return sorted(types)


# ---------------------------------------------------------------------------
# Calculation params helpers
# ---------------------------------------------------------------------------

def flatten_calc_params(calc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert the calculation dict to [{Param, Value}, ...] rows for Excel.
    Container sub-dicts are serialised as compact JSON strings.
    """
    rows: list[dict[str, Any]] = []
    for k, v in calc.items():
        if isinstance(v, pd.Timestamp):
            rows.append({"Param": k, "Value": v.strftime("%Y-%m-%d")})
        elif isinstance(v, dict):
            rows.append(
                {"Param": k, "Value": json.dumps(v, separators=(",", ":"), cls=CustomJsonEncoder)}
            )
        else:
            rows.append({"Param": k, "Value": str(v) if v is not None else ""})
    return rows


def unflatten_calc_params(raw_rows: list[list[Any]]) -> dict[str, Any]:
    """
    Reconstruct the calculation dict from [[Param, Value], ...] Excel rows.
    Header row (first row) is skipped if Param == 'Param'.
    """
    decoder = _make_as_internal()
    result: dict[str, Any] = {}

    for row in raw_rows:
        if len(row) < 2:
            continue
        param = str(row[0]).strip() if row[0] is not None else ""
        value = row[1]
        if not param or param == "Param":
            continue

        # Type coercion.
        if isinstance(value, datetime):
            value = pd.Timestamp(value)
        elif isinstance(value, float) and value == int(value):
            value = int(value)
        elif isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                value = json.loads(value, object_hook=decoder)
            except Exception:
                pass  # keep as string

        result[param] = value

    return result


# ---------------------------------------------------------------------------
# Risk factor helpers
# ---------------------------------------------------------------------------

def flatten_risk_factors(price_factors: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten price_factors dict to [{Factor_Name, Fields_JSON}, ...] rows."""
    rows: list[dict[str, Any]] = []
    for name, data in price_factors.items():
        rows.append(
            {
                "Factor_Name": name,
                "Fields_JSON": json.dumps(
                    data, separators=(",", ":"), cls=CustomJsonEncoder
                ),
            }
        )
    return rows


def update_risk_factors_from_rows(
    price_factors: dict[str, Any], raw_rows: list[list[Any]]
) -> None:
    """
    Write edited risk factor rows back into price_factors in-place.

    :param price_factors: ctx.current_cfg.params['Price Factors']
    :param raw_rows: [[Factor_Name, Fields_JSON], ...] from the RiskFactors sheet.
    """
    decoder = _make_as_internal()
    for row in raw_rows:
        if len(row) < 2:
            continue
        name = str(row[0]).strip() if row[0] else ""
        fields_json = str(row[1]).strip() if row[1] else ""
        if name in ("Factor_Name", "") or not fields_json:
            continue
        if name in price_factors:
            try:
                price_factors[name] = json.loads(fields_json, object_hook=decoder)
            except Exception as exc:
                logging.warning("Could not update risk factor %r: %s", name, exc)
