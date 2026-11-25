"""Common contract for every deterministic tool.

Every tool is a pure function of (validated params, local data cache, as_of date) and returns the SAME envelope:

    {tool, schema_version, status: OK|WARN|ERROR, params, as_of, data, warnings[], error, provenance}

- Errors never raise to the caller: they come back as status=ERROR with a machine-readable code (the LLM must be able
  to read them and the training data contains such cases).
- Nothing after `as_of` is ever read (look-ahead guard); `as_of` defaults to the last available trading day.
- Output is deterministic: no wall-clock times; provenance carries a hash of the result so replays can be compared.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from datetime import date, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

SCHEMA_VERSION = "1.1"
REGISTRY: dict[str, dict] = {}


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def warn(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def jsonable(x: Any, nd: int = 6) -> Any:
    """numpy/pandas -> plain JSON types; NaN/inf -> None; floats rounded."""
    if isinstance(x, dict):
        return {str(k): jsonable(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v, nd) for v in x]
    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.strftime("%Y-%m-%d") if not isinstance(x, pd.Timestamp) or x.hour == x.minute == 0 else x.isoformat()
    if x is pd.NaT:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)
    return x


def result_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def tool(name: str, params_model: type[BaseModel], description: str) -> Callable:
    """Register `fn(params) -> (data, warnings, provenance_extra)` as an MCP-style tool."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(**kwargs) -> dict:
            base = {"tool": name, "schema_version": SCHEMA_VERSION}
            try:
                p = params_model(**kwargs)
            except ValidationError as e:
                msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
                return {**base, "status": "ERROR", "params": jsonable(kwargs), "as_of": None, "data": None,
                        "warnings": [], "error": {"code": "INVALID_PARAMS", "message": msg}, "provenance": None}
            pj = p.model_dump(mode="json")
            try:
                data, warnings, prov = fn(p)
            except ToolError as e:
                return {**base, "status": "ERROR", "params": pj, "as_of": None, "data": None, "warnings": [],
                        "error": {"code": e.code, "message": e.message}, "provenance": None}
            data = jsonable(data)
            prov = jsonable(prov or {})
            as_of = prov.pop("as_of", None)
            prov["result_hash"] = result_hash(data)
            return {**base, "status": "WARN" if warnings else "OK", "params": pj, "as_of": as_of, "data": data,
                    "warnings": warnings, "error": None, "provenance": prov}

        REGISTRY[name] = {"name": name, "description": description, "params_model": params_model, "fn": wrapper}
        return wrapper

    return deco
