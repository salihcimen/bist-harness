"""Export the frozen tool schemas to mcp/tool_schemas.json (+ compact LLM view).

python -m bist.tools.export_schemas            -> write the file
tests/test_schema_freeze.py                    -> fails if the live tools no longer match the frozen file
The training-data generator and the MCP server must read ONLY this file.
"""
from __future__ import annotations

import hashlib
import json

from ..paths import ROOT
from . import REGISTRY, SCHEMA_VERSION  # noqa: F401  (import registers all tools)

OUT = ROOT / "mcp" / "tool_schemas.json"
ORDER = ["get_stock_history", "get_index_history", "calculate_indicators", "detect_reversal_events", "find_support_resistance",
         "rank_sector_relative_strength", "test_catch_up_history", "test_weekday_pattern", "get_fundamental_metrics", "calculate_fundamental_change", "analyze_disclosure_reactions",
         "get_macro_context", "analyze_macro_sensitivity", "get_news_events", "get_speech_features", "generate_strategy_variants", "run_backtest",
         "check_data_quality", "build_evidence_bundle"]

ENVELOPE = {
    "type": "object",
    "description": "Every tool returns this envelope. status ERROR => data is null and error={code,message}. warnings are machine-readable codes the answer must respect.",
    "properties": {
        "tool": {"type": "string"}, "schema_version": {"type": "string"}, "status": {"enum": ["OK", "WARN", "ERROR"]},
        "params": {"type": "object"}, "as_of": {"type": ["string", "null"]}, "data": {"type": ["object", "null"]},
        "warnings": {"type": "array", "items": {"type": "object", "properties": {"code": {"type": "string"}, "message": {"type": "string"}}}},
        "error": {"type": ["object", "null"], "properties": {"code": {"type": "string"}, "message": {"type": "string"}}},
        "provenance": {"type": ["object", "null"]},
    },
}


def _resolve(schema: dict, defs: dict) -> dict:
    """Inline $ref and strip titles so the schema is self-contained and compact."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            return _resolve(defs[schema["$ref"].split("/")[-1]], defs)
        return {k: _resolve(v, defs) for k, v in schema.items() if k not in ("title", "$defs")}
    if isinstance(schema, list):
        return [_resolve(x, defs) for x in schema]
    return schema


def _type_str(s: dict) -> str:
    if "enum" in s:
        return "|".join(map(str, s["enum"]))
    if "anyOf" in s:
        parts = [_type_str(x) for x in s["anyOf"] if x.get("type") != "null"]
        return "|".join(parts) if parts else "any"
    t = s.get("type", "any")
    if t == "array":
        return f"list[{_type_str(s.get('items', {}))}]"
    if s.get("format") == "date":
        return "date"
    return t


def build() -> dict:
    full, compact = [], []
    for name in ORDER:
        reg = REGISTRY[name]
        raw = reg["params_model"].model_json_schema()
        inp = _resolve(raw, raw.get("$defs", {}))
        full.append({"name": name, "description": reg["description"], "input_schema": inp})
        params = {}
        for k, v in inp.get("properties", {}).items():
            e = {"type": _type_str(v), "required": k in inp.get("required", [])}
            if "default" in v and v["default"] is not None:
                e["default"] = v["default"]
            params[k] = e
        compact.append({"name": name, "description": reg["description"].split(".")[0], "params": params})
    body = {"schema_version": SCHEMA_VERSION, "tools": full, "compact_tools": compact, "envelope": ENVELOPE}
    body["schema_hash"] = hashlib.sha256(json.dumps({k: body[k] for k in ("tools", "envelope")}, sort_keys=True).encode()).hexdigest()[:16]
    return body


def main():
    missing = [n for n in ORDER if n not in REGISTRY] + [n for n in REGISTRY if n not in ORDER]
    assert not missing, f"registry/order mismatch: {missing}"
    b = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(b, indent=1, ensure_ascii=False), encoding="utf-8")
    full_chars = len(json.dumps(b["tools"])); comp_chars = len(json.dumps(b["compact_tools"]))
    print(f"wrote {OUT}: {len(b['tools'])} tools, hash {b['schema_hash']}, full {full_chars} chars (~{full_chars // 4} tok), compact {comp_chars} chars (~{comp_chars // 4} tok)")


if __name__ == "__main__":
    main()
