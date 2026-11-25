import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bist.tools import REGISTRY, SCHEMA_VERSION  # noqa: E402
from bist.tools.export_schemas import ORDER, OUT, build  # noqa: E402
from bist.tools.news import get_news_events, get_speech_features  # noqa: E402


def test_frozen_schema_matches_live_tools():
    frozen = json.loads(OUT.read_text(encoding="utf-8"))
    live = build()
    assert frozen["schema_version"] == SCHEMA_VERSION == "1.1"
    assert frozen["schema_hash"] == live["schema_hash"], "tool schemas changed: bump SCHEMA_VERSION and re-export (this invalidates training data!)"


def test_all_tools_registered_in_declared_order():
    assert set(REGISTRY) == set(ORDER) and len(ORDER) == 19


def test_every_tool_description_and_param_is_documented():
    for t in json.loads(OUT.read_text(encoding="utf-8"))["tools"]:
        assert len(t["description"]) > 20, t["name"]
        for k, v in t["input_schema"].get("properties", {}).items():
            assert v.get("description"), f"{t['name']}.{k} has no description"


def test_news_and_speech_report_missing_source_instead_of_inventing():
    r = get_news_events(ticker="THYAO")
    assert r["status"] == "ERROR" and r["error"]["code"] == "SOURCE_NOT_AVAILABLE"
    r = get_speech_features(source_id="abc")
    assert r["error"]["code"] == "SOURCE_NOT_AVAILABLE"
