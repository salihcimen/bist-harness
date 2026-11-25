import pandas as pd

from bist.tools import store
from bist.tools.disclosures import analyze_disclosure_reactions
from conftest import assert_causal


def test_q4_tuprs_rows_by_publication_date():
    r = analyze_disclosure_reactions(ticker="TUPRS", start="2024-01-01", end="2026-09-10", as_of="2026-09-10")
    assert r["status"] != "ERROR", r["error"]
    d = r["data"]
    assert d["metrics"] == ["ebitda_margin_q", "net_debt_ebitda"] and len(d["rows"]) >= 9
    assert all("2024-01-01" <= x["first_published"][:10] <= "2026-09-10" for x in d["rows"])
    last = d["rows"][-1]
    bars = store.stock_bars("TUPRS", pd.Timestamp("2026-09-10"))
    pos = bars.index.get_indexer([pd.Timestamp(last["effective_trade_date"])], method="bfill")[0]
    assert last["returns"]["20d"]["complete"] == (pos + 19 < len(bars))
    codes = {w["code"] for w in r["warnings"]}
    if not last["returns"]["20d"]["complete"]:
        assert last["returns"]["20d"]["stock"] is None and "INCOMPLETE_FORWARD_WINDOW" in codes
    assert d["association_only"] == "no causal claim" and "TMS29_RESTATED_VALUES" in codes
    assert set(d["summary"]) == {"1d", "5d", "20d"}
    assert all(x["returns"]["1d"]["abnormal_vs_sector"] is not None for x in d["rows"] if x["returns"]["1d"]["complete"])


def test_disclosure_after_as_of_invisible():
    d = analyze_disclosure_reactions(ticker="TUPRS", start="2024-01-01", as_of="2026-07-01")["data"]
    assert d["rows"] and all(x["effective_trade_date"] <= "2026-07-01" for x in d["rows"])


def test_unknown_metric():
    assert analyze_disclosure_reactions(ticker="AKBNK", metrics=["ebitda_margin_q"])["error"]["code"] == "UNKNOWN_METRIC"


def test_bank_default_metrics():
    d = analyze_disclosure_reactions(ticker="AKBNK", start="2024-01-01", as_of="2026-09-10")["data"]
    assert d["kind"] == "bank" and d["metrics"] == ["roe_ttm", "nii_assets"]


def test_disclosures_causal(truncate_store):
    assert_causal(analyze_disclosure_reactions, "2026-09-10", truncate_store, ticker="TUPRS", start="2024-01-01")
