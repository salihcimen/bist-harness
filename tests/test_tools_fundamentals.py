import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bist.tools import store  # noqa: E402
from bist.tools.fundamentals import calculate_fundamental_change, get_fundamental_metrics, get_macro_context  # noqa: E402


def _cal(t, y, q):
    c = store.fund_calendar()
    return c[(c["ticker"] == t) & (c["year"] == y) & (c["quarter"] == q)].iloc[0]


def test_quarter_invisible_before_effective_date_and_visible_on_it():
    row = _cal("THYAO", 2021, 2)
    eff = row["effective_trade_date"]
    before = get_fundamental_metrics(ticker="THYAO", as_of=(eff - pd.Timedelta(days=1)).date())
    on = get_fundamental_metrics(ticker="THYAO", as_of=eff.date())
    assert (before["data"]["report"]["fiscal_year"], before["data"]["report"]["quarter"]) != (2021, 2)
    assert (on["data"]["report"]["fiscal_year"], on["data"]["report"]["quarter"]) == (2021, 2)
    assert on["data"]["report"]["effective_trade_date"] <= on["as_of"]


def test_future_period_request_is_rejected():
    r = get_fundamental_metrics(ticker="THYAO", as_of="2021-01-15", period="2023Q2")
    assert r["error"]["code"] == "PERIOD_NOT_PUBLIC"


def test_nominal_era_has_growth_and_tms29_withholds_it():
    nom = get_fundamental_metrics(ticker="EREGL", as_of="2021-06-30")
    assert nom["data"]["report"]["era"] == "nominal" and "revenue_yoy" in nom["data"]["metrics"]
    new = get_fundamental_metrics(ticker="EREGL", as_of="2025-06-30")
    assert new["data"]["report"]["era"] == "tms29_restated" and "revenue_yoy" not in new["data"]["metrics"]
    codes = {w["code"] for w in new["warnings"]}
    assert {"TMS29_RESTATED_VALUES", "GROWTH_WITHHELD"} <= codes


def test_bank_and_insurer_use_their_own_metric_sets():
    b = get_fundamental_metrics(ticker="AKBNK", as_of="2022-06-30")["data"]
    assert b["kind"] == "bank" and "nii_ttm" in b["metrics"] and "ebitda_ttm" not in b["metrics"] and b["valuation"] is None
    s = get_fundamental_metrics(ticker="TURSG", as_of="2022-06-30")["data"]
    assert s["kind"] == "insurer" and "premiums_written_net_ttm" in s["metrics"]


def test_fundamental_change_structure_and_era_boundary():
    r = calculate_fundamental_change(ticker="THYAO", as_of="2023-03-15")  # latest public may straddle the TMS29 boundary
    d = r["data"]
    assert set(d["ratio_changes"]["roe_ttm"]) == {"now", "vs_prev_quarter", "vs_prev_year"}
    r2 = calculate_fundamental_change(ticker="THYAO", as_of="2021-09-15")
    assert r2["data"]["growth_yoy"] is not None  # nominal era
    assert r2["data"]["compared_with"]["previous_year"] is not None


def test_nothing_public_yet_is_a_structured_error():
    r = get_fundamental_metrics(ticker="THYAO", as_of="2014-01-01")
    assert r["error"]["code"] == "NO_FUNDAMENTALS"


def test_valuation_present_for_industrial_only():
    d = get_fundamental_metrics(ticker="THYAO", as_of="2022-06-30")["data"]
    assert d["valuation"] is not None and d["valuation"]["market_cap_approx"] > 0


def test_macro_uses_only_strictly_earlier_observations():
    as_of = pd.Timestamp("2024-03-20")
    r = get_macro_context(as_of=as_of.date())
    for name, s in r["data"]["series"].items():
        if s:
            assert s["observation_date"] < "2024-03-20", name
    if store.tcmb().empty:  # v1.1: TCMB series are reported once data/processed/macro_tcmb.parquet exists
        assert any(w["code"] == "MACRO_INCOMPLETE" for w in r["warnings"])
        assert "policy_rate" in r["data"]["unavailable_series"]
    else:
        assert r["data"]["unavailable_series"] == [] and r["data"]["series"]["policy_rate"]["observation_date"] < "2024-03-20"
    assert r["data"]["series"]["fed_target_upper"]["value"] == 5.5  # 2024-03: 5.25-5.50
