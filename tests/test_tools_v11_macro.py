import pandas as pd
import pytest

from bist.tools import store
from bist.tools.fundamentals import get_macro_context
from conftest import assert_causal

needs_tcmb = pytest.mark.skipif(store.tcmb().empty, reason="data/processed/macro_tcmb.parquet missing (run python -m bist.fetch_evds)")


@needs_tcmb
def test_macro_range_q5():
    r = get_macro_context(as_of="2026-09-10", start="2024-01-01", series=["usdtry", "usdtry_tcmb", "policy_rate", "cpi_yoy"])
    assert r["status"] != "ERROR", r["error"]
    s = r["data"]["series"]
    assert set(s) == {"usdtry", "usdtry_tcmb", "policy_rate", "cpi_yoy"}
    assert s["usdtry_tcmb"]["range"]["n_obs"] > 400 and s["policy_rate"]["value"] is not None
    assert s["usdtry"]["range"]["last_date"] < "2026-09-10" and s["usdtry"]["range"]["first_date"] >= "2024-01-01"
    assert "MACRO_INCOMPLETE" not in {w["code"] for w in r["warnings"]}


@needs_tcmb
def test_cpi_publish_date_rule():
    d = get_macro_context(as_of="2026-09-02", series=["cpi_yoy"])["data"]["series"]["cpi_yoy"]
    assert d["observation_date"] <= "2026-07-31"          # August CPI is not public before early September


def test_macro_without_tcmb_file(monkeypatch):
    monkeypatch.setattr(store, "_tcmb", lambda: store.empty_tcmb())
    r = get_macro_context(as_of="2026-09-10")
    assert "policy_rate" in r["data"]["unavailable_series"] and r["status"] == "WARN"
    assert "MACRO_INCOMPLETE" in {w["code"] for w in r["warnings"]}


def test_macro_v10_unchanged():
    d = get_macro_context(as_of="2026-09-10")["data"]
    assert "range" not in d["series"]["usdtry"] and d["series"]["usdtry"]["observation_date"] < "2026-09-10"


def test_macro_unknown_series():
    assert get_macro_context(series=["bitcoin"])["error"]["code"] == "INVALID_PARAMS"


def test_macro_causal(truncate_store):
    assert_causal(get_macro_context, "2026-09-10", truncate_store, start="2024-01-01")


# ------------------------------------------------------------- analyze_macro_sensitivity (Q5)
import json  # noqa: E402

import numpy as np  # noqa: E402

from bist.tools.macro_sensitivity import analyze_macro_sensitivity, sensitivity_core  # noqa: E402


def test_sensitivity_core_perfect_beta():
    idx = pd.bdate_range("2024-01-01", periods=300)
    f = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, .01, 300)), index=idx)
    stock = 50 * (f / f.iloc[0]) ** 2
    d = sensitivity_core(stock, f * 0 + 1, f, "weekly", "pct")
    assert d["correlation"] > 0.99 and d["beta"] == pytest.approx(2, rel=0.1)
    assert d["xu100"]["correlation"] is None                      # flat index: undefined, not invented


def test_q5_eregl_usdtry():
    r = analyze_macro_sensitivity(ticker="EREGL", factor="usdtry", start="2024-01-01", end="2026-09-10", as_of="2026-09-10")
    assert r["status"] != "ERROR", r["error"]
    d = r["data"]
    assert d["n_periods"] > 100 and d["association_only"].startswith("correlation is not causation")
    assert "xu100" in d and [y["year"] for y in d["yearly"]] == [2024, 2025, 2026]
    body = json.dumps({k: v for k, v in d.items() if k != "association_only"}).lower()
    assert "target" not in body and "forecast" not in body


def test_sensitivity_monthly_cpi_is_stale_weekly():
    if store.tcmb().empty:
        pytest.skip("no TCMB data")
    r = analyze_macro_sensitivity(ticker="EREGL", factor="cpi_yoy", start="2024-01-01", as_of="2026-09-10")
    assert "STALE_FACTOR" in {w["code"] for w in r["warnings"]}


def test_sensitivity_policy_rate_missing_file(monkeypatch):
    monkeypatch.setattr(store, "_tcmb", lambda: store.empty_tcmb())
    assert analyze_macro_sensitivity(ticker="EREGL", factor="policy_rate")["error"]["code"] == "SOURCE_NOT_AVAILABLE"


def test_sensitivity_causal(truncate_store):
    assert_causal(analyze_macro_sensitivity, "2026-09-10", truncate_store, ticker="EREGL", factor="usdtry", start="2024-01-01")
