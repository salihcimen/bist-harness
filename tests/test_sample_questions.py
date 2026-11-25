"""Acceptance: the instructor's 5 sample questions (docs/sample_test_questions.txt) answered with the expected tools, the
question's date ranges and analysis date 2026-09-10. Every date in every result must be <= 2026-09-10 (no look-ahead)."""
import json
import re

import pandas as pd
import pytest

from bist.tools import store
from bist.tools.catchup import test_catch_up_history
from bist.tools.disclosures import analyze_disclosure_reactions
from bist.tools.events import detect_reversal_events
from bist.tools.fundamentals import calculate_fundamental_change, get_fundamental_metrics, get_macro_context
from bist.tools.indicators import calculate_indicators
from bist.tools.macro_sensitivity import analyze_macro_sensitivity
from bist.tools.research import rank_sector_relative_strength, test_weekday_pattern

AS_OF = "2026-09-10"
START = "2024-01-01"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def ok(env):
    assert env["status"] != "ERROR", (env["tool"], env["error"])
    dates = _DATE.findall(json.dumps({k: v for k, v in env.items() if k != "params"}, default=str))
    assert all(d <= AS_OF for d in dates), (env["tool"], max(dates))
    return env["data"]


def test_q1_sector_laggard_catch_up():
    d = ok(rank_sector_relative_strength(tickers=["BIMAS", "MGROS", "MAVI"], horizon_days=20, as_of=AS_OF))
    worst = d["ranking_lagging_first"][0]
    assert worst["lag_rank"] == 1 and worst["lag_gap"] is not None
    peers = [m["ticker"] for m in d["ranking_lagging_first"][1:]]
    h = ok(test_catch_up_history(ticker=worst["ticker"], peers=peers, lookback_days=20, start=START, end=AS_OF, as_of=AS_OF))
    for k in ("h5", "h10", "h20"):
        assert h["by_horizon"][k]["n"] > 0 and h["by_horizon"][k]["false_positive_rate"] is not None


def test_q2_weekday_sequence_is_vs_oos():
    d = ok(test_weekday_pattern(ticker="THYAO", pattern="weekday_sequence", down_days=["MON", "TUE", "WED"], up_days=["THU", "FRI"],
                                start=START, end=AS_OF, oos_start="2026-01-01", as_of=AS_OF))
    for seg in ("in_sample", "out_of_sample"):
        for k in ("occurrences", "mean", "median", "baseline_mean", "net_mean_after_cost", "p_value_bonferroni"):
            assert d[seg][k] is not None, (seg, k)
    assert d["in_sample"]["period"][1] <= "2025-12-31" and d["out_of_sample"]["period"][0] >= "2026-01-01"
    assert d["n_tests_in_family"] >= 1 and "market_wide_effect" in d


def test_q3_bollinger_touches_with_conditions():
    ok(calculate_indicators(ticker="ASELS", as_of=AS_OF))
    d = ok(detect_reversal_events(ticker="ASELS", event_types=["bollinger_lower_touch"], start=START, end=AS_OF, cooldown_days=0, as_of=AS_OF,
                                  condition_sets=[["rsi_trough", "high_volume"], ["supertrend_bearish", "below_ichimoku_cloud"]]))
    s = d["history_summary"]["bollinger_lower_touch"]
    assert s["count"] > 0 and s["h5"]["hit_rate"] is not None
    assert all(f"h{h}" in s for h in (1, 3, 5, 10)) and len(s["by_condition_set"]) == 2


def test_q4_tuprs_disclosures_reaction():
    ok(get_fundamental_metrics(ticker="TUPRS", as_of=AS_OF))
    ok(calculate_fundamental_change(ticker="TUPRS", as_of=AS_OF))
    d = ok(analyze_disclosure_reactions(ticker="TUPRS", start=START, end=AS_OF, as_of=AS_OF))
    row = next(r for r in d["rows"] if r["period"] == "2026Q2")
    bars = store.stock_bars("TUPRS", pd.Timestamp(AS_OF))
    pos = bars.index.searchsorted(pd.Timestamp(row["effective_trade_date"]))
    for h in (1, 5, 20):
        r = row["returns"][f"{h}d"]
        assert r["complete"] == (pos + h - 1 < len(bars))
        assert (r["stock"] is None) == (not r["complete"])
    assert d["summary"]["20d"]["improving"]["n"] + d["summary"]["20d"]["deteriorating"]["n"] > 0


def test_q5_macro_and_usdtry_sensitivity():
    if store.tcmb().empty:
        pytest.skip("TCMB data not built (python -m bist.fetch_evds)")
    d = ok(get_macro_context(as_of=AS_OF, start=START, series=["usdtry", "usdtry_tcmb", "policy_rate", "cpi_yoy"]))
    assert all(d["series"][s] and d["series"][s]["range"] for s in ("usdtry_tcmb", "policy_rate", "cpi_yoy"))
    s = ok(analyze_macro_sensitivity(ticker="EREGL", factor="usdtry", start=START, end=AS_OF, as_of=AS_OF))
    assert s["n_periods"] > 100 and s["correlation"] is not None and "no forecast" in s["association_only"]
