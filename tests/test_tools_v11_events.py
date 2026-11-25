import pandas as pd

from bist.tools.events import condition_flags, detect_reversal_events
from conftest import assert_causal

Q3_SETS = [["rsi_trough", "high_volume"], ["supertrend_bearish", "below_ichimoku_cloud"]]


def test_condition_flags_rsi_trough_and_volume():
    n = 25
    f = pd.DataFrame({"close": 10.0, "rsi": [50.0] * 21 + [40, 34, 33, 36], "volume": [100.0] * 23 + [160.0, 160.0],
                      "volume_missing": [False] * 24 + [True], "st_dir": 1, "ich_span_a": 9.0, "ich_span_b": 8.0,
                      "macd_hist": 0.1, "kama": 9.5}, index=pd.bdate_range("2024-01-01", periods=n))
    c = condition_flags(f)
    assert c["rsi_trough"].tolist()[-4:] == [False, True, True, False]
    assert c["high_volume"].tolist()[-2:] == [True, False]       # last day: volume missing -> False
    assert c["above_ichimoku_cloud"].iloc[-1] and c["above_kama"].iloc[-1] and c["supertrend_bullish"].iloc[-1]
    assert not c["below_ichimoku_cloud"].iloc[-1] and not c["macd_hist_negative"].iloc[-1]


def test_bollinger_touch_q3_asels():
    r = detect_reversal_events(ticker="ASELS", event_types=["bollinger_lower_touch"], start="2024-01-01", end="2026-09-10",
                               as_of="2026-09-10", cooldown_days=0, condition_sets=Q3_SETS)
    assert r["status"] != "ERROR", r["error"]
    d = r["data"]
    s = d["history_summary"]["bollinger_lower_touch"]
    assert s["count"] == len(d["events"]) > 0
    assert len(s["by_condition_set"]) == 2 and all(c["count"] <= s["count"] for c in s["by_condition_set"])
    assert s["h5"]["n"] + s["h5"]["n_excluded_incomplete"] == s["count"]
    assert d["window"] == ["2024-01-01", "2026-09-10"]
    assert all("2024-01-01" <= e["date"] <= "2026-09-10" and set(e["conditions"]) == {"rsi_trough", "high_volume", "supertrend_bearish", "below_ichimoku_cloud"} for e in d["events"])


def test_unknown_condition_rejected():
    r = detect_reversal_events(ticker="ASELS", condition_sets=[["moon_phase"]])
    assert r["error"]["code"] == "INVALID_PARAMS"


def test_reversal_v10_unchanged():
    d = detect_reversal_events(ticker="THYAO", as_of="2026-09-10")["data"]
    assert "events" not in d and "window" not in d
    assert "bollinger_lower_touch" not in d["history_summary"]      # new types only on explicit request


def test_reversal_window_causal(truncate_store):
    assert_causal(detect_reversal_events, "2026-09-10", truncate_store, ticker="ASELS", event_types=["bollinger_lower_touch"],
                  start="2024-01-01", condition_sets=[["rsi_trough", "high_volume"]])
