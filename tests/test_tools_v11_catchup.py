import numpy as np
import pandas as pd

from bist.tools import store
from bist.tools.catchup import catch_up_core, test_catch_up_history
from conftest import assert_causal


def test_catch_up_core_known_answer():
    idx = pd.bdate_range("2024-01-01", periods=120)
    peer = pd.Series(100 * 1.001 ** np.arange(120), index=idx)
    stock = peer.copy()
    stock.iloc[40:45] *= 0.9          # falls 10% behind for 5 bars, then fully closes the gap
    peers = pd.DataFrame({"P1": peer, "P2": peer * 1.0})
    d, w = catch_up_core(stock, peers, peer, peer * 0 == 0, 5, [5], 0.05, idx[0], idx[-1])
    h = d["by_horizon"]["h5"]
    assert d["n_events"] == 1 and h["narrowed_rate"] == 1.0 and h["closed_rate"] == 1.0
    assert h["false_positive_rate"] == 0.0 and d["events"][0]["date"] == idx[40]


def test_catch_up_q1_mavi():
    r = test_catch_up_history(ticker="MAVI", peers=["BIMAS", "MGROS"], start="2024-01-01", end="2026-09-10", as_of="2026-09-10")
    assert r["status"] == "WARN", r.get("error")
    d = r["data"]
    assert set(d["by_horizon"]) == {"h5", "h10", "h20"} and "CUSTOM_PEER_GROUP" in {x["code"] for x in r["warnings"]}
    assert all("2024-01-01" <= e["date"] <= "2026-09-10" for e in d["events"])
    first_incomplete = store.stock_bars("MAVI", pd.Timestamp("2026-09-10")).index[-20].strftime("%Y-%m-%d")
    for e in d["events"]:
        assert (e["rel_fwd"]["20d"] is None) == (e["date"] >= first_incomplete)
    h = d["by_horizon"]["h10"]
    assert h["n"] + h["n_excluded_incomplete"] == d["n_events"]
    assert set(d["sector_strong_at_event"]) == {"true", "false"}


def test_catch_up_invalid_params():
    assert test_catch_up_history(ticker="MAVI", horizons=[0])["status"] == "ERROR"
    assert test_catch_up_history(ticker="MAVI", peers=["NOPE"])["error"]["code"] == "UNKNOWN_TICKER"


def test_catch_up_small_peer_group_warns():
    r = test_catch_up_history(ticker="ASELS", as_of="2026-09-10")
    assert r["status"] == "WARN" and "PEER_GROUP_TOO_SMALL" in {x["code"] for x in r["warnings"]}
    assert r["data"]["by_horizon"] is None


def test_catch_up_causal(truncate_store):
    assert_causal(test_catch_up_history, "2026-09-10", truncate_store, ticker="MAVI", peers=["BIMAS", "MGROS"], start="2024-01-01")
