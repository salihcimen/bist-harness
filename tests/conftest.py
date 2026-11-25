import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bist.tools import store  # noqa: E402


@pytest.fixture
def truncate_store(monkeypatch):
    """cut('YYYY-MM-DD') removes every row dated after the cut from the data cache (prices, index, macro, TCMB, KAP calendar)."""

    def cut(day: str):
        ts = pd.Timestamp(day)
        for name, col in (("_stocks", "date"), ("_index", "date"), ("_macro", "date"), ("_tcmb", "publish_date"),
                          ("_fund_calendar", "first_pub_ts")):
            if not hasattr(store, name):
                continue
            full = getattr(store, name)()
            if col in full.columns:
                kept = full[full[col] <= ts + pd.Timedelta(hours=23, minutes=59)] if col == "first_pub_ts" else full[full[col] <= ts]
                monkeypatch.setattr(store, name, lambda kept=kept: kept)

    return cut


def assert_causal(fn, cut_day: str, truncate_store, **kw):
    """Result at as_of=cut_day must not change when everything after cut_day is deleted from the cache."""
    before = fn(as_of=cut_day, **kw)
    truncate_store(cut_day)
    after = fn(as_of=cut_day, **kw)
    assert before["status"] == after["status"], (before.get("error"), after.get("error"))
    assert before["data"] == after["data"]
