from datetime import date

import numpy as np
import pandas as pd
import pytest

from bist.tools.common import forward_window_stats, fwd_frame, resolve_range
from bist.tools.envelope import ToolError


def test_resolve_range_clips_end_and_defaults():
    a = pd.Timestamp("2026-09-10")
    s, e, w = resolve_range(None, date(2026, 12, 1), a, pd.Timestamp("2021-09-10"))
    assert (s, e) == (pd.Timestamp("2021-09-10"), a) and [x["code"] for x in w] == ["END_CLIPPED_TO_AS_OF"]


def test_resolve_range_rejects_inverted():
    with pytest.raises(ToolError) as ex:
        resolve_range(date(2026, 1, 5), date(2025, 1, 1), pd.Timestamp("2026-09-10"), pd.Timestamp("2021-01-01"))
    assert ex.value.code == "INVALID_DATE_RANGE"


def test_forward_window_excludes_and_counts_incomplete():
    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.Series(np.arange(100, 130, dtype=float), index=idx)
    fwd = fwd_frame(close, close * 0 + 1.0, [5, 20])
    ev = idx[[2, 15, 27]]
    st, w = forward_window_stats(ev, fwd, [5, 20], {"mean_excess_vs_xu100": "ex"})
    assert st["h5"]["n"] == 2 and st["h5"]["n_excluded_incomplete"] == 1
    assert st["h20"]["n"] == 1 and st["h20"]["n_excluded_incomplete"] == 2
    assert {x["code"] for x in w} == {"INCOMPLETE_FORWARD_WINDOW", "SMALL_SAMPLE"}
