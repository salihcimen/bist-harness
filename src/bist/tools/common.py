"""Shared v1.1 rules: date ranges clipped to as_of, forward-return windows that never look past as_of."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .envelope import ToolError, warn

COST_BPS_DEFAULT = 30
SMALL_N = 10


def resolve_range(start: date | None, end: date | None, as_of: pd.Timestamp, default_start: pd.Timestamp
                  ) -> tuple[pd.Timestamp, pd.Timestamp, list[dict]]:
    """[start, end] with end <= as_of. end > as_of is clipped (END_CLIPPED_TO_AS_OF); start > end is an error."""
    w = []
    e = pd.Timestamp(end) if end is not None else as_of
    if e > as_of:
        w.append(warn("END_CLIPPED_TO_AS_OF", f"end {e.date()} is after the analysis date; clipped to {as_of.date()}"))
        e = as_of
    s = pd.Timestamp(start) if start is not None else pd.Timestamp(default_start)
    if s > e:
        raise ToolError("INVALID_DATE_RANGE", f"start {s.date()} is after end {e.date()} (end is capped at as_of {as_of.date()})")
    return s, e, w


def fwd_frame(close: pd.Series, bench: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """fwd_h: close-to-close return over the next h bars; ex_h: minus the benchmark. NaN when t+h is beyond the series (= as_of)."""
    bench = bench.reindex(close.index).ffill()
    out = pd.DataFrame(index=close.index)
    for h in horizons:
        out[f"fwd_{h}"] = close.shift(-h) / close - 1
        out[f"ex_{h}"] = out[f"fwd_{h}"] - (bench.shift(-h) / bench - 1)
    return out


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def forward_window_stats(event_dates: pd.DatetimeIndex, fwd: pd.DataFrame, horizons: list[int], cols: dict[str, str],
                         main: str = "fwd", label: str = "") -> tuple[dict, list[dict]]:
    """Per horizon: n (completed windows only), n_excluded_incomplete, mean/median/hit_rate of `main`, mean of each extra column."""
    out, incomplete, small = {}, [], []
    ev = pd.DatetimeIndex(event_dates).intersection(fwd.index)
    for h in horizons:
        v = fwd.loc[ev, f"{main}_{h}"]
        r = v.dropna()
        row = {"n": int(len(r)), "n_excluded_incomplete": int(v.isna().sum()),
               "mean": _f(r.mean()) if len(r) else None, "median": _f(r.median()) if len(r) else None,
               "hit_rate": float((r > 0).mean()) if len(r) else None}
        for key, prefix in cols.items():
            x = fwd.loc[r.index, f"{prefix}_{h}"].dropna()
            row[key] = _f(x.mean()) if len(x) else None
        out[f"h{h}"] = row
        if row["n_excluded_incomplete"]:
            incomplete.append(f"{h}d: {row['n_excluded_incomplete']}")
        if 0 < row["n"] < SMALL_N:
            small.append(f"{h}d n={row['n']}")
    w = []
    pre = f"{label}: " if label else ""
    if incomplete:
        w.append(warn("INCOMPLETE_FORWARD_WINDOW", f"{pre}events whose forward window ends after as_of were excluded ({', '.join(incomplete)})"))
    if small:
        w.append(warn("SMALL_SAMPLE", f"{pre}fewer than {SMALL_N} completed observations ({', '.join(small)}); no reliable conclusion"))
    return out, w
