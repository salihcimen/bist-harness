"""test_catch_up_history: did similar lags vs. the peer median close afterwards? (sample Q1, S1 logic as a tool).

Event on day t: ticker's L-day return minus the peer-median L-day return <= -min_gap (uses bars <= t only); events are at
least L bars apart. Outcome over the next h bars: relative return (ticker - peer median). Windows ending after as_of are
excluded and counted.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import store
from .common import forward_window_stats, resolve_range
from .envelope import tool, warn
from .research import _group_members


def _rel_frame(stock: pd.Series, peers: pd.DataFrame, bench: pd.Series, horizons: list[int]) -> pd.DataFrame:
    peers = peers.reindex(stock.index).ffill()
    bench = bench.reindex(stock.index).ffill()
    out = pd.DataFrame(index=stock.index)
    for h in horizons:
        s = stock.shift(-h) / stock - 1
        m = (peers.shift(-h) / peers - 1).median(axis=1, skipna=True)
        out[f"rel_{h}"] = s - m
        out[f"ex_{h}"] = s - (bench.shift(-h) / bench - 1)
    return out


def _horizon_block(ev: pd.DatetimeIndex, fr: pd.DataFrame, gaps: pd.Series, horizons: list[int], label: str):
    st, w = forward_window_stats(ev, fr, horizons, {"mean_excess_vs_xu100": "ex"}, main="rel", label=label)
    out = {}
    for h in horizons:
        r = fr.loc[ev, f"rel_{h}"].dropna()
        s = st[f"h{h}"]
        narrowed = float((r > 0).mean()) if len(r) else None
        out[f"h{h}"] = {"n": s["n"], "n_excluded_incomplete": s["n_excluded_incomplete"], "narrowed_rate": narrowed,
                        "closed_rate": float((r >= gaps.loc[r.index].abs()).mean()) if len(r) else None,
                        "false_positive_rate": None if narrowed is None else 1 - narrowed,
                        "mean_rel": s["mean"], "median_rel": s["median"], "mean_excess_vs_xu100": s["mean_excess_vs_xu100"]}
    return out, w


def catch_up_core(stock: pd.Series, peers: pd.DataFrame, bench: pd.Series, sector_strong: pd.Series, lookback: int,
                  horizons: list[int], min_gap: float, start, end) -> tuple[dict, list[dict]]:
    peers_a = peers.reindex(stock.index).ffill()
    rel_l = stock / stock.shift(lookback) - 1 - (peers_a / peers_a.shift(lookback) - 1).median(axis=1, skipna=True)
    strong = sector_strong.reindex(stock.index).fillna(False).astype(bool)
    cand = (rel_l <= -abs(min_gap)) & (stock.index >= pd.Timestamp(start)) & (stock.index <= pd.Timestamp(end))
    picked, last = [], -10**9
    for i, v in enumerate(cand.to_numpy()):
        if v and i - last >= lookback:
            picked.append(stock.index[i])
            last = i
    ev = pd.DatetimeIndex(picked)
    fr = _rel_frame(stock, peers, bench, horizons)
    by_h, w = _horizon_block(ev, fr, rel_l, horizons, "catch-up history")
    split = {}
    for key, mask in (("true", strong.loc[ev].to_numpy()), ("false", ~strong.loc[ev].to_numpy())):
        split[key], _ = _horizon_block(ev[mask], fr, rel_l, horizons, "")
        split[key]["n_events"] = int(mask.sum())
    events = [{"date": d, "lag_gap": float(rel_l.loc[d]), "sector_strong": bool(strong.loc[d]),
               "rel_fwd": {f"{h}d": (None if np.isnan(fr.loc[d, f"rel_{h}"]) else float(fr.loc[d, f"rel_{h}"])) for h in horizons}}
              for d in ev]
    cur = rel_l.iloc[-1]
    return {"n_events": int(len(ev)), "current_lag_gap": None if np.isnan(cur) else float(cur), "by_horizon": by_h,
            "sector_strong_at_event": split, "events": events}, w


class CatchUpParams(BaseModel):
    ticker: str = Field(description="BIST ticker whose lag vs. its peers is studied, e.g. MAVI")
    peers: list[str] | None = Field(default=None, min_length=1, max_length=29, description="Explicit peer list (default: the ticker's peer group, ticker excluded).")
    lookback_days: int = Field(default=20, ge=5, le=60, description="L: trading days over which the lag is measured.")
    horizons: list[int] = Field(default=[5, 10, 20], min_length=1, max_length=5, description="Forward horizons (trading days) at which closing is checked.")
    min_lag_gap: float | None = Field(default=None, ge=0, le=1, description="Lag threshold as a fraction (0.05 = 5 points behind the peer median); default: the ticker's current L-day lag.")
    start: date | None = Field(default=None, description="First event date (default: 5 years before end).")
    end: date | None = Field(default=None, description="Last event date (default and maximum: as_of).")
    as_of: date | None = Field(default=None, description="Analysis date; nothing after it is used.")

    def model_post_init(self, _ctx):
        if any(h < 1 or h > 60 for h in self.horizons):
            raise ValueError("each horizon must be between 1 and 60 trading days")


@tool("test_catch_up_history", CatchUpParams,
      "History of similar lags: every past day the ticker trailed the peer median by at least the given gap over L days, and whether the gap narrowed or closed within 5/10/20 days (hit rate, false-positive rate, excess vs XU100, split by sector strength). Completed windows only.")
def test_catch_up_history(p: CatchUpParams):
    t, _ = store.resolve_ticker(p.ticker)
    as_of, w = store.resolve_as_of(p.as_of)
    if p.peers:
        _, peers, custom = _group_members(None, p.peers)
    else:
        _, peers, custom = _group_members(t, None)
    peers = [x for x in peers if x != t]
    bars = store.stock_bars(t, as_of)
    w += store.staleness_warnings(bars, as_of, t)
    start, end, wr = resolve_range(p.start, p.end, as_of, as_of - pd.DateOffset(years=5))
    w += wr
    if custom:
        w.append(warn("CUSTOM_PEER_GROUP", f"peers are the explicit list {', '.join(peers)}, not a predefined sector"))
    base = {"ticker": t, "peers": peers, "custom_group": custom, "lookback_days": p.lookback_days, "horizons": p.horizons,
            "window": [start, end]}
    prov = {"as_of": as_of, "last_bar_date": bars.index.max(), "sources": ["prices_clean.parquet", "index_clean.parquet"]}
    if len(peers) < 2:
        w.append(warn("PEER_GROUP_TOO_SMALL", f"{t} has {len(peers)} peer(s); a peer median needs at least 2, so catch-up history is not meaningful"))
        return {**base, "n_events": 0, "by_horizon": None, "sector_strong_at_event": None, "events": []}, w, prov
    close = bars["close"]
    pc = pd.DataFrame({x: store.stock_bars(x, as_of)["close"] for x in peers})
    bench = store.index_bars(as_of)["close"]
    L = p.lookback_days
    pa = pc.reindex(close.index).ffill()
    med_l = (pa / pa.shift(L) - 1).median(axis=1, skipna=True)
    ba = bench.reindex(close.index).ffill()
    strong = (med_l > (ba / ba.shift(L) - 1)) & (med_l > 0)
    rel_now = close.iloc[-1] / close.iloc[-1 - L] - 1 - med_l.iloc[-1] if len(close) > L else np.nan
    if p.min_lag_gap is not None:
        gap = p.min_lag_gap
    else:
        gap = abs(rel_now) if rel_now == rel_now and abs(rel_now) >= 0.005 else 0.02
    d, w2 = catch_up_core(close, pc, bench, strong, L, p.horizons, gap, start, end)
    w += w2
    if d["n_events"] < 10:
        w.append(warn("SMALL_SAMPLE", f"{t}: only {d['n_events']} lag events in the window; no reliable conclusion"))
    return {**base, "min_lag_gap_used": gap, **d}, w, prov


test_catch_up_history.__test__ = False  # not a pytest test (name starts with test_)
