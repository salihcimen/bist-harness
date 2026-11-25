"""analyze_disclosure_reactions: every quarterly report published (KAP first publication) in a window, its metric change and
the 1/5/20-day price reaction vs XU100 and the sector (sample Q4, S4 logic as a tool).

Reaction window for horizon h: close of the trading day before effective_trade_date -> close of the (h)th trading day from
effective_trade_date (the reaction day counts as day 1). A window that ends after as_of is marked complete=false and has no
numbers. Association only: no causal claim.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import store
from .common import resolve_range
from .envelope import ToolError, tool, warn
from .fundamentals import RATIO_COLS, available, era
from .research import _group_members

DEFAULT_METRICS = {"industrial": ["ebitda_margin_q", "net_debt_ebitda"], "bank": ["roe_ttm", "nii_assets"], "insurer": ["roa_ttm"]}
LOWER_IS_BETTER = {"net_debt_ebitda", "debt_equity"}


class DisclosureReactionParams(BaseModel):
    ticker: str = Field(description="BIST ticker, e.g. TUPRS")
    start: date | None = Field(default=None, description="First KAP publication date included (default: 3 years before end). Filter is on the publication date, not the quarter end.")
    end: date | None = Field(default=None, description="Last KAP publication date included (default and maximum: as_of).")
    metrics: list[str] | None = Field(default=None, min_length=1, max_length=4, description="Ratio metrics to report (default industrial: ebitda_margin_q, net_debt_ebitda; bank: roe_ttm, nii_assets; insurer: roa_ttm).")
    horizons: list[int] = Field(default=[1, 5, 20], min_length=1, max_length=4, description="Reaction horizons in trading days (reaction day = day 1).")
    peers: list[str] | None = Field(default=None, min_length=1, max_length=29, description="Peers for the sector-relative reaction (default: the ticker's peer group).")
    as_of: date | None = Field(default=None, description="Analysis date; reports not yet public and prices after it are never used.")

    def model_post_init(self, _ctx):
        if any(h < 1 or h > 60 for h in self.horizons):
            raise ValueError("each horizon must be between 1 and 60 trading days")


def _window_ret(close: pd.Series, pos: int, h: int) -> float | None:
    if pos - 1 < 0 or pos + h - 1 >= len(close):
        return None
    return float(close.iloc[pos + h - 1] / close.iloc[pos - 1] - 1)


def _improving(chg: dict, metrics: list[str]) -> bool | None:
    signs = []
    for m in metrics:
        v = chg.get(m)
        if v is None:
            return None
        signs.append((v < 0) if m in LOWER_IS_BETTER else (v > 0))
    return signs[0] if len(set(signs)) == 1 else None


@tool("analyze_disclosure_reactions", DisclosureReactionParams,
      "For every quarterly report first published on KAP in [start, end]: metric values and change vs. the previous quarter (improving flag), and the 1/5/20-day stock return, abnormal return vs XU100 and vs the sector median after the publication; summary of improving vs deteriorating quarters. Incomplete windows are flagged, never filled. Association only.")
def analyze_disclosure_reactions(p: DisclosureReactionParams):
    t, _ = store.resolve_ticker(p.ticker)
    as_of, w = store.resolve_as_of(p.as_of)
    start, end, wr = resolve_range(p.start, p.end, as_of, as_of - pd.DateOffset(years=3))
    w += wr
    av = available(t, as_of)
    if av.empty:
        raise ToolError("NO_FUNDAMENTALS", f"{t}: no financial report public on or before {as_of.date()}")
    kind = av.iloc[-1]["kind"]
    metrics = p.metrics or DEFAULT_METRICS[kind]
    bad = [m for m in metrics if m not in RATIO_COLS[kind]]
    if bad:
        raise ToolError("UNKNOWN_METRIC", f"{bad} not available for a {kind}. Valid: {', '.join(RATIO_COLS[kind])}")
    if p.peers:
        _, peers, _ = _group_members(None, p.peers)
    else:
        _, peers, _ = _group_members(t, None)
    peers = [x for x in peers if x != t]
    bars = store.stock_bars(t, as_of)
    close = bars["close"]
    xu = store.index_bars(as_of)["close"].reindex(close.index).ffill()
    pc = pd.DataFrame({x: store.stock_bars(x, as_of)["close"] for x in peers}).reindex(close.index).ffill() if peers else pd.DataFrame(index=close.index)
    pub = av["first_pub_ts"].dt.normalize()
    rows, incomplete, tms, late = [], [], False, []
    for i in np.flatnonzero(((pub >= start) & (pub <= end)).to_numpy()):
        r = av.iloc[i]
        prev = av.iloc[i - 1] if i > 0 else None
        e = era(int(r["year"]), int(r["month"]))
        tms |= e == "tms29_restated"
        per = f"{int(r['year'])}Q{int(r['quarter'])}"
        if bool(r["suspect_lag"]):
            late.append(per)
        same_era = prev is not None and era(int(prev["year"]), int(prev["month"])) == e
        vals = {m: (None if pd.isna(r.get(m)) else float(r[m])) for m in metrics}
        chg = {m: (float(r[m] - prev[m]) if same_era and not pd.isna(r.get(m)) and not pd.isna(prev.get(m)) else None) for m in metrics}
        pos = int(close.index.searchsorted(r["effective_trade_date"]))
        rets = {}
        for h in p.horizons:
            s = _window_ret(close, pos, h)
            done = s is not None
            if not done and pos - 1 >= 0:
                incomplete.append(f"{per} {h}d")
            xr = _window_ret(xu, pos, h) if done else None
            med = None
            if done and pc.shape[1] >= 2:
                vals_p = [_window_ret(pc[c], pos, h) for c in pc]
                vals_p = [v for v in vals_p if v is not None and not np.isnan(v)]
                med = float(np.median(vals_p)) if len(vals_p) >= 2 else None
            rets[f"{h}d"] = {"stock": s, "abnormal_vs_xu100": None if xr is None else s - xr,
                             "abnormal_vs_sector": None if med is None else s - med, "complete": done}
        rows.append({"period": per, "first_published": r["first_pub_ts"], "effective_trade_date": r["effective_trade_date"],
                     "era": e, "values": vals, "change_vs_prev_quarter": chg, "improving": _improving(chg, metrics), "returns": rets})
    summary = {}
    for h in p.horizons:
        k = f"{h}d"
        blk = {}
        for name, flag in (("improving", True), ("deteriorating", False)):
            x = [r["returns"][k]["abnormal_vs_xu100"] for r in rows if r["improving"] is flag and r["returns"][k]["complete"]]
            blk[name] = {"n": len(x), "mean_abnormal_vs_xu100": float(np.mean(x)) if x else None}
        a, b = blk["improving"]["mean_abnormal_vs_xu100"], blk["deteriorating"]["mean_abnormal_vs_xu100"]
        blk["diff_improving_minus_deteriorating"] = None if a is None or b is None else a - b
        summary[k] = blk
    if incomplete:
        w.append(warn("INCOMPLETE_FORWARD_WINDOW", f"{t}: reaction window ends after as_of for {', '.join(incomplete)}; reported as incomplete, no prices after as_of used"))
    if tms:
        w.append(warn("TMS29_RESTATED_VALUES", f"{t}: values from FY2022 on come from a source that restates history under TMS 29; they are not the first-published numbers (docs/data_layer_audit.md). Changes are computed only within the same era."))
    if late:
        w.append(warn("SUSPECT_LAG", f"{t}: unusually late publication for {', '.join(late)}; the publication date is uncertain"))
    if len(rows) < 10:
        w.append(warn("SMALL_SAMPLE", f"{t}: only {len(rows)} disclosures in the window; improving vs deteriorating comparison is anecdotal"))
    if len(peers) < 2:
        w.append(warn("PEER_GROUP_TOO_SMALL", f"{t}: fewer than 2 peers; sector-relative reaction not computed"))
    data = {"ticker": t, "kind": kind, "metrics": metrics, "peers": peers, "window": [start, end], "horizons": p.horizons,
            "reaction_convention": "close before the reaction day -> close of the h-th trading day from effective_trade_date (first trading day on which prices can reflect the KAP publication)",
            "rows": rows, "summary": summary, "association_only": "no causal claim"}
    return data, w, {"as_of": as_of, "last_bar_date": bars.index.max(),
                     "sources": ["fundamental_metrics.parquet (Is Yatirim)", "fundamental_calendar.parquet (KAP first publication)", "prices_clean.parquet", "index_clean.parquet"]}
