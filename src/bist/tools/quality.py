"""check_data_quality: the gate every analysis must pass before conclusions are drawn.

Verdicts (worst issue wins): PASS < PASS_WITH_WARNINGS < INVESTIGATE < ANALYSIS_UNSAFE < HARD_BLOCK.
The harness maps INVESTIGATE/ANALYSIS_UNSAFE/HARD_BLOCK to restricted final actions; this tool only reports.
"""
from __future__ import annotations

from datetime import date as Date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import fundamentals as fnd
from . import store
from .envelope import ToolError, tool, warn

LEVELS = ["PASS", "PASS_WITH_WARNINGS", "INVESTIGATE", "ANALYSIS_UNSAFE", "HARD_BLOCK"]
SEV = {"warning": 1, "investigate": 2, "unsafe": 3, "block": 4}
STALE_INVESTIGATE_DAYS = 5
STALE_UNSAFE_DAYS = 10
AS_OF_BEYOND_BLOCK_DAYS = 5


class EvidenceDate(BaseModel):
    source: str = Field(description="Name of the data/evidence item, e.g. 'fundamentals THYAO 2024Q2'")
    date: Date = Field(description="Date this evidence became public / was observed")


class CheckQualityParams(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=30, description="Tickers involved in the analysis")
    as_of: Date | None = Field(default=None, description="Analysis date")
    analysis_type: Literal["sector_catchup", "technical_reversal", "fundamental_reaction", "calendar_pattern", "general"] = Field(
        default="general", description="Decides which checks are mandatory (peer group for sector_catchup, fundamentals for fundamental_reaction).")
    uses_volume: bool = Field(default=False, description="True if the analysis relies on volume (zero-volume days then matter more).")
    evidence_dates: list[EvidenceDate] = Field(default_factory=list, description="Dates of the evidence actually used; any date after as_of is a leakage flag.")


def _issue(sev: str, code: str, scope: str, msg: str) -> dict:
    return {"severity": sev, "code": code, "scope": scope, "message": msg}


@tool("check_data_quality", CheckQualityParams,
      "Data-quality gate for an analysis: stale/missing data, unexplained extreme moves, zero-volume days, late/restated fundamentals, small peer groups, incomplete macro, look-ahead leakage. Returns a verdict PASS|PASS_WITH_WARNINGS|INVESTIGATE|ANALYSIS_UNSAFE|HARD_BLOCK.")
def check_data_quality(p: CheckQualityParams):
    issues: list[dict] = []
    last = store.data_last_date()
    req = pd.Timestamp(p.as_of) if p.as_of else last
    if req > last:
        gap = (req - last).days
        issues.append(_issue("block" if gap > AS_OF_BEYOND_BLOCK_DAYS else "warning", "AS_OF_BEYOND_DATA", "request",
                             f"as_of {req.date()} is {gap} days after the last data day {last.date()}; analysis can only use data up to {last.date()}"))
    as_of = min(req, last)
    checked = []
    for raw in p.tickers:
        try:
            t, _ = store.resolve_ticker(raw)
        except ToolError as e:
            issues.append(_issue("block", e.code, raw, e.message))
            continue
        checked.append(t)
        try:
            bars = store.stock_bars(t, as_of)
        except ToolError as e:
            issues.append(_issue("block", e.code, t, e.message))
            continue
        gap = (as_of - bars.index.max()).days
        if gap > STALE_UNSAFE_DAYS:
            issues.append(_issue("unsafe", "STALE_PRICES", t, f"last bar {bars.index.max().date()} is {gap} days before as_of"))
        elif gap > STALE_INVESTIGATE_DAYS:
            issues.append(_issue("investigate", "STALE_PRICES", t, f"last bar {bars.index.max().date()} is {gap} days before as_of"))
        if len(bars) < 60:
            issues.append(_issue("unsafe", "SHORT_HISTORY", t, f"only {len(bars)} bars"))
        elif len(bars) < 250:
            issues.append(_issue("warning", "SHORT_HISTORY", t, f"only {len(bars)} bars (< 1 year); long-window indicators are unreliable"))
        raw_bars = store.stock_bars(t, as_of, adjusted=False)
        recent = raw_bars.tail(20)
        nz = int(recent["volume_missing"].sum())
        if nz:
            issues.append(_issue("investigate" if (p.uses_volume or nz >= 5) else "warning", "ZERO_VOLUME_DAYS", t,
                                 f"{nz} zero-volume days in the last 20 bars"))
        big = raw_bars.tail(60)
        big = big[big["ret_1d"].abs() > 0.25]
        if len(big):
            issues.append(_issue("investigate", "EXTREME_MOVE", t, f"daily move above 25% on {', '.join(str(d.date()) for d in big.index[:3])} in the last 60 bars: possible unadjusted corporate action or bad tick"))
        if p.analysis_type in ("fundamental_reaction", "general"):
            av = fnd.available(t, as_of)
            if av.empty:
                if p.analysis_type == "fundamental_reaction":
                    issues.append(_issue("unsafe", "NO_FUNDAMENTALS", t, "no financial report public at as_of"))
            else:
                r = av.iloc[-1]
                for ww in fnd._flag_warnings(t, r, as_of):
                    c = ww["code"]
                    sev = "investigate" if c in ("LATE_FILING", "STALE_FUNDAMENTALS") else "warning"
                    issues.append(_issue(sev, c, t, ww["message"]))
    if p.analysis_type == "sector_catchup":
        u = store.universe()
        for t in checked:
            g = u.loc[u["ticker"] == t, "peer_group"].iloc[0]
            n = int((u["peer_group"] == g).sum())
            if n < 3:
                issues.append(_issue("investigate", "PEER_GROUP_TOO_SMALL", t, f"peer group {g} has {n} member(s); sector-relative analysis is not meaningful"))
    m = store.macro()
    latest_macro = m[m["date"] < as_of]["date"].max()
    if store.tcmb().empty:
        issues.append(_issue("warning", "MACRO_INCOMPLETE", "macro", f"missing: {', '.join(fnd.TCMB_SERIES)}"))
    if pd.notna(latest_macro) and (as_of - latest_macro).days > 7:
        issues.append(_issue("investigate", "STALE_MACRO", "macro", f"latest macro observation {latest_macro.date()}"))
    for e in p.evidence_dates:
        if pd.Timestamp(e.date) > as_of:
            issues.append(_issue("block", "LEAKAGE", e.source, f"evidence dated {e.date} is after as_of {as_of.date()} (look-ahead)"))
    worst = max((SEV[i["severity"]] for i in issues), default=0)
    verdict = LEVELS[worst]
    w = [warn("DATA_QUALITY_" + verdict, f"{sum(1 for i in issues if SEV[i['severity']] >= 2)} issue(s) at INVESTIGATE level or above")] if worst >= 2 else []
    data = {"verdict": verdict, "issues": issues, "tickers_checked": checked, "analysis_type": p.analysis_type,
            "n_issues_by_severity": {k: sum(1 for i in issues if i["severity"] == k) for k in SEV},
            "may_issue_candidate_signal": verdict in ("PASS", "PASS_WITH_WARNINGS")}
    return data, w, {"as_of": as_of, "sources": ["prices_clean.parquet", "fundamental_calendar.parquet", "macro_market_daily.parquet"]}
