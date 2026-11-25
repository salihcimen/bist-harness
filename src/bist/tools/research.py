"""rank_sector_relative_strength and test_weekday_pattern (calendar / multi-day patterns with significance diagnostics)."""
from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from . import store
from .common import resolve_range
from .envelope import ToolError, tool, warn

MIN_PEERS = 3
WEEKDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}


# ------------------------------------------------------------------ sector relative strength
class RankSectorParams(BaseModel):
    sector: str | None = Field(default=None, description="Peer group name (e.g. Financials, Retail_Consumer, Energy_Utilities, Metals_Mining, Chemicals, Auto_Durables, Construction_Materials, Transport, Telecom, Healthcare, Defense_Tech) or a ticker (its group is used).")
    tickers: list[str] | None = Field(default=None, min_length=1, max_length=30, description="Explicit peer list (e.g. the stocks named in the question); overrides `sector`.")
    horizon_days: int = Field(default=10, ge=1, le=60, description="Return horizon in trading days (typically 5, 10 or 20).")
    as_of: date | None = Field(default=None, description="Analysis date; nothing after it is used.")


def _group_members(sector: str | None, tickers: list[str] | None) -> tuple[str, list[str], bool]:
    """(group name, members, custom?). An explicit ticker list wins over `sector`; duplicates are dropped, unknown tickers raise."""
    if tickers:
        out = []
        for raw in tickers:
            t, _ = store.resolve_ticker(raw)
            if t not in out:
                out.append(t)
        return "custom", out, True
    if not sector:
        raise ToolError("MISSING_PARAM", "give `sector` (peer group or ticker) or an explicit `tickers` list")
    g, members = _group_of(sector)
    return g, members, False


def _group_of(sector: str) -> tuple[str, list[str]]:
    u = store.universe()
    s = sector.strip()
    by_group = {g.upper(): g for g in u["peer_group"].unique()}
    if s.upper() in by_group:
        g = by_group[s.upper()]
    elif s.upper().removesuffix(".IS") in set(u["ticker"]):
        g = u.loc[u["ticker"] == s.upper().removesuffix(".IS"), "peer_group"].iloc[0]
    else:
        raise ToolError("UNKNOWN_SECTOR", f"'{sector}' is not a peer group or ticker. Peer groups: {', '.join(sorted(by_group.values()))}")
    return g, u.loc[u["peer_group"] == g, "ticker"].tolist()


@tool("rank_sector_relative_strength", RankSectorParams,
      "Peer-group relative strength: each member's horizon return vs. the sector median and XU100, lag score and rank (1 = most lagging).")
def rank_sector_relative_strength(p: RankSectorParams):
    group, members, custom = _group_members(p.sector, p.tickers)
    as_of, w = store.resolve_as_of(p.as_of)
    h = p.horizon_days
    rets, last = {}, {}
    for t in members:
        b = store.stock_bars(t, as_of)
        w += store.staleness_warnings(b, as_of, t)
        if len(b) > h:
            rets[t] = float(b["close"].iloc[-1] / b["close"].iloc[-1 - h] - 1)
            last[t] = b.index.max()
    ib = store.index_bars(as_of)
    xu = float(ib["close"].iloc[-1] / ib["close"].iloc[-1 - h] - 1)
    if len(rets) < MIN_PEERS:
        w.append(warn("PEER_GROUP_TOO_SMALL", f"{group} has {len(rets)} usable member(s); fewer than {MIN_PEERS}, so sector-relative ranking is not meaningful"))
        data = {"sector": group, "horizon_days": h, "members": [{"ticker": t, "return": r} for t, r in rets.items()],
                "xu100_return": xu, "ranking_valid": False, "custom_group": custom}
        return data, w, {"as_of": as_of, "sources": ["prices_clean.parquet", "universe.csv (peer_group is our coarse mapping)"]}
    r = pd.Series(rets)
    med = float(r.median())
    sd = float(r.std(ddof=0)) or np.nan
    rows = []
    for t, v in r.items():
        rows.append({"ticker": t, "return": v, "excess_vs_sector_median": v - med, "lag_gap": v - med, "excess_vs_xu100": v - xu,
                     "lag_score": (med - v) / sd if sd == sd else None})
    rows.sort(key=lambda x: -(x["lag_score"] if x["lag_score"] is not None else -9))
    for i, x in enumerate(rows, 1):
        x["lag_rank"] = i
    data = {"sector": group, "horizon_days": h, "n_members": len(rows), "sector_median_return": med, "xu100_return": xu,
            "sector_vs_xu100": med - xu, "sector_strong": bool(med > xu and med > 0), "ranking_valid": True,
            "dispersion": sd, "ranking_lagging_first": rows, "custom_group": custom}
    if custom:
        w.append(warn("CUSTOM_PEER_GROUP", f"peer group is the explicit list {', '.join(members)}, not a predefined sector"))
    if len(rows) < 4:
        w.append(warn("SMALL_PEER_GROUP", f"{group} has only {len(rows)} members; median/dispersion are coarse"))
    return data, w, {"as_of": as_of, "last_bar_date": max(last.values()), "sources": ["prices_clean.parquet", "index_clean.parquet", "universe.csv (peer_group is our coarse mapping)"]}


# ------------------------------------------------------------------ weekday / multi-day patterns
Weekday = Literal["MON", "TUE", "WED", "THU", "FRI"]


class WeekdayPatternParams(BaseModel):
    ticker: str = Field(description="BIST ticker, e.g. THYAO")
    pattern: Literal["weekday_return", "bounce_after_down_day", "bounce_after_n_down_days", "weekday_sequence"] = Field(
        description="weekday_return: return on `weekday` vs other days. bounce_after_down_day: next-day return after a down `weekday`. bounce_after_n_down_days: next-day return after n consecutive down closes. weekday_sequence: in calendar weeks where every `down_days` day closed down, the cumulative return of `up_days` in the same week.")
    weekday: Weekday | None = Field(default=None, description="Required for weekday_return and bounce_after_down_day.")
    n_days: int = Field(default=3, ge=2, le=6, description="Consecutive down closes for bounce_after_n_down_days.")
    down_days: list[Weekday] | None = Field(default=None, min_length=1, max_length=4, description="weekday_sequence: days that must all close down, e.g. [MON, TUE, WED].")
    up_days: list[Weekday] | None = Field(default=None, min_length=1, max_length=4, description="weekday_sequence: days whose cumulative return is the outcome, e.g. [THU, FRI].")
    start: date | None = Field(default=None, description="Period start (default: 5 years before as_of).")
    end: date | None = Field(default=None, description="Period end (default and maximum: as_of).")
    oos_start: date | None = Field(default=None, description="Optional out-of-sample start: in-sample = [start, oos_start), out-of-sample = [oos_start, end], each with full statistics (replaces the 70/30 split).")
    cost_bps_round_trip: float = Field(default=30, ge=0, le=500, description="weekday_sequence: round-trip trading cost in basis points subtracted for the net mean.")
    as_of: date | None = Field(default=None, description="Analysis date; nothing after it is used.")
    n_tests_in_family: int = Field(default=5, ge=1, le=100, description="How many similar patterns were tried (Bonferroni correction). 5 = all weekdays.")


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    if len(a) < 2 or len(b) < 2:
        return None, None
    t, pv = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(pv)


def pattern_stats(close: pd.Series, pattern: str, weekday: str | None, n_days: int, start: pd.Timestamp, end: pd.Timestamp,
                  n_tests: int) -> dict:
    """Core engine on a close series (stock or index). Condition at day t, outcome measured at t (weekday_return) or t+1."""
    r = close.pct_change()
    df = pd.DataFrame({"ret": r, "dow": close.index.dayofweek})
    df["next_ret"] = df["ret"].shift(-1)
    df = df.loc[start:end]
    if pattern == "weekday_return":
        mask, val = df["dow"] == WEEKDAYS[weekday], "ret"
    elif pattern == "bounce_after_down_day":
        mask, val = (df["dow"] == WEEKDAYS[weekday]) & (df["ret"] < 0), "next_ret"
    else:
        run = ((r < 0).astype(int).rolling(n_days).sum() == n_days).reindex(df.index).fillna(False)
        mask, val = run, "next_ret"
    sub = df.dropna(subset=[val])
    mask = mask.reindex(sub.index).fillna(False).astype(bool)
    x, base, allv = sub.loc[mask, val].to_numpy(), sub.loc[~mask, val].to_numpy(), sub[val].to_numpy()
    tstat, pv = _welch(x, base)
    p_adj = min(1.0, pv * n_tests) if pv is not None else None
    cut = sub.index[int(len(sub) * 0.7)] if len(sub) else None

    def seg(idx_mask):
        a, b = sub.loc[mask & idx_mask, val], sub.loc[~mask & idx_mask, val]
        return {"n": int(len(a)), "mean_diff_vs_baseline": float(a.mean() - b.mean()) if len(a) and len(b) else None,
                "hit_rate": float((a > 0).mean()) if len(a) else None}

    ins, oos = (seg(sub.index <= cut), seg(sub.index > cut)) if cut is not None else ({}, {})
    yrs = []
    for _, g in sub.groupby(sub.index.year):
        m = mask.loc[g.index]
        if m.sum() >= 3 and (~m).sum() >= 3:
            yrs.append(float(g.loc[m, val].mean() - g.loc[~m, val].mean()))
    a, b = ins.get("mean_diff_vs_baseline"), oos.get("mean_diff_vs_baseline")
    consistent = bool(np.sign(a) == np.sign(b)) if a is not None and b is not None else None
    md = float(x.mean() - base.mean()) if len(x) and len(base) else None
    yrs_same = int(sum(1 for v in yrs if md is not None and np.sign(v) == np.sign(md)))
    regime_stable = bool(len(yrs) >= 3 and yrs_same / len(yrs) >= 0.6)
    return {
        "period": [sub.index.min(), sub.index.max()], "occurrences": int(len(x)),
        "mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None,
        "hit_rate": float((x > 0).mean()) if len(x) else None,
        "baseline_mean": float(allv.mean()) if len(allv) else None, "baseline_hit_rate": float((allv > 0).mean()) if len(allv) else None,
        "mean_diff_vs_rest": float(x.mean() - base.mean()) if len(x) and len(base) else None,
        "t_stat": tstat, "p_value": pv, "p_value_bonferroni": p_adj, "n_tests_in_family": n_tests,
        "in_sample_70pct": ins, "out_of_sample_30pct": oos, "oos_same_sign": consistent,
        "years_tested": len(yrs), "years_positive_diff": int(sum(1 for v in yrs if v > 0)),
        "years_same_sign_as_overall": yrs_same, "regime_stable": regime_stable,
        "significant_and_stable": bool(p_adj is not None and p_adj < 0.05 and consistent and regime_stable),
    }


MEASURED = {"weekday_return": "same-day close-to-close",
            "weekday_sequence": "cumulative close-to-close return of up_days within the same calendar week"}


def _seq_summary(out: pd.Series, cond: pd.Series, full: pd.Series, n_tests: int, cost_bps: float) -> dict:
    """out: up_days cumulative return per complete week (indexed by the week's last trading day); cond: condition-week mask."""
    x, rest = out[cond].to_numpy(), out[~cond].to_numpy()
    tstat, pv = _welch(x, rest)
    p_adj = min(1.0, pv * n_tests) if pv is not None else None
    md = float(x.mean() - rest.mean()) if len(x) and len(rest) else None
    yrs = []
    for _, g in out.groupby(out.index.year):
        m = cond.loc[g.index]
        if m.sum() >= 3 and (~m).sum() >= 3:
            yrs.append(float(g[m].mean() - g[~m].mean()))
    yrs_same = int(sum(1 for v in yrs if md is not None and np.sign(v) == np.sign(md)))
    mean = float(x.mean()) if len(x) else None
    return {"period": [out.index.min(), out.index.max()] if len(out) else [None, None], "n_weeks_total": int(len(out)),
            "occurrences": int(len(x)), "mean": mean, "median": float(np.median(x)) if len(x) else None,
            "hit_rate": float((x > 0).mean()) if len(x) else None, "full_pattern_count": int(full[cond].sum()),
            "baseline_mean": float(out.mean()) if len(out) else None, "baseline_hit_rate": float((out > 0).mean()) if len(out) else None,
            "mean_diff_vs_rest": md, "net_mean_after_cost": None if mean is None else mean - cost_bps / 1e4,
            "t_stat": tstat, "p_value": pv, "p_value_bonferroni": p_adj, "n_tests_in_family": n_tests,
            "years_tested": len(yrs), "years_positive_diff": int(sum(1 for v in yrs if v > 0)), "years_same_sign_as_overall": yrs_same,
            "regime_stable": bool(len(yrs) >= 3 and yrs_same / len(yrs) >= 0.6)}


def _weeks(close: pd.Series, down: list[str], up: list[str], start, end) -> tuple[pd.Series, pd.Series, pd.Series, int, int]:
    """Per complete calendar week (all down/up days traded): up_days cumulative return, condition mask, all-up mask; plus the number
    of weeks whose up days fall after `end` (incomplete forward window) and of weeks skipped for missing trading days (holidays)."""
    df = pd.DataFrame({"ret": close.pct_change(), "dow": close.index.dayofweek}).loc[start:end]
    wk = df.index.to_period("W-SUN")
    pv = df.assign(wk=wk).pivot_table(index="wk", columns="dow", values="ret", aggfunc="first").reindex(columns=range(5))
    last_day = pd.Series(df.index, index=wk).groupby(level=0).max()
    dcols, ucols = [WEEKDAYS[d] for d in down], [WEEKDAYS[d] for d in up]
    complete = pv[dcols + ucols].notna().all(axis=1)
    cut_off = pd.Series([pd.Period(x, "W-SUN").end_time.normalize() > pd.Timestamp(end) for x in pv.index], index=pv.index)
    incomplete = ~complete & pv[dcols].notna().all(axis=1) & cut_off
    n_incomplete, n_skipped = int(incomplete.sum()), int((~complete & ~incomplete).sum())
    pv = pv[complete]
    idx = pd.DatetimeIndex(last_day.loc[pv.index].to_numpy())
    cond = pd.Series((pv[dcols] < 0).all(axis=1).to_numpy(), index=idx)
    out = pd.Series(((1 + pv[ucols]).prod(axis=1) - 1).to_numpy(), index=idx)
    full = pd.Series((pv[ucols] > 0).all(axis=1).to_numpy(), index=idx)
    return out, cond, full, n_incomplete, n_skipped


def _significance(d: dict) -> None:
    d["significant_and_stable"] = bool(d["p_value_bonferroni"] is not None and d["p_value_bonferroni"] < 0.05
                                       and d["oos_same_sign"] and d["regime_stable"])


def sequence_stats(close: pd.Series, down: list[str], up: list[str], start: pd.Timestamp, end: pd.Timestamp, n_tests: int,
                   cost_bps: float) -> dict:
    """Calendar-week sequence pattern. Weeks missing any down/up trading day are skipped (holidays)."""
    out, cond, full, n_inc, n_skip = _weeks(close, down, up, start, end)
    d = _seq_summary(out, cond, full, n_tests, cost_bps)
    d["n_excluded_incomplete"], d["n_weeks_skipped_missing_days"] = n_inc, n_skip
    ins, oos = {}, {}
    if len(out):
        cut = out.index[int(len(out) * 0.7)]

        def seg(m):
            a, b = out[cond & m], out[~cond & m]
            return {"n": int(len(a)), "mean_diff_vs_baseline": float(a.mean() - b.mean()) if len(a) and len(b) else None,
                    "hit_rate": float((a > 0).mean()) if len(a) else None}
        ins, oos = seg(out.index <= cut), seg(out.index > cut)
    a, b = ins.get("mean_diff_vs_baseline"), oos.get("mean_diff_vs_baseline")
    d["in_sample_70pct"], d["out_of_sample_30pct"] = ins, oos
    d["oos_same_sign"] = bool(np.sign(a) == np.sign(b)) if a is not None and b is not None else None
    _significance(d)
    return d


def _stats(close: pd.Series, p: "WeekdayPatternParams", start, end) -> dict:
    if p.pattern == "weekday_sequence":
        return sequence_stats(close, p.down_days, p.up_days, start, end, p.n_tests_in_family, p.cost_bps_round_trip)
    return pattern_stats(close, p.pattern, p.weekday, p.n_days, start, end, p.n_tests_in_family)


def _apply_oos(d: dict, close: pd.Series, p: "WeekdayPatternParams", start, end, oos: pd.Timestamp) -> dict:
    """Explicit IS/OOS split: full statistics per segment; replaces the 70/30 fields."""
    ins = _stats(close, p, start, oos - pd.Timedelta(days=1))
    out = _stats(close, p, oos, end)
    for seg in (ins, out, d):
        for k in ("in_sample_70pct", "out_of_sample_30pct"):
            seg.pop(k, None)
    for seg in (ins, out):
        seg.pop("oos_same_sign", None)
        seg.pop("significant_and_stable", None)
    a, b = ins["mean_diff_vs_rest"], out["mean_diff_vs_rest"]
    d["in_sample"], d["out_of_sample"] = ins, out
    d["oos_same_sign"] = bool(np.sign(a) == np.sign(b)) if a is not None and b is not None else None
    _significance(d)
    return d


@tool("test_weekday_pattern", WeekdayPatternParams,
      "Tests a calendar/multi-day pattern: occurrences, conditional returns vs baseline, p-value with multiple-testing correction, in-sample vs out-of-sample, per-year stability and whether the effect is just market-wide (XU100 reference).")
def test_weekday_pattern(p: WeekdayPatternParams):
    t, _ = store.resolve_ticker(p.ticker)
    as_of, w = store.resolve_as_of(p.as_of)
    if p.pattern == "weekday_sequence":
        if not p.down_days or not p.up_days:
            raise ToolError("MISSING_PARAM", "pattern 'weekday_sequence' requires `down_days` and `up_days` (e.g. [MON,TUE,WED] and [THU,FRI])")
        if set(p.down_days) & set(p.up_days):
            raise ToolError("INVALID_PARAMS", "`down_days` and `up_days` must not overlap")
    elif p.pattern != "bounce_after_n_down_days" and p.weekday is None:
        raise ToolError("MISSING_PARAM", f"pattern '{p.pattern}' requires `weekday` (MON..FRI)")
    bars = store.stock_bars(t, as_of)
    w += store.staleness_warnings(bars, as_of, t)
    end = min(pd.Timestamp(p.end), as_of) if p.end else as_of
    start, end, wr = resolve_range(p.start, p.end, as_of, end - pd.DateOffset(years=5))  # default start: 5 years before the clipped end
    w += wr
    oos = pd.Timestamp(p.oos_start) if p.oos_start else None
    if oos is not None and not (start < oos <= end):
        raise ToolError("INVALID_DATE_RANGE", f"oos_start {oos.date()} must lie inside ({start.date()}, {end.date()}]")
    iclose = store.index_bars(as_of)["close"]
    d = _stats(bars["close"], p, start, end)
    ref = _stats(iclose, p, start, end)
    if oos is not None:
        d = _apply_oos(d, bars["close"], p, start, end, oos)
        ref = _apply_oos(ref, iclose, p, start, end, oos)
        if d["out_of_sample"]["occurrences"] < 10:
            w.append(warn("SMALL_SAMPLE", f"{t}: out-of-sample has only {d['out_of_sample']['occurrences']} occurrences"))
    market_wide = bool(ref["p_value"] is not None and ref["p_value"] < 0.05 and d["mean_diff_vs_rest"] is not None
                       and ref["mean_diff_vs_rest"] is not None and np.sign(ref["mean_diff_vs_rest"]) == np.sign(d["mean_diff_vs_rest"]))
    if d["occurrences"] < 10:
        w.append(warn("SMALL_SAMPLE", f"{t}: only {d['occurrences']} occurrences; no reliable conclusion possible"))
    if d.get("n_excluded_incomplete"):
        w.append(warn("INCOMPLETE_FORWARD_WINDOW", f"{t}: {d['n_excluded_incomplete']} week(s) whose up_days fall after the period end were excluded"))
    if market_wide:
        w.append(warn("MARKET_WIDE_EFFECT", "The same pattern is also significant in XU100 with the same sign: it is a market-wide effect, not specific to this stock (and it repeats across all stocks, so it is not independent evidence)."))
    if not d["significant_and_stable"] and d["occurrences"] >= 10:
        w.append(warn("NOT_SIGNIFICANT", "Pattern is not significant after multiple-testing correction and/or not stable out-of-sample; treat as noise, not as evidence."))
    d["stock_specific_evidence"] = bool(d["significant_and_stable"] and not market_wide)
    data = {"ticker": t, "pattern": p.pattern, "weekday": p.weekday,
            "n_days": p.n_days if p.pattern == "bounce_after_n_down_days" else None,
            "return_measured": MEASURED.get(p.pattern, "next-day close-to-close after the condition"),
            **d, "xu100_reference": {k: ref[k] for k in ("occurrences", "mean", "mean_diff_vs_rest", "p_value", "significant_and_stable")},
            "market_wide_effect": market_wide}
    if p.pattern == "weekday_sequence":
        data.update(down_days=p.down_days, up_days=p.up_days, cost_bps_round_trip=p.cost_bps_round_trip)
    if oos is not None:
        data["oos_start"] = oos
    return data, w, {"as_of": as_of, "last_bar_date": bars.index.max(), "sources": ["prices_clean.parquet", "index_clean.parquet"]}


test_weekday_pattern.__test__ = False  # not a pytest test (name starts with test_)
