"""get_fundamental_metrics / calculate_fundamental_change (point-in-time) and get_macro_context.

Point-in-time rule: a quarter is visible only from its effective_trade_date (first KAP publication, mapped to the first
trading day on which prices can reflect it). Quarters with no known publication timestamp are treated as NOT available.
TMS 29 rule (docs/data_layer_audit.md): from FY2022 on, Is Yatirim values are restated, not first-published; level-based
growth is therefore withheld there and only same-period ratios are reported.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import store
from .common import resolve_range
from .envelope import ToolError, tool, warn

COLS = {
    "industrial": ["revenue_ttm", "ebitda_ttm", "ni_ttm", "gross_margin_q", "ebitda_margin_q", "net_margin_q", "roe_ttm", "roa_ttm",
                   "net_debt_ebitda", "debt_equity", "current_ratio", "quick_ratio", "cash_conversion"],
    "bank": ["nii_ttm", "fees_ttm", "net_profit_ttm", "roe_ttm", "roa_ttm", "loan_deposit", "equity_assets", "nii_assets"],
    "insurer": ["premiums_written_net_ttm", "technical_balance_ttm", "net_profit_ttm", "roa_ttm"],
}
RATIO_COLS = {
    "industrial": ["gross_margin_q", "ebitda_margin_q", "net_margin_q", "roe_ttm", "roa_ttm", "net_debt_ebitda", "debt_equity", "current_ratio", "cash_conversion"],
    "bank": ["roe_ttm", "roa_ttm", "loan_deposit", "equity_assets", "nii_assets"],
    "insurer": ["roa_ttm"],
}
GROWTH_COLS = {"industrial": ["revenue_yoy", "ebitda_yoy", "ni_yoy"], "bank": ["ni_yoy"], "insurer": ["ni_yoy"]}
STALE_PERIOD_DAYS = 200


def era(year: int, month: int) -> str:
    return "nominal" if (year, month) <= (2022, 9) else "tms29_restated"


def available(ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Metric rows visible at as_of, oldest->newest, joined with publication info."""
    m = store.fund_metrics()
    m = m[m["ticker"] == ticker]
    cal = store.fund_calendar()
    cal = cal[cal["ticker"] == ticker][["year", "quarter", "first_pub_ts", "effective_trade_date", "lag_days", "suspect_lag", "corrected", "period_end"]]
    j = m.merge(cal, on=["year", "quarter"], how="left")
    j = j[j["effective_trade_date"].notna() & (j["effective_trade_date"] <= as_of)]
    return j.sort_values(["year", "month"]).reset_index(drop=True)


def _row_info(r: pd.Series) -> dict:
    return {"fiscal_year": int(r["year"]), "quarter": int(r["quarter"]), "period_end": r["period_end"], "first_published": r["first_pub_ts"],
            "effective_trade_date": r["effective_trade_date"], "lag_days": int(r["lag_days"]), "era": era(int(r["year"]), int(r["month"]))}


def _flag_warnings(ticker: str, r: pd.Series, as_of: pd.Timestamp) -> list[dict]:
    w = []
    if era(int(r["year"]), int(r["month"])) == "tms29_restated":
        w.append(warn("TMS29_RESTATED_VALUES", f"{ticker}: values for {int(r['year'])}Q{int(r['quarter'])} come from a source that restates history under inflation accounting (TMS 29), they are not the first-published numbers; use ratios, not levels or growth."))
    if bool(r["suspect_lag"]):
        w.append(warn("LATE_FILING", f"{ticker}: {int(r['year'])}Q{int(r['quarter'])} was published {int(r['lag_days'])} days after period end (unusually late); publication date is uncertain."))
    if (as_of - r["period_end"]).days > STALE_PERIOD_DAYS:
        w.append(warn("STALE_FUNDAMENTALS", f"{ticker}: latest available period ended {r['period_end'].date()}, over {STALE_PERIOD_DAYS} days before as_of."))
    if bool(r["corrected"]):
        w.append(warn("CORRECTED_FILING", f"{ticker}: a correction notice exists for {int(r['year'])}Q{int(r['quarter'])}; first-published values may differ."))
    return w


def _valuation(ticker: str, r: pd.Series, as_of: pd.Timestamp) -> tuple[dict | None, list[dict]]:
    if r["kind"] != "industrial":
        return None, []
    cap = store.fund_metrics().query("ticker == @ticker and paid_in_capital.notna() and paid_in_capital > 0", engine="python")
    if cap.empty:
        return None, []
    shares = float(cap.sort_values(["year", "month"]).iloc[-1]["paid_in_capital"])  # 1 TL nominal -> shares ~= capital
    px = float(store.stock_bars(ticker, as_of)["close"].iloc[-1])
    mcap = px * shares
    ebitda, ni, eq = r.get("ebitda_ttm"), r.get("ni_ttm"), r.get("eq")
    v = {"market_cap_approx": mcap,
         "pe_ttm": mcap / ni if ni and ni > 0 else None,
         "pb": mcap / eq if eq and eq > 0 else None,
         "ev_ebitda_ttm": (mcap + r["net_debt"]) / ebitda if ebitda and ebitda > 0 and not pd.isna(r.get("net_debt")) else None}
    return v, [warn("VALUATION_APPROX", "Market cap = split-adjusted price x latest known paid-in capital (1 TL nominal); approximate, ignores later rights issues.")]


class GetFundamentalParams(BaseModel):
    ticker: str = Field(description="BIST ticker, e.g. THYAO")
    as_of: date | None = Field(default=None, description="Analysis date: only quarters already public on that date are used.")
    period: str | None = Field(default=None, description="Optional specific quarter 'YYYYQn' (e.g. 2024Q2); it must be public at as_of. Default: latest public quarter.")


@tool("get_fundamental_metrics", GetFundamentalParams,
      "Point-in-time financial metrics of one stock (growth, profitability, leverage, liquidity, cash conversion; banks/insurer have their own set) with publication timestamps, data-era flag and valuation multiples.")
def get_fundamental_metrics(p: GetFundamentalParams):
    t, _ = store.resolve_ticker(p.ticker)
    as_of, w = store.resolve_as_of(p.as_of)
    av = available(t, as_of)
    if av.empty:
        raise ToolError("NO_FUNDAMENTALS", f"{t}: no financial report public on or before {as_of.date()}")
    if p.period:
        try:
            y, q = int(p.period[:4]), int(p.period[-1])
        except ValueError:
            raise ToolError("INVALID_PERIOD", f"period must look like 2024Q2, got '{p.period}'")
        sel = av[(av["year"] == y) & (av["quarter"] == q)]
        if sel.empty:
            raise ToolError("PERIOD_NOT_PUBLIC", f"{t} {p.period} is not public as of {as_of.date()} (latest public: {int(av.iloc[-1]['year'])}Q{int(av.iloc[-1]['quarter'])})")
        r = sel.iloc[-1]
    else:
        r = av.iloc[-1]
    kind = r["kind"]
    metrics = {c: r[c] for c in COLS[kind] if c in r}
    if era(int(r["year"]), int(r["month"])) == "nominal":
        metrics.update({c: r[c] for c in GROWTH_COLS[kind] if c in r})
    else:
        w.append(warn("GROWTH_WITHHELD", "YoY growth is withheld for the TMS 29 era (restated levels are not comparable across years); see ratio changes in calculate_fundamental_change."))
    w += _flag_warnings(t, r, as_of)
    val, vw = _valuation(t, r, as_of)
    w += vw
    if val is not None and era(int(r["year"]), int(r["month"])) == "tms29_restated":
        w.append(warn("VALUATION_TMS29_CAUTION", "P/E, P/B and EV/EBITDA combine a current market cap with restated (inflation-adjusted) earnings and book value; treat as indicative, compare within the same stock over time only with care."))
    if kind != "industrial":
        w.append(warn("NON_INDUSTRIAL_METRICS", f"{t} is a {kind}; industrial ratios (EBITDA, current ratio, net debt) do not apply and are not reported."))
    data = {"ticker": t, "kind": kind, "report": _row_info(r), "metrics": metrics, "valuation": val}
    return data, w, {"as_of": as_of, "sources": ["fundamental_metrics.parquet (Is Yatirim)", "fundamental_calendar.parquet (KAP first publication)"]}


class FundamentalChangeParams(BaseModel):
    ticker: str = Field(description="BIST ticker, e.g. THYAO")
    as_of: date | None = Field(default=None, description="Analysis date: only quarters already public on that date are used.")


@tool("calculate_fundamental_change", FundamentalChangeParams,
      "Change of the latest public quarter vs. the previous quarter and vs. the same quarter a year earlier: ratio deltas always; growth rates only in the nominal (pre-TMS 29) era.")
def calculate_fundamental_change(p: FundamentalChangeParams):
    t, _ = store.resolve_ticker(p.ticker)
    as_of, w = store.resolve_as_of(p.as_of)
    av = available(t, as_of)
    if av.empty:
        raise ToolError("NO_FUNDAMENTALS", f"{t}: no financial report public on or before {as_of.date()}")
    r = av.iloc[-1]
    kind = r["kind"]
    key = lambda row: (int(row["year"]), int(row["month"]))
    idx = {key(row): row for _, row in av.iterrows()}
    y, mth = key(r)
    prev_q = idx.get((y, mth - 3) if mth > 3 else (y - 1, 12))
    prev_y = idx.get((y - 1, mth))
    ratios = {}
    for c in RATIO_COLS[kind]:
        now = r[c]
        ratios[c] = {"now": now,
                     "vs_prev_quarter": (now - prev_q[c]) if prev_q is not None and not pd.isna(now) and not pd.isna(prev_q[c]) else None,
                     "vs_prev_year": (now - prev_y[c]) if prev_y is not None and not pd.isna(now) and not pd.isna(prev_y[c]) else None}
    growth = None
    if era(y, mth) == "nominal":
        growth = {c: r[c] for c in GROWTH_COLS[kind] if c in r}
    else:
        w.append(warn("GROWTH_WITHHELD", "YoY growth is withheld for the TMS 29 era (restated levels); only same-period ratio deltas are reported."))
    if prev_q is None:
        w.append(warn("NO_PREVIOUS_QUARTER", "Previous quarter is not available/public; quarter-over-quarter deltas are null."))
    if prev_y is None:
        w.append(warn("NO_PREVIOUS_YEAR", "Same quarter of the previous year is not available/public; year-over-year deltas are null."))
    if prev_y is not None and era(*key(prev_y)) != era(y, mth):
        w.append(warn("ERA_BOUNDARY", "The comparison quarter is in a different accounting era (pre/post TMS 29); ratio deltas across the boundary are indicative only."))
    w += _flag_warnings(t, r, as_of)
    data = {"ticker": t, "kind": kind, "report": _row_info(r),
            "compared_with": {"previous_quarter": f"{prev_q['year']:.0f}Q{prev_q['quarter']:.0f}" if prev_q is not None else None,
                              "previous_year": f"{prev_y['year']:.0f}Q{prev_y['quarter']:.0f}" if prev_y is not None else None},
            "ratio_changes": ratios, "growth_yoy": growth}
    return data, w, {"as_of": as_of, "sources": ["fundamental_metrics.parquet", "fundamental_calendar.parquet"]}


# ------------------------------------------------------------------ macro
PCT_SERIES = ["usdtry", "eurtry", "dxy", "brent", "gold", "sp500"]
LEVEL_SERIES = ["vix", "us10y", "fed_funds_effective", "fed_target_upper", "fed_target_lower"]
TCMB_SERIES = ["usdtry_tcmb", "policy_rate", "cpi_yoy"]
TCMB_PCT = {"usdtry_tcmb"}
MacroSeries = Literal["usdtry", "eurtry", "dxy", "brent", "gold", "sp500", "vix", "us10y", "fed_funds_effective", "fed_target_upper",
                      "fed_target_lower", "usdtry_tcmb", "policy_rate", "cpi_yoy"]


class MacroContextParams(BaseModel):
    as_of: date | None = Field(default=None, description="Analysis date. Values are taken from the last observation STRICTLY BEFORE this date (global closes happen after the BIST close; monthly CPI only after its publication date).")
    lookback_days: int = Field(default=20, ge=5, le=250, description="Window (calendar-day observations, ~trading days) for the change figures.")
    series: list[MacroSeries] | None = Field(default=None, min_length=1, max_length=14, description="Subset of series (default: all). TCMB: usdtry_tcmb (CBRT USD/TRY), policy_rate (CBRT funding rate), cpi_yoy (CPI y/y %).")
    start: date | None = Field(default=None, description="Optional period start: adds per-series start/end value, change, min/max and observation count over [start, end].")
    end: date | None = Field(default=None, description="Period end for the range figures (default and maximum: as_of).")


def series_obs(name: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """date,value of one macro series visible strictly before as_of (TCMB series: by publication date)."""
    if name in TCMB_SERIES:
        t = store.tcmb()
        d = t[(t["series"] == name) & (t["publish_date"] < as_of)]
    else:
        m = store.macro()
        d = m[(m["series"] == name) & (m["date"] < as_of)]
    return d[["date", "value"]].sort_values("date").reset_index(drop=True)


def _range(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    r = d[(d["date"] >= start) & (d["date"] <= end)]
    if r.empty:
        return None
    a, b = float(r["value"].iloc[0]), float(r["value"].iloc[-1])
    return {"start_value": a, "end_value": b, "change": b - a, "change_pct": b / a - 1 if a else None, "min": float(r["value"].min()),
            "max": float(r["value"].max()), "n_obs": int(len(r)), "first_date": r["date"].iloc[0], "last_date": r["date"].iloc[-1]}


@tool("get_macro_context", MacroContextParams,
      "Macro context at as_of (D-1 rule): USD/TRY, EUR/TRY, DXY, VIX, US 10Y, Fed funds and target range, Brent, gold, S&P 500, and TCMB series (CBRT USD/TRY, policy-rate proxy, CPI y/y by publication date) with changes; optional [start, end] range statistics per series.")
def get_macro_context(p: MacroContextParams):
    as_of, w = store.resolve_as_of(p.as_of)
    cutoff = as_of  # strictly before as_of
    have_tcmb = not store.tcmb().empty
    names = p.series or (PCT_SERIES + LEVEL_SERIES + (TCMB_SERIES if have_tcmb else []))
    rng = None
    if p.start is not None or p.end is not None:
        rs, re_, wr = resolve_range(p.start, p.end, as_of, as_of - pd.DateOffset(years=3))
        w += wr
        rng = (rs, re_)
    out, last_obs = {}, []
    for s in names:
        d = series_obs(s, cutoff)
        if d.empty:
            out[s] = None
            continue
        v = d["value"].to_numpy()
        n = p.lookback_days
        pct = s in PCT_SERIES or s in TCMB_PCT
        row = {"value": float(v[-1]), "observation_date": d["date"].iloc[-1]}
        if s == "cpi_yoy":
            row["unit"] = "percent y/y (monthly)"
        elif len(v) > n:
            row[f"change_{n}d_pct" if pct else f"change_{n}d"] = float(v[-1] / v[-1 - n] - 1) if pct else float(v[-1] - v[-1 - n])
        if rng is not None:
            row["range"] = _range(d, *rng)
        out[s] = row
        if s != "cpi_yoy":
            last_obs.append(d["date"].iloc[-1])
    m = store.macro()
    fed = out.get("fed_target_upper")
    if fed:
        d = m[(m["series"] == "fed_target_upper") & (m["date"] < cutoff)].sort_values("date")
        chg = d[d["value"].diff().fillna(0) != 0]
        fed["last_change_date"] = chg["date"].iloc[-1] if len(chg) else None
    vix = out.get("vix")
    regime = {"vix_regime": None if not vix else ("low" if vix["value"] < 15 else "normal" if vix["value"] < 25 else "high")}
    missing = [] if have_tcmb else TCMB_SERIES
    if missing:
        w.append(warn("MACRO_INCOMPLETE", f"Not available: {', '.join(missing)} (TCMB EVDS data not built: python -m bist.fetch_evds). Do not make statements about Turkish policy rates or inflation."))
    if last_obs and (cutoff - max(last_obs)).days > 7:
        w.append(warn("STALE_MACRO", f"Latest macro observation is {max(last_obs).date()}, more than 7 days before as_of."))
    data = {"series": out, "regime": regime, "unavailable_series": missing,
            "timing_rule": "each value is the last observation strictly before as_of (no same-day global close); monthly CPI is visible only after its publication date (assumed 3rd business day of the next month)"}
    if rng is not None:
        data["range_window"] = list(rng)
    src = ["macro_market_daily.parquet (Yahoo, NY Fed)"] + (["macro_tcmb.parquet (TCMB EVDS)"] if have_tcmb else [])
    return data, w, {"as_of": as_of, "sources": src}
