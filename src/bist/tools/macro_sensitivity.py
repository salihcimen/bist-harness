"""analyze_macro_sensitivity: historical co-movement of a stock with one macro factor (sample Q5).

Weekly (W-FRI) or monthly period returns of the stock and XU100 are paired with the factor's change over the same period
boundaries. The factor value used on trading day t is its last observation STRICTLY before t (D-1 rule; TCMB series by
publication date). Correlation, OLS beta, conditional means and per-year stability. Association only: no forecast.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import store
from .common import SMALL_N, resolve_range
from .envelope import ToolError, tool, warn
from .fundamentals import TCMB_SERIES

FACTORS = {  # factor -> (source series, kind)
    "usdtry": ("usdtry", "pct"), "eurtry": ("eurtry", "pct"), "brent": ("brent", "pct"), "gold": ("gold", "pct"),
    "dxy": ("dxy", "pct"), "sp500": ("sp500", "pct"), "vix": ("vix", "level"), "us10y": ("us10y", "level"),
    "policy_rate": ("policy_rate", "level"), "cpi_yoy": ("cpi_yoy", "level"),
}
SLOW_FACTORS = {"policy_rate", "cpi_yoy"}


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _fit(ret: np.ndarray, chg: np.ndarray) -> dict:
    r = _corr(ret, chg)
    if r is None:
        return {"correlation": None, "beta": None, "beta_t_stat": None}
    beta = float(np.cov(ret, chg, ddof=1)[0, 1] / np.var(chg, ddof=1))
    t = float(r * np.sqrt((len(ret) - 2) / (1 - r * r))) if abs(r) < 1 else None
    return {"correlation": r, "beta": beta, "beta_t_stat": t}


def sensitivity_core(stock: pd.Series, index: pd.Series, factor: pd.Series, frequency: str, factor_kind: str) -> dict:
    """stock/index: daily closes; factor: value usable on each trading day (already D-1 aligned to stock's index)."""
    rule = "W-FRI" if frequency == "weekly" else "ME"
    df = pd.DataFrame({"s": stock, "i": index.reindex(stock.index).ffill(), "f": factor.reindex(stock.index)})
    ends = df.groupby(df.index.to_period("W-FRI" if frequency == "weekly" else "M")).tail(1)
    ends = ends.dropna(subset=["s", "f"])
    sret = ends["s"].pct_change()
    iret = ends["i"].pct_change()
    chg = ends["f"].pct_change() if factor_kind == "pct" else ends["f"].diff()
    p = pd.DataFrame({"s": sret, "i": iret, "c": chg}).dropna()
    out = {"n_periods": int(len(p)), "period_rule": rule, **_fit(p["s"].to_numpy(), p["c"].to_numpy())}
    for name, m in (("factor_up", p["c"] > 0), ("factor_down", p["c"] < 0)):
        q = p[m]
        out[name] = {"n": int(len(q)), "mean_stock_return": float(q["s"].mean()) if len(q) else None,
                     "mean_excess_vs_xu100": float((q["s"] - q["i"]).mean()) if len(q) else None}
    yearly = []
    for y, g in p.groupby(p.index.year):
        yearly.append({"year": int(y), "n": int(len(g)), "correlation": _corr(g["s"].to_numpy(), g["c"].to_numpy()) if len(g) >= 4 else None})
    out["yearly"] = yearly
    signs = {np.sign(y["correlation"]) for y in yearly if y["correlation"] is not None}
    out["sign_stable"] = bool(len(signs) <= 1) if signs else None
    xf = _fit(p["i"].to_numpy(), p["c"].to_numpy())
    out["xu100"] = {"correlation": xf["correlation"], "beta": xf["beta"]}
    out["stock_specific"] = (None if out["correlation"] is None or xf["correlation"] is None
                             else bool(abs(out["correlation"] - xf["correlation"]) > 0.1))
    return out


def factor_series(name: str, as_of: pd.Timestamp) -> pd.Series:
    """value indexed by the date from which it is usable (TCMB: publish date), observations strictly before as_of."""
    if name in TCMB_SERIES:
        t = store.tcmb()
        d = t[(t["series"] == name) & (t["publish_date"] < as_of)]
        return pd.Series(d["value"].to_numpy(), index=pd.DatetimeIndex(d["publish_date"])).sort_index()
    m = store.macro()
    d = m[(m["series"] == name) & (m["date"] < as_of)]
    return pd.Series(d["value"].to_numpy(), index=pd.DatetimeIndex(d["date"])).sort_index()


def align_d1(trading_days: pd.DatetimeIndex, f: pd.Series) -> pd.Series:
    """For each trading day t: last factor value with date < t."""
    left = pd.DataFrame({"t": pd.DatetimeIndex(trading_days).astype("datetime64[ns]")})
    right = pd.DataFrame({"t": pd.DatetimeIndex(f.index).astype("datetime64[ns]"), "v": f.to_numpy()}).drop_duplicates("t", keep="last")
    j = pd.merge_asof(left, right, on="t", allow_exact_matches=False)
    return pd.Series(j["v"].to_numpy(), index=trading_days)


class MacroSensitivityParams(BaseModel):
    ticker: str = Field(description="BIST ticker, e.g. EREGL")
    factor: Literal["usdtry", "eurtry", "policy_rate", "cpi_yoy", "brent", "gold", "vix", "dxy", "sp500", "us10y"] = Field(
        description="Macro factor: usdtry/eurtry/brent/gold/dxy/sp500 as % change; vix, us10y, policy_rate (CBRT), cpi_yoy (CPI y/y) as level change.")
    start: date | None = Field(default=None, description="Period start (default: 3 years before end).")
    end: date | None = Field(default=None, description="Period end (default and maximum: as_of).")
    frequency: Literal["weekly", "monthly"] = Field(default="weekly", description="Return period.")
    as_of: date | None = Field(default=None, description="Analysis date; nothing after it is used.")


@tool("analyze_macro_sensitivity", MacroSensitivityParams,
      "Historical association between a stock and one macro factor over [start, end]: correlation, beta, mean stock return when the factor rose vs fell (and vs XU100), per-year stability and the same figures for XU100 (stock-specific or market-wide). D-1 aligned; association only, no forecast.")
def analyze_macro_sensitivity(p: MacroSensitivityParams):
    t, _ = store.resolve_ticker(p.ticker)
    as_of, w = store.resolve_as_of(p.as_of)
    start, end, wr = resolve_range(p.start, p.end, as_of, as_of - pd.DateOffset(years=3))
    w += wr
    src, kind = FACTORS[p.factor]
    if src in TCMB_SERIES and store.tcmb().empty:
        raise ToolError("SOURCE_NOT_AVAILABLE", f"{p.factor} needs TCMB EVDS data (python -m bist.fetch_evds); not built")
    f = factor_series(src, as_of)
    if f.empty:
        raise ToolError("NO_DATA", f"no observations of {p.factor} before {as_of.date()}")
    bars = store.stock_bars(t, as_of)
    w += store.staleness_warnings(bars, as_of, t)
    close = bars["close"].loc[start:end]
    idx = store.index_bars(as_of)["close"]
    d = sensitivity_core(close, idx, align_d1(close.index, f), p.frequency, kind)
    if d["n_periods"] < SMALL_N:
        w.append(warn("SMALL_SAMPLE", f"{t}: only {d['n_periods']} {p.frequency} periods; no reliable conclusion"))
    if d["sign_stable"] is False:
        w.append(warn("UNSTABLE_RELATIONSHIP", f"{t} vs {p.factor}: the sign of the yearly correlation changes across years; the relationship is not stable"))
    if p.factor in SLOW_FACTORS and p.frequency == "weekly":
        w.append(warn("STALE_FACTOR", f"{p.factor} changes rarely (monthly/at policy meetings); most weekly changes are zero, prefer frequency=monthly"))
    elif p.factor not in SLOW_FACTORS and (as_of - f.index.max()).days > 7:
        w.append(warn("STALE_FACTOR", f"last {p.factor} observation {f.index.max().date()} is more than 7 days before as_of"))
    data = {"ticker": t, "factor": p.factor, "factor_change": "percent change" if kind == "pct" else "level change",
            "frequency": p.frequency, "window": [start, end], **d,
            "timing_rule": "factor value on day t = last observation strictly before t (TCMB series by publication date)",
            "association_only": "correlation is not causation; no forecast"}
    return data, w, {"as_of": as_of, "last_bar_date": bars.index.max(),
                     "sources": ["prices_clean.parquet", "index_clean.parquet", "macro_tcmb.parquet (TCMB EVDS)" if src in TCMB_SERIES else "macro_market_daily.parquet"]}
