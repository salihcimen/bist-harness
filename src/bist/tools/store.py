"""Read-only access to the local data cache with a hard as_of cut (nothing after as_of is ever returned)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..paths import PROCESSED
from ..universe import load_universe
from .envelope import ToolError, warn

STALE_DAYS = 5  # calendar days between as_of and the last bar before a STALE_DATA warning


@lru_cache(maxsize=1)
def _universe() -> pd.DataFrame:
    return load_universe()


@lru_cache(maxsize=1)
def _stocks() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "prices_clean.parquet")


@lru_cache(maxsize=1)
def _index() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "index_clean.parquet")


def universe() -> pd.DataFrame:
    return _universe()


def tickers() -> list[str]:
    return _universe()["ticker"].tolist()


def resolve_ticker(raw: str) -> tuple[str, str]:
    """'thyao', 'THYAO.IS' -> ('THYAO', 'THYAO.IS'); unknown -> ToolError (never silently guess)."""
    t = raw.strip().upper().removesuffix(".IS")
    u = _universe()
    row = u[u["ticker"] == t]
    if row.empty:
        raise ToolError("UNKNOWN_TICKER", f"'{raw}' is not in the 30-stock universe. Valid tickers: {', '.join(tickers())}")
    return t, row.iloc[0]["yahoo_symbol"]


def data_last_date() -> pd.Timestamp:
    return _index()["date"].max()


def resolve_as_of(as_of) -> tuple[pd.Timestamp, list[dict]]:
    last = data_last_date()
    if as_of is None:
        return last, []
    ts = pd.Timestamp(as_of)
    if ts > last:
        return last, [warn("AS_OF_CLAMPED", f"as_of {ts.date()} is after the last available data day {last.date()}; using {last.date()}")]
    return ts, []


def _adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Dividend/split-consistent OHLC: scale O/H/L by adj_close/close (volume untouched)."""
    f = (df["adj_close"] / df["close"]).where(df["close"] > 0, 1.0)
    out = df.copy()
    for c in ("open", "high", "low"):
        out[c] = df[c] * f
    out["close"] = df["adj_close"]
    return out


def stock_bars(ticker: str, as_of: pd.Timestamp, adjusted: bool = True) -> pd.DataFrame:
    _, sym = resolve_ticker(ticker)
    df = _stocks()
    df = df[(df["symbol"] == sym) & (df["date"] <= as_of)].sort_values("date").set_index("date")
    if df.empty:
        raise ToolError("NO_DATA", f"No price data for {ticker} up to {as_of.date()}")
    return _adjust(df) if adjusted else df


def zero_volume_days(ticker: str, as_of: pd.Timestamp) -> int:
    _, sym = resolve_ticker(ticker)
    df = _stocks()
    return int(((df["symbol"] == sym) & df["volume_missing"] & (df["date"] <= as_of)).sum())


def index_bars(as_of: pd.Timestamp, adjusted: bool = False) -> pd.DataFrame:
    df = _index()
    df = df[df["date"] <= as_of].sort_values("date").set_index("date")
    if df.empty:
        raise ToolError("NO_DATA", f"No index data up to {as_of.date()}")
    return df


def staleness_warnings(bars: pd.DataFrame, as_of: pd.Timestamp, label: str) -> list[dict]:
    gap = (as_of - bars.index.max()).days
    out = []
    if gap > STALE_DAYS:
        out.append(warn("STALE_DATA", f"{label}: last bar {bars.index.max().date()} is {gap} days before as_of {as_of.date()}"))
    return out


@lru_cache(maxsize=1)
def _fund_metrics() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "fundamental_metrics.parquet")


@lru_cache(maxsize=1)
def _fund_calendar() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "fundamental_calendar.parquet")


@lru_cache(maxsize=1)
def _macro() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "macro_market_daily.parquet")


def fund_metrics() -> pd.DataFrame:
    return _fund_metrics()


def fund_calendar() -> pd.DataFrame:
    return _fund_calendar()


def macro() -> pd.DataFrame:
    return _macro()


TCMB_COLUMNS = ["date", "series", "value", "source", "publish_date"]


def empty_tcmb() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in ("date", "publish_date") else "float64" if c == "value" else "object")
                         for c in TCMB_COLUMNS})


@lru_cache(maxsize=1)
def _tcmb() -> pd.DataFrame:
    """TCMB EVDS series (python -m bist.fetch_evds); empty frame when the file was never built."""
    f = PROCESSED / "macro_tcmb.parquet"
    return pd.read_parquet(f) if f.exists() else empty_tcmb()


def tcmb() -> pd.DataFrame:
    return _tcmb()
