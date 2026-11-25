"""TCMB EVDS series -> data/processed/macro_tcmb.parquet (long format: date, series, value, source, publish_date).

python -m bist.fetch_evds        (needs EVDS_API_KEY in <project>/.env; the key is sent only in the `key` header, never printed)

Series (codes verified on EVDS3, 2026-11-21):
  usdtry_tcmb  TP.DK.USD.A.YTL      CBRT indicative USD/TRY buying rate, daily; publish_date = date
  policy_rate  TP.APIFON4           CBRT weighted average funding cost, daily (tracks the one-week repo policy rate; the repo
                                    series codes TP.PY.P01/P02 are rejected by the API); publish_date = date
  cpi_yoy      TP.TUKFIY2025.GENEL  CPI general index (2025=100), formulas=3 -> year-on-year % change, monthly; date = month end,
                                    publish_date ASSUMED = 3rd business day of the next month (TUIK calendar not loaded)
"""
from __future__ import annotations

import sys
import time
from datetime import date

import pandas as pd
import requests

from .kap_official import load_env
from .paths import PROCESSED

BASE = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
OUT = PROCESSED / "macro_tcmb.parquet"
START = date(2015, 1, 1)
SERIES = {  # name -> (code, formulas, frequency label, source note)
    "usdtry_tcmb": ("TP.DK.USD.A.YTL", None, "daily", "TCMB EVDS TP.DK.USD.A.YTL"),
    "policy_rate": ("TP.APIFON4", None, "daily", "TCMB EVDS TP.APIFON4 (weighted average funding cost, policy-rate proxy)"),
    "cpi_yoy": ("TP.TUKFIY2025.GENEL", 3, "monthly", "TCMB EVDS TP.TUKFIY2025.GENEL formulas=3 (y/y %); publish_date assumed: 3rd business day of next month"),
}


def fetch_series(code: str, start: date, end: date, key: str, formulas: int | None = None) -> pd.DataFrame:
    """[(raw_date, value)] for one EVDS series; empty values dropped."""
    url = f"{BASE}series={code}&startDate={start:%d-%m-%Y}&endDate={end:%d-%m-%Y}&type=json" + (f"&formulas={formulas}" if formulas else "")
    r = requests.get(url, headers={"key": key}, timeout=60)
    r.raise_for_status()
    col = code.replace(".", "_") + (f"-{formulas}" if formulas else "")
    rows = [(i["Tarih"], i.get(col)) for i in r.json().get("items", [])]
    df = pd.DataFrame(rows, columns=["raw", "value"]).dropna()
    df = df[df["value"].astype(str).str.strip() != ""]
    df["value"] = df["value"].astype(float)
    return df


def _third_business_day_next_month(month_end: pd.Timestamp) -> pd.Timestamp:
    first = month_end + pd.Timedelta(days=1)
    return pd.bdate_range(first, periods=3)[-1]


def build(key: str, end: date) -> pd.DataFrame:
    parts = []
    for name, (code, formulas, freq, note) in SERIES.items():
        chunks = []
        for y in range(START.year, end.year + 1):
            s, e = date(y, 1, 1), min(date(y, 12, 31), end)
            chunks.append(fetch_series(code, s, e, key, formulas))
            time.sleep(0.5)
        df = pd.concat(chunks, ignore_index=True)
        if freq == "daily":
            df["date"] = pd.to_datetime(df["raw"], format="%d-%m-%Y")
            df["publish_date"] = df["date"]
        else:
            df["date"] = pd.to_datetime(df["raw"], format="%Y-%m") + pd.offsets.MonthEnd(0)
            df["publish_date"] = df["date"].map(_third_business_day_next_month)
        df["series"], df["source"] = name, note
        parts.append(df[["date", "series", "value", "source", "publish_date"]].drop_duplicates(["date"]))
        print(f"{name}: {len(df)} rows, {df['date'].min().date()} .. {df['date'].max().date()}")
    return pd.concat(parts, ignore_index=True).sort_values(["series", "date"]).reset_index(drop=True)


def main() -> None:
    key = load_env().get("EVDS_API_KEY")
    if not key:
        raise SystemExit("EVDS_API_KEY not set in .env")
    out = build(key, date.today())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    sys.exit(main())
