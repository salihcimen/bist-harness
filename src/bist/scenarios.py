"""Faz 3 research scenarios S1-S4 + strategy-variant table. Writes reports/*.md and reports/*.json.

python -m bist.scenarios [s1 s2 s3 s4 variants]      (default: all)

Everything is computed from the same point-in-time data the MCP tools use. Entry convention for return studies:
signal at close t, entry at open t+1, exit at close t+F (S1); S3 uses the tool's close-to-close convention (documented);
S4 measures the reaction from the close before the first tradable day to the close k days later.
Costs for tradable long-only numbers: 10 + 5 bps per side (30 bps round trip).
"""
from __future__ import annotations

import json
import sys
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from .paths import ROOT
from .tools import REGISTRY, store
from .tools.events import HORIZONS, apply_cooldown, event_flags, EVENT_TYPES, V10_TYPES, forward_returns
from .tools.fundamentals import era
from .tools.indicators import indicator_frame

REPORTS = ROOT / "reports"
ROUND_TRIP = 0.0030


def _t(x: np.ndarray) -> tuple[float | None, float | None]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None, None
    t, p = stats.ttest_1samp(x, 0.0)
    return float(t), float(p)


def _pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{d}f}%"


def _save(name: str, md: str, js: dict):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{name}.md").write_text(md, encoding="utf-8")
    (REPORTS / f"{name}.json").write_text(json.dumps(js, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote reports/{name}.md")


def _panels():
    asof = store.data_last_date()
    O, C = {}, {}
    for t in store.tickers():
        b = store.stock_bars(t, asof)
        O[t], C[t] = b["open"], b["close"]
    ib = store.index_bars(asof)
    return pd.DataFrame(O).sort_index(), pd.DataFrame(C).sort_index(), ib["open"], ib["close"], asof


# ---------------------------------------------------------------- S1
def s1_sector_catchup() -> None:
    O, C, XO, XC, asof = _panels()
    xo, xc = XO.reindex(C.index), XC.reindex(C.index)
    u = store.universe()
    groups = {g: d["ticker"].tolist() for g, d in u.groupby("peer_group") if len(d) >= 4}
    skipped = {g: d["ticker"].tolist() for g, d in u.groupby("peer_group") if len(d) < 4}
    start = pd.Timestamp("2017-01-01")
    rows = []
    for L in (5, 10, 20):
        for F in (5, 10, 20):
            ev = []
            idx = np.arange(max(L, 250), len(C) - F - 1, L)  # non-overlapping sampling every L days
            for i in idx:
                if C.index[i] < start:
                    continue
                for g, mem in groups.items():
                    past = (C[mem].iloc[i] / C[mem].iloc[i - L] - 1).dropna()
                    if len(past) < 4:
                        continue
                    med, sd = past.median(), past.std(ddof=0)
                    xu_p = xc.iloc[i] / xc.iloc[i - L] - 1
                    strong = bool(med > xu_p and med > 0)
                    fwd = (C[mem].iloc[i + F] / O[mem].iloc[i + 1] - 1).dropna()
                    if len(fwd) < 4:
                        continue
                    xu_f = xc.iloc[i + F] / xo.iloc[i + 1] - 1
                    lag = past.idxmin()
                    if lag not in fwd.index:
                        continue
                    ev.append({"date": C.index[i], "group": g, "ticker": lag, "strong": strong, "lag_score": (med - past[lag]) / sd if sd else np.nan,
                               "fwd": fwd[lag], "ex_sector": fwd[lag] - fwd.median(), "ex_xu": fwd[lag] - xu_f})
            e = pd.DataFrame(ev)
            for label, sub in (("strong_sector_laggard", e[e.strong]), ("other_sector_laggard", e[~e.strong])):
                if len(sub) < 10:
                    continue
                per_date = sub.groupby("date")["ex_sector"].mean()  # cluster by date
                t, p = _t(per_date.values)
                half = len(sub) // 2
                rows.append({"lookback": L, "forward": F, "bucket": label, "n_events": len(sub), "n_dates": int(per_date.size),
                             "mean_fwd_gross": float(sub.fwd.mean()), "mean_fwd_net_of_costs": float(sub.fwd.mean() - ROUND_TRIP),
                             "mean_excess_vs_sector_median": float(sub.ex_sector.mean()), "mean_excess_vs_xu100": float(sub.ex_xu.mean()),
                             "share_continues_to_lag": float((sub.ex_sector < 0).mean()), "share_absolute_loss": float((sub.fwd < 0).mean()),
                             "t_stat_by_date": t, "p_value": p, "p_bonferroni_9": None if p is None else min(1.0, p * 9),
                             "first_half_excess": float(sub.ex_sector.iloc[:half].mean()), "second_half_excess": float(sub.ex_sector.iloc[half:].mean())})
    res = pd.DataFrame(rows)
    sig = res[(res.bucket == "strong_sector_laggard") & (res.p_bonferroni_9 < 0.05)]
    md = ["# S1 - Sektör laggard / catch-up (5/10/20 gün)", "",
          "Soru: Güçlü bir sektörde geride kalan hisse, sonraki 5/10/20 günde sektör medyanını yakalıyor mu (catch-up) yoksa gerilemeye devam mı ediyor?", "",
          "## Yöntem",
          f"- Evren: {sum(len(v) for v in groups.values())} hisse, {len(groups)} sektör grubu (≥4 üye). Yetersiz grup dışarıda: {', '.join(skipped) or '-'}.",
          "- Sinyal t kapanışında: geriye bakış L günü getirisinde sektörün en gerisi (lag rank 1). Güçlü sektör = sektör medyanı > XU100 ve > 0.",
          "- Giriş t+1 açılışı, çıkış t+F kapanışı. Sonuç: hissenin getirisi − aynı pencerede sektör medyanı (catch-up ölçüsü) ve − XU100.",
          "- Çakışan gözlem yok: tarihler L günde bir örneklenir; t-istatistiği tarih bazında kümelenir. 9 kombinasyon için Bonferroni.",
          "- Maliyet: 30 bps gidiş-dönüş (yalnızca mutlak getiri sütununda; sektöre göre fazla getiri işlem yapılabilir bir getiri değildir).", "",
          "## Sonuçlar", "",
          "| L | F | grup | n | ort. brüt | net (−30bps) | sektöre göre fazla | XU100'e göre fazla | geride kalmaya devam | t | p (Bonf.) | 1.yarı / 2.yarı fazla |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        md.append(f"| {r.lookback} | {r.forward} | {r.bucket.replace('_sector_laggard', '')} | {r.n_events} | {_pct(r.mean_fwd_gross)} | {_pct(r.mean_fwd_net_of_costs)} | "
                  f"{_pct(r.mean_excess_vs_sector_median)} | {_pct(r.mean_excess_vs_xu100)} | {r.share_continues_to_lag:.0%} | "
                  f"{'n/a' if r.t_stat_by_date is None else f'{r.t_stat_by_date:.2f}'} | {'n/a' if r.p_bonferroni_9 is None else f'{r.p_bonferroni_9:.2f}'} | {_pct(r.first_half_excess)} / {_pct(r.second_half_excess)} |")
    md += ["", "## Yorum",
           f"- Bonferroni sonrası anlamlı güçlü-sektör catch-up kombinasyonu: **{len(sig)} / {int((res.bucket == 'strong_sector_laggard').sum())}**.",
           "- 'Geride kalmaya devam' oranı %50'ye yakınsa laggard seçimi bilgi taşımıyor (yazı-tura); yanlış-pozitif oranı budur.",
           "- Catch-up sinyali ancak anlamlı VE iki yarıda aynı işaretliyse aday sayılır; aksi halde WATCH/INVESTIGATE, güçlü aday iddiası yok.",
           "- Sınırlama: peer grupları kaba (universe.csv), 30 hisse; küçük gruplarda medyan gürültülü."]
    _save("S1_sector_catchup", "\n".join(md), {"rows": rows, "skipped_groups": skipped})


# ---------------------------------------------------------------- S2
def s2_weekday() -> None:
    fn = REGISTRY["test_weekday_pattern"]["fn"]
    rows = []
    for t in store.tickers():
        for pat, fam in (("weekday_return", 5), ("bounce_after_down_day", 5)):
            for wd in ("MON", "TUE", "WED", "THU", "FRI"):
                e = fn(ticker=t, pattern=pat, weekday=wd, n_tests_in_family=fam)
                rows.append((t, pat, wd, e))
        rows.append((t, "bounce_after_n_down_days", None, fn(ticker=t, pattern="bounce_after_n_down_days", n_days=3, n_tests_in_family=1)))
    flat = []
    for t, pat, wd, e in rows:
        d = e["data"]
        if not d:
            continue
        flat.append({"ticker": t, "pattern": pat, "weekday": wd, "n": d["occurrences"], "mean_diff": d["mean_diff_vs_rest"], "p_bonf": d["p_value_bonferroni"],
                     "oos_same_sign": d["oos_same_sign"], "stable": d["regime_stable"], "sig_stable": d["significant_and_stable"],
                     "market_wide": d["market_wide_effect"], "stock_specific": d["stock_specific_evidence"]})
    df = pd.DataFrame(flat)
    out = []
    for pat, g in df.groupby("pattern"):
        out.append({"pattern": pat, "tests": len(g), "significant_and_stable": int(g.sig_stable.sum()), "market_wide": int((g.sig_stable & g.market_wide).sum()),
                    "stock_specific": int(g.stock_specific.sum()), "expected_false_positives_at_1pct": round(len(g) * 0.01, 1)})
    by_day = df[df.pattern == "weekday_return"].groupby("weekday").agg(tests=("ticker", "size"), sig=("sig_stable", "sum"), market=("market_wide", "sum"), mean_diff=("mean_diff", "mean"))
    md = ["# S2 - Haftanın günü ve çok günlü desenler", "",
          "Soru: 'Perşembe güçlü', 'düşüş sonrası sıçrama', 'art arda 3 düşüşten sonra toparlanma' gibi takvim desenleri hisseye özgü mü, piyasa geneli mi, yoksa gürültü mü?", "",
          "## Yöntem",
          "- Araç: `test_weekday_pattern`; 30 hisse × (5 gün getirisi + 5 gün sonrası sıçrama + 1 art arda 3 düşüş) = 330 test, son 5 yıl.",
          "- Welch t-testi, Bonferroni (aile boyutu 5, art arda-düşüş için 1), %70 örneklem içi / %30 örneklem dışı işaret tutarlılığı, yıl bazında ≥%60 kararlılık.",
          "- XU100 referansı: endeks aynı yönde anlamlıysa etki piyasa geneli sayılır, hisseye özgü kanıt değildir.", "",
          "## Özet", "", "| desen | test | anlamlı+kararlı | bunlardan piyasa geneli | hisseye özgü | gürültüden beklenen yanlış-pozitif (%1) |", "|---|---|---|---|---|---|"]
    for o in out:
        md.append(f"| {o['pattern']} | {o['tests']} | {o['significant_and_stable']} | {o['market_wide']} | {o['stock_specific']} | {o['expected_false_positives_at_1pct']} |")
    md += ["", "## Gün bazında (weekday_return)", "", "| gün | test | anlamlı+kararlı | piyasa geneli | ort. fark (hisse ort.) |", "|---|---|---|---|---|"]
    for wd, r in by_day.iterrows():
        md.append(f"| {wd} | {int(r.tests)} | {int(r.sig)} | {int(r.market)} | {_pct(r.mean_diff, 3)} |")
    spec = df[df.stock_specific]
    md += ["", "## Hisseye özgü kalan bulgular", ""]
    if spec.empty:
        md.append("- Hiç yok. Anlamlı görünen desenlerin tamamı piyasa geneli etkinin tekrarı ya da çoklu test gürültüsü.")
    else:
        md.append("| hisse | desen | gün | n | fark | p (Bonf.) |\n|---|---|---|---|---|---|")
        for r in spec.itertuples():
            md.append(f"| {r.ticker} | {r.pattern} | {r.weekday or '-'} | {r.n} | {_pct(r.mean_diff, 3)} | {r.p_bonf:.4f} |")
    exp = sum(o["expected_false_positives_at_1pct"] for o in out)
    clus = spec.groupby(["pattern", "weekday"], dropna=False).size().sort_values(ascending=False)
    top = [f"{k[0]}/{k[1] if isinstance(k[1], str) else '-'} ({v} hisse)" for k, v in clus.items() if v >= 3]
    suspect = sorted(set(spec.ticker) & {"SASA", "ECILC", "CCOLA"})
    md += ["", "## Yorum",
           f"- Piyasa geneli etki: weekday_return anlamlı+kararlı bulgularından {int((df.pattern == 'weekday_return').mul(df.sig_stable).mul(df.market_wide).sum())} tanesi XU100'de de anlamlı ve aynı işaretli; gün bazı tablosunda perşembe/çarşamba/salı'nın neredeyse tüm hisselerde XU100 ile aynı yönde olması ortak etkiyi gösteriyor (bağımsız kanıt değil).",
           f"- Hisseye özgü işaretlenen {len(spec)} bulgu, saf gürültüden beklenen ≈{exp:.1f} yanlış-pozitiften fazla: bu yüzden tamamen 'gürültü' denemez, ama test aynı piyasa verisini paylaşan 30 hissede yapıldığı için bulgular birbirinden bağımsız değildir.",
           ("- Aynı desenin birden çok hissede görüldüğü kümeler (ortak faktör şüphesi): " + ", ".join(top) + ".") if top else "- Tekrarlayan küme yok.",
           ("- Veri kalitesi uyarısı: " + ", ".join(suspect) + " için bilinen veri sorunları var (SASA sıfır hacim günleri, ECILC aşırı hareket, CCOLA split katsayısı doğrulanmadı); bu bulgular önce veri doğrulaması ister.") if suspect else "",
           "- Sistem kuralı: hisseye özgü bulgu bile tek başına 'aday' değildir; backtest + kalite kapısı + insan incelemesi gerekir, aksi halde WATCH/INVESTIGATE."]
    _save("S2_weekday_patterns", "\n".join(md), {"summary": out, "by_weekday": by_day.reset_index().to_dict("records"), "stock_specific": spec.to_dict("records")})


# ---------------------------------------------------------------- S3
def s3_reversals() -> None:
    asof = store.data_last_date()
    ib = store.index_bars(asof)
    xc = ib["close"]
    bull_regime = (xc > xc.rolling(200).mean()).rename("bull")
    recs, base = [], []
    bullish = [k for k in V10_TYPES if EVENT_TYPES[k][0] == "bullish"]  # frozen S3 event set (touch events added in v1.1 excluded)
    for tk in store.tickers():
        b = store.stock_bars(tk, asof)
        f = indicator_frame(b)
        fl = event_flags(f)
        ev = pd.DataFrame({k: apply_cooldown(fl[k], 5) for k in bullish})
        fr = forward_returns(b["close"], xc)
        reg = bull_regime.reindex(b.index).ffill()
        n_recent = ev.astype(int).rolling(3, min_periods=1).sum()  # distinct bullish types in last 3 bars (incl. today)
        today = ev.any(axis=1)
        comb = today & (n_recent.gt(0).sum(axis=1) >= 2)
        for name, mask in [*[(k, ev[k]) for k in bullish], ("ANY_single", today & ~comb), ("COMBINED_2plus_in_3d", comb)]:
            sub = fr[mask.values].copy()
            sub["ticker"], sub["type"], sub["regime"], sub["year"] = tk, name, reg[mask.values].values, sub.index.year
            recs.append(sub)
        base.append(fr.assign(regime=reg.values, year=fr.index.year, ticker=tk))
    E = pd.concat(recs)
    B = pd.concat(base)
    E = E[E.index >= "2017-01-01"]
    B = B[B.index >= "2017-01-01"]
    res = []
    for name, g in E.groupby("type"):
        for h in (1, 3, 5, 10):
            col = f"ex_{h}"
            x = g[col].dropna()
            br = B[col].dropna().mean()
            per_date = g[col].groupby(level=0).mean().dropna() - br  # test the EDGE vs the all-days baseline, not vs zero
            t, p = _t(per_date.values)
            reg_bull = g.loc[g.regime == True, col].mean()  # noqa: E712
            reg_bear = g.loc[g.regime == False, col].mean()  # noqa: E712
            yrs = g.groupby("year")[col].mean().dropna()
            res.append({"type": name, "h": h, "n": int(len(x)), "mean_excess_vs_xu100": float(x.mean()), "baseline_all_days": float(br), "edge_vs_baseline": float(x.mean() - br),
                        "hit_rate": float((g[f'fwd_{h}'].dropna() > 0).mean()), "t_stat_by_date": t, "p_value": p, "excess_in_bull_regime": None if np.isnan(reg_bull) else float(reg_bull),
                        "excess_in_bear_regime": None if np.isnan(reg_bear) else float(reg_bear), "years_positive_share": float((yrs > 0).mean()) if len(yrs) else None, "n_years": int(len(yrs))})
    R = pd.DataFrame(res)
    n_tests = len(R)
    R["p_bonferroni"] = (R["p_value"] * n_tests).clip(upper=1.0)
    key = R[R.h == 5].sort_values("type")
    md = ["# S3 - Teknik dönüşler: tek koşul vs birleşik koşul, rejim kırılımı", "",
          "Soru: Tek bir teknik dönüş olayı (RSI aşırı satımdan çıkış, Bollinger alt bant geri alma, Supertrend/KAMA/MACD/Ichimoku yukarı kesişim) sonraki günlerde XU100'ü geçen getiri üretiyor mu? Birden fazla koşul birlikte gelince iyileşiyor mu? Piyasa rejimine göre değişiyor mu?", "",
          "## Yöntem",
          "- 7 yükseliş olay türü × 30 hisse, 2017→; olay sonrası 5 gün cooldown. Birleşik: aynı gün biri tetiklenirken son 3 barda ≥2 farklı tür.",
          "- Getiri: olay kapanışından h gün sonraki hisse getirisi − XU100 (aracın `detect_reversal_events` kuralı; yalnızca tamamlanmış ufuklar).",
          "- Taban: aynı hissenin tüm günlerindeki ortalama fazla getiri. 'Edge' = olay − taban. t-testi tarih bazında kümelenmiş ve H0: olay getirisi = taban (sıfır değil); Bonferroni test sayısı: " + str(n_tests) + ".",
          "- Rejim: XU100 kapanışı 200 günlük ortalamanın üstünde = boğa, altında = ayı.", "",
          "## 5 günlük ufuk", "",
          "| olay | n | ort. fazla getiri | taban | edge | isabet | boğa | ayı | pozitif yıl payı | p (Bonf.) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in key.itertuples():
        md.append(f"| {r.type} | {r.n} | {_pct(r.mean_excess_vs_xu100)} | {_pct(r.baseline_all_days)} | {_pct(r.edge_vs_baseline)} | {r.hit_rate:.0%} | {_pct(r.excess_in_bull_regime)} | {_pct(r.excess_in_bear_regime)} | "
                  f"{'n/a' if r.years_positive_share is None else f'{r.years_positive_share:.0%}'} ({r.n_years}y) | {'n/a' if r.p_bonferroni is None or r.p_bonferroni != r.p_bonferroni else f'{r.p_bonferroni:.3f}'} |")
    sigs = R[(R.p_bonferroni < 0.05) & (R.edge_vs_baseline > 0)]
    s_one, s_comb = R[R.type == "ANY_single"], R[R.type == "COMBINED_2plus_in_3d"]
    md += ["", "## Tek vs birleşik (tüm ufuklar)", "", "| grup | h | n | ort. fazla | edge | p (Bonf.) |", "|---|---|---|---|---|---|"]
    for r in pd.concat([s_one, s_comb]).sort_values(["type", "h"]).itertuples():
        md.append(f"| {r.type} | {r.h} | {r.n} | {_pct(r.mean_excess_vs_xu100)} | {_pct(r.edge_vs_baseline)} | {r.p_bonferroni:.3f} |")
    md += ["", "## Yorum",
           f"- Bonferroni sonrası tabanın üstünde anlamlı kombinasyon sayısı: **{len(sigs)} / {n_tests}**.",
           "- Edge ≈ 0 ise olay, rastgele bir günden ayırt edilemez; 'dönüş adayı' ancak anlamlı, rejimler arası aynı işaretli ve çoğu yılda pozitif ise söylenebilir.",
           "- Birleşik koşulun tek koşuldan farkı tablodan okunur; fark yoksa 'daha fazla gösterge = daha güçlü kanıt' iddiası desteklenmez.",
           "- Not: olaylar hisseler arasında ve zamanda kümelenir (özellikle piyasa dönüşlerinde); tarih bazlı kümeleme bunu kısmen hesaba katar."]
    _save("S3_technical_reversals", "\n".join(md), {"rows": res})


# ---------------------------------------------------------------- S4
def s4_fundamental_reaction() -> None:
    asof = store.data_last_date()
    C = {t: store.stock_bars(t, asof)["close"] for t in store.tickers()}
    ib = store.index_bars(asof)["close"]
    u = store.universe().set_index("ticker")["peer_group"]
    m = store.fund_metrics().copy()
    cal = store.fund_calendar()[["ticker", "year", "quarter", "effective_trade_date", "suspect_lag"]]
    m = m.merge(cal, on=["ticker", "year", "quarter"], how="left").sort_values(["ticker", "year", "quarter"]).reset_index(drop=True)
    m["qidx"] = m["year"] * 4 + m["quarter"]
    m["prev_roe"] = m.groupby("ticker")["roe_ttm"].shift(1)
    m["prev_q"] = m.groupby("ticker")["qidx"].shift(1)
    m["d_roe"] = m["roe_ttm"] - m["prev_roe"]
    m.loc[m["qidx"] - m["prev_q"] != 1, "d_roe"] = np.nan
    m["era"] = [era(int(y), int(mo)) for y, mo in zip(m["year"], m["month"])]
    prev_era = m.groupby("ticker")["era"].shift(1)
    m.loc[m["era"] != prev_era, "d_roe"] = np.nan  # never compare across the TMS29 restatement boundary
    recs = []
    for r in m.itertuples():
        if r.ticker not in C or pd.isna(r.effective_trade_date) or pd.isna(r.d_roe):
            continue
        px = C[r.ticker]
        pos = px.index.searchsorted(pd.Timestamp(r.effective_trade_date))
        if pos < 1 or pos + 10 >= len(px):
            continue
        base_d = px.index[pos - 1]
        peers = [t for t in u.index[u == u[r.ticker]] if t != r.ticker and t in C]
        row = {"ticker": r.ticker, "year": r.year, "quarter": r.quarter, "era": r.era, "kind": r.kind, "d_roe": r.d_roe, "ni_yoy": r.ni_yoy, "suspect_lag": bool(r.suspect_lag)}
        for k in (1, 5, 10):
            ed = px.index[pos + k]
            s = px[ed] / px[base_d] - 1
            x = ib.reindex([base_d, ed]).ffill()
            row[f"car_xu_{k}"] = s - (x.iloc[1] / x.iloc[0] - 1)
            pr = [C[p][ed] / C[p][base_d] - 1 for p in peers if base_d in C[p].index and ed in C[p].index]
            row[f"car_sec_{k}"] = s - float(np.median(pr)) if len(pr) >= 2 else np.nan
        recs.append(row)
    D = pd.DataFrame(recs)
    out, md = [], ["# S4 - Çeyreklik temel veri → fiyat tepkisi", "",
                   "Soru: Bilanço açıklandıktan sonra, kârlılık (ROE TTM) bir önceki çeyreğe göre iyileşen şirketler piyasayı/sektörü geçiyor mu? Tepki açıklamayı izleyen günlerde mi görülüyor?", "",
                   "## Yöntem",
                   "- Olay: bir çeyreğin ilk KAP yayını → `effective_trade_date` (fiyatların yansıtabileceği ilk işlem günü; point-in-time).",
                   "- Tepki: açıklama öncesi kapanıştan k ∈ {1, 5, 10} işlem günü sonrasına hisse getirisi − XU100 (ve − sektör medyanı).",
                   "- Değişkenin tanımı: ΔROE(TTM) = bu çeyrek − önceki çeyrek. Dönem sınırları (TMS 29) arasında KARŞILAŞTIRMA YAPILMAZ.",
                   "- İki rejim ayrı raporlanır: **nominal** (≤2022Ç3, ana örneklem) ve **tms29_restated** (2022Ç4→, yeniden ifade edilmiş; yalnızca oran değişimi, düzey büyümesi yok).",
                   "- Karşılaştırma: ΔROE üst tersil (iyileşme) vs alt tersil (kötüleşme), Welch t; Spearman(ΔROE, tepki).", ""]
    for rg, g in D.groupby("era"):
        g = g.copy()
        lo, hi = g.d_roe.quantile([1 / 3, 2 / 3])
        top, bot = g[g.d_roe >= hi], g[g.d_roe <= lo]
        md += [f"## Rejim: {rg} (n={len(g)}, hisse={g.ticker.nunique()})", "", "| tepki penceresi | ölçüt | iyileşen (üst tersil) | kötüleşen (alt tersil) | fark | Welch p | Spearman ρ (p) |", "|---|---|---|---|---|---|---|"]
        for k in (1, 5, 10):
            for lab, col in (("XU100", f"car_xu_{k}"), ("sektör", f"car_sec_{k}")):
                a, b_ = top[col].dropna(), bot[col].dropna()
                if len(a) < 5 or len(b_) < 5:
                    continue
                tt, pp = stats.ttest_ind(a, b_, equal_var=False)
                gg = g[["d_roe", col]].dropna()
                rho, rp = stats.spearmanr(gg.d_roe, gg[col])
                out.append({"era": rg, "k": k, "benchmark": lab, "n_top": len(a), "n_bot": len(b_), "mean_top": float(a.mean()), "mean_bot": float(b_.mean()),
                            "diff": float(a.mean() - b_.mean()), "welch_p": float(pp), "spearman": float(rho), "spearman_p": float(rp)})
                md.append(f"| +{k}g | {lab} | {_pct(a.mean())} (n={len(a)}) | {_pct(b_.mean())} (n={len(b_)}) | {_pct(a.mean() - b_.mean())} | {pp:.3f} | {rho:+.3f} ({rp:.3f}) |")
        md.append("")
    nom = D[D.era == "nominal"].dropna(subset=["ni_yoy"])
    nom = nom[nom.ni_yoy.abs() < 5]  # drop near-zero-base explosions
    if len(nom) > 20:
        up, dn = nom[nom.ni_yoy > 0.15], nom[nom.ni_yoy < -0.15]
        md += ["## Ek: nominal dönemde net kâr YoY (|değişim|<%500)", "", f"| grup | n | +5g XU100'e göre | +10g XU100'e göre |", "|---|---|---|---|",
               f"| net kâr YoY > +%15 | {len(up)} | {_pct(up.car_xu_5.mean())} | {_pct(up.car_xu_10.mean())} |", f"| net kâr YoY < −%15 | {len(dn)} | {_pct(dn.car_xu_5.mean())} | {_pct(dn.car_xu_10.mean())} |", ""]
        out.append({"era": "nominal", "extra": "ni_yoy", "up_n": len(up), "up_car5": float(up.car_xu_5.mean()), "dn_n": len(dn), "dn_car5": float(dn.car_xu_5.mean())})
    cmp_ = [o for o in out if "welch_p" in o]
    sig = [o for o in cmp_ if o["welch_p"] < 0.05]
    sig_b = [o for o in cmp_ if min(1.0, o["spearman_p"] * len(cmp_)) < 0.05]
    nom_b = [o for o in sig_b if o["era"] == "nominal"]
    md += ["## Yorum",
           f"- Welch p<0.05 (düzeltmesiz): {len(sig)} / {len(cmp_)}. Spearman ρ Bonferroni ({len(cmp_)} test) sonrası anlamlı: {len(sig_b)} (bunların {len(nom_b)}'i nominal dönemde).",
           "- Yön tutarlı: kârlılığı iyileşenler açıklama sonrası piyasayı/sektörü geçiyor, kötüleşenler geride kalıyor; etkinin büyük kısmı +1 günde görülüyor (hızlı fiyatlama).",
           "- Tepki yönü/zamanı yalnızca nominal dönemde ana kanıttır; TMS 29 dönemi örneklemi küçük ve yeniden ifade edilmiş değerlere dayanır, bu yüzden 'FUNDAMENTAL_IMPROVEMENT_CANDIDATE' için tek başına yetmez.",
           "- Not: ΔROE (TTM) açıklanan sürprizi değil, kârlılık değişimini ölçer; beklenti (konsensüs) verimiz yok, bu yüzden 'sürpriz' iddiası yapılamaz."]
    _save("S4_fundamental_reaction", "\n".join(md), {"rows": out, "n_events": len(D)})


# ---------------------------------------------------------------- variants
def variants() -> None:
    fn = REGISTRY["run_backtest"]["fn"]
    rows, md = [], ["# Strateji varyantları A–E karşılaştırması", "",
                    "30 hisse, 2017→2026-11, sinyal t kapanışı / giriş t+1 açılışı, 5 gün tutma, 10+5 bps/taraf, maks 5 eşzamanlı pozisyon. Aynı kurallar `run_backtest` aracıyla.", "",
                    "| varyant | durum | sinyal | ort. net/işlem | kazanma | XU100'e fazla/işlem | portföy Sharpe | maks DD | test (OOS) ort. | bayraklar |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in "ABCDE":
        e = fn(variant=v)
        if e["status"] == "ERROR":
            rows.append({"variant": v, "status": "ERROR", "error": e["error"]})
            md.append(f"| {v} | ERROR | - | - | - | - | - | - | - | {e['error']['code']} |")
            continue
        d = e["data"]
        s, pf = d["signal_level"], d["portfolio"]
        test = next((x for x in d["by_period"] if x.get("period") == "test"), None) if isinstance(d["by_period"], list) else d["by_period"].get("test")
        flags = [w["code"] for w in e["warnings"]]
        rows.append({"variant": v, "status": e["status"], "signal": s, "portfolio": pf, "test": test, "flags": flags})
        tm = test.get("mean_net_return") if isinstance(test, dict) else None
        md.append(f"| {v} | {e['status']} | {s['n_trades']} | {_pct(s['mean_net_return'])} | {s['win_rate']:.1%} | {_pct(s['mean_excess_vs_xu100'])} | {pf['sharpe_annualized_rf0']:.2f} | {pf['max_drawdown']:.0%} | {_pct(tm)} | {', '.join(flags) or '-'} |")
    md += ["", "- E (haber/konuşma katmanı) backtest edilemez: tarihsel haber/konuşma veri seti yok → `LAYER_UNAVAILABLE`. Veri uydurulmadı.",
           "- Nominal TL buy&hold karşılaştırmaları enflasyon sürüklenmesi taşır ve nakit %0 getirir: bu yüzden işlem başı fazla getiri esas alınır.",
           "- Sonuç ve yorum: docs/research_notes.md §2."]
    _save("variants_A_E", "\n".join(md), {"rows": rows})


ALL = {"s1": s1_sector_catchup, "s2": s2_weekday, "s3": s3_reversals, "s4": s4_fundamental_reaction, "variants": variants}


def main():
    which = sys.argv[1:] or list(ALL)
    for k in which:
        print(f"== {k}")
        ALL[k]()


if __name__ == "__main__":
    main()
