# Araç şeması v1.1 — Uygulama planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hocanın 5 örnek sorusunu (ve benzerlerini) sorudaki tarih aralığıyla, uydurma sayı ve look-ahead olmadan cevaplayabilen araç şeması v1.1'i uygulayıp yeniden dondurmak.

**Architecture:** v1.0 araçlarına yalnız isteğe bağlı parametreler eklenir (geriye uyumlu); 3 yeni araç (`test_catch_up_history`, `analyze_disclosure_reactions`, `analyze_macro_sensitivity`) ayrı modüllerde, saf "çekirdek" fonksiyon (Series/DataFrame girer, dict çıkar) + ince `@tool` sarmalayıcı olarak yazılır. Ortak tarih aralığı/ileri pencere kuralları tek yardımcı modülde. TCMB verisi yeni `fetch_evds.py` ile parquet'e alınır, `store` üzerinden okunur. Sonda harness (durumlar, kanıt, policy, referans ajan) bağlanır ve şema yeniden dışa aktarılır.

**Tech Stack:** Python 3.11.9, pandas, numpy, scipy, pydantic v2, pytest; requests + python-dotenv (EVDS).

**Spec:** `docs/superpowers/specs/2026-11-19-tool-schema-v1.1-design.md`

## Global Constraints

- Çalıştırma: `cd /c/Projects/econ/bist-harness && PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests -q`
- v1.0 çağrılarının davranışı değişmez; mevcut 81 test her görevin sonunda yeşil kalır (yalnız `test_schema_freeze.py` beklenen sürüm/araç sayısı her görevde güncellenir).
- Hiçbir hesap as_of sonrasını okumaz. `end > as_of` → as_of'a kırpılır + `END_CLIPPED_TO_AS_OF` uyarısı.
- İleri getiri: t+h tarihi as_of'u aşan gözlem istatistiğe girmez; ufuk başına `n_excluded_incomplete`; en az biri varsa `INCOMPLETE_FORWARD_WINDOW`.
- Getiriler kapanıştan kapanışa; `cost_bps_round_trip` varsayılan 30.
- n < 10 → `SMALL_SAMPLE`.
- `SCHEMA_VERSION = "1.1"`; zarf sözleşmesi (`envelope.py`) değişmez.
- Koşullar serbest metin değil, sabit listeden (`Literal`) seçilir.
- `.env` asla okunup yazdırılmaz/commit edilmez; anahtar yalnız `os.environ["EVDS_API_KEY"]` (python-dotenv ile yüklenir) ve `key` header'ında kullanılır, URL'de asla.
- Her görev, şemayı değiştirdiyse sonunda `python -m bist.tools.export_schemas` çalıştırıp `mcp/tool_schemas.json`'u commit'e ekler.

## Review Focus

1. **Tatil/eksik gün içeren haftalar (`weekday_sequence`)**: `down_days`'ten biri işlem günü değilse hafta sayılmaz; çökme ya da yanlış hafta eşleşmesi olmaz. → Task 4 testi `test_weekday_sequence_skips_incomplete_weeks`.
2. **as_of'tan sonra yayımlanan ya da yayın zamanı bilinmeyen çeyrek (`analyze_disclosure_reactions`)**: satır olarak hiç görünmez. → Task 6 testi `test_disclosure_after_as_of_invisible`.
3. **Hisse listesinde evren dışı ya da tekrarlı ticker (`tickers`/`peers`)**: `UNKNOWN_TICKER` ERROR zarfı; tekrarlar tekilleştirilir. → Task 2 testi `test_custom_group_unknown_and_duplicate_tickers`.
4. **`start > end` ya da `start` as_of'tan sonra**: `INVALID_DATE_RANGE` ERROR zarfı, boş istatistik değil. → Task 1 testi `test_resolve_range_rejects_inverted`.
5. **TCMB parquet'i yokken (`fetch_evds` hiç çalışmamış)**: `get_macro_context` v1.0 gibi çalışır, TCMB serilerini `unavailable_series`'te bildirir; `analyze_macro_sensitivity(factor="policy_rate")` `SOURCE_NOT_AVAILABLE` ERROR döner. → Task 7 testi `test_macro_without_tcmb_file`.

---

## Dosya yapısı

| Dosya | Sorumluluk |
|---|---|
| `src/bist/tools/common.py` (yeni) | `resolve_range`, `forward_window_stats`, sabitler |
| `src/bist/tools/research.py` | `rank_sector_relative_strength` (+tickers, lag_gap), `test_weekday_pattern` (+weekday_sequence, oos_start) |
| `src/bist/tools/catchup.py` (yeni) | `test_catch_up_history` |
| `src/bist/tools/events.py` | `detect_reversal_events` (+bollinger touch, start/end, condition_sets, events) |
| `src/bist/tools/disclosures.py` (yeni) | `analyze_disclosure_reactions` |
| `src/bist/fetch_evds.py` (yeni) | EVDS → `data/processed/macro_tcmb.parquet` |
| `src/bist/tools/store.py` | `tcmb()` okuyucu |
| `src/bist/tools/fundamentals.py` | `get_macro_context` (+start/end/series, TCMB) |
| `src/bist/tools/macro_sensitivity.py` (yeni) | `analyze_macro_sensitivity` |
| `src/bist/tools/__init__.py`, `export_schemas.py` | kayıt ve `ORDER` |
| `src/bist/harness/{states,policy,reference}.py`, `src/bist/tools/evidence.py` | harness entegrasyonu |
| `tests/conftest.py` (yeni) | `truncate_store` fixture'ı (nedensellik testleri) |
| `tests/test_tools_v11_*.py`, `tests/test_sample_questions.py` (yeni) | testler |

---

### Task 1: Ortak yardımcılar, nedensellik fixture'ı, sürüm 1.1

**Files:**
- Create: `src/bist/tools/common.py`, `tests/conftest.py`, `tests/test_tools_v11_common.py`
- Modify: `src/bist/tools/envelope.py:25`, `tests/test_schema_freeze.py:16`

**Interfaces:**
- Produces:
  - `COST_BPS_DEFAULT = 30`, `SMALL_N = 10`
  - `resolve_range(start: date|None, end: date|None, as_of: pd.Timestamp, default_start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, list[dict]]` — `end` yoksa as_of; `end > as_of` → as_of + `END_CLIPPED_TO_AS_OF`; `start` yoksa `default_start`; `start > end` → `ToolError("INVALID_DATE_RANGE", ...)`.
  - `forward_window_stats(event_dates: pd.DatetimeIndex, fwd: pd.DataFrame, horizons: list[int], cols: dict[str, str]) -> tuple[dict, list[dict]]` — `fwd` sütunları `f"{key}_{h}"` biçiminde (örn. `fwd_5`, `ex_5`), tamamlanmamışlar NaN. Ufuk başına `{"n", "n_excluded_incomplete", "mean", "median", "hit_rate", <key>_mean ...}` döner; `cols` hangi ek sütunların ortalamasının raporlanacağını verir (örn. `{"mean_excess_vs_xu100": "ex"}`). Hariç tutulan varsa `INCOMPLETE_FORWARD_WINDOW` (ufuklar mesajda), herhangi bir ufukta `0 < n < SMALL_N` ise `SMALL_SAMPLE`.
  - `fwd_frame(close: pd.Series, bench: pd.Series, horizons: list[int]) -> pd.DataFrame` — `fwd_h = close.shift(-h)/close-1`, `ex_h = fwd_h − bench getirisi`; seri as_of'ta bittiği için tamamlanmamış pencere zaten NaN olur.
  - `tests/conftest.py`: fixture `truncate_store(monkeypatch)` → `cut(date_str)` çağrılınca `store._stocks/_index/_macro` (ve Task 7'den sonra `_tcmb`) çıktıları `date <= cut`, `_fund_calendar` `first_pub_ts <= cut` olacak şekilde monkeypatch'lenir. Yardımcı `assert_causal(fn, cut, truncate_store, **kw)`: `fn(as_of=cut, **kw)["data"]` kesmeden önce ve sonra eşit olmalı.

- [ ] **Step 1: Failing testler** (`tests/test_tools_v11_common.py`)

```python
def test_resolve_range_clips_end_and_defaults():
    a = pd.Timestamp("2026-09-10")
    s, e, w = resolve_range(None, date(2026, 12, 1), a, pd.Timestamp("2021-09-10"))
    assert (s, e) == (pd.Timestamp("2021-09-10"), a) and [x["code"] for x in w] == ["END_CLIPPED_TO_AS_OF"]

def test_resolve_range_rejects_inverted():
    with pytest.raises(ToolError) as ex:
        resolve_range(date(2026, 1, 5), date(2025, 1, 1), pd.Timestamp("2026-09-10"), pd.Timestamp("2021-01-01"))
    assert ex.value.code == "INVALID_DATE_RANGE"

def test_forward_window_excludes_and_counts_incomplete():
    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.Series(np.arange(100, 130, dtype=float), index=idx)
    fwd = fwd_frame(close, close * 0 + 1.0, [5, 20])
    ev = idx[[2, 15, 27]]                      # 27+5 > son bar, 15+20 ve 27+20 > son bar
    st, w = forward_window_stats(ev, fwd, [5, 20], {"mean_excess_vs_xu100": "ex"})
    assert st["h5"]["n"] == 2 and st["h5"]["n_excluded_incomplete"] == 1
    assert st["h20"]["n"] == 1 and st["h20"]["n_excluded_incomplete"] == 2
    assert {x["code"] for x in w} == {"INCOMPLETE_FORWARD_WINDOW", "SMALL_SAMPLE"}
```

- [ ] **Step 2:** `pytest tests/test_tools_v11_common.py -q` → FAIL (`ModuleNotFoundError: bist.tools.common`).
- [ ] **Step 3:** `common.py`'yi yukarıdaki imzalarla yaz; `conftest.py` fixture'ını ekle; `SCHEMA_VERSION = "1.1"`; `test_schema_freeze.py`'de `"1.0"` → `"1.1"`; `python -m bist.tools.export_schemas`.
- [ ] **Step 4:** `pytest tests -q` → hepsi PASS (81 + 3).
- [ ] **Step 5: Commit** — `git add src/bist/tools/common.py src/bist/tools/envelope.py tests/conftest.py tests/test_tools_v11_common.py tests/test_schema_freeze.py mcp/tool_schemas.json && git commit -m "feat(tools): v1.1 ortak tarih araligi ve ileri pencere yardimcilari"`

---

### Task 2: `rank_sector_relative_strength` — `tickers` ve `lag_gap` (Soru 1)

**Files:** Modify `src/bist/tools/research.py:20-74` · Test `tests/test_tools_v11_research.py`

**Interfaces:**
- Produces: `RankSectorParams.tickers: list[str] | None` (min 1, max 30). Verilirse `sector` yok sayılır; `data["custom_group"] = True`, `data["sector"] = "custom"`; tekrarlar tekilleştirilir, bilinmeyen ticker `UNKNOWN_TICKER`. Her üye satırında `lag_gap` (= `excess_vs_sector_median`). `tickers` yoksa `custom_group = False`, gerisi v1.0 ile aynı. `sector` artık `str | None` olur ama `tickers` da yoksa `MISSING_PARAM` ERROR.
- Ayrıca `_group_members(sector, tickers) -> tuple[str, list[str], bool]` (grup adı, üyeler, custom mu) — Task 3 kullanır.

- [ ] **Step 1: Failing testler**

```python
def test_custom_group_retail_q1():
    r = rank_sector_relative_strength(tickers=["BIMAS", "MGROS", "MAVI"], horizon_days=20, as_of="2026-09-10")
    d = r["data"]
    assert r["status"] in ("OK", "WARN") and d["custom_group"] and d["n_members"] == 3
    worst = d["ranking_lagging_first"][0]
    assert worst["lag_gap"] == pytest.approx(worst["excess_vs_sector_median"]) and worst["lag_gap"] <= 0

def test_custom_group_unknown_and_duplicate_tickers():
    assert rank_sector_relative_strength(tickers=["BIMAS", "XXXX", "MAVI"])["error"]["code"] == "UNKNOWN_TICKER"
    d = rank_sector_relative_strength(tickers=["BIMAS", "bimas", "MGROS", "MAVI"])["data"]
    assert d["n_members"] == 3

def test_rank_sector_v10_call_unchanged():
    d = rank_sector_relative_strength(sector="THYAO", horizon_days=10, as_of="2026-09-10")["data"]
    assert d["custom_group"] is False and d["sector"] == "Transport"

def test_rank_sector_causal(truncate_store):
    assert_causal(rank_sector_relative_strength, "2026-09-10", truncate_store, tickers=["BIMAS", "MGROS", "MAVI"], horizon_days=20)
```

- [ ] **Step 2:** çalıştır → FAIL (`tickers` bilinmeyen alan değil ama `custom_group` KeyError).
- [ ] **Step 3:** uygula; `export_schemas`.
- [ ] **Step 4:** `pytest tests -q` → PASS.
- [ ] **Step 5:** commit `feat(tools): rank_sector_relative_strength ozel grup ve lag_gap`.

---

### Task 3: YENİ `test_catch_up_history` (Soru 1)

**Files:** Create `src/bist/tools/catchup.py`, `tests/test_tools_v11_catchup.py` · Modify `src/bist/tools/__init__.py`, `export_schemas.py` (`ORDER`'a `rank_sector_relative_strength`'ten sonra ekle), `tests/test_schema_freeze.py` (araç sayısı 17)

**Interfaces:**
- Consumes: `common.resolve_range`, `common.forward_window_stats`, `research._group_members`.
- Produces:
  - `catch_up_core(stock: pd.Series, peers: pd.DataFrame, bench: pd.Series, sector_strong: pd.Series, lookback: int, horizons: list[int], min_gap: float, start, end) -> tuple[dict, list[dict]]` — `peers` sütun başına bir peer'in kapanışı (ticker hariç).
  - Tool params: `ticker: str`, `peers: list[str] | None`, `lookback_days: int = 20 (5..60)`, `horizons: list[int] = [5, 10, 20]` (her biri 1..60), `min_lag_gap: float | None` (0..1, kesir), `start`, `end` (varsayılan start = end − 5 yıl), `as_of`.
  - `data`: `ticker, peers, custom_group, lookback_days, min_lag_gap_used, current_lag_gap, n_events, by_horizon{"h5": {n, n_excluded_incomplete, narrowed_rate, closed_rate, false_positive_rate, mean_rel, median_rel, mean_excess_vs_xu100}}, sector_strong_at_event{"true": by_horizon, "false": by_horizon}, events[{date, lag_gap, sector_strong, rel_fwd: {"5d": x|null}}]`.

Algoritma (spec 5.2; S1 ile aynı mantık): `rel_L[t] = stock L-gün getirisi − peer L-gün getirilerinin medyanı`. `min_gap = |min_lag_gap|` ya da verilmezse `|rel_L[as_of]|` (o da < 0.005 ise 0.02 kullan ve `min_lag_gap_used`'ta göster). Olay: `start ≤ t ≤ end` ve `rel_L[t] ≤ −min_gap`, son olaydan en az L bar sonra. `rel_fwd_h[t] = stock h-gün ileri getirisi − peer medyan h-gün ileri getirisi` (t+h > as_of ise NaN). `narrowed = rel_fwd_h > 0`; `closed = rel_fwd_h ≥ |rel_L[t]|`. `sector_strong[t] = peer medyanı L-gün > XU100 L-gün ve > 0`. Uyarılar: `SMALL_SAMPLE`, `INCOMPLETE_FORWARD_WINDOW`, peer < 2 (ticker hariç) → `PEER_GROUP_TOO_SMALL` ERROR değil WARN + istatistik null, `CUSTOM_PEER_GROUP` peers verildiyse. Durum: `RUN_SECTOR_RELATIVE_ANALYSIS` (Task 9'da bağlanır).

- [ ] **Step 1: Failing testler**

```python
def test_catch_up_core_known_answer():
    idx = pd.bdate_range("2024-01-01", periods=120)
    peer = pd.Series(100 * 1.001 ** np.arange(120), index=idx)
    stock = peer.copy(); stock.iloc[40:45] *= 0.9    # 5 gün geride kal, sonra tam kapat
    stock.iloc[45:] = peer.iloc[45:]
    peers = pd.DataFrame({"P1": peer, "P2": peer * 1.0})
    d, w = catch_up_core(stock, peers, peer, peer * 0 == 0, 5, [5], 0.05, idx[0], idx[-1])
    assert d["n_events"] == 1 and d["by_horizon"]["h5"]["narrowed_rate"] == 1.0 and d["by_horizon"]["h5"]["closed_rate"] == 1.0
    assert d["by_horizon"]["h5"]["false_positive_rate"] == 0.0

def test_catch_up_q1_mavi():
    r = test_catch_up_history(ticker="MAVI", peers=["BIMAS", "MGROS"], start="2024-01-01", end="2026-09-10", as_of="2026-09-10")
    d = r["data"]
    assert set(d["by_horizon"]) == {"h5", "h10", "h20"} and "CUSTOM_PEER_GROUP" in {x["code"] for x in r["warnings"]}
    assert all(e["date"] <= "2026-09-10" for e in d["events"])
    late = [e for e in d["events"] if e["date"] > "2026-08-12"]
    assert all(e["rel_fwd"]["20d"] is None for e in late)

def test_catch_up_invalid_params():
    assert test_catch_up_history(ticker="MAVI", horizons=[0])["status"] == "ERROR"

def test_catch_up_causal(truncate_store):
    assert_causal(test_catch_up_history, "2026-09-10", truncate_store, ticker="MAVI", peers=["BIMAS", "MGROS"], start="2024-01-01")
```

(`late` eşiği: 2026-09-10'dan 20 işlem günü öncesinin kaba üst sınırı; testte tam tarihi `bars.index[-21]` ile hesaplamak daha doğruysa öyle yaz.)

- [ ] **Step 2:** çalıştır → FAIL (modül yok).
- [ ] **Step 3:** uygula; `__init__` importu, `ORDER`, freeze test sayısı 17; `test_catch_up_history.__test__ = False`; `export_schemas`.
- [ ] **Step 4:** `pytest tests -q` → PASS.
- [ ] **Step 5:** commit `feat(tools): test_catch_up_history`.

---

### Task 4: `test_weekday_pattern` — `weekday_sequence` ve `oos_start` (Soru 2)

**Files:** Modify `src/bist/tools/research.py:78-178` · Test `tests/test_tools_v11_research.py`

**Interfaces:**
- Produces: `pattern` Literal'a `"weekday_sequence"`; yeni alanlar `down_days: list[Weekday] | None`, `up_days: list[Weekday] | None`, `oos_start: date | None`, `cost_bps_round_trip: float = 30 (0..500)`. `weekday_sequence` için ikisi de zorunlu ve kesişmemeli; değilse `MISSING_PARAM` / `INVALID_PARAMS`.
- `sequence_stats(close: pd.Series, down: list[str], up: list[str], start, end, n_tests, cost_bps) -> dict`. Hafta = ISO takvim haftası. Koşul haftası: `down` günlerinin hepsi işlem günü ve hepsinde `ret < 0`. Sonuç: o haftada `up` günlerinin kümülatif getirisi `Π(1+r)−1` (`up` günlerinden biri eksikse hafta dışarıda). Baseline: `up` günleri tam olan tüm haftaların aynı getirisi. Dönen alanlar: `n_weeks_total, occurrences (koşul haftası), mean, median, hit_rate, full_pattern_count, baseline_mean, baseline_hit_rate, mean_diff_vs_rest, net_mean_after_cost (= mean − cost/1e4), t_stat, p_value, p_value_bonferroni, n_tests_in_family, years_tested, years_same_sign_as_overall, regime_stable, significant_and_stable` + (oos_start yoksa) `in_sample_70pct, out_of_sample_30pct, oos_same_sign` mevcut tanımla.
- `oos_start` verilirse (her desen için): `in_sample` = [start, oos_start), `out_of_sample` = [oos_start, end] için **tam** istatistik seti (`in_sample`, `out_of_sample` anahtarları), `oos_same_sign` bunlardan; `in_sample_70pct/out_of_sample_30pct` anahtarları yazılmaz. `oos_start` aralık dışındaysa `INVALID_DATE_RANGE`.
- `stock_specific_evidence`, `MARKET_WIDE_EFFECT`, `NOT_SIGNIFICANT` mantığı yeni desen için aynen (XU100 aynı çekirdekle).

- [ ] **Step 1: Failing testler**

```python
def test_sequence_stats_synthetic():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    r = pd.Series(0.0, index=idx)
    r[idx.dayofweek <= 2] = -0.01; r[idx.dayofweek >= 3] = 0.01
    close = (1 + r).cumprod() * 100
    d = sequence_stats(close, ["MON", "TUE", "WED"], ["THU", "FRI"], idx[0], idx[-1], 1, 30)
    assert d["occurrences"] == d["full_pattern_count"] > 40 and d["hit_rate"] == 1.0
    assert d["mean"] == pytest.approx(1.01 ** 2 - 1) and d["net_mean_after_cost"] == pytest.approx(d["mean"] - 0.003)

def test_weekday_sequence_skips_incomplete_weeks():
    idx = pd.bdate_range("2024-01-01", "2024-03-29").drop(pd.Timestamp("2024-01-02"))   # 1. haftada TUE yok
    close = pd.Series(np.linspace(100, 90, len(idx)), index=idx)
    d = sequence_stats(close, ["MON", "TUE", "WED"], ["THU", "FRI"], idx[0], idx[-1], 1, 30)
    assert d["n_weeks_total"] == 12                                # 13 haftadan biri düştü

def test_weekday_sequence_q2_is_oos():
    r = test_weekday_pattern(ticker="THYAO", pattern="weekday_sequence", down_days=["MON", "TUE", "WED"], up_days=["THU", "FRI"],
                             start="2024-01-01", end="2026-09-10", oos_start="2026-01-01", as_of="2026-09-10")
    d = r["data"]
    assert d["in_sample"]["period"][1] < "2026-01-01" <= d["out_of_sample"]["period"][0]
    assert "in_sample_70pct" not in d and "net_mean_after_cost" in d["in_sample"]

def test_weekday_sequence_requires_days():
    assert test_weekday_pattern(ticker="THYAO", pattern="weekday_sequence")["error"]["code"] == "MISSING_PARAM"

def test_weekday_v10_unchanged():
    d = test_weekday_pattern(ticker="THYAO", pattern="weekday_return", weekday="MON", as_of="2026-09-10")["data"]
    assert "in_sample_70pct" in d
```

- [ ] **Step 2–4:** FAIL → uygula (+`export_schemas`) → `pytest tests -q` PASS.
- [ ] **Step 5:** commit `feat(tools): weekday_sequence deseni ve oos_start`.

---

### Task 5: `detect_reversal_events` — Bollinger dokunuşu, tarih aralığı, koşul kümeleri (Soru 3)

**Files:** Modify `src/bist/tools/events.py:19-134` · Test `tests/test_tools_v11_events.py`

**Interfaces:**
- Produces:
  - `EVENT_TYPES`'a `"bollinger_lower_touch": ("bullish", "low touched or pierced the lower Bollinger band")`, `"bollinger_upper_touch": ("bearish", "high touched or pierced the upper Bollinger band")`; `event_flags`'e `low <= bb_lower`, `high >= bb_upper`.
  - `CONDITIONS: dict[str, Callable[[pd.DataFrame], pd.Series]]` (anahtarlar sabit: `rsi_oversold` RSI<30, `rsi_trough` RSI[t]<RSI[t−1] ve <35, `high_volume` volume > 1.5×20 günlük ortalama (o gün `volume_missing` ise False), `supertrend_bearish/bullish` st_dir −1/1, `below/above_ichimoku_cloud`, `macd_hist_negative/positive`, `below/above_kama`) ve `condition_flags(f: pd.DataFrame) -> pd.DataFrame`. `indicator_frame` `volume`/`volume_missing` taşımıyorsa bars'tan eklenir.
  - Params: `start: date|None`, `end: date|None`, `condition_sets: list[list[ConditionName]] | None` (en fazla 6 küme, küme başına 1–4 koşul), `horizons` sabit `HORIZONS = (1,3,5,10)`.
  - `start`/`end` verilirse istatistik penceresi `[start, end]` (`lookback_days` yok sayılır; `data["window"] = [start, end]`), verilmezse v1.0 davranışı ve `history_summary` aynen.
  - Her türün `history_summary[ty]` satırına ek: `n_excluded_incomplete` (ufuk başına, `h{h}` içinde) ve `by_condition_set: [{"conditions": [...], "count", "h1".."h10"}]` (`common.forward_window_stats` ile).
  - Yeni `events` listesi yalnız `start` verildiğinde: `[{date, type, conditions: {name: bool}, fwd_returns: {"1d".."10d": x|null}}]` (en fazla 300; aşılırsa kırpılır ve `EVENTS_TRUNCATED`).
  - Uyarılar: `INCOMPLETE_FORWARD_WINDOW`, `SMALL_SAMPLE` (mevcut <5 eşiği korunur, yeni pencere modunda <10), `VOLUME_MISSING_DAYS` (`high_volume` kullanıldıysa ve pencerede hacmi eksik gün varsa).

- [ ] **Step 1: Failing testler**

```python
def test_condition_flags_rsi_trough_and_volume():
    n = 25
    f = pd.DataFrame({"close": 10.0, "rsi": [50.0] * 21 + [40, 34, 33, 36], "volume": [100.0] * 23 + [160.0, 160.0],
                      "volume_missing": [False] * 24 + [True], "st_dir": 1, "ich_span_a": 9.0, "ich_span_b": 8.0,
                      "macd_hist": 0.1, "kama": 9.5}, index=pd.bdate_range("2024-01-01", periods=n))
    c = condition_flags(f)
    assert c["rsi_trough"].tolist()[-4:] == [False, True, True, False]
    assert c["high_volume"].tolist()[-2:] == [True, False]       # son gün hacim eksik → False
    assert c["above_ichimoku_cloud"].iloc[-1] and c["above_kama"].iloc[-1] and c["supertrend_bullish"].iloc[-1]

def test_bollinger_touch_q3_asels():
    r = detect_reversal_events(ticker="ASELS", event_types=["bollinger_lower_touch"], start="2024-01-01", end="2026-09-10",
                               as_of="2026-09-10", cooldown_days=0,
                               condition_sets=[["rsi_trough", "high_volume"], ["supertrend_bearish", "below_ichimoku_cloud"]])
    s = r["data"]["history_summary"]["bollinger_lower_touch"]
    assert s["count"] == len(r["data"]["events"]) > 0
    assert len(s["by_condition_set"]) == 2 and all(c["count"] <= s["count"] for c in s["by_condition_set"])
    assert s["h5"]["n"] + s["h5"]["n_excluded_incomplete"] == s["count"]
    assert r["data"]["window"] == ["2024-01-01", "2026-09-10"]

def test_unknown_condition_rejected():
    r = detect_reversal_events(ticker="ASELS", condition_sets=[["moon_phase"]])
    assert r["error"]["code"] == "INVALID_PARAMS"

def test_reversal_v10_unchanged():
    d = detect_reversal_events(ticker="THYAO", as_of="2026-09-10")["data"]
    assert "events" not in d and "window" not in d

def test_reversal_window_causal(truncate_store):
    assert_causal(detect_reversal_events, "2026-09-10", truncate_store, ticker="ASELS", event_types=["bollinger_lower_touch"],
                  start="2024-01-01", condition_sets=[["rsi_trough", "high_volume"]])
```

(`high_volume` ortalaması t günü dahil son 20 günün hacmi; 160 > 1.5 × ~103 → True.)

- [ ] **Step 2–4:** FAIL → uygula (+`export_schemas`) → PASS.
- [ ] **Step 5:** commit `feat(tools): bollinger touch, tarih araligi ve kosul kumeleri`.

---

### Task 6: YENİ `analyze_disclosure_reactions` (Soru 4)

**Files:** Create `src/bist/tools/disclosures.py`, `tests/test_tools_v11_disclosures.py` · Modify `__init__.py`, `export_schemas.ORDER` (`calculate_fundamental_change`'ten sonra), freeze test (18)

**Interfaces:**
- Consumes: `fundamentals.available`, `fundamentals.era`, `fundamentals.RATIO_COLS`, `common.*`, `research._group_members`.
- Produces: params `ticker`, `start`, `end` (KAP `first_pub_ts` tarihine göre; varsayılan start = end − 3 yıl), `metrics: list[str] | None` (varsayılan: industrial `["ebitda_margin_q", "net_debt_ebitda"]`, bank `["roe_ttm", "nii_assets"]`, insurer `["roa_ttm"]`; türün `RATIO_COLS`'unda olmayan metrik → `UNKNOWN_METRIC`), `horizons: list[int] = [1, 5, 20]`, `peers: list[str] | None`, `as_of`.
- `data`: `ticker, kind, metrics, peers, rows[{period "2026Q2", first_published, effective_trade_date, era, values{m}, change_vs_prev_quarter{m}, improving, returns{"1d": {stock, abnormal_vs_xu100, abnormal_vs_sector, complete}}}], summary{h: {improving: {n, mean_abnormal_vs_xu100}, deteriorating: {...}, diff}}, association_only: "no causal claim"`.
- Kurallar: satırlar `available(ticker, as_of)` üzerinden (yalnız yayın tarihi ≤ as_of olanlar) ve `start ≤ first_pub_ts.date() ≤ end`. Getiri tabanı: `effective_trade_date`'ten bir önceki işlem günü kapanışı → `effective_trade_date + (h−1)` işlem günü kapanışı (yani tepki günü dahil h gün). Pencere sonu as_of'u aşıyorsa `complete = False` ve getiriler null. `change_vs_prev_quarter`: önceki çeyrek farklı era'daysa (`era()` değişiyorsa) null. `improving`: industrial için `ebitda_margin_q` artışı ve `net_debt_ebitda` düşüşü ikisi aynı yönde → True/False, çelişirse ya da biri null ise null; diğer türlerde ilk metriğin artışı. Sektör anormal getirisi: peer'lerin (ticker hariç) aynı pencere medyanı, peer < 2 ise null. Uyarılar: `INCOMPLETE_FORWARD_WINDOW` (mesajda `2026Q2 20d` gibi), `TMS29_RESTATED_VALUES` (tms29 era satırı varsa), `SUSPECT_LAG`, `SMALL_SAMPLE`.

- [ ] **Step 1: Failing testler**

```python
def test_q4_tuprs_rows_by_publication_date():
    r = analyze_disclosure_reactions(ticker="TUPRS", start="2024-01-01", end="2026-09-10", as_of="2026-09-10")
    d = r["data"]
    assert d["metrics"] == ["ebitda_margin_q", "net_debt_ebitda"] and len(d["rows"]) >= 9
    assert all("2024-01-01" <= x["first_published"][:10] <= "2026-09-10" for x in d["rows"])
    last = d["rows"][-1]
    bars = store.stock_bars("TUPRS", pd.Timestamp("2026-09-10"))
    pos = bars.index.get_loc(pd.Timestamp(last["effective_trade_date"]))
    assert last["returns"]["20d"]["complete"] == (pos + 19 < len(bars))
    if not last["returns"]["20d"]["complete"]:
        assert last["returns"]["20d"]["stock"] is None and "INCOMPLETE_FORWARD_WINDOW" in {w["code"] for w in r["warnings"]}
    assert d["association_only"] == "no causal claim" and "TMS29_RESTATED_VALUES" in {w["code"] for w in r["warnings"]}

def test_disclosure_after_as_of_invisible():
    d = analyze_disclosure_reactions(ticker="TUPRS", start="2024-01-01", as_of="2026-07-01")["data"]
    assert all(x["first_published"][:10] <= "2026-07-01" for x in d["rows"])

def test_unknown_metric():
    assert analyze_disclosure_reactions(ticker="AKBNK", metrics=["ebitda_margin_q"])["error"]["code"] == "UNKNOWN_METRIC"

def test_disclosures_causal(truncate_store):
    assert_causal(analyze_disclosure_reactions, "2026-09-10", truncate_store, ticker="TUPRS", start="2024-01-01")
```

- [ ] **Step 2–4:** FAIL → uygula (+`export_schemas`) → PASS.
- [ ] **Step 5:** commit `feat(tools): analyze_disclosure_reactions`.

---

### Task 7: TCMB verisi (`fetch_evds.py`) ve `get_macro_context` aralık modu (Soru 5)

**Files:** Create `src/bist/fetch_evds.py`, `tests/test_tools_v11_macro.py` · Modify `src/bist/tools/store.py`, `src/bist/tools/fundamentals.py:171-213`, `tests/conftest.py` (`_tcmb` kesimi), `requirements*.txt` (python-dotenv yoksa)

**Interfaces:**
- `fetch_evds.py`: `fetch_series(code: str, start: date, end: date, frequency: int|None=None) -> pd.DataFrame` (EVDS3 `https://evds3.tcmb.gov.tr/igmevdsms-dis/series=<code>&startDate=DD-MM-YYYY&endDate=DD-MM-YYYY&type=json`, header `key`); `main()` → `data/processed/macro_tcmb.parquet` sütunları `date, series, value, source, publish_date`. Seriler: `usdtry_tcmb` ← `TP.DK.USD.A.YTL` (publish_date = date); `policy_rate` ← bir haftalık repo kodu doğrulanırsa o, değilse `TP.APIFON4` (`source` alanına hangi kod kullanıldığı yazılır; publish_date = date); `cpi_yoy` ← TÜFE yıllık % (kodu uygulayıcı EVDS'de doğrular; aylık; `publish_date` = sonraki ayın 3. iş günü, `source`'a "publish_date assumed: 3rd business day of next month"). Başlangıç 2015-01-01. Anahtar yoksa `SystemExit("EVDS_API_KEY not set")`.
- `store.tcmb() -> pd.DataFrame` (dosya yoksa boş DataFrame, aynı sütunlar); okuma `publish_date <= as_of` filtresi araçlarda yapılır (TÜFE için `date` değil `publish_date`).
- `MacroContextParams` ek: `start`, `end`, `series: list[MacroSeries] | None` (Literal: mevcut 11 seri + `usdtry_tcmb`, `policy_rate`, `cpi_yoy`). TCMB serileri dosya varsa `series` çıktısına girer (D-1 kuralı: `publish_date < as_of`), `MISSING_MACRO` / `MACRO_INCOMPLETE` yalnız dosya yoksa. `start` verilirse her seri için ek `range: {start_value, end_value, change, change_pct, min, max, n_obs, first_date, last_date}`.
- Bu görevde EVDS'yi gerçekten çalıştır: `python -m bist.fetch_evds` (anahtar `.env`'de). Çıktıda anahtar görünmemeli.

- [ ] **Step 1: Failing testler**

```python
def test_macro_range_q5():
    r = get_macro_context(as_of="2026-09-10", start="2024-01-01", series=["usdtry", "usdtry_tcmb", "policy_rate", "cpi_yoy"])
    s = r["data"]["series"]
    assert s["usdtry_tcmb"]["range"]["n_obs"] > 400 and s["policy_rate"]["value"] is not None
    assert s["usdtry"]["range"]["last_date"] < "2026-09-10"
    assert "MACRO_INCOMPLETE" not in {w["code"] for w in r["warnings"]}

def test_cpi_publish_date_rule():
    d = get_macro_context(as_of="2026-09-02", series=["cpi_yoy"])["data"]["series"]["cpi_yoy"]
    assert d["observation_date"] <= "2026-07-31"          # Ağustos TÜFE'si 3 Eylül'den önce görünmez

def test_macro_without_tcmb_file(monkeypatch):
    monkeypatch.setattr(store, "_tcmb", lambda: store.empty_tcmb())
    r = get_macro_context(as_of="2026-09-10")
    assert "policy_rate" in r["data"]["unavailable_series"] and r["status"] == "WARN"

def test_macro_v10_unchanged():
    d = get_macro_context(as_of="2026-09-10")["data"]
    assert "range" not in d["series"]["usdtry"]

def test_macro_causal(truncate_store):
    assert_causal(get_macro_context, "2026-09-10", truncate_store, start="2024-01-01")
```

(`store.empty_tcmb()` boş çerçeve üreticisi olarak `store.py`'ye eklenir; `_tcmb` dosya yoksa onu döndürür.)

- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** `fetch_evds.py` yaz ve çalıştır; `store` + `get_macro_context` uygula; `export_schemas`.
- [ ] **Step 4:** `pytest tests -q` → PASS.
- [ ] **Step 5:** commit `feat(data,tools): EVDS TCMB serileri ve get_macro_context aralik modu` (parquet `data/` altında, gitignore'da; commit'e girmez).

---

### Task 8: YENİ `analyze_macro_sensitivity` (Soru 5)

**Files:** Create `src/bist/tools/macro_sensitivity.py`, `tests/test_tools_v11_macro.py` (ekle) · Modify `__init__.py`, `ORDER` (`get_macro_context`'ten sonra), freeze test (19)

**Interfaces:**
- Consumes: `store.macro()`, `store.tcmb()`, `common.resolve_range`.
- Produces:
  - `sensitivity_core(stock: pd.Series, index: pd.Series, factor: pd.Series, frequency: str, factor_kind: Literal["pct","level"]) -> dict` — `factor` zaten D-1 hizalı (değer t günü için t'den kesin önceki son gözlem).
  - Params: `ticker`, `factor: Literal["usdtry","eurtry","policy_rate","cpi_yoy","brent","gold","vix","dxy","sp500","us10y"]`, `start`, `end` (varsayılan start = end − 3 yıl), `frequency: Literal["weekly","monthly"] = "weekly"`, `as_of`.
  - `data`: `ticker, factor, frequency, n_periods, correlation, beta, beta_t_stat, factor_up{n, mean_stock_return, mean_excess_vs_xu100}, factor_down{...}, yearly[{year, n, correlation}], sign_stable, xu100{correlation, beta}, stock_specific (|corr_stock − corr_xu100| > 0.1 ve beta işaretleri), association_only: "correlation is not causation; no forecast"`.
  - Dönemler: hafta sonu (W-FRI) / ay sonu (M) son işlem günü kapanışlarından getiriler; faktör değişimi aynı dönem sınırlarında D-1 değerlerinden (`pct` serilerde yüzde, `level` serilerde (vix, us10y, policy_rate, cpi_yoy) fark). Faktör değişimi 0 olan dönemler `factor_up/down`'a girmez.
  - Uyarılar: `SMALL_SAMPLE` (n_periods < 10), `UNSTABLE_RELATIONSHIP` (≥ 2 yıl test edilip korelasyon işareti yıllar arasında değişiyorsa), `STALE_FACTOR` (factor ∈ {cpi_yoy, policy_rate} ve frequency == "weekly"; ya da son gözlem as_of'tan > 7 gün önce). TCMB faktörü dosyada yoksa `SOURCE_NOT_AVAILABLE` ERROR.

- [ ] **Step 1: Failing testler**

```python
def test_sensitivity_core_perfect_beta():
    idx = pd.bdate_range("2024-01-01", periods=300)
    f = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, .01, 300)), index=idx)
    stock = 50 * (f / f.iloc[0]) ** 2                     # log-beta 2
    d = sensitivity_core(stock, f * 0 + 1, f, "weekly", "pct")
    assert d["correlation"] > 0.99 and d["beta"] == pytest.approx(2, rel=0.1)

def test_q5_eregl_usdtry():
    r = analyze_macro_sensitivity(ticker="EREGL", factor="usdtry", start="2024-01-01", end="2026-09-10", as_of="2026-09-10")
    d = r["data"]
    assert d["n_periods"] > 100 and d["association_only"].startswith("correlation is not causation")
    assert "xu100" in d and len(d["yearly"]) == 3
    assert "target" not in json.dumps(d).lower() and "forecast" not in json.dumps({k: v for k, v in d.items() if k != "association_only"}).lower()

def test_sensitivity_policy_rate_missing_file(monkeypatch):
    monkeypatch.setattr(store, "_tcmb", lambda: store.empty_tcmb())
    assert analyze_macro_sensitivity(ticker="EREGL", factor="policy_rate")["error"]["code"] == "SOURCE_NOT_AVAILABLE"

def test_sensitivity_causal(truncate_store):
    assert_causal(analyze_macro_sensitivity, "2026-09-10", truncate_store, ticker="EREGL", factor="usdtry", start="2024-01-01")
```

- [ ] **Step 2–4:** FAIL → uygula (+`export_schemas`) → PASS.
- [ ] **Step 5:** commit `feat(tools): analyze_macro_sensitivity`.

---

### Task 9: Harness entegrasyonu

**Files:** Modify `src/bist/harness/states.py`, `src/bist/tools/evidence.py`, `src/bist/harness/policy.py:101-200`, `src/bist/harness/reference.py`, `src/bist/harness/machine.py` (yalnız gerekirse: araç çağrılarının tarihlerini `check_data_quality.evidence_dates`'e taşıma) · Test `tests/test_harness_v11.py`

**Interfaces:**
- `ALLOWED_TOOLS`: `RUN_SECTOR_RELATIVE_ANALYSIS` += `test_catch_up_history`; `LOAD_FUNDAMENTALS` += `analyze_disclosure_reactions`; `LOAD_MACRO_CONTEXT` += `analyze_macro_sensitivity`.
- `EXTRACTORS` += `test_catch_up_history: _catchup`, `analyze_disclosure_reactions: _disclosures`, `analyze_macro_sensitivity: _macro_sens`; `_reversal` `by_condition_set` ve pencere modunu, `_pattern` `weekday_sequence` + `in_sample/out_of_sample`'ı, `_sector` `custom_group`'u, `_macro` TCMB serileri ve `range`'i işler. Kurallar: her sayı araç çıktısından `_p`/`_n` ile; `SMALL_SAMPLE` veya `INCOMPLETE_FORWARD_WINDOW` olan istatistik `commit=False` değil, ama n < 10 olan ufuk satırı `commit=False, reason="SMALL_SAMPLE"`; tamamlanmamış pencere kendi `kind="data_gap"` maddesi olarak (`"20-day window incomplete: 2026Q2"`), `commit=True` (bu bir olgudur). `_macro_sens` maddeleri `association_only` notunu cümlede taşır.
- `policy.signals`: `catchup.present` için ek şart — son `test_catch_up_history` çağrısı OK/WARN ve `by_horizon["h10"]` için `n >= 10` ve `narrowed_rate > 0.5`; çağrı yoksa ya da şart sağlanmazsa `present=False, partial=True` (`history_ok` alanı eklenir). Eşikler `CATCHUP_MIN_N = 10`, `CATCHUP_MIN_NARROWED = 0.5` sabitleri.
- `reference.plan(...)` yeni anahtar parametreler: `start: str|None = None`, `end: str|None = None`, `peers: list[str]|None = None`, `pattern_kw: dict|None = None` (ör. `down_days/up_days/oos_start`), `reversal_kw: dict|None = None` (ör. `event_types/condition_sets`), `macro_factor: str|None = None`. Çağrı eklemeleri: `sector_catchup`/`general` → `rank_sector_relative_strength` (peers verildiyse `tickers=[ticker]+peers`) + `test_catch_up_history`; `fundamental_reaction` → `analyze_disclosure_reactions`; `macro_factor` verildiyse → `analyze_macro_sensitivity`; `start/end` ilgili tüm aralık destekli araçlara geçer; `get_macro_context`'e `start` geçer. `STAGES["fundamental_reaction"]` aynı kalır (LOAD_FUNDAMENTALS zaten var); `macro_factor` verildiğinde `LOAD_MACRO_CONTEXT` her odakta istenir.
- `check_data_quality` çağrısı plan sırasında önce geldiği için: `reference.plan`, `start/end` verildiğinde `evidence_dates=[{"source": "analysis window end", "date": end}]` geçer (end > as_of ise sızıntı bayrağı mevcut mantıkla kalkar).

- [ ] **Step 1: Failing testler**

```python
def test_new_tools_only_in_their_states(db):   # test_harness.test_tool_only_in_its_state kalıbıyla
    ...  # test_catch_up_history SELECT_UNIVERSE/LOAD_FUNDAMENTALS'ta reddedilir, RUN_SECTOR_RELATIVE_ANALYSIS'te kabul

def test_catchup_candidate_needs_history(db):
    calls = [<rank env: MAVI lag_rank 1, sector_strong True>, <backtest SUPPORTS>]           # sentetik çağrı listesi
    assert policy.signals(calls, "MAVI", "2026-09-10")["catchup"]["present"] is False
    calls.append(<catch-up env: h10 n=12 narrowed_rate=0.6>)
    assert policy.signals(calls, "MAVI", "2026-09-10")["catchup"]["present"] is True

def test_evidence_reports_incomplete_window_as_data_gap(db):
    h = reference.run("Q4", "TUPRS", "fundamental_reaction", as_of="2026-09-10", db=db, start="2024-01-01", end="2026-09-10")
    b = next(c for c in h.calls if c["tool"] == "build_evidence_bundle")["envelope"]["data"]
    assert all(it["as_of"] <= "2026-09-10" for it in b["observed"] if it["as_of"])
    # TUPRS son çeyreğin 20g penceresi tamamlanmadıysa data_gap maddesi bulunur

@pytest.mark.parametrize("q,tk,focus,kw", SAMPLE_RUNS)   # Task 10'daki 5 sorunun reference.run argümanları
def test_reference_passes_validator_on_samples(db, q, tk, focus, kw):
    h = reference.run(q, tk, focus, as_of="2026-09-10", db=db, **kw)
    assert h.state == "HUMAN_REVIEW"    # run() validator reddinde RuntimeError fırlatır
```

(Sentetik çağrı sözlükleri `policy._last`'ın beklediği `{"tool", "envelope"}` biçiminde; testte küçük bir `env(tool, data, warnings=())` yardımcısıyla kur. `SAMPLE_RUNS` Task 10'da tanımlanır; bu görevde 5 satırı burada yaz, Task 10 import eder.)

- [ ] **Step 2–4:** FAIL → uygula → `pytest tests -q` PASS (mevcut `test_reference_agent_all_foci` dahil).
- [ ] **Step 5:** commit `feat(harness): v1.1 araclarini durumlara, kanita, policy ve referans ajana bagla`.

---

### Task 10: Örnek soru kabul testi, yeniden dondurma, dokümantasyon

**Files:** Create `tests/test_sample_questions.py` · Modify `mcp/tool_schemas.json` (export), `ROADMAP.md`, `CLAUDE.md`, memory `bist-sample-questions-gap.md`

**Interfaces:**
- `SAMPLE_RUNS` (Task 9'da `tests/test_harness_v11.py`'de tanımlı) — 5 soru için `reference.run` argümanları:
  - Q1 `("...", "MAVI", "sector_catchup", {"peers": ["BIMAS", "MGROS"], "horizon_days": 20, "start": "2024-01-01", "end": "2026-09-10"})` (hangi hissenin en geride olduğu testte `rank_sector_relative_strength(tickers=...)` ile önce bulunur; referans o hisseyi kullanır)
  - Q2 `("...", "THYAO", "calendar_pattern", {"pattern": "weekday_sequence", "pattern_kw": {"down_days": ["MON","TUE","WED"], "up_days": ["THU","FRI"], "oos_start": "2026-01-01"}, "start": "2024-01-01", "end": "2026-09-10"})`
  - Q3 `("...", "ASELS", "technical_reversal", {"reversal_kw": {"event_types": ["bollinger_lower_touch"], "condition_sets": [["rsi_trough","high_volume"],["supertrend_bearish","below_ichimoku_cloud"]], "cooldown_days": 0}, "start": "2024-01-01", "end": "2026-09-10"})`
  - Q4 `("...", "TUPRS", "fundamental_reaction", {"start": "2024-01-01", "end": "2026-09-10"})`
  - Q5 `("...", "EREGL", "general", {"macro_factor": "usdtry", "start": "2024-01-01", "end": "2026-09-10"})`

- [ ] **Step 1: `tests/test_sample_questions.py`** — her soru için beklenen araçları **doğrudan** çağır (as_of 2026-09-10), sonra:
  - tüm zarflar `status != "ERROR"`;
  - zarf JSON'undaki her `YYYY-MM-DD` dizgisi ≤ `"2026-09-10"` (regex ile tara; `params` hariç);
  - Q1: `rank_sector_relative_strength(tickers=[BIMAS,MGROS,MAVI], horizon_days=20)` + `test_catch_up_history(ticker=<en geride>, peers=<diğer ikisi>)`, `by_horizon` h5/h10/h20 dolu;
  - Q2: IS ve OOS segmentlerinin `occurrences`, `mean`, `median`, `net_mean_after_cost`, `p_value_bonferroni` alanları dolu;
  - Q3: `count`, `h5.hit_rate`, iki `by_condition_set` dolu;
  - Q4: TUPRS 2026Q2 satırı var ve `returns["20d"]["complete"]` Task 6'daki kuralla tutarlı;
  - Q5: `get_macro_context(..., series=["usdtry_tcmb","policy_rate","cpi_yoy"])` üç seriyi döndürür; `analyze_macro_sensitivity(EREGL, usdtry)` `n_periods > 100`.
- [ ] **Step 2:** `pytest tests/test_sample_questions.py -q` → PASS (önceki görevler tamamsa doğrudan geçmeli; geçmezse ilgili görevin hatası düzeltilir).
- [ ] **Step 3:** `python -m bist.tools.export_schemas`; çıktıdaki token bütçesini (tam/kompakt) ve durum başına en büyük şema boyutunu not et.
- [ ] **Step 4:** `ROADMAP.md`'ye "Faz 2b — şema v1.1" bölümü (19 araç, token bütçesi, örnek soru kabul testi, durum kutuları); `CLAUDE.md`'de "DONDURULDU (v1.0)" → "DONDURULDU (v1.1, 19 araç)", test sayısı, `fetch_evds` yeniden kurulum sırasına (`fetch_macro`'dan sonra) eklenir; memory `bist-sample-questions-gap.md` "v1.1 tamamlandı" olarak güncellenir.
- [ ] **Step 5:** `pytest tests -q` → tamamı PASS; commit `feat: arac semasi v1.1 donduruldu, ornek soru kabul testi`.
