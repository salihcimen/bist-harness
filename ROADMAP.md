# CSE-481 BIST 100 Agentic Harness + Fine-tuned Yerel LLM — ROADMAP

Durum anahtarı: [ ] başlamadı · [~] sürüyor · [x] bitti

## 0. Sabit kısıtlar
- GPU: RTX 4060 Laptop 8 GB VRAM → QLoRA, 3–4B model, kısa sekans (≤1536–2048 token)
- Disk: C: üzerinde ~127 GB boş (proje C:\Projects\econist-harness; HF cache `.cache/hf`)
- Ollama kurulu: `qwen3:4b-instruct-2507-q4_K_M` (base model = fine-tune'un hazır hâli) + `llama3.1:8b` (büyük hazır referans)
- Python 3.11.9 venv; torch 2.14.1+cu126, transformers, peft, trl, bitsandbytes kurulu
- Final sistem: bulut API yok, internet yok, tamamen yerel
- Notlandırma: canlı, önceden verilmeyen sorular. Metrikler: sayısal halüsinasyon, hatalı araç çağrısı,
  kanıtsız sonuç, kanıt bütünlüğü/tekrarlanabilirlik, kural ihlali (tavsiye, kalite/backtest atlama)

## Temel mimari kararı
Önce **deterministik araçlar + harness**, sonra **eğitim verisi**, en son **fine-tune**.
Sebep: eğitim verisi gerçek araç çıktılarından üretilecek; araç şemaları (isim, parametre, çıktı JSON)
Faz 2'de DONDURULMADAN veri üretimine geçilmez.

## Faz 0 — Kurulum ve zemin (≈ 0.5 gün)
- [x] Proje D:/bist-harness'a taşındı (131 GB boş)
- [x] Proje iskeleti: `src/ data/{raw,processed,db} mcp/ harness/ finetune/ eval/ reports/ logs/`
- [x] venv + requirements (torch 2.14+cu126, transformers, peft, trl, bitsandbytes, datasets, mcp, pyarrow, scipy, statsmodels)
- [~] GPU testi geçti (CUDA, matmul OK); QLoRA smoke test için model indirilmedi (Faz 6 öncesi)
- [x] 30'luk evren + sektör haritası dondur (`universe.csv`); PDF'teki banka örneği evren dışı (GARAN vb.) → sadece örnek
- [ ] Hocaya sorulacaklar: Fintables erişimi, haber kaynağı, onaylı video listesi, teslim tarihi
- **Çıktı:** çalışan ortam, `universe.csv`, GPU testi geçti

## Faz 1 — Veri katmanı, tamamen çevrimdışı önbellek (≈ 2–3 gün)
- [x] OHLCV (yfinance) 30 hisse + XU100 → Parquet (2015→). `clean_prices.py`: CCOLA split düzeltmesi, yarım bar atma, hacim bayrağı. AÇIK: SASA'da 202 sıfır-hacim günü, ECILC 2020-03-12 +%31.9 doğrulanmadı
- [x] Temel veri (Fintables YOK → KAP + İş Yatırım + resmi API): yayın zamanları, rakamlar, PIT takvim, çeyreklik oranlar (`fundamental_metrics.py`), resmi API ile doğrulama (`validate_fundamentals.py`). Denetim: docs/data_layer_audit.md. KRİTİK: İş Yatırım değerleri TMS29 döneminde (FY2022→) ilk yayın değeri değil → nominal dönem ana analiz, TMS29 ayrı rejim
- [x] Makro (piyasa tabanlı, anahtarsız): USD/TRY, EUR/TRY, DXY, VIX, ABD 10Y, Brent, altın, S&P500 (Yahoo) + Fed etkin faiz/hedef aralığı (NY Fed API; FRED bu ağdan erişilemiyor) → `data/processed/macro_market_daily.parquet` (`bist.fetch_macro`, 11 seri, 2015→). Kural: özellikler D-1 veya öncesi değerle hesaplanır (global kapanışlar BIST 18:00'den sonra)
- [x] Makro (anahtar gerekli; Faz 2b'de yapıldı): TCMB EVDS politika faizi + TÜFE (kullanıcı `.env`'e `EVDS_API_KEY` ekleyecek), TÜİK yayın takvimi; yayın zaman damgalarıyla
- [ ] Haber/konuşma: küçük, yasal, zaman damgalı örnek set (opsiyonel katman, en sona)
- [x] `check_data_quality` temeli (bkz. Faz 2)
- **Çıktı:** `data/processed/*.parquet` + veri sözlüğü taslağı. Sonrasında internet gerekmez.

## Faz 2 — Deterministik araçlar (≈ 4–5 gün) ← ŞEMA DONDURMA NOKTASI
Saf Python fonksiyonları, birim testli, tekrarlanabilir:
- [x] `calculate_indicators` (SMA/EMA/KAMA/RSI/MACD/Bollinger/ATR/Supertrend/Ichimoku) + `find_support_resistance` — `src/bist/tools/` (nedensellik testli: t anındaki değer yalnızca ≤t verisine bağlı)
- [x] `detect_reversal_events` + reversal strength (1/3/5/10 gün; yalnızca tamamlanmış ufuklar)
- [x] `rank_sector_relative_strength` (peer grubu <3 → PEER_GROUP_TOO_SMALL)
- [x] `test_weekday_pattern` (Bonferroni + OOS + yıl bazlı kararlılık + XU100 referansı; gürültüde yanlış-pozitif ≈%1 ölçüldü). BULGU: perşembe etkisi 30 hissede ortak = piyasa geneli (XU100 +%0,51), hisseye özgü kanıt değil
- [x] `get_fundamental_metrics`, `calculate_fundamental_change` (point-in-time; TMS29 döneminde büyüme geri tutulur, yalnızca oranlar)
- [x] `get_macro_context` (D-1 kuralı; TCMB/TÜİK eksikliğini açıkça bildirir) · `get_news_events`/`get_speech_features`: şema donduruldu, veri seti yokken SOURCE_NOT_AVAILABLE döner (model eksikliği bildirmeyi öğrenir); veri katmanı opsiyonel, en sona
- [x] `run_backtest` + `generate_strategy_variants` (A–D; E reddedilir: tarihsel haber/konuşma verisi yok). Sinyal t kapanış → giriş t+1 açılış, maliyet+slipaj, train/val/test + yıl kırılımı, az sinyal/OOS kararsızlık bayrakları. İLK BULGU: A–D hiçbiri sinyal başına XU100'e karşı pozitif fazla getiri üretmedi (−%0,02…−%0,22); B/C OOS'ta kararsız
- [x] `check_data_quality` (5 seviyeli karar; sızıntı bayrağı, bayat/eksik veri, geç/yeniden ifade edilmiş rapor, küçük peer grubu)
- [x] `build_evidence_bundle` (+ `session.py`): gözlenen vs karara bağlanan kanıt; cümlelerdeki sayılar araç çıktısından; kalite kapısı yoksa/engelliyse hiçbir şey commit edilmez
- [x] **Araç şemaları donduruldu** → `mcp/tool_schemas.json` (v1.0, 16 araç, tam ~4.9k token / kompakt ~1.9k token). Değişiklik testle yakalanır (`tests/test_schema_freeze.py`); değişirse eğitim verisi geçersiz olur
- **Çıktı:** 16 araç, 57 test yeşil, şema sürüm 1.0 ✅ (Faz 2 tamam)

## Faz 2b — Araç şeması v1.1 (hocanın örnek soruları) ✅ 2026-11-21
Gerekçe: hocanın 5 örnek sorusu (`docs/sample_test_questions.txt`) tarih aralığı boyunca olay bazlı istatistik istiyor. v1.0 bunu karşılamıyordu.
Tasarım: `docs/superpowers/specs/2026-11-19-tool-schema-v1.1-design.md`; plan: `docs/superpowers/plans/2026-11-20-tool-schema-v1.1.md`.
- [x] Ortak kurallar (`tools/common.py`): `start`/`end` (as_of'a kırpılır, `END_CLIPPED_TO_AS_OF`); tamamlanmamış ileri pencere hariç tutulur ve sayılır (`INCOMPLETE_FORWARD_WINDOW`); n<10 → `SMALL_SAMPLE`
- [x] `rank_sector_relative_strength`: `tickers` ile özel grup ve `lag_gap` alanı (S1)
- [x] YENİ `test_catch_up_history`: benzer gerilemeler 5/10/20 günde daraldı mı/kapandı mı, yanlış pozitif oranı, sektör gücüne göre kırılım (S1)
- [x] `test_weekday_pattern`: `weekday_sequence` deseni (örn. Pzt–Çar düşüş → Per–Cum), maliyet sonrası net getiri, `oos_start` ile tam IS/OOS (S2)
- [x] `detect_reversal_events`: `bollinger_lower/upper_touch`, `start`/`end`, sabit listeden `condition_sets`, olay listesi (S3)
- [x] YENİ `analyze_disclosure_reactions`: KAP yayın tarihine göre her çeyrek, metrik değişimi, 1/5/20 gün XU100'e ve sektöre göre anormal getiri, `complete` bayrağı (S4)
- [x] TCMB verisi: `python -m bist.fetch_evds` → `data/processed/macro_tcmb.parquet` (USD/TRY `TP.DK.USD.A.YTL`, politika faizi vekili `TP.APIFON4`, TÜFE yıllık `TP.TUKFIY2025.GENEL` formulas=3; TÜFE yayın tarihi = sonraki ayın 3. iş günü varsayımı). `get_macro_context`: `start`/`end`/`series` (S5)
- [x] YENİ `analyze_macro_sensitivity`: hisse ↔ makro faktör korelasyon/beta, faktör yükselir/düşerken getiri, yıllık kararlılık, XU100 kıyası; tahmin yok (S5)
- [x] Harness: yeni araçlar durumlarına bağlandı; kanıt cümleleri (`catchup_history`, `disclosure`, `macro_sensitivity`, `reversal_condition`, `data_gap`); `POTENTIAL_CATCH_UP_CANDIDATE` için `test_catch_up_history` şartı (10 günde n≥10 ve daralma >%50); `reference.plan` tarih aralığı/peers/desen/koşul/makro faktör alır
- [x] Kabul: `tests/test_sample_questions.py` (5 soru, analiz tarihi 2026-09-10, hiçbir çıktı tarihi sonrasını göstermez) ve `tests/test_harness_v11.py` (referans ajan 5 soruda doğrulayıcıdan geçer)
- **Çıktı:** şema **v1.1, 19 araç** (tam ~7.0k token / kompakt ~2.6k token; model her durumda yalnız o durumun araçlarını görür), 140 test yeşil ✅

## Faz 3 — 4 araştırma senaryosu + raporlar (≈ 3–4 gün, Faz 2 ile paralel yürür)
- [x] S1 Sektör laggard/catch-up (5/10/20 gün, yanlış pozitifler)
- [x] S2 Haftanın günü/çok günlü (OOS, multiple-testing)
- [x] S3 Teknik dönüşler (tek vs birleşik koşul, rejim kırılımı)
- [x] S4 Çeyreklik temel → fiyat tepkisi (anormal getiri vs XU100/sektör)
- [x] Strateji varyantları A–E backtest karşılaştırması
- **Çıktı:** 4 senaryo raporu + backtest tablosu → `reports/` (S1..S4, variants_A_E; `python -m bist.scenarios`). Makro/bağlam raporu: EVDS anahtarı gelince.

## Faz 4 — MCP sunucusu + durum makinesi + kanıt/log (≈ 3–4 gün)
- [x] MCP server (araçlar sarılır) → `bist.harness.mcp_server` (stdio; 16 araç yalnızca `call_tool` ile, durum makinesi üzerinden)
- [x] 17 durumlu harness (START…LOG_AND_CLOSE; spesifikasyondaki tüm durumlar): durum başına izinli araçlar, kalite/kanıt/backtest/gate atlanamaz, as_of sızıntısı engelli → `harness/machine.py`, `states.py`
- [x] Sonuç etiketi doğrulayıcı: 8 eylem + gate/kanıt izinleri + sayı/alıntı/tavsiye dili kontrolü → `harness/policy.py`
- [x] Committed Evidence Set, risk gate, insan onayı (zorunlu), SQLite karar logu (`data/db/decisions.sqlite`), `replay_analysis`, geçmiş vaka hafızası (`recall`)
- [x] LLM istemci katmanı: Ollama tool-calling döngüsü (`harness/llm_agent.py`)
- [x] v2 (2026-11-21): geçerli konuşma formatı (her araç çağrısından sonra tool mesajı), odak/tarih/dönem seçimi modelde, kompakt bağlam (16k), düzeltilen model girdisi hatası kapıyı düşürmez. **Base model ölçümü:** `docs/eval/baseline_base_model_2026-11-21.md` (5/5 doğru seçim ve doğrulanmış cevap; parametre hataları ve içerik zayıf → Faz 5 hedefleri)
- **Çıktı:** base model ile uçtan uca çalışan harness (henüz fine-tune yok = deney A/B/C temeli)

## Faz 5 — Eğitim verisi üretimi ✅ 2026-11-23
Tasarım `docs/superpowers/specs/2026-11-23-faz5-training-data-design.md`, dokümantasyon **`docs/finetune_data.md`**.
Komut: `python -m bist.finetune.generate --n 300 --seed 0` (deterministik, ~12 dk, önbellekli).
- [x] Format: referans izi, senaryolu model ile gerçek `LLMAgent` döngüsünde tekrar oynatılır; her asistan turu = bir örnek
      (girdi = ajanın gönderdiği sıkıştırılmış `context()` + araç tanımları, hedef = tool_call / next_state / final JSON). Qwen/HF mesaj biçimi
- [x] Dağılım: S1–S5 dengeli (54–68 soru); tur başına araç zincirleri; 22 hata-düzeltme izi (hatalı çağrı → ERROR → düzeltme)
- [x] Sınav uyumu (2026-11-23): 104 soru 2024-01-01–2026-09-10 çerçevesinde, 23 iki dönemli S2 (Q2), 13 iddialı tavsiye (Q5), S1'de temel veri (Q1)
- [x] Olumsuz/güvenlik örnekleri %22 (tavsiye, veri ötesi tarih → HARD_BLOCK, ileri dönem kırpma, kısa geçmiş, küçük peer grubu, uydurulamayacak sayı, kontrolü atlama)
- [x] Soru çeşitliliği: TR 149 / EN 151, senaryo×dil başına 12 kalıp, karışık tarih yazımı (yaklaşım A: yalnız şablon, paraphrase yok — kullanıcı kararı)
- [x] Açıklamalar referans şablonu (`policy.build_explanation`), sayılar araç çıktısından; tümü doğrulayıcıdan geçti
- [x] Ayrım: test = TCELL/FROTO/ENJSA/PGSUS veya analiz tarihi 2023 H1 veya yalnız-test kalıbı; 5 orijinal soru train/val'de yok
- [x] Harness: S1'de ana hisse sıralamadan sonra en geride kalana geçer (`harness_notes`), kalite/teknik tüm seçili hisselerde, sinyaller ana hisseye ait çağrılardan; `Harness.history`
- **Çıktı:** `finetune/data/{train,val,test}.jsonl` = 2.754 / 314 / 550 örnek (300 soru: 229/26/45), token maks 5.900

## Faz 6 — İnce ayar (≈ 2–3 gün, deneme dahil)
- **Seçilen model: `Qwen/Qwen3-4B-Instruct-2507`** (karar 2026-11-19, web araştırmasıyla teyitli). Gerekçe: yalnız non-thinking (`<think>` yok),
  BFCL-v3 61.9 (Qwen3-4B: 57.6), çok dilli (MMLU-ProX 61.6), Apache-2.0, 4-bit QLoRA 8 GB'a sığar, Ollama'da resmi etiket
  (`qwen3:4b-instruct-2507-q4_K_M`). HF ağırlıkları `.cache/hf` altında.
- Reddedilen: **Qwen3.5-4B** (daha yeni, ama Unsloth 4-bit QLoRA önermiyor → bf16 LoRA ≈10 GB VRAM > 8 GB; Qwen3.5 GGUF'ları Ollama'da çalışmıyor;
  multimodal + hibrit DeltaNet, transformers v5 gerekir). Yedek: Llama-3.2-3B-Instruct (Türkçe daha zayıf)
- Karşılaştırma (Faz 7): base Qwen3-4B-2507 vs fine-tuned + `llama3.1:8b` hazır büyük referans
- [x] QLoRA (4-bit NF4, r=16, alpha=32, lr≈2e-4, grad checkpointing, batch 1 + grad accum), loss yalnız asistan tokenlarında — `bist.finetune.train`, checkpoint+resume; 372 adım, 6 sa, eval_loss 0.0198→0.0026 (docs/finetune_train.md)
- [-] 2–3 hiperparametre denemesi — ATLANDI (koşu başına ≈6 sa, teslim tarihi); tek koşu, val kaybı monoton düştü
- [x] LoRA merge → GGUF (Q4_K_M) → `ollama create` ile yerel model — `bist.finetune.to_ollama` (llama.cpp convert + llama-quantize; Ollama/Windows safetensors alamıyor)
- **Çıktı:** `bist-analyst` Ollama modeli, eğitim logları ✅ 2026-11-24

## Faz 7 — Değerlendirme ve deneyler (≈ 2–3 gün)
- [ ] Otomatik değerlendirici: sayısal halüsinasyon, hatalı araç çağrısı, kanıtsız sonuç, izinli eylem uyumu,
      kural ihlali, tekrarlanabilirlik (aynı soru → aynı sonuç, temperature 0)
- [ ] **Base vs Fine-tuned** karşılaştırma tablosu (aynı test seti)
- [ ] Harness deneyleri A–E (Raw LLM → +Tools → +State → +Evidence/Gate → +Memory/Human)
- [ ] Hata analizi, bilinen sınırlamalar
- **Çıktı:** metrik tabloları, grafikler, sınırlamalar listesi

## Faz 8 — Rapor ve sunum (≈ 2 gün)
- [ ] Final rapor: temel model + gerekçe, veri hazırlama + örnek sayısı, QLoRA ayarları, yerel çalıştırma,
      base vs FT, sınırlamalar, 4 senaryo bulguları, A–E deneyi
- [ ] Veri sözlüğü, MCP tanımları, replay demosu
- [ ] Canlı demo provası, **internet kapalıyken** test; 20+ bilinmeyen soru ile kuru çalıştırma
- **Çıktı:** teslim paketi

## Toplam: ≈ 3–4 hafta (tek kişi, yoğun). Teslim tarihi netleşince sıkıştırılır.

## Riskler
| Risk | Önlem |
|---|---|
| Disk | Çözüldü: C: ~127 GB boş |
| 8 GB VRAM | 4B model 4-bit + kısa sekans + grad checkpointing; gerekirse Colab/Kaggle ücretsiz GPU yedeği (yalnız eğitim, veri hassas değil) |
| Fintables erişimi | Hocaya sor; yedek yfinance + disclosure tarihi tahmini (açıkça belgelenmiş) |
| Canlı testte bilinmeyen soru | Soru çeşitliliği + held-out test; araç çağrısı şemasını katı tut |
| Veri sızıntısı | Faz 2'de point-in-time testleri; gate HARD_BLOCK |
| Zaman | Haber/konuşma katmanı opsiyonel, en sona |
