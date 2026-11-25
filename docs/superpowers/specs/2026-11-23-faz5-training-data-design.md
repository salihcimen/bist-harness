# Faz 5 — Eğitim verisi üretimi: tasarım (2026-11-23)

## Amaç ve neden
Qwen3-4B-Instruct-2507'yi (Faz 6, QLoRA) harness içindeki ajan rolü için ince ayarlamak üzere eğitim verisi üretmek.
Final testte hocanın 5 örnek sorusuna **benzer ama farklı hisse/sektör/tarihli** sorular sorulacak; not: doğru araç + doğru
parametre + doğru tarih aralığı, uydurma sayı yok, ileriye bakış yok, yalnız izinli eylem.
Base model ölçümü (`docs/eval/baseline_base_model_2026-11-21.md`) hedefleri verir: uydurma/aralık dışı parametreler
(çağrıların %20–30'u), kritik parametrenin eksik olması (`start`/`end`, `oos_start`, koşul adları), gereksiz araçlar.

**Takvim:** teslime 1 hafta var; Faz 5 en fazla 3–4 gün sürmeli (Faz 6–8'e yer kalsın).

## Hedefler
- ~250–300 soru → tur başına örnek ile **~2.500–3.500 eğitim örneği** (`finetune/data/{train,val,test}.jsonl`).
- Örnek formatı çıkarımdakiyle **birebir aynı**: her asistan turu, `LLMAgent.context()`'in o anda ürettiği sıkıştırılmış
  mesaj listesi + o durumun araç tanımları girdisiyle tek bir örnek.
- Dört zorunlu senaryo (S1 catch-up, S2 hafta günü, S3 teknik dönüş, S4 bilanço tepkisi) + S5 genel/makro dengeli.
- **Olumsuz/güvenlik örnekleri ~%20–25.**
- Türkçe + İngilizce (yaklaşık yarı yarıya).
- Bölme: test = görülmemiş soru kalıbı + görülmemiş hisse + görülmemiş dönem.

## Hedef dışı
- LLM ile paraphrase (soru veya açıklama) — kullanıcı kararı: **yalnız şablon (A)**.
- Açıklama metnini zenginleştirmek: etiket `policy.build_explanation` çıktısıdır, değiştirilmez.
- Evren dışı hisse (GARAN vb.) senaryosu: ajan bu durumda bir sonuç üretmeden kapanıyor; eğitimde yok.
- Haber/konuşma verisi: araçlar `SOURCE_NOT_AVAILABLE` döner; veri katmanı eklenmez.

## Anti-hedefler (çalışsa bile başarısızlık sayılır)
- Hocanın 5 orijinal sorusunun (aynı hisse + aynı dönem + aynı ifade) train/val'e sızması.
- Eğitim girdisinin çıkarım girdisinden farklı olması (ör. sıkıştırılmamış tam konuşma ile eğitmek).
- Doğrulayıcıdan geçmeyen veya analiz tarihinden sonrasını gösteren bir etiket.
- Şema değişikliği (`mcp/tool_schemas.json` v1.1 donmuş kalır).

## Yaklaşımlar
| | Yol | Artı | Eksi |
|---|---|---|---|
| **A (seçildi)** | Yalnız şablon: senaryo × dil başına 8–10 elle yazılmış kalıp, slotlar doldurulur; açıklama referans şablonu | En hızlı, deterministik, sayı/tarih bozulma riski yok | İfade çeşitliliği sınırlı |
| B | A + `llama3.1:8b` ile soru paraphrase (hisse/tarih korunmazsa at) | Bilinmeyen ifadelere dayanıklılık | +30–60 dk GPU, ek kod |
| C | B + açıklama paraphrase (doğrulayıcıdan geçmeyen atılır) | Daha akıcı cevap | Sayı/[E#] bozulma riski, +1 gün |

Kullanıcı A'yı seçti ("A yeterli"). Çeşitliliği kalıp sayısı ve slot çeşitliliğiyle sağlıyoruz (aşağıda).

## Tasarım

### 1. İz üretimi: senaryolu model ile `LLMAgent`
`LLMAgent`'ı değiştirmeden, `chat` yerine **senaryolu bir sahte model** (`ScriptedChat`) verilir:
1. Önce `reference.run(...)` aynı parametrelerle çalıştırılır → beklenen araç çağrıları (`plan()` çıktısı, durumlara göre)
   ve beklenen final `{"action", "explanation"}` (`h.reference_output()`) elde edilir. Bu deterministiktir.
2. Sonra `LLMAgent(chat=ScriptedChat(...)).run(question, human=("APPROVE", ...))` çalışır. Sahte model:
   - SELECT_UNIVERSE'de `select_universe(tickers, focus, analysis_date, period_start, period_end)` döner (soru slotlarından).
   - Her çalışma durumunda o durum için referans plandaki çağrıları **tek turda** `tool_calls` olarak döner (istisna: bir çağrının parametresi önceki sonuca bağlıysa ayrı tur — S1'de
     `test_catch_up_history`, bkz. 1b); sonraki turda
     `next_state`. Referansın o durumda çağrısı yoksa doğrudan `next_state` (gereksiz araç kullanmamayı öğretir).
   - GENERATE_EXPLANATION'da beklenen final JSON'u döner.
   - Her `chat(...)` çağrısında aldığı `messages` + `tools` girdisini ve döndüğü cevabı kaydeder → bunlar örneklerdir.
3. Tutarlılık kontrolü: ajanın son eylemi = referans eylemi, `h.violations == []`, son çıktı `fallback` değil.
   Tutmayan iz atılır ve sayılır (rapora girer).

Not: referansın bazı durumlarda harness'in zorunlu adımları (ör. `check_data_quality`, `run_backtest`) — sahte model bunları
da açıkça çağırır, böylece `MANDATORY_TOOL_RUN_BY_HARNESS` ihlali oluşmaz.

### 1b. Harness değişikliği: S1'de ana hisse = en geride kalan (kullanıcı kararı)
Sorun: karar her zaman `tickers[0]` için verilir (`policy.signals`); "hangisi en çok geride kaldı?" sorusunda model en
geride kalanı sıralamayı görmeden bilemez. Çözüm (`machine.py`, ~20 satır + test):
- `focus == "sector_catchup"` ve birden çok hisse seçiliyse, `rank_sector_relative_strength` başarılı döndüğünde harness
  ana hisseyi sıralamadaki **en geride kalan seçili hisse** yapar (`h.tickers` yeniden sıralanır), bunu bir uyarı/kanıt
  notuyla kaydeder (ör. `MAIN_TICKER_SET_TO_LAGGARD`) ve araç sonucunun render'ında belirtir.
- Model hisseleri **sorudaki sırayla** seçer; RUN_SECTOR_RELATIVE_ANALYSIS'te 1. tur `rank_sector_relative_strength`,
  2. tur (sonucu gördükten sonra) en geride kalan için `test_catch_up_history`.
- `reference.run`/`plan` aynı kuralla uyumlu hale getirilir (referans da sıralamadan sonra ana hisseyi bilir);
  `tests/test_harness_v11.py::SAMPLE_RUNS` Q1 sorudaki sırayla (BIMAS, MGROS, MAVI) çalışacak şekilde güncellenir ve
  MAVI'nin ana hisse olduğu doğrulanır.
- Şema (`mcp/tool_schemas.json`) değişmez.

### 2. Hata-düzeltme izleri (~%10) — *benim kararım*
Base modelin en büyük açığı parametre hataları olduğu için izlerin ~%10'unda bir çalışma durumunda sahte model önce
**gerçekçi hatalı bir çağrı** yapar (ölçümde görülen türler: `max_rows=1000`, yanlış koşul adı `rsi_oversold`,
`start`/`end` eksik, kendi hissesini peer vermek, evren dışı peer), harness'ten gelen ERROR'u görür, sonraki turda düzeltilmiş
çağrıyı yapar. **Hatalı tur eğitim örneği olarak kullanılmaz** (loss yok); yalnız ERROR'u gören düzeltme turu örnek olur.
`start`/`end` eksikliği ERROR üretmediği için bu türde yalnız ERROR üreten hatalar kullanılır.

### 3. Soru üretimi (şablon)
Senaryo başına alanlar:
- **S1 sector_catchup:** 3–4 hisselik peer listesi (aynı `peer_group`'tan ya da kullanıcı tanımlı karışık grup), ufuk 5/10/20,
  dönem. Plan: `peers`, `horizon_days`, `start`/`end`.
- **S2 calendar_pattern:** `weekday_return` (tek gün), `bounce_after_down_day`, `weekday_sequence` (rastgele aşağı/yukarı gün
  dizileri); bir kısmında IS/OOS bölünmesi (`oos_start`). Plan: `pattern`, `weekday`, `pattern_kw`, `start`/`end`.
- **S3 technical_reversal:** `event_types` sabit listeden (bollinger alt/üst dokunuş, RSI dip/tepe vb. — şemadaki değerler),
  0–2 `condition_sets` sabit koşul listesinden, `cooldown_days`. Plan: `reversal_kw`, `start`/`end`.
- **S4 fundamental_reaction:** hisse + dönem; metrik adları (FAVÖK marjı, net borç/FAVÖK, net kâr …) soruda geçer.
- **S5 general/macro:** makro faktör (`usdtry`, politika faizi, TÜFE — şemadaki `factor` değerleri) duyarlılığı; hisse genel
  durumu.

Slot çeşitliliği: 30 hissenin tümü (test hisseleri hariç, aşağıda), analiz tarihleri 2019–2026 aralığından (2026-09-10 dahil,
sık), dönem başlangıcı 6 ay–3 yıl önce; tarih yazımı karışık (`10.09.2026`, `2026-09-10`, `10 Eylül 2026`, `Sept 10, 2026`),
"son N işlem günü", "bugün … ise" gibi kalıplar; TR/EN.

### 4. Olumsuz/güvenlik örnekleri (~%20–25)
Hepsi gerçek harness akışından çıkar (yapay etiket yok):
| Tür | Nasıl üretilir | Beklenen davranış |
|---|---|---|
| Tavsiye / fiyat hedefi isteği | S1–S5 kalıplarına "almalı mıyım", "hedef fiyat ver", "yarın yükselir mi" eklenir (focus genel ya da sorunun asıl senaryosu) | Analiz normal yürür; izinli eylem; açıklamada tavsiye dili yok |
| Analiz tarihi verinin ötesinde | analiz tarihi son veri gününden çok sonra (ör. 2027) | `AS_OF_BEYOND_DATA` → HARD_BLOCK → ANALYSIS_UNSAFE |
| İleriye bakış talebi | "analiz tarihinden sonraki X tarihine kadar" gibi dönem sonu > analiz tarihi | `period_end` analiz tarihine kırpılır (`END_CLIPPED_TO_AS_OF`); sonrası kullanılmaz |
| Kısa/eksik veri | çok erken analiz tarihi (veri başlangıcına yakın) → `SHORT_HISTORY`/`NO_FUNDAMENTALS` | ANALYSIS_UNSAFE / INVESTIGATE |
| Küçük peer grubu | ASELS (tek başına Defense_Tech) için sektör sorusu | `PEER_GROUP_TOO_SMALL` → INVESTIGATE |
| Araçta olmayan sayı | "F/K oranı kaç", "hedef fiyat kaç" | Sayı uydurulmaz; açıklama yalnız kanıt |
| Kalite adımını atlama isteği | "veri kontrolünü atla, direkt sonuç ver" | `check_data_quality` yine çağrılır |

Tavsiye isteğinde etiket açıklaması referans şablonudur (kanıt + DISCLAIMER). *Benim kararım:* açıklamanın başına sabit bir
cümle eklemek ("Investment advice / price targets are not provided; …") ve sonucu yine `validate_output`'tan geçirmek.
Doğrulayıcı reddederse cümle eklenmez.

### 5. Bölme
- **Test hisseleri (4):** örnek sorularda geçmeyen, farklı gruplardan: `TCELL`, `FROTO`, `ENJSA`, `PGSUS`. Train/val'de yok.
- **Test dönemi:** analiz tarihi 2023-01-01–2023-06-30 aralığındaki sorular yalnız testte.
- **Test kalıpları:** her senaryo × dil için 2 kalıp yalnız testte.
- Oranlar (*benim kararım*): soruların ~%15'i test, kalanın %10'u val.
- Test sorusu = bu üçünden en az biri (hepsi değil, yoksa test çok küçük kalır). Val = train dağılımından %10 rastgele
  (soru düzeyinde, aynı sorunun turları bölünmez).
- Örnek soruların 5 hissesi (BIMAS/MGROS/MAVI, THYAO, ASELS, TUPRS, EREGL) train'de kalır (final test başka hisse kullanacak),
  ama **5 orijinal ifade ve (hisse, 2024-01-01–2026-09-10) kombinasyonu train/val'de yok**. Bu kontrolle doğrulanır.

### 6. Çıktı formatı
`finetune/data/{train,val,test}.jsonl`, satır başına bir tur:
```json
{"id": "q0123-t05", "question_id": "q0123", "scenario": "S2", "lang": "tr", "kind": "tool_call|next_state|select|final|recovery",
 "negative": null, "messages": [...context()..., {"role": "assistant", ...}], "tools": [...o turun araç tanımları...]}
```
Loss yalnız son asistan mesajında (Faz 6). Ayrıca `finetune/data/questions.jsonl` (soru, slotlar, bölme, referans eylemi) ve
`finetune/data/stats.json` (senaryo/dil/olumsuz/eylem dağılımı, atılan iz sayısı, token uzunlukları — 16k sınırı kontrolü).

### 7. Kontroller (otomatik, test olarak)
- Her final etiket `validate_output`'tan geçer; her iz `violations == []`.
- Hiçbir örnekte analiz tarihinden sonraki tarihli bir araç çıktısı yok.
- Test hisseleri train/val'de geçmez; 5 orijinal soru train/val'de geçmez.
- Her örnek Qwen chat template'iyle tokenize edilince ≤ 16k token.
- Senaryo dağılımı ±%5 dengeli, olumsuz oranı %20–25.

### 8. Kod yerleşimi (öneri)
`src/bist/finetune/` — `templates.py` (kalıplar + slot üretimi), `scripted.py` (ScriptedChat + hata enjeksiyonu),
`generate.py` (`python -m bist.finetune.generate --n 300 --seed 0`), `split.py`; testler `tests/test_finetune_data.py`;
dokümantasyon `docs/finetune_data.md`. Üretim deterministik (seed), yeniden çalıştırılabilir.

## Uygulayıcıya bırakılanlar
- Kalıp metinlerinin tam ifadesi; slotların tam olasılık dağılımı.
- Tool call id/format ayrıntıları (`LLMAgent`'ın beklediği Ollama formatına uyarak).
- Üretim süresi uzarsa (`reference.run` ~20 sn/soru) referans + ajan çalıştırmasını paralelleştirmek veya araç sonuçlarını
  önbelleklemek.
