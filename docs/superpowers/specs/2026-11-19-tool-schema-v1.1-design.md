# Araç şeması v1.1 — hocanın örnek sorularını karşılayacak genişletme (tasarım)

Tarih: 2026-11-19 · Durum: taslak (kullanıcı onayı bekliyor) · Önceki: `mcp/tool_schemas.json` v1.0 (16 araç)

## 1. Amaç ve gerekçe
Hoca 5 örnek test sorusu verdi (`docs/sample_test_questions.txt`). Final test aynı tipte, farklı hisse, sektör ve tarihlerle
yapılacak: analiz tarihi verilir (örn. 10.09.2026), veri dönemi verilir (örn. 01.01.2024–10.09.2026), sonrasındaki hiçbir veri
kullanılamaz. Notlandırmada şunlara bakılıyor: doğru araç, **doğru parametre ve doğru tarih aralığı**, uydurma sayı, look-ahead,
kanıta dayalı sonuç, izinli eylem, tekrarlanabilirlik.

v1.0 araçları çoğunlukla "as_of anındaki özet + varsayılan geçmiş penceresi" mantığıyla kurulu. Örnek sorular ise **tarih
aralığı boyunca olay bazlı istatistik** istiyor. Proje kuralı gereği şema dondurulmadan eğitim verisi (Faz 5) üretilmez.
Bu yüzden önce şemayı v1.1'e genişletip yeniden donduruyoruz.

## 2. Hedefler, kapsam dışı, kaçınılacaklar
**Hedefler**
- 5 örnek sorunun her biri, beklenen araçlarla ve sorudaki tarih aralığıyla, uydurma sayı olmadan cevaplanabilsin.
- "Benzer ama farklı" sorular (başka hisse, sektör, tarih, gösterge kombinasyonu) aynı araçlarla cevaplanabilsin.
- Eksik veri (tamamlanmamış ileri getiri penceresi, yeniden ifade edilmiş bilanço, küçük örneklem) araç çıktısında
  makinece okunur bir uyarı olarak dönsün. Model bunu fark etmek zorunda kalmasın, araç söylesin.

**Kapsam dışı (sonraki işler)**
- `llm_agent` değişiklikleri: odak/tarih seçimini modele vermek, konuşma formatı, HTTP 400 hatası. Ayrı bir iş olarak bunun hemen ardından yapılacak.
- Faz 5 eğitim verisi, haber/konuşma veri katmanı.

**Kaçınılacaklar (çalışsa bile başarısızlık sayılır)**
- v1.0 çağrılarının davranışını değiştirmek. Mevcut 81 test ve eski parametreler aynen çalışmalı (yalnız ekleme).
- as_of sonrasına bakan herhangi bir hesap (look-ahead).
- Modelin serbest metinle koşul ya da formül yazdığı bir sorgu dili. Koşullar sabit bir listeden seçilir (yaklaşım A).
- Araç çıktısında kaynağı izlenemeyen sayı.

## 3. Yaklaşımlar (değerlendirilen)
- **A — mevcut araçlara isteğe bağlı parametreler + 3 yeni araç (SEÇİLDİ).** Geriye uyumlu, az sayıda ama esnek araç;
  koşullar sabit listeden seçilir. Bedeli: şema 16 → 19 araç, parametre sayısı artar.
- B — tek genel "istatistik sorgusu" aracı + küçük sorgu dili. En esnek seçenek, ama 4B model için hatalı ya da uydurma parametre riski yüksek. Bu da notlandırmada cezalandırılan bir hata türü. Reddedildi.
- C — her soru tipine dar, özel araç. Örneklerde kusursuz, ama soru biraz değişince yetersiz kalır. Reddedildi.

Kullanıcı 2026-11-19'da A'yı seçti.

## 4. Ortak kurallar (tüm yeni/değişen araçlar)
- Tarih aralığı: `start`, `end` (YYYY-MM-DD). `end` varsayılanı ve üst sınırı as_of'tur. `end > as_of` verilirse as_of'a kırpılır ve
  `END_CLIPPED_TO_AS_OF` uyarısı döner. `start` verilmezse aracın mevcut varsayılanı kullanılır.
- **İleri getiri penceresi:** t+h tarihi as_of'u aşıyorsa o gözlem istatistiğe girmez. Sayısı her ufuk için
  `n_excluded_incomplete` alanında raporlanır ve `INCOMPLETE_FORWARD_WINDOW` uyarısı döner. Hocanın 3. ve 4. sorusu bunu açıkça istiyor.
- Getiriler kapanıştan kapanışa ölçülür. İşlem edilebilir getiri gerektiğinde maliyet parametresi kullanılır: `cost_bps_round_trip`,
  varsayılan 30 bps (S1 ve backtest ile aynı).
- Küçük örneklem: n < 10 ise `SMALL_SAMPLE` uyarısı.
- Zarf sözleşmesi aynı kalır (`envelope.py`); `SCHEMA_VERSION = "1.1"`.

## 5. Değişiklikler, araç araç

### 5.1 `rank_sector_relative_strength` (Soru 1)
- Yeni isteğe bağlı parametre: `tickers: list[str]`. Verilirse peer grup bu liste olur ve `sector` yok sayılır (veri alanında
  `custom_group: true`). En az 3 geçerli hisse gerekir; değilse mevcut `PEER_GROUP_TOO_SMALL` davranışı.
- Çıktıda her hisse için `lag_gap` = hissenin getirisi − grup medyanı (mevcut `excess_vs_sector_median`, yeni adla da raporlanır).

### 5.2 YENİ `test_catch_up_history` (Soru 1)
Benzer gerilemeler geçmişte kapandı mı?
- Parametreler: `ticker`, `peers: list[str] | None` (yoksa ticker'ın peer grubu), `lookback_days` (geriye bakış L, varsayılan 20),
  `horizons` (varsayılan [5,10,20]), `min_lag_gap` (varsayılan: ticker'ın son L-günlük lag gap'i), `start`, `end`, `as_of`.
- Olay: t gününde ticker'ın L-günlük getirisi − peer medyanı ≤ −|min_lag_gap|. Örtüşmeyi önlemek için olaylar arasında en az L gün bırakılır (S1 ile aynı).
- Her ufuk h için:
  - `n` (yalnız tamamlanmış pencereler)
  - `narrowed_rate`: sonraki h günde ticker − medyan > 0
  - `closed_rate`: sonraki h gündeki göreli getiri ≥ |gap|
  - `false_positive_rate` = 1 − narrowed_rate
  - ortalama/medyan göreli getiri, XU100'e göre fazla getiri
  - `n_excluded_incomplete`
- Grup gücü ayrı raporlanır (`sector_strong_at_event` alt kırılımı; S1'deki tanım: medyan > XU100 ve > 0).
- Uyarılar: `SMALL_SAMPLE`, `INCOMPLETE_FORWARD_WINDOW`, `PEER_GROUP_TOO_SMALL`, `CUSTOM_PEER_GROUP`.
- Yerleştiği durum: `RUN_SECTOR_RELATIVE_ANALYSIS`.

### 5.3 `test_weekday_pattern` (Soru 2)
- Yeni `pattern = "weekday_sequence"`. Yeni parametreler: `down_days: list[weekday]` (örn. [MON,TUE,WED]) ve `up_days: list[weekday]` (örn. [THU,FRI]).
  - Koşul haftası: `down_days`'in hepsi düşüş kapanışı olan takvim haftası (eksik gün varsa o hafta dışarıda kalır).
  - Sonuç: o haftada `up_days`'in kümülatif getirisi.
  - Rapor: koşul haftası sayısı, ortalama ve medyan sonuç getirisi, `up_days`'in yükseldiği haftaların oranı, tam desen sayısı
    (hem düşüş hem yükseliş gerçekleşen haftalar), koşulsuz `up_days` getirisiyle kıyas (tüm haftalar), maliyet sonrası
    net ortalama (`cost_bps_round_trip`), Welch testi ve Bonferroni.
- Yeni isteğe bağlı `oos_start`: verilirse in-sample = [start, oos_start), out-of-sample = [oos_start, end]. Mevcut %70/%30 bölmesi
  yerine kullanılır ve iki segment için aynı tam istatistik seti döner. Model bunun yerine aracı iki ayrı tarih aralığıyla da çağırabilir; ikisi de geçerlidir.
- Mevcut XU100 referansı ve `MARKET_WIDE_EFFECT` mantığı yeni desen için de çalışır.

### 5.4 `detect_reversal_events` (Soru 3)
- Yeni olay türleri: `bollinger_lower_touch` (low ≤ alt bant) ve simetriği `bollinger_upper_touch` (high ≥ üst bant).
- Yeni parametreler: `start`, `end` (istatistik penceresi; verilirse `lookback_days` yerine geçer).
- Yeni `condition_sets: list[list[condition]]`. Her küme ayrı bir alt grup olarak raporlanır ve olay günü
  değerlendirilir (yalnız ≤ t verisi). Sabit koşul listesi:
  `rsi_oversold` (RSI<30), `rsi_trough` (RSI t'de yerel dip: RSI[t] < RSI[t−1] ve RSI[t] < 35; tamamlanmış dip t+1 verisi gerektireceği için kullanılmaz),
  `high_volume` (hacim > 20 günlük ortalamanın 1,5 katı; hacmi eksik günler bu koşulu sağlamaz), `supertrend_bearish`, `supertrend_bullish`,
  `below_ichimoku_cloud`, `above_ichimoku_cloud`, `macd_hist_negative`, `macd_hist_positive`, `below_kama`, `above_kama`.
- Çıktı: her olay türü × (tüm olaylar + her koşul kümesi) için sayı, 1/3/5/10 günlük ortalama/medyan getiri, hit rate,
  XU100'e göre fazla getiri, `n_excluded_incomplete`. Ayrıca `events` listesi (tarih, koşul bayrakları, ileri getiriler; tamamlanmamışlar null).
- Uyarılar: `SMALL_SAMPLE`, `INCOMPLETE_FORWARD_WINDOW`, `VOLUME_MISSING_DAYS` (koşulda hacim varsa).

### 5.5 YENİ `analyze_disclosure_reactions` (Soru 4)
- Parametreler: `ticker`, `start`, `end` (KAP **yayın** tarihine göre filtre), `metrics` (varsayılan: sanayi için
  [ebitda_margin_q, net_debt_ebitda]; banka/sigorta için kendi seti; geçersiz metrikte `UNKNOWN_METRIC` hatası),
  `horizons` (varsayılan [1,5,20]), `peers: list[str] | None` (sektör kıyası için; yoksa peer grubu), `as_of`.
- Her açıklama için bir satır:
  - dönem (YYYYQn), ilk KAP yayın zaman damgası, `effective_trade_date` (S4 ile aynı PIT kuralı)
  - metrik değerleri, bir önceki çeyreğe göre değişim (TMS29 rejim sınırı geçilmez), `improving: bool|null`
  - her ufuk için hisse getirisi, XU100'e göre anormal getiri, sektör medyanına göre anormal getiri (peer < 2 ise null), `complete: bool`
- Özet: iyileşen ve kötüleşen çeyreklerin ortalama anormal getirisi (yalnız tamamlanmış pencereler), n'ler, fark.
- Sabit not: `association_only: "no causal claim"`.
- Uyarılar: `INCOMPLETE_FORWARD_WINDOW` (hangi dönem ve ufuk olduğu belirtilir), `TMS29_RESTATED_VALUES` (FY2022 sonrası İş Yatırım değerleri
  ilk yayın değeri değildir; bkz. `docs/data_layer_audit.md`), `SUSPECT_LAG`, `SMALL_SAMPLE`.
- Yerleştiği durum: `LOAD_FUNDAMENTALS`.

### 5.6 `get_macro_context` + TCMB verisi (Soru 5)
- Yeni veri: `src/bist/fetch_evds.py` → `data/processed/macro_tcmb.parquet` (uzun format: date, series, value, source,
  `publish_date`). Seriler: TCMB politika faizi (tercihen bir haftalık repo; ulaşılamazsa `TP.APIFON4` ağırlıklı ortalama fonlama maliyeti),
  TCMB USD/TRY alış (`TP.DK.USD.A.YTL`) ve TÜFE yıllık değişim. Uç nokta: `https://evds3.tcmb.gov.tr/igmevdsms-dis/`, anahtar
  `.env`'deki `EVDS_API_KEY`, header `key`. TÜFE için **yayın tarihi** kuralı: ay verisi, TÜİK'in açıkladığı güne kadar görünmez
  (takvim yoksa ayın 3. iş günü varsayımı açıkça belgelenir).
- `get_macro_context` yeni parametreler: `start`, `end`, `series: list[str] | None`. Aralık verilirse her seri için başlangıç/bitiş değeri,
  değişim, min/max ve gözlem sayısı döner. Mevcut D-1 kuralı korunur; TCMB serileri artık `missing_sources`'ta değil.
- `store.py`'ye TCMB serilerini okuma eklenir.

### 5.7 YENİ `analyze_macro_sensitivity` (Soru 5)
- Parametreler: `ticker`, `factor` (usdtry, eurtry, policy_rate, cpi_yoy, brent, gold, vix, dxy, sp500, us10y), `start`, `end`,
  `frequency` (weekly | monthly; varsayılan weekly), `as_of`.
- Hesap: dönem getirileri, faktörün aynı dönemdeki değişimiyle (D-1 kuralıyla hizalı) eşleştirilir ve şunlar raporlanır:
  - korelasyon, beta (OLS) ve t-değeri
  - faktör yükselirken / düşerken hissenin ortalama getirisi ve XU100'e göre fazlası
  - yıl bazında korelasyon işaret kararlılığı
  - n
- Ayrıca XU100'ün aynı faktöre duyarlılığı raporlanır ("hisseye özgü mü, piyasa geneli mi").
- Sabit not: `association_only: "correlation is not causation; no forecast"`. Araç tahmin ya da fiyat hedefi üretmez.
- Uyarılar: `SMALL_SAMPLE`, `UNSTABLE_RELATIONSHIP` (yıllar arasında işaret değişiyorsa), `STALE_FACTOR` (ör. aylık TÜFE).
- Yerleştiği durum: `LOAD_MACRO_CONTEXT`.

## 6. Harness entegrasyonu
- `states.ALLOWED_TOOLS`: yeni araçlar 5.2, 5.5 ve 5.7'deki durumlara eklenir.
- `evidence.EXTRACTORS`: 3 yeni araç ve değişen araçların yeni alanları için deterministik kanıt cümleleri. Sayılar araç çıktısından
  formatlanır, uyarılar kanıt olarak geçer ("20 günlük pencere tamamlanmadı: 2026Q2").
- `policy`: `POTENTIAL_CATCH_UP_CANDIDATE` için `test_catch_up_history` desteği şart olur (ör. 10 günde narrowed_rate > %50, n ≥ 10,
  yanlış pozitif açıkça raporlanmış). Eşik değerler uygulayıcının kararı; testle sabitlenir.
- `reference.plan`: odaklara göre yeni araç çağrıları eklenir. Örnek: sector_catchup → `test_catch_up_history`;
  fundamental_reaction → `analyze_disclosure_reactions`; makro odaklı sorular → `analyze_macro_sensitivity`. Referans ajan
  tarih aralığı parametrelerini de alabilmeli (Faz 5 bunları kullanacak).
- `check_data_quality`: yeni araçların kullandığı tarihler `evidence_dates` üzerinden sızıntı kontrolüne girer.

## 7. Şemayı yeniden dondurma
- `export_schemas.py` ile `mcp/tool_schemas.json` v1.1 üretilir. `tests/test_schema_freeze.py` v1.1'e güncellenir.
- Token bütçesi raporlanır (v1.0: tam ~4.9k / kompakt ~1.9k). Per-state araç listesi sayesinde modelin bir anda gördüğü şema küçük kalmalı.
- ROADMAP'e "Faz 2b — şema v1.1" bölümü eklenir; CLAUDE.md'deki "DONDURULDU (v1.0)" ifadesi güncellenir.

## 8. Kabul ölçütleri
- Mevcut 81 test yeşil.
- Her yeni ya da değişen araç için birim testleri:
  - nedensellik: as_of sonrası veri kesilince sonuç değişmez
  - tamamlanmamış pencere hariç tutuluyor ve sayılıyor
  - geçersiz parametre ERROR zarfı döndürüyor
  - sentetik seride bilinen cevap
- **Örnek soru kabul betiği** (`tests/test_sample_questions.py`): 5 sorunun her biri, analiz tarihi 2026-09-10 ve sorudaki tarih
  aralıklarıyla, beklenen araçlar doğrudan çağrılarak çalıştırılır. Beklenen alanların dolu olduğu ve hiçbir çıktı tarihinin 2026-09-10'u aşmadığı doğrulanır.
  Soru 4'te TUPRS 2026Q2 için 20 günlük pencerenin `complete=false` olduğu ya da gerekçesiyle tamamlandığı doğrulanır.
  Soru 5'te USD/TRY ve TCMB serilerinin döndüğü doğrulanır.
- Referans ajan, 5 sorunun her biri için doğrulayıcıdan geçen bir sonuç üretir.

## 9. Uygulayıcıya bırakılanlar
- İç modül yerleşimi (ör. `catchup.py`, `disclosures.py`, `macro_sensitivity.py`) ve ortak yardımcılar (S1/S4 kodunun araç içine taşınması).
- EVDS'deki politika faizi ve TÜFE seri kodlarının kesinleştirilmesi (API'de doğrulanmış: `TP.DK.USD.A.YTL`, `TP.APIFON4`).
- Kanıt cümlelerinin tam metni ve policy eşikleri (testle sabitlenmek şartıyla).

## 10. Uygulayıcı kontrolünde verilen kararlar (Claude'un kararı, kullanıcı inceleyebilir)
- `rsi_trough` "tamamlanmış dip" yerine "t gününde düşmeye devam eden ve 35'in altındaki RSI" olarak tanımlandı. Gerçek dip t+1'i görmeyi gerektirir ve bu look-ahead olurdu.
- `weekday_sequence` takvim haftası bazında tanımlandı. Tatil içeren haftalarda eksik gün olursa o hafta sayılmaz.
- `test_catch_up_history`'de "kapandı" iki ayrı ölçüyle raporlanıyor: daraldı (göreli getiri > 0) ve tamamen kapandı (≥ |gap|).
  Hocanın sorusu hangisini kastettiğini söylemiyor; ikisi birlikte verilirse model doğru olanı alıntılayabilir.
- `high_volume` eşiği 20 günlük ortalamanın 1,5 katı (yaygın varsayılan); parametre olarak açılmadı, araç sayısını değil karmaşıklığı artırırdı.
- TÜFE için yayın tarihi kuralı uygulanır. TÜİK takvimi yoksa "ayın 3. iş günü" varsayımı kullanılır ve uyarıyla belgelenir.
- `analyze_disclosure_reactions`'a `peers` eklendi: TUPRS'un grubu (Energy_Utilities) elektrik şirketleri içeriyor, sektör kıyası
  bu yüzden kaba kalabilir. Model ya da kullanıcı daha uygun bir grup verebilir. Varsayılan peer grupları değiştirilmedi;
  hocanın sorusunda olduğu gibi listeyi açıkça veren sorular `tickers`/`peers` ile karşılanıyor.
