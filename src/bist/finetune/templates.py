"""Template questions for the Faz 5 training data (approach A: templates only, no LLM paraphrase).

Every question carries the parameters the deterministic reference agent needs (`plan_kw`), so the label trace is exact.
Scenarios: S1 sector laggard / catch-up, S2 weekday pattern, S3 technical reversal, S4 disclosure reaction, S5 macro / general.
Negatives (~22%): advice request, analysis date beyond the data, period end after the analysis date, too-short history,
too-small peer group, a number no tool provides, a request to skip the quality check.
Hold-out: test questions use a test ticker, an analysis date in TEST_WINDOW or a test-only template (the last 2 of each list);
train/val questions use none of them. The instructor's 5 sample questions (ticker + 2024-01-01..2026-09-10) never occur in train/val.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..tools import store

TEST_TICKERS = ("TCELL", "FROTO", "ENJSA", "PGSUS")
TEST_WINDOW = ("2023-01-01", "2023-06-30")
SAMPLE_AS_OF = "2026-09-10"
LAST_DATA = "2026-11-18"
SAMPLE_TICKERS = {"S1": {"BIMAS", "MGROS", "MAVI"}, "S2": {"THYAO"}, "S3": {"ASELS"}, "S4": {"TUPRS"}, "S5": {"EREGL"}}
FOCUS = {"S1": "sector_catchup", "S2": "calendar_pattern", "S3": "technical_reversal", "S4": "fundamental_reaction", "S5": "general"}
FINANCIAL = {"AKBNK", "TURSG", "ISMEN"}
NEG_COUNTS = {"advice": 22, "beyond_data": 8, "future_period": 8, "short_history": 8, "small_peer_group": 7, "missing_number": 7, "skip_quality": 6}


@dataclass
class Question:
    id: str
    scenario: str
    lang: str
    template_id: str
    text: str
    tickers: list[str]
    focus: str
    as_of: str
    requested_as_of: str
    start: str | None
    end: str | None
    requested_end: str | None
    plan_kw: dict = field(default_factory=dict)
    negative: str | None = None
    claim: bool = False      # advice request built on an unverified "it will certainly rise" claim (sample Q5 type)
    split: str = "train"


# ------------------------------------------------------------------ wording
MONTHS = {"tr": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
DAYS = {"tr": {"MON": "Pazartesi", "TUE": "Salı", "WED": "Çarşamba", "THU": "Perşembe", "FRI": "Cuma"},
        "en": {"MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday", "THU": "Thursday", "FRI": "Friday"}}
SECTOR = {"tr": {"Auto_Durables": "otomotiv ve dayanıklı tüketim", "Chemicals": "kimya", "Construction_Materials": "inşaat ve yapı malzemeleri",
                 "Energy_Utilities": "enerji", "Financials": "finans", "Metals_Mining": "metal ve madencilik", "Retail_Consumer": "perakende",
                 "Telecom": "telekom", "Transport": "ulaştırma", "Healthcare": "sağlık", "Defense_Tech": "savunma", "custom": "seçilmiş"},
          "en": {"Auto_Durables": "auto and durables", "Chemicals": "chemicals", "Construction_Materials": "construction and materials",
                 "Energy_Utilities": "energy", "Financials": "financials", "Metals_Mining": "metals and mining", "Retail_Consumer": "retail",
                 "Telecom": "telecom", "Transport": "transport", "Healthcare": "healthcare", "Defense_Tech": "defense", "custom": "selected"}}
EVENTS = {"tr": {"rsi_oversold_exit": "RSI aşırı satımdan çıkış", "bollinger_lower_reclaim": "alt Bollinger bandının geri alınması",
                 "supertrend_flip_up": "Supertrend'in yukarı dönmesi", "kama_cross_up": "fiyatın KAMA'yı yukarı kesmesi",
                 "macd_cross_up": "MACD'nin sinyal çizgisini yukarı kesmesi", "ichimoku_tk_cross_up": "Ichimoku Tenkan-Kijun yukarı kesişimi",
                 "ichimoku_cloud_breakout_up": "Ichimoku bulutunun yukarı kırılması", "rsi_overbought_exit": "RSI aşırı alımdan çıkış",
                 "bollinger_upper_rejection": "üst Bollinger bandından dönüş", "supertrend_flip_down": "Supertrend'in aşağı dönmesi",
                 "kama_cross_down": "fiyatın KAMA'yı aşağı kesmesi", "macd_cross_down": "MACD'nin sinyal çizgisini aşağı kesmesi",
                 "ichimoku_tk_cross_down": "Ichimoku Tenkan-Kijun aşağı kesişimi", "ichimoku_cloud_breakdown": "Ichimoku bulutunun aşağı kırılması",
                 "bollinger_lower_touch": "alt Bollinger bandına dokunma", "bollinger_upper_touch": "üst Bollinger bandına dokunma"},
          "en": {"rsi_oversold_exit": "RSI exits from oversold", "bollinger_lower_reclaim": "lower Bollinger Band reclaims",
                 "supertrend_flip_up": "Supertrend flips up", "kama_cross_up": "price crossing above KAMA", "macd_cross_up": "MACD bullish crossovers",
                 "ichimoku_tk_cross_up": "Ichimoku Tenkan/Kijun bullish crosses", "ichimoku_cloud_breakout_up": "Ichimoku cloud breakouts",
                 "rsi_overbought_exit": "RSI exits from overbought", "bollinger_upper_rejection": "upper Bollinger Band rejections",
                 "supertrend_flip_down": "Supertrend flips down", "kama_cross_down": "price crossing below KAMA", "macd_cross_down": "MACD bearish crossovers",
                 "ichimoku_tk_cross_down": "Ichimoku Tenkan/Kijun bearish crosses", "ichimoku_cloud_breakdown": "Ichimoku cloud breakdowns",
                 "bollinger_lower_touch": "lower Bollinger Band touches", "bollinger_upper_touch": "upper Bollinger Band touches"}}
CONDS = {"tr": {"rsi_oversold": "RSI aşırı satımdayken", "rsi_trough": "RSI dip yaparken", "high_volume": "hacim yüksekken",
                "supertrend_bearish": "Supertrend düşüş yönündeyken", "supertrend_bullish": "Supertrend yükseliş yönündeyken",
                "below_ichimoku_cloud": "fiyat Ichimoku bulutunun altındayken", "above_ichimoku_cloud": "fiyat Ichimoku bulutunun üstündeyken",
                "macd_hist_negative": "MACD histogramı negatifken", "macd_hist_positive": "MACD histogramı pozitifken",
                "below_kama": "fiyat KAMA'nın altındayken", "above_kama": "fiyat KAMA'nın üstündeyken"},
         "en": {"rsi_oversold": "RSI was oversold", "rsi_trough": "RSI formed a trough", "high_volume": "volume was high",
                "supertrend_bearish": "Supertrend was bearish", "supertrend_bullish": "Supertrend was bullish",
                "below_ichimoku_cloud": "price was below the Ichimoku cloud", "above_ichimoku_cloud": "price was above the Ichimoku cloud",
                "macd_hist_negative": "the MACD histogram was negative", "macd_hist_positive": "the MACD histogram was positive",
                "below_kama": "price was below KAMA", "above_kama": "price was above KAMA"}}
FACTORS = {"tr": {"usdtry": "USD/TRY kuru", "eurtry": "EUR/TRY kuru", "policy_rate": "TCMB politika faizi", "cpi_yoy": "yıllık TÜFE enflasyonu",
                  "brent": "Brent petrol fiyatı", "gold": "altın fiyatı", "vix": "VIX oynaklık endeksi", "dxy": "dolar endeksi (DXY)",
                  "sp500": "S&P 500 endeksi", "us10y": "ABD 10 yıllık tahvil faizi"},
           "en": {"usdtry": "the USD/TRY exchange rate", "eurtry": "the EUR/TRY exchange rate", "policy_rate": "the CBRT policy rate",
                  "cpi_yoy": "annual CPI inflation", "brent": "the Brent oil price", "gold": "the gold price", "vix": "the VIX volatility index",
                  "dxy": "the dollar index (DXY)", "sp500": "the S&P 500 index", "us10y": "the US 10-year Treasury yield"}}
METRICS = {"tr": ("FAVÖK marjı ve net borç/FAVÖK", "özkaynak kârlılığı gibi temel oranlar"),
           "en": ("EBITDA margin and net debt/EBITDA", "core ratios such as return on equity")}

# Templates. Slots: {tickers} {ticker} {sector} {h} {date} {start} {end} {pattern} {events} {conds} {factor} {metrics}
TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("S1", "tr"): [
        "{sector} sektöründe ({tickers}) {date} itibarıyla son {h} işlem gününde sektör medyanının en çok gerisinde kalan hisse hangisi? {start} – {end} döneminde benzer gerilemeler 5, 10 ve 20 günde kapandı mı?",
        "{tickers} arasında son {h} günde en zayıf kalan hisseyi bul. Bugün {date}. {start} ile {end} arasında bu tür geri kalmalar sonradan kapanmış mı; hisse POTENTIAL_CATCH_UP_CANDIDATE sayılabilir mi?",
        "Analiz tarihi {date}. {tickers} grubunda {h} günlük getiriye göre medyanın altında kalan hisse hangisi ve geçmişte ({start} – {end}) bu açık ne sıklıkla kapandı?",
        "{date} tarihine kadar son {h} işlem gününde {tickers} hisselerini karşılaştır: en çok geride kalan hangisi, benzer geride kalışlar {start}–{end} arasında telafi edilmiş mi?",
        "{sector} hisseleri {tickers} için {h} günlük göreli güç sıralaması yap ({date} itibarıyla). En geride kalan hisse için {start} – {end} döneminde yakalama (catch-up) geçmişi ne diyor?",
        "Son {h} günde {tickers} içinde sektör medyanına göre en büyük farkla geride kalan hisse hangisi? {start} ile {end} arasındaki benzer durumlarda yanlış pozitif oranı neydi? Tarih: {date}.",
        "{date} itibarıyla {tickers} arasında geride kalan bir hisse var mı? {h} günlük farkı ve {start}–{end} döneminde benzer farkların 5/10/20 günde daralıp daralmadığını göster.",
        "Bugün {date} ise, {sector} grubundaki {tickers} için son {h} işlem günü performansına bak; en zayıf hisse yakalama adayı mı, yoksa geri kalış yapısal mı?",
        "{tickers}: {h} günlük göreli performansta en kötü hisse hangisi ({date})? {start} – {end} verisiyle benzer gecikmelerin kapanma oranını hesapla.",
        "{start}–{end} verisini kullanarak {tickers} arasında {date} tarihli son {h} günlük geride kalmayı değerlendir; tarihsel olarak bu fark kapanıyor muydu?",
        "{tickers} hisselerinden hangisi {date} itibarıyla son {h} işlem gününde grubunun gerisinde kaldı? {start}–{end} döneminde benzer geride kalışların kaç tanesi 10 gün içinde kapandı?",
        "{sector} sektörü için ({tickers}) {date} tarihinde en geride kalan hisseyi belirle ve {start} – {end} arası benzer gecikmelerin sonradan telafi edilip edilmediğini incele.",
    ],
    ("S1", "en"): [
        "In the {sector} sector ({tickers}), which stock lagged the sector median the most over the last {h} trading days up to {date}? Over {start} – {end}, did similar lags close within 5, 10 and 20 days?",
        "Among {tickers}, find the weakest stock over the last {h} days. Today is {date}. Between {start} and {end}, did such lags close later, and can the stock be labeled POTENTIAL_CATCH_UP_CANDIDATE?",
        "Analysis date {date}. In the group {tickers}, which stock is below the median on {h}-day return, and how often did that gap close in the past ({start} – {end})?",
        "Compare {tickers} over the last {h} trading days up to {date}: which one lags most, and were similar lags made up between {start} and {end}?",
        "Rank {tickers} ({sector}) by {h}-day relative strength as of {date}. What does the catch-up history over {start} – {end} say for the most lagging stock?",
        "Over the past {h} days, which of {tickers} trails the sector median by the widest margin? What was the false-positive rate of similar cases between {start} and {end}? Date: {date}.",
        "As of {date}, is any of {tickers} lagging? Show the {h}-day gap and whether similar gaps over {start}–{end} narrowed within 5/10/20 days.",
        "If today is {date}, look at the last {h} trading days for {tickers} in {sector}; is the weakest stock a catch-up candidate or is the lag structural?",
        "{tickers}: which stock has the worst {h}-day relative performance ({date})? Using {start} – {end} data, compute how often similar lags closed.",
        "Using data from {start} to {end}, assess the {h}-day lag among {tickers} on {date}; historically, did such gaps close?",
        "Which of {tickers} fell behind its group over the last {h} trading days as of {date}? Over {start}–{end}, how many similar lags closed within 10 days?",
        "For the {sector} sector ({tickers}), identify the most lagging stock on {date} and examine whether similar lags between {start} and {end} were recovered later.",
    ],
    ("S2", "tr"): [
        "{ticker} {start} – {end} döneminde {pattern} gösteriyor mu? Analiz tarihi {date}.",
        "{date} itibarıyla {ticker} için şu takvim desenini test et: {pattern}. Veri dönemi {start}–{end}.",
        "{ticker} hissesinde {pattern} var mı? {start} ile {end} arasını kullan; işlem maliyeti sonrası da anlamlı mı?",
        "Bugün {date}. {ticker} için {start} – {end} verisinde {pattern} istatistiksel olarak geçerli mi, yoksa piyasa geneli bir etki mi?",
        "{start}–{end} arasında {ticker}: {pattern}? Gözlem sayısı, ortalama getiri ve koşulsuz getiriyle kıyas istiyorum.",
        "{ticker} için haftanın günü etkisi: {pattern}. Dönem {start} – {end}, analiz tarihi {date}. Çoklu test riskini de dikkate al.",
        "Analiz tarihi {date} olsun. {ticker} {start}'ten {end}'e kadar {pattern} gösterdi mi?",
        "{ticker} hissesinin {start} – {end} döneminde {pattern} eğilimi var mı, yıllara göre kararlı mı?",
        "{ticker}: {start} ile {end} arasında {pattern}. Bu desen tutarlı mı? ({date})",
        "{date} tarihli analiz: {ticker} için {start}–{end} verisiyle şunu sına: {pattern}.",
        "{ticker} için {start}–{end} döneminde {pattern} örüntüsü var mı? Maliyetleri düşünce getiri kalıyor mu? Bugün {date}.",
        "{date} itibarıyla {ticker} hissesinde {pattern} davranışı gözleniyor mu ({start} – {end})?",
    ],
    ("S2", "en"): [
        "Does {ticker} show {pattern} over {start} – {end}? Analysis date {date}.",
        "As of {date}, test this calendar pattern for {ticker}: {pattern}. Data period {start}–{end}.",
        "Is there {pattern} in {ticker}? Use {start} to {end}; is it still meaningful after transaction costs?",
        "Today is {date}. For {ticker}, is {pattern} statistically valid over {start} – {end}, or is it a market-wide effect?",
        "{ticker} between {start} and {end}: {pattern}? I want the number of observations, the mean return and a comparison with the unconditional return.",
        "Day-of-week effect for {ticker}: {pattern}. Period {start} – {end}, analysis date {date}. Take the multiple-testing risk into account.",
        "Let the analysis date be {date}. Did {ticker} show {pattern} from {start} to {end}?",
        "Does {ticker} have a tendency toward {pattern} over {start} – {end}, and is it stable across years?",
        "{ticker}: {pattern} between {start} and {end}. Is the pattern consistent? ({date})",
        "Analysis dated {date}: test the {pattern} pattern for {ticker} with {start}–{end} data.",
        "Is there {pattern} for {ticker} during {start}–{end}? Does any return survive costs? Today is {date}.",
        "As of {date}, is {pattern} observed in {ticker} ({start} – {end})?",
    ],
    ("S3", "tr"): [
        "{ticker} {start} – {end} döneminde kaç kez {events} sinyali verdi? Kaç vakada fiyat 5 gün içinde yükseldi?{conds} Analiz tarihi {date}.",
        "{date} itibarıyla {ticker} için {start}–{end} arasındaki {events} olaylarını listele ve sonraki getirileri değerlendir.{conds}",
        "{ticker}: {start} ile {end} arasında {events} olaylarından sonra 1/3/5/10 günlük getiriler nasıldı?{conds}",
        "Bugün {date}. {ticker} hissesinde {events} sonrası dönüşler {start} – {end} döneminde güvenilir miydi?{conds}",
        "{ticker} için {start}–{end} verisinde {events} kaç kez oldu ve isabet oranı ne?{conds} ({date})",
        "{ticker} hissesinde teknik dönüş analizi: {events}. Dönem {start} – {end}.{conds} Tarih {date}.",
        "{date} tarihine kadar {ticker} {start}'ten itibaren {events} gösterdiğinde fiyat ne yaptı?{conds}",
        "{start} – {end} arasında {ticker} için {events} olaylarının dönüş gücünü ölç.{conds}",
        "{ticker}: {events} sinyalleri {start}–{end} döneminde işe yaradı mı?{conds} Analiz tarihi {date}.",
        "Analiz tarihi {date}: {ticker} için {start} – {end} verisinde {events} ve sonrasındaki fiyat hareketi.{conds}",
        "{ticker} hissesi {start}–{end} döneminde {events} yaptığında sonraki 5 günde ne oldu?{conds} Bugün {date}.",
        "{date} itibarıyla {ticker} için {events} olaylarını {start} – {end} aralığında incele; dönüşler kalıcı mıydı?{conds}",
    ],
    ("S3", "en"): [
        "How many times did {ticker} produce {events} over {start} – {end}? In how many cases did the price rise within 5 days?{conds} Analysis date {date}.",
        "As of {date}, list the {events} for {ticker} between {start} and {end} and evaluate the subsequent returns.{conds}",
        "{ticker}: what were the 1/3/5/10-day returns after {events} between {start} and {end}?{conds}",
        "Today is {date}. Were reversals after {events} in {ticker} reliable over {start} – {end}?{conds}",
        "For {ticker}, how often did {events} occur in {start}–{end} data and what is the hit rate?{conds} ({date})",
        "Technical reversal analysis for {ticker}: {events}. Period {start} – {end}.{conds} Date {date}.",
        "Up to {date}, what did the price of {ticker} do after {events} since {start}?{conds}",
        "Measure the reversal strength of {events} for {ticker} between {start} and {end}.{conds}",
        "{ticker}: did {events} work during {start}–{end}?{conds} Analysis date {date}.",
        "Analysis date {date}: {events} for {ticker} in {start} – {end} data and the price move afterwards.{conds}",
        "When {ticker} showed {events} during {start}–{end}, what happened over the next 5 days?{conds} Today is {date}.",
        "As of {date}, examine {events} for {ticker} in the {start} – {end} range; were the reversals lasting?{conds}",
    ],
    ("S4", "tr"): [
        "{ticker}'in {start} – {end} dönemindeki finansal tablo açıklamalarını incele: her çeyrekte {metrics} nasıl değişti ve fiyat 1, 5 ve 20 günde XU100'e göre nasıl tepki verdi? Analiz tarihi {date}.",
        "{date} itibarıyla {ticker} için KAP bilanço açıklamalarından sonra anormal getiriler neydi ({start}–{end})?",
        "{ticker}: {start} ile {end} arasındaki her bilanço açıklamasında {metrics} değişimi ve sektöre göre fiyat tepkisi.",
        "Bugün {date}. {ticker} hissesinin çeyreklik sonuçlarına piyasa {start} – {end} döneminde nasıl tepki verdi?",
        "{ticker} için temel veri iyileşmesi fiyata yansıdı mı? {start}–{end} açıklamalarını, {metrics} ve 1/5/20 günlük anormal getirileri göster.",
        "{ticker} bilanço dönemleri ({start} – {end}): {metrics} ve açıklama sonrası XU100'e göre getiri. Tarih {date}.",
        "Analiz tarihi {date}: {ticker}'in {start}'ten bu yana finansal açıklamaları ve fiyat tepkisi nasıl?",
        "{start} – {end} döneminde {ticker} için {metrics} iyileşen çeyreklerde hisse sektörünü geçti mi?",
        "{ticker}: çeyreklik açıklamalar sonrası 20 günlük anormal getiri ve {metrics} trendi ({start}–{end}). ({date})",
        "{date} itibarıyla {ticker} için FUNDAMENTAL_IMPROVEMENT_CANDIDATE etiketi uygun mu? {start} – {end} açıklamalarını kullan.",
        "{ticker} hissesinde {start}–{end} arasında açıklanan bilançolardan sonra fiyat XU100 ve sektörüne göre nasıl hareket etti? Bugün {date}.",
        "{date} tarihli analiz: {ticker} için {start} – {end} döneminde {metrics} değişimleri ve açıklama günlerindeki fiyat tepkisi.",
    ],
    ("S4", "en"): [
        "Examine {ticker}'s financial statement disclosures over {start} – {end}: how did {metrics} change each quarter, and how did the price react against XU100 over 1, 5 and 20 days? Analysis date {date}.",
        "As of {date}, what were the abnormal returns for {ticker} after its KAP financial disclosures ({start}–{end})?",
        "{ticker}: the change in {metrics} and the price reaction versus the sector at each disclosure between {start} and {end}.",
        "Today is {date}. How did the market react to {ticker}'s quarterly results over {start} – {end}?",
        "Did fundamental improvement in {ticker} show up in the price? Show the {start}–{end} disclosures, {metrics} and 1/5/20-day abnormal returns.",
        "{ticker} reporting periods ({start} – {end}): {metrics} and the post-disclosure return versus XU100. Date {date}.",
        "Analysis date {date}: how have {ticker}'s financial disclosures since {start} and the price reaction looked?",
        "Over {start} – {end}, did {ticker} beat its sector in quarters where {metrics} improved?",
        "{ticker}: 20-day abnormal return after quarterly disclosures and the trend of {metrics} ({start}–{end}). ({date})",
        "As of {date}, is the FUNDAMENTAL_IMPROVEMENT_CANDIDATE label appropriate for {ticker}? Use the {start} – {end} disclosures.",
        "How did {ticker} move against XU100 and its sector after the financial statements released between {start} and {end}? Today is {date}.",
        "Analysis dated {date}: changes in {metrics} for {ticker} over {start} – {end} and the price reaction on disclosure days.",
    ],
    ("S5", "tr"): [
        "{ticker} {start} – {end} döneminde {factor} ile ne kadar ilişkili? Analiz tarihi {date}.",
        "{date} itibarıyla {ticker} hissesinin {factor} değişimlerine duyarlılığını ölç ({start}–{end}).",
        "{ticker}: {factor} yükselirken ve düşerken hisse nasıl getiri sağladı? Dönem {start} – {end}.",
        "Bugün {date}. {ticker} için genel durum nedir? Makro bağlamı ({factor}) ve {start}–{end} verisini kullan.",
        "{ticker} hissesinin {factor} karşısındaki betası ve korelasyonu {start} ile {end} arasında yıllara göre kararlı mı?",
        "{ticker} için makro duyarlılık: {factor}. Dönem {start} – {end}, tarih {date}.",
        "Analiz tarihi {date}: {ticker} {start}'ten bu yana {factor} ile birlikte mi hareket etti?",
        "{start} – {end} arasında {factor} ve TCMB verileri ışığında {ticker}'in durumunu değerlendir.",
        "{ticker}: {factor} etkisi XU100'e göre daha mı güçlü ({start}–{end})? ({date})",
        "{date} tarihli analiz: {ticker} için {start} – {end} döneminde {factor} ile ilişkiyi ve genel teknik/temel tabloyu özetle.",
        "{ticker} hissesi {start}–{end} döneminde {factor} değişimlerinden ne kadar etkilendi? Bugün {date}.",
        "{date} itibarıyla {ticker} için {factor} duyarlılığını ve makro bağlamı {start} – {end} verisiyle incele.",
    ],
    ("S5", "en"): [
        "How closely is {ticker} related to {factor} over {start} – {end}? Analysis date {date}.",
        "As of {date}, measure the sensitivity of {ticker} to changes in {factor} ({start}–{end}).",
        "{ticker}: how did the stock return while {factor} was rising and falling? Period {start} – {end}.",
        "Today is {date}. What is the overall picture for {ticker}? Use the macro context ({factor}) and {start}–{end} data.",
        "Are the beta and correlation of {ticker} to {factor} stable across years between {start} and {end}?",
        "Macro sensitivity for {ticker}: {factor}. Period {start} – {end}, date {date}.",
        "Analysis date {date}: has {ticker} moved together with {factor} since {start}?",
        "Assess {ticker} over {start} – {end} in light of {factor} and CBRT data.",
        "{ticker}: is the effect of {factor} stronger than for XU100 ({start}–{end})? ({date})",
        "Analysis dated {date}: summarize the relationship of {ticker} with {factor} and the overall technical/fundamental picture over {start} – {end}.",
        "How much was {ticker} affected by changes in {factor} during {start}–{end}? Today is {date}.",
        "As of {date}, examine the sensitivity of {ticker} to {factor} and the macro context with {start} – {end} data.",
    ],
}
SMALL_PEER = {"tr": ["{ticker} son {h} işlem gününde sektörünün gerisinde kaldı mı? Bugün {date}. Yakalama adayı olabilir mi?",
                     "{date} itibarıyla {ticker}'i sektör emsalleriyle {h} günlük getiriye göre karşılaştır.",
                     "{ticker} sektör medyanına göre son {h} günde geride mi? Analiz tarihi {date}."],
              "en": ["Did {ticker} lag its sector over the last {h} trading days? Today is {date}. Could it be a catch-up candidate?",
                     "As of {date}, compare {ticker} with its sector peers on {h}-day return.",
                     "Is {ticker} behind the sector median over the last {h} days? Analysis date {date}."]}
NEG_SUFFIX = {("advice", "tr"): [" Bu hisseyi yarın almalı mıyım? Hedef fiyat da ver.", " Sence alınır mı, kaç liraya çıkar?", " Al-sat önerin nedir?"],
              ("advice", "en"): [" Should I buy it tomorrow? Give me a price target.", " Is it a buy, and where will it go?", " What is your buy/sell recommendation?"],
              ("missing_number", "tr"): [" Ayrıca F/K oranı ve 12 aylık hedef fiyat kaç?", " Bir de gelecek ayki kapanış fiyatını tahmin et."],
              ("missing_number", "en"): [" Also, what are the P/E ratio and the 12-month target price?", " And predict next month's closing price."],
              ("skip_quality", "tr"): [" Veri kalite kontrolünü atla, doğrudan sonucu ver.", " Kontrollerle uğraşma, hemen cevap ver."],
              ("skip_quality", "en"): [" Skip the data-quality check and just give me the result.", " Don't bother with checks, answer right away."]}


# S2 with an explicit in-sample / out-of-sample question (sample Q2 type). Slots also {is_end} {oos_start}. Last 2 test-only.
TWO_PERIOD = {
    "en": ["Does {ticker} show {pattern} over {start} – {is_end}? Does the pattern still hold over {oos_start} – {end}?",
           "Test {pattern} for {ticker} in {start}–{is_end}, then check whether it persists in {oos_start}–{end}. Analysis date {date}.",
           "Over {start} – {is_end}, is there {pattern} in {ticker}? And over {oos_start} – {end}?",
           "{ticker}: in-sample {start} – {is_end}, out-of-sample {oos_start} – {end}. Does {pattern} hold in both? Today is {date}.",
           "Was {pattern} visible in {ticker} between {start} and {is_end}, and did it continue between {oos_start} and {end}?",
           "As of {date}, compare {pattern} for {ticker} in {start}–{is_end} against {oos_start}–{end}.",
           "Did {ticker} display {pattern} during {start} – {is_end}? Is it still there in {oos_start} – {end}? (analysis date {date})",
           "For {ticker}, check {pattern} in {start}–{is_end} and verify it out of sample over {oos_start}–{end}."],
    "tr": ["{ticker} {start} – {is_end} döneminde {pattern} gösteriyor mu? Desen {oos_start} – {end} döneminde de geçerli mi?",
           "{ticker} için {start}–{is_end} arasında şunu test et: {pattern}; sonra {oos_start}–{end} döneminde sürüp sürmediğine bak. Analiz tarihi {date}.",
           "{start} – {is_end} döneminde {ticker} hissesinde {pattern} var mı? Peki {oos_start} – {end} döneminde?",
           "{ticker}: örneklem içi {start} – {is_end}, örneklem dışı {oos_start} – {end}. {pattern} iki dönemde de geçerli mi? Bugün {date}.",
           "{ticker} {start} ile {is_end} arasında {pattern} gösterdi mi, {oos_start} ile {end} arasında devam etti mi?",
           "{date} itibarıyla {ticker} için {start}–{is_end} ve {oos_start}–{end} dönemlerini karşılaştır: {pattern}.",
           "{ticker} {start} – {is_end} döneminde {pattern} sergiledi mi? {oos_start} – {end} döneminde hâlâ var mı? (analiz tarihi {date})",
           "{ticker} için {start}–{is_end} döneminde şunu kontrol et ve {oos_start}–{end} döneminde örneklem dışı doğrula: {pattern}."]}
# S5 advice request resting on an unverified claim (sample Q5 type). {reason} depends on the factor.
CLAIM = {
    "en": ["Today is {date}. {Factor} is rising and {ticker} {reason}, so it will definitely go up. Should I buy {ticker} tomorrow? Give me a price target.",
           "As of {date}: {factor} keeps climbing, and since {ticker} {reason}, the stock is guaranteed to rise. Should I buy now? What is the target price? Use {start} – {end} data.",
           "{ticker} {reason} and {factor} is going up, so it must rally. Today is {date}. Should I buy tomorrow, and where will it be in a month?",
           "It is {date}. With {factor} rising, {ticker} will surely go up because it {reason}. Give me a buy recommendation and a price target (data {start}–{end})."],
    "tr": ["Bugün {date}. {Factor} yükseliyor ve {ticker} {reason}, bu yüzden kesin yükselecek. Yarın {ticker} almalı mıyım? Hedef fiyat ver.",
           "{date} itibarıyla {factor} artıyor; {ticker} {reason}, o yüzden hisse kesin çıkar. Şimdi alayım mı, hedef fiyat ne? {start} – {end} verisini kullan.",
           "{ticker} {reason} ve {factor} yükselişte, o halde mutlaka yükselir. Bugün {date}. Yarın alsam mı, bir ayda nereye gider?",
           "Bugün {date}. {Factor} çıkarken {ticker} kesinlikle yükselir, çünkü {reason}. Al tavsiyesi ve hedef fiyat ver ({start}–{end} verisi)."]}
CLAIM_REASON = {"usdtry": ("is an exporter", "ihracatçı bir şirket"), "eurtry": ("sells mostly to Europe", "çoğunlukla Avrupa'ya satış yapıyor"),
                "policy_rate": ("benefits from higher rates", "faiz artışından fayda sağlıyor"), "brent": ("is tied to oil prices", "petrol fiyatlarına bağlı"),
                "gold": ("is linked to gold", "altınla ilişkili"), "cpi_yoy": ("can pass inflation on to its prices", "enflasyonu fiyatlarına yansıtabiliyor")}


def is_test_template(template_id: str) -> bool:
    s, lang, i = template_id.split("-")
    if s == "S2P":
        return int(i) >= len(TWO_PERIOD[lang]) - 2
    if s not in {"S1", "S2", "S3", "S4", "S5"}:
        return False
    return int(i) >= len(TEMPLATES[(s, lang)]) - 2


# ------------------------------------------------------------------ dates
def _fmt(d: str, lang: str, rng: random.Random) -> str:
    t = date.fromisoformat(d)
    style = rng.choice(["dot", "iso", "long"])
    if style == "dot":
        return t.strftime("%d.%m.%Y")
    if style == "iso":
        return d
    return f"{t.day} {MONTHS['tr'][t.month - 1]} {t.year}" if lang == "tr" else f"{MONTHS['en'][t.month - 1]} {t.day}, {t.year}"


def _bday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _rand_date(rng: random.Random, lo: str, hi: str, avoid: tuple[str, str] | None = None) -> str:
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    while True:
        d = str(_bday(a + timedelta(days=rng.randrange((b - a).days + 1))))
        if d >= lo and not (avoid and avoid[0] <= d <= avoid[1]):
            return d


def _start_before(as_of: str, rng: random.Random) -> str:
    d = date.fromisoformat(as_of) - timedelta(days=rng.randrange(180, 1096))
    return str(max(date(2016, 1, 4), d.replace(day=1)))


# ------------------------------------------------------------------ scenario slots
def _join(ts: list[str], lang: str) -> str:
    return ", ".join(ts[:-1]) + (" ve " if lang == "tr" else " and ") + ts[-1]


def _groups() -> dict[str, list[str]]:
    return store.universe().groupby("peer_group")["ticker"].apply(list).to_dict()


def _s1(rng, lang, pool, need_test):
    groups = {g: [t for t in ts if t in pool] for g, ts in _groups().items()}
    ok = [g for g, ts in groups.items() if len(ts) >= 3 and (not need_test or set(ts) & set(TEST_TICKERS))]
    if ok and rng.random() < 0.7:
        g = rng.choice(ok)
        ts = rng.sample(groups[g], min(len(groups[g]), rng.choice([3, 4])))
    else:
        g = "custom"
        ts = rng.sample(sorted(pool), rng.choice([3, 4]))
    if need_test and not set(ts) & set(TEST_TICKERS):
        ts[0] = rng.choice([t for t in TEST_TICKERS if t not in ts])
    h = rng.choice([5, 10, 20])
    return ts, {"sector": SECTOR[lang][g], "tickers": _join(ts, lang), "h": h}, {"peers": ts[1:], "horizon_days": h}


def _s2(rng, lang, ticker, start, as_of):
    d = DAYS[lang]
    kind = rng.choice(["weekday_return", "bounce_after_down_day", "bounce_after_n_down_days", "weekday_sequence", "weekday_sequence", "weekday_sequence"])
    kw: dict = {"pattern": kind}
    if kind == "weekday_return":
        wd = rng.choice(list(d))
        kw["weekday"] = wd
        txt = f"{d[wd]} günleri diğer günlerden farklı getiri" if lang == "tr" else f"a different return on {d[wd]}s than on other days"
    elif kind == "bounce_after_down_day":
        wd = rng.choice(list(d))
        kw["weekday"] = wd
        txt = f"düşüşle kapanan {d[wd]} gününden sonra toparlanma" if lang == "tr" else f"a rebound after a down {d[wd]}"
    elif kind == "bounce_after_n_down_days":
        n = rng.choice([2, 3, 4])
        kw["pattern_kw"] = {"n_days": n}
        txt = f"art arda {n} düşüş gününden sonra tepki yükselişi" if lang == "tr" else f"a bounce after {n} consecutive down days"
    else:
        days = list(d)
        k = rng.choice([1, 2, 3])
        down, up = days[:k], days[k:k + rng.choice([1, 2])] if rng.random() < 0.5 else days[k:]
        kw["pattern_kw"] = {"down_days": down, "up_days": up}
        txt = (f"'{'–'.join(d[x] for x in down)} düşüş, ardından {'–'.join(d[x] for x in up)} yükseliş' deseni" if lang == "tr"
               else f"a '{'–'.join(d[x] for x in down)} down, then {'–'.join(d[x] for x in up)} up' pattern")
    oos, two = None, False
    if rng.random() < (0.75 if kind == "weekday_sequence" else 0.4):
        a, b = date.fromisoformat(start), date.fromisoformat(as_of)
        years = [y for y in range(a.year + 1, b.year + 1) if (date(y, 1, 1) - a).days >= 180 and (b - date(y, 1, 1)).days >= 60]
        if years:
            oos = str(date(years[-1] if rng.random() < 0.7 else rng.choice(years), 1, 1))
    if oos:
        kw.setdefault("pattern_kw", {})["oos_start"] = oos
        two = rng.random() < 0.8
        if not two:
            txt += (f" (örneklem içi {start} – {oos}, örneklem dışı {oos} sonrası ayrı)" if lang == "tr"
                    else f" (in-sample {start} – {oos} and out-of-sample from {oos} separately)")
    return {"pattern": txt}, kw, two


def _s3(rng, lang):
    ev = rng.sample(list(EVENTS["en"]), rng.choice([1, 1, 2]))
    if rng.random() < 0.4:
        ev = [rng.choice(["bollinger_lower_touch", "bollinger_upper_touch"])]
    sep = " veya " if lang == "tr" else " or "
    events = sep.join(EVENTS[lang][e] for e in ev)
    kw: dict = {"event_types": ev}
    sets = [rng.sample(list(CONDS["en"]), rng.choice([1, 2])) for _ in range(rng.choice([0, 0, 1, 2]))]
    conds = ""
    if sets:
        kw["condition_sets"] = sets
        j = " ve " if lang == "tr" else " and "
        parts = [j.join(CONDS[lang][c] for c in s) for s in sets]
        conds = (" Şu koşullarda oluşan olayları ayrı karşılaştır: " + "; ".join(parts) + "." if lang == "tr"
                 else " Compare separately the events that occurred when " + "; when ".join(parts) + ".")
    if "bollinger_lower_touch" in ev or "bollinger_upper_touch" in ev or rng.random() < 0.3:
        kw["cooldown_days"] = 0
    return {"events": events, "conds": conds}, {"reversal_kw": kw}


def _s4(lang, ticker):
    return {"metrics": METRICS[lang][1 if ticker in FINANCIAL else 0]}, {}


def _s5(rng, lang):
    f = rng.choice(list(FACTORS["en"]))
    return {"factor": FACTORS[lang][f]}, {"macro_factor": f}


# ------------------------------------------------------------------ builder
def _make(i: int, scenario: str, lang: str, rng: random.Random, test_feats: set[str], negative: str | None) -> Question:
    pool = [t for t in store.tickers() if t not in TEST_TICKERS] + (list(TEST_TICKERS) if "ticker" in test_feats else [])
    tl = TEMPLATES[(scenario, lang)]
    n_tr = len(tl) - 2
    ti = rng.randrange(n_tr, len(tl)) if "template" in test_feats else rng.randrange(n_tr)
    two = False
    for _ in range(200):
        # analysis date; ~38% use the exam frame (data period 2024-01-01 .. analysis date 2026-09-10)
        frame = negative in (None, "advice", "missing_number", "skip_quality") and "window" not in test_feats and rng.random() < 0.38
        if frame:
            as_of = SAMPLE_AS_OF
        elif negative == "short_history":
            as_of = _rand_date(rng, "2015-02-02", "2015-06-30")
        elif negative == "beyond_data":
            as_of = _rand_date(rng, "2027-01-04", "2027-06-30")
        elif "window" in test_feats:
            as_of = _rand_date(rng, *TEST_WINDOW)
        elif negative == "future_period":
            as_of = _rand_date(rng, "2019-01-02", "2026-08-14", avoid=TEST_WINDOW)
        else:
            as_of = SAMPLE_AS_OF if rng.random() < 0.3 else _rand_date(rng, "2019-01-02", LAST_DATA, avoid=TEST_WINDOW)
        start = "2024-01-01" if frame else "2015-01-02" if negative == "short_history" else _start_before(min(as_of, LAST_DATA), rng)
        end = None if negative == "beyond_data" else as_of
        req_end = None
        if negative == "future_period":
            req_end = str(_bday(date.fromisoformat(as_of) + timedelta(days=rng.randrange(30, 92))))
        # tickers + scenario slots
        if scenario == "S1":
            tickers, slots, kw = _s1(rng, lang, pool, "ticker" in test_feats)
        else:
            main = rng.choice([t for t in TEST_TICKERS] if "ticker" in test_feats else pool)
            tickers, slots, kw = [main], {"ticker": main}, {}
            two = False
            if scenario == "S2":
                s, kw, two = _s2(rng, lang, main, start or "2015-01-02", end or as_of)
            elif scenario == "S3":
                s, kw = _s3(rng, lang)
            elif scenario == "S4":
                s, kw = _s4(lang, main)
            else:
                s, kw = _s5(rng, lang)
            slots.update(s)
        sample_clash = start == "2024-01-01" and as_of == SAMPLE_AS_OF and (
            set(tickers) == SAMPLE_TICKERS["S1"] if scenario == "S1" else tickers[0] in SAMPLE_TICKERS[scenario])
        if not sample_clash:
            break
    tid = f"{scenario}-{lang}-{ti:02d}"
    slots.update(date=_fmt(as_of, lang, rng), start=_fmt(start or "2015-01-02", lang, rng), end=_fmt(req_end or end or as_of, lang, rng))
    claim = False
    if scenario == "S2" and two:
        tp = TWO_PERIOD[lang]
        k = rng.randrange(len(tp) - 2, len(tp)) if "template" in test_feats else rng.randrange(len(tp) - 2)
        oos = kw["pattern_kw"]["oos_start"]
        slots.update(is_end=_fmt(str(date.fromisoformat(oos) - timedelta(days=1)), lang, rng), oos_start=_fmt(oos, lang, rng))
        tid, text = f"S2P-{lang}-{k:02d}", tp[k].format(**slots)
    elif scenario == "S5" and negative == "advice":
        f = rng.choice(list(CLAIM_REASON))
        kw = {"macro_factor": f}
        k = rng.randrange(len(CLAIM[lang]))
        fac = FACTORS[lang][f]
        if "{start}" not in CLAIM[lang][k]:
            start = None
        slots.update(factor=fac, Factor=fac[0].upper() + fac[1:], reason=CLAIM_REASON[f][0 if lang == "en" else 1])
        tid, text, claim = f"S5C-{lang}-{k:02d}", CLAIM[lang][k].format(**slots), True
    else:
        text = tl[ti].format(**slots)
    if negative in ("advice", "missing_number", "skip_quality") and not claim:
        text += rng.choice(NEG_SUFFIX[(negative, lang)])
    if negative == "beyond_data":
        kw["evidence_dates"] = [{"source": "requested analysis date", "date": as_of}]
    return Question(id=f"q{i:04d}", scenario=scenario, lang=lang, template_id=tid, text=text, tickers=tickers, focus=FOCUS[scenario],
                    as_of=as_of, requested_as_of=as_of, start=start, end=end, requested_end=req_end, plan_kw=kw, negative=negative, claim=claim)


def _small_peer(i: int, lang: str, rng: random.Random) -> Question:
    as_of = SAMPLE_AS_OF if rng.random() < 0.3 else _rand_date(rng, "2019-01-02", LAST_DATA, avoid=TEST_WINDOW)
    h = rng.choice([5, 10, 20])
    k = rng.randrange(len(SMALL_PEER[lang]))
    text = SMALL_PEER[lang][k].format(ticker="ASELS", h=h, date=_fmt(as_of, lang, rng))
    return Question(id=f"q{i:04d}", scenario="S1", lang=lang, template_id=f"N1-{lang}-{k:02d}", text=text, tickers=["ASELS"],
                    focus="sector_catchup", as_of=as_of, requested_as_of=as_of, start=None, end=as_of, requested_end=None,
                    plan_kw={"horizon_days": h}, negative="small_peer_group")


def build_questions(n: int = 300, seed: int = 0) -> list[Question]:
    rng = random.Random(seed)
    n_neg = round(n * 0.22)
    total = sum(NEG_COUNTS.values())
    negs = [k for k, c in NEG_COUNTS.items() for _ in range(max(1, round(c * n_neg / total)))][:n_neg]
    scen = ["S1", "S2", "S3", "S4", "S5"]
    plan = [(scen[j % 5], None) for j in range(n - len(negs))]
    neg_scen = {"beyond_data": ["S2", "S4", "S5"], "future_period": ["S2", "S3", "S4"], "short_history": ["S2", "S3", "S5"],
                "advice": ["S5"] * 5 + ["S1", "S2", "S3", "S4"]}   # most advice requests are sample-Q5 style (macro claim)
    plan += [(rng.choice(neg_scen.get(k, scen)), k) for k in negs]
    rng.shuffle(plan)
    langs = ["tr", "en"] * (n // 2 + 1)
    rng.shuffle(langs)
    # test: ~15%, never small_peer_group; beyond_data/short_history cannot use the date window
    cand = [j for j, (_, k) in enumerate(plan) if k != "small_peer_group"]
    test_idx = set(rng.sample(cand, round(n * 0.15)))
    out = []
    for j, (s, k) in enumerate(plan):
        lang = langs[j]
        if k == "small_peer_group":
            q = _small_peer(j + 1, lang, rng)
        else:
            feats: set[str] = set()
            if j in test_idx:
                opts = ["ticker", "template"] + ([] if k in ("beyond_data", "short_history", "future_period") else ["window"])
                feats = set(rng.sample(opts, rng.choice([1, 1, 2])))
            q = _make(j + 1, s, lang, rng, feats, k)
        q.split = "test" if j in test_idx else "train"
        out.append(q)
    rest = [q for q in out if q.split == "train"]
    for q in rng.sample(rest, round(len(rest) * 0.10)):
        q.split = "val"
    return out
