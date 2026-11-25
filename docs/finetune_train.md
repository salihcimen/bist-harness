# Faz 6 — QLoRA eğitimi ve Ollama'ya aktarım

## Çalıştırma
```
HF_HOME=.cache/hf PYTHONPATH=src .venv/Scripts/python.exe -m bist.finetune.train            # tam koşu -> finetune/runs/qlora_r16/
... -m bist.finetune.train --resume                                                        # son checkpoint'ten devam
... -m bist.finetune.to_ollama --run qlora_r16 --name bist-analyst                         # merge -> GGUF -> Q4_K_M -> ollama create
```
`finetune/runs/` gitignore'da (checkpoint'ler, birleştirilmiş model, GGUF'lar, loglar).

## Kurulum
| | |
|---|---|
| Temel model | `Qwen/Qwen3-4B-Instruct-2507` (HF, `.cache/hf`) |
| Nicemleme | 4-bit NF4, double-quant, bf16 hesap (bitsandbytes) |
| LoRA | r=16, alpha=32, dropout 0.05; q/k/v/o/gate/up/down |
| Optimizasyon | lr 2e-4 cosine, %3 ısınma, paged_adamw_8bit, batch 1 × grad accum 16, 2 epoch = 372 adım |
| Bellek | gradient checkpointing (non-reentrant); fp32 upcast yok; kayıp yalnız hedef mesajın logit'lerinden (`logits_to_keep`) |
| Kayıp maskesi | yalnız son asistan mesajı (prompt = -100); `tests/test_finetune_train.py` |
| Checkpoint | 20 adımda bir (son 3 tutulur), `--resume` ile kaldığı yerden; doğrulama 60 adımda bir, 80 val satırı |
| Veri | train 2,976 satır (hiçbiri 8,192 token sınırında atılmadı), val 339 (80'i izleme için) |

## Sonuç (koşu `qlora_r16`, 2026-11-23/24)
- Süre 5 sa 58 dk (RTX 4060 Laptop 8 GB), en yüksek GPU belleği 6.43 GB. İki kesinti (doğrulama hatası düzeltmesi) checkpoint'ten devam ettirildi.
- train_loss (ortalama) 0.0075.
- eval_loss: 60 → 0.0198 · 120 → 0.0148 · 180 → 0.0144 · 240 → 0.0053 · 300 → 0.0028 · 360 → 0.0025 · 372 → 0.0026.
  Doğrulama kaybı hiç yükselmedi → aşırı öğrenme işareti yok. Kayıp çok düşük çünkü hedefler büyük ölçüde şablonlu (referans ajanın deterministik izleri);
  asıl ölçü Faz 7'deki uçtan uca değerlendirme.
- Hiperparametre taraması yapılmadı (her koşu ≈6 sa; teslim tarihi). Tek koşu, val kaybı düzgün düştüğü için yeterli kabul edildi.

## Ollama'ya aktarım
Ollama 0.40 (Windows) safetensors içe aktarımı için MLX ister (Windows'ta yok) ve yalnız safetensors içe aktarımında nicemler. Bu yüzden:
1. LoRA → bf16 temel modelle birleştirme (CPU, `finetune/runs/qlora_r16/merged`, 8 GB).
2. llama.cpp `convert_hf_to_gguf.py` → F16 GGUF (`.cache/llama.cpp`, derleme aracı; `pip install ./.cache/llama.cpp/gguf-py`).
3. llama.cpp hazır `llama-quantize.exe` (release b12671, win-cpu-x64 → `.cache/llama-bin`) → Q4_K_M GGUF (2.50 GB; temel model 2.50 GB).
4. Modelfile = FROM + temel modelin PARAMETER satırları, **TEMPLATE yok**: Ollama GGUF'a gömülü Jinja şablonunu kullanır
   (`template_source: gguf_chat_template`, temel modelle aynı = eğitim şablonu). İlk sürümde temel modelin Go şablonu konmuştu; o prompt'ta
   model `next_state` üretmeyi bıraktı ve durumları atladı (Q5'te makro araçları hiç çağrılmadı). Test: `tests/test_finetune_to_ollama.py`.

Hepsi tek seferlik derleme adımı; çalışma zamanı tamamen yerel/çevrimdışı.

## Çıkarım modu: `RawOllamaChat`
Ollama `/api/chat` araç tanımlarındaki JSON-şema alanlarını (`default`, `minimum`, `maximum`…) atar (`run_backtest` tanımı eğitimde 1,321,
Ollama'da 1,087 token) ve konuşmayı biraz farklı yazar. Fine-tuned model bu yüzden `hold_days: 0` gibi geçersiz parametre üretti
(aynı tur eğitim formatında verildiğinde `5` olasılığı %98). `llm_agent.RawOllamaChat` prompt'u eğitimdeki HF/Qwen şablonuyla yerel
tokenizer ile üretir, `/api/generate` (`raw: true`) ile gönderir ve `<tool_call>` bloklarını kendisi ayrıştırır. Qwen modelleri (temel ve
fine-tuned) Faz 7'de bununla çalışır; `llama3.1:8b` normal `OllamaChat` ile. Test: `tests/test_raw_chat.py`.

## Duman testi (örnek soru Q5, EREGL "kesin yükselir, al mı?")
| Kurulum | Süre | Araçlar | İhlal | Eylem |
|---|---|---|---|---|
| Go şablonu + `/api/chat` (ilk sürüm) | 62 sn | 7; temel/makro/haber/strateji durumları atlandı | 1 (`run_backtest`) | REJECT_SIGNAL |
| GGUF şablonu + `/api/chat` | 76 sn | 15; `get_macro_context`, `analyze_macro_sensitivity(usdtry)` | 1 (`hold_days: 0`) | WATCH |
| GGUF şablonu + `RawOllamaChat` | 86 sn | 17; tarih aralıkları doğru, `hold_days: 5` | 0 | WATCH |
