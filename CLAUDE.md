# BIST 100 Agentic Harness (CSE-481) — proje notları
- Kök: C:\Projects\econ\bist-harness (repo: github.com/erhanderya/bist-harness). Kaynak PDF/duyuru docs/ altında.
- Yol haritası: ROADMAP.md (durum kutularını güncel tut). Spec düz metin: docs/project_spec.txt
- Python: .venv\Scripts\python.exe (Python 3.11.9). pip cache: .cache/. Torch 2.14.1+cu126.
- Donanım: RTX 4060 Laptop 8GB. Ollama: `qwen3:4b-instruct-2507-q4_K_M` (`llm_agent.OllamaChat` varsayılanı) + `llama3.1:8b` (büyük referans).
- Fine-tune modeli: **Qwen/Qwen3-4B-Instruct-2507** (QLoRA 4-bit; HF cache `.cache/hf`, `HF_HOME=.cache/hf` ayarla). Qwen3.5-4B reddedildi (QLoRA önerilmiyor, Ollama'da çalışmıyor) — gerekçe ROADMAP Faz 6. Ollama sunucusu kapalıysa `ollama list` takılır; `curl http://127.0.0.1:11434/api/tags` ile kontrol et. Final sistem tamamen yerel/çevrimdışı; bulut LLM API YOK.
- Evren: data/universe.csv (30 hisse; sütunlar ticker, yahoo_symbol, sector, peer_group). `peer_group` PDF sektörlerinden türetilmiş 11 kaba grup (ASELS tek başına → "peer group too small" uyarısı).
- Veri (`data/` gitignore'da, repoda yok): yeniden kurulum sırası `fetch_prices` → `clean_prices` (data/corrections.csv: CCOLA.IS 2024-08-01 katsayı 11) → `fetch_macro` → `fetch_evds` (TCMB; `.env` EVDS_API_KEY) → `fetch_disclosure_times` → `fetch_fundamentals` → `fundamental_metrics` → `fundamental_calendar`. DİKKAT: `fetch_disclosure_times` başarısız pencereyi yamarken `--merge` verilmezse parquet'in üzerine yazar. `kap_harvest_official`/`validate_fundamentals` MKK anahtarı ister, araçlar kullanmaz.
- Kural: araç şemaları (mcp/tool_schemas.json) Faz 2'de dondurulmadan eğitim verisi üretme.
- Windows/Git Bash: dosya yolları /c/... ; Python'u `-I` ile değil venv ile çalıştır.
- Çalıştırma: `cd /c/Projects/econ/bist-harness && PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m <modül>`; test: `... -m pytest tests -q` (218 test; örnek soru kabulü: tests/test_sample_questions.py). Uzun Python dosyalarını bash heredoc yerine Write ile yaz.
- Araçlar: src/bist/tools/ (envelope.py sözleşme; her araç as_of alır, sonrasını görmez; hatalar status=ERROR zarfı). Şema: mcp/tool_schemas.json DONDURULDU (v1.1, 19 araç; ROADMAP Faz 2b). Ortak tarih aralığı/ileri pencere kuralları: tools/common.py. Hocanın 5 örnek sorusu: docs/sample_test_questions.txt.

## Harness (Faz 4)
- Paket: `src/bist/harness/` — `states.py` (17 durum + durum→araç izinleri), `machine.py` (Harness), `policy.py` (risk gate, referans karar, doğrulayıcı, açıklama şablonu), `logdb.py` (SQLite), `reference.py` (deterministik referans ajan = etiket kaynağı), `replay.py`, `llm_agent.py` (Ollama), `mcp_server.py`, `survey.py`.
- Referans çalıştırma: `reference.run(soru, "THYAO", focus, human=("APPROVE","neden"))` (v1.1 ek argümanlar: `start`, `end`, `peers`, `pattern_kw`, `reversal_kw`, `macro_factor`, örnekler `tests/test_harness_v11.py::SAMPLE_RUNS`); MCP: `python -m bist.harness.mcp_server`.
- Karar günlüğü yolu `BIST_DECISION_DB` ile değiştirilebilir. Testler: tests/test_harness.py, test_llm_agent.py, test_mcp_server.py.
- Kural: LLM çıktısı `policy.validate_output`'tan geçmeden kabul edilmez; hiçbir araç emir üretmez.

## Eğitim verisi (Faz 5)
- Paket: `src/bist/finetune/` — `templates.py` (şablon sorular, bölme), `scripted.py` (ScriptedChat: referans izi LLMAgent içinde), `export.py` (Qwen/HF format, token), `generate.py`, `candidates.py` (aday etiketi eki; tohumlar `finetune/candidate_seeds.json`, repoda).
- Üretim: `HF_HOME=.cache/hf ... -m bist.finetune.generate --n 300 --seed 0` (`finetune/data/` gitignore'da; önbellek `traces/`); `--check-only` kontroller. Doküman: docs/finetune_data.md.

## İnce ayar (Faz 6)
- `src/bist/finetune/train.py` (QLoRA, `--resume`), `to_ollama.py` (merge → GGUF → Q4_K_M → `ollama create bist-analyst`). Doküman: docs/finetune_train.md.
- Ollama/Windows safetensors içe aktaramaz (MLX ister) → llama.cpp: `.cache/llama.cpp` (convert + `gguf-py`), `.cache/llama-bin/llama-quantize.exe`. Modelfile temel Ollama modelinin `ollama show --modelfile` çıktısından.
- Çıktılar `finetune/runs/` (gitignore). Yerel model: `bist-analyst`. Qwen modelleri (temel + fine-tuned) `RawOllamaChat` ile çalıştır (eğitim şablonu, tam araç şeması); `/api/chat` şemayı kırpar. Modelfile'a TEMPLATE koyma (GGUF şablonu kullanılmalı).
