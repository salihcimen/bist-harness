"""Faz 6 export: merge the LoRA adapter into the bf16 base model, convert it to GGUF and register it in Ollama (Q4_K_M).

    python -m bist.finetune.to_ollama --run qlora_r16 --name bist-analyst

Ollama on Windows cannot import safetensors (its safetensors importer needs the MLX runtime), so the merged model goes through
llama.cpp's `convert_hf_to_gguf.py` (build tool, cloned once to `.cache/llama.cpp`) to an F16 GGUF; llama.cpp's prebuilt `llama-quantize`
(`.cache/llama-bin`) turns it into Q4_K_M, the same quantization type as the base model `qwen3:4b-instruct-2507-q4_K_M`
(Ollama only quantizes safetensors imports), and `ollama create` imports that file. The Modelfile takes
the base model's parameters but no TEMPLATE, so Ollama uses the chat template embedded in the GGUF, exactly like the base model:
prompts and tool calls are rendered as in training and as for the base model in Faz 7. Everything runs
locally (Ollama on 127.0.0.1).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .export import MODEL_ID

RUNS = Path("finetune/runs")
BASE_OLLAMA = "qwen3:4b-instruct-2507-q4_K_M"
CONVERT = Path(".cache/llama.cpp/convert_hf_to_gguf.py")
QUANTIZE = Path(".cache/llama-bin/llama-quantize.exe")   # llama.cpp release b12671, win-cpu-x64


def merge(adapter: Path, out: Path) -> Path:
    os.environ.setdefault("HF_HOME", str(Path(".cache/hf").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True).save_pretrained(str(out))
    return out


def to_gguf(merged: Path, out: Path) -> Path:
    if not CONVERT.exists():
        raise SystemExit(f"{CONVERT} missing: git clone --depth 1 https://github.com/ggml-org/llama.cpp .cache/llama.cpp "
                         "&& pip install ./.cache/llama.cpp/gguf-py")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    subprocess.run([sys.executable, str(CONVERT), str(merged), "--outfile", str(out), "--outtype", "f16"], check=True, env=env)
    return out


def quantize(src: Path, out: Path, quant: str) -> Path:
    if not QUANTIZE.exists():
        raise SystemExit(f"{QUANTIZE} missing: unzip llama.cpp's llama-<build>-bin-win-cpu-x64.zip release into .cache/llama-bin")
    subprocess.run([str(QUANTIZE), str(src), str(out), quant.upper()], check=True)
    return out


def base_modelfile(base: str = BASE_OLLAMA) -> str:
    return subprocess.run(["ollama", "show", "--modelfile", base], check=True, capture_output=True, text=True, encoding="utf-8").stdout


def modelfile(src: Path, base_text: str) -> str:
    """FROM `src` + the base model's PARAMETER lines (stop tokens, sampling), and NO TEMPLATE: the base Ollama model runs on the
    Jinja chat template embedded in its GGUF (`template_source: gguf_chat_template`), and so does ours (convert_hf_to_gguf embeds
    the HF template = the training format). A TEMPLATE line would switch Ollama to its Go template, which renders tool calls and
    tool results differently from training; with it the fine-tuned model stopped emitting `next_state` and skipped states."""
    params = [l for l in base_text.splitlines() if l.startswith("PARAMETER ")]
    return "\n".join([f"FROM {src.resolve()}", *params]) + "\n"


def main(run: str, name: str, quant: str = "q4_K_M") -> None:
    rd = RUNS / run
    merged = rd / "merged"
    if not (merged / "config.json").exists():
        print(f"merging {rd / 'final_adapter'} -> {merged}", flush=True)
        merge(rd / "final_adapter", merged)
    gguf = rd / "model-f16.gguf"
    if not gguf.exists():
        print(f"converting {merged} -> {gguf}", flush=True)
        to_gguf(merged, gguf)
    qgguf = rd / f"model-{quant.lower()}.gguf"
    if not qgguf.exists():
        print(f"quantizing {gguf} -> {qgguf}", flush=True)
        quantize(gguf, qgguf, quant)
    mf = rd / "Modelfile"
    mf.write_text(modelfile(qgguf, base_modelfile()), encoding="utf-8")
    subprocess.run(["ollama", "create", name, "-f", str(mf)], check=True)
    print(f"created ollama model {name}", flush=True)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="qlora_r16")
    ap.add_argument("--name", default="bist-analyst")
    ap.add_argument("--quant", default="q4_K_M")
    a = ap.parse_args()
    main(a.run, a.name, a.quant)


if __name__ == "__main__":
    _cli()
