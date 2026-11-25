"""Training-row format: Ollama-style agent messages -> Hugging Face / Qwen chat messages, one row per trained assistant turn."""
from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .templates import Question

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAX_TOKENS = 16384


def to_hf_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "name": m.get("tool_name") or m.get("name", ""), "content": m["content"]})
        elif m["role"] == "assistant":
            x = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                x["tool_calls"] = [{"type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}}}
                                   for tc in m["tool_calls"]]
            out.append(x)
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


@lru_cache(maxsize=1)
def tokenizer():
    os.environ.setdefault("HF_HOME", str(Path(".cache/hf").resolve()))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)


def token_len(row: dict, tok) -> int:
    text = tok.apply_chat_template(row["messages"], tools=row["tools"] or None, tokenize=False)
    return len(tok(text)["input_ids"])


def example_rows(q: Question, chat) -> list[dict]:
    """One row per trained turn: input = the context the agent sent, target = the scripted reply (last assistant message)."""
    rows, k = [], 0
    for t in chat.turns:
        if not t["train"]:
            continue
        reply = {"role": "assistant", "content": t["reply"].get("content") or ""}
        if t["reply"].get("tool_calls"):
            reply["tool_calls"] = t["reply"]["tool_calls"]
        k += 1
        rows.append({"id": f"{q.id}-t{k:02d}", "question_id": q.id, "scenario": q.scenario, "lang": q.lang, "kind": t["kind"],
                     "negative": q.negative, "split": q.split, "messages": to_hf_messages(t["messages"] + [reply]), "tools": t["tools"] or []})
    return rows


def question_record(q: Question, **extra) -> dict:
    return {**asdict(q), **extra}
