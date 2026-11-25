"""Faz 6: QLoRA fine-tuning of Qwen/Qwen3-4B-Instruct-2507 on the Faz 5 per-turn SFT rows, with resumable checkpoints.

    python -m bist.finetune.train --smoke                  # 20 steps: speed + memory check, throwaway output
    python -m bist.finetune.train                          # full run -> finetune/runs/<name>/ ; Ctrl+C any time
    python -m bist.finetune.train --resume                 # continue from the last checkpoint of the same run

Loss only on the target (last assistant message): prompt tokens are masked with -100. Checkpoints keep LoRA weights, optimizer,
scheduler, RNG and data position, so a stopped run resumes exactly where it was (`--resume`).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .export import MODEL_ID, tokenizer

DATA = Path("finetune/data")
RUNS = Path("finetune/runs")
IGNORE = -100


# ------------------------------------------------------------------ data
def encode(row: dict, tok, max_len: int = 8192) -> dict | None:
    """input_ids/labels for one row; labels = -100 on everything except the last assistant message. None if too long."""
    msgs, tools = row["messages"], row["tools"] or None
    prompt = tok.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError(f"{row.get('id')}: chat template prompt is not a prefix of the full text")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = p_ids + t_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [IGNORE] * len(p_ids) + t_ids}


def load_split(split: str, tok, max_len: int = 8192, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    out = [e for e in (encode(r, tok, max_len) for r in rows) if e]
    return out


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def last_checkpoint(run_dir: Path) -> Path | None:
    cks = [p for p in Path(run_dir).glob("checkpoint-*") if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    return max(cks, key=lambda p: int(p.name.split("-")[1])) if cks else None


# ------------------------------------------------------------------ model + training
def tail_loss(model, inputs: dict, labels, num_items: int | None = None):
    """Causal-LM loss computed from logits of the target tail only (`logits_to_keep`): the target is the last message, so the
    full [seq x 151k vocab] logit matrix (several GB for a 5k-token prompt) is never built. Equal to the model's own loss."""
    import torch
    import torch.nn.functional as F
    first = (labels != IGNORE).int().argmax(dim=1).min().item()
    keep = labels.shape[1] - first + 1          # +1: the position before the first target token predicts it
    logits = model(**inputs, logits_to_keep=keep).logits[:, :-1].float()
    tgt = labels[:, labels.shape[1] - keep + 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1).to(logits.device), ignore_index=IGNORE, reduction="sum")
    n = num_items if num_items is not None else (tgt != IGNORE).sum()
    return loss / (n.to(loss.device) if torch.is_tensor(n) else n)


def _trainer_cls():
    from transformers import Trainer

    class TailLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            labels = inputs.pop("labels")
            loss = tail_loss(model, inputs, labels, num_items_in_batch)
            return (loss, {"loss": loss}) if return_outputs else loss   # evaluation reads outputs as a dict

    return TailLossTrainer


def build_model(r: int, alpha: int, dropout: float, upcast: bool = False):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
                                                 local_files_only=True, attn_implementation="sdpa")
    if upcast:   # peft's helper casts every non-quantized weight to fp32 (more memory and slower on an 8 GB card)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, cfg)


def main(a) -> None:
    os.environ.setdefault("HF_HOME", str(Path(".cache/hf").resolve()))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # fully local: no Hub calls when saving/loading
    from transformers import TrainingArguments, set_seed
    set_seed(a.seed)
    tok = tokenizer()
    run_dir = RUNS / ("smoke" if a.smoke else a.name)
    train = load_split("train", tok, a.max_len, limit=a.smoke_rows if a.smoke else None)
    val = load_split("val", tok, a.max_len, limit=40 if a.smoke else a.val_rows)
    print(f"train {len(train)} rows, val {len(val)} rows -> {run_dir}", flush=True)
    model = build_model(a.r, a.alpha, a.dropout)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(run_dir), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=a.accum,
        num_train_epochs=a.epochs, max_steps=a.smoke_steps if a.smoke else -1, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, weight_decay=0.0, max_grad_norm=1.0, bf16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if a.smoke else 5, eval_strategy="steps", eval_steps=a.smoke_steps if a.smoke else a.eval_steps,
        save_strategy="steps", save_steps=a.smoke_steps if a.smoke else a.save_steps, save_total_limit=3,
        load_best_model_at_end=False, report_to=[], seed=a.seed, data_seed=a.seed, dataloader_num_workers=0,
        remove_unused_columns=False, prediction_loss_only=True)
    trainer = _trainer_cls()(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    resume = last_checkpoint(run_dir) if a.resume else None
    if a.resume and resume is None:
        print("no checkpoint found, starting fresh", flush=True)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    final = run_dir / "final_adapter"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    import torch
    stats = {"train_rows": len(train), "val_rows": len(val), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
             "log": trainer.state.log_history}
    (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "log"}), flush=True)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="qlora_r16")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--smoke-rows", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--val-rows", type=int, default=80)
    ap.add_argument("--eval-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())


if __name__ == "__main__":
    _cli()
