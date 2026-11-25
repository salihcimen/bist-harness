"""Faz 6 data path: loss mask covers exactly the target message; checkpoint lookup for resume."""
import json
from pathlib import Path

import pytest

from bist.finetune.export import tokenizer
from bist.finetune.train import IGNORE, collate, encode, last_checkpoint

ROW = {"id": "x", "tools": [{"type": "function", "function": {"name": "next_state", "description": "go on", "parameters": {"type": "object", "properties": {}}}}],
       "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "State LOAD_PRICES."},
                    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "next_state", "arguments": {}}}]}]}


@pytest.fixture(scope="module")
def tok():
    return tokenizer()


def test_labels_only_on_target(tok):
    e = encode(ROW, tok)
    assert len(e["input_ids"]) == len(e["labels"])
    target = [t for t, l in zip(e["input_ids"], e["labels"]) if l != IGNORE]
    text = tok.decode(target)
    assert "next_state" in text and "<|im_end|>" in text
    assert "State LOAD_PRICES" not in text and "sys" not in text
    k = next(i for i, l in enumerate(e["labels"]) if l != IGNORE)      # mask is one contiguous prompt prefix
    assert all(l == IGNORE for l in e["labels"][:k]) and all(l != IGNORE for l in e["labels"][k:])


def test_encode_matches_full_template(tok):
    e = encode(ROW, tok)
    full = tok(tok.apply_chat_template(ROW["messages"], tools=ROW["tools"], tokenize=False), add_special_tokens=False)["input_ids"]
    assert tok.decode(e["input_ids"]) == tok.decode(full)


def test_too_long_row_dropped(tok):
    assert encode(ROW, tok, max_len=10) is None


def test_real_rows_encode():
    p = Path("finetune/data/train.jsonl")
    if not p.exists():
        pytest.skip("dataset not generated")
    tok = tokenizer()
    for l in p.read_text(encoding="utf-8").splitlines()[:30]:
        e = encode(json.loads(l), tok)
        assert e and any(x != IGNORE for x in e["labels"])


def test_collate_pads_and_masks():
    b = collate([{"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3]}, {"input_ids": [4], "labels": [4]}], pad_id=0)
    assert b["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert b["labels"].tolist() == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_last_checkpoint(tmp_path):
    assert last_checkpoint(tmp_path) is None
    for n in (20, 100, 40):
        (tmp_path / f"checkpoint-{n}").mkdir()
    (tmp_path / "checkpoint-tmp").mkdir()
    assert last_checkpoint(tmp_path).name == "checkpoint-100"


def test_tail_loss_equals_full_loss():
    """The memory-saving loss (logits only for the target tail) equals the model's own loss on the full sequence."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from bist.finetune.train import tail_loss
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=8)).eval()
    ids = torch.randint(0, 50, (1, 12))
    labels = ids.clone()
    labels[:, :7] = IGNORE
    ref = m(input_ids=ids, labels=labels).loss
    got = tail_loss(m, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels)
    assert torch.allclose(ref, got, atol=1e-5)


def test_trainer_evaluate_runs(tmp_path):
    """Evaluation goes through compute_loss(return_outputs=True); it must not crash (it did at step 60 of the first run)."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments
    from bist.finetune.train import _trainer_cls
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=8))
    data = [{"input_ids": [1, 2, 3, 4, 5], "labels": [IGNORE, IGNORE, IGNORE, 4, 5]}] * 2
    args = TrainingArguments(output_dir=str(tmp_path), per_device_eval_batch_size=1, report_to=[], use_cpu=True,
                             remove_unused_columns=False, prediction_loss_only=True)
    tr = _trainer_cls()(model=m, args=args, eval_dataset=data, data_collator=lambda b: collate(b, 0))
    assert tr.evaluate()["eval_loss"] > 0
