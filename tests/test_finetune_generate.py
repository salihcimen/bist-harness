"""Faz 5 generator: HF/Qwen message format, per-turn rows, resumable cache, skipped questions, dataset checks."""
import json

import pytest

from bist.finetune import generate
from bist.finetune.export import to_hf_messages, token_len, tokenizer


def _rows(d, name):
    p = d / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def test_to_hf_messages_roundtrip():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_index_history", "arguments": {"max_rows": 5}}}]},
            {"role": "tool", "tool_name": "get_index_history", "content": "get_index_history: OK"}]
    hf = to_hf_messages(msgs)
    assert hf[2]["tool_calls"] == [{"type": "function", "function": {"name": "get_index_history", "arguments": {"max_rows": 5}}}]
    assert hf[3] == {"role": "tool", "name": "get_index_history", "content": "get_index_history: OK"}
    row = {"messages": hf + [{"role": "assistant", "content": "{}"}], "tools": []}
    assert token_len(row, tokenizer()) > 10


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("ft")
    stats = generate.main(n=6, seed=3, out_dir=d, error_share=0.5)
    return d, stats


def test_small_run(small):
    d, stats = small
    rows = _rows(d, "train") + _rows(d, "val") + _rows(d, "test")
    assert rows and stats["rows"] == len(rows)
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows)
    assert {"select", "final"} <= {r["kind"] for r in rows}
    assert all(r["n_tokens"] <= 16384 for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    finals = [r for r in rows if r["kind"] == "final"]
    assert all(json.loads(r["messages"][-1]["content"])["action"] for r in finals)
    assert generate.validate_dataset(d, full_mix=False) == []


def test_resume_uses_cache(small, monkeypatch):
    d, stats = small
    monkeypatch.setattr(generate, "replay", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replay called")))
    assert generate.main(n=6, seed=3, out_dir=d, error_share=0.5)["rows"] == stats["rows"]


def test_failing_question_is_skipped(tmp_path, monkeypatch):
    real = generate.replay

    def flaky(q, db, **kw):
        if q.id == "q0002":
            raise RuntimeError("boom")
        return real(q, db, **kw)

    monkeypatch.setattr(generate, "replay", flaky)
    stats = generate.main(n=3, seed=5, out_dir=tmp_path, error_share=0.0)
    assert [s["id"] for s in stats["skipped"]] == ["q0002"] and "boom" in stats["skipped"][0]["reason"]
    assert stats["questions_ok"] == 2


def test_validate_flags_future_date(small, tmp_path):
    d, _ = small
    bad = tmp_path / "bad"
    bad.mkdir()
    for f in ("questions.jsonl", "train.jsonl", "val.jsonl", "test.jsonl"):
        if (d / f).exists():
            (bad / f).write_text((d / f).read_text(encoding="utf-8"), encoding="utf-8")
    rows = _rows(bad, "train") or _rows(bad, "test")
    name = "train" if _rows(bad, "train") else "test"
    r = rows[0]
    r["messages"].insert(-1, {"role": "tool", "name": "x", "content": "x: OK last bar 2099-01-01"})
    (bad / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    assert any("2099-01-01" in p for p in generate.validate_dataset(bad, full_mix=False))


def test_cache_miss_when_question_changes(tmp_path, monkeypatch):
    generate.main(n=2, seed=3, out_dir=tmp_path, error_share=0.0)
    seen = []
    real = generate.replay
    monkeypatch.setattr(generate, "replay", lambda q, db, **kw: (seen.append(q.id), real(q, db, **kw))[1])
    generate.main(n=2, seed=4, out_dir=tmp_path, error_share=0.0)     # different seed -> different questions under the same ids
    assert seen == ["q0001", "q0002"]
    qs = [json.loads(l) for l in (tmp_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = _rows(tmp_path, "train") + _rows(tmp_path, "val") + _rows(tmp_path, "test")
    for q in qs:
        r = next(r for r in rows if r["question_id"] == q["id"])
        assert q["text"] in r["messages"][1]["content"]


def test_skipped_question_is_retried(tmp_path, monkeypatch):
    real = generate.replay
    monkeypatch.setattr(generate, "replay", lambda q, db, **kw: (_ for _ in ()).throw(RuntimeError("transient")) if q.id == "q0001" else real(q, db, **kw))
    assert generate.main(n=1, seed=5, out_dir=tmp_path, error_share=0.0)["questions_ok"] == 0
    monkeypatch.setattr(generate, "replay", real)
    assert generate.main(n=1, seed=5, out_dir=tmp_path, error_share=0.0)["questions_ok"] == 1


def test_error_choice_does_not_depend_on_cache():
    a = [generate.wants_error(0, f"q{i:04d}", 0.3) for i in range(1, 200)]
    assert a == [generate.wants_error(0, f"q{i:04d}", 0.3) for i in range(1, 200)]
    assert 0.2 < sum(a) / len(a) < 0.4
    assert a != [generate.wants_error(1, f"q{i:04d}", 0.3) for i in range(1, 200)]
