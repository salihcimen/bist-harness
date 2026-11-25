"""Faz 5 generator: template questions -> reference trace replayed through LLMAgent -> per-turn SFT rows.

    python -m bist.finetune.generate --n 300 --seed 0 [--out finetune/data] [--check-only] [--no-candidates]

Resumable: each question's rows are cached in <out>/traces/<id>.json (skipped questions too, with the reason).
Writes questions.jsonl, train.jsonl, val.jsonl, test.jsonl and stats.json. `validate_dataset` re-checks the written files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
import traceback
from collections import Counter
from pathlib import Path

from ..harness import policy
from ..harness.logdb import DecisionLog
from ..tools import session, store
from .export import MAX_TOKENS, example_rows, question_record, token_len, tokenizer
from .scripted import replay
from .candidates import candidate_questions
from .templates import TEST_TICKERS, build_questions

DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
SPLITS = ("train", "val", "test")


def _cleanup_sessions(before: set) -> None:
    for k in set(session._SESSIONS) - before:
        session._SESSIONS.pop(k, None)


def _trace(q, db, inject_pick) -> dict:
    before = set(session._SESSIONS)
    try:
        h, chat = replay(q, db, inject=inject_pick)
        rows = example_rows(q, chat)
        tok = tokenizer()
        for r in rows:
            r["n_tokens"] = token_len(r, tok)
        return {"id": q.id, "ok": True, "ref_action": chat.ref_action, "inject": getattr(chat, "inject_kind", None),
                "quality": policy.quality_verdict(h.calls), "rows": rows}
    except Exception as e:  # noqa: BLE001 - one bad question must not stop the run
        return {"id": q.id, "ok": False, "reason": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        _cleanup_sessions(before)


def wants_error(seed: int, qid: str, error_share: float) -> bool:
    """Whether a (train) question gets an injected error: a pure function of seed and id, so resumed and fresh runs agree."""
    return random.Random(f"{seed}-{qid}-err").random() < error_share


def cache_key(q, seed: int, error_share: float) -> str:
    raw = json.dumps([question_record(q), seed, error_share], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main(n: int = 300, seed: int = 0, out_dir: Path = Path("finetune/data"), error_share: float = 0.10, db_path: Path | None = None,
         candidates: bool = False) -> dict:
    out_dir = Path(out_dir)
    cache = out_dir / "traces"
    cache.mkdir(parents=True, exist_ok=True)
    db = DecisionLog(db_path or Path(tempfile.gettempdir()) / "bist_finetune_gen.sqlite")
    qs = build_questions(n, seed) + (candidate_questions() if candidates else [])   # c* ids: candidate-label supplement
    traces = {}
    for q in qs:
        f = cache / f"{q.id}.json"
        key = cache_key(q, seed, error_share)
        if f.exists():   # reuse only a successful trace of exactly this question and settings; failures are retried
            t = json.loads(f.read_text(encoding="utf-8"))
            if t.get("key") == key and t.get("ok"):
                traces[q.id] = t
                continue
        want_err = q.split == "train" and wants_error(seed, q.id, error_share)
        pick = (lambda kinds, r=random.Random(f"{seed}-{q.id}-kind"): r.choice(kinds) if kinds else None) if want_err else None
        t = {**_trace(q, db, pick), "key": key}
        f.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
        traces[q.id] = t
        print(f"{q.id} {q.split} {q.scenario} {q.negative or '-'} -> {t.get('ref_action') or 'SKIPPED ' + t.get('reason', '')}", flush=True)
    ok = [q for q in qs if traces[q.id]["ok"]]
    _write_jsonl(out_dir / "questions.jsonl", [question_record(q, ref_action=traces[q.id].get("ref_action"), ok=traces[q.id]["ok"],
                                                               inject=traces[q.id].get("inject")) for q in qs])
    by_split = {s: [r for q in ok if q.split == s for r in traces[q.id]["rows"]] for s in SPLITS}
    for s in SPLITS:
        _write_jsonl(out_dir / f"{s}.jsonl", by_split[s])
    rows = [r for s in SPLITS for r in by_split[s]]
    toks = sorted(r["n_tokens"] for r in rows) or [0]
    stats = {"questions": len(qs), "questions_ok": len(ok), "rows": len(rows),
             "skipped": [{"id": traces[q.id]["id"], "reason": traces[q.id]["reason"]} for q in qs if not traces[q.id]["ok"]],
             "rows_by_split": {s: len(by_split[s]) for s in SPLITS},
             "questions_by_split": dict(Counter(q.split for q in ok)), "by_scenario": dict(Counter(q.scenario for q in ok)),
             "by_lang": dict(Counter(q.lang for q in ok)), "by_negative": dict(Counter(q.negative or "none" for q in ok)),
             "candidate_questions": sum(q.id.startswith("c") for q in ok),
             "by_ref_action": dict(Counter(traces[q.id]["ref_action"] for q in ok)),
             "by_inject": dict(Counter(traces[q.id].get("inject") or "none" for q in ok)),
             "rows_by_kind": dict(Counter(r["kind"] for r in rows)),
             "tokens_max": toks[-1], "tokens_p95": toks[int(0.95 * (len(toks) - 1))], "tokens_mean": round(sum(toks) / len(toks))}
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def validate_dataset(out_dir: Path, full_mix: bool = True) -> list[str]:
    """[] if every check passes. full_mix=False skips the distribution checks (small test runs)."""
    out_dir = Path(out_dir)
    qs = {q["id"]: q for q in _read(out_dir / "questions.jsonl")}
    last = str(store.data_last_date().date())
    sample_txt = (Path("docs") / "sample_test_questions.txt").read_text(encoding="utf-8")
    originals = re.findall(r'"([^"]{40,})"', sample_txt)
    probs: list[str] = []
    for s in SPLITS:
        for r in _read(out_dir / f"{s}.jsonl"):
            q = qs.get(r["question_id"])
            if q is None:
                probs.append(f"{r['id']}: unknown question")
                continue
            if r["split"] != s or q["split"] != s:
                probs.append(f"{r['id']}: split mismatch")
            if r["messages"][-1]["role"] != "assistant":
                probs.append(f"{r['id']}: last message is not the assistant target")
            if r.get("n_tokens", 0) > MAX_TOKENS:
                probs.append(f"{r['id']}: {r['n_tokens']} tokens > {MAX_TOKENS}")
            if r["kind"] == "final":
                try:
                    a = json.loads(r["messages"][-1]["content"])["action"]
                    if a not in policy.ACTIONS:
                        probs.append(f"{r['id']}: action {a} not allowed")
                except Exception:
                    probs.append(f"{r['id']}: final target is not JSON")
            as_of = min(q["as_of"], last)
            allowed = set(DATE.findall(q["text"])) | {d for m in r["messages"] if m["role"] == "user" for d in DATE.findall(m["content"])}
            for m in r["messages"]:
                if m["role"] == "tool":
                    late = {d for d in DATE.findall(m["content"]) if d > as_of} - allowed
                    if late:
                        probs.append(f"{r['id']}: tool output dated after analysis date {as_of}: {sorted(late)}")
    for q in qs.values():
        if q["split"] != "test" and q.get("ok", True):
            if set(q["tickers"]) & set(TEST_TICKERS) or any(t in q["text"] for t in TEST_TICKERS):
                probs.append(f"{q['id']}: test ticker in {q['split']}")
            if q["text"] in originals:
                probs.append(f"{q['id']}: original sample question in {q['split']}")
    if full_mix:
        ok = [q for q in qs.values() if q.get("ok", True) and not q["id"].startswith("c")]   # the targeted supplement is outside the mix
        neg = sum(q["negative"] is not None for q in ok) / max(1, len(ok))
        if not 0.20 <= neg <= 0.25:
            probs.append(f"negative share {neg:.3f} outside 0.20-0.25")
        pos = [q for q in ok if q["negative"] is None]
        c = Counter(q["scenario"] for q in pos)
        for s in ("S1", "S2", "S3", "S4", "S5"):
            if abs(c[s] / max(1, len(pos)) - 0.2) > 0.05:
                probs.append(f"scenario {s} share {c[s] / max(1, len(pos)):.3f} outside 0.15-0.25")
    return probs


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("finetune/data"))
    ap.add_argument("--error-share", type=float, default=0.10)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-candidates", action="store_true", help="skip the candidate-label supplement (finetune/candidate_seeds.json)")
    a = ap.parse_args()
    if not a.check_only:
        st = main(a.n, a.seed, a.out, a.error_share, candidates=not a.no_candidates)
        print(json.dumps({k: v for k, v in st.items() if k != "skipped"}, ensure_ascii=False), f"skipped={len(st['skipped'])}")
    p = validate_dataset(a.out)
    print("OK" if not p else "PROBLEMS:\n" + "\n".join(p[:50]))


if __name__ == "__main__":
    _cli()
