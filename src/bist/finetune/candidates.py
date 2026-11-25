"""Candidate-label supplement: template questions whose reference answer is a *_CANDIDATE action.

Candidate conditions (signal + supporting backtest) are rare on real data, so the main random question set has almost none.
`scan` runs the reference agent over many deterministic `_make` draws (in parallel) and keeps the draws that end in a candidate;
the chosen draw keys are committed in `finetune/candidate_seeds.json`, so `candidate_questions()` rebuilds exactly the same
questions without re-scanning. Train/val draws use no held-out feature; test draws use a held-out ticker/template.

    python -m bist.finetune.candidates --n 1600 --workers 5    # scan + write finetune/candidate_seeds.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..harness import policy
from .templates import Question, _make

SEEDS_FILE = Path("finetune/candidate_seeds.json")
SCENARIOS = ("S1", "S3", "S4", "S5")
TEST_FEATS = (("ticker",), ("template",))


def make(key: dict) -> Question:
    """Rebuild one draw. key = {scenario, lang, feats: [..], i}."""
    feats = tuple(sorted(key["feats"]))
    rng = random.Random(f"cand-{key['scenario']}-{key['lang']}-{'+'.join(feats)}-{key['i']}")
    q = _make(0, key["scenario"], key["lang"], rng, set(feats), None)
    q.split = "test" if feats else "train"
    return q


def _keys(n: int) -> list[dict]:
    out = []
    for i in range(n):
        feats = list(TEST_FEATS[i // 4 % 2]) if i % 5 == 4 else []   # every 5th draw is a held-out (test) draw
        out.append({"scenario": SCENARIOS[i % 4], "lang": "tr" if i // 2 % 2 else "en", "feats": feats, "i": i})
    return out


def _scan_one(key: dict) -> dict | None:
    from ..harness.logdb import DecisionLog
    from ..tools import session
    from .scripted import reference_label
    db = DecisionLog(Path(tempfile.gettempdir()) / f"bist_cand_scan_{os.getpid()}.sqlite")
    before = set(session._SESSIONS)
    try:
        q = make(key)
        h, out = reference_label(q, db)
        return {**key, "action": out["action"]} if out["action"] in policy.CANDIDATES else None
    except Exception:  # noqa: BLE001 - a failing draw is simply not a hit
        return None
    finally:
        for k in set(session._SESSIONS) - before:
            session._SESSIONS.pop(k, None)


def scan(n: int, workers: int = 8) -> list[dict]:
    with ProcessPoolExecutor(workers) as ex:
        return [h for h in ex.map(_scan_one, _keys(n), chunksize=4) if h]


def select(hits: list[dict], per_action: int = 24, test_share: float = 0.2) -> list[dict]:
    """Up to `per_action` draws per candidate action, ~test_share of them held-out draws, alternating scenarios and languages."""
    out = []
    for a in sorted({h["action"] for h in hits}):
        hs = [h for h in hits if h["action"] == a]
        n_test = max(1, round(per_action * test_share))
        test = [h for h in hs if h["feats"]][:n_test]
        train = [h for h in hs if not h["feats"]][:per_action - len(test)]
        out += train + test
    return out


def candidate_questions(seeds: list[dict] | None = None, val_share: float = 0.10) -> list[Question]:
    if seeds is None:
        seeds = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))["seeds"] if SEEDS_FILE.exists() else []
    qs = []
    for j, k in enumerate(seeds, 1):
        q = make(k)
        q.id = f"c{j:04d}"
        qs.append(q)
    train = [q for q in qs if q.split == "train"]
    for q in random.Random("cand-val").sample(train, round(len(train) * val_share)):
        q.split = "val"
    return qs


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=5)   # more processes can exhaust the Windows page file
    ap.add_argument("--per-action", type=int, default=24)
    ap.add_argument("--select-only", action="store_true", help="re-select from the hits stored in the seeds file")
    a = ap.parse_args()
    hits = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))["all_hits"] if a.select_only else scan(a.n, a.workers)
    seeds = select(hits, a.per_action)
    SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEDS_FILE.write_text(json.dumps({"scanned": json.loads(SEEDS_FILE.read_text(encoding="utf-8"))["scanned"] if a.select_only else a.n, "hits": dict(Counter(h["action"] for h in hits)), "seeds": seeds,
                                      "all_hits": hits},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    by = Counter(f"{h['action']}|{'test' if h['feats'] else 'train'}" for h in hits)
    print(json.dumps({"hits": dict(by), "selected": len(seeds)}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
