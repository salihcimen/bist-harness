"""Candidate-label supplement: deterministic draws, held-out rules, committed seeds really end in a candidate action."""
import json
import tempfile
from pathlib import Path

import pytest

from bist.finetune import candidates, generate
from bist.finetune.candidates import SEEDS_FILE, candidate_questions, make, select
from bist.finetune.templates import SAMPLE_AS_OF, SAMPLE_TICKERS, TEST_TICKERS, TEST_WINDOW, is_test_template
from bist.harness import policy

SEEDS = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))["seeds"] if SEEDS_FILE.exists() else []
needs_seeds = pytest.mark.skipif(not SEEDS, reason="finetune/candidate_seeds.json not generated")


def _held(q):
    return [bool(set(q.tickers) & set(TEST_TICKERS)), TEST_WINDOW[0] <= q.as_of <= TEST_WINDOW[1], is_test_template(q.template_id)]


def test_make_is_deterministic():
    k = {"scenario": "S3", "lang": "tr", "feats": [], "i": 7}
    assert make(k).text == make(k).text


def test_keys_cover_scenarios_langs_and_test_draws():
    ks = candidates._keys(40)
    assert {k["scenario"] for k in ks} == set(candidates.SCENARIOS)
    assert {k["lang"] for k in ks} == {"tr", "en"}
    assert 0 < sum(bool(k["feats"]) for k in ks) < 20


def test_train_draws_not_held_out_and_test_draws_are():
    for k in candidates._keys(40):
        q = make(k)
        assert any(_held(q)) == bool(k["feats"]), (k, q)
        assert q.split == ("test" if k["feats"] else "train")


def test_select_balances_actions_and_keeps_test_share():
    hits = [{"action": a, "feats": f, "i": i} for i in range(30) for a, f in
            [("FUNDAMENTAL_IMPROVEMENT_CANDIDATE", []), ("TECHNICAL_REVERSAL_CANDIDATE", ["ticker"] if i % 3 == 0 else [])]]
    s = select(hits, per_action=10)
    by = {a: [h for h in s if h["action"] == a] for a in {h["action"] for h in s}}
    assert len(by["TECHNICAL_REVERSAL_CANDIDATE"]) == 10 and sum(bool(h["feats"]) for h in by["TECHNICAL_REVERSAL_CANDIDATE"]) == 2
    assert len(by["FUNDAMENTAL_IMPROVEMENT_CANDIDATE"]) == 10   # no test draws available -> all train


@needs_seeds
def test_candidate_questions_ids_splits_and_no_sample_clash():
    qs = candidate_questions()
    assert len(qs) == len(SEEDS) and len({q.id for q in qs}) == len(qs) and all(q.id.startswith("c") for q in qs)
    assert {q.split for q in qs} >= {"train", "test"}
    for q in qs:
        assert any(_held(q)) == (q.split == "test")
        if q.start == "2024-01-01" and q.as_of == SAMPLE_AS_OF and q.split != "test":
            assert (set(q.tickers) != SAMPLE_TICKERS["S1"]) if q.scenario == "S1" else q.tickers[0] not in SAMPLE_TICKERS[q.scenario]


@needs_seeds
def test_seeds_cover_several_candidate_actions():
    assert len({s["action"] for s in SEEDS}) >= 2 and len(SEEDS) >= 20


@needs_seeds
def test_committed_seeds_still_give_candidate_reference_action():
    for k in SEEDS[::max(1, len(SEEDS) // 4)][:4]:     # a few, the reference run is slow
        assert candidates._scan_one(k) is not None, k


def test_generate_mix_check_ignores_candidate_supplement(tmp_path):
    rec = lambda i, scen, neg: {"id": i, "scenario": scen, "negative": neg, "split": "train", "tickers": ["THYAO"], "text": "x", "ok": True}
    # main set: 40 balanced positives + 11 negatives (0.216); the 30 extra S3 candidates would break both shares if counted
    qs = [rec(f"q{j:04d}", ["S1", "S2", "S3", "S4", "S5"][j % 5], None) for j in range(40)]
    qs += [rec(f"q{j:04d}", "S5", "advice") for j in range(40, 51)]
    qs += [rec(f"c{j:04d}", "S3", None) for j in range(30)]
    (tmp_path / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs), encoding="utf-8")
    for s in generate.SPLITS:
        (tmp_path / f"{s}.jsonl").write_text("", encoding="utf-8")
    probs = generate.validate_dataset(tmp_path)
    assert not [p for p in probs if "share" in p], probs
