"""Faz 5 question templates: slots filled, universe tickers, mix, hold-out split, sample questions kept out of train/val."""
import re
from collections import Counter
from pathlib import Path

from bist.finetune.templates import TEMPLATES, TEST_TICKERS, TEST_WINDOW, build_questions, is_test_template
from bist.tools import store

QS = build_questions(300, seed=0)
SAMPLE_COMBOS = {"S1": {"BIMAS", "MGROS", "MAVI"}, "S2": {"THYAO"}, "S3": {"ASELS"}, "S4": {"TUPRS"}, "S5": {"EREGL"}}


def _in_window(d):
    return TEST_WINDOW[0] <= d <= TEST_WINDOW[1]


def test_count_and_ids_unique():
    assert len(QS) == 300 and len({q.id for q in QS}) == 300


def test_no_unfilled_slots():
    assert not [q.text for q in QS if re.search(r"\{\w*\}", q.text)]


def test_tickers_in_universe():
    uni = set(store.tickers())
    assert all(t in uni for q in QS for t in q.tickers)


def test_templates_per_lang():
    assert all(len(TEMPLATES[(s, l)]) >= 10 for s in ("S1", "S2", "S3", "S4", "S5") for l in ("tr", "en"))


def test_mix():
    neg = sum(q.negative is not None for q in QS) / len(QS)
    assert 0.20 <= neg <= 0.25
    tr = sum(q.lang == "tr" for q in QS) / len(QS)
    assert 0.40 <= tr <= 0.60
    test = sum(q.split == "test" for q in QS) / len(QS)
    assert 0.12 <= test <= 0.18
    pos = [q for q in QS if q.negative is None]
    c = Counter(q.scenario for q in pos)
    assert all(abs(c[s] / len(pos) - 0.20) <= 0.05 for s in ("S1", "S2", "S3", "S4", "S5")), c
    assert Counter(q.split for q in QS)["val"] > 0


def test_split_holdout():
    for q in QS:
        held = [bool(set(q.tickers) & set(TEST_TICKERS)), _in_window(q.as_of), is_test_template(q.template_id)]
        if q.split == "test":
            assert any(held), q
        else:
            assert not any(held), q


def test_sample_questions_not_in_train():
    text = Path("docs/sample_test_questions.txt").read_text(encoding="utf-8")
    originals = re.findall(r'"([^"]{40,})"', text)
    assert len(originals) >= 5
    for q in QS:
        assert q.text not in originals
        if q.split != "test" and q.start == "2024-01-01" and q.as_of == "2026-09-10":
            if q.scenario == "S1":
                assert set(q.tickers) != SAMPLE_COMBOS["S1"], q
            else:
                assert q.tickers[0] not in SAMPLE_COMBOS[q.scenario], q


def test_deterministic():
    assert [q.text for q in build_questions(50, 1)] == [q.text for q in build_questions(50, 1)]


def test_end_not_after_as_of():
    assert all(q.end is None or q.end <= q.as_of for q in QS)
    assert all(q.start is None or q.start < (q.end or q.as_of) for q in QS)


def test_negative_kinds_present_and_shaped():
    kinds = Counter(q.negative for q in QS if q.negative)
    assert set(kinds) == {"advice", "beyond_data", "future_period", "short_history", "small_peer_group", "missing_number", "skip_quality"}
    for q in QS:
        if q.negative == "beyond_data":
            assert q.requested_as_of > "2026-11-18" and q.plan_kw["evidence_dates"][0]["date"] == q.requested_as_of
        if q.negative == "future_period":
            assert q.requested_end > q.as_of and q.end == q.as_of
        if q.negative == "short_history":
            assert q.as_of < "2015-07-01"
        if q.negative == "small_peer_group":
            assert q.tickers == ["ASELS"] and q.focus == "sector_catchup" and "peers" not in q.plan_kw


def test_plan_kw_valid_for_reference_plan():
    from bist.harness import reference
    for q in QS[:60]:
        reference.plan(q.tickers[0], q.focus, start=q.start, end=q.end,
                       **{k: v for k, v in q.plan_kw.items()})   # raises on unknown kwargs


def test_exam_frame_is_common():
    frame = [q for q in QS if q.start == "2024-01-01" and q.as_of == "2026-09-10"]
    assert len(frame) / len(QS) >= 0.25


def test_two_period_weekday_questions():
    tp = [q for q in QS if q.template_id.startswith("S2P-")]
    assert len(tp) >= 15
    for q in tp:
        oos = q.plan_kw["pattern_kw"]["oos_start"]
        assert oos.endswith("-01-01") and q.start < oos < q.as_of
        prev_end = f"{int(oos[:4]) - 1}"
        assert prev_end in q.text and oos[:4] in q.text


def test_claim_advice_questions():
    cl = [q for q in QS if q.claim]
    assert len(cl) >= 10
    assert all(q.negative == "advice" and q.scenario == "S5" and q.plan_kw.get("macro_factor") for q in cl)
