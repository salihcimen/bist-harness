"""ScriptedChat: the reference trace replayed inside the real LLMAgent loop; recorded turns are the training examples."""
import pytest

from bist.finetune.scripted import ADVICE_PREFIX, ERROR_KINDS, applicable_errors, reference_label, replay
from bist.finetune.templates import Question
from bist.harness import policy
from bist.harness.logdb import DecisionLog

AS_OF = "2026-09-10"


@pytest.fixture()
def db(tmp_path):
    return DecisionLog(tmp_path / "d.sqlite")


def Q(scenario="S2", tickers=("THYAO",), focus="calendar_pattern", plan_kw=None, negative=None, as_of=AS_OF, start="2024-06-03", end=AS_OF,
      requested_end=None, text="Does THYAO show a Thursday effect?"):
    return Question(id="qT", scenario=scenario, lang="en", template_id=f"{scenario}-en-00", text=text, tickers=list(tickers), focus=focus,
                    as_of=as_of, requested_as_of=as_of, start=start, end=end, requested_end=requested_end,
                    plan_kw=plan_kw if plan_kw is not None else {"pattern": "weekday_return", "weekday": "THU"}, negative=negative)


S2 = Q()
S1 = Q("S1", ("BIMAS", "MGROS", "MAVI"), "sector_catchup", {"peers": ["MGROS", "MAVI"], "horizon_days": 20}, text="Which of BIMAS, MGROS, MAVI lagged?")
S3 = Q("S3", ("ASELS",), "technical_reversal", {"reversal_kw": {"event_types": ["bollinger_lower_touch"], "cooldown_days": 0}}, text="ASELS lower band touches?")
S5 = Q("S5", ("EREGL",), "general", {"macro_factor": "usdtry"}, text="EREGL vs USD/TRY?")


def _names(reply):
    return [tc["function"]["name"] for tc in reply.get("tool_calls") or []]


def test_replay_matches_reference(db):
    h, chat = replay(S2, db)
    assert h.final["action"] == reference_label(S2, db)[1]["action"]
    assert not h.final.get("fallback") and h.violations == []


def test_turn_inputs_are_agent_context(db):
    h, chat = replay(S2, db)
    assert all(t["messages"][0]["role"] == "system" and t["messages"][-1]["role"] in ("user", "tool") for t in chat.turns)
    assert chat.turns[0]["kind"] == "select" and chat.turns[-1]["kind"] == "final"
    assert "GENERATE_EXPLANATION" in chat.turns[-1]["messages"][-1]["content"]
    assert all(t["train"] for t in chat.turns)
    calls = [t for t in chat.turns if t["kind"] == "tool_call"]
    assert calls and all(_names(t["reply"])[-1] == "next_state" for t in calls)   # calls + next_state in one turn
    pat = next(t for t in calls if "test_weekday_pattern" in _names(t["reply"]))
    args = next(tc["function"]["arguments"] for tc in pat["reply"]["tool_calls"] if tc["function"]["name"] == "test_weekday_pattern")
    assert args["start"] == "2024-06-03" and args["end"] == AS_OF


def test_s1_two_turns(db):
    h, chat = replay(S1, db)
    i = next(k for k, t in enumerate(chat.turns) if "rank_sector_relative_strength" in _names(t["reply"]))
    assert "next_state" not in _names(chat.turns[i]["reply"])
    nxt = chat.turns[i + 1]["reply"]["tool_calls"]
    catch = next(tc for tc in nxt if tc["function"]["name"] == "test_catch_up_history")
    assert catch["function"]["arguments"]["ticker"] == h.tickers[0]
    sel = chat.turns[0]["reply"]["tool_calls"][0]["function"]["arguments"]
    assert sel["tickers"] == ["BIMAS", "MGROS", "MAVI"]


@pytest.mark.parametrize("kind", sorted(ERROR_KINDS))
def test_each_error_kind_errors_then_recovers(db, kind):
    q = {"max_rows": S2, "bad_condition": S3, "bad_event": S3, "unknown_peer": S1, "missing_weekday": S2, "bad_factor": S5}[kind]
    assert kind in applicable_errors(reference_label(q, db)[0].history)
    h, chat = replay(q, db, inject=kind)
    bad = [k for k, t in enumerate(chat.turns) if not t["train"]]
    assert len(bad) == 1
    k = bad[0]
    assert chat.turns[k + 1]["kind"] == "recovery" and chat.turns[k + 1]["train"]
    tool = _names(chat.turns[k]["reply"])[0]
    assert chat.turns[k + 1]["messages"][-1]["content"].startswith(f"{tool}: ERROR")
    assert h.final["action"] == reference_label(q, db)[1]["action"] and not h.final.get("fallback")


def test_advice_prefix_or_fallback(db):
    q = Q("S5", ("EREGL",), "general", {"macro_factor": "usdtry"}, negative="advice", text="Should I buy EREGL tomorrow? Give me a price target.")
    h, chat = replay(q, db)
    ref_h, final = reference_label(q, db)
    ok = policy.validate_output(final["action"], ADVICE_PREFIX + final["explanation"], ref_h.decision["permitted"], ref_h.evidence, q.text,
                                extra=[ref_h.as_of, str(ref_h.gate["reasons"])])["ok"]
    assert ok and h.final["explanation"].startswith(ADVICE_PREFIX)
    assert not h.final.get("fallback")


def test_beyond_data_unsafe(db):
    q = Q("S5", ("THYAO",), "general", {"macro_factor": "usdtry", "evidence_dates": [{"source": "requested analysis date", "date": "2027-03-01"}]},
          negative="beyond_data", as_of="2027-03-01", start="2025-03-03", end=None, text="As of 2027-03-01, THYAO vs USD/TRY?")
    h, chat = replay(q, db)
    assert h.final["action"] == "ANALYSIS_UNSAFE"
    sel = chat.turns[0]["reply"]["tool_calls"][0]["function"]["arguments"]
    assert sel["analysis_date"] == "2027-03-01"


def test_future_period_is_clipped(db):
    q = Q(requested_end="2026-11-30", negative="future_period", text="THYAO Thursday effect until 2026-11-30?")
    h, chat = replay(q, db)
    sel = chat.turns[0]["reply"]["tool_calls"][0]["function"]["arguments"]
    assert sel["period_end"] == "2026-11-30"
    assert "data period 2024-06-03 to 2026-09-10" in chat.turns[1]["messages"][-1]["content"] or any(
        "2026-09-10 (no data after it)" in m["content"] for m in chat.turns[1]["messages"] if m["role"] == "user")


def test_advice_prefix_passes_validator_wording():
    assert not policy._ADVICE.search(policy._DISCLAIM_STRIP.sub(" ", ADVICE_PREFIX)) and not policy._numbers(ADVICE_PREFIX)


def test_required_tool_error_in_reference_is_not_a_mismatch(db):
    # early 2015: run_backtest errors (too little data) for the reference too; the harness re-runs it and flags it -> not a model failure
    q = Q(plan_kw={"pattern": "weekday_sequence", "pattern_kw": {"down_days": ["MON", "TUE", "WED"], "up_days": ["THU", "FRI"]}},
          negative="short_history", as_of="2015-06-18", start="2015-01-02", end="2015-06-18", text="THYAO Mon-Wed down, Thu-Fri up, early 2015?")
    h, chat = replay(q, db)
    assert {v["code"] for v in h.violations} <= {"MANDATORY_TOOL_RUN_BY_HARNESS"}


def test_claim_note_in_label(db):
    from bist.finetune.scripted import CLAIM_NOTE
    q = Q("S5", ("EREGL",), "general", {"macro_factor": "usdtry"}, negative="advice",
          text="Today is 10.09.2026. USD/TRY is rising and EREGL is an exporter, so it will definitely go up. Should I buy EREGL tomorrow?")
    q.claim = True
    h, chat = replay(q, db)
    assert CLAIM_NOTE in h.final["explanation"] and not h.final.get("fallback")
    assert not policy._ADVICE.search(policy._DISCLAIM_STRIP.sub(" ", CLAIM_NOTE))
