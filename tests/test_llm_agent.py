"""LLMAgent drives the harness with a scripted fake model (no Ollama needed): proves the loop, the retry-on-violation path and the safe fallbacks."""
import json
import re

import pytest

from bist.harness import policy, reference
from bist.harness.llm_agent import LLMAgent, render
from bist.harness.logdb import DecisionLog


class FakeModel:
    """Behaves like a tool-calling model: calls the reference plan's tools per state; optionally hallucinates once in the final answer."""

    def __init__(self, ticker="AKBNK", focus="fundamental_reaction", hallucinate=False, skip_quality=False, garbage_final=False):
        self.ticker, self.focus = ticker, focus
        self.hallucinate, self.skip_quality, self.garbage = hallucinate, skip_quality, garbage_final
        self.seen_state: dict[str, int] = {}
        self.final_calls = 0
        self.plan = reference.plan(ticker, focus)

    def __call__(self, messages, tools=None, json_format=False):
        last = messages[-1]["content"]
        if json_format:
            self.final_calls += 1
            if self.garbage:
                return {"content": "not json"}
            perm = json.loads(re.search(r"Permitted actions: (\[.*?\])", last).group(1).replace("'", '"'))
            lines = [l for l in last.splitlines() if re.match(r"\[E\d+\]", l)]
            action = "ANALYSIS_UNSAFE" if perm == ["ANALYSIS_UNSAFE"] else "WATCH"
            text = f"Educational analytical state: {action}.\n" + "\n".join(lines[:3]) + "\n" + policy.DISCLAIMER
            if self.hallucinate and self.final_calls == 1:
                text += " The stock is up 99.9% this month."
            return {"content": json.dumps({"action": action, "explanation": text})}
        if "SELECT_UNIVERSE" in last:
            return {"tool_calls": [{"function": {"name": "select_universe", "arguments": {"tickers": [self.ticker]}}}]}
        state = re.match(r"State (\w+)\.", last).group(1) if re.match(r"State (\w+)\.", last) else None
        if state is None or state in ("RUN_BACKTEST_",):
            return {"content": ""}
        n = self.seen_state[state] = self.seen_state.get(state, 0) + 1
        if n > 1:
            return {"tool_calls": [{"function": {"name": "next_state", "arguments": {}}}]}
        if state == "CHECK_DATA_QUALITY" and self.skip_quality:
            return {"tool_calls": [{"function": {"name": "next_state", "arguments": {}}}]}
        if state == "RUN_BACKTEST":
            return {"tool_calls": [{"function": {"name": "run_backtest", "arguments": {"variant": "C"}}}]}
        calls = [{"function": {"name": t, "arguments": p}} for s, t, p in self.plan if s == state and t != "build_evidence_bundle"]
        return {"tool_calls": calls}


@pytest.fixture()
def db(tmp_path):
    return DecisionLog(tmp_path / "l.sqlite")


def test_scripted_model_completes_workflow(db):
    h = LLMAgent(FakeModel(), db=db).run("AKBNK bilanço sonrası?", focus="fundamental_reaction", human=("APPROVE", "ok"))
    assert h.closed and h.final["action"] in policy.ACTIONS
    tools = [c["tool"] for c in h.calls]
    assert tools.index("check_data_quality") < tools.index("build_evidence_bundle") < tools.index("run_backtest")
    assert h.violations == []


def test_hallucinated_number_is_rejected_then_corrected(db):
    m = FakeModel(hallucinate=True)
    h = LLMAgent(m, db=db).run("AKBNK?", focus="fundamental_reaction", human=("APPROVE", "ok"))
    rejected = [v for v in h.violations if v["code"] == "OUTPUT_REJECTED"]
    assert len(rejected) == 1 and "NUMBER_NOT_IN_EVIDENCE" in rejected[0]["message"]
    assert "99.9" not in h.final["explanation"] and m.final_calls == 2


def test_skipped_quality_check_is_run_by_harness(db):
    h = LLMAgent(FakeModel(skip_quality=True), db=db).run("AKBNK?", focus="fundamental_reaction", human=("APPROVE", "ok"))
    assert any(v["code"] == "MANDATORY_TOOL_RUN_BY_HARNESS" and v["tool"] == "check_data_quality" for v in h.violations)
    assert any(c["tool"] == "check_data_quality" for c in h.calls)


def test_unparseable_model_falls_back_safely_and_is_flagged(db):
    h = LLMAgent(FakeModel(garbage_final=True), db=db).run("AKBNK?", focus="fundamental_reaction", human=("APPROVE", "ok"))
    assert any(v["code"] == "LLM_FALLBACK" for v in h.violations) and h.final.get("fallback") is True
    assert h.final["action"] == h.decision["action"]


def test_render_is_compact(db):
    h = reference.run("q", "AKBNK", "fundamental_reaction", db=db)
    env = next(c["envelope"] for c in h.calls if c["tool"] == "get_fundamental_metrics")
    txt = render("get_fundamental_metrics", env)
    assert len(txt) < 900 and "AKBNK" in txt
    assert render("run_backtest", {"status": "ERROR", "error": {"code": "LAYER_UNAVAILABLE", "message": "m"}}).startswith("run_backtest: ERROR LAYER_UNAVAILABLE")


# ------------------------------------------------------------- v2: conversation format, model-chosen focus/period
class PlanningModel(FakeModel):
    """Also chooses focus, analysis date and the question's period in SELECT_UNIVERSE, and passes start/end to range-aware tools."""

    def __init__(self, sel=None, **kw):
        super().__init__(**kw)
        self.sel = sel or {}
        self.requests: list[list[dict]] = []

    def __call__(self, messages, tools=None, json_format=False):
        self.requests.append([dict(m) for m in messages])
        if "SELECT_UNIVERSE" in messages[-1]["content"] and not json_format:
            return {"tool_calls": [{"function": {"name": "select_universe", "arguments": {"tickers": [self.ticker], **self.sel}}}]}
        return super().__call__(messages, tools, json_format)


def _check_pairing(msgs):
    """Every assistant message with tool_calls is followed by exactly one tool message per call; no tool message is orphaned."""
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "tool":
            raise AssertionError(f"orphan tool message at {i}")
        if m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            assert all(x["role"] == "tool" for x in msgs[i + 1:i + 1 + k]), i
            assert all(x.get("tool_name") for x in msgs[i + 1:i + 1 + k]), i
            i += 1 + k
            continue
        i += 1


def test_conversation_keeps_assistant_tool_calls_before_results(db):
    m = PlanningModel()
    a = LLMAgent(m, db=db)
    a.run("AKBNK bilanço sonrası?", focus="fundamental_reaction", human=("APPROVE", "ok"))
    for req in m.requests:
        _check_pairing(req)
    _check_pairing(a.messages)
    assert sum(1 for x in a.messages if x["role"] == "tool") >= 5


def test_model_selects_focus_and_period(db):
    sel = {"focus": "technical_reversal", "analysis_date": "2026-09-10", "period_start": "2024-01-01", "period_end": "2026-09-10"}
    m = PlanningModel(sel=sel, ticker="ASELS", focus="technical_reversal")
    a = LLMAgent(m, db=db)
    h = a.run("ASELS alt Bollinger bandına kaç kez dokundu? Analiz tarihi 10.09.2026, dönem 01.01.2024 – 10.09.2026")
    assert h.focus == "technical_reversal" and h.as_of == "2026-09-10"
    assert a.selection["period_start"] == "2024-01-01"
    assert any("2024-01-01" in x["content"] for x in a.messages if x["role"] == "user")
    q = next(c for c in h.calls if c["tool"] == "check_data_quality")
    assert q["envelope"]["data"]["analysis_type"] == "technical_reversal"


def test_caller_focus_and_as_of_override_model(db):
    m = PlanningModel(sel={"focus": "calendar_pattern", "analysis_date": "2026-01-05"})
    h = LLMAgent(m, db=db).run("AKBNK?", as_of="2026-09-10", focus="fundamental_reaction", human=("APPROVE", "ok"))
    assert h.focus == "fundamental_reaction" and h.as_of == "2026-09-10"


def test_invalid_model_focus_falls_back_to_general(db):
    h = LLMAgent(PlanningModel(sel={"focus": "moon"}), db=db).run("AKBNK?", human=("APPROVE", "ok"))
    assert h.focus == "general"


def test_request_context_is_compact(db):
    """Earlier states' tool results are sent as their first line only; the final turn sees only the setup + committed evidence."""
    m = PlanningModel()
    a = LLMAgent(m, db=db)
    a.run("AKBNK bilanço sonrası?", focus="fundamental_reaction", human=("APPROVE", "ok"))
    full = sum(len(x["content"]) for x in a.messages)
    for req in m.requests:
        _check_pairing(req)
        last_prompt = max(i for i, x in enumerate(req) if x["role"] == "user")
        assert all(len(x["content"]) <= 240 for x in req[:last_prompt] if x["role"] == "tool")
    final = [r for r in m.requests if "GENERATE_EXPLANATION" in r[-1]["content"]][0]
    assert not any(x["role"] == "tool" and x["tool_name"] != "select_universe" for x in final)
    assert max(sum(len(x["content"]) for x in r) for r in m.requests) < full
