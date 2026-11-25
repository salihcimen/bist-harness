"""ScriptedChat: plays the model inside the real `LLMAgent` loop by returning the reference agent's tool calls state by state.

Every `chat(messages, tools)` call it receives is the exact compact context the real model sees at inference (`LLMAgent.context()`),
so each recorded turn is one training example (input = messages + tools, target = the reply).
Turn rules: per state one turn with all reference calls of that state plus `next_state`; a state without calls gets a `next_state`
turn. Own earlier turn (no `next_state`): `rank_sector_relative_strength` when `test_catch_up_history` follows (S1: the harness moves
the main ticker to the laggard after ranking), and an injected wrong call (error-recovery traces; that turn is not trained on).
"""
from __future__ import annotations

import copy
import json
import re
from typing import Callable

from ..harness import policy, reference
from ..harness.llm_agent import LLMAgent
from ..harness.machine import Harness
from .templates import Question

ADVICE_PREFIX = "No investment advice, buy/sell instruction or price forecast is given here; the following is historical evidence only.\n"
CLAIM_NOTE = "A certain price increase is not supported by the tool evidence; only the historical relationship is shown.\n"
CATCHUP_SPLIT = ("rank_sector_relative_strength", "test_catch_up_history")


class TraceMismatch(RuntimeError):
    pass


# ------------------------------------------------------------------ injected errors (each verified to give an ERROR envelope)
def _find(history, tool, cond: Callable[[dict], bool] = lambda p: True):
    return next(((s, t, p) for s, t, p in history if t == tool and cond(p)), None)


def _bad(tool, mutate, cond=lambda p: True):
    def f(history):
        hit = _find(history, tool, cond)
        return (hit[0], tool, mutate(dict(hit[2]))) if hit else None
    return f


ERROR_KINDS: dict[str, Callable] = {
    "max_rows": _bad("get_stock_history", lambda p: {**p, "max_rows": 1000}),
    "bad_condition": _bad("detect_reversal_events", lambda p: {**p, "condition_sets": [["rsi_minimum"]]}),
    "bad_event": _bad("detect_reversal_events", lambda p: {**p, "event_types": ["rsi_oversold"]}),
    "unknown_peer": _bad("rank_sector_relative_strength", lambda p: {**p, "tickers": [*p["tickers"], "HUBER"]}, lambda p: bool(p.get("tickers"))),
    "missing_weekday": _bad("test_weekday_pattern", lambda p: {k: v for k, v in p.items() if k != "weekday"}, lambda p: "weekday" in p),
    "bad_factor": _bad("analyze_macro_sensitivity", lambda p: {**p, "factor": p["factor"].replace("try", "_try") if "try" in p["factor"] else p["factor"] + "_index"}),
}


def applicable_errors(history: list[tuple[str, str, dict]]) -> list[str]:
    return [k for k, f in ERROR_KINDS.items() if f(history)]


# ------------------------------------------------------------------ the scripted model
def _tc(name: str, args: dict) -> dict:
    return {"function": {"name": name, "arguments": args}}


class ScriptedChat:
    def __init__(self, select_args: dict, history: list[tuple[str, str, dict]], final: dict, inject: tuple[str, str, dict] | None = None):
        self.select_args, self.history, self.final, self.inject = select_args, history, final, inject
        self.turns: list[dict] = []
        self._queue: dict[str, list[tuple[list[dict], str, bool]]] = {}
        self._recovery_next = False

    def _plan_state(self, state: str) -> list[tuple[list[dict], str, bool]]:
        calls = [(t, p) for s, t, p in self.history if s == state and t != "build_evidence_bundle"]
        nxt = _tc("next_state", {})
        turns: list[tuple[list[dict], str, bool]] = []
        if self.inject and self.inject[0] == state:
            turns.append(([_tc(self.inject[1], self.inject[2])], "injected", False))
        names = [t for t, _ in calls]
        if all(n in names for n in CATCHUP_SPLIT):
            k = names.index(CATCHUP_SPLIT[1])
            turns.append(([_tc(t, p) for t, p in calls[:k]], "tool_call", True))
            turns.append(([_tc(t, p) for t, p in calls[k:]] + [nxt], "tool_call", True))
        else:
            turns.append(([_tc(t, p) for t, p in calls] + [nxt], "tool_call" if calls else "next_state", True))
        return turns

    def __call__(self, messages: list[dict], tools: list[dict] | None = None, json_format: bool = False) -> dict:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if json_format:
            reply, kind, train = {"content": json.dumps(self.final, ensure_ascii=False)}, "final", True
        elif "SELECT_UNIVERSE" in last_user:
            reply, kind, train = {"tool_calls": [_tc("select_universe", self.select_args)]}, "select", True
        else:
            m = re.match(r"State (\w+)\.", last_user)
            state = m.group(1) if m else ""
            q = self._queue.setdefault(state, self._plan_state(state))
            calls, kind, train = q.pop(0) if q else ([_tc("next_state", {})], "next_state", True)
            if self._recovery_next and train:
                kind = "recovery"
            self._recovery_next = kind == "injected"
            reply = {"tool_calls": calls}
        self.turns.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools), "reply": copy.deepcopy(reply), "kind": kind, "train": train})
        return reply


# ------------------------------------------------------------------ reference label + replay
def _select_args(q: Question) -> dict:
    a = {"tickers": q.tickers, "focus": q.focus, "analysis_date": q.requested_as_of, "period_start": q.start, "period_end": q.requested_end or q.end}
    return {k: v for k, v in a.items() if v is not None}


def _with_prefix(q: Question, h: Harness, out: dict) -> dict:
    if q.negative != "advice":
        return out
    text = ADVICE_PREFIX + (CLAIM_NOTE if getattr(q, "claim", False) else "") + out["explanation"]
    ok = policy.validate_output(out["action"], text, h.decision["permitted"], h.evidence, q.text, extra=[h.as_of, str(h.gate["reasons"])])["ok"]
    return {**out, "explanation": text} if ok else out


def reference_label(q: Question, db) -> tuple[Harness, dict]:
    h = reference.run(q.text, q.tickers[0], q.focus, as_of=q.as_of, db=db, start=q.start, end=q.end, **q.plan_kw)
    return h, _with_prefix(q, h, h.reference_output())


def _run_agent(q: Question, db, history, final, inject) -> tuple[Harness, ScriptedChat]:
    chat = ScriptedChat(_select_args(q), history, final, inject)
    h = LLMAgent(chat=chat, db=db).run(q.text, human=("APPROVE", "training trace"))
    return h, chat


def replay(q: Question, db, inject: str | Callable[[list[str]], str | None] | None = None) -> tuple[Harness, ScriptedChat]:
    """Run the reference, then LLMAgent with ScriptedChat. With `inject`, the wrong call changes the evidence ids, so the label is the
    agent harness's own reference output (pass 1 ends in the flagged fallback; pass 2 returns that label and must validate).
    `inject` may be a callable that picks one of the applicable error kinds (or None) after the reference run."""
    ref_h, final = reference_label(q, db)
    if callable(inject):
        inject = inject(applicable_errors(ref_h.history))
    bad = ERROR_KINDS[inject](ref_h.history) if inject else None
    if inject and bad is None:
        raise TraceMismatch(f"error kind {inject} not applicable")
    if bad:
        h1, _ = _run_agent(q, db, ref_h.history, {"action": "", "explanation": ""}, bad)
        final = _with_prefix(q, h1, h1.reference_output())
    h, chat = _run_agent(q, db, ref_h.history, final, bad)
    chat.inject_kind = inject
    chat.ref_action = final["action"]
    problems = []
    if h.final is None or h.final.get("fallback"):
        problems.append("fallback")
    if h.final and h.final["action"] != final["action"]:
        problems.append(f"action {h.final['action']} != {final['action']}")
    ref_failed = {c["tool"] for c in ref_h.calls if c["envelope"]["status"] == "ERROR"}
    # a mandatory tool that fails on the data for the reference too (e.g. run_backtest with too little history) is re-run by the
    # harness and flagged; the scripted call was the correct one, so this is not a model failure
    real = [v for v in h.violations if not (v["code"] == "MANDATORY_TOOL_RUN_BY_HARNESS" and v["tool"] in ref_failed)]
    if real:
        problems.append(f"violations {[v['code'] for v in real]}")
    if policy.quality_verdict(h.calls) != policy.quality_verdict(ref_h.calls):
        problems.append("quality verdict differs")
    if problems:
        raise TraceMismatch(f"{q.id}: {'; '.join(problems)}")
    return h, chat
