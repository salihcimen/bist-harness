"""LLM agent: a local model (Ollama) drives the harness by choosing tool calls and writing the final conclusion.

Per state the model sees ONLY the tools that state exposes (+ a `next_state` pseudo-tool) and compact renderings of tool results
(the deterministic evidence statements, never raw 5 KB JSON). At GENERATE_EXPLANATION it must return {"action", "explanation"};
the harness validator rejects invented numbers / wrong actions / advice wording and the model gets the violations back (max 2 retries).
No cloud API is used anywhere: `OllamaChat` talks to http://localhost:11434.

Conversation format (v2): one growing message list. Every model turn is appended as an assistant message (with its tool_calls),
followed by exactly one `tool` message per call (`tool_name` set), so the transcript is valid for Ollama/OpenAI-style chat templates
and is the same format the Faz 5 training traces use (`LLMAgent.messages`). In SELECT_UNIVERSE the model chooses the ticker(s),
the focus, the analysis date and the question's data period; caller-supplied `as_of`/`focus` take precedence.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable

import pandas as pd

from ..tools import store
from ..tools.evidence import EXTRACTORS
from ..tools.export_schemas import OUT as SCHEMA_FILE
from . import policy
from .logdb import DecisionLog
from .machine import Harness, HarnessConfig
from .states import ALLOWED_TOOLS

CONTEXT_TOOL_LINE = 200  # chars kept of an earlier state's tool result / assistant text in the model context
FOCI = ["sector_catchup", "technical_reversal", "fundamental_reaction", "calendar_pattern", "general"]

SYSTEM = """You are the analysis agent of a BIST (Borsa Istanbul) educational analytics harness. You work in states; in each state you may call only the tools shown.
Rules: (1) numbers you write must come from tool output; never invent a number, a news item or a price. (2) Never give investment advice, price targets or buy/sell instructions.
(3) The only conclusions allowed are: HOLD, WATCH, INVESTIGATE, POTENTIAL_CATCH_UP_CANDIDATE, TECHNICAL_REVERSAL_CANDIDATE, FUNDAMENTAL_IMPROVEMENT_CANDIDATE, REJECT_SIGNAL, ANALYSIS_UNSAFE.
(4) If a tool reports an error, a missing source, or the data-quality gate blocks the analysis, say so; do not fill the gap.
(5) Use the analysis date and the data period of the question: pass `start`/`end` to every tool that accepts them; never use data after the analysis date.
Call `next_state` when you are done with a state."""

NEXT_TOOL = {"type": "function", "function": {"name": "next_state", "description": "Finish the current state and move to the next one.", "parameters": {"type": "object", "properties": {}}}}
SELECT_TOOL = {"type": "function", "function": {
    "name": "select_universe",
    "description": "Set up the analysis: the BIST ticker(s) the question is about, the analysis focus, the analysis date and the data period stated in the question.",
    "parameters": {"type": "object", "required": ["tickers", "focus"], "properties": {
        "tickers": {"type": "array", "items": {"type": "string"}, "description": "Ticker(s), main one first, e.g. [\"THYAO\"]; for a peer comparison list every stock named."},
        "focus": {"enum": FOCI, "description": "sector_catchup: laggard vs peers; technical_reversal: indicator/reversal events; fundamental_reaction: financial statements and price reaction; calendar_pattern: weekday/multi-day patterns; general: anything else (incl. macro, advice requests)."},
        "analysis_date": {"type": "string", "format": "date", "description": "Analysis date YYYY-MM-DD if the question gives one (e.g. 'Today is 10.09.2026' -> 2026-09-10)."},
        "period_start": {"type": "string", "format": "date", "description": "Start of the data period in the question, YYYY-MM-DD."},
        "period_end": {"type": "string", "format": "date", "description": "End of the data period in the question, YYYY-MM-DD (not after the analysis date)."}}}}}


class OllamaChat:
    def __init__(self, model: str = "qwen3:4b-instruct-2507-q4_K_M", host: str = "http://localhost:11434", temperature: float = 0.0, num_ctx: int = 16384, timeout: int = 300):
        self.model, self.host, self.temperature, self.num_ctx, self.timeout = model, host, temperature, num_ctx, timeout

    def __call__(self, messages: list[dict], tools: list[dict] | None = None, json_format: bool = False) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": self.temperature, "num_ctx": self.num_ctx}}
        if tools:
            body["tools"] = tools
        if json_format:
            body["format"] = "json"
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]


_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Qwen output -> (content, tool_calls). A block whose JSON does not parse keeps its raw arguments string, so LLMAgent reports
    MALFORMED_ARGUMENTS instead of silently dropping the call."""
    calls = []
    for body in _TOOL_CALL.findall(text):
        try:
            d = json.loads(body)
            calls.append({"function": {"name": d.get("name", ""), "arguments": d.get("arguments") or {}}})
        except ValueError:
            name = re.search(r'"name"\s*:\s*"([^"]*)"', body)
            calls.append({"function": {"name": name.group(1) if name else "unknown", "arguments": body}})
    content = _TOOL_CALL.sub("", text).strip() if calls else text.strip()
    return content, calls


class RawOllamaChat:
    """Same interface as OllamaChat, but the prompt is rendered with the HF/Qwen chat template used for training (local tokenizer)
    and sent to /api/generate with raw=true. Ollama's /api/chat drops JSON-schema fields (default/minimum/maximum...) from tool
    definitions and renders the conversation slightly differently, so the fine-tuned model would see another format than it
    was trained on. Use it for Qwen models (base and fine-tuned) so both get the identical, official template."""

    def __init__(self, model: str = "qwen3:4b-instruct-2507-q4_K_M", host: str = "http://localhost:11434", temperature: float = 0.0,
                 num_ctx: int = 16384, timeout: int = 300, num_predict: int = -1, post: Callable | None = None):
        self.model, self.host, self.temperature, self.num_ctx, self.timeout, self.num_predict = model, host, temperature, num_ctx, timeout, num_predict
        self._post = post or self._http

    def _http(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(f"{self.host}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        from ..finetune.export import to_hf_messages, tokenizer
        return tokenizer().apply_chat_template(to_hf_messages(messages), tools=tools or None, tokenize=False, add_generation_prompt=True)

    def __call__(self, messages: list[dict], tools: list[dict] | None = None, json_format: bool = False) -> dict:
        body = {"model": self.model, "prompt": self.prompt(messages, tools), "raw": True, "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.num_predict,
                            "stop": ["<|im_end|>", "<|im_start|>"]}}
        if json_format:
            body["format"] = "json"
        content, calls = parse_tool_calls(self._post("/api/generate", body)["response"])
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        return msg


def tool_defs(names: list[str]) -> list[dict]:
    sch = {t["name"]: t for t in json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))["tools"]}
    out = [NEXT_TOOL]
    for n in names:
        t = sch[n]
        out.append({"type": "function", "function": {"name": n, "description": t["description"], "parameters": t["input_schema"]}})
    return out


def render(tool: str, env: dict, call_id: str = "C0") -> str:
    """Compact text the model sees for a tool result."""
    if env["status"] == "ERROR":
        return f"{tool}: ERROR {env['error']['code']} - {env['error']['message']}"
    w = [x["code"] for x in env.get("warnings", [])]
    notes = "".join(f"\nHarness: {n['message']}" for n in env.get("harness_notes", []))
    return _render_body(tool, env, call_id, w) + notes


def _render_body(tool: str, env: dict, call_id: str, w: list[str]) -> str:
    fn = EXTRACTORS.get(tool)
    if fn:
        try:
            st = [it["statement"] for it in fn({"call_id": call_id, "tool": tool, "envelope": env})]
            return f"{tool}: {env['status']}{' warnings=' + ','.join(w) if w else ''}\n" + "\n".join(st)
        except Exception:
            pass
    return f"{tool}: {env['status']}{' warnings=' + ','.join(w) if w else ''}\n" + json.dumps(env["data"], default=str)[:700]


def _date(x) -> str | None:
    try:
        return str(pd.Timestamp(str(x)).date()) if x else None
    except (ValueError, TypeError):
        return None


class LLMAgent:
    def __init__(self, chat: Callable, db: DecisionLog | None = None, config: HarnessConfig | None = None, max_steps_per_state: int = 4, max_retries: int = 2):
        self.chat, self.db, self.cfg = chat, db, config
        self.max_steps, self.max_retries = max_steps_per_state, max_retries
        self.trace: list[dict] = []
        self.messages: list[dict] = []
        self.selection: dict = {}
        self._base_end: int | None = None    # end of system + question + selection + setup (always sent in full)
        self._seg_start: int | None = None   # first message of the current state (sent in full)
        self._final_start: int | None = None # first message of GENERATE_EXPLANATION (only base + this is sent)

    # ---------------------------------------------------------------- conversation helpers
    def _args(self, tc) -> tuple[str, dict]:
        f = tc["function"]
        a = f.get("arguments") or {}
        return f["name"], (json.loads(a) if isinstance(a, str) else a)

    def context(self) -> list[dict]:
        """Messages sent to the model for the next turn (deterministic, also used for training traces):
        base in full; earlier states compressed (tool results -> their first line = status + warnings); current state in full.
        In GENERATE_EXPLANATION only the base + the explanation turn (the committed evidence carries the facts)."""
        msgs = self.messages
        if self._base_end is None:
            return list(msgs)
        base = msgs[:self._base_end]
        if self._final_start is not None:
            return base + msgs[self._final_start:]
        seg = self._seg_start or self._base_end
        mid = []
        for x in msgs[self._base_end:seg]:
            if x["role"] == "tool":
                x = {**x, "content": x["content"].splitlines()[0][:CONTEXT_TOOL_LINE] if x["content"] else ""}
            elif x["role"] == "assistant" and len(x["content"]) > CONTEXT_TOOL_LINE:
                x = {**x, "content": x["content"][:CONTEXT_TOOL_LINE]}
            mid.append(x)
        return base + mid + msgs[seg:]

    def _ask(self, tools: list[dict] | None = None, json_format: bool = False) -> dict:
        """One model turn on the compact context; the reply is appended to the full transcript as an assistant message."""
        m = self.chat(self.context(), tools=tools, json_format=json_format) or {}
        calls = m.get("tool_calls") or []
        msg = {"role": "assistant", "content": m.get("content") or ""}
        if calls:
            msg["tool_calls"] = calls
        self.messages.append(msg)
        return {"content": msg["content"], "tool_calls": calls}

    def _tool_result(self, name: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_name": name, "content": content})

    def _user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    # ---------------------------------------------------------------- SELECT_UNIVERSE
    def _select(self) -> dict:
        """Model sets tickers/focus/date/period. Tickers are validated against the universe; results go back as tool messages."""
        self._user("State SELECT_UNIVERSE: call select_universe with the ticker(s), the focus, the analysis date and the data period of the question.")
        for _ in range(self.max_steps):
            m = self._ask(tools=[SELECT_TOOL])
            got = None
            for tc in m["tool_calls"]:
                try:
                    n, a = self._args(tc)
                except (ValueError, KeyError) as e:
                    self._tool_result("select_universe", f"ERROR MALFORMED_ARGUMENTS - {type(e).__name__}")
                    continue
                if n != "select_universe":
                    self._tool_result(n, "ERROR TOOL_NOT_ALLOWED_IN_STATE - only select_universe is available here")
                    continue
                ok, bad = [], []
                for t in a.get("tickers") or []:
                    try:
                        ok.append(store.resolve_ticker(str(t))[0])
                    except Exception:
                        bad.append(str(t))
                self.trace.append({"state": "SELECT_UNIVERSE", "tool": n, "args": a, "result": "ok" if ok and not bad else f"unknown {bad}"})
                if ok and not bad:
                    got = {**a, "tickers": list(dict.fromkeys(ok))}
                    self._tool_result(n, f"select_universe: OK tickers={got['tickers']}")
                else:
                    self._tool_result(n, f"select_universe: ERROR UNKNOWN_TICKER {bad or 'none given'}; valid: {', '.join(store.tickers())}")
            if got:
                return got
            if not m["tool_calls"]:
                self._user("Call select_universe now.")
        return {}

    # ---------------------------------------------------------------- analysis states
    def _work_state(self, h: Harness) -> None:
        tdefs = tool_defs(ALLOWED_TOOLS[h.state])
        self._seg_start = len(self.messages)
        self._user(f"State {h.state}. Allowed tools: {', '.join(ALLOWED_TOOLS[h.state])}. Tickers: {', '.join(h.tickers)}. "
                   "Call the tool(s) needed for the question (skip the state if none is needed), then next_state.")
        for _ in range(self.max_steps):
            m = self._ask(tools=tdefs)
            if not m["tool_calls"]:
                return
            done = False
            for tc in m["tool_calls"]:
                try:
                    n, a = self._args(tc)
                except (ValueError, KeyError) as e:
                    self._tool_result(tc.get("function", {}).get("name", "unknown"), f"ERROR MALFORMED_ARGUMENTS - {type(e).__name__}")
                    continue
                if n == "next_state":
                    done = True
                    self._tool_result(n, "next_state: OK")
                    continue
                env = h.call(n, **a)
                self.trace.append({"state": h.state, "tool": n, "args": a, "status": env["status"]})
                self._tool_result(n, render(n, env, h.calls[-1]["call_id"] if env["status"] != "ERROR" and h.calls else "C0"))
            if done:
                return

    def _ensure_required(self, h: Harness) -> None:
        """Mandatory tools the model did not call are run by the harness with safe defaults and flagged (the model cannot skip them)."""
        defaults = {"CHECK_DATA_QUALITY": ("check_data_quality", {"tickers": h.tickers, "analysis_type": h.focus}), "BUILD_EVIDENCE": ("build_evidence_bundle", {}),
                    "RUN_BACKTEST": ("run_backtest", {"variant": "D"})}
        d = defaults.get(h.state)
        if d and not (h.state == "RUN_BACKTEST" and h.blocked()) and not any(c["tool"] == d[0] and c["envelope"]["status"] != "ERROR" for c in h.calls):
            if h.state != "BUILD_EVIDENCE":   # evidence building is always done by the harness itself; not a model failure
                h.violations.append({"state": h.state, "tool": d[0], "code": "MANDATORY_TOOL_RUN_BY_HARNESS", "message": f"model did not complete {d[0]} correctly; default parameters used"})
            h.call(d[0], **d[1])

    # ---------------------------------------------------------------- run
    def run(self, question: str, as_of: str | None = None, focus: str | None = None, human: tuple[str, str] | None = None) -> Harness:
        self._base_end = self._seg_start = self._final_start = None
        self.messages = [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Question: {question}" + (f"\nAnalysis date: {as_of}" if as_of else "")}]
        sel = self._select()
        self.selection = sel
        f = focus or (sel.get("focus") if sel.get("focus") in FOCI else "general")
        h = Harness(question, as_of=as_of or _date(sel.get("analysis_date")), focus=f, db=self.db, config=self.cfg)
        h.advance()                                                       # START -> SELECT_UNIVERSE
        if not sel.get("tickers") or not h.select_universe(sel["tickers"]).get("ok"):
            h.violations.append({"state": h.state, "tool": None, "code": "NO_TICKER_SELECTED", "message": "model did not select a valid ticker"})
            return h
        ps, pe = _date(sel.get("period_start")), _date(sel.get("period_end"))
        if pe and pe > h.as_of:
            pe = h.as_of
        period = f"; data period {ps or 'not stated'} to {pe or h.as_of} (pass start/end to tools that accept them)" if (ps or pe) else ""
        self._user(f"Analysis set up: tickers {', '.join(h.tickers)}; focus {h.focus}; analysis date {h.as_of} (no data after it){period}.")
        self._base_end = len(self.messages)
        while h.state != "RISK_GATE":
            if h.state not in ("SELECT_UNIVERSE",) and ALLOWED_TOOLS[h.state] and h.state != "BUILD_EVIDENCE" and not (h.state == "RUN_BACKTEST" and h.blocked()):
                self._work_state(h)
            self._ensure_required(h)
            if not h.advance()["ok"]:
                raise RuntimeError(f"harness refused to advance from {h.state}")
        g = h.risk_gate()
        h.advance()                                                       # -> GENERATE_EXPLANATION
        ev = h.evidence
        obs = {i["id"]: i for i in ev["observed"]}
        ctx = ("State GENERATE_EXPLANATION. Committed evidence (cite as [E#], quote only these numbers):\n" + "\n".join(f"[{i}] {obs[i]['statement']}" for i in ev["committed_ids"])
               + f"\nRisk gate: {g['verdict']} {g['reasons']}\nPermitted actions: {g['permitted_actions']}\n"
               + 'Return JSON: {"action": <one permitted action>, "explanation": <short text citing [E#] ids; end with: ' + policy.DISCLAIMER + '>}')
        self._final_start = len(self.messages)
        self._user(ctx)
        for attempt in range(self.max_retries + 1):
            m = self._ask(json_format=True)
            try:
                out = json.loads(m["content"])
                res = h.propose(str(out.get("action", "")), str(out.get("explanation", "")))
            except Exception as e:  # malformed JSON
                res = {"ok": False, "violations": [f"MALFORMED_OUTPUT:{type(e).__name__}"]}
                h.violations.append({"state": h.state, "tool": None, "code": "OUTPUT_REJECTED", "message": res["violations"][0]})
            self.trace.append({"state": "GENERATE_EXPLANATION", "attempt": attempt, "ok": res["ok"], "violations": res["violations"]})
            if res["ok"]:
                break
            self._user("Your previous answer was rejected: " + "; ".join(res["violations"]) + ". Fix it.\n" + ctx)
        if h.final is None:                                               # safe failure instead of an unvalidated conclusion
            ref = h.reference_output()
            h.violations.append({"state": h.state, "tool": None, "code": "LLM_FALLBACK", "message": "model output never passed validation; reference output used and flagged"})
            h.final = {**ref, "fallback": True}
        h.advance()
        if human:
            h.review(*human)
            h.advance()
            h.close()
        return h
