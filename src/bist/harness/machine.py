"""The stateful harness. One `Harness` object = one analysis walking START -> ... -> LOG_AND_CLOSE.

Enforced here (not left to the language model):
  * a tool can only be called in a state that exposes it, and never with an as_of later than the analysis date;
  * states advance forward only; mandatory states (quality check, evidence, backtest, risk gate, explanation, human review) cannot be skipped;
  * if the data-quality verdict blocks the analysis the analysis short-circuits to BUILD_EVIDENCE/RISK_GATE (backtest waived) and ends in ANALYSIS_UNSAFE;
  * the final output must pass `policy.validate_output` (8 allowed actions, permitted by gate+evidence, numbers only from evidence, no advice wording);
  * a human decision is required before the analysis can be closed. The system never produces or sends an order.
`HarnessConfig` switches individual protections off to build the experiment ladder (raw tools -> +states -> +evidence/gate -> +memory/human).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..tools import REGISTRY, SCHEMA_VERSION, session, store
from ..tools.envelope import jsonable
from . import policy
from .logdb import DecisionLog
from .states import ALLOWED_TOOLS, MANDATORY, REQUIRED_TOOL, STATES

REVIEW_DECISIONS = ("APPROVE", "REJECT", "DEFER")


@dataclass
class HarnessConfig:
    enforce_permissions: bool = True   # tool only in its state, as_of clamp
    enforce_order: bool = True         # forward-only, mandatory states can't be skipped
    enforce_gate: bool = True          # validator may not output an action the gate/evidence forbids
    enforce_grounding: bool = True     # numbers/citations/advice checks on the explanation
    require_human: bool = True


def _err(code: str, msg: str, tool: str = "", params: dict | None = None) -> dict:
    return {"tool": tool, "schema_version": SCHEMA_VERSION, "status": "ERROR", "params": params or {}, "as_of": None, "data": None, "warnings": [],
            "error": {"code": code, "message": msg}, "provenance": None}


class Harness:
    def __init__(self, question: str, as_of: str | None = None, focus: str = "general", analysis_id: str | None = None,
                 db: DecisionLog | None = None, config: HarnessConfig | None = None):
        self.cfg = config or HarnessConfig()
        self.id = analysis_id or f"A-{uuid.uuid4().hex[:10]}"
        self.question, self.focus = question, focus
        self.as_of = str(store.resolve_as_of(as_of)[0].date())
        self.db = db or DecisionLog()
        self.state = "START"
        self.tickers: list[str] = []
        self.evidence: dict | None = None
        self.gate: dict | None = None
        self.decision: dict | None = None
        self.final: dict | None = None
        self.review_record: dict | None = None
        self.history: list[tuple[str, str, dict]] = []  # (state, tool, caller params) of every call that reached the tool
        self.violations: list[dict] = []     # blocked tool calls / rejected outputs (metrics: invalid tool-call rate, unsupported conclusions)
        self._seq = 0
        self._tseq = 0
        self.closed = False
        session.new_analysis(self.id, as_of=self.as_of, question=question)
        self.db.start(self.id, question, self.as_of, focus)

    # ---------------------------------------------------------------- introspection
    @property
    def calls(self) -> list[dict]:
        return session.calls(self.id)

    def allowed_tools(self) -> list[str]:
        return list(ALLOWED_TOOLS[self.state]) if self.cfg.enforce_permissions else sorted(REGISTRY)

    def status(self) -> dict:
        return {"analysis_id": self.id, "state": self.state, "allowed_tools": self.allowed_tools(), "tickers": self.tickers, "as_of": self.as_of,
                "next_states": self.next_states(), "blocked": self.blocked(), "calls": len(self.calls)}

    def blocked(self) -> bool:
        return policy.quality_verdict(self.calls) in policy.BLOCKING

    # ---------------------------------------------------------------- tools
    def select_universe(self, tickers: list[str]) -> dict:
        if self.state != "SELECT_UNIVERSE" and self.cfg.enforce_order:
            return self._violation("select_universe", "WRONG_STATE", f"select_universe is only available in SELECT_UNIVERSE (state: {self.state})")
        res, bad = [], []
        for t in tickers:
            try:
                res.append(store.resolve_ticker(t)[0])
            except Exception:
                bad.append(t)
        if bad or not res:
            return self._violation("select_universe", "UNKNOWN_TICKER", f"not in the 30-stock universe: {bad or tickers}")
        self.tickers = res
        sec = store.universe().set_index("ticker").loc[res[0], "peer_group"]
        self.db.exec("UPDATE analyses SET tickers=?, sector=? WHERE id=?", ",".join(res), sec, self.id)
        return {"ok": True, "tickers": res, "sector": sec}

    def call(self, tool: str, **params) -> dict:
        if self.closed:
            return _err("ANALYSIS_CLOSED", "analysis is closed", tool, params)
        if self.cfg.enforce_permissions and tool not in ALLOWED_TOOLS[self.state]:
            return self._violation(tool, "TOOL_NOT_ALLOWED_IN_STATE",
                                   f"'{tool}' is not available in state {self.state}; allowed here: {ALLOWED_TOOLS[self.state] or 'none'}", params)
        if tool not in REGISTRY:
            return self._violation(tool, "UNKNOWN_TOOL", f"'{tool}' is not a registered tool", params)
        caller_params = dict(params)
        if tool == "build_evidence_bundle":
            params = {"analysis_id": self.id}
        a = params.get("as_of")
        if a is not None and self.cfg.enforce_permissions and str(a)[:10] > self.as_of:
            return self._violation(tool, "AS_OF_AFTER_ANALYSIS_DATE", f"as_of {a} is later than the analysis date {self.as_of} (look-ahead is not allowed)", params)
        t = params.get("ticker")
        if t and self.tickers and self.cfg.enforce_permissions:
            try:
                if store.resolve_ticker(str(t))[0] not in self.tickers and tool != "rank_sector_relative_strength":
                    return self._violation(tool, "TICKER_NOT_IN_SELECTION", f"{t} was not selected in SELECT_UNIVERSE ({self.tickers})", params)
            except Exception:
                pass
        env = session.run(self.id, tool, **params)
        self.history.append((self.state, tool, caller_params))
        c = self.calls[-1]
        if tool == "rank_sector_relative_strength" and env["status"] != "ERROR" and self.focus == "sector_catchup" and len(self.tickers) > 1:
            self._main_to_laggard(env, c)
        self._seq += 1
        self.db.tool_call(self.id, self._seq, c["call_id"], self.state, tool, env.get("params") or params, env)
        if tool == "build_evidence_bundle" and env["status"] != "ERROR":
            self.evidence = env["data"]
            self.db.evidence(self.id, self.evidence["observed"])
        return env

    def _main_to_laggard(self, env: dict, rec: dict) -> None:
        """S1: 'which selected stock lagged most?' -- the model cannot know before ranking, so the harness makes the most lagging
        selected ticker the main one (decisions are taken for tickers[0]) and says so in the tool result."""
        rows = (env.get("data") or {}).get("ranking_lagging_first") or []
        lag = next((m["ticker"] for m in rows if m.get("ticker") in self.tickers), None)
        if not lag or lag == self.tickers[0]:
            return
        old = self.tickers[0]
        self.tickers = [lag] + [t for t in self.tickers if t != lag]
        # a harness annotation, kept apart from the tool's own status/warnings so replay still reproduces the tool result
        w = {"code": "MAIN_TICKER_SET_TO_LAGGARD", "message": f"main ticker set to {lag} (most lagging selected ticker; was {old})"}
        for e in {id(env): env, id(rec["envelope"]): rec["envelope"]}.values():
            e.setdefault("harness_notes", []).append(w)
        self.db.exec("UPDATE analyses SET tickers=? WHERE id=?", ",".join(self.tickers), self.id)

    def _violation(self, tool, code, msg, params=None) -> dict:
        env = _err(code, msg, tool, params)
        self.violations.append({"state": self.state, "tool": tool, "code": code, "message": msg})
        self._seq += 1
        self.db.tool_call(self.id, self._seq, None, self.state, tool, params or {}, env, allowed=False)
        return env

    # ---------------------------------------------------------------- transitions
    def next_states(self) -> list[str]:
        if self.closed:
            return []
        i = STATES.index(self.state)
        if not self.cfg.enforce_order:
            return STATES[i + 1:]
        out = []
        for s in STATES[i + 1:]:
            out.append(s)
            if s in MANDATORY and not self._waived(s):
                break
        if self.blocked():  # short-circuit: nothing but evidence -> gate -> explanation -> review -> close
            out = [s for s in out if s in ("BUILD_EVIDENCE", "RISK_GATE", "GENERATE_EXPLANATION", "HUMAN_REVIEW", "LOG_AND_CLOSE")]
        return out

    def _waived(self, s: str) -> bool:
        return s == "RUN_BACKTEST" and self.blocked()

    def _can_leave(self) -> str | None:
        s = self.state
        if s == "SELECT_UNIVERSE" and not self.tickers:
            return "select_universe must be called first"
        req = REQUIRED_TOOL.get(s)
        if req and not (s == "RUN_BACKTEST" and self.blocked()) and not any(c["tool"] == req for c in self.calls):
            return f"{s} can not be left before '{req}' was called"
        if s == "RISK_GATE" and self.gate is None:
            return "risk_gate() must be evaluated before leaving RISK_GATE"
        if s == "GENERATE_EXPLANATION" and self.final is None:
            return "a validated conclusion (propose) is required before leaving GENERATE_EXPLANATION"
        if s == "HUMAN_REVIEW" and self.cfg.require_human and self.review_record is None:
            return "a human review decision is required before closing"
        return None

    def advance(self, to: str | None = None) -> dict:
        legal = self.next_states()
        if not legal:
            return {"ok": False, "state": self.state, "error": "no further state"}
        target = to or legal[0]
        if target not in STATES:
            return self._deny(target, f"'{target}' is not a state")
        if self.cfg.enforce_order:
            if target not in legal:
                skipped = [s for s in STATES[STATES.index(self.state) + 1:STATES.index(target) if target in STATES else 0] if s in MANDATORY and not self._waived(s)]
                why = f"cannot skip mandatory state(s) {skipped}" if skipped and STATES.index(target) > STATES.index(self.state) else f"'{target}' is not reachable from {self.state}; legal: {legal}"
                return self._deny(target, why)
            why = self._can_leave()
            if why:
                return self._deny(target, why)
        self._tseq += 1
        self.db.transition(self.id, self._tseq, self.state, target, True)
        self.state = target
        self.db.exec("UPDATE analyses SET final_state=? WHERE id=?", target, self.id)
        return {"ok": True, "state": target, "allowed_tools": self.allowed_tools()}

    def _deny(self, target, why) -> dict:
        self._tseq += 1
        self.db.transition(self.id, self._tseq, self.state, target, False, why)
        self.violations.append({"state": self.state, "tool": None, "code": "ILLEGAL_TRANSITION", "message": why})
        return {"ok": False, "state": self.state, "error": why}

    # ---------------------------------------------------------------- gate / decision / output
    def risk_gate(self) -> dict:
        if self.state != "RISK_GATE" and self.cfg.enforce_order:
            return {"error": f"risk_gate is evaluated in RISK_GATE (state: {self.state})"}
        self.gate = policy.risk_gate(self.calls)
        self.decision = policy.decide(self.calls, self.tickers[0] if self.tickers else "", self.focus, self.as_of, self.gate)
        return {**self.gate, "permitted_actions": self.decision["permitted"]}

    def reference_output(self) -> dict:
        """Reference conclusion and explanation (the label for training / the oracle for evaluation)."""
        assert self.decision and self.evidence, "evidence and gate must exist"
        return {"action": self.decision["action"], "explanation": policy.build_explanation(self.decision["action"], self.evidence, self.gate, self.decision["rationale"])}

    def propose(self, action: str, explanation: str) -> dict:
        if self.state != "GENERATE_EXPLANATION" and self.cfg.enforce_order:
            return {"ok": False, "violations": [f"WRONG_STATE:{self.state}"]}
        if self.decision is None or self.evidence is None:
            return {"ok": False, "violations": ["NO_GATE_OR_EVIDENCE"]}
        permitted = self.decision["permitted"] if self.cfg.enforce_gate else policy.ACTIONS
        res = policy.validate_output(action, explanation, permitted, self.evidence, self.question,
                                     extra=[self.as_of, str(self.gate["reasons"])]) if (self.cfg.enforce_grounding or self.cfg.enforce_gate) else {"ok": True, "violations": []}
        if not self.cfg.enforce_grounding:
            res["violations"] = [x for x in res["violations"] if x.startswith("ACTION_")]
            res["ok"] = not res["violations"]
        if not self.cfg.enforce_gate:
            res["violations"] = [x for x in res["violations"] if not x.startswith("ACTION_NOT_PERMITTED")]
            res["ok"] = not res["violations"]
        if res["ok"]:
            self.final = {"action": action, "explanation": explanation}
        else:
            self.violations.append({"state": self.state, "tool": None, "code": "OUTPUT_REJECTED", "message": "; ".join(res["violations"]), "action": action})
        return res

    def review(self, decision: str, reason: str, reviewer: str = "human") -> dict:
        if self.state != "HUMAN_REVIEW" and self.cfg.enforce_order:
            return {"ok": False, "error": f"review is recorded in HUMAN_REVIEW (state: {self.state})"}
        if decision not in REVIEW_DECISIONS:
            return {"ok": False, "error": f"decision must be one of {REVIEW_DECISIONS}"}
        if not reason.strip():
            return {"ok": False, "error": "a reason is required"}
        self.review_record = {"reviewer": reviewer, "decision": decision, "reason": reason}
        self.db.review(self.id, reviewer, decision, reason)
        return {"ok": True, **self.review_record}

    def close(self) -> dict:
        if self.state != "LOG_AND_CLOSE" and self.cfg.enforce_order:
            return {"ok": False, "error": f"close in LOG_AND_CLOSE (state: {self.state})"}
        rec = {"analysis_id": self.id, "question": self.question, "as_of": self.as_of, "focus": self.focus, "tickers": self.tickers,
               "gate": {k: self.gate[k] for k in ("verdict", "reasons")} if self.gate else None, "action": (self.final or {}).get("action"),
               "reference_action": (self.decision or {}).get("action"), "n_tool_calls": len(self.calls), "violations": self.violations,
               "committed_ids": (self.evidence or {}).get("committed_ids"), "human_review": self.review_record,
               "memory": self.db.recall(self.tickers[0]) if self.tickers else []}
        import json
        self.db.finish(self.id, final_state="LOG_AND_CLOSE", gate_verdict=(self.gate or {}).get("verdict"), action=(self.final or {}).get("action"),
                       reference_action=(self.decision or {}).get("action"), explanation=(self.final or {}).get("explanation"),
                       violations=json.dumps(self.violations, ensure_ascii=False), schema_version=SCHEMA_VERSION, record=json.dumps(rec, default=str, ensure_ascii=False))
        self.closed = True
        return {"ok": True, **rec}
