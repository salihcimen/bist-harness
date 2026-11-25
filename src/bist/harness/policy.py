"""Risk gate, permitted-action policy, reference decision, explanation template and the output validator.

Everything here is deterministic and reads only the tool envelopes recorded for the analysis. The reference decision
(`decide`) is the label source for the training data and the oracle for evaluation; the validator is what turns the LLM
into a constrained component: it may only emit one of 8 actions, only an action permitted by the gate and the evidence,
and may quote only numbers that exist in the evidence (no advice wording, no uncommitted evidence cited).
"""
from __future__ import annotations

import re

ACTIONS = ["HOLD", "WATCH", "INVESTIGATE", "POTENTIAL_CATCH_UP_CANDIDATE", "TECHNICAL_REVERSAL_CANDIDATE",
           "FUNDAMENTAL_IMPROVEMENT_CANDIDATE", "REJECT_SIGNAL", "ANALYSIS_UNSAFE"]
CANDIDATES = {"POTENTIAL_CATCH_UP_CANDIDATE", "TECHNICAL_REVERSAL_CANDIDATE", "FUNDAMENTAL_IMPROVEMENT_CANDIDATE"}
VERDICT_RANK = {"PASS": 0, "PASS_WITH_WARNINGS": 1, "INVESTIGATE": 2, "ANALYSIS_UNSAFE": 3, "HARD_BLOCK": 4}
BLOCKING = {"ANALYSIS_UNSAFE", "HARD_BLOCK"}
FRESH_DAYS = 20  # a fundamental improvement is "new information" for this many calendar days after it became tradable
MAX_DRAWDOWN_WARN = 0.5


def _ok(env: dict) -> bool:
    return env.get("status") in ("OK", "WARN") and env.get("data") is not None


def _last(calls: list[dict], tool: str) -> dict | None:
    for c in reversed(calls):
        if c["tool"] == tool:
            return c["envelope"]
    return None


def _last_for(calls: list[dict], tool: str, ticker: str) -> dict | None:
    """Last call of `tool` about `ticker` (results without a ticker field count for any ticker)."""
    for c in reversed(calls):
        if c["tool"] == tool:
            t = (c["envelope"].get("data") or {}).get("ticker")
            if t is None or t == ticker:
                return c["envelope"]
    return None


def _codes(env: dict | None) -> set[str]:
    return {w["code"] for w in (env or {}).get("warnings", [])}


# ------------------------------------------------------------------ gate
def quality_verdict(calls: list[dict]) -> str | None:
    vs = [c["envelope"]["data"]["verdict"] for c in calls if c["tool"] == "check_data_quality" and _ok(c["envelope"])]
    return max(vs, key=VERDICT_RANK.get) if vs else None


def backtest_view(calls: list[dict]) -> dict | None:
    """Condensed view of the last backtest: SUPPORTS / REFUTES / INCONCLUSIVE and the numbers behind it."""
    env = _last(calls, "run_backtest")
    if env is None:
        return None
    if not _ok(env):
        return {"ran": True, "error": env["error"]["code"], "verdict": "UNAVAILABLE"}
    d = env["data"]
    codes = _codes(env)
    if not d.get("signal_level"):
        return {"ran": True, "variant": d["spec"]["name"], "verdict": "INCONCLUSIVE", "flags": sorted(codes), "n_trades": 0}
    te, s, p = d["by_period"]["test"], d["signal_level"], d["portfolio"]
    test_mean = te.get("mean_net_return")
    if "UNSTABLE_OUT_OF_SAMPLE" in codes or (test_mean is not None and te["n_trades"] >= 10 and test_mean < 0):
        verdict = "REFUTES"
    elif codes & {"FEW_SIGNALS", "SMALL_TEST_SAMPLE"} or test_mean is None:
        verdict = "INCONCLUSIVE"
    else:
        # a strategy only "supports" a candidate if it is positive out-of-sample AND beats XU100 per trade (nominal-TL drift makes raw positives meaningless)
        ex = s.get("mean_excess_vs_xu100")
        verdict = "SUPPORTS" if (test_mean > 0 and ex is not None and ex > 0) else "REFUTES"
    return {"ran": True, "variant": d["spec"]["name"], "verdict": verdict, "flags": sorted(codes), "n_trades": s["n_trades"], "test_mean": test_mean,
            "excess_vs_xu100": s.get("mean_excess_vs_xu100"), "max_drawdown": p.get("max_drawdown"), "sharpe": p.get("sharpe_annualized_rf0")}


# Errors caused by the caller's parameters (not by the data): a later successful call of the same tool recovers them.
MODEL_INPUT_ERRORS = {"INVALID_PARAMS", "MISSING_PARAM", "INVALID_DATE_RANGE", "INVALID_PERIOD", "UNKNOWN_METRIC", "UNKNOWN_EVENT_TYPE",
                      "UNKNOWN_SECTOR", "UNKNOWN_TICKER", "TOOL_NOT_ALLOWED_IN_STATE", "TICKER_NOT_IN_SELECTION", "AS_OF_AFTER_ANALYSIS_DATE"}


def risk_gate(calls: list[dict], require_backtest: bool = True) -> dict:
    q = quality_verdict(calls)
    reasons: list[str] = []
    if q is None:
        return {"verdict": "ANALYSIS_UNSAFE", "reasons": ["QUALITY_NOT_RUN: no data-quality check was performed"], "quality": None, "backtest": None}
    verdict = q
    for c in calls:
        if c["tool"] == "check_data_quality" and _ok(c["envelope"]):
            reasons += [f"{i['code']}({i['severity']})" for i in c["envelope"]["data"]["issues"] if i["severity"] != "warning"][:5]
    bt = backtest_view(calls)

    def bump(v):
        nonlocal verdict
        if VERDICT_RANK[v] > VERDICT_RANK[verdict]:
            verdict = v

    if verdict not in BLOCKING:
        if bt is None:
            if require_backtest:
                bump("INVESTIGATE"); reasons.append("BACKTEST_NOT_RUN")
        elif bt.get("verdict") == "UNAVAILABLE":
            bump("INVESTIGATE"); reasons.append(f"BACKTEST_UNAVAILABLE({bt['error']})")
        else:
            for f in bt.get("flags", []):
                if f in ("UNSTABLE_OUT_OF_SAMPLE", "FEW_SIGNALS", "SMALL_TEST_SAMPLE"):
                    bump("PASS_WITH_WARNINGS"); reasons.append(f)
            if bt.get("max_drawdown") is not None and bt["max_drawdown"] < -MAX_DRAWDOWN_WARN:
                bump("PASS_WITH_WARNINGS"); reasons.append(f"HIGH_DRAWDOWN({bt['max_drawdown'] * 100:.0f}%)")
        succeeded = {c["tool"] for c in calls if _ok(c["envelope"])}
        for c in calls:
            env = c["envelope"]
            if env["status"] != "ERROR" or c["tool"] in ("get_news_events", "get_speech_features", "run_backtest"):
                continue
            if (env.get("error") or {}).get("code") in MODEL_INPUT_ERRORS:
                if c["tool"] not in succeeded:   # the agent never got a valid call through: an analysis step is missing, data is fine
                    bump("PASS_WITH_WARNINGS"); reasons.append("TOOL_CALL_INVALID")
            else:
                bump("INVESTIGATE"); reasons.append("TOOL_ERROR")
    return {"verdict": verdict, "reasons": sorted(set(reasons)), "quality": q, "backtest": bt}


# ------------------------------------------------------------------ signals
CATCHUP_MIN_N = 10          # POTENTIAL_CATCH_UP_CANDIDATE needs >= this many completed 10-day windows of similar past lags
CATCHUP_MIN_NARROWED = 0.5  # ... and the gap must have narrowed in more than this share of them


def catchup_history_ok(env: dict | None) -> bool:
    if not env or not _ok(env):
        return False
    h = (env["data"].get("by_horizon") or {}).get("h10") or {}
    return (h.get("n") or 0) >= CATCHUP_MIN_N and (h.get("narrowed_rate") or 0) > CATCHUP_MIN_NARROWED


def signals(calls: list[dict], ticker: str, as_of: str | None) -> dict:
    s = {"fund": {"present": False, "partial": False, "deterioration": False}, "reversal": {"present": False, "partial": False},
         "catchup": {"present": False, "partial": False}, "pattern": {"present": False, "market_wide": False, "tested": False}}
    fc = _last_for(calls, "calculate_fundamental_change", ticker)
    if fc and _ok(fc) and not (_codes(fc) & {"LATE_FILING", "STALE_FUNDAMENTALS"}):
        d = fc["data"]
        rc = d["ratio_changes"]
        dv = lambda k: (rc.get(k) or {}).get("vs_prev_quarter")
        roe = dv("roe_ttm") if "roe_ttm" in rc else dv("roa_ttm")
        others = [dv(k) for k in ("ebitda_margin_q", "net_margin_q", "roa_ttm", "equity_assets") if dv(k) is not None]
        nd = dv("net_debt_ebitda")
        up = (roe is not None and roe > 0)
        confirm = sum(1 for x in others if x > 0) + (1 if nd is not None and nd < 0 else 0)
        down = (roe is not None and roe < 0)
        eff = d["report"].get("effective_trade_date")
        fresh = True
        if eff and as_of:
            import pandas as pd
            fresh = (pd.Timestamp(as_of) - pd.Timestamp(eff)).days <= FRESH_DAYS
        s["fund"].update(present=bool(up and confirm >= 1 and fresh), partial=bool(up and not (confirm >= 1 and fresh)), deterioration=bool(down))
    rv = _last_for(calls, "detect_reversal_events", ticker)
    if rv and _ok(rv):
        d = rv["data"]
        bull = [e for e in d["recent_events"] if e["direction"] == "bullish"]
        if bull:
            good = False
            for ty in {e["type"] for e in bull}:
                h = d["history_summary"].get(ty, {}).get("h5", {})
                if h.get("n", 0) >= 10 and (h.get("mean_excess_vs_xu100") or -1) > 0 and (h.get("hit_rate") or 0) >= 0.5:
                    good = True
            s["reversal"].update(present=True, history_ok=good)
        elif d["recent_events"]:
            s["reversal"]["partial"] = True
    ind = _last_for(calls, "calculate_indicators", ticker)
    if ind and _ok(ind) and not s["reversal"]["present"] and ind["data"]["states"].get("rsi_zone") == "oversold":
        s["reversal"]["partial"] = True
    sec = _last(calls, "rank_sector_relative_strength")
    if sec and _ok(sec) and sec["data"].get("ranking_valid"):
        d = sec["data"]
        me = next((m for m in d["ranking_lagging_first"] if m["ticker"] == ticker), None)
        if me and me["lag_rank"] <= 2:
            hist_ok = catchup_history_ok(_last_for(calls, "test_catch_up_history", ticker))
            present = bool(d["sector_strong"]) and hist_ok
            s["catchup"].update(present=present, partial=not present, history_ok=hist_ok)
    for c in calls:
        if c["tool"] == "test_weekday_pattern" and _ok(c["envelope"]):
            d = c["envelope"]["data"]
            s["pattern"]["tested"] = True
            if d["stock_specific_evidence"]:
                s["pattern"]["present"] = True
            elif d["market_wide_effect"]:
                s["pattern"]["market_wide"] = True
    return s


# ------------------------------------------------------------------ decision
def decide(calls: list[dict], ticker: str, focus: str, as_of: str | None, gate: dict | None = None) -> dict:
    """Reference decision: returns {action, permitted, rationale}. permitted = every action the validator will accept."""
    g = gate or risk_gate(calls)
    v = g["verdict"]
    if v in BLOCKING:
        return {"action": "ANALYSIS_UNSAFE", "permitted": ["ANALYSIS_UNSAFE"], "rationale": f"gate {v}", "gate": g}
    if v == "INVESTIGATE":
        return {"action": "INVESTIGATE", "permitted": ["INVESTIGATE", "WATCH", "REJECT_SIGNAL"], "rationale": "gate INVESTIGATE", "gate": g}
    S = signals(calls, ticker, as_of)
    bt = g["backtest"] or {}
    supported = bt.get("verdict") == "SUPPORTS"
    refuted = bt.get("verdict") == "REFUTES"
    cands, rejected, partial = [], [], []
    if S["fund"]["present"]:
        cands.append("FUNDAMENTAL_IMPROVEMENT_CANDIDATE")
    elif S["fund"]["deterioration"] and focus == "fundamental_reaction":
        rejected.append("fund_deterioration")
    elif S["fund"]["partial"]:
        partial.append("fund")
    if S["reversal"]["present"]:
        (cands if (S["reversal"].get("history_ok") and supported) else rejected if (focus == "technical_reversal" and (refuted or not S["reversal"].get("history_ok"))) else partial).append(
            "TECHNICAL_REVERSAL_CANDIDATE" if (S["reversal"].get("history_ok") and supported) else "reversal")
    elif S["reversal"]["partial"]:
        partial.append("reversal")
    if S["catchup"]["present"]:
        (cands if supported else rejected if (focus == "sector_catchup" and (refuted or bt.get("verdict") == "INCONCLUSIVE")) else partial).append(
            "POTENTIAL_CATCH_UP_CANDIDATE" if supported else "catchup")
    elif S["catchup"]["partial"]:
        partial.append("catchup")
    if S["pattern"]["present"]:
        partial.append("pattern")
    # contradiction: a bullish candidate while the latest fundamentals deteriorate
    if cands and cands != ["FUNDAMENTAL_IMPROVEMENT_CANDIDATE"] and S["fund"]["deterioration"]:
        return {"action": "INVESTIGATE", "permitted": ["INVESTIGATE", "WATCH", "REJECT_SIGNAL", "HOLD"], "rationale": "technical/sector signal contradicts deteriorating fundamentals", "gate": g}
    permitted = ["HOLD", "WATCH", "INVESTIGATE", "REJECT_SIGNAL"] + cands
    order = {"fundamental_reaction": ["FUNDAMENTAL_IMPROVEMENT_CANDIDATE"], "technical_reversal": ["TECHNICAL_REVERSAL_CANDIDATE"],
             "sector_catchup": ["POTENTIAL_CATCH_UP_CANDIDATE"]}.get(focus, [])
    if cands:
        pick = next((x for x in order if x in cands), None) or next((x for x in ["FUNDAMENTAL_IMPROVEMENT_CANDIDATE", "TECHNICAL_REVERSAL_CANDIDATE", "POTENTIAL_CATCH_UP_CANDIDATE"] if x in cands))
        if focus != "general" and not any(x in cands for x in order):
            # candidate exists only for a different hypothesis than the one asked about: the asked hypothesis has no support
            return {"action": "REJECT_SIGNAL" if focus in ("technical_reversal", "sector_catchup", "fundamental_reaction") else "WATCH", "permitted": permitted,
                    "rationale": f"asked hypothesis ({focus}) is not supported; other candidate: {pick}", "gate": g}
        return {"action": pick, "permitted": permitted, "rationale": "supported candidate", "gate": g}
    if rejected:
        return {"action": "REJECT_SIGNAL", "permitted": permitted, "rationale": f"hypothesis not supported ({', '.join(rejected)})", "gate": g}
    if focus == "calendar_pattern" and S["pattern"]["market_wide"] and not S["pattern"]["present"]:
        return {"action": "REJECT_SIGNAL", "permitted": permitted, "rationale": "calendar pattern is a market-wide effect, not stock-specific", "gate": g}
    if focus == "calendar_pattern" and S["pattern"]["tested"] and not S["pattern"]["present"]:
        return {"action": "REJECT_SIGNAL", "permitted": permitted, "rationale": "calendar pattern not significant/stable after multiple-testing correction", "gate": g}
    if partial:
        return {"action": "WATCH", "permitted": permitted, "rationale": f"partial signals only ({', '.join(partial)})", "gate": g}
    return {"action": "HOLD", "permitted": permitted, "rationale": "no actionable signal", "gate": g}


# ------------------------------------------------------------------ explanation + validation
MAX_EVIDENCE_LINES = 9  # reference explanation: committed facts quoted (plus up to MAX_GAP_LINES data-gap notes)
MAX_GAP_LINES = 2
DISCLAIMER = "Historical evidence only; educational analytical state, not investment advice and not a trade instruction."
_NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CITE = re.compile(r"\bE(\d+)\b")
_ADVICE = re.compile(r"\b(you should (buy|sell)|i recommend|we recommend|buy now|sell now|price target|guaranteed|guarantee[sd]?|will (certainly |definitely )?(rise|go up|fall|drop|double)|sure (profit|win)|risk[- ]free|"
                     r"almalısın\w*|satmalısın\w*|alın\b|satın\b|tavsiye ederim|garanti(li)?\b|kesin (yüksel|kazan|düş)\w*|yatırım yap(ın|malısın)\w*)", re.I)
_DISCLAIM_STRIP = re.compile(r"(not (an? )?(investment|financial) advice|not (an? )?trade instruction|yatırım tavsiyesi değildir|yatırım tavsiyesi (içermez|niteliğinde değildir))", re.I)


def _numbers(text: str) -> set[float]:
    t = _DATE.sub(" ", text)
    out = set()
    for m in _NUM.findall(t):
        out.add(abs(float(m.replace("−", "-").replace(",", ".").lstrip("+"))))
    return out


def allowed_numbers(bundle: dict, question: str = "", extra: list[str] | None = None) -> set[float]:
    pool = [question, *(extra or [])]
    for it in bundle["observed"]:
        pool.append(it["statement"])
        if it.get("reason"):
            pool.append(it["reason"])
    return set().union(*(_numbers(x) for x in pool)) if pool else set()


def build_explanation(action: str, bundle: dict, gate: dict, rationale: str = "") -> str:
    """Reference explanation: cites committed evidence ids with the tool's own statements; quotes nothing else."""
    obs = {it["id"]: it for it in bundle["observed"]}
    committed = [obs[i] for i in bundle["committed_ids"]]
    order = ["fundamental_change", "fundamental", "disclosure_summary", "catchup_history", "macro_sensitivity", "reversal_condition", "reversal_history",
             "reversal", "sector", "pattern", "pattern_stats", "disclosure", "technical", "backtest", "valuation", "macro", "macro_gap", "quality"]
    gaps = [it for it in committed if it["kind"] == "data_gap"][:MAX_GAP_LINES]
    committed = [it for it in committed if it["kind"] != "data_gap"]
    committed.sort(key=lambda it: order.index(it["kind"]) if it["kind"] in order else 99)
    lines = [f"Educational analytical state: {action}."]
    if committed or gaps:
        lines.append("Evidence used:")
        lines += [f"[{it['id']}] {it['statement']}" for it in committed[:MAX_EVIDENCE_LINES] + gaps]
    else:
        lines.append("No evidence was committed for this conclusion.")
    kind = {it["id"]: it["kind"] for it in bundle["observed"]}
    notes = list(dict.fromkeys(f"{kind[x['id']]} ({x['reason']})" for x in bundle["excluded"] if x["reason"] and not x["reason"].startswith("data-quality verdict")))[:4]
    if notes:
        lines.append("Observed but not used as evidence: " + "; ".join(notes) + ".")
    if gate["reasons"] or gate["verdict"] != "PASS":
        lines.append(f"Risk gate: {gate['verdict']}" + (f" ({', '.join(gate['reasons'])})" if gate["reasons"] else "") + ".")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def validate_output(action: str, explanation: str, permitted: list[str], bundle: dict, question: str = "", extra: list[str] | None = None) -> dict:
    """Returns {ok, violations[]}; violations are machine-readable codes (used as metrics: unsupported-conclusion, numeric hallucination, ...)."""
    v: list[str] = []
    if action not in ACTIONS:
        v.append(f"ACTION_NOT_ALLOWED:{action}")
    elif action not in permitted:
        v.append(f"ACTION_NOT_PERMITTED_BY_EVIDENCE_OR_GATE:{action}")
    committed = set(bundle["committed_ids"])
    all_ids = {it["id"] for it in bundle["observed"]}
    cited = {f"E{n}" for n in _CITE.findall(explanation)}
    bad = sorted(c for c in cited if c in all_ids and c not in committed)
    if bad:
        v.append("CITES_UNCOMMITTED_EVIDENCE:" + ",".join(bad))
    ghost = sorted(c for c in cited if c not in all_ids)
    if ghost:
        v.append("CITES_UNKNOWN_EVIDENCE:" + ",".join(ghost))
    if action in CANDIDATES and not (cited & committed):
        v.append("CANDIDATE_WITHOUT_COMMITTED_EVIDENCE")
    body = _CITE.sub(" ", explanation)
    unknown = sorted(n for n in _numbers(body) if n not in allowed_numbers(bundle, question, extra) and not any(abs(n - a) < 1e-9 for a in allowed_numbers(bundle, question, extra)))
    if unknown:
        v.append("NUMBER_NOT_IN_EVIDENCE:" + ",".join(f"{x:g}" for x in unknown[:6]))
    if _ADVICE.search(_DISCLAIM_STRIP.sub(" ", explanation)):
        v.append("INVESTMENT_ADVICE_LANGUAGE")
    return {"ok": not v, "violations": v}
