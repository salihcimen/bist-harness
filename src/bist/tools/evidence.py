"""build_evidence_bundle: turns the recorded tool results of one analysis into

  observed  = every deterministic fact the tools returned (as plain-language statements WITH the numbers), and
  committed = the subset that may support a conclusion (stock-specific, sufficiently sampled, stable, not blocked by the gate).

The statements are the ONLY numbers the language model may quote: its explanation must be derivable from them.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from . import session
from .common import SMALL_N
from .envelope import tool, warn

BLOCKING = {"ANALYSIS_UNSAFE", "HARD_BLOCK"}


def _p(x, nd=1, sign=True):
    return "n/a" if x is None else f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def _n(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _item(kind, call, statement, values=None, commit=True, reason=None, ticker=None):
    return {"kind": kind, "ticker": ticker, "statement": statement, "values": values or {}, "source_call": call["call_id"],
            "source_tool": call["tool"], "as_of": call["envelope"].get("as_of"), "commit": commit, "reason": reason}


def _codes(env):
    return {w["code"] for w in env.get("warnings", [])}


def _indicators(c):
    d, t = c["envelope"]["data"], c["envelope"]["data"]["ticker"]
    v, s = d["values"], d["states"]
    out = []
    ab = lambda b: "above" if b else "below"
    if s.get("close_above_sma_50") is not None and s.get("close_above_sma_200") is not None:
        out.append(_item("technical", c, f"{t}: close {_n(v['close'])} is {ab(s['close_above_sma_50'])} SMA50 ({_n(v['sma_50'])}) and {ab(s['close_above_sma_200'])} SMA200 ({_n(v['sma_200'])}); SMA50 is {ab(s.get('sma50_above_sma200'))} SMA200.", {"close": v["close"]}, ticker=t))
    if s.get("supertrend_direction"):
        out.append(_item("technical", c, f"{t}: Supertrend is {s['supertrend_direction']}, flipped {s['supertrend_days_since_flip']} trading days ago.", ticker=t))
    out.append(_item("technical", c, f"{t}: RSI(14) is {_n(v.get('rsi'), 1)} ({s.get('rsi_zone')}); MACD histogram is {'positive' if s.get('macd_hist_positive') else 'negative'}{' with a signal-line cross in the last 3 bars' if s.get('macd_cross_last_3_bars') else ''}.", {"rsi": v.get("rsi")}, ticker=t))
    out.append(_item("technical", c, f"{t}: Bollinger %B is {_n(v.get('bb_pct_b'))} (zone: {s.get('bollinger_zone')}); close is {ab(s.get('close_above_kama'))} KAMA, KAMA slope {s.get('kama_slope_5d')}.", {"bb_pct_b": v.get("bb_pct_b")}, ticker=t))
    if s.get("ichimoku_price_vs_cloud"):
        out.append(_item("technical", c, f"{t}: price is {s['ichimoku_price_vs_cloud']} the Ichimoku cloud ({s['ichimoku_cloud_color']} cloud); Tenkan is {ab(s['ichimoku_tenkan_above_kijun'])} Kijun.", ticker=t))
    return out


def _reversal(c):
    d, env = c["envelope"]["data"], c["envelope"]
    t, out = d["ticker"], []
    ev = d["recent_events"]
    if not ev:
        out.append(_item("reversal", c, f"{t}: no reversal event in the recent window.", ticker=t))
    for e in ev:
        out.append(_item("reversal", c, f"{t}: {e['direction']} reversal event '{e['type']}' on {e['date']} (close {_n(e['close'])}).", {"type": e["type"], "date": e["date"]}, ticker=t))
    if d.get("window"):
        return out + _reversal_window(c, d, t)
    seen = {e["type"] for e in ev}
    for ty in seen:
        h = d["history_summary"][ty]["h5"]
        ok = h["n"] >= 10
        out.append(_item("reversal_history", c, f"{t}: historically after '{ty}' ({d['history_summary'][ty]['count']} occurrences in the lookback), the 5-day forward return averaged {_p(h['mean'])} (median {_p(h['median'])}, hit rate {_p(h['hit_rate'], 0, False)}, mean excess vs XU100 {_p(h['mean_excess_vs_xu100'])}, n={h['n']}).",
                         {"type": ty, "n": h["n"]}, commit=ok, reason=None if ok else "SMALL_SAMPLE (fewer than 10 completed observations)", ticker=t))
    return out


def _small(n):
    return None if n >= SMALL_N else f"SMALL_SAMPLE (fewer than {SMALL_N} completed observations)"


def _gaps(c, what):
    """One data_gap item per call for INCOMPLETE_FORWARD_WINDOW: a fact the answer must state (windows ending after the analysis date)."""
    msgs = [w["message"].split("(")[-1].rstrip(")") for w in c["envelope"].get("warnings", []) if w["code"] == "INCOMPLETE_FORWARD_WINDOW"]
    if not msgs:
        return []
    return [_item("data_gap", c, f"{what}: events whose forward window ends after the analysis date were excluded ({'; '.join(msgs[:4])}).", commit=True)]


def _hstats(s):
    return ", ".join(f"{h}d mean {_p(s[f'h{h}']['mean'])}" for h in (1, 3, 5, 10))


def _reversal_window(c, d, t):
    w0, w1 = d["window"]
    out = []
    for ty, s in d["history_summary"].items():
        h5 = s["h5"]
        out.append(_item("reversal_history", c, f"{t}: '{ty}' occurred {s['count']} times between {w0} and {w1}; forward returns {_hstats(s)}; within 5 days the price rose in {_p(h5['hit_rate'], 0, False)} of cases "
                                                 f"(median {_p(h5['median'])}, mean excess vs XU100 {_p(h5['mean_excess_vs_xu100'])}, n={h5['n']}, {h5['n_excluded_incomplete']} with incomplete windows excluded).",
                         {"type": ty, "n": h5["n"]}, commit=h5["n"] >= SMALL_N, reason=_small(h5["n"]), ticker=t))
        for cs in s.get("by_condition_set", []):
            name = " & ".join(cs["conditions"])
            h = cs["h5"]
            if cs["count"] == 0:
                out.append(_item("reversal_condition", c, f"{t}: no '{ty}' event between {w0} and {w1} occurred together with [{name}].", {"conditions": cs["conditions"]}, ticker=t))
                continue
            out.append(_item("reversal_condition", c, f"{t}: '{ty}' with [{name}]: {cs['count']} events; forward returns {_hstats(cs)}; within 5 days the price rose in {_p(h['hit_rate'], 0, False)} of cases (n={h['n']}).",
                             {"conditions": cs["conditions"], "n": h["n"]}, commit=h["n"] >= SMALL_N, reason=_small(h["n"]), ticker=t))
    return out + _gaps(c, f"{t} reversal statistics")


def _catchup(c):
    d = c["envelope"]["data"]
    t = d["ticker"]
    if d.get("by_horizon") is None:
        return [_item("catchup_history", c, f"{t}: catch-up history not computed (fewer than 2 peers).", commit=False, reason="PEER_GROUP_TOO_SMALL", ticker=t)]
    w0, w1 = d["window"]
    out = [_item("sector", c, f"{t}: current {d['lookback_days']}-day return vs the peer median ({', '.join(d['peers'])}) is {_p(d['current_lag_gap'])}.", {"lag_gap": d["current_lag_gap"]}, ticker=t)]
    for k, h in d["by_horizon"].items():
        out.append(_item("catchup_history", c, f"{t}: {d['n_events']} past lags of at least {_p(d['min_lag_gap_used'], 1, False)} behind the peer median between {w0} and {w1}; within {k[1:]} days the gap narrowed in "
                                               f"{_p(h['narrowed_rate'], 0, False)} and fully closed in {_p(h['closed_rate'], 0, False)} of cases (false positives {_p(h['false_positive_rate'], 0, False)}, "
                                               f"mean relative return {_p(h['mean_rel'])}, n={h['n']}, {h['n_excluded_incomplete']} incomplete windows excluded).",
                         {"horizon": k, "n": h["n"]}, commit=h["n"] >= SMALL_N, reason=_small(h["n"]), ticker=t))
    return out + _gaps(c, f"{t} catch-up history")


def _disclosures(c):
    d = c["envelope"]["data"]
    t, out = d["ticker"], []
    hz = [f"{h}d" for h in d["horizons"]]
    for r in d["rows"]:
        mets = "; ".join(f"{m} {_n(r['values'][m], 3)} (change vs previous quarter {_n(r['change_vs_prev_quarter'][m], 3)})" for m in d["metrics"])
        rets = ", ".join(f"{k} {_p(r['returns'][k]['abnormal_vs_xu100'])} vs XU100 / {_p(r['returns'][k]['abnormal_vs_sector'])} vs sector" if r["returns"][k]["complete"]
                         else f"{k} incomplete (window ends after the analysis date)" for k in hz)
        imp = {True: "improving", False: "deteriorating", None: "mixed/undetermined"}[r["improving"]]
        out.append(_item("disclosure", c, f"{t} {r['period']} first published on KAP {str(r['first_published'])[:10]} (tradable {r['effective_trade_date']}): {mets}; {imp}; abnormal returns {rets}.",
                         {"period": r["period"]}, ticker=t))
    for k in hz:
        s = d["summary"][k]
        n = s["improving"]["n"] + s["deteriorating"]["n"]
        out.append(_item("disclosure_summary", c, f"{t}: mean {k} abnormal return vs XU100 after improving quarters {_p(s['improving']['mean_abnormal_vs_xu100'])} (n={s['improving']['n']}) vs deteriorating quarters "
                                                  f"{_p(s['deteriorating']['mean_abnormal_vs_xu100'])} (n={s['deteriorating']['n']}); association only, no causal claim.",
                         {"horizon": k, "n": n}, commit=n >= SMALL_N, reason=_small(n), ticker=t))
    inc = [f"{r['period']} {k}" for r in d["rows"] for k in hz if not r["returns"][k]["complete"]]
    if inc:
        out.append(_item("data_gap", c, f"{t}: price-reaction window not complete as of the analysis date for {', '.join(inc)}; no later prices were used.", ticker=t))
    if "TMS29_RESTATED_VALUES" in _codes(c["envelope"]):
        out.append(_item("data_gap", c, f"{t}: fundamentals from FY2022 on are restated (TMS 29), not first-published values; changes are compared within the same era only.", ticker=t))
    return out


def _macro_sens(c):
    d = c["envelope"]["data"]
    t, f = d["ticker"], d["factor"]
    w0, w1 = d["window"]
    up, dn = d["factor_up"], d["factor_down"]
    scope = {True: "differs from the market (stock-specific)", False: "is similar to the market (not stock-specific)", None: "cannot be compared with the market"}[d["stock_specific"]]
    stmt = (f"{t} vs {f} ({d['frequency']}, {d['n_periods']} periods, {w0} to {w1}): correlation {_n(d['correlation'])}, beta {_n(d['beta'])} (t {_n(d['beta_t_stat'])}); "
            f"when {f} rose ({up['n']} periods) {t} averaged {_p(up['mean_stock_return'])} ({_p(up['mean_excess_vs_xu100'])} vs XU100), when it fell ({dn['n']} periods) {_p(dn['mean_stock_return'])}; "
            f"XU100 correlation {_n(d['xu100']['correlation'])}, so the relationship {scope}; yearly correlation sign stable: {d['sign_stable']}. Correlation is not causation; no forecast.")
    ok = d["n_periods"] >= SMALL_N and d["sign_stable"] is not False
    reason = None if ok else ("UNSTABLE_RELATIONSHIP" if d["sign_stable"] is False else _small(d["n_periods"]))
    return [_item("macro_sensitivity", c, stmt, {"factor": f, "n": d["n_periods"]}, commit=ok, reason=reason, ticker=t)]


def _sector(c):
    d = c["envelope"]["data"]
    label = f"Peer group ({', '.join(m['ticker'] for m in d.get('ranking_lagging_first', d.get('members', [])))})" if d.get("custom_group") else f"Sector {d['sector']}"
    if not d.get("ranking_valid"):
        return [_item("sector", c, f"{label}: peer group too small for a relative ranking.", commit=False, reason="PEER_GROUP_TOO_SMALL")]
    out = [_item("sector", c, f"{label} ({d['n_members']} stocks): median {d['horizon_days']}-day return {_p(d['sector_median_return'])} vs XU100 {_p(d['xu100_return'])}; sector is {'strong' if d['sector_strong'] else 'not strong'} relative to the index.", {"sector_strong": d["sector_strong"]})]
    for m in d["ranking_lagging_first"][:3]:
        out.append(_item("sector", c, f"{m['ticker']}: {d['horizon_days']}-day return {_p(m['return'])}, {_p(m['excess_vs_sector_median'])} vs sector median (lag score {_n(m['lag_score'])}, lag rank {m['lag_rank']} of {d['n_members']}).", {"lag_rank": m["lag_rank"]}, ticker=m["ticker"]))
    return out


def _pattern(c):
    d = c["envelope"]["data"]
    name = d["pattern"] + (f" {d['weekday']}" if d.get("weekday") else "")
    if d.get("down_days"):
        name = f"{'-'.join(d['down_days'])} down, then {'-'.join(d['up_days'])} return"
    stmt = (f"{d['ticker']}: pattern '{name}' occurred {d['occurrences']} times; mean {_p(d['mean'], 2)} vs baseline {_p(d['baseline_mean'], 2)}, hit rate {_p(d['hit_rate'], 0, False)}; "
            f"p-value after multiple-testing correction {_n(d['p_value_bonferroni'], 3)}; out-of-sample same sign: {d['oos_same_sign']}; years with same sign {d['years_same_sign_as_overall']}/{d['years_tested']}; "
            f"XU100 shows the same effect: {d['market_wide_effect']}.")
    verdict = ("stock-specific, significant and stable" if d["stock_specific_evidence"] else
               "market-wide effect, not stock-specific" if d["market_wide_effect"] else "not significant or not stable: treat as noise")
    stmt += f" Verdict: {verdict}."
    ok = bool(d["stock_specific_evidence"])
    reason = None if ok else ("MARKET_WIDE_EFFECT (not stock-specific)" if d["market_wide_effect"] else "NOT_SIGNIFICANT_OR_UNSTABLE")
    out = [_item("pattern", c, stmt, {"p_adj": d["p_value_bonferroni"]}, commit=ok, reason=reason, ticker=d["ticker"])]
    if not ok:  # the rejection itself must be quotable: descriptive facts + verdict, citable, never support for a candidate
        out.append(_item("pattern_stats", c, f"{d['ticker']}: pattern '{name}' descriptive statistics: {d['occurrences']} occurrences, mean {_p(d['mean'], 2)}, "
                                             f"median {_p(d.get('median'), 2)} vs unconditional {_p(d['baseline_mean'], 2)}, hit rate {_p(d['hit_rate'], 0, False)}. Verdict: {verdict}.",
                         {"verdict": verdict}, ticker=d["ticker"]))
    for key, lab in (("in_sample", "in-sample"), ("out_of_sample", "out-of-sample")):
        s = d.get(key)
        if not s:
            continue
        net = f", net of {_n(d['cost_bps_round_trip'], 0)} bps round-trip costs {_p(s['net_mean_after_cost'], 2)}" if s.get("net_mean_after_cost") is not None else ""
        out.append(_item("pattern", c, f"{d['ticker']}: pattern '{name}' {lab} {s['period'][0]} to {s['period'][1]}: {s['occurrences']} occurrences, mean {_p(s['mean'], 2)}, median {_p(s['median'], 2)} vs unconditional {_p(s['baseline_mean'], 2)}, "
                                    f"hit rate {_p(s['hit_rate'], 0, False)}{net}; p-value after multiple-testing correction {_n(s['p_value_bonferroni'], 3)}; overall significant and stable: {d['significant_and_stable']}.",
                         {"segment": key, "n": s["occurrences"]}, ticker=d["ticker"]))  # descriptive segment facts are citable; the verdict item above carries significance
    if not d.get("in_sample") and d.get("net_mean_after_cost") is not None:
        out.append(_item("pattern", c, f"{d['ticker']}: pattern '{name}' mean net of {_n(d['cost_bps_round_trip'], 0)} bps round-trip costs {_p(d['net_mean_after_cost'], 2)}.", ticker=d["ticker"]))
    return out


def _fundamentals(c):
    d, env = c["envelope"]["data"], c["envelope"]
    t, r, m = d["ticker"], d["report"], d["metrics"]
    codes = _codes(env)
    uncertain = bool(codes & {"LATE_FILING", "STALE_FUNDAMENTALS"})
    reason = "LATE_FILING/STALE_FUNDAMENTALS: publication timing uncertain" if uncertain else None
    if d["kind"] == "industrial":
        body = (f"ROE TTM {_p(m.get('roe_ttm'), 1, False)}, EBITDA margin {_p(m.get('ebitda_margin_q'), 1, False)}, net margin {_p(m.get('net_margin_q'), 1, False)}, "
                f"net debt/EBITDA {_n(m.get('net_debt_ebitda'))}, current ratio {_n(m.get('current_ratio'))}, cash conversion {_n(m.get('cash_conversion'))}")
    elif d["kind"] == "bank":
        body = f"ROE TTM {_p(m.get('roe_ttm'), 1, False)}, ROA TTM {_p(m.get('roa_ttm'), 2, False)}, loans/deposits {_n(m.get('loan_deposit'))}, equity/assets {_p(m.get('equity_assets'), 1, False)}"
    else:
        body = f"ROA TTM {_p(m.get('roa_ttm'), 2, False)}"
    era = " (inflation-accounting era: ratios only, restated source)" if r["era"] == "tms29_restated" else ""
    out = [_item("fundamental", c, f"{t}: latest public quarter {r['fiscal_year']}Q{r['quarter']} (first published {str(r['first_published'])[:10]}, effective {r['effective_trade_date']}){era}: {body}.", {"quarter": f"{r['fiscal_year']}Q{r['quarter']}"}, commit=not uncertain, reason=reason, ticker=t)]
    if d.get("valuation") and d["valuation"].get("pe_ttm") is not None:
        v = d["valuation"]
        tms = "VALUATION_TMS29_CAUTION" in codes
        out.append(_item("valuation", c, f"{t}: approximate P/E {_n(v['pe_ttm'], 1)}, P/B {_n(v.get('pb'))}, EV/EBITDA {_n(v.get('ev_ebitda_ttm'), 1)}.", commit=not tms and not uncertain, reason="VALUATION_TMS29_CAUTION" if tms else reason, ticker=t))
    return out


def _fchange(c):
    d, env = c["envelope"]["data"], c["envelope"]
    t, rc, r = d["ticker"], d["ratio_changes"], d["report"]
    uncertain = bool(_codes(env) & {"LATE_FILING", "STALE_FUNDAMENTALS"})
    parts = []
    for k, label, pct in (("ebitda_margin_q", "EBITDA margin", True), ("net_margin_q", "net margin", True), ("roe_ttm", "ROE TTM", True),
                          ("net_debt_ebitda", "net debt/EBITDA", False), ("current_ratio", "current ratio", False), ("roa_ttm", "ROA TTM", True), ("loan_deposit", "loans/deposits", False)):
        if k in rc and rc[k]["vs_prev_quarter"] is not None:
            dv = rc[k]["vs_prev_quarter"]
            parts.append(f"{label} {dv * 100:+.1f} pp" if pct else f"{label} {dv:+.2f}")
    g = d.get("growth_yoy")
    gtxt = f" YoY growth: revenue {_p(g.get('revenue_yoy'))}, EBITDA {_p(g.get('ebitda_yoy'))}, net income {_p(g.get('ni_yoy'))}." if g else " YoY growth is not reported for this period (inflation-accounting restatement)."
    return [_item("fundamental_change", c, f"{t}: {r['fiscal_year']}Q{r['quarter']} vs previous quarter ({d['compared_with']['previous_quarter']}): " + ("; ".join(parts) or "no comparable ratios") + "." + gtxt,
                  commit=not uncertain, reason="LATE_FILING/STALE_FUNDAMENTALS" if uncertain else None, ticker=t)]


def _macro(c):
    d = c["envelope"]["data"]["series"]
    out = []
    u, fx = d.get("usdtry"), d.get("eurtry")
    k = next((x for x in (u or {}) if x.startswith("change_")), None)
    if u:
        out.append(_item("macro", c, f"USD/TRY {_n(u['value'])} (observed {u['observation_date']}), {k.replace('change_', '').replace('_pct', '')} change {_p(u.get(k))}; EUR/TRY {_n(fx['value'])}.", {"usdtry": u["value"]}))
    f = d.get("fed_target_upper")
    if f:
        low = d.get("fed_target_lower")
        out.append(_item("macro", c, f"Fed funds target range {_n(low['value'])}-{_n(f['value'])}% (last change {str(f.get('last_change_date'))[:10]}); US 10Y yield {_n((d.get('us10y') or {}).get('value'))}%.", {"fed_upper": f["value"]}))
    v = d.get("vix")
    if v:
        out.append(_item("macro", c, f"VIX {_n(v['value'], 1)} ({c['envelope']['data']['regime']['vix_regime']} regime); DXY {_n((d.get('dxy') or {}).get('value'), 1)}; Brent {_n((d.get('brent') or {}).get('value'), 1)}.", {"vix": v["value"]}))
    pr, cpi, ut = d.get("policy_rate"), d.get("cpi_yoy"), d.get("usdtry_tcmb")
    if pr or cpi or ut:
        parts = []
        if ut:
            parts.append(f"CBRT USD/TRY {_n(ut['value'], 4)} (observed {ut['observation_date']})")
        if pr:
            parts.append(f"CBRT weighted average funding cost (policy-rate proxy) {_n(pr['value'])}% (observed {pr['observation_date']})")
        if cpi:
            parts.append(f"CPI inflation {_n(cpi['value'])}% y/y (month ending {cpi['observation_date']})")
        out.append(_item("macro", c, "TCMB: " + "; ".join(parts) + ".", {"policy_rate": (pr or {}).get("value")}))
    for name in ("usdtry", "usdtry_tcmb", "policy_rate", "cpi_yoy"):
        r = (d.get(name) or {}).get("range")
        if r:
            out.append(_item("macro", c, f"{name} from {r['first_date']} to {r['last_date']}: {_n(r['start_value'], 4)} -> {_n(r['end_value'], 4)} (change {_n(r['change'], 4)}, {_p(r['change_pct'])}), min {_n(r['min'], 4)}, max {_n(r['max'], 4)}, {r['n_obs']} observations.",
                             {"series": name}))
    miss = ", ".join(c["envelope"]["data"]["unavailable_series"])
    if miss:
        out.append(_item("macro_gap", c, f"Not available: {miss}. No statement about Turkish policy rates or inflation can be made.", commit=True))
    return out


def _backtest(c):
    d, env = c["envelope"]["data"], c["envelope"]
    if not d.get("signal_level"):
        return [_item("backtest", c, f"Backtest '{d['spec']['name']}': no trades.", commit=False, reason="NO_TRADES")]
    s, te, p = d["signal_level"], d["by_period"]["test"], d["portfolio"]
    bad = _codes(env) & {"FEW_SIGNALS", "SMALL_TEST_SAMPLE", "UNSTABLE_OUT_OF_SAMPLE"}
    stmt = (f"Backtest '{d['spec']['name']}' ({str(d['period'][0])[:10]} to {str(d['period'][1])[:10]}, entry at next open, round-trip cost {d['assumptions']['round_trip_cost_bps']:.0f} bps): "
            f"{s['n_trades']} trades, mean net return per trade {_p(s['mean_net_return'], 2)}, win rate {_p(s['win_rate'], 0, False)}, mean excess vs XU100 {_p(s['mean_excess_vs_xu100'], 2)}; "
            f"out-of-sample test period: {te['n_trades']} trades, mean {_p(te.get('mean_net_return'), 2)}; portfolio Sharpe {_n(p['sharpe_annualized_rf0'])}, max drawdown {_p(p['max_drawdown'], 0, False)}.")
    return [_item("backtest", c, stmt, {"variant": d["spec"]["name"], "n_trades": s["n_trades"]}, commit=not bad, reason=", ".join(sorted(bad)) or None)]


def _quality(c):
    d = c["envelope"]["data"]
    n = d["n_issues_by_severity"]
    top = [f"{i['code']}({i['scope']})" for i in d["issues"] if i["severity"] != "warning"][:4]
    return [_item("quality", c, f"Data-quality gate verdict: {d['verdict']} ({n['warning']} warnings, {n['investigate']} investigate, {n['unsafe']} unsafe, {n['block']} block){'; ' + ', '.join(top) if top else ''}.", {"verdict": d["verdict"]})]


EXTRACTORS = {"test_catch_up_history": _catchup, "analyze_disclosure_reactions": _disclosures, "analyze_macro_sensitivity": _macro_sens,
              "calculate_indicators": _indicators, "detect_reversal_events": _reversal, "rank_sector_relative_strength": _sector,
              "test_weekday_pattern": _pattern, "get_fundamental_metrics": _fundamentals, "calculate_fundamental_change": _fchange,
              "get_macro_context": _macro, "run_backtest": _backtest, "check_data_quality": _quality}


class BuildEvidenceParams(BaseModel):
    analysis_id: str = Field(description="Id of the running analysis whose recorded tool results are turned into evidence.")


@tool("build_evidence_bundle", BuildEvidenceParams,
      "Builds the observed evidence set (all deterministic facts with numbers) and the committed evidence set (stock-specific, sufficiently sampled, stable, gate-approved) from the tool results recorded for an analysis.")
def build_evidence_bundle(p: BuildEvidenceParams):
    log = session.calls(p.analysis_id)
    sess_as_of = session.session_as_of(p.analysis_id)
    items, verdict = [], None
    for c in log:
        env = c["envelope"]
        if env["status"] == "ERROR":
            items.append(_item("tool_error", c, f"Tool {c['tool']} failed: {env['error']['code']} - {env['error']['message']}", commit=False, reason="TOOL_ERROR"))
            continue
        fn = EXTRACTORS.get(c["tool"])
        if fn:
            items += fn(c)
        if c["tool"] == "check_data_quality":
            verdict = env["data"]["verdict"]
    if sess_as_of:
        for it in items:
            if it["as_of"] and it["as_of"] > sess_as_of:
                it["commit"], it["reason"] = False, "LEAKAGE: evidence dated after the analysis date"
    gate_msg = None
    if verdict is None:
        gate_msg = "check_data_quality was not run: nothing can be committed"
    elif verdict in BLOCKING:
        gate_msg = f"data-quality verdict {verdict}: nothing can be committed"
    for it in items:
        if gate_msg and it["kind"] != "quality":
            it["commit"], it["reason"] = False, gate_msg
    for i, it in enumerate(items, 1):
        it["id"] = f"E{i}"
    committed = [it for it in items if it["commit"]]
    excluded = [{"id": it["id"], "reason": it["reason"]} for it in items if not it["commit"]]
    w = []
    if gate_msg:
        w.append(warn("EVIDENCE_BLOCKED", gate_msg))
    data = {"analysis_id": p.analysis_id, "quality_verdict": verdict, "n_observed": len(items), "n_committed": len(committed),
            "observed": items, "committed_ids": [it["id"] for it in committed], "excluded": excluded}
    return data, w, {"as_of": sess_as_of, "sources": ["recorded tool results of the analysis"]}
