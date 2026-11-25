"""S1: the harness makes the most lagging selected ticker the main ticker after ranking; Harness.history; requested-date evidence."""
import pytest

from bist.harness import policy, reference
from bist.harness.logdb import DecisionLog

AS_OF = "2026-09-10"
WIN = {"start": "2024-01-01", "end": AS_OF}
RETAIL = ["BIMAS", "MGROS", "MAVI"]


@pytest.fixture()
def db(tmp_path):
    return DecisionLog(tmp_path / "d.sqlite")


def _rank(h):
    return next(c for c in h.calls if c["tool"] == "rank_sector_relative_strength")["envelope"]


def _laggard(env, selected):
    return next(m["ticker"] for m in env["data"]["ranking_lagging_first"] if m["ticker"] in set(selected))


def _codes(env):
    return {w["code"] for w in env.get("harness_notes", [])}


def test_q1_question_order_main_becomes_laggard(db):
    h = reference.run("Q1", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, peers=["MGROS", "MAVI"], horizon_days=20, **WIN)
    rank = _rank(h)
    laggard = _laggard(rank, RETAIL)
    assert h.tickers[0] == laggard
    catch = next(c for c in h.calls if c["tool"] == "test_catch_up_history")
    assert catch["envelope"]["params"]["ticker"] == laggard
    assert set(catch["envelope"]["params"]["peers"]) == set(RETAIL) - {laggard}
    assert ("MAIN_TICKER_SET_TO_LAGGARD" in _codes(rank)) == (laggard != "BIMAS")


def test_no_reorder_when_first_is_laggard(db):
    h = reference.run("Q1", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, peers=["MGROS", "MAVI"], horizon_days=20, **WIN)
    lag = h.tickers[0]
    rest = [t for t in RETAIL if t != lag]
    h2 = reference.run("Q1", lag, "sector_catchup", as_of=AS_OF, db=db, peers=rest, horizon_days=20, **WIN)
    assert h2.tickers[0] == lag
    assert "MAIN_TICKER_SET_TO_LAGGARD" not in _codes(_rank(h2))


def test_no_reorder_outside_sector_catchup_or_single_ticker(db):
    h = reference.run("q", "THYAO", "technical_reversal", as_of=AS_OF, db=db, **WIN)
    assert h.tickers == ["THYAO"]
    h2 = reference.run("q", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, **WIN)   # sector mode, one selected ticker
    assert h2.tickers == ["BIMAS"]
    assert "MAIN_TICKER_SET_TO_LAGGARD" not in _codes(_rank(h2))


def test_history_records_states(db):
    h = reference.run("q", "THYAO", "calendar_pattern", as_of=AS_OF, db=db, **WIN)
    pairs = {(s, t) for s, t, _ in h.history}
    assert ("CHECK_DATA_QUALITY", "check_data_quality") in pairs
    assert ("RUN_PATTERN_AND_REVERSAL_ANALYSIS", "test_weekday_pattern") in pairs
    assert all(t != "select_universe" for _, t, _ in h.history)
    assert len(h.history) == len(h.calls)


def test_requested_date_beyond_data_blocks(db):
    h = reference.run("q", "THYAO", "general", as_of="2027-03-01", db=db,
                      evidence_dates=[{"source": "requested analysis date", "date": "2027-03-01"}])
    assert policy.quality_verdict(h.calls) == "HARD_BLOCK"
    assert h.final["action"] == "ANALYSIS_UNSAFE"


def test_s1_quality_and_indicators_cover_every_selected_ticker(db):
    h = reference.run("Q1", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, peers=["MGROS", "MAVI"], horizon_days=20, **WIN)
    dq = next(c for c in h.calls if c["tool"] == "check_data_quality")["envelope"]["params"]
    assert set(dq["tickers"]) == set(RETAIL)
    assert {c["envelope"]["data"]["ticker"] for c in h.calls if c["tool"] == "calculate_indicators"} == set(RETAIL)


def _env(tool, data, codes=()):
    return {"tool": tool, "envelope": {"status": "WARN" if codes else "OK", "data": data, "warnings": [{"code": c, "message": c} for c in codes]}}


def test_signals_use_only_the_main_tickers_calls():
    rank = _env("rank_sector_relative_strength", {"ranking_valid": True, "sector_strong": True, "custom_group": True,
                                                   "ranking_lagging_first": [{"ticker": "MAVI", "lag_rank": 1}, {"ticker": "BIMAS", "lag_rank": 2}]})
    strong = {"ticker": "BIMAS", "n_events": 40, "by_horizon": {"h10": {"n": 40, "narrowed_rate": 0.9, "closed_rate": 0.6}}}
    hist_bimas = _env("test_catch_up_history", strong)
    ind_bimas = _env("calculate_indicators", {"ticker": "BIMAS", "states": {"rsi_zone": "oversold"}})
    s = policy.signals([rank, hist_bimas, ind_bimas], "MAVI", AS_OF)
    assert s["catchup"].get("history_ok") is False
    assert s["reversal"]["partial"] is False


def test_main_ticker_note_reaches_the_model(db):
    from bist.harness.llm_agent import render
    h = reference.run("Q1", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, peers=["MGROS", "MAVI"], horizon_days=20, **WIN)
    rank = _rank(h)
    if h.tickers[0] != "BIMAS":
        assert f"main ticker set to {h.tickers[0]}" in render("rank_sector_relative_strength", rank)


def test_replay_ok_after_reorder_in_sector_mode(db):
    from bist.harness.machine import Harness
    from bist.harness.replay import replay_analysis
    h = Harness("q", as_of=AS_OF, focus="sector_catchup", db=db)
    h.advance()
    h.select_universe(["MGROS", "BIMAS", "MAVI"])
    h.advance()
    h.advance()
    h.call("check_data_quality", tickers=["MGROS", "BIMAS", "MAVI"], analysis_type="sector_catchup")
    assert h.advance("RUN_SECTOR_RELATIVE_ANALYSIS")["ok"]
    env = h.call("rank_sector_relative_strength", sector="MGROS", horizon_days=20)
    assert env["status"] == "OK"
    assert replay_analysis(h.id, db)["mismatched_calls"] == []


def test_s1_loads_fundamentals_of_the_main_ticker(db):
    h = reference.run("Q1", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, peers=["MGROS", "MAVI"], horizon_days=20, **WIN)
    fm = [c["envelope"]["data"]["ticker"] for c in h.calls if c["tool"] in ("get_fundamental_metrics", "calculate_fundamental_change") and c["envelope"]["data"]]
    assert fm and set(fm) == {h.tickers[0]}
