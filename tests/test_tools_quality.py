import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bist.tools.quality import check_data_quality  # noqa: E402


def codes(r):
    return {i["code"] for i in r["data"]["issues"]}


def test_clean_case_is_pass_with_warnings_only_because_macro_incomplete():
    r = check_data_quality(tickers=["THYAO"], as_of="2023-05-10", analysis_type="technical_reversal")
    from bist.tools import store
    if store.tcmb().empty:  # v1.1: with TCMB data built the clean case is a plain PASS
        assert r["data"]["verdict"] == "PASS_WITH_WARNINGS" and "MACRO_INCOMPLETE" in codes(r)
    else:
        assert r["data"]["verdict"] == "PASS" and "MACRO_INCOMPLETE" not in codes(r)
    assert r["data"]["may_issue_candidate_signal"] is True


def test_unknown_ticker_is_hard_block():
    r = check_data_quality(tickers=["THYAO", "FAKE1"], as_of="2023-05-10")
    assert r["data"]["verdict"] == "HARD_BLOCK" and "UNKNOWN_TICKER" in codes(r)
    assert r["data"]["may_issue_candidate_signal"] is False


def test_leakage_flag_blocks():
    r = check_data_quality(tickers=["THYAO"], as_of="2023-05-10",
                           evidence_dates=[{"source": "fundamentals THYAO 2023Q2", "date": "2023-08-10"}])
    assert r["data"]["verdict"] == "HARD_BLOCK" and "LEAKAGE" in codes(r)


def test_far_future_as_of_blocks_but_near_future_only_warns():
    assert check_data_quality(tickers=["THYAO"], as_of="2030-01-01")["data"]["verdict"] == "HARD_BLOCK"


def test_small_peer_group_needs_investigation_for_sector_analysis():
    r = check_data_quality(tickers=["ASELS"], as_of="2023-05-10", analysis_type="sector_catchup")
    assert r["data"]["verdict"] == "INVESTIGATE" and "PEER_GROUP_TOO_SMALL" in codes(r)
    # same stock, a non-sector analysis: not a problem
    assert "PEER_GROUP_TOO_SMALL" not in codes(check_data_quality(tickers=["ASELS"], as_of="2023-05-10", analysis_type="technical_reversal"))


def test_fundamental_reaction_without_public_report_is_unsafe():
    r = check_data_quality(tickers=["THYAO"], as_of="2014-06-01", analysis_type="fundamental_reaction")
    assert r["data"]["verdict"] in ("ANALYSIS_UNSAFE", "HARD_BLOCK")  # no report public (and/or no price history yet)


def test_tms29_period_gives_warning_for_fundamental_analysis():
    r = check_data_quality(tickers=["EREGL"], as_of="2025-06-30", analysis_type="fundamental_reaction")
    assert "TMS29_RESTATED_VALUES" in codes(r)


def test_late_filing_triggers_investigate():
    # BIMAS 2023Q4 was filed ~130 days late (published May 2024); right after publication the latest report is that late one
    r = check_data_quality(tickers=["BIMAS"], as_of="2024-05-31", analysis_type="fundamental_reaction")
    assert "LATE_FILING" in codes(r) and r["data"]["verdict"] in ("INVESTIGATE", "ANALYSIS_UNSAFE", "HARD_BLOCK")
