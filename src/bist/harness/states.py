"""The 17 workflow states (START .. LOG_AND_CLOSE), which MCP tools each state exposes, and which states can never be skipped."""
from __future__ import annotations

STATES = ["START", "SELECT_UNIVERSE", "LOAD_MARKET_DATA", "CHECK_DATA_QUALITY", "CALCULATE_TECHNICALS", "RUN_SECTOR_RELATIVE_ANALYSIS",
          "RUN_PATTERN_AND_REVERSAL_ANALYSIS", "LOAD_FUNDAMENTALS", "LOAD_MACRO_CONTEXT", "LOAD_NEWS_AND_SPEECH_CONTEXT", "BUILD_EVIDENCE",
          "PREVIEW_STRATEGIES", "RUN_BACKTEST", "RISK_GATE", "GENERATE_EXPLANATION", "HUMAN_REVIEW", "LOG_AND_CLOSE"]

ALLOWED_TOOLS: dict[str, list[str]] = {
    "START": [],
    "SELECT_UNIVERSE": [],
    "LOAD_MARKET_DATA": ["get_stock_history", "get_index_history"],
    "CHECK_DATA_QUALITY": ["check_data_quality"],
    "CALCULATE_TECHNICALS": ["calculate_indicators"],
    "RUN_SECTOR_RELATIVE_ANALYSIS": ["rank_sector_relative_strength", "test_catch_up_history"],
    "RUN_PATTERN_AND_REVERSAL_ANALYSIS": ["detect_reversal_events", "find_support_resistance", "test_weekday_pattern"],
    "LOAD_FUNDAMENTALS": ["get_fundamental_metrics", "calculate_fundamental_change", "analyze_disclosure_reactions"],
    "LOAD_MACRO_CONTEXT": ["get_macro_context", "analyze_macro_sensitivity"],
    "LOAD_NEWS_AND_SPEECH_CONTEXT": ["get_news_events", "get_speech_features"],
    "BUILD_EVIDENCE": ["build_evidence_bundle"],
    "PREVIEW_STRATEGIES": ["generate_strategy_variants"],
    "RUN_BACKTEST": ["run_backtest"],
    "RISK_GATE": [],
    "GENERATE_EXPLANATION": [],
    "HUMAN_REVIEW": [],
    "LOG_AND_CLOSE": [],
}

# A mandatory state can not be skipped; it can only be left after its required tool was called.
MANDATORY = {"SELECT_UNIVERSE", "CHECK_DATA_QUALITY", "BUILD_EVIDENCE", "RUN_BACKTEST", "RISK_GATE", "GENERATE_EXPLANATION", "HUMAN_REVIEW", "LOG_AND_CLOSE"}
REQUIRED_TOOL = {"CHECK_DATA_QUALITY": "check_data_quality", "BUILD_EVIDENCE": "build_evidence_bundle", "RUN_BACKTEST": "run_backtest"}
# Optional analysis states that the quality gate makes meaningless once the data are blocked.
ALL_TOOLS = sorted({t for v in ALLOWED_TOOLS.values() for t in v})
