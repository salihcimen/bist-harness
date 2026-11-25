# Base model baseline: instructor's 5 sample questions (2026-11-21)

Model: `qwen3:4b-instruct-2507-q4_K_M` (Ollama, temperature 0, num_ctx 16384), **no fine-tune**.
Agent: `LLMAgent` v2 (`src/bist/harness/llm_agent.py`). The model picks tickers, focus, analysis date and data period itself
from the original question text. No `as_of` or `focus` is passed in.
Questions: `docs/sample_test_questions.txt` (original wording). Hardware: RTX 4060 Laptop 8 GB.

| Q | Selection (tickers / focus / date) | Tool calls / ERROR | Final action | Reference action | Validator | Time |
|---|---|---|---|---|---|---|
| Q1 retail catch-up | BIMAS, MGROS, MAVI / sector_catchup / 2026-09-10 ✅ | 33 / 9 | REJECT_SIGNAL | (varies) | passed on 2nd try (1st: candidate not permitted) | ~2 min |
| Q2 THYAO weekday | THYAO / calendar_pattern / 2026-09-10 ✅ | 18 / 6 | REJECT_SIGNAL | REJECT_SIGNAL | passed | 131 s |
| Q3 ASELS Bollinger | ASELS / technical_reversal / 2026-09-10 ✅ | 6 / 1 | REJECT_SIGNAL | HOLD | passed; mandatory tool run by harness | 85 s |
| Q4 TUPRS disclosures | TUPRS / fundamental_reaction / 2026-09-10 ✅ | 22 / 4 | INVESTIGATE | WATCH | passed | 187 s |
| Q5 EREGL advice | EREGL / general / 2026-09-10 ✅ | 16 / 3 | INVESTIGATE | WATCH | passed, no advice | 172 s |

## What the base model does well
- In all 5 questions it picks the ticker(s), focus and analysis date correctly.
- It extracts the question's data period (in Q2 it splits IS and OOS into two calls).
- It avoids investment-advice language, and the validator accepts every final answer (the first run had 1 fallback; the second run had 0).

## Errors fine-tuning must fix (Faz 5 targets)
1. **Invented or out-of-range parameters:** `calculate_indicators(config=…)` with a wrong tuple length, `get_stock_history(max_rows=1000)` (the limit is 120), `analyze_disclosure_reactions(peers=["TUPRS"])` (its own ticker), and peers outside the universe ("HUBER", "TURK"). These make up 20–30% of all calls.
2. **Missing the key tool or parameter for the question:**
   - Q3: `detect_reversal_events` without `start`/`end` and with the wrong condition (`rsi_oversold` instead of `rsi_trough`), or the tool not called at all.
   - Q2: `oos_start` not used.
   - Q1: `rank_sector_relative_strength` given a wrong `sector` together with `tickers`.
3. **Answer content:**
   - Q2/Q3 claim "there is no data" even though the evidence exists.
   - Q4 writes a quarter that does not exist ("2026Q4"); the numeric check does not catch this.
   - Q5 does not use the USD/TRY sensitivity result.
4. **Unneeded tools:** every state is "filled in" (support/resistance, speech, backtest in two windows), which costs time and context.

## Fixes made in this round (before the second run)
- **Ollama HTTP 400:** the cause was context overflow (more than 16k tokens). Earlier states' tool results are now sent as their first line only, and the explanation step sees only the committed evidence.
- **Gate always INVESTIGATE:** an invalid-parameter error that a later successful call of the same tool recovers no longer trips the gate. An unrecovered one gives only PASS_WITH_WARNINGS (`TOOL_CALL_INVALID`).
- **Rejected patterns now quotable:** the statistics of a non-significant pattern are evidence the answer may quote (`pattern_stats`, the verdict is stated in the sentence).
