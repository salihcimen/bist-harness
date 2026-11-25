# Faz 5 Training Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate ~2,500–3,500 per-turn SFT examples (`finetune/data/{train,val,test}.jsonl`) from ~300 template questions by replaying the deterministic reference agent through the real `LLMAgent` loop.

**Architecture:** For each template question, `reference.run` produces the correct tool calls (with their states) and the label conclusion. A `ScriptedChat` then plays the model inside an unmodified `LLMAgent.run`, returning those calls state by state; every `chat(messages, tools)` call it receives is recorded, so each training example's input is exactly the compressed `context()` the real model sees at inference. One small harness change makes S1 ("which stock lagged most?") work: after ranking, the harness sets the main ticker to the most lagging selected ticker.

**Tech Stack:** Python 3.11 (`.venv`), pandas, transformers tokenizer (Qwen/Qwen3-4B-Instruct-2507, local HF cache), pytest.

**Spec:** `docs/superpowers/specs/2026-11-23-faz5-training-data-design.md`

## Global Constraints
- Run: `cd /c/Projects/econ/bist-harness && PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m <module>`; tests `... -m pytest tests -q`. Set `BIST_DECISION_DB` to a temp file in generation and tests.
- `mcp/tool_schemas.json` (v1.1) must not change (`tests/test_schema_freeze.py` must stay green).
- No cloud API, no LLM paraphrase (approach A: templates only).
- Every label passes `policy.validate_output`; no tool output dated after the analysis date in any example.
- The 5 original sample questions (wording, and ticker + 2024-01-01–2026-09-10 combination) never appear in train/val.
- Test tickers `TCELL, FROTO, ENJSA, PGSUS`; test analysis-date window 2023-01-01–2023-06-30; last 2 templates of each (scenario, lang) are test-only. ~15% of questions are test; val = 10% of the rest, split by question.
- Negatives ~20–25% of questions; TR/EN roughly 50/50; S1–S5 balanced (±5%) among non-negatives.
- Every example ≤ 16,384 tokens under the Qwen chat template (with tools).
- Long Python files are written with the Write tool, not bash heredocs.

## Review Focus
- S1 when the first-named ticker already is the laggard → no reorder and no `MAIN_TICKER_SET_TO_LAGGARD` warning (test in Task 1).
- `rank_sector_relative_strength` in sector mode with a single selected ticker, or in a non-`sector_catchup` focus → main ticker unchanged (test in Task 1).
- A template with an unfilled `{slot}` or a ticker not in `data/universe.csv` → caught at question build time (test in Task 2).
- The advice-refusal prefix rejected by the validator → label falls back to the plain reference explanation instead of a fallback/failed trace (test in Task 3).
- One question raising inside a tool during generation → skipped and counted in `stats.json`, run continues; re-running resumes from cached traces (test in Task 4).

---

### Task 1: Harness — S1 main ticker = laggard; state log; quality evidence dates

**Files:**
- Modify: `src/bist/harness/machine.py` (`Harness.__init__`, `Harness.call`)
- Modify: `src/bist/harness/reference.py` (`plan`, `run`)
- Modify: `tests/test_harness_v11.py` (`SAMPLE_RUNS` Q1)
- Test: `tests/test_harness_laggard.py`

**Interfaces:**
- Produces: `Harness.history: list[tuple[str, str, dict]]` — `(state, tool, params)` of every call that reached `session.run` (not permission violations), in order; params as passed by the caller.
- Produces: warning code `MAIN_TICKER_SET_TO_LAGGARD` appended to the `rank_sector_relative_strength` envelope's `warnings` (so `render` and evidence show it) with message `main ticker set to {new} (most lagging selected ticker; was {old})`.
- Produces: `reference.plan(..., evidence_dates: list[dict] | None = None)` — appended to the `check_data_quality` params' `evidence_dates` (after the existing analysis-window-end entry).
- Produces: in `reference.run`, for `focus == "sector_catchup"`, every planned call after `rank_sector_relative_strength` whose `ticker` equals the original main uses `h.tickers[0]` instead, and `test_catch_up_history.peers` becomes `h.tickers[1:]`.

- [ ] **Step 1: Write the failing tests** in `tests/test_harness_laggard.py` (fixture `db` as in `tests/test_harness_v11.py`, `AS_OF = "2026-09-10"`, `WIN = {"start": "2024-01-01", "end": AS_OF}`):

```python
def test_q1_question_order_main_becomes_laggard(db):
    h = reference.run("Q1", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, peers=["MGROS", "MAVI"], horizon_days=20, **WIN)
    rank = next(c for c in h.calls if c["tool"] == "rank_sector_relative_strength")["envelope"]
    laggard = next(m["ticker"] for m in rank["data"]["ranking_lagging_first"] if m["ticker"] in {"BIMAS", "MGROS", "MAVI"})
    assert h.tickers[0] == laggard
    catch = next(c for c in h.calls if c["tool"] == "test_catch_up_history")
    assert catch["envelope"]["params"]["ticker"] == laggard
    assert set(catch["envelope"]["params"]["peers"]) == {"BIMAS", "MGROS", "MAVI"} - {laggard}
    assert ("MAIN_TICKER_SET_TO_LAGGARD" in {w["code"] for w in rank["warnings"]}) == (laggard != "BIMAS")

def test_no_reorder_when_first_is_laggard(db):   # put the laggard first -> no warning
    ...  # run once to find the laggard, run again with it first; assert tickers[0] unchanged and no MAIN_TICKER_SET_TO_LAGGARD

def test_no_reorder_outside_sector_catchup_or_single_ticker(db):
    h = reference.run("q", "THYAO", "technical_reversal", as_of=AS_OF, db=db, **WIN)
    assert h.tickers == ["THYAO"]
    h2 = reference.run("q", "BIMAS", "sector_catchup", as_of=AS_OF, db=db, **WIN)   # sector mode, one selected ticker
    assert h2.tickers == ["BIMAS"]

def test_history_records_states(db):
    h = reference.run("q", "THYAO", "calendar_pattern", as_of=AS_OF, db=db, **WIN)
    assert ("CHECK_DATA_QUALITY", "check_data_quality") in {(s, t) for s, t, _ in h.history}
    assert all(t != "select_universe" for _, t, _ in h.history)

def test_requested_date_beyond_data_blocks(db):
    h = reference.run("q", "THYAO", "general", as_of="2027-03-01", db=db,
                      evidence_dates=[{"source": "requested analysis date", "date": "2027-03-01"}])
    assert policy.quality_verdict(h.calls) == "HARD_BLOCK" and h.final["action"] == "ANALYSIS_UNSAFE"
```

- [ ] **Step 2: Run** `... -m pytest tests/test_harness_laggard.py -q` — Expected: FAIL (`history` missing, `evidence_dates` unexpected kwarg, Q1 main stays BIMAS).

- [ ] **Step 3: Implement.** In `Harness.call`, after `session.run` returns a non-ERROR envelope for `rank_sector_relative_strength` with `self.focus == "sector_catchup"` and `len(self.tickers) > 1`: take the first `ranking_lagging_first` entry whose ticker is in `self.tickers`; if it differs from `self.tickers[0]`, move it to the front and append the warning to `env["warnings"]` (check that `self.calls[-1]["envelope"]` is the same object; if not, update both). Append to `self.history` for every call that reached `session.run`. In `reference.plan` add `evidence_dates`; in `reference.run` apply the ticker/peers substitution described in Interfaces.

- [ ] **Step 4: Update `SAMPLE_RUNS` Q1** in `tests/test_harness_v11.py` to `("Q1 ...", "BIMAS", "sector_catchup", {"peers": ["MGROS", "MAVI"], "horizon_days": 20, **WIN})` (question order), and any Q1-specific assertion that assumed `MAVI` as passed ticker to use `h.tickers[0]`.

- [ ] **Step 5: Run** `... -m pytest tests -q` — Expected: all pass (149 + new).

- [ ] **Step 6: Commit** `feat(harness): S1 ana hisse en geride kalana gecer; Harness.history; plan evidence_dates`

---

### Task 2: Question templates and sampling

**Files:**
- Create: `src/bist/finetune/__init__.py`, `src/bist/finetune/templates.py`
- Test: `tests/test_finetune_templates.py`

**Interfaces:**
- Produces:
```python
@dataclass
class Question:
    id: str                 # "q0001"
    scenario: str           # "S1".."S5"
    lang: str               # "tr" | "en"
    template_id: str        # "S2-tr-03"
    text: str
    tickers: list[str]      # question order, canonical (no .IS)
    focus: str              # reference/LLMAgent focus
    as_of: str              # analysis date passed to reference.run (<= last data day, except beyond_data)
    requested_as_of: str    # date as written in the question (== as_of unless beyond_data)
    start: str | None
    end: str | None         # tool end (<= as_of)
    requested_end: str | None  # period end as written (> as_of for future_period)
    plan_kw: dict           # extra reference.plan kwargs (peers, horizon_days, pattern, weekday, pattern_kw, reversal_kw, macro_factor, evidence_dates)
    negative: str | None    # None | advice | beyond_data | future_period | short_history | small_peer_group | missing_number | skip_quality
    split: str              # "train" | "val" | "test"
TEMPLATES: dict[tuple[str, str], list[str]]   # (scenario, lang) -> >= 10 templates; last 2 are test-only
NEG_SUFFIX: dict[tuple[str, str], list[str]]  # (negative, lang) -> phrases appended / templates for negatives
TEST_TICKERS = ("TCELL", "FROTO", "ENJSA", "PGSUS")
TEST_WINDOW = ("2023-01-01", "2023-06-30")
def build_questions(n: int = 300, seed: int = 0) -> list[Question]
```

Slot rules (spec §3): S1 3–4 tickers from one `peer_group` (or a mixed custom group), `horizon_days` ∈ {5,10,20}; S2 `pattern` ∈ {weekday_return, bounce_after_down_day, bounce_after_n_down_days, weekday_sequence}, ~half of weekday_sequence with `oos_start`; S3 `event_types` 1–2 from the schema list (incl. `bollinger_lower_touch`/`bollinger_upper_touch`), 0–2 `condition_sets` of 1–2 conditions from the schema enum, `cooldown_days` ∈ {0, default}; S4 fundamental_reaction (metric names in the text: FAVÖK marjı / EBITDA margin, net borç/FAVÖK / net debt/EBITDA); S5 `general` + `macro_factor` from the schema enum. Analysis dates 2019-01-02–2026-11-18 (weekday, ≥25% exactly 2026-09-10); period start 6 months–3 years before; date formats mixed (`10.09.2026`, `2026-09-10`, `10 Eylül 2026`, `Sept 10, 2026`, "son N işlem günü"/"last N trading days"). Negatives (spec §4): advice/missing_number/skip_quality = a normal question + phrase; beyond_data = requested date 2027-01..2027-06, `as_of` = that date (harness clamps), `plan_kw["evidence_dates"] = [{"source": "requested analysis date", "date": requested_as_of}]`; future_period = `requested_end` 1–3 months after `as_of`, `end = as_of`; short_history = as_of in 2015-02..2015-06; small_peer_group = ASELS, `sector_catchup`, no peers. Mix: ~22% negatives (advice ≈ 1/3 of them), split as in Global Constraints: train/val questions use no test ticker, no test-window date, no test-only template; each test question has at least one of the three.

- [ ] **Step 1: Write the failing tests** in `tests/test_finetune_templates.py`:
```python
qs = build_questions(300, seed=0)
def test_count_and_ids_unique(): assert len(qs) == 300 and len({q.id for q in qs}) == 300
def test_no_unfilled_slots(): assert not any(re.search(r"\{\w*\}", q.text) for q in qs)
def test_tickers_in_universe(): assert all(t in set(store.tickers()) for q in qs for t in q.tickers)
def test_templates_per_lang(): assert all(len(TEMPLATES[(s, l)]) >= 10 for s in ("S1","S2","S3","S4","S5") for l in ("tr","en"))
def test_mix(): # negatives 20-25%, tr share 40-60%, test share 12-18%, each S1..S5 within ±5 pts of 20% of non-negatives
def test_split_holdout(): # train/val: no TEST_TICKERS, no as_of in TEST_WINDOW, no test-only template; test: >=1 of the three
def test_sample_questions_not_in_train(): # none of the 5 texts in docs/sample_test_questions.txt; no train/val q with (start,as_of)==("2024-01-01","2026-09-10") and tickers[0] in the sample's ticker for its scenario
def test_deterministic(): assert [q.text for q in build_questions(50, 1)] == [q.text for q in build_questions(50, 1)]
def test_end_not_after_as_of(): assert all(q.end is None or q.end <= q.as_of for q in qs if q.negative != "beyond_data")
```
- [ ] **Step 2: Run** `... -m pytest tests/test_finetune_templates.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `templates.py` per the interface and slot rules.
- [ ] **Step 4: Run** the test file — Expected: PASS.
- [ ] **Step 5: Commit** `feat(finetune): soru sablonlari ve ornekleme`

---

### Task 3: ScriptedChat — reference replay inside LLMAgent

**Files:**
- Create: `src/bist/finetune/scripted.py`
- Test: `tests/test_finetune_scripted.py`

**Interfaces:**
- Consumes: `Question` (Task 2); `Harness.history` (Task 1); `reference.run`; `LLMAgent`.
- Produces:
```python
ERROR_KINDS: dict[str, Callable]  # name -> (history, question) -> (state, bad_tool, bad_params) | None
class ScriptedChat:
    def __init__(self, select_args: dict, history: list[tuple[str, str, dict]], final: dict, inject: str | None = None): ...
    def __call__(self, messages: list[dict], tools: list[dict] | None = None, json_format: bool = False) -> dict: ...
    turns: list[dict]   # {"messages": deep copy of input, "tools": tools, "reply": returned dict, "kind": str, "train": bool}
def reference_label(q: Question, db) -> tuple[Harness, dict]   # runs reference.run with q's fields; returns (h, final {"action","explanation"})
def replay(q: Question, db, inject: str | None = None) -> tuple[Harness, ScriptedChat]   # runs LLMAgent(chat=ScriptedChat(...)).run(q.text, human=("APPROVE", "training trace"))
```
Behavior (spec §1–2):
- `select_args = {"tickers": q.tickers, "focus": q.focus, "analysis_date": q.requested_as_of, "period_start": q.start, "period_end": q.requested_end or q.end}` (drop None).
- State turns are recognized like `tests/test_llm_agent.py::FakeModel` (`"SELECT_UNIVERSE"`, `State (\w+)\.`, `json_format`). Per state: one turn with all `history` calls for that state (excluding `build_evidence_bundle`) **plus `next_state` in the same turn**; no calls → a turn with only `next_state`. Exceptions get their own earlier turn without `next_state`: `rank_sector_relative_strength` when the state also has `test_catch_up_history` (the catch-up call comes in the next turn), and the injected bad call.
- Kinds: `select`, `tool_call`, `next_state`, `final`, `recovery` (the turn after an injected error); the injected bad turn has `train=False`, all others `train=True`.
- `ERROR_KINDS` (each must produce an ERROR envelope from the harness/tool; drop any that does not, verified by test): `max_rows` (get_stock_history `max_rows=1000`), `bad_condition` (detect_reversal_events condition name `rsi_minimum`), `unknown_peer` (rank_sector_relative_strength `tickers` + `"HUBER"`), `self_peer` (analyze_disclosure_reactions `peers=[ticker]`). Applicable only if the history contains that tool.
- Final: `{"action", "explanation"}` from `reference_label`; for `q.negative == "advice"` prefix the explanation with `"Investment advice, buy/sell instructions and price targets are not provided; the following is historical evidence only.\n"` only if `policy.validate_output` accepts it on the reference harness; otherwise no prefix.
- Consistency (raise `TraceMismatch`): `h.final["action"] == final["action"]`, `not h.final.get("fallback")`, `h.violations == []` (except none expected), quality verdict equal to the reference's.

- [ ] **Step 1: Write the failing tests** (`db` fixture; use one cheap question per case, e.g. THYAO calendar_pattern 2026-09-10):
```python
def test_replay_matches_reference(db): h, chat = replay(q_s2, db); assert h.final["action"] == reference_label(q_s2, db)[1]["action"]
def test_turn_inputs_are_agent_context(db): # every recorded turn's messages[0]["role"] == "system" and the last message is a user/tool message; final turn kind == "final" and its messages contain "GENERATE_EXPLANATION"
def test_s1_two_turns(db): # BIMAS/MGROS/MAVI q: the rank turn has no next_state; next turn calls test_catch_up_history for h.tickers[0]
def test_each_error_kind_errors_then_recovers(db, kind): # parametrize over ERROR_KINDS: a train=False turn, followed by a train=True "recovery" turn; tool result after the bad turn starts with f"{tool}: ERROR"; final action == reference
def test_advice_prefix_or_fallback(db): # advice q: final explanation starts with the prefix iff validate_output accepts it; h.final not fallback
def test_beyond_data_unsafe(db): # beyond_data q: final action ANALYSIS_UNSAFE
```
- [ ] **Step 2: Run** `... -m pytest tests/test_finetune_scripted.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `scripted.py`.
- [ ] **Step 4: Run** the test file — Expected: PASS.
- [ ] **Step 5: Commit** `feat(finetune): ScriptedChat ile referans izinin LLMAgent icinde tekrari`

---

### Task 4: Generator, export format, dataset checks

**Files:**
- Create: `src/bist/finetune/export.py`, `src/bist/finetune/generate.py`
- Test: `tests/test_finetune_generate.py`

**Interfaces:**
- Consumes: `build_questions` (Task 2), `replay`, `ERROR_KINDS`, `TraceMismatch` (Task 3).
- Produces:
```python
def to_hf_messages(messages: list[dict]) -> list[dict]   # Ollama format -> Qwen/HF: tool_calls [{"type":"function","function":{"name","arguments": dict}}], tool msgs {"role":"tool","name":..,"content":..}
def example_rows(q: Question, chat: ScriptedChat) -> list[dict]  # one row per train=True turn (schema below)
def token_len(row: dict, tok) -> int     # tok.apply_chat_template(row["messages"], tools=[t["function"] for t in row["tools"]], tokenize=True)
def validate_dataset(out_dir: Path) -> list[str]  # [] if all checks pass, else problem strings
def main(n: int = 300, seed: int = 0, out_dir: Path = Path("finetune/data"), error_share: float = 0.10) -> dict  # returns stats
# CLI: python -m bist.finetune.generate --n 300 --seed 0 [--out finetune/data] [--check-only]
```
Row schema (spec §6): `{"id": f"{q.id}-t{k:02d}", "question_id", "scenario", "lang", "kind", "negative", "split", "messages": to_hf_messages(turn.messages + [reply as assistant msg]), "tools": turn.tools or []}`.
Generation: per question, cache the rows in `finetune/data/traces/{q.id}.json` (skip if present → resumable); ~`error_share` of train questions (seeded) get an applicable error kind; exceptions/`TraceMismatch` → question skipped and listed in `stats["skipped"]` with the reason. Writes `questions.jsonl` (asdict(Question) + `ref_action`), `train.jsonl`, `val.jsonl`, `test.jsonl`, `stats.json` (counts by split/scenario/lang/kind/negative/ref_action, skipped, token max/p95). Tokenizer: `AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)` with `HF_HOME=.cache/hf`.
`validate_dataset` checks (spec §7): every `final` row's action ∈ `policy.ACTIONS` and its question's trace had no violations (recorded in cache); no date > the question's `as_of` appears in any tool message (regex `\d{4}-\d{2}-\d{2}`, except dates that also appear in the question text / user prompt); test tickers absent from train/val rows; the 5 sample texts absent; tokens ≤ 16384; negatives 20–25% and scenario balance per Global Constraints.

- [ ] **Step 1: Write the failing tests:**
```python
def test_to_hf_messages_roundtrip(): # an Ollama-style assistant msg with tool_calls arguments dict and a tool msg with tool_name -> HF shape; tokenizer applies without error
def test_small_run(tmp_path, monkeypatch): # main(n=6, seed=3, out_dir=tmp_path) -> train/val/test files exist, every row's last message is assistant, validate_dataset(tmp_path) == []
def test_resume_uses_cache(tmp_path): # second main() call with same args does not call replay (monkeypatch replay to raise) and gives the same stats
def test_failing_question_is_skipped(tmp_path, monkeypatch): # replay raises for one id -> stats["skipped"] has it, others written
```
- [ ] **Step 2: Run** `... -m pytest tests/test_finetune_generate.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `export.py` (to_hf_messages, token_len, example_rows) and `generate.py` (main, validate_dataset, CLI).
- [ ] **Step 4: Run** the test file, then the full suite — Expected: PASS.
- [ ] **Step 5: Commit** `feat(finetune): veri uretici, HF formati, veri seti kontrolleri`

---

### Task 5: Full generation, documentation, roadmap

**Files:**
- Create: `docs/finetune_data.md`
- Modify: `ROADMAP.md` (Faz 5 boxes; Faz 1 EVDS box → [x]), `CLAUDE.md` (test count, finetune package line), `.gitignore` (`finetune/data/traces/`)

- [ ] **Step 1: Run the full generation in the background:** `BIST_DECISION_DB=$TEMP/gen.sqlite HF_HOME=.cache/hf PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m bist.finetune.generate --n 300 --seed 0` — Expected: `stats.json` with 2,500–3,500 train+val+test rows, skipped ≤ 5% of questions. If rows fall outside the range, adjust `--n` (not the per-turn rules) and rerun (cache makes it incremental).
- [ ] **Step 2: Run** `... -m bist.finetune.generate --check-only` — Expected: prints `OK` (validate_dataset returned []).
- [ ] **Step 3: Write `docs/finetune_data.md`:** pipeline (reference → ScriptedChat → LLMAgent context), row schema, the numbers from `stats.json` (per split/scenario/lang/kind/negative/action, token max/p95, skipped with reasons), hold-out rules, negative types and expected behavior, the S1 laggard rule, the beyond-data `evidence_dates` rule (deviation from spec: harness clamps as_of, so the requested date is passed as evidence → LEAKAGE → HARD_BLOCK), how to regenerate.
- [ ] **Step 4: Update ROADMAP** Faz 5 items to [x] with the real counts, Faz 1 EVDS box to [x]; **CLAUDE.md** test count and a line for `src/bist/finetune/` + generate command.
- [ ] **Step 5: Run** `... -m pytest tests -q` — Expected: all pass.
- [ ] **Step 6: Commit** `feat(finetune): Faz 5 egitim verisi uretildi; dokumantasyon` — commit `finetune/data/{train,val,test,questions}.jsonl` + `stats.json` only if total size < 50 MB, else add them to `.gitignore` and note the regeneration command in the doc.
