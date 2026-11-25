# BIST 100 Agentic Financial Analytics Harness (CSE-481)

Educational project: 30 BIST stocks, deterministic analysis tools (frozen schema v1.0), a stateful harness
(17 states, quality gate, backtest, risk gate, human review, SQLite decision log, replay), an MCP server,
and a planned QLoRA fine-tune of a small local LLM (Qwen3-4B-Instruct-2507, served via Ollama) that calls the tools and explains only from tool output.
Nothing here is investment advice; no tool creates orders.

- `src/bist/tools/` deterministic MCP tools + envelope/provenance
- `src/bist/harness/` state machine, policy/validator, decision log, replay, MCP server, LLM agent, survey
- `src/bist/scenarios.py` research scenarios S1–S4; results in `reports/` and `docs/research_notes.md`
- `ROADMAP.md` phases and status; `CLAUDE.md` working notes

## Setup
```
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```
Market/fundamental data (`data/`) and logs are **not** in the repo (third-party sources: Yahoo/Is Yatirim/KAP).
Rebuild them with the fetch scripts in `src/bist/` (see ROADMAP Faz 1); tests that need data will fail without it.
API keys go in `.env` (see `.env.example`), never in git.
