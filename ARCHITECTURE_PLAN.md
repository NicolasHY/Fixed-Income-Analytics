# Architecture Plan — Mapping to the 5-Layer GenAI Architecture

This document maps the current Company EM FI codebase onto the canonical
five-layer GenAI architecture and flags the tight-coupling points that must
be broken before a refactor.

## Current files at a glance

| Path | Role today |
|------|-----------|
| `app.py` | Streamlit dashboard (UI + analytics glue + chart prep) |
| `chatbot.py` | Local Ollama chat helper (system prompt + LangChain wiring) |
| `main.ipynb` | End-to-end pipeline: load → PCA/regime → VaR → briefings → outputs |
| `src/data_loader.py` | Raw-CSV loading, yield-change construction, portfolio P&L proxy |
| `src/pca_regime.py` | PCA, GMM regime detection, alert engine |
| `src/risk_free.py` | FRED API client for SOFR/€STR, on-disk cache |
| `src/ui_theme.py` | Streamlit CSS theming |
| `config/funds.yaml` | Country universe, fund defs, VaR/PCA/regime/alert params, briefing prompt + showcase dates |
| `data/raw/<Country>/…csv` | Investing.com yield downloads |
| `data/output/*.{csv,json,png,pdf}` | Pre-computed pipeline artifacts the dashboard reads |
| `private/{gemini,fred}_key.txt` | Plain-text secrets (gitignored) |
| `tests/` | Unit + integration tests over synthetic data |
| `export_pdf.py`, `run_dashboard.bat`, `Dockerfile` | Build / packaging |

## Mapping to the five layers

### 1. Application / API layer — *user-facing surface*

- `app.py` — Streamlit pages: Home, Pipeline Health, Data Load, PCA & Regime,
  VaR Engine, Portfolios, Alert History, Daily Briefings, Chatbot.
- `chatbot.py` — chat-page handler exposing `stream_chat(history)`.
- `export_pdf.py`, `run_dashboard.bat`, `Dockerfile` — delivery/packaging.

**Today's problem.** This layer also performs analytics: `app.py`
re-computes portfolio carry, `_quick_stats`, VaR overlays, and Plotly figures
inline. There is no stable "API" between the UI and the analytical engine —
the UI knows the shape of every CSV/JSON sidecar in `data/output/`.

### 2. Orchestration / Routing layer — *who calls what, with what context*

- `main.ipynb` is the de-facto orchestrator: it sequences ingestion → PCA →
  regime → alerts → VaR → briefings, threads config through each step, and
  writes sidecar artifacts.
- `chatbot._to_lc_messages` is the only real "router" in the codebase: it
  prepends the system prompt and maps `{role, content}` dicts onto LangChain
  message objects.
- No tool-routing, no retrieval-then-generate composition, no run-level
  logging beyond `data/output/pipeline_log.json` (built by the notebook).

**Today's problem.** Orchestration lives in notebook cells and Streamlit
page bodies — there is no callable orchestration module, so the same logic
(load → derive P&L → compute VaR) is re-implemented partially in `app.py`
(see `_load_portfolio_data`) and partially in `main.ipynb`.

### 3. Data / RAG layer — *facts the model and the analytics consume*

- **Structured data ingestion**: `src/data_loader.py` (yields, changes,
  portfolio definitions) and `src/risk_free.py` (FRED rates + on-disk cache
  at `data/output/risk_free_rates.csv`).
- **Derived artifacts**: everything in `data/output/` — country CSVs,
  `health_check.json`, `pipeline_log.json`, `alert_history.json`,
  `var_*.{csv,json,png}`, `regime_classification.png`, `pca_*.png`,
  `sample_briefings.json`.
- **Briefing "retrieval"**: `build_daily_payload(...)` inside the notebook
  hand-crafts a compact JSON of regime label, top alerts, PC moves, VaR
  numbers, etc., for a given date — this is the closest thing to a RAG
  retrieval step.
- `config/funds.yaml` doubles as both a config file and a knowledge base
  (macro events, showcase dates, briefing system prompt).

**Today's problem.** There is no retrieval interface — the briefing payload
is constructed inline in the notebook, so the chatbot (which legitimately
*would* benefit from grounding in the same facts) has no way to call it.
The chatbot system prompt admits this explicitly: *"If asked about specific
live numbers … say you don't have access to them."*

### 4. Models / Inference layer — *the actual LLM and analytical engines*

- **LLM inference (offline / scheduled)**: `main.ipynb` calls
  `genai.Client(api_key=…)` directly on Gemini 2.0-flash for showcase-date
  briefings; results are cached in `data/output/sample_briefings.json`.
- **LLM inference (interactive)**: `chatbot.py` calls a local
  `langchain_ollama.ChatOllama(model="qwen3.6")`.
- **Quant inference engines** (these are also "models" in the broad sense):
  - PCA + GMM in `src/pca_regime.py`
  - VaR engines (parametric / parametric-t / historical / Monte Carlo / stressed
    / factor-decomposition) — currently inlined in `main.ipynb` and partially
    duplicated in `tests/test_var.py`.

**Today's problem.** Two separate LLM clients (Gemini SDK in the notebook,
LangChain-Ollama in `chatbot.py`), each with its own hard-coded model name
and system prompt. There is no shared `LLMClient` interface, no provider
swap, no observability hook, and no place to add retries or token-budget
controls.

### 5. Infrastructure layer — *secrets, storage, runtime, CI*

- **Secrets**: `private/gemini_key.txt`, `private/fred_key.txt` — plain-text
  files read with `open(...)`.
- **Storage**: local filesystem only. `data/raw/` (inputs), `data/output/`
  (artifacts + LLM cache + risk-free cache).
- **Runtime**: Streamlit (`app.py`), Jupyter (`main.ipynb`), pytest.
- **Packaging**: `Dockerfile`, `run_dashboard.bat`, `requirements.txt`.
- **CI / observability**: none beyond the local `pipeline_log.json` /
  `health_check.json` written by the notebook.

**Today's problem.** Secrets and paths are read with literals scattered
across modules (e.g. `open("private/gemini_key.txt")` in the notebook,
`Path("data/output")` in `app.py`); there is no settings abstraction.

## Tight-coupling hotspots to break first

The refactor's first job is to drive a stake between layers that are
currently fused. In rough priority order:

1. **`app.py` ↔ analytical engine.** The Streamlit pages re-implement
   portfolio carry, quick stats, VaR table formatting, and figure
   construction. → Extract a thin `services/` layer (e.g.
   `services/portfolios.py`, `services/var.py`) returning plain dataframes
   so the UI only renders.

2. **`main.ipynb` ↔ everything.** The notebook is currently the
   orchestrator, the engine *and* the storage writer. → Move ingestion,
   PCA/regime, VaR and briefing generation into callable modules
   (`src/var_engine.py`, `src/briefing.py`, `src/orchestrator.py`) and let
   the notebook be a thin driver.

3. **LLM clients ↔ business code.** Both `chatbot.py` (LangChain-Ollama)
   and the notebook (Gemini SDK) instantiate provider-specific clients with
   hard-coded model names and prompts. → Introduce a single
   `src/llm_client.py` interface with `complete()` / `stream()` so providers
   are swappable and prompts/models come from config.

4. **Briefing payload ↔ notebook.** `build_daily_payload(...)` is the
   "retrieval" step but it lives in a notebook cell, so the chatbot can't
   reuse it. → Lift it into `src/briefing_context.py` (the Data/RAG layer)
   and expose it to both the scheduled briefing and the chat path.

5. **Secrets/paths ↔ literals.** Every module reads its own files with
   string paths. → Centralise into `src/settings.py` (key loading, output
   directory, model names) so we can swap to env-vars / a vault later.

6. **Sidecar JSON/CSV ↔ UI.** The dashboard depends on the *exact* schema
   of `var_stress_windows.json`, `var_multi_nu_table.csv`, etc., produced
   by the notebook. → Define typed result objects in the service layer and
   write the sidecars as a *serialisation* of those objects, not as the
   primary interface.

7. **Test helpers ↔ notebook logic.** `tests/test_var.py` re-implements
   VaR formulas because the notebook does not export them. → Once a real
   `src/var_engine.py` exists, the tests import it and the duplication
   disappears.

## Refactor sequencing (preview, not part of this task)

1. Lock current behaviour with characterization tests *(this PR)*.
2. Extract `src/var_engine.py`, `src/briefing_context.py`,
   `src/llm_client.py`, `src/settings.py` behind stable function
   signatures. Tests must stay green.
3. Rewire `app.py` and `main.ipynb` onto the new modules.
4. Add a real chatbot grounding path: chatbot pulls today's briefing
   payload through `briefing_context` instead of disclaiming.
