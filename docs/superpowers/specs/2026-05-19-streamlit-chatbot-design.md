# Streamlit Chatbot Page — Design

**Date:** 2026-05-19
**Status:** Approved (design phase)

## Goal

Add a basic chatbot to the Streamlit dashboard, backed by a locally-installed
LLM via Ollama. No data leaves the user's machine. The existing standalone
script `chatbot.py` has two blocking bugs and is CLI-only — it needs to be
refactored into a module the Streamlit app can import.

## Scope

In:

- New "Chatbot" page in the sidebar nav (9th entry in `PAGES`).
- Token-streamed responses (Ollama streaming → `st.write_stream`).
- Per-session conversation history (`st.session_state`).
- Fixed system prompt describing the platform and the user (analyst).
- Two-file change: refactor `chatbot.py`, append a page block to `app.py`.

Out:

- Live data access (no injection of VaR results, alerts, briefings).
- Cross-session persistence.
- Model switcher, temperature/top-p controls.
- Streaming cancellation, edit/regenerate, multi-conversation tabs.
- The CLI `__main__` entry point in `chatbot.py` (removed — module only).

## Bugs in current `chatbot.py`

1. `response.content` — `OllamaLLM` (the legacy completion class) returns a
   plain `str`. Accessing `.content` raises `AttributeError`. Fix by switching
   to `ChatOllama`, which uses message objects and natively supports a system
   prompt + chat history.
2. CLI-only — `input()`/`print()` cannot run inside a Streamlit app. Fix by
   exposing a `stream_chat(messages)` generator the page can pass to
   `st.write_stream`.

## Architecture

```
chatbot.py  (refactored, ~25 lines)
  ├── SYSTEM_PROMPT  — generic project-context system message
  ├── MODEL_NAME     — "qwen3.6"
  ├── get_chat_model()         — returns a ChatOllama instance
  └── stream_chat(messages)    — yields response chunks (strings) for streaming

app.py  (one new page block appended)
  ├── PAGES += ["Chatbot"]
  └── elif page == "Chatbot":
         render history → chat_input → stream reply → append to history
```

## Data flow

```
user types in st.chat_input
       │
       ▼
append {"role":"user","content":...} to st.session_state["chatbot_messages"]
       │
       ▼
build [SystemMessage, HumanMessage, AIMessage, ...] from history
       │
       ▼
ChatOllama.stream(messages) → yields AIMessageChunk objects
       │
       ▼
stream_chat() unwraps each chunk to its .content (string) and yields it
       │
       ▼
st.write_stream(...) renders progressively, returns the joined final text
       │
       ▼
append {"role":"assistant","content":final} to history
```

## UI

- Page title + one-line subtitle: *"Local assistant — runs on Ollama, no data
  leaves your machine."*
- "Clear conversation" button above the chat (resets the history list in
  session state).
- `st.chat_message("user" / "assistant")` for each prior turn.
- `st.chat_input("Ask anything…")` at the bottom.

## System prompt (sanitized)

> *You are an assistant embedded in an Emerging Market sovereign fixed income
> analytics dashboard. The platform analyzes EM sovereign bond yields across
> local-currency and hard-currency universes. The dashboard has pages: Home,
> Pipeline Health, Data Load, PCA & Regime, VaR Engine, Portfolios, Alert
> History, Daily Briefings, and Chatbot (this page). The user is a financial
> analyst. Be concise. If asked about specific live numbers (today's VaR,
> current alerts), say you don't have access to them and point to the
> relevant page.*

## Error handling

- Ollama daemon not running → first `.stream()` call raises a connection
  error. Catch in the page block and show `st.error(...)` with a hint to
  start Ollama. Do not crash the whole app.
- Model not installed (`qwen3.6` missing locally) → same path; the error
  message from `ollama` is surfaced.

## Files touched

- `chatbot.py` — full rewrite (~25 lines).
- `app.py` — two edits: add `"Chatbot"` to `PAGES`, append new
  `elif page == "Chatbot":` block at the end of the page dispatch chain.

## Non-goals / explicitly deferred

- Logging conversations.
- Authentication or rate limiting (local app, single user).
- Markdown rendering customization beyond Streamlit defaults.
