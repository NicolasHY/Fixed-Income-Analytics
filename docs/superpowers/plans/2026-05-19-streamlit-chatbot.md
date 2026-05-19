# Streamlit Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken CLI `chatbot.py` with a small importable module, and add a "Chatbot" page to the existing Streamlit app that streams responses from a local Ollama model (`qwen3.6`).

**Architecture:** `chatbot.py` becomes a pure module exposing constants (`MODEL_NAME`, `SYSTEM_PROMPT`), a model factory (`get_chat_model`), and a token-streaming generator (`stream_chat`) that takes plain `{"role","content"}` history dicts. The Streamlit page lives in `app.py` next to the other page blocks, holds per-session history in `st.session_state["chatbot_messages"]`, and renders streamed output via `st.write_stream`.

**Tech Stack:** Python, Streamlit 1.57, `langchain-ollama` (`ChatOllama`), `langchain-core` messages, Ollama daemon running locally with `qwen3.6` pulled.

---

## File Structure

| File | Status | Responsibility |
|------|--------|---------------|
| `chatbot.py` | **Rewrite** | Module: model factory + streaming helper + system prompt. No CLI, no `__main__`. |
| `app.py` | **Modify** | Two edits: (1) add `"Chatbot"` to `PAGES`; (2) append a new `elif page == "Chatbot":` block at the end of the page dispatch chain. |
| `tests/test_chatbot.py` | **Create** | Unit tests for the module's constants, message conversion, and stream wrapper (Ollama is mocked — no daemon required). |

---

## Task 1: Tests for the chatbot module (TDD — fails first)

**Files:**
- Create: `tests/test_chatbot.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_chatbot.py`:

```python
"""Tests for the chatbot module. Mocks Ollama — no live daemon required."""
from unittest.mock import patch, MagicMock

import pytest


def test_constants_are_nonempty_strings():
    import chatbot
    assert isinstance(chatbot.MODEL_NAME, str) and chatbot.MODEL_NAME
    assert isinstance(chatbot.SYSTEM_PROMPT, str) and len(chatbot.SYSTEM_PROMPT) > 50


def test_system_prompt_describes_dashboard_pages():
    """The system prompt must mention dashboard pages so the model can route users."""
    import chatbot
    sp = chatbot.SYSTEM_PROMPT
    for page in ["VaR Engine", "PCA & Regime", "Alert History", "Daily Briefings"]:
        assert page in sp, f"system prompt missing page reference: {page}"


def test_to_lc_messages_prepends_system_and_maps_roles():
    import chatbot
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "what is VaR?"},
    ]
    msgs = chatbot._to_lc_messages(history)
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == chatbot.SYSTEM_PROMPT
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "hello"
    assert isinstance(msgs[2], AIMessage) and msgs[2].content == "hi there"
    assert isinstance(msgs[3], HumanMessage) and msgs[3].content == "what is VaR?"
    assert len(msgs) == 4


def test_stream_chat_yields_chunk_contents():
    """stream_chat must unwrap each AIMessageChunk and yield its .content as a string."""
    import chatbot

    fake_chunks = [MagicMock(content="Hel"), MagicMock(content="lo"), MagicMock(content="!")]
    fake_model = MagicMock()
    fake_model.stream.return_value = iter(fake_chunks)

    with patch.object(chatbot, "get_chat_model", return_value=fake_model):
        out = list(chatbot.stream_chat([{"role": "user", "content": "hi"}]))

    assert out == ["Hel", "lo", "!"]
    # Verify the messages passed to .stream() are langchain message objects.
    sent = fake_model.stream.call_args[0][0]
    assert sent[0].content == chatbot.SYSTEM_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chatbot.py -v`
Expected: tests fail with `AttributeError: module 'chatbot' has no attribute '_to_lc_messages'` (or similar) — confirms tests exercise behaviour that does not yet exist in the rewritten module.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/test_chatbot.py
git commit -m "test(chatbot): add unit tests for module API (TDD red)"
```

---

## Task 2: Rewrite `chatbot.py` as a module

**Files:**
- Modify (full rewrite): `chatbot.py`

- [ ] **Step 1: Rewrite `chatbot.py`**

Full file contents:

```python
"""Local-LLM chatbot helper for the Streamlit dashboard.

Exposes a small interface so the Streamlit page only deals with plain
{"role", "content"} dicts and never imports langchain types directly.
"""
from collections.abc import Iterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

MODEL_NAME = "qwen3.6"

SYSTEM_PROMPT = (
    "You are an assistant embedded in an Emerging Market sovereign fixed "
    "income analytics dashboard. The platform analyzes EM sovereign bond "
    "yields across local-currency and hard-currency universes. The dashboard "
    "has pages: Home, Pipeline Health, Data Load, PCA & Regime, VaR Engine, "
    "Portfolios, Alert History, Daily Briefings, and Chatbot (this page). "
    "The user is a financial analyst. Be concise. If asked about specific "
    "live numbers (today's VaR, current alerts), say you don't have access "
    "to them and point to the relevant page."
)


def get_chat_model() -> ChatOllama:
    """Return a ChatOllama handle for the configured local model."""
    return ChatOllama(model=MODEL_NAME)


def _to_lc_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert {role, content} dicts into langchain message objects.

    Always prepends the system prompt. Unknown roles are coerced to user.
    """
    msgs: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            msgs.append(AIMessage(content=content))
        else:
            msgs.append(HumanMessage(content=content))
    return msgs


def stream_chat(history: list[dict]) -> Iterator[str]:
    """Yield response tokens from the model, one chunk's content at a time.

    Intended for use with Streamlit's `st.write_stream`.
    """
    model = get_chat_model()
    for chunk in model.stream(_to_lc_messages(history)):
        yield chunk.content
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_chatbot.py -v`
Expected: all four tests PASS.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `pytest tests/ -q`
Expected: all tests pass; no new failures elsewhere.

- [ ] **Step 4: Commit the module rewrite**

```powershell
git add chatbot.py
git commit -m "feat(chatbot): refactor to a Streamlit-friendly module (TDD green)"
```

---

## Task 3: Add "Chatbot" to the page navigation

**Files:**
- Modify: `app.py:570-573` (the `PAGES` list)

- [ ] **Step 1: Append "Chatbot" to PAGES**

Find this in `app.py`:

```python
PAGES = [
    "Home", "Pipeline Health", "Data Load", "PCA & Regime",
    "VaR Engine", "Portfolios", "Alert History", "Daily Briefings",
]
```

Change to:

```python
PAGES = [
    "Home", "Pipeline Health", "Data Load", "PCA & Regime",
    "VaR Engine", "Portfolios", "Alert History", "Daily Briefings",
    "Chatbot",
]
```

- [ ] **Step 2: Quick syntax check**

Run: `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read())"`
Expected: no output (parse succeeds).

- [ ] **Step 3: Do NOT commit yet**

The page block in Task 4 must land in the same commit as the nav entry — otherwise selecting "Chatbot" in the sidebar between commits would render an empty page.

---

## Task 4: Add the Chatbot page block to `app.py`

**Files:**
- Modify: `app.py` — append a new `elif page == "Chatbot":` block immediately after the `Alert History` block (current end of dispatch, around line 2036).

- [ ] **Step 1: Add the chatbot import**

Open `app.py`, locate the import section near the top (look for existing imports like `from src.data_loader import ...`), and add this import alongside them:

```python
import chatbot
```

If there's no existing alphabetical ordering convention in that block, just add it on its own line after the other stdlib/third-party imports.

- [ ] **Step 2: Append the page block**

At the **end of `app.py`** (after the `Alert History` block, current EOF around line 2036), append:

```python

# ── Chatbot ───────────────────────────────────────────────────────────────────
elif page == "Chatbot":
    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = []

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown(
            "<div style='color:#64748b; font-size:0.9rem; margin-bottom:8px;'>"
            "Local assistant — runs on Ollama, no data leaves your machine."
            "</div>",
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chatbot_messages"] = []
            st.rerun()

    # Render prior turns.
    for msg in st.session_state["chatbot_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about the dashboard or general questions…")
    if user_input:
        st.session_state["chatbot_messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(
                    chatbot.stream_chat(st.session_state["chatbot_messages"])
                )
                st.session_state["chatbot_messages"].append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as exc:
                st.error(
                    f"Could not reach the local Ollama model "
                    f"(`{chatbot.MODEL_NAME}`). Is `ollama serve` running and the "
                    f"model pulled?\n\n**Details:** `{exc}`"
                )
```

- [ ] **Step 3: Syntax check**

Run: `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read())"`
Expected: no output.

- [ ] **Step 4: Import smoke test**

Run: `python -c "import chatbot; print(chatbot.MODEL_NAME, '|', len(chatbot.SYSTEM_PROMPT))"`
Expected: prints `qwen3.6 | <some integer ≥ ~400>`.

- [ ] **Step 5: Commit nav entry + page block together**

```powershell
git add app.py
git commit -m "feat(app): add Chatbot page backed by local Ollama (qwen3.6)"
```

---

## Task 5: Manual end-to-end verification

**Files:** none (manual smoke test)

This step cannot be automated because Streamlit + Ollama require a browser and a running daemon.

- [ ] **Step 1: Ensure Ollama is running and the model is pulled**

Run: `ollama list`
Expected: a row containing `qwen3.6` (with any tag). If absent, run `ollama pull qwen3.6`.

- [ ] **Step 2: Launch the Streamlit app**

Run (PowerShell, from project root): `.\.venv\Scripts\streamlit.exe run app.py`
Expected: browser opens at `http://localhost:8501`.

- [ ] **Step 3: Verify the Chatbot page**

In the sidebar, click **Chatbot**. Verify:
- The "Local assistant — runs on Ollama…" subtitle appears.
- "Clear conversation" button is visible top-right.
- A chat input box is at the bottom.

- [ ] **Step 4: Send a test message**

Type "What pages does this dashboard have?" and press Enter.
Expected: tokens stream into the assistant bubble within a few seconds; final reply mentions at least three of the dashboard pages.

- [ ] **Step 5: Verify history + clear button**

Send a follow-up like "And what does the PCA page do?" — confirm the reply references the prior turn (proves history is being sent). Click **Clear conversation** — chat empties; further messages start fresh.

- [ ] **Step 6: Verify graceful failure**

Stop Ollama (`ollama stop` in another terminal, or kill the process). Send a message. Expected: `st.error` panel with the "Could not reach the local Ollama model…" message; the rest of the app remains usable.

- [ ] **Step 7: Stop the app and commit nothing**

This task produces no code changes — just confidence the integration works.

---

## Done criteria

- All tests in `tests/test_chatbot.py` pass.
- Full suite (`pytest tests/`) still passes.
- The Streamlit app starts and the Chatbot page produces a streamed reply from `qwen3.6` for at least one sample message.
- Stopping Ollama produces a visible error instead of a crashed app.
