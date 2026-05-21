"""Chat helper for the Streamlit dashboard.

Thin orchestration glue between the dashboard Application layer and the
unified Models layer (:mod:`src.llm_client`). It owns:

* the system prompt that grounds the assistant in the dashboard's pages
* the ``{role, content}`` dict → :class:`ChatMessage` conversion

It does *not* own LLM provider details — those live in
:mod:`src.llm_client`. To swap Ollama for Gemini, only this module's
``get_chat_model`` factory needs to change.
"""
from __future__ import annotations

from collections.abc import Iterator

# Back-compat imports — exposed via _to_lc_messages() for tests and callers
# written before src.llm_client existed. New code should use the
# ChatMessage path through src.llm_client.
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from src.llm_client import ChatMessage, LLMClient, OllamaLLMClient
from src.orchestration.chat import orchestrate_chat

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


def get_chat_model() -> LLMClient:
    """Return an :class:`LLMClient` handle for the configured local model."""
    return OllamaLLMClient(model=MODEL_NAME)


def _to_chat_messages(history: list[dict]) -> list[ChatMessage]:
    """Convert ``{role, content}`` dicts into a provider-neutral message list.

    Always prepends the system prompt. Unknown roles are coerced to ``"user"``.
    """
    msgs: list[ChatMessage] = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
    ]
    for m in history:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        msgs.append(ChatMessage(role=role, content=m.get("content", "")))
    return msgs


def _to_lc_messages(history: list[dict]) -> list[BaseMessage]:
    """[Back-compat] Convert ``{role, content}`` dicts → langchain messages.

    Kept for tests and external callers written before :mod:`src.llm_client`
    existed. New code should use :func:`_to_chat_messages` and pass the
    result to an :class:`LLMClient`.
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
    """Yield response tokens from the model, one string per chunk.

    Delegates to :func:`src.orchestration.chat.orchestrate_chat` so the
    chat path goes through the same Orchestration layer as the briefing
    path. The model handle is acquired from :func:`get_chat_model` (still
    patchable in tests).

    Intended for use with Streamlit's ``st.write_stream``.
    """
    client = get_chat_model()
    yield from orchestrate_chat(history, SYSTEM_PROMPT, client)
