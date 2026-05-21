"""
Chat orchestrator.

Minimal today: prepend a system prompt, map roles, stream from the
Models layer. The orchestration boundary exists so that future grounding
(retrieving briefing context from :mod:`src.data.briefing_context`
before calling the model) can be added here without touching
:mod:`chatbot` or :mod:`app`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from src.llm_client import ChatMessage, LLMClient


def _build_messages(history: Iterable[dict], system_prompt: str) -> list[ChatMessage]:
    msgs: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
    ]
    for m in history:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        msgs.append(ChatMessage(role=role, content=m.get("content", "")))
    return msgs


def orchestrate_chat(
    history: Iterable[dict],
    system_prompt: str,
    llm_client: LLMClient,
) -> Iterator[str]:
    """Stream a chat response.

    Parameters
    ----------
    history : iterable of {role, content} dicts
        Prior turns. ``role`` may be ``"system"``, ``"user"``,
        ``"assistant"``; unknown roles are coerced to ``"user"``.
    system_prompt : str
        Prepended as the first message regardless of the history.
    llm_client : LLMClient
        Caller-supplied so tests can mock and so the application layer
        can pick the provider (Ollama for local chat, Gemini if it ever
        needs cloud chat, …).
    """
    yield from llm_client.stream(_build_messages(history, system_prompt))
