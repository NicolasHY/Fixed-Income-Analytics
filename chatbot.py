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
