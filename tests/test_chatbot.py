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


def test_stream_chat_yields_strings_from_llm_client():
    """stream_chat delegates to LLMClient.stream() and yields its strings directly.

    Provider-specific chunk-unwrapping (langchain's AIMessageChunk.content,
    Gemini's response.text) is now the LLMClient's responsibility; the
    chatbot just forwards strings.
    """
    import chatbot
    from src.llm_client import ChatMessage

    fake_client = MagicMock()
    fake_client.stream.return_value = iter(["Hel", "lo", "!"])

    with patch.object(chatbot, "get_chat_model", return_value=fake_client):
        out = list(chatbot.stream_chat([{"role": "user", "content": "hi"}]))

    assert out == ["Hel", "lo", "!"]
    # Messages are now provider-neutral ChatMessage dataclasses with the
    # system prompt prepended.
    sent = fake_client.stream.call_args[0][0]
    assert isinstance(sent[0], ChatMessage)
    assert sent[0].role == "system"
    assert sent[0].content == chatbot.SYSTEM_PROMPT
    assert sent[1].role == "user"
    assert sent[1].content == "hi"
