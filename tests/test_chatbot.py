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
