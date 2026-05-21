"""
Unit tests for the unified Models layer (``src/llm_client.py``).

These tests cover the boundary that the rest of the codebase will rely on
after the Models-layer extraction:

* ``ChatMessage`` is a provider-neutral, frozen dataclass.
* ``OllamaLLMClient`` converts ``ChatMessage`` lists into langchain message
  objects and unwraps streamed chunks back into strings.
* ``GeminiLLMClient`` flattens the system + user messages into Gemini's
  single-prompt format and extracts ``.text`` from response chunks.
* ``get_llm_client`` returns the right provider class and raises on
  unknown names.

Provider SDKs (``langchain_ollama`` and ``google.genai``) are mocked at
import time so these tests run without network access or model daemons.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# Make ``src`` importable regardless of pytest invocation directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm_client import (
    ChatMessage,
    GeminiLLMClient,
    LLMClient,
    OllamaLLMClient,
    get_llm_client,
)


# --------------------------------------------------------------------------- #
# ChatMessage                                                                 #
# --------------------------------------------------------------------------- #

class TestChatMessage:

    def test_is_frozen(self):
        m = ChatMessage(role="user", content="hi")
        with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
            m.role = "assistant"  # type: ignore[misc]

    def test_equality_by_value(self):
        assert ChatMessage("user", "hi") == ChatMessage("user", "hi")
        assert ChatMessage("user", "hi") != ChatMessage("assistant", "hi")


# --------------------------------------------------------------------------- #
# OllamaLLMClient                                                             #
# --------------------------------------------------------------------------- #

class TestOllamaLLMClient:
    """Mock ``langchain_ollama.ChatOllama`` so no daemon is required."""

    def _patched_client(self, fake_chat: MagicMock) -> OllamaLLMClient:
        # Build a stub langchain_ollama module exposing ChatOllama.
        fake_module = ModuleType("langchain_ollama")
        fake_module.ChatOllama = MagicMock(return_value=fake_chat)
        with patch.dict(sys.modules, {"langchain_ollama": fake_module}):
            return OllamaLLMClient(model="qwen3.6")

    def test_to_langchain_maps_roles(self):
        from langchain_core.messages import (
            AIMessage, HumanMessage, SystemMessage,
        )

        msgs = OllamaLLMClient._to_langchain([
            ChatMessage("system", "S"),
            ChatMessage("user", "U"),
            ChatMessage("assistant", "A"),
            ChatMessage("unknown", "X"),
        ])

        assert isinstance(msgs[0], SystemMessage) and msgs[0].content == "S"
        assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "U"
        assert isinstance(msgs[2], AIMessage) and msgs[2].content == "A"
        # Unknown roles fall through to HumanMessage.
        assert isinstance(msgs[3], HumanMessage) and msgs[3].content == "X"

    def test_complete_invokes_chat_and_returns_content(self):
        fake_chat = MagicMock()
        fake_chat.invoke.return_value = MagicMock(content="full-response")

        client = self._patched_client(fake_chat)
        out = client.complete([ChatMessage("user", "hi")])

        assert out == "full-response"
        # First positional arg to invoke() is the converted langchain list.
        sent = fake_chat.invoke.call_args[0][0]
        assert len(sent) == 1
        assert sent[0].content == "hi"

    def test_stream_unwraps_chunk_content(self):
        fake_chat = MagicMock()
        fake_chat.stream.return_value = iter([
            MagicMock(content="Hel"),
            MagicMock(content="lo"),
            MagicMock(content="!"),
        ])

        client = self._patched_client(fake_chat)
        out = list(client.stream([ChatMessage("user", "hi")]))

        assert out == ["Hel", "lo", "!"]

    def test_stream_preserves_message_order(self):
        fake_chat = MagicMock()
        fake_chat.stream.return_value = iter([])

        client = self._patched_client(fake_chat)
        list(client.stream([
            ChatMessage("system", "S"),
            ChatMessage("user", "U1"),
            ChatMessage("assistant", "A1"),
            ChatMessage("user", "U2"),
        ]))

        sent = fake_chat.stream.call_args[0][0]
        assert [m.content for m in sent] == ["S", "U1", "A1", "U2"]


# --------------------------------------------------------------------------- #
# GeminiLLMClient                                                             #
# --------------------------------------------------------------------------- #

class TestGeminiLLMClient:
    """Mock ``google.genai.Client`` so no API key/network is needed."""

    def _patched_client(self, fake_genai_client: MagicMock) -> GeminiLLMClient:
        fake_module = ModuleType("google.genai")
        fake_module.Client = MagicMock(return_value=fake_genai_client)
        parent = ModuleType("google")
        parent.genai = fake_module
        with patch.dict(sys.modules, {
            "google": parent,
            "google.genai": fake_module,
        }):
            return GeminiLLMClient(model="gemini-2.0-flash", api_key="fake-key")

    def test_flatten_concatenates_system_and_rest(self):
        prompt = GeminiLLMClient._flatten([
            ChatMessage("system", "You are an analyst."),
            ChatMessage("user", "What is VaR?"),
        ])
        assert "You are an analyst." in prompt
        assert "What is VaR?" in prompt
        # System content must appear before user content.
        assert prompt.index("analyst") < prompt.index("VaR")

    def test_flatten_without_system_returns_user_only(self):
        prompt = GeminiLLMClient._flatten([
            ChatMessage("user", "hello"),
        ])
        assert prompt == "hello"

    def test_complete_calls_generate_content_and_returns_text(self):
        fake_resp = MagicMock(text="Gemini reply.")
        fake_genai_client = MagicMock()
        fake_genai_client.models.generate_content.return_value = fake_resp

        client = self._patched_client(fake_genai_client)
        out = client.complete([
            ChatMessage("system", "S"),
            ChatMessage("user", "U"),
        ])

        assert out == "Gemini reply."
        kwargs = fake_genai_client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.0-flash"
        assert "S" in kwargs["contents"] and "U" in kwargs["contents"]

    def test_stream_yields_chunk_text_only_when_present(self):
        fake_genai_client = MagicMock()
        fake_genai_client.models.generate_content_stream.return_value = iter([
            MagicMock(text="Hel"),
            MagicMock(text=None),   # empty chunks must be skipped
            MagicMock(text="lo"),
        ])

        client = self._patched_client(fake_genai_client)
        out = list(client.stream([ChatMessage("user", "hi")]))

        assert out == ["Hel", "lo"]


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

class TestGetLLMClient:

    def test_returns_ollama_instance(self):
        fake_module = ModuleType("langchain_ollama")
        fake_module.ChatOllama = MagicMock()
        with patch.dict(sys.modules, {"langchain_ollama": fake_module}):
            client = get_llm_client("ollama", model="qwen3.6")
        assert isinstance(client, OllamaLLMClient)
        assert isinstance(client, LLMClient)

    def test_returns_gemini_instance(self):
        fake_module = ModuleType("google.genai")
        fake_module.Client = MagicMock()
        parent = ModuleType("google")
        parent.genai = fake_module
        with patch.dict(sys.modules, {
            "google": parent,
            "google.genai": fake_module,
        }):
            client = get_llm_client("gemini", model="gemini-2.0-flash",
                                    api_key="fake")
        assert isinstance(client, GeminiLLMClient)
        assert isinstance(client, LLMClient)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_client("openai", model="gpt-4")
