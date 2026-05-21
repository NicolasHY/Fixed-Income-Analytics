"""
Models layer: unified LLM client interface.

This module is the *only* place the codebase imports an LLM SDK.
Everything else (the chatbot, the briefing generator, the orchestrator) sees
a single ``LLMClient`` interface and a provider-neutral ``ChatMessage`` type.

Before this layer, ``chatbot.py`` imported ``langchain_ollama`` directly and
``main.ipynb`` instantiated ``genai.Client(...)`` inline — two SDKs, two
message formats, zero swap-ability. After this layer, providers are
selected via ``get_llm_client(provider, ...)`` and business code is
SDK-agnostic.

Providers
---------
``OllamaLLMClient``  — local inference via ``langchain_ollama.ChatOllama``
                       (used by the dashboard chatbot).
``GeminiLLMClient``  — cloud inference via ``google.genai``
                       (used by the daily briefing generator).

SDK imports are kept inside the provider constructors so the rest of the
codebase doesn't pay the import cost, and so the absence of one SDK does
not break callers that only use the other.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Provider-neutral message type                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChatMessage:
    """Provider-neutral chat message.

    role : one of ``"system"``, ``"user"``, ``"assistant"``.
    """
    role: str
    content: str


# --------------------------------------------------------------------------- #
# Abstract base                                                               #
# --------------------------------------------------------------------------- #

class LLMClient(ABC):
    """Provider-agnostic chat completion interface."""

    @abstractmethod
    def complete(self, messages: list[ChatMessage]) -> str:
        """Return the full completion as a single string."""

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield completion chunks as strings (one chunk per yield)."""


# --------------------------------------------------------------------------- #
# Ollama provider — local inference                                           #
# --------------------------------------------------------------------------- #

class OllamaLLMClient(LLMClient):
    """Local Ollama provider — used by the interactive chatbot."""

    def __init__(self, model: str):
        from langchain_ollama import ChatOllama  # lazy: heavy import
        self.model = model
        self._chat = ChatOllama(model=model)

    @staticmethod
    def _to_langchain(messages: list[ChatMessage]) -> list:
        from langchain_core.messages import (
            AIMessage, HumanMessage, SystemMessage,
        )
        out: list = []
        for m in messages:
            if m.role == "system":
                out.append(SystemMessage(content=m.content))
            elif m.role == "assistant":
                out.append(AIMessage(content=m.content))
            else:
                out.append(HumanMessage(content=m.content))
        return out

    def complete(self, messages: list[ChatMessage]) -> str:
        return self._chat.invoke(self._to_langchain(messages)).content

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        for chunk in self._chat.stream(self._to_langchain(messages)):
            yield chunk.content


# --------------------------------------------------------------------------- #
# Gemini provider — cloud inference                                           #
# --------------------------------------------------------------------------- #

class GeminiLLMClient(LLMClient):
    """Google Gemini provider — used for scheduled daily briefings.

    Gemini's API does not have separate system/user roles; system messages
    are concatenated as a prefix to the user prompt. Assistant messages are
    treated as additional context (also concatenated) — this matches how
    the notebook calls Gemini today.
    """

    def __init__(self, model: str, api_key: str):
        from google import genai  # lazy: heavy import
        self.model = model
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _flatten(messages: list[ChatMessage]) -> str:
        system_parts = [m.content for m in messages if m.role == "system"]
        rest_parts = [m.content for m in messages if m.role != "system"]
        system = "\n\n".join(s for s in system_parts if s)
        rest = "\n\n".join(r for r in rest_parts if r)
        return f"{system}\n\n{rest}".strip() if system else rest

    def complete(self, messages: list[ChatMessage]) -> str:
        resp = self._client.models.generate_content(
            model=self.model, contents=self._flatten(messages),
        )
        return resp.text

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        for chunk in self._client.models.generate_content_stream(
            model=self.model, contents=self._flatten(messages),
        ):
            if chunk.text:
                yield chunk.text


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def get_llm_client(provider: str, **kwargs) -> LLMClient:
    """Construct an ``LLMClient`` for the named provider.

    Parameters
    ----------
    provider : {"ollama", "gemini"}
    **kwargs : provider-specific construction args.
        * ollama  : ``model``
        * gemini  : ``model``, ``api_key``
    """
    if provider == "ollama":
        return OllamaLLMClient(**kwargs)
    if provider == "gemini":
        return GeminiLLMClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
