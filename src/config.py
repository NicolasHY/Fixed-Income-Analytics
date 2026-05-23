"""
Infrastructure layer: resolved configuration and secrets.

This module is the *only* place runtime configuration is allowed to
touch ``os.environ`` or read secret files. Business code asks for a
resolved value (an API key, a model id, …) and either receives it or
gets an informative ``ConfigError`` — it never inspects the filesystem
or environment directly.

Why a separate module?
----------------------
Before this layer, the notebook held ``open("private/gemini_key.txt")``
on the production execution path. That meant:

* Containers and CI couldn't inject a key without bind-mounting a file.
* Missing-key failures surfaced as opaque ``FileNotFoundError`` traces
  half-way through the briefing pipeline.
* Three call sites (notebook, Streamlit page, future scheduler) each
  re-implemented the same "read the file" pattern.

Centralising in ``src/config.py`` collapses those three call sites to
one, and gives every entry point the same precedence rules:

    1. environment variable                 (container / CI injection)
    2. file path declared in funds.yaml     (dev-machine default)
    3. ``ConfigError`` naming both          (loud, early, actionable)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #

class ConfigError(RuntimeError):
    """Raised when a required configuration value cannot be resolved.

    Messages always name the environment variable that callers can set
    and, when applicable, the file path that was tried — so the fix is
    obvious without reading the resolver source.
    """


# --------------------------------------------------------------------------- #
# Generic resolver                                                            #
# --------------------------------------------------------------------------- #

def _read_secret_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def resolve_secret(
    env_var: str,
    *,
    fallback_path: str | Path | None = None,
    name: Optional[str] = None,
) -> str:
    """Resolve a secret string via env-first, file-fallback strategy.

    Parameters
    ----------
    env_var : str
        Environment variable consulted first. Whitespace-only values
        are treated as unset.
    fallback_path : str | Path, optional
        File whose stripped contents are used if ``env_var`` is unset.
    name : str, optional
        Human-readable label for error messages (defaults to ``env_var``).
    """
    label = name or env_var
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val
    if fallback_path is not None:
        file_val = _read_secret_file(fallback_path)
        if file_val:
            return file_val
    detail = (
        f" or place the secret in {fallback_path}"
        if fallback_path is not None
        else ""
    )
    raise ConfigError(
        f"Could not resolve {label}: set the {env_var} environment "
        f"variable{detail}."
    )


# --------------------------------------------------------------------------- #
# Gemini provider configuration                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GeminiConfig:
    """Resolved Gemini provider configuration."""
    api_key: str
    model: str


def get_gemini_config(config: dict | None = None) -> GeminiConfig:
    """Resolve Gemini API configuration.

    ``model`` is read from ``config["briefing"]["model"]``. ``api_key``
    is resolved env-first via :func:`resolve_secret`, falling back to
    the file path at ``config["briefing"]["api_key_path"]``.

    Parameters
    ----------
    config : dict, optional
        Parsed funds.yaml. If ``None``, :func:`src.data_loader.load_config`
        is invoked.
    """
    if config is None:
        from src.data_loader import load_config
        config = load_config()

    briefing = config.get("briefing") or {}
    model = briefing.get("model")
    if not model:
        raise ConfigError(
            "briefing.model is missing from funds.yaml — cannot build a "
            "Gemini client without a model id."
        )

    api_key = resolve_secret(
        "GEMINI_API_KEY",
        fallback_path=briefing.get("api_key_path"),
        name="Gemini API key",
    )
    return GeminiConfig(api_key=api_key, model=model)


def build_gemini_client(config: dict | None = None):
    """Build a :class:`GeminiLLMClient` from resolved configuration.

    The notebook, Streamlit pages, and any scheduler should call this
    rather than constructing the client themselves — it guarantees the
    env-first / file-fallback precedence is uniform across entry points.
    """
    from src.llm_client import GeminiLLMClient
    cfg = get_gemini_config(config)
    return GeminiLLMClient(model=cfg.model, api_key=cfg.api_key)
