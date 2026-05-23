"""
Tests for ``src/config.py`` — the infrastructure / config layer.

The config layer is the only place runtime configuration is allowed to
touch ``os.environ`` or local file paths. Business code asks for a
resolved value (an API key, a model id, …) and either gets it or sees
an informative ``ConfigError``.

Resolution precedence under test:
    1. environment variable (containers / CI inject here)
    2. file path declared in funds.yaml (dev-machine default)
    3. ``ConfigError`` naming both the env var and the path that was tried.
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

from src.config import (
    ConfigError,
    GeminiConfig,
    build_gemini_client,
    get_gemini_config,
    resolve_secret,
)


# --------------------------------------------------------------------------- #
# resolve_secret — generic env-first, file-fallback resolver                  #
# --------------------------------------------------------------------------- #

class TestResolveSecret:

    def test_env_var_wins_over_file(self, tmp_path, monkeypatch):
        f = tmp_path / "secret.txt"
        f.write_text("from-file")
        monkeypatch.setenv("MY_SECRET", "from-env")

        assert resolve_secret("MY_SECRET", fallback_path=f) == "from-env"

    def test_falls_back_to_file_when_env_missing(self, tmp_path, monkeypatch):
        f = tmp_path / "secret.txt"
        f.write_text("from-file\n")  # trailing newline must be stripped
        monkeypatch.delenv("MY_SECRET", raising=False)

        assert resolve_secret("MY_SECRET", fallback_path=f) == "from-file"

    def test_whitespace_only_env_is_ignored(self, tmp_path, monkeypatch):
        f = tmp_path / "secret.txt"
        f.write_text("from-file")
        monkeypatch.setenv("MY_SECRET", "   ")

        assert resolve_secret("MY_SECRET", fallback_path=f) == "from-file"

    def test_empty_file_falls_through(self, tmp_path, monkeypatch):
        f = tmp_path / "secret.txt"
        f.write_text("   \n")  # whitespace-only content
        monkeypatch.delenv("MY_SECRET", raising=False)

        with pytest.raises(ConfigError):
            resolve_secret("MY_SECRET", fallback_path=f)

    def test_raises_when_nothing_resolves(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_SECRET", raising=False)
        bogus = tmp_path / "does-not-exist"

        with pytest.raises(ConfigError) as exc:
            resolve_secret("MISSING_SECRET", fallback_path=bogus, name="My Key")

        msg = str(exc.value)
        assert "My Key" in msg
        assert "MISSING_SECRET" in msg
        assert str(bogus) in msg

    def test_raises_without_fallback_path(self, monkeypatch):
        monkeypatch.delenv("ONLY_ENV", raising=False)

        with pytest.raises(ConfigError) as exc:
            resolve_secret("ONLY_ENV")

        assert "ONLY_ENV" in str(exc.value)


# --------------------------------------------------------------------------- #
# get_gemini_config — composed Gemini provider configuration                  #
# --------------------------------------------------------------------------- #

class TestGetGeminiConfig:

    def _cfg(self, tmp_path: Path, key_contents: str = "key-from-file") -> dict:
        f = tmp_path / "gemini_key.txt"
        f.write_text(key_contents)
        return {
            "briefing": {
                "model": "gemini-2.0-flash",
                "api_key_path": str(f),
            }
        }

    def test_returns_gemini_config_dataclass(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        out = get_gemini_config(self._cfg(tmp_path))

        assert isinstance(out, GeminiConfig)
        assert out.model == "gemini-2.0-flash"
        assert out.api_key == "key-from-file"

    def test_env_var_overrides_file_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "key-from-env")

        out = get_gemini_config(self._cfg(tmp_path))

        assert out.api_key == "key-from-env"

    def test_raises_when_model_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        with pytest.raises(ConfigError, match="briefing.model"):
            get_gemini_config({"briefing": {"api_key_path": "x"}})

    def test_raises_when_briefing_section_missing(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        with pytest.raises(ConfigError, match="briefing.model"):
            get_gemini_config({})

    def test_raises_when_key_unresolvable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = {
            "briefing": {
                "model": "gemini-2.0-flash",
                "api_key_path": str(tmp_path / "nope"),
            }
        }

        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            get_gemini_config(cfg)

    def test_loads_yaml_when_no_config_passed(self, tmp_path, monkeypatch):
        """If the caller passes no config dict, the YAML loader is used."""
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        fake = {"briefing": {"model": "gemini-from-yaml"}}
        with patch("src.data_loader.load_config", return_value=fake):
            out = get_gemini_config()
        assert out.model == "gemini-from-yaml"
        assert out.api_key == "k"


# --------------------------------------------------------------------------- #
# build_gemini_client — convenience constructor                               #
# --------------------------------------------------------------------------- #

class TestBuildGeminiClient:

    def test_builds_with_resolved_settings(self, tmp_path, monkeypatch):
        f = tmp_path / "g.txt"
        f.write_text("file-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = {
            "briefing": {
                "model": "gemini-2.0-flash",
                "api_key_path": str(f),
            }
        }

        fake_module = ModuleType("google.genai")
        fake_module.Client = MagicMock()
        parent = ModuleType("google")
        parent.genai = fake_module
        with patch.dict(sys.modules, {
            "google": parent,
            "google.genai": fake_module,
        }):
            client = build_gemini_client(cfg)

        assert client.model == "gemini-2.0-flash"
        fake_module.Client.assert_called_once_with(api_key="file-key")

    def test_env_var_wins_in_builder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        cfg = {"briefing": {"model": "gemini-2.0-flash"}}

        fake_module = ModuleType("google.genai")
        fake_module.Client = MagicMock()
        parent = ModuleType("google")
        parent.genai = fake_module
        with patch.dict(sys.modules, {
            "google": parent,
            "google.genai": fake_module,
        }):
            build_gemini_client(cfg)

        fake_module.Client.assert_called_once_with(api_key="env-key")
