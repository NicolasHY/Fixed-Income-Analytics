"""
Characterization tests for the Orchestration layer.

Three surfaces to lock:

1. :func:`build_analytics_state` — full pipeline runs cleanly against the
   real raw CSVs and the returned dataclass exposes every field the
   briefing flow and the dashboard rely on.
2. :func:`generate_briefing` + :func:`run_briefing_workflow` — Data layer
   is called, Models layer is invoked, cache round-trips.
3. :func:`orchestrate_chat` — builds the message list (system prompt
   prepended, role coercion) and forwards to the model client.

The LLM provider is always mocked (no network, no daemons). Real raw
CSVs are sliced to ``<= 2025-12-31`` for stable snapshots.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.briefing_store import load_briefings
from src.llm_client import ChatMessage, LLMClient
from src.orchestration import (
    AnalyticsState,
    DEFAULT_BRIEFING_SYSTEM_PROMPT,
    build_analytics_state,
    generate_briefing,
    orchestrate_chat,
    run_briefing_workflow,
)


SLICE_END = pd.Timestamp("2025-12-31")
STRESS_DATE = "2022-09-23"


# --------------------------------------------------------------------------- #
# Shared fixtures                                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def state() -> AnalyticsState:
    return build_analytics_state(slice_end=SLICE_END)


# --------------------------------------------------------------------------- #
# 1. build_analytics_state                                                    #
# --------------------------------------------------------------------------- #

class TestBuildAnalyticsState:
    """The full pipeline must run end-to-end and produce a populated state."""

    def test_returns_analytics_state(self, state):
        assert isinstance(state, AnalyticsState)

    def test_universe_loaded(self, state):
        assert set(state.change_dfs) == {
            "Brazil", "Mexico", "South Africa", "Poland",
            "Colombia", "Hungary", "Romania",
        }

    def test_lc_portfolio_built(self, state):
        # LC fund proxy uses the 4 local-currency countries from lc_fund config.
        assert set(state.proxy_dy.columns) == {
            "Brazil", "Mexico", "South Africa", "Poland",
        }
        assert state.duration == pytest.approx(5.22)
        # Weights normalise to 1.
        assert sum(state.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_pca_regime_alerts_populated(self, state):
        assert set(state.pca_results) == set(state.change_dfs)
        assert {"regime", "regime_proba", "regime_label"} <= set(
            state.regime_features.columns
        )
        assert len(state.all_alerts) > 0, "Expected at least one alert day"

    def test_mc_var_snapshot(self, state):
        """Same values as ``tests/test_var_engine.py``'s snapshot — proves
        the orchestrator wires the engine with the right seed/config."""
        assert state.var_95 == pytest.approx(0.00510399, abs=1e-6)
        assert state.var_99 == pytest.approx(0.00972852, abs=1e-6)

    def test_var99_exceeds_var95(self, state):
        assert state.var_99 > state.var_95


# --------------------------------------------------------------------------- #
# 2. Briefing workflow                                                        #
# --------------------------------------------------------------------------- #

class TestGenerateBriefing:

    def test_calls_data_then_model(self, state):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.complete.return_value = "Headline: Rates rallied across LATAM."

        date_out, text = generate_briefing(state, STRESS_DATE, fake_llm)

        assert date_out == STRESS_DATE
        assert text == "Headline: Rates rallied across LATAM."

        # Models layer was called once with a system + user pair.
        fake_llm.complete.assert_called_once()
        sent = fake_llm.complete.call_args[0][0]
        assert len(sent) == 2
        assert isinstance(sent[0], ChatMessage)
        assert sent[0].role == "system"
        assert sent[1].role == "user"
        # The user message contains the payload as JSON (Data layer output).
        assert STRESS_DATE in sent[1].content
        assert "regime" in sent[1].content
        assert "curve_moves_bps" in sent[1].content

    def test_uses_config_system_prompt_when_present(self, state):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.complete.return_value = "ok"
        # state.config["briefing"]["system_prompt"] is the EM analyst prompt.
        generate_briefing(state, STRESS_DATE, fake_llm)
        sent = fake_llm.complete.call_args[0][0]
        configured = state.config["briefing"]["system_prompt"]
        assert sent[0].content == configured

    def test_explicit_system_prompt_overrides_config(self, state):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.complete.return_value = "ok"
        generate_briefing(state, STRESS_DATE, fake_llm,
                          system_prompt="OVERRIDE PROMPT")
        sent = fake_llm.complete.call_args[0][0]
        assert sent[0].content == "OVERRIDE PROMPT"


class TestRunBriefingWorkflow:

    def test_skips_cached_dates(self, state, tmp_path):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.complete.return_value = "fresh briefing"

        cache_path = tmp_path / "briefings.json"
        # Pre-seed the cache with the stress date.
        from src.data.briefing_store import save_briefings
        save_briefings({STRESS_DATE: "already-cached"}, cache_path)

        out = run_briefing_workflow(state, [STRESS_DATE], fake_llm,
                                    cache_path=cache_path)

        fake_llm.complete.assert_not_called()
        assert out[STRESS_DATE] == "already-cached"

    def test_generates_and_persists_missing_dates(self, state, tmp_path):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.complete.return_value = "generated text"
        cache_path = tmp_path / "briefings.json"

        out = run_briefing_workflow(state, [STRESS_DATE], fake_llm,
                                    cache_path=cache_path)

        fake_llm.complete.assert_called_once()
        assert out[STRESS_DATE] == "generated text"
        # Persisted to disk.
        assert load_briefings(cache_path) == {STRESS_DATE: "generated text"}

    def test_continues_on_provider_failure(self, state, tmp_path):
        """One date fails, the next succeeds — the cache still saves the good one."""
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.complete.side_effect = [
            RuntimeError("rate-limited"),
            "second date worked",
        ]
        cache_path = tmp_path / "briefings.json"

        out = run_briefing_workflow(
            state, [STRESS_DATE, "2023-06-15"], fake_llm,
            cache_path=cache_path,
        )

        # First date failed → not in cache; second succeeded → in cache.
        assert STRESS_DATE not in out
        assert out["2023-06-15"] == "second date worked"
        assert load_briefings(cache_path) == {"2023-06-15": "second date worked"}

    def test_default_system_prompt_is_nonempty(self):
        assert isinstance(DEFAULT_BRIEFING_SYSTEM_PROMPT, str)
        assert len(DEFAULT_BRIEFING_SYSTEM_PROMPT) > 200
        assert "EM" in DEFAULT_BRIEFING_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# 3. orchestrate_chat                                                         #
# --------------------------------------------------------------------------- #

class TestOrchestrateChat:

    def test_prepends_system_prompt(self):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.stream.return_value = iter(["Hel", "lo"])
        out = list(orchestrate_chat(
            [{"role": "user", "content": "hi"}],
            "SYSTEM",
            fake_llm,
        ))
        assert out == ["Hel", "lo"]
        sent = fake_llm.stream.call_args[0][0]
        assert sent[0].role == "system" and sent[0].content == "SYSTEM"
        assert sent[1].role == "user" and sent[1].content == "hi"

    def test_role_coercion(self):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.stream.return_value = iter([])
        list(orchestrate_chat(
            [
                {"role": "assistant", "content": "prior turn"},
                {"role": "tool", "content": "should be coerced to user"},
            ],
            "SYS",
            fake_llm,
        ))
        sent = fake_llm.stream.call_args[0][0]
        assert sent[1].role == "assistant"
        assert sent[2].role == "user"
        assert sent[2].content == "should be coerced to user"

    def test_empty_history_still_yields_messages(self):
        fake_llm = MagicMock(spec=LLMClient)
        fake_llm.stream.return_value = iter(["x"])
        out = list(orchestrate_chat([], "SYS", fake_llm))
        assert out == ["x"]
        sent = fake_llm.stream.call_args[0][0]
        assert len(sent) == 1 and sent[0].role == "system"
