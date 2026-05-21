"""
Briefing workflow orchestrator.

The single-date entry point :func:`generate_briefing` is the hub the
ARCHITECTURE_PLAN.md described: it calls the Data layer
(:func:`src.data.build_daily_payload`) to assemble the grounding context,
then invokes the Models layer (an :class:`LLMClient`) to produce a
human-readable briefing.

:func:`run_briefing_workflow` wraps that for the showcase-date loop the
notebook used to run inline — it consults the cache on disk, skips dates
already present, and persists new entries via
:func:`src.data.briefing_store.save_briefings`.

The Gemini SYSTEM_PROMPT lives in ``config/funds.yaml`` under
``briefing.system_prompt``; :data:`DEFAULT_BRIEFING_SYSTEM_PROMPT` is the
fallback if the config doesn't supply one.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.data import (
    DEFAULT_BRIEFING_CACHE,
    build_daily_payload,
    load_briefings,
    save_briefings,
)
from src.llm_client import ChatMessage, LLMClient
from src.orchestration.pipeline import AnalyticsState

logger = logging.getLogger(__name__)


DEFAULT_BRIEFING_SYSTEM_PROMPT = (
    "You are a Fixed Income analyst at an EM debt asset manager.\n"
    "You produce daily morning briefings for the portfolio manager of an "
    "EM local-currency sovereign bond fund.\n\n"
    "Given structured daily analytics data (regime classification, yield "
    "curve moves, PCA scores, VaR metrics, and alerts), produce a briefing "
    "with:\n\n"
    "1. **Headline** (one sentence, max 15 words): the single most "
    "important thing today.\n"
    "2. **Regime context** (1-2 sentences): current regime, confidence, "
    "what it implies.\n"
    "3. **Notable country moves** (2-3 sentences): which countries moved, "
    "direction, magnitude in bps, and any PCA z-score flags.\n"
    "4. **Risk update** (1-2 sentences): portfolio P&L, proximity to VaR "
    "bands, any breaches.\n"
    "5. **Alerts** (if any): list triggered alerts with severity.\n\n"
    "Rules:\n"
    "- Be precise and quantitative. No filler.\n"
    "- Use basis points for yield moves.\n"
    "- Flag anything that warrants PM attention with [ACTION NEEDED] prefix.\n"
    "- Total length: 8-12 sentences maximum.\n"
    '- If no alerts fired, say "No alerts triggered."\n'
)


def _resolve_system_prompt(state: AnalyticsState) -> str:
    return (
        state.config.get("briefing", {}).get("system_prompt")
        or DEFAULT_BRIEFING_SYSTEM_PROMPT
    )


def generate_briefing(
    state: AnalyticsState,
    date,
    llm_client: LLMClient,
    system_prompt: Optional[str] = None,
) -> tuple[str, str]:
    """Produce one briefing.

    Returns ``(date_str, briefing_text)`` so callers can drop the result
    directly into the cache dict.
    """
    payload = build_daily_payload(
        date,
        state.regime_features,
        state.pca_results,
        state.change_dfs,
        state.portfolio_pnl,
        state.all_alerts,
        state.var_95,
        state.var_99,
    )
    prompt = system_prompt or _resolve_system_prompt(state)
    user_msg = (
        "Generate the daily briefing for this data:\n\n"
        f"{json.dumps(payload, indent=2, default=str)}"
    )
    messages = [
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=user_msg),
    ]
    text = llm_client.complete(messages)
    return payload["date"], text


def run_briefing_workflow(
    state: AnalyticsState,
    dates: Iterable,
    llm_client: LLMClient,
    *,
    cache_path: str | Path = DEFAULT_BRIEFING_CACHE,
    system_prompt: Optional[str] = None,
    sleep_seconds: float = 0.0,
) -> dict[str, str]:
    """Generate briefings for ``dates``, using the on-disk cache.

    Already-cached dates are skipped (no API call). Failed dates are
    logged and skipped. The cache is rewritten at the end.

    Returns the full cache dict (cached + newly generated).
    """
    briefings = load_briefings(cache_path)
    n_cached = len(briefings)
    generated = 0
    failed: list[tuple[str, str]] = []

    for d in dates:
        target = str(pd.Timestamp(d).date())
        if target in briefings:
            logger.info("%s: cached (skipping API call)", target)
            continue
        try:
            date_out, text = generate_briefing(
                state, d, llm_client, system_prompt=system_prompt,
            )
            briefings[date_out] = text
            generated += 1
            logger.info("%s: generated via API", target)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception as exc:  # noqa: BLE001 — surface every provider error
            failed.append((target, f"{type(exc).__name__}: {exc}"))
            logger.warning("%s: failed - %s", target, exc)

    save_briefings(briefings, cache_path)
    logger.info(
        "Briefings: %d cached, %d generated, %d failed (total: %d)",
        n_cached, generated, len(failed), len(briefings),
    )
    return briefings
