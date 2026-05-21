"""Orchestration layer — the sole coordinator of Data and Models layers.

This package is the only place in the codebase that knows the *sequence*
of analytical steps. Everything else (the dashboard, the notebook, the
chatbot, the briefing scheduler) calls into this layer and receives
prepared state or streamed output.

Three orchestration surfaces are exposed:

* :class:`AnalyticsState` + :func:`build_analytics_state` — runs the
  full pipeline (load → PCA → regime → alerts → portfolio P&L → VaR)
  and returns a single dataclass that downstream consumers grab fields
  off of.
* :func:`generate_briefing` / :func:`run_briefing_workflow` — calls the
  Data layer (``build_daily_payload``) and the Models layer (an
  ``LLMClient``) to produce briefings. Handles caching via
  ``src.data.briefing_store``.
* :func:`orchestrate_chat` — minimal chat orchestrator (system prompt +
  history → streamed tokens). Foundation for future grounding (where
  the orchestrator would call ``build_daily_payload`` before
  ``client.stream``).
"""

from src.orchestration.briefing import (
    DEFAULT_BRIEFING_SYSTEM_PROMPT,
    generate_briefing,
    run_briefing_workflow,
)
from src.orchestration.chat import orchestrate_chat
from src.orchestration.health import (
    HEALTH_CHECK_PATH,
    PIPELINE_LOG_PATH,
    build_health_check,
    run_pipeline_step,
    write_health_check,
    write_pipeline_log,
)
from src.orchestration.pipeline import AnalyticsState, build_analytics_state

__all__ = [
    "AnalyticsState",
    "build_analytics_state",
    "generate_briefing",
    "run_briefing_workflow",
    "orchestrate_chat",
    "DEFAULT_BRIEFING_SYSTEM_PROMPT",
    "run_pipeline_step",
    "build_health_check",
    "write_pipeline_log",
    "write_health_check",
    "PIPELINE_LOG_PATH",
    "HEALTH_CHECK_PATH",
]
