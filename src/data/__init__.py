"""Data layer — structured retrieval and on-disk persistence.

This package owns every static-data parser / local-file loader that used to
live inline in ``main.ipynb`` and the Streamlit pages. The rule going
forward is: if a module reads or writes a project artifact (CSV / JSON in
``data/output/``, briefing cache, alert history, etc.), it lives here.

Public surface:

* :func:`build_daily_payload` — assemble per-date analytics into a JSON
  payload the LLM (or any consumer) can ground in.
* :func:`load_briefings` / :func:`save_briefings` — round-trip the
  briefing cache (``data/output/sample_briefings.json``).

This package deliberately contains *no* vector-store or embedding code.
When semantic search becomes a real requirement, a sibling module
(e.g. ``src/data/vector_index.py``) will be added behind a clean interface
without touching the structured retrieval here.
"""
from src.data.briefing_context import build_daily_payload
from src.data.briefing_store import (
    DEFAULT_BRIEFING_CACHE,
    load_briefings,
    save_briefings,
)
from src.data.var_artifacts import (
    DEFAULT_OUTPUT_DIR,
    load_alert_history,
    load_country_outputs,
    load_decomposition,
    load_health_check,
    load_multi_nu,
    load_pipeline_log,
    load_stress_data,
)
from src.data.cache_keys import data_version

__all__ = [
    "build_daily_payload",
    "load_briefings",
    "save_briefings",
    "DEFAULT_BRIEFING_CACHE",
    "DEFAULT_OUTPUT_DIR",
    "load_stress_data",
    "load_multi_nu",
    "load_decomposition",
    "load_pipeline_log",
    "load_health_check",
    "load_alert_history",
    "load_country_outputs",
    "data_version",
]
