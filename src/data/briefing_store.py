"""
On-disk persistence for cached LLM briefings.

The briefing pipeline in ``main.ipynb`` generates daily briefings via
Gemini and caches them by date so we don't re-bill the API on every
notebook re-run. The Streamlit "Daily Briefings" page also reads this
file to display historical briefings.

Both reader and writer are deliberately stateless functions: each call
opens, parses, optionally writes, and closes. Path is overridable so
tests can use ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_BRIEFING_CACHE: Path = Path("data/output/sample_briefings.json")


def load_briefings(path: str | Path = DEFAULT_BRIEFING_CACHE) -> dict[str, str]:
    """Return ``{date_str: briefing_text}``. Empty dict if file missing.

    Mirrors the notebook's ``CACHE_PATH`` load block — the only difference
    is that a missing file returns ``{}`` instead of leaving ``briefings``
    undefined.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_briefings(
    briefings: dict[str, str],
    path: str | Path = DEFAULT_BRIEFING_CACHE,
) -> None:
    """Write the briefing cache to disk (indent=2, UTF-8).

    Creates parent directories if needed. Overwrites the existing file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(briefings, f, indent=2)
