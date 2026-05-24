"""Cache-key helpers for the Streamlit dashboard.

``data_version`` produces a hashable signature of a data directory so it
can be passed as an argument into ``@st.cache_data`` loaders. When any
file under the directory changes, the signature changes and Streamlit
invalidates the cached result automatically.
"""
from __future__ import annotations

from pathlib import Path


def data_version(path: str | Path) -> tuple[tuple[str, float], ...]:
    """Hashable signature of every file under *path*: sorted (name, mtime).

    Walks recursively (raw data lives in per-country subfolders). Stats
    files only — never reads them — so it is cheap to call on every
    Streamlit rerun. Returns an empty tuple for a missing directory so a
    not-yet-generated data folder is still a stable, valid cache key.
    An existing-but-empty directory also returns ``()``; the collision is
    intentional and safe because dashboard loaders return ``None`` when
    their files are absent, making "missing dir" and "empty dir" the same
    "no data" state.
    Files that vanish mid-walk are skipped rather than raising.
    """
    root = Path(path)
    if not root.exists():
        return ()
    entries: list[tuple[str, float]] = []
    for p in root.rglob("*"):
        try:
            if p.is_file():
                entries.append((p.as_posix(), p.stat().st_mtime))
        except OSError:
            continue
    return tuple(sorted(entries))
