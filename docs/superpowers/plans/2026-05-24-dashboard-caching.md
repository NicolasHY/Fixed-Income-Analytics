# Dashboard Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Streamlit dashboard instant when switching sidebar pages and fast on first load after a server restart, by caching every disk loader and disk-persisting the expensive ones, with automatic invalidation when the notebook regenerates data.

**Architecture:** A pure `data_version(path)` helper returns a hashable signature (sorted `(filename, mtime)` walked recursively) of a data directory. It is computed once per rerun and passed as a regular argument into each `@st.cache_data` loader, so the cache auto-invalidates when files change. The three expensive raw-CSV-derived loaders use `persist="disk"` so their results survive server restarts. A sidebar "Refresh data" button clears the cache manually.

**Tech Stack:** Python, Streamlit (`@st.cache_data`), pandas, pytest.

---

## File Structure

- **Create** `src/data/cache_keys.py` — pure `data_version(path)` helper (no Streamlit import, so it is unit-testable).
- **Modify** `src/data/__init__.py` — export `data_version`.
- **Modify** `app.py` — import the helper, compute version keys, add/retrofit cache wrappers, add the Refresh button.
- **Create** `tests/test_cache_version.py` — unit tests for `data_version`.

Why a separate module: the test must run without Streamlit, and importing `app.py` triggers `st.set_page_config()` and other top-level Streamlit calls. Keeping the pure helper in `src/data/` (the project's data-layer home for file readers) lets the test import just the helper.

---

## Task 1: `data_version` helper (TDD)

**Files:**
- Create: `src/data/cache_keys.py`
- Modify: `src/data/__init__.py`
- Test: `tests/test_cache_version.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache_version.py`:

```python
"""Unit tests for the data_version cache-key helper."""
import os

from src.data import data_version


def test_missing_directory_returns_empty(tmp_path):
    assert data_version(tmp_path / "does_not_exist") == ()


def test_stable_when_unchanged(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    assert data_version(tmp_path) == data_version(tmp_path)


def test_changes_when_file_added(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    before = data_version(tmp_path)
    (tmp_path / "b.csv").write_text("y")
    assert data_version(tmp_path) != before


def test_changes_when_file_removed(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("x")
    (tmp_path / "b.csv").write_text("y")
    before = data_version(tmp_path)
    a.unlink()
    assert data_version(tmp_path) != before


def test_changes_when_mtime_changes(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("x")
    before = data_version(tmp_path)
    os.utime(a, (1_000_000_000, 1_000_000_000))  # fixed, deterministically different mtime
    assert data_version(tmp_path) != before


def test_recurses_into_subdirectories(tmp_path):
    sub = tmp_path / "Brazil"
    sub.mkdir()
    bond = sub / "bond.csv"
    bond.write_text("x")
    before = data_version(tmp_path)
    os.utime(bond, (1_000_000_000, 1_000_000_000))
    assert data_version(tmp_path) != before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache_version.py -v`
Expected: FAIL — `ImportError: cannot import name 'data_version' from 'src.data'`.

- [ ] **Step 3: Write the helper**

Create `src/data/cache_keys.py`:

```python
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
```

- [ ] **Step 4: Export it from the data package**

In `src/data/__init__.py`, add the import (after the existing `from src.data.var_artifacts import (...)` block, around line 35) and add `"data_version"` to `__all__`.

Add import line:

```python
from src.data.cache_keys import data_version
```

Add to the `__all__` list:

```python
    "data_version",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_cache_version.py -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 6: Commit**

```bash
git add src/data/cache_keys.py src/data/__init__.py tests/test_cache_version.py
git commit -m "feat(cache): add data_version cache-key helper"
```

---

## Task 2: Retrofit the existing cached loaders in `app.py`

Adds the version-key import, computes the two version keys once per rerun, gives the four already-cached loaders a `version` parameter (so they invalidate on data change), and disk-persists `_load_portfolio_data`. All call sites are updated in the same commit so the app stays runnable.

**Files:**
- Modify: `app.py` (import line ~22; cached-loader block ~600-622; call sites ~706, ~1190, ~988, ~1047, ~1085)

- [ ] **Step 1: Add `data_version` to the data-layer import**

Find (around line 22):

```python
from src.data import load_briefings as _load_briefings_from_disk
```

Replace with:

```python
from src.data import load_briefings as _load_briefings_from_disk, data_version
```

- [ ] **Step 2: Compute version keys and retrofit the four cached loaders**

Find the block (around lines 600-622):

```python
@st.cache_data
def _load_stress_data():
    return _load_stress_data_from_disk(OUT)


@st.cache_data
def _load_multi_nu():
    return _load_multi_nu_from_disk(OUT)


@st.cache_data
def _load_decomposition():
    return _load_decomposition_from_disk(OUT)


@st.cache_data(show_spinner="Loading portfolio data…")
def _load_portfolio_data():
    return build_portfolio_views()
```

Replace with:

```python
# Data-version cache keys — recomputed every rerun (cheap, stat-only).
# Passing these into a cached loader as a regular (non-underscore)
# argument makes Streamlit fold them into the cache key, so the cache
# auto-invalidates whenever the notebook regenerates the data files.
_OUT_VER = data_version(OUT)
_RAW_VER = data_version("data/raw")


@st.cache_data
def _load_stress_data(version):
    return _load_stress_data_from_disk(OUT)


@st.cache_data
def _load_multi_nu(version):
    return _load_multi_nu_from_disk(OUT)


@st.cache_data
def _load_decomposition(version):
    return _load_decomposition_from_disk(OUT)


@st.cache_data(show_spinner="Loading portfolio data…", persist="disk")
def _load_portfolio_data(version):
    return build_portfolio_views()
```

- [ ] **Step 3: Update the call sites for these four loaders**

Home page (around line 706):

```python
        _home_ports = _load_portfolio_data()
```
→
```python
        _home_ports = _load_portfolio_data(_RAW_VER)
```

Portfolios page (around line 1190):

```python
        portfolio_results = _load_portfolio_data()
```
→
```python
        portfolio_results = _load_portfolio_data(_RAW_VER)
```

VaR Engine tab 2 (around line 988):

```python
        data = _load_stress_data()
```
→
```python
        data = _load_stress_data(_OUT_VER)
```

VaR Engine tab 3 (around line 1047):

```python
        data = _load_multi_nu()
```
→
```python
        data = _load_multi_nu(_OUT_VER)
```

VaR Engine tab 4 (around line 1085):

```python
        data = _load_decomposition()
```
→
```python
        data = _load_decomposition(_OUT_VER)
```

- [ ] **Step 4: Syntax-check the app**

Run: `python -m py_compile app.py`
Expected: no output (exit 0). Any `SyntaxError` means an edit was misapplied.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(cache): version-key existing loaders, disk-persist portfolio data"
```

---

## Task 3: Cache the five currently-uncached loaders in `app.py`

These five loaders are re-read from disk on every rerun today — the direct cause of sidebar-page-switch lag. Add cached wrappers (in-session; small JSON/CSV) and switch the pages to call them.

**Files:**
- Modify: `app.py` (new wrappers after the block from Task 2; call sites ~807, ~830, ~857, ~1135, ~1881)

- [ ] **Step 1: Add the five cached wrappers**

Immediately after the `_load_portfolio_data` definition added in Task 2, insert:

```python
@st.cache_data
def _load_health_check(version):
    return _load_health_check_from_disk(OUT)


@st.cache_data
def _load_pipeline_log(version):
    return _load_pipeline_log_from_disk(OUT)


@st.cache_data
def _load_country_outputs(countries, version):
    return _load_country_outputs_from_disk(list(countries), OUT)


@st.cache_data
def _load_alert_history(version):
    return _load_alert_history_from_disk(OUT)


@st.cache_data
def _load_briefings(version):
    return _load_briefings_from_disk(OUT / "sample_briefings.json")
```

- [ ] **Step 2: Update the Pipeline Health call sites (around lines 807 and 830)**

```python
    checks = _load_health_check_from_disk(OUT)
```
→
```python
    checks = _load_health_check(_OUT_VER)
```

```python
    log = _load_pipeline_log_from_disk(OUT)
```
→
```python
    log = _load_pipeline_log(_OUT_VER)
```

- [ ] **Step 3: Update the Data Load call site (around line 857)**

`st.cache_data` cannot hash a Python `list`, so pass `COUNTRIES` as a tuple.

```python
    country_dfs, missing = _load_country_outputs_from_disk(COUNTRIES, OUT)
```
→
```python
    country_dfs, missing = _load_country_outputs(tuple(COUNTRIES), _OUT_VER)
```

- [ ] **Step 4: Update the Daily Briefings call site (around line 1135)**

```python
    briefings = _load_briefings_from_disk(OUT / "sample_briefings.json")
```
→
```python
    briefings = _load_briefings(_OUT_VER)
```

- [ ] **Step 5: Update the Alert History call site (around line 1881)**

```python
    alerts = _load_alert_history_from_disk(OUT)
```
→
```python
    alerts = _load_alert_history(_OUT_VER)
```

- [ ] **Step 6: Syntax-check the app**

Run: `python -m py_compile app.py`
Expected: no output (exit 0).

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat(cache): cache health, pipeline log, country outputs, alerts, briefings"
```

---

## Task 4: Disk-persist the two raw-CSV loaders in the Risk Statistics tab

`_load_yield_levels` and `_load_rf_data` are defined inside the Portfolios `tab_risk` block. Give them a `version` parameter, `persist="disk"`, and pass the module-level version keys at their call sites.

**Files:**
- Modify: `app.py` (`_load_yield_levels` def ~1437 + call ~1454; `_load_rf_data` def ~1647 + call ~1658)

- [ ] **Step 1: Disk-persist `_load_yield_levels`**

Find (around line 1437):

```python
        @st.cache_data(show_spinner="Loading yield levels…")
        def _load_yield_levels():
```
→
```python
        @st.cache_data(show_spinner="Loading yield levels…", persist="disk")
        def _load_yield_levels(version):
```

Find its call site (around line 1454):

```python
            yield_levels = _load_yield_levels()
```
→
```python
            yield_levels = _load_yield_levels(_RAW_VER)
```

- [ ] **Step 2: Disk-persist `_load_rf_data`**

Find (around line 1647):

```python
        @st.cache_data(show_spinner="Loading risk-free rates…")
        def _load_rf_data():
```
→
```python
        @st.cache_data(show_spinner="Loading risk-free rates…", persist="disk")
        def _load_rf_data(version):
```

Find its call site (around line 1658):

```python
        rf_data = _load_rf_data()
```
→
```python
        rf_data = _load_rf_data(_OUT_VER)
```

- [ ] **Step 3: Syntax-check the app**

Run: `python -m py_compile app.py`
Expected: no output (exit 0).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(cache): disk-persist yield-levels and risk-free-rate loaders"
```

---

## Task 5: Add the "Refresh data" sidebar button

A manual safety valve that clears both the in-session and the on-disk persisted caches.

**Files:**
- Modify: `app.py` (sidebar block, immediately before the `Stop Server` button ~659)

- [ ] **Step 1: Insert the Refresh button**

Find the Stop Server button (around line 659):

```python
    if st.button("Stop Server", key="stop_server_btn", type="secondary"):
```

Insert immediately before that line (same indentation — inside the `with st.sidebar:` block):

```python
    if st.button("Refresh data", key="refresh_data_btn", type="secondary"):
        st.cache_data.clear()
        st.rerun()

```

- [ ] **Step 2: Syntax-check the app**

Run: `python -m py_compile app.py`
Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(cache): add Refresh data button to clear caches"
```

---

## Task 6: Verification

**Files:** none (verification only)

- [ ] **Step 1: Run the new helper tests**

Run: `pytest tests/test_cache_version.py -v`
Expected: 6 passed.

- [ ] **Step 2: Run the full test suite (no regressions)**

Run: `pytest tests/`
Expected: all tests pass (same count as before plus the 6 new ones).

- [ ] **Step 3: Final syntax check**

Run: `python -m py_compile app.py`
Expected: no output (exit 0).

- [ ] **Step 4: Manual dashboard checklist**

Start the app: `streamlit run app.py`, then verify:

- Visit each sidebar page once (Home, Pipeline Health, Data Load, PCA & Regime, VaR Engine, Portfolios, Alert History, Daily Briefings). The first visit may show a spinner.
- Re-visit pages and switch between them → now **instant**, no reload spinner.
- Stop the server (Ctrl-C or the Stop Server button) and run `streamlit run app.py` again → the **first** Home/Portfolios load is fast (served from the disk-persisted cache), not a cold rebuild.
- Touch an output file to simulate a notebook re-run, e.g. PowerShell `(Get-Item data\output\health_check.json).LastWriteTime = Get-Date`, then reload the Pipeline Health page → it reloads with fresh data.
- Click **Refresh data** in the sidebar → the app reruns and all caches are rebuilt.

- [ ] **Step 5: Mark the plan complete (no commit needed — verification only)**

---

## Self-Review Notes

- **Spec coverage:** version helper (Task 1) ✓; cache the 5 uncached loaders (Task 3) ✓; version-key the 3 existing in-session caches (Task 2 — stress/multi_nu/decomposition) ✓; disk-persist the 3 expensive loaders — `_load_portfolio_data` (Task 2), `_load_yield_levels` + `_load_rf_data` (Task 4) ✓; Refresh button (Task 5) ✓; unit test + manual checklist (Tasks 1, 6) ✓; out-of-scope items left untouched ✓.
- **Hashability:** `COUNTRIES` is converted to a tuple at the `_load_country_outputs` call site (Step 3, Task 3) because `@st.cache_data` cannot hash a list.
- **Recursion:** `data_version` walks recursively so it detects content changes inside `data/raw/<Country>/` subfolders (raw CSVs are nested), not just top-level files.
- **No shadowing:** the cache-key parameter is named `version`, distinct from the imported `data_version` function.
- **Runnable between commits:** every `app.py` task updates loader definitions and their call sites together, so the app never references a removed `*_from_disk` direct call or a wrong signature mid-plan.
