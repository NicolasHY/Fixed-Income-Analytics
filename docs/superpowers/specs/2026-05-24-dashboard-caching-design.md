# Dashboard Caching — Design

**Date:** 2026-05-24
**Status:** Approved (pending spec review)
**Author:** Nicolas Henry (with Claude Code)

## Goal

Make the Streamlit dashboard (`app.py`) feel instant when switching
sidebar pages, and fast on the first load after starting the server.
Streamlit re-runs the entire script top-to-bottom on every interaction
(page switch, inner-tab click, widget change), so any uncached disk read
or heavy computation is paid again each time.

Two pain points (confirmed with the user):

1. **Switching sidebar pages** is slow — several loaders re-read CSV/JSON
   from disk on every rerun.
2. **First load after starting** is slow — the expensive portfolio P&L
   build runs cold, and an in-session cache is wiped when the server
   stops.

## Scope (decided)

| Decision | Choice |
|----------|--------|
| Approach | In-session caching for all loaders **+ disk persistence** for the expensive ones |
| Invalidation | Auto, via a data-version cache key (file mtimes) **+** a manual "Refresh data" button |
| Disk-persisted loaders | `_load_portfolio_data`, `_load_yield_levels`, `_load_rf_data` |
| In-session-only loaders | the small JSON/CSV artifact loaders |
| Out of scope | Caching `_risk_stats` / Plotly figure construction on the Portfolios page |

## Root cause

`@st.cache_data` already wraps `_load_portfolio_data`, `_load_stress_data`,
`_load_multi_nu`, `_load_decomposition`, `_load_yield_levels`, and
`_load_rf_data`. But:

- Five loaders are **not cached at all** and re-read disk every rerun:
  `load_health_check`, `load_pipeline_log`, `load_country_outputs`,
  `load_alert_history`, `load_briefings`.
- The existing in-session caches **do not survive a server restart**, so
  the first load after `streamlit run` is always cold.
- The existing caches have **no invalidation key**, so after the notebook
  regenerates `data/output/` within a long-running session they can serve
  stale numbers.

## Design

### 1. Data-version helper (new, pure)

```python
def _data_version(path: str | Path) -> tuple[tuple[str, float], ...]:
    """Hashable signature of a directory's files: sorted (name, mtime).

    Cheap — stats files, never reads them. Returns () for a missing dir
    so a not-yet-generated data folder is a stable, valid key.
    """
```

Computed once near the top of each rerun:

- `_OUT_VER = _data_version("data/output")` — notebook artifacts
- `_RAW_VER = _data_version("data/raw")` — source country CSVs

The signature is passed into each cached loader as a **regular argument**
(no leading underscore) so Streamlit includes it in the cache key. When a
file changes, its mtime changes, the signature changes, and the cache
auto-invalidates and reloads. When nothing changed, it is an instant
cache hit.

### 2. Cache wrappers

Every disk loader is wrapped in `@st.cache_data` taking a `data_version`
argument. The expensive raw-CSV-derived loaders also persist to disk.

| Loader | Caching | Version key | Reads |
|---|---|---|---|
| `_load_health_check` (new wrapper) | in-session | `data/output` | `health_check.json` |
| `_load_pipeline_log` (new wrapper) | in-session | `data/output` | `pipeline_log.json` |
| `_load_country_outputs` (new wrapper) | in-session | `data/output` | 7× `<Country>.csv` |
| `_load_alert_history` (new wrapper) | in-session | `data/output` | `alert_history.json` |
| `_load_briefings` (new wrapper) | in-session | `data/output` | `sample_briefings.json` |
| `_load_stress_data` (add key) | in-session | `data/output` | VaR sidecars |
| `_load_multi_nu` (add key) | in-session | `data/output` | multi-ν sidecars |
| `_load_decomposition` (add key) | in-session | `data/output` | decomposition sidecars |
| `_load_portfolio_data` (add key + persist) | **`persist="disk"`** | `data/raw` | all raw CSVs + P&L proxy |
| `_load_yield_levels` (add key + persist) | **`persist="disk"`** | `data/raw` | all raw CSVs |
| `_load_rf_data` (add key + persist) | **`persist="disk"`** | `data/output` | `risk_free_rates.csv` / FRED |

The three `persist="disk"` loaders are what make the **first load after
restart** fast — Streamlit pickles their results to its on-disk cache, so
they survive `streamlit run` restarts. All three return only
pandas objects / plain dicts / `None`, which pickle cleanly.

`_load_yield_levels` and `_load_rf_data` are defined inside the
Portfolios `tab_risk` block today. To cache them keyed on a version
computed at the top of the script, the `data_version` argument is passed
in from the module-level value; the function definitions can stay where
they are.

### 3. Manual "Refresh data" button

Added to the sidebar near "Stop Server":

```python
if st.button("Refresh data", key="refresh_data_btn", type="secondary"):
    st.cache_data.clear()
    st.rerun()
```

`st.cache_data.clear()` wipes both the in-session and the on-disk
persisted caches — a safety valve if anything ever looks stale.

## Data flow / invalidation

```
every rerun:
  _OUT_VER = _data_version("data/output")   # cheap stat-only scan
  _RAW_VER = _data_version("data/raw")
        │
        ▼  passed as cache key into each loader
  files unchanged  → cache key identical → instant cache hit
  notebook re-run  → file mtimes change  → key changes → reload
  Refresh button   → st.cache_data.clear() → full reload
        │
  server restart   → in-session caches gone, but the three
                     persist="disk" loaders read from disk cache → fast first load
```

## Error handling

- Loaders keep returning `None` on missing files — unchanged. The
  pages' existing "run the notebook first" warnings still fire.
- `_data_version` tolerates a missing directory (returns `()`), so a
  fresh checkout with no `data/` folder does not crash.
- Disk-persisted returns are all picklable (pandas + plain dict + `None`).

## Testing

- **Unit** (`tests/test_cache_version.py`, synthetic temp files, no
  Streamlit): the signature is stable across calls when files are
  unchanged; changes when a file's content/mtime changes; changes when a
  file is added or removed; returns `()` for a missing directory.
- **Manual checklist:**
  - Switch sidebar pages repeatedly → instant after the first visit.
  - Stop and restart the server → first page load is fast (disk cache).
  - Touch / regenerate a file in `data/output/` → affected page reloads
    with fresh numbers.
  - Click "Refresh data" → everything reloads.

## Out of scope

- Caching `_risk_stats` and the Plotly figure construction on the
  Portfolios page. The user did not report inner-tab lag, so this stays
  minimal; it can be revisited if the Portfolios tabs ever feel slow.
- Reducing the redundant raw-CSV reads inside `build_portfolio_views` /
  `_apply_daily_carry` (a separate optimisation).
