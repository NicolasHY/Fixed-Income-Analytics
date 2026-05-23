# Excel Report Export — Design

**Date:** 2026-05-23
**Status:** Approved (pending spec review)
**Author:** Nicolas Henry (with Claude Code)

## Goal

Export the EM Fixed Income analytics into a polished, consistent Excel
workbook driven by a **reusable, hand-editable `.xlsx` template**. The
template defines all visual chrome (branding, colours, fonts, titles,
headers, disclaimer) so future exports stay consistent and the template
can be tweaked in Excel without touching code.

## Scope (decided)

| Decision | Choice |
|----------|--------|
| Report sections | Cover & pipeline health, Risk/VaR summary, PCA & regime, Alerts |
| Portfolio coverage | **Both** portfolios (EM Hard-Currency + EM Local-Currency) |
| Charts | **Native Excel charts** (real `BarChart`/`LineChart`, editable in Excel) |
| Template structure | **Blueprint + clone** — one styled blueprint sheet cloned per portfolio |
| Rendering module home | New `src/reporting/` package |
| Headline VaR / breach rate | **Monte Carlo t-copula** (matches existing pipeline contract) |
| Excel library | `openpyxl` (can load+modify a template; xlsxwriter is write-only) |
| Data source | Computed **fresh** per portfolio (no reliance on single-portfolio CSVs in `data/output/`) |

## Key structural insight

PCA, regime detection, and the alert scan run on **raw country yields**,
not fund weights — they are **portfolio-independent**. Only the cover
metadata and the risk/VaR numbers differ per fund. The report therefore
splits cleanly into:

- **Shared sheets:** `PCA & Regime`, `Alerts` (computed once)
- **Per-portfolio sheets:** cloned from one blueprint, one per fund

## Architecture & data flow

```
config/funds.yaml
   │
   ▼  build_report_state(config, data_dir)      ← src/orchestration/report_state.py  (NEW)
   ├─ load raw yields once (load_all_countries_combined)
   ├─ PCA + regime + alerts        (shared — portfolio-independent)
   └─ for pdef in config["portfolios"]:
        build_portfolio_pnl_from_def → pnl, proxy_dy
        full VaR suite via src/quant/var_engine.py:
          • compute_parametric_var (95/99)
          • compute_historical_var (config windows)
          • compute_stressed_var   (each stress window)
          • compute_multi_nu_var_table
          • compute_mc_t_copula_var (HEADLINE: var_95/99, cvar_95/99)
          • compute_factor_idio_decomposition
          • kupiec_pof + christoffersen_test (vs historical VaR)
        build_health_check(pipeline_log, pnl, regime_features, var_95)
   │
   ▼  ReportState (data contract)
   │
templates/report_template.xlsx  +  write_report(state, template, out)   ← src/reporting/excel_report.py (NEW)
   │
   ▼
data/output/em_fi_report_<YYYY-MM-DD>.xlsx
```

Two responsibilities, cleanly split:
- **Orchestration computes** → produces `ReportState`.
- **Reporting renders** → consumes `ReportState`, writes Excel.
The exporter does no analytics; the orchestrator does no Excel.

## Components

### 1. `src/orchestration/report_state.py` (NEW)

Dataclasses forming the report's data contract:

```python
@dataclass
class PortfolioRisk:
    name: str            # e.g. "EM Hard Currency Sustainable" (cover/title)
    short: str           # e.g. "HC" — sheet-name prefix
    label: str           # e.g. "Hard Currency" — short display name
    aum_eur: float
    duration: float
    weights: dict[str, float]      # normalised
    var_95: float; var_99: float   # MC t-copula headline
    cvar_95: float; cvar_99: float
    parametric: dict               # {0.95: {...}, 0.99: {...}}
    historical: dict               # {window_label: {0.95:..., 0.99:...}}
    stressed: dict                 # {window_name: {VaR_95, VaR_99, CVaR_95, CVaR_99}}
    multi_nu: pd.DataFrame         # df index nu, cols VaR/CVaR
    decomposition: dict            # pct_systematic/idiosyncratic, betas df, var_total...
    backtests: dict                # kupiec + christoffersen results
    health: list[dict]             # build_health_check output (per-portfolio)

@dataclass
class ReportState:
    run_date: pd.Timestamp
    data_end: pd.Timestamp
    config: dict
    # shared
    pca_results: dict
    regime_features: pd.DataFrame
    explained_variance: list[float]
    loadings: pd.DataFrame         # per-country PC betas
    alerts: dict                   # run_alert_scan output
    # per portfolio
    portfolios: list[PortfolioRisk]
```

`build_report_state(config=None, data_dir="data/raw", slice_end=None)`:
- Mirrors the structure of `build_analytics_state` but iterates
  `config["portfolios"]` and assembles the full VaR suite per fund.
- Reuses existing functions only; no analytics logic is duplicated.
- `slice_end` supported for deterministic tests (same convention as
  `build_analytics_state`).
- No API keys required (no LLM / FRED).

`pipeline_log` for `build_health_check` is a minimal synthetic log (the
report is not a full pipeline run); freshness/breach/regime cards are the
meaningful ones. The "all steps passed" / "runtime" cards are populated
from a single nominal entry so the RAG table renders consistently.

### 2. `templates/report_template.xlsx` (NEW) + `scripts/build_report_template.py` (NEW)

Because the template can't be hand-drawn by an agent, a generator script
emits it with openpyxl. The user opens it in Excel to tweak, and can
re-run the script to regenerate from scratch. The script is the
authoritative description of the template's structure.

Template sheets:

- **Cover** — brand title, colours/fonts, logo (`assets/app_icon.ico`
  converted/embedded if usable; otherwise text title), metadata labels
  with text tokens `{{run_date}}`, `{{data_end}}`, `{{n_portfolios}}`,
  and a disclaimer block.
- **`_tmpl_Portfolio`** (blueprint, deleted from final output) — styled:
  - Title row with `{{portfolio_name}}`, `{{aum}}`, `{{duration}}` tokens.
  - **Health (RAG) block** with header styling + marker `{{HEALTH_TABLE}}`.
  - **Risk block**: VaR/CVaR matrix, stressed scenarios, multi-nu table,
    decomposition summary — each with a header and a marker cell
    (`{{VAR_TABLE}}`, `{{STRESSED_TABLE}}`, `{{MULTINU_TABLE}}`,
    `{{DECOMP_TABLE}}`).
  - Frozen panes, column widths, brand fills on headers.
- **PCA & Regime** (shared) — explained-variance table marker
  `{{EXPVAR_TABLE}}`, loadings/betas marker `{{LOADINGS_TABLE}}`, current
  regime summary marker `{{REGIME_TABLE}}`, chart anchor area.
- **Alerts** (shared) — alert-history table marker `{{ALERTS_TABLE}}`.

### 3. `src/reporting/excel_report.py` (NEW)

`write_report(state, template_path, output_path)`:
1. `load_workbook(template_path)`.
2. Fill **Cover** tokens.
3. For each `PortfolioRisk`: `copy_worksheet(_tmpl_Portfolio)`, rename to
   `"{short} — {label}"` (e.g. `HC — Hard Currency`; capped at Excel's
   31-char sheet-name limit), replace tokens, write each table at its
   marker cell, apply row styling.
4. Fill shared **PCA & Regime** and **Alerts** sheets.
5. Add **native Excel charts** referencing written ranges:
   - Per portfolio: VaR-by-method `BarChart`; stressed-scenario `BarChart`.
   - PCA & Regime: explained-variance `BarChart`.
6. Delete `_tmpl_Portfolio`; order sheets (Cover → portfolios → shared);
   `save(output_path)`.

Helpers (small, single-purpose, unit-testable):
`_replace_tokens(ws, mapping)`, `_find_marker(ws, token) -> (row, col)`,
`_write_table(ws, anchor, df, *, header_style)`, `_style_header(...)`,
`_add_bar_chart(ws, data_ref, cats_ref, anchor, title)`.

### 4. `export_excel.py` (repo-root CLI, NEW)

Mirrors `export_pdf.py`. Args: `--template` (default
`templates/report_template.xlsx`), `--output` (default
`data/output/em_fi_report_<date>.xlsx`), `--data-dir`. Calls
`build_report_state` then `write_report`, prints the resolved path.

### 5. `requirements.txt`

Add `openpyxl`.

## Why sentinel markers, not defined names

`openpyxl.copy_worksheet` duplicates cell styles, column widths, and
merged cells but **not** workbook-level defined names or charts. So:
- Anchors use in-cell sentinel tokens located by scanning the cloned
  sheet — robust across clones.
- Charts are added by code **after** cloning, so the blueprint stays
  chart-free (nothing to lose in the copy).

## Error handling

- Missing template or `_tmpl_Portfolio` sheet → explicit error naming the
  expected path/sheet.
- Missing raw data → propagates unchanged from `data_loader`.
- A test asserts **no `{{...}}` token survives** in the output, catching
  unfilled placeholders.

## Testing (`tests/test_excel_report.py`, synthetic data only)

Consistent with the repo convention (`conftest.py` synthetic fixtures, no
real market data).

- **Template generator:** `build_report_template.py` produces a workbook
  that loads, has the expected sheet names, and contains the expected
  marker cells.
- **Renderer:** build a synthetic `ReportState` → `write_report` to a temp
  path → reopen with openpyxl and assert:
  - expected sheet names present; one risk sheet per portfolio;
    `_tmpl_Portfolio` removed.
  - key cells match expected values (headline VaR, a stressed number, a
    health detail string).
  - ≥1 chart on each chart-bearing sheet (`len(ws._charts) >= 1`).
  - no surviving `{{` token anywhere.
- **State builder (light):** `build_report_state(slice_end=...)` on the
  existing synthetic raw data returns one `PortfolioRisk` per configured
  portfolio with finite headline VaR. (Skipped gracefully if raw data
  absent, matching other data-dependent tests.)

## Out of scope (YAGNI)

- Refactoring `main.ipynb` to consume `build_report_state` (the notebook
  keeps its current cells; the new function is additive). Optional future
  cleanup, noted but not done here.
- PDF/HTML report formats (the `src/reporting/` package leaves room, but
  only Excel is built now).
- LLM briefing text in the workbook (no API dependency in the report).
