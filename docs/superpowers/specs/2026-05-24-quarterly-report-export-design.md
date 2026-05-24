# Quarterly Report Export — Design Spec

**Date:** 2026-05-24  
**Feature:** Export Quarterly Report button in the Portfolios tab  
**Status:** Approved

---

## Overview

Add a self-contained quarterly report export to the Portfolios page of the Streamlit dashboard. A quarter dropdown + download button let the user export a pre-filled copy of `Report_Template.xlsx` covering any completed calendar quarter visible in the data. All Excel generation logic lives in a new `src/report_generator.py` module; `app.py` only owns the UI controls.

---

## 1. UI — Portfolios Page

**Placement:** A compact control row rendered at the top of the `elif page == "Portfolios":` block, above `st.tabs(...)`. It renders inside a `section-card`-styled container.

**Controls:**
- `st.selectbox` — quarter options, e.g. `"Q1 2026 (Jan – Mar 2026)"`. Populated by `get_available_quarters()`. Defaults to the index of the most recently completed quarter.
- `st.download_button` — label `"Export Quarterly Report (.xlsx)"`. On click, calls `generate_quarterly_report(...)` and serves the returned bytes with `mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"` and a filename like `EM_FI_Q1_2026_Report.xlsx`.

The download button is a native Streamlit download button (no intermediate spinner needed — generation is fast enough in-process).

---

## 2. Template Modifications — `Report_Template.xlsx`

Three new sheets are added to the workbook. Existing sheets (Raw_Data, Performance_Report, Alerts_Review) are unchanged in structure.

### Sheet: `Risk_Summary`

Professional two-column layout (Portfolio 1 | Portfolio 2) with a dark-navy header banner and three clearly labelled sections separated by blank rows and bold section titles:

| Section | Rows |
|---|---|
| Return Metrics | Cumulative Log Return (%), Annualised Return (%), Carry — Wtd Avg Yield (%), Roll-Down Return (est. %) |
| Risk & Ratio Metrics | Annualised Volatility (%), Maximum Drawdown (%), Sharpe Ratio (rf = €STR), Sortino Ratio (rf = €STR), Sharpe (rf = 0, ref), Sortino (MAR = 0, ref), Calmar Ratio |
| Bond Analytics | AUM (EUR), Modified Duration (yrs), DV01 (% of NAV per 1bp), DV01 (EUR per 1bp), Convexity (yrs²), YTM — Wtd Avg Benchmark (%), Yield Curve Slope (long−short, %) |

Static values only (no formulas) — Python writes pre-computed numbers.

### Sheet: `VaR_Analysis`

Two sub-tables stacked vertically, each with portfolio columns and confidence-level rows:

- **VaR & CVaR as % of NAV** — Parametric, Historical, Monte Carlo VaR and CVaR at α=5% and α=1% (i.e. 95% and 99% confidence), for both portfolios.
- **VaR & CVaR in EUR (based on AUM)** — Same grid scaled by AUM.

Static values only.

### Sheet: `KRD`

- Header banner.
- Per-country modified duration table: Country | MD (par bond, yrs) | KRD — Portfolio 1 (yrs) | KRD — Portfolio 2 (yrs).
- Below: Latest Benchmark Yield Snapshot table: Country | Yield (%) — latest available — for each portfolio's constituent countries.

Static values only.

### Formatting conventions (applied by Python via openpyxl):

- **Header banner row:** fill `#0d1b2a`, font white, bold, 13pt.
- **Section title rows:** fill `#1b3a5c`, font white, bold, 10pt.
- **Column header rows:** fill `#e2e8f0`, font `#0f172a`, bold, 9pt.
- **Data rows:** alternating white / `#f6f8fb`, font `#0f172a`, 9pt.
- **Number formats:** percentages as `0.00%`, EUR values as `€#,##0`, durations as `0.00`, ratios as `0.00`.
- **Column widths:** metric label col ~40, value cols ~22 each.
- **Borders:** thin `#cbd5e1` around all data cells.

---

## 3. New Module — `src/report_generator.py`

### `get_available_quarters(pnl: pd.Series) -> list[dict]`

Returns a list of dicts, one per completed calendar quarter present in the PnL index, in reverse chronological order:

```python
{"label": "Q1 2026 (Jan – Mar 2026)", "start": date(2026, 1, 1), "end": date(2026, 3, 31)}
```

A quarter is "completed" if its end date is before today. The most recently completed quarter is index 0.

### `generate_quarterly_report(quarter: dict, p1: dict, p2: dict, alert_history: dict, rs1: dict, rs2: dict, rf_data) -> bytes`

Parameters:
- `quarter` — one entry from `get_available_quarters()`
- `p1`, `p2` — portfolio view dicts from `build_portfolio_views()`
- `alert_history` — raw alert dict from `load_alert_history()`
- `rs1`, `rs2` — risk-stat dicts computed by the `_risk_stats()` helper (extracted into a shared helper or re-computed inside the generator)
- `rf_data` — risk-free rate DataFrame (may be None)

Returns the Excel file as `bytes` for `st.download_button`.

**Internal steps:**
1. Load `Report_Template.xlsx` into an openpyxl workbook (`keep_vba=False`, `data_only=False` to preserve array formulas).
2. Build the Raw_Data rows for the quarter (same logic as the standalone rebuild script: HC + LC daily P&L, rolling VaR, regime from alert_history, NAV indexed to 1000 at quarter start).
3. Clear and rewrite the Raw_Data table; update its `ref` to match the new row count.
4. Update `Performance_Report!D5` (Report_Start_Date) and `Performance_Report!D6` (Report_End_Date).
5. Write Risk_Summary, VaR_Analysis, KRD sheets with pre-computed values and formatting.
6. Save to a `BytesIO` buffer and return `.getvalue()`.

### Risk-stat extraction

`_risk_stats()` currently lives inline in `app.py`. To avoid circular imports or code duplication, `report_generator.py` will accept pre-computed `rs1`/`rs2` dicts (the caller passes them in). The app already has these available at the point where it renders the Risk Statistics tab.

---

## 4. Data Flow

```
app.py  Portfolios page
  │
  ├─ get_available_quarters(p1["pnl"])  →  quarter options list
  │
  └─ on download click:
       generate_quarterly_report(
           quarter, p1, p2,
           alert_history, rs1, rs2, rf_data
       )  →  bytes  →  st.download_button
```

`rs1` / `rs2` are computed once (already happen for the Risk Statistics tab render) and passed through. No duplicate computation.

---

## 5. Files Changed

| File | Change |
|---|---|
| `Report_Template.xlsx` | Add Risk_Summary, VaR_Analysis, KRD sheets with formatting |
| `src/report_generator.py` | New module: `get_available_quarters`, `generate_quarterly_report` |
| `app.py` | Add quarter selector + download button at top of Portfolios page; pass `rs1`/`rs2` to generator |

---

## 6. Out of Scope

- Embedded Plotly charts in the Excel file.
- Email delivery.
- Scheduling / automated generation.
- Any changes to the existing three template sheets (Raw_Data, Performance_Report, Alerts_Review).
