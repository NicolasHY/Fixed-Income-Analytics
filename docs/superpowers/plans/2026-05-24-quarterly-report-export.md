# Quarterly Report Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a quarter selector + download button to the Portfolios page that exports a pre-filled copy of `Report_Template.xlsx` covering the selected calendar quarter, including three new professional analytics sheets.

**Architecture:** All Excel-generation logic lives in a new `src/report_generator.py` (self-contained — loads its own data, no Streamlit coupling). A new `src/services/risk_stats.py` extracts the risk-stat computation that currently sits inline in `app.py`, so both the dashboard and the generator share one implementation. `app.py` is updated to import from both modules, move two cached loaders to module scope, pre-compute `rs1`/`rs2` before the tabs, and render the export UI above the existing tabs.

**Tech Stack:** Python 3.12, openpyxl (already installed), pandas, numpy, scipy, Streamlit — no new dependencies.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/services/risk_stats.py` | **Create** | `compute_risk_stats(pdef, pnl, yield_levels, rf_data) -> dict` |
| `src/report_generator.py` | **Create** | `get_available_quarters(pnl)`, sheet-writer helpers, `generate_quarterly_report(quarter)` |
| `tests/test_risk_stats.py` | **Create** | Unit tests for `compute_risk_stats` |
| `tests/test_report_generator.py` | **Create** | Unit + integration tests for the generator |
| `Report_Template.xlsx` | **Modify** | Add Risk_Summary, VaR_Analysis, KRD sheets via Python script |
| `app.py` | **Modify** | Move two loaders to module scope; import shared functions; add export UI above tabs |

---

## Task 1 — Create feature branch

**Files:** none

- [ ] **Create and switch to branch**

```bash
git checkout -b feature/quarterly-report-export
```

- [ ] **Verify**

```bash
git branch --show-current
# Expected: feature/quarterly-report-export
```

---

## Task 2 — Extract `compute_risk_stats` to `src/services/risk_stats.py`

**Files:**
- Create: `src/services/risk_stats.py`
- Create: `tests/test_risk_stats.py`

This extracts the `_risk_stats()` function currently defined inline inside `with tab_risk:` in `app.py` (lines 1581–1762). The new module-level function is identical in logic but takes explicit parameters and adds `c_vals` to the return dict (needed by the KRD sheet writer).

- [ ] **Write the failing tests** (`tests/test_risk_stats.py`)

```python
"""Tests for src/services/risk_stats.py — compute_risk_stats()."""
import numpy as np
import pandas as pd
import pytest
from src.services.risk_stats import compute_risk_stats


@pytest.fixture
def pdef():
    return {
        "name": "Test Fund",
        "weights": {"Brazil": 50.0, "Mexico": 50.0},
        "effective_duration": 5.0,
        "benchmark_maturity": "5Y",
        "aum_eur": 1_000_000,
    }


@pytest.fixture
def pnl():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    return pd.Series(rng.normal(0.0002, 0.003, 500), index=dates)


@pytest.fixture
def yield_levels(pnl):
    rng = np.random.default_rng(99)
    idx = pnl.index
    return {
        "Brazil": pd.DataFrame(
            {"5Y": rng.uniform(8, 12, len(idx)), "3Y": rng.uniform(7, 10, len(idx))},
            index=idx,
        ),
        "Mexico": pd.DataFrame(
            {"5Y": rng.uniform(7, 10, len(idx)), "3Y": rng.uniform(6, 9, len(idx))},
            index=idx,
        ),
    }


def test_returns_required_keys(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    for key in [
        "ann_ret", "ann_vol", "max_dd", "sharpe_zero", "sortino_zero", "calmar",
        "mod_dur", "dv01", "dv01_eur", "krd", "var_rows", "var_rows_eur",
        "c_vals", "carry", "rolldown", "convexity", "ytm", "yc_slope",
        "current_estr", "current_sofr", "avg_estr",
    ]:
        assert key in result, f"Missing key: {key}"


def test_vol_is_positive(pdef, pnl, yield_levels):
    assert compute_risk_stats(pdef, pnl, yield_levels)["ann_vol"] > 0


def test_max_drawdown_non_positive(pdef, pnl, yield_levels):
    assert compute_risk_stats(pdef, pnl, yield_levels)["max_dd"] <= 0


def test_var_rows_length_and_positivity(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    assert len(result["var_rows"]) == 2          # α=5% and α=10%
    for vr in result["var_rows"]:
        assert vr["Param VaR (%)"] > 0
        assert vr["Hist VaR (%)"] > 0


def test_krd_sums_to_mod_dur(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    krd_sum = sum(result["krd"].values())
    assert abs(krd_sum - result["mod_dur"]) < 1e-6


def test_dv01_eur_scales_by_aum(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    expected = result["dv01"] / 100 * pdef["aum_eur"] / 100
    # dv01_eur = mod_dur * 0.0001 * aum ; dv01 = mod_dur * 0.01
    # => dv01_eur = dv01 / 100 * aum
    assert abs(result["dv01_eur"] - result["dv01"] / 100 * pdef["aum_eur"]) < 1e-6


def test_no_rf_data_falls_back_to_zero_rf(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels, rf_data=None)
    assert result["sharpe"] == result["sharpe_zero"]
    assert result["sortino"] == result["sortino_zero"]
```

- [ ] **Run tests — verify they fail** (module doesn't exist yet)

```bash
pytest tests/test_risk_stats.py -v
# Expected: ImportError / ModuleNotFoundError
```

- [ ] **Create `src/services/risk_stats.py`**

```python
"""
Shared risk-stat computation used by both app.py (Risk Statistics tab)
and src/report_generator.py (quarterly Excel export).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_risk_stats(
    pdef: dict,
    pnl: pd.Series,
    yield_levels: dict,
    rf_data=None,
) -> dict:
    """
    Compute return, risk, bond-analytics and VaR statistics for one portfolio.

    Parameters
    ----------
    pdef : dict
        One entry from config["portfolios"] — must have keys:
        weights, effective_duration, benchmark_maturity, aum_eur.
    pnl : pd.Series
        Daily portfolio P&L as a decimal fraction, business-date index.
    yield_levels : dict[str, pd.DataFrame]
        {country: wide DataFrame of yield levels (%) indexed by date}.
    rf_data : pd.DataFrame or None
        Risk-free rate table with columns estr_pct, sofr_pct; None → rf = 0.

    Returns
    -------
    dict with keys: ann_ret, cum_log, carry, rolldown, ann_vol, max_dd,
        sharpe, sortino, sharpe_zero, sortino_zero, calmar,
        mod_dur, dv01, dv01_eur, convexity, ytm, yc_slope,
        krd, md_by_c, c_vals,
        var_rows, var_rows_eur,
        current_estr, current_sofr, avg_estr, aum.
    """
    def _par_md(yield_pct: float, T: int) -> float:
        y = yield_pct / 100
        return float(T) if y <= 0 else (1 - (1 + y) ** (-T)) / y

    raw_w: dict[str, float] = pdef["weights"]
    tot_w = sum(raw_w.values())
    w = {k: v / tot_w for k, v in raw_w.items()}
    D = float(pdef["effective_duration"])
    mat = pdef["benchmark_maturity"]
    mat_n = int(mat[:-1])
    n = len(pnl)
    aum = float(pdef.get("aum_eur", 0))

    # ── Return metrics ────────────────────────────────────────────────────────
    cum_log = float(np.log1p(pnl).sum() * 100)
    ann_ret = float(((1 + pnl).prod() ** (252 / n) - 1) * 100)

    # ── Carry: portfolio-weighted latest benchmark yield ──────────────────────
    c_vals: dict[str, float] = {}
    for c in w:
        if c in yield_levels and mat in yield_levels[c].columns:
            s = yield_levels[c][mat].dropna()
            if len(s) > 0:
                c_vals[c] = float(s.iloc[-1])
    if c_vals:
        ws_ = sum(w[c] for c in c_vals)
        carry = sum(w[c] * c_vals[c] for c in c_vals) / ws_
    else:
        carry = np.nan

    # ── Modified duration (par bond approximation) ───────────────────────────
    md_by_c: dict[str, float] = {c: _par_md(c_vals[c], mat_n) for c in c_vals}
    for c in w:
        if c not in md_by_c:
            md_by_c[c] = D

    # ── Roll-down ─────────────────────────────────────────────────────────────
    rd_vals: dict[str, float] = {}
    for c in w:
        if c not in yield_levels:
            continue
        avail_nums = sorted(int(x[:-1]) for x in yield_levels[c].columns)
        shorter = [m for m in avail_nums if m < mat_n]
        if not shorter:
            continue
        ns = max(shorter)
        sub = yield_levels[c][[mat, f"{ns}Y"]].dropna()
        if len(sub) == 0:
            continue
        row_ = sub.iloc[-1]
        slope = (row_[mat] - row_[f"{ns}Y"]) / (mat_n - ns)
        rd_vals[c] = md_by_c[c] * slope
    if rd_vals:
        ws_rd = sum(w[c] for c in rd_vals)
        rolldown = sum(w[c] * rd_vals[c] for c in rd_vals) / ws_rd
    else:
        rolldown = np.nan

    # ── Volatility & drawdown ─────────────────────────────────────────────────
    ann_vol = float(pnl.std() * np.sqrt(252) * 100)
    cum_s = (1 + pnl).cumprod()
    max_dd = float(((cum_s / cum_s.cummax()) - 1).min() * 100)

    # ── Portfolio duration analytics ──────────────────────────────────────────
    ws_md = sum(w[c] for c in md_by_c)
    mod_dur = sum(w[c] * md_by_c[c] for c in md_by_c) / ws_md
    dv01 = mod_dur * 0.01
    dv01_eur = mod_dur * 0.0001 * aum if aum else np.nan
    ytm = carry

    conv_by_c: dict[str, float] = {}
    for c in md_by_c:
        if c in c_vals:
            y_f = c_vals[c] / 100
            d_mac = md_by_c[c] * (1 + y_f)
            conv_by_c[c] = d_mac * (d_mac + 1) / (1 + y_f) ** 2
    if conv_by_c:
        ws_cv = sum(w[c] for c in conv_by_c)
        convexity = sum(w[c] * conv_by_c[c] for c in conv_by_c) / ws_cv
    else:
        convexity = np.nan

    sl_vals: dict[str, float] = {}
    for c in w:
        if c not in yield_levels:
            continue
        lr = yield_levels[c].dropna(how="all").iloc[-1].dropna()
        avail = sorted(lr.index, key=lambda x: int(x[:-1]))
        if len(avail) >= 2:
            sl_vals[c] = float(lr[avail[-1]] - lr[avail[0]])
    if sl_vals:
        ws_sl = sum(w[c] for c in sl_vals)
        yc_slope = sum(w[c] * sl_vals[c] for c in sl_vals) / ws_sl
    else:
        yc_slope = np.nan

    krd = {c: w[c] * md_by_c.get(c, D) for c in w}

    # ── Ratios rf = 0 ─────────────────────────────────────────────────────────
    sharpe_zero = (ann_ret / 100) / (ann_vol / 100) if ann_vol > 0 else np.nan
    ds_zero = float(np.mean(np.minimum(pnl, 0.0) ** 2))
    sortino_zero = (
        (ann_ret / 100) / (np.sqrt(ds_zero) * np.sqrt(252)) if ds_zero > 0 else np.nan
    )
    calmar = (ann_ret / 100) / abs(max_dd / 100) if max_dd != 0 else np.nan

    # ── Ratios rf = €STR ──────────────────────────────────────────────────────
    current_estr = np.nan
    current_sofr = np.nan
    avg_estr = np.nan
    sharpe = sharpe_zero
    sortino = sortino_zero
    if rf_data is not None:
        try:
            from src.risk_free import align_rf_to_pnl
            rf_estr = align_rf_to_pnl(rf_data, pnl, column="estr_pct")
            common = pnl.index.intersection(rf_estr.index)
            excess = pnl.loc[common] - rf_estr.loc[common]
            n_exc = len(excess)
            ann_exc = float(((1 + excess).prod() ** (252 / n_exc) - 1) * 100)
            exc_vol = float(excess.std() * np.sqrt(252) * 100)
            sharpe = (ann_exc / 100) / (exc_vol / 100) if exc_vol > 0 else np.nan
            ds_rf = float(np.mean(np.minimum(excess, 0.0) ** 2))
            sortino = (
                (ann_exc / 100) / (np.sqrt(ds_rf) * np.sqrt(252)) if ds_rf > 0 else np.nan
            )
            avg_estr = float(
                rf_data["estr_pct"].reindex(pnl.index, method="ffill").dropna().mean()
            )
            current_estr = float(rf_data["estr_pct"].dropna().iloc[-1])
            current_sofr = float(rf_data["sofr_pct"].dropna().iloc[-1])
        except Exception:
            pass

    # ── VaR / CVaR ────────────────────────────────────────────────────────────
    mu_p, sig_p = pnl.mean(), pnl.std()
    np.random.seed(42)
    sims = np.random.normal(mu_p, sig_p, 50_000)
    var_rows = []
    for alpha in [0.05, 0.10]:
        z = norm.ppf(alpha)
        pv = -(mu_p + z * sig_p) * 100
        pcv = -(mu_p - sig_p * norm.pdf(-z) / alpha) * 100
        q_ = float(np.quantile(pnl, alpha))
        hv = -q_ * 100
        tmask = pnl <= q_
        hcv = -float(pnl[tmask].mean()) * 100 if tmask.any() else np.nan
        mcv = -float(np.percentile(sims, alpha * 100)) * 100
        var_rows.append({
            "α": f"{int(alpha * 100)}%",
            "Confidence": f"{int((1 - alpha) * 100)}%",
            "Param VaR (%)": round(pv, 4),
            "Param CVaR (%)": round(pcv, 4),
            "Hist VaR (%)": round(hv, 4),
            "Hist CVaR (%)": round(hcv, 4),
            "MC VaR (%)": round(mcv, 4),
        })

    var_rows_eur = [
        {
            **vr,
            "Param VaR (EUR)": vr["Param VaR (%)"] / 100 * aum if aum else np.nan,
            "Hist VaR (EUR)": vr["Hist VaR (%)"] / 100 * aum if aum else np.nan,
            "MC VaR (EUR)": vr["MC VaR (%)"] / 100 * aum if aum else np.nan,
            "Param CVaR (EUR)": vr["Param CVaR (%)"] / 100 * aum if aum else np.nan,
        }
        for vr in var_rows
    ]

    return dict(
        cum_log=cum_log, ann_ret=ann_ret, carry=carry, rolldown=rolldown,
        ann_vol=ann_vol, max_dd=max_dd,
        sharpe=sharpe, sortino=sortino,
        sharpe_zero=sharpe_zero, sortino_zero=sortino_zero,
        calmar=calmar, mod_dur=mod_dur, dv01=dv01, dv01_eur=dv01_eur,
        aum=aum, convexity=convexity, md_by_c=md_by_c,
        ytm=ytm, yc_slope=yc_slope, krd=krd, c_vals=c_vals,
        var_rows=var_rows, var_rows_eur=var_rows_eur,
        current_estr=current_estr, current_sofr=current_sofr, avg_estr=avg_estr,
    )
```

- [ ] **Run tests — verify they pass**

```bash
pytest tests/test_risk_stats.py -v
# Expected: 7 passed
```

- [ ] **Commit**

```bash
git add src/services/risk_stats.py tests/test_risk_stats.py
git commit -m "feat(report): extract compute_risk_stats to src/services/risk_stats.py"
```

---

## Task 3 — Add three new sheets to `Report_Template.xlsx`

**Files:**
- Modify: `Report_Template.xlsx` (run a one-off script, then delete the script)

These sheets are pre-formatted placeholders. The generator will delete and recreate them with live data on each export. Creating them in the template now means the file shows the intended structure when opened in Excel.

- [ ] **Run the scaffold script** (paste into a Python REPL or save as `_add_sheets.py` and run once)

```python
"""One-off script — run once, then delete."""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

NAVY    = "0D1B2A"
NAVY_M  = "1B3A5C"
GRAY_L  = "E2E8F0"
WHITE   = "FFFFFF"
TEXT_D  = "0F172A"

def _hfill(): return PatternFill("solid", fgColor=NAVY)
def _sfill(): return PatternFill("solid", fgColor=NAVY_M)
def _cfill(): return PatternFill("solid", fgColor=GRAY_L)
def _border():
    s = Side(style="thin", color="CBD5E1")
    return Border(left=s, right=s, top=s, bottom=s)

def _banner(ws, row, text, n_cols):
    ws.cell(row=row, column=1).value = text
    ws.cell(row=row, column=1).font = Font(name="Calibri", bold=True, size=13, color=WHITE)
    ws.row_dimensions[row].height = 28
    for c in range(1, n_cols + 1):
        ws.cell(row=row, column=c).fill = _hfill()
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)

def _section(ws, row, text, n_cols):
    ws.cell(row=row, column=1).value = text
    ws.cell(row=row, column=1).font = Font(name="Calibri", bold=True, size=10, color=WHITE)
    ws.row_dimensions[row].height = 18
    for c in range(1, n_cols + 1):
        ws.cell(row=row, column=c).fill = _sfill()
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)

def _col_header(ws, row, headers):
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.fill = _cfill()
        cell.font = Font(name="Calibri", bold=True, size=9, color=TEXT_D)
        cell.border = _border()
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 30

def _placeholder_row(ws, row, label, n_val_cols, alt=False):
    fill = PatternFill("solid", fgColor="F6F8FB") if alt else PatternFill("solid", fgColor=WHITE)
    ws.cell(row=row, column=1).value = label
    ws.cell(row=row, column=1).font = Font(name="Calibri", size=9, color=TEXT_D)
    ws.cell(row=row, column=1).border = _border()
    ws.cell(row=row, column=1).fill = fill
    for c in range(2, 2 + n_val_cols):
        ws.cell(row=row, column=c).value = "—"
        ws.cell(row=row, column=c).font = Font(name="Calibri", size=9, color="94A3B8")
        ws.cell(row=row, column=c).border = _border()
        ws.cell(row=row, column=c).fill = fill
        ws.cell(row=row, column=c).alignment = Alignment(horizontal="center")

wb = openpyxl.load_workbook("Report_Template.xlsx")

# ── Risk_Summary ──────────────────────────────────────────────────────────────
if "Risk_Summary" in wb.sheetnames:
    del wb["Risk_Summary"]
ws = wb.create_sheet("Risk_Summary")
_banner(ws, 1, "EM FIXED INCOME — RISK SUMMARY", 3)
ws.column_dimensions["A"].width = 42
ws.column_dimensions["B"].width = 22
ws.column_dimensions["C"].width = 22
_col_header(ws, 2, ["METRIC", "EM Hard Currency Sustainable", "EM Local Currency Sustainable"])
_section(ws, 3, "RETURN METRICS", 3)
for i, lbl in enumerate(["Cumulative Log Return (%)", "Annualised Return (%)",
                          "Carry — Wtd Avg Yield (%)", "Roll-Down Return (est. %)"], start=4):
    _placeholder_row(ws, i, lbl, 2, alt=(i % 2 == 1))
_section(ws, 8, "RISK & RATIO METRICS", 3)
for i, lbl in enumerate(["Annualised Volatility (%)", "Maximum Drawdown (%)",
                          "Sharpe Ratio (rf = €STR)", "Sortino Ratio (rf = €STR)",
                          "Sharpe Ratio (rf = 0, ref)", "Sortino Ratio (MAR = 0, ref)",
                          "Calmar Ratio"], start=9):
    _placeholder_row(ws, i, lbl, 2, alt=(i % 2 == 1))
_section(ws, 16, "BOND ANALYTICS", 3)
for i, lbl in enumerate(["AUM (EUR)", "Modified Duration (yrs)", "DV01 (% of NAV per 1bp)",
                          "DV01 (EUR per 1bp parallel)", "Convexity (yrs²)",
                          "YTM — Wtd Avg Benchmark (%)", "Yield Curve Slope (long−short, %)"],
                         start=17):
    _placeholder_row(ws, i, lbl, 2, alt=(i % 2 == 1))

# ── VaR_Analysis ─────────────────────────────────────────────────────────────
if "VaR_Analysis" in wb.sheetnames:
    del wb["VaR_Analysis"]
ws2 = wb.create_sheet("VaR_Analysis")
_banner(ws2, 1, "EM FIXED INCOME — VAR & CVAR ANALYSIS", 7)
for col, w_val in zip("ABCDEFG", [28, 14, 20, 20, 18, 18, 18]):
    ws2.column_dimensions[col].width = w_val
_section(ws2, 2, "VAR & CVAR AS % OF NAV", 7)
_col_header(ws2, 3, ["Portfolio", "Confidence", "Param VaR (%)", "Param CVaR (%)",
                      "Hist VaR (%)", "Hist CVaR (%)", "MC VaR (%)"])
for i in range(4, 8):
    _placeholder_row(ws2, i, "—", 6, alt=(i % 2 == 1))
ws2.cell(row=9, column=1).value = None
_section(ws2, 10, "VAR & CVAR IN EUR (BASED ON AUM)", 7)
_col_header(ws2, 11, ["Portfolio", "Confidence", "Param VaR (EUR)", "Param CVaR (EUR)",
                       "Hist VaR (EUR)", "MC VaR (EUR)", ""])
for i in range(12, 16):
    _placeholder_row(ws2, i, "—", 6, alt=(i % 2 == 1))

# ── KRD ───────────────────────────────────────────────────────────────────────
if "KRD" in wb.sheetnames:
    del wb["KRD"]
ws3 = wb.create_sheet("KRD")
_banner(ws3, 1, "EM FIXED INCOME — KEY-RATE DURATION", 4)
for col, w_val in zip("ABCD", [22, 22, 26, 26]):
    ws3.column_dimensions[col].width = w_val
_section(ws3, 2, "KEY-RATE DURATION BY COUNTRY (yrs)", 4)
_col_header(ws3, 3, ["Country", "MD (par bond, yrs)",
                      "KRD — EM HC Sustainable", "KRD — EM LC Sustainable"])
for i in range(4, 11):
    _placeholder_row(ws3, i, "—", 3, alt=(i % 2 == 1))
ws3.cell(row=12, column=1).value = None
_section(ws3, 13, "LATEST BENCHMARK YIELD SNAPSHOT (%)", 4)
_col_header(ws3, 14, ["Country", "Latest Yield (%)", "As of", ""])
for i in range(15, 22):
    _placeholder_row(ws3, i, "—", 3, alt=(i % 2 == 1))

wb.save("Report_Template.xlsx")
print("Done — Risk_Summary, VaR_Analysis, KRD sheets added.")
```

- [ ] **Run the script**

```bash
cd "C:\Users\Nicolas\OneDrive - SKEMA Business School\FMI\DPAM\Project"
python _add_sheets.py
```

Expected output: `Done — Risk_Summary, VaR_Analysis, KRD sheets added.`

- [ ] **Verify in Python**

```python
import openpyxl
wb = openpyxl.load_workbook("Report_Template.xlsx")
print(wb.sheetnames)
# Expected: ['Raw_Data', 'Performance_Report', 'Alerts_Review', 'Risk_Summary', 'VaR_Analysis', 'KRD']
```

- [ ] **Delete the script**

```bash
del _add_sheets.py
```

- [ ] **Commit**

```bash
git add Report_Template.xlsx
git commit -m "feat(report): add Risk_Summary, VaR_Analysis, KRD placeholder sheets to template"
```

---

## Task 4 — Create `src/report_generator.py` — Part 1: quarters utility + raw-data builder

**Files:**
- Create: `src/report_generator.py`
- Create: `tests/test_report_generator.py`

- [ ] **Write the failing tests for `get_available_quarters`** (`tests/test_report_generator.py`)

```python
"""Tests for src/report_generator.py."""
import io
from datetime import date

import numpy as np
import pandas as pd
import openpyxl
import pytest

from src.report_generator import get_available_quarters


def _make_pnl(start: str, end: str) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start, end)
    return pd.Series(rng.normal(0.0002, 0.003, len(idx)), index=idx)


def test_returns_nonempty_list():
    pnl = _make_pnl("2024-01-02", "2026-04-30")
    quarters = get_available_quarters(pnl)
    assert isinstance(quarters, list)
    assert len(quarters) > 0


def test_newest_first():
    pnl = _make_pnl("2024-01-02", "2026-04-30")
    quarters = get_available_quarters(pnl)
    for i in range(len(quarters) - 1):
        assert quarters[i]["start"] > quarters[i + 1]["start"]


def test_all_quarters_completed():
    pnl = _make_pnl("2024-01-02", "2026-04-30")
    today = date.today()
    for q in get_available_quarters(pnl):
        assert q["end"] < today


def test_label_format():
    pnl = _make_pnl("2025-01-02", "2025-06-30")
    quarters = get_available_quarters(pnl)
    labels = [q["label"] for q in quarters]
    assert any("Q1 2025" in lbl for lbl in labels)
    assert any("Q2 2025" in lbl for lbl in labels)


def test_required_keys():
    pnl = _make_pnl("2024-01-02", "2026-04-30")
    q = get_available_quarters(pnl)[0]
    assert "label" in q
    assert "start" in q
    assert "end" in q
    assert isinstance(q["start"], date)
    assert isinstance(q["end"], date)
```

- [ ] **Run tests — verify they fail**

```bash
pytest tests/test_report_generator.py -v
# Expected: ImportError
```

- [ ] **Create `src/report_generator.py`** with `get_available_quarters` and `_populate_raw_data`

```python
"""
Quarterly Excel report generator.

Public API
----------
get_available_quarters(pnl)      -> list[dict]
generate_quarterly_report(quarter) -> bytes
"""
from __future__ import annotations

import io
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import (
    Alignment, Border, Font, PatternFill, Side,
)
from openpyxl.utils import get_column_letter

TEMPLATE_PATH = Path(__file__).parent.parent / "Report_Template.xlsx"
DATA_OUT = Path(__file__).parent.parent / "data" / "output"

# ── Colour palette (matches dashboard CSS tokens) ────────────────────────────
_NAVY   = "0D1B2A"
_NAVY_M = "1B3A5C"
_GRAY_L = "E2E8F0"
_ALT    = "F6F8FB"
_WHITE  = "FFFFFF"
_TEXT   = "0F172A"
_MUTED  = "94A3B8"

# ── Style factories ───────────────────────────────────────────────────────────

def _hfill() -> PatternFill:
    return PatternFill("solid", fgColor=_NAVY)

def _sfill() -> PatternFill:
    return PatternFill("solid", fgColor=_NAVY_M)

def _cfill() -> PatternFill:
    return PatternFill("solid", fgColor=_GRAY_L)

def _rowfill(alt: bool) -> PatternFill:
    return PatternFill("solid", fgColor=_ALT if alt else _WHITE)

def _border() -> Border:
    s = Side(style="thin", color="CBD5E1")
    return Border(left=s, right=s, top=s, bottom=s)

def _hfont(size: int = 13) -> Font:
    return Font(name="Calibri", bold=True, size=size, color=_WHITE)

def _sfont() -> Font:
    return Font(name="Calibri", bold=True, size=10, color=_WHITE)

def _cfont() -> Font:
    return Font(name="Calibri", bold=True, size=9, color=_TEXT)

def _dfont() -> Font:
    return Font(name="Calibri", size=9, color=_TEXT)

def _mfont() -> Font:
    return Font(name="Calibri", size=9, color=_MUTED)

# ── Quarter utilities ─────────────────────────────────────────────────────────

def get_available_quarters(pnl: pd.Series) -> list[dict]:
    """
    Return completed calendar quarters present in pnl, newest first.

    A quarter is included when its end date is strictly before today
    and the quarter overlaps the pnl date range.

    Returns
    -------
    list of dict with keys: label (str), start (date), end (date).
    """
    today = date.today()
    pnl_min = pnl.index.min().date()
    pnl_max = pnl.index.max().date()

    month_abbr = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    results: list[dict] = []
    for period in pd.period_range(pnl_min, today, freq="Q"):
        q_start = period.start_time.date()
        q_end   = period.end_time.date()

        if q_end >= today:          # incomplete quarter
            continue
        if q_end < pnl_min:        # entirely before data
            continue
        if q_start > pnl_max:      # entirely after data
            continue

        q_num = (q_start.month - 1) // 3 + 1
        label = (
            f"Q{q_num} {q_start.year} "
            f"({month_abbr[q_start.month]} – {month_abbr[q_end.month]} {q_start.year})"
        )
        results.append({"label": label, "start": q_start, "end": q_end})

    return list(reversed(results))
```

- [ ] **Run quarter tests — verify they pass**

```bash
pytest tests/test_report_generator.py -v
# Expected: 5 passed
```

- [ ] **Continue in `src/report_generator.py`** — add the `_populate_raw_data` helper immediately after the existing code

```python

# ── Raw-data builder ──────────────────────────────────────────────────────────

def _populate_raw_data(wb: openpyxl.Workbook, quarter: dict, p1: dict, p2: dict,
                       alert_history: dict) -> None:
    """Write the selected quarter's daily rows into the Raw_Data table."""
    # Build a full-history regime series and forward-fill
    regime_series = pd.Series({d: v["regime"] for d, v in alert_history.items()})
    regime_series.index = pd.to_datetime(regime_series.index)
    regime_series = regime_series.sort_index()
    all_bdays = pd.date_range(regime_series.index.min(), regime_series.index.max(), freq="B")
    regime_full = regime_series.reindex(all_bdays).ffill().bfill()

    q_start = pd.Timestamp(quarter["start"])
    q_end   = pd.Timestamp(quarter["end"])

    hc_pnl = p1["pnl"]
    lc_pnl = p2["pnl"]

    # Rolling 252-day 5th-percentile VaR (computed on full history for accuracy)
    def _var95(series: pd.Series) -> pd.Series:
        return series.rolling(252, min_periods=30).quantile(0.05).abs()

    hc_var = _var95(hc_pnl)
    lc_var = _var95(lc_pnl)

    # NAV indexed to 1000 at first trading day of the quarter
    def _nav(series: pd.Series, start: pd.Timestamp) -> pd.Series:
        sub = series[series.index >= start]
        return (1 + sub).cumprod() * 1000

    hc_nav = _nav(hc_pnl, q_start)
    lc_nav = _nav(lc_pnl, q_start)

    hc_sub = hc_pnl[(hc_pnl.index >= q_start) & (hc_pnl.index <= q_end)]
    lc_sub = lc_pnl[(lc_pnl.index >= q_start) & (lc_pnl.index <= q_end)]

    rows: list[dict] = []
    for dt in sorted(set(hc_sub.index) | set(lc_sub.index)):
        regime = regime_full.get(dt, "Normal")
        for ft, src, var_s, nav_s in [
            ("Hard Currency", hc_sub, hc_var, hc_nav),
            ("Local Currency", lc_sub, lc_var, lc_nav),
        ]:
            if dt not in src.index:
                continue
            pnl_val = float(round(src[dt], 6))
            nav_val = float(round(nav_s.get(dt, 1000.0), 2))
            var_raw = var_s.get(dt, np.nan)
            var_val = float(round(var_raw, 6)) if not pd.isna(var_raw) else 0.005
            rows.append({
                "Date":             dt.to_pydatetime(),
                "Fund_Type":        ft,
                "Daily_Return":     pnl_val,
                "Nav":              nav_val,
                "VaR_95_Estimate":  var_val,
                "Actual_PnL":       pnl_val,
                "Regime_State":     regime,
            })

    ws = wb["Raw_Data"]
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.value = None

    COLS = ["Date", "Fund_Type", "Daily_Return", "Nav",
            "VaR_95_Estimate", "Actual_PnL", "Regime_State"]
    for r_idx, record in enumerate(rows, start=2):
        for c_idx, col in enumerate(COLS, start=1):
            ws.cell(row=r_idx, column=c_idx).value = record[col]

    for tbl in ws.tables.values():
        if tbl.name == "tbl_RawData":
            tbl.ref = f"A1:G{len(rows) + 1}"
```

- [ ] **Commit**

```bash
git add src/report_generator.py tests/test_report_generator.py
git commit -m "feat(report): get_available_quarters + _populate_raw_data"
```

---

## Task 5 — Add sheet-writer helpers and `generate_quarterly_report` to `src/report_generator.py`

**Files:**
- Modify: `src/report_generator.py` (append helpers and main function)
- Modify: `tests/test_report_generator.py` (add integration test)

- [ ] **Add the integration test** to `tests/test_report_generator.py`

```python
# add this import at the top of the file
from src.report_generator import generate_quarterly_report   # noqa: F401 (used below)


def test_generate_quarterly_report_valid_xlsx():
    """Integration test — requires Report_Template.xlsx and data/output/ on disk."""
    q = {"label": "Q1 2026 (Jan – Mar 2026)", "start": date(2026, 1, 1), "end": date(2026, 3, 31)}
    result = generate_quarterly_report(q)
    assert isinstance(result, bytes)
    assert len(result) > 5_000

    wb = openpyxl.load_workbook(io.BytesIO(result))
    for sheet in ["Raw_Data", "Performance_Report", "Alerts_Review",
                  "Risk_Summary", "VaR_Analysis", "KRD"]:
        assert sheet in wb.sheetnames, f"Missing sheet: {sheet}"

    # Raw_Data must have data rows for Q1 2026
    ws = wb["Raw_Data"]
    data_rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    assert len(data_rows) > 0

    # Report dates should reflect the quarter
    ws_perf = wb["Performance_Report"]
    assert ws_perf["D5"].value is not None   # Report_Start_Date set
    assert ws_perf["D6"].value is not None   # Report_End_Date set
```

- [ ] **Run integration test — verify it fails**

```bash
pytest tests/test_report_generator.py::test_generate_quarterly_report_valid_xlsx -v
# Expected: ImportError (generate_quarterly_report not yet defined)
```

- [ ] **Append sheet-writers and the main generator** to `src/report_generator.py`

```python

# ── Sheet layout helpers ──────────────────────────────────────────────────────

def _banner(ws, row: int, text: str, n_cols: int) -> None:
    ws.cell(row=row, column=1).value = text
    ws.cell(row=row, column=1).font = _hfont()
    ws.cell(row=row, column=1).alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[row].height = 28
    for c in range(1, n_cols + 1):
        ws.cell(row=row, column=c).fill = _hfill()
    if n_cols > 1:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)


def _section(ws, row: int, text: str, n_cols: int) -> None:
    ws.cell(row=row, column=1).value = text
    ws.cell(row=row, column=1).font = _sfont()
    ws.row_dimensions[row].height = 18
    for c in range(1, n_cols + 1):
        ws.cell(row=row, column=c).fill = _sfill()
    if n_cols > 1:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)


def _col_headers(ws, row: int, headers: list[str]) -> None:
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.fill = _cfill()
        cell.font = _cfont()
        cell.border = _border()
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 30


def _data_cell(ws, row: int, col: int, value, alt: bool,
               fmt: str | None = None, align: str = "left") -> None:
    cell = ws.cell(row=row, column=col, value=value)
    cell.fill = _rowfill(alt)
    cell.font = _dfont()
    cell.border = _border()
    cell.alignment = Alignment(horizontal=align, vertical="center")
    if fmt:
        cell.number_format = fmt


def _fmt_val(val, fmt_str: str) -> float | str:
    """Return val as-is (let Excel format it) if numeric, else 'N/A'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return val


# ── Risk Summary sheet ────────────────────────────────────────────────────────

def _write_risk_summary(wb: openpyxl.Workbook, rs1: dict, rs2: dict,
                         pn1: str, pn2: str, quarter_label: str) -> None:
    if "Risk_Summary" in wb.sheetnames:
        del wb["Risk_Summary"]
    ws = wb.create_sheet("Risk_Summary")
    wb.move_sheet("Risk_Summary", offset=-(len(wb.sheetnames) - 4))

    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 22

    _banner(ws, 1, f"EM FIXED INCOME — RISK SUMMARY   |   {quarter_label}", 3)
    _col_headers(ws, 2, ["METRIC", pn1, pn2])

    def _row(r, label, v1, v2, fmt=None, alt=False):
        _data_cell(ws, r, 1, label, alt)
        _data_cell(ws, r, 2, _fmt_val(v1, fmt), alt, fmt, align="center")
        _data_cell(ws, r, 3, _fmt_val(v2, fmt), alt, fmt, align="center")

    _section(ws, 3, "RETURN METRICS", 3)
    _row(4, "Cumulative Log Return (%)",   rs1["cum_log"],  rs2["cum_log"],  "0.00", alt=False)
    _row(5, "Annualised Return (%)",        rs1["ann_ret"],  rs2["ann_ret"],  "0.00", alt=True)
    _row(6, "Carry — Wtd Avg Yield (%)",   rs1["carry"],    rs2["carry"],    "0.00", alt=False)
    _row(7, "Roll-Down Return (est. %)",   rs1["rolldown"], rs2["rolldown"], "0.00", alt=True)

    _section(ws, 8, "RISK & RATIO METRICS", 3)
    _row(9,  "Annualised Volatility (%)",      rs1["ann_vol"],     rs2["ann_vol"],     "0.00",  alt=False)
    _row(10, "Maximum Drawdown (%)",           rs1["max_dd"],      rs2["max_dd"],      "0.00",  alt=True)
    _row(11, "Sharpe Ratio (rf = €STR)",       rs1["sharpe"],      rs2["sharpe"],      "0.00",  alt=False)
    _row(12, "Sortino Ratio (rf = €STR)",      rs1["sortino"],     rs2["sortino"],     "0.00",  alt=True)
    _row(13, "Sharpe Ratio (rf = 0, ref)",     rs1["sharpe_zero"], rs2["sharpe_zero"], "0.00",  alt=False)
    _row(14, "Sortino Ratio (MAR = 0, ref)",   rs1["sortino_zero"],rs2["sortino_zero"],"0.00",  alt=True)
    _row(15, "Calmar Ratio",                   rs1["calmar"],      rs2["calmar"],      "0.00",  alt=False)

    _section(ws, 16, "BOND ANALYTICS", 3)
    _row(17, "AUM (EUR)",                       rs1["aum"],       rs2["aum"],       '#,##0',   alt=False)
    _row(18, "Modified Duration (yrs)",         rs1["mod_dur"],   rs2["mod_dur"],   "0.00",    alt=True)
    _row(19, "DV01 (% of NAV per 1bp)",         rs1["dv01"],      rs2["dv01"],      "0.0000",  alt=False)
    _row(20, "DV01 (EUR per 1bp parallel)",     rs1["dv01_eur"],  rs2["dv01_eur"],  '#,##0',   alt=True)
    _row(21, "Convexity (yrs²)",                rs1["convexity"], rs2["convexity"], "0.0",     alt=False)
    _row(22, "YTM — Wtd Avg Benchmark (%)",     rs1["ytm"],       rs2["ytm"],       "0.00",    alt=True)
    _row(23, "Yield Curve Slope (long−short,%)",rs1["yc_slope"],  rs2["yc_slope"],  "0.00",    alt=False)


# ── VaR Analysis sheet ────────────────────────────────────────────────────────

def _write_var_analysis(wb: openpyxl.Workbook, rs1: dict, rs2: dict,
                         pn1: str, pn2: str, quarter_label: str) -> None:
    if "VaR_Analysis" in wb.sheetnames:
        del wb["VaR_Analysis"]
    ws = wb.create_sheet("VaR_Analysis")
    wb.move_sheet("VaR_Analysis", offset=-(len(wb.sheetnames) - 5))

    for col, w in zip("ABCDEFG", [28, 14, 20, 20, 18, 18, 18]):
        ws.column_dimensions[col].width = w

    _banner(ws, 1, f"EM FIXED INCOME — VAR & CVAR ANALYSIS   |   {quarter_label}", 7)

    # % section
    _section(ws, 2, "VAR & CVAR AS % OF PORTFOLIO NAV", 7)
    _col_headers(ws, 3, ["Portfolio", "Confidence",
                          "Param VaR (%)", "Param CVaR (%)",
                          "Hist VaR (%)", "Hist CVaR (%)", "MC VaR (%)"])
    r = 4
    for pname, rs in [(pn1, rs1), (pn2, rs2)]:
        for vr in rs["var_rows"]:
            alt = (r % 2 == 0)
            _data_cell(ws, r, 1, pname,             alt, align="left")
            _data_cell(ws, r, 2, vr["Confidence"],  alt, align="center")
            _data_cell(ws, r, 3, vr["Param VaR (%)"],  alt, "0.0000", "center")
            _data_cell(ws, r, 4, vr["Param CVaR (%)"], alt, "0.0000", "center")
            _data_cell(ws, r, 5, vr["Hist VaR (%)"],   alt, "0.0000", "center")
            _data_cell(ws, r, 6, vr["Hist CVaR (%)"],  alt, "0.0000", "center")
            _data_cell(ws, r, 7, vr["MC VaR (%)"],     alt, "0.0000", "center")
            r += 1

    r += 1  # blank row

    # EUR section
    _section(ws, r, "VAR & CVAR IN EUR (BASED ON AUM)", 7)
    r += 1
    _col_headers(ws, r, ["Portfolio", "Confidence",
                          "Param VaR (EUR)", "Param CVaR (EUR)",
                          "Hist VaR (EUR)", "MC VaR (EUR)", ""])
    r += 1
    for pname, rs in [(pn1, rs1), (pn2, rs2)]:
        for vr in rs["var_rows_eur"]:
            alt = (r % 2 == 0)
            _data_cell(ws, r, 1, pname,                      alt, align="left")
            _data_cell(ws, r, 2, vr["Confidence"],           alt, align="center")
            _data_cell(ws, r, 3, vr["Param VaR (EUR)"],      alt, '#,##0', "center")
            _data_cell(ws, r, 4, vr["Param CVaR (EUR)"],     alt, '#,##0', "center")
            _data_cell(ws, r, 5, vr["Hist VaR (EUR)"],       alt, '#,##0', "center")
            _data_cell(ws, r, 6, vr["MC VaR (EUR)"],         alt, '#,##0', "center")
            ws.cell(row=r, column=7).fill = _rowfill(alt)
            r += 1


# ── KRD sheet ─────────────────────────────────────────────────────────────────

def _write_krd(wb: openpyxl.Workbook, rs1: dict, rs2: dict,
               pn1: str, pn2: str, quarter_label: str) -> None:
    if "KRD" in wb.sheetnames:
        del wb["KRD"]
    ws = wb.create_sheet("KRD")
    wb.move_sheet("KRD", offset=-(len(wb.sheetnames) - 6))

    for col, w in zip("ABCD", [22, 22, 26, 26]):
        ws.column_dimensions[col].width = w

    _banner(ws, 1, f"EM FIXED INCOME — KEY-RATE DURATION   |   {quarter_label}", 4)

    _section(ws, 2, "KEY-RATE DURATION BY COUNTRY (yrs)", 4)
    _col_headers(ws, 3, ["Country", "MD (par bond, yrs)",
                          f"KRD — {pn1} (yrs)", f"KRD — {pn2} (yrs)"])

    FLAG = {"Brazil": "🇧🇷", "Mexico": "🇲🇽", "South Africa": "🇿🇦", "Poland": "🇵🇱",
            "Colombia": "🇨🇴", "Hungary": "🇭🇺", "Romania": "🇷🇴"}

    all_countries = sorted(set(rs1["krd"]) | set(rs2["krd"]))
    for i, c in enumerate(all_countries):
        r = 4 + i
        alt = (i % 2 == 1)
        label = f"{FLAG.get(c, '')} {c}"
        md = rs1["md_by_c"].get(c, 0)
        _data_cell(ws, r, 1, label,                 alt)
        _data_cell(ws, r, 2, md,                    alt, "0.000", "center")
        _data_cell(ws, r, 3, rs1["krd"].get(c, 0), alt, "0.0000", "center")
        _data_cell(ws, r, 4, rs2["krd"].get(c, 0), alt, "0.0000", "center")

    snap_start = 4 + len(all_countries) + 2
    _section(ws, snap_start, "LATEST BENCHMARK YIELD SNAPSHOT (%)", 4)
    _col_headers(ws, snap_start + 1, ["Country", "Latest Yield (%)", "As of", ""])
    for i, c in enumerate(all_countries):
        r = snap_start + 2 + i
        alt = (i % 2 == 1)
        label = f"{FLAG.get(c, '')} {c}"
        y_val = rs1["c_vals"].get(c) or rs2["c_vals"].get(c)
        _data_cell(ws, r, 1, label,                           alt)
        _data_cell(ws, r, 2, y_val if y_val is not None else "N/A", alt, "0.00", "center")
        _data_cell(ws, r, 3, "Latest available",              alt, align="center")
        ws.cell(row=r, column=4).fill = _rowfill(alt)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_quarterly_report(quarter: dict) -> bytes:
    """
    Build a quarterly Excel report for the given quarter dict.

    Parameters
    ----------
    quarter : dict
        One entry from get_available_quarters() —
        must have keys: label (str), start (date), end (date).

    Returns
    -------
    bytes — the xlsx file contents, ready for st.download_button.
    """
    from src.data import load_briefings as _unused  # noqa: ensure src.data importable
    from src.data.var_artifacts import (
        load_alert_history as _load_alerts,
    )
    from src.data_loader import (
        load_config,
        load_all_countries_combined,
        load_country_yields,
    )
    from src.risk_free import load_risk_free_rates
    from src.services.portfolios import build_portfolio_views
    from src.services.risk_stats import compute_risk_stats

    warnings.filterwarnings("ignore")

    cfg = load_config()
    portfolio_results = build_portfolio_views()
    p1, p2 = portfolio_results[0], portfolio_results[1]

    alert_history = _load_alerts(DATA_OUT) or {}

    # Yield levels for risk-stat computation
    all_countries = (
        cfg["countries"]["local_currency"] + cfg["countries"]["hard_currency"]
    )
    excluded = cfg.get("excluded_series", {})
    yield_levels: dict = {}
    for country in all_countries:
        try:
            df = load_country_yields(country, "data/raw")
            excl = excluded.get(country, [])
            df = df.drop(columns=[c for c in excl if c in df.columns])
            yield_levels[country] = df
        except Exception:
            pass

    rf_data = None
    try:
        fred_cfg = cfg.get("fred", {})
        key = open(fred_cfg.get("key_path", "private/fred_key.txt")).read().strip()
        rf_data = load_risk_free_rates(
            fred_cfg.get("output_path", "data/output/risk_free_rates.csv"),
            fred_api_key=key,
        )
    except Exception:
        pass

    rs1 = compute_risk_stats(p1["def"], p1["pnl"], yield_levels, rf_data)
    rs2 = compute_risk_stats(p2["def"], p2["pnl"], yield_levels, rf_data)
    pn1, pn2 = p1["def"]["name"], p2["def"]["name"]

    # Load and populate workbook
    wb = openpyxl.load_workbook(TEMPLATE_PATH)

    _populate_raw_data(wb, quarter, p1, p2, alert_history)

    ws_perf = wb["Performance_Report"]
    ws_perf["D5"] = datetime.combine(quarter["start"], datetime.min.time())
    ws_perf["D6"] = datetime.combine(quarter["end"],   datetime.min.time())

    lbl = quarter["label"]
    _write_risk_summary(wb, rs1, rs2, pn1, pn2, lbl)
    _write_var_analysis(wb, rs1, rs2, pn1, pn2, lbl)
    _write_krd(wb, rs1, rs2, pn1, pn2, lbl)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
```

- [ ] **Run the integration test**

```bash
pytest tests/test_report_generator.py -v
# Expected: all tests pass (including integration test)
```

- [ ] **Commit**

```bash
git add src/report_generator.py tests/test_report_generator.py
git commit -m "feat(report): sheet writers + generate_quarterly_report"
```

---

## Task 6 — Refactor `app.py`: shared imports + pre-compute `rs1`/`rs2` before tabs

**Files:**
- Modify: `app.py`

Two `@st.cache_data` functions (`_load_yield_levels`, `_load_rf_data`) are currently defined **inside** `with tab_risk:`, which re-registers them on every rerun. Move them to module scope alongside the other loaders. Remove the inline `_risk_stats` closure and use the shared `compute_risk_stats` instead. Pre-compute `rs1`/`rs2` before the tabs so they're available for the export button.

- [ ] **Add imports at the top of `app.py`** (after the existing `from src.services.portfolios import ...` line, ~line 33)

```python
from src.services.risk_stats import compute_risk_stats
from src.report_generator import get_available_quarters, generate_quarterly_report
```

- [ ] **Move `_load_yield_levels` to module scope** — find it at ~line 1554 (inside `with tab_risk:`), cut the whole decorated function, and paste it at module scope near the other `@st.cache_data` loaders (~line 647). The function body is unchanged:

```python
@st.cache_data(show_spinner="Loading yield levels…", persist="disk")
def _load_yield_levels(version):
    cfg = load_config()
    all_c = cfg["countries"]["local_currency"] + cfg["countries"]["hard_currency"]
    excluded = cfg.get("excluded_series", {})
    out = {}
    for country in all_c:
        try:
            df = load_country_yields(country, "data/raw")
            excl = excluded.get(country, [])
            df = df.drop(columns=[c for c in excl if c in df.columns])
            out[country] = df
        except Exception:
            pass
    return out
```

- [ ] **Move `_load_rf_data` to module scope** — find it at ~line 1769, cut and paste beside `_load_yield_levels`:

```python
@st.cache_data(show_spinner="Loading risk-free rates…", persist="disk")
def _load_rf_data(version):
    cfg = load_config()
    key_path = cfg.get("fred", {}).get("key_path", "private/fred_key.txt")
    out_path  = cfg.get("fred", {}).get("output_path", "data/output/risk_free_rates.csv")
    try:
        key = open(key_path).read().strip()
        return load_risk_free_rates(out_path, fred_api_key=key)
    except Exception:
        return None
```

- [ ] **Add `_cached_report` at module scope** (right after `_load_rf_data`):

```python
@st.cache_data(show_spinner="Preparing quarterly report…")
def _cached_report(q_start_iso: str, q_end_iso: str, _ver: str) -> bytes:
    from datetime import date as _date
    return generate_quarterly_report({
        "label": f"{q_start_iso} – {q_end_iso}",
        "start": _date.fromisoformat(q_start_iso),
        "end":   _date.fromisoformat(q_end_iso),
    })
```

- [ ] **In the Portfolios section (`elif page == "Portfolios":`), after loading `p1`/`p2` (~line 1309) and before `tab_weights, tab_perf ...`, add the pre-computation block**:

```python
    # Pre-compute risk stats and yield levels once (used by both tab_risk and export button).
    yield_levels = _load_yield_levels(_RAW_VER)
    rf_data      = _load_rf_data(_OUT_VER)
    rs1 = compute_risk_stats(p1["def"], p1["pnl"], yield_levels, rf_data)
    rs2 = compute_risk_stats(p2["def"], p2["pnl"], yield_levels, rf_data)
```

- [ ] **Inside `with tab_risk:`, replace the calls that previously loaded yield_levels, rf_data, and ran `_risk_stats`** — those three lines (`yield_levels = ...`, `rf_data = _load_rf_data(...)`, `rs1 = _risk_stats(...)`, `rs2 = _risk_stats(...)`) should be deleted since the values are now pre-computed above. The rest of `tab_risk` (which just reads from `rs1`/`rs2`) is unchanged.

Also **delete the inline `_risk_stats` function definition** (it ran from ~line 1581 to ~line 1762).

- [ ] **Run the full test suite to verify nothing broke**

```bash
pytest tests/ -v --ignore=tests/test_characterization.py -q
# Expected: all existing tests pass
```

- [ ] **Commit**

```bash
git add app.py
git commit -m "refactor(app): move loaders to module scope, use shared compute_risk_stats"
```

---

## Task 7 — Add the export UI to the Portfolios page

**Files:**
- Modify: `app.py`

- [ ] **Add the export control row** immediately after the pre-computation block from Task 6 and before the `tab_weights, tab_perf, ... = st.tabs(...)` line:

```python
    # ── Quarterly export control bar ─────────────────────────────────────────
    quarters = get_available_quarters(p1["pnl"])
    if quarters:
        st.markdown("<div class='section-card' style='padding:16px 24px; margin-bottom:16px;'><h3 style='margin-bottom:12px;'>Quarterly Report Export</h3>", unsafe_allow_html=True)
        _exp_col1, _exp_col2 = st.columns([3, 1])
        with _exp_col1:
            _q_idx = st.selectbox(
                "Select quarter",
                range(len(quarters)),
                format_func=lambda i: quarters[i]["label"],
                label_visibility="visible",
            )
        with _exp_col2:
            _q = quarters[_q_idx]
            _q_num = (_q["start"].month - 1) // 3 + 1
            _fname = f"EM_FI_Q{_q_num}_{_q['start'].year}_Report.xlsx"
            _report_bytes = _cached_report(
                _q["start"].isoformat(),
                _q["end"].isoformat(),
                _RAW_VER,
            )
            st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Export Quarterly Report",
                data=_report_bytes,
                file_name=_fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help=f"Download {_q['label']} as a pre-filled Excel report.",
            )
        st.markdown("</div>", unsafe_allow_html=True)
```

- [ ] **Start the dashboard and verify the export bar renders**

```bash
streamlit run app.py
```

Navigate to the **Portfolios** page. Verify:
1. A "Quarterly Report Export" card appears above the Weights / Cumulative Performance / … tabs.
2. The dropdown lists completed quarters (e.g. Q1 2026, Q4 2025, …) newest-first.
3. Clicking **Export Quarterly Report** triggers a download.

- [ ] **Open the downloaded file in Excel.** Verify:
   - All 6 sheets are present (Raw_Data, Performance_Report, Alerts_Review, Risk_Summary, VaR_Analysis, KRD).
   - Raw_Data rows match the selected quarter's date range.
   - Performance_Report and Alerts_Review formulas compute (no `#REF!` or `#VALUE!` errors).
   - Risk_Summary, VaR_Analysis, KRD contain formatted numeric data with navy headers.

- [ ] **Commit**

```bash
git add app.py
git commit -m "feat(portfolios): quarterly report export button with quarter selector"
```

---

## Task 8 — Final cleanup and branch-ready commit

**Files:** all modified files

- [ ] **Run the full test suite one last time**

```bash
pytest tests/ -v --ignore=tests/test_characterization.py -q
# Expected: all tests pass
```

- [ ] **Check git status — nothing untracked**

```bash
git status
# Expected: nothing to commit, working tree clean
```

- [ ] **Push the branch**

```bash
git push -u origin feature/quarterly-report-export
```
