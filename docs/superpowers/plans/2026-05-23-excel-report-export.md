# Excel Report Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the EM Fixed Income analytics for both portfolios into a polished `.xlsx` workbook driven by a reusable, hand-editable Excel template.

**Architecture:** A new orchestration entry point `build_report_state()` computes a `ReportState` (shared PCA/regime/alerts + a per-portfolio risk bundle). A new `src/reporting/excel_report.py` renders that state into a workbook by loading `templates/report_template.xlsx`, cloning a blueprint sheet per portfolio, filling tokens/tables at sentinel marker cells, and adding native Excel charts. A generator script emits the template so it can be regenerated or hand-edited.

**Tech Stack:** Python, pandas, numpy, scikit-learn, statsmodels, scipy (existing); `openpyxl` (new) for reading/writing the template and native charts; pytest for tests.

**Spec:** `docs/superpowers/specs/2026-05-23-excel-report-export-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` (modify) | Add `openpyxl`. |
| `config/funds.yaml` (modify) | Add `short` + `label` to each portfolio for sheet naming. |
| `src/pca_regime.py` (modify) | Add `build_panel_factor_scores()` (panel PCA → global factor scores + explained variance). |
| `src/orchestration/report_state.py` (create) | `PortfolioRisk` + `ReportState` dataclasses; `build_portfolio_risk()` (pure) and `build_report_state()` (I/O wrapper). |
| `scripts/build_report_template.py` (create) | `build_template(path)` — emits `templates/report_template.xlsx`. |
| `templates/report_template.xlsx` (create, generated) | The editable template (Cover, `_tmpl_Portfolio`, PCA & Regime, Alerts). |
| `src/reporting/__init__.py` (create) | Package marker. |
| `src/reporting/excel_report.py` (create) | Rendering helpers + `write_report()` + `generate_report()`. |
| `export_excel.py` (create) | Repo-root CLI mirroring `export_pdf.py`. |
| `tests/test_report_state.py` (create) | Tests for panel PCA, `build_portfolio_risk`, `build_report_state`. |
| `tests/test_excel_report.py` (create) | Tests for template generator + rendering helpers + `write_report`. |

**Key conventions discovered (must match):**
- `build_portfolio_pnl_from_def(change_dfs, pdef)` returns `(pnl: Series, proxy_dy: DataFrame)`; weights are normalised over **all** raw weights, `proxy_dy` keeps only countries present with the benchmark maturity.
- VaR helpers in `src/quant/var_engine.py` return **positive fractions** (e.g. `0.005` = 0.5% loss).
- `compute_parametric_var(pnl, confidence) -> {"VaR","CVaR","mu","sigma"}`.
- `compute_historical_var(pnl, window, confidence) -> {"VaR","CVaR","n_obs"}`.
- `compute_stressed_var(pnl, start, end, confidence) -> {"VaR","CVaR","n_obs","start","end"}`.
- `compute_multi_nu_var_table(pnl) -> DataFrame` indexed by nu (`4,5,8,20,'inf'`), cols `["VaR 95%","VaR 99%","CVaR 95%","CVaR 99%"]`.
- `compute_mc_t_copula_var(proxy_dy, weights, duration, n_sims, copula_dof, seed) -> {"VaR_95","VaR_99","CVaR_95","CVaR_99"}`.
- `compute_factor_idio_decomposition(yield_changes, factor_scores, weights) -> {"B","D","Sigma_F","var_systematic","var_idiosyncratic","var_total","pct_systematic","pct_idiosyncratic"}`; `weights` aligned to `yield_changes.columns`.
- `kupiec_pof(returns, VaR, p) -> {...,"violation_rate","p_value","reject_H0"}`; `christoffersen_test(returns, VaR) -> {"LR_ind","p_value","reject_independence"}`.
- `build_health_check(pipeline_log, portfolio_pnl, regime_features, var_95, now=None) -> list[{"check","status","detail"}]` (5 cards).
- Panel PCA (notebook cell 24): concat each country's `dy` with columns renamed `f"{country}_{mat}"`, `dropna()`, `StandardScaler`, `PCA(5)`, take first 3 scores named `"PC1 (global level)"`, `"PC2 (global slope)"`, `"PC3 (global curvature)"`.
- Decomposition (notebook cell 47): regress each country's benchmark-maturity change in `proxy_dy` on the panel scores aligned to `proxy_dy.index`.

---

## Task 1: Add openpyxl dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Add `openpyxl` on its own line at the end of `requirements.txt` (after `statsmodels`).

- [ ] **Step 2: Install it**

Run: `pip install openpyxl`
Expected: `Successfully installed openpyxl-...`

- [ ] **Step 3: Verify import**

Run: `python -c "import openpyxl; print(openpyxl.__version__)"`
Expected: a version number prints, no error.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add openpyxl for Excel report export"
```

---

## Task 2: Add portfolio short/label to config

**Files:**
- Modify: `config/funds.yaml` (the two entries under `portfolios:`)

- [ ] **Step 1: Add fields to each portfolio**

In `config/funds.yaml`, add `short` and `label` keys to each portfolio entry (directly under `id`):

For `portfolio_1` (EM Hard Currency Sustainable):
```yaml
    id: "portfolio_1"
    short: "HC"
    label: "Hard Currency"
```

For `portfolio_2` (EM Local Currency Sustainable):
```yaml
    id: "portfolio_2"
    short: "LC"
    label: "Local Currency"
```

- [ ] **Step 2: Verify config still loads and fields are present**

Run:
```bash
python -c "from src.data_loader import load_config; p=load_config()['portfolios']; print([(x['short'],x['label']) for x in p])"
```
Expected: `[('HC', 'Hard Currency'), ('LC', 'Local Currency')]`

- [ ] **Step 3: Commit**

```bash
git add config/funds.yaml
git commit -m "config: add short/label to portfolios for report sheet naming"
```

---

## Task 3: Panel factor scores helper in pca_regime

**Files:**
- Modify: `src/pca_regime.py` (add a function after `run_pca_all_countries`, near line 72)
- Test: `tests/test_report_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_state.py`:

```python
"""Tests for the report-state orchestration layer (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from src.pca_regime import build_panel_factor_scores


def test_build_panel_factor_scores_shape(synthetic_yield_changes):
    scores, evr = build_panel_factor_scores(synthetic_yield_changes, n_components=3)
    assert list(scores.columns) == [
        "PC1 (global level)", "PC2 (global slope)", "PC3 (global curvature)"
    ]
    assert len(scores) > 0
    assert np.isfinite(scores.values).all()
    # Explained variance ratios are a descending-ish vector that sums to <= 1
    assert evr[0] >= evr[1]
    assert 0 < evr[:3].sum() <= 1.0 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_state.py::test_build_panel_factor_scores_shape -v`
Expected: FAIL with `ImportError: cannot import name 'build_panel_factor_scores'`.

- [ ] **Step 3: Implement the function**

In `src/pca_regime.py`, add after `run_pca_all_countries` (after line 72):

```python
def build_panel_factor_scores(
    change_dfs: dict[str, pd.DataFrame],
    n_components: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Global ('panel') PCA across every country-maturity series.

    Mirrors main.ipynb cell 24: stack all per-country yield-change frames
    side by side (columns renamed ``f"{country}_{mat}"``), drop rows with
    any NaN, standardise, run a 5-component PCA, and return the first
    ``n_components`` scores as global factors.

    Returns
    -------
    scores : DataFrame
        Panel factor scores, columns
        ``["PC1 (global level)", "PC2 (global slope)", "PC3 (global curvature)"]``
        truncated to ``n_components``.
    explained_var : ndarray
        Explained-variance ratios of the fitted PCA (length up to 5).
    """
    parts = []
    for country, dy in sorted(change_dfs.items()):
        renamed = dy.rename(columns={c: f"{country}_{c}" for c in dy.columns})
        parts.append(renamed)
    panel = pd.concat(parts, axis=1).dropna()

    scaler = StandardScaler()
    panel_std = scaler.fit_transform(panel)
    k = min(5, panel.shape[1])
    pca = PCA(n_components=k)
    scores = pca.fit_transform(panel_std)

    labels = [
        "PC1 (global level)", "PC2 (global slope)", "PC3 (global curvature)",
    ][:n_components]
    scores_df = pd.DataFrame(
        scores[:, :n_components], index=panel.index, columns=labels,
    )
    return scores_df, pca.explained_variance_ratio_
```

(`StandardScaler`, `PCA`, `np`, `pd` are already imported at the top of `pca_regime.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report_state.py::test_build_panel_factor_scores_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pca_regime.py tests/test_report_state.py
git commit -m "feat: add panel factor scores helper (global PCA)"
```

---

## Task 4: ReportState dataclasses + build_portfolio_risk (pure)

**Files:**
- Modify: `tests/conftest.py` (add a shared `report_test_config` fixture)
- Create: `src/orchestration/report_state.py`
- Test: `tests/test_report_state.py` (add a fixture + test)

> **Why conftest:** `tests/` has no `__init__.py` and there is no pytest
> config, so cross-module test imports (`from tests.test_report_state import …`)
> are unreliable. The repo's convention is shared fixtures in `conftest.py`;
> both new test modules consume `report_test_config` from there.

- [ ] **Step 1a: Add the shared config fixture to conftest.py**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def report_test_config():
    """Config adapted to the 4 synthetic countries (Brazil/Mexico/SA/Poland).

    Shared by tests/test_report_state.py and tests/test_excel_report.py.
    """
    return {
        "pca": {"n_components": 3},
        "regime": {
            "gmm": {"max_components": 4, "covariance_type": "full",
                    "n_init": 3, "random_state": 42},
            "labels": {0: "Normal", 1: "Stress", 2: "Risk-On", 3: "Other"},
            "colors": {},
        },
        "alerts": {
            "pc_zscore_threshold": 3.0, "pc_zscore_high_threshold": 4.0,
            "vol_percentile": 0.90, "vol_trailing_window": 252,
            "curve_zscore_threshold": 3.0, "curve_zscore_high_threshold": 4.0,
            "curve_rolling_window": 60,
        },
        "var": {
            "confidence_levels": [0.95, 0.99],
            "historical_windows": {"1Y": 252, "2Y": 504},
            "monte_carlo": {"n_simulations": 2000, "random_seed": 42,
                            "copula_dof": 5},
            "stress_windows": {"COVID": ["2022-10-01", "2022-12-15"]},
            "primary_stress_window": "COVID",
        },
        "portfolios": [
            {"name": "Test HC", "id": "p1", "short": "HC",
             "label": "Hard Currency",
             "aum_eur": 1.0e7,
             "weights": {"Brazil": 4, "Mexico": 3,
                         "South Africa": 4, "Poland": 5},
             "effective_duration": 6.0, "benchmark_maturity": "5Y"},
            {"name": "Test LC", "id": "p2", "short": "LC",
             "label": "Local Currency",
             "aum_eur": 5.0e9,
             "weights": {"Brazil": 10, "Mexico": 8,
                         "South Africa": 7, "Poland": 6},
             "effective_duration": 5.2, "benchmark_maturity": "5Y"},
        ],
        "macro_events": {},
    }
```

- [ ] **Step 1b: Add the failing test**

Append to `tests/test_report_state.py`:

```python
from src.pca_regime import (
    build_regime_features, fit_gmm, run_alert_scan, run_pca_all_countries,
)
from src.orchestration.report_state import (
    PortfolioRisk, build_portfolio_risk,
)


@pytest.fixture
def shared_state(synthetic_yield_changes, report_test_config):
    cfg = report_test_config
    pca_results = run_pca_all_countries(
        synthetic_yield_changes, n_components=cfg["pca"]["n_components"])
    regime_features = fit_gmm(build_regime_features(pca_results), cfg)
    panel_scores, _evr = build_panel_factor_scores(synthetic_yield_changes, 3)
    return cfg, regime_features, panel_scores


def test_build_portfolio_risk_fields(synthetic_yield_changes, shared_state):
    cfg, regime_features, panel_scores = shared_state
    pdef = cfg["portfolios"][0]
    risk = build_portfolio_risk(
        pdef, synthetic_yield_changes, panel_scores, regime_features, cfg)

    assert isinstance(risk, PortfolioRisk)
    assert risk.short == "HC" and risk.label == "Hard Currency"
    # Headline VaR positive and ordered (99% >= 95%)
    assert risk.var_95 > 0 and risk.var_99 >= risk.var_95
    # Parametric / historical / stressed populated
    assert set(risk.parametric.keys()) == {0.95, 0.99}
    assert "1Y" in risk.historical and "2Y" in risk.historical
    assert "COVID" in risk.stressed
    assert risk.stressed["COVID"]["n_obs"] > 0
    # multi-nu table shape
    assert list(risk.multi_nu.columns) == [
        "VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%"]
    # decomposition shares sum to 100
    d = risk.decomposition
    assert abs(d["pct_systematic"] + d["pct_idiosyncratic"] - 100.0) < 1e-6
    assert list(d["betas"].index) == list(risk.weights.keys())
    # backtests + health
    assert "violation_rate" in risk.backtests["kupiec_95"]
    assert len(risk.health) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_state.py::test_build_portfolio_risk_fields -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.orchestration.report_state'`.

- [ ] **Step 3: Implement dataclasses + build_portfolio_risk**

Create `src/orchestration/report_state.py`:

```python
"""Report-state orchestration: assemble everything the Excel report renders.

Splits cleanly from rendering. ``build_portfolio_risk`` is pure (operates on
in-memory data, no I/O) so it is unit-testable with synthetic fixtures.
``build_report_state`` is the thin I/O wrapper that loads raw data once,
computes the shared PCA/regime/alerts, then assembles one
:class:`PortfolioRisk` per configured portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import (
    build_portfolio_pnl_from_def,
    load_all_countries_combined,
    load_config,
)
from src.orchestration.health import build_health_check
from src.pca_regime import (
    build_panel_factor_scores,
    build_regime_features,
    fit_gmm,
    run_alert_scan,
    run_pca_all_countries,
)
from src.quant.var_engine import (
    christoffersen_test,
    compute_factor_idio_decomposition,
    compute_historical_var,
    compute_mc_t_copula_var,
    compute_multi_nu_var_table,
    compute_parametric_var,
    compute_stressed_var,
    kupiec_pof,
)


@dataclass
class PortfolioRisk:
    """Per-portfolio risk bundle rendered onto one cloned report sheet."""

    name: str
    short: str
    label: str
    aum_eur: float
    duration: float
    weights: dict[str, float]

    # Headline (MC t-copula)
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    parametric: dict          # {0.95: {"VaR","CVaR"}, 0.99: {...}}
    historical: dict          # {"1Y": {0.95: {...}, 0.99: {...}}, ...}
    stressed: dict            # {window: {"VaR_95","VaR_99","CVaR_95","CVaR_99","n_obs"}}
    multi_nu: pd.DataFrame
    decomposition: dict       # pct_*, var_*, "betas" (DataFrame)
    backtests: dict           # {"kupiec_95": {...}, "christoffersen": {...}}
    health: list[dict]


@dataclass
class ReportState:
    """Everything the Excel report needs, computed once."""

    run_date: pd.Timestamp
    data_end: pd.Timestamp
    config: dict
    panel_explained_var: np.ndarray
    regime_features: pd.DataFrame
    alerts: dict
    portfolios: list[PortfolioRisk] = field(default_factory=list)


def _nominal_pipeline_log() -> list[dict]:
    """Minimal log so build_health_check's step/runtime cards render GREEN.

    The report is not a full pipeline run; freshness / breach-rate /
    regime-stability are the meaningful cards.
    """
    return [{"step": "report", "status": "success", "runtime_seconds": 0.0}]


def build_portfolio_risk(
    pdef: dict,
    change_dfs: dict[str, pd.DataFrame],
    panel_scores: pd.DataFrame,
    regime_features: pd.DataFrame,
    config: dict,
    now: Optional[pd.Timestamp] = None,
) -> PortfolioRisk:
    """Assemble the full VaR suite + health for one portfolio. Pure."""
    pnl, proxy_dy = build_portfolio_pnl_from_def(change_dfs, pdef)

    raw_w = pdef["weights"]
    total = sum(raw_w.values())
    weights = {c: raw_w[c] / total for c in proxy_dy.columns}
    w_vec = np.array([weights[c] for c in proxy_dy.columns])
    duration = pdef["effective_duration"]

    # Parametric
    parametric = {
        cl: {k: compute_parametric_var(pnl, cl)[k] for k in ("VaR", "CVaR")}
        for cl in config["var"]["confidence_levels"]
    }

    # Historical at each configured window
    historical = {}
    for label, window in config["var"]["historical_windows"].items():
        historical[label] = {
            cl: {k: compute_historical_var(pnl, window, cl)[k]
                 for k in ("VaR", "CVaR")}
            for cl in config["var"]["confidence_levels"]
        }

    # Stressed at each configured window
    stressed = {}
    for wname, (start, end) in config["var"]["stress_windows"].items():
        s95 = compute_stressed_var(pnl, start, end, 0.95)
        s99 = compute_stressed_var(pnl, start, end, 0.99)
        stressed[wname] = {
            "VaR_95": s95["VaR"], "VaR_99": s99["VaR"],
            "CVaR_95": s95["CVaR"], "CVaR_99": s99["CVaR"],
            "n_obs": s95["n_obs"],
        }

    multi_nu = compute_multi_nu_var_table(pnl)

    # Headline MC t-copula
    mc_cfg = config["var"]["monte_carlo"]
    mc = compute_mc_t_copula_var(
        proxy_dy=proxy_dy, weights=w_vec, duration=duration,
        n_sims=mc_cfg["n_simulations"], copula_dof=mc_cfg["copula_dof"],
        seed=mc_cfg["random_seed"],
    )

    # Factor / idiosyncratic decomposition vs panel factors
    common = panel_scores.index.intersection(proxy_dy.index)
    decomp = compute_factor_idio_decomposition(
        proxy_dy.loc[common], panel_scores.loc[common], w_vec)
    var_empirical = float(np.var(proxy_dy.loc[common].values @ w_vec, ddof=1))
    betas = pd.DataFrame(
        decomp["B"], index=list(weights.keys()),
        columns=list(panel_scores.columns))
    decomposition = {
        "pct_systematic": decomp["pct_systematic"],
        "pct_idiosyncratic": decomp["pct_idiosyncratic"],
        "var_systematic": decomp["var_systematic"],
        "var_idiosyncratic": decomp["var_idiosyncratic"],
        "var_total": decomp["var_total"],
        "var_empirical": var_empirical,
        "betas": betas,
    }

    # Backtests vs the headline 95% VaR
    backtests = {
        "kupiec_95": kupiec_pof(pnl, mc["VaR_95"], 0.05),
        "christoffersen": christoffersen_test(pnl, mc["VaR_95"]),
    }

    health = build_health_check(
        _nominal_pipeline_log(), pnl, regime_features, mc["VaR_95"], now=now)

    return PortfolioRisk(
        name=pdef["name"], short=pdef["short"], label=pdef["label"],
        aum_eur=float(pdef["aum_eur"]), duration=float(duration),
        weights=weights,
        var_95=mc["VaR_95"], var_99=mc["VaR_99"],
        cvar_95=mc["CVaR_95"], cvar_99=mc["CVaR_99"],
        parametric=parametric, historical=historical, stressed=stressed,
        multi_nu=multi_nu, decomposition=decomposition,
        backtests=backtests, health=health,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report_state.py::test_build_portfolio_risk_fields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/report_state.py tests/test_report_state.py
git commit -m "feat: PortfolioRisk dataclass and build_portfolio_risk assembler"
```

---

## Task 5: build_report_state (I/O wrapper)

**Files:**
- Modify: `src/orchestration/report_state.py` (add `build_report_state`)
- Test: `tests/test_report_state.py` (add a test using `change_dfs` injection)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report_state.py`:

```python
from src.orchestration.report_state import ReportState, build_report_state


def test_build_report_state_with_injected_data(
        synthetic_yield_changes, report_test_config):
    state = build_report_state(
        config=report_test_config, change_dfs=synthetic_yield_changes)

    assert isinstance(state, ReportState)
    assert len(state.portfolios) == 2
    assert {p.short for p in state.portfolios} == {"HC", "LC"}
    assert all(np.isfinite(p.var_95) for p in state.portfolios)
    assert len(state.panel_explained_var) >= 3
    assert "regime" in state.regime_features.columns
    assert isinstance(state.alerts, dict)
    assert state.data_end == max(
        df.index.max() for df in synthetic_yield_changes.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_state.py::test_build_report_state_with_injected_data -v`
Expected: FAIL with `ImportError: cannot import name 'build_report_state'`.

- [ ] **Step 3: Implement build_report_state**

Append to `src/orchestration/report_state.py`:

```python
def build_report_state(
    config: Optional[dict] = None,
    data_dir: str = "data/raw",
    slice_end: Optional[pd.Timestamp] = None,
    change_dfs: Optional[dict[str, pd.DataFrame]] = None,
) -> ReportState:
    """Run the full per-portfolio analytics and assemble a ReportState.

    Parameters
    ----------
    config
        Output of ``load_config``; loaded from default if omitted.
    data_dir
        Root of the raw-data tree (ignored if ``change_dfs`` is given).
    slice_end
        Optional upper bound; all per-country frames are sliced to
        ``<= slice_end`` for deterministic snapshots.
    change_dfs
        Pre-loaded yield-change frames. When provided, raw-data loading is
        skipped (used by tests with synthetic data).
    """
    if config is None:
        config = load_config()

    if change_dfs is None:
        change_dfs = load_all_countries_combined(config, data_dir=data_dir)
    if slice_end is not None:
        change_dfs = {c: df.loc[:slice_end] for c, df in change_dfs.items()}

    # Shared, portfolio-independent inference
    pca_results = run_pca_all_countries(
        change_dfs, n_components=config["pca"]["n_components"])
    regime_features = fit_gmm(build_regime_features(pca_results), config)
    alerts = run_alert_scan(
        regime_features, pca_results, change_dfs, config,
        config.get("macro_events"))
    panel_scores, panel_evr = build_panel_factor_scores(
        change_dfs, n_components=config["pca"]["n_components"])

    portfolios = [
        build_portfolio_risk(
            pdef, change_dfs, panel_scores, regime_features, config)
        for pdef in config["portfolios"]
    ]

    data_end = max(df.index.max() for df in change_dfs.values())
    return ReportState(
        run_date=pd.Timestamp.now().normalize(),
        data_end=data_end,
        config=config,
        panel_explained_var=panel_evr,
        regime_features=regime_features,
        alerts=alerts,
        portfolios=portfolios,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report_state.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/report_state.py tests/test_report_state.py
git commit -m "feat: build_report_state orchestration wrapper"
```

---

## Task 6: Template generator

**Files:**
- Create: `scripts/build_report_template.py`
- Create (generated): `templates/report_template.xlsx`
- Test: `tests/test_excel_report.py`

The blueprint sheet uses fixed, generously-spaced anchor rows (each data
table is small and bounded: health ≤ 6 rows, VaR ≤ 7, stressed ≤ 4,
multi-ν ≤ 7, decomposition summary ≤ 6, betas ≤ 8). Section titles sit one
row above each sentinel marker; the renderer overwrites the marker cell with
the table's top-left.

- [ ] **Step 1: Write the failing test**

Create `tests/test_excel_report.py`:

```python
"""Tests for the Excel template generator and report renderer."""
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from scripts.build_report_template import (
    BLUEPRINT_SHEET, MARKERS, build_template,
)


def test_template_generator_structure(tmp_path):
    out = tmp_path / "tmpl.xlsx"
    build_template(out)
    assert out.exists()

    wb = openpyxl.load_workbook(out)
    assert "Cover" in wb.sheetnames
    assert BLUEPRINT_SHEET in wb.sheetnames
    assert "PCA & Regime" in wb.sheetnames
    assert "Alerts" in wb.sheetnames

    # Every declared marker token exists exactly where expected
    found = set()
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.strip() in MARKERS:
                    found.add(cell.value.strip())
    assert MARKERS <= found
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_excel_report.py::test_template_generator_structure -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_report_template'`.

(If `scripts/` lacks `__init__.py`, the import still works because pytest adds the rootdir to `sys.path`; no package marker needed for `scripts`.)

- [ ] **Step 3: Implement the generator**

Create `scripts/build_report_template.py`:

```python
"""Generate the reusable Excel report template.

Run directly to (re)create ``templates/report_template.xlsx``. Open the
result in Excel to tweak branding/styling; the renderer
(``src.reporting.excel_report``) fills tokens and tables at the sentinel
marker cells defined here.
"""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

TEMPLATE_PATH = Path("templates/report_template.xlsx")
BLUEPRINT_SHEET = "_tmpl_Portfolio"

# Brand palette
BRAND = "1F3864"        # deep navy
BRAND_LIGHT = "D9E1F2"  # light blue
WHITE = "FFFFFF"

# Sentinel markers the renderer locates and writes at.
MARKERS = {
    "{{HEALTH_TABLE}}", "{{VAR_TABLE}}", "{{STRESSED_TABLE}}",
    "{{MULTINU_TABLE}}", "{{DECOMP_TABLE}}", "{{BETAS_TABLE}}",
    "{{EXPVAR_TABLE}}", "{{REGIME_TABLE}}", "{{ALERTS_TABLE}}",
}

_TITLE_FONT = Font(name="Calibri", size=18, bold=True, color=WHITE)
_SECTION_FONT = Font(name="Calibri", size=12, bold=True, color=BRAND)
_LABEL_FONT = Font(name="Calibri", size=10, italic=True, color="595959")
_BRAND_FILL = PatternFill("solid", fgColor=BRAND)


def _title_bar(ws, text, row=1, span=7):
    """Full-width brand title bar at ``row``."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    cell = ws.cell(row=row, column=1, value=text)
    cell.font = _TITLE_FONT
    cell.fill = _BRAND_FILL
    cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[row].height = 28


def _section(ws, text, row):
    cell = ws.cell(row=row, column=1, value=text)
    cell.font = _SECTION_FONT


def _marker(ws, token, row, col=1):
    ws.cell(row=row, column=col, value=token)


def _set_widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _build_cover(ws):
    _title_bar(ws, "EM Fixed Income — Risk Report", row=1)
    ws.cell(row=3, column=1, value="Run date:").font = _LABEL_FONT
    ws.cell(row=3, column=2, value="{{run_date}}")
    ws.cell(row=4, column=1, value="Data through:").font = _LABEL_FONT
    ws.cell(row=4, column=2, value="{{data_end}}")
    ws.cell(row=5, column=1, value="Portfolios:").font = _LABEL_FONT
    ws.cell(row=5, column=2, value="{{n_portfolios}}")
    ws.cell(row=7, column=1,
            value=("Local & hard-currency EM sovereign bond risk analytics. "
                   "VaR/CVaR, stressed scenarios, factor decomposition, "
                   "PCA-based regime detection and alerts.")).font = _LABEL_FONT
    ws.cell(row=20, column=1,
            value=("Generated automatically from data/raw. Duration-based "
                   "P&L proxy; figures are estimates for internal use only."
                   )).font = _LABEL_FONT
    _set_widths(ws, [16, 60])


def _build_blueprint(ws):
    _title_bar(ws, "{{portfolio_name}}", row=1)
    ws.cell(row=2, column=1,
            value="AUM (EUR): {{aum}}   |   Effective duration: {{duration}}"
            ).font = _LABEL_FONT

    _section(ws, "Pipeline Health", row=4)
    _marker(ws, "{{HEALTH_TABLE}}", row=5)

    _section(ws, "Value-at-Risk by Method (1-day, fraction of NAV)", row=12)
    _marker(ws, "{{VAR_TABLE}}", row=13)

    _section(ws, "Stressed Scenarios", row=21)
    _marker(ws, "{{STRESSED_TABLE}}", row=22)

    _section(ws, "Student-t Sensitivity (multi-ν)", row=27)
    _marker(ws, "{{MULTINU_TABLE}}", row=28)

    _section(ws, "Factor / Idiosyncratic Decomposition", row=36)
    _marker(ws, "{{DECOMP_TABLE}}", row=37)
    _section(ws, "Factor Betas", row=45)
    _marker(ws, "{{BETAS_TABLE}}", row=46)

    ws.freeze_panes = "A3"
    _set_widths(ws, [30, 16, 16, 16, 16, 16, 16])


def _build_pca_regime(ws):
    _title_bar(ws, "PCA & Regime (shared)", row=1)
    _section(ws, "Panel PCA — Explained Variance", row=3)
    _marker(ws, "{{EXPVAR_TABLE}}", row=4)
    _section(ws, "Regime — Recent History", row=14)
    _marker(ws, "{{REGIME_TABLE}}", row=15)
    ws.freeze_panes = "A3"
    _set_widths(ws, [24, 18, 18, 18])


def _build_alerts(ws):
    _title_bar(ws, "Alerts (shared)", row=1)
    _section(ws, "Recent Alert Days", row=3)
    _marker(ws, "{{ALERTS_TABLE}}", row=4)
    ws.freeze_panes = "A3"
    _set_widths(ws, [14, 16, 14, 10, 40])


def build_template(path: Path | str = TEMPLATE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    _build_cover(wb.active)
    wb.active.title = "Cover"
    _build_blueprint(wb.create_sheet(BLUEPRINT_SHEET))
    _build_pca_regime(wb.create_sheet("PCA & Regime"))
    _build_alerts(wb.create_sheet("Alerts"))

    wb.save(path)
    return path


if __name__ == "__main__":
    out = build_template()
    print(f"Template written to {out.resolve()}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_excel_report.py::test_template_generator_structure -v`
Expected: PASS.

- [ ] **Step 5: Generate the committed template**

Run: `python scripts/build_report_template.py`
Expected: `Template written to ...templates/report_template.xlsx`.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_report_template.py templates/report_template.xlsx tests/test_excel_report.py
git commit -m "feat: Excel report template generator and template"
```

---

## Task 7: Rendering helpers

**Files:**
- Create: `src/reporting/__init__.py`
- Create: `src/reporting/excel_report.py` (helpers only in this task)
- Test: `tests/test_excel_report.py` (add helper tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_excel_report.py`:

```python
from openpyxl import Workbook

from src.reporting.excel_report import (
    find_marker, replace_tokens, write_dataframe,
)


def test_find_marker_and_replace_tokens():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Hello {{name}}"
    ws["B3"] = "{{TABLE}}"
    assert find_marker(ws, "{{TABLE}}") == (3, 2)
    replace_tokens(ws, {"{{name}}": "World"})
    assert ws["A1"].value == "Hello World"
    with pytest.raises(KeyError):
        find_marker(ws, "{{MISSING}}")


def test_write_dataframe_returns_bbox():
    wb = Workbook()
    ws = wb.active
    df = pd.DataFrame(
        {"VaR 95%": [0.005, 0.006], "VaR 99%": [0.009, 0.010]},
        index=["Parametric", "Historical"])
    top, left, bottom, right = write_dataframe(
        ws, (2, 1), df, index_header="Method", number_format="0.000%")
    # header row + 2 data rows
    assert (top, left) == (2, 1)
    assert bottom == 4 and right == 3
    assert ws.cell(row=2, column=1).value == "Method"
    assert ws.cell(row=3, column=1).value == "Parametric"
    assert ws.cell(row=3, column=2).value == pytest.approx(0.005)
    assert ws.cell(row=3, column=2).number_format == "0.000%"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_excel_report.py::test_find_marker_and_replace_tokens -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.reporting'`.

- [ ] **Step 3: Implement the package + helpers**

Create `src/reporting/__init__.py`:

```python
"""Report rendering (Excel, and room for other formats later)."""
```

Create `src/reporting/excel_report.py` (helpers section):

```python
"""Render a ReportState into the Excel template.

Loads templates/report_template.xlsx, fills cover tokens, clones the
blueprint sheet once per portfolio, writes tables at sentinel markers,
adds native Excel charts, and saves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from src.orchestration.report_state import (
    PortfolioRisk, ReportState, build_report_state,
)

DEFAULT_TEMPLATE = Path("templates/report_template.xlsx")
BLUEPRINT_SHEET = "_tmpl_Portfolio"

_HEADER_FILL = PatternFill("solid", fgColor="1F3864")
_HEADER_FONT = Font(bold=True, color="FFFFFF")


def find_marker(ws, token: str) -> tuple[int, int]:
    """Return (row, col) of the cell whose stripped value equals ``token``."""
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, str) and cell.value.strip() == token:
                return cell.row, cell.column
    raise KeyError(f"Marker {token!r} not found in sheet {ws.title!r}")


def replace_tokens(ws, mapping: dict[str, str]) -> None:
    """In-place replace any ``{{token}}`` substrings in string cells."""
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, str) and "{{" in cell.value:
                new = cell.value
                for token, value in mapping.items():
                    if token in new:
                        new = new.replace(token, str(value))
                if new != cell.value:
                    cell.value = new


def write_dataframe(
    ws, anchor: tuple[int, int], df: pd.DataFrame, *,
    index_header: Optional[str] = None,
    number_format: Optional[str] = None,
) -> tuple[int, int, int, int]:
    """Write ``df`` (with its index as the first column) starting at ``anchor``.

    Returns the bounding box ``(top_row, left_col, bottom_row, right_col)``.
    Header row is styled with the brand fill.
    """
    r0, c0 = anchor
    # Header row
    ws.cell(row=r0, column=c0, value=index_header or df.index.name or "")
    for j, col in enumerate(df.columns, start=1):
        ws.cell(row=r0, column=c0 + j, value=str(col))
    for cc in range(c0, c0 + len(df.columns) + 1):
        hc = ws.cell(row=r0, column=cc)
        hc.fill = _HEADER_FILL
        hc.font = _HEADER_FONT
    # Body
    for i, (idx, srow) in enumerate(df.iterrows(), start=1):
        ws.cell(row=r0 + i, column=c0, value=str(idx))
        for j, col in enumerate(df.columns, start=1):
            val = srow[col]
            cell = ws.cell(row=r0 + i, column=c0 + j,
                           value=float(val) if pd.notna(val)
                           and not isinstance(val, str) else val)
            if number_format and pd.notna(val) and not isinstance(val, str):
                cell.number_format = number_format
    bottom = r0 + len(df)
    right = c0 + len(df.columns)
    return r0, c0, bottom, right
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_excel_report.py::test_find_marker_and_replace_tokens tests/test_excel_report.py::test_write_dataframe_returns_bbox -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reporting/__init__.py src/reporting/excel_report.py tests/test_excel_report.py
git commit -m "feat: Excel rendering helpers (markers, tokens, dataframe writer)"
```

---

## Task 8: write_report end-to-end

**Files:**
- Modify: `src/reporting/excel_report.py` (add table builders + `write_report` + `generate_report`)
- Test: `tests/test_excel_report.py` (add an end-to-end test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_excel_report.py`. It reuses the report-state fixtures by building a state with injected synthetic data:

```python
from scripts.build_report_template import build_template as _build_template
from src.orchestration.report_state import build_report_state
from src.reporting.excel_report import write_report


def _no_tokens_remain(wb) -> bool:
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and "{{" in cell.value:
                    return False
    return True


def test_write_report_end_to_end(
        tmp_path, synthetic_yield_changes, report_test_config):
    template = tmp_path / "tmpl.xlsx"
    _build_template(template)
    state = build_report_state(
        config=report_test_config, change_dfs=synthetic_yield_changes)

    out = tmp_path / "report.xlsx"
    write_report(state, template_path=template, output_path=out)
    assert out.exists()

    wb = openpyxl.load_workbook(out)
    # Blueprint removed; one risk sheet per portfolio
    assert "_tmpl_Portfolio" not in wb.sheetnames
    assert "HC — Hard Currency" in wb.sheetnames
    assert "LC — Local Currency" in wb.sheetnames
    assert "Cover" in wb.sheetnames
    # No unfilled tokens anywhere
    assert _no_tokens_remain(wb)
    # At least one native chart on a portfolio sheet and on PCA & Regime
    assert len(wb["HC — Hard Currency"]._charts) >= 1
    assert len(wb["PCA & Regime"]._charts) >= 1
    # Cover shows portfolio count
    cover_vals = [c.value for row in wb["Cover"].iter_rows() for c in row]
    assert 2 in cover_vals or "2" in cover_vals
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_excel_report.py::test_write_report_end_to_end -v`
Expected: FAIL with `ImportError: cannot import name 'write_report'`.

- [ ] **Step 3: Implement table builders, write_report, generate_report**

Append to `src/reporting/excel_report.py`:

```python
def _var_method_table(p: PortfolioRisk) -> pd.DataFrame:
    """Rows = methods, cols = VaR/CVaR at 95/99 (fractions)."""
    rows = {}
    rows["Parametric"] = [
        p.parametric[0.95]["VaR"], p.parametric[0.99]["VaR"],
        p.parametric[0.95]["CVaR"], p.parametric[0.99]["CVaR"]]
    for label, h in p.historical.items():
        rows[f"Historical {label}"] = [
            h[0.95]["VaR"], h[0.99]["VaR"], h[0.95]["CVaR"], h[0.99]["CVaR"]]
    rows["Monte Carlo (t-copula)"] = [
        p.var_95, p.var_99, p.cvar_95, p.cvar_99]
    return pd.DataFrame.from_dict(
        rows, orient="index",
        columns=["VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%"])


def _stressed_table(p: PortfolioRisk) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {w: [d["VaR_95"], d["VaR_99"], d["CVaR_95"], d["CVaR_99"], d["n_obs"]]
         for w, d in p.stressed.items()},
        orient="index",
        columns=["VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%", "Obs"])


def _health_table(p: PortfolioRisk) -> pd.DataFrame:
    return pd.DataFrame(
        [[h["status"], h["detail"]] for h in p.health],
        index=[h["check"] for h in p.health],
        columns=["Status", "Detail"])


def _decomp_summary_table(p: PortfolioRisk) -> pd.DataFrame:
    d = p.decomposition
    return pd.DataFrame(
        {"Value": [d["pct_systematic"], d["pct_idiosyncratic"],
                   d["var_systematic"], d["var_idiosyncratic"],
                   d["var_total"], d["var_empirical"]]},
        index=["Systematic %", "Idiosyncratic %", "Var systematic",
               "Var idiosyncratic", "Var total", "Var empirical"])


def _regime_recent_table(state: ReportState, n: int = 20) -> pd.DataFrame:
    rf = state.regime_features.tail(n)
    return pd.DataFrame(
        {"Regime": rf["regime_label"].values,
         "Confidence": rf["regime_proba"].values},
        index=[d.date().isoformat() for d in rf.index])


def _expvar_table(state: ReportState) -> pd.DataFrame:
    evr = state.panel_explained_var
    return pd.DataFrame(
        {"Explained Variance": [float(x) for x in evr]},
        index=[f"PC{i + 1}" for i in range(len(evr))])


def _alerts_table(state: ReportState, n: int = 50) -> pd.DataFrame:
    days = sorted(state.alerts.keys(), reverse=True)[:n]
    rows = []
    for d in days:
        a = state.alerts[d]
        top = a["alerts"][0]["detail"] if a["alerts"] else ""
        rows.append([a["regime"], a["max_severity"], a["n_alerts"], top])
    return pd.DataFrame(
        rows, index=days,
        columns=["Regime", "Max severity", "# Alerts", "Top alert"])


def _add_bar_chart(ws, bbox, value_col_offset, title, anchor_cell):
    """Bar chart of one value column vs the row labels in ``bbox``."""
    top, left, bottom, right = bbox
    data = Reference(ws, min_col=left + value_col_offset, min_row=top,
                     max_row=bottom)
    cats = Reference(ws, min_col=left, min_row=top + 1, max_row=bottom)
    chart = BarChart()
    chart.type = "col"
    chart.title = title
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 7
    chart.width = 12
    ws.add_chart(chart, anchor_cell)


def _fill_portfolio_sheet(ws, p: PortfolioRisk) -> None:
    replace_tokens(ws, {
        "{{portfolio_name}}": p.name,
        "{{aum}}": f"{p.aum_eur:,.0f}",
        "{{duration}}": f"{p.duration:.2f}",
    })
    write_dataframe(ws, find_marker(ws, "{{HEALTH_TABLE}}"),
                    _health_table(p), index_header="Check")
    var_bbox = write_dataframe(ws, find_marker(ws, "{{VAR_TABLE}}"),
                               _var_method_table(p), index_header="Method",
                               number_format="0.000%")
    write_dataframe(ws, find_marker(ws, "{{STRESSED_TABLE}}"),
                    _stressed_table(p), index_header="Window",
                    number_format="0.000%")
    write_dataframe(ws, find_marker(ws, "{{MULTINU_TABLE}}"),
                    p.multi_nu, index_header="nu", number_format="0.000%")
    write_dataframe(ws, find_marker(ws, "{{DECOMP_TABLE}}"),
                    _decomp_summary_table(p), index_header="Metric")
    write_dataframe(ws, find_marker(ws, "{{BETAS_TABLE}}"),
                    p.decomposition["betas"], index_header="Country",
                    number_format="0.0000")
    # Native chart: VaR 95% by method (value_col_offset=1 -> first data col)
    _add_bar_chart(ws, var_bbox, value_col_offset=1,
                   title="VaR 95% by method", anchor_cell="I4")


def _fill_pca_regime_sheet(ws, state: ReportState) -> None:
    ev_bbox = write_dataframe(ws, find_marker(ws, "{{EXPVAR_TABLE}}"),
                              _expvar_table(state), index_header="Component",
                              number_format="0.0%")
    write_dataframe(ws, find_marker(ws, "{{REGIME_TABLE}}"),
                    _regime_recent_table(state), index_header="Date",
                    number_format="0.0%")
    _add_bar_chart(ws, ev_bbox, value_col_offset=1,
                   title="Panel PCA explained variance", anchor_cell="F4")


def write_report(
    state: ReportState,
    template_path: Path | str = DEFAULT_TEMPLATE,
    output_path: Path | str = None,
) -> Path:
    """Render ``state`` into the template and save the workbook."""
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    if output_path is None:
        output_path = Path(
            f"data/output/em_fi_report_{state.run_date.date()}.xlsx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = load_workbook(template_path)
    if BLUEPRINT_SHEET not in wb.sheetnames:
        raise KeyError(
            f"Template missing blueprint sheet {BLUEPRINT_SHEET!r}: "
            f"{template_path}")

    # Cover
    replace_tokens(wb["Cover"], {
        "{{run_date}}": state.run_date.date().isoformat(),
        "{{data_end}}": state.data_end.date().isoformat(),
        "{{n_portfolios}}": len(state.portfolios),
    })

    blueprint = wb[BLUEPRINT_SHEET]
    for p in state.portfolios:
        ws = wb.copy_worksheet(blueprint)
        ws.title = f"{p.short} — {p.label}"[:31]
        _fill_portfolio_sheet(ws, p)

    _fill_pca_regime_sheet(wb["PCA & Regime"], state)
    write_dataframe(wb["Alerts"], find_marker(wb["Alerts"], "{{ALERTS_TABLE}}"),
                    _alerts_table(state), index_header="Date")

    del wb[BLUEPRINT_SHEET]
    # Order: Cover, portfolio sheets, PCA & Regime, Alerts
    order = (["Cover"]
             + [f"{p.short} — {p.label}"[:31] for p in state.portfolios]
             + ["PCA & Regime", "Alerts"])
    wb._sheets.sort(key=lambda s: order.index(s.title)
                    if s.title in order else len(order))

    wb.save(output_path)
    return output_path


def generate_report(
    template_path: Path | str = DEFAULT_TEMPLATE,
    output_path: Path | str = None,
    data_dir: str = "data/raw",
) -> Path:
    """Build the report state from live data and render it. CLI entry point."""
    state = build_report_state(data_dir=data_dir)
    return write_report(state, template_path=template_path,
                        output_path=output_path)
```

- [ ] **Step 4: Run the full Excel test module**

Run: `pytest tests/test_excel_report.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reporting/excel_report.py tests/test_excel_report.py
git commit -m "feat: write_report renders ReportState into Excel with native charts"
```

---

## Task 9: CLI entry point

**Files:**
- Create: `export_excel.py`
- Test: `tests/test_excel_report.py` (add a CLI smoke test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_excel_report.py`:

```python
def test_export_excel_cli_callable(
        tmp_path, synthetic_yield_changes, report_test_config, monkeypatch):
    import export_excel
    from src.orchestration.report_state import build_report_state

    # Force the CLI's state build to use synthetic data
    monkeypatch.setattr(
        export_excel, "build_report_state",
        lambda **kw: build_report_state(
            config=report_test_config, change_dfs=synthetic_yield_changes))

    template = tmp_path / "tmpl.xlsx"
    _build_template(template)
    out = tmp_path / "cli_report.xlsx"
    rc = export_excel.main(["--template", str(template), "--output", str(out)])
    assert rc == 0
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_excel_report.py::test_export_excel_cli_callable -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_excel'`.

- [ ] **Step 3: Implement the CLI**

Create `export_excel.py`:

```python
"""Export the EM Fixed Income risk report to Excel.

Usage:
    python export_excel.py [--template PATH] [--output PATH] [--data-dir DIR]

Builds the analytics for every portfolio in config/funds.yaml and renders
them into a workbook using templates/report_template.xlsx.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.orchestration.report_state import build_report_state
from src.reporting.excel_report import DEFAULT_TEMPLATE, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the risk report to Excel.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                        help="Path to the .xlsx template.")
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/output/em_fi_report_<date>.xlsx).")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Root of the raw-data tree.")
    args = parser.parse_args(argv)

    template = Path(args.template)
    if not template.exists():
        print(f"ERROR: template not found: {template}", file=sys.stderr)
        print("Run: python scripts/build_report_template.py", file=sys.stderr)
        return 1

    print("Building report state from", args.data_dir, "…")
    state = build_report_state(data_dir=args.data_dir)
    out = write_report(state, template_path=template, output_path=args.output)
    print(f"Report written to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_excel_report.py::test_export_excel_cli_callable -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: all tests PASS (no regressions in existing VaR/PCA/alert tests).

- [ ] **Step 6: Commit**

```bash
git add export_excel.py tests/test_excel_report.py
git commit -m "feat: export_excel CLI entry point"
```

---

## Task 10: Generate a real report + document

**Files:**
- Generated (not committed unless desired): `data/output/em_fi_report_<date>.xlsx`
- Modify: `CLAUDE.md` (Commands section)
- Modify: `README.md` (usage note)

- [ ] **Step 1: Generate against live data**

Run: `python export_excel.py`
Expected: `Report written to: ...data/output/em_fi_report_<today>.xlsx`. If raw
data is missing, the data loader raises a clear error — that is expected on a
machine without `data/raw`.

- [ ] **Step 2: Manually open the file**

Open the generated `.xlsx` in Excel and confirm: Cover tokens filled; one
`HC — …` and one `LC — …` sheet; VaR/stressed/multi-ν/decomposition tables
populated; charts render; PCA & Regime and Alerts sheets populated; no
`{{…}}` text anywhere.

- [ ] **Step 3: Document the command**

In `CLAUDE.md`, under the ```powershell Commands block, add:

````markdown
# (Re)generate the Excel report template
python scripts/build_report_template.py

# Export the risk report to Excel (both portfolios)
python export_excel.py
````

In `README.md`, add a short bullet under usage noting that
`python export_excel.py` produces `data/output/em_fi_report_<date>.xlsx`
from `templates/report_template.xlsx`, and that the template is editable in
Excel or regenerable via `scripts/build_report_template.py`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document Excel report export commands"
```

(Decide separately whether to commit a sample generated workbook under
`data/output/`; by default it is left untracked like the other generated
artifacts.)

---

## Notes for the implementer

- **`copy_worksheet` fidelity:** It copies cell values, styles, column
  widths, merged cells, and freeze panes — but *not* charts or images. The
  blueprint is intentionally chart-free; charts are added by code after
  cloning. Do not add native charts to the blueprint in the template.
- **VaR values are fractions.** Cells use number formats like `0.000%` so
  `0.005` displays as `0.500%`. Do not pre-multiply by 100.
- **Sheet-name limit:** Excel caps sheet names at 31 chars; the renderer
  slices `f"{short} — {label}"` to 31. Keep `short`/`label` short in config.
- **Determinism in tests:** `build_health_check` defaults `now` to
  `pd.Timestamp.now()`, so the freshness card's GREEN/YELLOW status depends
  on the synthetic data's last date relative to today. Tests assert
  structure (5 cards) and field presence, not a specific freshness colour.
- **Bounded table rows:** The blueprint's fixed anchor spacing assumes the
  documented max table sizes. If a future config adds many stress windows or
  historical windows, widen the spacing in `_build_blueprint`.
```
