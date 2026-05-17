# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EM Fixed Income Analytics Suite for the Company. Analyzes Emerging Market sovereign bond yields across local-currency (Brazil, Mexico, South Africa, Poland) and hard-currency (Colombia, Hungary, Romania) universes. The primary deliverable is `main.ipynb`.

## Commands

```powershell
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_var.py

# Run a single test class or function
pytest tests/test_var.py::TestParametricVaR
pytest tests/test_var.py::TestParametricVaR::test_var_is_positive

# Run the notebook (execute all cells)
jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```

## Architecture

### Data Flow

```
data/raw/<Country>/<Country> <N>-Year Bond Yield Historical Data.csv
         ↓  (src/data_loader.py)
    yield levels → daily changes (first-diff) → change_dfs dict
         ↓
    portfolio P&L proxy  (duration approx: ΔP/P ≈ −D_eff × weighted_avg_Δy/100)
         ↓
    Module 1: PCA + GMM → regime_features, pca_results
    Module 2: VaR Engine (parametric / historical / Monte Carlo) + backtesting
    Module 3: LLM briefing via Google Gemini
         ↓
    data/output/  (CSVs, PNGs, JSONs)
```

### Module Map

| Module | Location | Purpose |
|--------|----------|---------|
| Data ingestion | `src/data_loader.py` | Load raw CSVs, compute yield changes, build portfolio P&L proxy |
| PCA + Regime | `src/pca_regime.py` | PCA on daily yield changes, GMM regime detection (BIC-selected k), alert engine |
| VaR Engine | `main.ipynb` (Module 2) | Parametric, historical, Monte Carlo VaR/CVaR + Kupiec/Christoffersen backtests |
| LLM Briefing | `main.ipynb` (Module 3/4) | Daily PM briefing via Gemini with disk-based caching |

### Configuration (`config/funds.yaml`)

Single source of truth for all parameters: country universe, fund weights, VaR confidence levels, PCA component count, GMM hyperparameters, alert thresholds, Gemini model config, and macro event annotations. All analytical code reads from this config — never hardcode thresholds or weights in code.

### Key Design Decisions

- **Yield changes, not levels**: PCA is run on daily first-differences (bps changes) per country, not yield levels.
- **Config-driven alerts**: All z-score thresholds (`pc_zscore_threshold: 3.0`, `curve_zscore_threshold: 3.0`) and window lengths (`vol_trailing_window: 252`, `curve_rolling_window: 60`) live in `config/funds.yaml`.
- **BIC for GMM component selection**: `_select_n_components_bic()` in `pca_regime.py` selects the number of regime states automatically unless overridden.
- **Duration approximation for P&L**: `build_portfolio_pnl()` uses `ΔP/P ≈ −D_eff × (weighted_avg_Δy / 100)` with fund weights normalized to sum to 1.
- **LLM caching**: Briefings are cached to `data/output/sample_briefings.json` keyed by date to avoid redundant API calls.
- **Gemini API key**: Loaded from `private/gemini_key.txt` (gitignored). The `private/` directory is not tracked.

### Tests

Tests in `tests/` use synthetic data only (no real market data required). `conftest.py` provides shared fixtures (`synthetic_yield_changes`, `portfolio_pnl`, `pca_results`, `regime_features`). The VaR helpers in `test_var.py` are self-contained copies of notebook logic — if notebook VaR logic changes, update the test helpers too.

### Raw Data Format

CSVs downloaded manually from Investing.com. Expected path pattern:
```
data/raw/<Country>/<Country> <N>-Year Bond Yield Historical Data.csv
```
Columns used: `Date` (MM/DD/YYYY), `Price` (yield level in %). Some countries have maturities excluded due to coverage gaps — see `excluded_series` in `config/funds.yaml`.
