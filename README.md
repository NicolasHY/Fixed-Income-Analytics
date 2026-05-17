# Company EM Fixed Income Intelligence Platform

Interview demonstration project for the **VIE Data Scientist** role at Company Fixed Income (Brussels).

Target funds: **Company L Bonds EM Sustainable** (PLBEMSA LX) · **Company L Bonds EM Hard Currency Sustainable** (DPBEMHF LX)

---

## JD-to-Module Mapping

| JD Deliverable | Module | Status |
|---|---|---|
| AI: automate interpretation of FI performance/risk — daily PM insights | Module 4: LLM Daily Briefing Engine | ✅ |
| AI: intelligent monitoring agents for data platform reliability | Module 5: Pipeline Health Monitor | ✅ |
| AI: prototype AI financial tools (research assistants, valuation) | Module 6: EM Bond Research Assistant | 🔧 stretch |
| Standardize/centralize VaR reporting frameworks | Module 2: Multi-Method VaR Engine | ✅ |
| Comparative analysis across Fixed Income portfolios | Module 3: HC vs. LC Dashboard | ⏭️ skipped* |
| Industrialize data pipelines, containerize services | Module 7: Architecture & Deployment | ✅ |

*HC fund position-level Bloomberg data unavailable; the framework extends naturally once data is accessible.

---

## Modules

### Module 1 — Yield Curve PCA & Regime Detection
Daily yield changes for 7 EM countries (Brazil, Mexico, South Africa, Poland, Colombia, Hungary, Romania) decomposed into **level / slope / curvature** (PC1–PC3 capture >90% of variance per country). A Gaussian Mixture Model with BIC-selected k detects market regimes and an automated alert engine fires on regime shifts, PC z-score breaches (>3σ), vol spikes (>90th percentile trailing 1Y), and country curve outliers (>3σ vs. 60-day rolling mean).

### Module 2 — Multi-Method VaR Engine
Portfolio P&L proxy via duration approximation: `ΔP/P ≈ −D_eff × Δy`. Three methods implemented:
- **Parametric** — Normal and Student-t, rolling 1Y covariance
- **Monte Carlo** — Multivariate Normal + Student-t copula (10,000 simulations)
- **Historical Simulation** — 1Y / 2Y / 3Y lookback windows

Backtesting: **Kupiec POF** (proportion of failures) and **Christoffersen independence** tests.

### Module 4 — LLM Daily Briefing Engine
Structured JSON payload (regime, PCA scores, VaR figures, alerts, curve moves) sent to **Google Gemini 2.0 Flash** with a fixed-income analyst system prompt. Results cached to `data/output/sample_briefings.json` by date. Five historical showcase dates pre-generated.

### Module 5 — Pipeline Health Monitor
Five-step pipeline wrapped in structured logging (step name, timestamp, runtime, output shape, error). Health checks: all steps passed · data freshness · runtime within bounds · VaR breach rate (3–8% tolerance) · regime stability. Output: `GREEN / YELLOW / RED` per check, written to `data/output/health_check.json`.

### Module 7 — Architecture & Deployment
Dockerized, tested, and config-driven. See below.

---

## Quick Start

```bash
# Local
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Main analysis
jupyter notebook main.ipynb

# Test suite (no real market data required)
pytest tests/ -v

# Streamlit dashboard (reads pre-generated outputs)
streamlit run app.py

# Docker
docker build -t company-fi .
docker run -p 8501:8501 company-fi
# → open http://localhost:8501
```

---

## Repository Structure

```
company-em-fi-intelligence/
├── config/
│   └── funds.yaml          ← single source of truth: countries, weights, thresholds
├── data/
│   ├── raw/                ← Investing.com CSVs, one folder per country
│   └── output/             ← generated artefacts (charts, briefings, logs)
├── src/
│   ├── data_loader.py      ← CSV ingestion, yield changes, portfolio P&L proxy
│   └── pca_regime.py       ← PCA, GMM, alert engine (reusable modules)
├── tests/
│   ├── conftest.py         ← synthetic data fixtures
│   ├── test_var.py         ← parametric / MC / historical VaR + backtests
│   ├── test_pca.py         ← PCA decomposition properties
│   └── test_alerts.py      ← alert engine structure and severity logic
├── app.py                  ← Streamlit dashboard
├── main.ipynb              ← end-to-end analysis (all modules)
├── Dockerfile
└── requirements.txt
```

---

## Data

Yield curves downloaded manually from [Investing.com](https://www.investing.com/rates-bonds/) for 7 EM countries (2015–present). Path convention: `data/raw/<Country>/<Country> <N>-Year Bond Yield Historical Data.csv`. Excluded series (poor coverage): 2Y Colombia, 20Y Hungary.

All parameters — country universe, fund weights, VaR confidence levels, GMM hyperparameters, alert thresholds — are centralized in `config/funds.yaml`.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | pandas, numpy |
| PCA / GMM | scikit-learn |
| Regime (HMM) | hmmlearn |
| VaR / Statistics | scipy |
| LLM briefing | google-genai (Gemini 2.0 Flash) |
| Visualization | matplotlib |
| Dashboard | Streamlit |
| Containerization | Docker |
| Testing | pytest |

---

*Nicolas Henry — SKEMA Business School, FMI programme*
