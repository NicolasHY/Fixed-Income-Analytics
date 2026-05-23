# EM Fixed Income Intelligence Platform

> **Anonymized portfolio project.** A production-style analytics suite built around a real Emerging Markets fixed-income asset manager's sovereign-bond funds. The client's identity and any non-public details have been removed; all market data shown is from public sources (Investing.com, Bloomberg).

An end-to-end EM sovereign-bond analytics stack:

- **Yield-curve PCA & regime detection** — daily yield changes decomposed into level/slope/curvature; Gaussian Mixture Model (BIC-selected k) with an automated alert engine.
- **Multi-method VaR engine** — parametric (Normal & Student-t), Monte Carlo, and historical simulation, with Kupiec POF and Christoffersen independence backtests.
- **LLM daily-briefing engine** — structured risk payload summarized by an LLM into a PM-ready briefing, cached by date.
- **Pipeline health monitor** — structured logging and GREEN/YELLOW/RED checks on freshness, runtime, and VaR breach rates.
- **Local research assistant** — offline chatbot backed by a local Ollama runtime (no API key, no data leaves the machine).

Config-driven (`config/funds.yaml` is the single source of truth), tested, and containerized.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

jupyter notebook main.ipynb     # end-to-end analysis
pytest tests/ -v                # test suite (synthetic data, no market data needed)
streamlit run app.py            # dashboard (reads pre-generated outputs)
```

## Tech stack

pandas · numpy · scikit-learn · hmmlearn · scipy · statsmodels · Plotly · Streamlit · FRED API · Google Gemini · Ollama · Docker · pytest

---

*Nicolas Henry — SKEMA Business School, FMI programme*
