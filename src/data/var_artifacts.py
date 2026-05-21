"""
Readers for the on-disk artifacts produced by the analytical pipeline
(``main.ipynb`` cells 49 / 57 / 58 / 34) and consumed by the Streamlit
dashboard.

Before this module the dashboard hard-coded every artifact path and
schema inline in ``app.py``. Lifting these into the Data layer means
the Application layer no longer knows the file layout — if a sidecar
file's name or columns change, only this module updates.

Every loader returns ``None`` when its required files are missing, so
callers can fall back to a "run the notebook first" warning without
needing exception handling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_OUTPUT_DIR: Path = Path("data/output")


# --------------------------------------------------------------------------- #
# VaR sidecars (written by main.ipynb cell 49)                                #
# --------------------------------------------------------------------------- #

def load_stress_data(
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Optional[dict]:
    """Stressed-VaR overlay: portfolio P&L series, stress windows, summary table."""
    out_dir = Path(out_dir)
    pnl_path = out_dir / "var_portfolio_pnl.csv"
    win_path = out_dir / "var_stress_windows.json"
    sum_path = out_dir / "var_stressed_summary.csv"
    if not (pnl_path.exists() and win_path.exists() and sum_path.exists()):
        return None
    pnl = pd.read_csv(pnl_path, index_col=0, parse_dates=True)["pnl"]
    with open(win_path, encoding="utf-8") as f:
        windows = json.load(f)
    summary = pd.read_csv(sum_path, index_col=0)
    return {"pnl": pnl, "windows": windows, "summary": summary}


def load_multi_nu(
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Optional[dict]:
    """Multi-nu parametric-t sensitivity grid: table + MLE-fit nu."""
    out_dir = Path(out_dir)
    table_path = out_dir / "var_multi_nu_table.csv"
    fit_path = out_dir / "var_multi_nu_fit.json"
    if not (table_path.exists() and fit_path.exists()):
        return None
    table = pd.read_csv(table_path, index_col=0)
    with open(fit_path, encoding="utf-8") as f:
        nu_fit = json.load(f)["nu_fit"]
    return {"table": table, "nu_fit": nu_fit}


def load_decomposition(
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Optional[dict]:
    """Factor / idiosyncratic variance decomposition: scalar summary + per-series betas."""
    out_dir = Path(out_dir)
    json_path = out_dir / "var_decomposition.json"
    betas_path = out_dir / "var_decomposition_betas.csv"
    if not (json_path.exists() and betas_path.exists()):
        return None
    with open(json_path, encoding="utf-8") as f:
        scalars = json.load(f)
    betas = pd.read_csv(betas_path, index_col=0)
    return {"scalars": scalars, "betas": betas}


# --------------------------------------------------------------------------- #
# Pipeline / health artifacts (written by main.ipynb cells 57 / 58)           #
# --------------------------------------------------------------------------- #

def load_pipeline_log(
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Optional[list]:
    p = Path(out_dir) / "pipeline_log.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_health_check(
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Optional[list]:
    p = Path(out_dir) / "health_check.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Per-country exported yield CSVs (written by main.ipynb early in the run)    #
# --------------------------------------------------------------------------- #

def load_country_outputs(
    countries: list[str],
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load each ``data/output/<Country>.csv`` (yield levels, wide-format).

    Returns ``(country_dfs, missing)`` where ``missing`` lists countries
    whose CSV is not on disk.
    """
    out_dir = Path(out_dir)
    country_dfs: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for country in countries:
        p = out_dir / f"{country}.csv"
        if not p.exists():
            missing.append(country)
            continue
        country_dfs[country] = pd.read_csv(p, index_col=0, parse_dates=True)
    return country_dfs, missing


# --------------------------------------------------------------------------- #
# Alert history (written by main.ipynb cell 34)                               #
# --------------------------------------------------------------------------- #

def load_alert_history(
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Optional[dict]:
    p = Path(out_dir) / "alert_history.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
