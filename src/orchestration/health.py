"""
Pipeline-step runner + health-check assembly (extracted from
``main.ipynb`` cells 57 & 58).

Two concerns are deliberately separated:

* :func:`run_pipeline_step` — wraps any callable in a timing +
  error-capture record. The record format matches the schema the
  Streamlit Pipeline Health page expects, so callers (notebook today,
  cron jobs / app.py tomorrow) can drop it straight into
  ``pipeline_log.json``.
* :func:`build_health_check` — collapses a run log + the analytics
  state into the dashboard's green / yellow / red status cards. Pure
  function: every input is explicit (including ``now``, for tests).

Persistence helpers (:func:`write_pipeline_log`,
:func:`write_health_check`) live here too so callers don't have to know
the artifact paths.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

PIPELINE_LOG_PATH: Path = Path("data/output/pipeline_log.json")
HEALTH_CHECK_PATH: Path = Path("data/output/health_check.json")


# --------------------------------------------------------------------------- #
# Step runner                                                                 #
# --------------------------------------------------------------------------- #

def run_pipeline_step(
    name: str,
    func: Callable,
    *args,
    **kwargs,
) -> tuple[Any, dict]:
    """Run one pipeline step with timing + structured logging.

    Returns ``(result, entry)``. ``entry`` has keys:
    ``step``, ``timestamp``, ``status``, ``runtime_seconds``,
    ``output_shape``, ``error``. On failure, ``result`` is ``None`` and
    ``entry["status"] == "failure"`` with ``entry["error"]`` populated.
    """
    entry: dict = {
        "step": name,
        "timestamp": str(pd.Timestamp.now()),
        "status": None,
        "runtime_seconds": None,
        "output_shape": None,
        "error": None,
    }
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        entry["status"] = "success"
        entry["runtime_seconds"] = round(time.time() - t0, 2)
        if isinstance(result, pd.DataFrame):
            entry["output_shape"] = list(result.shape)
        elif isinstance(result, dict):
            entry["output_shape"] = len(result)
        elif isinstance(result, (list, np.ndarray)):
            entry["output_shape"] = len(result)
        return result, entry
    except Exception as exc:  # noqa: BLE001 — surface every failure to the log
        entry["status"] = "failure"
        entry["runtime_seconds"] = round(time.time() - t0, 2)
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return None, entry


# --------------------------------------------------------------------------- #
# Health check                                                                #
# --------------------------------------------------------------------------- #

def build_health_check(
    pipeline_log: list[dict],
    portfolio_pnl: pd.Series,
    regime_features: pd.DataFrame,
    var_95: float,
    now: Optional[pd.Timestamp] = None,
) -> list[dict]:
    """Five dashboard status cards (pipeline / freshness / runtime / breach / regime).

    Parameters
    ----------
    pipeline_log
        Output of repeated :func:`run_pipeline_step` calls.
    portfolio_pnl
        Daily fund-proxy P&L (fraction, e.g. ``0.002``).
    regime_features
        Output of :func:`src.pca_regime.fit_gmm` — must include an integer
        ``"regime"`` column.
    var_95
        Headline 95% VaR (positive fraction) used for the breach-rate card.
    now
        Override for the freshness reference timestamp. Defaults to
        ``pd.Timestamp.now()``. Tests pin it so the YELLOW/RED cutoffs are
        deterministic.

    Returns
    -------
    list of dict
        Each entry has ``check`` (label), ``status``
        (``"GREEN"`` / ``"YELLOW"`` / ``"RED"``), and ``detail``.
    """
    if now is None:
        now = pd.Timestamp.now()
    checks: list[dict] = []

    # 1. All steps passed
    failed = [e for e in pipeline_log if e["status"] != "success"]
    n_total = len(pipeline_log)
    checks.append({
        "check": "All pipeline steps passed",
        "status": "GREEN" if not failed else "RED",
        "detail": (
            f"{len(failed)} failed" if failed
            else f"All {n_total} steps OK"
        ),
    })

    # 2. Data freshness
    latest_date = portfolio_pnl.index.max()
    staleness = (now - latest_date).days
    checks.append({
        "check": "Data freshness",
        "status": (
            "GREEN" if staleness <= 7
            else "YELLOW" if staleness <= 30
            else "RED"
        ),
        "detail": f"Latest data: {latest_date.date()} ({staleness} days ago)",
    })

    # 3. Runtime anomalies
    runtimes = {
        e["step"]: e["runtime_seconds"] for e in pipeline_log
        if e["runtime_seconds"] is not None
    }
    if runtimes:
        slowest_name = max(runtimes, key=runtimes.get)
        max_runtime = runtimes[slowest_name]
    else:
        slowest_name, max_runtime = "n/a", 0
    checks.append({
        "check": "Runtime within bounds",
        "status": "GREEN" if max_runtime < 30 else "YELLOW",
        "detail": f"Slowest step: {slowest_name} ({max_runtime}s)",
    })

    # 4. VaR breach rate (trailing 250 days)
    recent_pnl = portfolio_pnl.iloc[-250:]
    breach_rate = float((recent_pnl < -var_95).mean())
    checks.append({
        "check": "VaR 95% breach rate (250d)",
        "status": "GREEN" if 0.03 <= breach_rate <= 0.08 else "YELLOW",
        "detail": f"{breach_rate:.1%} (target: 5%, tolerance: 3-8%)",
    })

    # 5. Regime stability (last 20 days)
    recent_regimes = regime_features["regime"].iloc[-20:]
    n_transitions = int((recent_regimes.diff() != 0).sum() - 1)
    checks.append({
        "check": "Regime stability (20d)",
        "status": "GREEN" if n_transitions <= 3 else "YELLOW",
        "detail": f"{n_transitions} transitions in last 20 trading days",
    })

    return checks


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #

def write_pipeline_log(
    log: list[dict], path: str | Path = PIPELINE_LOG_PATH,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def write_health_check(
    checks: list[dict], path: str | Path = HEALTH_CHECK_PATH,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
