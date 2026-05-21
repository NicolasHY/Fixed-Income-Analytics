"""
Structured retrieval for the briefing pipeline.

This is the project's "RAG without embeddings": given a trading date and
the analytics already produced by the upstream pipeline (regime detection,
PCA, VaR, alert engine), assemble a JSON-serialisable dict that an LLM
can be grounded in.

Before this module, ``build_daily_payload`` lived inline in
``main.ipynb`` cell 53. It could not be reused by the chatbot or the
Streamlit pages because those callers do not execute notebook cells. This
module is the extraction — the notebook still has its own copy until the
Orchestration step rewires it, and tests pin the two to behave identically.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_daily_payload(
    date,
    regime_features: pd.DataFrame,
    pca_results: dict[str, dict],
    change_dfs: dict[str, pd.DataFrame],
    portfolio_pnl: pd.Series,
    all_alerts: dict[str, dict],
    var_95: float,
    var_99: float,
) -> dict[str, Any]:
    """Pack all analytics for a single date into a structured dict for the LLM.

    Parameters
    ----------
    date
        Anything ``pd.Timestamp`` accepts (``"2022-09-23"``, ``Timestamp``…).
    regime_features
        Output of ``src.pca_regime.fit_gmm`` — must contain ``regime_label``,
        ``regime_proba``, ``avg_level``, ``dispersion``, ``real_vol``.
    pca_results
        ``{country: {"scores": DataFrame, …}}`` from
        ``src.pca_regime.run_pca_all_countries``.
    change_dfs
        ``{country: DataFrame}`` of daily yield changes in bps.
    portfolio_pnl
        Daily portfolio P&L series (fraction, not percent).
    all_alerts
        ``{date_str: payload}`` from ``src.pca_regime.run_alert_scan``.
    var_95, var_99
        Monte Carlo / t-copula VaR estimates for the date (fractions).

    Returns
    -------
    dict
        JSON-serialisable. Keys are some subset of:
        ``{"date", "regime", "curve_moves_bps", "pc_scores",
        "portfolio", "alerts"}``. ``date``, ``curve_moves_bps``,
        ``pc_scores`` and ``alerts`` are always present; ``regime`` and
        ``portfolio`` are only present if the date is in their indexes.
    """
    date = pd.Timestamp(date)
    payload: dict[str, Any] = {"date": str(date.date())}

    # Regime
    if date in regime_features.index:
        row = regime_features.loc[date]
        payload["regime"] = {
            "label": row["regime_label"],
            "confidence": round(row["regime_proba"], 3),
            "avg_level_shock": round(row["avg_level"], 3),
            "dispersion": round(row["dispersion"], 3),
            "realized_vol": round(row["real_vol"], 3),
        }

    # Per-country curve moves
    curve_moves: dict[str, dict[str, float]] = {}
    for country, dy in change_dfs.items():
        if date in dy.index:
            moves = dy.loc[date].to_dict()
            curve_moves[country] = {
                mat: round(v, 3) for mat, v in moves.items()
                if not np.isnan(v)
            }
    payload["curve_moves_bps"] = curve_moves

    # Per-country PCA scores
    pc_scores: dict[str, dict[str, float]] = {}
    for country, res in pca_results.items():
        if date in res["scores"].index:
            scores = res["scores"].loc[date]
            pc_scores[country] = {
                "PC1_level": round(scores.iloc[0], 2),
                "PC2_slope": round(scores.iloc[1], 2),
                "PC3_curvature": round(scores.iloc[2], 2),
            }
    payload["pc_scores"] = pc_scores

    # Portfolio P&L
    if date in portfolio_pnl.index:
        pnl = portfolio_pnl.loc[date]
        payload["portfolio"] = {
            "daily_pnl_pct": round(pnl * 100, 4),
            "var_95_mc_tcopula": round(var_95 * 100, 4),
            "var_99_mc_tcopula": round(var_99 * 100, 4),
            "var_breach_95": pnl < -var_95,
            "var_breach_99": pnl < -var_99,
        }

    # Alerts
    date_str = str(date.date())
    if date_str in all_alerts:
        payload["alerts"] = all_alerts[date_str]["alerts"]
    else:
        payload["alerts"] = []

    return payload
