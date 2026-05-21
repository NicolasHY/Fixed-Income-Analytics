"""
Analytical pipeline orchestrator.

Composes the Data layer (``src.data_loader``) with the inference engines
(``src.pca_regime`` and ``src.quant.var_engine``) into a single
``build_analytics_state(config)`` entry point that returns an
:class:`AnalyticsState` ready to feed the briefing flow or the dashboard.

The dataclass is the *contract* between the orchestrator and its
consumers: any field a consumer needs (regime features, PCA results, VaR
levels, …) is reachable through it. The notebook and the chatbot both
build this state once and share it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import (
    build_portfolio_pnl,
    load_all_countries_combined,
    load_config,
)
from src.pca_regime import (
    build_regime_features,
    fit_gmm,
    run_alert_scan,
    run_pca_all_countries,
)
from src.quant.var_engine import compute_mc_t_copula_var


@dataclass
class AnalyticsState:
    """Frozen-in-time output of one full pipeline run."""

    # Inputs
    config: dict
    change_dfs: dict[str, pd.DataFrame]

    # Portfolio (LC fund proxy by default)
    portfolio_pnl: pd.Series
    proxy_dy: pd.DataFrame
    weights: dict[str, float]
    duration: float

    # PCA + regime + alerts
    pca_results: dict[str, dict]
    regime_features: pd.DataFrame
    all_alerts: dict[str, dict]

    # VaR (MC t-copula by default; the headline numbers the briefing uses)
    var_95: float
    var_99: float
    cvar_95: float = field(default=float("nan"))
    cvar_99: float = field(default=float("nan"))


def build_analytics_state(
    config: Optional[dict] = None,
    data_dir: str = "data/raw",
    slice_end: Optional[pd.Timestamp] = None,
) -> AnalyticsState:
    """Run the full analytical pipeline and return the assembled state.

    Parameters
    ----------
    config : dict, optional
        Output of ``src.data_loader.load_config``. Loaded from the default
        location if omitted.
    data_dir : str
        Root of the raw-data tree (passed through to
        ``load_all_countries_combined``).
    slice_end : pd.Timestamp, optional
        If provided, all per-country DataFrames are sliced to
        ``<= slice_end`` before downstream computation. Used by tests to
        keep snapshots stable as new data lands.
    """
    if config is None:
        config = load_config()

    # --- Data layer -------------------------------------------------------
    raw = load_all_countries_combined(config, data_dir=data_dir)
    if slice_end is not None:
        raw = {c: df.loc[:slice_end] for c, df in raw.items()}

    portfolio_pnl, proxy_dy = build_portfolio_pnl(raw, config)
    fund_cfg = config["lc_fund"]
    raw_weights = fund_cfg["weights"]
    total = sum(raw_weights.values())
    weights = {k: v / total for k, v in raw_weights.items()}
    duration = fund_cfg["effective_duration"]

    # --- Inference engines ------------------------------------------------
    pca_results = run_pca_all_countries(
        raw, n_components=config["pca"]["n_components"],
    )
    regime_features = fit_gmm(build_regime_features(pca_results), config)
    all_alerts = run_alert_scan(
        regime_features, pca_results, raw, config,
        config.get("macro_events"),
    )

    mc_cfg = config["var"]["monte_carlo"]
    w_vec = np.array([weights[c] for c in proxy_dy.columns])
    mc = compute_mc_t_copula_var(
        proxy_dy=proxy_dy,
        weights=w_vec,
        duration=duration,
        n_sims=mc_cfg["n_simulations"],
        copula_dof=mc_cfg["copula_dof"],
        seed=mc_cfg["random_seed"],
    )

    return AnalyticsState(
        config=config,
        change_dfs=raw,
        portfolio_pnl=portfolio_pnl,
        proxy_dy=proxy_dy,
        weights=weights,
        duration=duration,
        pca_results=pca_results,
        regime_features=regime_features,
        all_alerts=all_alerts,
        var_95=mc["VaR_95"],
        var_99=mc["VaR_99"],
        cvar_95=mc["CVaR_95"],
        cvar_99=mc["CVaR_99"],
    )
