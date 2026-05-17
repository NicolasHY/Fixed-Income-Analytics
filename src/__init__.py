"""
Company EM Fixed Income Intelligence Platform — Source Package.

Public import surface for the analytical pipeline.
All parameters are read from config/funds.yaml — no hardcoded numbers.
"""

from .data_loader import (
    load_config,
    load_all_countries,
    load_all_countries_combined,
    build_portfolio_pnl,
    build_portfolio_pnl_from_def,
)
from .pca_regime import run_pca, run_pca_all_countries, build_regime_features, fit_gmm, generate_alerts
from .risk_free import (
    fetch_and_cache_risk_free_rates,
    load_risk_free_rates,
    daily_rf_from_annual,
    align_rf_to_pnl,
)

__version__ = "0.1.0"
__all__ = [
    # data
    "load_config",
    "load_all_countries",
    "load_all_countries_combined",
    "build_portfolio_pnl",
    "build_portfolio_pnl_from_def",
    # pca / regime
    "run_pca",
    "run_pca_all_countries",
    "build_regime_features",
    "fit_gmm",
    "generate_alerts",
    # risk-free rates
    "fetch_and_cache_risk_free_rates",
    "load_risk_free_rates",
    "daily_rf_from_annual",
    "align_rf_to_pnl",
]
