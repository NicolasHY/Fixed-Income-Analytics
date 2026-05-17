"""
Company EM Fixed Income Intelligence Platform — Source Package.

Public import surface for the analytical pipeline.
All parameters are read from config/funds.yaml — no hardcoded numbers.
"""

from .data_loader import load_config, load_all_countries, build_portfolio_pnl
from .pca_regime import run_pca, run_pca_all_countries, build_regime_features, fit_gmm, generate_alerts
from .var_engine import (
    compute_parametric_var,
    compute_historical_var,
    compute_monte_carlo_var,
    run_all_var_methods,
    kupiec_pof,
    christoffersen_test,
)
from .briefing_engine import build_briefing_payload, generate_daily_briefing, generate_showcase_briefings
from .pipeline_monitor import run_with_monitoring, log_run, check_pipeline_health

__version__ = "0.1.0"
__all__ = [
    # data
    "load_config",
    "load_all_countries",
    "build_portfolio_pnl",
    # pca / regime
    "run_pca",
    "run_pca_all_countries",
    "build_regime_features",
    "fit_gmm",
    "generate_alerts",
    # var
    "compute_parametric_var",
    "compute_historical_var",
    "compute_monte_carlo_var",
    "run_all_var_methods",
    "kupiec_pof",
    "christoffersen_test",
    # briefing
    "build_briefing_payload",
    "generate_daily_briefing",
    "generate_showcase_briefings",
    # monitor
    "run_with_monitoring",
    "log_run",
    "check_pipeline_health",
]
