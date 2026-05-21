"""Quant engines — analytical functions that are not LLM-backed.

Today this package holds the VaR engine (parametric / historical / Monte
Carlo / stressed / multi-nu / factor decomposition + Kupiec/Christoffersen
backtests). When more quant primitives are extracted out of the notebook
(e.g. duration approximations, attribution decompositions) they live here
as peer modules.

The Orchestration layer (``src/orchestration/``) calls into this package
the same way it calls into the Models layer (``src/llm_client.py``) — both
are equal-footing inference engines that the orchestrator coordinates.
"""

from src.quant.var_engine import (
    christoffersen_test,
    compute_factor_idio_decomposition,
    compute_historical_var,
    compute_mc_normal_multivariate_var,
    compute_mc_t_copula_var,
    compute_monte_carlo_var,
    compute_multi_nu_var_table,
    compute_parametric_var,
    compute_stressed_var,
    kupiec_pof,
)

__all__ = [
    "compute_parametric_var",
    "compute_multi_nu_var_table",
    "compute_historical_var",
    "compute_stressed_var",
    "compute_monte_carlo_var",
    "compute_mc_normal_multivariate_var",
    "compute_mc_t_copula_var",
    "compute_factor_idio_decomposition",
    "kupiec_pof",
    "christoffersen_test",
]
