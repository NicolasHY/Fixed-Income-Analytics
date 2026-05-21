"""
Characterization tests for the new pieces of ``src/quant/var_engine.py``
that are not already covered by ``tests/test_var.py``.

Specifically: the Monte Carlo Student-t copula VaR — extracted from
``main.ipynb`` cell 44 — must be:

* deterministic for a given seed (so the briefing payload is reproducible),
* numerically identical to a fresh inline re-implementation of the
  notebook's code path (the parity test).

The single-asset functions are already exercised by ``tests/test_var.py``
through the new import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as t_dist

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader import (
    build_portfolio_pnl,
    load_all_countries_combined,
    load_config,
)
from src.quant.var_engine import compute_mc_t_copula_var


SLICE_END = pd.Timestamp("2025-12-31")
DATA_DIR = "data/raw"


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def proxy_dy(cfg):
    raw = load_all_countries_combined(cfg, data_dir=DATA_DIR)
    sliced = {c: df.loc[:SLICE_END] for c, df in raw.items()}
    _, proxy = build_portfolio_pnl(sliced, cfg)
    return proxy


@pytest.fixture(scope="module")
def lc_w_vec(cfg, proxy_dy):
    raw_w = cfg["lc_fund"]["weights"]
    tot = sum(raw_w.values())
    w = {k: v / tot for k, v in raw_w.items()}
    return np.array([w[c] for c in proxy_dy.columns])


@pytest.fixture(scope="module")
def lc_duration(cfg):
    return cfg["lc_fund"]["effective_duration"]


# --------------------------------------------------------------------------- #
# Structural & determinism                                                    #
# --------------------------------------------------------------------------- #

class TestMcTCopulaStructure:

    def test_returns_expected_keys(self, proxy_dy, lc_w_vec, lc_duration):
        out = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration)
        assert set(out) == {"VaR_95", "VaR_99", "CVaR_95", "CVaR_99"}

    def test_all_values_positive_fractions(self, proxy_dy, lc_w_vec, lc_duration):
        out = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration)
        for k, v in out.items():
            assert v > 0, f"{k} should be a positive loss magnitude"
            assert v < 0.10, f"{k} = {v:.4%} is implausible for the LC fund"

    def test_var99_exceeds_var95(self, proxy_dy, lc_w_vec, lc_duration):
        out = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration)
        assert out["VaR_99"] > out["VaR_95"]

    def test_cvar_exceeds_var(self, proxy_dy, lc_w_vec, lc_duration):
        out = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration)
        assert out["CVaR_95"] > out["VaR_95"]
        assert out["CVaR_99"] > out["VaR_99"]


class TestMcTCopulaDeterminism:

    def test_seed_reproducibility(self, proxy_dy, lc_w_vec, lc_duration):
        a = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration, seed=42)
        b = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration, seed=42)
        assert a == b

    def test_different_seed_changes_output(self, proxy_dy, lc_w_vec, lc_duration):
        a = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration, seed=42)
        b = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration, seed=99)
        assert a["VaR_95"] != b["VaR_95"]


class TestMcTCopulaSnapshot:
    """Pin the MC t-copula VaR levels for the LC fund on the canonical slice.

    Notebook cell 44 produces these same numbers when run with the same
    seed (42) on the same data — if either changes, the briefing payload
    changes, and we want to know.
    """

    def test_lc_fund_var_levels(self, proxy_dy, lc_w_vec, lc_duration):
        out = compute_mc_t_copula_var(
            proxy_dy, lc_w_vec, lc_duration,
            n_sims=10000, copula_dof=5, seed=42,
        )
        assert out["VaR_95"] == pytest.approx(0.00510399, abs=1e-6)
        assert out["VaR_99"] == pytest.approx(0.00972852, abs=1e-6)
        assert out["CVaR_95"] == pytest.approx(0.00833501, abs=1e-6)
        assert out["CVaR_99"] == pytest.approx(0.01484935, abs=1e-6)


# --------------------------------------------------------------------------- #
# Parity with the notebook's inline implementation                            #
# --------------------------------------------------------------------------- #

def _notebook_inline_mc_t(proxy_dy, w_vec, D_eff, n_sims=10000,
                          nu_copula=5, seed=42):
    """Verbatim copy of cell 44's logic — used as the regression oracle."""
    np.random.seed(seed)
    mu_vec = proxy_dy.mean().values
    cov_mat = proxy_dy.cov().values

    sim_dy_normal = np.random.multivariate_normal(mu_vec, cov_mat, size=n_sims)
    _ = -D_eff * (sim_dy_normal @ w_vec) / 100  # warm-up sink, ignored

    marginal_params = {}
    for col in proxy_dy.columns:
        nu, loc, scale = t_dist.fit(proxy_dy[col].dropna())
        marginal_params[col] = (nu, loc, scale)

    U = np.zeros_like(proxy_dy.values, dtype=float)
    for i, col in enumerate(proxy_dy.columns):
        nu, loc, scale = marginal_params[col]
        U[:, i] = t_dist.cdf(proxy_dy[col].values, df=nu, loc=loc, scale=scale)
    U = np.clip(U, 1e-6, 1 - 1e-6)
    Z = norm.ppf(U)
    corr_matrix = np.corrcoef(Z.T)

    chi2_samples = np.random.chisquare(nu_copula, size=n_sims)
    W = nu_copula / chi2_samples

    Z_sim = np.random.multivariate_normal(
        np.zeros(proxy_dy.shape[1]), corr_matrix, size=n_sims,
    )
    T_sim = np.sqrt(W[:, None]) * Z_sim

    sim_dy_copula = np.zeros_like(T_sim)
    for i, col in enumerate(proxy_dy.columns):
        nu_m, loc_m, scale_m = marginal_params[col]
        u_sim = t_dist.cdf(T_sim[:, i], df=nu_copula)
        sim_dy_copula[:, i] = t_dist.ppf(u_sim, df=nu_m, loc=loc_m, scale=scale_m)

    sim_pnl_copula = -D_eff * (sim_dy_copula @ w_vec) / 100

    return {
        "VaR_95": float(-np.percentile(sim_pnl_copula, 5)),
        "VaR_99": float(-np.percentile(sim_pnl_copula, 1)),
        "CVaR_95": float(
            -sim_pnl_copula[sim_pnl_copula <= np.percentile(sim_pnl_copula, 5)].mean()
        ),
        "CVaR_99": float(
            -sim_pnl_copula[sim_pnl_copula <= np.percentile(sim_pnl_copula, 1)].mean()
        ),
    }


def test_mc_t_copula_parity_with_notebook_inline(proxy_dy, lc_w_vec, lc_duration):
    """Module output must equal a fresh in-test re-implementation of cell 44."""
    expected = _notebook_inline_mc_t(proxy_dy, lc_w_vec, lc_duration)
    actual = compute_mc_t_copula_var(proxy_dy, lc_w_vec, lc_duration)
    for k in expected:
        assert actual[k] == pytest.approx(expected[k], abs=1e-12), k
