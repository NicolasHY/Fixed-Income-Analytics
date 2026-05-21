"""
VaR engine — parametric / historical / Monte Carlo / stressed / decomposition
plus Kupiec POF and Christoffersen independence backtests.

Before this module these formulas existed in two places (cells in
``main.ipynb`` and inline helpers in ``tests/test_var.py``). Both copies
have been replaced by imports from here, so the engine is single-sourced.

Function signatures match the original test_var.py helpers exactly so the
existing VaR test suite (18 tests) ports over without modification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t as t_dist


# --------------------------------------------------------------------------- #
# Parametric VaR                                                              #
# --------------------------------------------------------------------------- #

def compute_parametric_var(pnl_series: pd.Series, confidence: float = 0.95) -> dict:
    """Parametric VaR under a normal-distribution assumption."""
    mu = pnl_series.mean()
    sigma = pnl_series.std()
    alpha = 1 - confidence
    var = -(mu + norm.ppf(alpha) * sigma)
    cvar = -(mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha)
    return {"VaR": var, "CVaR": cvar, "mu": mu, "sigma": sigma}


def compute_multi_nu_var_table(
    pnl_series: pd.Series, nus=(4, 5, 8, 20),
) -> pd.DataFrame:
    """
    Parametric VaR/CVaR at 95% and 99% for Student-t with the variance
    correction scale = sigma * sqrt((nu - 2) / nu), plus a normal ('inf') row.

    Returns a DataFrame indexed by nu (with 'inf' as the last row), columns
    = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%'].
    """
    mu = pnl_series.mean()
    sigma = pnl_series.std()
    rows = []
    for nu in nus:
        if nu <= 2:
            raise ValueError(f"nu must be > 2 for variance correction, got {nu}")
        s = sigma * np.sqrt((nu - 2) / nu)
        var_95 = -(mu + t_dist.ppf(0.05, df=nu) * s)
        var_99 = -(mu + t_dist.ppf(0.01, df=nu) * s)
        cvar_95 = -pnl_series[pnl_series <= -var_95].mean()
        cvar_99 = -pnl_series[pnl_series <= -var_99].mean()
        rows.append([nu, var_95, var_99, cvar_95, cvar_99])

    # Normal row (nu -> infinity)
    var_95_n = -(mu + norm.ppf(0.05) * sigma)
    var_99_n = -(mu + norm.ppf(0.01) * sigma)
    cvar_95_n = -(mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05)
    cvar_99_n = -(mu - sigma * norm.pdf(norm.ppf(0.01)) / 0.01)
    rows.append(["inf", var_95_n, var_99_n, cvar_95_n, cvar_99_n])

    df = pd.DataFrame(rows, columns=["nu", "VaR 95%", "VaR 99%",
                                     "CVaR 95%", "CVaR 99%"])
    df = df.set_index("nu")
    return df


# --------------------------------------------------------------------------- #
# Historical simulation VaR                                                   #
# --------------------------------------------------------------------------- #

def compute_historical_var(
    pnl_series: pd.Series, window: int = 252, confidence: float = 0.95,
) -> dict:
    """Historical simulation VaR over the trailing ``window`` observations."""
    sample = pnl_series.iloc[-window:]
    alpha = 1 - confidence
    q = np.quantile(sample, alpha)
    return {
        "VaR": -q,
        "CVaR": -sample[sample <= q].mean(),
        "n_obs": len(sample),
    }


def compute_stressed_var(pnl_series, start, end, confidence: float = 0.95) -> dict:
    """Historical VaR/CVaR over a specific stress window (inclusive bounds)."""
    sample = pnl_series.loc[start:end]
    if len(sample) == 0:
        raise ValueError(f"Empty stress window: {start} to {end}")
    alpha = 1 - confidence
    q = np.quantile(sample, alpha)
    return {
        "VaR": -q,
        "CVaR": -sample[sample <= q].mean(),
        "n_obs": len(sample),
        "start": str(sample.index.min().date()),
        "end": str(sample.index.max().date()),
    }


# --------------------------------------------------------------------------- #
# Monte Carlo VaR                                                             #
# --------------------------------------------------------------------------- #

def compute_monte_carlo_var(
    pnl_series: pd.Series,
    n_sims: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Monte Carlo VaR under a normal P&L assumption (single-series)."""
    np.random.seed(seed)
    mu = pnl_series.mean()
    sigma = pnl_series.std()
    sim_pnl = np.random.normal(mu, sigma, n_sims)
    alpha = 1 - confidence
    q = np.percentile(sim_pnl, alpha * 100)
    return {
        "VaR": -q,
        "CVaR": -sim_pnl[sim_pnl <= q].mean(),
        "n_sims": n_sims,
    }


def compute_mc_normal_multivariate_var(
    proxy_dy: pd.DataFrame,
    weights: np.ndarray,
    duration: float,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict:
    """Monte Carlo VaR under multivariate-normal yield changes.

    Companion to :func:`compute_mc_t_copula_var` — uses the same
    portfolio inputs but a thin-tailed Gaussian assumption. Returns
    ``{"VaR_95", "VaR_99", "CVaR_95", "CVaR_99"}`` as positive fractions.
    """
    np.random.seed(seed)
    w_vec = np.asarray(weights, dtype=float)
    mu_vec = proxy_dy.mean().values
    cov_mat = proxy_dy.cov().values
    sim_dy = np.random.multivariate_normal(mu_vec, cov_mat, size=n_sims)
    sim_pnl = -duration * (sim_dy @ w_vec) / 100
    return {
        "VaR_95": float(-np.percentile(sim_pnl, 5)),
        "VaR_99": float(-np.percentile(sim_pnl, 1)),
        "CVaR_95": float(
            -sim_pnl[sim_pnl <= np.percentile(sim_pnl, 5)].mean()
        ),
        "CVaR_99": float(
            -sim_pnl[sim_pnl <= np.percentile(sim_pnl, 1)].mean()
        ),
    }


def compute_mc_t_copula_var(
    proxy_dy: pd.DataFrame,
    weights: np.ndarray,
    duration: float,
    n_sims: int = 10000,
    copula_dof: int = 5,
    seed: int = 42,
) -> dict:
    """Monte Carlo VaR under a Student-t copula with t-distributed marginals.

    Extracted verbatim from ``main.ipynb`` cell 44. Designed to be
    bit-reproducible: same seed, same RNG call sequence (``multivariate_normal``
    for the seed warm-up, then ``chisquare``, then ``multivariate_normal``
    again for ``Z_sim``).

    Parameters
    ----------
    proxy_dy : DataFrame
        Per-country daily yield changes (columns = countries, in % points).
    weights : ndarray
        Portfolio weights aligned to ``proxy_dy.columns`` (sum to 1).
    duration : float
        Effective portfolio duration (years).
    n_sims, copula_dof, seed : tuning knobs.

    Returns
    -------
    dict
        ``{"VaR_95", "VaR_99", "CVaR_95", "CVaR_99"}`` as positive fractions
        (e.g. 0.005 = 0.5% loss).
    """
    np.random.seed(seed)
    w_vec = np.asarray(weights, dtype=float)

    # A. Multivariate-normal warm-up (must run first to consume RNG state
    # the same way the notebook did — even though we only use the t-copula
    # path's outputs below).
    mu_vec = proxy_dy.mean().values
    cov_mat = proxy_dy.cov().values
    sim_dy_normal = np.random.multivariate_normal(mu_vec, cov_mat, size=n_sims)
    _ = -duration * (sim_dy_normal @ w_vec) / 100  # discarded; preserves RNG order

    # B. Student-t copula
    # Fit marginal t-distributions per country
    marginal_params = {}
    for col in proxy_dy.columns:
        nu_m, loc_m, scale_m = t_dist.fit(proxy_dy[col].dropna())
        marginal_params[col] = (nu_m, loc_m, scale_m)

    # PIT to uniform via marginal CDFs, then to standard normal
    U = np.zeros_like(proxy_dy.values, dtype=float)
    for i, col in enumerate(proxy_dy.columns):
        nu_m, loc_m, scale_m = marginal_params[col]
        U[:, i] = t_dist.cdf(proxy_dy[col].values, df=nu_m, loc=loc_m, scale=scale_m)
    U = np.clip(U, 1e-6, 1 - 1e-6)
    Z = norm.ppf(U)
    corr_matrix = np.corrcoef(Z.T)

    # Simulate from the copula
    chi2_samples = np.random.chisquare(copula_dof, size=n_sims)
    W = copula_dof / chi2_samples
    Z_sim = np.random.multivariate_normal(
        np.zeros(len(proxy_dy.columns)), corr_matrix, size=n_sims,
    )
    T_sim = np.sqrt(W[:, None]) * Z_sim  # samples in standard-normal space

    # Transform back to original marginals via inverse CDF
    sim_dy_copula = np.zeros_like(T_sim)
    for i, col in enumerate(proxy_dy.columns):
        nu_m, loc_m, scale_m = marginal_params[col]
        u_sim = t_dist.cdf(T_sim[:, i], df=copula_dof)
        sim_dy_copula[:, i] = t_dist.ppf(u_sim, df=nu_m, loc=loc_m, scale=scale_m)

    sim_pnl = -duration * (sim_dy_copula @ w_vec) / 100

    var_95 = -np.percentile(sim_pnl, 5)
    var_99 = -np.percentile(sim_pnl, 1)
    cvar_95 = -sim_pnl[sim_pnl <= np.percentile(sim_pnl, 5)].mean()
    cvar_99 = -sim_pnl[sim_pnl <= np.percentile(sim_pnl, 1)].mean()
    return {
        "VaR_95": float(var_95),
        "VaR_99": float(var_99),
        "CVaR_95": float(cvar_95),
        "CVaR_99": float(cvar_99),
    }


# --------------------------------------------------------------------------- #
# Factor / idiosyncratic variance decomposition                               #
# --------------------------------------------------------------------------- #

def compute_factor_idio_decomposition(
    yield_changes: pd.DataFrame,
    factor_scores: pd.DataFrame,
    weights,
) -> dict:
    """Decompose Var(w' delta_y) into systematic + idiosyncratic via OLS per series.

    Cross-series residual correlation is ignored (matches the equity project
    methodology and the spec). The caller should display empirical
    Var(w' dy) alongside the decomposition total for transparency.
    """
    import statsmodels.api as sm

    common = yield_changes.index.intersection(factor_scores.index)
    Y = yield_changes.loc[common]
    F = factor_scores.loc[common]
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != Y.shape[1]:
        raise ValueError(f"weights length {w.shape[0]} != n_series {Y.shape[1]}")

    n_series = Y.shape[1]
    n_factors = F.shape[1]
    B = np.zeros((n_series, n_factors))
    resid_var = np.zeros(n_series)

    X = sm.add_constant(F.values)
    for i, col in enumerate(Y.columns):
        model = sm.OLS(Y[col].values, X).fit()
        B[i, :] = model.params[1:]
        resid_var[i] = model.resid.var(ddof=1)

    Sigma_F = F.cov().values
    D = np.diag(resid_var)

    var_systematic = float(w @ B @ Sigma_F @ B.T @ w)
    var_idiosyncratic = float(w @ D @ w)
    var_total = var_systematic + var_idiosyncratic
    pct_systematic = 100.0 * var_systematic / var_total
    pct_idiosyncratic = 100.0 * var_idiosyncratic / var_total

    return {
        "B": B, "D": D, "Sigma_F": Sigma_F,
        "var_systematic": var_systematic,
        "var_idiosyncratic": var_idiosyncratic,
        "var_total": var_total,
        "pct_systematic": pct_systematic,
        "pct_idiosyncratic": pct_idiosyncratic,
    }


# --------------------------------------------------------------------------- #
# Backtests                                                                   #
# --------------------------------------------------------------------------- #

def kupiec_pof(returns, VaR, p, alpha: float = 0.05) -> dict:
    """Kupiec Proportion-of-Failures test."""
    T = len(returns)
    violations = (returns < -VaR)
    N = int(violations.sum())
    p_hat = N / T

    if N == 0:
        LR = -2 * T * np.log(1 - p)
    elif N == T:
        LR = -2 * T * np.log(p)
    else:
        LR = -2 * (
            N * np.log(p) + (T - N) * np.log(1 - p)
            - N * np.log(p_hat) - (T - N) * np.log(1 - p_hat)
        )

    p_value = 1 - chi2.cdf(LR, df=1)
    crit = chi2.ppf(1 - alpha, df=1)
    reject = LR > crit

    return {
        "T": T, "N": N, "expected": p * T, "violation_rate": p_hat,
        "LR": LR, "p_value": p_value, "reject_H0": reject,
    }


def christoffersen_test(returns, VaR, alpha: float = 0.05) -> dict:
    """Christoffersen independence test for VaR breach clustering."""
    violations = (returns < -VaR).astype(int).values
    T = len(violations)

    n00 = n01 = n10 = n11 = 0
    for i in range(1, T):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0:
            n10 += 1
        elif violations[i-1] == 1 and violations[i] == 1:
            n11 += 1

    if n00 + n01 == 0 or n10 + n11 == 0 or n01 + n11 == 0:
        return {"LR_ind": np.nan, "p_value": np.nan,
                "reject_independence": False}

    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p_hat = (n01 + n11) / (T - 1)

    L_ind = (n00 + n10) * np.log(1 - p_hat) + (n01 + n11) * np.log(p_hat)
    L_markov = n00 * np.log(1 - p01) + n01 * np.log(p01)
    if n10 > 0:
        L_markov += n10 * np.log(1 - p11)
    if n11 > 0:
        L_markov += n11 * np.log(p11)

    LR_ind = -2 * (L_ind - L_markov)
    p_value = 1 - chi2.cdf(LR_ind, df=1)

    return {
        "LR_ind": LR_ind, "p_value": p_value,
        "reject_independence": p_value < alpha,
    }
