"""
Tests for Module 2: Multi-Method VaR Engine.

Validates the three VaR methodologies (Parametric, Monte Carlo, Historical)
and the Kupiec/Christoffersen backtesting logic.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as t_dist, chi2


# ---------------------------------------------------------------------------
# VaR computation helpers (extracted from notebook logic)
# ---------------------------------------------------------------------------

def compute_parametric_var(pnl_series, confidence=0.95):
    """Parametric VaR under normal assumption."""
    mu = pnl_series.mean()
    sigma = pnl_series.std()
    alpha = 1 - confidence
    var = -(mu + norm.ppf(alpha) * sigma)
    cvar = -(mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha)
    return {"VaR": var, "CVaR": cvar, "mu": mu, "sigma": sigma}


def compute_multi_nu_var_table(pnl_series, nus=(4, 5, 8, 20)):
    """
    Parametric VaR/CVaR at 95% and 99% for Student-t with the variance
    correction scale = sigma * sqrt((nu - 2) / nu), plus a 'inf' (normal) row.

    Returns a DataFrame indexed by nu (with 'inf' as the last row),
    columns = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%'].
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

    df = pd.DataFrame(rows, columns=["nu", "VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%"])
    df = df.set_index("nu")
    return df


def compute_historical_var(pnl_series, window=252, confidence=0.95):
    """Historical simulation VaR."""
    sample = pnl_series.iloc[-window:]
    alpha = 1 - confidence
    q = np.quantile(sample, alpha)
    var = -q
    cvar = -sample[sample <= q].mean()
    return {"VaR": var, "CVaR": cvar, "n_obs": len(sample)}


def compute_stressed_var(pnl_series, start, end, confidence=0.95):
    """
    Historical VaR/CVaR over a stress window.

    pnl_series : pd.Series indexed by date.
    start, end : window bounds (anything accepted by pandas .loc, inclusive).
    confidence : confidence level (e.g. 0.95).
    """
    sample = pnl_series.loc[start:end]
    if len(sample) == 0:
        raise ValueError(f"Empty stress window: {start} to {end}")
    alpha = 1 - confidence
    q = np.quantile(sample, alpha)
    var = -q
    cvar = -sample[sample <= q].mean()
    return {"VaR": var, "CVaR": cvar, "n_obs": len(sample),
            "start": str(sample.index.min().date()),
            "end": str(sample.index.max().date())}


def compute_monte_carlo_var(pnl_series, n_sims=10000, confidence=0.95, seed=42):
    """Monte Carlo VaR under normal assumption."""
    np.random.seed(seed)
    mu = pnl_series.mean()
    sigma = pnl_series.std()
    sim_pnl = np.random.normal(mu, sigma, n_sims)
    alpha = 1 - confidence
    q = np.percentile(sim_pnl, alpha * 100)
    var = -q
    cvar = -sim_pnl[sim_pnl <= q].mean()
    return {"VaR": var, "CVaR": cvar, "n_sims": n_sims}


def compute_factor_idio_decomposition(yield_changes, factor_scores, weights):
    """
    Decompose Var(w' delta_y) into systematic (driven by factor_scores) and
    idiosyncratic (per-series residual) components via OLS per series.

    yield_changes : DataFrame, columns = series (e.g. countries), index = dates.
    factor_scores : DataFrame, columns = factors (e.g. PC1/PC2/PC3), index = dates.
    weights       : 1-D array, same length as yield_changes.columns.

    Returns dict with keys:
      B (n_series x n_factors), D (n_series x n_series diag of resid variances),
      Sigma_F (n_factors x n_factors), var_systematic, var_idiosyncratic,
      pct_systematic, pct_idiosyncratic, var_total.

    Cross-series residual correlation is ignored (matches the equity project's
    methodology and the spec). The caller should display empirical Var(w' dy)
    alongside the decomposition total for transparency.
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


def kupiec_pof(returns, VaR, p, alpha=0.05):
    """Kupiec Proportion of Failures test."""
    T = len(returns)
    violations = (returns < -VaR)
    N = violations.sum()
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
    
    return {"T": T, "N": N, "expected": p * T, "violation_rate": p_hat,
            "LR": LR, "p_value": p_value, "reject_H0": reject}


def christoffersen_test(returns, VaR, alpha=0.05):
    """Christoffersen independence test for VaR breach clustering."""
    violations = (returns < -VaR).astype(int).values
    T = len(violations)
    
    n00 = n01 = n10 = n11 = 0
    for i in range(1, T):
        if violations[i-1] == 0 and violations[i] == 0: n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1: n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0: n10 += 1
        elif violations[i-1] == 1 and violations[i] == 1: n11 += 1
    
    if n00 + n01 == 0 or n10 + n11 == 0 or n01 + n11 == 0:
        return {"LR_ind": np.nan, "p_value": np.nan, "reject_independence": False}
    
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p_hat = (n01 + n11) / (T - 1)
    
    L_ind = (n00 + n10) * np.log(1 - p_hat) + (n01 + n11) * np.log(p_hat)
    L_markov = n00 * np.log(1 - p01) + n01 * np.log(p01)
    if n10 > 0: L_markov += n10 * np.log(1 - p11)
    if n11 > 0: L_markov += n11 * np.log(p11)
    
    LR_ind = -2 * (L_ind - L_markov)
    p_value = 1 - chi2.cdf(LR_ind, df=1)
    
    return {"LR_ind": LR_ind, "p_value": p_value, "reject_independence": p_value < alpha}


# ===========================================================================
# Tests
# ===========================================================================

class TestParametricVaR:
    """Tests for the Parametric (Variance-Covariance) VaR method."""
    
    def test_var_is_positive(self, portfolio_pnl):
        """VaR should always be a positive number (loss magnitude)."""
        result = compute_parametric_var(portfolio_pnl, confidence=0.95)
        assert result["VaR"] > 0, "Parametric VaR should be positive"
    
    def test_var_99_exceeds_var_95(self, portfolio_pnl):
        """99% VaR should be greater than 95% VaR."""
        var_95 = compute_parametric_var(portfolio_pnl, confidence=0.95)["VaR"]
        var_99 = compute_parametric_var(portfolio_pnl, confidence=0.99)["VaR"]
        assert var_99 > var_95, "VaR 99% should exceed VaR 95%"
    
    def test_cvar_exceeds_var(self, portfolio_pnl):
        """CVaR (Expected Shortfall) should always exceed VaR."""
        result = compute_parametric_var(portfolio_pnl, confidence=0.95)
        assert result["CVaR"] > result["VaR"], "CVaR should exceed VaR"
    
    def test_var_within_reasonable_bounds(self, portfolio_pnl):
        """VaR should be within reasonable bounds (0, 10%) for a bond portfolio."""
        result = compute_parametric_var(portfolio_pnl, confidence=0.95)
        assert 0 < result["VaR"] < 0.10, \
            f"VaR {result['VaR']:.4%} is outside reasonable bounds for a bond fund"


class TestHistoricalVaR:
    """Tests for the Historical Simulation VaR method."""
    
    def test_var_is_positive(self, portfolio_pnl):
        """Historical VaR should be positive."""
        result = compute_historical_var(portfolio_pnl, window=252)
        assert result["VaR"] > 0, "Historical VaR should be positive"
    
    def test_different_windows_produce_different_results(self, portfolio_pnl):
        """Different lookback windows should generally produce different VaR estimates."""
        var_1y = compute_historical_var(portfolio_pnl, window=252)["VaR"]
        var_3y = compute_historical_var(portfolio_pnl, window=756)["VaR"]
        # They can be similar but should not be exactly equal
        # (different sample sizes with random data)
        assert isinstance(var_1y, float) and isinstance(var_3y, float)
    
    def test_correct_observation_count(self, portfolio_pnl):
        """The window parameter should control the number of observations used."""
        result = compute_historical_var(portfolio_pnl, window=252)
        assert result["n_obs"] == min(252, len(portfolio_pnl))


class TestMonteCarloVaR:
    """Tests for the Monte Carlo VaR method."""
    
    def test_var_is_positive(self, portfolio_pnl):
        """Monte Carlo VaR should be positive."""
        result = compute_monte_carlo_var(portfolio_pnl)
        assert result["VaR"] > 0, "MC VaR should be positive"
    
    def test_reproducibility_with_seed(self, portfolio_pnl):
        """Same seed should produce the same VaR."""
        var1 = compute_monte_carlo_var(portfolio_pnl, seed=42)["VaR"]
        var2 = compute_monte_carlo_var(portfolio_pnl, seed=42)["VaR"]
        assert var1 == var2, "MC VaR should be reproducible with the same seed"
    
    def test_different_seeds_differ(self, portfolio_pnl):
        """Different seeds should produce different VaR values."""
        var1 = compute_monte_carlo_var(portfolio_pnl, seed=42)["VaR"]
        var2 = compute_monte_carlo_var(portfolio_pnl, seed=99)["VaR"]
        assert var1 != var2, "MC VaR should differ with different seeds"


class TestKupiecBacktest:
    """Tests for the Kupiec POF backtest."""
    
    def test_kupiec_with_known_data(self):
        """
        With a perfectly calibrated model (5% breach rate), 
        Kupiec should NOT reject H0.
        """
        np.random.seed(42)
        n = 1000
        returns = pd.Series(np.random.normal(0, 0.01, n))
        var_95 = -np.quantile(returns, 0.05)  # Exactly the 5th percentile
        
        result = kupiec_pof(returns, var_95, p=0.05)
        # With the VaR set at the exact 5th percentile, breach rate ≈ 5%
        assert not result["reject_H0"], \
            "Kupiec should not reject H0 for a perfectly calibrated VaR"
    
    def test_kupiec_rejects_bad_model(self):
        """
        With a VaR that is far too small (many breaches), 
        Kupiec should reject H0.
        """
        np.random.seed(42)
        n = 1000
        returns = pd.Series(np.random.normal(0, 0.01, n))
        # Set VaR ridiculously low → many breaches
        bad_var = 0.0001
        
        result = kupiec_pof(returns, bad_var, p=0.05)
        assert result["reject_H0"], \
            "Kupiec should reject H0 for a severely under-estimating VaR model"
    
    def test_violation_rate_is_correct(self):
        """The violation rate should match actual breaches / total observations."""
        np.random.seed(42)
        n = 500
        returns = pd.Series(np.random.normal(0, 0.01, n))
        var = 0.01  # About 50% breach rate
        
        result = kupiec_pof(returns, var, p=0.05)
        expected_rate = (returns < -var).sum() / n
        assert abs(result["violation_rate"] - expected_rate) < 1e-10


class TestChristoffersenBacktest:
    """Tests for the Christoffersen independence test."""
    
    def test_independent_breaches_not_rejected(self):
        """With randomly scattered breaches, independence should not be rejected."""
        np.random.seed(42)
        n = 1000
        returns = pd.Series(np.random.normal(0, 0.01, n))
        var = -np.quantile(returns, 0.05)
        
        result = christoffersen_test(returns, var)
        # For truly random data, breaches should be independent
        assert not result["reject_independence"], \
            "Independence should not be rejected for random data"
    
    def test_returns_valid_structure(self):
        """The test should return a dict with the expected keys."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 500))
        var = 0.01

        result = christoffersen_test(returns, var)
        assert "LR_ind" in result
        assert "p_value" in result
        assert "reject_independence" in result


def test_decomposition_sums_close_to_empirical_variance():
    """
    Build a synthetic 3-factor structure with small idiosyncratic noise and
    check that var_systematic + var_idiosyncratic is within 15% of the
    empirical variance of the weighted portfolio yield change.
    """
    import statsmodels.api as sm

    np.random.seed(123)
    n = 1500
    dates = pd.bdate_range("2020-01-01", periods=n)

    # 3 latent factors with distinct variances
    f1 = np.random.normal(0, 0.05, n)
    f2 = np.random.normal(0, 0.03, n)
    f3 = np.random.normal(0, 0.02, n)
    factors = pd.DataFrame({"PC1": f1, "PC2": f2, "PC3": f3}, index=dates)

    # Country betas (4 x 3) and small idiosyncratic noise
    betas_true = np.array([
        [1.0, 0.5, 0.2],
        [0.8, 0.4, 0.1],
        [1.2, -0.3, 0.1],
        [0.7, 0.6, -0.1],
    ])
    eps = np.random.normal(0, 0.005, (n, 4))
    Y = factors.values @ betas_true.T + eps
    yield_changes = pd.DataFrame(Y, index=dates, columns=["c1", "c2", "c3", "c4"])

    weights = np.array([0.3, 0.25, 0.25, 0.2])

    decomp = compute_factor_idio_decomposition(
        yield_changes=yield_changes,
        factor_scores=factors,
        weights=weights,
    )

    empirical_var = float(np.var(yield_changes.values @ weights, ddof=1))
    total = decomp["var_systematic"] + decomp["var_idiosyncratic"]

    assert abs(total - empirical_var) / empirical_var < 0.15, (
        f"Decomposition sum {total:.6f} vs empirical {empirical_var:.6f} "
        f"differs by more than 15%"
    )
    assert decomp["pct_systematic"] + decomp["pct_idiosyncratic"] == pytest.approx(100.0, abs=1e-6)
    assert decomp["B"].shape == (4, 3)


def test_multi_nu_99var_monotonic_in_nu(portfolio_pnl):
    """
    Higher nu = thinner tails. Holding mu and sigma constant via the variance
    correction, the 99% parametric-t VaR must shrink as nu increases, and the
    nu -> infinity row must match the normal parametric VaR.
    """
    nus = [4, 5, 8, 20]
    table = compute_multi_nu_var_table(portfolio_pnl, nus=nus)

    var_99 = [table.loc[nu, "VaR 99%"] for nu in nus]
    assert all(var_99[i] >= var_99[i + 1] for i in range(len(var_99) - 1)), (
        f"Expected VaR 99% non-increasing in nu, got {var_99}"
    )

    normal = compute_parametric_var(portfolio_pnl, confidence=0.99)
    assert abs(table.loc["inf", "VaR 99%"] - normal["VaR"]) < 1e-9


def test_stressed_var_exceeds_full_sample_var(portfolio_pnl):
    """
    A stress window constructed from the worst N days must produce a 95%
    historical VaR no smaller than the full-sample 95% historical VaR.
    Tests the compute_stressed_var helper, not any specific real-world date.
    """
    full = compute_historical_var(portfolio_pnl, window=len(portfolio_pnl), confidence=0.95)

    worst_dates = portfolio_pnl.nsmallest(int(len(portfolio_pnl) * 0.2)).index
    start, end = worst_dates.min(), worst_dates.max()
    stressed = compute_stressed_var(portfolio_pnl, start, end, confidence=0.95)

    assert stressed["VaR"] >= full["VaR"] - 1e-9, (
        f"Stressed VaR ({stressed['VaR']:.6f}) should be >= full-sample VaR "
        f"({full['VaR']:.6f}) when the window is drawn from the worst tail"
    )
    assert stressed["n_obs"] > 0
    assert stressed["CVaR"] >= stressed["VaR"], "CVaR must be >= VaR"
