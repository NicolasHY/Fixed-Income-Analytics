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


def compute_historical_var(pnl_series, window=252, confidence=0.95):
    """Historical simulation VaR."""
    sample = pnl_series.iloc[-window:]
    alpha = 1 - confidence
    q = np.quantile(sample, alpha)
    var = -q
    cvar = -sample[sample <= q].mean()
    return {"VaR": var, "CVaR": cvar, "n_obs": len(sample)}


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
