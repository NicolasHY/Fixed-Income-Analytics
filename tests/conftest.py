"""
Shared fixtures for the EM Fixed Income Analytics test suite.

Provides synthetic data for testing VaR, PCA, and Alert modules
without requiring actual market data.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, t as t_dist, chi2


# ---------------------------------------------------------------------------
# Synthetic yield curve data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_yield_changes():
    """
    Generate synthetic daily yield change DataFrames for 4 countries.
    Simulates ~3 years of daily data with different volatility regimes.
    """
    np.random.seed(42)
    n_days = 756  # ~3 years of trading days
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    maturities = ["2Y", "5Y", "10Y"]
    
    countries = {}
    for i, country in enumerate(["Brazil", "Mexico", "South Africa", "Poland"]):
        # Base volatility varies by country
        base_vol = 0.05 + 0.02 * i
        data = {}
        for j, mat in enumerate(maturities):
            # Maturity adds vol
            vol = base_vol + 0.01 * j
            changes = np.random.normal(0, vol, n_days)
            # Inject a stress period (days 200-250)
            changes[200:250] *= 3.0
            data[mat] = changes
        countries[country] = pd.DataFrame(data, index=dates)
    
    return countries


@pytest.fixture
def portfolio_pnl(synthetic_yield_changes):
    """
    Build a synthetic portfolio P&L series from yield changes,
    mimicking the LC fund proxy from the notebook.
    """
    lc_weights = {
        "Brazil": 0.303, "Mexico": 0.264,
        "South Africa": 0.234, "Poland": 0.198
    }
    D_eff = 5.22
    benchmark_mat = "5Y"
    
    proxy_dy = pd.DataFrame({
        country: synthetic_yield_changes[country][benchmark_mat]
        for country in lc_weights
    })
    proxy_dy = proxy_dy.dropna()
    
    w_vec = np.array([lc_weights[c] for c in proxy_dy.columns])
    portfolio_dy = proxy_dy @ w_vec
    pnl = -D_eff * (portfolio_dy / 100)
    
    return pnl


@pytest.fixture
def pca_results(synthetic_yield_changes):
    """
    Run PCA on the synthetic yield changes, mimicking the notebook's approach.
    Returns a dict of {country: {scores, explained_var, loadings}}.
    """
    results = {}
    for country, dy in synthetic_yield_changes.items():
        dy_clean = dy.dropna()
        scaler = StandardScaler()
        dy_std = scaler.fit_transform(dy_clean)
        
        n_components = min(3, dy_clean.shape[1])
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(dy_std)
        
        col_names = [
            f"PC{i+1} ({'level' if i==0 else 'slope' if i==1 else 'curve'})"
            for i in range(n_components)
        ]
        scores_df = pd.DataFrame(scores, index=dy_clean.index, columns=col_names)
        
        results[country] = {
            "scores": scores_df,
            "explained_var": pca.explained_variance_ratio_,
            "loadings": pca.components_,
            "n_components": n_components,
        }
    
    return results


@pytest.fixture
def regime_features(pca_results, synthetic_yield_changes):
    """
    Build regime features from PCA results, mimicking the notebook.
    """
    # Get common dates
    dates = None
    for country, res in pca_results.items():
        if dates is None:
            dates = res["scores"].index
        else:
            dates = dates.intersection(res["scores"].index)
    
    # Average PC1 across countries
    pc1_scores = pd.DataFrame({
        country: res["scores"].iloc[:, 0]
        for country, res in pca_results.items()
    })
    
    features = pd.DataFrame(index=dates)
    features["avg_level"] = pc1_scores.mean(axis=1)
    features["dispersion"] = pc1_scores.std(axis=1)
    
    # Realized vol (rolling 20-day std of avg PC1)
    features["real_vol"] = features["avg_level"].rolling(20).std()
    features = features.dropna()
    
    # Fit GMM
    scaler = StandardScaler()
    X = scaler.fit_transform(features[["avg_level", "dispersion", "real_vol"]])
    
    gmm = GaussianMixture(n_components=3, covariance_type="full",
                           n_init=5, random_state=42)
    features["regime"] = gmm.fit_predict(X)
    features["regime_proba"] = gmm.predict_proba(X).max(axis=1)
    
    # Simple labels
    labels = {0: "Normal", 1: "Stress", 2: "Risk-On"}
    features["regime_label"] = features["regime"].map(labels)
    
    return features
