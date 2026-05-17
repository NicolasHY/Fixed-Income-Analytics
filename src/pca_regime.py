"""
Module: pca_regime.py
=====================
Yield Curve PCA, GMM Regime Detection, and Alert Engine (Module 1).

All thresholds come from config['pca'], config['regime'], config['alerts'].
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_PC_LABELS = ["PC1 (level)", "PC2 (slope)", "PC3 (curvature)"]


# ---------------------------------------------------------------------------
# 1.2  PCA
# ---------------------------------------------------------------------------

def run_pca(dy_df: pd.DataFrame, n_components: int = 3) -> dict[str, Any]:
    """
    Run PCA on daily yield changes for a single country.

    Returns dict with: scores (DataFrame), explained_var, loadings, loadings_df,
    n_components, scaler, pca object.
    Column names match the notebook and conftest.py fixtures.
    """
    dy_clean = dy_df.dropna()
    n_comp = min(n_components, dy_clean.shape[1])

    scaler = StandardScaler()
    dy_std = scaler.fit_transform(dy_clean)

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(dy_std)

    col_names = _PC_LABELS[:n_comp]
    scores_df = pd.DataFrame(scores, index=dy_clean.index, columns=col_names)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=dy_clean.columns,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )

    return {
        "scores": scores_df,
        "explained_var": pca.explained_variance_ratio_,
        "loadings": pca.components_,
        "loadings_df": loadings_df,
        "n_components": n_comp,
        "scaler": scaler,
        "pca": pca,
    }


def run_pca_all_countries(
    change_dfs: dict[str, pd.DataFrame],
    n_components: int = 3,
) -> dict[str, dict]:
    """Run PCA independently for each country. Returns {country: result_dict}."""
    return {country: run_pca(dy, n_components=n_components)
            for country, dy in change_dfs.items()}


# ---------------------------------------------------------------------------
# 1.3  Regime detection
# ---------------------------------------------------------------------------

def build_regime_features(pca_results: dict[str, dict]) -> pd.DataFrame:
    """
    Build GMM input features from PCA results.

    Features: avg_level (mean PC1 across countries), dispersion (std PC1),
    real_vol (20-day rolling std of avg_level). Rows with NaN dropped.
    """
    dates = None
    for res in pca_results.values():
        idx = res["scores"].index
        dates = idx if dates is None else dates.intersection(idx)

    pc1_panel = pd.DataFrame({
        country: res["scores"].iloc[:, 0].reindex(dates)
        for country, res in pca_results.items()
    })

    features = pd.DataFrame(index=dates)
    features["avg_level"] = pc1_panel.mean(axis=1)
    features["dispersion"] = pc1_panel.std(axis=1)
    features["real_vol"] = features["avg_level"].rolling(20).std()
    return features.dropna()


def _select_n_components_bic(X: np.ndarray, max_k: int, cov_type: str,
                              n_init: int, random_state: int) -> int:
    """Select GMM k by BIC."""
    bic_scores = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                              n_init=n_init, random_state=random_state)
        gmm.fit(X)
        bic_scores.append((k, gmm.bic(X)))
    return min(bic_scores, key=lambda t: t[1])[0]


def fit_gmm(
    features: pd.DataFrame,
    config: dict,
    n_components: int | None = None,
) -> pd.DataFrame:
    """
    Fit a GMM on regime features. Adds regime, regime_proba, regime_label columns.
    n_components selected by BIC if not provided. Config drives all hyperparameters.
    """
    reg_cfg = config["regime"]["gmm"]
    label_map: dict[int, str] = {int(k): v for k, v in config["regime"]["labels"].items()}

    feature_cols = ["avg_level", "dispersion", "real_vol"]
    scaler = StandardScaler()
    X = scaler.fit_transform(features[feature_cols])

    if n_components is None:
        n_components = _select_n_components_bic(
            X, reg_cfg["max_components"], reg_cfg["covariance_type"],
            reg_cfg["n_init"], reg_cfg["random_state"],
        )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=reg_cfg["covariance_type"],
        n_init=reg_cfg["n_init"],
        random_state=reg_cfg["random_state"],
    )
    gmm.fit(X)

    out = features.copy()
    out["regime"] = gmm.predict(X)
    out["regime_proba"] = gmm.predict_proba(X).max(axis=1)
    out["regime_label"] = out["regime"].map(label_map).fillna("Unknown")
    return out


# ---------------------------------------------------------------------------
# 1.4  Alert engine
# ---------------------------------------------------------------------------

def generate_alerts(
    date: pd.Timestamp,
    regime_features: pd.DataFrame,
    pca_results: dict[str, dict],
    change_dfs: dict[str, pd.DataFrame],
    config: dict,
    macro_events: dict[str, str] | None = None,
) -> dict | None:
    """
    Generate structured alert payload for a single trading date.

    Alert types: regime_shift, pc_zscore_breach, vol_spike, country_curve_outlier.
    All thresholds driven by config['alerts']. Returns None if date not in index.
    """
    if date not in regime_features.index:
        return None

    a = config["alerts"]
    row = regime_features.loc[date]
    alerts: list[dict] = []

    # 1. Regime transition
    loc = regime_features.index.get_loc(date)
    if loc > 0:
        prev = regime_features.iloc[loc - 1]
        if row["regime"] != prev["regime"]:
            alerts.append({
                "type": "regime_shift",
                "severity": "high",
                "detail": (
                    f"Regime changed from {prev['regime_label']} to {row['regime_label']} "
                    f"(confidence: {row['regime_proba']:.0%})"
                ),
            })

    # 2. PC score z-score breach
    for country, res in pca_results.items():
        scores = res["scores"]
        if date not in scores.index:
            continue
        for col in scores.columns:
            val = float(scores.loc[date, col])
            if abs(val) > a["pc_zscore_threshold"]:
                alerts.append({
                    "type": "pc_zscore_breach",
                    "severity": "high" if abs(val) >= a["pc_zscore_high_threshold"] else "medium",
                    "detail": f"{country} {col} z-score = {val:.2f} ({'spike' if val > 0 else 'drop'})",
                    "country": country,
                })

    # 3. Realised vol spike
    vol_series = regime_features["real_vol"]
    vol_loc = vol_series.index.get_loc(date)
    if vol_loc >= a["vol_trailing_window"]:
        trailing = vol_series.iloc[vol_loc - a["vol_trailing_window"]: vol_loc]
        p_thresh = trailing.quantile(a["vol_percentile"])
        if row["real_vol"] > p_thresh:
            alerts.append({
                "type": "vol_spike",
                "severity": "medium",
                "detail": (
                    f"Realised vol ({row['real_vol']:.3f}) exceeds "
                    f"{a['vol_percentile']:.0%} percentile ({p_thresh:.3f})"
                ),
            })

    # 4. Country curve outlier
    for country, dy in change_dfs.items():
        if date not in dy.index:
            continue
        c_loc = dy.index.get_loc(date)
        if c_loc < a["curve_rolling_window"]:
            continue
        window = dy.iloc[c_loc - a["curve_rolling_window"]: c_loc]
        today = dy.loc[date]
        rolling_mean = window.mean()
        rolling_std = window.std()
        for mat in dy.columns:
            if rolling_std[mat] == 0:
                continue
            z = (today[mat] - rolling_mean[mat]) / rolling_std[mat]
            if abs(z) > a["curve_zscore_threshold"]:
                alerts.append({
                    "type": "country_curve_outlier",
                    "severity": "medium" if abs(z) >= a["curve_zscore_high_threshold"] else "low",
                    "detail": f"{country} {mat} move = {today[mat]:+.2f} bps (z={z:.1f} vs {a['curve_rolling_window']}d rolling)",
                    "country": country,
                })

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    alerts.sort(key=lambda x: severity_rank.get(x["severity"], 0), reverse=True)
    max_severity = alerts[0]["severity"] if alerts else "none"

    payload: dict = {
        "date": str(date.date()),
        "regime": row["regime_label"],
        "regime_confidence": round(float(row["regime_proba"]), 3),
        "n_alerts": len(alerts),
        "max_severity": max_severity,
        "alerts": alerts,
    }
    if macro_events:
        macro_note = macro_events.get(str(date.date()))
        if macro_note:
            payload["macro_event"] = macro_note

    return payload


def run_alert_scan(
    regime_features: pd.DataFrame,
    pca_results: dict[str, dict],
    change_dfs: dict[str, pd.DataFrame],
    config: dict,
    macro_events: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Run generate_alerts over all dates. Returns {date_str: payload} for alert days."""
    all_alerts: dict[str, dict] = {}
    for date in regime_features.index:
        result = generate_alerts(date, regime_features, pca_results,
                                 change_dfs, config, macro_events)
        if result and result["n_alerts"] > 0:
            all_alerts[result["date"]] = result
    logger.info("Alert scan: %d alert days / %d total.", len(all_alerts), len(regime_features))
    return all_alerts
