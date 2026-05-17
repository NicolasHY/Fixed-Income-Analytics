"""
Tests for Module 1.4: Automated Alert Engine.

Validates that the alert system correctly detects regime transitions,
PC z-score breaches, volatility spikes, and country-level outliers.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Alert generation logic (extracted from notebook)
# ---------------------------------------------------------------------------

def generate_alerts(date, regime_features, pca_results, change_dfs):
    """
    Generate structured alerts for a single trading day.
    
    Mirrors the notebook's generate_alerts function, simplified
    for testing (MACRO_EVENTS not needed for unit tests).
    """
    alerts = []
    
    if date not in regime_features.index:
        return None
    
    row = regime_features.loc[date]
    
    # 1. Regime transition
    loc = regime_features.index.get_loc(date)
    if loc > 0:
        prev = regime_features.iloc[loc - 1]
        if row["regime"] != prev["regime"]:
            alerts.append({
                "type": "regime_shift",
                "severity": "high",
                "detail": f"Regime changed from {prev['regime_label']} to {row['regime_label']} "
                          f"(confidence: {row['regime_proba']:.0%})",
            })
    
    # 2. PC score z-score > 3 on any component (per country)
    for country, res in pca_results.items():
        scores = res["scores"]
        if date not in scores.index:
            continue
        for col in scores.columns:
            val = scores.loc[date, col]
            if abs(val) > 3.0:
                direction = "spike" if val > 0 else "drop"
                alerts.append({
                    "type": "pc_zscore_breach",
                    "severity": "medium" if abs(val) < 4 else "high",
                    "detail": f"{country} {col} z-score = {val:.2f} ({direction})",
                    "country": country,
                })
    
    # 3. Realized vol crosses 90th percentile (trailing 1Y = 252 days)
    vol_series = regime_features["real_vol"]
    vol_loc = vol_series.index.get_loc(date)
    if vol_loc >= 252:
        trailing_window = vol_series.iloc[vol_loc - 252 : vol_loc]
        p90 = trailing_window.quantile(0.9)
        if row["real_vol"] > p90:
            alerts.append({
                "type": "vol_spike",
                "severity": "medium",
                "detail": f"Realized vol ({row['real_vol']:.3f}) exceeds 90th percentile "
                          f"of trailing 1Y ({p90:.3f})",
            })
    
    # 4. Single-country curve move > 3 std from 60-day rolling mean
    for country, dy in change_dfs.items():
        if date not in dy.index:
            continue
        loc_c = dy.index.get_loc(date)
        if loc_c < 60:
            continue
        window = dy.iloc[loc_c - 60 : loc_c]
        today = dy.loc[date]
        rolling_mean = window.mean()
        rolling_std = window.std()
        for mat in dy.columns:
            if rolling_std[mat] == 0:
                continue
            z = (today[mat] - rolling_mean[mat]) / rolling_std[mat]
            if abs(z) > 3.0:
                alerts.append({
                    "type": "country_curve_outlier",
                    "severity": "low" if abs(z) < 4 else "medium",
                    "detail": f"{country} {mat} move = {today[mat]:+.2f} bps "
                              f"(z={z:.1f} vs 60d rolling)",
                    "country": country,
                })
    
    # Summary
    severity_order = {"high": 3, "medium": 2, "low": 1}
    alerts.sort(key=lambda a: severity_order.get(a["severity"], 0), reverse=True)
    max_severity = alerts[0]["severity"] if alerts else "none"
    
    return {
        "date": str(date.date()),
        "regime": row["regime_label"],
        "regime_confidence": round(row["regime_proba"], 3),
        "n_alerts": len(alerts),
        "max_severity": max_severity,
        "alerts": alerts,
    }


# ===========================================================================
# Tests
# ===========================================================================

class TestAlertStructure:
    """Tests for the basic structure and format of generated alerts."""
    
    def test_returns_none_for_invalid_date(self, regime_features, pca_results,
                                            synthetic_yield_changes):
        """Alert engine should return None for a date not in the data."""
        fake_date = pd.Timestamp("1900-01-01")
        result = generate_alerts(fake_date, regime_features, pca_results,
                                  synthetic_yield_changes)
        assert result is None
    
    def test_valid_date_returns_dict(self, regime_features, pca_results,
                                     synthetic_yield_changes):
        """A valid date should return a properly structured dict."""
        # Use a date well into the series (past the 252-day warmup)
        date = regime_features.index[300]
        result = generate_alerts(date, regime_features, pca_results,
                                  synthetic_yield_changes)
        
        assert result is not None
        assert "date" in result
        assert "regime" in result
        assert "regime_confidence" in result
        assert "n_alerts" in result
        assert "max_severity" in result
        assert "alerts" in result
    
    def test_n_alerts_matches_list_length(self, regime_features, pca_results,
                                           synthetic_yield_changes):
        """n_alerts field should match the actual number of alerts."""
        date = regime_features.index[300]
        result = generate_alerts(date, regime_features, pca_results,
                                  synthetic_yield_changes)
        
        assert result is not None
        assert result["n_alerts"] == len(result["alerts"])
    
    def test_regime_confidence_in_range(self, regime_features, pca_results,
                                         synthetic_yield_changes):
        """Regime confidence should be between 0 and 1."""
        date = regime_features.index[300]
        result = generate_alerts(date, regime_features, pca_results,
                                  synthetic_yield_changes)
        
        assert result is not None
        assert 0 <= result["regime_confidence"] <= 1.0


class TestAlertSeverity:
    """Tests for alert severity classification and ordering."""
    
    def test_alerts_sorted_by_severity(self, regime_features, pca_results,
                                        synthetic_yield_changes):
        """Alerts should be sorted by severity (high first)."""
        severity_order = {"high": 3, "medium": 2, "low": 1}
        
        # Check multiple dates to find one with alerts
        for i in range(300, min(500, len(regime_features))):
            date = regime_features.index[i]
            result = generate_alerts(date, regime_features, pca_results,
                                      synthetic_yield_changes)
            if result and result["n_alerts"] > 1:
                severities = [severity_order[a["severity"]] for a in result["alerts"]]
                assert severities == sorted(severities, reverse=True), \
                    f"Alerts on {date} are not sorted by severity"
                return  # Found a date with multiple alerts, test passed
        
        # If no date has multiple alerts, skip gracefully
        pytest.skip("No dates found with multiple alerts for severity sorting test")
    
    def test_max_severity_is_correct(self, regime_features, pca_results,
                                      synthetic_yield_changes):
        """max_severity should match the highest severity in the alerts list."""
        severity_rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
        
        for i in range(300, min(500, len(regime_features))):
            date = regime_features.index[i]
            result = generate_alerts(date, regime_features, pca_results,
                                      synthetic_yield_changes)
            if result and result["n_alerts"] > 0:
                expected_max = max(result["alerts"],
                                   key=lambda a: severity_rank[a["severity"]])
                assert result["max_severity"] == expected_max["severity"]
                return
        
        pytest.skip("No dates found with alerts for max severity test")


class TestAlertTypes:
    """Tests for specific alert type detection."""
    
    def test_regime_shift_detected(self, regime_features, pca_results,
                                    synthetic_yield_changes):
        """
        When the regime changes between consecutive days, 
        a regime_shift alert should be generated.
        """
        # Find a date where regime changes
        for i in range(1, len(regime_features)):
            if regime_features.iloc[i]["regime"] != regime_features.iloc[i-1]["regime"]:
                date = regime_features.index[i]
                result = generate_alerts(date, regime_features, pca_results,
                                          synthetic_yield_changes)
                if result is None:
                    continue
                alert_types = [a["type"] for a in result["alerts"]]
                assert "regime_shift" in alert_types, \
                    f"regime_shift alert missing on {date} despite regime change"
                
                # Regime shift should be high severity
                shift_alerts = [a for a in result["alerts"] if a["type"] == "regime_shift"]
                assert shift_alerts[0]["severity"] == "high"
                return
        
        pytest.skip("No regime transitions found in synthetic data")
    
    def test_alert_types_are_valid(self, regime_features, pca_results,
                                    synthetic_yield_changes):
        """All alert types should be from the known set."""
        valid_types = {"regime_shift", "pc_zscore_breach", "vol_spike",
                       "country_curve_outlier"}
        
        for i in range(300, min(500, len(regime_features))):
            date = regime_features.index[i]
            result = generate_alerts(date, regime_features, pca_results,
                                      synthetic_yield_changes)
            if result and result["n_alerts"] > 0:
                for alert in result["alerts"]:
                    assert alert["type"] in valid_types, \
                        f"Unknown alert type: {alert['type']}"
    
    def test_pc_zscore_breach_has_country(self, regime_features, pca_results,
                                           synthetic_yield_changes):
        """PC z-score breach alerts should include a country field."""
        for i in range(60, min(400, len(regime_features))):
            date = regime_features.index[i]
            result = generate_alerts(date, regime_features, pca_results,
                                      synthetic_yield_changes)
            if result:
                for alert in result["alerts"]:
                    if alert["type"] == "pc_zscore_breach":
                        assert "country" in alert, \
                            "pc_zscore_breach alert must include country"
    
    def test_country_curve_outlier_has_country(self, regime_features, pca_results,
                                                synthetic_yield_changes):
        """Country curve outlier alerts should include a country field."""
        for i in range(60, min(400, len(regime_features))):
            date = regime_features.index[i]
            result = generate_alerts(date, regime_features, pca_results,
                                      synthetic_yield_changes)
            if result:
                for alert in result["alerts"]:
                    if alert["type"] == "country_curve_outlier":
                        assert "country" in alert, \
                            "country_curve_outlier alert must include country"


class TestAlertSeverityLevels:
    """Tests for correct severity level assignment."""
    
    def test_severity_values_are_valid(self, regime_features, pca_results,
                                        synthetic_yield_changes):
        """All severity values should be from {low, medium, high}."""
        valid_severities = {"low", "medium", "high"}
        
        for i in range(300, min(500, len(regime_features))):
            date = regime_features.index[i]
            result = generate_alerts(date, regime_features, pca_results,
                                      synthetic_yield_changes)
            if result and result["n_alerts"] > 0:
                for alert in result["alerts"]:
                    assert alert["severity"] in valid_severities, \
                        f"Invalid severity: {alert['severity']}"
