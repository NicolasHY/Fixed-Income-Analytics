"""Tests for src/services/risk_stats.py — compute_risk_stats()."""
import numpy as np
import pandas as pd
import pytest
from src.services.risk_stats import compute_risk_stats


@pytest.fixture
def pdef():
    return {
        "name": "Test Fund",
        "weights": {"Brazil": 50.0, "Mexico": 50.0},
        "effective_duration": 5.0,
        "benchmark_maturity": "5Y",
        "aum_eur": 1_000_000,
    }


@pytest.fixture
def pnl():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    return pd.Series(rng.normal(0.0002, 0.003, 500), index=dates)


@pytest.fixture
def yield_levels(pnl):
    rng = np.random.default_rng(99)
    idx = pnl.index
    return {
        "Brazil": pd.DataFrame(
            {"5Y": rng.uniform(8, 12, len(idx)), "3Y": rng.uniform(7, 10, len(idx))},
            index=idx,
        ),
        "Mexico": pd.DataFrame(
            {"5Y": rng.uniform(7, 10, len(idx)), "3Y": rng.uniform(6, 9, len(idx))},
            index=idx,
        ),
    }


def test_returns_required_keys(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    for key in [
        "ann_ret", "ann_vol", "max_dd", "sharpe_zero", "sortino_zero", "calmar",
        "mod_dur", "dv01", "dv01_eur", "krd", "var_rows", "var_rows_eur",
        "c_vals", "carry", "rolldown", "convexity", "ytm", "yc_slope",
        "current_estr", "current_sofr", "avg_sofr",
    ]:
        assert key in result, f"Missing key: {key}"


def test_vol_is_positive(pdef, pnl, yield_levels):
    assert compute_risk_stats(pdef, pnl, yield_levels)["ann_vol"] > 0


def test_max_drawdown_non_positive(pdef, pnl, yield_levels):
    assert compute_risk_stats(pdef, pnl, yield_levels)["max_dd"] <= 0


def test_var_rows_length_and_positivity(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    assert len(result["var_rows"]) == 2          # α=5% and α=10%
    for vr in result["var_rows"]:
        assert vr["Param VaR (%)"] > 0
        assert vr["Hist VaR (%)"] > 0


def test_krd_sums_to_mod_dur(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    krd_sum = sum(result["krd"].values())
    assert abs(krd_sum - result["mod_dur"]) < 1e-6


def test_dv01_eur_scales_by_aum(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels)
    # dv01_eur = mod_dur * 0.0001 * aum; dv01 = mod_dur * 0.01
    # => dv01_eur = dv01 / 100 * aum
    assert abs(result["dv01_eur"] - result["dv01"] / 100 * pdef["aum_eur"]) < 1e-6


def test_no_rf_data_falls_back_to_zero_rf(pdef, pnl, yield_levels):
    result = compute_risk_stats(pdef, pnl, yield_levels, rf_data=None)
    assert result["sharpe"] == result["sharpe_zero"]
    assert result["sortino"] == result["sortino_zero"]
