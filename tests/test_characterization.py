"""
Characterization tests for the Company EM FI platform.

Purpose
-------
These tests lock down the *current* output of the core analytical workflows
before any refactor that splits the codebase into the five GenAI layers
(Application / Orchestration / Data-RAG / Models-Inference / Infrastructure).

They are intentionally written against the public surface of each module
(``src.data_loader``, ``src.pca_regime``, ``src.risk_free``, ``chatbot``) and
the real ``config/funds.yaml`` so a regression in any of them will fail here.

Stability notes
---------------
* Tests that consume real CSVs under ``data/raw/`` slice every series to
  ``<= 2025-12-31`` so they remain stable as new data is appended.
* GMM-based regime tests assert *structure* (number of components, label set,
  probability range) — not the count per cluster, because sklearn's cluster
  numbering is permutation-invariant across versions.
* Floating-point assertions use ``pytest.approx`` with explicit tolerances
  computed from one initial run of the pipeline; bumping a value here means
  the underlying analytics changed and the diff should be reviewed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (containing ``src`` and ``chatbot.py``) is importable
# regardless of where pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    build_portfolio_pnl_from_def,
    load_all_countries_combined,
    load_config,
    load_yield_changes,
)
from src.pca_regime import (
    build_regime_features,
    fit_gmm,
    generate_alerts,
    run_pca,
    run_pca_all_countries,
)
from src.risk_free import daily_rf_from_annual


# --------------------------------------------------------------------------- #
# Shared fixtures (real-data-backed, sliced for stability)                    #
# --------------------------------------------------------------------------- #

SLICE_END = pd.Timestamp("2025-12-31")
DATA_DIR = "data/raw"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def change_dfs(cfg) -> dict[str, pd.DataFrame]:
    """All-country yield-change DataFrames sliced to a fixed end date."""
    raw = load_all_countries_combined(cfg, data_dir=DATA_DIR)
    return {country: df.loc[:SLICE_END] for country, df in raw.items()}


@pytest.fixture(scope="module")
def pca_all(change_dfs) -> dict[str, dict]:
    return run_pca_all_countries(change_dfs, n_components=3)


@pytest.fixture(scope="module")
def regime_df(pca_all, cfg) -> pd.DataFrame:
    features = build_regime_features(pca_all)
    return fit_gmm(features, cfg)


# --------------------------------------------------------------------------- #
# 1. Config layer — locks the keys the rest of the system reads               #
# --------------------------------------------------------------------------- #

class TestConfigCharacterization:
    """Lock the structural shape of config/funds.yaml so refactors don't drop keys."""

    REQUIRED_TOP_KEYS = {
        "alerts", "briefing", "countries", "excluded_series", "fred",
        "lc_fund", "macro_events", "pca", "pipeline", "portfolios",
        "regime", "var",
    }

    def test_top_level_keys(self, cfg):
        assert set(cfg.keys()) >= self.REQUIRED_TOP_KEYS

    def test_lc_fund_shape(self, cfg):
        assert cfg["lc_fund"]["effective_duration"] == pytest.approx(5.22)
        assert cfg["lc_fund"]["benchmark_maturity"] == "5Y"
        assert set(cfg["lc_fund"]["weights"]) == {
            "Brazil", "Mexico", "South Africa", "Poland",
        }

    def test_portfolio_definitions(self, cfg):
        ids = {p["id"] for p in cfg["portfolios"]}
        assert ids == {"portfolio_1", "portfolio_2"}
        for p in cfg["portfolios"]:
            assert {"name", "id", "weights",
                    "effective_duration", "benchmark_maturity"} <= set(p)

    def test_var_confidence_levels(self, cfg):
        assert cfg["var"]["confidence_levels"] == [0.95, 0.99]

    def test_pca_n_components(self, cfg):
        assert cfg["pca"]["n_components"] == 3

    def test_regime_labels_have_four_states(self, cfg):
        # 0..3 → Normal / EM Crisis / Rate Shock / Risk-On
        assert set(cfg["regime"]["labels"].keys()) == {0, 1, 2, 3}

    def test_alert_thresholds_pinned(self, cfg):
        a = cfg["alerts"]
        assert a["pc_zscore_threshold"] == 3.0
        assert a["curve_zscore_threshold"] == 3.0
        assert a["vol_trailing_window"] == 252
        assert a["curve_rolling_window"] == 60


# --------------------------------------------------------------------------- #
# 2. Data-ingestion layer — raw CSV loading + portfolio P&L proxy             #
# --------------------------------------------------------------------------- #

class TestDataIngestionCharacterization:
    """Snapshot the outputs of the ingestion functions on real raw CSVs."""

    def test_brazil_yield_changes_shape(self):
        dy = load_yield_changes("Brazil", data_dir=DATA_DIR).loc[:SLICE_END]
        assert dy.shape == (2617, 5)
        assert list(dy.columns) == ["2Y", "3Y", "5Y", "8Y", "10Y"]

    def test_brazil_5y_summary_stats(self):
        dy = load_yield_changes("Brazil", data_dir=DATA_DIR).loc[:SLICE_END]
        # Snapshot computed from a single deterministic pipeline run.
        assert float(dy["5Y"].mean()) == pytest.approx(-0.00218112, abs=1e-6)
        assert float(dy["5Y"].std()) == pytest.approx(0.14866129, abs=1e-6)

    def test_universe_loaded(self, change_dfs):
        assert set(change_dfs) == {
            "Brazil", "Mexico", "South Africa", "Poland",
            "Colombia", "Hungary", "Romania",
        }

    def test_lc_portfolio_pnl_shape_and_stats(self, change_dfs, cfg):
        pdef = next(p for p in cfg["portfolios"] if p["id"] == "portfolio_2")
        pnl, proxy_dy = build_portfolio_pnl_from_def(change_dfs, pdef)

        assert len(pnl) == 1422
        assert float(pnl.mean()) == pytest.approx(0.000101858, abs=1e-7)
        assert float(pnl.std()) == pytest.approx(0.003024849, abs=1e-7)

        # Proxy_dy must use all six LC-funded countries (Colombia is excluded
        # from portfolio_2 weights; HC fund uses the 7th).
        assert set(proxy_dy.columns) == set(pdef["weights"])

    def test_hc_portfolio_pnl_uses_correct_universe(self, change_dfs, cfg):
        pdef = next(p for p in cfg["portfolios"] if p["id"] == "portfolio_1")
        pnl, proxy_dy = build_portfolio_pnl_from_def(change_dfs, pdef)
        assert len(pnl) > 0
        # HC fund weights include Colombia, LC fund does not.
        assert "Colombia" in proxy_dy.columns


# --------------------------------------------------------------------------- #
# 3. Models/inference layer — PCA + GMM regime engine                         #
# --------------------------------------------------------------------------- #

class TestPCACharacterization:
    """Lock per-country PCA decomposition output."""

    def test_brazil_pca_explained_variance(self, change_dfs):
        pca = run_pca(change_dfs["Brazil"], n_components=3)
        evr = pca["explained_var"]
        assert evr[0] == pytest.approx(0.75839302, abs=1e-5)
        assert evr[1] == pytest.approx(0.11482089, abs=1e-5)
        assert evr[2] == pytest.approx(0.06236713, abs=1e-5)

    def test_pca_score_columns_are_labelled(self, change_dfs):
        pca = run_pca(change_dfs["Brazil"], n_components=3)
        assert list(pca["scores"].columns) == [
            "PC1 (level)", "PC2 (slope)", "PC3 (curvature)",
        ]

    def test_pca_loadings_have_unit_norm(self, pca_all):
        for country, res in pca_all.items():
            for i in range(res["n_components"]):
                norm = float(np.linalg.norm(res["loadings"][i]))
                assert norm == pytest.approx(1.0, abs=1e-6), country

    def test_brazil_pc1_is_level_factor(self, change_dfs):
        # PC1 on yield-change PCA for a sovereign curve must load positively
        # (or all-negatively, sign is arbitrary) on every maturity.
        pca = run_pca(change_dfs["Brazil"], n_components=3)
        pc1 = pca["loadings"][0]
        same_sign = np.all(pc1 > 0) or np.all(pc1 < 0)
        assert same_sign, f"PC1 should be a level factor; loadings={pc1}"


class TestRegimeCharacterization:
    """Lock the structural shape of regime detection — not cluster IDs."""

    def test_regime_features_shape(self, pca_all):
        features = build_regime_features(pca_all)
        assert list(features.columns) == ["avg_level", "dispersion", "real_vol"]
        assert len(features) == 1262

    def test_fit_gmm_adds_expected_columns(self, regime_df):
        for col in ("regime", "regime_proba", "regime_label"):
            assert col in regime_df.columns

    def test_regime_count_matches_config(self, regime_df, cfg):
        # BIC selects 2..max_components inclusive; current snapshot fits 4.
        n_regimes = regime_df["regime"].nunique()
        assert 2 <= n_regimes <= cfg["regime"]["gmm"]["max_components"]
        assert n_regimes == 4

    def test_regime_labels_are_from_config(self, regime_df, cfg):
        valid_labels = set(cfg["regime"]["labels"].values()) | {"Unknown"}
        assert set(regime_df["regime_label"].unique()) <= valid_labels

    def test_regime_proba_in_unit_interval(self, regime_df):
        assert regime_df["regime_proba"].between(0.0, 1.0).all()


# --------------------------------------------------------------------------- #
# 4. Alert engine — payload structure on a known date                         #
# --------------------------------------------------------------------------- #

class TestAlertCharacterization:
    """Lock the alert payload structure produced by ``generate_alerts``."""

    REQUIRED_KEYS = {
        "date", "regime", "regime_confidence",
        "n_alerts", "max_severity", "alerts",
    }
    VALID_ALERT_TYPES = {
        "regime_shift", "pc_zscore_breach",
        "vol_spike", "country_curve_outlier",
    }
    VALID_SEVERITIES = {"low", "medium", "high"}

    def test_payload_keys(self, regime_df, pca_all, change_dfs, cfg):
        date = regime_df.index[400]
        payload = generate_alerts(date, regime_df, pca_all,
                                  change_dfs, cfg, cfg.get("macro_events"))
        assert payload is not None
        assert set(payload.keys()) >= self.REQUIRED_KEYS
        assert payload["date"] == str(date.date())

    def test_invalid_date_returns_none(self, regime_df, pca_all,
                                       change_dfs, cfg):
        assert generate_alerts(
            pd.Timestamp("1900-01-01"), regime_df, pca_all,
            change_dfs, cfg, None,
        ) is None

    def test_alert_entries_are_well_formed(self, regime_df, pca_all,
                                           change_dfs, cfg):
        # Scan a slice and assert every alert entry conforms.
        for date in regime_df.index[300:500]:
            payload = generate_alerts(date, regime_df, pca_all,
                                      change_dfs, cfg, cfg.get("macro_events"))
            if payload is None or payload["n_alerts"] == 0:
                continue
            for alert in payload["alerts"]:
                assert alert["type"] in self.VALID_ALERT_TYPES
                assert alert["severity"] in self.VALID_SEVERITIES
                assert "detail" in alert
                if alert["type"] in ("pc_zscore_breach",
                                     "country_curve_outlier"):
                    assert "country" in alert

    def test_max_severity_matches_alerts_list(self, regime_df, pca_all,
                                              change_dfs, cfg):
        rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
        for date in regime_df.index[300:500]:
            payload = generate_alerts(date, regime_df, pca_all,
                                      change_dfs, cfg, cfg.get("macro_events"))
            if not payload or payload["n_alerts"] == 0:
                continue
            top = max(payload["alerts"], key=lambda a: rank[a["severity"]])
            assert payload["max_severity"] == top["severity"]


# --------------------------------------------------------------------------- #
# 5. Risk-free helper — pure math, no network                                 #
# --------------------------------------------------------------------------- #

class TestRiskFreeCharacterization:

    def test_daily_rf_from_annual_zero(self):
        assert daily_rf_from_annual(pd.Series([0.0])).iloc[0] == 0.0

    def test_daily_rf_from_annual_known_values(self):
        out = daily_rf_from_annual(pd.Series([5.0, 10.0])).tolist()
        # (1 + r/100) ** (1/252) - 1
        assert out[0] == pytest.approx(0.00019363, abs=1e-7)
        assert out[1] == pytest.approx(0.00037829, abs=1e-7)


# --------------------------------------------------------------------------- #
# 6. LLM-orchestration surface — chatbot message routing                      #
# --------------------------------------------------------------------------- #

class TestChatbotRoutingCharacterization:
    """The single orchestration boundary we have today is message mapping."""

    def test_empty_history_still_yields_system_prompt(self):
        import chatbot
        from langchain_core.messages import SystemMessage

        msgs = chatbot._to_lc_messages([])
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == chatbot.SYSTEM_PROMPT

    def test_unknown_role_defaults_to_human(self):
        import chatbot
        from langchain_core.messages import HumanMessage

        msgs = chatbot._to_lc_messages(
            [{"role": "tool", "content": "ignored-role"}]
        )
        # System prompt + 1 human-mapped message
        assert len(msgs) == 2
        assert isinstance(msgs[1], HumanMessage)
        assert msgs[1].content == "ignored-role"

    def test_system_prompt_dashboard_page_grounding(self):
        import chatbot
        # The chatbot is supposed to "route" users to the right page.
        for page in ("VaR Engine", "PCA & Regime",
                     "Alert History", "Daily Briefings"):
            assert page in chatbot.SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# 7. End-to-end smoke characterization                                        #
# --------------------------------------------------------------------------- #

def test_end_to_end_smoke(change_dfs, pca_all, regime_df, cfg):
    """
    Smoke test that mirrors the notebook pipeline order:
    config → all-country changes → PCA → regime features → alerts.

    If any boundary changes shape under a refactor, this test fails fast.
    """
    # 1. Universe size
    assert len(change_dfs) == 7

    # 2. PCA result per country
    assert set(pca_all) == set(change_dfs)
    for country, res in pca_all.items():
        assert res["scores"].shape[1] == 3
        # explained variance is monotone decreasing
        evr = res["explained_var"]
        assert all(evr[i] >= evr[i + 1] for i in range(len(evr) - 1)), country

    # 3. Regime DataFrame inherits feature dates and adds 3 cols.
    assert {"regime", "regime_proba", "regime_label"} <= set(regime_df.columns)
    assert regime_df.index.max() <= SLICE_END

    # 4. At least one alert day must exist in the recent slice.
    found_alert = False
    for date in regime_df.index[-200:]:
        payload = generate_alerts(date, regime_df, pca_all,
                                  change_dfs, cfg, cfg.get("macro_events"))
        if payload and payload["n_alerts"] > 0:
            found_alert = True
            break
    assert found_alert, "No alerts in the last 200 days — pipeline likely broken"
