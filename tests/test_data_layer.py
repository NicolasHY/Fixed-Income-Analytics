"""
Characterization tests for the Data layer (``src/data/``).

Locks the output of ``build_daily_payload`` and the round-trip of
``load_briefings`` / ``save_briefings``. The notebook still holds its own
copy of ``build_daily_payload`` (will be removed in the Orchestration
step); these tests give us a regression net to detect any drift between
the two copies until then.

Real-data fixtures match those in ``tests/test_characterization.py``: raw
CSVs are sliced to ``<= 2025-12-31`` so snapshots stay stable as new data
arrives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import (
    DEFAULT_BRIEFING_CACHE,
    build_daily_payload,
    load_briefings,
    save_briefings,
)
from src.data_loader import (
    build_portfolio_pnl_from_def,
    load_all_countries_combined,
    load_config,
)
from src.pca_regime import (
    build_regime_features,
    fit_gmm,
    run_alert_scan,
    run_pca_all_countries,
)


SLICE_END = pd.Timestamp("2025-12-31")
DATA_DIR = "data/raw"
STRESS_DATE = "2022-09-23"  # In every index; one of the briefing showcase dates.


# --------------------------------------------------------------------------- #
# Shared real-data fixtures (module-scoped to amortise the pipeline cost)     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def change_dfs(cfg):
    raw = load_all_countries_combined(cfg, data_dir=DATA_DIR)
    return {c: df.loc[:SLICE_END] for c, df in raw.items()}


@pytest.fixture(scope="module")
def lc_pnl(cfg, change_dfs):
    pdef = next(p for p in cfg["portfolios"] if p["id"] == "portfolio_2")
    pnl, _ = build_portfolio_pnl_from_def(change_dfs, pdef)
    return pnl


@pytest.fixture(scope="module")
def pca_all(change_dfs):
    return run_pca_all_countries(change_dfs, n_components=3)


@pytest.fixture(scope="module")
def regime_df(pca_all, cfg):
    return fit_gmm(build_regime_features(pca_all), cfg)


@pytest.fixture(scope="module")
def all_alerts(regime_df, pca_all, change_dfs, cfg):
    return run_alert_scan(regime_df, pca_all, change_dfs, cfg,
                          cfg.get("macro_events"))


@pytest.fixture(scope="module")
def payload(regime_df, pca_all, change_dfs, lc_pnl, all_alerts):
    """Payload for the stress date with stub VaR levels."""
    return build_daily_payload(
        STRESS_DATE, regime_df, pca_all, change_dfs,
        lc_pnl, all_alerts, var_95=0.005, var_99=0.008,
    )


# --------------------------------------------------------------------------- #
# build_daily_payload — structure                                             #
# --------------------------------------------------------------------------- #

class TestPayloadStructure:

    def test_date_round_trips(self, payload):
        assert payload["date"] == STRESS_DATE

    def test_required_top_level_keys(self, payload):
        # date, curve_moves_bps, pc_scores, alerts always present.
        # regime + portfolio present only if date in those indexes.
        assert set(payload.keys()) >= {
            "date", "curve_moves_bps", "pc_scores", "alerts",
            "regime", "portfolio",
        }

    def test_regime_block_shape(self, payload, cfg):
        r = payload["regime"]
        assert set(r.keys()) == {
            "label", "confidence", "avg_level_shock",
            "dispersion", "realized_vol",
        }
        # label must come from the configured regime set.
        valid_labels = set(cfg["regime"]["labels"].values()) | {"Unknown"}
        assert r["label"] in valid_labels
        assert 0.0 <= float(r["confidence"]) <= 1.0

    def test_curve_moves_cover_full_universe(self, payload, change_dfs):
        # 2022-09-23 is a business day with data for every loaded country.
        assert set(payload["curve_moves_bps"]) == set(change_dfs)
        for country, moves in payload["curve_moves_bps"].items():
            assert moves, f"{country} has no maturities"
            assert all(isinstance(v, float) for v in moves.values())

    def test_pc_scores_block_per_country(self, payload, change_dfs):
        assert set(payload["pc_scores"]) == set(change_dfs)
        for country, scores in payload["pc_scores"].items():
            assert set(scores.keys()) == {
                "PC1_level", "PC2_slope", "PC3_curvature",
            }

    def test_portfolio_block_shape(self, payload):
        p = payload["portfolio"]
        assert set(p.keys()) == {
            "daily_pnl_pct", "var_95_mc_tcopula", "var_99_mc_tcopula",
            "var_breach_95", "var_breach_99",
        }

    def test_alerts_is_a_list(self, payload):
        assert isinstance(payload["alerts"], list)


# --------------------------------------------------------------------------- #
# build_daily_payload — numerical snapshots                                   #
# --------------------------------------------------------------------------- #

class TestPayloadNumericalSnapshots:
    """Snapshot values computed from a single deterministic pipeline run."""

    def test_brazil_5y_curve_move_bps(self, payload):
        assert payload["curve_moves_bps"]["Brazil"]["5Y"] == pytest.approx(
            0.13, abs=1e-6,
        )

    def test_brazil_pc1_score(self, payload):
        assert payload["pc_scores"]["Brazil"]["PC1_level"] == pytest.approx(
            1.57, abs=1e-6,
        )

    def test_daily_pnl_pct(self, payload):
        # P&L for 2022-09-23 on the LC fund ≈ -0.6091% (rounded to 4dp).
        assert float(payload["portfolio"]["daily_pnl_pct"]) == pytest.approx(
            -0.6091, abs=1e-4,
        )

    def test_var_levels_pipe_through(self, payload):
        # var_95=0.005 → 0.5%, var_99=0.008 → 0.8%
        assert payload["portfolio"]["var_95_mc_tcopula"] == pytest.approx(0.5)
        assert payload["portfolio"]["var_99_mc_tcopula"] == pytest.approx(0.8)

    def test_var_breach_logic(self, payload):
        # pnl ≈ -0.006091, var_95=0.005 → breach;  var_99=0.008 → no breach.
        assert bool(payload["portfolio"]["var_breach_95"]) is True
        assert bool(payload["portfolio"]["var_breach_99"]) is False


# --------------------------------------------------------------------------- #
# build_daily_payload — edge cases                                            #
# --------------------------------------------------------------------------- #

class TestPayloadEdgeCases:

    def test_unknown_date_returns_minimal_payload(self, regime_df, pca_all,
                                                   change_dfs, lc_pnl,
                                                   all_alerts):
        payload = build_daily_payload(
            "1900-01-01", regime_df, pca_all, change_dfs,
            lc_pnl, all_alerts, var_95=0.005, var_99=0.008,
        )
        assert payload["date"] == "1900-01-01"
        # Date not in any index → no regime, no portfolio block.
        assert "regime" not in payload
        assert "portfolio" not in payload
        # Curve moves / PC scores are present but empty.
        assert payload["curve_moves_bps"] == {}
        assert payload["pc_scores"] == {}
        # Alerts default to an empty list when date_str not in cache.
        assert payload["alerts"] == []

    def test_accepts_timestamp_input(self, regime_df, pca_all, change_dfs,
                                     lc_pnl, all_alerts):
        ts = pd.Timestamp(STRESS_DATE)
        a = build_daily_payload(
            ts, regime_df, pca_all, change_dfs,
            lc_pnl, all_alerts, var_95=0.005, var_99=0.008,
        )
        b = build_daily_payload(
            STRESS_DATE, regime_df, pca_all, change_dfs,
            lc_pnl, all_alerts, var_95=0.005, var_99=0.008,
        )
        assert a == b

    def test_payload_is_json_serialisable(self, payload):
        # The notebook serialises with default=str to coerce numpy bool/float;
        # the dict itself must at minimum survive default=str without errors.
        json.dumps(payload, default=str)


# --------------------------------------------------------------------------- #
# briefing_store — round-trip                                                 #
# --------------------------------------------------------------------------- #

class TestBriefingStore:

    def test_default_path_points_at_real_artifact(self):
        # Defensive: matches the path written by main.ipynb cell 55.
        assert str(DEFAULT_BRIEFING_CACHE).replace("\\", "/") == (
            "data/output/sample_briefings.json"
        )

    def test_load_returns_empty_dict_when_missing(self, tmp_path):
        assert load_briefings(tmp_path / "nope.json") == {}

    def test_save_then_load_round_trip(self, tmp_path):
        path = tmp_path / "subdir" / "briefings.json"
        briefings = {
            "2022-09-23": "Headline: gilt crisis spills into EM.",
            "2023-06-15": "Regime: Normal. No alerts triggered.",
        }
        save_briefings(briefings, path)
        assert path.exists()
        assert load_briefings(path) == briefings

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deeply" / "nested" / "out.json"
        save_briefings({"2024-01-01": "x"}, path)
        assert path.exists()

    def test_load_reads_existing_real_cache(self):
        # If the project's real cache exists, the loader must parse it.
        if not DEFAULT_BRIEFING_CACHE.exists():
            pytest.skip("No real briefing cache on disk")
        briefings = load_briefings()
        assert isinstance(briefings, dict)
        # Keys should look like ISO dates; values are non-empty strings.
        for k, v in briefings.items():
            assert isinstance(k, str) and len(k) == 10
            assert isinstance(v, str) and v


# --------------------------------------------------------------------------- #
# Parity with the notebook copy                                               #
# --------------------------------------------------------------------------- #

def test_module_payload_matches_inline_implementation(
    regime_df, pca_all, change_dfs, lc_pnl, all_alerts,
):
    """The extracted module must produce a byte-identical payload to a
    fresh, inline re-implementation of the notebook's logic — this is the
    regression net protecting the two copies until orchestration removes
    the notebook duplicate.
    """
    # Inline copy of cell 53's body (no shared imports) — keep in sync if
    # the notebook copy is intentionally changed.
    def _notebook_inline(date, regime_features, pca_results, change_dfs,
                         portfolio_pnl, all_alerts, VaR_95, VaR_99):
        date = pd.Timestamp(date)
        payload = {"date": str(date.date())}

        if date in regime_features.index:
            row = regime_features.loc[date]
            payload["regime"] = {
                "label": row["regime_label"],
                "confidence": round(row["regime_proba"], 3),
                "avg_level_shock": round(row["avg_level"], 3),
                "dispersion": round(row["dispersion"], 3),
                "realized_vol": round(row["real_vol"], 3),
            }

        curve_moves = {}
        for country, dy in change_dfs.items():
            if date in dy.index:
                moves = dy.loc[date].to_dict()
                curve_moves[country] = {
                    mat: round(v, 3) for mat, v in moves.items()
                    if not np.isnan(v)
                }
        payload["curve_moves_bps"] = curve_moves

        pc_scores = {}
        for country, res in pca_results.items():
            if date in res["scores"].index:
                scores = res["scores"].loc[date]
                pc_scores[country] = {
                    "PC1_level": round(scores.iloc[0], 2),
                    "PC2_slope": round(scores.iloc[1], 2),
                    "PC3_curvature": round(scores.iloc[2], 2),
                }
        payload["pc_scores"] = pc_scores

        if date in portfolio_pnl.index:
            pnl = portfolio_pnl.loc[date]
            payload["portfolio"] = {
                "daily_pnl_pct": round(pnl * 100, 4),
                "var_95_mc_tcopula": round(VaR_95 * 100, 4),
                "var_99_mc_tcopula": round(VaR_99 * 100, 4),
                "var_breach_95": pnl < -VaR_95,
                "var_breach_99": pnl < -VaR_99,
            }

        date_str = str(date.date())
        if date_str in all_alerts:
            payload["alerts"] = all_alerts[date_str]["alerts"]
        else:
            payload["alerts"] = []

        return payload

    args = (STRESS_DATE, regime_df, pca_all, change_dfs,
            lc_pnl, all_alerts, 0.005, 0.008)
    a = build_daily_payload(*args)
    b = _notebook_inline(*args)

    assert json.dumps(a, default=str) == json.dumps(b, default=str)
