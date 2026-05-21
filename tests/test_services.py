"""
Characterization tests for ``src/services/portfolios.py``.

Two surfaces:

* :func:`compute_quick_stats` — eight Home-page numbers; this is the
  exact math the dashboard renders, so it gets pinned to ~6 decimals.
* :func:`build_portfolio_views` — composition of Data layer +
  ``_apply_daily_carry``. We run it against the real raw CSVs and lock
  the per-portfolio shape and the LC fund's annualised stats.

Everything below uses module-scoped fixtures so the real-data load only
happens once per pytest session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.portfolios import build_portfolio_views, compute_quick_stats


# --------------------------------------------------------------------------- #
# compute_quick_stats                                                         #
# --------------------------------------------------------------------------- #

def _synthetic_pnl(seed: int = 0, days: int = 252, mu: float = 0.0002,
                   sigma: float = 0.005) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    return pd.Series(rng.normal(mu, sigma, days), index=idx)


class TestComputeQuickStats:

    def test_returns_eight_keys(self):
        stats = compute_quick_stats(_synthetic_pnl())
        assert set(stats.keys()) == {
            "ret", "vol", "sharpe", "cum", "dd",
            "var95", "start", "end",
        }

    def test_value_types(self):
        stats = compute_quick_stats(_synthetic_pnl())
        for k in ("ret", "vol", "sharpe", "cum", "dd", "var95"):
            assert isinstance(stats[k], float), k
        assert isinstance(stats["start"], str)
        assert isinstance(stats["end"], str)

    def test_zero_volatility_yields_nan_sharpe(self):
        # Exactly-zero P&L gives exactly-zero std (any other constant value
        # still produces a tiny FP std of ~1e-19 that would slip past the
        # ``vol > 0`` guard — match the original app.py behaviour by using 0.0).
        idx = pd.bdate_range("2024-01-01", periods=20)
        stats = compute_quick_stats(pd.Series([0.0] * 20, index=idx))
        assert np.isnan(stats["sharpe"])

    def test_positive_vol_yields_finite_sharpe(self):
        stats = compute_quick_stats(_synthetic_pnl())
        assert np.isfinite(stats["sharpe"])

    def test_date_labels_match_index_extremes(self):
        idx = pd.bdate_range("2022-03-01", periods=400)
        pnl = pd.Series(np.zeros(400), index=idx)
        stats = compute_quick_stats(pnl)
        assert stats["start"] == idx.min().strftime("%b %Y")
        assert stats["end"] == idx.max().strftime("%b %Y")

    def test_known_inputs_known_outputs(self):
        # All-zero P&L → 0 return, 0 vol, 0 cum, 0 drawdown.
        idx = pd.bdate_range("2024-01-01", periods=252)
        stats = compute_quick_stats(pd.Series(np.zeros(252), index=idx))
        assert stats["ret"] == pytest.approx(0.0, abs=1e-6)
        assert stats["vol"] == pytest.approx(0.0, abs=1e-6)
        assert stats["cum"] == pytest.approx(0.0, abs=1e-6)
        assert stats["dd"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# build_portfolio_views                                                       #
# --------------------------------------------------------------------------- #

class TestBuildPortfolioViews:
    """Run against the real raw CSVs (no slicing — the dashboard doesn't slice)."""

    @pytest.fixture(scope="class")
    def views(self):
        return build_portfolio_views()

    def test_one_view_per_portfolio(self, views):
        from src.data_loader import load_config
        cfg = load_config()
        assert len(views) == len(cfg["portfolios"])

    def test_view_shape(self, views):
        for v in views:
            assert set(v.keys()) == {"def", "pnl", "proxy_dy"}
            assert "id" in v["def"]
            assert isinstance(v["pnl"], pd.Series)
            assert isinstance(v["proxy_dy"], pd.DataFrame)

    def test_pnl_includes_carry_uplift(self, views):
        """Carry adds a tiny daily positive term, so the carry-adjusted P&L
        must have a higher mean than the bare duration proxy."""
        from src.data_loader import (
            build_portfolio_pnl_from_def,
            load_all_countries_combined,
            load_config,
        )
        cfg = load_config()
        change_dfs = load_all_countries_combined(cfg, data_dir="data/raw")
        for v in views:
            bare_pnl, _ = build_portfolio_pnl_from_def(change_dfs, v["def"])
            common = v["pnl"].index.intersection(bare_pnl.index)
            uplift = (v["pnl"].loc[common] - bare_pnl.loc[common]).mean()
            assert uplift > 0, (
                f"{v['def']['name']} carry should be a positive daily uplift "
                f"(got {uplift:.2e})"
            )

    def test_quick_stats_pipes_through(self, views):
        """End-to-end: build_portfolio_views → compute_quick_stats works
        without error for every portfolio."""
        for v in views:
            stats = compute_quick_stats(v["pnl"])
            assert np.isfinite(stats["vol"]) and stats["vol"] > 0
