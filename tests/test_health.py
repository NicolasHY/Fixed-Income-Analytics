"""
Characterization tests for ``src/orchestration/health.py``.

Three surfaces:

* :func:`run_pipeline_step` — entry shape on success/failure, timing field.
* :func:`build_health_check` — 5 cards, status logic for each, deterministic
  via the ``now`` override.
* :func:`write_pipeline_log` / :func:`write_health_check` — atomic disk
  round-trip into ``tmp_path``.
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

from src.orchestration.health import (
    HEALTH_CHECK_PATH,
    PIPELINE_LOG_PATH,
    build_health_check,
    run_pipeline_step,
    write_health_check,
    write_pipeline_log,
)


# --------------------------------------------------------------------------- #
# run_pipeline_step                                                           #
# --------------------------------------------------------------------------- #

class TestRunPipelineStep:

    def test_success_entry_shape(self):
        result, entry = run_pipeline_step("noop", lambda: {"a": 1})
        assert result == {"a": 1}
        assert entry["step"] == "noop"
        assert entry["status"] == "success"
        assert entry["error"] is None
        assert entry["runtime_seconds"] is not None
        assert entry["output_shape"] == 1  # len({"a": 1})

    def test_output_shape_for_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        _, entry = run_pipeline_step("df", lambda: df)
        assert entry["output_shape"] == [3, 1]

    def test_output_shape_for_list(self):
        _, entry = run_pipeline_step("lst", lambda: [1, 2, 3, 4])
        assert entry["output_shape"] == 4

    def test_failure_captures_error(self):
        def boom():
            raise ValueError("kaboom")
        result, entry = run_pipeline_step("boom", boom)
        assert result is None
        assert entry["status"] == "failure"
        assert entry["error"] == "ValueError: kaboom"
        assert entry["runtime_seconds"] is not None

    def test_passes_args_and_kwargs(self):
        result, _ = run_pipeline_step(
            "add", lambda a, b, c=0: a + b + c, 1, 2, c=3,
        )
        assert result == 6


# --------------------------------------------------------------------------- #
# build_health_check                                                          #
# --------------------------------------------------------------------------- #

def _ok_log(n: int = 5) -> list[dict]:
    return [{
        "step": f"step_{i}", "status": "success",
        "runtime_seconds": 1.0, "output_shape": 1,
        "error": None, "timestamp": "fake",
    } for i in range(n)]


def _stable_pnl(end="2025-12-31", days=300, sigma=0.005) -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end=end, periods=days)
    return pd.Series(rng.normal(0, sigma, days), index=idx)


def _flat_regimes(days=20, value=0) -> pd.DataFrame:
    return pd.DataFrame({"regime": [value] * days})


class TestBuildHealthCheck:

    def test_returns_five_cards(self):
        checks = build_health_check(
            _ok_log(), _stable_pnl(), _flat_regimes(),
            var_95=0.01, now=pd.Timestamp("2026-01-01"),
        )
        assert len(checks) == 5
        for card in checks:
            assert set(card.keys()) == {"check", "status", "detail"}
            assert card["status"] in {"GREEN", "YELLOW", "RED"}

    def test_pipeline_green_when_all_steps_succeed(self):
        checks = build_health_check(
            _ok_log(), _stable_pnl(), _flat_regimes(),
            var_95=0.01, now=pd.Timestamp("2026-01-01"),
        )
        assert checks[0]["status"] == "GREEN"
        assert "All 5 steps OK" in checks[0]["detail"]

    def test_pipeline_red_when_a_step_fails(self):
        log = _ok_log(3) + [{
            "step": "bad", "status": "failure",
            "runtime_seconds": 0.1, "output_shape": None,
            "error": "X", "timestamp": "fake",
        }]
        checks = build_health_check(
            log, _stable_pnl(), _flat_regimes(),
            var_95=0.01, now=pd.Timestamp("2026-01-01"),
        )
        assert checks[0]["status"] == "RED"
        assert "1 failed" in checks[0]["detail"]

    def test_freshness_thresholds(self):
        idx = pd.bdate_range(end="2026-01-01", periods=10)
        pnl = pd.Series(0.001, index=idx)
        # 3 days stale → GREEN
        c = build_health_check(_ok_log(), pnl, _flat_regimes(),
                               var_95=0.01, now=pd.Timestamp("2026-01-04"))
        assert c[1]["status"] == "GREEN"
        # 20 days stale → YELLOW
        c = build_health_check(_ok_log(), pnl, _flat_regimes(),
                               var_95=0.01, now=pd.Timestamp("2026-01-21"))
        assert c[1]["status"] == "YELLOW"
        # 60 days stale → RED
        c = build_health_check(_ok_log(), pnl, _flat_regimes(),
                               var_95=0.01, now=pd.Timestamp("2026-03-02"))
        assert c[1]["status"] == "RED"

    def test_runtime_yellow_when_slow(self):
        log = _ok_log()
        log[2]["runtime_seconds"] = 45.0
        checks = build_health_check(
            log, _stable_pnl(), _flat_regimes(),
            var_95=0.01, now=pd.Timestamp("2026-01-01"),
        )
        assert checks[2]["status"] == "YELLOW"
        assert "45" in checks[2]["detail"]

    def test_var_breach_card_yellow_outside_tolerance(self):
        # PnL much larger than var_95 → breach rate near 50% → YELLOW
        rng = np.random.default_rng(1)
        pnl = pd.Series(rng.normal(0, 0.05, 300),
                        index=pd.bdate_range(end="2026-01-01", periods=300))
        checks = build_health_check(
            _ok_log(), pnl, _flat_regimes(),
            var_95=0.001, now=pd.Timestamp("2026-01-02"),
        )
        assert checks[3]["status"] == "YELLOW"

    def test_regime_stability_yellow_when_too_many_transitions(self):
        # 10 alternating regimes in last 20 days = 9 transitions → YELLOW
        regimes = pd.DataFrame({"regime": [0, 1] * 10})
        checks = build_health_check(
            _ok_log(), _stable_pnl(), regimes,
            var_95=0.01, now=pd.Timestamp("2026-01-01"),
        )
        assert checks[4]["status"] == "YELLOW"


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #

class TestPersistence:

    def test_pipeline_log_round_trip(self, tmp_path):
        log = _ok_log()
        path = tmp_path / "nested" / "pipeline_log.json"
        write_pipeline_log(log, path)
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8")) == log

    def test_health_check_round_trip(self, tmp_path):
        checks = build_health_check(
            _ok_log(), _stable_pnl(), _flat_regimes(),
            var_95=0.01, now=pd.Timestamp("2026-01-01"),
        )
        path = tmp_path / "deep" / "health.json"
        write_health_check(checks, path)
        assert json.loads(path.read_text(encoding="utf-8")) == checks

    def test_default_paths_match_dashboard(self):
        # Defensive: the dashboard reads these exact paths.
        assert str(PIPELINE_LOG_PATH).replace("\\", "/") == (
            "data/output/pipeline_log.json"
        )
        assert str(HEALTH_CHECK_PATH).replace("\\", "/") == (
            "data/output/health_check.json"
        )
