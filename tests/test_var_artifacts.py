"""
Characterization tests for ``src/data/var_artifacts.py``.

Three concerns:

* Every loader returns ``None`` when its required files are missing.
* Every loader produces the dict shape the Streamlit dashboard expects
  when the files DO exist (driven by synthetic fixtures in ``tmp_path``).
* The default output directory points where the dashboard reads from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.var_artifacts import (
    DEFAULT_OUTPUT_DIR,
    load_alert_history,
    load_country_outputs,
    load_decomposition,
    load_health_check,
    load_multi_nu,
    load_pipeline_log,
    load_stress_data,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Missing-file behaviour                                                      #
# --------------------------------------------------------------------------- #

class TestMissingFilesReturnNone:

    def test_stress_data(self, tmp_path):
        assert load_stress_data(tmp_path) is None

    def test_multi_nu(self, tmp_path):
        assert load_multi_nu(tmp_path) is None

    def test_decomposition(self, tmp_path):
        assert load_decomposition(tmp_path) is None

    def test_pipeline_log(self, tmp_path):
        assert load_pipeline_log(tmp_path) is None

    def test_health_check(self, tmp_path):
        assert load_health_check(tmp_path) is None

    def test_alert_history(self, tmp_path):
        assert load_alert_history(tmp_path) is None


# --------------------------------------------------------------------------- #
# Happy-path shape (synthetic artifacts in tmp_path)                          #
# --------------------------------------------------------------------------- #

class TestLoadStressData:

    def test_returns_pnl_windows_summary(self, tmp_path):
        # pnl
        idx = pd.bdate_range("2024-01-01", periods=10)
        _write_csv(tmp_path / "var_portfolio_pnl.csv",
                   pd.DataFrame({"pnl": [0.001] * 10}, index=idx))
        # windows
        _write_json(tmp_path / "var_stress_windows.json", {
            "primary_stress": "COVID",
            "windows": {"COVID": {"start": "2020-02-19", "end": "2020-05-15",
                                  "VaR_95": 0.01}},
            "reference": {"hist_full_VaR_95": 0.005,
                          "parametric_normal_VaR_95": 0.004},
        })
        # summary
        _write_csv(tmp_path / "var_stressed_summary.csv",
                   pd.DataFrame({"Historical 3Y": [0.005],
                                 "Stressed COVID": [0.012]},
                                index=["VaR 95%"]))

        out = load_stress_data(tmp_path)
        assert set(out) == {"pnl", "windows", "summary"}
        assert isinstance(out["pnl"], pd.Series)
        assert out["windows"]["primary_stress"] == "COVID"
        assert isinstance(out["summary"], pd.DataFrame)


class TestLoadMultiNu:

    def test_returns_table_and_nu_fit(self, tmp_path):
        _write_csv(tmp_path / "var_multi_nu_table.csv",
                   pd.DataFrame({"VaR 95%": [0.005, 0.0045],
                                 "VaR 99%": [0.010, 0.0090]},
                                index=[4, "inf"]))
        _write_json(tmp_path / "var_multi_nu_fit.json", {"nu_fit": 4.7})

        out = load_multi_nu(tmp_path)
        assert set(out) == {"table", "nu_fit"}
        assert out["nu_fit"] == pytest.approx(4.7)
        assert isinstance(out["table"], pd.DataFrame)


class TestLoadDecomposition:

    def test_returns_scalars_and_betas(self, tmp_path):
        _write_json(tmp_path / "var_decomposition.json", {
            "pct_systematic": 88.5, "pct_idiosyncratic": 11.5,
        })
        _write_csv(tmp_path / "var_decomposition_betas.csv",
                   pd.DataFrame({"PC1": [0.5, 0.4], "PC2": [0.1, 0.2]},
                                index=["Brazil", "Mexico"]))

        out = load_decomposition(tmp_path)
        assert set(out) == {"scalars", "betas"}
        assert out["scalars"]["pct_systematic"] == pytest.approx(88.5)
        assert isinstance(out["betas"], pd.DataFrame)


class TestLoadPipelineLog:

    def test_round_trip(self, tmp_path):
        log = [{"step": "s1", "status": "success", "runtime_seconds": 1.2}]
        _write_json(tmp_path / "pipeline_log.json", log)
        assert load_pipeline_log(tmp_path) == log


class TestLoadHealthCheck:

    def test_round_trip(self, tmp_path):
        checks = [{"check": "x", "status": "GREEN", "detail": "ok"}]
        _write_json(tmp_path / "health_check.json", checks)
        assert load_health_check(tmp_path) == checks


class TestLoadAlertHistory:

    def test_round_trip(self, tmp_path):
        alerts = {"2024-01-01": {"alerts": [], "max_severity": "none"}}
        _write_json(tmp_path / "alert_history.json", alerts)
        assert load_alert_history(tmp_path) == alerts


class TestLoadCountryOutputs:

    def test_returns_dfs_and_missing(self, tmp_path):
        idx = pd.bdate_range("2024-01-01", periods=5)
        _write_csv(tmp_path / "Brazil.csv",
                   pd.DataFrame({"5Y": [10, 11, 12, 13, 14]}, index=idx))
        # Mexico missing on purpose
        dfs, missing = load_country_outputs(["Brazil", "Mexico"], tmp_path)
        assert set(dfs) == {"Brazil"}
        assert isinstance(dfs["Brazil"], pd.DataFrame)
        assert missing == ["Mexico"]

    def test_empty_universe(self, tmp_path):
        dfs, missing = load_country_outputs([], tmp_path)
        assert dfs == {} and missing == []


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #

def test_default_output_dir_matches_dashboard():
    assert str(DEFAULT_OUTPUT_DIR).replace("\\", "/") == "data/output"
