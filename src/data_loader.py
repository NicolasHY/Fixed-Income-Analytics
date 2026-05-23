"""
Module: data_loader.py
======================
Handles all data ingestion and preprocessing for the EM FI platform.

Raw data layout (from Investing.com downloads):
    data/raw/<Country>/<Country> <N>-Year Bond Yield Historical Data.csv
    Columns: Date, Price, Open, High, Low, Change %

All country/fund/threshold parameters come from config/funds.yaml.
"""

from __future__ import annotations

import re
import glob
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config" / "funds.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load and return the centralized funds.yaml configuration."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Raw CSV loading
# ---------------------------------------------------------------------------

def _parse_maturity(filename: str) -> str | None:
    """
    Extract maturity label (e.g. '5Y') from a filename like:
        'Brazil 5-Year Bond Yield Historical Data.csv'
    Returns None if the pattern does not match.
    """
    match = re.search(r"(\d+)-Year", filename)
    return f"{match.group(1)}Y" if match else None


def load_country_yields(country: str, data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """
    Load all maturity CSVs for a single country and pivot into a wide DataFrame.

    Parameters
    ----------
    country : str
        Country name matching the subdirectory under data_dir (e.g. 'Brazil').
    data_dir : path-like
        Root of the raw data directory.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by business Date, columns = maturities (e.g. '2Y', '5Y', …).
        Values are yield levels in percent (as published by Investing.com).
    """
    data_dir = Path(data_dir)
    pattern = str(data_dir / country / "*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No CSV files found for {country} in {data_dir / country}. "
            "Run data collection first."
        )

    frames = []
    for fpath in files:
        mat = _parse_maturity(Path(fpath).name)
        if mat is None:
            logger.warning("Could not parse maturity from %s — skipping.", fpath)
            continue

        df = pd.read_csv(fpath, usecols=["Date", "Price"])
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df = df.rename(columns={"Price": mat})
        frames.append(df)

    if not frames:
        raise ValueError(f"All CSV files for {country} failed to parse.")

    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide[sorted(wide.columns, key=lambda x: int(x[:-1]))]  # sort maturities numerically
    return wide


def load_yield_changes(
    country: str,
    data_dir: str | Path = "data/raw",
    excluded_maturities: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load yield levels and compute daily changes (first differences) for one country.

    Parameters
    ----------
    country : str
    data_dir : path-like
    excluded_maturities : list of str, optional
        Maturity labels to drop (e.g. ['2Y'] for Colombia per the config).

    Returns
    -------
    pd.DataFrame
        Daily Δy in percentage points, business-date index.
    """
    yields = load_country_yields(country, data_dir)

    if excluded_maturities:
        yields = yields.drop(columns=[m for m in excluded_maturities if m in yields.columns])

    changes = yields.diff().dropna()
    logger.info(
        "%s: %d obs after differencing (%.1f%% kept).",
        country, len(changes), 100 * len(changes) / len(yields),
    )
    return changes


def load_all_countries(
    config: dict,
    data_dir: str | Path = "data/raw",
    country_list: str = "local_currency",
) -> dict[str, pd.DataFrame]:
    """
    Load yield changes for all countries in the config's country list.

    Parameters
    ----------
    config : dict
        Output of load_config().
    data_dir : path-like
    country_list : {'local_currency', 'hard_currency'}
        Which country universe to load.

    Returns
    -------
    dict[str, pd.DataFrame]
        {country_name: yield_changes_df}
    """
    countries = config["countries"][country_list]
    excluded = config.get("excluded_series", {})
    change_dfs: dict[str, pd.DataFrame] = {}

    for country in countries:
        try:
            change_dfs[country] = load_yield_changes(
                country,
                data_dir=data_dir,
                excluded_maturities=excluded.get(country),
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", country, e)

    if not change_dfs:
        raise RuntimeError("No country data could be loaded. Check data/raw/ directory.")

    return change_dfs


# ---------------------------------------------------------------------------
# Portfolio P&L proxy
# ---------------------------------------------------------------------------

def build_portfolio_pnl(
    change_dfs: dict[str, pd.DataFrame],
    config: dict,
    benchmark_mat: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build the LC fund P&L proxy via duration approximation.

    Formula:  ΔP/P ≈ −D_eff × (weighted_avg_Δy / 100)

    Parameters
    ----------
    change_dfs : dict[str, pd.DataFrame]
        Output of load_all_countries().
    config : dict
        Output of load_config(). Uses lc_fund.weights, lc_fund.effective_duration,
        and lc_fund.benchmark_maturity.
    benchmark_mat : str, optional
        Override the maturity to use (default: config lc_fund.benchmark_maturity).

    Returns
    -------
    pnl : pd.Series
        Daily portfolio P&L as a fraction (not percent), business-date index.
    proxy_dy : pd.DataFrame
        The per-country yield change series used (for diagnostics).
    """
    fund_cfg = config["lc_fund"]
    raw_weights: dict[str, float] = fund_cfg["weights"]
    D_eff: float = fund_cfg["effective_duration"]
    mat: str = benchmark_mat or fund_cfg["benchmark_maturity"]

    # Normalise weights to sum to 1
    total = sum(raw_weights.values())
    weights = {k: v / total for k, v in raw_weights.items()}

    # Select benchmark maturity for each country
    proxy_dy = pd.DataFrame({
        country: change_dfs[country][mat]
        for country in weights
        if country in change_dfs and mat in change_dfs[country].columns
    }).dropna()

    w_vec = np.array([weights[c] for c in proxy_dy.columns])
    portfolio_dy = proxy_dy @ w_vec          # weighted avg daily yield change (bps / 100)
    pnl = -D_eff * (portfolio_dy / 100)      # duration-price approximation

    logger.info(
        "Portfolio P&L proxy: %d obs, mean=%.4f%%, std=%.4f%%",
        len(pnl), pnl.mean() * 100, pnl.std() * 100,
    )
    return pnl, proxy_dy


def build_portfolio_pnl_from_def(
    change_dfs: dict[str, pd.DataFrame],
    portfolio_def: dict,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build P&L proxy for a portfolio defined by an explicit dict (from the
    ``portfolios`` section of funds.yaml).

    Parameters
    ----------
    change_dfs : dict[str, pd.DataFrame]
        Yield-change DataFrames keyed by country name.
    portfolio_def : dict
        One entry from ``config["portfolios"]``. Must contain ``weights``,
        ``effective_duration``, and ``benchmark_maturity``.

    Returns
    -------
    pnl : pd.Series
        Daily P&L as a fraction.
    proxy_dy : pd.DataFrame
        Per-country yield change series used.
    """
    raw_weights: dict[str, float] = portfolio_def["weights"]
    D_eff: float = portfolio_def["effective_duration"]
    mat: str = portfolio_def["benchmark_maturity"]

    total = sum(raw_weights.values())
    weights = {k: v / total for k, v in raw_weights.items()}

    proxy_dy = pd.DataFrame({
        country: change_dfs[country][mat]
        for country in weights
        if country in change_dfs and mat in change_dfs[country].columns
    }).dropna()

    w_vec = np.array([weights[c] for c in proxy_dy.columns])
    portfolio_dy = proxy_dy @ w_vec
    pnl = -D_eff * (portfolio_dy / 100)

    logger.info(
        "%s P&L: %d obs, mean=%.4f%%, std=%.4f%%",
        portfolio_def.get("name", "portfolio"),
        len(pnl), pnl.mean() * 100, pnl.std() * 100,
    )
    return pnl, proxy_dy


def load_all_countries_combined(
    config: dict,
    data_dir: str | Path = "data/raw",
) -> dict[str, pd.DataFrame]:
    """
    Load yield changes for every country present in data/raw, spanning both
    the local-currency and hard-currency universes defined in config.

    Returns
    -------
    dict[str, pd.DataFrame]
        {country_name: yield_changes_df}
    """
    all_countries = (
        config["countries"]["local_currency"]
        + config["countries"]["hard_currency"]
    )
    excluded = config.get("excluded_series", {})
    change_dfs: dict[str, pd.DataFrame] = {}

    for country in all_countries:
        try:
            change_dfs[country] = load_yield_changes(
                country,
                data_dir=data_dir,
                excluded_maturities=excluded.get(country),
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", country, e)

    if not change_dfs:
        raise RuntimeError("No country data could be loaded. Check data/raw/ directory.")

    return change_dfs
