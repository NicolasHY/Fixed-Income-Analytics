"""
Module: risk_free.py
====================
Fetches and caches SOFR (USD) and €STR/EONIA (EUR) risk-free rate series
from the St. Louis Federal Reserve FRED API.

USD series  : SOFR (from 2018-04-02), backfilled with DFF (Fed Funds daily)
EUR series  : €STR / ECBESTRVOLWGTTRMDMNRT (from 2019-10-02), backfilled with EONIA / EONYA

All rates are returned as annualised percentages, as published by FRED.
Use ``daily_rf_from_annual()`` to convert to daily fraction for return calculations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs
_SOFR   = "SOFR"                    # Secured Overnight Financing Rate (USD)
_DFF    = "DFF"                     # Effective Federal Funds Rate – backfill for SOFR
_ESTR   = "ECBESTRVOLWGTTRMDMNRT"   # Euro Short-Term Rate (€STR)
_EONIA  = "EONYA"                   # EONIA overnight rate – backfill for €STR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_series(api_key: str, series_id: str, start: str = "2018-01-01") -> pd.Series:
    """Fetch one FRED series and return a date-indexed Series (annualised %)."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "sort_order": "asc",
    }
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    s = pd.to_numeric(df["value"], errors="coerce").dropna()
    s.name = series_id
    logger.info("Fetched %s: %d observations (%s → %s).",
                series_id, len(s), s.index[0].date(), s.index[-1].date())
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_and_cache_risk_free_rates(
    fred_api_key: str,
    output_path: str | Path = "data/output/risk_free_rates.csv",
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """
    Download SOFR+DFF (USD) and €STR+EONIA (EUR) from FRED, combine them
    into a single DataFrame, forward-fill weekends/holidays, and save to CSV.

    Parameters
    ----------
    fred_api_key : str
    output_path  : path-like
        Destination CSV (created if missing).
    start : str
        Earliest date to fetch (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Columns: ``sofr_pct``, ``estr_pct`` – both annualised percent.
    """
    logger.info("Fetching risk-free rates from FRED (start=%s)…", start)

    # ── USD: SOFR, backfilled with DFF ──────────────────────────────────────
    sofr = _fetch_series(fred_api_key, _SOFR, start=start)
    dff  = _fetch_series(fred_api_key, _DFF,  start=start)
    usd_rf = sofr.combine_first(dff)
    usd_rf.name = "sofr_pct"

    # ── EUR: €STR, backfilled with EONIA ────────────────────────────────────
    try:
        estr = _fetch_series(fred_api_key, _ESTR, start=start)
    except Exception as exc:
        logger.warning("Could not fetch €STR (%s); will use EONIA only.", exc)
        estr = pd.Series(dtype=float, name=_ESTR)

    try:
        eonia = _fetch_series(fred_api_key, _EONIA, start=start)
    except Exception as exc:
        logger.warning("Could not fetch EONIA (%s).", exc)
        eonia = pd.Series(dtype=float, name=_EONIA)

    eur_rf = estr.combine_first(eonia)
    eur_rf.name = "estr_pct"

    # ── Combine, forward-fill, save ──────────────────────────────────────────
    out = pd.concat([usd_rf, eur_rf], axis=1).sort_index()
    out = out.ffill()          # propagate weekend / holiday values

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path)
    logger.info(
        "Risk-free rates cached to %s (%d rows, %s → %s).",
        output_path, len(out), out.index[0].date(), out.index[-1].date(),
    )
    return out


def load_risk_free_rates(
    output_path: str | Path = "data/output/risk_free_rates.csv",
    fred_api_key: str | None = None,
) -> pd.DataFrame:
    """
    Load the cached risk-free rate CSV.  If the file does not exist and a
    FRED key is supplied, fetch and cache it first.

    Returns
    -------
    pd.DataFrame
        Columns: ``sofr_pct``, ``estr_pct`` (annualised %).
    """
    output_path = Path(output_path)
    if not output_path.exists():
        if fred_api_key:
            return fetch_and_cache_risk_free_rates(fred_api_key, output_path)
        raise FileNotFoundError(
            f"{output_path} not found. Provide a FRED API key or run "
            "fetch_and_cache_risk_free_rates() first."
        )
    df = pd.read_csv(output_path, index_col=0, parse_dates=True)
    logger.info("Loaded risk-free rates from %s (%d rows).", output_path, len(df))
    return df


def daily_rf_from_annual(annual_pct: pd.Series) -> pd.Series:
    """
    Convert an annualised rate series (as percent) to a daily return fraction.

    Formula: r_daily = (1 + r_annual/100)^(1/252) - 1
    """
    return (1 + annual_pct / 100) ** (1 / 252) - 1


def align_rf_to_pnl(
    rf_df: pd.DataFrame,
    pnl: pd.Series,
    column: str = "estr_pct",
) -> pd.Series:
    """
    Forward-fill the chosen risk-free rate column to the P&L index and
    return daily fraction returns aligned to the P&L dates.

    Parameters
    ----------
    rf_df   : output of load_risk_free_rates()
    pnl     : daily P&L series (fraction)
    column  : 'estr_pct' (EUR, default) or 'sofr_pct' (USD)
    """
    annual = rf_df[column].reindex(pnl.index, method="ffill").dropna()
    return daily_rf_from_annual(annual)
