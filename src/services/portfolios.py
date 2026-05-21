"""
Dashboard portfolio service.

Two entry points the Home page needs:

* :func:`build_portfolio_views` — loads every portfolio in the config,
  builds the duration P&L proxy via the Data layer, and adds a daily
  carry term. Returns the exact ``[{def, pnl, proxy_dy}, ...]`` shape
  ``app.py`` was assembling inline.
* :func:`compute_quick_stats` — the eight top-line numbers shown on the
  Home-page portfolio cards (annualised return, vol, Sharpe rf=0,
  cumulative return, max drawdown, parametric 95% VaR, start, end).

Both functions are pure (no Streamlit, no I/O side-effects beyond
reading raw CSVs) so they're trivial to unit-test.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_loader import (
    build_portfolio_pnl_from_def,
    load_all_countries_combined,
    load_config,
    load_country_yields,
)


def _apply_daily_carry(
    pnl: pd.Series, pdef: dict, data_dir: str,
) -> pd.Series:
    """Add ``portfolio_weighted_yield / 100 / 252`` to the rate-change P&L.

    Converts the pure duration proxy into a total-return proxy. For the
    HC fund this uses local-currency yields as a carry approximation
    (actual USD bond yields are lower — the proxy remains an estimate).
    """
    mat = pdef["benchmark_maturity"]
    raw_w = pdef["weights"]
    tot_w = sum(raw_w.values())
    w_norm = {k: v / tot_w for k, v in raw_w.items()}
    carry_parts: dict[str, pd.Series] = {}
    for country, wt in w_norm.items():
        try:
            lvl = load_country_yields(country, data_dir=data_dir)
            if mat in lvl.columns:
                carry_parts[country] = lvl[mat] * wt
        except Exception:
            continue
    if not carry_parts:
        return pnl
    port_yield_pct = (
        pd.DataFrame(carry_parts).sum(axis=1).reindex(pnl.index).ffill()
    )
    return pnl + (port_yield_pct / 100 / 252)


def build_portfolio_views(
    config: Optional[dict] = None,
    data_dir: str = "data/raw",
) -> list[dict]:
    """Build per-portfolio analytics views for the dashboard Home page.

    Returns a list of ``{"def", "pnl", "proxy_dy"}`` dicts, one per
    portfolio in ``config["portfolios"]``. ``pnl`` is the duration-based
    rate proxy plus daily carry.
    """
    if config is None:
        config = load_config()
    change_dfs = load_all_countries_combined(config, data_dir=data_dir)
    results: list[dict] = []
    for pdef in config["portfolios"]:
        pnl, proxy_dy = build_portfolio_pnl_from_def(change_dfs, pdef)
        pnl_with_carry = _apply_daily_carry(pnl, pdef, data_dir=data_dir)
        results.append({
            "def": pdef,
            "pnl": pnl_with_carry,
            "proxy_dy": proxy_dy,
        })
    return results


def compute_quick_stats(pnl: pd.Series) -> dict:
    """Eight top-line numbers shown on a Home-page portfolio card.

    All percentages are in human units (``5.3`` = 5.3%) to match the
    dashboard's display formatting. Sharpe is dimensionless and assumes
    ``rf = 0``. VaR is the one-day 95% parametric (normal) estimate.
    """
    n = len(pnl)
    ann = np.sqrt(252)
    ret = float(((1 + pnl).prod() ** (252 / n) - 1) * 100)
    vol = float(pnl.std() * ann * 100)
    sharpe = ret / vol if vol > 0 else float("nan")
    cum = float(((1 + pnl).cumprod() - 1).iloc[-1] * 100)
    roll = (1 + pnl).cumprod()
    dd = float(((roll / roll.cummax()) - 1).min() * 100)
    var95 = float(-(pnl.mean() + norm.ppf(0.05) * pnl.std()) * 100)
    return {
        "ret": ret,
        "vol": vol,
        "sharpe": sharpe,
        "cum": cum,
        "dd": dd,
        "var95": var95,
        "start": pnl.index.min().strftime("%b %Y"),
        "end": pnl.index.max().strftime("%b %Y"),
    }
