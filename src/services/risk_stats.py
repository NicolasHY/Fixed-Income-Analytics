"""
Shared risk-stat computation used by both app.py (Risk Statistics tab)
and src/report_generator.py (quarterly Excel export).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_risk_stats(
    pdef: dict,
    pnl: pd.Series,
    yield_levels: dict,
    rf_data=None,
) -> dict:
    """
    Compute return, risk, bond-analytics and VaR statistics for one portfolio.

    Parameters
    ----------
    pdef : dict
        One entry from config["portfolios"] — must have keys:
        weights, effective_duration, benchmark_maturity, aum_eur.
    pnl : pd.Series
        Daily portfolio P&L as a decimal fraction, business-date index.
    yield_levels : dict[str, pd.DataFrame]
        {country: wide DataFrame of yield levels (%) indexed by date}.
    rf_data : pd.DataFrame or None
        Risk-free rate table with columns estr_pct, sofr_pct; None → rf = 0.

    Returns
    -------
    dict with keys: ann_ret, cum_log, carry, rolldown, ann_vol, max_dd,
        sharpe, sortino, sharpe_zero, sortino_zero, calmar,
        mod_dur, dv01, dv01_eur, convexity, ytm, yc_slope,
        krd, md_by_c, c_vals,
        var_rows, var_rows_eur,
        current_estr, current_sofr, avg_sofr, aum.
    """
    def _par_md(yield_pct: float, T: int) -> float:
        y = yield_pct / 100
        return float(T) if y <= 0 else (1 - (1 + y) ** (-T)) / y

    raw_w: dict[str, float] = pdef["weights"]
    tot_w = sum(raw_w.values())
    w = {k: v / tot_w for k, v in raw_w.items()}
    D = float(pdef["effective_duration"])
    mat = pdef["benchmark_maturity"]
    mat_n = int(mat[:-1])
    n = len(pnl)
    aum = float(pdef.get("aum_eur", 0))

    # ── Return metrics ────────────────────────────────────────────────────────
    cum_log = float(np.log1p(pnl).sum() * 100)
    ann_ret = float(((1 + pnl).prod() ** (252 / n) - 1) * 100)

    # ── Carry: portfolio-weighted latest benchmark yield ──────────────────────
    c_vals: dict[str, float] = {}
    for c in w:
        if c in yield_levels and mat in yield_levels[c].columns:
            s = yield_levels[c][mat].dropna()
            if len(s) > 0:
                c_vals[c] = float(s.iloc[-1])
    if c_vals:
        ws_ = sum(w[c] for c in c_vals)
        carry = sum(w[c] * c_vals[c] for c in c_vals) / ws_
    else:
        carry = np.nan

    # ── Modified duration (par bond approximation) ───────────────────────────
    md_by_c: dict[str, float] = {c: _par_md(c_vals[c], mat_n) for c in c_vals}
    for c in w:
        if c not in md_by_c:
            md_by_c[c] = D

    # ── Roll-down ─────────────────────────────────────────────────────────────
    rd_vals: dict[str, float] = {}
    for c in w:
        if c not in yield_levels:
            continue
        avail_nums = sorted(int(x[:-1]) for x in yield_levels[c].columns)
        shorter = [m for m in avail_nums if m < mat_n]
        if not shorter:
            continue
        ns = max(shorter)
        sub = yield_levels[c][[mat, f"{ns}Y"]].dropna()
        if len(sub) == 0:
            continue
        row_ = sub.iloc[-1]
        slope = (row_[mat] - row_[f"{ns}Y"]) / (mat_n - ns)
        rd_vals[c] = md_by_c[c] * slope
    if rd_vals:
        ws_rd = sum(w[c] for c in rd_vals)
        rolldown = sum(w[c] * rd_vals[c] for c in rd_vals) / ws_rd
    else:
        rolldown = np.nan

    # ── Volatility & drawdown ─────────────────────────────────────────────────
    ann_vol = float(pnl.std() * np.sqrt(252) * 100)
    cum_s = (1 + pnl).cumprod()
    max_dd = float(((cum_s / cum_s.cummax()) - 1).min() * 100)

    # ── Portfolio duration analytics ──────────────────────────────────────────
    ws_md = sum(w[c] for c in md_by_c)
    mod_dur = sum(w[c] * md_by_c[c] for c in md_by_c) / ws_md
    dv01 = mod_dur * 0.01
    dv01_eur = mod_dur * 0.0001 * aum if aum else np.nan
    ytm = carry

    conv_by_c: dict[str, float] = {}
    for c in md_by_c:
        if c in c_vals:
            y_f = c_vals[c] / 100
            d_mac = md_by_c[c] * (1 + y_f)
            conv_by_c[c] = d_mac * (d_mac + 1) / (1 + y_f) ** 2
    if conv_by_c:
        ws_cv = sum(w[c] for c in conv_by_c)
        convexity = sum(w[c] * conv_by_c[c] for c in conv_by_c) / ws_cv
    else:
        convexity = np.nan

    sl_vals: dict[str, float] = {}
    for c in w:
        if c not in yield_levels:
            continue
        lr = yield_levels[c].dropna(how="all").iloc[-1].dropna()
        avail = sorted(lr.index, key=lambda x: int(x[:-1]))
        if len(avail) >= 2:
            sl_vals[c] = float(lr[avail[-1]] - lr[avail[0]])
    if sl_vals:
        ws_sl = sum(w[c] for c in sl_vals)
        yc_slope = sum(w[c] * sl_vals[c] for c in sl_vals) / ws_sl
    else:
        yc_slope = np.nan

    krd = {c: w[c] * md_by_c.get(c, D) for c in w}

    # ── Ratios rf = 0 ─────────────────────────────────────────────────────────
    sharpe_zero = (ann_ret / 100) / (ann_vol / 100) if ann_vol > 0 else np.nan
    ds_zero = float(np.mean(np.minimum(pnl, 0.0) ** 2))
    sortino_zero = (
        (ann_ret / 100) / (np.sqrt(ds_zero) * np.sqrt(252)) if ds_zero > 0 else np.nan
    )
    calmar = (ann_ret / 100) / abs(max_dd / 100) if max_dd != 0 else np.nan

    # ── Ratios rf = SOFR overnight (fall back to €STR) ───────────────────────
    current_estr = np.nan
    current_sofr = np.nan
    avg_sofr     = np.nan
    rf_label = "0"
    sharpe  = sharpe_zero
    sortino = sortino_zero
    if rf_data is not None:
        from src.risk_free import align_rf_to_pnl
        # Overnight rate: SOFR preferred, €STR as fallback
        if "sofr_pct" in rf_data.columns and rf_data["sofr_pct"].dropna().shape[0] > 0:
            rf_col, rf_label = "sofr_pct", "SOFR"
        elif "estr_pct" in rf_data.columns and rf_data["estr_pct"].dropna().shape[0] > 0:
            rf_col, rf_label = "estr_pct", "€STR"
        else:
            rf_col = None
        if rf_col:
            try:
                rf_series = align_rf_to_pnl(rf_data, pnl, column=rf_col)
                common = pnl.index.intersection(rf_series.index)
                excess = pnl.loc[common] - rf_series.loc[common]
                n_exc = len(excess)
                ann_exc = float(((1 + excess).prod() ** (252 / n_exc) - 1) * 100)
                exc_vol = float(excess.std() * np.sqrt(252) * 100)
                sharpe = (ann_exc / 100) / (exc_vol / 100) if exc_vol > 0 else np.nan
                ds_rf = float(np.mean(np.minimum(excess, 0.0) ** 2))
                sortino = (
                    (ann_exc / 100) / (np.sqrt(ds_rf) * np.sqrt(252)) if ds_rf > 0 else np.nan
                )
                avg_sofr = float(
                    rf_data[rf_col].reindex(pnl.index, method="ffill").dropna().mean()
                )
            except Exception:
                pass
        try:
            current_sofr = float(rf_data["sofr_pct"].dropna().iloc[-1])
            current_estr = float(rf_data["estr_pct"].dropna().iloc[-1])
        except Exception:
            pass

    # ── VaR / CVaR ────────────────────────────────────────────────────────────
    mu_p, sig_p = pnl.mean(), pnl.std()
    np.random.seed(42)
    sims = np.random.normal(mu_p, sig_p, 50_000)
    var_rows = []
    for alpha in [0.05, 0.10]:
        z = norm.ppf(alpha)
        pv = -(mu_p + z * sig_p) * 100
        pcv = -(mu_p - sig_p * norm.pdf(-z) / alpha) * 100
        q_ = float(np.quantile(pnl, alpha))
        hv = -q_ * 100
        tmask = pnl <= q_
        hcv = -float(pnl[tmask].mean()) * 100 if tmask.any() else np.nan
        mcv = -float(np.percentile(sims, alpha * 100)) * 100
        var_rows.append({
            "α": f"{int(alpha * 100)}%",
            "Confidence": f"{int((1 - alpha) * 100)}%",
            "Param VaR (%)": round(pv, 4),
            "Param CVaR (%)": round(pcv, 4),
            "Hist VaR (%)": round(hv, 4),
            "Hist CVaR (%)": round(hcv, 4),
            "MC VaR (%)": round(mcv, 4),
        })

    var_rows_eur = [
        {
            **vr,
            "Param VaR (EUR)": vr["Param VaR (%)"] / 100 * aum if aum else np.nan,
            "Hist VaR (EUR)": vr["Hist VaR (%)"] / 100 * aum if aum else np.nan,
            "MC VaR (EUR)": vr["MC VaR (%)"] / 100 * aum if aum else np.nan,
            "Param CVaR (EUR)": vr["Param CVaR (%)"] / 100 * aum if aum else np.nan,
        }
        for vr in var_rows
    ]

    return dict(
        cum_log=cum_log, ann_ret=ann_ret, carry=carry, rolldown=rolldown,
        ann_vol=ann_vol, max_dd=max_dd,
        sharpe=sharpe, sortino=sortino,
        sharpe_zero=sharpe_zero, sortino_zero=sortino_zero,
        calmar=calmar, mod_dur=mod_dur, dv01=dv01, dv01_eur=dv01_eur,
        aum=aum, convexity=convexity, md_by_c=md_by_c,
        ytm=ytm, yc_slope=yc_slope, krd=krd, c_vals=c_vals,
        var_rows=var_rows, var_rows_eur=var_rows_eur,
        current_estr=current_estr, current_sofr=current_sofr,
        avg_sofr=avg_sofr, rf_label=rf_label,
    )
