"""
EM Fixed Income Intelligence Platform — Streamlit Dashboard

Offline-capable demo: reads pre-generated outputs from data/output/.
Run:  streamlit run app.py
"""

import json
import os
import sys
import threading
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import load_config, load_all_countries_combined, build_portfolio_pnl_from_def, load_country_yields
from src.risk_free import load_risk_free_rates, align_rf_to_pnl

st.set_page_config(
    page_title="EM FI Intelligence",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'/>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2a3b 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span {
        color: #c9d6e3 !important;
    }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
        gap: 12px;
    }

    /* ── Sidebar nav: turn radio buttons into clickable link rows ── */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        gap: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        display: flex !important;
        align-items: center;
        padding: 7px 12px;
        margin: 0;
        border-radius: 6px;
        cursor: pointer;
        border-left: 3px solid transparent;
        transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
    }
    /* hide the native radio dot */
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    /* chevron indicator */
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label::before {
        content: "›";
        color: #4a6a85;
        margin-right: 10px;
        font-size: 1.15rem;
        line-height: 1;
        transition: color 0.12s ease, transform 0.12s ease;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label p {
        color: #8ab4d4 !important;
        font-size: 0.92rem !important;
        margin: 0 !important;
        transition: color 0.12s ease;
    }
    /* hover */
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background: rgba(126,200,227,0.08);
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover::before {
        color: #7ec8e3;
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover p {
        color: #ffffff !important;
    }
    /* selected state */
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has([aria-checked="true"]),
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(126,200,227,0.16) 0%, rgba(126,200,227,0) 100%);
        border-left-color: #7ec8e3;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has([aria-checked="true"])::before,
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked)::before {
        color: #7ec8e3;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has([aria-checked="true"]) p,
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Main background ── */
    .stApp {
        background-color: #f0f4f8;
    }

    /* ── Top header bar ── */
    .company-header {
        background: linear-gradient(90deg, #0d1b2a 0%, #1b3a5c 100%);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .company-header h1 {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .company-header p {
        color: #8ab4d4 !important;
        margin: 4px 0 0 0;
        font-size: 0.85rem;
    }
    /* higher specificity to beat the main-content text rule */
    section[data-testid="stMain"] .company-header h1 { color: #ffffff !important; }
    section[data-testid="stMain"] .company-header p  { color: #8ab4d4 !important; }
    .company-badge {
        background: #1e4d7b;
        color: #7ec8e3;
        border: 1px solid #2e6ea8;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* ── Equal-height columns ── */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    [data-testid="column"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    [data-testid="column"] .stat-card,
    [data-testid="column"] .health-card {
        flex: 1;
    }

    /* ── Stat cards ── */
    .stat-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-top: 4px solid #1b3a5c;
        text-align: center;
        height: 100%;
        box-sizing: border-box;
    }
    .stat-card .label {
        font-size: 0.78rem;
        color: #6b7c93;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .stat-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0d1b2a;
    }
    .stat-card.green { border-top-color: #16a34a; }
    .stat-card.amber { border-top-color: #d97706; }
    .stat-card.red   { border-top-color: #dc2626; }
    .stat-card.blue  { border-top-color: #1b3a5c; }

    /* ── Status pill badges ── */
    .pill {
        display: inline-block;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .pill-green  { background: #dcfce7; color: #15803d; }
    .pill-yellow { background: #fef9c3; color: #92400e; }
    .pill-red    { background: #fee2e2; color: #991b1b; }
    .pill-blue   { background: #dbeafe; color: #1e40af; }
    .pill-gray   { background: #f1f5f9; color: #475569; }

    /* ── Section cards ── */
    .section-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 24px 28px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }
    .section-card h3 {
        color: #0d1b2a;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e2e8f0;
    }

    /* ── Health check cards ── */
    .health-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        text-align: center;
        height: 100%;
        box-sizing: border-box;
    }
    .health-card .hc-icon { font-size: 2rem; }
    .health-card .hc-name {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7c93;
        margin: 6px 0 4px 0;
    }
    .health-card .hc-status {
        font-size: 1rem;
        font-weight: 700;
    }
    .health-card .hc-detail {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* ── Briefing container ── */
    .briefing-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 28px 32px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-left: 5px solid #1b3a5c;
        line-height: 1.75;
        color: #1e293b;
    }

    /* ── Event context pill ── */
    .event-context {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: #1e40af;
        margin-bottom: 16px;
    }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 20px 0;
    }

    /* ── Main content text (fixes white-on-light background) ── */
    section[data-testid="stMain"] p,
    section[data-testid="stMain"] li,
    section[data-testid="stMain"] ol,
    section[data-testid="stMain"] ul,
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4,
    section[data-testid="stMain"] strong,
    section[data-testid="stMain"] em {
        color: #1e293b !important;
    }

    /* ── Sidebar expand arrow (visible when sidebar is collapsed) ── */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] * {
        color: #1b3a5c !important;
    }

    /* ── Sidebar collapse arrow (visible when sidebar is open) — always show ── */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
    }
    [data-testid="stSidebarCollapseButton"] * {
        color: #c9d6e3 !important;
    }

    /* ── Home page section label ── */
    .home-section-label {
        font-size: 0.75rem;
        color: #6b7c93;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 28px 0 12px 0;
        font-weight: 600;
    }

    /* ── Home nav container cards ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 100%) !important;
        border: 1px solid #1b3a5c !important;
        border-radius: 10px !important;
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.20) !important;
        border-color: #2e6ea8 !important;
    }
    /* Force white text on all child markdown / captions inside the nav cards */
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] strong,
    [data-testid="stVerticalBlockBorderWrapper"] small,
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"],
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] p {
        color: #ffffff !important;
    }

    /* ── Sidebar: flex layout so the stop-server button can pin to the bottom ── */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        min-height: calc(100vh - 5rem);
    }
    .sidebar-spacer {
        flex: 1 1 auto;
        min-height: 16px;
    }

    /* ── Sidebar title ── */
    .sidebar-title {
        font-size: 1.3rem;
        color: #7ec8e3;
        font-weight: 700;
        letter-spacing: 0.01em;
        white-space: nowrap;
        line-height: 1;
    }

    /* ── Sidebar Universe expander (dark theme override) ── */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid #1e3a5f;
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary span,
    section[data-testid="stSidebar"] details summary {
        color: #c9d6e3 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        text-transform: none;
        letter-spacing: 0.01em;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] svg {
        color: #8ab4d4 !important;
        fill: #8ab4d4 !important;
    }

    /* ── Compact "Stop Server" button (sidebar secondary) ── */
    section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
        font-size: 0.7rem !important;
        padding: 3px 12px !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
        width: auto !important;
        background: transparent !important;
        border: 1px solid #3a5577 !important;
        color: #8ab4d4 !important;
        opacity: 0.75;
    }
    section[data-testid="stSidebar"] .stButton:has(button[kind="secondary"]) {
        text-align: center;
        margin-top: 8px;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        opacity: 1;
        border-color: #c47a7a !important;
        color: #e89a9a !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

OUT = Path("data/output")

PAGES = [
    "Home", "Pipeline Health", "Data Load", "PCA & Regime",
    "VaR Engine", "Portfolios", "Alert History", "Daily Briefings",
]
_NAV_KEY = "_nav"

if _NAV_KEY not in st.session_state:
    st.session_state[_NAV_KEY] = "Home"


@st.cache_data(show_spinner="Loading portfolio data…")
def _load_portfolio_data():
    cfg = load_config()
    change_dfs = load_all_countries_combined(cfg, data_dir="data/raw")
    results = []
    for pdef in cfg["portfolios"]:
        pnl, proxy_dy = build_portfolio_pnl_from_def(change_dfs, pdef)

        # ── Add daily carry: portfolio-weighted benchmark yield / 252 ─────────
        # Converts the pure rate-change proxy into a total-return proxy.
        # For the HC fund this uses local-currency yields as a carry approximation
        # (actual USD bond yields are lower — the proxy remains an estimate).
        mat = pdef["benchmark_maturity"]
        raw_w = pdef["weights"]
        tot_w = sum(raw_w.values())
        w_norm = {k: v / tot_w for k, v in raw_w.items()}
        carry_parts: dict[str, pd.Series] = {}
        for country, wt in w_norm.items():
            try:
                lvl = load_country_yields(country, data_dir="data/raw")
                if mat in lvl.columns:
                    carry_parts[country] = lvl[mat] * wt
            except Exception:
                pass
        if carry_parts:
            port_yield_pct = (
                pd.DataFrame(carry_parts).sum(axis=1).reindex(pnl.index).ffill()
            )
            pnl = pnl + (port_yield_pct / 100 / 252)

        results.append({"def": pdef, "pnl": pnl, "proxy_dy": proxy_dy})
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 12px 0 22px 0;'>
        <div class='sidebar-title'>EM Fixed Income</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem; color:#4a6a85; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "",
        PAGES,
        key=_NAV_KEY,
        label_visibility="collapsed",
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    with st.expander("Investable Universe", expanded=False):
        st.markdown("""
        <div style='font-size:0.8rem; color:#8ab4d4; line-height:2;'>
            🇧🇷 Brazil<br>
            🇲🇽 Mexico<br>
            🇿🇦 South Africa<br>
            🇵🇱 Poland<br>
            🇨🇴 Colombia<br>
            🇭🇺 Hungary<br>
            🇷🇴 Romania
        </div>
        """, unsafe_allow_html=True)

    # Spacer pushes the Stop Server button to the very bottom of the sidebar.
    st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)

    if st.button("Stop Server", key="stop_server_btn", type="secondary"):
        # Try to close the browser tab (only works on JS-opened windows);
        # fall back to replacing the page with a "Server stopped" message.
        components.html(
            """
            <script>
                const finish = () => {
                    try { window.top.close(); } catch (e) {}
                    try { window.parent.window.close(); } catch (e) {}
                    try { window.close(); } catch (e) {}
                    try {
                        window.parent.document.documentElement.innerHTML =
                            '<body style="display:flex;align-items:center;justify-content:center;'
                          + 'height:100vh;font-family:system-ui,sans-serif;background:#f0f4f8;margin:0;">'
                          + '<div style="text-align:center;">'
                          + '<h1 style="color:#0d1b2a;font-size:1.5rem;margin-bottom:8px;">Server stopped</h1>'
                          + '<p style="color:#64748b;font-size:0.95rem;">You can safely close this tab.</p>'
                          + '</div></body>';
                    } catch (e) {}
                };
                setTimeout(finish, 300);
            </script>
            """,
            height=0,
        )
        # Delay the kill so the browser receives the response and runs the JS.
        threading.Timer(0.8, lambda: os._exit(0)).start()
        st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='company-header'>
    <div>
        <h1>EM Fixed Income Intelligence Platform</h1>
        <p>Emerging Markets Sovereign Bond Analytics &nbsp;·&nbsp; Daily Dashboard</p>
    </div>
    <div class='company-badge'>{page.upper()}</div>
</div>
""", unsafe_allow_html=True)

# ── Home ──────────────────────────────────────────────────────────────────────
if page == "Home":

    # ── Portfolio Snapshot ────────────────────────────────────────────────────
    st.markdown("<div class='home-section-label'>Portfolio Snapshot</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:8px;
                padding:10px 16px; font-size:0.82rem; color:#92400e; margin-bottom:14px;">
        <strong>Rate &amp; carry proxy</strong> — Metrics are computed from a yield-change
        duration model with daily coupon accrual added. <strong>FX return is excluded</strong>
        (a significant driver for the LC fund). The HC fund uses local-currency sovereign yields
        as a proxy for its USD-denominated holdings — treat its absolute return as an
        approximation. For verified NAV performance refer to fund factsheets.
    </div>
    """, unsafe_allow_html=True)

    try:
        _home_ports = _load_portfolio_data()
        _hp1, _hp2 = _home_ports[0], _home_ports[1]

        def _quick_stats(pnl):
            n = len(pnl)
            ann = np.sqrt(252)
            ret = float(((1 + pnl).prod() ** (252 / n) - 1) * 100)
            vol = float(pnl.std() * ann * 100)
            sharpe = ret / vol if vol > 0 else np.nan
            cum = float(((1 + pnl).cumprod() - 1).iloc[-1] * 100)
            roll = (1 + pnl).cumprod()
            dd = float(((roll / roll.cummax()) - 1).min() * 100)
            var95 = float(-(pnl.mean() + norm.ppf(0.05) * pnl.std()) * 100)
            start = pnl.index.min().strftime("%b %Y")
            end   = pnl.index.max().strftime("%b %Y")
            return {"ret": ret, "vol": vol, "sharpe": sharpe, "cum": cum,
                    "dd": dd, "var95": var95, "start": start, "end": end}

        _hqs1 = _quick_stats(_hp1["pnl"])
        _hqs2 = _quick_stats(_hp2["pnl"])

        pc_col1, pc_col2 = st.columns(2)
        for _col, _pdata, _qs in [(pc_col1, _hp1, _hqs1), (pc_col2, _hp2, _hqs2)]:
            with _col:
                _pn = _pdata["def"]["name"]
                _aum = _pdata["def"].get("aum_eur", 0)
                _aum_str = f"€{_aum/1e6:.0f}M" if _aum else "N/A"
                _rc = "#4ade80" if _qs["ret"] > 0 else "#f87171"
                _cc = "#4ade80" if _qs["cum"] > 0 else "#f87171"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 100%);
                            border-radius: 12px; padding: 22px 24px; color: #fff;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.12);">
                    <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 4px;">{_pn}</div>
                    <div style="font-size: 0.78rem; color: #8ab4d4; margin-bottom: 16px;">
                        AUM: <strong style="color:#fff;">{_aum_str}</strong>
                        &nbsp;·&nbsp; {_qs['start']} – {_qs['end']}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px;">
                        <div>
                            <div style="font-size: 0.65rem; color: #8ab4d4; text-transform:uppercase; letter-spacing:0.05em;">Ann. Rtn (proxy)</div>
                            <div style="font-size: 1.05rem; font-weight: 700; color:{_rc};">{_qs['ret']:+.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.65rem; color: #8ab4d4; text-transform:uppercase; letter-spacing:0.05em;">Ann. Volatility</div>
                            <div style="font-size: 1.05rem; font-weight: 700;">{_qs['vol']:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.65rem; color: #8ab4d4; text-transform:uppercase; letter-spacing:0.05em;">Sharpe rf=0 (proxy)</div>
                            <div style="font-size: 1.05rem; font-weight: 700;">{_qs['sharpe']:.2f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.65rem; color: #8ab4d4; text-transform:uppercase; letter-spacing:0.05em;">Max Drawdown</div>
                            <div style="font-size: 1.05rem; font-weight: 700; color:#f87171;">{_qs['dd']:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.65rem; color: #8ab4d4; text-transform:uppercase; letter-spacing:0.05em;">95% VaR (daily)</div>
                            <div style="font-size: 1.05rem; font-weight: 700;">{_qs['var95']:.3f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.65rem; color: #8ab4d4; text-transform:uppercase; letter-spacing:0.05em;">Cum. Rtn (proxy)</div>
                            <div style="font-size: 1.05rem; font-weight: 700; color:{_cc};">{_qs['cum']:+.2f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    except Exception:
        st.info("Portfolio data not yet available — run the notebook to generate outputs.")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Navigation Grid ───────────────────────────────────────────────────────
    st.markdown("<div class='home-section-label'>Explore the Dashboard</div>", unsafe_allow_html=True)

    _NAV_ITEMS = [
        ("Pipeline Health", "🔧", "System status, health checks, and execution log"),
        ("Data Load",       "📂", "Inspect raw yield data, country coverage, and time-series"),
        ("PCA & Regime",    "📊", "Factor loadings, PC scores, and GMM market regimes"),
        ("VaR Engine",      "⚠️",  "P&L bands with parametric, historical and Monte Carlo VaR"),
        ("Portfolios",      "💼", "Full analytics: performance, risk stats, DV01, and KRD"),
        ("Alert History",   "🔔", "Regime-shift and factor-alert log with severity filters"),
        ("Daily Briefings", "📝", "LLM-generated PM briefings for key market dates"),
    ]

    for _row in [_NAV_ITEMS[:4], _NAV_ITEMS[4:]]:
        _nav_cols = st.columns(4)
        for _j, (_pg, _ic, _ds) in enumerate(_row):
            with _nav_cols[_j]:
                with st.container(border=True):
                    st.markdown(f"**{_ic}&nbsp; {_pg}**")
                    st.caption(_ds)
                    if st.button("Open →", key=f"navbtn_{_pg}", use_container_width=True):
                        st.session_state[_NAV_KEY] = _pg
                        st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Pipeline Health ───────────────────────────────────────────────────────────
elif page == "Pipeline Health":
    health_path = OUT / "health_check.json"
    if not health_path.exists():
        st.warning("health_check.json not found. Run Module 4 (Pipeline Health Monitor) first.")
    else:
        with open(health_path) as f:
            checks = json.load(f)

        COLOR_MAP = {
            "GREEN":  ("🟢", "hc-status' style='color:#16a34a", "pill-green"),
            "YELLOW": ("🟡", "hc-status' style='color:#d97706", "pill-yellow"),
            "RED":    ("🔴", "hc-status' style='color:#dc2626", "pill-red"),
        }

        cols = st.columns(len(checks))
        for i, check in enumerate(checks):
            icon, style, _ = COLOR_MAP[check["status"]]
            with cols[i]:
                st.markdown(f"""
                <div class='health-card'>
                    <div class='hc-icon'>{icon}</div>
                    <div class='hc-name'>{check['check']}</div>
                    <div class='{style}'>{check['status']}</div>
                    <div class='hc-detail'>{check['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

    log_path = OUT / "pipeline_log.json"
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'><h3>Step-by-step Execution Log</h3>", unsafe_allow_html=True)

        df_log = pd.DataFrame(log)

        def fmt_status(s):
            if s == "success":
                return "✅ success"
            return "❌ failure"

        df_log["status"] = df_log["status"].map(fmt_status)
        st.dataframe(
            df_log[["step", "status", "runtime_seconds", "output_shape", "error"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "step": st.column_config.TextColumn("Step", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "runtime_seconds": st.column_config.NumberColumn("Runtime (s)", format="%.2f", width="small"),
                "output_shape": st.column_config.TextColumn("Output Shape", width="small"),
                "error": st.column_config.TextColumn("Error", width="large"),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ── Data Load ─────────────────────────────────────────────────────────────────
elif page == "Data Load":
    COUNTRIES = ["Brazil", "Mexico", "South Africa", "Poland", "Colombia", "Hungary", "Romania"]
    FLAG = {"Brazil": "🇧🇷", "Mexico": "🇲🇽", "South Africa": "🇿🇦", "Poland": "🇵🇱",
             "Colombia": "🇨🇴", "Hungary": "🇭🇺", "Romania": "🇷🇴"}

    summary_rows = []
    country_dfs = {}
    missing = []
    for country in COUNTRIES:
        path = OUT / f"{country}.csv"
        if not path.exists():
            missing.append(country)
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        country_dfs[country] = df
        summary_rows.append({
            "Country": f"{FLAG.get(country, '')} {country}",
            "Start": df.index.min().strftime("%Y-%m-%d"),
            "End":   df.index.max().strftime("%Y-%m-%d"),
            "Obs":   len(df),
            "Maturities": ", ".join(df.columns.tolist()),
        })

    if missing:
        st.warning(f"Missing output CSVs for: {', '.join(missing)}. Run the notebook first.")

    if summary_rows:
        st.markdown("<div class='section-card'><h3>Loaded Data Summary</h3>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Country":    st.column_config.TextColumn("Country",    width="medium"),
                "Start":      st.column_config.TextColumn("Start",      width="small"),
                "End":        st.column_config.TextColumn("End",        width="small"),
                "Obs":        st.column_config.NumberColumn("Obs",      width="small"),
                "Maturities": st.column_config.TextColumn("Maturities", width="large"),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><h3>Yield Levels — Select Country</h3>", unsafe_allow_html=True)
        sel_country = st.selectbox("Country", list(country_dfs.keys()),
                                   format_func=lambda c: f"{FLAG.get(c, '')} {c}",
                                   label_visibility="collapsed")
        sel_df = country_dfs[sel_country]
        st.line_chart(sel_df, use_container_width=True, height=320)
        st.caption(f"Yield levels (%) — {sel_country} — {len(sel_df)} observations")
        st.markdown("</div>", unsafe_allow_html=True)

# ── PCA & Regime ──────────────────────────────────────────────────────────────
elif page == "PCA & Regime":
    PCA_COUNTRY_FILES = {
        "Brazil":       OUT / "pca_scores_brazil.png",
        "Mexico":       OUT / "pca_scores_mexico.png",
        "South Africa": OUT / "pca_scores_south_africa.png",
        "Poland":       OUT / "pca_scores_poland.png",
        "Colombia":     OUT / "pca_scores_colombia.png",
        "Hungary":      OUT / "pca_scores_hungary.png",
        "Romania":      OUT / "pca_scores_romania.png",
    }

    tab1, tab2, tab3, tab4 = st.tabs(["Loadings", "PC Scores", "Explained Variance", "Regime"])

    with tab1:
        st.subheader("Per-country PCA loadings")
        img = OUT / "pca_loadings.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.warning("pca_loadings.png not found. Run the notebook first.")

        st.subheader("Panel PCA loadings (cross-country)")
        img2 = OUT / "pca_panel_loadings.png"
        if img2.exists():
            st.image(str(img2), use_container_width=True)
        else:
            st.warning("pca_panel_loadings.png not found.")

    with tab2:
        st.subheader("PC Score time series — select country")
        FLAG2 = {"Brazil": "🇧🇷", "Mexico": "🇲🇽", "South Africa": "🇿🇦", "Poland": "🇵🇱",
                  "Colombia": "🇨🇴", "Hungary": "🇭🇺", "Romania": "🇷🇴"}
        sel = st.selectbox("Country", list(PCA_COUNTRY_FILES.keys()),
                            format_func=lambda c: f"{FLAG2.get(c, '')} {c}",
                            label_visibility="collapsed")
        img3 = PCA_COUNTRY_FILES[sel]
        if img3.exists():
            st.image(str(img3), use_container_width=True)
        else:
            st.warning(f"PC scores chart not found for {sel}. Run the notebook first.")

    with tab3:
        st.subheader("Explained variance by country and panel")
        img4 = OUT / "pca_explained_variance.png"
        if img4.exists():
            st.image(str(img4), use_container_width=True)
        else:
            st.warning("pca_explained_variance.png not found.")

    with tab4:
        st.subheader("GMM Regime classification over time")
        img5 = OUT / "regime_classification.png"
        if img5.exists():
            st.image(str(img5), use_container_width=True)
        else:
            st.warning("regime_classification.png not found. Run the notebook first.")

# ── VaR Engine ────────────────────────────────────────────────────────────────
elif page == "VaR Engine":
    img_var = OUT / "var_pnl_bands.png"
    if not img_var.exists():
        st.warning("var_pnl_bands.png not found. Run Module 2 (VaR Engine) in the notebook first.")
    else:
        st.markdown("<div class='section-card'><h3>Portfolio P&L with VaR / CVaR Bands</h3>", unsafe_allow_html=True)
        st.image(str(img_var), use_container_width=True)
        st.caption(
            "LC Fund P&L proxy (duration approximation: ΔP/P ≈ −D_eff × weighted_avg_Δy/100) "
            "with parametric, historical and Monte Carlo VaR/CVaR bands. "
            "Backtested via Kupiec POF and Christoffersen independence tests."
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ── Daily Briefings ───────────────────────────────────────────────────────────
elif page == "Daily Briefings":
    briefings_path = OUT / "sample_briefings.json"
    if not briefings_path.exists():
        st.warning("sample_briefings.json not found. Run Module 3 (Daily Briefing Engine) first.")
    else:
        with open(briefings_path) as f:
            briefings = json.load(f)

        EVENT_META = {
            "2022-03-18": ("Post-Russia invasion EM stress",      "Elevated EM sovereign risk amid sanctions fallout and commodity shock"),
            "2022-09-23": ("UK gilt crisis / EM contagion",        "LDI-driven gilt sell-off sparking global EM spread widening"),
            "2023-06-15": ("Fed hiking cycle plateau",             "Terminal rate debate peaks; DM-EM rate differential at cycle extremes"),
            "2024-09-18": ("Fed pivot — first cut",                "FOMC delivers first 50bp cut, triggering EM local currency rally"),
            "2025-01-22": ("Trump 2.0 early policy moves",         "Dollar strength and tariff risk repricing in EM FX and rates"),
        }

        dates = list(briefings.keys())

        def label(d):
            meta = EVENT_META.get(d)
            return f"{d}  —  {meta[0]}" if meta else d

        left, right = st.columns([1, 2])
        with left:
            st.markdown("<div class='section-card'><h3>Select Date</h3>", unsafe_allow_html=True)
            selected_idx = st.radio(
                "",
                range(len(dates)),
                format_func=lambda i: label(dates[i]),
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            selected = dates[selected_idx]
            meta = EVENT_META.get(selected)
            if meta:
                st.markdown(f"""
                <div class='event-context'>
                    <strong>📌 Market context:</strong> {meta[1]}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='briefing-box'>
                <div style='font-size:0.75rem; color:#94a3b8; text-transform:uppercase;
                            letter-spacing:0.08em; margin-bottom:14px;'>
                    PM Briefing &nbsp;·&nbsp; {selected}
                </div>
            """, unsafe_allow_html=True)
            st.markdown(briefings[selected])
            st.markdown("</div>", unsafe_allow_html=True)

# ── Portfolios ────────────────────────────────────────────────────────────────
elif page == "Portfolios":

    PORT_COLORS = ["#1b3a5c", "#e67e22"]

    try:
        portfolio_results = _load_portfolio_data()
    except Exception as exc:
        st.error(f"Could not load portfolio data: {exc}")
        st.stop()

    p1 = portfolio_results[0]
    p2 = portfolio_results[1]

    st.markdown("""
    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:8px;
                padding:10px 16px; font-size:0.82rem; color:#92400e; margin-bottom:16px;">
        <strong>Rate &amp; carry proxy</strong> — All P&amp;L and performance metrics
        use a yield-change duration model plus daily coupon accrual
        (ΔP/P ≈ −D_eff × Δy/100 + y_t/252).
        <strong>FX return is excluded</strong> — material for the LC fund.
        The HC fund uses local-currency yields as a proxy for its USD-denominated holdings.
        Risk metrics (VaR, vol, DV01) are internally consistent with this proxy;
        return and Sharpe figures are estimates, not NAV-based performance.
    </div>
    """, unsafe_allow_html=True)

    tab_weights, tab_perf, tab_var, tab_compare, tab_risk = st.tabs(
        ["Weights", "Cumulative Performance", "VaR", "P&L Comparison", "Risk Statistics"]
    )

    # ── Weights ──────────────────────────────────────────────────────────────
    with tab_weights:
        countries = list(p1["def"]["weights"].keys())
        raw1 = [p1["def"]["weights"][c] for c in countries]
        raw2 = [p2["def"]["weights"][c] for c in countries]
        tot1, tot2 = sum(raw1), sum(raw2)
        pct1 = [v / tot1 * 100 for v in raw1]
        pct2 = [v / tot2 * 100 for v in raw2]

        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            name=p1["def"]["name"], x=countries, y=pct1,
            marker_color=PORT_COLORS[0], text=[f"{v:.1f}%" for v in pct1],
            textposition="outside",
        ))
        fig_w.add_trace(go.Bar(
            name=p2["def"]["name"], x=countries, y=pct2,
            marker_color=PORT_COLORS[1], text=[f"{v:.1f}%" for v in pct2],
            textposition="outside",
        ))
        fig_w.update_layout(
            barmode="group",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(title="Weight (%)", gridcolor="#e2e8f0"),
            xaxis=dict(title="Country"),
            margin=dict(t=60, b=40, l=60, r=20),
            height=400,
        )
        st.markdown("<div class='section-card'><h3>Portfolio Weights by Country</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_w, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Weight table
        df_w = pd.DataFrame({
            "Country": countries,
            f"{p1['def']['name']} (%)": [f"{v:.2f}" for v in pct1],
            f"{p2['def']['name']} (%)": [f"{v:.2f}" for v in pct2],
        })
        st.markdown("<div class='section-card'><h3>Weight Table</h3>", unsafe_allow_html=True)
        st.dataframe(df_w, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Cumulative Performance ────────────────────────────────────────────────
    with tab_perf:
        common_idx = p1["pnl"].index.intersection(p2["pnl"].index)
        cum1 = (1 + p1["pnl"].loc[common_idx]).cumprod() - 1
        cum2 = (1 + p2["pnl"].loc[common_idx]).cumprod() - 1

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum1.index, y=cum1 * 100,
            name=p1["def"]["name"], line=dict(color=PORT_COLORS[0], width=2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum2.index, y=cum2 * 100,
            name=p2["def"]["name"], line=dict(color=PORT_COLORS[1], width=2),
        ))
        fig_cum.add_hline(y=0, line_color="#94a3b8", line_dash="dot", line_width=1)
        fig_cum.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(title="Cumulative Return (%)", gridcolor="#e2e8f0", zeroline=False),
            xaxis=dict(gridcolor="#e2e8f0"),
            margin=dict(t=60, b=40, l=70, r=20), height=400,
        )
        st.markdown("<div class='section-card'><h3>Cumulative Return (%)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_cum, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Rolling 60-day volatility
        vol1 = p1["pnl"].loc[common_idx].rolling(60).std() * np.sqrt(252) * 100
        vol2 = p2["pnl"].loc[common_idx].rolling(60).std() * np.sqrt(252) * 100

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol1.index, y=vol1,
            name=p1["def"]["name"], line=dict(color=PORT_COLORS[0], width=1.5),
        ))
        fig_vol.add_trace(go.Scatter(
            x=vol2.index, y=vol2,
            name=p2["def"]["name"], line=dict(color=PORT_COLORS[1], width=1.5),
        ))
        fig_vol.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(title="Annualised Volatility — 60d rolling (%)", gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0"),
            margin=dict(t=60, b=40, l=70, r=20), height=340,
        )
        st.markdown("<div class='section-card'><h3>Rolling 60-Day Volatility (annualised)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_vol, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Summary stats
        def _stats(pnl):
            ann = np.sqrt(252)
            ret = pnl.mean() * 252 * 100
            vol = pnl.std() * ann * 100
            sharpe = (pnl.mean() / pnl.std()) * ann if pnl.std() > 0 else np.nan
            cum = ((1 + pnl).cumprod() - 1).iloc[-1] * 100
            roll = (1 + pnl).cumprod()
            dd = ((roll / roll.cummax()) - 1).min() * 100
            return {"Ann. Return (%)": f"{ret:.2f}", "Ann. Vol (%)": f"{vol:.2f}",
                    "Sharpe": f"{sharpe:.2f}", "Total Return (%)": f"{cum:.2f}",
                    "Max Drawdown (%)": f"{dd:.2f}"}

        s1, s2 = _stats(p1["pnl"]), _stats(p2["pnl"])
        metrics = list(s1.keys())
        df_stats = pd.DataFrame({
            "Metric": metrics,
            p1["def"]["name"]: [s1[m] for m in metrics],
            p2["def"]["name"]: [s2[m] for m in metrics],
        })
        st.markdown("<div class='section-card'><h3>Performance Summary</h3>", unsafe_allow_html=True)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── VaR ──────────────────────────────────────────────────────────────────
    with tab_var:
        conf_levels = [0.95, 0.99]

        def _parametric_var(pnl, alpha):
            mu, sigma = pnl.mean(), pnl.std()
            var  = -(mu + norm.ppf(alpha) * sigma)
            cvar = -(mu - sigma * norm.pdf(norm.ppf(alpha)) / (1 - alpha))
            return var * 100, cvar * 100

        def _historical_var(pnl, alpha):
            var  = float(-np.quantile(pnl, 1 - alpha))
            tail = pnl[pnl <= np.quantile(pnl, 1 - alpha)]
            cvar = float(-tail.mean()) if len(tail) > 0 else np.nan
            return var * 100, cvar * 100

        rows = []
        for conf in conf_levels:
            for pdef_key, pnl in [(p1["def"]["name"], p1["pnl"]),
                                   (p2["def"]["name"], p2["pnl"])]:
                pvar_p, pcvar_p = _parametric_var(pnl, conf)
                pvar_h, pcvar_h = _historical_var(pnl, conf)
                rows.append({
                    "Portfolio": pdef_key,
                    "Confidence": f"{conf*100:.0f}%",
                    "Param VaR (%)": f"{pvar_p:.3f}",
                    "Param CVaR (%)": f"{pcvar_p:.3f}",
                    "Hist VaR (%)": f"{pvar_h:.3f}",
                    "Hist CVaR (%)": f"{pcvar_h:.3f}",
                })

        df_var = pd.DataFrame(rows)
        st.markdown("<div class='section-card'><h3>Daily VaR & CVaR (as % of portfolio value)</h3>", unsafe_allow_html=True)
        st.dataframe(df_var, use_container_width=True, hide_index=True)
        st.caption("Parametric VaR assumes normally distributed daily P&L. Historical VaR uses empirical quantiles.")
        st.markdown("</div>", unsafe_allow_html=True)

        # VaR bar chart
        fig_var = go.Figure()
        for conf in conf_levels:
            subset = df_var[df_var["Confidence"] == f"{conf*100:.0f}%"]
            fig_var.add_trace(go.Bar(
                name=f"Param VaR {conf*100:.0f}%",
                x=subset["Portfolio"].tolist(),
                y=subset["Param VaR (%)"].astype(float).tolist(),
                marker_color=PORT_COLORS if conf == 0.95 else ["#5b7fa6", "#f0a864"],
            ))
        fig_var.update_layout(
            barmode="group",
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            yaxis=dict(title="Daily VaR (%)", gridcolor="#e2e8f0"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=60, b=40, l=60, r=20), height=360,
        )
        st.markdown("<div class='section-card'><h3>Parametric VaR Comparison</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_var, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── P&L Comparison ───────────────────────────────────────────────────────
    with tab_compare:
        common_idx2 = p1["pnl"].index.intersection(p2["pnl"].index)
        pnl1 = p1["pnl"].loc[common_idx2] * 100
        pnl2 = p2["pnl"].loc[common_idx2] * 100

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=pnl1.index, y=pnl1,
            name=p1["def"]["name"],
            line=dict(color=PORT_COLORS[0], width=1),
            opacity=0.85,
        ))
        fig_pnl.add_trace(go.Scatter(
            x=pnl2.index, y=pnl2,
            name=p2["def"]["name"],
            line=dict(color=PORT_COLORS[1], width=1),
            opacity=0.85,
        ))
        fig_pnl.add_hline(y=0, line_color="#94a3b8", line_dash="dot", line_width=1)
        fig_pnl.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(title="Daily P&L (%)", gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0"),
            margin=dict(t=60, b=40, l=70, r=20), height=400,
        )
        st.markdown("<div class='section-card'><h3>Daily P&L Overlay</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_pnl, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Difference series
        diff = (pnl1 - pnl2).dropna()
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(
            x=diff.index, y=diff,
            name="P1 − P2",
            marker_color=np.where(diff >= 0, PORT_COLORS[0], PORT_COLORS[1]).tolist(),
        ))
        fig_diff.add_hline(y=0, line_color="#94a3b8", line_dash="dot", line_width=1)
        fig_diff.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            yaxis=dict(title="P1 − P2 daily P&L (%)", gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0"),
            margin=dict(t=40, b=40, l=70, r=20), height=320,
            showlegend=False,
        )
        st.markdown("<div class='section-card'><h3>Daily P&L Difference (Portfolio 1 − Portfolio 2)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_diff, use_container_width=True)
        corr = float(pnl1.corr(pnl2))
        st.caption(f"Portfolio correlation: {corr:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Risk Statistics ───────────────────────────────────────────────────────
    with tab_risk:

        @st.cache_data(show_spinner="Loading yield levels…")
        def _load_yield_levels():
            cfg = load_config()
            all_c = cfg["countries"]["local_currency"] + cfg["countries"]["hard_currency"]
            excluded = cfg.get("excluded_series", {})
            out = {}
            for country in all_c:
                try:
                    df = load_country_yields(country, "data/raw")
                    excl = excluded.get(country, [])
                    df = df.drop(columns=[c for c in excl if c in df.columns])
                    out[country] = df
                except Exception:
                    pass
            return out

        try:
            yield_levels = _load_yield_levels()
        except Exception as _ye:
            st.warning(f"Could not load yield levels: {_ye}")
            yield_levels = {}

        def _fmt(val, fmt=".2f"):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:{fmt}}"

        def _risk_stats(pdef, pnl, lvls, rf_data=None):
            raw_w = pdef["weights"]
            tot_w = sum(raw_w.values())
            w = {k: v / tot_w for k, v in raw_w.items()}
            D = float(pdef["effective_duration"])
            mat = pdef["benchmark_maturity"]
            mat_n = int(mat[:-1])
            n = len(pnl)

            # ── Return metrics ──────────────────────────────────────────────
            cum_log = float(np.log1p(pnl).sum() * 100)
            ann_ret = float(((1 + pnl).prod() ** (252 / n) - 1) * 100)

            # Carry: portfolio-weighted latest benchmark yield
            c_vals = {}
            for c in w:
                if c in lvls and mat in lvls[c].columns:
                    s = lvls[c][mat].dropna()
                    if len(s) > 0:
                        c_vals[c] = float(s.iloc[-1])
            if c_vals:
                ws = sum(w[c] for c in c_vals)
                carry = sum(w[c] * c_vals[c] for c in c_vals) / ws
            else:
                carry = np.nan

            # Per-country modified duration — par bond approximation
            # MD_i = [1 − (1 + y_i)^(−T)] / y_i  (par-priced bond, annual compounding)
            def _par_md(yield_pct, T):
                y = yield_pct / 100
                return float(T) if y <= 0 else (1 - (1 + y) ** (-T)) / y

            md_by_c = {c: _par_md(c_vals[c], mat_n) for c in c_vals}
            for c in w:               # fallback to config D_eff if no yield data
                if c not in md_by_c:
                    md_by_c[c] = D

            # Roll-down: per-country MD × slope_per_year toward next shorter maturity
            rd_vals = {}
            for c in w:
                if c not in lvls:
                    continue
                avail_nums = sorted([int(x[:-1]) for x in lvls[c].columns])
                shorter = [m for m in avail_nums if m < mat_n]
                if not shorter:
                    continue
                ns = max(shorter)
                ns_str = f"{ns}Y"
                sub = lvls[c][[mat, ns_str]].dropna()
                if len(sub) == 0:
                    continue
                row = sub.iloc[-1]
                slope = (row[mat] - row[ns_str]) / (mat_n - ns)
                rd_vals[c] = md_by_c[c] * slope
            if rd_vals:
                ws_rd = sum(w[c] for c in rd_vals)
                rolldown = sum(w[c] * rd_vals[c] for c in rd_vals) / ws_rd
            else:
                rolldown = np.nan

            # ── Risk metrics ────────────────────────────────────────────────
            ann_vol = float(pnl.std() * np.sqrt(252) * 100)
            cum_s = (1 + pnl).cumprod()
            max_dd = float(((cum_s / cum_s.cummax()) - 1).min() * 100)

            # Duration & bond analytics (portfolio-weighted from per-country par bond MD)
            aum = float(pdef.get("aum_eur", 0))

            ws_md  = sum(w[c] for c in md_by_c)
            mod_dur = sum(w[c] * md_by_c[c] for c in md_by_c) / ws_md

            dv01     = mod_dur * 0.01                          # % of NAV per 1bp shift
            dv01_eur = mod_dur * 0.0001 * aum if aum else np.nan  # EUR per 1bp shift

            ytm = carry  # latest portfolio-weighted benchmark yield as YTM proxy

            # Per-country convexity: C_i = D_mac_i × (D_mac_i + 1) / (1 + y_i)²
            conv_by_c = {}
            for c in md_by_c:
                if c in c_vals:
                    y_f_c   = c_vals[c] / 100
                    d_mac_c = md_by_c[c] * (1 + y_f_c)
                    conv_by_c[c] = d_mac_c * (d_mac_c + 1) / (1 + y_f_c) ** 2
            if conv_by_c:
                ws_cv     = sum(w[c] for c in conv_by_c)
                convexity = sum(w[c] * conv_by_c[c] for c in conv_by_c) / ws_cv
            else:
                convexity = np.nan

            # Yield curve slope: (max_maturity − min_maturity) yield, portfolio-weighted
            sl_vals = {}
            for c in w:
                if c not in lvls:
                    continue
                lr = lvls[c].dropna(how="all").iloc[-1].dropna()
                avail = sorted(lr.index, key=lambda x: int(x[:-1]))
                if len(avail) >= 2:
                    sl_vals[c] = float(lr[avail[-1]] - lr[avail[0]])
            if sl_vals:
                ws_sl = sum(w[c] for c in sl_vals)
                yc_slope = sum(w[c] * sl_vals[c] for c in sl_vals) / ws_sl
            else:
                yc_slope = np.nan

            # Key-rate duration by country: w_i × MD_i (per-country par bond duration)
            krd = {c: w[c] * md_by_c.get(c, D) for c in w}

            # ── Ratios (rf = 0 baseline) ─────────────────────────────────────
            sharpe_zero = (ann_ret / 100) / (ann_vol / 100) if ann_vol > 0 else np.nan
            ds_zero = float(np.mean(np.minimum(pnl, 0.0) ** 2))
            sortino_zero = (ann_ret / 100) / (np.sqrt(ds_zero) * np.sqrt(252)) if ds_zero > 0 else np.nan
            calmar = (ann_ret / 100) / abs(max_dd / 100) if max_dd != 0 else np.nan

            # ── Ratios (rf = €STR) ───────────────────────────────────────────
            current_estr = np.nan
            current_sofr = np.nan
            sharpe = sharpe_zero
            sortino = sortino_zero
            avg_estr = np.nan
            if rf_data is not None:
                try:
                    rf_estr = align_rf_to_pnl(rf_data, pnl, column="estr_pct")
                    common = pnl.index.intersection(rf_estr.index)
                    excess = pnl.loc[common] - rf_estr.loc[common]
                    n_exc = len(excess)
                    ann_exc = float(((1 + excess).prod() ** (252 / n_exc) - 1) * 100)
                    exc_vol = float(excess.std() * np.sqrt(252) * 100)
                    sharpe = (ann_exc / 100) / (exc_vol / 100) if exc_vol > 0 else np.nan
                    ds_rf = float(np.mean(np.minimum(excess, 0.0) ** 2))
                    sortino = (ann_exc / 100) / (np.sqrt(ds_rf) * np.sqrt(252)) if ds_rf > 0 else np.nan
                    avg_estr = float(rf_data["estr_pct"].reindex(pnl.index, method="ffill").dropna().mean())
                    current_estr = float(rf_data["estr_pct"].dropna().iloc[-1])
                    current_sofr = float(rf_data["sofr_pct"].dropna().iloc[-1])
                except Exception:
                    pass

            # ── VaR / CVaR ──────────────────────────────────────────────────
            mu_p, sig_p = pnl.mean(), pnl.std()
            np.random.seed(42)
            sims = np.random.normal(mu_p, sig_p, 50_000)
            var_rows = []
            for alpha in [0.05, 0.10]:
                z = norm.ppf(alpha)
                pv  = -(mu_p + z * sig_p) * 100
                pcv = -(mu_p - sig_p * norm.pdf(-z) / alpha) * 100
                q   = float(np.quantile(pnl, alpha))
                hv  = -q * 100
                tmask = pnl <= q
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

            # Dollar VaR rows (reuse var_rows, scale by AUM)
            var_rows_eur = []
            for vr in var_rows:
                var_rows_eur.append({
                    **vr,
                    "Param VaR (EUR)":  vr["Param VaR (%)"]  / 100 * aum if aum else np.nan,
                    "Hist VaR (EUR)":   vr["Hist VaR (%)"]   / 100 * aum if aum else np.nan,
                    "MC VaR (EUR)":     vr["MC VaR (%)"]     / 100 * aum if aum else np.nan,
                    "Param CVaR (EUR)": vr["Param CVaR (%)"] / 100 * aum if aum else np.nan,
                })

            return dict(
                cum_log=cum_log, ann_ret=ann_ret, carry=carry, rolldown=rolldown,
                ann_vol=ann_vol, max_dd=max_dd,
                sharpe=sharpe, sortino=sortino,
                sharpe_zero=sharpe_zero, sortino_zero=sortino_zero,
                calmar=calmar, mod_dur=mod_dur, dv01=dv01, dv01_eur=dv01_eur,
                aum=aum, convexity=convexity, md_by_c=md_by_c,
                ytm=ytm, yc_slope=yc_slope, krd=krd,
                var_rows=var_rows, var_rows_eur=var_rows_eur,
                current_estr=current_estr, current_sofr=current_sofr, avg_estr=avg_estr,
            )

        @st.cache_data(show_spinner="Loading risk-free rates…")
        def _load_rf_data():
            cfg = load_config()
            key_path = cfg.get("fred", {}).get("key_path", "private/fred_key.txt")
            out_path  = cfg.get("fred", {}).get("output_path", "data/output/risk_free_rates.csv")
            try:
                key = open(key_path).read().strip()
                return load_risk_free_rates(out_path, fred_api_key=key)
            except Exception:
                return None

        rf_data = _load_rf_data()
        rs1 = _risk_stats(p1["def"], p1["pnl"], yield_levels, rf_data)
        rs2 = _risk_stats(p2["def"], p2["pnl"], yield_levels, rf_data)
        pn1, pn2 = p1["def"]["name"], p2["def"]["name"]

        FLAG_R = {"Brazil": "🇧🇷", "Mexico": "🇲🇽", "South Africa": "🇿🇦", "Poland": "🇵🇱",
                  "Colombia": "🇨🇴", "Hungary": "🇭🇺", "Romania": "🇷🇴"}

        # ── 1. Return Metrics ──────────────────────────────────────────────
        st.markdown("<div class='section-card'><h3>Return Metrics</h3>", unsafe_allow_html=True)
        df_ret = pd.DataFrame([
            ("Cumulative Log Return (%)",  _fmt(rs1["cum_log"]),   _fmt(rs2["cum_log"])),
            ("Annualised Return (%)",       _fmt(rs1["ann_ret"]),   _fmt(rs2["ann_ret"])),
            ("Carry — Wtd Avg Yield (%)",  _fmt(rs1["carry"]),     _fmt(rs2["carry"])),
            ("Roll-Down Return (est. %)",  _fmt(rs1["rolldown"]),  _fmt(rs2["rolldown"])),
        ], columns=["Metric", pn1, pn2])
        st.dataframe(df_ret, use_container_width=True, hide_index=True)
        st.caption(
            "Carry = portfolio-weighted latest benchmark yield. "
            "Roll-down ≈ D_eff × annual curve slope to next shorter maturity, portfolio-weighted. "
            "South Africa excluded from roll-down (no maturity shorter than 5Y available in data)."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 2. Risk & Ratio Metrics ────────────────────────────────────────
        estr_now = rs1["current_estr"]
        sofr_now = rs1["current_sofr"]
        avg_e    = rs1["avg_estr"]
        has_rf   = not np.isnan(estr_now)
        rf_label = f"€STR ({estr_now:.2f}%)" if has_rf else "0 (no rf data)"

        # Rate context banner
        if has_rf:
            st.markdown(f"""
            <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                        padding:10px 16px;font-size:0.85rem;color:#1e40af;margin-bottom:12px;'>
                <strong>Risk-free rates (FRED, latest):</strong>
                &nbsp; €STR = <strong>{estr_now:.3f}%</strong>
                &nbsp;|&nbsp; SOFR = <strong>{sofr_now:.3f}%</strong>
                &nbsp;|&nbsp; Average €STR over portfolio history = <strong>{avg_e:.3f}%</strong>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-card'><h3>Risk & Ratio Metrics</h3>", unsafe_allow_html=True)
        df_risk_tbl = pd.DataFrame([
            ("Annualised Volatility (%)",       _fmt(rs1["ann_vol"]),       _fmt(rs2["ann_vol"])),
            ("Maximum Drawdown (%)",            _fmt(rs1["max_dd"]),        _fmt(rs2["max_dd"])),
            (f"Sharpe Ratio (rf = {rf_label})", _fmt(rs1["sharpe"]),        _fmt(rs2["sharpe"])),
            (f"Sortino Ratio (rf = {rf_label})",_fmt(rs1["sortino"]),       _fmt(rs2["sortino"])),
            ("Sharpe Ratio (rf = 0, ref)",      _fmt(rs1["sharpe_zero"]),   _fmt(rs2["sharpe_zero"])),
            ("Sortino Ratio (MAR = 0, ref)",    _fmt(rs1["sortino_zero"]),  _fmt(rs2["sortino_zero"])),
            ("Calmar Ratio",                    _fmt(rs1["calmar"]),        _fmt(rs2["calmar"])),
        ], columns=["Metric", pn1, pn2])
        st.dataframe(df_risk_tbl, use_container_width=True, hide_index=True)
        st.caption(
            f"Sharpe and Sortino use daily excess returns over €STR (EUR risk-free, source: FRED). "
            f"Sortino denominator = annualised downside semi-deviation of excess returns (√(E[min(excess,0)²]) × √252). "
            f"Calmar = annualised total return / |max drawdown|. "
            f"rf=0 rows shown for reference."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 3. Bond Analytics ──────────────────────────────────────────────
        def _fmt_eur(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            if abs(val) >= 1_000_000:
                return f"€{val/1_000_000:,.3f}M"
            return f"€{val:,.0f}"

        st.markdown("<div class='section-card'><h3>Bond Analytics</h3>", unsafe_allow_html=True)
        df_bond = pd.DataFrame([
            ("AUM (EUR)",                        _fmt_eur(rs1["aum"]),          _fmt_eur(rs2["aum"])),
            ("Modified Duration (yrs)",          _fmt(rs1["mod_dur"]),          _fmt(rs2["mod_dur"])),
            ("DV01 (% of NAV per 1bp parallel)", _fmt(rs1["dv01"], ".4f"),      _fmt(rs2["dv01"], ".4f")),
            ("DV01 (EUR per 1bp parallel)",      _fmt_eur(rs1["dv01_eur"]),     _fmt_eur(rs2["dv01_eur"])),
            ("Convexity — approx (yrs²)",        _fmt(rs1["convexity"], ".1f"), _fmt(rs2["convexity"], ".1f")),
            ("YTM — Wtd Avg Benchmark (%)",      _fmt(rs1["ytm"]),              _fmt(rs2["ytm"])),
            ("Yield Curve Slope (long−short, %)",_fmt(rs1["yc_slope"]),         _fmt(rs2["yc_slope"])),
        ], columns=["Metric", pn1, pn2])
        st.dataframe(df_bond, use_container_width=True, hide_index=True)
        st.caption(
            "Modified Duration = portfolio-weighted average of per-country par bond duration: "
            "MD_i = [1 − (1 + y_i)^(−T)] / y_i (annual compounding, bond priced at par). "
            "DV01 (EUR) = MD_portfolio × 0.0001 × AUM. "
            "Convexity per country: C_i = D_mac_i × (D_mac_i+1) / (1+y_i)², then portfolio-weighted. "
            "YTM = portfolio-weighted latest benchmark yield."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 4. VaR / CVaR ─────────────────────────────────────────────────
        st.markdown("<div class='section-card'><h3>VaR & CVaR — Daily (% of portfolio NAV)</h3>", unsafe_allow_html=True)
        var_long = []
        for pname_v, rs_v in [(pn1, rs1), (pn2, rs2)]:
            for vrow in rs_v["var_rows"]:
                var_long.append({
                    "Portfolio":      pname_v,
                    "α":              vrow["α"],
                    "Confidence":     vrow["Confidence"],
                    "Param VaR (%)":  f"{vrow['Param VaR (%)']:.4f}",
                    "Param CVaR (%)": f"{vrow['Param CVaR (%)']:.4f}",
                    "Hist VaR (%)":   f"{vrow['Hist VaR (%)']:.4f}",
                    "Hist CVaR (%)":  f"{vrow['Hist CVaR (%)']:.4f}",
                    "MC VaR (%)":     f"{vrow['MC VaR (%)']:.4f}",
                })
        st.dataframe(pd.DataFrame(var_long), use_container_width=True, hide_index=True)
        st.caption(
            "Parametric: Normal P&L assumption, z-score method. "
            "CVaR (Normal) = −(μ − σ × φ(z_α) / α). "
            "Historical: empirical quantile / tail mean. "
            "Monte Carlo: 50,000 draws from N(μ, σ²), seed = 42."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><h3>VaR & CVaR — Daily (EUR, based on AUM)</h3>", unsafe_allow_html=True)
        var_eur_long = []
        for pname_v, rs_v in [(pn1, rs1), (pn2, rs2)]:
            for vrow in rs_v["var_rows_eur"]:
                var_eur_long.append({
                    "Portfolio":        pname_v,
                    "α":                vrow["α"],
                    "Confidence":       vrow["Confidence"],
                    "Param VaR (EUR)":  _fmt_eur(vrow["Param VaR (EUR)"]),
                    "Param CVaR (EUR)": _fmt_eur(vrow["Param CVaR (EUR)"]),
                    "Hist VaR (EUR)":   _fmt_eur(vrow["Hist VaR (EUR)"]),
                    "MC VaR (EUR)":     _fmt_eur(vrow["MC VaR (EUR)"]),
                })
        st.dataframe(pd.DataFrame(var_eur_long), use_container_width=True, hide_index=True)
        st.caption(
            f"EUR VaR = VaR (%) × AUM. "
            f"AUM: {pn1} = {_fmt_eur(rs1['aum'])}, {pn2} = {_fmt_eur(rs2['aum'])}."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 5. Key-Rate Duration by Country ───────────────────────────────
        st.markdown("<div class='section-card'><h3>Key-Rate Duration by Country (yrs)</h3>", unsafe_allow_html=True)
        all_krd_c = sorted(set(rs1["krd"]) | set(rs2["krd"]))
        krd_labels = [f"{FLAG_R.get(c, '')} {c}" for c in all_krd_c]

        fig_krd = go.Figure()
        fig_krd.add_trace(go.Bar(
            name=pn1, x=krd_labels,
            y=[rs1["krd"].get(c, 0) for c in all_krd_c],
            marker_color=PORT_COLORS[0],
            text=[f"{rs1['krd'].get(c, 0):.3f}" for c in all_krd_c],
            textposition="outside",
        ))
        fig_krd.add_trace(go.Bar(
            name=pn2, x=krd_labels,
            y=[rs2["krd"].get(c, 0) for c in all_krd_c],
            marker_color=PORT_COLORS[1],
            text=[f"{rs2['krd'].get(c, 0):.3f}" for c in all_krd_c],
            textposition="outside",
        ))
        fig_krd.update_layout(
            barmode="group",
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(title="KRD (yrs)", gridcolor="#e2e8f0"),
            margin=dict(t=60, b=40, l=60, r=20), height=380,
        )
        st.plotly_chart(fig_krd, use_container_width=True)

        krd_tbl = pd.DataFrame({
            "Country":              krd_labels,
            "MD (par bond, yrs)":   [f"{rs1['md_by_c'].get(c, 0):.3f}" for c in all_krd_c],
            f"KRD — {pn1} (yrs)":  [f"{rs1['krd'].get(c, 0):.4f}" for c in all_krd_c],
            f"KRD — {pn2} (yrs)":  [f"{rs2['krd'].get(c, 0):.4f}" for c in all_krd_c],
        })
        st.dataframe(krd_tbl, use_container_width=True, hide_index=True)
        st.caption(
            "MD (par bond) = per-country modified duration using latest 5Y yield: "
            "[1 − (1+y)^(−5)] / y. Both portfolios share the same MD_i; "
            "KRDs differ only because of different weights. "
            "KRD_i = w_i × MD_i. Sum = portfolio modified duration."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 6. Country Yield-Change Correlation Matrix ─────────────────────
        st.markdown("<div class='section-card'><h3>Country Yield-Change Correlation Matrix</h3>", unsafe_allow_html=True)
        mat_corr = p1["def"]["benchmark_maturity"]
        full_dy = {}
        for country in all_krd_c:
            if country in yield_levels and mat_corr in yield_levels[country].columns:
                full_dy[country] = yield_levels[country][mat_corr].diff().dropna()

        if len(full_dy) >= 2:
            corr_df = pd.DataFrame(full_dy).dropna().corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=[f"{FLAG_R.get(c, '')} {c}" for c in corr_df.columns],
                y=[f"{FLAG_R.get(c, '')} {c}" for c in corr_df.index],
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
                texttemplate="%{text}",
                colorbar=dict(title="ρ"),
            ))
            fig_corr.update_layout(
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                margin=dict(t=40, b=80, l=130, r=20),
                height=440,
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(
                f"Daily first-differences of {mat_corr} benchmark yields, all common available dates. "
                "Correlation is market-structural and identical for both portfolios; "
                "portfolio risk depends on this matrix weighted by each portfolio's KRD vector."
            )
        else:
            st.warning("Not enough data to build correlation matrix.")
        st.markdown("</div>", unsafe_allow_html=True)

# ── Alert History ─────────────────────────────────────────────────────────────
elif page == "Alert History":
    alert_path = OUT / "alert_history.json"
    if not alert_path.exists():
        st.warning("alert_history.json not found. Run Module 1.4 (Alert Engine) first.")
    else:
        with open(alert_path) as f:
            alerts = json.load(f)

        records = []
        for date_str, payload in alerts.items():
            for alert in payload.get("alerts", []):
                records.append({
                    "date": date_str,
                    "type": alert["type"],
                    "severity": alert["severity"],
                    "regime": payload.get("regime", ""),
                    "detail": alert.get("detail", ""),
                })

        if not records:
            st.info("No alerts found in alert_history.json.")
        else:
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", ascending=False)

            total   = len(alerts)
            high    = int((df["severity"] == "high").sum())
            med     = int((df["severity"] == "medium").sum())
            shifts  = int((df["type"] == "regime_shift").sum())

            c1, c2, c3, c4 = st.columns(4)
            for col, val, label_, color in [
                (c1, total,  "Alert Days",     "blue"),
                (c2, high,   "High Severity",  "red"),
                (c3, med,    "Medium Severity","amber"),
                (c4, shifts, "Regime Shifts",  "green"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='stat-card {color}'>
                        <div class='label'>{label_}</div>
                        <div class='value'>{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            left, right = st.columns([1, 3])
            with left:
                st.markdown("<div class='section-card'><h3>Filters</h3>", unsafe_allow_html=True)
                sev_filter = st.multiselect(
                    "Severity",
                    ["high", "medium", "low"],
                    default=["high", "medium"],
                )
                type_filter = st.multiselect(
                    "Type",
                    sorted(df["type"].unique().tolist()),
                    default=sorted(df["type"].unique().tolist()),
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                filtered = df.copy()
                if sev_filter:
                    filtered = filtered[filtered["severity"].isin(sev_filter)]
                if type_filter:
                    filtered = filtered[filtered["type"].isin(type_filter)]

                SEV_PILL = {
                    "high":   "<span class='pill pill-red'>high</span>",
                    "medium": "<span class='pill pill-yellow'>medium</span>",
                    "low":    "<span class='pill pill-gray'>low</span>",
                }

                st.markdown(f"<div style='font-size:0.82rem; color:#64748b; margin-bottom:8px;'>{len(filtered)} alerts shown</div>", unsafe_allow_html=True)
                st.dataframe(
                    filtered[["date", "severity", "type", "regime", "detail"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "date":     st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                        "severity": st.column_config.TextColumn("Severity", width="small"),
                        "type":     st.column_config.TextColumn("Type",     width="medium"),
                        "regime":   st.column_config.TextColumn("Regime",   width="small"),
                        "detail":   st.column_config.TextColumn("Detail",   width="large"),
                    },
                )
