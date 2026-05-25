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
from src.data import load_briefings as _load_briefings_from_disk, data_version
from src.data.var_artifacts import (
    load_alert_history as _load_alert_history_from_disk,
    load_country_outputs as _load_country_outputs_from_disk,
    load_decomposition as _load_decomposition_from_disk,
    load_health_check as _load_health_check_from_disk,
    load_multi_nu as _load_multi_nu_from_disk,
    load_pipeline_log as _load_pipeline_log_from_disk,
    load_stress_data as _load_stress_data_from_disk,
)
from src.risk_free import load_risk_free_rates, align_rf_to_pnl
from src.services.portfolios import build_portfolio_views, compute_quick_stats
from src.services.risk_stats import compute_risk_stats
from src.report_generator import get_available_quarters, generate_quarterly_report
from src.ui_theme import apply_theme
import chatbot

st.set_page_config(
    page_title="EM FI Intelligence",
    # Empty transparent SVG → blank tab icon (suppresses the Streamlit default).
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'/>",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Design tokens ─────────────────────────────────────────────────── */
    :root {
        --font-ui:        "Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont,
                          "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --c-bg:           #f6f8fb;
        --c-surface:      #ffffff;
        --c-border:       #e2e8f0;
        --c-border-soft:  #eef2f7;
        --c-text:         #0f172a;
        --c-text-muted:   #64748b;
        --c-navy-900:     #0d1b2a;
        --c-navy-700:     #1b3a5c;
        --c-azure-300:    #7ec8e3;
        --c-azure-200:    #8ab4d4;
        --c-warn:         #b45309;
        --c-warn-soft:    #fef3c7;
        --c-warn-border:  #fcd34d;
        --c-info:         #1e40af;
        --c-info-soft:    #eff6ff;
        --c-info-border:  #bfdbfe;
        --e-1:            0 1px 2px rgba(15,23,42,0.04), 0 1px 3px rgba(15,23,42,0.06);
        --e-2:            0 2px 4px rgba(15,23,42,0.05), 0 4px 12px rgba(15,23,42,0.08);
    }

    /* ── Base font ── */
    /* NOTE: never apply a wildcard font override to the sidebar —
       Streamlit uses Material Icons there and !important would break them. */
    html, body,
    [data-testid="stMain"] *,
    button, input, select, textarea,
    .stDataFrame, .stDataFrame * {
        font-family: var(--font-ui) !important;
    }

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
        background-color: var(--c-bg);
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
        box-shadow: var(--e-1);
        border-top: 4px solid #1b3a5c;
        text-align: center;
        height: 100%;
        box-sizing: border-box;
        transition: box-shadow 0.18s ease, transform 0.18s ease;
    }
    .stat-card:hover {
        box-shadow: var(--e-2);
        transform: translateY(-1px);
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
        box-shadow: var(--e-1);
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
        box-shadow: var(--e-1);
        text-align: center;
        min-height: 228px;
        box-sizing: border-box;
        flex: 1 1 auto !important;
    }
    /* Make the row stretch its columns, and propagate flex down through
       every wrapper between the column and .health-card so the card
       actually fills the row's height (not just hits the min-height floor). */
    [data-testid="stHorizontalBlock"]:has(.health-card) {
        align-items: stretch !important;
    }
    [data-testid="column"]:has(.health-card),
    [data-testid="column"]:has(.health-card) > div,
    [data-testid="column"]:has(.health-card) [data-testid="stVerticalBlock"],
    [data-testid="column"]:has(.health-card) [data-testid="stVerticalBlock"] > div,
    [data-testid="column"]:has(.health-card) [data-testid="stElementContainer"],
    [data-testid="column"]:has(.health-card) [data-testid="stMarkdown"],
    [data-testid="column"]:has(.health-card) [data-testid="stMarkdownContainer"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        min-height: 0 !important;
        height: auto !important;
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
        box-shadow: var(--e-1);
        border-left: 5px solid var(--c-azure-300);
        line-height: 1.75;
        color: var(--c-text);
        font-size: 0.93rem;
    }

    /* ── Event context (reuses info-banner tokens) ── */
    .event-context {
        background: var(--c-info-soft);
        border: 1px solid var(--c-info-border);
        border-left: 3px solid var(--c-info);
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: var(--c-info);
        margin-bottom: 16px;
        line-height: 1.55;
    }

    /* ── Disclaimer banner (caution / amber) ── */
    .disclaimer-banner {
        background: var(--c-warn-soft);
        border: 1px solid var(--c-warn-border);
        border-left: 3px solid var(--c-warn);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: var(--c-warn);
        margin-bottom: 14px;
        line-height: 1.55;
    }

    /* ── Disclaimer footnote (muted, low-emphasis) ── */
    .disclaimer-footnote {
        background: transparent;
        border-top: 1px solid #1b2a3f;
        padding: 10px 0 0 0;
        margin-top: 18px;
        font-size: 0.72rem;
        color: #5d6e83;
        line-height: 1.5;
        font-style: normal;
    }
    .disclaimer-footnote strong {
        color: #7a8a9e;
        font-weight: 600;
    }

    /* ── Info banner (contextual / blue) ── */
    .info-banner {
        background: var(--c-info-soft);
        border: 1px solid var(--c-info-border);
        border-left: 3px solid var(--c-info);
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: var(--c-info);
        margin-bottom: 12px;
        line-height: 1.55;
    }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 20px 0;
    }

    /* ── Main content text (fixes white-on-light background) ── */
    /* strong/em intentionally excluded — they inherit from parent so dark
       card inline colors (color:#fff) are not overridden. */
    section[data-testid="stMain"] p,
    section[data-testid="stMain"] li,
    section[data-testid="stMain"] ol,
    section[data-testid="stMain"] ul,
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4 {
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

    /* ── Home nav container cards — equal height ── */
    /* Step 1: stretch all columns in a nav-card row to the tallest one */
    [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) {
        align-items: stretch !important;
    }
    /* Step 2: pass height down through every intermediate wrapper */
    [data-testid="column"]:has([data-testid="stVerticalBlockBorderWrapper"]),
    [data-testid="column"]:has([data-testid="stVerticalBlockBorderWrapper"]) > div,
    [data-testid="column"]:has([data-testid="stVerticalBlockBorderWrapper"]) > div > [data-testid="stVerticalBlock"] {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        min-height: 0 !important;
    }
    /* Step 3: the bordered card fills its column */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 100%) !important;
        border: 1px solid #1b3a5c !important;
        border-radius: 10px !important;
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
        flex: 1 !important;
        height: 100% !important;
        box-sizing: border-box !important;
    }
    /* Step 4: inner block is flex col so button pins to bottom */
    [data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        flex: 1 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] .stButton {
        margin-top: auto !important;
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

    /* ── Tab headers — equal-width, fully clickable ── */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        gap: 0;
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {
        flex: 1 1 0;
        min-width: 0;
        justify-content: center;
        text-align: center;
        padding: 10px 12px;
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] {
        width: 100%;
        text-align: center;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ── App footer (custom; replaces hidden Streamlit footer) ── */
    .app-footer {
        margin: 48px 0 8px 0;
        padding-top: 16px;
        border-top: 1px solid var(--c-border);
        text-align: center;
        font-size: 0.74rem;
        letter-spacing: 0.02em;
        color: var(--c-text-muted);
    }
</style>
""", unsafe_allow_html=True)

OUT = Path("data/output")

# Editable build tag shown in the page footer. Bump on notable releases.
APP_VERSION = "1.0"

PAGES = [
    "Home", "Pipeline Health", "Data Load", "PCA & Regime",
    "VaR Engine", "Portfolios", "Alert History", "Daily Briefings",
    "Chatbot",
]
_NAV_KEY = "_nav"
_NAV_PENDING_KEY = "_nav_pending"

if _NAV_KEY not in st.session_state:
    st.session_state[_NAV_KEY] = "Home"

# Apply any pending nav request from a previous run (e.g. home-page Open buttons)
# BEFORE the sidebar radio widget is instantiated with key=_NAV_KEY — otherwise
# Streamlit raises "cannot modify widget value after instantiation".
if _NAV_PENDING_KEY in st.session_state:
    st.session_state[_NAV_KEY] = st.session_state.pop(_NAV_PENDING_KEY)


# Streamlit-cached wrappers around the Data + Services layers. The actual
# JSON/CSV parsing and portfolio carry math live in src/data/var_artifacts.py
# and src/services/portfolios.py — these wrappers only own the @st.cache_data
# session caching the dashboard needs.

# Data-version cache keys — recomputed every rerun (cheap, stat-only).
# Passing these into a cached loader as a regular (non-underscore)
# argument makes Streamlit fold them into the cache key, so the cache
# auto-invalidates whenever the notebook regenerates the data files.
# Note: persist="disk" caches survive a server restart, so they also
# survive code changes to the underlying loader modules (the cache key
# hashes the wrapper body, not the imported helpers). After pulling new
# code, use the sidebar "Refresh data" button to force a rebuild.
_OUT_VER = data_version(OUT)
_RAW_VER = data_version("data/raw")


@st.cache_data(show_spinner="Loading stress scenarios…")
def _load_stress_data(version):
    return _load_stress_data_from_disk(OUT)


@st.cache_data(show_spinner="Loading parametric-t grid…")
def _load_multi_nu(version):
    return _load_multi_nu_from_disk(OUT)


@st.cache_data(show_spinner="Loading variance decomposition…")
def _load_decomposition(version):
    return _load_decomposition_from_disk(OUT)


@st.cache_data(show_spinner="Loading portfolio data…", persist="disk")
def _load_portfolio_data(version):
    return build_portfolio_views()


@st.cache_data(show_spinner="Loading pipeline health…")
def _load_health_check(version):
    return _load_health_check_from_disk(OUT)


@st.cache_data(show_spinner="Loading execution log…")
def _load_pipeline_log(version):
    return _load_pipeline_log_from_disk(OUT)


@st.cache_data(show_spinner="Loading country data…")
def _load_country_outputs(countries, version):
    return _load_country_outputs_from_disk(list(countries), OUT)


@st.cache_data(show_spinner="Loading alert history…")
def _load_alert_history(version):
    return _load_alert_history_from_disk(OUT)


@st.cache_data(show_spinner="Loading briefings…")
def _load_briefings(version):
    return _load_briefings_from_disk(OUT / "sample_briefings.json")


@st.cache_data(show_spinner="Loading yield levels…", persist="disk")
def _load_yield_levels(version):
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


# Keyed on _OUT_VER because the rf rates live in data/output.
@st.cache_data(show_spinner="Loading risk-free rates…", persist="disk")
def _load_rf_data(version):
    cfg = load_config()
    key_path = cfg.get("fred", {}).get("key_path", "private/fred_key.txt")
    out_path  = cfg.get("fred", {}).get("output_path", "data/output/risk_free_rates.csv")
    try:
        key = open(key_path).read().strip()
        return load_risk_free_rates(out_path, fred_api_key=key)
    except Exception:
        return None


@st.cache_data(show_spinner="Preparing quarterly report…")
def _cached_report(q_start_iso: str, q_end_iso: str, _ver: str) -> bytes:
    from datetime import date as _date
    return generate_quarterly_report({
        "label": f"{q_start_iso} – {q_end_iso}",
        "start": _date.fromisoformat(q_start_iso),
        "end":   _date.fromisoformat(q_end_iso),
    })


# ── Shared UI helpers ───────────────────────────────────────────────────────────
def _csv_download(df, filename, *, key, label="⬇ Download CSV", index=False):
    """Render a compact CSV export button for a displayed table.

    Pass the raw (numeric) DataFrame, not a string-formatted display copy, so the
    exported file keeps full precision.
    """
    st.download_button(
        label,
        # utf-8-sig writes a BOM so Excel renders €, em-dashes, etc. correctly.
        df.to_csv(index=index).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
        help=f"Export this table as {filename}",
    )


def _page_footer():
    """Slim, consistent footer rendered at the bottom of every page."""
    st.markdown(
        f"""
        <div class='app-footer'>
            EM Fixed Income Intelligence Platform
            &nbsp;·&nbsp; v{APP_VERSION}
            &nbsp;·&nbsp; Internal use only — not investment advice
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    if st.button(
        "Refresh data",
        key="refresh_data_btn",
        type="secondary",
        help="Clear cached data and reload the latest analytics outputs from disk.",
    ):
        st.cache_data.clear()
        st.toast("Data caches cleared — reloading latest outputs.", icon="✅")
        st.rerun()

    if st.button(
        "Stop Server",
        key="stop_server_btn",
        type="secondary",
        help="Shut down the dashboard server and close this tab.",
    ):
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

    try:
        _home_ports = _load_portfolio_data(_RAW_VER)
        _hp1, _hp2 = _home_ports[0], _home_ports[1]

        _hqs1 = compute_quick_stats(_hp1["pnl"])
        _hqs2 = compute_quick_stats(_hp2["pnl"])

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
                    st.markdown(f"**{_pg}**")
                    st.markdown(
                        f"<div style='height:64px;font-size:0.8rem;"
                        f"color:#8ab4d4;line-height:1.45;padding:2px 0 6px 0;"
                        f"display:-webkit-box;-webkit-line-clamp:3;"
                        f"-webkit-box-orient:vertical;overflow:hidden;'>{_ds}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Open →", key=f"navbtn_{_pg}", use_container_width=True):
                        st.session_state[_NAV_PENDING_KEY] = _pg
                        st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer-footnote'>
        <strong>A note on the numbers.</strong> Returns and risk figures here come from
        a duration-based approximation on yield changes, with daily coupon accrual added.
        FX is not included, which matters quite a bit for the LC fund. For the HC fund,
        local-currency sovereign yields stand in for the actual USD holdings, so absolute
        returns are indicative rather than exact. The official factsheets are the source
        for NAV performance.
    </div>
    """, unsafe_allow_html=True)

# ── Pipeline Health ───────────────────────────────────────────────────────────
elif page == "Pipeline Health":
    checks = _load_health_check(_OUT_VER)
    if checks is None:
        st.warning("health_check.json not found. Run Module 4 (Pipeline Health Monitor) first.")
    else:
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

    log = _load_pipeline_log(_OUT_VER)
    if log is not None:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'><h3>Step-by-step Execution Log</h3>", unsafe_allow_html=True)

        df_log = pd.DataFrame(log)

        st.dataframe(
            df_log[["step", "status", "runtime_seconds", "output_shape", "error"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "step": st.column_config.TextColumn("Step", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "runtime_seconds": st.column_config.NumberColumn("Runtime (s)", format="%.2f", width="small"),
                "output_shape": st.column_config.TextColumn("Output Shape", width="small"),
                "error": st.column_config.TextColumn("Error", width="small"),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ── Data Load ─────────────────────────────────────────────────────────────────
elif page == "Data Load":
    COUNTRIES = ["Brazil", "Mexico", "South Africa", "Poland", "Colombia", "Hungary", "Romania"]
    FLAG = {"Brazil": "🇧🇷", "Mexico": "🇲🇽", "South Africa": "🇿🇦", "Poland": "🇵🇱",
             "Colombia": "🇨🇴", "Hungary": "🇭🇺", "Romania": "🇷🇴"}

    country_dfs, missing = _load_country_outputs(tuple(COUNTRIES), _OUT_VER)
    summary_rows = []
    for country, df in country_dfs.items():
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
                "Country":    st.column_config.TextColumn("Country",    width="small"),
                "Start":      st.column_config.TextColumn("Start",      width="small"),
                "End":        st.column_config.TextColumn("End",        width="small"),
                "Obs":        st.column_config.NumberColumn("Obs",      width="small"),
                "Maturities": st.column_config.TextColumn("Maturities", width="large"),
            },
        )
        # Export plain country names (the displayed table prepends a flag emoji).
        _csv_download(
            pd.DataFrame(summary_rows).assign(Country=list(country_dfs.keys())),
            "data_load_summary.csv",
            key="dl_data_summary",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><h3>Yield Levels — Select Country</h3>", unsafe_allow_html=True)
        sel_country = st.selectbox("Country", list(country_dfs.keys()),
                                   format_func=lambda c: f"{FLAG.get(c, '')} {c}",
                                   label_visibility="collapsed")
        sel_df = country_dfs[sel_country]
        # Mask non-physical yield ticks (Mexico 30Y has a handful of >95% spikes
        # from bad Investing.com prints in 2018-2019). Anything above 30% is
        # well outside even crisis-level EM sovereign yields.
        sel_df_plot = sel_df.where((sel_df >= -5) & (sel_df <= 30))
        n_clipped = int(sel_df.notna().sum().sum() - sel_df_plot.notna().sum().sum())
        st.line_chart(sel_df_plot, use_container_width=True, height=320)
        caption = f"Yield levels (%) — {sel_country} — {len(sel_df)} observations"
        if n_clipped:
            caption += f" ({n_clipped} outlier tick{'s' if n_clipped > 1 else ''} hidden)"
        st.caption(caption)
        _csv_download(
            sel_df,
            f"yields_{sel_country.replace(' ', '_')}.csv",
            key="dl_country_yields",
            index=True,
        )
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
        st.caption("How each PC weights the maturities of a single country's curve — typically PC1 ≈ level, PC2 ≈ slope, PC3 ≈ curvature.")
        img = OUT / "pca_loadings.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.warning("pca_loadings.png not found. Run the notebook first.")

        st.subheader("Panel PCA loadings (cross-country)")
        st.caption("How each PC weights every country–maturity pair across the EM panel, surfacing common factors that move yields jointly across markets.")
        img2 = OUT / "pca_panel_loadings.png"
        if img2.exists():
            st.image(str(img2), use_container_width=True)
        else:
            st.warning("pca_panel_loadings.png not found.")

    with tab2:
        st.subheader("PC Score time series — select country")
        st.caption("Daily values of each PC for the selected country — how strongly level/slope/curvature moves were expressed each day.")
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
        st.caption("Share of daily yield-change variance captured by each PC — shows how many components are needed to summarise curve dynamics.")
        img4 = OUT / "pca_explained_variance.png"
        if img4.exists():
            st.image(str(img4), use_container_width=True)
        else:
            st.warning("pca_explained_variance.png not found.")

    with tab4:
        st.subheader("GMM Regime classification over time")
        st.caption("Daily regime labels from a Gaussian Mixture fit on PC scores (k chosen by BIC) — segments history into distinct market states.")
        img5 = OUT / "regime_classification.png"
        if img5.exists():
            st.image(str(img5), use_container_width=True)
        else:
            st.warning("regime_classification.png not found. Run the notebook first.")

# ── VaR Engine ────────────────────────────────────────────────────────────────
elif page == "VaR Engine":
    tab1, tab2, tab3, tab4 = st.tabs([
        "P&L Bands", "Stressed VaR", "Parametric-t Sensitivity", "Risk Decomposition"
    ])

    with tab1:
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

    with tab2:
        data = _load_stress_data(_OUT_VER)
        if data is None:
            st.warning("Stressed VaR artifacts not found. Run Module 2 (VaR Engine) in the notebook first.")
        else:
            pnl = data["pnl"]
            windows = data["windows"]
            summary = data["summary"]
            primary = windows["primary_stress"]
            window_names = list(windows["windows"].keys())
            selected = st.selectbox(
                "Stress window to overlay",
                options=window_names,
                index=window_names.index(primary),
            )
            w = windows["windows"][selected]
            stress_slice = pnl.loc[w["start"]:w["end"]]
            VaR_full = windows["reference"]["hist_full_VaR_95"]
            VaR_param_n = windows["reference"]["parametric_normal_VaR_95"]
            VaR_stress = w["VaR_95"]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pnl, histnorm="probability density", opacity=0.5,
                marker_color="#4682B4", name=f"Full sample (n={len(pnl)})",
                nbinsx=80,
            ))
            fig.add_trace(go.Histogram(
                x=stress_slice, histnorm="probability density", opacity=0.5,
                marker_color="#DC143C", name=f"{selected} (n={len(stress_slice)})",
                nbinsx=20,
            ))
            fig.add_vline(x=-VaR_full, line_dash="dash", line_color="#4682B4",
                          annotation_text=f"Hist VaR 95% (full): {VaR_full:.2%}",
                          annotation_position="top right")
            fig.add_vline(x=-VaR_param_n, line_dash="dot", line_color="black",
                          annotation_text=f"Param normal VaR 95%: {VaR_param_n:.2%}",
                          annotation_position="top right")
            fig.add_vline(x=-VaR_stress, line_dash="dash", line_color="#DC143C",
                          annotation_text=f"Stressed VaR 95% ({selected}): {VaR_stress:.2%}",
                          annotation_position="top right")
            fig.update_layout(
                barmode="overlay",
                title="LC Fund P&L Distribution: Full Sample vs Stress",
                xaxis_title="Daily portfolio P&L",
                yaxis_title="Density",
                legend=dict(x=0.01, y=0.99),
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(summary.map(lambda x: f"{x:.4%}"), use_container_width=True)
            _csv_download(summary, "var_stress_summary.csv", key="dl_var_stress", index=True)

            ratio_99 = summary.loc["VaR 99%", f"Stressed {selected}"] / summary.loc["VaR 99%", "Historical 3Y"]
            st.caption(
                f"{selected} stress 99% VaR is **{ratio_99:.1f}×** the full-sample historical 99% VaR — "
                "models calibrated on the full sample under-price crisis-conditional tail risk."
            )

    with tab3:
        data = _load_multi_nu(_OUT_VER)
        if data is None:
            st.warning("Multi-ν parametric-t artifacts not found. Run Module 2 (VaR Engine) in the notebook first.")
        else:
            table = data["table"]
            nu_fit = data["nu_fit"]
            st.markdown(f"**MLE-fitted ν = {nu_fit:.1f}** &nbsp; · &nbsp; comparison grid:")
            st.dataframe(table.map(lambda x: f"{x:.4%}"), use_container_width=True)
            _csv_download(table, "var_multi_nu_table.csv", key="dl_var_nu", index=True)

            # 99% VaR vs nu line chart. 'inf' plotted at x=30 with tick labeled '∞'.
            x_numeric = [4, 5, 8, 20, 30]
            x_labels = ["4", "5", "8", "20", "∞"]
            y_99 = table["VaR 99%"].tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_numeric, y=y_99,
                mode="lines+markers",
                marker=dict(size=10, color="#1f77b4"),
                line=dict(color="#1f77b4", width=2),
                hovertemplate="ν = %{text}<br>99% VaR = %{y:.4%}<extra></extra>",
                text=x_labels,
            ))
            fig.update_layout(
                title="99% VaR vs degrees of freedom",
                xaxis=dict(title="ν (Student-t degrees of freedom)",
                           tickvals=x_numeric, ticktext=x_labels),
                yaxis=dict(title="99% VaR", tickformat=".2%"),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Variance correction `σ · √((ν−2)/ν)` keeps the scaled-t standard deviation "
                "matched to the sample, so rows compare like-for-like. Lower ν → fatter tails "
                "→ larger 99% VaR. The ν → ∞ row reproduces the normal parametric VaR."
            )

    with tab4:
        data = _load_decomposition(_OUT_VER)
        if data is None:
            st.warning("Decomposition artifacts not found. Run Module 2 (VaR Engine) in the notebook first.")
        else:
            scalars = data["scalars"]
            betas = data["betas"]

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[scalars["pct_systematic"], scalars["pct_idiosyncratic"]],
                    y=["Systematic (PCs)", "Idiosyncratic"],
                    orientation="h",
                    marker_color=["#1f77b4", "#7f7f7f"],
                    text=[f"{scalars['pct_systematic']:.2f}%",
                          f"{scalars['pct_idiosyncratic']:.2f}%"],
                    textposition="auto",
                ))
                fig.update_layout(
                    title="Daily Yield-Change Variance Decomposition",
                    xaxis=dict(title="% of total decomposed variance", range=[0, 100]),
                    height=320,
                    margin=dict(l=10, r=10, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Decomposition total",   f"{scalars['var_total']:.4g}")
                st.metric("Empirical Var(w'Δy)",   f"{scalars['var_empirical']:.4g}")
                st.metric("Residual-corr gap",     f"{scalars['residual_corr_gap_pct']:+.2f}%")

            st.markdown("**β matrix (country × PC):**")
            st.dataframe(betas.map(lambda x: f"{x:.3f}"), use_container_width=True)
            _csv_download(betas, "var_decomposition_betas.csv", key="dl_var_betas", index=True)

            st.latex(
                r"\mathrm{Var}(\mathbf{w}^\top \Delta y) "
                r"= \mathbf{w}^\top B \Sigma_F B^\top \mathbf{w} "
                r"+ \mathbf{w}^\top D \mathbf{w}"
            )

            st.caption(
                f"Systematic share: **{scalars['pct_systematic']:.2f}%** — the fraction of the "
                "LC fund's daily yield-change variance driven by the global EM rate factors "
                "(PC1 level, PC2 slope, PC3 curvature). A high systematic share means "
                "diversification across the four LC countries is limited."
            )

# ── Daily Briefings ───────────────────────────────────────────────────────────
elif page == "Daily Briefings":
    briefings = _load_briefings(_OUT_VER)
    if not briefings:
        st.warning("sample_briefings.json not found. Run Module 3 (Daily Briefing Engine) first.")
    else:
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
        portfolio_results = _load_portfolio_data(_RAW_VER)
    except Exception as exc:
        st.error(f"Could not load portfolio data: {exc}")
        st.stop()

    p1 = portfolio_results[0]
    p2 = portfolio_results[1]

    # Pre-compute risk stats and yield levels once (used by both tab_risk and export button).
    yield_levels = _load_yield_levels(_RAW_VER)
    rf_data      = _load_rf_data(_OUT_VER)
    rs1 = compute_risk_stats(p1["def"], p1["pnl"], yield_levels, rf_data)
    rs2 = compute_risk_stats(p2["def"], p2["pnl"], yield_levels, rf_data)

    # ── Quarterly export control bar ──────────────────────────────────────────
    quarters = get_available_quarters(p1["pnl"])
    if quarters:
        st.markdown(
            "<div class='section-card' style='padding:16px 24px; margin-bottom:16px;'>"
            "<h3 style='margin-bottom:12px;'>Quarterly Report Export</h3>",
            unsafe_allow_html=True,
        )
        _exp_col1, _exp_col2 = st.columns([3, 1])
        with _exp_col1:
            _q_idx = st.selectbox(
                "Select quarter",
                range(len(quarters)),
                format_func=lambda i: quarters[i]["label"],
                label_visibility="visible",
            )
        with _exp_col2:
            _q = quarters[_q_idx]
            _q_num = (_q["start"].month - 1) // 3 + 1
            _fname = f"EM_FI_Q{_q_num}_{_q['start'].year}_Report.xlsx"
            _report_bytes = _cached_report(
                _q["start"].isoformat(),
                _q["end"].isoformat(),
                _RAW_VER,
            )
            st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Export Quarterly Report",
                data=_report_bytes,
                file_name=_fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help=f"Download {_q['label']} as a pre-filled Excel report.",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    tab_weights, tab_perf, tab_var, tab_compare, tab_risk = st.tabs(
        ["Weights", "Cumulative Performance", "VaR", "P&L Comparison", "Risk Statistics"]
    )

    # ── Weights ──────────────────────────────────────────────────────────────
    with tab_weights:
        w1 = p1["def"]["weights"]
        w2 = p2["def"]["weights"]
        countries = sorted(set(w1) | set(w2))
        raw1 = [w1.get(c, 0.0) for c in countries]
        raw2 = [w2.get(c, 0.0) for c in countries]
        tot1, tot2 = sum(raw1), sum(raw2)
        pct1 = [v / tot1 * 100 if tot1 else 0.0 for v in raw1]
        pct2 = [v / tot2 * 100 if tot2 else 0.0 for v in raw2]

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
        _csv_download(df_w, "portfolio_weights.csv", key="dl_pf_weights")
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
        _csv_download(df_stats, "portfolio_performance_summary.csv", key="dl_pf_perf")
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
        _csv_download(df_var, "portfolio_var_cvar_summary.csv", key="dl_pf_var_summary")
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

        def _fmt(val, fmt=".2f"):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:{fmt}}"

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
        _csv_download(df_ret, "portfolio_return_metrics.csv", key="dl_pf_ret")
        st.caption("Carry = portfolio-weighted latest benchmark yield. Roll-down approximates the 1-year return from rolling down the curve, using each country's par-bond modified duration and the curve slope between the benchmark maturity T and the next-shorter available maturity T*ᵢ:")
        st.latex(r"\text{Roll-down} \;\approx\; \sum_i w_i \cdot \text{MD}_i \cdot \frac{y_T^{(i)} - y_{T_i^*}^{(i)}}{T - T_i^*}")
        st.caption("T = portfolio benchmark maturity; T*ᵢ = next-shorter available maturity for country i. South Africa excluded (no maturity shorter than 5Y available in data).")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 2. Risk & Ratio Metrics ────────────────────────────────────────
        estr_now = rs1["current_estr"]
        sofr_now = rs1["current_sofr"]
        ust_now  = rs1["current_ust"]
        avg_e    = rs1["avg_estr"]
        avg_u    = rs1["avg_ust"]
        rf_lbl   = rs1.get("rf_label", "0")
        has_rf   = not np.isnan(ust_now)
        rf_label = f"{rf_lbl} ({ust_now:.2f}%)" if has_rf else "0 (no rf data)"

        # Rate context banner
        if has_rf:
            estr_str = f"€STR = <strong>{estr_now:.3f}%</strong>&nbsp;|&nbsp;" if not np.isnan(estr_now) else ""
            sofr_str = f"SOFR = <strong>{sofr_now:.3f}%</strong>&nbsp;|&nbsp;" if not np.isnan(sofr_now) else ""
            st.markdown(f"""
            <div class='info-banner'>
                <strong>Risk-free rates (FRED, latest):</strong>
                &nbsp; {rf_lbl} = <strong>{ust_now:.3f}%</strong>
                &nbsp;|&nbsp; {estr_str}{sofr_str}
                Avg {rf_lbl} over portfolio history = <strong>{avg_u:.3f}%</strong>
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
        _csv_download(df_risk_tbl, "portfolio_risk_metrics.csv", key="dl_pf_risk")
        st.caption(f"Sharpe and Sortino use daily excess returns over the UST constant-maturity yield matching each portfolio's benchmark maturity (source: FRED). Falls back to SOFR if the UST series is unavailable. Sortino denominator = annualised downside semi-deviation of excess returns:")
        st.latex(r"\text{Sortino denominator} \;=\; \sqrt{\mathbb{E}\!\left[\min(\text{excess},\, 0)^2\right]} \cdot \sqrt{252}")
        st.latex(r"\text{Calmar} \;=\; \frac{\text{Annualised total return}}{\lvert \text{Max drawdown} \rvert}")
        st.caption("rf = 0 rows shown for reference.")
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
        _csv_download(df_bond, "portfolio_bond_analytics.csv", key="dl_pf_bond")
        st.caption("Modified Duration = portfolio-weighted average of per-country par bond duration (annual compounding, bond priced at par):")
        st.latex(r"\text{MD}_i \;=\; \frac{1 - (1 + y_i)^{-T}}{y_i}")
        st.latex(r"\text{DV01 (EUR)} \;=\; \text{MD}_{\text{portfolio}} \times 0.0001 \times \text{AUM}")
        st.caption("Convexity per country (then portfolio-weighted):")
        st.latex(r"C_i \;=\; \frac{D_{\text{mac},i} \cdot (D_{\text{mac},i} + 1)}{(1 + y_i)^{2}}")
        st.caption("YTM = portfolio-weighted latest benchmark yield.")
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
        _csv_download(pd.DataFrame(var_long), "portfolio_var_cvar_pct.csv", key="dl_pf_var_pct")
        st.caption("Parametric: Normal P&L assumption, z-score method (μ, σ are the sample mean and standard deviation of daily P&L; z_α = Φ⁻¹(α); φ is the standard Normal pdf):")
        st.latex(r"\text{VaR}_{\text{Normal}} \;=\; -\left(\mu + z_\alpha \cdot \sigma\right)")
        st.latex(r"\text{CVaR}_{\text{Normal}} \;=\; -\left(\mu - \sigma \cdot \frac{\varphi(z_\alpha)}{\alpha}\right)")
        st.caption("Historical: empirical α-quantile of P&L for VaR, mean of the left α-tail for CVaR. Monte Carlo: 50,000 draws from a Normal distribution fitted to the sample mean and variance (seed = 42):")
        st.latex(r"\mathcal{N}\!\left(\mu,\, \sigma^{2}\right)")
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
        _csv_download(pd.DataFrame(var_eur_long), "portfolio_var_cvar_eur.csv", key="dl_pf_var_eur")
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
        # Export plain country names (krd_labels prepend a flag emoji for display).
        _csv_download(
            krd_tbl.assign(Country=all_krd_c),
            "portfolio_key_rate_duration.csv",
            key="dl_pf_krd",
        )
        st.caption("MD (par bond) = per-country modified duration at the benchmark maturity T, using each country's latest benchmark yield (par bond, annual compounding):")
        st.latex(r"\text{MD}_i \;=\; \frac{1 - (1 + y_i)^{-T}}{y_i}")
        st.caption("Both portfolios share the same MDᵢ; KRDs differ only because of different weights:")
        st.latex(r"\text{KRD}_i \;=\; w_i \cdot \text{MD}_i")
        st.latex(r"\sum_i \text{KRD}_i \;=\; \text{MD}_{\text{portfolio}}")
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

    st.markdown("""
    <div class='disclaimer-footnote'>
        <strong>Rate &amp; carry proxy.</strong> All P&amp;L and performance metrics
        use a yield-change duration model plus daily coupon accrual:
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\frac{\Delta P}{P} \;\approx\; -\,D_{\text{eff}} \cdot \frac{\Delta y}{100} \;+\; \frac{y_t}{252}")
    st.markdown("""
    <div class='disclaimer-footnote' style='border-top:none; margin-top:4px; padding-top:0;'>
        <strong>FX return is excluded</strong> — material for the LC fund.
        The HC fund uses local-currency yields as a proxy for its USD-denominated holdings.
        Risk metrics (VaR, vol, DV01) are internally consistent with this proxy;
        return and Sharpe figures are estimates, not NAV-based performance.
    </div>
    """, unsafe_allow_html=True)

# ── Alert History ─────────────────────────────────────────────────────────────
elif page == "Alert History":
    alerts = _load_alert_history(_OUT_VER)
    if alerts is None:
        st.warning("alert_history.json not found. Run Module 1.4 (Alert Engine) first.")
    else:
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
                        "type":     st.column_config.TextColumn("Type",     width="small"),
                        "regime":   st.column_config.TextColumn("Regime",   width="small"),
                        "detail":   st.column_config.TextColumn("Detail",   width="large"),
                    },
                )
                _csv_download(
                    filtered[["date", "severity", "type", "regime", "detail"]],
                    "alert_history.csv",
                    key="dl_alerts",
                )

# ── Chatbot ───────────────────────────────────────────────────────────────────
elif page == "Chatbot":
    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = []

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown(
            "<div style='color:#64748b; font-size:0.9rem; margin-bottom:8px;'>"
            "Local assistant — runs on Ollama, no data leaves your machine."
            "</div>",
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button(
            "Clear conversation",
            use_container_width=True,
            help="Delete all messages in this chat and start over.",
        ):
            st.session_state["chatbot_messages"] = []
            st.toast("Conversation cleared.", icon="🗑️")
            st.rerun()

    for msg in st.session_state["chatbot_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about the dashboard or general questions…")
    if user_input:
        st.session_state["chatbot_messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(
                "<span style='color:#64748b; font-style:italic;'>"
                "Thinking… this usually takes a few seconds."
                "</span>",
                unsafe_allow_html=True,
            )
            try:
                reply = ""
                for chunk in chatbot.stream_chat(
                    st.session_state["chatbot_messages"]
                ):
                    reply += chunk
                    placeholder.markdown(reply)
                if not reply:
                    placeholder.empty()
                st.session_state["chatbot_messages"].append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as exc:
                placeholder.empty()
                st.error(
                    f"Could not reach the local Ollama model "
                    f"(`{chatbot.MODEL_NAME}`). Is `ollama serve` running and the "
                    f"model pulled?\n\n**Details:** `{exc}`"
                )

# ── Footer (rendered on every page) ─────────────────────────────────────────────
_page_footer()
