"""
EM Fixed Income Intelligence Platform — Streamlit Dashboard

Offline-capable demo: reads pre-generated outputs from data/output/.
Run:  streamlit run app.py
"""

import json
import os
import streamlit as st
import pandas as pd
from pathlib import Path

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

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

OUT = Path("data/output")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0 24px 0;'>
        <div style='font-size:0.72rem; color:#5a8aaa; letter-spacing:0.08em; text-transform:uppercase;'>EM Fixed Income</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem; color:#4a6a85; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Pipeline Health", "Data Load", "PCA & Regime", "VaR Engine", "Alert History", "Daily Briefings"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("Stop Server", use_container_width=True, type="secondary"):
        os._exit(0)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#4a6a85; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>Universe</div>
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

# ── Pipeline Health ───────────────────────────────────────────────────────────
if page == "Pipeline Health":
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
