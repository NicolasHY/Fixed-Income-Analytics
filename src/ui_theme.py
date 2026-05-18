"""
Company brand theme for Plotly charts.

Call `apply_theme()` once at app startup; every subsequent
`go.Figure()` will inherit the template automatically.
See BRAND.md for the full token specification.
"""

import plotly.graph_objects as go
import plotly.io as pio

CHART_COLORS = ["#1b3a5c", "#e67e22", "#16a34a", "#7ec8e3", "#9333ea", "#64748b"]

_FONT_STACK = (
    '"Inter","SF Pro Text",-apple-system,BlinkMacSystemFont,'
    '"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif'
)

_COMPANY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=_FONT_STACK, size=12, color="#0f172a"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=CHART_COLORS,
        hoverlabel=dict(
            bgcolor="#0d1b2a",
            bordercolor="#1b3a5c",
            font=dict(family=_FONT_STACK, color="#c9d6e3", size=12),
        ),
        legend=dict(
            orientation="h",
            y=1.08,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12),
        ),
        xaxis=dict(
            gridcolor="#e2e8f0",
            linecolor="#e2e8f0",
            zeroline=False,
            showgrid=True,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            linecolor="#e2e8f0",
            zeroline=False,
            showgrid=True,
            tickfont=dict(size=11),
        ),
        margin=dict(t=56, r=20, b=44, l=64),
    )
)


def apply_theme() -> None:
    """Register and activate the Company Plotly template."""
    pio.templates["company"] = _COMPANY_TEMPLATE
    pio.templates.default = "company"
