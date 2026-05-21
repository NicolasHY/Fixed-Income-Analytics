"""Application-tier services.

Lives between the Data + Orchestration layers and the Streamlit pages.
Functions here return ready-to-render data structures (DataFrames, plain
dicts of numbers) without any Streamlit calls. The Application layer
(``app.py``) does nothing but render the result.

Right now this is just the dashboard portfolio view + quick stats; new
services land here as more inline logic gets lifted out of ``app.py``.
"""

from src.services.portfolios import build_portfolio_views, compute_quick_stats

__all__ = ["build_portfolio_views", "compute_quick_stats"]
