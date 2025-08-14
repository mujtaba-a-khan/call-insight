"""Re-exports for Streamlit UI components."""
from .kpis import render_kpis
from .charts import render_charts
from .calls_table import render_calls_table
from .details_drawer import render_details

__all__ = ["render_kpis", "render_charts", "render_calls_table", "render_details"]
