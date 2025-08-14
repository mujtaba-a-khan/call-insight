# app/components/charts.py
# Charts for Call Insights:
# - Donut: Connected vs Disconnected
# - Bars: Call Types, Outcomes
# - Consistent colors; optionally save PNGs under artifacts/charts
#
# Notes
# • Uses Plotly for light interactivity in Streamlit.
# • If "kaleido" is installed, figures are also exported as PNGs.
# • Unknowns are hidden by default unless the caller sets show_unknowns=True.


from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_COLORS: Dict[str, str] = {
    "Connected": "#16A34A",
    "Disconnected": "#9CA3AF",
    "Inquiry": "#3B82F6",
    "Billing/Sales": "#14B8A6",
    "Support": "#F59E0B",
    "Complaint": "#EF4444",
    "Resolved": "#16A34A",
    "Callback": "#8B5CF6",
    "Refund": "#F59E0B",
    "Sale-close": "#14B8A6",
    "Unknown": "#6B7280",
}


def _maybe_save(fig, path: Path) -> None:
    """Save plotly figure to PNG if kaleido is available; otherwise skip silently."""
    try:
        import plotly.io as pio  # noqa: WPS433 (import inside fn is intentional)
        path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, str(path), format="png", scale=2)
    except Exception:
        # No kaleido or write error — ignore; interactive chart still renders in Streamlit.
        pass


def render_charts(
    df: pd.DataFrame,
    *,
    show_unknowns: bool = False,
    color_map: Optional[Dict[str, str]] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Render the three charts. Optionally export PNGs to `save_dir`.

    Parameters
    ----------
    df : DataFrame
        Must include columns: Connected, Type, Outcome
    show_unknowns : bool
        If False, Unknown categories are filtered from bars.
    color_map : dict
        Label -> hex color. Falls back to DEFAULT_COLORS.
    save_dir : Path
        If provided, PNGs are written here (requires `kaleido`).
    """
    if df.empty:
        st.info("Upload audio/CSV to see charts.")
        return

    colors = {**DEFAULT_COLORS, **(color_map or {})}

    # ---- Connected vs Disconnected (Donut)
    c_counts = (
        df["Connected"]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    # Ensure stable order
    conn_order = [lab for lab in ["Connected", "Disconnected", "Unknown"] if lab in c_counts["label"].unique()]
    fig_conn = px.pie(
        c_counts,
        names="label",
        values="count",
        hole=0.5,
        title="Connected vs Disconnected",
        color="label",
        category_orders={"label": conn_order},
        color_discrete_map={lab: colors.get(lab, "#9CA3AF") for lab in conn_order},
    )
    st.plotly_chart(fig_conn, use_container_width=True)
    if save_dir:
        _maybe_save(fig_conn, Path(save_dir) / "connected_vs_disconnected.png")

    # ---- Call Types (Bar)
    t_counts = (
        df["Type"]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("Type")
        .reset_index(name="count")
    )
    if not show_unknowns:
        t_counts = t_counts[t_counts["Type"] != "Unknown"]
    type_order = [lab for lab in ["Inquiry", "Billing/Sales", "Support", "Complaint", "Unknown"] if lab in t_counts["Type"].unique()]

    fig_types = px.bar(
        t_counts,
        x="Type",
        y="count",
        title="Call Types",
        color="Type",
        category_orders={"Type": type_order},
        color_discrete_map={lab: colors.get(lab, "#9CA3AF") for lab in type_order},
    )
    fig_types.update_layout(xaxis_title=None, yaxis_title="Count")
    st.plotly_chart(fig_types, use_container_width=True)
    if save_dir:
        _maybe_save(fig_types, Path(save_dir) / "call_types.png")

    # ---- Outcomes (Bar)
    o_counts = (
        df["Outcome"]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("Outcome")
        .reset_index(name="count")
    )
    if not show_unknowns:
        o_counts = o_counts[o_counts["Outcome"] != "Unknown"]
    outcome_order = [lab for lab in ["Resolved", "Callback", "Refund", "Sale-close", "Unknown"] if lab in o_counts["Outcome"].unique()]

    fig_out = px.bar(
        o_counts,
        x="Outcome",
        y="count",
        title="Outcomes",
        color="Outcome",
        category_orders={"Outcome": outcome_order},
        color_discrete_map={lab: colors.get(lab, "#9CA3AF") for lab in outcome_order},
    )
    fig_out.update_layout(xaxis_title=None, yaxis_title="Count")
    st.plotly_chart(fig_out, use_container_width=True)
    if save_dir:
        _maybe_save(fig_out, Path(save_dir) / "outcomes.png")

    st.caption("English-only. Unknowns hidden by default (toggle in sidebar).")
