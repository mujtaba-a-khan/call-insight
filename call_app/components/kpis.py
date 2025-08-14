# app/components/kpis.py
# KPI tiles for Call Insights.
# - Shows: Total included calls, % Connected, Top Call Type, Top Outcome
# - Handles empty data gracefully
# - Returns a small summary object you can reuse elsewhere

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st


@dataclass
class KpiSummary:
    total_calls: int
    percent_connected: float
    top_type: Optional[str]
    top_outcome: Optional[str]
    excluded_count: int


def _top_value(series: pd.Series, show_unknowns: bool) -> Optional[str]:
    if series.empty:
        return None
    s = series.dropna().astype(str)
    if not show_unknowns:
        s = s[s != "Unknown"]
    if s.empty:
        return None
    return s.mode().iloc[0]


def render_kpis(
    df: pd.DataFrame,
    excluded_df: pd.DataFrame | None = None,
    show_unknowns: bool = False,
) -> KpiSummary:
    """
    Render KPI tiles and return a summary.
    Expected columns in `df`: Connected, Type, Outcome
    """
    excluded_count = int(len(excluded_df)) if excluded_df is not None else 0
    total = int(len(df))

    connected = int((df["Connected"] == "Connected").sum()) if "Connected" in df.columns else 0
    disconnected = int((df["Connected"] == "Disconnected").sum()) if "Connected" in df.columns else 0
    denom = max(1, connected + disconnected)
    pct_connected = (connected / denom) * 100.0

    top_type = _top_value(df.get("Type", pd.Series(dtype=str)), show_unknowns)
    top_outcome = _top_value(df.get("Outcome", pd.Series(dtype=str)), show_unknowns)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Calls (included)", f"{total}")
    c2.metric("% Connected", f"{pct_connected:.1f}%")
    c3.metric("Top Call Type", top_type or "—")
    c4.metric("Top Outcome", top_outcome or "—")

    st.caption(f"Excluded (non-English/empty/<10s/STT fail): **{excluded_count}**")

    return KpiSummary(
        total_calls=total,
        percent_connected=pct_connected,
        top_type=top_type,
        top_outcome=top_outcome,
        excluded_count=excluded_count,
    )
