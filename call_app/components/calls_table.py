# app/components/calls_table.py
# Calls table with quick search, export, and row selection.
# - Shows the main operational columns
# - Simple "search all columns" text box
# - Export the visible (filtered) rows as CSV
# - Returns the selected row as a dict (or None)

import io
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st


DEFAULT_COLUMNS: List[str] = [
    "call_id",
    "start_time",
    "duration_seconds",
    "Connected",
    "Type",
    "Outcome",
    "agent_id",
    "campaign",
]


def _coerce_str(x: Any) -> str:
    try:
        return "" if pd.isna(x) else str(x)
    except Exception:
        return ""


def _filter_by_search(df: pd.DataFrame, cols: Iterable[str], query: str) -> pd.DataFrame:
    """Case-insensitive contains filter across the provided columns."""
    if not query:
        return df
    q = query.strip().lower()
    if not q:
        return df
    mask = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            col = df[c].map(_coerce_str).str.lower()
            mask = mask | col.str.contains(q, na=False)
    return df[mask]


def render_calls_table(
    df: pd.DataFrame,
    *,
    show_columns: Optional[List[str]] = None,
    page_size: int = 25,
) -> Optional[Dict[str, Any]]:
    """
    Render the calls table UI and return the selected row (as a dict) or None.

    Parameters
    ----------
    df : DataFrame
        Should include columns in DEFAULT_COLUMNS (missing ones are ignored gracefully).
    show_columns : list[str]
        Columns to display; defaults to DEFAULT_COLUMNS intersection with df.
    page_size : int
        Max rows to display (soft cap; Streamlit's table is virtualized).
    """
    st.subheader("Calls")

    if df is None or df.empty:
        st.info("No calls to display yet.")
        return None

    cols = show_columns or [c for c in DEFAULT_COLUMNS if c in df.columns]

    # --- Controls
    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
    with c1:
        search = st.text_input("Search (all columns)", placeholder="customer, agent, campaign, type...")
    with c2:
        page_size = int(st.number_input("Page size", min_value=10, max_value=200, value=page_size, step=5))
    with c3:
        sort_desc = st.toggle("Newest first", value=True)

    # --- Filter + sort
    work = _filter_by_search(df, cols, search).copy()
    if "start_time" in work.columns:
        work["__sort_time__"] = pd.to_datetime(work["start_time"], errors="coerce")
        work = work.sort_values("__sort_time__", ascending=not sort_desc).drop(columns="__sort_time__", errors="ignore")

    visible = work[cols].head(page_size)

    # --- Export visible rows
    csv_bytes = visible.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export visible rows (CSV)",
        data=csv_bytes,
        file_name="calls_visible.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --- Show table
    st.dataframe(visible, use_container_width=True, hide_index=True)

    # --- Row selection (by call_id)
    selected = None
    if "call_id" in work.columns:
        with st.expander("Details"):
            options = ["(none)"] + work["call_id"].astype(str).tolist()
            choice = st.selectbox("Select call_id", options, index=0)
            if choice != "(none)":
                row = work[work["call_id"].astype(str) == choice].iloc[0].to_dict()
                selected = {k: row.get(k) for k in df.columns}  # preserve original keys

    return selected
