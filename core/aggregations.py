# core/aggregations.py
# Aggregations & KPIs for charts and summaries.
# no Streamlit here, so it's easy to unit test.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Kpis:
    total_included: int
    connected: int
    disconnected: int
    pct_connected: float
    top_type: Optional[str]
    top_outcome: Optional[str]


def _mode_or_none(s: pd.Series) -> Optional[str]:
    if s is None or s.empty:
        return None
    s = s.dropna().astype(str)
    if s.empty:
        return None
    return s.mode().iloc[0]


def compute_kpis(df: pd.DataFrame, *, show_unknowns: bool = False) -> Kpis:
    """Compute headline KPIs for the dashboard."""
    total = int(len(df))

    conn_col = df.get("Connected", pd.Series(dtype=str)).astype(str)
    connected = int((conn_col == "Connected").sum())
    disconnected = int((conn_col == "Disconnected").sum())
    denom = max(1, connected + disconnected)
    pct = (connected / denom) * 100.0

    type_col = df.get("Type", pd.Series(dtype=str)).astype(str)
    out_col = df.get("Outcome", pd.Series(dtype=str)).astype(str)

    if not show_unknowns:
        type_col = type_col[type_col != "Unknown"]
        out_col = out_col[out_col != "Unknown"]

    return Kpis(
        total_included=total,
        connected=connected,
        disconnected=disconnected,
        pct_connected=pct,
        top_type=_mode_or_none(type_col),
        top_outcome=_mode_or_none(out_col),
    )


def connection_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts for Connected vs Disconnected (and Unknown if present)."""
    vc = (
        df.get("Connected", pd.Series(dtype=str))
        .astype(str)
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("Connected")
        .reset_index(name="count")
    )
    # Stable order if the labels exist
    order = ["Connected", "Disconnected", "Unknown"]
    vc["order"] = vc["Connected"].apply(lambda x: order.index(x) if x in order else 99)
    return vc.sort_values("order").drop(columns="order")


def type_breakdown(df: pd.DataFrame, *, show_unknowns: bool = False) -> pd.DataFrame:
    """Return counts for call types."""
    vc = (
        df.get("Type", pd.Series(dtype=str))
        .astype(str)
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("Type")
        .reset_index(name="count")
    )
    if not show_unknowns:
        vc = vc[vc["Type"] != "Unknown"]
    order = ["Inquiry", "Billing/Sales", "Support", "Complaint", "Unknown"]
    vc["order"] = vc["Type"].apply(lambda x: order.index(x) if x in order else 99)
    return vc.sort_values("order").drop(columns="order")


def outcome_breakdown(df: pd.DataFrame, *, show_unknowns: bool = False) -> pd.DataFrame:
    """Return counts for outcomes."""
    vc = (
        df.get("Outcome", pd.Series(dtype=str))
        .astype(str)
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("Outcome")
        .reset_index(name="count")
    )
    if not show_unknowns:
        vc = vc[vc["Outcome"] != "Unknown"]
    order = ["Resolved", "Callback", "Refund", "Sale-close", "Unknown"]
    vc["order"] = vc["Outcome"].apply(lambda x: order.index(x) if x in order else 99)
    return vc.sort_values("order").drop(columns="order")


def by_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-agent breakdown: total, connected%, top type, top outcome.
    Returns an empty DataFrame if agent_id missing.
    """
    if "agent_id" not in df.columns or df.empty:
        return pd.DataFrame(columns=["agent_id", "total", "pct_connected", "top_type", "top_outcome"])

    g = df.groupby("agent_id", dropna=False)
    con = (g["Connected"].apply(lambda s: (s == "Connected").sum())).rename("connected")
    dis = (g["Connected"].apply(lambda s: (s == "Disconnected").sum())).rename("disconnected")
    total = g.size().rename("total")

    pct = ((con / (con + dis).clip(lower=1)) * 100.0).rename("pct_connected")

    # top mode ignoring Unknown where possible
    def _top_mode(series: pd.Series) -> Optional[str]:
        s = series.dropna().astype(str)
        if s.empty:
            return None
        s_wo = s[s != "Unknown"]
        return (_mode_or_none(s_wo) or _mode_or_none(s))

    top_type = g["Type"].apply(_top_mode).rename("top_type")
    top_out = g["Outcome"].apply(_top_mode).rename("top_outcome")

    out = pd.concat([total, pct, top_type, top_out], axis=1).reset_index()
    return out.sort_values("total", ascending=False)


def by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-campaign breakdown: total, connected%, top type, top outcome.
    Returns an empty DataFrame if campaign missing.
    """
    if "campaign" not in df.columns or df.empty:
        return pd.DataFrame(columns=["campaign", "total", "pct_connected", "top_type", "top_outcome"])

    g = df.groupby("campaign", dropna=False)
    con = (g["Connected"].apply(lambda s: (s == "Connected").sum())).rename("connected")
    dis = (g["Connected"].apply(lambda s: (s == "Disconnected").sum())).rename("disconnected")
    total = g.size().rename("total")

    pct = ((con / (con + dis).clip(lower=1)) * 100.0).rename("pct_connected")

    def _top_mode(series: pd.Series) -> Optional[str]:
        s = series.dropna().astype(str)
        if s.empty:
            return None
        s_wo = s[s != "Unknown"]
        return (_mode_or_none(s_wo) or _mode_or_none(s))

    top_type = g["Type"].apply(_top_mode).rename("top_type")
    top_out = g["Outcome"].apply(_top_mode).rename("top_outcome")

    out = pd.concat([total, pct, top_type, top_out], axis=1).reset_index()
    return out.sort_values("total", ascending=False)
