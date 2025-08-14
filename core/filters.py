# core/filters.py
# Pure-pandas filters for Calls and Q&A results.
# - Sidebar filters: date range, agent, campaign
# - Q&A filters: schema-bounded spec (date_from/to, type_any, outcome_any, agents,
#   campaigns, products, min/max amount)
# - Timezone-agnostic: comparisons are done in UTC; day bounds are built in
#   Europe/Berlin local time and converted to UTC.

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, Optional, List
from zoneinfo import ZoneInfo
import re

import pandas as pd

LOCAL_TZ = ZoneInfo("Europe/Berlin")


# ----------------------------- helpers ---------------------------------------
def _ensure_ts(x) -> pd.Timestamp:
    """Coerce arbitrary input to a pandas ``Timestamp``.

    Args:
        x: Any object that pandas can parse into a timestamp.

    Returns:
        A pandas ``Timestamp`` (may be tz-naive if input lacked tz info).
    """
    if not isinstance(x, pd.Timestamp):
        x = pd.to_datetime(x, errors="coerce")
    return x


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a tz-aware UTC ``Timestamp`` (assume LOCAL_TZ when naive).

    Args:
        ts: Input timestamp (tz-aware or naive).

    Returns:
        UTC-localized timestamp.
    """
    ts = _ensure_ts(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TZ)
    return ts.tz_convert("UTC")


def _local_day_bounds(
    date_from: Optional[date], date_to: Optional[date]
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Build half-open local day bounds ``[start_local, next_day_local)``.

    If one bound is None, the other is used for both.

    Args:
        date_from: Inclusive start (date).
        date_to: Inclusive end (date).

    Returns:
        A tuple of local-time bounds (tz-aware). ``(None, None)`` if both inputs are None.
    """
    if date_from is None and date_to is None:
        return None, None
    if date_from is None:
        date_from = date_to
    if date_to is None:
        date_to = date_from
    start_local = pd.Timestamp(date_from).normalize().tz_localize(LOCAL_TZ)
    end_local = (pd.Timestamp(date_to).normalize() + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    return start_local, end_local


def _normalize_dates(
    df: pd.DataFrame, date_from: Optional[date], date_to: Optional[date]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert optional naive day inputs to inclusive half-open UTC bounds.

    The returned bounds are ``[start_utc, end_utc)`` suitable for vectorized
    comparisons on UTC timestamps.

    Args:
        df: DataFrame with a ``start_time`` column.
        date_from: Inclusive start date, or ``None``.
        date_to: Inclusive end date, or ``None``.

    Returns:
        The UTC bounds as tz-aware timestamps.
    """
    ts_utc = pd.to_datetime(df.get("start_time"), errors="coerce", utc=True)

    # If the column is entirely NaT, return a range that admits everything.
    if ts_utc.isna().all():
        return (
            pd.Timestamp.min.tz_localize("UTC"),
            pd.Timestamp.max.tz_localize("UTC"),
        )

    # Derive missing bounds from data using LOCAL_TZ day boundaries.
    if date_from is None or date_to is None:
        ts_local = ts_utc.dt.tz_convert(LOCAL_TZ).dropna()
        if ts_local.empty:
            return (
                pd.Timestamp.min.tz_localize("UTC"),
                pd.Timestamp.max.tz_localize("UTC"),
            )
        if date_from is None:
            start_local = ts_local.min().normalize()
        else:
            start_local = pd.Timestamp(date_from).normalize().tz_localize(LOCAL_TZ)
        if date_to is None:
            end_local = ts_local.max().normalize() + pd.Timedelta(days=1)
        else:
            end_local = (pd.Timestamp(date_to).normalize() + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    else:
        start_local, end_local = _local_day_bounds(date_from, date_to)

    # Convert local bounds to UTC for comparison.
    start_utc = _to_utc(start_local)
    end_utc = _to_utc(end_local)
    return start_utc, end_utc


# ----------------------------- sidebar ---------------------------------------
@dataclass(frozen=True)
class SidebarFilters:
    """Filters originating from the Streamlit sidebar.

    Attributes:
        date_from: Inclusive start day (local).
        date_to: Inclusive end day (local).
        agent: Exact ``agent_id`` to include (None = all).
        campaign: Exact campaign to include (None = all).
    """
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    agent: Optional[str] = None
    campaign: Optional[str] = None


def apply_sidebar_filters(df: pd.DataFrame, f: SidebarFilters) -> pd.DataFrame:
    """Apply standard dashboard filters (date, agent, campaign).

    Notes:
        * Date range is inclusive by day (LOCAL_TZ), compared in UTC.
        * Agent and campaign are exact matches when provided.

    Args:
        df: Source calls dataframe.
        f: Sidebar filter values.

    Returns:
        Filtered dataframe (new object).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Date range (normalize both sides to UTC).
    if "start_time" in out.columns:
        start_utc, end_utc = _normalize_dates(out, f.date_from, f.date_to)
        st_col = pd.to_datetime(out["start_time"], errors="coerce", utc=True)
        out = out[(st_col >= start_utc) & (st_col < end_utc)]

    # Agent (exact, sidebar uses authoritative IDs).
    if f.agent and "agent_id" in out.columns:
        out = out[out["agent_id"] == f.agent]

    # Campaign (exact).
    if f.campaign and "campaign" in out.columns:
        out = out[out["campaign"] == f.campaign]

    return out


# ----------------------------- Q&A spec --------------------------------------
def apply_qna_filter_spec(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """Apply a structured Q&A spec to the calls dataframe.

    Recognized keys (schema-aligned):
        date_from, date_to:
            Day strings in "YYYY-MM-DD" format (LOCAL_TZ day bounds, compared in UTC).
        type_any:
            List[str] among {"Inquiry","Billing/Sales","Support","Complaint"}.
        outcome_any:
            List[str] among {"Resolved","Callback","Refund","Sale-close","Unknown"}.
        agents:
            List[str]; matches any of the columns {agent_id, agent_name, agent} if present.
        campaigns:
            List[str]; matches the 'campaign' column (case-insensitive exact).
        products:
            List[str]; substring match against 'product_name' (case-insensitive).
        min_amount, max_amount:
            Numeric lower/upper bounds for 'amount' if present.

    Args:
        df: Source calls dataframe.
        spec: Parsed filter specification.

    Returns:
        Filtered dataframe (new object; original is not modified).
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    mask = pd.Series(True, index=out.index, dtype=bool)

    # ---------- helpers ----------
    def _norm(s: pd.Series) -> pd.Series:
        """Normalize a string series for robust equality tests.

        Args:
            s: Input pandas Series.

        Returns:
            Lowercased, trimmed string Series (NaN handled).
        """
        return s.astype(str).str.strip().str.casefold()

    def _match_labels(series: pd.Series, wanted: List[str]) -> pd.Series:
        """Case/space-insensitive membership test for categorical labels.

        Args:
            series: Column to test (e.g., 'Type', 'Outcome').
            wanted: Desired label values.

        Returns:
            Boolean Series (True when value is in 'wanted').
        """
        if not wanted:
            return pd.Series(True, index=series.index, dtype=bool)
        targets = {w.strip().casefold() for w in wanted if isinstance(w, str) and w.strip()}
        return _norm(series).isin(targets)

    # ---------- date range ----------
    dfrom = spec.get("date_from")
    dto = spec.get("date_to", dfrom)
    try:
        date_from = pd.to_datetime(dfrom).date() if dfrom else None
        date_to = pd.to_datetime(dto).date() if dto else None
    except Exception:
        date_from = date_to = None

    if "start_time" in out.columns and (date_from or date_to):
        start_utc, end_utc = _normalize_dates(out, date_from, date_to)
        st_col = pd.to_datetime(out["start_time"], errors="coerce", utc=True)
        mask &= (st_col >= start_utc) & (st_col < end_utc)

    # ---------- type / outcome ----------
    if spec.get("type_any") and "Type" in out.columns:
        mask &= _match_labels(out["Type"], list(spec.get("type_any", [])))

    if spec.get("outcome_any") and "Outcome" in out.columns:
        mask &= _match_labels(out["Outcome"], list(spec.get("outcome_any", [])))

    # ---------- agents (ID or name) ----------
    if spec.get("agents"):
        want = {w.strip().casefold() for w in spec["agents"] if isinstance(w, str) and w.strip()}
        agent_cols = [c for c in ("agent_id", "agent_name", "agent") if c in out.columns]
        if agent_cols:
            any_match = pd.Series(False, index=out.index, dtype=bool)
            for c in agent_cols:
                any_match |= _norm(out[c]).isin(want)
            mask &= any_match

    # ---------- campaigns ----------
    if spec.get("campaigns") and "campaign" in out.columns:
        want = {w.strip().casefold() for w in spec["campaigns"] if isinstance(w, str) and w.strip()}
        mask &= _norm(out["campaign"]).isin(want)

    # ---------- products (substring) ----------
    if spec.get("products") and "product_name" in out.columns:
        values = [str(p).strip().lower() for p in spec["products"] if isinstance(p, str) and p.strip()]
        if values:
            pn = out["product_name"].astype(str).str.lower()
            sub = pd.Series(False, index=out.index, dtype=bool)
            for v in values:
                sub |= pn.str.contains(re.escape(v), na=False)
            mask &= sub

    # ---------- amount range ----------
    if "amount" in out.columns:
        amt = pd.to_numeric(out["amount"], errors="coerce")
        if spec.get("min_amount") is not None:
            mask &= amt >= float(spec["min_amount"])
        if spec.get("max_amount") is not None:
            mask &= amt <= float(spec["max_amount"])

    return out.loc[mask].reset_index(drop=True)
