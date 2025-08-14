# core/schema.py
# Canonical schema and helpers for the Calls table.
# Keep this file tiny and dependency-free (pandas only). Everything else builds on it.

from typing import Dict, List

import pandas as pd

# ---- Enumerations (used across UI, rules, and tests)
CALL_CONNECTION = ["Connected", "Disconnected"]
CALL_TYPES = ["Inquiry", "Billing/Sales", "Support", "Complaint", "Unknown"]
CALL_OUTCOMES = ["Resolved", "Callback", "Refund", "Sale-close", "Unknown"]

# ---- Canonical column order (raw + derived)
COLUMNS_ORDER: List[str] = [
    "call_id",
    "start_time",
    "duration_seconds",
    "transcript",
    "agent_id",
    "campaign",
    "customer_name",
    "product_name",
    "quantity",
    "amount",
    "order_id",
    # derived labels
    "Connected",
    "Type",
    "Outcome",
]

# ---- Target dtypes (nullable where sensible)
DTYPES: Dict[str, str] = {
    "call_id": "string",
    "start_time": "datetime64[ns]",
    "duration_seconds": "float64",
    "transcript": "string",
    "agent_id": "string",
    "campaign": "string",
    "customer_name": "string",
    "product_name": "string",
    "quantity": "Int64",     # nullable integer
    "amount": "Float64",     # nullable float
    "order_id": "string",
    "Connected": "string",
    "Type": "string",
    "Outcome": "string",
}


def empty_calls_df() -> pd.DataFrame:
    """Return an empty Calls DataFrame with the canonical columns & dtypes."""
    df = pd.DataFrame(columns=COLUMNS_ORDER)
    # apply dtypes (datetime stays NaT)
    for col, dt in DTYPES.items():
        if dt == "datetime64[ns]":
            df[col] = pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
        else:
            df[col] = df[col].astype(dt)
    return df[COLUMNS_ORDER]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns with NA values, preserving existing data."""
    out = df.copy()
    for col in COLUMNS_ORDER:
        if col not in out.columns:
            out[col] = pd.NA
    # keep a stable column order
    return out[COLUMNS_ORDER]


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to canonical dtypes; tolerate bad inputs gracefully."""
    out = ensure_columns(df)

    # datetime
    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True)

    # numerics
    out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce").astype("float64")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").astype("Int64")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").astype("Float64")

    # strings (use pandas "string" dtype for consistency)
    str_cols = [
        "call_id",
        "transcript",
        "agent_id",
        "campaign",
        "customer_name",
        "product_name",
        "order_id",
        "Connected",
        "Type",
        "Outcome",
    ]
    for c in str_cols:
        out[c] = out[c].astype("string")

    return out


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize label values into known sets; unknowns become 'Unknown'.
    Does not *assign* labels — that’s handled by labeling_rules.py.
    """
    out = df.copy()
    if "Type" in out.columns:
        out["Type"] = out["Type"].where(out["Type"].isin(CALL_TYPES), "Unknown").astype("string")
    if "Outcome" in out.columns:
        out["Outcome"] = out["Outcome"].where(out["Outcome"].isin(CALL_OUTCOMES), "Unknown").astype("string")
    if "Connected" in out.columns:
        out["Connected"] = out["Connected"].where(out["Connected"].isin(CALL_CONNECTION), pd.NA).astype("string")
    return out


def normalize_calls_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full normalization pass:
      1) ensure required columns exist
      2) parse start_time as tz-aware UTC (handles ISO with offsets)
      3) coerce dtypes (duration, amounts, etc.)
      4) normalize label values (Type/Outcome, etc.)
      5) reorder columns if COLUMNS_ORDER is available
    """
    # 1) make sure all expected columns exist
    out = ensure_columns(df)

    # 2) enforce tz-aware UTC for start_time
    if "start_time" in out.columns:
        out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True)

    # 3) coerce numeric/text dtypes using your existing helper
    out = coerce_dtypes(out)

    # 4) normalize labels using your existing helper
    out = normalize_labels(out)

    # 5) optional: keep a consistent column order if constant is defined
    try:
        out = out[[c for c in COLUMNS_ORDER if c in out.columns]]
    except NameError:
        # COLUMNS_ORDER not defined in this module; ignore
        pass

    return out


def validate_calls_df(df: pd.DataFrame) -> List[str]:
    """
    Lightweight validation. Returns a list of human-readable warnings (empty if OK).
    """
    warnings: List[str] = []
    missing = [c for c in COLUMNS_ORDER if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns added as empty: {', '.join(missing)}")

    # sanity checks
    if "duration_seconds" in df.columns and (df["duration_seconds"].lt(0).any()):
        warnings.append("Some rows have negative duration_seconds (will be treated as 0).")
    if "start_time" in df.columns and pd.to_datetime(df["start_time"], errors="coerce", utc=True).isna().all():
        warnings.append("start_time could not be parsed for any rows (check your timestamps).")

    return warnings
