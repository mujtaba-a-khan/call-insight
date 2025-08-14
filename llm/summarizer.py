# llm/summarizer.py
# Deterministic result summarization with optional local LLM polish.
#
# Usage:
#   text = summarize_results(
#       df_display,              # the DataFrame slice you are showing
#       spec,                    # structured filter spec
#       tz="Europe/Berlin",
#       llm=llm,                 # optional LocalLlm
#       pii_mask=True,
#       total_scope=scope_count, # ALL rows that matched before any display capping
#       display_count=len(df_display),
#   )
#
# Notes:
#   - This module never sends PII or full tables to an LLM.
#   - If no LLM is available, it returns a deterministic summary string.

from __future__ import annotations

from typing import Dict, Optional, List
import pandas as pd

from llm.runner import LocalLlm


def _pick_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    """Return the first column that exists in df, or None.

    Args:
        df: Input DataFrame.
        *names: Candidate column names in preference order.

    Returns:
        Name of the first matching column, or None.
    """
    for n in names:
        if n in df.columns:
            return n
    return None


def _deterministic_summary(df_display: pd.DataFrame,
                           spec: Dict,
                           total_scope: Optional[int],
                           display_count: Optional[int]) -> str:
    """Build a concise summary for the visible slice, aware of the full scope.

    Args:
        df_display: DataFrame currently shown in the UI (already sliced).
        spec: Structured filter spec used to produce the scope.
        total_scope: Size of the full in-scope set (before any display capping).
        display_count: Number of rows being shown (len(df_display)).

    Returns:
        Compact, human-readable summary string.
    """
    scope_n = int(total_scope) if total_scope is not None else int(len(df_display))
    shown_n = int(display_count) if display_count is not None else int(len(df_display))

    parts: List[str] = []
    if shown_n < scope_n:
        parts.append(f"Showing top {shown_n} of {scope_n} calls.")
    else:
        parts.append(f"{scope_n} calls in scope.")

    if spec.get("date_from") and spec.get("date_to"):
        parts.append(f"Date range: {spec['date_from']} → {spec['date_to']}.")

    # Light tallies over the displayed slice (fast and representative).
    type_col = _pick_col(df_display, "Type", "type")
    if type_col:
        vc = df_display[type_col].value_counts(dropna=False).to_dict()
        if vc:
            top = sorted(vc.items(), key=lambda x: (-x[1], str(x[0])))[:3]
            parts.append("Top types: " + ", ".join(f"{k} ({v})" for k, v in top) + ".")

    out_col = _pick_col(df_display, "Outcome", "outcome")
    if out_col:
        vc = df_display[out_col].value_counts(dropna=False).to_dict()
        if vc:
            top = sorted(vc.items(), key=lambda x: (-x[1], str(x[0])))[:3]
            parts.append("Top outcomes: " + ", ".join(f"{k} ({v})" for k, v in top) + ".")

    return " ".join(parts)


def summarize_results(df_display: pd.DataFrame,
                      spec: Dict,
                      tz: Optional[str] = None,
                      llm: Optional[LocalLlm] = None,
                      pii_mask: bool = True,
                      total_scope: Optional[int] = None,
                      display_count: Optional[int] = None) -> str:
    """Summarize the currently displayed rows, with awareness of the full scope.

    Args:
        df_display: The DataFrame that the table is showing (after any Top-K/cap).
        spec: Structured filter spec used to produce the scope.
        tz: IANA timezone string (reserved for future use).
        llm: Optional local LLM to lightly rewrite wording; content is unchanged.
        pii_mask: Ignored by the text summary; table masking handled elsewhere.
        total_scope: Number of rows that match the filters before display capping.
        display_count: Number of rows currently displayed (len(df_display)).

    Returns:
        One-sentence summary string.
    """
    base = _deterministic_summary(df_display, spec, total_scope, display_count)

    # If no model configured or it fails, return the deterministic string.
    if not llm:
        return base

    prompt = (
        "Rewrite the analytics summary for clarity without adding or removing facts.\n"
        f"Summary: {base}\n"
        "Return one sentence."
    )
    out = (llm.complete(prompt) or "").strip()
    return out if out else base
