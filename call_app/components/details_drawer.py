# app/components/details_drawer.py
# Per-call details panel:
# - Header badges: Connected / Type / Outcome
# - "Why these labels?" → shows up to N matched phrases from rules
# - Transcript preview with simple highlights
# - Copyable one-liner summary
#
# Usage:
#   from app.components.details_drawer import render_details
#   render_details(row_dict, rules_cfg, color_map=..., pii_mask=True)


import html
import textwrap
from typing import Dict, Iterable, List, Optional

import streamlit as st


DEFAULT_COLORS = {
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


def _mask_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return name
    parts = name.split()
    masked = []
    for p in parts:
        if len(p) <= 1:
            masked.append(p + "*")
        else:
            masked.append(p[0] + "*" * (len(p) - 1))
    return " ".join(masked)


def _badge(label: str, value: str, colors: Dict[str, str]) -> str:
    """Return small HTML badge markup."""
    color = colors.get(value, "#9CA3AF")
    return (
        f'<span style="display:inline-block;padding:2px 8px;margin-right:6px;'
        f'border-radius:999px;font-size:12px;background:{color};color:#0B1220;">'
        f'<b>{html.escape(label)}:</b> {html.escape(value)}</span>'
    )


def _collect_hits(transcript: str, rules: Dict) -> List[str]:
    """Collect a few matched phrases from type & outcome rules."""
    if not isinstance(transcript, str):
        return []
    t = transcript.lower()
    hits: List[str] = []

    # Type phrases
    for bucket, keywords in (rules.get("types") or {}).items():
        for kw in keywords:
            if kw.lower() in t:
                hits.append(f"[{bucket}] …{kw}…")

    # Outcome phrases (tail-focused rule in core, but we still surface phrases)
    outs = rules.get("outcomes") or {}
    for label in ["Resolved", "Callback", "Refund", "Sale-close"]:
        for phrase in outs.get(label, []):
            if phrase.lower() in t:
                hits.append(f"[{label}] …{phrase}…")

    # Deduplicate in a stable way and cap to 5
    seen = set()
    deduped = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
        if len(deduped) >= 5:
            break
    return deduped


def _highlight(text: str, needles: Iterable[str]) -> str:
    """
    Very light highlighter: wraps matched phrases with <mark>.
    Streamlit will render this via unsafe_allow_html=True.
    """
    if not isinstance(text, str) or not text:
        return ""
    safe = html.escape(text)
    # Sort needles by length to avoid nested overlaps
    sorted_needles = sorted({n for n in needles if n}, key=len, reverse=True)
    for n in sorted_needles:
        n_safe = html.escape(n)
        safe = safe.replace(n_safe, f"<mark>{n_safe}</mark>")
    return safe


def render_details(
    row: Dict,
    rules_cfg: Dict,
    *,
    color_map: Optional[Dict[str, str]] = None,
    pii_mask: bool = True,
) -> None:
    """
    Render a details drawer for a single call.

    Parameters
    ----------
    row : dict
        Row data from the Calls table (expects keys like call_id, transcript, Connected, Type, Outcome, etc.)
    rules_cfg : dict
        Loaded YAML rules (configs/rules.yml)
    color_map : dict
        Optional label -> color hex mapping
    pii_mask : bool
        If True, masks customer_name in any previews
    """
    colors = {**DEFAULT_COLORS, **(color_map or {})}

    # --- Header badges
    badges_html = []
    for key in ("Connected", "Type", "Outcome"):
        val = row.get(key) or "Unknown"
        badges_html.append(_badge(key, str(val), colors))
    st.markdown(" ".join(badges_html), unsafe_allow_html=True)

    # --- Why these labels?
    with st.expander("Why these labels?"):
        transcript = row.get("transcript") or ""
        hits = _collect_hits(transcript, rules_cfg)
        if hits:
            st.write("\n".join(f"- {h}" for h in hits))
        else:
            st.write("No specific phrases matched; labels may have come from duration/token thresholds.")

    # --- Transcript preview with highlights
    with st.expander("Transcript"):
        preview = textwrap.shorten(transcript, width=6000, placeholder=" …")
        needles = [h.split("…")[1] for h in _collect_hits(transcript, rules_cfg)]  # extract phrase part
        html_preview = _highlight(preview, needles)
        st.markdown(f"<div style='white-space:pre-wrap'>{html_preview}</div>", unsafe_allow_html=True)

    # --- Copyable one-liner summary
    cust = row.get("customer_name")
    if pii_mask and isinstance(cust, str):
        cust = _mask_name(cust)
    summary = (
        f"Call {row.get('call_id')} • {row.get('start_time')} • "
        f"{row.get('Connected')}/{row.get('Type')}/{row.get('Outcome')}"
        + (f" • {cust}" if cust else "")
    )
    st.text_area("Copy call summary", value=summary, height=60)
