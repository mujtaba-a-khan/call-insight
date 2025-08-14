# core/labeling_rules.py
# Deterministic labeling per spec:
# 1) Connected vs Disconnected
# 2) Call Type: Inquiry, Billing/Sales, Support, Complaint (tie-break priority)
# 3) Outcome: Resolved, Callback, Refund, Sale-close (tail-window with specificity priority)
#
# Notes:
# • English-only gate & <10s hard exclusion are handled in ingestion; this file only *labels* rows.
# • No diarization in V1. Rule B (two speakers) is reserved for V2.
# • Keep this module side-effect free and easy to unit test.

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .utils import (
    english_token_count,
    contains_any,
    soft_negation_guard,
    tail_fraction,
    tokenize_english,
)


# -----------------------------
# Connected vs Disconnected
# -----------------------------
def connected_label(duration_s: float, transcript: str, rules: Dict) -> str:
    cfg = rules.get("connected", {})
    short_disc = int(cfg.get("short_disconnected_s", 10))
    cutoff = int(cfg.get("duration_cutoff_s", 30))
    min_tokens = int(cfg.get("min_tokens", 40))
    cues: List[str] = list(cfg.get("disconnection_cues", []))

    # Edge rule: duration < short_disconnected_s ⇒ Disconnected
    if float(duration_s or 0) < short_disc:
        return "Disconnected"

    text = transcript or ""
    # Clear cues of non-connection
    if contains_any(text, cues):
        return "Disconnected"

    # Main rule: duration ≥ cutoff and tokens ≥ min_tokens ⇒ Connected
    if float(duration_s or 0) >= cutoff and english_token_count(text) >= min_tokens:
        return "Connected"

    return "Disconnected"


# -----------------------------
# Call Type (Bucket with tie-break)
# -----------------------------
def _bucket_score(text_lc: str, keywords: Iterable[str]) -> int:
    """Count keyword hits with a soft negation guard."""
    score = 0
    for kw in keywords:
        kw = str(kw).lower()
        if kw and kw in text_lc and soft_negation_guard(text_lc, kw):
            score += 1
    return score


def type_label(transcript: str, rules: Dict) -> str:
    t = (transcript or "").lower()
    buckets: Dict[str, List[str]] = rules.get("types", {})
    if not buckets:
        return "Unknown"

    scores = {name: _bucket_score(t, kws) for name, kws in buckets.items()}
    if all(v == 0 for v in scores.values()):
        return "Unknown"

    # Resolve ties by configured priority
    tie_break = rules.get("type_tie_break", ["Complaint", "Support", "Billing/Sales", "Inquiry"])

    # Choose max score; if ties, pick the highest-priority per tie_break
    max_score = max(scores.values())
    tied = [k for k, v in scores.items() if v == max_score]
    if len(tied) == 1:
        return tied[0]

    for pref in tie_break:
        if pref in tied:
            return pref
    # Fallback: stable order
    return sorted(tied)[0]


# -----------------------------
# Outcome (Tail-window + specificity priority)
# -----------------------------
def outcome_label(transcript: str, rules: Dict) -> str:
    if not transcript:
        return "Unknown"
    outs = rules.get("outcomes", {})
    frac = float(rules.get("outcome_window_frac", 0.25))

    tail_text = tail_fraction(transcript, frac).lower()

    # Specificity priority (Refund/Sale-close > Callback > Resolved)
    priority = ["Refund", "Sale-close", "Callback", "Resolved"]

    # Negations that explicitly block Refund
    negations = [n.lower() for n in outs.get("negations", [])]
    refund_blocked = any(n in tail_text for n in negations)

    for label in priority:
        phrases = [p.lower() for p in outs.get(label, [])]
        if not phrases:
            continue
        # At least one phrase must appear (with a soft negation guard)
        for phrase in phrases:
            if phrase in tail_text and soft_negation_guard(tail_text, phrase):
                if label == "Refund" and refund_blocked:
                    break  # blocked; try next label in priority
                return label

    return "Unknown"


# -----------------------------
# Public API
# -----------------------------
def apply_labels(
    df: pd.DataFrame,
    rules: Dict,
    *,
    duration_cutoff_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply all labels to a copy of df. Expects columns: duration_seconds, transcript.
    Returns a new DataFrame with Connected, Type, Outcome columns set.
    """
    rules = dict(rules or {})
    if duration_cutoff_override is not None:
        rules.setdefault("connected", {})
        rules["connected"]["duration_cutoff_s"] = int(duration_cutoff_override)

    out = df.copy()

    # Connected
    out["Connected"] = [
        connected_label(float(d or 0.0), str(t or ""), rules)
        for d, t in zip(out.get("duration_seconds", []), out.get("transcript", []))
    ]

    # Type
    out["Type"] = [type_label(str(t or ""), rules) for t in out.get("transcript", [])]

    # Outcome
    out["Outcome"] = [outcome_label(str(t or ""), rules) for t in out.get("transcript", [])]

    return out


def explain_matches(transcript: str, rules: Dict, max_items: int = 5) -> List[str]:
    """
    Optional helper: surface up to N matched phrases for UI explanations.
    Returns a list like: ["[Support] …error…", "[Refund] …processed the refund…"]
    """
    if not transcript:
        return []
    t = transcript.lower()
    results: List[str] = []

    # Type phrases
    for bucket, kws in (rules.get("types") or {}).items():
        for kw in kws:
            kw_l = kw.lower()
            if kw_l in t and soft_negation_guard(t, kw_l):
                results.append(f"[{bucket}] …{kw}…")
                if len(results) >= max_items:
                    return results

    # Outcome phrases
    outs = rules.get("outcomes") or {}
    for label in ["Resolved", "Callback", "Refund", "Sale-close"]:
        for phrase in outs.get(label, []):
            ph_l = phrase.lower()
            if ph_l in t and soft_negation_guard(t, ph_l):
                results.append(f"[{label}] …{phrase}…")
                if len(results) >= max_items:
                    return results

    return results[:max_items]
