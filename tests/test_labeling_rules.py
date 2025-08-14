# tests/test_labeling_rules.py

import pandas as pd

from core.labeling_rules import (
    apply_labels,
    connected_label,
    outcome_label,
    type_label,
)

RULES = {
    "connected": {
        "duration_cutoff_s": 30,
        "min_tokens": 40,
        "short_disconnected_s": 10,
        "disconnection_cues": ["voicemail", "leave a message", "the number you have dialed"],
    },
    "types": {
        "Inquiry": ["information", "question about", "how to"],
        "Billing/Sales": ["invoice", "payment", "order", "subscription"],
        "Support": ["error", "not working", "bug", "fix"],
        "Complaint": ["complaint", "unacceptable", "poor service"],
    },
    "type_tie_break": ["Complaint", "Support", "Billing/Sales", "Inquiry"],
    "outcomes": {
        "Resolved": ["issue is resolved", "should work now"],
        "Callback": ["schedule a callback", "ring you back"],
        "Refund": ["processed the refund", "issued a refund", "credited your account"],
        "Sale-close": ["order placed", "subscription activated", "payment processed"],
        "negations": ["refund denied", "no refund"],
    },
    "outcome_window_frac": 0.25,
}


def _tok(n: int, word="hello"):
    return " ".join([word] * n)


# ---------- Connected ----------

def test_connected_duration_and_tokens():
    # Long enough duration + tokens
    text = _tok(45)
    assert connected_label(40, text, RULES) == "Connected"
    # Not enough tokens
    assert connected_label(40, _tok(10), RULES) == "Disconnected"
    # Very short duration
    assert connected_label(5, _tok(100), RULES) == "Disconnected"


def test_connected_disconnection_cues():
    t = "we reached voicemail please leave a message after the tone"
    assert connected_label(120, t, RULES) == "Disconnected"


# ---------- Type ----------

def test_type_simple_and_tie_break():
    assert type_label("there is an error and it is not working", RULES) == "Support"
    # Contains both support and complaint hits → tie-break picks Complaint
    t = "this is unacceptable, I want to file a complaint. also there is an error"
    assert type_label(t, RULES) == "Complaint"


# ---------- Outcome ----------

def test_outcome_tail_window_and_negation():
    # Refund appears in tail
    t = _tok(100) + " processed the refund yesterday"
    assert outcome_label(t, RULES) == "Refund"
    # Refund phrase but blocked by negation
    t2 = _tok(100) + " refund denied due to policy"
    assert outcome_label(t2, RULES) != "Refund"


# ---------- DataFrame apply ----------

def test_apply_labels_dataframe():
    df = pd.DataFrame(
        {
            "call_id": ["a", "b", "c"],
            "start_time": pd.to_datetime(["2025-08-13", "2025-08-14", "2025-08-15"]),
            "duration_seconds": [35, 8, 50],
            "transcript": [
                "customer reports an error; should work now after reset",
                "voicemail please leave a message",
                "order placed and subscription activated",
            ],
        }
    )
    out = apply_labels(df, RULES)
    assert set(["Connected", "Type", "Outcome"]).issubset(out.columns)

    row_a = out.loc[out["call_id"] == "a"].iloc[0]
    assert row_a["Connected"] in {"Connected", "Disconnected"}
    assert row_a["Type"] in {"Support", "Complaint", "Billing/Sales", "Inquiry", "Unknown"}
    assert row_a["Outcome"] in {"Resolved", "Callback", "Refund", "Sale-close", "Unknown"}

    # Spot-check expected labels
    assert out.loc[out["call_id"] == "b", "Connected"].item() == "Disconnected"
    assert out.loc[out["call_id"] == "c", "Outcome"].item() == "Sale-close"
