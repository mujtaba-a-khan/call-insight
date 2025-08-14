# tests/test_aggregations.py

import pandas as pd

from core.aggregations import (
    Kpis,
    compute_kpis,
    connection_breakdown,
    type_breakdown,
    outcome_breakdown,
)

def _df():
    return pd.DataFrame(
        {
            "call_id": ["a", "b", "c", "d", "e"],
            "start_time": pd.to_datetime(
                ["2025-08-13", "2025-08-13", "2025-08-14", "2025-08-14", "2025-08-15"]
            ),
            "Connected": ["Connected", "Disconnected", "Connected", "Connected", "Disconnected"],
            "Type": ["Support", "Inquiry", "Billing/Sales", "Support", "Unknown"],
            "Outcome": ["Resolved", "Callback", "Refund", "Sale-close", "Unknown"],
        }
    )


def test_compute_kpis_basic():
    df = _df()
    k = compute_kpis(df, show_unknowns=False)
    assert isinstance(k, Kpis)
    assert k.total_included == 5
    # 3 connected out of 5 where labels present -> 3/(3+2)=60%
    assert round(k.pct_connected, 1) == 60.0
    # Top type: Support appears twice
    assert k.top_type == "Support"
    # Outcomes are all unique except Unknown -> with Unknown hidden, top could be any of the uniques;
    # our helper chooses mode; with all counts equal, pandas picks lexicographic min -> "Callback" or "Refund" or "Resolved" or "Sale-close"
    assert k.top_outcome in {"Resolved", "Callback", "Refund", "Sale-close"}


def test_connection_breakdown_order():
    df = _df()
    br = connection_breakdown(df)
    assert br["Connected"].tolist()[:2] == ["Connected", "Disconnected"]
    assert br["count"].sum() == len(df)


def test_type_breakdown_hide_unknowns():
    df = _df()
    br = type_breakdown(df, show_unknowns=False)
    assert "Unknown" not in br["Type"].values
    # With Unknown shown, length should be +1
    br2 = type_breakdown(df, show_unknowns=True)
    assert len(br2) == len(br) + 1


def test_outcome_breakdown_hide_unknowns():
    df = _df()
    br = outcome_breakdown(df, show_unknowns=False)
    assert "Unknown" not in br["Outcome"].values
    br2 = outcome_breakdown(df, show_unknowns=True)
    assert len(br2) == len(br) + 1
