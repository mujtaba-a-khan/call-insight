# tests/test_filters.py

import pandas as pd

from core.filters import SidebarFilters, apply_qna_filter_spec, apply_sidebar_filters


def _df():
    return pd.DataFrame(
        {
            "call_id": ["a", "b", "c", "d"],
            "start_time": pd.to_datetime(
                ["2025-08-13 09:00", "2025-08-13 18:30", "2025-08-14 10:00", "2025-08-15 12:00"]
            ),
            "agent_id": ["Alice", "Bob", "Alice", "Cara"],
            "campaign": ["Sales-EMEA", "Support-Q3", "Sales-EMEA", "Sales-NA"],
            "Type": ["Billing/Sales", "Support", "Inquiry", "Complaint"],
            "Outcome": ["Sale-close", "Resolved", "Refund", "Callback"],
            "product_name": ["Widget Pro", "Starter Plan", "Widget Lite", "Gadget"],
            "amount": [120.0, 0.0, 25.5, 300.0],
        }
    )


# ---------- Sidebar filters ----------

def test_sidebar_date_range_inclusive_by_day():
    df = _df()
    f = SidebarFilters(date_from=pd.Timestamp("2025-08-13").date(), date_to=pd.Timestamp("2025-08-14").date())
    out = apply_sidebar_filters(df, f)
    # Includes all rows on 13th and 14th, excludes 15th
    assert set(out["call_id"]) == {"a", "b", "c"}


def test_sidebar_agent_and_campaign_filters():
    df = _df()
    f = SidebarFilters(agent="Alice", campaign="Sales-EMEA")
    out = apply_sidebar_filters(df, f)
    assert set(out["call_id"]) == {"a", "c"}
    # With campaign narrowed, only "a" and "c" exist anyway
    f2 = SidebarFilters(agent="Alice", campaign="Sales-NA")
    out2 = apply_sidebar_filters(df, f2)
    assert out2.empty


# ---------- Q&A spec filters ----------

def test_qna_basic_date_and_category_filters():
    df = _df()
    spec = {
        "date_from": "2025-08-13",
        "date_to": "2025-08-13",
        "type_any": ["Billing/Sales"],
        "outcome_any": ["Sale-close"],
    }
    out = apply_qna_filter_spec(df, spec)
    # Only row "a" matches both type and outcome on 13th
    assert set(out["call_id"]) == {"a"}


def test_qna_agents_campaigns_products_amount():
    df = _df()
    spec = {
        "agents": ["Alice"],
        "campaigns": ["Sales-EMEA"],
        "products": ["widget"],  # substring match, case-insensitive
        "min_amount": 50.0,
        "max_amount": 200.0,
    }
    out = apply_qna_filter_spec(df, spec)
    # Rows a (Widget Pro, 120) and c (Widget Lite, 25.5) — but c fails min_amount
    assert set(out["call_id"]) == {"a"}


def test_qna_amount_only_and_unknown_columns_tolerant():
    df = _df().drop(columns=["product_name"])
    spec = {"min_amount": 100.0}
    out = apply_qna_filter_spec(df, spec)
    assert set(out["call_id"]) == {"a", "d"}  # 120 and 300


def test_qna_handles_empty_df():
    out = apply_qna_filter_spec(pd.DataFrame(), {"date_from": "2025-08-13"})
    assert out.empty
