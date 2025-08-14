# tests/test_interpreter.py

from llm.interpreter import Catalog, build_filter_spec


def test_build_filter_spec_sales_example():
    question = "Sales-related calls on 2025-08-13; who was the customer and what they purchased?"
    cfg = {
        "timezone": "Europe/Berlin",
        "synonyms": {
            "sales-related": {"type_any": ["Billing/Sales"], "outcome_any": ["Sale-close"]},
        },
        "filter_schema": {"allowed_keys": ["date_from", "date_to", "type_any", "outcome_any", "agents", "campaigns", "products", "min_amount", "max_amount"]},
    }
    spec, notes = build_filter_spec(question, cfg, catalog=Catalog(agents=["Alice"], campaigns=["Q3-Sales"]))
    assert spec["date_from"] == "2025-08-13" and spec["date_to"] == "2025-08-13"
    assert "Billing/Sales" in spec.get("type_any", []) or "Sale-close" in spec.get("outcome_any", [])
    assert isinstance(notes, str)


def test_build_filter_spec_aliases_and_amounts():
    q = "refunds last week from $50 to $200 for agent Alice in campaign Sales-EMEA"
    cfg = {"timezone": "Europe/Berlin", "synonyms": {"refunds": {"outcome_any": ["Refund"]}}, "entity_match": {"score_threshold": 0.8}}
    spec, _ = build_filter_spec(q, cfg, catalog=Catalog(agents=["Alice", "Bob"], campaigns=["Sales-EMEA", "Support-Q3"]))
    assert spec["outcome_any"] == ["Refund"]
    assert "date_from" in spec and "date_to" in spec
    assert spec["min_amount"] == 50.0 and spec["max_amount"] == 200.0
    assert spec["agents"] == ["Alice"]
    assert spec["campaigns"] == ["Sales-EMEA"]


def test_build_filter_spec_no_categories():
    q = "show me everything for yesterday"
    spec, notes = build_filter_spec(q, {"timezone": "Europe/Berlin"})
    assert "date_from" in spec and "date_to" in spec
    # no categories means empty lists
    assert spec.get("type_any", []) == []
    assert spec.get("outcome_any", []) == []
    assert "No specific categories" in notes or notes == ""
