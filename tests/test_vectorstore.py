# tests/test_vectorstore.py

import pandas as pd

from vectorstore.chroma.indexer import build_or_update_index
from vectorstore.chroma.backend import ChromaConfig, open_collection, query


def test_chroma_index_and_query(tmp_path):
    # Prepare tiny doc set
    docs = pd.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "text": [
                "order placed subscription activated payment processed",
                "processed the refund credited your account reversal posted",
                "troubleshoot error not working reset configuration failed",
            ],
            "metadata": [
                {"call_id": "a", "Type": "Billing/Sales", "Outcome": "Sale-close"},
                {"call_id": "b", "Type": "Billing/Sales", "Outcome": "Refund"},
                {"call_id": "c", "Type": "Support", "Outcome": "Resolved"},
            ],
        }
    )

    # Build index in a temp persistent dir
    n = build_or_update_index(docs=docs, collection_name="calls", persist_dir=tmp_path, rebuild=True)
    assert n == 3

    # Open collection and run a couple of queries
    col = open_collection(ChromaConfig(persist_dir=tmp_path, collection="calls"))
    q1 = query(col, text="order placed subscription", top_k=3)
    q2 = query(col, text="processed refund credited", top_k=3)
    q3 = query(col, text="error not working reset", top_k=3)

    # We expect the relevant IDs to be among the results
    assert "a" in q1["ids"]
    assert "b" in q2["ids"]
    assert "c" in q3["ids"]
