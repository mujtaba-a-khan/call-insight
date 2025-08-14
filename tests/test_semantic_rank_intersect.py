# tests/test_semantic_rank_intersect.py
# Ensures semantic_rank_intersect() reorders by semantic IDs and intersects correctly.

import pandas as pd

import core.semantic_search as ss


class _DummyCollection:
    """Minimal stand-in for a Chroma collection used by the monkeypatched query."""
    pass


def test_semantic_rank_intersect_orders_and_intersects(monkeypatch):
    """Given a filtered DataFrame, ensure we re-rank by semantic order and intersect IDs.

    We monkeypatch core.semantic_search.vs_query to return a fixed ID order and
    core.semantic_search.open_collection to avoid touching the real vector store.
    """
    # Build a small filtered scope
    df = pd.DataFrame(
        {
            "call_id": ["A", "B", "C", "D"],
            "Type": ["Sales", "Support", "Billing", "Sales"],
            "Outcome": ["Resolved", "Callback", "Refund", "Resolved"],
        }
    )

    # Fake vectorstore config (values are ignored by our stubs)
    vs_cfg = {
        "persist_dir": "./tmp",
        "collection": "calls",
        "distance": "cosine",
        "embedder": "hash",
    }

    # Monkeypatch the VS calls to avoid external deps
    def _fake_open_collection(cfg, rebuild=False):  # noqa: ARG001
        return _DummyCollection()

    def _fake_query(collection, *, text, top_k=5, where=None):  # noqa: ARG001
        # Pretend the semantic search says C is best, then A, then E (not in df)
        return {"ids": ["C", "A", "E"], "documents": [], "metadatas": [], "distances": [0.1, 0.3, 0.9]}

    monkeypatch.setattr(ss, "open_collection", _fake_open_collection)
    monkeypatch.setattr(ss, "vs_query", _fake_query)

    ranked = ss.semantic_rank_intersect(
        df,
        question="who asked for a refund?",
        vs_cfg=vs_cfg,
        top_k=5,
        id_col="call_id",
    )

    # Expect only intersection with ["C", "A"] and in that order
    assert ranked["call_id"].tolist() == ["C", "A"]
