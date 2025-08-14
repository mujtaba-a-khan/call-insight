# tests/test_chroma_backend_hash.py
# Verifies basic upsert/query flow against Chroma using the dependency-free hash embedder.

import tempfile
from pathlib import Path
from typing import Dict, Any, List

from vectorstore.chroma.backend import (
    ChromaConfig,
    open_collection,
    add_documents,
    query as vs_query,
)


def test_backend_upsert_and_query_with_hash_embedder():
    """Index three short docs with the hash embedder and retrieve by similarity.

    This test uses a real Chroma persistent client pointed at a temporary
    directory. No network or heavy ML dependencies are required.
    """
    with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmp:
        cfg = ChromaConfig(
            persist_dir=Path(tmp),
            collection="utest_calls",
            distance="cosine",
            embedder="hash",
            # dim/seed only affect the hash embedder
            # defaults are fine, but we keep them explicit in tests for clarity
            dim=384,
            seed=13,
        )
        col = open_collection(cfg, rebuild=True)

        ids = ["c1", "c2", "c3"]
        docs = [
            "refund policy and returns",
            "technical support for home router setup",
            "billing issue found on monthly invoice",
        ]
        metas: List[Dict[str, Any]] = [
            {"call_id": "c1"},
            {"call_id": "c2"},
            {"call_id": "c3"},
        ]

        n = add_documents(col, ids=ids, texts=docs, metadatas=metas)
        assert n == len(ids)

        res = vs_query(col, text="requesting a refund for returned item", top_k=2)
        # Should return up to 2 IDs, including c1 as the closest match
        assert isinstance(res, dict)
        assert "ids" in res and len(res["ids"]) <= 2
        assert "c1" in res["ids"]
