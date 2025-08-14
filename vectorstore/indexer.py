# vectorstore/chroma/indexer.py
# High-level indexer used by scripts/build_vector_index.py
# - Opens (or rebuilds) a Chroma collection
# - Sends documents through our local embedder
# - Returns number of documents indexed

"""Compatibility shim for legacy imports."""
from .chroma.indexer import build_or_update_index  # re-export

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from .chroma.backend import ChromaConfig, add_documents, open_collection


def build_or_update_index(
    *,
    docs: pd.DataFrame,              # columns: doc_id, text, metadata
    collection_name: str = "calls",
    persist_dir: Path = Path("./vectorstore/chroma"),
    rebuild: bool = False,
) -> int:
    cfg = ChromaConfig(persist_dir=persist_dir, collection=collection_name)
    col = open_collection(cfg, rebuild=rebuild)

    if docs.empty:
        return 0

    # Ensure required columns
    for c in ["doc_id", "text"]:
        if c not in docs.columns:
            raise ValueError(f"docs missing required column: {c}")

    # Fill metadata
    if "metadata" not in docs.columns:
        docs["metadata"] = [{} for _ in range(len(docs))]

    n = add_documents(
        col,
        ids=list(docs["doc_id"].astype(str).values),
        texts=list(docs["text"].astype(str).values),
        metadatas=list(docs["metadata"].values),
    )
    return int(n)
