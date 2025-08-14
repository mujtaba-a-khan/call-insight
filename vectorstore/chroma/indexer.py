# vectorstore/chroma/indexer.py
# Builds or updates a Chroma collection from a pandas DataFrame.

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .backend import (
    ChromaConfig,
    open_collection,
    add_documents,
)

def _extract_columns(
    df: pd.DataFrame,
    *,
    id_field: str,
    text_field: str,
    meta_field: str,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Extract payload columns from a DataFrame.

    Args:
        df: Source dataframe.
        id_field: Column to use as the document ID (e.g., "doc_id" or "call_id").
        text_field: Column with document text (e.g., "text", "transcript", or "summary_text").
        meta_field: Column with per-row metadata dicts.

    Returns:
        Tuple of (ids, texts, metadatas) ready for upsert.

    Raises:
        KeyError: If the required columns are missing.
        ValueError: If any ID is null/empty or any text is missing.
    """
    missing = [c for c in (id_field, text_field) if c not in df.columns]
    if missing:
        raise KeyError(f"Required column(s) missing: {missing}. Have: {list(df.columns)}")

    ids = df[id_field].astype(str).tolist()
    texts = df[text_field].astype(str).tolist()

    # Normalize metadata: accept dicts or build empty dicts
    if meta_field in df.columns:
        metas_series = df[meta_field]
        # Fill NaN/None with {}
        metas = [
            (m if isinstance(m, dict) else {}) for m in metas_series.fillna({}).tolist()
        ]
    else:
        metas = [{} for _ in ids]

    # Basic sanity checks
    if any(not i for i in ids):
        raise ValueError("Found empty document ID(s) in input.")
    if any((t is None) or (str(t).strip() == "") for t in texts):
        raise ValueError("Found empty document text in input.")

    return ids, texts, metas


def build_or_update_index(
    *,
    docs: pd.DataFrame,
    collection_name: str = "calls",
    persist_dir: Path = Path("./vectorstore/chroma"),
    rebuild: bool = False,
    vs_cfg: Optional[Dict[str, Any]] = None,
    id_field: str = "doc_id",
    text_field: str = "text",
    meta_field: str = "metadata",
) -> int:
    """Build or update a Chroma index from a DataFrame.

    This function is intentionally small: it converts the incoming DataFrame
    into (ids, texts, metadatas), opens a persistent collection using the
    configured embedder, and performs an upsert.

    Args:
        docs: DataFrame with at least the ID and text columns.
        collection_name: Chosen collection name (e.g., "calls").
        persist_dir: On-disk path where Chroma stores its DB.
        rebuild: If True, drop and recreate the collection before indexing.
        vs_cfg: Optional vectorstore config dict. Recognized keys:
            - distance: "cosine" | "l2" | "ip"
            - embedder: "hash" | "ollama" | "sbert"
            - embedder_model: Ollama model name
            - sbert_model: sentence-transformers model id
        id_field: Column to use as document ID (default "doc_id").
        text_field: Column with the document text (default "text").
        meta_field: Column with per-row metadata dicts (default "metadata").

    Returns:
        Number of documents upserted.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If payload validation fails.
    """
    # Map DataFrame columns into payload lists
    ids, texts, metas = _extract_columns(
        docs, id_field=id_field, text_field=text_field, meta_field=meta_field
    )

    # Build Chroma configuration from provided vectorstore settings
    cfg_kwargs: Dict[str, Any] = {
        "persist_dir": persist_dir,
        "collection": collection_name,
    }
    if vs_cfg:
        # Only pass through keys we recognize; ignore extras to stay robust.
        if "distance" in vs_cfg:
            cfg_kwargs["distance"] = vs_cfg["distance"]
        if "embedder" in vs_cfg:
            cfg_kwargs["embedder"] = vs_cfg["embedder"]
        if "embedder_model" in vs_cfg:
            cfg_kwargs["embedder_model"] = vs_cfg["embedder_model"]
        if "sbert_model" in vs_cfg:
            cfg_kwargs["sbert_model"] = vs_cfg["sbert_model"]

    col = open_collection(ChromaConfig(**cfg_kwargs), rebuild=rebuild)

    # Upsert documents
    return add_documents(col, ids=ids, texts=texts, metadatas=metas)
