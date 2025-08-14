# core/semantic_search.py
# Utilities for semantic re-ranking of Q&A results using the local Chroma index.

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from vectorstore.chroma.backend import ChromaConfig, open_collection, query as vs_query


def load_vectorstore_cfg(app_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load vectorstore config with sensible fallbacks.

    Args:
        app_cfg: Parsed app.yml content.

    Returns:
        Vectorstore configuration. Prefer configs/vectorstore.yml if present,
        else the vectorstore block from app.yml. Missing keys are fine.
    """
    import yaml, os
    vs_path = Path("configs/vectorstore.yml")
    if vs_path.exists():
        raw = yaml.safe_load(vs_path.read_text(encoding="utf-8")) or {}
        return (raw.get("vectorstore") or {}) if isinstance(raw, dict) else {}
    return app_cfg.get("vectorstore", {}) if isinstance(app_cfg, dict) else {}


def semantic_rank_intersect(
    df: pd.DataFrame,
    *,
    question: str,
    vs_cfg: Dict[str, Any],
    top_k: int = 50,
    id_col: str = "call_id",
) -> pd.DataFrame:
    """Intersect a filtered calls DataFrame with semantics-ranked IDs from Chroma.

    The function **does not** expand the scope; it only re-ranks (and potentially
    reduces) `df` by intersecting with semantic top-k IDs from the vector index.

    Args:
        df: Already-filtered calls table (client-side filters/interpreter).
        question: Natural-language query entered by the user.
        vs_cfg: Vectorstore configuration dict (persist_dir, collection, embedder, etc.).
        top_k: Number of nearest neighbors to request from Chroma.
        id_col: Column in `df` that matches the vector doc IDs (usually "call_id").

    Returns:
        A copy of `df` re-ranked by semantic distance where possible. If the
        vector store is unavailable or `question` is empty, returns `df` unchanged.
    """
    if df.empty or not question.strip() or id_col not in df.columns:
        return df

    try:
        cfg = ChromaConfig(
            persist_dir=Path(vs_cfg.get("persist_dir", "./vectorstore/chroma")),
            collection=vs_cfg.get("collection", "calls"),
            distance=vs_cfg.get("distance", "cosine"),
            embedder=vs_cfg.get("embedder", "hash"),
            embedder_model=vs_cfg.get("embedder_model", "nomic-embed-text"),
            sbert_model=vs_cfg.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        col = open_collection(cfg, rebuild=False)
        res = vs_query(col, text=question, top_k=int(top_k))
        sem_ids = [str(x) for x in (res.get("ids") or [])]
        if not sem_ids:
            return df

        # Intersect and stable-rank by semantic order.
        order = {cid: i for i, cid in enumerate(sem_ids)}
        out = df[df[id_col].astype(str).isin(order.keys())].copy()
        if out.empty:
            # No overlap between filter scope and semantics; keep original.
            return df
        out["_rank"] = out[id_col].astype(str).map(lambda x: order.get(x, 10**9))
        out = out.sort_values("_rank", kind="stable").drop(columns=["_rank"])
        return out
    except Exception:
        # Fail open: keep deterministic results if vectorstore not ready.
        return df
