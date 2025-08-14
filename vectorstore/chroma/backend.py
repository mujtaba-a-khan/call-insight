# vectorstore/chroma/backend.py
# Local-first wrapper around ChromaDB with pluggable embedding backends.
# - Ships with a dependency-free hashing embedder.
# - Opt-in semantic quality via Ollama or sentence-transformers (both run locally).
# - Persistent index on disk; safe to rebuild when needed.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb  # type: ignore

# Local embedders
from ..embedders_local import OllamaEmbedder, SentenceTransformerEmbedder  # noqa: F401
# Lightweight hash embedder
from ..embedder import HashingEmbedder  # noqa: F401


@dataclass(frozen=True)
class ChromaConfig:
    """Configuration for a persistent Chroma collection.

    Args:
        persist_dir: Filesystem path where Chroma stores its data.
        collection: Logical collection name.
        dim: Vector dimensionality for the hashing embedder (ignored by Ollama/SBERT).
        seed: Hashing embedder seed for reproducibility.
        distance: Vector metric to use: "cosine", "l2", or "ip".
        embedder: Which embedding backend to use: "hash" | "ollama" | "sbert".
        embedder_model: Ollama embedding model name (e.g., "nomic-embed-text").
        sbert_model: Sentence-Transformers model id (e.g., "sentence-transformers/all-MiniLM-L6-v2").
    """
    persist_dir: Path
    collection: str = "calls"
    dim: int = 384
    seed: int = 13
    distance: str = "cosine"
    embedder: str = "hash"
    embedder_model: str = "nomic-embed-text"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------------------
# Client / Collection helpers
# --------------------------------------------------------------------------------------


def _client(persist_dir: Path) -> "chromadb.api.client.PersistentClient":
    """Create or reuse a persistent Chroma client rooted at ``persist_dir``.

    Ensures the directory exists before instantiation.

    Args:
        persist_dir: Path to the on-disk storage location.

    Returns:
        A Chroma ``PersistentClient`` instance.
    """
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _collection(
    client: "chromadb.api.client.PersistentClient",
    name: str,
    embedder: Any,
    *,
    recreate: bool = False,
    distance: str = "cosine",
):
    """Open or create a collection bound to an embedding function.

    Tries to set HNSW space via collection metadata; falls back cleanly if the
    installed Chroma version ignores or changes the parameter.

    Args:
        client: Persistent Chroma client.
        name: Collection name.
        embedder: Callable embedding function (texts -> List[List[float]]).
        recreate: If True, drop and recreate the collection.
        distance: Space metric ("cosine", "l2", "ip").

    Returns:
        A Chroma collection object.
    """
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            # Non-fatal if the collection doesn't exist yet.
            pass

    try:
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedder,
            metadata={"hnsw:space": distance},
        )
    except Exception:
        # Older/newer API variants may not accept metadata; retry without it.
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedder,
        )


def open_collection(cfg: ChromaConfig, *, rebuild: bool = False):
    """Open a persistent collection using the configured embedder.

    Args:
        cfg: Chroma configuration.
        rebuild: If True, drop and recreate the collection from scratch.

    Returns:
        A Chroma collection bound to the chosen embedding backend.

    Raises:
        ValueError: If an unknown embedder is specified.
    """
    client = _client(cfg.persist_dir)

    # Pick an embedding backend:
    # - "hash": zero extra deps, good for smoke tests and keyword-ish matching.
    # - "ollama": strong local semantic embeddings (requires Ollama + pulled model).
    # - "sbert": strong semantic embeddings via sentence-transformers.
    if cfg.embedder == "hash":
        ef = HashingEmbedder(dim=cfg.dim, seed=cfg.seed)
    elif cfg.embedder == "ollama":
        ef = OllamaEmbedder(model=cfg.embedder_model)
    elif cfg.embedder in ("sbert", "sentence-transformers"):
        ef = SentenceTransformerEmbedder(model=cfg.sbert_model)
    else:
        raise ValueError(f"Unknown embedder '{cfg.embedder}'")

    return _collection(
        client,
        cfg.collection,
        ef,
        recreate=rebuild,
        distance=cfg.distance,
    )


# --------------------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------------------


def add_documents(
    collection: Any,
    *,
    ids: Sequence[str],
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """Upsert documents into the collection.

    Prefers ``upsert`` when available to avoid duplicate key errors; falls back to
    ``add`` for older Chroma versions.

    Args:
        collection: Chroma collection object.
        ids: Stable unique IDs per document (e.g., call_id).
        texts: Document text payloads (e.g., transcript or summary+transcript).
        metadatas: Optional list of per-doc metadata dicts.

    Returns:
        Number of documents processed.
    """
    n = len(ids)
    if n == 0:
        return 0

    payload_metas = list(metadatas or [{} for _ in ids])
    try:
        collection.upsert(
            ids=list(ids),
            documents=list(texts),
            metadatas=payload_metas,
        )
    except AttributeError:
        collection.add(
            ids=list(ids),
            documents=list(texts),
            metadatas=payload_metas,
        )
    return n


def delete_ids(collection: Any, ids: Iterable[str]) -> int:
    """Delete documents by ID.

    Args:
        collection: Chroma collection.
        ids: Iterable of document IDs.

    Returns:
        Number of IDs attempted for deletion (0 if call is a no-op or fails).
    """
    to_delete = list(ids)
    if not to_delete:
        return 0
    try:
        collection.delete(ids=to_delete)
        return len(to_delete)
    except Exception:
        # Non-fatal; return 0 to signal nothing removed.
        return 0


def wipe(collection_name: str, persist_dir: Path) -> None:
    """Drop an entire collection.

    Safe to call even if the collection does not exist.

    Args:
        collection_name: Name of the collection to remove.
        persist_dir: Root path of the Chroma storage.
    """
    client = _client(persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------------------


def query(
    collection: Any,
    *,
    text: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Similarity search over the collection.

    Requests distances explicitly to keep behavior consistent across Chroma versions.

    Args:
        collection: Chroma collection.
        text: Query text from the user.
        top_k: Number of nearest neighbors to return.
        where: Optional metadata filter applied server-side (e.g., {"agent": "Sara"}).

    Returns:
        Dict with flat lists:
            - "ids": List[str]
            - "distances": List[float]
            - "documents": List[str]
            - "metadatas": List[dict]
    """
    if not text or top_k <= 0:
        return {"ids": [], "distances": [], "documents": [], "metadatas": []}

    kwargs: Dict[str, Any] = {
        "query_texts": [text],
        "n_results": int(top_k),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    res = collection.query(**kwargs)

    # Some Chroma versions wrap outputs as [[...]]; normalize to 1-D lists.
    def _first(x: Any, default: Any):
        return (x or [default])[0] if isinstance(x, list) else (x or default)

    return {
        "ids": _first(res.get("ids"), []),
        "distances": _first(res.get("distances"), []),
        "documents": _first(res.get("documents"), []),
        "metadatas": _first(res.get("metadatas"), []),
    }
