# vectorstore/embedders_local.py
# Local embedders compatible with Chroma's embedding_function interface.

from typing import Sequence, List, Optional


class OllamaEmbedder:
    """EmbeddingFunction that calls a local Ollama embedding model.

    Requires the `ollama` Python package and the Ollama daemon running locally.
    """

    def __init__(self, model: str = "nomic-embed-text", normalize: bool = True, batch_size: int = 64) -> None:
        self.model = model
        self.normalize = bool(normalize)
        self.batch_size = int(batch_size)

        try:
            import ollama  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Ollama Python client not available. Install the 'llm' extra: pip install .[llm]"
            ) from e

    def name(self) -> str:
        """Stable identifier used by Chroma to detect embedder changes."""
        return f"ollama/{self.model}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Chroma EmbeddingFunction interface.

        Args:
            input: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        import math
        import ollama  # type: ignore

        texts: List[str] = list(input or [])
        if not texts:
            return []

        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            res = ollama.embed(model=self.model, input=batch)
            # Normalize return shape across ollama versions
            vecs = res.get("embeddings") or res.get("data") or []
            if isinstance(vecs, dict) and "embedding" in vecs:  # single-input legacy form
                vecs = [vecs["embedding"]]
            if self.normalize:
                for j, v in enumerate(vecs):
                    norm = math.sqrt(sum(x * x for x in v)) or 1.0
                    vecs[j] = [x / norm for x in v]
            out.extend(vecs)
        return out


class SentenceTransformerEmbedder:
    """EmbeddingFunction using a local Sentence-Transformers model.

    Requires `sentence-transformers` (install via the 'sbert' extra).
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True) -> None:
        self.model = model
        self.normalize = bool(normalize)

        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Sentence-Transformers not available. Install the 'sbert' extra: pip install .[sbert]"
            ) from e

        # Lazy-load the heavy model at first call
        self._sbert = None

    def name(self) -> str:
        """Stable identifier used by Chroma to detect embedder changes."""
        return f"sbert/{self.model}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Chroma EmbeddingFunction interface.

        Args:
            input: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        from sentence_transformers import SentenceTransformer

        texts: List[str] = list(input or [])
        if not texts:
            return []

        if self._sbert is None:
            self._sbert = SentenceTransformer(self.model)

        vecs = self._sbert.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=False,
        )
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]