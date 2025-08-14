# vectorstore/chroma/embedder.py
# Ultra-light, local embedder that avoids heavy ML deps.
# Strategy:
#   - Tokenize text into lowercase words
#   - Hash each token into a fixed-dimensional vector space (feature hashing)
#   - L2-normalize the result
#
# This is *not* SOTA for semantic search, but it's fast, deterministic, and
# works offline with zero extra dependencies. You can swap this out later
# for a sentence-transformer while keeping the same Chroma interface.


import hashlib
import math
import re
import random
from typing import Iterable, List, Sequence

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]{1,30}")


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _hash_token(token: str, dim: int, seed: int) -> int:
    # Use sha1 for stability; combine seed with token
    h = hashlib.sha1(f"{seed}:{token}".encode("utf-8")).digest()
    # take 4 bytes -> int
    idx = int.from_bytes(h[:4], byteorder="little", signed=False)
    return idx % dim


class HashingEmbedder:
    """Dependency-free hash-based embedder (NOT semantic).

    Produces deterministic pseudo-embeddings by hashing tokens into a fixed-size vector.
    Useful for integration tests or when no ML deps should be installed.
    """

    def __init__(self, dim: int = 384, seed: int = 13) -> None:
        self.dim = int(dim)
        self.seed = int(seed)  # kept for ID stability; no RNG used in hashing

    def name(self) -> str:
        """Stable identifier used by Chroma to detect embedder changes."""
        return f"hash/d{self.dim}-s{self.seed}"

    def _tokenize(self, text: str) -> List[str]:
        return (text or "").lower().split()

    def _bucket(self, token: str) -> int:
        # Map a token to a dimension using a stable hash
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % self.dim

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Chroma EmbeddingFunction interface (>= 0.4.16).

        Args:
            input: List of strings to embed.

        Returns:
            List of float vectors (len == dim) in the same order as `input`.
        """
        out: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            toks = self._tokenize(text)
            for tok in toks:
                j = self._bucket(tok)
                vec[j] += 1.0
            # L2 normalize to keep distances comparable
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out