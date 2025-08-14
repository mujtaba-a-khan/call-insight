# core/utils.py
# Small, dependency-light helpers used across core modules.
# Keep this file stable; other modules should not reimplement these.

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


# ---- Paths & filesystem ----------------------------------------------------


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it doesn't exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- Time ------------------------------------------------------------------


def now_iso(tz: Optional[str] = None) -> str:
    """
    Return an ISO-8601 timestamp. If tz is provided (e.g., 'Europe/Berlin'),
    we try to use it; otherwise fall back to local time.
    """
    try:
        import zoneinfo  # Python 3.9+
        z = zoneinfo.ZoneInfo(tz) if tz else None
        return datetime.now(tz=z).isoformat(timespec="seconds")
    except Exception:
        return datetime.now().astimezone().isoformat(timespec="seconds")


# ---- Hashing & caching keys -------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    """Hex digest of bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cache_key_from_parts(*parts: str) -> str:
    """
    Build a stable key from arbitrary string parts (e.g., filename, mtime, params).
    """
    joined = "||".join(parts)
    return sha256_bytes(joined.encode("utf-8"))


# ---- Text utilities ---------------------------------------------------------


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'_-]*")  # simple English-ish tokens


def normalize_ws(text: str) -> str:
    """Collapse whitespace to single spaces; strip ends."""
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", text).strip()


def tokenize_english(text: str) -> List[str]:
    """
    Return a list of lightweight English-ish tokens.
    This is deterministic and fast; no external NLP dependencies.
    """
    if not isinstance(text, str) or not text:
        return []
    return _TOKEN_RE.findall(text)


def english_token_count(text: str) -> int:
    """Convenience wrapper used by rules/exclusions."""
    return len(tokenize_english(text))


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    """Case-insensitive substring search for any of the phrases."""
    if not isinstance(text, str) or not text:
        return False
    t = text.lower()
    for p in phrases:
        if p and str(p).lower() in t:
            return True
    return False


def soft_negation_guard(text: str, phrase: str) -> bool:
    """
    Heuristic: returns True if `phrase` appears *without* a nearby negation.
    Negations considered: no, not, never, without, deny/denied/denies.
    """
    if not isinstance(text, str) or not phrase:
        return False
    t = text.lower()
    p = phrase.lower()
    idx = t.find(p)
    if idx == -1:
        return False
    window_start = max(0, idx - 20)
    window = t[window_start:idx]
    return not re.search(r"\b(no|not|never|without|deny|denied|denies)\b", window)


# ---- PII masking ------------------------------------------------------------


def mask_name(name: str) -> str:
    """
    Mask a name like "Jane Smith" -> "J*** S****".
    Keeps length hints without exposing the full string.
    """
    if not isinstance(name, str) or not name.strip():
        return name
    parts = [p for p in name.strip().split() if p]
    masked = []
    for p in parts:
        if len(p) <= 1:
            masked.append(p + "*")
        else:
            masked.append(p[0] + "*" * (len(p) - 1))
    return " ".join(masked)


# ---- Text windowing ---------------------------------------------------------


def tail_fraction(text: str, frac: float) -> str:
    """
    Return the last `frac` fraction of tokens from text (0<frac<=1).
    Used by outcome rules to focus on the tail of the transcript.
    """
    frac = max(0.0, min(1.0, float(frac or 0.25)))
    toks = tokenize_english(text)
    if not toks:
        return ""
    start = int(len(toks) * (1.0 - frac))
    return " ".join(toks[start:])
