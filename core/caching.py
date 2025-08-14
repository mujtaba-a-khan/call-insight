# core/caching.py
# Small, readable disk cache helpers.
# We keep caches human-inspectable (JSON alongside small blobs) and safe to delete.
#
# Layout (relative to project root):
#   cache/
#     stt/      # transcripts + metadata keyed by (filename, mtime, params)
#     labels/   # deterministic labels keyed by (transcript_hash, rules)
#
# Typical use:
#   from core.caching import stt_cache_key, labels_cache_key, load_json, save_json
#   key = stt_cache_key(path, mtime, model="small.en", compute="int8")
#   data = load_json(stt_dir / f"{key}.json")
#   if not data: ...run STT...; save_json(stt_dir / f"{key}.json", payload)

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import cache_key_from_parts, ensure_dir, sha256_bytes


# --- Project-relative cache roots ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ensure_dir(PROJECT_ROOT / "cache")
STT_DIR = ensure_dir(CACHE_ROOT / "stt")
LABELS_DIR = ensure_dir(CACHE_ROOT / "labels")


# --- Generic JSON IO -------------------------------------------------------------

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file; return None if missing or malformed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        # Corrupt cache entries can be ignored; callers may rebuild them.
        return None


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file atomically (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def remove(path: Path) -> None:
    """Best-effort delete; ignore if absent."""
    try:
        path.unlink(missing_ok=True)  # py3.8+: use try/except instead
    except Exception:
        pass


# --- Key builders (stable) ------------------------------------------------------

def stt_cache_key(
    *,
    file_name: str,
    file_mtime_ns: int,
    file_size: int,
    model: str,
    compute: str,
) -> str:
    """
    Build a stable key for STT results based on the *input file identity* and STT params.
    Use precise mtime (ns) + size to survive renames and catch edits.
    """
    return cache_key_from_parts("stt", file_name, str(file_mtime_ns), str(file_size), model, compute)


def labels_cache_key(
    *,
    transcript_hash: str,
    duration_cutoff_s: int,
    min_tokens: int,
    outcome_tail_frac: float,
) -> str:
    """
    Build a stable key for deterministic labels. If any rule changes, key changes.
    """
    return cache_key_from_parts(
        "labels",
        transcript_hash,
        f"cut={duration_cutoff_s}",
        f"tok={min_tokens}",
        f"tail={outcome_tail_frac}",
    )


# --- High-level helpers for STT & labels ---------------------------------------

@dataclass(frozen=True)
class SttCachePaths:
    json: Path  # main payload (transcript, segments, meta)


def stt_paths(key: str) -> SttCachePaths:
    """Compute STT cache file paths for a given key."""
    return SttCachePaths(json=STT_DIR / f"{key}.json")


def read_stt(key: str) -> Optional[Dict[str, Any]]:
    """
    Return STT payload or None.
    Payload shape (suggested):
      {
        "transcript": "...",
        "segments": [...],         # optional
        "duration_s": 123.4,       # optional
        "model": "small.en",
        "compute": "int8",
        "source": {"name": "...", "mtime_ns": 0, "size": 0}
      }
    """
    return load_json(stt_paths(key).json)


def write_stt(key: str, payload: Dict[str, Any]) -> None:
    save_json(stt_paths(key).json, payload)


@dataclass(frozen=True)
class LabelsCachePaths:
    json: Path  # labels payload per transcript hash + rules


def labels_paths(key: str) -> LabelsCachePaths:
    return LabelsCachePaths(json=LABELS_DIR / f"{key}.json")


def read_labels(key: str) -> Optional[Dict[str, Any]]:
    """
    Return labels payload or None.
    Payload shape (suggested):
      {
        "Connected": "Connected|Disconnected",
        "Type": "Inquiry|Billing/Sales|Support|Complaint|Unknown",
        "Outcome": "Resolved|Callback|Refund|Sale-close|Unknown",
        "explain": { "type_hits": [...], "outcome_hits": [...] }  # optional
      }
    """
    return load_json(labels_paths(key).json)


def write_labels(key: str, payload: Dict[str, Any]) -> None:
    save_json(labels_paths(key).json, payload)


# --- Convenience utilities ------------------------------------------------------

def transcript_hash(text: str) -> str:
    """Hash the transcript text to use as a durable content key."""
    if not isinstance(text, str):
        text = ""
    return sha256_bytes(text.encode("utf-8"))


def clear_cache(kind: str = "all") -> None:
    """
    Remove cached entries.
      kind = "stt" | "labels" | "all"
    """
    def _rm_dir(path: Path):
        if not path.exists():
            return
        for p in path.glob("**/*"):
            if p.is_file():
                with suppress(Exception):
                    p.unlink()
    from contextlib import suppress

    if kind in ("stt", "all"):
        _rm_dir(STT_DIR)
    if kind in ("labels", "all"):
        _rm_dir(LABELS_DIR)
