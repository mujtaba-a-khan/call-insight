# scripts/build_vector_index.py
# Build or update the local vector index from labeled calls.
# - Reads snapshots (artifacts/state/*.parquet|csv) or optional CSV globs
# - Ensures labels exist using deterministic rules
# - Prepares compact docs (id, text, metadata)
# - Hands docs to the Chroma indexer with a pluggable local embedder

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from core.labeling_rules import apply_labels
from core.schema import ensure_columns, normalize_calls_df
from core.storage import LocalStorage
from vectorstore.chroma.indexer import build_or_update_index # unified indexer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dict.

    Args:
        path: Filesystem path to a YAML file.

    Returns:
        A dictionary parsed from YAML. Returns an empty dict on errors or if the
        file does not exist.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _load_vectorstore_cfg(app_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load vectorstore configuration with a sensible fallback.

    Prefers a dedicated ``configs/vectorstore.yml`` if present. Otherwise it
    falls back to the ``vectorstore`` block in ``configs/app.yml``.

    Args:
        app_cfg: The already-loaded application configuration (from app.yml).

    Returns:
        The vectorstore configuration dictionary. Missing keys are fine; the
        indexer applies defaults.
    """
    vs_path = CFG_DIR / "vectorstore.yml"
    if vs_path.exists():
        cfg = _load_yaml(vs_path)
        return cfg.get("vectorstore", {}) if isinstance(cfg, dict) else {}
    return app_cfg.get("vectorstore", {}) if isinstance(app_cfg, dict) else {}


def _load_calls_from_snapshots() -> pd.DataFrame:
    """Load the latest normalized calls table from local snapshots.

    Returns:
        A DataFrame of calls. Can be empty if no snapshots exist yet.
    """
    storage = LocalStorage()
    calls, _ = storage.load_snapshots()
    return calls


def _load_calls_from_csv_glob(glob_pattern: str) -> pd.DataFrame:
    """Load and normalize calls from a CSV glob when snapshots are absent.

    Args:
        glob_pattern: Glob pattern for CSVs, e.g. ``"data/input_csv/*.csv"``.

    Returns:
        A normalized DataFrame of calls. Empty if no files matched or all failed.
    """
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            # Skip unreadable files; continue with others.
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return normalize_calls_df(ensure_columns(df))


def _prepare_docs(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Transform the calls table into the minimal doc schema for indexing.

    The indexer expects three columns:
        - ``doc_id`` (string): unique document ID (mapped from ``call_id``)
        - ``text`` (string): concatenated text from selected fields
        - ``metadata`` (dict): lightweight metadata per document

    Args:
        df: Source DataFrame of calls (already normalized).
        fields: Column names to concatenate into the ``text`` field.

    Returns:
        A DataFrame with columns ``doc_id``, ``text``, ``metadata``.
        Empty if no non-empty text rows remain.
    """
    if df.empty:
        return df

    # Ensure expected text fields exist; missing ones become empty strings.
    for f in fields:
        if f not in df.columns:
            df[f] = ""

    # Build the text payload per row, skipping missing/blank parts.
    def _row_text(row: pd.Series) -> str:
        parts: List[str] = []
        for f in fields:
            v = row.get(f)
            if pd.notna(v):
                s = str(v).strip()
                if s:
                    parts.append(s)
        return "\n".join(parts)

    # Robust metadata builder (avoid KeyError if optional columns are missing).
    def _row_meta(row: pd.Series) -> Dict[str, Any]:
        return {
            "call_id": str(row.get("call_id", "")),
            "start_time": str(row.get("start_time", "")),
            "agent_id": str(row.get("agent_id", "")),
            "agent": str(row.get("agent", "")),
            "campaign": str(row.get("campaign", "")),
            "type": str(row.get("Type", "")),
            "outcome": str(row.get("Outcome", "")),
            "product": str(row.get("product", "")),
        }

    out = pd.DataFrame(
        {
            "doc_id": df["call_id"].astype(str),
            "text": df.apply(_row_text, axis=1),
            "metadata": df.apply(_row_meta, axis=1),
        }
    )

    # Drop rows with empty text to keep the index meaningful.
    out = out[out["text"].astype(str).str.strip() != ""]
    return out.reset_index(drop=True)


def main(argv: List[str] | None = None) -> int:
    """CLI entry point to build or update the local vector index.

    Workflow:
        1. Load app and rules configs.
        2. Read calls from snapshots, or from CSV glob if requested/needed.
        3. Ensure labels exist (Connected/Type/Outcome).
        4. Prepare compact docs (id, text, metadata).
        5. Hand off to the Chroma indexer with vectorstore configuration.

    Args:
        argv: Optional list of CLI arguments for testing.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Build or update the local vector index from labeled calls."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Index at most N rows (0 = no limit).",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="transcript",
        help="Comma-separated fields to concatenate into the indexed text "
             '(e.g., "transcript,product_name").',
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help='Optional CSV glob used when snapshots are empty (e.g., "data/input_csv/*.csv").',
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Override collection name (defaults to configs/vectorstore.yml or app.yml).",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="",
        help="Override persist dir (defaults to configs/vectorstore.yml or app.yml).",
    )
    args = parser.parse_args(argv)

    # Load configs
    app_cfg = _load_yaml(CFG_DIR / "app.yml")
    rules_cfg = _load_yaml(CFG_DIR / "rules.yml")
    vs_cfg = _load_vectorstore_cfg(app_cfg)

    # Resolve persistence/collection from CLI overrides or config defaults
    persist_dir = Path(args.persist_dir or vs_cfg.get("persist_dir", "./vectorstore/chroma"))
    collection_name = args.collection or vs_cfg.get("collection", "calls")

    # 1) Load data
    df = _load_calls_from_snapshots()
    if df.empty and args.input_csv:
        df = _load_calls_from_csv_glob(args.input_csv)

    if df.empty:
        print("No data found to index. Load snapshots or provide --input-csv.")
        return 0

    # 2) Ensure labels exist (some CSVs may not have them yet)
    if not {"Connected", "Type", "Outcome"}.issubset(df.columns):
        df = apply_labels(df, rules_cfg)

    # 3) Optional row cap for fast iteration
    if args.limit and args.limit > 0:
        df = df.head(int(args.limit))

    # 4) Prepare docs (id, text, metadata)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    docs = _prepare_docs(df, fields)
    if docs.empty:
        print("Nothing to index after preparing documents (empty text).")
        return 0

    # 5) Build/update the index
    persist_dir.mkdir(parents=True, exist_ok=True)
    n = build_or_update_index(
        docs=docs,
        collection_name=collection_name,
        persist_dir=persist_dir,
        rebuild=bool(args.rebuild),
        vs_cfg=vs_cfg,
        id_field="doc_id",
        text_field="text",
        meta_field="metadata",
    )

    print(f"[vectorstore] Indexed {n} document(s) into '{collection_name}' at '{persist_dir}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
