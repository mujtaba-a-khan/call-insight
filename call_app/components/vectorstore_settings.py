# call_app/components/vectorstore_settings.py
# Streamlit Settings panel for the local vector store (Chroma).
# Lets you pick the embedder, tweak storage/collection, choose which fields to index,
# and rebuild the index using the in-memory calls DataFrame.

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

from vectorstore.chroma.indexer import build_or_update_index


DEFAULT_FIELDS = ["title", "summary", "transcript", "product_name"]


def _load_vs_cfg(app_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load vectorstore config, preferring ``configs/vectorstore.yml``.

    Falls back to the ``vectorstore`` block in ``configs/app.yml`` when a dedicated
    file is not present.

    Args:
        app_cfg: Parsed application config (from configs/app.yml).

    Returns:
        Vectorstore configuration dictionary (possibly empty).
    """
    vs_yml = Path("configs/vectorstore.yml")
    if vs_yml.exists():
        try:
            raw = yaml.safe_load(vs_yml.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                return raw.get("vectorstore", {}) or {}
        except Exception:
            pass
    return (app_cfg.get("vectorstore") or {}) if isinstance(app_cfg, dict) else {}


def _save_vs_cfg(cfg: Dict[str, Any]) -> None:
    """Persist vectorstore config to ``configs/vectorstore.yml``.

    Args:
        cfg: Vectorstore configuration to write under the ``vectorstore`` key.
    """
    payload = {"vectorstore": cfg}
    path = Path("configs/vectorstore.yml")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def _prepare_docs_from_df(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Create (doc_id, text, metadata) rows from the calls DataFrame.

    Args:
        df: Source calls DataFrame.
        fields: Column names to concatenate into the document ``text`` payload.

    Returns:
        DataFrame with columns: ``doc_id`` (str), ``text`` (str), ``metadata`` (dict).
    """
    df = df.copy()
    # Ensure all requested text fields exist
    for f in fields:
        if f not in df.columns:
            df[f] = ""

    def _row_text(row: pd.Series) -> str:
        parts: List[str] = []
        for f in fields:
            v = row.get(f)
            if pd.notna(v):
                s = str(v).strip()
                if s:
                    parts.append(s)
        return "\n".join(parts)

    def _row_meta(row: pd.Series) -> Dict[str, Any]:
        return {
            "call_id": str(row.get("call_id", "")),
            "start_time": str(row.get("start_time", "")),
            "agent_id": str(row.get("agent_id", "")),
            "agent": str(row.get("agent", "")),
            "campaign": str(row.get("campaign", "")),
            "type": str(row.get("Type", "")),
            "outcome": str(row.get("Outcome", "")),
            "product": str(row.get("product_name", row.get("product", ""))),
        }

    docs = pd.DataFrame(
        {
            "doc_id": df["call_id"].astype(str),
            "text": df.apply(_row_text, axis=1),
            "metadata": df.apply(_row_meta, axis=1),
        }
    )
    # Keep only rows with non-empty text
    docs = docs[docs["text"].astype(str).str.strip() != ""]
    return docs.reset_index(drop=True)


def render_vectorstore_settings(df_calls: pd.DataFrame, app_cfg: Dict[str, Any]) -> None:
    """Render a Vectorstore (Chroma) settings panel with rebuild controls.

    The panel supports:
      - Selecting embedder: ``hash`` | ``ollama`` | ``sbert``
      - Tweaking ``persist_dir``, ``collection``, ``distance``
      - Choosing which columns to index as text
      - Rebuilding the index in-place using the current in-memory dataset

    Args:
        df_calls: Labeled calls DataFrame currently loaded in the app.
        app_cfg: Application configuration (already loaded from configs/app.yml).
    """
    st.markdown("### Vectorstore (Chroma) Settings")

    vs_cfg = _load_vs_cfg(app_cfg)
    # Current values with sensible defaults
    enabled = bool(vs_cfg.get("enabled", True))
    persist_dir = str(vs_cfg.get("persist_dir", "./vectorstore/chroma"))
    collection = str(vs_cfg.get("collection", "calls"))
    distance = str(vs_cfg.get("distance", "cosine"))
    embedder = str(vs_cfg.get("embedder", "ollama"))
    embedder_model = str(vs_cfg.get("embedder_model", "nomic-embed-text"))
    sbert_model = str(vs_cfg.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2"))

    # --- Controls ------------------------------------------------------------
    enabled = st.toggle("Enable vectorstore", value=enabled, help="Turn semantic index on/off globally.")

    cols1 = st.columns(3)
    with cols1[0]:
        embedder = st.selectbox(
            "Embedder",
            options=["hash", "ollama", "sbert"],
            index=["hash", "ollama", "sbert"].index(embedder if embedder in ["hash", "ollama", "sbert"] else "ollama"),
        )
    with cols1[1]:
        distance = st.selectbox(
            "Distance",
            options=["cosine", "l2", "ip"],
            index=["cosine", "l2", "ip"].index(distance if distance in ["cosine", "l2", "ip"] else "cosine"),
        )
    with cols1[2]:
        collection = st.text_input("Collection name", value=collection)

    cols2 = st.columns(2)
    with cols2[0]:
        persist_dir = st.text_input("Persist dir", value=persist_dir, help="On-disk location for Chroma DB.")
    with cols2[1]:
        if embedder == "ollama":
            embedder_model = st.text_input(
                "Ollama embedder model", value=embedder_model, help='Example: "nomic-embed-text"'
            )
        elif embedder == "sbert":
            sbert_model = st.text_input(
                "Sentence-Transformers model",
                value=sbert_model,
                help='Example: "sentence-transformers/all-MiniLM-L6-v2"',
            )
        else:
            st.text_input("Hash embedder (no params)", value="dim=384, seed=13", disabled=True)

    # Field selection for indexing
    present_defaults = [f for f in DEFAULT_FIELDS if f in df_calls.columns]
    fields = st.multiselect(
        "Fields to index (concatenated into document text)",
        options=sorted(df_calls.columns.tolist()),
        default=present_defaults or ["transcript"],
    )

    st.divider()

    cols3 = st.columns(3)
    with cols3[0]:
        save_clicked = st.button("Save settings", type="secondary", use_container_width=True)
    with cols3[1]:
        rebuild_clicked = st.button("Rebuild index now", type="primary", use_container_width=True)
    with cols3[2]:
        limit = st.number_input(
            "Rebuild limit (rows)", min_value=0, max_value=1_000_000, value=0, help="0 = no limit"
        )

    # --- Actions -------------------------------------------------------------
    if save_clicked:
        new_cfg = {
            "enabled": enabled,
            "provider": "chroma",
            "persist_dir": persist_dir,
            "collection": collection,
            "distance": distance,
            "embedder": embedder,
            "embedder_model": embedder_model,
            "sbert_model": sbert_model,
        }
        _save_vs_cfg(new_cfg)
        st.success("Vectorstore settings saved to configs/vectorstore.yml")

    if rebuild_clicked:
        if df_calls is None or df_calls.empty:
            st.warning("No data loaded. Import calls first, then rebuild the index.")
            return

        df_to_index = df_calls.copy()
        if limit and int(limit) > 0:
            df_to_index = df_to_index.head(int(limit))

        docs = _prepare_docs_from_df(df_to_index, fields=fields)
        if docs.empty:
            st.info("Nothing to index (no non-empty text after field concatenation).")
            return

        vs_cfg_now = {
            "persist_dir": persist_dir,
            "collection": collection,
            "distance": distance,
            "embedder": embedder,
            "embedder_model": embedder_model,
            "sbert_model": sbert_model,
        }
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        n = build_or_update_index(
            docs=docs,
            collection_name=collection,
            persist_dir=Path(persist_dir),
            rebuild=True,  # Explicit rebuild per the button
            vs_cfg=vs_cfg_now,
            id_field="doc_id",
            text_field="text",
            meta_field="metadata",
        )
        st.success(f"Rebuilt index: {n} document(s) → collection '{collection}' @ '{persist_dir}'")
