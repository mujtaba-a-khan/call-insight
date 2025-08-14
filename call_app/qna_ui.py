# call_app/qna_ui.py
# Local Q&A surface over the already-labeled Calls DataFrame.
# ─────────────────────────────────────────────────────────────────────────────
# Design
#   1) User asks a question in natural language.
#   2) Deterministic interpreter → bounded filter JSON (no data leaves the app).
#   3) Apply the filter locally on pandas → this is the *full scope*.
#   4) Optional: semantic re-ranking (local Chroma) reorders the scope.
#      - Top-K caps what we DISPLAY after re-ranking; it never shrinks scope.
#   5) Optional: local LLM (Ollama) summarizes the displayed slice.
#
# Answer contract
#   • Scope line (true total in-scope rows)
#   • One-sentence summary: “Showing top X of Y calls …”
#   • Result table (≤cap rows; Download CSV for view + all in scope)
#   • Scope note (how categories were interpreted) + timezone note
#   • Data availability note (if customer/product columns are missing)
#
# PII
#   • Default masks customer_name in both the summary and the table (toggle to unmask).
#   • We never log the raw question. Audit log stores only: {ts, filter_spec, row_count}.


import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from core.filters import apply_qna_filter_spec
from core.semantic_search import load_vectorstore_cfg, semantic_rank_intersect
from core.utils import ensure_dir, now_iso
from llm.interpreter import Catalog, build_filter_spec
from llm.runner import LlmConfig, LocalLlm
from llm.summarizer import summarize_results
from call_app.components.semantic_rerank import present_with_semantics


# Columns shown in the compact preview table (subset; only displayed if present).
_PREVIEW_COLS = [
    "call_id",
    "start_time",
    "agent_id",
    "campaign",
    "Type",
    "Outcome",
    "customer_name",
    "product_name",
    "quantity",
    "amount",
    "order_id",
]


# ────────────────────────────── Utilities ─────────────────────────────────────


def _mask_name(name: str) -> str:
    """Mask a full name deterministically.

    Converts "Jane Smith" → "J*** S****".

    Args:
        name: Raw name string.

    Returns:
        Masked name string. Returns the input as-is for non-strings/empty values.
    """
    if not isinstance(name, str) or not name.strip():
        return name
    parts = name.split()
    masked: List[str] = []
    for p in parts:
        if len(p) <= 1:
            masked.append(p + "*")
        else:
            masked.append(p[0] + "*" * (len(p) - 1))
    return " ".join(masked)


def _catalog_from_df(df: pd.DataFrame) -> Catalog:
    """Build a small entity catalog for substring matching.

    Args:
        df: Calls DataFrame.

    Returns:
        Catalog with unique lists for agents (IDs or names), campaigns, and products.
    """
    def _col_vals(name: str) -> List[str]:
        if name not in df.columns:
            return []
        s = df[name].dropna().astype(str).str.strip()
        return [v for v in s.tolist() if v]

    # Agents may be provided as IDs (A003) or names (e.g., "Sara").
    agent_pool = set(_col_vals("agent_id")) | set(_col_vals("agent_name")) | set(_col_vals("agent"))
    agents = sorted([a for a in agent_pool if a])

    camps = sorted([c for c in _col_vals("campaign") if c])
    prods = sorted([p for p in _col_vals("product_name") if p])

    return Catalog(agents=agents, campaigns=camps, products=prods)


def _llm_from_cfg(llm_cfg: Dict[str, Any]) -> Optional[LocalLlm]:
    """Create a LocalLlm wrapper if runner settings request Ollama.

    Tolerates absence of Ollama; returns None if runner is not configured.

    Args:
        llm_cfg: LLM configuration dictionary (from configs/llm.yml).

    Returns:
        LocalLlm instance if available/desired, otherwise None.
    """
    model = (llm_cfg or {}).get("model") or {}
    if str(model.get("runner", "none")).lower() != "ollama":
        return None
    cfg = LlmConfig(
        runner="ollama",
        name=str(model.get("name", "llama3:instruct")),
        temperature=float((model.get("params") or {}).get("temperature", 0.2)),
        max_tokens=int((model.get("params") or {}).get("num_predict", 256)),
        timeout_s=int(((llm_cfg.get("model") or {}).get("timeouts") or {}).get("request_s", 20)),
    )
    # Optional custom endpoint support (if you add model.endpoint in YAML)
    endpoint = str((model.get("endpoint") or "")).strip()
    if endpoint:
        cfg.endpoint = endpoint
    return LocalLlm(cfg)


def _qna_log(path: Path, spec: Dict[str, Any], row_count: int) -> None:
    """Append a single JSON line audit record.

    Note:
        We do **not** store the raw question to avoid sensitive data tracking.

    Args:
        path: Destination file path (*.jsonl).
        spec: Deterministic filter spec used for the query.
        row_count: Number of rows matched before capping.
    """
    ensure_dir(path.parent)
    rec = {"ts": now_iso(), "filter": spec, "row_count": int(row_count)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _prep_result_table(df: pd.DataFrame, show_pii: bool, cap: int) -> pd.DataFrame:
    """Produce a compact, ordered preview table with optional PII masking.

    Args:
        df: Source DataFrame with the rows you intend to display.
        show_pii: If False, mask customer names.
        cap: Row cap for the preview table.

    Returns:
        A DataFrame containing at most ``cap`` rows and the preview columns.
    """
    cols = [c for c in _PREVIEW_COLS if c in df.columns]
    out = df[cols].head(int(cap)).copy()
    if "customer_name" in out.columns and not show_pii:
        out["customer_name"] = out["customer_name"].astype(str).map(_mask_name)
    return out


def _data_availability_note(df: pd.DataFrame) -> str:
    """Return a short note if key columns are missing."""
    missing: List[str] = []
    if "customer_name" not in df.columns:
        missing.append("customer")
    if "product_name" not in df.columns:
        missing.append("product")
    if not missing:
        return ""
    return f"Note: {', '.join(missing)} columns are not available in the dataset."


def _reorder_with_chroma(df_scope: pd.DataFrame,
                         question: str,
                         vs_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return df_scope reordered by local semantic relevance.

    Important:
        - We request up to **len(df_scope)** candidates so we don't lose rows.
        - If the vectorstore is disabled/misconfigured, we simply return df_scope.

    Args:
        df_scope: All rows that match the structured spec.
        question: User's query text (free-form).
        vs_cfg: Vectorstore configuration (from configs/app.yml).

    Returns:
        DataFrame with the same rows/columns as df_scope but in ranked order.
    """
    try:
        if not (vs_cfg or {}).get("enabled"):
            return df_scope
        search_k = int(len(df_scope)) or 1  # do not shrink scope here
        ranked = semantic_rank_intersect(
            df_scope,
            question=question,
            vs_cfg=vs_cfg,
            top_k=search_k,
            id_col="call_id",
        )
        if ranked is None or ranked.empty or (len(ranked) != len(df_scope)):
            return df_scope
        return ranked
    except Exception:
        return df_scope


# ────────────────────────────── Public UI -------------------------------------


def render_qna_ui(df_all: pd.DataFrame, *, app_cfg: Dict[str, Any], llm_cfg: Dict[str, Any]) -> None:
    """Render the Q&A panel. All computation is local and bounded to the current dataset.

    Workflow:
        1) Interpret natural language → filter spec (deterministic).
        2) Apply filter locally on pandas → full scope.
        3) Optional: semantic re-ranking (local Chroma) reorders the scope.
        4) Optional: local LLM summary (or deterministic summary).

    Args:
        df_all: Labeled calls DataFrame for the active session.
        app_cfg: Application configuration (from configs/app.yml).
        llm_cfg: LLM/Q&A configuration (from configs/llm.yml).
    """
    st.subheader("Q&A")
    st.caption("Ask about your uploaded & labeled data. Runs offline.")

    # Controls: question + PII toggle
    left, right = st.columns([0.75, 0.25])
    with left:
        question = st.text_input(
            "Ask about your data…",
            placeholder="e.g., sales-related calls on 2025-08-13; who was the customer and what they purchased?",
        )
    with right:
        show_pii = st.toggle(
            "Show PII (unmask names)",
            value=not bool(app_cfg.get("pii_mask", True)),
            help="Names are masked by default. Toggle to unmask.",
        )

    # Runner status badges
    llm = _llm_from_cfg(llm_cfg)
    badge = "Local LLM: **available**" if (llm and llm.available()) else "Local LLM: _disabled_"
    st.caption(badge)

    # Semantic search controls (local Chroma; re-ranks within the filtered scope)
    vs_cfg = load_vectorstore_cfg(app_cfg)
    with st.expander(
        "Semantic search (local Chroma)",
        expanded=bool((vs_cfg or {}).get("enabled", True)),
    ):
        use_semantic = st.toggle(
            "Enable semantic re-ranking",
            value=bool((vs_cfg or {}).get("enabled", True)),
            help="Re-order the filtered rows using local embeddings WITHOUT shrinking scope.",
        )
        top_k = st.number_input(
            "Show top K after re-ranking",
            min_value=1,
            max_value=2000,
            value=int((llm_cfg.get("semantic_top_k") or 50)),
            step=1,
            help="Caps only what is DISPLAYED after re-ranking. The true scope is never reduced.",
        )
        show_sem_debug = st.checkbox(
            "Show semantic debug (IDs only)",
            value=False,
            help="Print the ranked call_id order (top of table) to verify behavior.",
        )

    # Result row cap for the preview table
    cap = int(llm_cfg.get("result_rows_cap", 50) or 50)
    ask = st.button("Run", type="primary", use_container_width=True)
    if not ask:
        return

    if df_all is None or df_all.empty:
        st.info("No data loaded yet. Upload audio or CSV first.")
        return

    # 1) Interpret → filter spec (deterministic; no data leaves the app)
    catalog = _catalog_from_df(df_all)
    spec, notes = build_filter_spec(question, llm_cfg, catalog=catalog)

    # 2) Apply filter locally → *full scope* (deterministic)
    df_scope = apply_qna_filter_spec(df_all, spec)
    scope_count = int(len(df_scope))
    if scope_count == 0:
        st.info("No calls matched your query.")
        log_dir = Path((app_cfg.get("paths") or {}).get("qna_log_dir", "./artifacts/logs/qna"))
        _qna_log(Path(log_dir) / "qna.log.jsonl", spec, scope_count)
        return

    # 3) Optional semantic re-ranking — reorder the scope; do not shrink it.
    if use_semantic and question.strip():
        ranked_scope = _reorder_with_chroma(df_scope, question, vs_cfg)
    else:
        ranked_scope = df_scope

    # 3b) Decide what we DISPLAY: Top-K after re-ranking (if enabled), else entire scope.
    display_df, _ = present_with_semantics(
        df_scope=ranked_scope,
        question=question,
        semantic_enabled=use_semantic and bool(question.strip()),
        top_k=int(top_k),
        reorder_fn=None,  # Reordering already applied above.
    )

    # 3c) Apply the compact table cap on top of the display slice.
    display_df = display_df.head(cap).reset_index(drop=True)
    display_count = int(len(display_df))

    # 4) Summarize the displayed slice, but reference the TRUE scope count.
    tz = llm_cfg.get("timezone", "Europe/Berlin")
    summary = summarize_results(
        display_df,
        spec,
        tz=tz,
        llm=llm,
        pii_mask=not show_pii,
        total_scope=scope_count,
        display_count=display_count,
    )

    # ───────────── UI: scope line + summary + table + downloads + notes ─────────────

    # True scope line (never affected by Top-K or cap)
    st.caption(f"**{scope_count} calls in scope.**")

    # One-sentence summary of the displayed slice
    st.write(summary)

    # Visible table (masked if needed)
    table_view = _prep_result_table(display_df, show_pii=show_pii, cap=cap)
    st.dataframe(table_view, use_container_width=True, hide_index=True)

    # Optional semantic caption + debug
    if use_semantic and question.strip():
        st.caption(
            f"Semantic re-ranking active. Showing top **{min(int(top_k), scope_count, cap)}** rows after re-ordering."
        )
        if show_sem_debug and "call_id" in display_df.columns:
            st.code("Re-ranked call_id (head): " + ", ".join(display_df["call_id"].astype(str).tolist()))

    # Downloads: (a) the view; (b) the full scope (unmasked)
    csv_view = table_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download displayed rows (CSV)",
        data=csv_view,
        file_name="qna_result_view.csv",
        mime="text/csv",
        use_container_width=True,
    )

    cols_for_all = [c for c in _PREVIEW_COLS if c in df_scope.columns]
    if not cols_for_all:
        cols_for_all = list(df_scope.columns)
    csv_all = df_scope[cols_for_all].copy().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download all in scope (CSV)",
        data=csv_all,
        file_name="qna_result_all_in_scope.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Scope + availability notes
    scope_bits: List[str] = []
    if notes:
        scope_bits.append(notes)
    scope_bits.append(f"Timezone: {tz}")
    avail = _data_availability_note(df_all)
    if avail:
        scope_bits.append(avail)
    st.caption(" • ".join(scope_bits))

    # 5) Audit log (no question text)
    log_dir = Path((app_cfg.get("paths") or {}).get("qna_log_dir", "./artifacts/logs/qna"))
    _qna_log(Path(log_dir) / "qna.log.jsonl", spec, scope_count)