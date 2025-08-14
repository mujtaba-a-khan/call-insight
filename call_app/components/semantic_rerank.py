# call_app/components/semantic_rerank.py
# Semantic re-ranking helper for the Q&A surface.
# Purpose:
#   - Never shrink the *scope* of structured results.
#   - Optionally reorder rows semantically and cap the *display* to Top-K.
#
# Typical wiring in qna_ui.py:
#   from call_app.components.semantic_rerank import present_with_semantics
#   df_view, scope_count = present_with_semantics(
#       df_scope=df_scope,                 # ALL rows that match the structured spec
#       question=user_question,            # free-text from the input box
#       semantic_enabled=ui_semantic_on,   # toggle value
#       top_k=ui_top_k,                    # slider value (int)
#       reorder_fn=local_chroma_rerank_fn  # optional; returns df_scope in ranked order
#   )
#   st.caption(f"{scope_count} calls in scope.")
#   st.dataframe(df_view)
#
# Notes:
#   - This keeps analytics accurate: scope_count == len(df_scope).
#   - Top-K limits only what you *show* after re-ranking, not how many rows match.

from __future__ import annotations

from typing import Callable, Tuple
import pandas as pd


def present_with_semantics(
    df_scope: pd.DataFrame,
    question: str,
    semantic_enabled: bool,
    top_k: int,
    reorder_fn: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None,
) -> Tuple[pd.DataFrame, int]:
    """Return a display DataFrame and the full in-scope count.

    Args:
        df_scope: Fully filtered DataFrame from structured spec
                  (this already includes date/campaign/product/type/outcome/agent/amount).
        question: User's free-text question. Used only if semantic re-ranking is enabled.
        semantic_enabled: Whether to apply semantic ranking.
        top_k: Maximum rows to *display* after re-ranking. Ignored if semantic is off.
        reorder_fn: Optional function that returns a *reordered* view of df_scope based
                    on the question. It must not add/remove rows.

    Returns:
        (df_view, scope_count):
            df_view: DataFrame for display. If semantic is enabled, at most top_k rows.
            scope_count: The total number of rows in scope (len(df_scope)).
    """
    scope_count = int(len(df_scope))
    if scope_count == 0 or not semantic_enabled:
        # No re-ranking; show everything and keep the true scope count.
        return df_scope, scope_count

    ranked = df_scope
    if callable(reorder_fn):
        try:
            ranked = reorder_fn(df_scope, question)
            # Safety: ensure we didn't lose/gain rows (defensive programming).
            if len(ranked) != scope_count:
                ranked = df_scope
        except Exception:
            # Fail open: if the reranker errors, fall back to original ordering.
            ranked = df_scope

    # Cap only what we *display*; never the in-scope count.
    k = max(1, int(top_k)) if top_k is not None else scope_count
    return ranked.iloc[: min(k, scope_count)], scope_count
