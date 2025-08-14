# call_app/app.py
# Streamlit single-page app for Call Insights (Prototype — Local)
# ─────────────────────────────────────────────────────────────────────────────
# Responsibilities
#   • Load configs (app.yml, rules.yml, llm.yml) and cached snapshots
#   • Let users upload audio/CSV; run offline STT + deterministic labeling
#   • Provide filters, KPIs, charts, a calls table, and a details drawer
#   • Offer a local “LLM Q&A” section and a Vectorstore (Chroma) settings panel
#
# Implementation notes
#   • No external services. STT uses faster-whisper *_en models locally.
#   • Storage is in-memory with snapshots under ./artifacts/state.
#   • Vector DB is optional; fully local via Chroma, configured in-app.

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yaml

from call_app.components import (
    render_kpis,
    render_charts,
    render_calls_table,
    render_details,
)
from call_app.components.vectorstore_settings import render_vectorstore_settings
from core.filters import _normalize_dates  # timezone-safe day bounds (UTC)
from core.ingestion import (
    AudioIngestConfig,
    CsvIngestConfig,
    ingest_audio_files,
    ingest_csv,
)
from core.labeling_rules import apply_labels
from core.schema import normalize_calls_df  # noqa: F401
from core.storage import LocalStorage
from core.utils import ensure_dir, now_iso
from .qna_ui import render_qna_ui

# Ensure project root is importable when launched via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Config directory location.
CFG_DIR = PROJECT_ROOT / "configs"


@st.cache_data(show_spinner=False)
def _load_yaml(path: Path) -> Dict:
    """Load a YAML file into a dictionary.

    Args:
        path: Filesystem path to a YAML file.

    Returns:
        Dict parsed from YAML. Empty dict on missing file or parse error.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _load_configs() -> Tuple[Dict, Dict, Dict]:
    """Load application, rules, and LLM configuration, merging vectorstore settings.

    Returns:
        Tuple of dictionaries: (app_cfg, rules_cfg, llm_cfg).
    """
    app_cfg = _load_yaml(CFG_DIR / "app.yml")
    rules_cfg = _load_yaml(CFG_DIR / "rules.yml")
    llm_cfg = _load_yaml(CFG_DIR / "llm.yml")

    # Merge configs/vectorstore.yml → app_cfg["vectorstore"] (if present).
    vs_file = _load_yaml(CFG_DIR / "vectorstore.yml")
    vs_block = vs_file.get("vectorstore") if isinstance(vs_file, dict) else None
    if isinstance(vs_block, dict) and vs_block:
        app_cfg.setdefault("vectorstore", {})
        app_cfg["vectorstore"].update(vs_block)

    return app_cfg, rules_cfg, llm_cfg


def _bootstrap_state() -> None:
    """Initialize session state with storage and cached DataFrames."""
    if "storage" not in st.session_state:
        st.session_state.storage = LocalStorage()
        calls, excluded = st.session_state.storage.load_snapshots()
        st.session_state.calls_df = calls
        st.session_state.excluded_df = excluded
    st.session_state.setdefault("selected_row", None)


def _process_csv_uploads(
    files: List[st.runtime.uploaded_file_manager.UploadedFile], app_cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process uploaded CSV transcripts via the ingestion pipeline.

    Args:
        files: Uploaded CSV files from Streamlit.
        app_cfg: Application configuration dict.

    Returns:
        Tuple of (included_df, excluded_df).
    """
    if not files:
        return pd.DataFrame(), pd.DataFrame(columns=["call_id", "reason"])

    inc_frames, exc_frames = [], []
    gate = CsvIngestConfig(min_tokens=int((app_cfg.get("connected") or {}).get("min_tokens", 20)))
    for f in files:
        bio = BytesIO(f.read())
        inc, exc = ingest_csv(bio, gate)
        inc_frames.append(inc)
        exc_frames.append(exc)

    inc_all = pd.concat(inc_frames, ignore_index=True) if inc_frames else pd.DataFrame()
    exc_all = pd.concat(exc_frames, ignore_index=True) if exc_frames else pd.DataFrame(columns=["call_id", "reason"])
    return inc_all, exc_all


def _process_audio_uploads(
    files: List[st.runtime.uploaded_file_manager.UploadedFile], app_cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transcode and process uploaded audio files with local Whisper STT."""
    if not files:
        return pd.DataFrame(), pd.DataFrame(columns=["call_id", "reason"])

    tmp_upload_dir = (
        Path((app_cfg.get("paths") or {}).get("audio_wav_dir", "./artifacts/audio_wav")).parent / "tmp_uploads"
    )
    ensure_dir(tmp_upload_dir)

    saved_paths: List[Path] = []
    for f in files:
        dst = tmp_upload_dir / f.name
        dst.write_bytes(f.read())
        saved_paths.append(dst)

    # Build STT params from app.yml
    stt_cfg = (app_cfg.get("stt") or {})
    params = AudioIngestConfig(
        min_tokens=int((app_cfg.get("connected") or {}).get("min_tokens", 20)),
        short_disconnected_s=int((app_cfg.get("connected") or {}).get("short_disconnected_s", 10)),
        timezone=str(app_cfg.get("timezone", "Europe/Berlin")),
        stt_params=None,
        wav_out_dir=Path((app_cfg.get("paths") or {}).get("audio_wav_dir", "./artifacts/audio_wav")),
    )
    from core.stt_whisper import SttParams  # local import to avoid slow import at top-level

    params = AudioIngestConfig(
        min_tokens=params.min_tokens,
        short_disconnected_s=params.short_disconnected_s,
        timezone=params.timezone,
        stt_params=SttParams(
            model_name=str(stt_cfg.get("model", "small.en")),
            compute_type=str(stt_cfg.get("compute_type", "int8")),
            num_workers=int(stt_cfg.get("num_workers", 1)),
        ),
        wav_out_dir=params.wav_out_dir,
    )

    inc, exc = ingest_audio_files(saved_paths, params)
    return inc, exc


@st.cache_data(show_spinner=False)
def _apply_labels_cached(
    df: pd.DataFrame, rules_cfg: Dict, duration_cutoff_override: Optional[int]
) -> pd.DataFrame:
    """Apply labeling rules with caching."""
    if df is None or df.empty:
        return df
    return apply_labels(df, rules_cfg, duration_cutoff_override=duration_cutoff_override)


def _prefer_existing(primary: Optional[pd.Series], fallback: pd.Series) -> pd.Series:
    """Prefer user-provided labels over rule-generated ones.

    Args:
        primary: Series with user-provided labels (may be None).
        fallback: Series produced by deterministic rules.

    Returns:
        Series where non-empty, non-'Unknown' values from `primary` are kept;
        otherwise values from `fallback`.
    """
    if primary is None:
        return fallback
    p = primary.astype(str)
    mask = p.str.strip().ne("").fillna(False) & p.str.lower().ne("unknown")
    return primary.where(mask, fallback)


def _summary_csv_bytes(df: pd.DataFrame, show_unknowns: bool, context: Dict) -> bytes:
    """Create a CSV export combining connection/type/outcome counts + context."""
    if df is None:
        return b""

    conn = df.get("Connected", pd.Series(dtype=str)).astype(str).fillna("Unknown").value_counts(dropna=False)
    conn_df = conn.rename_axis("label").reset_index(name="count")
    conn_df.insert(0, "section", "Connection")

    typ = df.get("Type", pd.Series(dtype=str)).astype(str).fillna("Unknown").value_counts(dropna=False)
    if not show_unknowns and "Unknown" in typ.index:
        typ = typ.drop(index="Unknown")
    typ_df = typ.rename_axis("label").reset_index(name="count")
    typ_df.insert(0, "section", "Types")

    outc = df.get("Outcome", pd.Series(dtype=str)).astype(str).fillna("Unknown").value_counts(dropna=False)
    if not show_unknowns and "Unknown" in outc.index:
        outc = outc.drop(index="Unknown")
    out_df = outc.rename_axis("label").reset_index(name="count")
    out_df.insert(0, "section", "Outcomes")

    all_df = pd.concat([conn_df, typ_df, out_df], ignore_index=True)
    ctx_rows = [{"section": "Context", "label": k, "count": v} for k, v in context.items()]
    spacer = pd.DataFrame([{"section": "", "label": "", "count": ""}])
    out_full = pd.concat([all_df, spacer, pd.DataFrame(ctx_rows)], ignore_index=True)
    return out_full.to_csv(index=False).encode("utf-8")


def main() -> None:
    """Streamlit entry point for the Call Insights single-page app."""
    st.set_page_config(page_title="Call Insights (Prototype — Local)", layout="wide")
    _bootstrap_state()
    app_cfg, rules_cfg, llm_cfg = _load_configs()

    colors = (app_cfg.get("ui") or {}).get("colors", {})
    artifacts_dir = ensure_dir((app_cfg.get("paths") or {}).get("artifacts_dir", "./artifacts"))
    charts_dir = ensure_dir((app_cfg.get("paths") or {}).get("charts_dir", "./artifacts/charts"))
    exports_dir = ensure_dir((app_cfg.get("paths") or {}).get("exports_dir", "./artifacts/exports"))

    # Header
    left, right = st.columns([0.75, 0.25])
    with left:
        st.title("Call Insights (Prototype — Local)")
        st.caption("English-only • No external services • All processing on your machine")
    with right:
        st.write("")
        st.write(f"Last updated: {now_iso(app_cfg.get('timezone', 'Europe/Berlin'))}")

    # Uploads
    with st.expander("Upload Audio / CSV", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            audio_files = st.file_uploader(
                "Audio files",
                type=(app_cfg.get("ingestion") or {}).get("accepted_audio_ext", ["wav", "mp3", "m4a", "flac"]),
                accept_multiple_files=True,
                help="Recommended ≤30–45 min per file. Converted to 16 kHz mono WAV locally.",
            )
        with c2:
            csv_files = st.file_uploader(
                "Transcripts CSV (optional)",
                type=["csv"],
                accept_multiple_files=True,
                help="Columns: call_id,start_time,duration_seconds,transcript,[agent_id],[campaign],…",
            )

        process = st.button("Process uploads", use_container_width=True)
        if process:
            with st.spinner("Processing uploads locally…"):
                inc_csv, exc_csv = _process_csv_uploads(csv_files or [], app_cfg)
                inc_aud, exc_aud = _process_audio_uploads(audio_files or [], app_cfg)

                included = (
                    pd.concat([inc_csv, inc_aud], ignore_index=True)
                    if not (inc_csv.empty and inc_aud.empty)
                    else pd.DataFrame()
                )
                excluded = (
                    pd.concat([exc_csv, exc_aud], ignore_index=True)
                    if not (exc_csv.empty and exc_aud.empty)
                    else pd.DataFrame(columns=["call_id", "reason"])
                )

                if not included.empty:
                    # Preserve CSV-provided labels where present; fill gaps via rules.
                    existing_type = included["Type"].copy() if "Type" in included.columns else None
                    existing_outcome = included["Outcome"].copy() if "Outcome" in included.columns else None

                    labeled = apply_labels(included, rules_cfg)

                    if existing_type is not None and "Type" in labeled.columns:
                        labeled["Type"] = _prefer_existing(existing_type, labeled["Type"])
                    if existing_outcome is not None and "Outcome" in labeled.columns:
                        labeled["Outcome"] = _prefer_existing(existing_outcome, labeled["Outcome"])

                    st.session_state.storage.append_calls(labeled, snapshot=True)
                    st.session_state.calls_df = st.session_state.storage.calls_df

                if not excluded.empty:
                    st.session_state.storage.append_excluded(excluded, snapshot=True)
                    st.session_state.excluded_df = st.session_state.storage.excluded_df

            st.success(
                f"Added {len(included)} row(s). Excluded: {len(excluded)}. "
                "Snapshots updated under ./artifacts/state."
            )

    # Sidebar: filters and settings for cut off call
    with st.sidebar:
        st.header("Filters")
        calls = st.session_state.calls_df
        LOCAL_TZ = ZoneInfo(app_cfg.get("timezone", "Europe/Berlin"))

        if not calls.empty:
            st_local = pd.to_datetime(calls["start_time"], errors="coerce", utc=True).dt.tz_convert(LOCAL_TZ)
            min_date = st_local.min().date()
            max_date = st_local.max().date()
        else:
            now_local = pd.Timestamp.now(tz=LOCAL_TZ)
            min_date = max_date = now_local.date()

        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            help="Inclusive date range over start_time (local timezone).",
        )

        # Robustly handle one/two/none date selection.
        if isinstance(date_range, (list, tuple)):
            if len(date_range) == 2:
                date_from, date_to = date_range
            elif len(date_range) == 1:
                date_from = date_to = date_range[0]
            else:
                date_from = date_to = None
        else:
            date_from = date_to = date_range

        agent = st.selectbox(
            "Agent",
            options=["(All)"]
            + sorted([a for a in calls.get("agent_id", pd.Series([], dtype=str)).dropna().unique().tolist() if a]),
            index=0,
        )
        campaign = st.selectbox(
            "Campaign",
            options=["(All)"]
            + sorted([c for c in calls.get("campaign", pd.Series([], dtype=str)).dropna().unique().tolist() if c]),
            index=0,
        )

        st.divider()
        st.header("Settings")
        cutoff = int(
            st.slider(
                "Duration cutoff (s) for Connected",
                min_value=10,
                max_value=120,
                value=int((app_cfg.get("connected") or {}).get("duration_cutoff_s", 30)),
                step=5,
            )
        )
        st.caption("Connected = duration ≥ cutoff and transcript ≥ 40 tokens")

    # Read current toggle value from session state (so we can compute viz_df first).
    show_unknowns_default = bool(app_cfg.get("show_unknowns_default", False))
    show_unknowns = bool(st.session_state.get("show_unknowns", show_unknowns_default))

    # Compute filtered + labeled views (UTC)
    df_all = st.session_state.calls_df
    if not df_all.empty:
        labeled = _apply_labels_cached(df_all, rules_cfg, duration_cutoff_override=cutoff)

        # Keep CSV-provided labels where present.
        if "Type" in df_all.columns and "Type" in labeled.columns:
            labeled["Type"] = _prefer_existing(df_all["Type"], labeled["Type"])
        if "Outcome" in df_all.columns and "Outcome" in labeled.columns:
            labeled["Outcome"] = _prefer_existing(df_all["Outcome"], labeled["Outcome"])

        start_utc, end_utc = _normalize_dates(labeled, date_from, date_to)
        st_col_utc = pd.to_datetime(labeled["start_time"], errors="coerce", utc=True)

        mask = (st_col_utc >= start_utc) & (st_col_utc < end_utc)
        if agent != "(All)":
            mask &= labeled["agent_id"] == agent
        if campaign != "(All)":
            mask &= labeled["campaign"] == campaign

        # kpi_df: used for KPIs ONLY
        kpi_df = labeled.loc[mask].reset_index(drop=True)

        # viz_df: charts + table + export; controlled by toggle
        viz_df = kpi_df.copy()
        if not show_unknowns:
            for col in ("Type", "Outcome"):
                if col in viz_df.columns:
                    viz_df = viz_df[viz_df[col].astype(str).str.strip().ne("Unknown")]
    else:
        kpi_df = df_all
        viz_df = df_all

    # KPIs 
    render_kpis(kpi_df, excluded_df=st.session_state.excluded_df, show_unknowns=True)

    # Charts 
    render_charts(viz_df, show_unknowns=show_unknowns, color_map=colors, save_dir=charts_dir)

    # Using a stable key ensures the next rerun picks the new value to recompute viz_df.
    st.toggle(
        "Show Unknowns",
        key="show_unknowns",
        value=show_unknowns_default 
        if "show_unknowns" not in st.session_state 
        else st.session_state["show_unknowns"],
        help="Include rows labeled 'Unknown' in charts and table. KPIs are unaffected.",
    )

    # Calls table + details drawer
    selected = render_calls_table(viz_df, page_size=25)

    if selected:
        render_details(selected, rules_cfg, color_map=colors, pii_mask=bool(app_cfg.get("pii_mask", True)))

    # Summary CSV export 
    context = {
        "date_from": str(date_from),
        "date_to": str(date_to),
        "agent": agent,
        "campaign": campaign,
        "cutoff_s": cutoff,
        "show_unknowns": bool(st.session_state.get("show_unknowns", show_unknowns_default)),
    }
    summary_bytes = _summary_csv_bytes(viz_df, bool(context["show_unknowns"]), context)
    st.download_button(
        "Download Summary (CSV)",
        data=summary_bytes,
        file_name="summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    # Q&A Surface — question-driven
    render_qna_ui(
        st.session_state.calls_df,        # df_all
        app_cfg=app_cfg,
        llm_cfg=llm_cfg,
        )

    st.divider()
    # Vectorstore (Chroma) Settings Panel
    with st.expander("Vectorstore (Chroma) Settings", expanded=False):
        render_vectorstore_settings(st.session_state.calls_df, app_cfg)

    # Footer
    st.divider()
    st.caption(
        f"English-only. Connected if duration ≥ {cutoff}s and transcript ≥ 40 tokens | Developed by Mujtaba Khan"
    )

if __name__ == "__main__":
    main()
