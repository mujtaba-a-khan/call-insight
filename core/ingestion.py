# core/ingestion.py
# Data intake & validation
# - CSV loader with schema normalization and English-only gate
# - Audio batch processor: convert → STT (faster-whisper) → row assembly
# - Returns (included_df, excluded_df) where excluded has reasons
#
# Rules implemented here (V1):
# • Exclude if transcript is empty or < min_tokens (English-only gate)
# • Do NOT exclude short calls; rows with duration < short_disconnected_s remain included
#   (they will be labeled "Disconnected" later by labeling_rules.py)
#
# Notes:
# • Labeling (Connected/Type/Outcome) happens in core/labeling_rules.py.
# • Start times for audio uploads default to "now" if not provided elsewhere.

import pandas as pd
from typing import IO

from dataclasses import dataclass, field
from .stt_whisper import SttParams
from pathlib import Path
from typing import Iterable, List, Tuple

from .audio_preprocess import to_wav_mono_16k
from .schema import COLUMNS_ORDER, coerce_dtypes, ensure_columns, normalize_calls_df
from .stt_whisper import SttParams, transcribe_local
from .utils import english_token_count, ensure_dir, now_iso


@dataclass(frozen=True)
class CsvIngestConfig:
    min_tokens: int = 20
    short_disconnected_s: int = 10  # kept for reference; not used for exclusion
    timezone: str = "Europe/Berlin"


@dataclass(frozen=True)
class AudioIngestConfig:
    min_tokens: int = 20
    short_disconnected_s: int = 10
    timezone: str = "Europe/Berlin"
    #stt_params: SttParams = SttParams()

    stt_params: SttParams = field(
        default_factory=lambda: SttParams(model_name="base")
    )

    tmp_upload_dir: Path = Path("./artifacts/tmp_uploads")   # where raw uploads may be staged
    wav_out_dir: Path = Path("./artifacts/audio_wav")        # normalized WAV output
    


def ingest_csv(
    csv_source: str | Path | IO[bytes],
    cfg: CsvIngestConfig = CsvIngestConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a CSV with (at least) columns:
      call_id, start_time (ISO), duration_seconds, transcript, [agent_id], [campaign], ...
    Returns (included_df, excluded_df) where excluded_df has columns: call_id, reason.
    """
    df = pd.read_csv(csv_source)
    df = coerce_dtypes(ensure_columns(df))

    excluded_rows = []
    keep_rows = []

    for _, row in df.iterrows():
        transcript = (row.get("transcript") or "")
        tok = english_token_count(str(transcript))

        if tok < cfg.min_tokens:
            excluded_rows.append({"call_id": row.get("call_id"), "reason": "short/non-English"})
            continue

        keep_rows.append(row)

    included_df = pd.DataFrame(keep_rows, columns=df.columns).reset_index(drop=True) if keep_rows else df.iloc[0:0].copy()
    excluded_df = pd.DataFrame(excluded_rows, columns=["call_id", "reason"]).reset_index(drop=True)

    # Normalize columns/dtypes for the happy path
    included_df = normalize_calls_df(included_df)

    #Test line to parse UTC
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    return included_df, excluded_df


def ingest_audio_files(
    audio_paths: Iterable[Path],
    cfg: AudioIngestConfig = AudioIngestConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Batch process audio files:
      - Convert to wav mono 16k
      - Transcribe offline (faster-whisper *_en)
      - Apply English-only gate (exclude < min_tokens)
    Returns (included_df, excluded_df).
    """
    ensure_dir(cfg.wav_out_dir)

    rows: List[dict] = []
    excluded_rows: List[dict] = []

    for src in audio_paths:
        src = Path(src)
        if not src.exists():
            excluded_rows.append({"call_id": src.name, "reason": "file not found"})
            continue

        try:
            # Normalize + STT
            wav_path, duration_s = to_wav_mono_16k(src, cfg.wav_out_dir, overwrite=False)
            payload = transcribe_local(src, wav_out_dir=cfg.wav_out_dir, params=cfg.stt_params, overwrite_cache=False)
            transcript = payload.get("transcript", "") or ""
            # Use duration from STT if available
            duration_s = float(payload.get("duration_s") or duration_s or 0.0)
        except Exception as e:  # STT or conversion failure
            excluded_rows.append({"call_id": src.name, "reason": f"stt-fail: {e.__class__.__name__}"})
            continue

        # English-only gate
        if english_token_count(transcript) < cfg.min_tokens:
            excluded_rows.append({"call_id": src.name, "reason": "short/non-English"})
            continue

        rows.append(
            {
                "call_id": src.stem,
                "start_time": now_iso(cfg.timezone),
                "duration_seconds": duration_s,
                "transcript": transcript,
                "agent_id": pd.NA,
                "campaign": pd.NA,
                "customer_name": pd.NA,
                "product_name": pd.NA,
                "quantity": pd.NA,
                "amount": pd.NA,
                "order_id": pd.NA,
                # Derived labels are assigned later by labeling_rules.py
            }
        )

    included_df = pd.DataFrame(rows, columns=COLUMNS_ORDER)
    included_df = normalize_calls_df(included_df)

    excluded_df = pd.DataFrame(excluded_rows, columns=["call_id", "reason"]).reset_index(drop=True)
    return included_df, excluded_df
