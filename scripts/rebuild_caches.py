# scripts/rebuild_caches.py
# End-to-end local processing:
# 1) Collect audio files (and/or CSV transcripts)
# 2) Convert + STT (cached) + English-only gate
# 3) Apply deterministic labels
# 4) Save snapshots for the app (artifacts/state)
#
# Usage examples:
#   callinsights-rebuild-caches --audio-dir data/input_audio --snapshot
#   callinsights-rebuild-caches --csv-glob "data/input_csv/*.csv" --cutoff 45
#   callinsights-rebuild-caches --clear-cache stt --audio-dir data/input_audio
#
# Notes:
# - All work is offline; STT uses faster-whisper *_en models.
# - Caches live under ./cache/{stt,labels}. Safe to delete anytime.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from core.caching import clear_cache
from core.ingestion import (
    AudioIngestConfig,
    CsvIngestConfig,
    ingest_audio_files,
    ingest_csv,
)
from core.labeling_rules import apply_labels
from core.schema import ensure_columns, normalize_calls_df
from core.stt_whisper import SttParams
from core.storage import LocalStorage
from core.utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = PROJECT_ROOT / "configs"


# ---------- helpers ----------

def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _gather_audio(audio_dir: Path, exts: List[str]) -> List[Path]:
    paths: List[Path] = []
    for ext in exts:
        paths.extend(sorted(audio_dir.glob(f"*.{ext}")))
        # also search nested dirs: dir/**/*.ext
        paths.extend(sorted(audio_dir.glob(f"**/*.{ext}")))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        if p.exists() and p.is_file() and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _load_many_csv(glob_pattern: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load many CSVs and combine included/excluded results."""
    files = sorted(Path().glob(glob_pattern))
    included_frames: List[pd.DataFrame] = []
    excluded_frames: List[pd.DataFrame] = []
    for f in files:
        try:
            inc, exc = ingest_csv(f, CsvIngestConfig())
            included_frames.append(inc)
            excluded_frames.append(exc)
        except Exception as e:
            excluded_frames.append(pd.DataFrame([{"call_id": f.name, "reason": f"csv-fail: {e.__class__.__name__}"}]))
    inc_all = pd.concat(included_frames, ignore_index=True) if included_frames else pd.DataFrame()
    exc_all = pd.concat(excluded_frames, ignore_index=True) if excluded_frames else pd.DataFrame(columns=["call_id", "reason"])
    return inc_all, exc_all


# ---------- CLI ----------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild local caches (audio → STT → labels) and save snapshots.")
    parser.add_argument("--audio-dir", type=str, default="data/input_audio", help="Directory containing audio files.")
    parser.add_argument("--csv-glob", type=str, default="", help='Optional CSV glob, e.g. "data/input_csv/*.csv".')
    parser.add_argument("--snapshot", action="store_true", help="Persist results to artifacts/state.")
    parser.add_argument("--cutoff", type=int, default=0, help="Override duration cutoff (seconds) for Connected label.")
    parser.add_argument(
        "--clear-cache",
        type=str,
        choices=["stt", "labels", "all"],
        default="",
        help="Optional: clear caches before processing.",
    )
    args = parser.parse_args(argv)

    app_cfg = _load_yaml(CFG_DIR / "app.yml")
    rules_cfg = _load_yaml(CFG_DIR / "rules.yml")

    # Optional: clear caches up front
    if args.clear_cache:
        clear_cache(args.clear_cache)
        print(f"Cleared cache: {args.clear_cache}")

    # Ingestion config (derive from app.yml where possible)
    stt_cfg = (app_cfg.get("stt") or {})
    stt_params = SttParams(
        model_name=str(stt_cfg.get("model", "small.en")),
        compute_type=str(stt_cfg.get("compute_type", "int8")),
        num_workers=int(stt_cfg.get("num_workers", 1)),
    )
    audio_cfg = AudioIngestConfig(
        min_tokens=int((app_cfg.get("ingestion") or {}).get("min_tokens", 20)),
        short_disconnected_s=int((app_cfg.get("connected") or {}).get("short_disconnected_s", 10)),
        timezone=str(app_cfg.get("timezone", "Europe/Berlin")),
        stt_params=stt_params,
        wav_out_dir=Path((app_cfg.get("paths") or {}).get("audio_wav_dir", "./artifacts/audio_wav")),
    )

    # Collect audio files
    audio_dir = Path(args.audio_dir)
    accepted_exts = list((app_cfg.get("ingestion") or {}).get("accepted_audio_ext", ["wav", "mp3", "m4a", "flac"]))
    audio_paths = _gather_audio(audio_dir, accepted_exts)

    # Process audio
    inc_audio = pd.DataFrame()
    exc_audio = pd.DataFrame(columns=["call_id", "reason"])
    if audio_paths:
        print(f"Found {len(audio_paths)} audio file(s) under {audio_dir}. Processing…")
        inc_audio, exc_audio = ingest_audio_files(audio_paths, audio_cfg)
    else:
        print(f"No audio files found in {audio_dir} (extensions: {accepted_exts}).")

    # Process CSVs (optional)
    inc_csv = pd.DataFrame()
    exc_csv = pd.DataFrame(columns=["call_id", "reason"])
    if args.csv_glob:
        print(f"Loading CSVs from '{args.csv_glob}' …")
        inc_csv, exc_csv = _load_many_csv(args.csv_glob)

    # Combine
    included = pd.concat([inc_audio, inc_csv], ignore_index=True) if not inc_audio.empty or not inc_csv.empty else pd.DataFrame()
    excluded = pd.concat([exc_audio, exc_csv], ignore_index=True) if not exc_audio.empty or not exc_csv.empty else pd.DataFrame(columns=["call_id", "reason"])

    if included.empty:
        print("Nothing to process. (No included rows from audio/CSV.)")
        # Still write empty snapshots if requested
        storage = LocalStorage()
        storage.set_calls(included, snapshot=args.snapshot)
        storage.set_excluded(excluded, snapshot=args.snapshot)
        return 0

    # Label deterministically
    duration_override = int(args.cutoff) if args.cutoff and args.cutoff > 0 else None
    labeled = apply_labels(included, rules_cfg, duration_cutoff_override=duration_override)

    # Save snapshots (optional)
    storage = LocalStorage()
    storage.set_calls(labeled, snapshot=args.snapshot)
    storage.set_excluded(excluded, snapshot=args.snapshot)

    # Report
    connected = int((labeled.get("Connected", pd.Series(dtype=str)).astype(str) == "Connected").sum())
    disconnected = int((labeled.get("Connected", pd.Series(dtype=str)).astype(str) == "Disconnected").sum())
    print(
        f"Processed {len(labeled)} included row(s) "
        f"({connected} Connected, {disconnected} Disconnected). "
        f"Excluded: {len(excluded)}."
    )
    if args.snapshot:
        print("Snapshots saved to artifacts/state/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
