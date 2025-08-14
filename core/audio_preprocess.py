# core/audio_preprocess.py
# Local audio normalization for STT:
# - Convert any supported input (wav/mp3/m4a/flac) to 16 kHz, mono, 16-bit PCM WAV
# - Return the output WAV path and duration (seconds)
# - No external services; requires ffmpeg installed on the system
#
# Typical use:
#   from pathlib import Path
#   from core.audio_preprocess import to_wav_mono_16k, audio_duration_s
#   wav_path, dur = to_wav_mono_16k(Path("data/input_audio/call.m4a"), Path("artifacts/audio_wav"))
#
# Notes:
# - We use pydub (ffmpeg under the hood). Ensure ffmpeg is on PATH:
#     macOS:  brew install ffmpeg
#     Ubuntu: sudo apt-get install ffmpeg

import os
from pathlib import Path
from typing import Optional, Tuple

from pydub import AudioSegment  # type: ignore
from pydub.utils import which

# Ensure pydub can find ffmpeg (works for Homebrew on macOS)
_ffmpeg = which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
if not _ffmpeg:
    raise RuntimeError(
        "FFmpeg not found. Install via 'brew install ffmpeg' or set FFMPEG path."
    )
AudioSegment.converter = _ffmpeg

from .utils import ensure_dir


TARGET_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # bytes → 16-bit PCM


def _infer_stem(src: Path) -> str:
    """Return a safe stem for the output filename."""
    # .stem only removes the last suffix; keep it simple and predictable.
    return src.stem


def audio_duration_s(path: Path) -> float:
    """Fast-ish duration probe via pydub (uses ffmpeg)."""
    seg = AudioSegment.from_file(path)
    return round(seg.duration_seconds, 3)


def to_wav_mono_16k(src_path: Path, out_dir: Path, *, overwrite: bool = False) -> Tuple[Path, float]:
    """
    Convert `src_path` to mono, 16 kHz, 16-bit PCM WAV in `out_dir`.

    Returns
    -------
    (wav_path, duration_seconds)

    Raises
    ------
    RuntimeError if conversion fails (e.g., missing ffmpeg or bad file).
    """
    src_path = Path(src_path)
    out_dir = ensure_dir(out_dir)

    if not src_path.exists():
        raise RuntimeError(f"Audio file not found: {src_path}")

    stem = _infer_stem(src_path)
    wav_path = out_dir / f"{stem}.wav"

    if wav_path.exists() and not overwrite:
        # Trust existing normalized file; still compute duration to return.
        try:
            return wav_path, audio_duration_s(wav_path)
        except Exception:
            # Fall through to re-export if probing failed.
            pass

    try:
        seg = AudioSegment.from_file(src_path)
        # Normalize: channels, rate, sample width (16-bit)
        seg = seg.set_channels(TARGET_CHANNELS)
        seg = seg.set_frame_rate(TARGET_RATE)
        seg = seg.set_sample_width(TARGET_SAMPLE_WIDTH)

        # Export as PCM WAV
        export_params = {
            "format": "wav",
            "parameters": ["-acodec", "pcm_s16le"],  # explicit 16-bit PCM
        }
        seg.export(wav_path, **export_params)

        dur = round(seg.duration_seconds, 3)
        return wav_path, dur
    except FileNotFoundError as e:
        # Common when ffmpeg is not installed / not on PATH
        raise RuntimeError(
            "ffmpeg not found. Please install it and ensure it's on PATH "
            "(e.g., `brew install ffmpeg` or `sudo apt-get install ffmpeg`)."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to convert '{src_path}' to WAV: {e}") from e


def save_bytes_and_convert(
    data: bytes,
    original_name: str,
    tmp_dir: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> Tuple[Path, float]:
    """
    Convenience for Streamlit uploads:
    - Writes the uploaded bytes to `tmp_dir/original_name`
    - Converts it to mono 16 kHz WAV in `out_dir`
    - Returns (wav_path, duration_seconds)
    """
    tmp_dir = ensure_dir(tmp_dir)
    out_dir = ensure_dir(out_dir)

    # Preserve the original extension so ffmpeg can detect the format
    src_path = tmp_dir / original_name
    src_path.write_bytes(data)

    return to_wav_mono_16k(src_path, out_dir, overwrite=overwrite)