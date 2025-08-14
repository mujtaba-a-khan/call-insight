# core/stt_whisper.py
# Offline English-only STT using faster-whisper (CTranslate2).
# - Converts input audio to 16 kHz mono WAV (via core.audio_preprocess)
# - Transcribes with an *_en model (small.en/base.en) — no language detection
# - Caches results on disk using core.caching (keyed by file identity + params)
#
# Typical use:
#   from pathlib import Path
#   from core.stt_whisper import transcribe_local
#   payload = transcribe_local(Path("data/input_audio/foo.mp3"),
#   wav_out_dir = Path("artifacts/audio_wav"),
#   model_name = "small.en", compute_type="int8")
#   print(payload["transcript"])
#
# Cache payload shape:
# {
#   "transcript": "full text ...",
#   "segments": [{"start": 0.00, "end": 2.34, "text": "..."}, ...],
#   "duration_s": 123.456,
#   "model": "small.en",
#   "compute": "int8",
#   "source": {"name": "foo.mp3", "mtime_ns": 1690000000, "size": 1234567}
# }

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .audio_preprocess import to_wav_mono_16k
from .caching import (
    STT_DIR,
    read_stt,
    stt_cache_key,
    write_stt,
)
from .utils import ensure_dir


# Lazily import faster_whisper so the rest of the app works without it.
def _lazy_import_whisper():
    try:
        from faster_whisper import WhisperModel  # type: ignore
        return WhisperModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is not installed or failed to import. "
            "Install it via `pip install faster-whisper`."
        ) from e


@dataclass
class SttParams:
    model_name: str = "small.en"   # English-only model
    compute_type: str = "int8"     # int8 | int8_float16 | float16 | float32
    num_workers: int = 1
    beam_size: int = 1
    vad_filter: bool = False       # keep False for V1
    temperature: float = 0.0


def _file_identity(path: Path) -> Tuple[str, int, int]:
    """Return (name, mtime_ns, size) for caching keys."""
    stat = path.stat()
    return (path.name, getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)), stat.st_size)


def _build_cache_key(src_audio: Path, params: SttParams) -> str:
    name, mtime_ns, size = _file_identity(src_audio)
    return stt_cache_key(
        file_name=name,
        file_mtime_ns=mtime_ns,
        file_size=size,
        model=params.model_name,
        compute=params.compute_type,
    )


def _load_model(params: SttParams):
    WhisperModel = _lazy_import_whisper()
    # device selection is handled internally by faster-whisper; keeping it simple for portability.
    return WhisperModel(
        params.model_name,
        compute_type=params.compute_type,
        num_workers=params.num_workers,
    )


def _run_transcribe(
    wav_path: Path,
    params: SttParams,
) -> Tuple[str, List[Dict[str, float | str]], float]:
    """
    Run faster-whisper on a normalized WAV file.
    Returns (full_text, segments, duration_s).
    """
    model = _load_model(params)

    # english-only; disable language detection
    segments, info = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=params.beam_size,
        vad_filter=params.vad_filter,
        temperature=params.temperature,
    )

    seg_rows: List[Dict[str, float | str]] = []
    full_text_parts: List[str] = []

    for seg in segments:
        text = seg.text.strip()
        if text:
            full_text_parts.append(text)
        seg_rows.append({"start": float(seg.start or 0.0), "end": float(seg.end or 0.0), "text": text})

    full_text = " ".join(full_text_parts).strip()
    duration_s = float(getattr(info, "duration", 0.0) or 0.0)

    return full_text, seg_rows, duration_s


def transcribe_local(
    src_audio_path: Path,
    *,
    wav_out_dir: Path,
    params: Optional[SttParams] = None,
    overwrite_cache: bool = False,
) -> Dict:
    """
    Convert + transcribe an audio file locally with caching.

    Parameters
    ----------
    src_audio_path : Path
        Input file (.wav/.mp3/.m4a/.flac)
    wav_out_dir : Path
        Where normalized WAVs are stored (e.g., artifacts/audio_wav)
    params : SttParams
        STT parameters (model/compute/etc.)
    overwrite_cache : bool
        If true, ignore cache and re-run STT.

    Returns
    -------
    dict payload (see header docstring for shape)
    """
    src_audio_path = Path(src_audio_path)
    params = params or SttParams()

    # 1) Cache lookup
    key = _build_cache_key(src_audio_path, params)
    if not overwrite_cache:
        cached = read_stt(key)
        if cached:
            return cached

    # 2) Normalize audio
    wav_path, duration_s = to_wav_mono_16k(src_audio_path, wav_out_dir, overwrite=False)

    # 3) Transcribe
    text, segments, duration_from_model = _run_transcribe(wav_path, params)
    if duration_from_model > 0:
        duration_s = duration_from_model

    payload = {
        "transcript": text,
        "segments": segments,
        "duration_s": duration_s,
        "model": params.model_name,
        "compute": params.compute_type,
        "source": {
            "name": src_audio_path.name,
            "mtime_ns": _file_identity(src_audio_path)[1],
            "size": _file_identity(src_audio_path)[2],
        },
    }

    # 4) Save cache and return
    ensure_dir(STT_DIR)
    write_stt(key, payload)
    return payload
