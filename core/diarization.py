# core/diarization.py
# Optional, off-by-default diarization (2-speaker estimate).
# Safe to keep in the repo: it only imports heavy deps when enabled=True.
#
# Requirements (only if you plan to enable it):
#   pip install '.[diarization]'
# which provides: resemblyzer (embeddings) + scikit-learn (clustering)
#
# Usage (V2 idea; not used in V1):
#   cfg = DiarizationConfig(enabled=True, k_speakers=2)
#   segments = diarize_wav(Path("artifacts/audio_wav/foo.wav"), cfg)
#   -> [{"start": 0.00, "end": 1.50, "speaker": "A"}, {"start": 1.50, "end": 3.00, "speaker": "B"}, ...]


from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class DiarizationConfig:
    enabled: bool = False          # V1 keeps this False
    k_speakers: int = 2            # target clusters (2 = agent + customer)
    window_s: float = 1.5          # sliding window size for embeddings
    hop_s: float = 0.75            # hop size between windows
    min_segment_s: float = 0.75    # merge tiny runs into neighbors


def diarize_wav(wav_path: Path, cfg: DiarizationConfig) -> List[Dict]:
    """
    Lightweight diarization using speaker embeddings + clustering.
    Returns a list of segments: [{start, end, speaker}], speaker in {"A","B","C",...}.
    If cfg.enabled is False, returns [].

    Notes:
    - Heuristic only; not used for labeling in V1.
    - If dependencies are missing, raises a friendly RuntimeError with install hint.
    """
    if not cfg.enabled:
        return []

    # Import heavy deps lazily so the rest of the app stays lightweight.
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Diarization requires 'resemblyzer'. Install optional extra: pip install '.[diarization]'"
        ) from e

    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Diarization requires 'scikit-learn'. Install optional extra: pip install '.[diarization]'"
        ) from e

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # 1) Load & embed with sliding windows
    wav = preprocess_wav(str(wav_path))
    encoder = VoiceEncoder()
    # We ask for partial embeddings (one per window). Some versions return
    # (utt_embed, partial_embeds, slices); others may return (partial_embeds, slices).
    # We handle both shapes conservatively.
    partial_embeds = None
    try:
        _, partial_embeds, _ = encoder.embed_utterance(wav, return_partials=True)
    except TypeError:
        # Older API: returns only partials
        partial_embeds = encoder.embed_utterance(wav, return_partials=True)  # type: ignore

    if partial_embeds is None or len(partial_embeds) == 0:
        return []

    # 2) Cluster partial embeddings into k speakers
    k = max(1, int(cfg.k_speakers))
    k = min(k, len(partial_embeds))  # can't have more clusters than windows
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(partial_embeds)

    # 3) Convert window labels to time segments and merge consecutive same-speaker windows
    win = float(cfg.window_s)
    hop = float(cfg.hop_s)

    def label_to_speaker(idx: int) -> str:
        # 0->A, 1->B, 2->C ...
        return chr(ord("A") + (idx % 26))

    segments: List[Dict] = []
    if len(labels) == 0:
        return segments

    # Build raw segments from windows
    cur_label = int(labels[0])
    cur_start = 0.0
    for i in range(1, len(labels)):
        if int(labels[i]) != cur_label:
            # Window i starts at time i*hop; previous segment ends at that time + window size
            seg_end = i * hop + win  # approximate; windows overlap
            segments.append({"start": cur_start, "end": max(cur_start + hop, seg_end), "speaker": label_to_speaker(cur_label)})
            cur_label = int(labels[i])
            cur_start = i * hop
    # Last segment
    last_end = (len(labels) - 1) * hop + win
    segments.append({"start": cur_start, "end": last_end, "speaker": label_to_speaker(cur_label)})

    # 4) Clean-up: merge tiny segments and adjacent same-speaker segments
    merged: List[Dict] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] or (seg["end"] - seg["start"] < cfg.min_segment_s):
            # merge into previous
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg)

    # Ensure monotonic times and non-negative durations
    out: List[Dict] = []
    t_cursor = 0.0
    for seg in merged:
        start = max(t_cursor, float(seg["start"]))
        end = max(start, float(seg["end"]))
        if end - start > 1e-3:
            out.append({"start": round(start, 2), "end": round(end, 2), "speaker": seg["speaker"]})
            t_cursor = end

    return out
