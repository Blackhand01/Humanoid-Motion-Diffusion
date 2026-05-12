"""Audio feature extraction and temporal alignment for AIST++ music conditioning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re

import librosa
import numpy as np


LOGGER = logging.getLogger(__name__)
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".mp4")


@dataclass(frozen=True)
class AudioFeatures:
    """Aligned audio features for one motion clip.

    Shapes:
        frame_features: [motion_frames, 14], containing 12 chroma channels,
            normalized beat pulse, and normalized tempo.
        beat_mask: [motion_frames], binary nearest-beat indicator.
        chroma: [motion_frames, 12], interpolated chromatic energy.
    """

    frame_features: np.ndarray
    beat_mask: np.ndarray
    chroma: np.ndarray
    tempo_bpm: float
    beat_times: np.ndarray


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    mean = feature.mean(axis=0, keepdims=True)
    std = feature.std(axis=0, keepdims=True)
    return (feature - mean) / np.clip(std, 1e-6, None)


def infer_aist_music_id(motion_filename: str | Path) -> str:
    """Infer AIST++ music id from a motion filename.

    Example:
        gBR_sBM_cAll_d04_mBR0_ch08.pkl -> mBR0
    """
    stem = Path(motion_filename).stem
    match = re.search(r"_(m[A-Z]{2}\d+)_", f"_{stem}_")
    if match is None:
        raise ValueError(f"Could not infer AIST++ music id from filename: {motion_filename}")
    return match.group(1)


def _normalize_extensions(allowed_extensions: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    extensions = allowed_extensions or DEFAULT_AUDIO_EXTENSIONS
    return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions)


def find_audio_for_motion(
    motion_filename: str | Path,
    audio_root: str | Path = "audio",
    allowed_extensions: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    """Find the local .wav corresponding to an AIST++ motion filename.

    The function searches recursively under ``audio_root`` in this order:
    exact motion sequence id, exact music id, then music-id prefix. Exact
    sequence audio is preferred because AIST++ videos can contain sequence-level
    offsets even when they share the same underlying track.
    """
    motion_stem = Path(motion_filename).stem
    music_id = infer_aist_music_id(motion_filename)
    root = Path(audio_root).expanduser()
    if not root.exists():
        return None

    extensions = _normalize_extensions(allowed_extensions)
    audio_files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    for target_stem in (motion_stem, music_id):
        exact = [path for path in audio_files if path.stem == target_stem]
        if exact:
            return exact[0]
    prefixed = [path for path in audio_files if path.stem.startswith(music_id)]
    return prefixed[0] if prefixed else None


def _interpolate_feature_to_motion(
    feature: np.ndarray,
    feature_times: np.ndarray,
    motion_frame_count: int,
    motion_fps: float,
) -> np.ndarray:
    """Interpolate feature [channels, audio_frames] to [motion_frames, channels]."""
    if feature.ndim != 2:
        raise ValueError(f"Expected feature [channels, frames], received {feature.shape}")
    if motion_frame_count <= 0:
        raise ValueError("motion_frame_count must be positive")

    motion_times = np.arange(motion_frame_count, dtype=np.float32) / float(motion_fps)
    aligned = np.stack(
        [
            np.interp(
                motion_times,
                feature_times,
                feature[channel_idx],
                left=float(feature[channel_idx, 0]),
                right=float(feature[channel_idx, -1]),
            )
            for channel_idx in range(feature.shape[0])
        ],
        axis=1,
    )
    return aligned.astype(np.float32)


def _beat_mask_to_motion(beat_times: np.ndarray, motion_frame_count: int, motion_fps: float) -> np.ndarray:
    """Create a binary beat mask aligned to motion frames."""
    beat_mask = np.zeros((motion_frame_count,), dtype=np.float32)
    for beat_time in beat_times:
        frame = int(round(float(beat_time) * float(motion_fps)))
        if 0 <= frame < motion_frame_count:
            beat_mask[frame] = 1.0
    return beat_mask


def extract_audio_features(
    audio_path: str | Path,
    motion_frame_count: int,
    motion_fps: float,
    sample_rate: int = 22050,
    hop_length: int = 512,
) -> AudioFeatures:
    """Extract beat, tempo, and chroma features aligned to motion frames.

    Args:
        audio_path: Path to an audio file supported by librosa.
        motion_frame_count: Number of target motion frames.
        motion_fps: Motion frame rate in Hz.
        sample_rate: Audio sample rate used by librosa.
        hop_length: STFT hop length for chroma/beat extraction.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    tempo_raw, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    tempo = float(np.asarray(tempo_raw).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_mask = _beat_mask_to_motion(beat_times, motion_frame_count, motion_fps)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
    chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    aligned_chroma = _normalize_feature(
        _interpolate_feature_to_motion(chroma, chroma_times, motion_frame_count, motion_fps)
    )

    tempo_feature = np.full((motion_frame_count, 1), tempo / 240.0, dtype=np.float32)
    beat_feature = beat_mask[:, None]
    frame_features = np.concatenate([aligned_chroma, beat_feature, tempo_feature], axis=1).astype(np.float32)

    return AudioFeatures(
        frame_features=frame_features,
        beat_mask=beat_mask,
        chroma=aligned_chroma,
        tempo_bpm=tempo,
        beat_times=beat_times.astype(np.float32),
    )


def _audio_cache_key(
    audio_path: Path,
    motion_frame_count: int,
    motion_fps: float,
    sample_rate: int,
    hop_length: int,
) -> str:
    stat = audio_path.stat()
    payload = {
        "path": str(audio_path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "motion_frame_count": int(motion_frame_count),
        "motion_fps": float(motion_fps),
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def load_or_extract_audio_features(
    audio_path: str | Path,
    motion_frame_count: int,
    motion_fps: float,
    sample_rate: int = 22050,
    hop_length: int = 512,
    cache_dir: str | Path | None = None,
) -> AudioFeatures:
    """Extract audio features with an on-disk NumPy cache.

    Librosa feature extraction is expensive enough to dominate local MPS
    debugging when many clips reuse the same song. The cache key includes the
    file mtime/size and target alignment parameters, so stale features are not
    reused after audio replacement or config changes.
    """
    path = Path(audio_path).expanduser()
    if cache_dir is None:
        return extract_audio_features(path, motion_frame_count, motion_fps, sample_rate, hop_length)

    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{path.stem}_{_audio_cache_key(path, motion_frame_count, motion_fps, sample_rate, hop_length)}.npz"
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            return AudioFeatures(
                frame_features=cached["frame_features"].astype(np.float32),
                beat_mask=cached["beat_mask"].astype(np.float32),
                chroma=cached["chroma"].astype(np.float32),
                tempo_bpm=float(cached["tempo_bpm"].reshape(-1)[0]),
                beat_times=cached["beat_times"].astype(np.float32),
            )
        except Exception as exc:
            LOGGER.warning("Ignoring corrupt audio feature cache %s: %s", cache_path, exc)

    features = extract_audio_features(path, motion_frame_count, motion_fps, sample_rate, hop_length)
    np.savez_compressed(
        cache_path,
        frame_features=features.frame_features,
        beat_mask=features.beat_mask,
        chroma=features.chroma,
        tempo_bpm=np.array([features.tempo_bpm], dtype=np.float32),
        beat_times=features.beat_times,
    )
    return features
