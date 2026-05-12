"""Tests for audio feature extraction and motion alignment."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from embodied_motion_flow.audio.audio_processor import (
    extract_audio_features,
    find_audio_for_motion,
    infer_aist_music_id,
    load_or_extract_audio_features,
)


def _write_sine_wav(path: Path, sample_rate: int = 22050, seconds: float = 1.0) -> None:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_extract_audio_features_aligns_to_motion_frames(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_sine_wav(audio_path)
    features = extract_audio_features(audio_path, motion_frame_count=30, motion_fps=30.0)
    assert features.frame_features.shape == (30, 14)
    assert features.chroma.shape == (30, 12)
    assert features.beat_mask.shape == (30,)
    assert np.isfinite(features.frame_features).all()


def test_find_audio_for_motion_uses_aist_music_id(tmp_path: Path) -> None:
    audio_path = tmp_path / "mBR0_track.wav"
    _write_sine_wav(audio_path)
    motion_name = "gBR_sBM_cAll_d04_mBR0_ch08.pkl"
    assert infer_aist_music_id(motion_name) == "mBR0"
    assert find_audio_for_motion(motion_name, tmp_path) == audio_path


def test_find_audio_prefers_exact_sequence_audio(tmp_path: Path) -> None:
    exact = tmp_path / "gBR_sBM_cAll_d04_mBR0_ch08.wav"
    music = tmp_path / "mBR0_track.wav"
    _write_sine_wav(exact)
    _write_sine_wav(music)
    assert find_audio_for_motion("gBR_sBM_cAll_d04_mBR0_ch08.pkl", tmp_path) == exact


def test_audio_feature_cache_round_trip(tmp_path: Path) -> None:
    audio_path = tmp_path / "mBR0.wav"
    cache_dir = tmp_path / "cache"
    _write_sine_wav(audio_path)
    first = load_or_extract_audio_features(audio_path, motion_frame_count=30, motion_fps=30.0, cache_dir=cache_dir)
    second = load_or_extract_audio_features(audio_path, motion_frame_count=30, motion_fps=30.0, cache_dir=cache_dir)
    assert first.frame_features.shape == (30, 14)
    assert np.allclose(first.frame_features, second.frame_features)
    assert len(list(cache_dir.glob("*.npz"))) == 1
