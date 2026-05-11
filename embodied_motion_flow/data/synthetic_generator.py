"""Synthetic 12-DOF humanoid trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from embodied_motion_flow.config import DataConfig


JOINT_NAMES: tuple[str, ...] = (
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


@dataclass(frozen=True)
class SyntheticBatch:
    """Container for generated motion trajectories.

    Shapes:
        trajectories: [batch, time, joints]
        anomaly_labels: [batch]
    """

    trajectories: np.ndarray
    anomaly_labels: np.ndarray
    modes: list[str]


def _time_axis(sequence_length: int, sample_rate_hz: float) -> np.ndarray:
    return np.arange(sequence_length, dtype=np.float32) / sample_rate_hz


def _walking_pattern(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    traj = np.zeros((t.shape[0], 12), dtype=np.float32)
    phase = 2.0 * np.pi * freq_hz * t + rng.uniform(0.0, 2.0 * np.pi)
    traj[:, 0] = amplitude * 0.8 * np.sin(phase)
    traj[:, 3] = amplitude * 0.8 * np.sin(phase + np.pi)
    traj[:, 1] = amplitude * 1.1 * np.maximum(0.0, np.sin(phase + 0.2))
    traj[:, 4] = amplitude * 1.1 * np.maximum(0.0, np.sin(phase + np.pi + 0.2))
    traj[:, 2] = amplitude * 0.6 * np.sin(phase + 0.5)
    traj[:, 5] = amplitude * 0.6 * np.sin(phase + np.pi + 0.5)
    traj[:, 6] = amplitude * 0.35 * np.sin(phase + np.pi)
    traj[:, 9] = amplitude * 0.35 * np.sin(phase)
    traj[:, 7] = amplitude * 0.20 * np.sin(phase + 1.4)
    traj[:, 10] = amplitude * 0.20 * np.sin(phase + np.pi + 1.4)
    traj[:, 8] = amplitude * 0.16 * np.sin(phase + 2.1)
    traj[:, 11] = amplitude * 0.16 * np.sin(phase + np.pi + 2.1)
    return traj


def _reaching_pattern(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    traj = np.zeros((t.shape[0], 12), dtype=np.float32)
    phase = 2.0 * np.pi * freq_hz * t + rng.uniform(0.0, np.pi)
    envelope = 0.5 * (1.0 + np.sin(phase))
    side = rng.choice([0, 1])
    sh = 6 if side == 0 else 9
    el = 7 if side == 0 else 10
    wr = 8 if side == 0 else 11
    traj[:, sh] = amplitude * (0.2 + 1.0 * envelope)
    traj[:, el] = amplitude * (0.9 - 0.5 * envelope)
    traj[:, wr] = amplitude * (0.4 + 0.7 * envelope)
    traj[:, 0] = amplitude * 0.2 * np.sin(phase + 0.7)
    traj[:, 3] = amplitude * 0.2 * np.sin(phase + np.pi + 0.7)
    traj[:, 1] = amplitude * 0.15 * np.maximum(0.0, np.sin(phase + 0.4))
    traj[:, 4] = amplitude * 0.15 * np.maximum(0.0, np.sin(phase + np.pi + 0.4))
    return traj


def _idle_stabilization_pattern(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    traj = np.zeros((t.shape[0], 12), dtype=np.float32)
    slow = 2.0 * np.pi * freq_hz * 0.5 * t + rng.uniform(0.0, 2.0 * np.pi)
    micro = 2.0 * np.pi * freq_hz * 3.0 * t + rng.uniform(0.0, 2.0 * np.pi)
    traj[:, 0] = amplitude * 0.15 * np.sin(slow)
    traj[:, 3] = amplitude * 0.15 * np.sin(slow + np.pi)
    traj[:, 6] = amplitude * 0.12 * np.sin(slow + 0.3)
    traj[:, 9] = amplitude * 0.12 * np.sin(slow + np.pi + 0.3)
    traj += (amplitude * 0.03 * np.sin(micro))[:, None]
    return traj


def _inspection_pattern(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    traj = np.zeros((t.shape[0], 12), dtype=np.float32)
    phase = 2.0 * np.pi * freq_hz * 0.7 * t + rng.uniform(0.0, np.pi)
    head_tilt_like = amplitude * 0.4 * np.sin(phase)
    traj[:, 6] = amplitude * 0.65 + 0.15 * head_tilt_like
    traj[:, 7] = amplitude * 0.45 + 0.1 * np.sin(phase + 0.6)
    traj[:, 8] = amplitude * 0.2 + 0.1 * np.sin(phase + 1.3)
    traj[:, 9] = amplitude * 0.35 + 0.1 * np.sin(phase + np.pi)
    traj[:, 10] = amplitude * 0.25 + 0.08 * np.sin(phase + np.pi + 0.8)
    traj[:, 11] = amplitude * 0.08 + 0.05 * np.sin(phase + np.pi + 1.2)
    traj[:, 0] = amplitude * 0.1 * np.sin(phase * 0.5)
    traj[:, 3] = amplitude * 0.1 * np.sin(phase * 0.5 + np.pi)
    return traj


def _recovery_pattern(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    traj = np.zeros((t.shape[0], 12), dtype=np.float32)
    damping = np.exp(-2.0 * t / max(t.max(), 1e-6))
    phase = 2.0 * np.pi * freq_hz * 1.3 * t + rng.uniform(0.0, 2.0 * np.pi)
    osc = damping * np.sin(phase)
    traj[:, 0] = amplitude * 0.7 * osc
    traj[:, 3] = amplitude * 0.7 * np.sin(phase + np.pi) * damping
    traj[:, 1] = amplitude * 0.8 * np.maximum(0.0, osc)
    traj[:, 4] = amplitude * 0.8 * np.maximum(0.0, np.sin(phase + np.pi) * damping)
    traj[:, 6] = amplitude * 0.4 * np.sin(phase + 0.9) * damping
    traj[:, 9] = amplitude * 0.4 * np.sin(phase + np.pi + 0.9) * damping
    traj[:, 7] = amplitude * 0.25 * np.sin(phase + 1.5) * damping
    traj[:, 10] = amplitude * 0.25 * np.sin(phase + np.pi + 1.5) * damping
    return traj


def _nominal_mode_generators() -> dict[str, Callable[[np.ndarray, float, float, np.random.Generator], np.ndarray]]:
    return {
        "walking": _walking_pattern,
        "reaching": _reaching_pattern,
        "idle_stabilization": _idle_stabilization_pattern,
        "inspection_poses": _inspection_pattern,
        "recovery_motions": _recovery_pattern,
    }


def _apply_anomalies(
    trajectory: np.ndarray,
    data_cfg: DataConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject anomalous effects in-place and return trajectory."""
    anomaly_cfg = data_cfg.anomalies
    joint_lower = np.asarray(data_cfg.joint_limits.lower, dtype=np.float32)
    joint_upper = np.asarray(data_cfg.joint_limits.upper, dtype=np.float32)
    seq_len, num_joints = trajectory.shape

    if rng.random() < anomaly_cfg.unstable_pose_probability:
        center = rng.integers(low=seq_len // 6, high=seq_len - seq_len // 6)
        half_width = max(2, seq_len // 20)
        start = max(0, center - half_width)
        end = min(seq_len, center + half_width)
        unstable_joints = rng.choice(num_joints, size=rng.integers(2, 5), replace=False)
        direction = rng.choice([-1.0, 1.0], size=unstable_joints.shape[0])
        violation = direction * rng.uniform(0.4, 1.0, size=unstable_joints.shape[0])
        trajectory[start:end, unstable_joints] += violation

    if rng.random() < anomaly_cfg.sensor_spike_probability:
        n_spikes = max(1, seq_len // 16)
        spike_steps = rng.integers(0, seq_len, size=n_spikes)
        spike_joints = rng.integers(0, num_joints, size=n_spikes)
        spike_values = rng.normal(0.0, anomaly_cfg.spike_scale, size=n_spikes)
        trajectory[spike_steps, spike_joints] += spike_values.astype(np.float32)

    if rng.random() < anomaly_cfg.jitter_probability:
        jitter = rng.normal(0.0, anomaly_cfg.jitter_scale, size=(seq_len, num_joints)).astype(np.float32)
        high_freq = np.sign(np.sin(np.linspace(0.0, 25.0 * np.pi, seq_len, dtype=np.float32)))[:, None]
        trajectory += high_freq * jitter

    if rng.random() < anomaly_cfg.abrupt_acceleration_probability:
        pivot = rng.integers(seq_len // 4, 3 * seq_len // 4)
        step = rng.normal(0.0, anomaly_cfg.spike_scale * 0.5, size=(num_joints,)).astype(np.float32)
        trajectory[pivot:] += step[None, :]

    if rng.random() < anomaly_cfg.long_tail_probability:
        drift = rng.normal(0.0, anomaly_cfg.drift_scale, size=(num_joints,)).astype(np.float32)
        time_gain = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)[:, None]
        trajectory += time_gain * drift[None, :]

    # Explicit range-limit violations for long-tail outliers.
    outlier_joints = rng.choice(num_joints, size=rng.integers(1, 4), replace=False)
    for joint in outlier_joints:
        if rng.random() < 0.5:
            trajectory[:, joint] += (joint_upper[joint] - joint_lower[joint]) * rng.uniform(0.2, 0.6)
        else:
            trajectory[:, joint] -= (joint_upper[joint] - joint_lower[joint]) * rng.uniform(0.2, 0.6)

    return trajectory


def generate_synthetic_batch(
    batch_size: int,
    data_cfg: DataConfig,
    seed: int,
) -> SyntheticBatch:
    """Generate synthetic 12-DOF trajectories with nominal and anomalous motions.

    Args:
        batch_size: Number of trajectories in the batch.
        data_cfg: Data config block.
        seed: RNG seed for deterministic generation.
    """
    rng = np.random.default_rng(seed)
    t = _time_axis(data_cfg.sequence_length, data_cfg.sample_rate_hz)
    mode_generators = _nominal_mode_generators()
    enabled_modes = [
        mode_name
        for mode_name in mode_generators.keys()
        if data_cfg.motion_modes.get(mode_name, None) is not None
        and data_cfg.motion_modes[mode_name].enabled
    ]
    if not enabled_modes:
        raise ValueError("At least one motion mode must be enabled in config.")

    trajectories = np.zeros((batch_size, data_cfg.sequence_length, data_cfg.num_joints), dtype=np.float32)
    anomaly_labels = np.zeros((batch_size,), dtype=np.int64)
    modes: list[str] = []

    for idx in range(batch_size):
        mode_name = enabled_modes[idx % len(enabled_modes)] if rng.random() < 0.4 else rng.choice(enabled_modes)
        mode_cfg = data_cfg.motion_modes[mode_name]
        generator = mode_generators[mode_name]
        amplitude_scale = mode_cfg.amplitude_scale * rng.uniform(0.85, 1.15)
        frequency_hz = mode_cfg.frequency_hz * rng.uniform(0.9, 1.1)
        sample = generator(t, frequency_hz, amplitude_scale, rng)

        # Add low-level sensor noise to all trajectories.
        sample += rng.normal(
            loc=0.0,
            scale=data_cfg.anomalies.sensor_noise_std * 0.35,
            size=sample.shape,
        ).astype(np.float32)

        if data_cfg.anomalies.enabled and rng.random() < data_cfg.anomalies.anomaly_fraction:
            sample = _apply_anomalies(sample, data_cfg, rng)
            anomaly_labels[idx] = 1

        trajectories[idx] = sample
        modes.append(mode_name)

    return SyntheticBatch(trajectories=trajectories, anomaly_labels=anomaly_labels, modes=modes)
