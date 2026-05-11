"""3D SMPL skeleton rendering utilities for AIST++ validation."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

SMPL_REST_OFFSETS = np.array(
    [
        [0.00, 0.00, 0.00],
        [-0.09, -0.10, 0.00],
        [0.09, -0.10, 0.00],
        [0.00, 0.12, 0.00],
        [0.00, -0.42, 0.02],
        [0.00, -0.42, 0.02],
        [0.00, 0.16, 0.00],
        [0.00, -0.43, -0.01],
        [0.00, -0.43, -0.01],
        [0.00, 0.16, 0.00],
        [0.00, -0.08, 0.12],
        [0.00, -0.08, 0.12],
        [0.00, 0.16, 0.00],
        [-0.06, 0.10, 0.00],
        [0.06, 0.10, 0.00],
        [0.00, 0.14, 0.02],
        [-0.18, 0.02, 0.00],
        [0.18, 0.02, 0.00],
        [-0.28, 0.00, 0.00],
        [0.28, 0.00, 0.00],
        [-0.25, 0.00, 0.00],
        [0.25, 0.00, 0.00],
        [-0.08, 0.00, 0.00],
        [0.08, 0.00, 0.00],
    ],
    dtype=np.float32,
)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotations [N, 3] to rotation matrices [N, 3, 3]."""
    vectors = np.asarray(axis_angle, dtype=np.float32).reshape(-1, 3)
    angles = np.linalg.norm(vectors, axis=1, keepdims=True)
    axes = vectors / np.clip(angles, 1e-8, None)
    x = axes[:, 0]
    y = axes[:, 1]
    z = axes[:, 2]
    zeros = np.zeros_like(x)
    skew = np.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    eye = np.eye(3, dtype=np.float32)[None, :, :]
    sin = np.sin(angles)[:, :, None]
    cos = np.cos(angles)[:, :, None]
    return eye + sin * skew + (1.0 - cos) * np.matmul(skew, skew)


def smpl_axis_angle_to_joints(pose: np.ndarray) -> np.ndarray:
    """Forward-kinematics approximation from SMPL axis-angle pose to joints [24, 3]."""
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape != (72,):
        raise ValueError(f"Expected one SMPL pose with shape [72], received {arr.shape}")

    local_rot = axis_angle_to_matrix(arr.reshape(24, 3))
    global_rot = np.zeros((24, 3, 3), dtype=np.float32)
    joints = np.zeros((24, 3), dtype=np.float32)

    for joint_idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx < 0:
            global_rot[joint_idx] = local_rot[joint_idx]
            joints[joint_idx] = SMPL_REST_OFFSETS[joint_idx]
            continue
        global_rot[joint_idx] = global_rot[parent_idx] @ local_rot[joint_idx]
        joints[joint_idx] = joints[parent_idx] + global_rot[parent_idx] @ SMPL_REST_OFFSETS[joint_idx]

    return joints


def smpl_motion_to_joints(motion: np.ndarray) -> np.ndarray:
    """Convert motion [T, 72] to approximate 3D SMPL joint positions [T, 24, 3]."""
    arr = np.asarray(motion, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 72:
        raise ValueError(f"Expected SMPL motion [T, 72], received {arr.shape}")
    return np.stack([smpl_axis_angle_to_joints(frame) for frame in arr], axis=0)


def _draw_smpl_frame(axis: plt.Axes, joints: np.ndarray, title: str) -> None:
    axis.clear()
    for joint_idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx < 0:
            continue
        segment = joints[[parent_idx, joint_idx]]
        axis.plot(segment[:, 0], segment[:, 2], segment[:, 1], "-o", linewidth=2.0, markersize=3.5)

    axis.set_title(title)
    axis.set_xlim(-1.1, 1.1)
    axis.set_ylim(-1.1, 1.1)
    axis.set_zlim(-1.2, 1.2)
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.set_zlabel("y")
    axis.view_init(elev=18.0, azim=-70.0)


def save_smpl_static_frame(motion: np.ndarray, output_path: Path, frame_index: int = 0, dpi: int = 140) -> Path:
    """Save a static 3D SMPL skeleton frame from motion [T, 72]."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joints = smpl_motion_to_joints(motion)
    frame = int(np.clip(frame_index, 0, joints.shape[0] - 1))
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    axis = fig.add_subplot(111, projection="3d")
    _draw_smpl_frame(axis, joints[frame], f"SMPL 24-joint frame {frame}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def save_smpl_3d_animation(
    motion: np.ndarray,
    output_path: Path,
    fps: int,
    max_frames: int,
    dpi: int = 140,
) -> Path:
    """Save an MP4 3D SMPL skeleton animation from motion [T, 72]."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joints = smpl_motion_to_joints(motion)
    frame_count = min(joints.shape[0], max_frames)
    frame_indices = np.linspace(0, joints.shape[0] - 1, frame_count, dtype=np.int64)

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    axis = fig.add_subplot(111, projection="3d")
    frames: list[np.ndarray] = []
    for display_idx, frame_idx in enumerate(frame_indices):
        _draw_smpl_frame(axis, joints[frame_idx], f"SMPL 24-joint frame {display_idx + 1}/{frame_count}")
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        frames.append(frame.copy())

    plt.close(fig)
    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)
    return output_path
