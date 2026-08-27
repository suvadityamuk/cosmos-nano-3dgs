from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TrajectoryConfig


@dataclass(frozen=True)
class CameraTrajectory:
    """Absolute OpenCV camera poses and the corresponding Cosmos raw actions."""

    poses_c2w: np.ndarray
    raw_actions: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray

    def __post_init__(self) -> None:
        if self.poses_c2w.ndim != 3 or self.poses_c2w.shape[1:] != (4, 4):
            raise ValueError(f"poses_c2w must have shape [T, 4, 4], got {self.poses_c2w.shape}")
        if self.raw_actions.shape != (len(self.poses_c2w) - 1, 9):
            raise ValueError("raw_actions must have shape [T-1, 9]")

    def write(self, directory: str | Path) -> dict[str, Path]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        poses_path = root / "commanded_poses_c2w.npy"
        actions_path = root / "camera_actions.json"
        metadata_path = root / "trajectory.json"
        np.save(poses_path, self.poses_c2w)
        actions_path.write_text(json.dumps(self.raw_actions.tolist()) + "\n")
        metadata_path.write_text(
            json.dumps(
                {
                    "camera_convention": "OpenCV camera-to-world; +x right, +y down, +z forward",
                    "pose_convention": "backward_framewise",
                    "rotation_format": "rot6d_columns",
                    "num_frames": len(self.poses_c2w),
                    "num_actions": len(self.raw_actions),
                    "azimuth_deg": self.azimuth_deg.tolist(),
                    "elevation_deg": self.elevation_deg.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        return {"poses": poses_path, "actions": actions_path, "metadata": metadata_path}


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / norm


def look_at_c2w(
    center: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """Create an OpenCV camera-to-world pose looking from *center* at *target*."""

    up = np.asarray(world_up if world_up is not None else (0.0, 0.0, 1.0), dtype=np.float64)
    forward = _normalize(np.asarray(target, dtype=np.float64) - np.asarray(center, dtype=np.float64))
    right = _normalize(np.cross(forward, up))
    down = _normalize(np.cross(forward, right))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack((right, down, forward), axis=1)
    pose[:3, 3] = center
    return pose


def rotation_matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Encode a rotation as the column-based 6D representation used by Cosmos."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in shape [3, 3]")
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode Cosmos column-based rot6d and project it to SO(3)."""

    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError("rot6d must end in dimension 6")
    columns = values.reshape(*values.shape[:-1], 2, 3)
    col0 = columns[..., 0, :]
    col1 = columns[..., 1, :]
    col0 = col0 / np.clip(np.linalg.norm(col0, axis=-1, keepdims=True), 1e-8, None)
    col1 = col1 - np.sum(col0 * col1, axis=-1, keepdims=True) * col0
    col1 = col1 / np.clip(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8, None)
    col2 = np.cross(col0, col1)
    return np.stack((col0, col1, col2), axis=-1)


def poses_to_actions(poses_c2w: np.ndarray) -> np.ndarray:
    """Convert absolute poses to backward-framewise translation + rot6d actions."""

    poses = np.asarray(poses_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("poses_c2w must have shape [T>=2, 4, 4]")
    deltas = np.linalg.inv(poses[:-1]) @ poses[1:]
    return np.concatenate((deltas[:, :3, 3], rotation_matrix_to_rot6d(deltas[:, :3, :3])), axis=-1).astype(np.float32)


def actions_to_poses(raw_actions: np.ndarray, initial_pose_c2w: np.ndarray | None = None) -> np.ndarray:
    """Integrate backward-framewise Cosmos actions into absolute poses."""

    actions = np.asarray(raw_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 9:
        raise ValueError("raw_actions must have shape [T, 9]")
    current = np.asarray(initial_pose_c2w if initial_pose_c2w is not None else np.eye(4), dtype=np.float64)
    poses = [current.copy()]
    for action in actions:
        delta = np.eye(4, dtype=np.float64)
        delta[:3, 3] = action[:3]
        delta[:3, :3] = rot6d_to_rotation_matrix(action[3:])
        current = current @ delta
        poses.append(current.copy())
    return np.stack(poses)


def make_closed_helical_trajectory(config: TrajectoryConfig) -> CameraTrajectory:
    """Create a closed orbit with sinusoidal elevation.

    The elevation completes one sinusoid while azimuth completes ``turns`` turns,
    visiting eye-level, high, eye-level, low, then closing at the initial pose.
    """

    frame_count = config.num_actions + 1
    phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    azimuth = 2.0 * np.pi * config.turns * phase
    elevation = np.deg2rad(config.elevation_amplitude_deg) * np.sin(2.0 * np.pi * phase)
    target = np.asarray(config.target_xyz, dtype=np.float64)
    horizontal = config.radius_m * np.cos(elevation)
    centers = np.stack(
        (
            target[0] + horizontal * np.cos(azimuth),
            target[1] + horizontal * np.sin(azimuth),
            target[2] + config.radius_m * np.sin(elevation),
        ),
        axis=-1,
    )
    poses = np.stack([look_at_c2w(center, target) for center in centers])
    actions = poses_to_actions(poses)
    return CameraTrajectory(
        poses_c2w=poses.astype(np.float32),
        raw_actions=actions,
        azimuth_deg=np.rad2deg(azimuth).astype(np.float32),
        elevation_deg=np.rad2deg(elevation).astype(np.float32),
    )
