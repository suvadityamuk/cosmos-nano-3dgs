from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def image_sharpness(path: str | Path) -> float:
    """Variance of a discrete Laplacian, without requiring OpenCV."""

    image = np.asarray(Image.open(path).convert("L").resize((256, 256)), dtype=np.float32)
    laplacian = -4.0 * image[1:-1, 1:-1] + image[:-2, 1:-1] + image[2:, 1:-1] + image[1:-1, :-2] + image[1:-1, 2:]
    return float(np.var(laplacian))


def select_keyframes(
    frame_paths: list[str | Path] | tuple[str | Path, ...],
    count: int,
    blur_threshold: float = 20.0,
) -> tuple[np.ndarray, dict[int, float]]:
    """Select angularly uniform sharp frames while retaining both closure frames."""

    total = len(frame_paths)
    if total < 3:
        raise ValueError("at least three frames are required")
    count = min(max(3, count), total)
    scores = {index: image_sharpness(path) for index, path in enumerate(frame_paths)}
    uniform = np.unique(np.rint(np.linspace(0, total - 1, count)).astype(np.int64))
    selected = {0, total - 1}
    selected.update(index for index in uniform if scores[int(index)] >= blur_threshold)
    for index in sorted(range(total), key=lambda item: scores[item], reverse=True):
        if len(selected) >= count:
            break
        selected.add(index)
    return np.asarray(sorted(selected), dtype=np.int64), scores


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    residuals: np.ndarray

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return self.scale * np.einsum("ij,...j->...i", self.rotation, values) + self.translation

    def transform_c2w(self, poses: np.ndarray) -> np.ndarray:
        values = np.asarray(poses, dtype=np.float64)
        transformed = values.copy()
        transformed[:, :3, :3] = self.rotation[None] @ values[:, :3, :3]
        transformed[:, :3, 3] = self.transform_points(values[:, :3, 3])
        return transformed


def _umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have equal shape [N, 3]")
    if len(source) < 3:
        raise ValueError("at least three points are needed for similarity alignment")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    source_variance = np.sum(source_centered**2) / len(source)
    if source_variance < 1e-12:
        raise ValueError("source camera centers are degenerate")
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance) if with_scale else 1.0
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def robust_similarity_alignment(
    source: np.ndarray,
    target: np.ndarray,
    *,
    residual_threshold: float | None = None,
    max_trials: int = 256,
    seed: int = 0,
) -> SimilarityTransform:
    """Estimate a Sim(3) with deterministic RANSAC followed by inlier refinement."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("source and target must have equal shape [N>=3, 3]")
    target_extent = np.linalg.norm(target - target.mean(axis=0), axis=1).max(initial=0.0)
    threshold = residual_threshold if residual_threshold is not None else max(target_extent * 0.08, 1e-4)
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(source), dtype=bool)
    best_error = float("inf")
    for _ in range(max_trials):
        indices = rng.choice(len(source), 3, replace=False)
        try:
            scale, rotation, translation = _umeyama(source[indices], target[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue
        predicted = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(predicted - target, axis=1)
        inliers = residuals <= threshold
        error = float(residuals[inliers].mean()) if inliers.any() else float("inf")
        if inliers.sum() > best_inliers.sum() or (inliers.sum() == best_inliers.sum() and error < best_error):
            best_inliers = inliers
            best_error = error
    if best_inliers.sum() < 3:
        best_inliers[:] = True
    scale, rotation, translation = _umeyama(source[best_inliers], target[best_inliers])
    predicted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    inliers = residuals <= threshold
    return SimilarityTransform(scale, rotation, translation, inliers, residuals)


def rotation_geodesic_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.shape[-2:] != (3, 3):
        raise ValueError("rotations must have equal shape [..., 3, 3]")
    relative = np.swapaxes(left, -1, -2) @ right
    return np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(left.shape[:-2])


def invert_poses(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("poses must have shape [N, 3, 4] or [N, 4, 4]")
    if values.shape[1:] == (3, 4):
        homogeneous = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
        homogeneous[:, :3] = values
        values = homogeneous
    return np.linalg.inv(values)
