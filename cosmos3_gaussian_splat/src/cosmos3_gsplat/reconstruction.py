from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import GeometryConfig
from .vggt_backend import VGGTGeometryResult


@dataclass(frozen=True)
class ReconstructionResult:
    image_paths: tuple[Path, ...]
    poses_c2w: np.ndarray
    intrinsics: np.ndarray
    colmap_dir: Path
    metrics: dict[str, float | int | str | bool]
    root: Path

    def write(self) -> Path:
        metadata_path = self.root / "reconstruction.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "image_paths": [str(path) for path in self.image_paths],
                    "colmap_dir": str(self.colmap_dir),
                    "camera_convention": "OpenCV camera-to-world",
                    "metrics": self.metrics,
                },
                indent=2,
            )
            + "\n"
        )
        np.savez_compressed(
            self.root / "reconstruction.npz",
            poses_c2w=self.poses_c2w,
            intrinsics=self.intrinsics,
        )
        return metadata_path

    @classmethod
    def read(cls, root: str | Path) -> ReconstructionResult:
        directory = Path(root)
        metadata = json.loads((directory / "reconstruction.json").read_text())
        arrays = np.load(directory / "reconstruction.npz")
        return cls(
            image_paths=tuple(Path(path) for path in metadata["image_paths"]),
            poses_c2w=arrays["poses_c2w"],
            intrinsics=arrays["intrinsics"],
            colmap_dir=Path(metadata["colmap_dir"]),
            metrics=metadata["metrics"],
            root=directory,
        )


def _write_colmap_text(
    *,
    sparse_dir: Path,
    image_paths: list[Path],
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    width, height = Image.open(image_paths[0]).size
    median_k = np.median(intrinsics, axis=0)
    cameras = (
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {median_k[0, 0]:.12g} {median_k[1, 1]:.12g} "
        f"{median_k[0, 2]:.12g} {median_k[1, 2]:.12g}\n"
    )
    (sparse_dir / "cameras.txt").write_text(cameras)
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_id, (path, c2w) in enumerate(zip(image_paths, poses_c2w, strict=True), start=1):
        w2c = np.linalg.inv(c2w)
        quat_xyzw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        qx, qy, qz, qw = quat_xyzw
        tx, ty, tz = w2c[:3, 3]
        image_lines.append(
            f"{image_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {tx:.12g} {ty:.12g} {tz:.12g} 1 {path.name}"
        )
        image_lines.append("")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n")
    point_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for point_id, (point, color) in enumerate(zip(points, colors, strict=True), start=1):
        point_lines.append(
            f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {int(color[0])} {int(color[1])} {int(color[2])} 0"
        )
    (sparse_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n")


def _run_colmap_diagnostic(image_dir: Path, output_dir: Path) -> dict[str, float | int | str | bool]:
    """Run classical SfM as a non-blocking consistency diagnostic."""

    try:
        import pycolmap
    except ImportError:
        return {"colmap_status": "unavailable", "colmap_registered_views": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    try:
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(image_dir),
            camera_model="PINHOLE",
            camera_mode=pycolmap.CameraMode.SINGLE,
        )
        pycolmap.match_exhaustive(database_path=str(database_path))
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(output_dir / "sparse"),
        )
        if not reconstructions:
            return {"colmap_status": "no_reconstruction", "colmap_registered_views": 0}
        best = max(reconstructions.values(), key=lambda reconstruction: reconstruction.num_reg_images())
        registered = int(best.num_reg_images())
        return {
            "colmap_status": "complete",
            "colmap_registered_views": registered,
            "colmap_points": int(best.num_points3D()),
            "colmap_mean_reprojection_error": float(best.compute_mean_reprojection_error()),
        }
    except Exception as error:  # diagnostic failure must not block VGGT
        return {
            "colmap_status": "failed",
            "colmap_registered_views": 0,
            "colmap_error": f"{type(error).__name__}: {error}",
        }


def build_reconstruction(
    geometry: VGGTGeometryResult,
    config: GeometryConfig,
    output_dir: str | Path,
) -> ReconstructionResult:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    accepted = geometry.accepted_mask
    image_paths = [geometry.training_image_paths[index] for index in np.flatnonzero(accepted)]
    poses = geometry.aligned_poses_c2w[accepted]
    intrinsics = geometry.intrinsics[accepted]
    shared_k = np.repeat(np.median(intrinsics, axis=0, keepdims=True), len(intrinsics), axis=0).astype(np.float32)

    images_dir = root / "images"
    images_dir.mkdir(exist_ok=True)
    copied_paths: list[Path] = []
    for source in image_paths:
        destination = images_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied_paths.append(destination)
    sparse_dir = root / "sparse" / "0"
    _write_colmap_text(
        sparse_dir=sparse_dir,
        image_paths=copied_paths,
        poses_c2w=poses,
        intrinsics=shared_k,
        points=geometry.points,
        colors=geometry.colors,
    )
    metrics: dict[str, float | int | str | bool] = {
        "reconstruction_source": "vggt_aligned",
        "reconstruction_views": len(copied_paths),
        "shared_focal_x": float(shared_k[0, 0, 0]),
        "shared_focal_y": float(shared_k[0, 1, 1]),
        "vggt_bundle_adjustment_status": "deferred_to_bounded_gsplat_pose_optimization",
    }
    if config.run_colmap_diagnostic:
        metrics.update(_run_colmap_diagnostic(images_dir, root / "colmap_diagnostic"))
    result = ReconstructionResult(
        image_paths=tuple(copied_paths),
        poses_c2w=poses.astype(np.float32),
        intrinsics=shared_k,
        colmap_dir=sparse_dir,
        metrics=metrics,
        root=root,
    )
    result.write()
    return result
