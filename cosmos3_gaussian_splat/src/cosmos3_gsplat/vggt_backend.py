from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import GeometryConfig
from .geometry import (
    invert_poses,
    robust_similarity_alignment,
    rotation_geodesic_degrees,
    select_keyframes,
)
from .telemetry import measure_stage, release_gpu_memory


def write_point_cloud_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(points, dtype="<f4").reshape(-1, 3)
    rgb = np.clip(np.asarray(colors).reshape(-1, 3), 0, 255).astype("u1")
    if len(xyz) != len(rgb):
        raise ValueError("points and colors must have equal length")
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["r"], vertex["g"], vertex["b"] = rgb.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with destination.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)
    return destination


def _load_vggt_padded_mask(path: str | Path, target_size: int = 518) -> np.ndarray:
    image = Image.open(path).convert("L")
    width, height = image.size
    if width >= height:
        resized_width = target_size
        resized_height = round(height * (target_size / width) / 14) * 14
    else:
        resized_height = target_size
        resized_width = round(width * (target_size / height) / 14) * 14
    resized = image.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    canvas = Image.new("L", (target_size, target_size), 0)
    canvas.paste(resized, ((target_size - resized_width) // 2, (target_size - resized_height) // 2))
    return np.asarray(canvas) >= 128


def _pycolmap_image_extrinsic(image) -> np.ndarray:
    """Read a 3x4 W2C matrix across PyCOLMAP 3.x/4.x API differences."""

    transform = image.cam_from_world
    if callable(transform):
        transform = transform()
    matrix = transform.matrix
    if callable(matrix):
        matrix = matrix()
    return np.asarray(matrix)[:3]


def _run_vggt_bundle_adjustment(
    *,
    images,
    depth_confidence: np.ndarray,
    points_3d: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str | bool]] | None:
    """Use VGGT's official track predictor and PyCOLMAP conversion for one BA pass."""

    try:
        import pycolmap
        import torch
        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
        from vggt.dependency.track_predict import predict_tracks
    except ImportError:
        return None
    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            ),
        ):
            tracks, visibility, _, track_points, track_colors = predict_tracks(
                images,
                conf=depth_confidence,
                points_3d=points_3d,
                masks=None,
                max_query_pts=4096,
                query_frame_num=min(8, len(images)),
                keypoint_extractor="aliked+sp",
                fine_tracking=True,
            )
        reconstruction, _ = batch_np_matrix_to_pycolmap(
            track_points,
            extrinsics_w2c,
            intrinsics,
            tracks,
            np.asarray(images.shape[-2:]),
            masks=visibility > 0.2,
            max_reproj_error=8.0,
            shared_camera=True,
            camera_type="PINHOLE",
            points_rgb=track_colors,
        )
        if reconstruction is None:
            return None
        before = float(reconstruction.compute_mean_reprojection_error())
        pycolmap.bundle_adjustment(reconstruction, pycolmap.BundleAdjustmentOptions())
        after = float(reconstruction.compute_mean_reprojection_error())
        image_ids = sorted(reconstruction.images)
        refined_extrinsics = np.stack(
            [_pycolmap_image_extrinsic(reconstruction.images[image_id]) for image_id in image_ids]
        )
        refined_intrinsics = np.stack(
            [
                np.asarray(reconstruction.cameras[reconstruction.images[image_id].camera_id].calibration_matrix())
                for image_id in image_ids
            ]
        )
        point_ids = sorted(reconstruction.points3D)
        refined_points = np.stack([np.asarray(reconstruction.points3D[point_id].xyz) for point_id in point_ids])
        refined_colors = np.stack([np.asarray(reconstruction.points3D[point_id].color) for point_id in point_ids])
        sparse_dir = output_dir / "ba_sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        reconstruction.write(sparse_dir)
        return (
            refined_extrinsics.astype(np.float32),
            refined_intrinsics.astype(np.float32),
            refined_points.astype(np.float32),
            refined_colors.astype(np.uint8),
            {
                "vggt_bundle_adjustment_status": "complete",
                "vggt_ba_reprojection_before": before,
                "vggt_ba_reprojection_after": after,
                "vggt_ba_points": len(refined_points),
            },
        )
    except Exception as error:
        return (
            extrinsics_w2c,
            intrinsics,
            points_3d.reshape(-1, 3)[:0],
            np.empty((0, 3), dtype=np.uint8),
            {
                "vggt_bundle_adjustment_status": "failed",
                "vggt_ba_error": f"{type(error).__name__}: {error}",
            },
        )


@dataclass(frozen=True)
class VGGTGeometryResult:
    keyframe_indices: np.ndarray
    keyframe_paths: tuple[Path, ...]
    training_image_paths: tuple[Path, ...]
    accepted_mask: np.ndarray
    commanded_poses_c2w: np.ndarray
    predicted_poses_c2w: np.ndarray
    aligned_poses_c2w: np.ndarray
    intrinsics: np.ndarray
    depths: np.ndarray
    depth_confidence: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    metrics: dict[str, float | int | str | bool]
    root: Path

    @property
    def accepted_indices(self) -> np.ndarray:
        return np.flatnonzero(self.accepted_mask)

    def write(self) -> dict[str, Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        arrays_path = self.root / "vggt_geometry.npz"
        np.savez_compressed(
            arrays_path,
            keyframe_indices=self.keyframe_indices,
            accepted_mask=self.accepted_mask,
            commanded_poses_c2w=self.commanded_poses_c2w,
            predicted_poses_c2w=self.predicted_poses_c2w,
            aligned_poses_c2w=self.aligned_poses_c2w,
            intrinsics=self.intrinsics,
            depths=self.depths,
            depth_confidence=self.depth_confidence,
            points=self.points,
            colors=self.colors,
        )
        point_cloud_path = write_point_cloud_ply(self.root / "vggt_points.ply", self.points, self.colors)
        metadata_path = self.root / "geometry.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "keyframe_paths": [str(path) for path in self.keyframe_paths],
                    "training_image_paths": [str(path) for path in self.training_image_paths],
                    "accepted_keyframe_positions": self.accepted_indices.tolist(),
                    "metrics": self.metrics,
                },
                indent=2,
            )
            + "\n"
        )
        return {"arrays": arrays_path, "point_cloud": point_cloud_path, "metadata": metadata_path}

    @classmethod
    def read(cls, root: str | Path) -> VGGTGeometryResult:
        directory = Path(root)
        arrays = np.load(directory / "vggt_geometry.npz")
        metadata = json.loads((directory / "geometry.json").read_text())
        return cls(
            keyframe_indices=arrays["keyframe_indices"],
            keyframe_paths=tuple(Path(path) for path in metadata["keyframe_paths"]),
            training_image_paths=tuple(Path(path) for path in metadata["training_image_paths"]),
            accepted_mask=arrays["accepted_mask"],
            commanded_poses_c2w=arrays["commanded_poses_c2w"],
            predicted_poses_c2w=arrays["predicted_poses_c2w"],
            aligned_poses_c2w=arrays["aligned_poses_c2w"],
            intrinsics=arrays["intrinsics"],
            depths=arrays["depths"],
            depth_confidence=arrays["depth_confidence"],
            points=arrays["points"],
            colors=arrays["colors"],
            metrics=metadata["metrics"],
            root=directory,
        )


class VGGTBackend:
    def __init__(self, config: GeometryConfig) -> None:
        self.config = config

    def reconstruct(
        self,
        *,
        frame_paths: list[str | Path] | tuple[str | Path, ...],
        commanded_poses_c2w: np.ndarray,
        output_dir: str | Path,
        object_mask: str | Path | None = None,
    ) -> VGGTGeometryResult:
        try:
            import torch
            from vggt.models.vggt import VGGT
            from vggt.utils.geometry import unproject_depth_map_to_point_map
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except ImportError as error:
            raise RuntimeError("VGGT reconstruction requires the package's 'gpu' optional dependencies") from error
        if not torch.cuda.is_available():
            raise RuntimeError("VGGT reconstruction requires a CUDA GPU")
        if len(frame_paths) != len(commanded_poses_c2w):
            raise ValueError("frame_paths and commanded_poses_c2w must have equal length")

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        indices, sharpness = select_keyframes(frame_paths, self.config.num_keyframes, self.config.blur_threshold)
        selected_paths = tuple(Path(frame_paths[index]) for index in indices)
        selected_commands = np.asarray(commanded_poses_c2w)[indices]
        metrics: dict[str, float | int | str | bool] = {
            "candidate_frames": len(frame_paths),
            "selected_keyframes": len(indices),
            "sharpness_min": min(sharpness.values()),
            "sharpness_median": float(np.median(list(sharpness.values()))),
        }
        model = None
        ba_result = None
        with measure_stage("vggt_geometry") as stage_metrics:
            try:
                model = (
                    VGGT.from_pretrained(
                        self.config.model_id,
                        revision=self.config.model_revision,
                        map_location="cpu",
                        strict=True,
                    )
                    .to("cuda")
                    .eval()
                )
                images = load_and_preprocess_images([str(path) for path in selected_paths], mode="pad").to("cuda")
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    batched_images = images[None]
                    tokens, patch_start_index = model.aggregator(batched_images)
                    pose_encoding = model.camera_head(tokens)[-1]
                    extrinsics_w2c, intrinsics = pose_encoding_to_extri_intri(pose_encoding, batched_images.shape[-2:])
                    depth, confidence = model.depth_head(tokens, batched_images, patch_start_index)
                points = unproject_depth_map_to_point_map(
                    depth.squeeze(0), extrinsics_w2c.squeeze(0), intrinsics.squeeze(0)
                )
                images_np = images.detach().float().cpu().numpy()
                extrinsics_np = extrinsics_w2c.squeeze(0).detach().float().cpu().numpy()
                intrinsics_np = intrinsics.squeeze(0).detach().float().cpu().numpy()
                depths_np = depth.squeeze(0).detach().float().cpu().numpy()
                confidence_np = confidence.squeeze(0).detach().float().cpu().numpy()
                points_np = np.asarray(points, dtype=np.float32)
                if self.config.use_bundle_adjustment:
                    ba_result = _run_vggt_bundle_adjustment(
                        images=images,
                        depth_confidence=confidence_np,
                        points_3d=points_np,
                        extrinsics_w2c=extrinsics_np,
                        intrinsics=intrinsics_np,
                        output_dir=root,
                    )
            finally:
                del model
                release_gpu_memory()
            metrics.update(stage_metrics)

        command_extent = max(
            float(np.linalg.norm(selected_commands[:, :3, 3] - selected_commands[:, :3, 3].mean(0), axis=1).max()),
            1e-6,
        )
        metrics["vggt_bundle_adjustment_status"] = "disabled"
        if ba_result is not None:
            ba_extrinsics, ba_intrinsics, _, _, ba_metrics = ba_result
            metrics.update(ba_metrics)
            if ba_metrics["vggt_bundle_adjustment_status"] == "complete" and len(ba_extrinsics) == len(indices):
                ba_predicted_c2w = invert_poses(ba_extrinsics)
                ba_similarity = robust_similarity_alignment(
                    ba_predicted_c2w[:, :3, 3],
                    selected_commands[:, :3, 3],
                    residual_threshold=self.config.max_center_residual_ratio * command_extent,
                )
                ba_aligned_c2w = ba_similarity.transform_c2w(ba_predicted_c2w)
                ba_rotation_residuals = rotation_geodesic_degrees(
                    ba_aligned_c2w[:, :3, :3],
                    selected_commands[:, :3, :3],
                )
                reprojection_improved = np.isfinite(float(ba_metrics["vggt_ba_reprojection_after"])) and float(
                    ba_metrics["vggt_ba_reprojection_after"]
                ) <= float(ba_metrics["vggt_ba_reprojection_before"])
                pose_within_bounds = (
                    float(np.median(ba_similarity.residuals)) <= self.config.max_center_residual_ratio * command_extent
                    and float(np.median(ba_rotation_residuals)) <= self.config.max_rotation_residual_deg
                )
                if reprojection_improved and pose_within_bounds:
                    extrinsics_np = ba_extrinsics
                    intrinsics_np = ba_intrinsics
                    points_np = unproject_depth_map_to_point_map(depths_np, extrinsics_np, intrinsics_np)
                    metrics["vggt_ba_accepted"] = True
                else:
                    metrics["vggt_ba_accepted"] = False
            else:
                metrics["vggt_ba_accepted"] = False

        predicted_c2w = invert_poses(extrinsics_np)
        similarity = robust_similarity_alignment(
            predicted_c2w[:, :3, 3],
            selected_commands[:, :3, 3],
            residual_threshold=self.config.max_center_residual_ratio * command_extent,
        )
        aligned_c2w = similarity.transform_c2w(predicted_c2w)
        aligned_points = similarity.transform_points(points_np)
        rotation_residuals = rotation_geodesic_degrees(aligned_c2w[:, :3, :3], selected_commands[:, :3, :3])
        within_command_bounds = (similarity.residuals <= self.config.max_center_residual_ratio * command_extent) & (
            rotation_residuals <= self.config.max_rotation_residual_deg
        )
        finite_views = (
            np.isfinite(predicted_c2w).all(axis=(1, 2))
            & np.isfinite(intrinsics_np).all(axis=(1, 2))
            & np.isfinite(depths_np).all(axis=tuple(range(1, depths_np.ndim)))
        )
        required_views = min(self.config.min_accepted_views, len(indices))
        pose_prior_reliable = int(within_command_bounds.sum()) >= required_views
        accepted = within_command_bounds if pose_prior_reliable else finite_views
        if int(accepted.sum()) < required_views:
            raise RuntimeError(
                f"Only {int(accepted.sum())}/{len(indices)} VGGT views are finite; need {required_views}"
            )

        if images_np.min() < 0:
            images_np = (images_np + 1.0) / 2.0
        colors = np.clip(np.moveaxis(images_np, 1, -1) * 255.0, 0, 255).astype(np.uint8)
        root = Path(output_dir)
        training_images_dir = root / "images"
        training_images_dir.mkdir(parents=True, exist_ok=True)
        training_image_paths: list[Path] = []
        for position, image_array in enumerate(colors):
            image_path = training_images_dir / f"image_{position + 1:04d}.png"
            Image.fromarray(image_array).save(image_path)
            training_image_paths.append(image_path)
        if depths_np.ndim == 4 and depths_np.shape[-1] == 1:
            depths_np = depths_np[..., 0]
        if confidence_np.ndim == 4 and confidence_np.shape[-1] == 1:
            confidence_np = confidence_np[..., 0]
        point_mask = np.isfinite(aligned_points).all(axis=-1) & np.isfinite(depths_np)
        confidence_threshold = float(np.quantile(confidence_np[point_mask], self.config.depth_confidence_quantile))
        point_mask &= confidence_np >= confidence_threshold
        point_mask &= accepted[:, None, None]
        if object_mask is not None:
            height, width = depths_np.shape[-2:]
            mask = _load_vggt_padded_mask(object_mask)
            if mask.shape != (height, width):
                mask = np.asarray(
                    Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST),
                    dtype=bool,
                )
            point_mask &= mask[None]
        flat_points = aligned_points[point_mask]
        flat_colors = colors[point_mask]
        if len(flat_points) > self.config.max_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(flat_points), self.config.max_points, replace=False)
            flat_points = flat_points[keep]
            flat_colors = flat_colors[keep]

        metrics.update(
            {
                "geometry_model": self.config.model_id,
                "geometry_model_revision": self.config.model_revision,
                "accepted_views": int(accepted.sum()),
                "views_within_command_bounds": int(within_command_bounds.sum()),
                "pose_prior_reliable": pose_prior_reliable,
                "pose_gate_fallback": not pose_prior_reliable,
                "sim3_scale": similarity.scale,
                "center_residual_median": float(np.median(similarity.residuals)),
                "center_residual_max": float(similarity.residuals.max()),
                "rotation_residual_deg_median": float(np.median(rotation_residuals)),
                "rotation_residual_deg_max": float(rotation_residuals.max()),
                "predicted_loop_closure_translation": float(
                    np.linalg.norm(aligned_c2w[0, :3, 3] - aligned_c2w[-1, :3, 3])
                ),
                "predicted_loop_closure_rotation_deg": float(
                    rotation_geodesic_degrees(aligned_c2w[0:1, :3, :3], aligned_c2w[-1:, :3, :3])[0]
                ),
                "depth_confidence_threshold": confidence_threshold,
                "initial_point_count": len(flat_points),
            }
        )
        result = VGGTGeometryResult(
            keyframe_indices=indices,
            keyframe_paths=selected_paths,
            training_image_paths=tuple(training_image_paths),
            accepted_mask=accepted,
            commanded_poses_c2w=selected_commands.astype(np.float32),
            predicted_poses_c2w=predicted_c2w.astype(np.float32),
            aligned_poses_c2w=aligned_c2w.astype(np.float32),
            intrinsics=intrinsics_np.astype(np.float32),
            depths=depths_np.astype(np.float32),
            depth_confidence=confidence_np.astype(np.float32),
            points=flat_points.astype(np.float32),
            colors=flat_colors,
            metrics=metrics,
            root=root,
        )
        result.write()
        return result
