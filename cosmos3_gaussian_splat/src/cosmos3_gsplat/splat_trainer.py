from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .config import SplatConfig
from .reconstruction import ReconstructionResult
from .telemetry import measure_stage, release_gpu_memory
from .vggt_backend import VGGTGeometryResult

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GaussianSplatResult:
    artifacts: dict[str, str]
    metrics: dict[str, float | int | str | bool]


def _initial_log_scales(points: np.ndarray) -> np.ndarray:
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(4, len(points)))
    if distances.ndim == 1:
        average = np.full(len(points), np.median(distances))
    else:
        average = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1))
    floor = max(float(np.median(average)) * 0.01, 1e-6)
    return np.log(np.clip(average, floor, None))[:, None].repeat(3, axis=1).astype(np.float32)


def _load_images(paths: tuple[Path, ...]) -> np.ndarray:
    arrays = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in paths]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"training images must have one shape, got {shapes}")
    return np.stack(arrays)


def _skew(vectors):
    import torch

    zeros = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors.unbind(-1)
    return torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(*vectors.shape[:-1], 3, 3)


_SSIM_WINDOW_CACHE: dict[tuple, object] = {}


def _l1_loss(predicted, target):
    import torch.nn.functional as functional

    return functional.l1_loss(predicted, target, reduction="none")


def _create_ssim_window(window_size, channels, device, dtype):
    import torch

    positions = torch.arange(window_size, device=device, dtype=torch.float32)
    gaussian = torch.exp(-((positions - window_size // 2) ** 2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian.unsqueeze(1).mm(gaussian.unsqueeze(0)).float()
    return window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous().to(dtype)


def _torch_ssim_map(image1, image2, window, window_size, channels):
    import torch.nn.functional as functional

    padding = window_size // 2
    mean1 = functional.conv2d(image1, window, padding=padding, groups=channels)
    mean2 = functional.conv2d(image2, window, padding=padding, groups=channels)
    mean1_squared = mean1.pow(2)
    mean2_squared = mean2.pow(2)
    mean_product = mean1 * mean2
    variance1 = functional.conv2d(image1 * image1, window, padding=padding, groups=channels) - mean1_squared
    variance2 = functional.conv2d(image2 * image2, window, padding=padding, groups=channels) - mean2_squared
    covariance = functional.conv2d(image1 * image2, window, padding=padding, groups=channels) - mean_product
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mean_product + c1) * (2 * covariance + c2)) / (
        (mean1_squared + mean2_squared + c1) * (variance1 + variance2 + c2)
    )


def _ssim_loss(image1, image2, window_size=11):
    """Numerically identical fallback to gsplat main's ``ssim_loss``."""

    try:
        from fused_ssim import fused_ssim

        if window_size == 11 and not image2.requires_grad and not image1.is_cpu:
            return 1.0 - fused_ssim(image1, image2, padding="valid")
    except (ImportError, NotImplementedError):
        pass
    channels = image1.shape[1]
    key = (window_size, channels, image1.device, image1.dtype)
    if key not in _SSIM_WINDOW_CACHE:
        _SSIM_WINDOW_CACHE[key] = _create_ssim_window(
            window_size,
            channels,
            image1.device,
            image1.dtype,
        )
    window = _SSIM_WINDOW_CACHE[key]
    return 1.0 - _torch_ssim_map(image1, image2, window, window_size, channels).mean()


def _corrected_c2w(base_c2w, pose_delta):
    import torch

    rotation_delta = torch.matrix_exp(_skew(pose_delta[:, 3:]))
    result = torch.eye(4, device=base_c2w.device, dtype=base_c2w.dtype).repeat(len(base_c2w), 1, 1)
    result[:, :3, :3] = rotation_delta @ base_c2w[:, :3, :3]
    result[:, :3, 3] = base_c2w[:, :3, 3] + pose_delta[:, :3]
    return result


class GaussianSplatTrainer:
    def __init__(self, config: SplatConfig) -> None:
        self.config = config

    def train(
        self,
        *,
        geometry: VGGTGeometryResult,
        reconstruction: ReconstructionResult,
        steps: int,
        output_dir: str | Path,
    ) -> GaussianSplatResult:
        try:
            import gsplat
            import torch
            from gsplat import export_splats
            from gsplat.rendering import rasterization
            from gsplat.strategy import DefaultStrategy
        except ImportError as error:
            raise RuntimeError("Gaussian training requires the package's 'gpu' optional dependencies") from error
        if not torch.cuda.is_available():
            raise RuntimeError("Gaussian splat training requires a CUDA GPU")
        has_3dgs = getattr(gsplat, "has_3dgs", None)
        if callable(has_3dgs):
            cuda_kernels_available = bool(has_3dgs())
        else:
            from gsplat.cuda._backend import _C

            cuda_kernels_available = _C is not None
        if not cuda_kernels_available:
            raise RuntimeError("gsplat CUDA kernels are unavailable; use a CUDA devel image with nvcc")

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda")
        rng = np.random.default_rng(0)
        max_initial_points = 20_000 if steps <= self.config.test_steps else 100_000
        points = geometry.points
        colors = geometry.colors.astype(np.float32) / 255.0
        if len(points) > max_initial_points:
            keep = rng.choice(len(points), max_initial_points, replace=False)
            points, colors = points[keep], colors[keep]
        if len(points) < 100:
            raise RuntimeError(f"not enough VGGT points to initialize splats: {len(points)}")
        log_scales = _initial_log_scales(points)
        count = len(points)
        parameters = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.from_numpy(points).to(device)),
                "scales": torch.nn.Parameter(torch.from_numpy(log_scales).to(device)),
                "quats": torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(count, 1)),
                "opacities": torch.nn.Parameter(
                    torch.full(
                        (count,),
                        math.log(self.config.initial_opacity / (1.0 - self.config.initial_opacity)),
                        device=device,
                    )
                ),
                "sh0": torch.nn.Parameter(
                    torch.from_numpy(((colors - 0.5) / SH_C0).astype(np.float32)).to(device)[:, None, :]
                ),
            }
        )
        learning_rates = {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
        }
        optimizers = {
            name: torch.optim.Adam([{"params": parameters[name], "lr": learning_rates[name]}], eps=1e-15)
            for name in parameters
        }
        camera_centers = reconstruction.poses_c2w[:, :3, 3]
        scene_scale = max(
            float(np.linalg.norm(camera_centers - camera_centers.mean(0), axis=1).max()),
            1e-3,
        )
        strategy = DefaultStrategy(
            refine_start_iter=min(500, max(10, steps // 10)),
            refine_stop_iter=steps,
            reset_every=max(300, min(3_000, steps // 2)),
            refine_every=max(25, min(100, steps // 20)),
            verbose=True,
        )
        strategy.check_sanity(parameters, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

        accepted_positions = np.flatnonzero(geometry.accepted_mask)
        images_np = _load_images(reconstruction.image_paths)
        images = torch.from_numpy(images_np).to(device)
        base_c2w = torch.from_numpy(reconstruction.poses_c2w).to(device)
        command_c2w = torch.from_numpy(geometry.commanded_poses_c2w[geometry.accepted_mask]).to(device)
        base_k = torch.from_numpy(reconstruction.intrinsics).to(device)
        depths = torch.from_numpy(geometry.depths[accepted_positions]).to(device)
        confidence = torch.from_numpy(geometry.depth_confidence[accepted_positions]).to(device)
        depth_scale = float(geometry.metrics["sim3_scale"])
        depths = depths * depth_scale
        confidence_threshold = float(geometry.metrics["depth_confidence_threshold"])
        pose_delta = torch.nn.Parameter(torch.zeros((len(images), 6), device=device))
        log_focal = torch.nn.Parameter(torch.zeros((), device=device))
        pose_optimizer = torch.optim.Adam([pose_delta], lr=self.config.pose_learning_rate)
        focal_optimizer = torch.optim.Adam([log_focal], lr=self.config.focal_learning_rate)
        effective_pose_prior_weight = self.config.pose_prior_weight * (
            1.0 if geometry.metrics.get("pose_prior_reliable", False) else 0.01
        )
        holdout = np.arange(len(images)) % self.config.holdout_stride == 0
        train_indices = np.flatnonzero(~holdout)
        if len(train_indices) == 0:
            train_indices = np.arange(len(images))
            holdout[:] = False
        width, height = images_np.shape[2], images_np.shape[1]
        best_loss = float("inf")
        stale_steps = 0
        completed_steps = 0
        metrics: dict[str, float | int | str | bool] = {}
        with measure_stage("gsplat_training") as stage_metrics:
            for step in range(steps):
                image_index = int(train_indices[step % len(train_indices)])
                for optimizer in (*optimizers.values(), pose_optimizer):
                    optimizer.zero_grad(set_to_none=True)
                corrected_c2w = _corrected_c2w(base_c2w, pose_delta)
                focal_scale = torch.exp(log_focal)
                corrected_k = base_k.clone()
                corrected_k[:, 0, 0] = base_k[:, 0, 0] * focal_scale
                corrected_k[:, 1, 1] = base_k[:, 1, 1] * focal_scale
                rendered, alpha, info = rasterization(
                    means=parameters["means"],
                    quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
                    scales=torch.exp(parameters["scales"]),
                    opacities=torch.sigmoid(parameters["opacities"]),
                    colors=parameters["sh0"],
                    viewmats=torch.linalg.inv(corrected_c2w[image_index : image_index + 1]),
                    Ks=corrected_k[image_index : image_index + 1],
                    width=width,
                    height=height,
                    packed=True,
                    absgrad=strategy.absgrad,
                    sh_degree=0,
                    render_mode="RGB+ED",
                )
                strategy.step_pre_backward(parameters, optimizers, strategy_state, step, info)
                predicted_rgb = rendered[..., :3]
                target_rgb = images[image_index : image_index + 1]
                rgb_l1 = _l1_loss(predicted_rgb, target_rgb).mean()
                rgb_ssim = _ssim_loss(
                    predicted_rgb.permute(0, 3, 1, 2),
                    target_rgb.permute(0, 3, 1, 2),
                )
                loss = torch.lerp(rgb_l1, rgb_ssim, self.config.ssim_weight)
                predicted_depth = rendered[..., 3]
                target_depth = depths[image_index]
                depth_mask = confidence[image_index] >= confidence_threshold
                if depth_mask.any() and self.config.depth_loss_weight > 0:
                    valid_predicted = predicted_depth[0][depth_mask]
                    valid_target = target_depth[depth_mask]
                    depth_l1 = torch.mean(torch.abs(valid_predicted - valid_target))
                    loss = loss + self.config.depth_loss_weight * depth_l1 / scene_scale
                center_prior = torch.nn.functional.smooth_l1_loss(corrected_c2w[:, :3, 3], command_c2w[:, :3, 3])
                rotation_prior = torch.mean(torch.square(corrected_c2w[:, :3, :3] - command_c2w[:, :3, :3]))
                loss = loss + effective_pose_prior_weight * (center_prior + rotation_prior)
                loss.backward()
                for optimizer in (*optimizers.values(), pose_optimizer):
                    optimizer.step()
                strategy.step_post_backward(
                    parameters,
                    optimizers,
                    strategy_state,
                    step,
                    info,
                    packed=True,
                )
                if step % self.config.focal_update_every == 0:
                    self._finite_difference_focal_step(
                        log_focal=log_focal,
                        optimizer=focal_optimizer,
                        epsilon=self.config.focal_finite_difference_epsilon,
                        prior_weight=self.config.focal_prior_weight,
                        parameters=parameters,
                        pose_c2w=corrected_c2w[image_index : image_index + 1].detach(),
                        base_k=base_k[image_index : image_index + 1],
                        target=target_rgb,
                        width=width,
                        height=height,
                        rasterization=rasterization,
                    )
                completed_steps = step + 1
                current_loss = float(loss.detach())
                if current_loss < best_loss - 1e-5:
                    best_loss = current_loss
                    stale_steps = 0
                else:
                    stale_steps += 1
                if step > strategy.refine_start_iter and stale_steps >= self.config.early_stop_patience:
                    break
            metrics.update(stage_metrics)

        corrected_c2w = _corrected_c2w(base_c2w, pose_delta).detach()
        corrected_k = base_k.detach().clone()
        corrected_k[:, 0, 0] *= torch.exp(log_focal.detach())
        corrected_k[:, 1, 1] *= torch.exp(log_focal.detach())
        shn = torch.empty((len(parameters["means"]), 0, 3), device=device)
        gaussian_ply_path = root / "gaussian_splat.ply"
        export_splats(
            means=parameters["means"].detach(),
            scales=parameters["scales"].detach(),
            quats=torch.nn.functional.normalize(parameters["quats"].detach(), dim=-1),
            opacities=parameters["opacities"].detach(),
            sh0=parameters["sh0"].detach(),
            shN=shn,
            format="ply",
            save_to=str(gaussian_ply_path),
        )
        legacy_splat_path = root / "splat.ply"
        shutil.copy2(gaussian_ply_path, legacy_splat_path)
        compact_splat_path = root / "gaussian_splat.splat"
        export_splats(
            means=parameters["means"].detach(),
            scales=parameters["scales"].detach(),
            quats=torch.nn.functional.normalize(parameters["quats"].detach(), dim=-1),
            opacities=parameters["opacities"].detach(),
            sh0=parameters["sh0"].detach(),
            shN=shn,
            format="splat",
            save_to=str(compact_splat_path),
        )
        checkpoint_path = root / "splat.pt"
        torch.save(
            {
                "splats": {name: value.detach().cpu() for name, value in parameters.items()},
                "poses_c2w": corrected_c2w.cpu(),
                "intrinsics": corrected_k.cpu(),
                "step": completed_steps,
            },
            checkpoint_path,
        )
        render_path, commanded_render_path, side_by_side_path, evaluation = self._render_outputs(
            parameters=parameters,
            poses_c2w=corrected_c2w,
            commanded_poses_c2w=command_c2w,
            intrinsics=corrected_k,
            images=images,
            holdout=holdout,
            width=width,
            height=height,
            fps=30,
            output_dir=root,
            rasterization=rasterization,
        )
        metrics.update(
            {
                "training_steps_requested": steps,
                "training_steps_completed": completed_steps,
                "early_stopped": completed_steps < steps,
                "best_training_loss": best_loss,
                "initial_splats": count,
                "final_splats": len(parameters["means"]),
                "focal_scale": float(torch.exp(log_focal.detach())),
                "effective_pose_prior_weight": effective_pose_prior_weight,
                **evaluation,
            }
        )
        metrics_path = root / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        release_gpu_memory()
        return GaussianSplatResult(
            artifacts={
                "splat": str(gaussian_ply_path),
                "splat_ply": str(gaussian_ply_path),
                "splat_compact": str(compact_splat_path),
                "splat_legacy_ply": str(legacy_splat_path),
                "checkpoint": str(checkpoint_path),
                "render": str(render_path),
                "commanded_render": str(commanded_render_path),
                "comparison": str(side_by_side_path),
                "metrics": str(metrics_path),
            },
            metrics=metrics,
        )

    @staticmethod
    def _finite_difference_focal_step(
        *,
        log_focal,
        optimizer,
        epsilon: float,
        prior_weight: float,
        parameters,
        pose_c2w,
        base_k,
        target,
        width: int,
        height: int,
        rasterization,
    ) -> None:
        """Optimize shared focal length despite gsplat treating K as non-differentiable."""

        import torch

        losses = []
        with torch.no_grad():
            for offset in (-epsilon, epsilon):
                scale = torch.exp(log_focal.detach() + offset)
                k = base_k.clone()
                k[:, 0, 0] = base_k[:, 0, 0] * scale
                k[:, 1, 1] = base_k[:, 1, 1] * scale
                rendered, _, _ = rasterization(
                    means=parameters["means"],
                    quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
                    scales=torch.exp(parameters["scales"]),
                    opacities=torch.sigmoid(parameters["opacities"]),
                    colors=parameters["sh0"],
                    viewmats=torch.linalg.inv(pose_c2w),
                    Ks=k,
                    width=width,
                    height=height,
                    packed=True,
                    sh_degree=0,
                )
                data_loss = torch.mean(torch.abs(rendered[..., :3] - target))
                losses.append(data_loss + prior_weight * torch.square(log_focal.detach() + offset))
        gradient = (losses[1] - losses[0]) / (2.0 * epsilon)
        optimizer.zero_grad(set_to_none=True)
        log_focal.grad = gradient.reshape_as(log_focal)
        optimizer.step()
        with torch.no_grad():
            log_focal.clamp_(math.log(0.67), math.log(1.5))

    @staticmethod
    def _render_outputs(
        *,
        parameters,
        poses_c2w,
        commanded_poses_c2w,
        intrinsics,
        images,
        holdout: np.ndarray,
        width: int,
        height: int,
        fps: int,
        output_dir: Path,
        rasterization,
    ) -> tuple[Path, Path, Path, dict[str, float]]:
        import torch

        renders: list[np.ndarray] = []
        commanded_renders: list[np.ndarray] = []
        comparisons: list[np.ndarray] = []
        heldout_mse: list[float] = []
        with torch.inference_mode():
            for index in range(len(poses_c2w)):
                rendered, _, _ = rasterization(
                    means=parameters["means"],
                    quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
                    scales=torch.exp(parameters["scales"]),
                    opacities=torch.sigmoid(parameters["opacities"]),
                    colors=parameters["sh0"],
                    viewmats=torch.linalg.inv(poses_c2w[index : index + 1]),
                    Ks=intrinsics[index : index + 1],
                    width=width,
                    height=height,
                    packed=True,
                    sh_degree=0,
                )
                predicted = rendered[0].clamp(0, 1)
                target = images[index].clamp(0, 1)
                if holdout[index]:
                    heldout_mse.append(float(torch.mean(torch.square(predicted - target))))
                predicted_np = (predicted.cpu().numpy() * 255).astype(np.uint8)
                target_np = (target.cpu().numpy() * 255).astype(np.uint8)
                renders.append(predicted_np)
                comparisons.append(np.concatenate((target_np, predicted_np), axis=1))
                commanded, _, _ = rasterization(
                    means=parameters["means"],
                    quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
                    scales=torch.exp(parameters["scales"]),
                    opacities=torch.sigmoid(parameters["opacities"]),
                    colors=parameters["sh0"],
                    viewmats=torch.linalg.inv(commanded_poses_c2w[index : index + 1]),
                    Ks=intrinsics[index : index + 1],
                    width=width,
                    height=height,
                    packed=True,
                    sh_degree=0,
                )
                commanded_renders.append((commanded[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        render_path = output_dir / "render_orbit.mp4"
        commanded_render_path = output_dir / "render_commanded_orbit.mp4"
        comparison_path = output_dir / "generated_vs_splat.mp4"
        iio.imwrite(render_path, np.stack(renders), fps=fps, codec="libx264")
        iio.imwrite(commanded_render_path, np.stack(commanded_renders), fps=fps, codec="libx264")
        iio.imwrite(comparison_path, np.stack(comparisons), fps=fps, codec="libx264")
        mean_mse = float(np.mean(heldout_mse)) if heldout_mse else float("nan")
        psnr = -10.0 * math.log10(max(mean_mse, 1e-12)) if heldout_mse else float("nan")
        return render_path, commanded_render_path, comparison_path, {"heldout_mse": mean_mse, "heldout_psnr": psnr}
