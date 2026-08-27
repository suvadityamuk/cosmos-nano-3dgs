from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from .config import PipelineConfig
from .cosmos_backend import CosmosCameraGenerator
from .manifest import STAGES, RunLayout, RunManifest, StageName, utc_now
from .reconstruction import ReconstructionResult, build_reconstruction
from .trajectory import CameraTrajectory, make_closed_helical_trajectory
from .vggt_backend import VGGTBackend, VGGTGeometryResult


@dataclass(frozen=True)
class Cosmos3GaussianSplatPipelineOutput:
    run_id: str
    run_dir: Path
    manifest_path: Path
    generated_video: Path | None = None
    cameras_path: Path | None = None
    point_cloud_path: Path | None = None
    splat_path: Path | None = None
    render_video: Path | None = None
    metrics_path: Path | None = None
    report_path: Path | None = None
    bucket_uri: str | None = None


class Cosmos3GaussianSplatPipeline:
    """Composable Cosmos 3 → VGGT → gsplat pipeline with Diffusers-style loading."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        geometry_model: str = "facebook/VGGT-1B",
        config: PipelineConfig | None = None,
    ) -> Cosmos3GaussianSplatPipeline:
        base = config or PipelineConfig()
        configured = replace(
            base,
            generation=replace(base.generation, model_id=model_id),
            geometry=replace(base.geometry, model_id=geometry_model),
        )
        return cls(configured)

    def __call__(
        self,
        *,
        prompt: str,
        image: str | Path | Image.Image,
        output_dir: str | Path,
        trajectory: CameraTrajectory | None = None,
        object_mask: str | Path | None = None,
        stages: Iterable[StageName] = STAGES,
        resume: bool = True,
    ) -> Cosmos3GaussianSplatPipelineOutput:
        requested = tuple(stages)
        unknown = set(requested) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
        layout = RunLayout(Path(output_dir))
        layout.create()
        run_config = replace(self.config, prompt=prompt)
        if layout.manifest.exists() and resume:
            manifest = RunManifest.read(layout.manifest)
            stored_config = PipelineConfig.read_json(layout.config)
            if stored_config != run_config:
                raise ValueError("cannot resume with a different configuration or prompt")
        else:
            run_config.write_json(layout.config)
            manifest = RunManifest.create(
                str(layout.config),
                run_id=os.environ.get("COSMOS3_RUN_ID") or layout.root.name,
            )
            manifest.write(layout.manifest)

        camera_trajectory = trajectory or make_closed_helical_trajectory(run_config.trajectory)
        camera_trajectory.write(layout.generated)
        reference_path = self._persist_reference(image, layout.generated / "reference.png")
        mask_path = self._persist_mask(object_mask, layout.generated / "object_mask.png")

        if "generate" in requested and not manifest.is_complete("generate"):
            with manifest.running("generate", layout.manifest) as record:
                result = CosmosCameraGenerator(run_config.generation).run(
                    prompt=prompt,
                    image=reference_path,
                    trajectory=camera_trajectory,
                    output_dir=layout.generated,
                )
                record.artifacts.update(
                    {
                        "video": str(result.video_path),
                        "frames": str(result.frames_dir),
                        "metadata": str(result.metadata_path),
                        "poses": str(layout.generated / "commanded_poses_c2w.npy"),
                        "actions": str(layout.generated / "camera_actions.json"),
                    }
                )
                record.metrics.update(result.metrics)
                record.metrics["frames"] = len(result.frame_paths)

        geometry_result: VGGTGeometryResult | None = None
        reconstruction: ReconstructionResult | None = None
        if "geometry" in requested and not manifest.is_complete("geometry"):
            frame_paths = self._generated_frame_paths(layout)
            geometry_config = run_config.geometry
            if run_config.profile == "test":
                geometry_config = replace(
                    geometry_config,
                    num_keyframes=min(8, len(frame_paths)),
                    min_accepted_views=min(3, len(frame_paths)),
                    run_colmap_diagnostic=False,
                )
            with manifest.running("geometry", layout.manifest) as record:
                geometry_result = VGGTBackend(geometry_config).reconstruct(
                    frame_paths=frame_paths,
                    commanded_poses_c2w=camera_trajectory.poses_c2w,
                    output_dir=layout.geometry / "vggt",
                    object_mask=mask_path,
                )
                reconstruction = build_reconstruction(
                    geometry_result,
                    geometry_config,
                    layout.geometry / "reconstruction",
                )
                record.artifacts.update(
                    {
                        "geometry": str(layout.geometry / "vggt" / "vggt_geometry.npz"),
                        "point_cloud": str(layout.geometry / "vggt" / "vggt_points.ply"),
                        "colmap": str(reconstruction.colmap_dir),
                        "reconstruction": str(reconstruction.root / "reconstruction.json"),
                    }
                )
                record.metrics.update(geometry_result.metrics)
                record.metrics.update(reconstruction.metrics)

        if "splat" in requested and not manifest.is_complete("splat"):
            from .splat_trainer import GaussianSplatTrainer

            geometry_result = geometry_result or VGGTGeometryResult.read(layout.geometry / "vggt")
            reconstruction = reconstruction or ReconstructionResult.read(layout.geometry / "reconstruction")
            with manifest.running("splat", layout.manifest) as record:
                splat_result = GaussianSplatTrainer(run_config.splat).train(
                    geometry=geometry_result,
                    reconstruction=reconstruction,
                    steps=run_config.training_steps,
                    output_dir=layout.splat,
                )
                record.artifacts.update(splat_result.artifacts)
                record.metrics.update(splat_result.metrics)

        if "report" in requested and not manifest.is_complete("report"):
            from .report import build_report

            with manifest.running("report", layout.manifest) as record:
                report_result = build_report(layout)
                record.artifacts.update(report_result.artifacts)
                record.metrics.update(report_result.metrics)

        if all(manifest.is_complete(stage) for stage in STAGES):
            bucket_uri = os.environ.get("HF_ARTIFACT_URI")
            layout.completion_manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "run_id": manifest.run_id,
                        "completed_at": utc_now(),
                        "manifest": str(layout.manifest),
                        "bucket_uri": bucket_uri,
                    },
                    indent=2,
                )
                + "\n"
            )
        return self._output(layout, manifest)

    @staticmethod
    def _persist_reference(image: str | Path | Image.Image, destination: Path) -> Path:
        if isinstance(image, Image.Image):
            image.convert("RGB").save(destination)
        else:
            source = Path(image)
            if not source.is_file():
                raise FileNotFoundError(source)
            Image.open(source).convert("RGB").save(destination)
        return destination

    @staticmethod
    def _persist_mask(mask: str | Path | None, destination: Path) -> Path | None:
        if mask is None:
            return None
        source = Path(mask)
        if not source.is_file():
            raise FileNotFoundError(source)
        Image.open(source).convert("L").save(destination)
        return destination

    @staticmethod
    def _generated_frame_paths(layout: RunLayout) -> tuple[Path, ...]:
        paths = tuple(sorted((layout.generated / "frames").glob("frame_*.png")))
        if not paths:
            raise FileNotFoundError(f"no generated frames under {layout.generated / 'frames'}")
        return paths

    @staticmethod
    def _output(layout: RunLayout, manifest: RunManifest) -> Cosmos3GaussianSplatPipelineOutput:
        def existing(path: Path) -> Path | None:
            return path if path.exists() else None

        return Cosmos3GaussianSplatPipelineOutput(
            run_id=manifest.run_id,
            run_dir=layout.root,
            manifest_path=layout.manifest,
            generated_video=existing(layout.generated / "cosmos_orbit.mp4"),
            cameras_path=existing(layout.geometry / "reconstruction" / "reconstruction.npz"),
            point_cloud_path=existing(layout.geometry / "vggt" / "vggt_points.ply"),
            splat_path=existing(layout.splat / "gaussian_splat.ply") or existing(layout.splat / "splat.ply"),
            render_video=existing(layout.splat / "render_orbit.mp4"),
            metrics_path=existing(layout.report / "metrics.json"),
            report_path=existing(layout.report / "index.html"),
            bucket_uri=os.environ.get("HF_ARTIFACT_URI"),
        )
