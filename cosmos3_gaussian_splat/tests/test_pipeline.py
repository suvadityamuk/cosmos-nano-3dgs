from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from cosmos3_gsplat.config import GeometryConfig, PipelineConfig
from cosmos3_gsplat.cosmos_backend import CosmosGenerationResult
from cosmos3_gsplat.manifest import RunManifest
from cosmos3_gsplat.pipeline import Cosmos3GaussianSplatPipeline
from cosmos3_gsplat.splat_trainer import GaussianSplatResult
from cosmos3_gsplat.vggt_backend import VGGTGeometryResult


def test_pipeline_completes_and_resumes_stages(monkeypatch, tmp_path: Path) -> None:
    calls = {"generate": 0, "geometry": 0, "splat": 0}

    def fake_generate(self, *, prompt, image, trajectory, output_dir):
        calls["generate"] += 1
        root = Path(output_dir)
        frames = root / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(len(trajectory.poses_c2w)):
            path = frames / f"frame_{index:04d}.png"
            Image.new("RGB", (32, 32), (index % 255, 100, 150)).save(path)
            paths.append(path)
        video = root / "cosmos_orbit.mp4"
        video.write_bytes(b"video")
        metadata = root / "generation.json"
        metadata.write_text("{}")
        return CosmosGenerationResult(video, frames, tuple(paths), metadata, {"fake_generation": True})

    def fake_reconstruct(self, *, frame_paths, commanded_poses_c2w, output_dir, object_mask=None):
        calls["geometry"] += 1
        selected = np.array([0, len(frame_paths) // 2, len(frame_paths) - 1])
        paths = tuple(Path(frame_paths[index]) for index in selected)
        poses = commanded_poses_c2w[selected]
        intrinsics = np.repeat(
            np.array([[[20, 0, 16], [0, 20, 16], [0, 0, 1]]], dtype=np.float32),
            3,
            axis=0,
        )
        result = VGGTGeometryResult(
            keyframe_indices=selected,
            keyframe_paths=paths,
            training_image_paths=paths,
            accepted_mask=np.ones(3, dtype=bool),
            commanded_poses_c2w=poses,
            predicted_poses_c2w=poses,
            aligned_poses_c2w=poses,
            intrinsics=intrinsics,
            depths=np.ones((3, 32, 32), dtype=np.float32),
            depth_confidence=np.ones((3, 32, 32), dtype=np.float32),
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            metrics={"sim3_scale": 1.0, "depth_confidence_threshold": 0.5, "accepted_views": 3},
            root=Path(output_dir),
        )
        result.write()
        return result

    def fake_train(self, *, geometry, reconstruction, steps, output_dir):
        calls["splat"] += 1
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for name, filename in (
            ("splat", "splat.ply"),
            ("render", "render_orbit.mp4"),
            ("commanded_render", "render_commanded_orbit.mp4"),
            ("comparison", "generated_vs_splat.mp4"),
            ("metrics", "metrics.json"),
        ):
            path = root / filename
            path.write_text("{}" if filename.endswith(".json") else name)
            artifacts[name] = str(path)
        return GaussianSplatResult(artifacts, {"fake_training": True})

    monkeypatch.setattr("cosmos3_gsplat.pipeline.CosmosCameraGenerator.run", fake_generate)
    monkeypatch.setattr("cosmos3_gsplat.pipeline.VGGTBackend.reconstruct", fake_reconstruct)
    monkeypatch.setattr("cosmos3_gsplat.splat_trainer.GaussianSplatTrainer.train", fake_train)
    config = PipelineConfig(
        geometry=replace(GeometryConfig(), num_keyframes=3, min_accepted_views=3, run_colmap_diagnostic=False),
        profile="test",
    )
    pipeline = Cosmos3GaussianSplatPipeline(config)
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "brown").save(reference)
    run_dir = tmp_path / "run"

    output = pipeline(prompt=config.prompt, image=reference, output_dir=run_dir)
    assert output.splat_path is not None
    assert output.report_path is not None
    assert (run_dir / "complete.json").is_file()
    assert all(record.status == "complete" for record in RunManifest.read(output.manifest_path).stages.values())

    pipeline(prompt=config.prompt, image=reference, output_dir=run_dir)
    assert calls == {"generate": 1, "geometry": 1, "splat": 1}
