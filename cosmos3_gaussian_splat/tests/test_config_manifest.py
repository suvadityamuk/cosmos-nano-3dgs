import json
from pathlib import Path

import pytest

from cosmos3_gsplat.config import PipelineConfig, TrajectoryConfig
from cosmos3_gsplat.manifest import RunLayout, RunManifest


def test_config_round_trip(tmp_path: Path) -> None:
    config = PipelineConfig(trajectory=TrajectoryConfig(radius_m=2.25), profile="test")
    path = config.write_json(tmp_path / "config.json")
    loaded = PipelineConfig.read_json(path)
    assert loaded == config
    assert loaded.training_steps == loaded.splat.test_steps


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrajectoryConfig(radius_m=0)


def test_manifest_records_success_atomically(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    layout.create()
    config = PipelineConfig()
    config.write_json(layout.config)
    manifest = RunManifest.create(str(layout.config), run_id="test-run")
    manifest.write(layout.manifest)

    with manifest.running("generate", layout.manifest) as record:
        artifact = layout.generated / "video.mp4"
        artifact.write_bytes(b"video")
        record.artifacts["video"] = str(artifact)
        record.metrics["frames"] = 61

    loaded = RunManifest.read(layout.manifest)
    assert loaded.is_complete("generate")
    assert loaded.stages["generate"].metrics["frames"] == 61
    assert not list(layout.root.glob(".*.tmp"))


def test_manifest_records_failure(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    layout.create()
    PipelineConfig().write_json(layout.config)
    manifest = RunManifest.create(str(layout.config))

    with pytest.raises(RuntimeError), manifest.running("geometry", layout.manifest):
        raise RuntimeError("bad geometry")

    payload = json.loads(layout.manifest.read_text())
    assert payload["status"] == "failed"
    assert payload["stages"]["geometry"]["status"] == "failed"
    assert "bad geometry" in payload["stages"]["geometry"]["error"]
