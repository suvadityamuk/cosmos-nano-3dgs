from pathlib import Path

import numpy as np

from cosmos3_gsplat.config import PipelineConfig
from cosmos3_gsplat.manifest import RunLayout, RunManifest
from cosmos3_gsplat.report import build_report
from cosmos3_gsplat.splat_trainer import _initial_log_scales
from tests.test_reconstruction import _synthetic_geometry


def test_initial_scales_are_finite_for_duplicate_points() -> None:
    points = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    scales = _initial_log_scales(points)
    assert scales.shape == (4, 3)
    assert np.isfinite(scales).all()


def test_report_uses_relative_artifact_links(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    layout.create()
    PipelineConfig().write_json(layout.config)
    manifest = RunManifest.create(str(layout.config), run_id="report-test")
    manifest.stages["generate"].metrics["frames"] = 61
    manifest.write(layout.manifest)
    geometry = _synthetic_geometry(tmp_path)
    target = layout.geometry / "vggt"
    target.mkdir(parents=True)
    for source in geometry.root.iterdir():
        if source.is_file():
            (target / source.name).write_bytes(source.read_bytes())
    (layout.generated / "cosmos_orbit.mp4").write_bytes(b"video")
    (layout.splat / "render_orbit.mp4").write_bytes(b"video")
    (layout.splat / "render_commanded_orbit.mp4").write_bytes(b"video")
    (layout.splat / "generated_vs_splat.mp4").write_bytes(b"video")
    (layout.splat / "splat.ply").write_bytes(b"ply")

    result = build_report(layout)
    html = Path(result.artifacts["report"]).read_text()
    assert "../generated/cosmos_orbit.mp4" in html
    assert "../splat/render_commanded_orbit.mp4" in html
    assert "generate.frames" in html
