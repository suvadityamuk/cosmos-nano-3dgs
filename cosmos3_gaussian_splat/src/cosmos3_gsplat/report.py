from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jinja2 import Template
from PIL import Image, ImageDraw

from .manifest import RunLayout, RunManifest


@dataclass(frozen=True)
class ReportResult:
    artifacts: dict[str, str]
    metrics: dict[str, float | int | str | bool]


def _relative(target: Path, parent: Path) -> str:
    return os.path.relpath(target, parent)


def _trajectory_plot(geometry_path: Path, output_path: Path) -> Path:
    arrays = np.load(geometry_path)
    commanded = arrays["commanded_poses_c2w"][:, :3, 3]
    aligned = arrays["aligned_poses_c2w"][:, :3, 3]
    canvas = Image.new("RGB", (960, 480), "white")
    draw = ImageDraw.Draw(canvas)

    def panel(values_a: np.ndarray, values_b: np.ndarray, x0: int, axes: tuple[int, int], label: str) -> None:
        all_values = np.concatenate((values_a[:, axes], values_b[:, axes]), axis=0)
        minimum = all_values.min(axis=0)
        maximum = all_values.max(axis=0)
        span = np.maximum(maximum - minimum, 1e-6)

        def project(values: np.ndarray) -> list[tuple[float, float]]:
            normalized = (values[:, axes] - minimum) / span
            return [(x0 + 30 + 400 * float(x), 430 - 380 * float(y)) for x, y in normalized]

        draw.rectangle((x0 + 20, 20, x0 + 450, 450), outline="#999999")
        draw.text((x0 + 30, 30), label, fill="black")
        draw.line(project(values_a), fill="#1769aa", width=3)
        draw.line(project(values_b), fill="#d32f2f", width=3)

    panel(commanded, aligned, 0, (0, 1), "Top view (XY)")
    panel(commanded, aligned, 480, (0, 2), "Side view (XZ)")
    draw.line((20, 468, 60, 468), fill="#1769aa", width=3)
    draw.text((66, 458), "commanded", fill="black")
    draw.line((190, 468, 230, 468), fill="#d32f2f", width=3)
    draw.text((236, 458), "VGGT aligned", fill="black")
    canvas.save(output_path)
    return output_path


def build_report(layout: RunLayout) -> ReportResult:
    layout.report.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest.read(layout.manifest)
    metrics: dict[str, float | int | str | bool] = {}
    for stage, record in manifest.stages.items():
        for name, value in record.metrics.items():
            metrics[f"{stage}.{name}"] = value
    metrics_path = layout.report / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    trajectory_path = _trajectory_plot(
        layout.geometry / "vggt" / "vggt_geometry.npz",
        layout.report / "trajectories.png",
    )
    generated_video = layout.generated / "cosmos_orbit.mp4"
    render_video = layout.splat / "render_orbit.mp4"
    commanded_render_video = layout.splat / "render_commanded_orbit.mp4"
    comparison_video = layout.splat / "generated_vs_splat.mp4"
    splat_path = layout.splat / "gaussian_splat.ply"
    compact_splat_path = layout.splat / "gaussian_splat.splat"
    html = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cosmos 3 Gaussian Splat — {{ run_id }}</title>
  <style>
    body { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; font-family: system-ui, sans-serif; color: #202124; }
    video, img { width: 100%; border-radius: 8px; background: #111; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
    table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
    td { border-bottom: 1px solid #ddd; padding: 0.35rem; vertical-align: top; }
    td:first-child { font-family: ui-monospace, monospace; width: 55%; }
  </style>
</head>
<body>
  <h1>Cosmos 3 Gaussian Splat MVP</h1>
  <p>Run <code>{{ run_id }}</code>. Blue cameras are commanded; red cameras are VGGT-aligned.</p>
  <img src="{{ trajectory }}" alt="Commanded and VGGT-aligned camera trajectories">
  <div class="grid">
    <section><h2>Cosmos generated orbit</h2><video controls loop src="{{ generated }}"></video></section>
    <section><h2>Splat render at refined cameras</h2><video controls loop src="{{ render }}"></video></section>
    <section>
      <h2>Splat render at commanded cameras</h2>
      <video controls loop src="{{ commanded_render }}"></video>
    </section>
  </div>
  <section><h2>Generated vs. reconstructed</h2><video controls loop src="{{ comparison }}"></video></section>
  <p>
    <a href="{{ splat }}">Download Gaussian splat PLY</a> ·
    <a href="{{ compact_splat }}">Download compact .splat</a>
  </p>
  <h2>Metrics</h2>
  <table>{% for key, value in metrics.items() %}<tr><td>{{ key }}</td><td>{{ value }}</td></tr>{% endfor %}</table>
</body>
</html>
"""
    ).render(
        run_id=manifest.run_id,
        trajectory=_relative(trajectory_path, layout.report),
        generated=_relative(generated_video, layout.report),
        render=_relative(render_video, layout.report),
        commanded_render=_relative(commanded_render_video, layout.report),
        comparison=_relative(comparison_video, layout.report),
        splat=_relative(splat_path, layout.report),
        compact_splat=_relative(compact_splat_path, layout.report),
        metrics=metrics,
    )
    report_path = layout.report / "index.html"
    report_path.write_text(html)
    return ReportResult(
        artifacts={
            "report": str(report_path),
            "metrics": str(metrics_path),
            "trajectory_plot": str(trajectory_path),
        },
        metrics={"report_metrics": len(metrics)},
    )
