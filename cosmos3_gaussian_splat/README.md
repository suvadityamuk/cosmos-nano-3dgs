# Cosmos 3 Gaussian Splat MVP

This standalone research demo turns one reference image into a plausible Gaussian-splat asset:

1. Cosmos 3 generates a camera-controlled, closed orbit from the reference.
2. VGGT estimates the apparent cameras, intrinsics, dense depth, and confidence.
3. The VGGT cameras are Sim(3)-aligned with the commanded camera trajectory.
4. Optional COLMAP SfM records an independent consistency diagnostic.
5. `gsplat` optimizes Gaussians, bounded camera corrections, and one shared focal correction.

This reconstructs a **plausible completion**, not the unknowable ground-truth back/underside of a
single-view object.

## Publish as a standalone repository

This directory is a complete repository root: it includes `pyproject.toml`, `uv.lock`, tests,
GitHub Actions, job launchers, and its own license. It does not import files from the parent Cosmos
checkout. To publish it separately:

```bash
cp -a demos/cosmos3_gaussian_splat /path/to/cosmos3-gaussian-splat
cd /path/to/cosmos3-gaussian-splat
git init
uv sync --frozen --extra test
uv run pytest
```

Use a fine-grained `HF_TOKEN` through your local/CI secret manager; never commit it.

## Public API

```python
from cosmos3_gsplat import Cosmos3GaussianSplatPipeline

pipe = Cosmos3GaussianSplatPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    geometry_model="facebook/VGGT-1B",
)
result = pipe(
    prompt="A stationary wooden chair. The camera orbits while the chair remains still.",
    image="chair.png",
    output_dir="outputs/chair",
)
print(result.splat_path, result.report_path)
```

The same API has a direct script entrypoint:

```bash
uv run python run_demo.py \
  --image chair.png \
  --mask chair_mask.png \
  --prompt "A stationary chair while the camera moves around it." \
  --output-dir outputs/chair \
  --profile test
```

The stages `generate`, `geometry`, `splat`, and `report` are independently resumable. The final
`complete.json` is written only after every stage succeeds.

## Requirements and access

- The full pipeline requires CUDA. The supported job target is one Hugging Face Jobs
  `a100-large` (80 GB VRAM).
- Accept access for `nvidia/Cosmos3-Nano` and
  `nvidia/Cosmos-1.0-Guardrail`, then authenticate locally with `hf auth login`.
- `facebook/VGGT-1B` is non-commercial. This MVP is for research/evaluation.
- Keep the input object centered and mostly visible. An optional foreground mask improves the
  initial point cloud and splat loss.

The Job uses BF16, Cosmos model CPU offload, and strictly sequential model lifetimes:
Cosmos is released before VGGT loads, and VGGT is released before `gsplat` training.

## Local CPU checks

The geometry math, manifests, CLI, and job submission code do not import CUDA packages:

```bash
uv sync --extra test
uv run pytest
uv run ruff check src tests
uv run cosmos3-gsplat trajectory --output-dir /tmp/cosmos3-trajectory
```

Install the `gpu` extra only on a CUDA machine:

```bash
uv sync --extra gpu
```

The project lock routes Linux `torch`/`torchvision` through the CUDA 12.8 PyTorch index used by
the A100 Job target. Diffusers, VGGT source, LightGlue, and stable `gsplat==1.5.3`
are pinned; the public VGGT checkpoint revision is pinned in `GeometryConfig`.
`gsplat` CUDA kernels compile lazily on the Job, so the Docker image must include `nvcc`.

## Submit an A100 Job

The submitter creates/reuses a private HF bucket, syncs source/input files to a read-only job
volume, and mounts the artifact bucket read-write at `/artifacts`.

```bash
uv run cosmos3-gsplat-submit \
  --reference-image ./chair.png \
  --prompt "A stationary wooden chair. The camera orbits while the chair remains completely still." \
  --bucket YOUR_NAMESPACE/cosmos3-gsplat-artifacts \
  --profile test \
  --wait \
  --download-dir ./outputs/test
```

After the test job succeeds, run `--profile full`. The default Docker image is the CUDA 12 NGC
PyTorch image recommended by the Cosmos cookbook, and the timeout is four hours. Override either
with `--image` or `--timeout`.

Each run writes to:

```text
hf://buckets/<namespace>/<bucket>/runs/<run-id>/
├── config.json
├── manifest.json
├── complete.json
├── job.log
├── generated/
│   ├── cosmos_orbit.mp4
│   ├── commanded_poses_c2w.npy
│   ├── camera_actions.json
│   └── frames/
├── geometry/
│   ├── vggt/
│   └── reconstruction/
├── splat/
│   ├── gaussian_splat.ply
│   ├── gaussian_splat.splat
│   ├── splat.ply
│   ├── splat.pt
│   ├── render_orbit.mp4
│   └── generated_vs_splat.mp4
└── report/
    ├── index.html
    ├── metrics.json
    └── trajectories.png
```

Retrieve a run directly:

```bash
hf buckets sync \
  hf://buckets/<namespace>/<bucket>/runs/<run-id> \
  ./outputs/<run-id>
```

Interrupted runs retain their stage manifests and artifacts but have no `complete.json`. Re-submit
with the same run ID to resume.

## Diagnostics

The report includes:

- commanded versus VGGT-aligned camera paths;
- translation, rotation, and loop-closure residuals;
- confidence-approved view and initial point counts;
- classical COLMAP registration/reprojection results when available;
- training/held-out rendering metrics;
- per-stage timing and peak CPU/GPU memory;
- generated orbit, splat orbit, and side-by-side comparison videos.

Classical SfM is intentionally diagnostic rather than blocking: generated views can violate rigid
multi-view assumptions even when VGGT produces a usable dense initialization.

## Camera-control boundary found during validation

The released `camera_pose` example is a far-field fly-through: about 53 m of translation with only
~0.27 degrees of rotation per frame. A 60-action 360-degree object orbit requires ~6 degrees per
frame, well outside that distribution. Runtime controls confirmed:

- NVIDIA's exact lighthouse example produces stable, sustained forward camera motion.
- A chair test preserving the example translations and using an in-range 0.25-degree yaw per frame
  preserves identity but still collapses to a nearly static wobble with no useful parallax.

This is a model capability boundary, not action serialization: action files match the submitted
A100 tensors, integrate back to the requested poses, and use the documented rot6d/backward-framewise
convention. Smoothing or reducing the orbit stabilizes appearance only by removing viewpoint change.
Do not launch a long splat optimization unless a generation test first demonstrates real
wide-baseline parallax.

For a true single-image object orbit, use a multiview-specific generator or a geometry-guided
render/inpaint loop, or fine-tune Cosmos on near-field object-centric trajectories. The two-case
control is implemented by `python -m cosmos3_gsplat.camera_diagnostics` and
`jobs/camera_diagnostics.sh`.
