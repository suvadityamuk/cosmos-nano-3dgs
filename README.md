# Cosmos3 to Gaussian Splat

This experimental cookbook composes three systems:

1. Cosmos3-Nano forward dynamics generates video from one reference image and a camera-pose action sequence.
2. VGGT estimates the cameras and dense depth that the generated pixels actually support.
3. `gsplat` initializes and optimizes a Gaussian representation, exporting both
   `gaussian_splat.ply` and compact `gaussian_splat.splat`.

The runnable tutorial is
[`run_cosmos3_to_gaussian_splat.ipynb`](./run_cosmos3_to_gaussian_splat.ipynb).
The implementation lives in the self-contained
[`cosmos3_gaussian_splat`](./cosmos3_gaussian_splat)
folder, which can be copied into a separate GitHub repository without the rest of this checkout.

## Prerequisites

- Linux with an NVIDIA GPU. The validated target is one A100 80 GB through Hugging Face Jobs.
- `uv` and Python 3.11 or 3.12.
- Hugging Face access to `nvidia/Cosmos3-Nano` and `nvidia/Cosmos-1.0-Guardrail`.
- A fine-grained `HF_TOKEN` supplied through environment/secret management.

See the shared [Cosmos3 environment setup](https://github.com/NVIDIA/cosmos/tree/main/README.md) for general backend requirements.

## Run the notebook

From this directory:

```bash
jupyter lab run_cosmos3_to_gaussian_splat.ipynb
```

The notebook separates CPU-safe setup/trajectory inspection from GPU stages. Set
`RUN_GPU_STAGES = True` only in a suitable GPU environment. The corresponding script invocation is:

```bash
cd ./cosmos3_gaussian_splat
uv sync --frozen --extra gpu
uv run python run_demo.py \
  --image /path/to/chair.png \
  --mask /path/to/chair_mask.png \
  --prompt "A stationary chair while the camera moves around it." \
  --output-dir outputs/chair \
  --profile smoke
```

For HF Jobs and private bucket persistence, use `cosmos3-gsplat-submit` as documented in the
standalone [README](.cosmos3_gaussian_splat/README.md).

## Output

Each completed run contains:

- the generated Cosmos video and individual frames;
- commanded, VGGT-estimated, and refined cameras;
- VGGT depth/confidence and initialization point cloud;
- COLMAP-compatible camera/point files;
- `gaussian_splat.ply`, `gaussian_splat.splat`, and a training checkpoint;
- refined/commanded render videos, metrics, and an HTML report.

## Gotchas and limitations

- The current released `camera_pose` model supports general cinematic camera motion, but a close,
geometry-preserving object orbit is not demonstrated. A 60-step, 360-degree video output is outside the rotation distribution of the public example. Reducing the action to an in-distribution 15-degree shallow turn preserved chair identity but still produced essentially no parallax.

- Treat command/VGGT disagreement as a quality problem. Do not run long Gaussian optimization when
generation has not produced genuine wide-baseline views. There can be better outputs observed by finetuning on camera trajectories. 
