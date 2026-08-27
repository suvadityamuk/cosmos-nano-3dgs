from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .config import GeometryConfig
from .cosmos_backend import _materialize_guardrail_nltk_data, _normalise_video_frames
from .telemetry import release_gpu_memory
from .trajectory import actions_to_poses, rot6d_to_rotation_matrix
from .vggt_backend import VGGTBackend

SHALLOW_YAW_ROT6D = np.asarray(
    [
        0.9999904807207345,
        0.0,
        -0.004363309284746571,
        0.0,
        1.0,
        0.0,
    ],
    dtype=np.float32,
)


def _action_metrics(actions: np.ndarray) -> dict[str, object]:
    rotations = rot6d_to_rotation_matrix(actions[:, 3:])
    angles = np.degrees(np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    return {
        "translation_mean_abs": np.abs(actions[:, :3]).mean(axis=0).tolist(),
        "rotation_deg_mean": float(angles.mean()),
        "rotation_deg_max": float(angles.max()),
        "rotation_deg_total": float(angles.sum()),
    }


def run_diagnostics(
    *,
    control_image: Path,
    control_prompt: str,
    control_actions_path: Path,
    chair_image: Path,
    chair_mask: Path | None,
    output_dir: Path,
) -> dict[str, object]:
    import torch
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video, load_image

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_actions = np.asarray(json.loads(control_actions_path.read_text()), dtype=np.float32)
    shallow_actions = reference_actions.copy()
    shallow_actions[:, 3:] = SHALLOW_YAW_ROT6D
    cases = [
        ("known_distribution_control", control_prompt, control_image, reference_actions),
        (
            "chair_reference_envelope_arc",
            "The camera moves smoothly leftward in a gentle shallow arc "
            "while the chair and lighting remain completely still.",
            chair_image,
            shallow_actions,
        ),
    ]
    token = os.environ.get("HF_TOKEN") or None
    pipe = Cosmos3OmniPipeline.from_pretrained(
        "nvidia/Cosmos3-Nano",
        torch_dtype=torch.bfloat16,
        enable_safety_checker=True,
        token=token,
    )
    _materialize_guardrail_nltk_data(token)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=10.0,
        use_karras_sigmas=False,
    )
    pipe.enable_model_cpu_offload()
    results: dict[str, object] = {}
    chair_frame_paths: tuple[Path, ...] = ()
    for name, prompt, image_path, actions in cases:
        case_dir = output_dir / name
        frames_dir = case_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        result = pipe(
            prompt=prompt,
            action=CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=len(actions),
                domain_name="camera_pose",
                resolution_tier=480,
                raw_actions=torch.as_tensor(actions),
                image=load_image(str(image_path)).convert("RGB"),
                view_point="ego_view",
            ),
            fps=30,
            num_inference_steps=30,
            guidance_scale=1.0,
            use_system_prompt=False,
            enable_safety_check=True,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        frames = _normalise_video_frames(result.video)
        frame_paths = []
        for index, frame in enumerate(frames):
            path = frames_dir / f"frame_{index:04d}.png"
            frame.save(path)
            frame_paths.append(path)
        video_path = case_dir / "video.mp4"
        export_to_video(frames, str(video_path), fps=30, macro_block_size=1)
        poses = actions_to_poses(actions)
        np.save(case_dir / "commanded_poses_c2w.npy", poses)
        (case_dir / "actions.json").write_text(json.dumps(actions.tolist()) + "\n")
        results[name] = {
            "video": str(video_path),
            "frames": len(frames),
            "actions": _action_metrics(actions),
        }
        if name == "chair_reference_envelope_arc":
            chair_frame_paths = tuple(frame_paths)
            chair_poses = poses
    pipe.maybe_free_model_hooks()
    del pipe
    release_gpu_memory()

    geometry = VGGTBackend(
        GeometryConfig(
            num_keyframes=8,
            min_accepted_views=3,
            use_bundle_adjustment=False,
            run_colmap_diagnostic=False,
        )
    ).reconstruct(
        frame_paths=chair_frame_paths,
        commanded_poses_c2w=chair_poses,
        output_dir=output_dir / "chair_reference_envelope_arc" / "vggt",
        object_mask=chair_mask,
    )
    results["chair_reference_envelope_arc"]["vggt"] = geometry.metrics
    summary_path = output_dir / "diagnostics.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "complete.json").write_text(json.dumps({"status": "complete"}, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-image", type=Path, required=True)
    parser.add_argument("--control-prompt-file", type=Path, required=True)
    parser.add_argument("--control-actions", type=Path, required=True)
    parser.add_argument("--chair-image", type=Path, required=True)
    parser.add_argument("--chair-mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostics(
        control_image=args.control_image,
        control_prompt=args.control_prompt_file.read_text().strip(),
        control_actions_path=args.control_actions,
        chair_image=args.chair_image,
        chair_mask=args.chair_mask,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
