from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .manifest import STAGES, RunManifest
from .pipeline import Cosmos3GaussianSplatPipeline
from .trajectory import make_closed_helical_trajectory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosmos3-gsplat")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one or more pipeline stages")
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--mask", type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--profile", choices=("test", "full"), default="test")
    run.add_argument("--stages", default="all", help="all or comma-separated generate,geometry,splat,report")
    run.add_argument("--no-resume", action="store_true")

    trajectory = subparsers.add_parser("trajectory", help="write the default trajectory without a GPU")
    trajectory.add_argument("--output-dir", type=Path, required=True)
    trajectory.add_argument("--config", type=Path)

    inspect = subparsers.add_parser("inspect", help="print a run manifest")
    inspect.add_argument("manifest", type=Path)
    return parser


def _load_config(path: Path | None, profile: str | None = None) -> PipelineConfig:
    config = PipelineConfig.read_json(path) if path else PipelineConfig()
    return replace(config, profile=profile) if profile else config


def _parse_stages(value: str):
    if value == "all":
        return STAGES
    stages = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(stages) - set(STAGES)
    if invalid:
        raise SystemExit(f"unknown stages: {', '.join(sorted(invalid))}")
    return stages


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        config = _load_config(args.config, args.profile)
        pipeline = Cosmos3GaussianSplatPipeline.from_pretrained(
            config.generation.model_id,
            geometry_model=config.geometry.model_id,
            config=config,
        )
        result = pipeline(
            prompt=args.prompt,
            image=args.image,
            object_mask=args.mask,
            output_dir=args.output_dir,
            stages=_parse_stages(args.stages),
            resume=not args.no_resume,
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "manifest": str(result.manifest_path),
                    "bucket_uri": result.bucket_uri,
                    "splat": str(result.splat_path) if result.splat_path else None,
                    "report": str(result.report_path) if result.report_path else None,
                },
                indent=2,
            )
        )
    elif args.command == "trajectory":
        config = _load_config(args.config)
        paths = make_closed_helical_trajectory(config.trajectory).write(args.output_dir)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "inspect":
        print(json.dumps(RunManifest.read(args.manifest), default=lambda value: value.__dict__, indent=2))
