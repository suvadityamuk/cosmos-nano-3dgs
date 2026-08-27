#!/usr/bin/env python3
"""Run the standalone Cosmos 3 → VGGT → Gaussian splat pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cosmos3_gsplat import Cosmos3GaussianSplatPipeline, PipelineConfig
from cosmos3_gsplat.manifest import STAGES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--profile", choices=("test", "full"), default="test")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig.read_json(args.config) if args.config else PipelineConfig()
    config = replace(config, profile=args.profile)
    stages = STAGES if args.stages == "all" else tuple(item.strip() for item in args.stages.split(","))
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
        stages=stages,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "manifest": str(result.manifest_path),
                "splat_ply": str(result.splat_path) if result.splat_path else None,
                "report": str(result.report_path) if result.report_path else None,
                "bucket_uri": result.bucket_uri,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
