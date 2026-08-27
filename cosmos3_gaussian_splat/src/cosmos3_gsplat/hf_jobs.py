from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex

DEFAULT_IMAGE = "nvcr.io/nvidia/pytorch:25.06-py3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosmos3-gsplat-submit")
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bucket", default="cosmos3-gsplat-artifacts")
    parser.add_argument("--namespace")
    parser.add_argument("--run-id")
    parser.add_argument("--profile", choices=("test", "full"), default="test")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for the HF Job")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--download-dir", type=Path)
    return parser


def submit_job(
    *,
    reference_image: Path,
    prompt: str,
    project_dir: Path,
    bucket: str,
    namespace: str | None,
    run_id: str | None,
    profile: str,
    docker_image: str,
    timeout: str,
    mask: Path | None = None,
):
    from huggingface_hub import HfApi, Volume, get_token

    token = get_token()
    if not token:
        raise RuntimeError("Authenticate locally with `hf auth login` before submitting a Job")
    api = HfApi()
    owner = namespace or api.whoami(token=token)["name"]
    bucket_id = bucket if "/" in bucket else f"{owner}/{bucket}"
    bucket_url = api.create_bucket(bucket_id=bucket_id, private=True, exist_ok=True, token=token)
    if not api.bucket_info(bucket_id=bucket_id, token=token).private:
        raise RuntimeError(f"Artifact bucket {bucket_id!r} exists but is public; refusing to upload run data")
    resolved_run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{token_hex(4)}"
    run_prefix = f"runs/{resolved_run_id}"
    api.batch_bucket_files(bucket_id=bucket_id, add=[(b"", f"{run_prefix}/.keep")], token=token)

    project_dir = project_dir.resolve()
    if not (project_dir / "pyproject.toml").is_file():
        raise FileNotFoundError(f"pyproject.toml not found under project directory: {project_dir}")
    with tempfile.TemporaryDirectory(prefix="cosmos3-gsplat-input-") as temporary:
        staging = Path(temporary)
        staged_project = staging / "project"
        shutil.copytree(
            project_dir,
            staged_project,
            ignore=shutil.ignore_patterns(
                ".venv",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                "*.pyc",
                "outputs",
                "artifacts",
            ),
        )
        source_volume = api.sync_job_volume(
            staged_project,
            "/workspace",
            remote_name="cosmos3-gaussian-splat-source",
            read_only=True,
            namespace=owner,
            token=token,
        )
        staged_input = staging / "input"
        staged_input.mkdir()
        from PIL import Image

        Image.open(reference_image).convert("RGB").save(staged_input / "reference.png")
        (staged_input / "prompt.txt").write_text(prompt)
        if mask:
            Image.open(mask).convert("L").save(staged_input / "mask.png")
        input_volume = api.sync_job_volume(
            staged_input,
            "/inputs",
            remote_name=f"cosmos3-gsplat-input-{resolved_run_id}",
            read_only=True,
            namespace=owner,
            token=token,
        )
    artifact_volume = Volume(
        type="bucket",
        source=bucket_id,
        mount_path="/artifacts",
        path=run_prefix,
        read_only=False,
    )
    artifact_uri = f"hf://buckets/{bucket_id}/{run_prefix}"
    command = [
        "bash",
        "/workspace/jobs/entrypoint.sh",
        "--image",
        "/inputs/reference.png",
        "--prompt-file",
        "/inputs/prompt.txt",
        "--output-dir",
        "/artifacts",
        "--profile",
        profile,
    ]
    if mask:
        command.extend(["--mask", "/inputs/mask.png"])
    job = api.run_job(
        image=docker_image,
        command=command,
        flavor="a100-large",
        timeout=timeout,
        name=f"cosmos3-gsplat-{resolved_run_id}",
        labels={"project": "cosmos3-gsplat", "profile": profile, "run-id": resolved_run_id},
        env={
            "HF_ARTIFACT_URI": artifact_uri,
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "COSMOS3_RUN_ID": resolved_run_id,
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "MAX_JOBS": "12",
            "BUILD_EXPERIMENTAL": "0",
        },
        secrets={"HF_TOKEN": token},
        volumes=[source_volume, input_volume, artifact_volume],
        namespace=owner,
        token=token,
    )
    return (
        api,
        job,
        {
            "run_id": resolved_run_id,
            "bucket_id": bucket_id,
            "bucket_url": bucket_url.url,
            "artifact_uri": artifact_uri,
            "namespace": owner,
        },
    )


def main() -> None:
    args = _parser().parse_args()
    api, job, metadata = submit_job(
        reference_image=args.reference_image,
        prompt=args.prompt,
        project_dir=args.project_dir,
        bucket=args.bucket,
        namespace=args.namespace,
        run_id=args.run_id,
        profile=args.profile,
        docker_image=args.image,
        timeout=args.timeout,
        mask=args.mask,
    )
    payload = {**metadata, "job_id": job.id, "job_url": job.url, "status": job.status.stage}
    print(json.dumps(payload, indent=2))
    if args.wait:
        final = api.wait_for_job(job.id, timeout=None, namespace=metadata["namespace"])
        for line in api.fetch_job_logs(job_id=job.id, namespace=metadata["namespace"], tail=200):
            print(line, end="" if line.endswith("\n") else "\n")
        payload["status"] = final.status.stage
        print(json.dumps(payload, indent=2))
        if final.status.stage != "COMPLETED":
            raise SystemExit(f"HF Job ended with status {final.status.stage}")
        completion = list(
            api.get_bucket_paths_info(
                bucket_id=metadata["bucket_id"],
                paths=[f"runs/{metadata['run_id']}/complete.json"],
            )
        )
        if not completion:
            raise SystemExit("HF Job completed but the artifact bucket has no complete.json")
        if args.download_dir:
            args.download_dir.mkdir(parents=True, exist_ok=True)
            api.sync_bucket(metadata["artifact_uri"], str(args.download_dir))
