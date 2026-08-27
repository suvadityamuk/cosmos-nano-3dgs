from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from cosmos3_gsplat.hf_jobs import submit_job


class FakeApi:
    last_instance = None

    def __init__(self) -> None:
        self.run_kwargs = None
        self.bucket_files = []
        FakeApi.last_instance = self

    def whoami(self, token=None):
        return {"name": "tester"}

    def create_bucket(self, **kwargs):
        assert kwargs["private"] is True
        assert kwargs["exist_ok"] is True
        return SimpleNamespace(url="https://huggingface.co/buckets/tester/demo")

    def bucket_info(self, **kwargs):
        return SimpleNamespace(private=True)

    def batch_bucket_files(self, **kwargs):
        self.bucket_files.extend(kwargs["add"])

    def sync_job_volume(self, source, mount_path, **kwargs):
        from huggingface_hub import Volume

        return Volume(
            type="bucket",
            source="tester/jobs-artifacts",
            mount_path=mount_path,
            path=kwargs["remote_name"],
            read_only=kwargs["read_only"],
        )

    def run_job(self, **kwargs):
        self.run_kwargs = kwargs
        return SimpleNamespace(
            id="job-id",
            url="https://huggingface.co/jobs/tester/job-id",
            status=SimpleNamespace(stage="RUNNING"),
        )


def test_submit_job_mounts_explicit_artifact_bucket(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "secret-token")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
    reference = tmp_path / "chair.png"
    Image.new("RGB", (16, 16), "brown").save(reference)

    _, job, metadata = submit_job(
        reference_image=reference,
        prompt="A chair",
        project_dir=project,
        bucket="demo",
        namespace=None,
        run_id="chair-run",
        profile="test",
        docker_image="pytorch:test",
        timeout="1h",
    )

    assert job.id == "job-id"
    assert metadata["artifact_uri"] == "hf://buckets/tester/demo/runs/chair-run"
    api = FakeApi.last_instance
    artifact_volume = api.run_kwargs["volumes"][-1]
    assert artifact_volume.source == "tester/demo"
    assert artifact_volume.mount_path == "/artifacts"
    assert artifact_volume.path == "runs/chair-run"
    assert artifact_volume.read_only is False
    assert api.run_kwargs["flavor"] == "a100-large"
    assert api.run_kwargs["secrets"] == {"HF_TOKEN": "secret-token"}
    assert api.run_kwargs["env"]["HF_ARTIFACT_URI"] == metadata["artifact_uri"]
    assert api.run_kwargs["env"]["COSMOS3_RUN_ID"] == "chair-run"
    assert api.run_kwargs["env"]["TORCH_CUDA_ARCH_LIST"] == "8.0"
    assert "/artifacts" in api.run_kwargs["command"]
    assert (b"", "runs/chair-run/.keep") in api.bucket_files
