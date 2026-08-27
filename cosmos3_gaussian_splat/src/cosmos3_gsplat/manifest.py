from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

StageName = Literal["generate", "geometry", "splat", "report"]
StageStatus = Literal["pending", "running", "complete", "failed"]
STAGES: tuple[StageName, ...] = ("generate", "geometry", "splat", "report")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class StageRecord:
    status: StageStatus = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    updated_at: str
    config_path: str
    status: StageStatus = "pending"
    stages: dict[str, StageRecord] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def create(cls, config_path: str, run_id: str | None = None) -> RunManifest:
        now = utc_now()
        return cls(
            run_id=run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
            created_at=now,
            updated_at=now,
            config_path=config_path,
            stages={stage: StageRecord() for stage in STAGES},
        )

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        raw = json.loads(Path(path).read_text())
        raw["stages"] = {name: StageRecord(**record) for name, record in raw["stages"].items()}
        return cls(**raw)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now()
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        temporary.replace(destination)
        return destination

    def is_complete(self, stage: StageName) -> bool:
        return self.stages[stage].status == "complete"

    def record_artifact(self, stage: StageName, name: str, path: str | Path) -> None:
        self.stages[stage].artifacts[name] = str(path)

    @contextmanager
    def running(self, stage: StageName, manifest_path: str | Path) -> Iterator[StageRecord]:
        record = self.stages[stage]
        record.status = "running"
        record.started_at = utc_now()
        record.completed_at = None
        record.error = None
        self.status = "running"
        started = time.monotonic()
        self.write(manifest_path)
        try:
            yield record
        except Exception as error:
            record.status = "failed"
            record.error = f"{type(error).__name__}: {error}"
            record.completed_at = utc_now()
            record.duration_seconds = time.monotonic() - started
            self.status = "failed"
            self.write(manifest_path)
            raise
        else:
            record.status = "complete"
            record.completed_at = utc_now()
            record.duration_seconds = time.monotonic() - started
            self.status = "complete" if all(self.is_complete(item) for item in STAGES) else "running"
            self.write(manifest_path)


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def completion_manifest(self) -> Path:
        return self.root / "complete.json"

    @property
    def generated(self) -> Path:
        return self.root / "generated"

    @property
    def geometry(self) -> Path:
        return self.root / "geometry"

    @property
    def splat(self) -> Path:
        return self.root / "splat"

    @property
    def report(self) -> Path:
        return self.root / "report"

    def create(self) -> None:
        for path in (self.root, self.generated, self.geometry, self.splat, self.report):
            path.mkdir(parents=True, exist_ok=True)
