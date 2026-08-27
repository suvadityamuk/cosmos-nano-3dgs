from __future__ import annotations

import gc
import resource
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


def cpu_peak_gib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def gpu_metrics() -> dict[str, float | str | bool]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class StageTelemetry:
    name: str
    duration_seconds: float
    cpu_peak_gib: float
    gpu: dict[str, float | str | bool]

    def to_metrics(self) -> dict[str, float | str | bool]:
        metrics: dict[str, float | str | bool] = {
            "stage": self.name,
            "duration_seconds": self.duration_seconds,
            "cpu_peak_gib": self.cpu_peak_gib,
        }
        metrics.update(self.gpu)
        return metrics


@contextmanager
def measure_stage(name: str) -> Iterator[dict[str, float | str | bool]]:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    started = time.monotonic()
    metrics: dict[str, float | str | bool] = {}
    try:
        yield metrics
    finally:
        telemetry = StageTelemetry(
            name=name,
            duration_seconds=time.monotonic() - started,
            cpu_peak_gib=cpu_peak_gib(),
            gpu=gpu_metrics(),
        )
        metrics.update(telemetry.to_metrics())
