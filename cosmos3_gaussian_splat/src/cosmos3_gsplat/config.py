from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryConfig:
    """Configuration for the closed object-centric camera trajectory."""

    radius_m: float = 1.0
    elevation_amplitude_deg: float = 12.0
    turns: float = 1.0
    num_actions: int = 60
    target_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.turns <= 0:
            raise ValueError("turns must be positive")
        if self.num_actions < 2:
            raise ValueError("num_actions must be at least 2")
        if not 0 <= self.elevation_amplitude_deg < 89:
            raise ValueError("elevation_amplitude_deg must be in [0, 89)")


@dataclass(frozen=True)
class GenerationConfig:
    model_id: str = "nvidia/Cosmos3-Nano"
    fps: int = 30
    resolution_tier: int = 480
    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    flow_shift: float = 10.0
    seed: int = 0
    enable_guardrails: bool = True
    cpu_offload: bool = True

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.resolution_tier not in (256, 480, 704, 720):
            raise ValueError("resolution_tier must be one of 256, 480, 704, 720")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")


@dataclass(frozen=True)
class GeometryConfig:
    model_id: str = "facebook/VGGT-1B"
    model_revision: str = "860abec7937da0a4c03c41d3c269c366e82abdf9"
    num_keyframes: int = 32
    min_accepted_views: int = 16
    blur_threshold: float = 20.0
    depth_confidence_quantile: float = 0.35
    max_points: int = 250_000
    max_center_residual_ratio: float = 0.35
    max_rotation_residual_deg: float = 35.0
    use_bundle_adjustment: bool = True
    run_colmap_diagnostic: bool = True

    def __post_init__(self) -> None:
        if self.num_keyframes < 3:
            raise ValueError("num_keyframes must be at least 3")
        if not 3 <= self.min_accepted_views <= self.num_keyframes:
            raise ValueError("min_accepted_views must be between 3 and num_keyframes")
        if not 0 <= self.depth_confidence_quantile < 1:
            raise ValueError("depth_confidence_quantile must be in [0, 1)")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")


@dataclass(frozen=True)
class SplatConfig:
    test_steps: int = 100
    full_steps: int = 7_000
    holdout_stride: int = 8
    initial_opacity: float = 0.1
    pose_learning_rate: float = 1e-5
    pose_prior_weight: float = 1e-3
    focal_learning_rate: float = 1e-5
    focal_prior_weight: float = 1e-4
    focal_update_every: int = 50
    focal_finite_difference_epsilon: float = 0.01
    depth_loss_weight: float = 0.05
    ssim_weight: float = 0.2
    early_stop_patience: int = 800

    def __post_init__(self) -> None:
        if self.test_steps <= 0 or self.full_steps <= 0:
            raise ValueError("training steps must be positive")
        if self.holdout_stride < 2:
            raise ValueError("holdout_stride must be at least 2")
        if not 0 < self.initial_opacity < 1:
            raise ValueError("initial_opacity must be in (0, 1)")
        if not 0 <= self.ssim_weight <= 1:
            raise ValueError("ssim_weight must be in [0, 1]")
        if self.focal_update_every <= 0 or self.focal_finite_difference_epsilon <= 0:
            raise ValueError("focal finite-difference settings must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    splat: SplatConfig = field(default_factory=SplatConfig)
    prompt: str = "A stationary chair. The camera orbits the chair while the chair remains completely still."
    profile: str = "full"

    def __post_init__(self) -> None:
        if self.profile not in ("test", "full"):
            raise ValueError("profile must be 'test' or 'full'")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")

    @property
    def training_steps(self) -> int:
        return self.splat.test_steps if self.profile == "test" else self.splat.full_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return destination

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        trajectory_data = dict(data.get("trajectory", {}))
        if "target_xyz" in trajectory_data:
            trajectory_data["target_xyz"] = tuple(trajectory_data["target_xyz"])
        return cls(
            trajectory=TrajectoryConfig(**trajectory_data),
            generation=GenerationConfig(**data.get("generation", {})),
            geometry=GeometryConfig(**data.get("geometry", {})),
            splat=SplatConfig(**data.get("splat", {})),
            prompt=data.get("prompt", cls.prompt),
            profile=data.get("profile", "full"),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> PipelineConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
