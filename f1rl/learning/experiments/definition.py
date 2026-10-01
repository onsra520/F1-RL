from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real
from pathlib import Path

from ...sandbox.geometry.vector import Vec3
from ...sandbox.perception.rangefinder import RangefinderSpec
from ...sandbox.perception.sensor_suite import SensorSuite

_MAX_SEED = 2**32 - 1
_ALLOWED_STATUSES = {"draft", "frozen"}
_ALLOWED_TRAINING_STATUSES = {"pilot_owned", "frozen"}
_V1_SEMANTICS = {
    "algorithm": "ppo",
    "policy": "MlpPolicy",
    "action": "steering-throttle-brake0-v1",
    "observation": "rangefinders-local-navigation-v1",
    "reward": "target-progress-collision-time-v1",
    "race": "single-lap-ordered-checkpoints-v1",
}


def _optional_real(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number or None")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite real number or None")
    return number


def _require_positive_optional(value: object, name: str) -> float | None:
    number = _optional_real(value, name)
    if number is not None and number <= 0.0:
        raise ValueError(f"{name} must be positive when resolved")
    return number


def _require_non_negative_optional(value: object, name: str) -> float | None:
    number = _optional_real(value, name)
    if number is not None and number < 0.0:
        raise ValueError(f"{name} must be non-negative when resolved")
    return number


def _validate_seed_tuple(values: object, name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
            raise ValueError(f"{name} seeds must be integers in [0, {_MAX_SEED}]")


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    id: str
    protocol_version: int
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("experiment id must be a non-blank string")
        if isinstance(self.protocol_version, bool) or not isinstance(self.protocol_version, int) or self.protocol_version <= 0:
            raise ValueError("protocol_version must be a positive integer")
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError('status must be "draft" or "frozen"')


@dataclass(frozen=True, slots=True)
class ExperimentSemantics:
    algorithm: str
    policy: str
    action: str
    observation: str
    reward: str
    race: str

    def __post_init__(self) -> None:
        for name, expected in _V1_SEMANTICS.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r} for protocol v1")


@dataclass(frozen=True, slots=True)
class VehicleExperimentConfig:
    wheelbase: float | None = None
    ride_height: float | None = None
    max_steering_angle: float | None = None
    max_acceleration: float | None = None
    max_braking: float | None = None
    rolling_drag: float | None = None
    air_drag: float | None = None
    max_ground_speed: float | None = None
    gravity: float | None = None

    def __post_init__(self) -> None:
        for name in ("wheelbase", "ride_height", "gravity"):
            object.__setattr__(self, name, _require_positive_optional(getattr(self, name), name))
        steering = _optional_real(self.max_steering_angle, "max_steering_angle")
        if steering is not None and not 0.0 <= steering < pi / 2.0:
            raise ValueError("max_steering_angle must be in [0, pi/2) when resolved")
        object.__setattr__(self, "max_steering_angle", steering)
        for name in ("max_acceleration", "max_braking", "rolling_drag", "air_drag", "max_ground_speed"):
            object.__setattr__(self, name, _require_non_negative_optional(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class SimulationExperimentConfig:
    fixed_dt: float | None = None
    max_surface_distance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_dt", _require_positive_optional(self.fixed_dt, "fixed_dt"))
        object.__setattr__(
            self,
            "max_surface_distance",
            _require_non_negative_optional(self.max_surface_distance, "max_surface_distance"),
        )


@dataclass(frozen=True, slots=True)
class RangefinderExperimentSpec:
    id: str
    local_direction: tuple[float, float, float]
    max_distance: float

    def __post_init__(self) -> None:
        if not isinstance(self.local_direction, tuple) or len(self.local_direction) != 3:
            raise ValueError("local_direction must be a 3-tuple")
        components: list[float] = []
        for component in self.local_direction:
            if isinstance(component, bool) or not isinstance(component, Real) or not isfinite(float(component)):
                raise ValueError("local_direction components must be finite real numbers")
            components.append(float(component))
        if isinstance(self.max_distance, bool) or not isinstance(self.max_distance, Real):
            raise ValueError("max_distance must be a finite positive number")
        max_distance = float(self.max_distance)
        RangefinderSpec(self.id, Vec3(*components), max_distance)
        object.__setattr__(self, "local_direction", tuple(components))
        object.__setattr__(self, "max_distance", max_distance)

    def _domain_spec(self) -> RangefinderSpec:
        return RangefinderSpec(self.id, Vec3(*self.local_direction), self.max_distance)


@dataclass(frozen=True, slots=True)
class ObservationExperimentConfig:
    velocity_scale: float | None = None
    target_distance_scale: float | None = None
    rangefinders: tuple[RangefinderExperimentSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "velocity_scale",
            _require_positive_optional(self.velocity_scale, "velocity_scale"),
        )
        object.__setattr__(
            self,
            "target_distance_scale",
            _require_positive_optional(self.target_distance_scale, "target_distance_scale"),
        )
        if not isinstance(self.rangefinders, tuple) or not all(
            isinstance(spec, RangefinderExperimentSpec) for spec in self.rangefinders
        ):
            raise ValueError("rangefinders must be a tuple of RangefinderExperimentSpec values")
        SensorSuite(tuple(spec._domain_spec() for spec in self.rangefinders))


@dataclass(frozen=True, slots=True)
class RewardExperimentConfig:
    target_progress_weight_per_meter: float | None = None
    collision_penalty: float | None = None
    time_penalty_per_second: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "target_progress_weight_per_meter",
            "collision_penalty",
            "time_penalty_per_second",
        ):
            object.__setattr__(self, name, _require_non_negative_optional(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class EpisodeExperimentConfig:
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class TrainingExperimentConfig:
    runtime_config: Path
    protocol_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_config, Path):
            raise ValueError("runtime_config must be Path")
        if self.protocol_status not in _ALLOWED_TRAINING_STATUSES:
            raise ValueError('protocol_status must be "pilot_owned" or "frozen"')


@dataclass(frozen=True, slots=True)
class SeedProtocol:
    ppo: tuple[int, ...]
    sampling: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_seed_tuple(self.ppo, "ppo")
        _validate_seed_tuple(self.sampling, "sampling")


@dataclass(frozen=True, slots=True)
class TrackExperimentConfig:
    manifest: Path

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, Path):
            raise ValueError("manifest must be Path")


@dataclass(frozen=True, slots=True)
class EvaluationExperimentConfig:
    deterministic_policy: bool

    def __post_init__(self) -> None:
        if not isinstance(self.deterministic_policy, bool):
            raise ValueError("deterministic_policy must be bool")


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    metadata: ExperimentMetadata
    semantics: ExperimentSemantics
    vehicle: VehicleExperimentConfig
    simulation: SimulationExperimentConfig
    observation: ObservationExperimentConfig
    reward: RewardExperimentConfig
    episode: EpisodeExperimentConfig
    training: TrainingExperimentConfig
    seeds: SeedProtocol
    tracks: TrackExperimentConfig
    evaluation: EvaluationExperimentConfig

    def __post_init__(self) -> None:
        expected = (
            (self.metadata, ExperimentMetadata, "metadata"),
            (self.semantics, ExperimentSemantics, "semantics"),
            (self.vehicle, VehicleExperimentConfig, "vehicle"),
            (self.simulation, SimulationExperimentConfig, "simulation"),
            (self.observation, ObservationExperimentConfig, "observation"),
            (self.reward, RewardExperimentConfig, "reward"),
            (self.episode, EpisodeExperimentConfig, "episode"),
            (self.training, TrainingExperimentConfig, "training"),
            (self.seeds, SeedProtocol, "seeds"),
            (self.tracks, TrackExperimentConfig, "tracks"),
            (self.evaluation, EvaluationExperimentConfig, "evaluation"),
        )
        for value, expected_type, name in expected:
            if not isinstance(value, expected_type):
                raise ValueError(f"{name} must be {expected_type.__name__}")
