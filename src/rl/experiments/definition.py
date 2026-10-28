from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from ...simulation.geometry.vector import Vec3
from ...simulation.perception.rangefinder import RangefinderSpec
from ...simulation.perception.sensor_suite import SensorSuite

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
_V2_SEMANTICS = {
    "algorithm": "ppo",
    "policy": "MlpPolicy",
    "action": "steering-throttle-brake0-v1",
    "observation": "rangefinders-local-route-navigation-v1",
    "reward": "route-highwater-progress-collision-time-offtrack-v1",
    "race": "single-lap-ordered-checkpoints-v1",
    "dynamics": "ground-no-slip-air-ballistic-v1",
    "navigation": "authored-surface-polyline-endpoint-v1",
}
_SEMANTICS_BY_VERSION = {1: _V1_SEMANTICS, 2: _V2_SEMANTICS}
_ROOT_KEYS = {"experiment", "semantics", "environment", "training", "seeds", "tracks", "evaluation"}
_VEHICLE_KEYS = {
    "wheelbase",
    "ride_height",
    "max_steering_angle",
    "max_acceleration",
    "max_braking",
    "rolling_drag",
    "air_drag",
    "max_ground_speed",
    "gravity",
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


def _exact_mapping(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if set(value) != keys:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys, key=str)
        raise ValueError(f"{name} keys must match exactly; missing={missing}, unknown={unknown}")
    return value


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _reference(value: object, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank path string")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _rangefinders(value: object) -> tuple[RangefinderExperimentSpec, ...]:
    result: list[RangefinderExperimentSpec] = []
    for index, item in enumerate(_sequence(value, "environment.observation.rangefinders")):
        mapping = _exact_mapping(
            item,
            {"id", "local_direction", "max_distance"},
            f"environment.observation.rangefinders[{index}]",
        )
        direction = _sequence(
            mapping["local_direction"],
            f"environment.observation.rangefinders[{index}].local_direction",
        )
        if len(direction) != 3:
            raise ValueError("rangefinder local_direction must contain exactly three values")
        result.append(
            RangefinderExperimentSpec(
                mapping["id"],
                tuple(direction),  # type: ignore[arg-type]
                mapping["max_distance"],  # type: ignore[arg-type]
            )
        )
    return tuple(result)


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
    dynamics: str | None = None
    navigation: str | None = None

    def __post_init__(self) -> None:
        # A legacy v1 tuple and the authored-route v2 tuple are indivisible.
        # Do not permit mixing an observation from one version with the reward
        # from another, or quietly defaulting the v2 dynamics/navigation.
        if not any(
            all(getattr(self, name) == value for name, value in expected.items())
            and all(
                getattr(self, name) is None
                for name in ("dynamics", "navigation")
                if name not in expected
            )
            for expected in _SEMANTICS_BY_VERSION.values()
        ):
            raise ValueError("experiment semantics must exactly match one supported protocol version")


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
    route_progress_weight_per_meter: float | None = None
    offtrack_penalty: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "target_progress_weight_per_meter",
            "collision_penalty",
            "time_penalty_per_second",
            "route_progress_weight_per_meter",
            "offtrack_penalty",
        ):
            object.__setattr__(self, name, _require_non_negative_optional(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class OffTrackExperimentConfig:
    lateral_limit_meters: float
    max_consecutive_steps: int

    def __post_init__(self) -> None:
        limit = _require_positive_optional(self.lateral_limit_meters, "offtrack.lateral_limit_meters")
        if limit is None:
            raise ValueError("offtrack.lateral_limit_meters is required")
        object.__setattr__(self, "lateral_limit_meters", limit)
        if (
            type(self.max_consecutive_steps) is not int
            or self.max_consecutive_steps <= 0
        ):
            raise ValueError("offtrack.max_consecutive_steps must be a positive integer")


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
    training_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, Path):
            raise ValueError("manifest must be Path")
        for name, values in (
            ("training_ids", self.training_ids),
            ("heldout_ids", self.heldout_ids),
        ):
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a non-empty tuple")
            if not all(isinstance(track_id, str) and track_id.strip() for track_id in values):
                raise ValueError(f"{name} must contain non-blank string IDs")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} values must be unique")
        if set(self.training_ids) & set(self.heldout_ids):
            raise ValueError("training_ids and heldout_ids must be disjoint")


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
    offtrack: OffTrackExperimentConfig | None = None

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
        version = self.metadata.protocol_version
        if version not in _SEMANTICS_BY_VERSION:
            raise ValueError("unsupported experiment protocol_version")
        required = _SEMANTICS_BY_VERSION[version]
        if any(getattr(self.semantics, key) != value for key, value in required.items()):
            raise ValueError("semantics do not match experiment protocol_version")
        if version == 1:
            if (
                self.offtrack is not None
                or self.reward.route_progress_weight_per_meter is not None
                or self.reward.offtrack_penalty is not None
            ):
                raise ValueError("protocol v1 must not include C2 route/offtrack fields")
        else:
            if (
                not isinstance(self.offtrack, OffTrackExperimentConfig)
                or self.reward.target_progress_weight_per_meter is not None
            ):
                raise ValueError("protocol v2 requires offtrack and forbids legacy target progress")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentDefinition:
        if not isinstance(path, (str, Path)):
            raise ValueError("path must be str or Path")
        config_path = Path(path)
        if not config_path.is_file():
            raise ValueError("experiment definition path must identify an existing file")
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError("failed to load experiment definition YAML") from error

        if not isinstance(document, dict):
            raise ValueError("experiment definition root must be a mapping")
        metadata = _exact_mapping(document.get("experiment"), {"id", "protocol_version", "status"}, "experiment")
        version = metadata["protocol_version"]
        if type(version) is not int or version not in _SEMANTICS_BY_VERSION:
            raise ValueError("unsupported experiment protocol_version")
        expected_root = _ROOT_KEYS | ({"offtrack"} if version == 2 else set())
        root = _exact_mapping(document, expected_root, "root")
        semantics = _exact_mapping(root["semantics"], set(_SEMANTICS_BY_VERSION[version]), "semantics")
        environment = _exact_mapping(
            root["environment"],
            {"vehicle", "simulation", "observation", "reward", "episode"},
            "environment",
        )
        vehicle = _exact_mapping(environment["vehicle"], _VEHICLE_KEYS, "environment.vehicle")
        simulation = _exact_mapping(
            environment["simulation"],
            {"fixed_dt", "max_surface_distance"},
            "environment.simulation",
        )
        observation = _exact_mapping(
            environment["observation"],
            {"velocity_scale", "target_distance_scale", "rangefinders"},
            "environment.observation",
        )
        progress_key = "route_progress" if version == 2 else "target_progress"
        reward = _exact_mapping(
            environment["reward"],
            {progress_key, "collision", "time"} | ({"offtrack"} if version == 2 else set()),
            "environment.reward",
        )
        progress = _exact_mapping(
            reward[progress_key],
            {"weight_per_meter"},
            f"environment.reward.{progress_key}",
        )
        offtrack_reward = (
            _exact_mapping(reward["offtrack"], {"penalty"}, "environment.reward.offtrack")
            if version == 2 else None
        )
        offtrack = (
            _exact_mapping(root["offtrack"], {"lateral_limit_meters", "max_consecutive_steps"}, "offtrack")
            if version == 2 else None
        )
        collision = _exact_mapping(
            reward["collision"],
            {"penalty"},
            "environment.reward.collision",
        )
        time = _exact_mapping(
            reward["time"],
            {"penalty_per_second"},
            "environment.reward.time",
        )
        episode = _exact_mapping(environment["episode"], {"max_steps"}, "environment.episode")
        training = _exact_mapping(root["training"], {"runtime_config", "protocol_status"}, "training")
        seeds = _exact_mapping(root["seeds"], {"ppo", "sampling"}, "seeds")
        tracks = _exact_mapping(
            root["tracks"],
            {"manifest", "training_ids", "heldout_ids"},
            "tracks",
        )
        evaluation = _exact_mapping(root["evaluation"], {"deterministic_policy"}, "evaluation")
        base = config_path.resolve().parent

        return cls(
            metadata=ExperimentMetadata(metadata["id"], metadata["protocol_version"], metadata["status"]),
            semantics=ExperimentSemantics(**semantics),
            vehicle=VehicleExperimentConfig(**vehicle),
            simulation=SimulationExperimentConfig(**simulation),
            observation=ObservationExperimentConfig(
                velocity_scale=observation["velocity_scale"],
                target_distance_scale=observation["target_distance_scale"],
                rangefinders=_rangefinders(observation["rangefinders"]),
            ),
            reward=RewardExperimentConfig(
                target_progress_weight_per_meter=progress["weight_per_meter"] if version == 1 else None,
                collision_penalty=collision["penalty"],
                time_penalty_per_second=time["penalty_per_second"],
                route_progress_weight_per_meter=progress["weight_per_meter"] if version == 2 else None,
                offtrack_penalty=offtrack_reward["penalty"] if offtrack_reward is not None else None,
            ),
            episode=EpisodeExperimentConfig(episode["max_steps"]),
            training=TrainingExperimentConfig(
                _reference(training["runtime_config"], base, "training.runtime_config"),
                training["protocol_status"],
            ),
            seeds=SeedProtocol(
                tuple(_sequence(seeds["ppo"], "seeds.ppo")),
                tuple(_sequence(seeds["sampling"], "seeds.sampling")),
            ),
            tracks=TrackExperimentConfig(
                _reference(tracks["manifest"], base, "tracks.manifest"),
                tuple(_sequence(tracks["training_ids"], "tracks.training_ids")),
                tuple(_sequence(tracks["heldout_ids"], "tracks.heldout_ids")),
            ),
            evaluation=EvaluationExperimentConfig(evaluation["deterministic_policy"]),
            offtrack=OffTrackExperimentConfig(**offtrack) if offtrack is not None else None,
        )

    def require_frozen(self) -> None:
        if self.metadata.status != "frozen":
            raise ValueError("experiment status must be frozen")

        required_scalars = (
            ("vehicle.wheelbase", self.vehicle.wheelbase),
            ("vehicle.ride_height", self.vehicle.ride_height),
            ("vehicle.max_steering_angle", self.vehicle.max_steering_angle),
            ("vehicle.max_acceleration", self.vehicle.max_acceleration),
            ("vehicle.max_braking", self.vehicle.max_braking),
            ("vehicle.rolling_drag", self.vehicle.rolling_drag),
            ("vehicle.air_drag", self.vehicle.air_drag),
            ("vehicle.max_ground_speed", self.vehicle.max_ground_speed),
            ("vehicle.gravity", self.vehicle.gravity),
            ("simulation.fixed_dt", self.simulation.fixed_dt),
            ("simulation.max_surface_distance", self.simulation.max_surface_distance),
            ("observation.velocity_scale", self.observation.velocity_scale),
            ("observation.target_distance_scale", self.observation.target_distance_scale),
            (
                "reward.progress_weight_per_meter",
                self.reward.route_progress_weight_per_meter
                if self.metadata.protocol_version == 2
                else self.reward.target_progress_weight_per_meter,
            ),
            ("reward.collision_penalty", self.reward.collision_penalty),
            ("reward.time_penalty_per_second", self.reward.time_penalty_per_second),
            *(
                (("reward.offtrack_penalty", self.reward.offtrack_penalty),)
                if self.metadata.protocol_version == 2
                else ()
            ),
            ("episode.max_steps", self.episode.max_steps),
        )
        unresolved = [name for name, value in required_scalars if value is None]
        if unresolved:
            raise ValueError(f"frozen experiment has unresolved fields: {unresolved}")

        if not self.observation.rangefinders:
            raise ValueError("frozen experiment requires at least one rangefinder")
        if self.training.protocol_status != "frozen":
            raise ValueError("training protocol must be frozen")
        if self.metadata.protocol_version == 2:
            # This cannot substitute for a reviewed/fingerprinted future
            # release profile, but prevents mechanically promoting the
            # original exploratory WP7 pilot or mutable root runtime config.
            profile = self.training.runtime_config
            if (
                profile.name in {"config.yml", "wp7-pilot-200k.yml"}
                or "recovery" in profile.parts
            ):
                raise ValueError("protocol v2 frozen requires a non-pilot final training profile")
        if len(self.seeds.ppo) != len(self.seeds.sampling) or len(self.seeds.ppo) < 3:
            raise ValueError("frozen experiment requires at least three paired PPO and sampling seeds")
        if len(set(self.seeds.ppo)) != len(self.seeds.ppo):
            raise ValueError("frozen experiment PPO seeds must be unique")
        if len(set(self.seeds.sampling)) != len(self.seeds.sampling):
            raise ValueError("frozen experiment sampling seeds must be unique")
        if not self.training.runtime_config.is_file():
            raise ValueError("frozen experiment runtime_config must identify an existing file")
        if not self.tracks.manifest.is_file():
            raise ValueError("frozen experiment manifest must identify an existing file")

        from .split import resolve_track_split

        resolve_track_split(self.tracks)
        if self.evaluation.deterministic_policy is not True:
            raise ValueError("frozen experiment requires deterministic policy evaluation")
