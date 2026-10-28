"""Build only TRAIN scenarios for a non-frozen WP7 reconstruction candidate.

The legacy ExperimentDefinition is *not* silently relabeled as final C2:
its original v1 semantic contract stays draft and this explicit candidate
owns the newly reconstructed route/highwater/offtrack behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from ..core.simulation import SimulationConfig
from ..simulation.geometry.vector import Vec3
from ..simulation.perception.rangefinder import RangefinderSpec
from ..simulation.perception.sensor_suite import SensorSuite
from ..simulation.vehicle import VehicleParameters
from ..tracks import build_initial_world_state, load_track_asset
from .episode import EpisodeConfig
from .experiments import ExperimentDefinition, resolve_track_split
from .observation import ObservationConfig, ObservationEncoder
from .offtrack import OffTrackConfig
from .rewards import CollisionPenalty, OffTrackPenalty, RewardComposer, RouteProgressReward, TimePenalty
from src.training.env_factory import F1RLEnvFactory, TrainingScenario, TrainingScenarioSet


_BEHAVIOR = {
    "dynamics": "ground-no-slip-air-ballistic-v1",
    "navigation": "authored-surface-polyline-endpoint-v1",
    "observation": "rangefinders-local-route-navigation-v1",
    "reward": "route-highwater-progress-collision-time-offtrack-v1",
    "action": "steering-throttle-brake0-v1",
    "race": "single-lap-ordered-checkpoints-v1",
}
_REWARDS = {
    "route_progress_weight_per_meter",
    "collision_penalty",
    "time_penalty_per_second",
    "offtrack_penalty",
}


def _mapping(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} keys must match exactly: {sorted(expected)}")
    return value


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _positive(value: object, name: str) -> float:
    number = _nonnegative(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return number


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    status: str
    behavior: dict[str, str]
    experiment: ExperimentDefinition
    route_progress_weight_per_meter: float
    collision_penalty: float
    time_penalty_per_second: float
    offtrack_penalty: float
    offtrack: OffTrackConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecoveryCandidate:
        source = Path(path).resolve()
        if not source.is_file():
            raise ValueError("candidate config path must be a file")
        try:
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError("failed to read candidate YAML") from error
        data = _mapping(
            document,
            {"schema_version", "status", "base_experiment", "behavior", "rewards", "offtrack"},
            "candidate",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("candidate schema_version must be 1")
        if data["status"] != "candidate":
            raise ValueError("only candidate status is allowed during WP7 recovery")
        behavior = _mapping(data["behavior"], set(_BEHAVIOR), "behavior")
        if behavior != _BEHAVIOR:
            raise ValueError("candidate behavior semantic identities have drifted")
        base = data["base_experiment"]
        if not isinstance(base, str) or not base.strip():
            raise ValueError("base_experiment must be a non-empty path")
        experiment = ExperimentDefinition.from_yaml(source.parent / base)
        if experiment.metadata.status != "draft" or experiment.training.protocol_status != "pilot_owned":
            raise ValueError("recovery requires the original draft, pilot-owned experiment")
        if experiment.seeds.ppo or experiment.seeds.sampling:
            raise ValueError("recovery must not consume frozen final replicate seeds")
        rewards = _mapping(data["rewards"], _REWARDS, "rewards")
        offtrack = _mapping(data["offtrack"], {"lateral_limit_meters", "max_consecutive_steps"}, "offtrack")
        return cls(
            status="candidate",
            behavior=dict(behavior),
            experiment=experiment,
            route_progress_weight_per_meter=_nonnegative(rewards["route_progress_weight_per_meter"], "route_progress_weight_per_meter"),
            collision_penalty=_nonnegative(rewards["collision_penalty"], "collision_penalty"),
            time_penalty_per_second=_nonnegative(rewards["time_penalty_per_second"], "time_penalty_per_second"),
            offtrack_penalty=_nonnegative(rewards["offtrack_penalty"], "offtrack_penalty"),
            offtrack=OffTrackConfig(
                _positive(offtrack["lateral_limit_meters"], "lateral_limit_meters"),
                offtrack["max_consecutive_steps"],
            ),
        )


def _resolved(value: object, name: str) -> Any:
    if value is None:
        raise ValueError(f"unresolved experiment field {name}")
    return value


def build_training_scenarios(candidate: RecoveryCandidate) -> TrainingScenarioSet:
    """Build the four approved TRAIN tracks; no HELD-OUT env or rollout is created."""
    if not isinstance(candidate, RecoveryCandidate) or candidate.status != "candidate":
        raise ValueError("expected an explicitly non-frozen RecoveryCandidate")
    experiment = candidate.experiment
    if experiment.metadata.status != "draft" or experiment.training.protocol_status != "pilot_owned":
        raise ValueError("cannot train a reconstructed candidate from frozen experiment")
    split = resolve_track_split(experiment.tracks)  # Static manifest and split validation only.
    v = experiment.vehicle
    vehicle = VehicleParameters(
        **{
            name: _resolved(getattr(v, name), f"vehicle.{name}")
            for name in (
                "wheelbase", "ride_height", "max_steering_angle", "max_acceleration",
                "max_braking", "rolling_drag", "air_drag", "max_ground_speed", "gravity"
            )
        }
    )
    sim = experiment.simulation
    simulation = SimulationConfig(
        _resolved(sim.fixed_dt, "simulation.fixed_dt"),
        _resolved(sim.max_surface_distance, "simulation.max_surface_distance"),
    )
    obs = experiment.observation
    sensors = SensorSuite(
        tuple(
            RangefinderSpec(spec.id, Vec3(*spec.local_direction), spec.max_distance)
            for spec in obs.rangefinders
        )
    )
    encoder = ObservationEncoder(
        ObservationConfig(
            sensors,
            _resolved(obs.velocity_scale, "observation.velocity_scale"),
            _resolved(obs.target_distance_scale, "observation.target_distance_scale"),
        )
    )
    episode = EpisodeConfig(_resolved(experiment.episode.max_steps, "episode.max_steps"))
    rewards = RewardComposer((
        RouteProgressReward(candidate.route_progress_weight_per_meter),
        CollisionPenalty(candidate.collision_penalty),
        TimePenalty(candidate.time_penalty_per_second),
        OffTrackPenalty(candidate.offtrack_penalty),
    ))
    scenarios: list[TrainingScenario] = []
    for entry in split.training:  # NEVER iterate over split.heldout when creating environments.
        asset = load_track_asset(experiment.tracks.manifest.parent / entry.path)
        if asset.schema_version != 2 or asset.navigation is None:
            raise ValueError(f"TRAIN track {entry.track_id} requires authored navigation v2")
        if (asset.track_id, asset.world_sha256, asset.content_sha256) != (
            entry.track_id, entry.world_sha256, entry.content_sha256
        ):
            raise ValueError(f"TRAIN track fingerprint mismatch for {entry.track_id}")
        factory = F1RLEnvFactory(
            track=asset.world,
            initial_state=build_initial_world_state(
                asset, max_surface_distance=simulation.max_surface_distance
            ),
            vehicle_parameters=vehicle,
            simulation_config=simulation,
            observation_encoder=encoder,
            reward_composer=rewards,
            episode_config=episode,
            navigation=asset.navigation,
            offtrack_config=candidate.offtrack,
        )
        scenarios.append(TrainingScenario(entry.track_id, factory))
    return TrainingScenarioSet(tuple(scenarios))
