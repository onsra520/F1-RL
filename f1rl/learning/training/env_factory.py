from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from ...runtime.simulation import SimulationConfig
from ...runtime.physics import PhysicsConfig
from ...sandbox.body_collision import VehicleBodyGeometry
from ...sandbox.ramp_support import PlanarRampPolicy
from ...sandbox.track.model import TrackWorld
from ...sandbox.vehicle import VehicleParameters
from ...sandbox.world import WorldState
from ...track_assets.model import AuthoredNavigation
from ..episode import EpisodeConfig
from ..gym_environment import F1RLEnv
from ..observation import ObservationEncoder
from ..offtrack import OffTrackConfig
from ..rewards import RewardComposer
from ..wp7_race import WP7RaceConfig


@dataclass(frozen=True, slots=True)
class F1RLEnvFactory:
    track: TrackWorld
    initial_state: WorldState
    vehicle_parameters: VehicleParameters
    simulation_config: SimulationConfig
    observation_encoder: ObservationEncoder
    reward_composer: RewardComposer
    episode_config: EpisodeConfig
    navigation: AuthoredNavigation | None = None
    offtrack_config: OffTrackConfig | None = None
    body_geometry: VehicleBodyGeometry | None = None
    ramp_support_policy: PlanarRampPolicy | None = None
    physics_config: PhysicsConfig | None = None
    race_config: WP7RaceConfig | None = None

    def __post_init__(self) -> None:
        values = (
            (self.track, TrackWorld, "track"),
            (self.initial_state, WorldState, "initial_state"),
            (self.vehicle_parameters, VehicleParameters, "vehicle_parameters"),
            (self.simulation_config, SimulationConfig, "simulation_config"),
            (self.observation_encoder, ObservationEncoder, "observation_encoder"),
            (self.reward_composer, RewardComposer, "reward_composer"),
            (self.episode_config, EpisodeConfig, "episode_config"),
        )
        for value, expected_type, name in values:
            if not isinstance(value, expected_type):
                raise ValueError(f"{name} must be {expected_type.__name__}")
        if self.navigation is not None and not isinstance(self.navigation, AuthoredNavigation):
            raise ValueError("navigation must be AuthoredNavigation or None")
        if self.offtrack_config is not None and not isinstance(self.offtrack_config, OffTrackConfig):
            raise ValueError("offtrack_config must be OffTrackConfig or None")
        if self.body_geometry is not None and not isinstance(self.body_geometry, VehicleBodyGeometry):
            raise ValueError("body_geometry must be VehicleBodyGeometry or None")
        if self.ramp_support_policy is not None and (self.body_geometry is None or not isinstance(self.ramp_support_policy, PlanarRampPolicy)):
            raise ValueError('ramp_support_policy requires body geometry and PlanarRampPolicy')
        if self.physics_config is not None:
            if not isinstance(self.physics_config, PhysicsConfig):
                raise ValueError("physics_config must be PhysicsConfig or None")
            if self.body_geometry is not None or self.ramp_support_policy is not None:
                raise ValueError("physics_config cannot be mixed with legacy body/ramp options")
            self.physics_config.validate_state(self.initial_state, self.vehicle_parameters)
        if self.race_config is not None:
            if not isinstance(self.race_config, WP7RaceConfig):
                raise ValueError("race_config must be WP7RaceConfig or None")
            if self.physics_config is None or self.physics_config.mode not in (
                "wp7_suspension_v1", "wp7_grounded_yaw_v1",
            ) or self.navigation is None or self.offtrack_config is not None:
                raise ValueError("WP7 race requires grounded physics/navigation without C2 offtrack")

    @property
    def requires_scientific_admission(self) -> bool:
        return self.body_geometry is not None or (
            self.physics_config is not None and self.physics_config.mode != "legacy_v1"
        )

    def create(self) -> F1RLEnv:
        return F1RLEnv(
            track=self.track,
            initial_state=self.initial_state,
            vehicle_parameters=self.vehicle_parameters,
            simulation_config=self.simulation_config,
            observation_encoder=self.observation_encoder,
            reward_composer=self.reward_composer,
            episode_config=self.episode_config,
            navigation=self.navigation,
            offtrack_config=self.offtrack_config,
            body_geometry=self.body_geometry,
            ramp_support_policy=self.ramp_support_policy,
            physics_config=self.physics_config,
            race_config=self.race_config,
        )


@dataclass(frozen=True, slots=True)
class TrainingScenario:
    track_id: str
    factory: F1RLEnvFactory

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.factory, F1RLEnvFactory):
            raise ValueError("factory must be F1RLEnvFactory")


def _spaces_compatible(first: gym.Space, second: gym.Space) -> bool:
    if type(first) is not type(second):
        return False
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if isinstance(first, gym.spaces.Box):
        return (
            isinstance(second, gym.spaces.Box)
            and np.array_equal(first.low, second.low)
            and np.array_equal(first.high, second.high)
        )
    return first == second


@dataclass(frozen=True, slots=True)
class TrainingScenarioSet:
    scenarios: tuple[TrainingScenario, ...]
    _action_space: gym.Space = field(init=False, repr=False, compare=False)
    _observation_space: gym.Space = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scenarios, tuple) or not self.scenarios:
            raise ValueError("scenarios must be a non-empty tuple")
        if not all(isinstance(scenario, TrainingScenario) for scenario in self.scenarios):
            raise ValueError("scenarios must contain TrainingScenario values")
        track_ids = tuple(scenario.track_id for scenario in self.scenarios)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("scenario track_id values must be unique")

        reference = self.scenarios[0].factory
        shared_fields = (
            "vehicle_parameters",
            "simulation_config",
            "observation_encoder",
            "reward_composer",
            "episode_config",
            "offtrack_config",
            "body_geometry",
            "ramp_support_policy",
            "physics_config",
            "race_config",
        )
        for scenario in self.scenarios[1:]:
            if any(
                getattr(scenario.factory, name) != getattr(reference, name)
                for name in shared_fields
            ):
                raise ValueError("scenario factories must share the non-track environment contract")

        canonical_action_space: gym.Space | None = None
        canonical_observation_space: gym.Space | None = None
        for scenario in self.scenarios:
            env = scenario.factory.create()
            try:
                action_space = env.action_space
                observation_space = env.observation_space
                if not isinstance(action_space, gym.Space) or not isinstance(observation_space, gym.Space):
                    raise ValueError("scenario environments must expose Gym spaces")
                if canonical_action_space is None:
                    canonical_action_space = action_space
                    canonical_observation_space = observation_space
                elif not _spaces_compatible(canonical_action_space, action_space) or not _spaces_compatible(
                    canonical_observation_space, observation_space
                ):
                    raise ValueError("scenario environments must expose compatible Gym spaces")
            finally:
                env.close()

        assert canonical_action_space is not None
        assert canonical_observation_space is not None
        object.__setattr__(self, "_action_space", canonical_action_space)
        object.__setattr__(self, "_observation_space", canonical_observation_space)

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space
