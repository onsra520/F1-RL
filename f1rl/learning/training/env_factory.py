from __future__ import annotations

from dataclasses import dataclass

from ...runtime.simulation import SimulationConfig
from ...sandbox.track.model import TrackWorld
from ...sandbox.vehicle import VehicleParameters
from ...sandbox.world import WorldState
from ..episode import EpisodeConfig
from ..gym_environment import F1RLEnv
from ..observation import ObservationEncoder
from ..rewards import RewardComposer


@dataclass(frozen=True, slots=True)
class F1RLEnvFactory:
    track: TrackWorld
    initial_state: WorldState
    vehicle_parameters: VehicleParameters
    simulation_config: SimulationConfig
    observation_encoder: ObservationEncoder
    reward_composer: RewardComposer
    episode_config: EpisodeConfig

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

    def create(self) -> F1RLEnv:
        return F1RLEnv(
            track=self.track,
            initial_state=self.initial_state,
            vehicle_parameters=self.vehicle_parameters,
            simulation_config=self.simulation_config,
            observation_encoder=self.observation_encoder,
            reward_composer=self.reward_composer,
            episode_config=self.episode_config,
        )
