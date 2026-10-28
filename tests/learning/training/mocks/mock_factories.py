from __future__ import annotations

from src.rl import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from src.training.env_factory import F1RLEnvFactory, TrainingScenario, TrainingScenarioSet
from src.core.simulation import SimulationConfig
from src.simulation.entities.finish_line import FinishLine
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.state import InteractionState
from src.simulation.perception.sensor_suite import SensorSuite
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.model import TrackWorld
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.track.state import TrackState
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.vehicle import VehicleParameters, VehicleState
from src.simulation.world import WorldState


def make_factory(*, finish_z: float = 100.0, max_steps: int = 100) -> F1RLEnvFactory:
    road = PlanarSurfacePatch(
        "road",
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        10.0,
        200.0,
    )
    finish = FinishLine(
        "finish",
        OrientedBoxVolume(
            Vec3(0.0, 0.3, finish_z),
            Vec3(5.0, 5.0, 0.25),
            Quaternion.identity(),
        ),
    )
    track = TrackWorld(surfaces=(road,), finish_lines=(finish,))
    state = WorldState(
        VehicleState(
            Vec3(0.0, 0.3, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    parameters = VehicleParameters(
        2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81
    )
    encoder = ObservationEncoder(
        ObservationConfig(SensorSuite(()), 20.0, 100.0)
    )
    return F1RLEnvFactory(
        track=track,
        initial_state=state,
        vehicle_parameters=parameters,
        simulation_config=SimulationConfig(0.1, 0.5),
        observation_encoder=encoder,
        reward_composer=RewardComposer(()),
        episode_config=EpisodeConfig(max_steps),
    )


def make_scenario_set(*, max_steps: int = 100) -> TrainingScenarioSet:
    return TrainingScenarioSet(
        (
            TrainingScenario("alpha", make_factory(finish_z=100.0, max_steps=max_steps)),
            TrainingScenario("beta", make_factory(finish_z=101.0, max_steps=max_steps)),
            TrainingScenario("gamma", make_factory(finish_z=102.0, max_steps=max_steps)),
        )
    )
