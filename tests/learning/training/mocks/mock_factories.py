from __future__ import annotations

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.training.env_factory import F1RLEnvFactory, TrainingScenario, TrainingScenarioSet
from f1rl.runtime.simulation import SimulationConfig
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState


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
