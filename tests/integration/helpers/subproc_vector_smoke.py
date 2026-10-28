import json

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from src.learning.training.config import VectorEnvConfig
from src.learning.training.env_factory import F1RLEnvFactory
from src.learning.training.vector_env import make_vector_env
from src.runtime.simulation import SimulationConfig
from src.sandbox.entities.finish_line import FinishLine
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.state import InteractionState
from src.sandbox.perception.sensor_suite import SensorSuite
from src.sandbox.rules.scoring import ScoreState
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.state import TrackState
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleParameters, VehicleState
from src.sandbox.world import WorldState


def _factory() -> F1RLEnvFactory:
    road = PlanarSurfacePatch(
        "road", Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0), 10.0, 200.0,
    )
    finish = FinishLine(
        "finish",
        OrientedBoxVolume(
            Vec3(0.0, 0.3, 100.0), Vec3(5.0, 5.0, 0.25), Quaternion.identity()
        ),
    )
    track = TrackWorld(surfaces=(road,), finish_lines=(finish,))
    state = WorldState(
        VehicleState(
            Vec3(0.0, 0.3, 0.0), Vec3(0.0, 0.0, 0.0),
            Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    parameters = VehicleParameters(2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81)
    encoder = ObservationEncoder(ObservationConfig(SensorSuite(()), 20.0, 100.0))
    return F1RLEnvFactory(
        track, state, parameters, SimulationConfig(0.1, 0.5),
        encoder, RewardComposer(()), EpisodeConfig(100),
    )


def main() -> None:
    vector = None
    reset_shape: tuple[int, ...] | None = None
    reward_shape: tuple[int, ...] | None = None
    done_shape: tuple[int, ...] | None = None
    closed = False
    try:
        vector = make_vector_env(
            _factory(),
            VectorEnvConfig(n_envs=2, backend="subproc", start_method="spawn"),
        )
        observations = vector.reset()
        _, rewards, dones, _ = vector.step(np.zeros((2, 2), dtype=np.float32))
        reset_shape = observations.shape
        reward_shape = rewards.shape
        done_shape = dones.shape
    finally:
        if vector is not None:
            vector.close()
            closed = True

    print(json.dumps({
        "backend": "subproc",
        "n_envs": 2,
        "reset_shape": list(reset_shape or ()),
        "reward_shape": list(reward_shape or ()),
        "done_shape": list(done_shape or ()),
        "closed": closed,
    }))


if __name__ == "__main__":
    main()
