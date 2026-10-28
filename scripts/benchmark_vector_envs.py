from __future__ import annotations

import argparse
import json
from time import perf_counter

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.rl import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from src.training.config import VectorEnvConfig
from src.training.env_factory import F1RLEnvFactory
from src.training.vector_env import make_vector_env
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
        encoder, RewardComposer(()), EpisodeConfig(1000),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark F1-RL vector environment throughput")
    parser.add_argument("--backend", choices=("dummy", "subproc"), required=True)
    parser.add_argument("--n-envs", type=_positive_int, nargs="+", required=True)
    parser.add_argument("--transitions", type=_positive_int, default=1024)
    return parser


def _measure(backend: str, n_envs: int, transition_budget: int) -> dict[str, object]:
    vector = None
    transitions = 0
    start = perf_counter()
    try:
        vector = make_vector_env(
            _factory(),
            VectorEnvConfig(n_envs=n_envs, backend=backend, start_method="spawn"),
        )
        vector.reset()
        actions = np.zeros((n_envs, 2), dtype=np.float32)
        start = perf_counter()
        while transitions < transition_budget:
            vector.step(actions)
            transitions += n_envs
        elapsed = perf_counter() - start
    finally:
        if vector is not None:
            vector.close()
    return {
        "backend": backend,
        "n_envs": n_envs,
        "transitions": transitions,
        "elapsed_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
    }


def main() -> None:
    args = _parser().parse_args()
    for n_envs in args.n_envs:
        result = _measure(args.backend, n_envs, args.transitions)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
