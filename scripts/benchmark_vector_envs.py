"""Bounded synthetic legacy vector throughput diagnostic; never trains PPO."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter, process_time
from uuid import uuid4

# The script is launched as a file, so its repository root is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.gym_environment import F1RLEnv
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


def _make_synthetic_legacy_env() -> F1RLEnv:
    road = PlanarSurfacePatch(
        "road", Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0), 10.0, 200.0,
    )
    finish = FinishLine(
        "finish",
        OrientedBoxVolume(
            Vec3(0.0, 0.3, 100.0), Vec3(5.0, 5.0, 0.25), Quaternion.identity(),
        ),
    )
    track = TrackWorld(surfaces=(road,), finish_lines=(finish,))
    initial_state = WorldState(
        VehicleState(
            Vec3(0.0, 0.3, 0.0), Vec3(0.0, 0.0, 0.0), Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0), True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    vehicle_parameters = VehicleParameters(2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81)
    observation_encoder = ObservationEncoder(
        ObservationConfig(SensorSuite(()), 20.0, 100.0),
    )
    return F1RLEnv(
        track, initial_state, vehicle_parameters, SimulationConfig(0.1, 0.5),
        observation_encoder, RewardComposer(()), EpisodeConfig(max_steps=200),
    )


def _make_vector_env(backend: str, n_envs: int) -> VecEnv:
    factories = [_make_synthetic_legacy_env for _ in range(n_envs)]
    if backend == "dummy":
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic legacy-only vector performance diagnostic")
    parser.add_argument("--backend", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=412)
    args = parser.parse_args()
    if not (1 <= args.n_envs <= 16 and 1 <= args.transitions <= 10000):
        parser.error("diagnostic limits: 1-16 environments and 1-10000 transitions")
    if args.transitions % args.n_envs != 0:
        parser.error("transitions must be divisible by n-envs for exact accounting")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be an integer in [0, 2**32 - 1]")

    if "f1rl.learning.training.runner" in sys.modules:
        raise RuntimeError("training runner must not be imported by this diagnostic")
    env: VecEnv | None = None
    digest = hashlib.sha256()
    actions = np.zeros((args.n_envs, 2), dtype=np.float32)
    cpu_started = process_time()
    total_started = perf_counter()
    step_seconds = 0.0
    close_completed = False
    try:
        env = _make_vector_env(args.backend, args.n_envs)
        env.seed(args.seed)
        observations = env.reset()
        digest.update(np.ascontiguousarray(observations, dtype="<f4").tobytes())
        step_started = perf_counter()
        for _ in range(args.transitions // args.n_envs):
            observations, rewards, dones, _ = env.step(actions)
            digest.update(np.ascontiguousarray(observations, dtype="<f4").tobytes())
            digest.update(np.ascontiguousarray(rewards, dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(dones, dtype=np.uint8).tobytes())
        step_seconds = max(perf_counter() - step_started, 1e-9)
    finally:
        if env is not None:
            env.close()
        close_completed = True
    elapsed = max(perf_counter() - total_started, 1e-9)
    controller_cpu_seconds = process_time() - cpu_started
    if not isfinite(elapsed) or not isfinite(controller_cpu_seconds):
        raise RuntimeError("benchmark clock returned a non-finite duration")
    print(json.dumps({
        "schema": "legacy_vector_diagnostic_v2",
        "trace_id": uuid4().hex,
        "backend": args.backend,
        "environment_kind": "synthetic_legacy",
        "start_method": "spawn",
        "n_envs": args.n_envs,
        "seed": args.seed,
        "transitions": args.transitions,
        "vector_steps": args.transitions // args.n_envs,
        "step_seconds": step_seconds,
        "elapsed_seconds": elapsed,
        "controller_cpu_seconds": controller_cpu_seconds,
        "transitions_per_second": args.transitions / step_seconds,
        "end_to_end_transitions_per_second": args.transitions / elapsed,
        "checksum_sha256": digest.hexdigest(),
        "training_permitted": False,
        "heldout_permitted": False,
        "learning_called": False,
        "checkpoint_created": False,
        "close_completed": close_completed,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
