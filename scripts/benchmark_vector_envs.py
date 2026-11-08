"""Bounded vector-env throughput diagnostic; NOT PPO training or scientific evidence."""
from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter

# The script is launched as a file, so its repository root is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from f1rl.learning.training import VectorEnvConfig, make_vector_env
from tests.learning.training.test_vector_env import _factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy-only vector performance diagnostic")
    parser.add_argument("--backend", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=4)
    args = parser.parse_args()
    if not (1 <= args.n_envs <= 16 and 1 <= args.transitions <= 10000):
        parser.error("diagnostic limits: 1-16 environments and 1-10000 transitions")
    env = make_vector_env(
        _factory(), VectorEnvConfig(
            n_envs=args.n_envs, backend=args.backend, start_method="spawn",
        ),
    )
    started = perf_counter()
    count = 0
    try:
        env.reset()
        action_shape = env.action_space.shape
        for _ in range(ceil(args.transitions / args.n_envs)):
            actions = np.zeros((args.n_envs, *action_shape), dtype=np.float32)
            env.step(actions)
            count += args.n_envs
    finally:
        env.close()
    elapsed = max(perf_counter() - started, 1e-9)
    print(json.dumps({
        "schema": "legacy_vector_diagnostic_v1",
        "backend": args.backend, "n_envs": args.n_envs,
        "transitions": count, "elapsed_seconds": elapsed,
        "transitions_per_second": count / elapsed,
        "training_permitted": False, "heldout_permitted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
