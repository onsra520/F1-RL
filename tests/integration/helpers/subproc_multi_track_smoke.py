import json
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from f1rl.learning.training.config import VectorEnvConfig
from f1rl.learning.training.vector_env import make_multi_track_vector_env
from tests.learning.training.test_vector_env import _scenario_set


def main() -> None:
    vector = None
    closed = False
    try:
        vector = make_multi_track_vector_env(
            _scenario_set(),
            VectorEnvConfig(n_envs=2, backend="subproc", start_method="spawn"),
            sampling_seed=123,
        )
        observations = vector.reset()
        next_observations, rewards, dones, infos = vector.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        payload = {
            "backend": "subproc",
            "n_envs": vector.num_envs,
            "reset_shape": list(observations.shape),
            "step_shape": list(next_observations.shape),
            "reward_shape": list(rewards.shape),
            "done_shape": list(dones.shape),
            "track_ids": [info["track_id"] for info in infos],
            "closed": False,
        }
    finally:
        if vector is not None:
            vector.close()
            closed = True
    payload["closed"] = closed
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
