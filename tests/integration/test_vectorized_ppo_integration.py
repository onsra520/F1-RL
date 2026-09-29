import numpy as np

from f1rl.learning.training.config import VectorEnvConfig
from f1rl.learning.training.vector_env import make_vector_env
from tests.learning.training.test_vector_env import _factory


def test_real_dummy_vector_env_reset_step_smoke() -> None:
    vector = make_vector_env(_factory(), VectorEnvConfig(n_envs=4, backend="dummy"))
    try:
        observations = vector.reset()
        next_observations, rewards, dones, infos = vector.step(
            np.zeros((4, 2), dtype=np.float32)
        )
        assert observations.shape == (4, 9)
        assert next_observations.shape == (4, 9)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
    finally:
        vector.close()
