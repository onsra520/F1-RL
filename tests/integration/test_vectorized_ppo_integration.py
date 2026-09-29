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


def test_real_vectorized_ppo_training_save_load_and_continue(tmp_path) -> None:
    from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel

    first_vector = make_vector_env(_factory(), VectorEnvConfig(n_envs=2))
    second_vector = make_vector_env(_factory(), VectorEnvConfig(n_envs=2))
    try:
        config = PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device="cpu")
        model = PPOModel.create(first_vector, config)
        assert model.learn(8) is model
        assert model.num_timesteps >= 8

        archive = model.save(tmp_path / "vector-policy")
        loaded = PPOModel.load(archive, env=second_vector, device="cpu")
        before = loaded.num_timesteps
        assert loaded.learn(8) is loaded
        assert loaded.num_timesteps >= before + 8
    finally:
        first_vector.close()
        second_vector.close()
