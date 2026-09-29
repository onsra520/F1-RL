import json
from pathlib import Path
import subprocess
import sys

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


def test_guarded_subproc_vector_smoke_script() -> None:
    helper = Path("tests/integration/helpers/subproc_vector_smoke.py")
    completed = subprocess.run(
        [sys.executable, str(helper)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["backend"] == "subproc"
    assert payload["n_envs"] == 2
    assert payload["reset_shape"] == [2, 9]
    assert payload["reward_shape"] == [2]
    assert payload["done_shape"] == [2]
    assert payload["closed"] is True


def _run_benchmark(backend: str, n_envs: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_vector_envs.py",
            "--backend",
            backend,
            "--n-envs",
            str(n_envs),
            "--transitions",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_vector_benchmark_cli_reports_positive_diagnostics_for_dummy() -> None:
    payload = _run_benchmark("dummy", 1)
    assert payload["backend"] == "dummy"
    assert payload["n_envs"] == 1
    assert payload["transitions"] >= 4
    assert payload["elapsed_seconds"] > 0.0
    assert payload["transitions_per_second"] > 0.0


def test_vector_benchmark_cli_reports_positive_diagnostics_for_subproc() -> None:
    payload = _run_benchmark("subproc", 2)
    assert payload["backend"] == "subproc"
    assert payload["n_envs"] == 2
    assert payload["transitions"] >= 4
    assert payload["elapsed_seconds"] > 0.0
    assert payload["transitions_per_second"] > 0.0
