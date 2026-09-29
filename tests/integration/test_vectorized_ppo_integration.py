import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from f1rl.learning.training.config import VectorEnvConfig
from f1rl.learning.training.vector_env import make_vector_env
from tests.learning.training.test_vector_env import _factory, _scenario_set


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


def test_guarded_multi_track_subproc_smoke_script() -> None:
    helper = Path("tests/integration/helpers/subproc_multi_track_smoke.py")
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(helper)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=child_env,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["backend"] == "subproc"
    assert payload["n_envs"] == 2
    assert payload["reset_shape"] == [2, 9]
    assert payload["step_shape"] == [2, 9]
    assert payload["reward_shape"] == [2]
    assert payload["done_shape"] == [2]
    assert len(payload["track_ids"]) == 2
    assert all(isinstance(track_id, str) for track_id in payload["track_ids"])
    assert payload["closed"] is True


def test_real_multi_track_ppo_training_smoke() -> None:
    from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
    from f1rl.learning.training.vector_env import make_multi_track_vector_env

    vector = make_multi_track_vector_env(
        _scenario_set(),
        VectorEnvConfig(n_envs=2),
        sampling_seed=123,
    )
    try:
        model = PPOModel.create(
            vector,
            PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device="cpu"),
        )
        assert model.learn(8) is model
        assert model.num_timesteps >= 8
    finally:
        vector.close()


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


def test_training_package_exports_exact_optional_surface() -> None:
    import f1rl.learning as learning
    import f1rl.learning.training as training

    assert training.__all__ == (
        "F1RLEnvFactory",
        "TrainingScenario",
        "TrainingScenarioSet",
        "VectorEnvConfig",
        "VectorTrainingConfig",
        "make_multi_track_vector_env",
        "make_vector_env",
    )
    for name in training.__all__:
        assert getattr(training, name) is not None
        assert not hasattr(learning, name)
    assert not hasattr(training, "_RankedEnvFactory")


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_vector_production_dependency_boundaries_are_exact() -> None:
    sb3_owners: list[str] = []
    torch_owners: list[str] = []
    for path in Path("f1rl").rglob("*.py"):
        modules = _direct_imports(path)
        if any(module == "stable_baselines3" or module.startswith("stable_baselines3.") for module in modules):
            sb3_owners.append(path.as_posix())
        if any(module == "torch" or module.startswith("torch.") for module in modules):
            torch_owners.append(path.as_posix())

    assert sorted(sb3_owners) == [
        "f1rl/learning/algorithms/ppo.py",
        "f1rl/learning/training/vector_env.py",
    ]
    assert torch_owners == []
    assert "stable_baselines3" not in _direct_imports(
        Path("f1rl/learning/training/env_factory.py")
    )
    assert all(
        not module.startswith("f1rl.learning")
        for path in (*Path("f1rl/sandbox").rglob("*.py"), *Path("f1rl/runtime").rglob("*.py"))
        for module in _direct_imports(path)
    )


def test_training_docs_describe_vectorized_contract() -> None:
    text = Path("f1rl/learning/training/README.md").read_text(encoding="utf-8").lower()
    for token in (
        "same-track",
        "dummy",
        "subproc",
        "spawn",
        "__main__",
        "vectorized-ppo.yml",
        "close()",
        "diagnostic",
        "sampling_seed",
        "track_id",
        "canonical",
        "non-track",
    ):
        assert token in text


def test_algorithm_docs_describe_single_and_vector_training_contract() -> None:
    text = Path("f1rl/learning/algorithms/README.md").read_text(encoding="utf-8").lower()
    assert "f1rlenv" in text
    assert "vecenv" in text
    assert "single-observation" in text


def test_learning_docs_mark_vectorized_ppo_implemented_and_future_layers_pending() -> None:
    text = Path("f1rl/learning/README.md").read_text(encoding="utf-8").lower()
    assert "vectorized ppo" in text
    assert "implemented" in text
    assert "multi-track" in text
    assert "training orchestration" in text
    assert "evaluation" in text
    assert "multi-track sampling is implemented" in text
    assert "unseen-track generalization experiments" in text


\n