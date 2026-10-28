from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.rl.algorithms.ppo import PPOConfig
from src.training.config import VectorEnvConfig, VectorTrainingConfig


def test_vector_env_config_defaults_are_exact() -> None:
    config = VectorEnvConfig()
    assert config.n_envs == 8
    assert config.backend == "dummy"
    assert config.start_method == "spawn"


def test_vector_env_config_is_frozen() -> None:
    config = VectorEnvConfig()
    with pytest.raises(FrozenInstanceError):
        config.n_envs = 4  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "8", None])
def test_vector_env_config_rejects_invalid_counts(value: object) -> None:
    with pytest.raises(ValueError):
        VectorEnvConfig(n_envs=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["", "auto", "process", "Dummy", 1, None])
def test_vector_env_config_rejects_invalid_backends(backend: object) -> None:
    with pytest.raises(ValueError):
        VectorEnvConfig(backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize("start_method", ["", "   ", 1, None])
def test_vector_env_config_rejects_invalid_start_methods(start_method: object) -> None:
    with pytest.raises(ValueError):
        VectorEnvConfig(start_method=start_method)  # type: ignore[arg-type]


def test_subproc_requires_spawn() -> None:
    with pytest.raises(ValueError):
        VectorEnvConfig(backend="subproc", start_method="fork")


def test_dummy_stores_non_empty_start_method_without_runtime_semantics() -> None:
    config = VectorEnvConfig(backend="dummy", start_method="forkserver")
    assert config.start_method == "forkserver"


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "vector.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_checked_in_vector_yaml_matches_defaults() -> None:
    loaded = VectorTrainingConfig.from_yaml("configs/training/vectorized-ppo.yml")
    assert loaded == VectorTrainingConfig(PPOConfig(), VectorEnvConfig())


def test_combined_yaml_omitted_fields_use_defaults(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "ppo:\n  n_steps: 8\n  batch_size: 4\nvector_env:\n  n_envs: 4\n",
    )
    loaded = VectorTrainingConfig.from_yaml(path)
    assert loaded.ppo.n_steps == 8
    assert loaded.ppo.batch_size == 4
    assert loaded.ppo.device == "cpu"
    assert loaded.vector_env.n_envs == 4
    assert loaded.vector_env.backend == "dummy"
    assert loaded.vector_env.start_method == "spawn"


@pytest.mark.parametrize(
    "text",
    [
        "{}\n",
        "ppo: {}\n",
        "vector_env: {}\n",
        "ppo: []\nvector_env: {}\n",
        "ppo: {}\nvector_env: []\n",
        "ppo: {}\nvector_env: {}\nother: 1\n",
        "ppo:\n  unknown: 1\nvector_env: {}\n",
        "ppo: {}\nvector_env:\n  unknown: 1\n",
        "ppo:\n  n_steps: [8\nvector_env: {}\n",
        "- not-a-mapping\n",
        "\n",
    ],
)
def test_combined_yaml_rejects_malformed_or_unknown_documents(
    tmp_path: Path,
    text: str,
) -> None:
    with pytest.raises(ValueError):
        VectorTrainingConfig.from_yaml(_write_yaml(tmp_path, text))


@pytest.mark.parametrize("path", ["", "   ", 123, None, object()])
def test_combined_yaml_rejects_invalid_path_values(path: object) -> None:
    with pytest.raises(ValueError):
        VectorTrainingConfig.from_yaml(path)  # type: ignore[arg-type]


def test_combined_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        VectorTrainingConfig.from_yaml(tmp_path / "missing.yml")


def test_standalone_ppo_yaml_contract_is_unchanged(tmp_path: Path) -> None:
    assert PPOConfig.from_yaml("configs/training/ppo-baseline.yml") == PPOConfig()

    extra_root = _write_yaml(
        tmp_path,
        "ppo: {}\nvector_env:\n  n_envs: 8\n",
    )
    with pytest.raises(ValueError):
        PPOConfig.from_yaml(extra_root)
