from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.rl.algorithms.ppo import PPOConfig
from src.training.config import TrainingRunConfig, VectorEnvConfig


def _yaml(
    *,
    training: str = "{}",
    ppo: str = "{}",
    vector_env: str = "{}",
) -> str:
    return f"training: {training}\nppo: {ppo}\nvector_env: {vector_env}\n"


def test_training_run_config_defaults_are_exact() -> None:
    config = TrainingRunConfig(ppo=PPOConfig(), vector_env=VectorEnvConfig())

    assert config.total_timesteps == 50_000
    assert config.checkpoint_every_steps == 25_000
    assert config.sampling_seed == 0
    assert config.artifacts_root == Path("artifacts/runs")


def test_training_run_config_is_frozen() -> None:
    config = TrainingRunConfig(ppo=PPOConfig(), vector_env=VectorEnvConfig())
    with pytest.raises(FrozenInstanceError):
        config.total_timesteps = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_timesteps", True),
        ("total_timesteps", 0),
        ("total_timesteps", -1),
        ("checkpoint_every_steps", True),
        ("checkpoint_every_steps", 0),
        ("checkpoint_every_steps", -1),
        ("sampling_seed", True),
        ("sampling_seed", 1.5),
        ("sampling_seed", "1"),
        ("sampling_seed", -1),
        ("sampling_seed", 2**32),
    ],
)
def test_training_run_config_rejects_invalid_scalar_fields(field: str, value: object) -> None:
    kwargs = {
        "ppo": PPOConfig(),
        "vector_env": VectorEnvConfig(),
        field: value,
    }
    with pytest.raises(ValueError):
        TrainingRunConfig(**kwargs)


def test_training_run_config_accepts_disabled_checkpointing_and_boundary_seeds() -> None:
    config = TrainingRunConfig(
        ppo=PPOConfig(seed=2**32 - 1),
        vector_env=VectorEnvConfig(),
        checkpoint_every_steps=None,
        sampling_seed=2**32 - 1,
    )
    assert config.checkpoint_every_steps is None
    assert config.ppo.seed == 2**32 - 1
    assert config.sampling_seed == 2**32 - 1


def test_yaml_omitted_training_fields_use_defaults_and_resolve_relative_root(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(_yaml(), encoding="utf-8")

    config = TrainingRunConfig.from_yaml(path)

    assert config.ppo == PPOConfig()
    assert config.vector_env == VectorEnvConfig()
    assert config.total_timesteps == 50_000
    assert config.checkpoint_every_steps == 25_000
    assert config.sampling_seed == 0
    assert config.artifacts_root == tmp_path / "artifacts" / "runs"


def test_yaml_resolves_explicit_relative_artifacts_root_against_config_parent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.yml"
    path.parent.mkdir()
    path.write_text(_yaml(training="{artifacts_root: outputs/runs}"), encoding="utf-8")

    config = TrainingRunConfig.from_yaml(path)

    assert config.artifacts_root == path.parent / "outputs" / "runs"


def test_yaml_preserves_absolute_artifacts_root(tmp_path: Path) -> None:
    destination = (tmp_path / "absolute-runs").resolve()
    path = tmp_path / "config.yml"
    path.write_text(_yaml(training=f"{{artifacts_root: '{destination.as_posix()}'}}"), encoding="utf-8")

    config = TrainingRunConfig.from_yaml(path)
    assert config.artifacts_root == destination


@pytest.mark.parametrize(
    "text",
    [
        "[]\n",
        "training: {}\nppo: {}\n",
        "training: {}\nppo: {}\nvector_env: {}\nextra: {}\n",
        "training: []\nppo: {}\nvector_env: {}\n",
        "training: {}\nppo: []\nvector_env: {}\n",
        "training: {}\nppo: {}\nvector_env: []\n",
        "training: {unknown: 1}\nppo: {}\nvector_env: {}\n",
        "training: {}\nppo: {unknown: 1}\nvector_env: {}\n",
        "training: {}\nppo: {}\nvector_env: {unknown: 1}\n",
        "training: {artifacts_root: ''}\nppo: {}\nvector_env: {}\n",
        "training: {artifacts_root: '   '}\nppo: {}\nvector_env: {}\n",
    ],
)
def test_yaml_rejects_malformed_unknown_or_blank_documents(tmp_path: Path, text: str) -> None:
    path = tmp_path / "config.yml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        TrainingRunConfig.from_yaml(path)


@pytest.mark.parametrize("path", ["", "   ", 123, None, object()])
def test_yaml_rejects_invalid_path_values(path: object) -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig.from_yaml(path)


def test_yaml_rejects_missing_file_and_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig.from_yaml(tmp_path / "missing.yml")
    with pytest.raises(ValueError):
        TrainingRunConfig.from_yaml(tmp_path)


def test_checked_in_root_yaml_matches_v1_defaults() -> None:
    config = TrainingRunConfig.from_yaml(Path("config.yml"))

    assert config.ppo == PPOConfig()
    assert config.vector_env == VectorEnvConfig()
    assert config.total_timesteps == 50_000
    assert config.checkpoint_every_steps == 25_000
    assert config.sampling_seed == 0
    assert config.artifacts_root == Path("artifacts/runs").resolve()
