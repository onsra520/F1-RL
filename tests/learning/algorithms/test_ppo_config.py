from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from f1rl.learning.algorithms.ppo import PPOConfig


def test_ppo_config_defaults_match_baseline() -> None:
    assert PPOConfig() == PPOConfig(
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=0,
        device="cpu",
    )


def test_ppo_config_is_immutable() -> None:
    config = PPOConfig()
    with pytest.raises(FrozenInstanceError):
        config.device = "cuda"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("learning_rate", float("nan")),
        ("learning_rate", True),
        ("n_steps", 1),
        ("n_steps", 2.5),
        ("n_steps", True),
        ("batch_size", 1),
        ("batch_size", True),
        ("n_epochs", 0),
        ("n_epochs", 1.5),
        ("n_epochs", True),
        ("gamma", -0.01),
        ("gamma", 1.01),
        ("gamma", True),
        ("gae_lambda", -0.01),
        ("gae_lambda", 1.01),
        ("clip_range", 0.0),
        ("clip_range", 1.01),
        ("ent_coef", -0.01),
        ("vf_coef", -0.01),
        ("max_grad_norm", 0.0),
        ("max_grad_norm", float("inf")),
        ("seed", True),
        ("seed", 1.5),
    ],
)
def test_ppo_config_rejects_invalid_field_values(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)


def test_batch_size_must_fit_and_divide_rollout() -> None:
    with pytest.raises(ValueError):
        PPOConfig(n_steps=8, batch_size=16)
    with pytest.raises(ValueError):
        PPOConfig(n_steps=8, batch_size=3)


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_supported_devices_are_accepted(device: str) -> None:
    assert PPOConfig(device=device).device == device


@pytest.mark.parametrize("device", ["", "gpu", "CUDA", 1, None])
def test_unsupported_devices_are_rejected(device: object) -> None:
    with pytest.raises(ValueError):
        PPOConfig(device=device)


def test_seed_may_be_none() -> None:
    assert PPOConfig(seed=None).seed is None


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ppo.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_checked_in_yaml_matches_defaults() -> None:
    assert PPOConfig.from_yaml("configs/training/ppo-baseline.yml") == PPOConfig()


def test_yaml_omitted_fields_use_defaults(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "ppo:\n  n_steps: 8\n  batch_size: 4\n")
    config = PPOConfig.from_yaml(path)
    assert config.n_steps == 8
    assert config.batch_size == 4
    assert config.learning_rate == PPOConfig().learning_rate
    assert config.device == "cpu"


@pytest.mark.parametrize("device", ["cuda", "auto"])
def test_yaml_accepts_accelerator_device_modes(tmp_path: Path, device: str) -> None:
    path = _write_yaml(tmp_path, f"ppo:\n  device: {device}\n")
    assert PPOConfig.from_yaml(path).device == device


def test_empty_root_mapping_uses_defaults(tmp_path: Path) -> None:
    assert PPOConfig.from_yaml(_write_yaml(tmp_path, "{}\n")) == PPOConfig()


@pytest.mark.parametrize(
    "text",
    [
        "- not-a-mapping\n",
        "ppo: []\n",
        "other: 1\n",
        "ppo:\n  unknown: 1\n",
        "ppo:\n  n_steps: 1\n",
        "ppo:\n  n_steps: [8\n",
    ],
)
def test_yaml_rejects_invalid_documents(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        PPOConfig.from_yaml(_write_yaml(tmp_path, text))


def test_yaml_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PPOConfig.from_yaml(tmp_path / "missing.yml")


def test_yaml_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PPOConfig.from_yaml(tmp_path)


@pytest.mark.parametrize("path", [None, 123, object()])
def test_yaml_rejects_invalid_path_types(path: object) -> None:
    with pytest.raises(ValueError):
        PPOConfig.from_yaml(path)
