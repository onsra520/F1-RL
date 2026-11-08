from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from f1rl.local_demo.runtime import exploratory_models
from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.exploratory_models import (
    BrakeZeroExploratoryPolicy,
    ExploratoryModel,
    discover_exploratory_models,
    validate_observation_encoder,
)


def test_discovery_requires_the_registered_archive_digest(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    archive = tmp_path / ".archives" / "F1-RL" / "wp7-exploratory" / "run" / "model.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"known model")
    digest = sha256(b"known model").hexdigest()
    monkeypatch.setattr(exploratory_models, "ARCHIVES", (("wp7-calibration", "run/model.zip", digest),))

    models, paths, warnings = discover_exploratory_models(root)

    assert models == (ExploratoryModel("wp7-calibration", digest, len(b"known model")),)
    assert paths["wp7-calibration"] == archive
    assert warnings == ()


def test_missing_exploratory_archives_remain_optional(tmp_path):
    models, paths, warnings = discover_exploratory_models(tmp_path)

    assert models == ()
    assert paths == {}
    assert len(warnings) == 2
    assert all("unavailable" in warning for warning in warnings)


def test_brake_zero_adapter_preserves_legacy_actions_and_rejects_invalid_output():
    policy = BrakeZeroExploratoryPolicy(
        SimpleNamespace(predict=lambda _observation, deterministic: np.asarray([-0.4, 0.7], dtype=np.float32))
    )

    np.testing.assert_allclose(policy.predict(np.zeros(16, dtype=np.float32)), [-0.4, 0.7, 0.0])

    invalid = BrakeZeroExploratoryPolicy(SimpleNamespace(predict=lambda *_args, **_kwargs: [0.0, 2.0]))
    with pytest.raises(ValueError, match="invalid"):
        invalid.predict(np.zeros(16, dtype=np.float32))


def test_exploratory_loader_rejects_observation_space_mismatch(tmp_path, monkeypatch):
    import f1rl.learning.algorithms.ppo as ppo_module

    archive = tmp_path / "model.zip"
    archive.write_bytes(b"archive")
    env = SimpleNamespace(
        observation_space=gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32),
        action_space=gym.spaces.Box(
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(exploratory_models, "validate_observation_encoder", lambda _encoder: None)
    monkeypatch.setattr(
        ppo_module.PPOModel,
        "inspect_spaces",
        staticmethod(lambda _path: (gym.spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32), None)),
    )

    with pytest.raises(ValueError, match="observation space"):
        exploratory_models.load_exploratory_policy(archive, sha256(b"archive").hexdigest(), env, object())


def test_exploratory_controller_checks_current_canonical_sensor_contract():
    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    encoder = catalog.track_factory("straight-sprint-v1").observation_encoder
    validate_observation_encoder(encoder)

    changed = SimpleNamespace(
        config=SimpleNamespace(
            sensor_suite=encoder.config.sensor_suite,
            velocity_scale=20.0,
            target_distance_scale=encoder.config.target_distance_scale,
        ),
        feature_names=encoder.feature_names,
    )
    with pytest.raises(ValueError, match="observation semantics"):
        validate_observation_encoder(changed)
