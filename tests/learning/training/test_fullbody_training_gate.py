"""Prospective full-body mode must not silently train using legacy protocol."""
from dataclasses import replace

import pytest

from f1rl.learning.algorithms.ppo import PPOConfig
from f1rl.learning.training.config import TrainingRunConfig, VectorEnvConfig
from f1rl.learning.training.env_factory import TrainingScenario, TrainingScenarioSet
from f1rl.learning.training.runner import TrainingRunner
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.vector import Vec3
from tests.learning.training.mocks.mock_factories import make_factory, make_scenario_set


_BODY = VehicleBodyGeometry(1.6, 1.2, 4.0, Vec3(0., .5, 0.))


def test_singletrack_fullbody_cannot_train_without_scientific_approval(tmp_path) -> None:
    factory = replace(make_factory(), body_geometry=_BODY)
    root = tmp_path / "new-runs"
    runner = TrainingRunner(TrainingRunConfig(
        PPOConfig(seed=1), VectorEnvConfig(n_envs=1), artifacts_root=root,
    ))
    with pytest.raises(ValueError, match="full-body candidate training is blocked"):
        runner.run_single(factory)
    assert not root.exists()


def test_multitrack_fullbody_cannot_train_without_scientific_approval(tmp_path) -> None:
    baseline = make_scenario_set()
    scenarios = TrainingScenarioSet(tuple(
        TrainingScenario(s.track_id, replace(s.factory, body_geometry=_BODY))
        for s in baseline.scenarios
    ))
    root = tmp_path / "new-runs"
    runner = TrainingRunner(TrainingRunConfig(
        PPOConfig(seed=1), VectorEnvConfig(n_envs=1), artifacts_root=root,
    ))
    with pytest.raises(ValueError, match="full-body candidate training is blocked"):
        runner.run_multi(scenarios)
    assert not root.exists()


def test_prospective_multitrack_requires_identical_body_geometry() -> None:
    baseline = make_scenario_set()
    a, b = baseline.scenarios[:2]
    with pytest.raises(ValueError, match="non-track environment contract"):
        TrainingScenarioSet((
            TrainingScenario(a.track_id, replace(a.factory, body_geometry=_BODY)),
            TrainingScenario(b.track_id, replace(b.factory, body_geometry=None)),
        ))
