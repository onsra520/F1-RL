"""WP7 final model audit must execute before a training run is marked completed."""
import json
from pathlib import Path

import pytest

import f1rl.learning.training.runner as runner_module
from f1rl.learning.algorithms.ppo import PPOConfig
from f1rl.learning.training.config import TrainingRunConfig, VectorEnvConfig
from f1rl.learning.training.runner import TrainingRunner
from tests.learning.training.mocks.mock_factories import make_scenario_set
from tests.learning.training.mocks.mock_ppo import MockPPOModel
from tests.learning.training.mocks.mock_vec_env import MockVecEnv


def _config(root:Path)->TrainingRunConfig:
    return TrainingRunConfig(
        ppo=PPOConfig(seed=0),
        vector_env=VectorEnvConfig(n_envs=8),
        total_timesteps=200_000,
        checkpoint_every_steps=25_000,
        sampling_seed=0,
        artifacts_root=root,
    )


def _mocks(monkeypatch):
    MockPPOModel.reset()
    MockPPOModel.progress_points=tuple(range(25_000, 200_001, 25_000))
    monkeypatch.setattr(runner_module,"PPOModel",MockPPOModel)
    monkeypatch.setattr(runner_module,"make_multi_track_vector_env",lambda *args:MockVecEnv())


def test_final_observer_sees_archive_before_runner_completes(tmp_path,monkeypatch):
    _mocks(monkeypatch)
    final_seen=[]
    def verify(path:Path,steps:int)->None:
        assert path.is_file()
        assert path.name.endswith("-final-model.zip")
        assert steps==200_000
        state=json.loads((path.parent/"state.json").read_text("utf-8"))
        assert state["status"]=="running"
        assert state["final_model"] is None
        final_seen.append(path)
    result=TrainingRunner(_config(tmp_path/"runs"),final_model_observer=verify).run_multi(make_scenario_set())
    assert final_seen==[result.final_model_path]
    state=json.loads((result.run_directory/"state.json").read_text("utf-8"))
    assert state["status"]=="completed"


def test_final_observer_error_aborts_success_and_removes_unverified_final(tmp_path,monkeypatch):
    _mocks(monkeypatch)
    def reject(path,steps):
        assert path.is_file()
        raise RuntimeError("final model evaluation failed")
    with pytest.raises(RuntimeError,match="final model evaluation failed"):
        TrainingRunner(_config(tmp_path/"runs"),final_model_observer=reject).run_multi(make_scenario_set())
    run=next((tmp_path/"runs").iterdir())
    state=json.loads((run/"state.json").read_text("utf-8"))
    assert state["status"]=="failed"
    assert state["final_model"] is None
    assert not list(run.glob("*final-model.zip"))
    assert len(list((run/"checkpoints").glob("*.zip")))==7


def test_final_observer_invalid_input_rejected(tmp_path):
    with pytest.raises(ValueError,match="final_model_observer"):
        TrainingRunner(_config(tmp_path/"runs"),final_model_observer=object())
