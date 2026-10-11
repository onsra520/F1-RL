"""WP7 pilot plumbing: mock-tested saved-checkpoint observer; never runs PPO."""
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


def _config(root):
    return TrainingRunConfig(
        ppo=PPOConfig(seed=0),
        vector_env=VectorEnvConfig(n_envs=8),
        total_timesteps=100_000,
        checkpoint_every_steps=25_000,
        sampling_seed=0,
        artifacts_root=root,
    )


def _install_mocks(monkeypatch):
    MockPPOModel.reset()
    MockPPOModel.progress_points=(25_000,50_000,75_000,100_000)
    monkeypatch.setattr(runner_module,"PPOModel",MockPPOModel)
    monkeypatch.setattr(runner_module,"make_multi_track_vector_env",lambda *args:MockVecEnv())


def test_checkpoint_observer_runs_after_each_durable_model_save(tmp_path,monkeypatch):
    _install_mocks(monkeypatch)
    seen=[]
    def inspect_saved_checkpoint(path:Path,actual_steps:int)->None:
        assert path.is_file()
        assert path.read_text(encoding="utf-8")=="mock model"
        seen.append((actual_steps,path))
    runner=TrainingRunner(_config(tmp_path/"runs"),checkpoint_observer=inspect_saved_checkpoint)
    result=runner.run_multi(make_scenario_set())
    assert [step for step,_ in seen]==[25_000,50_000,75_000]
    assert [p for _,p in seen]==list(result.checkpoint_paths)
    assert result.actual_total_timesteps==100_000
    state=json.loads((result.run_directory/"state.json").read_text(encoding="utf-8"))
    assert state["status"]=="completed"
    assert state["latest_checkpoint"].endswith("step75000.zip")


def test_checkpoint_observer_failure_aborts_and_preserves_failure_state(tmp_path,monkeypatch):
    _install_mocks(monkeypatch)
    failure=RuntimeError("checkpoint audit failed")
    def reject(path:Path,actual_steps:int)->None:
        assert path.is_file()
        raise failure
    runner=TrainingRunner(_config(tmp_path/"runs"),checkpoint_observer=reject)
    with pytest.raises(RuntimeError,match="checkpoint audit failed"):
        runner.run_multi(make_scenario_set())
    run=next((tmp_path/"runs").iterdir())
    state=json.loads((run/"state.json").read_text(encoding="utf-8"))
    assert state["status"]=="failed"
    assert state["final_model"] is None
    assert len(list((run/"checkpoints").glob("*.zip")))==1


def test_checkpoint_observer_invalid_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError,match="checkpoint_observer"):
        TrainingRunner(_config(tmp_path/"runs"),checkpoint_observer=123)


def test_no_observer_keeps_previous_runner_behavior(tmp_path,monkeypatch):
    _install_mocks(monkeypatch)
    result=TrainingRunner(_config(tmp_path/"runs")).run_multi(make_scenario_set())
    assert len(result.checkpoint_paths)==3
