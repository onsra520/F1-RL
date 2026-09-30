import json
from pathlib import Path

import yaml

from f1rl.learning.training import TrainingRunConfig, TrainingRunner
from tests.learning.training.test_vector_env import _scenario_set


def test_real_multi_track_training_orchestration_writes_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "artifacts" / "runs"
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "total_timesteps": 8,
                    "checkpoint_every_steps": 4,
                    "sampling_seed": 123,
                    "artifacts_root": str(artifacts_root),
                },
                "ppo": {
                    "n_steps": 4,
                    "batch_size": 4,
                    "n_epochs": 1,
                    "seed": 42,
                    "device": "cpu",
                },
                "vector_env": {"n_envs": 2, "backend": "dummy", "start_method": "spawn"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = TrainingRunConfig.from_yaml(config_path)
    scenarios = _scenario_set()
    result = TrainingRunner(config).run_multi(scenarios)

    assert result.requested_total_timesteps == 8
    assert result.actual_total_timesteps >= 8
    assert len(result.checkpoint_paths) == 1
    checkpoint = result.checkpoint_paths[0]
    assert checkpoint.is_file()
    assert "ppo-multitrack-seed42-sample123-step" in checkpoint.name
    assert result.final_model_path.is_file()
    assert result.final_model_path.name == "ppo-multitrack-seed42-sample123-final-model.zip"

    state_path = result.run_directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["actual_total_timesteps"] == result.actual_total_timesteps
    assert state["training_tracks"] == [scenario.track_id for scenario in scenarios.scenarios]
    assert state["sampling_seed"] == 123
    assert result.run_directory / state["latest_checkpoint"] == checkpoint
    assert result.run_directory / state["final_model"] == result.final_model_path
