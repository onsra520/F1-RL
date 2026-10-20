from pathlib import Path
import json

import numpy as np

from f1rl.learning.training.telemetry import TrainingTelemetryRecorder
from f1rl.learning.training.vector_env import _SampledF1RLEnv
from f1rl.local_demo.runner import _validate_smoke_telemetry
from f1rl.research.wp7_protocol import build_protocol, prepare_training_scenarios


ROOT = Path(__file__).resolve().parents[3]


class _Model:
    num_timesteps = 1
    _n_updates = 0

    def __init__(self, env):
        self._env = env

    def weights_sha256(self):
        return "a" * 64

    def get_env(self):
        return self._env


def test_smoke_telemetry_separates_terminal_transition_from_autoreset_episode(tmp_path):
    protocol = build_protocol(ROOT, profile="spatial_rotating_owner_v3")
    scenarios, _ = prepare_training_scenarios(ROOT, protocol, "multitrack-seed0")
    env = _SampledF1RLEnv(scenarios, sampling_seed=31, rank=0)
    env.set_training_telemetry_enabled(True)
    recorder = None
    try:
        observation, reset_info = env.reset(seed=17)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        terminal_observation, reward, _, _, terminal_info = env.step(action)
        terminal_info["track_id"] = reset_info["track_id"]
        terminal_info["terminal_observation"] = terminal_observation.copy()
        autoreset_observation, autoreset_info = env.reset()
        autoreset_info["track_id"] = env._current_scenario.track_id
        vec = type("Vec", (), {"reset_infos": [autoreset_info]})()
        model = _Model(vec)
        contracts = env.training_reset_contracts
        identity = {
            "run_id": "smoke-autoreset-unit",
            "source_commit": "b" * 40,
            "policy_id": "local-demo-smoke-C-seed0",
            "split": "TRAIN",
            "track_ids": list(contracts),
            "seed_identity": {"training_reset": 1000},
            "reset_contracts": contracts,
        }
        recorder = TrainingTelemetryRecorder(
            tmp_path / "smoke.jsonl", run_identity=identity, max_timesteps=1,
            capture_autoreset=True,
        )
        recorder.on_training_start(model, total_timesteps=1, n_envs=1)
        recorder.on_step({
            "obs": np.asarray([observation], dtype=np.float32),
            "new_obs": np.asarray([autoreset_observation], dtype=np.float32),
            "actions": np.asarray([action], dtype=np.float32),
            "clipped_actions": np.asarray([action], dtype=np.float32),
            "rewards": np.asarray([reward], dtype=np.float32),
            "dones": np.asarray([True]), "infos": [terminal_info],
        }, model)
        recorder.close()

        row = json.loads((tmp_path / "smoke.jsonl").read_text().splitlines()[-1])
        assert row["episode_index"] == 0
        assert row["seed_identity"]["train_reset"]["reset_index"] == 0
        np.testing.assert_array_equal(row["next_observation"], terminal_observation)
        np.testing.assert_array_equal(row["autoreset_observation"], autoreset_observation)
        assert row["autoreset_identity"]["reset_index"] == 1
        assert row["autoreset_identity"]["track_id"] == autoreset_info["track_id"]
        assert row["autoreset_episode_id"].endswith("episode-1")
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def test_smoke_telemetry_auditor_enforces_exact_rollout_and_episode_continuity(tmp_path):
    run_id = "smoke-auditor-unit"
    reset0 = {"seed": 1000, "rank": 0, "reset_index": 0, "draw_seed": 6,
              "track_id": "straight-sprint-v1", "version": "wp7_train_reset_research_v1",
              "configuration_sha256": "a" * 64, "world_sha256": "b" * 64}
    autoreset = {"seed": 1000, "rank": 0, "reset_index": 1, "draw_seed": 7,
                 "track_id": "slalom-v1", "version": "wp7_train_reset_research_v1",
                 "configuration_sha256": "c" * 64, "world_sha256": "d" * 64}
    events = [{"event_type": "run", "run_id": run_id,
               "schema_version": "f1rl_local_demo_smoke_telemetry_v1"},
              {"event_type": "training_start", "run_id": run_id,
               "schema_version": "f1rl_local_demo_smoke_telemetry_v1",
               "policy_weights_sha256": "a" * 64}]
    for index in range(1, 257):
        terminal = index == 1
        episode = 0 if terminal else 1
        track = "straight-sprint-v1" if terminal else "slalom-v1"
        observation = ([0.] * 16) if index == 1 else ([2.] * 16 if index == 2 else [float(index)] * 16)
        next_observation = [1.] * 16 if index == 1 else [float(index + 1)] * 16
        events.append({
            "event_type": "step", "schema_version": "f1rl_local_demo_smoke_telemetry_v1",
            "global_timestep": index, "env_index": 0, "episode_index": episode,
            "episode_id": f"{run_id}/env-0/episode-{episode}", "track_id": track,
            "seed_identity": {"train_reset": reset0 if terminal else autoreset},
            "terminated": terminal, "truncated": False,
            "autoreset_identity": autoreset if terminal else None,
            "autoreset_observation": [2.] * 16 if terminal else None,
            "autoreset_episode_id": f"{run_id}/env-0/episode-1" if terminal else None,
            "observation": observation, "next_observation": next_observation,
            "raw_ppo_action": [0.] * 3, "action_sent_to_environment": [0.] * 3,
        })
    events.append({"event_type": "ppo_update", "schema_version": "f1rl_local_demo_smoke_telemetry_v1",
                   "ppo_epoch_updates": 10, "policy_weights_sha256": "c" * 64,
                   "metrics": {"train/loss": 1., "train/policy_gradient_loss": .1,
                               "train/value_loss": .9}})
    path = tmp_path / "auditor.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    result = _validate_smoke_telemetry(path, run_id)

    assert result["step_rows"] == 256
    assert result["automatic_resets_verified"] == 1
    assert result["episodes_terminated_or_truncated"] == 1
    assert result["episodes_started"] == 2
