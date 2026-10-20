from pathlib import Path
import json

import numpy as np

from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.learning.training.config import VectorEnvConfig
from f1rl.learning.training.telemetry import TrainingTelemetryRecorder
from f1rl.learning.training.vector_env import _SampledF1RLEnv, make_multi_track_vector_env
from f1rl.local_demo.runner import (
    _candidate_reward,
    _install_demo_reward,
    _validate_smoke_telemetry,
)
from f1rl.research.simulation_only_protocol import load_simulation_only_manifest
from f1rl.research.wp7_protocol import (
    build_protocol,
    load_registration,
    prepare_scenarios,
    prepare_training_scenarios,
)


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

    # A coherent 256-step audit must still fail closed if the recorded
    # automatic reset is missing or its identity is changed after the run.
    for name, mutate in (
        ("missing-reset", lambda event: event.update(autoreset_identity=None)),
        ("spoofed-reset", lambda event: event["autoreset_identity"].update(seed=1001)),
    ):
        corrupted = [json.loads(json.dumps(event)) for event in events]
        terminal = next(event for event in corrupted if event.get("event_type") == "step")
        mutate(terminal)
        corrupted_path = tmp_path / f"{name}.jsonl"
        corrupted_path.write_text(
            "".join(json.dumps(event) + "\n" for event in corrupted), encoding="utf-8"
        )
        try:
            _validate_smoke_telemetry(corrupted_path, run_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"smoke telemetry auditor accepted {name} identity")


def test_real_train_curriculum_termination_preserves_dummy_vec_autoreset_identity(tmp_path):
    manifest, _ = load_simulation_only_manifest(ROOT)
    _, reward_config = _candidate_reward(manifest, "C")
    registration = load_registration(ROOT, profile="spatial_rotating_owner_v3")
    scenarios = prepare_scenarios(ROOT, registration, seed_index=0)
    assert tuple(scenario.track_id for scenario in scenarios.scenarios) == TRAIN_IDS
    vec = make_multi_track_vector_env(
        scenarios, VectorEnvConfig(n_envs=1, backend="dummy"), sampling_seed=0
    )
    _install_demo_reward(vec, reward_config)
    vec.env_method("set_training_telemetry_enabled", True)
    contracts = vec.get_attr("training_reset_contracts")[0]

    class VecModel:
        num_timesteps = 0
        _n_updates = 0

        def __init__(self):
            self._env = vec

        def get_env(self):
            return self._env

        def weights_sha256(self):
            return "a" * 64

    model = VecModel()
    recorder = TrainingTelemetryRecorder(
        tmp_path / "real-dummy-vec-smoke.jsonl",
        run_identity={
            "run_id": "real-dummy-vec-smoke", "source_commit": "b" * 40,
            "policy_id": "local-demo-smoke-C-seed0", "split": "TRAIN",
            "track_ids": list(TRAIN_IDS),
            "seed_identity": {"ppo": 0, "sampling": 0, "training_reset": 1000},
            "reset_contracts": contracts,
        },
        max_timesteps=32, capture_autoreset=True,
    )
    try:
        observation = vec.reset()
        recorder.on_training_start(model, total_timesteps=32, n_envs=1)
        action = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        terminal_info = None
        terminal_timestep = None
        for timestep in range(1, 33):
            next_observation, rewards, dones, infos = vec.step(action)
            model.num_timesteps = timestep
            recorder.on_step({
                "obs": observation,
                "new_obs": next_observation,
                "actions": action,
                "clipped_actions": action,
                "rewards": rewards,
                "dones": dones,
                "infos": infos,
            }, model)
            observation = next_observation
            if dones[0] and terminal_timestep is None:
                terminal_info = infos[0]
                terminal_timestep = timestep
            elif terminal_timestep is not None:
                break
        assert terminal_info is not None
        assert terminal_info["demo_training"]["curriculum_terminal"] is True
        assert terminal_info["demo_training"]["lateral_distance_m"] > 3.0
        assert terminal_info["TimeLimit.truncated"] is False
        assert np.asarray(terminal_info["terminal_observation"]).shape == (16,)

        recorder.close()
        rows = [json.loads(line) for line in (tmp_path / "real-dummy-vec-smoke.jsonl").read_text().splitlines()]
        steps = [row for row in rows if row["event_type"] == "step"]
        terminal_index = next(index for index, row in enumerate(steps)
                              if row["demo_training"]["curriculum_terminal"])
        terminal = steps[terminal_index]
        following = steps[terminal_index + 1]
        assert terminal["termination_reason"] == "demo_curriculum_boundary"
        np.testing.assert_array_equal(terminal["next_observation"], terminal_info["terminal_observation"])
        np.testing.assert_array_equal(terminal["autoreset_observation"], following["observation"])
        assert following["episode_index"] == terminal["episode_index"] + 1
        assert terminal["autoreset_identity"] == following["seed_identity"]["train_reset"]
        assert terminal["autoreset_episode_id"] == following["episode_id"]
    finally:
        recorder.close()
        vec.close()
