"""Provenance checks against actual registered TRAIN environment transitions."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from f1rl.learning.training.telemetry import TrainingTelemetryRecorder
from f1rl.learning.training.vector_env import _SampledF1RLEnv
from f1rl.research.wp7_protocol import build_protocol, prepare_training_scenarios


ROOT = Path(__file__).resolve().parents[3]
PROFILE = "spatial_rotating_owner_v3"
POLICY = "singletrack-straight-sprint-v1-seed0"


class _Model:
    num_timesteps = 1
    _n_updates = 0

    def weights_sha256(self):
        return "a" * 64


def _actual_train_transition():
    protocol = build_protocol(ROOT, profile=PROFILE)
    scenarios, _ = prepare_training_scenarios(ROOT, protocol, POLICY)
    scenario = scenarios.scenarios[0]
    env = scenario.factory.create()
    try:
        env.set_training_telemetry_enabled(True)
        observation, _ = env.reset(seed=17)
        raw_action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(raw_action)
        transition = {
            "obs": np.asarray([observation], dtype=np.float32),
            "actions": np.asarray([raw_action], dtype=np.float32),
            "clipped_actions": np.asarray([raw_action], dtype=np.float32),
            "rewards": np.asarray([reward], dtype=np.float32),
            "dones": np.asarray([terminated or truncated], dtype=bool),
            "infos": [info],
        }
        return env, transition
    except BaseException:
        env.close()
        raise


def _actual_multitrack_batch():
    protocol = build_protocol(ROOT, profile=PROFILE)
    scenarios, _ = prepare_training_scenarios(ROOT, protocol, "multitrack-seed0")
    envs, observations, actions, rewards, dones, infos = [], [], [], [], [], []
    try:
        for rank, scenario in enumerate(scenarios.scenarios[:2]):
            env = scenario.factory.create(reset_rank=rank, reset_sequence=0)
            env.set_training_telemetry_enabled(True)
            observation, _ = env.reset(seed=17 + rank)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_observation, reward, terminated, truncated, info = env.step(action)
            info["track_id"] = scenario.track_id
            envs.append(env)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
        return scenarios, envs, {
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "clipped_actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "infos": infos,
        }
    except BaseException:
        for env in envs:
            env.close()
        raise


@pytest.mark.parametrize(("declared_seed", "accepted"), [(1000, True), (1001, False)])
def test_declared_run_reset_seed_must_match_actual_train_transition(
    tmp_path, declared_seed, accepted
):
    env, transition = _actual_train_transition()
    recorder = None
    try:
        if not accepted:
            actual_reset = transition["infos"][0]["train_reset"]
            assert actual_reset["seed"] == 1000
            declared_contract = replace(env._reset_config, seed=declared_seed).identity_for_track(
                env._reset_track_id
            )
            recorder = TrainingTelemetryRecorder(
                tmp_path / f"telemetry-{declared_seed}.jsonl",
                run_identity={
                    "run_id": f"seed-binding-{declared_seed}",
                    "source_commit": "b" * 40,
                    "policy_id": POLICY,
                    "split": "TRAIN",
                    "track_ids": [env._reset_track_id],
                    "seed_identity": {"ppo": 0, "training_reset": declared_seed},
                    "reset_contracts": {env._reset_track_id: declared_contract},
                },
                max_timesteps=2,
            )
            recorder.on_training_start(_Model(), total_timesteps=1, n_envs=1)
            with pytest.raises(ValueError, match="declared training reset seed"):
                recorder.on_step(transition, _Model())
            recorder.close()
            return
        recorder = TrainingTelemetryRecorder(
            tmp_path / f"telemetry-{declared_seed}.jsonl",
            run_identity={
                "run_id": f"seed-binding-{declared_seed}",
                "source_commit": "b" * 40,
                "policy_id": POLICY,
                "split": "TRAIN",
                "track_ids": [env._reset_track_id],
                "seed_identity": {"ppo": 0, "training_reset": declared_seed},
                "reset_contracts": {env._reset_track_id: env.training_reset_contract},
            },
            max_timesteps=2,
        )
        recorder.on_training_start(_Model(), total_timesteps=1, n_envs=1)
        recorder.on_step(transition, _Model())
        assert recorder.step_rows == 1
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


@pytest.mark.parametrize("field", ["seed", "rank", "reset_index", "draw_seed", "track_id",
                                    "configuration_sha256", "world_sha256", "version"])
def test_actual_train_transition_rejects_spoofed_reset_identity(tmp_path, field):
    env, transition = _actual_train_transition()
    try:
        reset = dict(transition["infos"][0]["train_reset"])
        wrong_version = ("wp7_train_reset_lifecycle_research_v1"
                         if reset["version"] == "wp7_train_reset_research_v1"
                         else "wp7_train_reset_research_v1")
        reset[field] = {"seed": 1001, "rank": 1, "reset_index": 1, "draw_seed": 1,
                        "track_id": "right-turn-v1", "configuration_sha256": "e" * 64,
                        "world_sha256": "f" * 64,
                        "version": wrong_version}[field]
        transition["infos"][0]["train_reset"] = reset
        transition["infos"][0]["training_telemetry"]["seed_identity"] = dict(reset)
        recorder = TrainingTelemetryRecorder(
            tmp_path / f"spoof-{field}.jsonl",
            run_identity={
                "run_id": f"spoof-{field}", "source_commit": "b" * 40,
                "policy_id": POLICY, "split": "TRAIN",
                "track_ids": [env._reset_track_id],
                "seed_identity": {"ppo": 0, "training_reset": 1000},
                "reset_contracts": {env._reset_track_id: env.training_reset_contract},
            }, max_timesteps=2)
        recorder.on_training_start(_Model(), total_timesteps=1, n_envs=1)
        with pytest.raises(ValueError, match="reset"):
            recorder.on_step(transition, _Model())
        recorder.close()
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["missing_reset", "missing_state_identity", "boolean_seed"])
def test_actual_train_transition_fails_closed_on_missing_or_malformed_identity(tmp_path, mutation):
    env, transition = _actual_train_transition()
    recorder = None
    try:
        info = transition["infos"][0]
        if mutation == "missing_reset":
            info.pop("train_reset")
        elif mutation == "missing_state_identity":
            info["training_telemetry"].pop("seed_identity")
        else:
            info["train_reset"]["seed"] = True
            info["training_telemetry"]["seed_identity"] = dict(info["train_reset"])
        recorder = TrainingTelemetryRecorder(
            tmp_path / f"invalid-{mutation}.jsonl",
            run_identity={
                "run_id": f"invalid-{mutation}", "source_commit": "b" * 40,
                "policy_id": POLICY, "split": "TRAIN",
                "track_ids": [env._reset_track_id],
                "seed_identity": {"ppo": 0, "training_reset": 1000},
                "reset_contracts": {env._reset_track_id: env.training_reset_contract},
            }, max_timesteps=2)
        recorder.on_training_start(_Model(), total_timesteps=1, n_envs=1)
        with pytest.raises(ValueError, match="reset seed identity"):
            recorder.on_step(transition, _Model())
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def test_actual_multienvironment_multitrack_transitions_match_authoritative_contract(tmp_path):
    scenarios, envs, transition = _actual_multitrack_batch()
    track_ids = [scenario.track_id for scenario in scenarios.scenarios[:2]]
    contracts = {scenario.track_id: scenario.factory.reset_config.identity_for_track(scenario.track_id)
                 for scenario in scenarios.scenarios[:2]}
    recorder = TrainingTelemetryRecorder(
        tmp_path / "multitrack.jsonl",
        run_identity={"run_id": "multitrack", "source_commit": "b" * 40,
                      "policy_id": "multitrack-seed0", "split": "TRAIN",
                      "track_ids": track_ids,
                      "seed_identity": {"ppo": 0, "training_reset": 1000},
                      "reset_contracts": contracts},
        max_timesteps=4)
    try:
        recorder.on_training_start(_Model(), total_timesteps=2, n_envs=2)
        recorder.on_step(transition, _Model())
        assert recorder.step_rows == 2
    finally:
        recorder.close()
        for env in envs:
            env.close()


def test_sampled_train_wrappers_expose_contracts_and_record_ranked_transitions(tmp_path):
    protocol = build_protocol(ROOT, profile=PROFILE)
    scenarios, _ = prepare_training_scenarios(ROOT, protocol, "multitrack-seed0")
    wrappers, observations, actions, rewards, dones, infos = [], [], [], [], [], []
    try:
        for rank in range(2):
            env = _SampledF1RLEnv(scenarios, sampling_seed=31, rank=rank)
            env.set_training_telemetry_enabled(True)
            observation, _ = env.reset(seed=17 + rank)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            wrappers.append(env)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
        contracts = wrappers[0].training_reset_contracts
        transition = {
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "clipped_actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool), "infos": infos,
        }
        recorder = TrainingTelemetryRecorder(
            tmp_path / "sampled-multitrack.jsonl",
            run_identity={"run_id": "sampled", "source_commit": "b" * 40,
                          "policy_id": "multitrack-seed0", "split": "TRAIN",
                          "track_ids": list(contracts), "seed_identity": {"training_reset": 1000},
                          "reset_contracts": contracts}, max_timesteps=4)
        recorder.on_training_start(_Model(), total_timesteps=2, n_envs=2)
        recorder.on_step(transition, _Model())
        assert recorder.step_rows == 2
        assert all(info["track_id"] in contracts for info in infos)
        recorder.close()
    finally:
        for env in wrappers:
            env.close()


def test_training_telemetry_toggle_preserves_transition_and_seven_key_evaluation_telemetry():
    protocol = build_protocol(ROOT, profile=PROFILE)
    scenarios, _ = prepare_training_scenarios(ROOT, protocol, POLICY)
    outputs = []
    envs = []
    try:
        for enabled in (False, True):
            env = scenarios.scenarios[0].factory.create()
            env.set_training_telemetry_enabled(enabled)
            observation, reset_info = env.reset(seed=17)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            result = env.step(action)
            outputs.append((observation, reset_info, result))
            envs.append(env)
        off, on = outputs
        np.testing.assert_array_equal(off[0], on[0])
        np.testing.assert_array_equal(off[2][0], on[2][0])
        assert off[2][1:4] == on[2][1:4]
        off_info, on_info = off[2][4], on[2][4]
        assert set(off_info["telemetry"]) == set(on_info["telemetry"]) == {
            "vehicle_position", "vehicle_velocity", "vehicle_speed", "grounded",
            "obstacle_hit_ids", "applied_boost_pad_ids", "applied_slow_zone_ids",
        }
        assert {key: value for key, value in off_info.items() if key != "training_telemetry"} == {
            key: value for key, value in on_info.items() if key != "training_telemetry"
        }
        assert "training_telemetry" not in off_info
        assert on_info["training_telemetry"]["schema_version"] == "f1rl_training_state_v1"
    finally:
        for env in envs:
            env.close()
