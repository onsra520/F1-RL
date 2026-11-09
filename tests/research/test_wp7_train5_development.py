from types import SimpleNamespace
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.wp7_train5_v4_development import episode_outcome, wilson_interval

ROOT = Path(__file__).resolve().parents[2]


def test_wilson_interval_is_conditional_and_nonzero_for_zero_successes():
    low, high = wilson_interval(0, 60)
    assert low == pytest.approx(0.0, abs=1e-15)
    assert high == pytest.approx(0.06017, abs=0.0001)


def test_episode_outcome_rejects_ambiguous_or_uncertified_terminal():
    assert episode_outcome(SimpleNamespace(success=True, truncated=False), {}) == "finished"
    assert episode_outcome(SimpleNamespace(success=False, truncated=True), {}) == "valid_step_limit"
    with pytest.raises(ValueError, match="unclassified"):
        episode_outcome(SimpleNamespace(success=False, truncated=False), {"research_outcome": {"reason": "unknown"}})


def test_registered_roll_envelope_terminal_is_reported_separately(monkeypatch):
    import f1rl.learning.roll_envelope_registered as roll

    monkeypatch.setattr(roll, "validate_registered_roll_envelope_info", lambda info: True)
    info = {"research_outcome": {"reason": "tilt_envelope_exit", "valid_sample": True}}
    assert episode_outcome(SimpleNamespace(success=False, truncated=False), info) == "tilt_envelope_exit"


def test_fake_stochastic_episode_uses_distinct_repeatable_policy_and_reset_seeds(monkeypatch):
    from scripts import wp7_train5_v4_development as development

    @dataclass(frozen=True)
    class Reset:
        seed: int

    @dataclass(frozen=True)
    class Factory:
        reset_config: Reset | None
        reset_track_id: str | None

    observed = []

    def fake_evaluate(**kwargs):
        observed.append((kwargs["factory"].reset_config.seed, kwargs["reset_seed"],
                         kwargs["deterministic"], float(torch.rand(1))))
        kwargs["step_observer"]({"terminated": True, "truncated": False,
                                 "info": {"research_outcome": {"version": "wp7_episode_outcome_v1",
                                                               "valid_sample": True, "reason": "finish"}}})
        return SimpleNamespace(success=True, truncated=False, episode_return=1.5)

    monkeypatch.setattr(development, "evaluate_episode", fake_evaluate)
    factory = Factory(Reset(5), "straight-sprint-v1")
    args = dict(factory=factory, policy=object(), track_id="straight-sprint-v1", condition="perturbation",
                environment_seed=123, policy_seed=456)
    first = development.evaluate_one(**args)
    second = development.evaluate_one(**args)
    assert first == second == {"outcome": "finished", "valid_physics": True, "episode_return": 1.5}
    assert observed[0] == observed[1]
    assert observed[0][:3] == (123, 123, False)


def test_incomplete_batch_is_rejected_before_evaluation_output(tmp_path):
    from scripts.wp7_train5_v4_development import evaluate_batch

    (tmp_path / "batch-result.json").write_text(json.dumps({"schema": "train5_internal_batch_result_v1", "runs": []}), encoding="utf8")
    with pytest.raises(ValueError):
        evaluate_batch(ROOT, tmp_path, tmp_path)
    assert not (tmp_path / "evaluation-journal.jsonl").exists()


def test_invalid_terminal_is_not_reported_as_valid(monkeypatch):
    from scripts import wp7_train5_v4_development as development

    @dataclass(frozen=True)
    class Factory:
        reset_config: object
        reset_track_id: str | None

    def invalid(**kwargs):
        kwargs["step_observer"]({"terminated": True, "truncated": False,
                                 "info": {"research_outcome": {"version": "wp7_episode_outcome_v1", "valid_sample": False}}})
        return SimpleNamespace(success=True, truncated=False, episode_return=1.0)

    monkeypatch.setattr(development, "evaluate_episode", invalid)
    with pytest.raises(ValueError, match="invalid or uncertified"):
        development.evaluate_one(factory=Factory(None, None), policy=object(), track_id="straight-sprint-v1",
                                 condition="nominal", environment_seed=123, policy_seed=456)


def test_wrong_checkpoint_step_is_not_accepted_for_development():
    from scripts.wp7_train5_v4_development import validate_loaded_policy

    space = object()
    policy = SimpleNamespace(num_timesteps=200000, _model=SimpleNamespace(observation_space=space, action_space=space),
                             policy_weights_finite=lambda: True)
    with pytest.raises(ValueError, match="policy steps"):
        validate_loaded_policy(policy, SimpleNamespace(observation_space=space, action_space=space))


def test_fake_six_run_batch_produces_all_3600_train_records(tmp_path, monkeypatch):
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.research import wp7_train5_internal_execution as execution
    from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS, prepare_v4_registered_runner
    from scripts import wp7_train5_batch as batch
    from scripts import wp7_train5_v4_development as development
    from scripts import wp7_train5_v4_factory_preview as preview

    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: key.public_key().public_bytes_raw())
    monkeypatch.setattr(batch, "_require_clean_source", lambda root, head: None)
    plan = batch.build_plan(ROOT, tmp_path / "runs", tmp_path, valid_hours=168)
    envelopes = batch.sign_plan(ROOT, plan, key, now=datetime.now(timezone.utc))
    output = tmp_path / "runs"
    output.mkdir()
    entries = []
    for run, envelope in zip(plan["runs"], envelopes, strict=True):
        cohort, replicate = run["cohort"], run["replicate"]
        reference = prepare_v4_registered_runner(ROOT, output, cohort=cohort, replicate=replicate)
        directory = output / f"{cohort}-{replicate}"
        directory.mkdir()
        model = directory / "final.zip"
        model.write_bytes(f"fake-{cohort}-{replicate}".encode())
        state = {"status": "completed", "run_id": directory.name, "execution_scope": "INTERNAL_SIMULATION_ONLY",
                 "ppo": asdict(reference._config.ppo), "vector_env": asdict(reference._config.vector_env),
                 "sampling_seed": run["seeds"]["sampling"], "training_reset_seed": run["seeds"]["perturbation"],
                 "training_tracks": plan["track_ids"], "initial_checkpoint": reference._initial_checkpoint,
                 "mode": "multitrack", "requested_total_timesteps": 200000, "actual_total_timesteps": 200704,
                 "checkpoint_every_steps": None, "latest_checkpoint": None, "final_model": model.name,
                 "internal_authorization": envelope}
        (directory / "state.json").write_text(json.dumps(state), encoding="utf8")
        entries.append({"cohort": cohort, "replicate": replicate, "run_directory": str(directory),
                        "final_model": str(model), "actual_steps": 200704, "sha256": sha256(model.read_bytes()).hexdigest()})
    result = {"schema": "train5_internal_batch_result_v1", "source_commit": plan["source_commit"],
              "protocol_sha256": plan["protocol_sha256"], "manifest_sha256": plan["manifest_sha256"],
              "runs": entries, "frozen_checkpoint_sha256": FROZEN_CHECKPOINTS,
              "scientific_admission": False, "heldout_evaluated": False}
    (output / "batch-result.json").write_text(json.dumps(result), encoding="utf8")
    observation_space, action_space = object(), object()
    scenario_set = SimpleNamespace(scenarios=[SimpleNamespace(track_id=track, factory=object()) for track in plan["track_ids"]],
                                   observation_space=observation_space, action_space=action_space)
    monkeypatch.setattr(preview, "preview_registered_factories", lambda *args, **kwargs: scenario_set)
    fake_policy = SimpleNamespace(_model=SimpleNamespace(observation_space=observation_space, action_space=action_space),
                                  num_timesteps=200704, policy_weights_finite=lambda: True)
    monkeypatch.setattr(PPOModel, "load", lambda *args, **kwargs: fake_policy)
    monkeypatch.setattr(development, "evaluate_one", lambda **kwargs: {
        "outcome": "finished", "valid_physics": True, "episode_return": 1.0})
    monkeypatch.setattr(development.os, "fsync", lambda fd: None)
    report = development.evaluate_batch(ROOT, output, tmp_path)
    assert report["summary"]["input_episode_count"] == 3600
    assert len(report["cells"]) == 60
    assert report["cohort_policy_summaries"]["scratch"]["mean"] == 1.0
    assert report["scientific_admission"] is False and report["heldout_evaluated"] is False
    events = (output / "evaluation-journal.jsonl").read_text(encoding="utf8").splitlines()
    assert len(events) == 7200
    with pytest.raises(ValueError, match="already started"):
        development.evaluate_batch(ROOT, output, tmp_path)
