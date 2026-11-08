from dataclasses import replace
from pathlib import Path

import pytest

from f1rl.learning.action import ActionContract
from f1rl.learning.gym_environment import F1RLEnv
from f1rl.learning.rewards import RewardComposer
from f1rl.learning.training.env_factory import TrainingScenario, TrainingScenarioSet
from f1rl.track_assets.canonical import fingerprint_track_world
import scripts.wp7_train5_v4_factory_preview as preview

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "audit", lambda *_: {"external_witness_status": "FOUR_EXTERNAL_WITNESSES_SHA256_VERIFIED"})
    monkeypatch.setattr(F1RLEnv, "reset", lambda *args, **kwargs: pytest.fail("preview must not consume reset seeds"))
    monkeypatch.setattr(F1RLEnv, "step", lambda *args, **kwargs: pytest.fail("preview must not run physics"))
    return preview.preview_registered_factories(ROOT, tmp_path, cohort="train5_from_scratch", replicate=0)


def test_preview_uses_five_train_tracks_and_keeps_research_guards(prepared, tmp_path):
    assert tuple(row.track_id for row in prepared.scenarios) == preview.TRAIN5_IDS
    assert all(row.factory.requires_scientific_admission for row in prepared.scenarios)
    assert all(row.factory.observation_encoder.size == 16 for row in prepared.scenarios)
    assert all(row.factory.action_contract.version == "steering-throttle-brake-v2" for row in prepared.scenarios)
    preview.validate_registered_factories(ROOT, tmp_path, prepared, cohort="train5_from_scratch", replicate=0)


@pytest.mark.parametrize("cohort", ["train5_from_scratch", "train5_from_frozen_checkpoint"])
@pytest.mark.parametrize("replicate", [0, 1, 2])
def test_each_registered_replicate_uses_its_assigned_reset_without_sampling(tmp_path, monkeypatch, cohort, replicate):
    monkeypatch.setattr(preview, "audit", lambda *_: {"external_witness_status": "FOUR_EXTERNAL_WITNESSES_SHA256_VERIFIED"})
    monkeypatch.setattr(F1RLEnv, "reset", lambda *args, **kwargs: pytest.fail("preview must not consume reset seeds"))
    protocol, _ = preview.load_v4_protocol(ROOT)
    seed = next(row["seeds"]["perturbation"] for row in protocol["runs"] if row["cohort"] == cohort and row["replicate"] == replicate)
    scenarios = preview.preview_registered_factories(ROOT, tmp_path, cohort=cohort, replicate=replicate)
    assert all(row.factory.reset_config.seed == seed for row in scenarios.scenarios)


@pytest.mark.parametrize("field", ["action", "observation", "reward", "physics"])
def test_rejects_shared_environment_contract_drift(prepared, tmp_path, field):
    altered = []
    for row in prepared.scenarios:
        factory = row.factory
        if field == "action":
            factory = replace(factory, action_contract=ActionContract("steering-throttle-brake0-v1"))
        elif field == "observation":
            encoder = factory.observation_encoder
            factory = replace(factory, observation_encoder=replace(encoder, config=replace(encoder.config, velocity_scale=encoder.config.velocity_scale * 2)))
        elif field == "reward":
            factory = replace(factory, reward_composer=RewardComposer(()))
        else:
            factory = replace(factory, physics_config=replace(factory.physics_config, ground_motion_version="spatial_rotating_compliant_lifecycle_v3"))
        altered.append(TrainingScenario(row.track_id, factory))
    with pytest.raises(ValueError, match="scenario geometry, configuration or reset"):
        preview.validate_registered_factories(ROOT, tmp_path, TrainingScenarioSet(tuple(altered)), cohort="train5_from_scratch", replicate=0)


def test_rejects_altered_geometry_even_with_matching_caller_registry(prepared, tmp_path):
    original = prepared.scenarios[0]
    first_surface = original.factory.track.surfaces[0]
    changed_track = replace(original.factory.track, surfaces=(replace(first_surface, width=first_surface.width + 1), *original.factory.track.surfaces[1:]))
    reset = original.factory.reset_config
    entries = tuple(replace(entry, world_sha256=fingerprint_track_world(changed_track)) if entry.track_id == original.track_id else entry for entry in reset.training_registry)
    changed_reset = replace(reset, training_registry=entries)
    rows = tuple(TrainingScenario(row.track_id, replace(row.factory,
        track=changed_track if row.track_id == original.track_id else row.factory.track,
        reset_config=changed_reset)) for row in prepared.scenarios)
    with pytest.raises(ValueError, match="scenario geometry, configuration or reset"):
        preview.validate_registered_factories(ROOT, tmp_path, TrainingScenarioSet(rows), cohort="train5_from_scratch", replicate=0)


def test_refuses_unverified_external_witnesses(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "audit", lambda *_: {"external_witness_status": "NOT_VERIFIED_EXTERNAL_ROOT_NOT_PROVIDED"})
    with pytest.raises(ValueError, match="witnesses"):
        preview.preview_registered_factories(ROOT, tmp_path, cohort="train5_from_scratch", replicate=0)


def test_matching_factories_still_cannot_obtain_training_admission(prepared, tmp_path):
    with pytest.raises(ValueError, match="admission is not granted"):
        preview.require_registered_factory_admission(ROOT, tmp_path, prepared, cohort="train5_from_scratch", replicate=0)
