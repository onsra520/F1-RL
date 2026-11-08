"""Canonical C2 protocol v2: strict version separation and honest draft gate.

No held-out environments are constructed. Track identifiers are metadata only.
"""
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from f1rl.learning.experiments import (
    ExperimentDefinition,
    ExperimentSemantics,
    OffTrackExperimentConfig,
    RewardExperimentConfig,
    SeedProtocol,
)

V1 = Path("configs/experiments/ppo-generalization-v1.yml")
V2 = Path("configs/experiments/ppo-generalization-c2-v2.yml")


def _load_modified(tmp_path: Path, edit) -> ExperimentDefinition:
    document = yaml.safe_load(V2.read_text(encoding="utf-8"))
    edit(document)
    destination = tmp_path / "candidate.yml"
    # Absolute refs preserve the checked-in manifest and runtime profile
    # while varying only the contract under test.
    document["tracks"]["manifest"] = str(Path("assets/tracks/manifest.json").resolve())
    document["training"]["runtime_config"] = str(
        Path("configs/recovery/wp7-pilot-200k.yml").resolve()
    )
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return ExperimentDefinition.from_yaml(destination)


def test_original_v1_stays_strict_and_unchanged() -> None:
    legacy = ExperimentDefinition.from_yaml(V1)
    assert legacy.metadata.protocol_version == 1
    assert legacy.metadata.status == "draft"
    assert legacy.semantics.dynamics is None
    assert legacy.semantics.navigation is None
    assert legacy.reward.route_progress_weight_per_meter is None
    assert legacy.reward.offtrack_penalty is None
    assert legacy.offtrack is None


def test_c2_v2_is_a_complete_distinct_draft_contract() -> None:
    old = ExperimentDefinition.from_yaml(V1)
    new = ExperimentDefinition.from_yaml(V2)
    assert new.metadata.id == "ppo-generalization-c2-v2"
    assert new.metadata.protocol_version == 2
    assert new.metadata.status == "draft"
    assert new.training.protocol_status == "pilot_owned"
    assert new.training.runtime_config.name == "wp7-pilot-200k.yml"
    assert new.vehicle == old.vehicle
    assert new.simulation == old.simulation
    assert new.observation == old.observation
    assert new.episode == old.episode
    assert new.tracks == old.tracks
    assert new.reward == RewardExperimentConfig(
        target_progress_weight_per_meter=None,
        collision_penalty=5.0,
        time_penalty_per_second=0.1,
        route_progress_weight_per_meter=1.0,
        offtrack_penalty=2.0,
    )
    assert new.offtrack == OffTrackExperimentConfig(6.0, 10)
    assert new.semantics == ExperimentSemantics(
        algorithm="ppo",
        policy="MlpPolicy",
        action="steering-throttle-brake0-v1",
        observation="rangefinders-local-route-navigation-v1",
        reward="route-highwater-progress-collision-time-offtrack-v1",
        race="single-lap-ordered-checkpoints-v1",
        dynamics="ground-no-slip-air-ballistic-v1",
        navigation="authored-surface-polyline-endpoint-v1",
    )


def test_v2_draft_cannot_be_frozen_or_readied_by_changing_only_status() -> None:
    definition = ExperimentDefinition.from_yaml(V2)
    with pytest.raises(ValueError, match="experiment status must be frozen"):
        definition.require_frozen()
    with pytest.raises(ValueError, match="training protocol must be frozen"):
        replace(definition, metadata=replace(definition.metadata, status="frozen")).require_frozen()
    with pytest.raises(ValueError, match="non-pilot final training profile"):
        replace(
            definition,
            metadata=replace(definition.metadata, status="frozen"),
            training=replace(definition.training, protocol_status="frozen"),
        ).require_frozen()


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_protocol_version_must_be_exact_supported_integer(tmp_path: Path, version) -> None:
    with pytest.raises(ValueError, match="protocol_version"):
        _load_modified(tmp_path, lambda doc: doc["experiment"].update(protocol_version=version))


@pytest.mark.parametrize("key", ["dynamics", "navigation", "reward", "observation"])
def test_v2_semantics_are_atomic(tmp_path: Path, key: str) -> None:
    legacy = yaml.safe_load(V1.read_text(encoding="utf-8"))["semantics"][key] if key in ("reward", "observation") else None

    def corrupt(doc) -> None:
        doc["semantics"][key] = legacy if legacy is not None else "not-the-versioned-identity"

    with pytest.raises(ValueError, match="semantics"):
        _load_modified(tmp_path, corrupt)


def test_mismatched_metadata_and_semantics_rejected_even_when_instantiated() -> None:
    v1 = ExperimentDefinition.from_yaml(V1)
    v2 = ExperimentDefinition.from_yaml(V2)
    with pytest.raises(ValueError, match="semantics"):
        replace(v1, metadata=replace(v1.metadata, protocol_version=2))
    with pytest.raises(ValueError, match="semantics"):
        replace(v2, metadata=replace(v2.metadata, protocol_version=1))


@pytest.mark.parametrize("mutation", [
    lambda d: d.pop("offtrack"),
    lambda d: d["offtrack"].update(max_consecutive_steps=0),
    lambda d: d["offtrack"].update(lateral_limit_meters=-1),
    lambda d: d["offtrack"].update(unknown=42),
    lambda d: d["environment"]["reward"].pop("offtrack"),
    lambda d: d["environment"]["reward"].pop("route_progress"),
    lambda d: d["environment"]["reward"].update(target_progress={"weight_per_meter": 1.0}),
    lambda d: d["environment"]["reward"]["offtrack"].update(penalty=-1),
])
def test_v2_offtrack_and_route_reward_fail_closed(tmp_path: Path, mutation) -> None:
    with pytest.raises(ValueError):
        _load_modified(tmp_path, mutation)


def test_v1_document_cannot_silently_gain_c2_fields(tmp_path: Path) -> None:
    old = yaml.safe_load(V1.read_text(encoding="utf-8"))
    old["offtrack"] = {"lateral_limit_meters": 6.0, "max_consecutive_steps": 10}
    path = tmp_path / "mutated-v1.yml"
    path.write_text(yaml.safe_dump(old), encoding="utf-8")
    with pytest.raises(ValueError, match="root keys"):
        ExperimentDefinition.from_yaml(path)


def test_v2_unresolved_route_reward_is_not_freezeable() -> None:
    v2 = ExperimentDefinition.from_yaml(V2)
    with pytest.raises(ValueError, match="unresolved"):
        replace(
            v2,
            metadata=replace(v2.metadata, status="frozen"),
            training=replace(v2.training, protocol_status="frozen"),
            seeds=SeedProtocol((100, 101, 102), (200, 201, 202)),
            reward=replace(v2.reward, route_progress_weight_per_meter=None),
        ).require_frozen()


def test_canonical_c2_matches_actual_train_candidate_without_heldout_rollouts() -> None:
    from f1rl.learning.experiments.c2_alignment import verify_draft_c2_alignment
    from f1rl.learning.reconstruction import RecoveryCandidate

    candidate = RecoveryCandidate.from_yaml(Path("configs/recovery/wp7-c2-candidate.yml"))
    verify_draft_c2_alignment(ExperimentDefinition.from_yaml(V2), candidate)


@pytest.mark.parametrize("field,value", [
    ("wheelbase", 2.5),
    ("ride_height", 0.7),
])
def test_candidate_alignment_rejects_environment_drift(field, value) -> None:
    from f1rl.learning.experiments.c2_alignment import verify_draft_c2_alignment
    from f1rl.learning.reconstruction import RecoveryCandidate

    canonical = ExperimentDefinition.from_yaml(V2)
    candidate = RecoveryCandidate.from_yaml(Path("configs/recovery/wp7-c2-candidate.yml"))
    modified = replace(canonical, vehicle=replace(canonical.vehicle, **{field: value}))
    with pytest.raises(ValueError, match="environment"):
        verify_draft_c2_alignment(modified, candidate)


def test_candidate_alignment_rejects_reward_and_corridor_drift() -> None:
    from f1rl.learning.experiments.c2_alignment import verify_draft_c2_alignment
    from f1rl.learning.reconstruction import RecoveryCandidate

    canonical = ExperimentDefinition.from_yaml(V2)
    candidate = RecoveryCandidate.from_yaml(Path("configs/recovery/wp7-c2-candidate.yml"))
    for modified in (
        replace(canonical, reward=replace(canonical.reward, route_progress_weight_per_meter=1.5)),
        replace(canonical, reward=replace(canonical.reward, offtrack_penalty=3.0)),
        replace(canonical, offtrack=replace(canonical.offtrack, lateral_limit_meters=3.0)),
        replace(canonical, offtrack=replace(canonical.offtrack, max_consecutive_steps=5)),
    ):
        with pytest.raises(ValueError, match="reward/offtrack"):
            verify_draft_c2_alignment(modified, candidate)


def test_candidate_alignment_rejects_fake_freeze_or_seed_promotion() -> None:
    from f1rl.learning.experiments.c2_alignment import verify_draft_c2_alignment
    from f1rl.learning.reconstruction import RecoveryCandidate

    canonical = ExperimentDefinition.from_yaml(V2)
    candidate = RecoveryCandidate.from_yaml(Path("configs/recovery/wp7-c2-candidate.yml"))
    for modified in (
        replace(canonical, metadata=replace(canonical.metadata, status="frozen")),
        replace(canonical, training=replace(canonical.training, protocol_status="frozen")),
        replace(canonical, seeds=SeedProtocol((100, 101, 102), (200, 201, 202))),
    ):
        with pytest.raises(ValueError, match="draft"):
            verify_draft_c2_alignment(modified, candidate)


def test_v2_frozen_contract_requires_independent_seed_replicates(tmp_path: Path) -> None:
    definition = ExperimentDefinition.from_yaml(V2)
    final_profile = tmp_path / "ppo-generalization-c2-v2-training.yml"
    final_profile.write_text("schema_version: 2\n", encoding="utf-8")
    staged = replace(
        definition,
        metadata=replace(definition.metadata, status="frozen"),
        training=replace(definition.training, protocol_status="frozen", runtime_config=final_profile),
    )
    with pytest.raises(ValueError, match="at least three paired"):
        staged.require_frozen()
    with pytest.raises(ValueError, match="at least three paired"):
        replace(staged, seeds=SeedProtocol((100, 101, 102), (200, 201))).require_frozen()
