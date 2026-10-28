"""Research protocol reconstruction, never admission or a PPO rollout."""

from pathlib import Path

import pytest

from src.research.wp7_protocol import (
    build_protocol,
    load_registration,
    prepare_evaluation_factory,
    prepare_scenarios,
    prepare_training_scenarios,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("seed_index,expected", [(0, 1000), (1, 1001), (2, 1002)])
def test_each_replicate_binds_only_its_registered_train_reset_seed(seed_index, expected):
    registered = load_registration(ROOT)
    scenarios = prepare_scenarios(ROOT, registered, seed_index=seed_index)
    assert len(scenarios.scenarios) == 4
    assert {s.factory.reset_config.seed for s in scenarios.scenarios} == {expected}
    assert all(s.factory.requires_scientific_admission for s in scenarios.scenarios)


@pytest.mark.parametrize("bad_index", [-1, 3, True, None, 1.0])
def test_implicit_or_invalid_train_seed_selection_is_rejected(bad_index):
    with pytest.raises(ValueError, match="seed_index"):
        prepare_scenarios(ROOT, load_registration(ROOT), seed_index=bad_index)


@pytest.mark.parametrize("seed_index", [0, 1, 2])
def test_registered_policy_factory_tracks_and_seed_bundle_match(seed_index):
    protocol = build_protocol(ROOT)
    multitrack, multi = prepare_training_scenarios(ROOT, protocol, f"multitrack-seed{seed_index}")
    straight, single = prepare_training_scenarios(
        ROOT, protocol, f"singletrack-straight-sprint-v1-seed{seed_index}"
    )
    expected = 1000 + seed_index
    assert multi["seed_bundle"] == protocol["registration"]["seed_roster"][seed_index]
    assert multi["paired_seed_index"] == seed_index
    assert multi["track_ids"] == [s.track_id for s in multitrack.scenarios]
    assert multi["training_authorized"] is False
    assert {s.factory.reset_config.seed for s in multitrack.scenarios} == {expected}
    assert len(straight.scenarios) == 1
    assert single["track_ids"] == ["straight-sprint-v1"]
    assert straight.scenarios[0].factory.reset_config.seed == expected
    assert multi["reset_sha256"] == single["reset_sha256"]


def test_evaluation_factory_preserves_training_replicate_identity_without_rollout():
    protocol = build_protocol(ROOT)
    for index in range(3):
        _, metadata = prepare_evaluation_factory(
            ROOT, protocol, f"singletrack-straight-sprint-v1-seed{index}",
            "slalom-v1", "nominal", 0,
        )
        assert metadata["paired_seed_index"] == index
        assert metadata["training_seed_bundle"]["perturbation"] == 1000 + index
        assert metadata["training_authorized"] is False


def test_unregistered_policy_is_rejected_before_scenario_preparation():
    protocol = build_protocol(ROOT)
    with pytest.raises(ValueError, match="unregistered training policy"):
        prepare_training_scenarios(ROOT, protocol, "multitrack-seed999")
