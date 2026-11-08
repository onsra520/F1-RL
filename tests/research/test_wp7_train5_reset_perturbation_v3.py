"""TRAIN-5 five-track perturbed reset engineering tests; no learner invoked."""
from pathlib import Path
import json
import math
import numpy as np
import pytest

from f1rl.research.wp7_train5_reset_perturbation_v3 import (
    FIXTURE_SEEDS, MAX_ATTEMPTS, RESET_VERSION,
    prepare_train5_perturbed_scenarios, fixture_identity,
)
from f1rl.research.wp7_train5_v2_protocol import TRAIN5_IDS

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(r"D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106")


def _factory_set(seed):
    return prepare_train5_perturbed_scenarios(
        ROOT, evidence_root=EVIDENCE, perturbation_seed=seed,
    )


def test_five_train_factories_have_registered_lifecycle_reset_not_training_admission():
    scenarios = _factory_set(FIXTURE_SEEDS[0])
    assert tuple(s.track_id for s in scenarios.scenarios) == TRAIN5_IDS
    assert scenarios.observation_space.shape == (16,)
    assert scenarios.action_space.shape == (3,)
    configs = {s.factory.reset_config.sha256 for s in scenarios.scenarios}
    assert len(configs) == 1
    for scenario in scenarios.scenarios:
        f = scenario.factory
        assert f.reset_config.seed == FIXTURE_SEEDS[0]
        assert f.reset_config.version == RESET_VERSION
        assert f.reset_config.max_attempts == MAX_ATTEMPTS
        assert f.reset_track_id == scenario.track_id
        assert f.requires_scientific_admission is True
        assert f.physics_config.ground_motion_version == "spatial_road_edge_partial_lifecycle_v4"
        assert f.race_config.version == "wp7_directed_lifecycle_crash_contract_v4"


def test_actual_five_track_perturbed_reset_deterministic_and_independent():
    scenarios = _factory_set(FIXTURE_SEEDS[0])
    draws = {}
    for scenario in scenarios.scenarios:
        meta = []
        for _ in range(2):
            env = scenario.factory.create(reset_rank=1, reset_sequence=0)
            try:
                obs, info = env.reset(seed=FIXTURE_SEEDS[0])
                assert env.observation_space.contains(obs)
                metadata = info["train_reset"]
                assert metadata["version"] == RESET_VERSION
                assert metadata["seed"] == FIXTURE_SEEDS[0]
                assert metadata["track_id"] == scenario.track_id
                assert metadata["rank"] == 1 and metadata["reset_index"] == 0
                assert -0.5 <= metadata["lateral_offset_m"] <= 0.5
                assert -5.0 <= metadata["heading_jitter_deg"] <= 5.0
                assert math.isfinite(metadata["lateral_offset_m"])
                assert math.isfinite(metadata["heading_jitter_deg"])
                assert metadata["scientific_evidence"] is False
                meta.append(metadata)
            finally:
                env.close()
        assert meta[0] == meta[1]
        draws[scenario.track_id] = meta[0]["draw_seed"]
    assert len(set(draws.values())) == 5


def test_reset_sequence_changes_independent_draws_without_consuming_training_seeds():
    scenarios = _factory_set(FIXTURE_SEEDS[1])
    for scenario in scenarios.scenarios:
        env = scenario.factory.create(reset_rank=0, reset_sequence=0)
        try:
            first, info_first = env.reset(seed=FIXTURE_SEEDS[1])
            second, info_second = env.reset(seed=FIXTURE_SEEDS[1])
            assert env.observation_space.contains(first)
            assert env.observation_space.contains(second)
            assert info_first["train_reset"]["draw_seed"] != info_second["train_reset"]["draw_seed"]
            assert info_first["train_reset"]["reset_index"] == 0
            assert info_second["train_reset"]["reset_index"] == 1
        finally:
            env.close()
    reg = json.loads((ROOT / "configs/research/wp7-train5-paired-v2-registration.json").read_text("utf8"))
    reserved = {
        seed for group in reg["proposed_training_seeds"].values()
        for row in group for seed in row.values()
    }
    assert len(reserved) == 18
    assert not (set(FIXTURE_SEEDS) & reserved)


@pytest.mark.parametrize("invalid", [-1, 2**32, 2026111001, 2026112203, 3.14, True])
def test_reset_adapter_refuses_invalid_or_reserved_seeds(invalid):
    with pytest.raises(ValueError):
        _factory_set(invalid)


def test_fixture_identity_is_non_authorizing_and_source_bound():
    x = fixture_identity(ROOT, EVIDENCE, FIXTURE_SEEDS[0])
    assert x["factory_count"] == 5
    assert x["training_permission"] is False
    assert x["scientific_validation"] is False
    assert x["used_reserved_training_seeds"] is False
    assert set(x["tracks"]) == set(TRAIN5_IDS)
