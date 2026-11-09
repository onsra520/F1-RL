from copy import deepcopy
from pathlib import Path

import pytest

from f1rl.research.wp7_train5_v4_admission import (
    V4AdmissionError, assess_v4_admission, development_seeds,
    load_v4_protocol, require_v4_training_admission, training_vector_seed_overlaps, validate_v4_protocol,
)

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_allocations_and_development_seeds_are_deterministic():
    protocol, digest = load_v4_protocol(ROOT)
    seeds = development_seeds(ROOT, protocol, digest)
    assert len(seeds) == len(set(seeds.values())) == 3600
    assert seeds == development_seeds(ROOT, protocol, digest)
    reserved = {seed for run in protocol["runs"] for seed in run["seeds"].values()}
    assert len(reserved) == 18
    assert not reserved.intersection(seeds.values())
    vector_seeds = {run["seeds"]["ppo"] + rank for run in protocol["runs"] for rank in range(8)}
    assert not vector_seeds.intersection(seeds.values())
    assert not training_vector_seed_overlaps(protocol)
    assert [run["initial_checkpoint"] for run in protocol["runs"][3:]] == ["C-seed0", "B-seed1", "A-seed2"]


def test_stochastic_policy_seeds_are_reproducible_and_separate_from_environment():
    from f1rl.research.wp7_train5_v4_admission import development_policy_seeds

    protocol, digest = load_v4_protocol(ROOT)
    env = development_seeds(ROOT, protocol, digest)
    policy = development_policy_seeds(ROOT, protocol, digest)
    assert policy == development_policy_seeds(ROOT, protocol, digest)
    assert set(policy) == set(env)
    assert len(set(policy.values())) == 3600
    assert not set(policy.values()).intersection(env.values())
    reserved = {seed for run in protocol["runs"] for seed in run["seeds"].values()}
    reserved.update(seed for run in protocol["runs"] for seed in range(
        run["seeds"]["ppo"], run["seeds"]["ppo"] + protocol["budget"]["vector_envs"]))
    assert not set(policy.values()).intersection(reserved)


@pytest.mark.parametrize("mutation", ["seed", "checkpoint", "threshold", "permission", "exception"])
def test_protocol_rejects_unregistered_changes(mutation):
    protocol, _ = load_v4_protocol(ROOT)
    protocol = deepcopy(protocol)
    if mutation == "seed":
        protocol["runs"][0]["seeds"]["ppo"] += 1
    elif mutation == "checkpoint":
        protocol["runs"][3]["initial_checkpoint"] = "A-seed2"
    elif mutation == "threshold":
        protocol["development"]["thresholds"]["nominal_finish_rate_min"] = .8
    elif mutation == "permission":
        protocol["permissions"]["training"] = True
    else:
        protocol["seed_exception"]["logs_recovered"] = True
    with pytest.raises(V4AdmissionError):
        validate_v4_protocol(ROOT, protocol)


def test_wrong_protocol_hash_cannot_select_development_seeds():
    protocol, _ = load_v4_protocol(ROOT)
    with pytest.raises(V4AdmissionError, match="saved protocol"):
        development_seeds(ROOT, protocol, "0" * 64)


def test_admission_remains_closed_with_forged_review():
    result = assess_v4_admission(ROOT, review_envelope={"training_permitted": True}, run_manifest_sha256="0" * 64)
    assert result["training_permitted"] is False
    assert result["heldout_permitted"] is False
    assert result["signatures_match_repository_registry"] is False
    assert result["registry_independently_authenticated"] is False
    assert len(result["open_compatibility_gates"]) == 9
    assert result["training_vector_seed_overlap_count"] == 0
    assert "NEW_SEED_GLOBAL_REGISTRY_OPEN" in result["blockers"]
    with pytest.raises(V4AdmissionError, match="not granted"):
        require_v4_training_admission(ROOT)


def test_matching_untrusted_registry_does_not_grant_admission(monkeypatch):
    import f1rl.research.wp7_train5_v4_admission as admission
    monkeypatch.setattr(admission, "verify_repository_run_reviews", lambda *args, **kwargs: {})
    result = admission.assess_v4_admission(ROOT, review_envelope={}, run_manifest_sha256="0" * 64)
    assert result["signatures_match_repository_registry"] is True
    assert result["registry_independently_authenticated"] is False
    assert result["training_permitted"] is False


def test_changed_bound_source_is_rejected(monkeypatch):
    protocol, _ = load_v4_protocol(ROOT)
    target = ROOT / "f1rl/research/wp7_train5_v4_runner.py"
    original = Path.read_bytes

    def read_bytes(path):
        contents = original(path)
        return contents + b"changed" if path == target else contents

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(V4AdmissionError, match="protocol source changed"):
        validate_v4_protocol(ROOT, protocol)
