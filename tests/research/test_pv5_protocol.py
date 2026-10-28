from pathlib import Path
import json

import pytest

from src.research.wp7_protocol import build_protocol, validate_protocol, evaluation_seed
from src.research.wp7_protocol import capture_runtime_manifest, validate_runtime_manifest
from src.research.wp7_protocol import prepare_evaluation_factory
from src.research.wp7_protocol import load_registration

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_binds_actual_runtime_contracts_and_all_required_cells():
    protocol = build_protocol(ROOT)
    assert len(protocol["policies"]) == 15
    assert protocol["required_cells"] == 120
    assert protocol["minimum_evaluation_episodes"] == 7200
    assert protocol["training_authorized"] is False
    assert len(protocol["tracks"]) == 4
    assert all(len(t["race_sha256"]) == 64 for t in protocol["tracks"])
    assert len(protocol["physics_sha256"]) == 64
    assert "flight_race_reset_integration_missing" in protocol["registration_blockers"]
    validate_protocol(protocol, ROOT)
    validate_protocol(json.loads(json.dumps(protocol)), ROOT)


@pytest.mark.parametrize("field", ["tracks", "physics_sha256", "required_cells", "registration"])
def test_missing_or_tampered_protocol_cannot_be_used_as_provenance(field):
    protocol = build_protocol(ROOT)
    protocol.pop(field)
    with pytest.raises(ValueError, match="protocol"):
        validate_protocol(protocol, ROOT)


def test_evaluation_stream_is_identity_bound_and_does_not_pool_cells():
    protocol = build_protocol(ROOT)
    first = evaluation_seed(protocol, "multitrack-seed0", "slalom-v1", "randomized_train", 0)
    assert first == evaluation_seed(protocol, "multitrack-seed0", "slalom-v1", "randomized_train", 0)
    assert first != evaluation_seed(protocol, "multitrack-seed0", "slalom-v1", "randomized_train", 1)
    with pytest.raises(ValueError):
        evaluation_seed(protocol, "multitrack-seed0", "heldout-v1", "nominal", 0)


def test_future_runtime_capture_is_explicit_and_artifact_drift_is_rejected(tmp_path):
    protocol = build_protocol(ROOT)
    model = tmp_path / "synthetic.zip"
    model.write_bytes(b"synthetic hash fixture, not trained model")
    artifacts = {"checkpoint-0": model}
    manifest = capture_runtime_manifest(ROOT, protocol, artifacts=artifacts)
    assert manifest["scope"] == "current_capture_not_historical_provenance"
    assert manifest["runtime"]["cpu"]["logical_processors"] > 0
    assert manifest["runtime"]["packages"]["torch"]
    assert manifest["runtime"]["packages"]["stable-baselines3"]
    validate_runtime_manifest(manifest, protocol, ROOT, artifacts=artifacts)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        validate_runtime_manifest(manifest, protocol, ROOT, artifacts=artifacts)


def test_missing_runtime_dependencies_cannot_pass_manifest_validation():
    protocol = build_protocol(ROOT)
    manifest = capture_runtime_manifest(ROOT, protocol)
    manifest["runtime"]["packages"].pop("torch")
    with pytest.raises(ValueError, match="manifest"):
        validate_runtime_manifest(manifest, protocol, ROOT)


def test_boolean_and_numeric_json_types_are_not_interchangeable():
    protocol = build_protocol(ROOT)
    protocol["training_authorized"] = 0
    with pytest.raises(ValueError, match="protocol"):
        validate_protocol(protocol, ROOT)


def test_nominal_and_randomized_cells_construct_distinct_reset_contracts_without_rollout():
    protocol = build_protocol(ROOT)
    nominal, first = prepare_evaluation_factory(ROOT, protocol, "multitrack-seed0", "slalom-v1", "nominal", 0)
    randomized, second = prepare_evaluation_factory(ROOT, protocol, "multitrack-seed0", "slalom-v1", "randomized_train", 0)
    assert nominal.reset_config is None
    assert randomized.reset_config.seed == second["environment_seed"]
    assert first["environment_seed"] != second["environment_seed"]
    assert second["deterministic_prediction"] is False
    assert randomized.requires_scientific_admission


@pytest.mark.parametrize("change", ["duplicate_seed", "bool_seed", "threshold", "unknown_parameter"])
def test_registration_rejects_malformed_scientific_inputs(tmp_path, change):
    document = load_registration(ROOT)
    if change == "duplicate_seed":
        document["seed_roster"][1]["ppo"] = 0
    elif change == "bool_seed":
        document["seed_roster"][0]["sampling"] = False
    elif change == "threshold":
        document["evaluation"]["minimum_lower_bound"] = .8
    else:
        document["engineering_suspension"]["inferred_measurement"] = 1.
    path = tmp_path / "configs/research/wp7-pv5-protocol-v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registration(tmp_path)
