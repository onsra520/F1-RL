"""Keep the frozen TRAIN-5 registration while binding the compatible source revision."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "configs/research/wp7-train5-source-compatibility-v2.json"
PORTABLE = ROOT / "configs/research/wp7-train5-paired-v2-portable-source-revision-v2.json"
PORTABLE_V1 = ROOT / "configs/research/wp7-train5-paired-v2-portable-source-revision-v1.json"
HISTORICAL = ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
V1_PRESERVATION_COMMIT = "d34a58341c55540d37d8743621be5cc2b4154bec"


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _git_blob(commit, path):
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
        check=True, capture_output=True,
    ).stdout


def test_source_compatibility_record_preserves_frozen_registration_and_anchors_real_blobs():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    portable = json.loads(PORTABLE.read_text(encoding="utf-8"))
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))

    v1_blob = _git_blob(
        V1_PRESERVATION_COMMIT,
        "configs/research/wp7-train5-paired-v2-portable-source-revision-v1.json",
    )
    assert v1_blob == _git_blob(
        "a6015c2ce9e18f45bc231d669f57ac97816a08cb",
        "configs/research/wp7-train5-paired-v2-portable-source-revision-v1.json",
    )
    assert hashlib.sha256(v1_blob).hexdigest() == (
        "5a809d1ed45f2b4af4f8cd29e52a84b3abc17566f32462ed809b830ae1edfe98"
    )
    assert PORTABLE_V1.read_bytes().replace(b"\r\n", b"\n") == v1_blob

    assert record["sha256"] == _canonical_sha(
        {key: value for key, value in record.items() if key != "sha256"}
    )
    assert record["schema"] == "f1rl_m5_source_lock_compatibility_v2"
    assert record["status"] == "ENGINEERING_COMPATIBILITY_ONLY"
    assert record["training_permission"] is False
    assert record["smoke_permission"] is False
    assert record["heldout_permission"] is False
    assert record["frozen_historical_registration_modified"] is False
    assert record["verification"]["source_lock"] == "PASS_44_OF_44"
    assert record["verification"]["physics_v4_ppo_contracts"] == "PASS_156"
    assert portable["source_anchor_commit"] == record["compatibility_anchor_commit"]
    assert record["preserved_source_sha256"] == historical["source_sha256"]
    historical_blob = _git_blob(
        record["compatibility_anchor_commit"], record["historical_registration"]
    )
    historical_worktree = HISTORICAL.read_bytes()
    assert record["historical_registration_worktree_relation"] == "CRLF_to_LF_only"
    assert historical_worktree.replace(b"\r\n", b"\n") == historical_blob
    assert hashlib.sha256(historical_blob).hexdigest() == record["historical_registration_sha256"]

    files = record["compatibility_source_files"]
    assert set(files) == {
        "f1rl/learning/gym_environment.py",
        "f1rl/learning/training/env_factory.py",
        "f1rl/research/wp7_train5_v2_protocol.py",
        "f1rl/runtime/physics.py",
    }
    for path, binding in files.items():
        before = hashlib.sha256(_git_blob(record["historical_anchor_commit"], path)).hexdigest()
        after = hashlib.sha256(_git_blob(record["compatibility_anchor_commit"], path)).hexdigest()
        assert binding["before_git_blob_sha256"] == before
        assert binding["after_git_blob_sha256"] == after == portable["git_blob_sha256"][path]

    assert files["f1rl/research/wp7_train5_v2_protocol.py"]["classification"] == (
        "git_blob_unchanged_worktree_line_endings_normalized"
    )
    assert all(
        files[path]["classification"] == "opt_in_layered_v5_support_only_preserve_v4"
        for path in (
            "f1rl/learning/gym_environment.py",
            "f1rl/learning/training/env_factory.py",
            "f1rl/runtime/physics.py",
        )
    )


def test_portable_source_lock_validates_without_external_witness_access():
    from f1rl.research import wp7_train5_v2_protocol as protocol

    registration = protocol.validate_preview(
        ROOT,
        registration_file=ROOT / protocol.REGISTRATION,
        evidence_root=ROOT,
        check_evidence=False,
    )

    assert len(registration["source_sha256"]) == 44
    assert registration["source_anchor_commit"] == "d34a58341c55540d37d8743621be5cc2b4154bec"
    assert registration["training_permission"] is False
    assert registration["heldout_permission"] is False
