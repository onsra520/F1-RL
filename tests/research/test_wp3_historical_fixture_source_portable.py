"""Portable exact-byte WP3 fixture provenance checks for the Physics-v4 handoff.

These assert archival identity and fail-closed status ONLY; they do not
authorize registration on the unrelated Git branch or read HELD-OUT.
"""
from hashlib import sha256
from pathlib import Path
import json

from f1rl.learning.crash_terminal_registered import _sha


ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/"docs/research"
HISTORIC={
    "WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json":
        "80f7db86f4e12a07521e70b7725630b66ca709ae7fe50ff9f52f84bf45b7f3a5",
    "WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt":
        "f550d9a53e264fe4a8cc9ffc15dc932f61bdd1cc5048406b86c9f7c6d0e60142",
    "WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json":
        "086f27aef269d2cb999f121b008b3e168168fce274907a513585dfbf0641729d",
    "WP3-V4L-RIGHT-TURN-PATCH-BRACKET-20261024.json":
        "517705a28a743cd5ce6791cb2cb827263df91f9bad2db5fdadebed02b2010b9f",
    "WP3-V4N-RAMP-BUMP-TRACE-20261024.json":
        "0392e1d9f2e049a281eaf7ffb2ff28ce8ed388f5bcbb86163b8139b79c0a0286",
    "WP3-V4M-SLALOM-GROUNDED-CONTACT-20261024.json":
        "4ad0336bf992f4ea443998b815763da218efd01a78ff96f64ad8e7c11c4bb104",
}


def test_exact_six_historical_wp3_git_blobs_match_cross_approved_sha():
    assert len(HISTORIC)==6
    for name,expected in HISTORIC.items():
        path=DIRECTORY/name
        assert path.is_file(),name
        assert sha256(path.read_bytes()).hexdigest()==expected,name


def test_original_technical_registry_is_self_consistent_but_not_signoff():
    record=json.loads(
        (DIRECTORY/"WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json")
        .read_text(encoding="utf8")
    )
    assert record["record_sha256"]==_sha({
        k:v for k,v in record.items() if k!="record_sha256"
    })
    assert record["scientific_scoring_approved"] is False
    assert record["owner_scientific_signoff"] is False
    assert record["full_5760_episode_evaluation_authorized"] is False
    assert record["registered_original_checkpoint_count"]==12


def test_original_three_invalid_replays_cannot_be_promoted_to_release():
    for name,token in (
        ("WP3-V4L-RIGHT-TURN-PATCH-BRACKET-20261024.json","contact"),
        ("WP3-V4N-RAMP-BUMP-TRACE-20261024.json","bump"),
        ("WP3-V4M-SLALOM-GROUNDED-CONTACT-20261024.json","obstacle"),
    ):
        raw=json.loads((DIRECTORY/name).read_text(encoding="utf8"))
        assert raw["full_5760_eval_admission"]=="BLOCKED"
        assert len(raw["cases"])==1
        entry=raw["cases"][0]
        assert entry["scientific_evidence"] is False
        assert token in entry["error"].lower()


def test_original_registration_source_is_not_implicitly_promoted():
    record=json.loads(
        (DIRECTORY/"WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json")
        .read_text(encoding="utf8")
    )
    assert record["source_base_commit"]=="ce0e55d7a428826b2d4ed3218641105bd0206f4b"
    assert record["scientific_signoff"] is False
    assert record["full_5760_eval_admission"]=="BLOCKED"
    assert record["heldout_access"] is False
    assert record["registered_evaluation_executed"] is False
    assert len(record["cases"])==3
    assert all(row["scientific_evidence"] is False for row in record["cases"])


def test_heldout_manifest_can_validate_only_slalom_train_asset():
    from f1rl.track_assets import load_track_manifest

    # Explicit selector validates the registered TRAIN Slalom SHA and geometry
    # while the rest of the historical manifest remains metadata-only.
    m=load_track_manifest(
        ROOT/"assets/tracks/manifest.json",validate_asset_ids=("slalom-v1",)
    )
    slalom=next(x for x in m.tracks if x.track_id=="slalom-v1")
    assert slalom.path=="training/slalom-v1.json"
    assert slalom.world_sha256=="6071829f2eb790ada04c5e59b22efb205e3a20d59fbe670e68fb78ed3eb304e0"
