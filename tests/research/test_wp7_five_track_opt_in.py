"""Opt-in five-track TRAIN smoke preserves historical four-track research."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.research.wp7_protocol import load_registration, prepare_scenarios
from f1rl.track_assets import load_track_asset, load_track_manifest

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = "spatial_rotating_owner_v3"
FIVE_IDS = (*TRAIN_IDS, "left-turn-train-v1")
CANDIDATE = ROOT / "configs/recovery/wp7-c2-five-track.yml"
MANIFEST = ROOT / "assets/tracks/manifest-five-track.json"
FROZEN = {
    "configs/experiments/ppo-generalization-v1.yml": "b00257a0b4e20fd2c512f867be661aa35ea9c1427dc0b61c1302384df2b1f84f",
    "configs/recovery/wp7-c2-candidate.yml": "370a4a6b38954683434292bf82e16403b5aa6a4119ceb5d97d147c3142e283f9",
    "assets/tracks/manifest.json": "5a0ffc507fffef6fcc782d7acf4a16f543dbd89794e96df143e2dcd6e87b0dda",
    "assets/tracks/training/right-turn-v1.json": "a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e",
}


def _build(**options):
    reg = load_registration(ROOT, profile=REGISTRATION)
    return prepare_scenarios(ROOT, reg, seed_index=0, **options)


def test_historical_default_four_unchanged():
    assert TRAIN_IDS == (
        "straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"
    )
    scenarios = _build()
    assert tuple(x.track_id for x in scenarios.scenarios) == TRAIN_IDS
    assert tuple(x.track_id for x in scenarios.scenarios[0].factory.reset_config.training_registry) == TRAIN_IDS
    for name, digest in FROZEN.items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_five_track_selection_fails_closed_on_exact_heldout_fingerprint():
    manifest = load_track_manifest(ROOT / "assets/tracks/manifest.json", validate_asset_ids=TRAIN_IDS)
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    proposed = metadata["tracks"][4]
    # HELD-OUT metadata only; NEVER load evaluation geometry.
    heldout = next(x for x in manifest.tracks if x.track_id == "left-turn-v1")
    assert proposed["world_sha256"] == heldout.world_sha256
    assert proposed["content_sha256"] == heldout.content_sha256
    with pytest.raises(ValueError, match="TRAIN-5 BLOCKED.*left-turn-v1"):
        _build(candidate_file=CANDIDATE, manifest_path=MANIFEST, track_ids=FIVE_IDS)


@pytest.mark.parametrize("overrides", [
    {"candidate_file": CANDIDATE},
    {"manifest_path": MANIFEST, "track_ids": FIVE_IDS},
    {"candidate_file": CANDIDATE, "track_ids": FIVE_IDS},
    {"candidate_file": CANDIDATE, "manifest_path": MANIFEST},
    {"candidate_file": ROOT/"configs/recovery/wp7-c2-candidate.yml", "manifest_path": MANIFEST, "track_ids": FIVE_IDS},
    {"candidate_file": CANDIDATE, "manifest_path": ROOT/"assets/tracks/manifest.json", "track_ids": FIVE_IDS},
    {"candidate_file": CANDIDATE, "manifest_path": MANIFEST, "track_ids": (*TRAIN_IDS, "left-turn-v1")},
    {"candidate_file": CANDIDATE, "manifest_path": MANIFEST, "track_ids": (*TRAIN_IDS, "left-turn-train-v1", "left-turn-train-v1")},
    {"candidate_file": CANDIDATE, "manifest_path": MANIFEST, "track_ids": tuple(reversed(FIVE_IDS))},
    {"candidate_file": CANDIDATE, "manifest_path": MANIFEST, "track_ids": []},
    {"candidate_file": CANDIDATE, "manifest_path": MANIFEST, "track_ids": ["", *FIVE_IDS]},
])
def test_inconsistent_explicit_five_track_selection_rejected_before_any_geometry(monkeypatch, overrides):
    import f1rl.research.wp7_protocol as module
    monkeypatch.setattr(module, "build_training_scenarios", lambda _: pytest.fail("Geometry accessed before guard"))
    with pytest.raises((ValueError, TypeError)):
        _build(**overrides)


def test_reflection_is_exact_and_heldout_geometry_not_loaded(monkeypatch):
    right = json.loads((ROOT / "assets/tracks/training/right-turn-v1.json").read_text())
    left = json.loads((ROOT / "assets/tracks/training/left-turn-train-v1.json").read_text())
    assert left["track_id"] == "left-turn-train-v1"
    left["track_id"] = right["track_id"]
    expected = deepcopy(right)
    for row in expected["surfaces"]:
        row["origin"][0] *= -1
        row["forward"][0] *= -1
    expected["spawn"]["position"][0] *= -1
    for key in ("slow_zones", "checkpoints", "finish_lines"):
        for row in expected[key]:
            row["volume"]["center"][0] *= -1
    for point in expected["navigation"]["points"]:
        point[0] *= -1
    assert left == expected
    assert load_track_asset(ROOT/"assets/tracks/training/left-turn-train-v1.json").track_id == "left-turn-train-v1"
    original = load_track_asset
    def checked(path):
        assert "/evaluation/" not in Path(path).as_posix()
        return original(path)
    monkeypatch.setattr("f1rl.learning.reconstruction.load_track_asset", checked)
    with pytest.raises(ValueError, match="TRAIN-5 BLOCKED.*HELD-OUT|TRAIN-5 BLOCKED.*left-turn-v1"):
        _build(candidate_file=CANDIDATE, manifest_path=MANIFEST, track_ids=FIVE_IDS)


def test_original_full06_historical_evidence_unmodified():
    report = ROOT / "docs/.superpowers/handoff/evidence/phase2a-root-cause-audit-20261106/report.md"
    assert "5,760" in report.read_text(encoding="utf-8", errors="replace")
    assert "96/96" in report.read_text(encoding="utf-8", errors="replace")
